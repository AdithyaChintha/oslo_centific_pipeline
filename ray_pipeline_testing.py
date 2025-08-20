# Simple Ray Pipeline with Time-Based Detection Integration
import ray
import os
import gc
import torch
import requests
import json
from pathlib import Path
from utils.logger import get_logger

# Import setup and all necessary Ray tasks
from setup.cosmos.setup import setup_cosmos
from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
from ray_jobs.scene_det import detect_scenes
from ray_jobs.run_yolodetect_task import run_yolodetect_on_shard
from ray_jobs.audio_diarization_pii import process_audio_diarization

# Import new ray jobs
from ray_jobs.nsfw_det_new import process_video_chunks_for_nsfw
from ray_jobs.motion_energy import compute_motion_energy
from ray_jobs.face_age_detector import process_video_chunks_for_face_detection

logger = get_logger("SimplifiedUnifiedPipeline")

# def download_nsfw_model(model_dir: str = "models/nsfw"):
#     """Download NSFW detection model and labels if not present"""
#     os.makedirs(model_dir, exist_ok=True)
    
#     model_path = os.path.join(model_dir, "nsfw_model.onnx")
#     labels_path = os.path.join(model_dir, "labels.json")
    
#     # Check if files already exist
#     if os.path.exists(model_path) and os.path.exists(labels_path):
#         logger.info(f"NSFW model already exists at {model_path}")
#         return model_path, labels_path
    
    # try:
    #     # Download model
    #     if not os.path.exists(model_path):
    #         logger.info("Downloading NSFW detection model...")
    #         model_url = "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_checkpoint.onnx"
    #         response = requests.get(model_url, stream=True)
    #         response.raise_for_status()
            
    #         with open(model_path, 'wb') as f:
    #             for chunk in response.iter_content(chunk_size=8192):
    #                 f.write(chunk)
    #         logger.info(f"Model downloaded to {model_path}")
        
    #     # Create labels file if not exists
    #     if not os.path.exists(labels_path):
    #         labels = {"0": "safe", "1": "nsfw"}
    #         with open(labels_path, 'w') as f:
    #             json.dump(labels, f)
    #         logger.info(f"Labels file created at {labels_path}")
        
    #     return model_path, labels_path
        
    # except Exception as e:
    #     logger.error(f"Failed to download NSFW model: {e}")
    #     return None, None

def clear_gpu_memory():
    """Clear GPU memory between tasks"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")

def extract_flagged_segments(task_result, task_type, shard_index, shard_offset_sec):
    """Extract flagged segments from task results"""
    segments = []
    
    try:
        if not task_result:
            return segments
            
        # Different tasks have different result formats
        if task_type == "audio":
            # Check for PII detections in audio result
            pii_detections = task_result.get('pii_detections', [])
            for pii in pii_detections:
                segments.append({
                    "start_time": shard_offset_sec + pii.get('start_time', 0),
                    "end_time": shard_offset_sec + pii.get('end_time', 0),
                    "task_type": "audio_pii",
                    "confidence": 0.8,
                    "flag_type": "pii_detected",
                    "priority": "high",
                    "description": f"PII detected: {pii.get('entity_type', 'unknown')}",
                    "shard_index": shard_index + 1
                })
                
        elif task_type == "yolo":
            # YOLO doesn't return segments directly, could process events file here
            # For now, we'll skip detailed YOLO segment extraction
            pass
            
        elif task_type == "scene":
            success = task_result.get('processing_info', {}).get('success', False)
            if not success:
                logger.warning(f"Scene detection failed for shard {shard_index}: {task_result.get('processing_info', {}).get('error', 'Unknown error')}")
                return segments
            
            # FIXED: Look for 'scenes' instead of 'flagged_segments'
            scene_data = task_result.get('scenes', [])
            for scene in scene_data:
                segments.append({
                    "start_time": shard_offset_sec + scene.get('start_time', 0),
                    "end_time": shard_offset_sec + scene.get('end_time', 60),
                    "task_type": "scene_detection",
                    "confidence": 0.8,  # Default confidence for scene detection
                    "flag_type": "scene_content",
                    "priority": "medium",
                    "description": scene.get('description', 'Scene detected')[:200],  # Truncate long descriptions
                    "shard_index": shard_index + 1
                })
                
        elif task_type == "nsfw":
            # NSFW returns flagged_segments
            nsfw_segments = task_result.get('flagged_segments', [])
            for seg in nsfw_segments:
                segments.append({
                    "start_time": seg.get('start_time', 0),  # Already has offset
                    "end_time": seg.get('end_time', 0),
                    "task_type": "nsfw_detection",
                    "confidence": seg.get('confidence', 0.7),
                    "flag_type": "nsfw_content",
                    "priority": seg.get('priority', 'high'),
                    "description": seg.get('description', 'NSFW content detected'),
                    "shard_index": shard_index + 1
                })
                
        elif task_type == "motion":
            # Motion energy returns segments with start/end times
            motion_segments = task_result.get('segments', [])
            for seg in motion_segments:
                activity_type = seg.get('activity_type', '').lower()
                # Only flag high activity segments
                if 'high' in activity_type or seg.get('avg_motion_energy', 0) > 0.7:
                    segments.append({
                        "start_time": shard_offset_sec + seg.get('start_time', 0),
                        "end_time": shard_offset_sec + seg.get('end_time', 0),
                        "task_type": "motion_energy",
                        "confidence": seg.get('confidence', 0.6),
                        "flag_type": "high_motion",
                        "priority": "medium",
                        "description": f"High motion activity: {activity_type}",
                        "shard_index": shard_index + 1
                    })
                    
        elif task_type == "face":
            # Face detection returns flagged_segments
            face_segments = task_result.get('flagged_segments', [])
            for seg in face_segments:
                segments.append({
                    "start_time": seg.get('start_time', 0),  # Already has offset
                    "end_time": seg.get('end_time', 0),
                    "task_type": "face_detection",
                    "confidence": seg.get('confidence', 0.7),
                    "flag_type": seg.get('flag_type', 'face_detected'),
                    "priority": seg.get('priority', 'medium'),
                    "description": seg.get('description', 'Face detected'),
                    "shard_index": shard_index + 1
                })
                
    except Exception as e:
        logger.warning(f"Error extracting segments from {task_type}: {e}")
    
    return segments

def merge_overlapping_segments(segments, merge_threshold=1.0):
    """Merge overlapping or very close segments"""
    if not segments:
        return segments
    
    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x['start_time'])
    merged = []
    
    for segment in sorted_segments:
        if not merged:
            merged.append(segment)
            continue
        
        last_segment = merged[-1]
        
        # Check if segments overlap or are very close
        if segment['start_time'] <= last_segment['end_time'] + merge_threshold:
            # Merge segments
            last_segment['end_time'] = max(last_segment['end_time'], segment['end_time'])
            
            # Combine task types
            if isinstance(last_segment.get('task_type'), str):
                combined_tasks = [last_segment['task_type']]
            else:
                combined_tasks = last_segment.get('task_type', [])
                
            if segment['task_type'] not in combined_tasks:
                combined_tasks.append(segment['task_type'])
            
            last_segment['task_type'] = combined_tasks if len(combined_tasks) > 1 else combined_tasks[0]
            last_segment['description'] = f"Combined flags: {', '.join(combined_tasks)}"
            
            # Use highest priority
            priorities = ['high', 'medium', 'low']
            current_priority = priorities.index(last_segment.get('priority', 'low'))
            new_priority = priorities.index(segment.get('priority', 'low'))
            if new_priority < current_priority:
                last_segment['priority'] = segment['priority']
        else:
            merged.append(segment)
    
    return merged

def pipeline_main(input_video_path: str, output_dir: str):
    """
    Simplified unified Ray pipeline for video analysis with time-based detection segments
    """
    ray.init()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- STAGE A: INITIAL SETUP ---
    logger.info("Starting simplified pipeline setup...")
    setup_cosmos()
    
    # Download NSFW model if needed
    nsfw_model_path = '/tmp/nsfw_model.onnx'
    
    logger.info("Setup complete.")

    # --- STAGE B: VIDEO PREPARATION ---
    logger.info(f"Preparing video: {input_video_path}")
    
    # Convert .insv to .mp4 if necessary
    if input_video_path.lower().endswith('.insv'):
        mp4_path_ref = convert_insv_to_dual_mp4.remote(input_video_path)
        mp4_path = ray.get(mp4_path_ref)
        logger.info(f"Converted {input_video_path} to {mp4_path}")
    else:
        mp4_path = input_video_path

    # Split the video into shards
    shards_dir = os.path.join(output_dir, "video_shards")
    shard_paths_ref = split_video_into_shards.remote(mp4_path, output_dir=shards_dir, duration_sec=60)
    shard_paths = ray.get(shard_paths_ref)
    logger.info(f"Video split into {len(shard_paths)} shards in {shards_dir}")

    # --- STAGE C: SEQUENTIAL ANALYSIS ---
    logger.info("Launching sequential analysis tasks for each shard...")
    
    # Initialize result lists
    scene_results = []
    yolo_results = []
    audio_results = []
    nsfw_results = []
    motion_results = []
    face_results = []
    
    # All flagged segments across all shards
    all_flagged_segments = []
    
    # Define the prompt for scene detection
    prompt_path = "config/cosmos_prompt.yaml"
    
    # Verify prompt file exists
    if not os.path.exists(prompt_path):
        logger.error(f"Prompt file not found: {prompt_path}")
        logger.info("Creating default prompt file...")
        os.makedirs("config", exist_ok=True)
        
        default_prompt = {
            "system_prompt": "You are an AI assistant that analyzes video content and identifies different scenes or activities.",
            "user_prompt": "Please analyze this video and describe the different scenes or activities you observe. Focus on identifying distinct segments and their content."
        }
        
        import yaml
        with open(prompt_path, 'w') as f:
            yaml.dump(default_prompt, f, default_flow_style=False)
        logger.info(f"Created default prompt file at {prompt_path}")

    # Process each shard sequentially
    for i, shard_path in enumerate(shard_paths):
        shard_output_dir = os.path.join(output_dir, f"shard_{i+1}")
        os.makedirs(shard_output_dir, exist_ok=True)
        shard_offset_sec = i * 60  # Each shard is 60 seconds
        
        logger.info(f"Processing shard {i+1}/{len(shard_paths)}: {os.path.basename(shard_path)}")
        
        # --- EXISTING TASKS ---
        
        # Task 1: Audio Diarization & PII
        logger.info(f"  - Running audio diarization for shard {i+1}...")
        try:
            audio_output_dir = os.path.join(shard_output_dir, "audio_output")
            audio_task = process_audio_diarization.remote([shard_path], audio_output_dir)
            audio_result = ray.get(audio_task)
            audio_results.append(audio_result)
            
            # Extract flagged segments
            segments = extract_flagged_segments(audio_result, "audio", i, shard_offset_sec)
            all_flagged_segments.extend(segments)
            
            logger.info(f"  - ✅ Audio diarization completed for shard {i+1}")
        except Exception as e:
            logger.error(f"  - ❌ Audio diarization failed for shard {i+1}: {e}")
            audio_results.append(None)
        
        # clear_gpu_memory()
        
        # Task 2: YOLO Detection
        logger.info(f"  - Running YOLO detection for shard {i+1}...")
        try:
            yolo_output_dir = os.path.join(shard_output_dir, "yolo_output")
            yolo_task = run_yolodetect_on_shard.remote(shard_path, yolo_output_dir)
            yolo_result = ray.get(yolo_task)
            yolo_results.append(yolo_result)
            
            # Extract flagged segments (minimal for now)
            segments = extract_flagged_segments(yolo_result, "yolo", i, shard_offset_sec)
            all_flagged_segments.extend(segments)
            
            logger.info(f"  - ✅ YOLO detection completed for shard {i+1}")
        except Exception as e:
            logger.error(f"  - ❌ YOLO detection failed for shard {i+1}: {e}")
            yolo_results.append(None)
        
        clear_gpu_memory()
        
        # Task 3: Scene Detection
        logger.info(f"  - Running scene detection for shard {i+1}...")
        try:
            scene_output_dir = os.path.join(shard_output_dir, "scene_output")
            scene_task = detect_scenes.remote(shard_path, prompt_path, scene_output_dir)
            scene_result = ray.get(scene_task)
            scene_results.append(scene_result)
            
            # FIXED: Check if scene detection actually succeeded
            if scene_result and scene_result.get('processing_info', {}).get('success', False):
                # Extract flagged segments only if successful
                segments = extract_flagged_segments(scene_result, "scene", i, shard_offset_sec)
                all_flagged_segments.extend(segments)
                
                # ADDED: Verify result file was saved
                result_file = os.path.join(scene_output_dir, "scene_detection_results.json")
                if os.path.exists(result_file):
                    logger.info(f"  - Scene detection completed successfully for shard {i+1} - Results saved to {result_file}")
                    logger.info(f"    Extracted {len(segments)} scene segments")
                else:
                    logger.warning(f"  - Scene detection completed but no result file found at {result_file}")
            else:
                error_msg = scene_result.get('processing_info', {}).get('error', 'Unknown error') if scene_result else 'No result returned'
                logger.error(f"  - Scene detection failed for shard {i+1}: {error_msg}")
        except Exception as e:
            logger.error(f"  - ❌ Scene detection failed for shard {i+1}: {e}")
            scene_results.append(None)
        
        clear_gpu_memory()

        # Task 4: NSFW Detection - FIXED PARAMETERS
        logger.info(f"  - Running NSFW detection for shard {i+1}...")
        try:
            nsfw_output_dir = os.path.join(shard_output_dir, "nsfw_output")
            os.makedirs(nsfw_output_dir, exist_ok=True)
            
            # Fix: Use correct parameters that match nsfw_det.py function signature
            nsfw_task = process_video_chunks_for_nsfw.remote(
                chunk_paths=[shard_path],
                model_path=nsfw_model_path,  # Use the downloaded model path
                confidence_threshold=0.5,
                chunk_duration_sec=60
            )
            nsfw_result = ray.get(nsfw_task)
            nsfw_results.append(nsfw_result)
            
            # Extract flagged segments and save results
            if nsfw_result and nsfw_result.get("success"):
                segments = extract_flagged_segments(nsfw_result, "nsfw", i, shard_offset_sec)
                all_flagged_segments.extend(segments)
                
                # Save detailed NSFW results with segments
                nsfw_detailed_result = {
                    "shard_index": i + 1,
                    "shard_file": shard_path,
                    "shard_offset_seconds": shard_offset_sec,
                    "nsfw_result": nsfw_result,
                    "extracted_segments": segments
                }
                
                if nsfw_result and nsfw_result.get("success"):
                    video_name = os.path.splitext(os.path.basename(shard_path))[0]
                    nsfw_file = os.path.join(nsfw_output_dir, f"{video_name}_nsfw_results.json")
                    with open(nsfw_file, 'w') as f:
                        json.dump(nsfw_result, f, indent=2)
                
                # Log what we found
                total_detections = nsfw_result.get('total_nsfw_detections', 0)
                flagged_segments = len(nsfw_result.get('flagged_segments', []))
                logger.info(f"  - ✅ NSFW detection completed for shard {i+1}")
                logger.info(f"    Found {total_detections} NSFW detections in {flagged_segments} segments")
                
                # Log segment details for debugging
                for seg in nsfw_result.get('flagged_segments', []):
                    logger.info(f"    Segment: {seg.get('start_time', 0):.1f}s - {seg.get('end_time', 0):.1f}s "
                               f"(confidence: {seg.get('confidence', 0):.2f})")
            else:
                error_msg = nsfw_result.get('error', 'Unknown error') if nsfw_result else 'No result returned'
                logger.warning(f"  - ⚠️ NSFW detection failed for shard {i+1}: {error_msg}")
                
        except Exception as e:
            logger.error(f"  - ❌ NSFW detection failed for shard {i+1}: {e}")
            nsfw_results.append(None)
        
        clear_gpu_memory()
        
        # Task 5: Motion Energy Analysis
        logger.info(f"  - Running motion energy analysis for shard {i+1}...")
        try:
            motion_output_dir = os.path.join(shard_output_dir, "motion_output")
            os.makedirs(motion_output_dir, exist_ok=True)
            
            motion_task = compute_motion_energy.remote(
                [shard_path], sensitivity_level="medium", save_detailed_data=False
            )
            motion_result = ray.get(motion_task)
            motion_results.append(motion_result)
            
            # Extract flagged segments
            if motion_result and motion_result.get("success"):
                segments = extract_flagged_segments(motion_result, "motion", i, shard_offset_sec)
                all_flagged_segments.extend(segments)
                
                # Save motion energy results
                if motion_result and motion_result.get("success"):
                    video_name = os.path.splitext(os.path.basename(shard_path))[0]
                    motion_file = os.path.join(motion_output_dir, f"{video_name}_motion_results.json")
                    with open(motion_file, 'w') as f:
                        json.dump(motion_result, f, indent=2)
                    
            logger.info(f"  - ✅ Motion energy analysis completed for shard {i+1}")
        except Exception as e:
            logger.error(f"  - ❌ Motion energy analysis failed for shard {i+1}: {e}")
            motion_results.append(None)
        
        clear_gpu_memory()
        
        # Task 6: Face Age Detection
        logger.info(f"  - Running face age detection for shard {i+1}...")
        try:
            face_output_dir = os.path.join(shard_output_dir, "face_output")
            os.makedirs(face_output_dir, exist_ok=True)
            
            face_task = process_video_chunks_for_face_detection.remote(
                [shard_path], config=None, frame_interval=30, 
                save_frames=False, chunk_duration_sec=60
            )
            face_result = ray.get(face_task)
            face_results.append(face_result)
            
            # Extract flagged segments
            if face_result and face_result.get("success"):
                segments = extract_flagged_segments(face_result, "face", i, shard_offset_sec)
                all_flagged_segments.extend(segments)
                
                # Save face detection results
                if face_result and face_result.get("success"):
                    video_name = os.path.splitext(os.path.basename(shard_path))[0]
                    face_file = os.path.join(face_output_dir, f"{video_name}_face_results.json")
                    with open(face_file, 'w') as f:
                        json.dump(face_result, f, indent=2)
                    
            logger.info(f"  - ✅ Face age detection completed for shard {i+1}")
        except Exception as e:
            logger.error(f"  - ❌ Face age detection failed for shard {i+1}: {e}")
            face_results.append(None)
        
        clear_gpu_memory()
        
        logger.info(f"✅ Completed processing for shard {i+1}/{len(shard_paths)}")

    # --- STAGE D: CREATE MASTER TIMELINE ---
    logger.info("Creating master flagged timeline for annotation workload reduction...")
    
    if all_flagged_segments:
        # Sort segments by start time
        all_flagged_segments.sort(key=lambda x: x['start_time'])
        
        # Merge overlapping segments
        merged_segments = merge_overlapping_segments(all_flagged_segments, merge_threshold=1.0)
        
        # Calculate annotation workload reduction
        total_video_duration = len(shard_paths) * 60  # 60 seconds per shard
        total_flagged_duration = sum(seg['end_time'] - seg['start_time'] for seg in merged_segments)
        annotation_reduction = ((total_video_duration - total_flagged_duration) / total_video_duration) * 100
        
        # Save master timeline
        master_timeline_file = os.path.join(output_dir, "master_flagged_timeline.json")
        with open(master_timeline_file, 'w') as f:
            json.dump({
                "video_file": input_video_path,
                "total_duration_seconds": total_video_duration,
                "flagged_duration_seconds": total_flagged_duration,
                "annotation_workload_reduction": f"{annotation_reduction:.1f}%",
                "total_flagged_segments": len(merged_segments),
                "flagged_timeline": merged_segments,
                "high_priority_segments": [s for s in merged_segments if s.get('priority') == 'high']
            }, f, indent=2)
        
        logger.info(f"📊 Master timeline created: {len(merged_segments)} segments")
        logger.info(f"🎯 Annotation workload reduction: {annotation_reduction:.1f}%")
        logger.info(f"⚡ Only {total_flagged_duration:.1f}s of {total_video_duration}s needs manual review")
    else:
        logger.info("No flagged segments found across all shards")
        annotation_reduction = 0
        merged_segments = []

    # --- STAGE E: RESULTS SUMMARY ---
    logger.info("All shards processed. Generating summary...")
    
    # Calculate success rates
    successful_audio = sum(1 for result in audio_results if result is not None)
    successful_yolo = sum(1 for result in yolo_results if result is not None)
    successful_scene = sum(1 for result in scene_results if result is not None and result.get('processing_info', {}).get('success', False))
    successful_nsfw = sum(1 for result in nsfw_results if result is not None and result.get("success"))
    successful_motion = sum(1 for result in motion_results if result is not None and result.get("success"))
    successful_face = sum(1 for result in face_results if result is not None and result.get("success"))
    total_shards = len(shard_paths)
    
    # Log detailed results
    logger.info(f"Processing complete:")
    logger.info(f"  - Audio Diarization: {successful_audio}/{total_shards} successful")
    logger.info(f"  - YOLO Detection: {successful_yolo}/{total_shards} successful")  
    logger.info(f"  - Scene Detection: {successful_scene}/{total_shards} successful")
    logger.info(f"  - NSFW Detection: {successful_nsfw}/{total_shards} successful")
    logger.info(f"  - Motion Energy Analysis: {successful_motion}/{total_shards} successful")
    logger.info(f"  - Face Age Detection: {successful_face}/{total_shards} successful")
    
    total_tasks = total_shards * 6  # 6 tasks per shard
    successful_tasks = successful_audio + successful_yolo + successful_scene + successful_nsfw + successful_motion + successful_face
    
    # Create summary
    results_summary = {
        'processing_summary': {
            'total_shards': total_shards,
            'successful_audio': successful_audio,
            'successful_yolo': successful_yolo,
            'successful_scene': successful_scene,
            'successful_nsfw': successful_nsfw,
            'successful_motion': successful_motion,
            'successful_face': successful_face,
            'overall_success_rate': f"{(successful_tasks / total_tasks * 100):.1f}%"
        },
        'annotation_summary': {
            'total_flagged_segments': len(merged_segments),
            'flagged_duration_seconds': sum(seg['end_time'] - seg['start_time'] for seg in merged_segments) if merged_segments else 0,
            'annotation_workload_reduction': f"{annotation_reduction:.1f}%",
            'high_priority_segments': len([s for s in merged_segments if s.get('priority') == 'high']) if merged_segments else 0
        }
    }
    
    # Save summary to file
    summary_file = os.path.join(output_dir, "pipeline_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Pipeline summary saved to {summary_file}")
    logger.info("🚀 Simplified unified pipeline complete!")
    
    return results_summary

if __name__ == "__main__":
    # Define the input video and the main output directory
    INPUT_VIDEO = "/home/nvcoe_admin/code/oslo/whole_pipeline_testing/kamwai_chan_input_videos/vaccum_floor_1GB.mp4"
    OUTPUT_DIR = "outputs/simplified_unified_pipeline_output"
    
    try:
        results = pipeline_main(INPUT_VIDEO, OUTPUT_DIR)
        
        print(f"\n🎉 Simplified Pipeline completed!")
        print(f"📊 Processing Summary:")
        for task in ['audio', 'yolo', 'scene', 'nsfw', 'motion', 'face']:
            count = results['processing_summary'][f'successful_{task}']
            total = results['processing_summary']['total_shards']
            print(f"   {task.title()}: {count}/{total} successful")
        
        print(f"\n🎯 Annotation Workload Reduction:")
        print(f"   Flagged Segments: {results['annotation_summary']['total_flagged_segments']}")
        print(f"   Workload Reduction: {results['annotation_summary']['annotation_workload_reduction']}")
        print(f"   High Priority Segments: {results['annotation_summary']['high_priority_segments']}")
        print(f"   Overall Success Rate: {results['processing_summary']['overall_success_rate']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        raise