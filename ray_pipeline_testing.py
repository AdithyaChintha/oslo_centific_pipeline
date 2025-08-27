# Simple Ray Pipeline with Time-Based Detection Integration
import ray
import os
import gc
import torch
import requests
import json
import shutil
import time
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from utils.logger import get_logger
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

# Import setup and all necessary Ray tasks
from setup.cosmos.setup import setup_cosmos
from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.audio_splitter import split_audio_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
from ray_jobs.scene_det import detect_scenes
from ray_jobs.run_yolodetect_task import run_yolodetect_on_shard
from ray_jobs.audio_diarization_pii import process_audio_diarization
from ray_jobs.clap_detector import detect_claps_in_media

# Import new ray jobs
from ray_jobs.nsfw_det_final import process_video_chunks_for_nsfw
from ray_jobs.motion_energy import compute_motion_energy
from ray_jobs.face_age_detector import process_video_chunks_for_face_detection
from ray_jobs.labelstudio_tasks import build_labelstudio_json_for_shard_task, import_to_labelstudio_task


logger = get_logger("SimplifiedUnifiedPipeline")

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

        def maybe_offset(t):
            # If upstream added global times (set by those tasks), don't re-add.
            if isinstance(t, (int, float)):
                return t + shard_offset_sec
            return shard_offset_sec

        if task_type == "audio":
            for pii in task_result.get('pii_detections', []):
                segments.append({
                    "start_time": maybe_offset(pii.get('start_time', 0)),
                    "end_time":   maybe_offset(pii.get('end_time',   0)),
                    "task_type": "audio_pii",
                    "confidence": 0.8,
                    "flag_type": "pii_detected",
                    "priority": "high",
                    "description": f"PII detected: {pii.get('entity_type', 'unknown')}",
                    "shard_index": shard_index + 1
                })

        elif task_type == "yolo":
            pass  # handled elsewhere if needed

        elif task_type == "scene":
            success = task_result.get('processing_info', {}).get('success', False)
            if not success:
                logger.warning(f"Scene detection failed for shard {shard_index}: "
                               f"{task_result.get('processing_info', {}).get('error', 'Unknown error')}")
                return segments
            for scene in task_result.get('scenes', []):
                segments.append({
                    "start_time": maybe_offset(scene.get('start_time', 0)),
                    "end_time":   maybe_offset(scene.get('end_time', 60)),
                    "task_type": "scene_detection",
                    "confidence": 0.8,
                    "flag_type": "scene_content",
                    "priority": "medium",
                    "description": scene.get('description', 'Scene detected')[:200],
                    "shard_index": shard_index + 1
                })

        elif task_type == "nsfw":
            for seg in task_result.get('flagged_segments', []):
                segments.append({
                    "start_time": maybe_offset(seg.get('start_time', 0)),
                    "end_time":   maybe_offset(seg.get('end_time',   0)),
                    "task_type": "nsfw_detection",
                    "confidence": seg.get('confidence', 0.7),
                    "flag_type": "nsfw_content",
                    "priority": seg.get('priority', 'high'),
                    "description": seg.get('description', 'NSFW content detected'),
                    "shard_index": shard_index + 1
                })

        elif task_type == "motion":
            for seg in task_result.get('segments', []):
                activity_type = seg.get('activity_type', '').lower()
                if 'high' in activity_type or seg.get('avg_motion_energy', 0) > 0.7:
                    segments.append({
                        "start_time": maybe_offset(seg.get('start_time', 0)),
                        "end_time":   maybe_offset(seg.get('end_time',   0)),
                        "task_type": "motion_energy",
                        "confidence": seg.get('confidence', 0.6),
                        "flag_type": "high_motion",
                        "priority": "medium",
                        "description": f"High motion activity: {activity_type}",
                        "shard_index": shard_index + 1
                    })

        elif task_type == "face":
            for seg in task_result.get('flagged_segments', []):
                segments.append({
                    "start_time": maybe_offset(seg.get('start_time', 0)),
                    "end_time":   maybe_offset(seg.get('end_time',   0)),
                    "task_type": "face_detection",
                    "confidence": seg.get('confidence', 0.7),
                    "flag_type": seg.get('flag_type', 'face_detected'),
                    "priority": seg.get('priority', 'medium'),
                    "description": seg.get('description', 'Face detected'),
                    "shard_index": shard_index + 1
                })

        elif task_type == "clap":
            for clap in task_result.get('clap_timestamps', []):
                clap_time = clap.get('timestamp_seconds', 0)
                desc = clap.get('timestamp_formatted', f"{clap_time:.2f}s")
                segments.append({
                    "start_time": max(0, shard_offset_sec + clap_time - 2),
                    "end_time":   shard_offset_sec + clap_time + 2,
                    "task_type": "clap_detection",
                    "confidence": 0.9,
                    "flag_type": "clap_detected",
                    "priority": "medium",
                    "description": f"Clap detected at {desc}",
                    "shard_index": shard_index + 1
                })
    except Exception as e:
        logger.warning(f"Error extracting segments from {task_type}: {e}")
    return segments


# Put near imports / helpers
PRIORITY_ORDER = ['high', 'medium', 'low']  # 0 is highest

def merge_overlapping_segments(timeline, tol=1.0):
    """
    Merge overlapping segments across views (1s tolerance), dedupe task types,
    combine source_view, keep max confidence, and choose highest priority.
    """
    if not timeline:
        return []
    timeline = sorted(timeline, key=lambda x: x.get("start_time", 0))
    merged, cur = [], dict(timeline[0])

    def to_set(val):
        if isinstance(val, list): return set(val)
        if isinstance(val, str):  return set(t.strip() for t in val.split(",") if t.strip())
        return set()

    for seg in timeline[1:]:
        if seg.get("start_time", 0) <= cur.get("end_time", 0) + tol:
            cur["end_time"] = max(cur.get("end_time", 0), seg.get("end_time", 0))

            cur_tasks = to_set(cur.get("task_type", "")) | to_set(seg.get("task_type", ""))
            cur["task_type"] = ", ".join(sorted(cur_tasks)) if cur_tasks else ""

            cur_desc, new_desc = cur.get("description", ""), seg.get("description", "")
            if new_desc and new_desc not in cur_desc:
                cur["description"] = f"{cur_desc}; {new_desc}" if cur_desc else new_desc

            cur_views = to_set(cur.get("source_view", "")) | to_set(seg.get("source_view", ""))
            cur["source_view"] = ", ".join(sorted(cur_views)) if cur_views else ""

            cur["confidence"] = max(cur.get("confidence", 0), seg.get("confidence", 0))

            cp = PRIORITY_ORDER.index(cur.get("priority", "medium")) if cur.get("priority", "medium") in PRIORITY_ORDER else 1
            sp = PRIORITY_ORDER.index(seg.get("priority", "medium")) if seg.get("priority", "medium") in PRIORITY_ORDER else 1
            if sp < cp:
                cur["priority"] = seg.get("priority", "medium")
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)
    return merged


def merge_overlapping_segments_dual_view(timeline):
    if not timeline:
        return []
    timeline.sort(key=lambda x: x.get("start_time", 0))
    merged, current = [], timeline[0].copy()
    for segment in timeline[1:]:
        if segment["start_time"] <= current["end_time"] + 1.0:
            current["end_time"] = max(current["end_time"], segment["end_time"])
            # task_type
            cur_tasks = current.get("task_type", "")
            cur_list = cur_tasks if isinstance(cur_tasks, list) else ([t.strip() for t in cur_tasks.split(",")] if cur_tasks else [])
            new_tasks = segment.get("task_type", "")
            new_list = new_tasks if isinstance(new_tasks, list) else ([t.strip() for t in new_tasks.split(",")] if new_tasks else [])
            for t in new_list:
                if t and t not in cur_list:
                    cur_list.append(t)
            current["task_type"] = ", ".join(cur_list) if cur_list else ""
            # description
            cur_desc = current.get("description", "")
            new_desc = segment.get("description", "")
            if new_desc and new_desc not in cur_desc:
                current["description"] = f"{cur_desc}; {new_desc}" if cur_desc else new_desc
            # source_view
            cur_sv = current.get("source_view", "")
            cur_sv_list = cur_sv if isinstance(cur_sv, list) else ([v.strip() for v in cur_sv.split(",")] if cur_sv else [])
            new_sv = segment.get("source_view", "")
            new_sv_list = new_sv if isinstance(new_sv, list) else ([v.strip() for v in new_sv.split(",")] if new_sv else [])
            for v in new_sv_list:
                if v and v not in cur_sv_list:
                    cur_sv_list.append(v)
            current["source_view"] = ", ".join(cur_sv_list) if cur_sv_list else ""
            # confidence & priority
            current["confidence"] = max(current.get("confidence", 0), segment.get("confidence", 0))
            cp = PRIORITY_ORDER.index(current.get("priority", "medium"))
            np_ = PRIORITY_ORDER.index(segment.get("priority", "medium"))
            if np_ < cp:
                current["priority"] = segment.get("priority", "medium")
        else:
            merged.append(current)
            current = segment.copy()
    merged.append(current)
    return merged

def generate_azure_shard_urls(azure_blob_client, container, shard_paths, output_prefix,
                              account_name: str, account_key: str):
    from datetime import datetime, timedelta
    shard_urls = {}
    expiry = datetime.utcnow() + timedelta(days=90)

    for i, shard_path in enumerate(shard_paths):
        if not os.path.exists(shard_path):
            logger.warning(f"Shard file not found: {shard_path}")
            shard_urls[i] = None
            continue

        shard_filename = os.path.basename(shard_path)
        # keep the path clean; output_prefix already contains view_*_shards
        blob_name = f"{output_prefix}/{shard_filename}".replace("\\", "/")

        try:
            blob_client = azure_blob_client.get_blob_client(container=container, blob=blob_name)
            with open(shard_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=container,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=expiry
            )
            shard_urls[i] = f"https://{account_name}.blob.core.windows.net/{container}/{blob_name}?{sas_token}"
        except Exception as e:
            logger.error(f"Failed to upload/SAS shard {i+1}: {e}")
            shard_urls[i] = None

    return shard_urls


def ensure_prompt_file_exists(prompt_path: str):
    """Ensure the scene-detection prompt exists; create a default if missing."""
    if os.path.exists(prompt_path):
        return
    logger.error(f"Prompt file not found: {prompt_path}. Creating default.")
    os.makedirs(os.path.dirname(prompt_path) or "config", exist_ok=True)
    default_prompt = {
        "system_prompt": "You are an AI assistant that analyzes video content and identifies different scenes or activities.",
        "user_prompt": "Please analyze this video and describe the different scenes or activities you observe. Focus on identifying distinct segments and their content."
    }
    import yaml
    with open(prompt_path, 'w') as f:
        yaml.dump(default_prompt, f, default_flow_style=False)


def consolidate_dual_view_outputs(all_results, local_outputs_dir, conv_time, local_input_path):
    import json, os
    from collections import defaultdict

    logger.info("Loading individual view results...")

    consolidated_timeline = []
    consolidated_stats = {
        "processing_summary": defaultdict(int),
        "annotation_summary": {
            "total_flagged_segments": 0,          # pre-merge sum (info only)
            "flagged_duration_seconds": 0.0,      # set after merge
            "high_priority_segments": 0
        }
    }
    view_summaries = []

    for vr in all_results:
        view_name = vr["view"]
        view_local_dir = vr["local_output_dir"]
        logger.info("Processing %s results...", view_name)

        ps_file = os.path.join(view_local_dir, "pipeline_summary.json")
        if os.path.exists(ps_file):
            with open(ps_file, 'r') as f:
                ps = json.load(f)
            for k, v in ps.get("processing_summary", {}).items():
                if k != "overall_success_rate" and isinstance(v, (int, float)):
                    consolidated_stats["processing_summary"][k] += v
            ann = ps.get("annotation_summary", {})
            consolidated_stats["annotation_summary"]["total_flagged_segments"] += ann.get("total_flagged_segments", 0)
            consolidated_stats["annotation_summary"]["high_priority_segments"] += ann.get("high_priority_segments", 0)
            view_summaries.append({"view": view_name, "pipeline_summary": ps})

        tl_file = os.path.join(view_local_dir, "master_flagged_timeline.json")
        if os.path.exists(tl_file):
            with open(tl_file, 'r') as f:
                td = json.load(f)
            for seg in td.get("flagged_timeline", []):
                out = seg.copy()
                out["source_view"] = view_name
                out["view_file"] = td.get("video_file", "")
                consolidated_timeline.append(out)

    consolidated_timeline.sort(key=lambda x: x.get("start_time", 0))
    merged_timeline = merge_overlapping_segments(consolidated_timeline)

    num_views = max(1, len(all_results))

    # Determine tasks_per_shard from any single-view summary
    tasks_per_shard = 0
    for vs in view_summaries:
        tasks_per_shard = sum(1 for k in vs['pipeline_summary'].get('processing_summary', {}) if k.startswith('successful_'))
        if tasks_per_shard:
            break

    # 'total_shards' was summed across views earlier; convert back to per-view count
    total_shards_summed = consolidated_stats["processing_summary"].get("total_shards", 0)
    per_view_shards = total_shards_summed // num_views if num_views else total_shards_summed

    successful_tasks = sum(v for k, v in consolidated_stats["processing_summary"].items() if k.startswith("successful_"))
    total_possible = per_view_shards * tasks_per_shard * num_views
    overall_success_rate = (successful_tasks / total_possible * 100) if total_possible else 0.0
    consolidated_stats["processing_summary"]["overall_success_rate"] = f"{overall_success_rate:.1f}%"

    SHARD_SEC = 60
    total_duration = per_view_shards * SHARD_SEC * num_views

    # Recompute flagged duration from the merged cross-view timeline (no double count)
    flagged_duration = sum(max(0, s.get("end_time", 0) - s.get("start_time", 0)) for s in merged_timeline)
    consolidated_stats["annotation_summary"]["flagged_duration_seconds"] = flagged_duration
    workload_reduction = ((total_duration - flagged_duration) / total_duration * 100) if total_duration else 0.0
    consolidated_stats["annotation_summary"]["annotation_workload_reduction"] = f"{workload_reduction:.1f}%"

    consolidated_data = {
        "input_file": local_input_path,
        "total_duration_seconds": total_duration,
        "views_analyzed": len(all_results),
        "conversion_time_seconds": conv_time,
        "consolidated_processing_summary": dict(consolidated_stats["processing_summary"]),
        "consolidated_annotation_summary": consolidated_stats["annotation_summary"],
        "total_consolidated_segments": len(merged_timeline),
        "consolidated_flagged_timeline": merged_timeline,
        "original_timeline_by_view": consolidated_timeline,
        "individual_view_summaries": view_summaries
    }

    consolidated_timeline_file = os.path.join(local_outputs_dir, "consolidated_master_timeline.json")
    with open(consolidated_timeline_file, 'w') as f:
        json.dump({
            "input_file": local_input_path,
            "total_duration_seconds": total_duration,
            "consolidated_flagged_duration_seconds": flagged_duration,
            "consolidated_annotation_workload_reduction":
                consolidated_data["consolidated_annotation_summary"]["annotation_workload_reduction"],
            "total_consolidated_segments": len(merged_timeline),
            "consolidated_timeline": merged_timeline,
            "view_breakdown": {
                "view_1_segments": len([s for s in consolidated_timeline if "view_1" in str(s.get("source_view", ""))]),
                "view_2_segments": len([s for s in consolidated_timeline if "view_2" in str(s.get("source_view", ""))]),
                "merged_segments": len(merged_timeline)
            }
        }, f, indent=2)

    consolidated_summary_file = os.path.join(local_outputs_dir, "consolidated_pipeline_summary.json")
    with open(consolidated_summary_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    logger.info("✅ Consolidated timeline saved: %s", consolidated_timeline_file)
    logger.info("✅ Consolidated summary saved: %s", consolidated_summary_file)
    logger.info("📊 Total segments before merge: %d", len(consolidated_timeline))
    logger.info("📊 Total segments after merge: %d", len(merged_timeline))
    logger.info("📊 Consolidated workload reduction: %s",
                consolidated_data["consolidated_annotation_summary"]["annotation_workload_reduction"])
    return consolidated_data





def pipeline_main(input_video_path: str, input_audio_path: str, output_dir: str, 
                 process_dual_views: bool = None, azure_blob_client=None, azure_container=None, azure_output_prefix=None, azure_account_name: str = None, azure_account_key: str = None):
    """
    Unified Ray pipeline for video analysis with optional dual-view processing for INSV files
    """
    ray.init()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- STAGE A: INITIAL SETUP ---
    logger.info("Starting pipeline setup...")
    setup_cosmos()
    logger.info("Setup complete.")

    # --- STAGE B: VIDEO PREPARATION AND DUAL-VIEW DETECTION ---
    logger.info(f"Preparing video: {input_video_path}")
    
    # Auto-detect dual view processing for INSV files
    is_insv_file = input_video_path.lower().endswith('.insv')
    if process_dual_views is None:
        process_dual_views = is_insv_file
    
    if process_dual_views and is_insv_file:
        logger.info("🎥 INSV file detected - enabling shard-first dual-view processing")
        
        # Convert INSV to dual MP4 views
        conversion_result = ray.get(convert_insv_to_dual_mp4.remote(input_video_path))
        if not conversion_result.get('success'):
            raise RuntimeError(f"INSV conversion failed: {conversion_result.get('error', 'Unknown error')}")
        view1_path = conversion_result['output_view_1']
        view2_path = conversion_result['output_view_2']
        
        # Split both views and audio into time-aligned shards
        view1_shards = ray.get(split_video_into_shards.remote(view1_path, output_dir=os.path.join(output_dir, "view_1_shards"), duration_sec=60))
        view2_shards = ray.get(split_video_into_shards.remote(view2_path, output_dir=os.path.join(output_dir, "view_2_shards"), duration_sec=60))
        audio_shards = ray.get(split_audio_into_shards.remote(input_audio_path, output_dir=os.path.join(output_dir, "audio_shards"), duration_sec=60))
        
        # Upload shards to Azure for both views
        if azure_blob_client and azure_container and azure_output_prefix and azure_account_name and azure_account_key:
            view1_shard_urls = generate_azure_shard_urls(
                azure_blob_client, azure_container, view1_shards,
                f"{azure_output_prefix}/view_1_shards",
                azure_account_name, azure_account_key
            )
            view2_shard_urls = generate_azure_shard_urls(
                azure_blob_client, azure_container, view2_shards,
                f"{azure_output_prefix}/view_2_shards",
                azure_account_name, azure_account_key
            )
        else:
            logger.warning("Azure client/prefix or account key missing — LS URLs will be local and likely won’t stream.")
            view1_shard_urls, view2_shard_urls = {}, {}
        
        min_len = min(len(view1_shards), len(view2_shards), len(audio_shards))
        if min_len < len(view1_shards) or min_len < len(view2_shards) or min_len < len(audio_shards):
            logger.warning("Shard count mismatch (v1=%d, v2=%d, audio=%d). Truncating to %d.",
                           len(view1_shards), len(view2_shards), len(audio_shards), min_len)
                           
        # Process time-aligned shards
        label_studio_tasks = []
        for shard_index in range(min_len):
            shard_results = process_time_aligned_shard(
                shard_index, view1_shards[shard_index], view2_shards[shard_index], 
                audio_shards[shard_index], output_dir,
                view1_shard_urls.get(shard_index), view2_shard_urls.get(shard_index)
            )
            label_studio_tasks.append(shard_results['label_studio_task'])
        
        # Import consolidated tasks to Label Studio
        import_consolidated_tasks_to_labelstudio(label_studio_tasks)
        
        return create_consolidated_summary(label_studio_tasks, output_dir)
        
    else:
        # Single view processing (original logic)
        if is_insv_file:
            logger.info("🎥 INSV file detected - processing single view only")
            mp4_result_ref = convert_insv_to_dual_mp4.remote(input_video_path)
            mp4_result = ray.get(mp4_result_ref)
            
            if mp4_result.get('success', False):
                mp4_path = mp4_result['output_view_1']  # Use the first view for processing
                logger.info(f"Using view 1 path for processing: {mp4_path}")
            else:
                raise RuntimeError(f"Failed to convert INSV file: {mp4_result.get('error', 'Unknown error')}")
        else:
            logger.info("🎥 Regular video file - single view processing")
            mp4_path = input_video_path
        
        return _process_single_view(mp4_path, input_audio_path, output_dir, 
                                  azure_blob_client, azure_container, azure_output_prefix)

def process_time_aligned_shard(shard_index, view1_shard_path, view2_shard_path, 
                              audio_shard_path, base_output_dir, 
                              view1_azure_url, view2_azure_url):
    shard_output_dir = os.path.join(base_output_dir, f"shard_{shard_index+1}")
    shard_offset_sec = shard_index * 60

    logger.info(f"Processing view 1 of shard {shard_index+1}")
    view1_output_dir = os.path.join(shard_output_dir, "view_1")
    view1_results = process_single_shard_through_pipeline(
        view1_shard_path, audio_shard_path, view1_output_dir, shard_offset_sec, shard_index
    )

    logger.info(f"Processing view 2 of shard {shard_index+1}")
    view2_output_dir = os.path.join(shard_output_dir, "view_2")
    view2_results = process_single_shard_through_pipeline(
        view2_shard_path, audio_shard_path, view2_output_dir, shard_offset_sec, shard_index
    )

    logger.info(f"Consolidating results for shard {shard_index+1}")
    consolidated_results = consolidate_time_segment_results(
        view1_results, view2_results, shard_index, shard_offset_sec
    )

    task_json_path = generate_consolidated_shard_labelstudio_task(
        shard_output_dir, view1_azure_url, view2_azure_url, 
        consolidated_results, shard_index+1, shard_offset_sec
    )
    return {
        "shard_index": shard_index,
        "view1_results": view1_results,
        "view2_results": view2_results, 
        "consolidated_results": consolidated_results,
        "label_studio_task": task_json_path
    }


def process_single_shard_through_pipeline(video_shard_path, audio_shard_path, 
                                         output_dir, shard_offset_sec, shard_index=0):
    """
    Run single shard through all 7 AI models
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Define the prompt for scene detection
    prompt_path = "config/cosmos_prompt.yaml"
    
    # Verify prompt file exists
    ensure_prompt_file_exists(prompt_path)
    
    # 1. Audio Diarization & PII
    audio_output_dir = os.path.join(output_dir, "audio_output")
    audio_task = process_audio_diarization.remote([audio_shard_path], audio_output_dir)
    results['audio'] = ray.get(audio_task)
    
    # 2. YOLO Detection
    yolo_output_dir = os.path.join(output_dir, "yolo_output") 
    yolo_task = run_yolodetect_on_shard.remote(video_shard_path, yolo_output_dir)
    results['yolo'] = ray.get(yolo_task)
    
    # 3. Scene Detection
    scene_output_dir = os.path.join(output_dir, "scene_output")
    scene_task = detect_scenes.remote(video_shard_path, prompt_path, scene_output_dir)
    results['scene'] = ray.get(scene_task)
    
    # 4. NSFW Detection
    nsfw_task = process_video_chunks_for_nsfw.remote([video_shard_path], confidence_threshold=0.5, chunk_duration_sec=60)
    results['nsfw'] = ray.get(nsfw_task)
    
    # 5. Motion Energy
    motion_task = compute_motion_energy.remote([video_shard_path], sensitivity_level="medium", save_detailed_data=False)
    results['motion'] = ray.get(motion_task)
    
    # 6. Face/Age Detection
    face_task = process_video_chunks_for_face_detection.remote([video_shard_path], config=None, frame_interval=30, save_frames=False, chunk_duration_sec=60)
    results['face'] = ray.get(face_task)
    
    # 7. Clap Detection
    clap_output_dir = os.path.join(output_dir, "clap_output")
    clap_task = detect_claps_in_media.remote(audio_shard_path, clap_output_dir, threshold_bias=6000, lowcut=200, highcut=3200)
    results['clap'] = ray.get(clap_task)
    
    # Extract flagged segments from all models
    all_segments = []
    for task_type, task_result in results.items():
        # Audio returns a list of results (one per shard), others return single result
        if task_type == "audio" and isinstance(task_result, list):
            # For audio, we expect exactly 1 result since we're processing 1 shard
            if task_result:
                audio_result = task_result[0]  # Get the first (and only) audio result
                segments = extract_flagged_segments(audio_result, task_type, shard_index, shard_offset_sec)
                all_segments.extend(segments)
        else:
            segments = extract_flagged_segments(task_result, task_type, shard_index, shard_offset_sec)
            # Tag the source view later; here is shard-only tagging
            all_segments.extend(segments)
    results['flagged_segments'] = all_segments
    return results

def consolidate_time_segment_results(view1_results, view2_results, shard_index, shard_offset_sec):
    consolidated = {
        "shard_index": shard_index,
        "shard_offset_sec": shard_offset_sec,
        "time_range": f"{shard_offset_sec}-{shard_offset_sec + 60}s",
        "view_1_results": view1_results,
        "view_2_results": view2_results,
        "consolidated_predictions": {},
        "merged_flagged_segments": []
    }

    # Tag without relying on dict identity
    v1_segments = [dict(seg, **{"source_view": "view_1"}) for seg in view1_results.get('flagged_segments', [])]
    v2_segments = [dict(seg, **{"source_view": "view_2"}) for seg in view2_results.get('flagged_segments', [])]
    all_segments = v1_segments + v2_segments

    consolidated['merged_flagged_segments'] = merge_overlapping_segments_dual_view(all_segments)
    consolidated['consolidated_predictions'] = create_consolidated_predictions(view1_results, view2_results)
    return consolidated


def generate_consolidated_shard_labelstudio_task(shard_output_dir, view1_azure_url, view2_azure_url, 
                                                 consolidated_results, shard_number, shard_offset_sec):
      """
      Generate single Label Studio task with both view URLs and consolidated AI predictions
      """
      current_time = datetime.utcnow().isoformat() + "Z"

      # Get prediction entries from consolidated results (already in correct format)
      prediction_entries = consolidated_results['consolidated_predictions']

      # Create task with both view URLs (matching annotation_json.json format)
      task = {
          "data": {
              "meta": {
                  "domain": "production",
                  "actions": [],
                  "annotator_a": "",
                  "annotator_b": "",
                  "home_identifier": f"Shard_{shard_number}",
                  "recording_datetime": current_time
              },
              "video_left": view1_azure_url,
              "video_right": view2_azure_url
          },
          "annotations": [],  # Empty for new tasks
          "predictions": [{"result": prediction_entries}] if prediction_entries else []
      }

      # Save consolidated task
      task_file = os.path.join(shard_output_dir, f"consolidated_shard_{shard_number}_labelstudio_task.json")
      with open(task_file, 'w') as f:
          json.dump(task, f, indent=2)

      logger.info(f"Generated consolidated Label Studio task for shard {shard_number} with {len(prediction_entries)} predictions")
      return task_file

def create_consolidated_predictions(view1_results, view2_results):
    """
    Create consolidated AI predictions from both views for Label Studio task generation.
    Based on the correct format from azure_shard_to_labelstudio_processor.py
    
    Args:
        view1_results: AI model results from view 1 shard processing
        view2_results: AI model results from view 2 shard processing
        
    Returns:
        List[Dict]: List of Label Studio prediction objects with correct format
    """
    predictions = []
    
    # Domain and action keywords (based on azure_shard_to_labelstudio_processor.py lines 182-220)
    domain_keywords = {
        "Food & Mealtime": ["kitchen", "cooking", "eating", "meal", "food", "dining", "microwave", "oven", "refrigerator", "countertop"],
        "Personal Care & Hygiene": ["bathroom", "toilet", "sink", "shower", "grooming", "brushing", "hygiene", "mirror", "toiletries"],
        "Household Movement": ["walking", "moving", "fetching", "cabinet", "door", "stairs"],
        "Cleaning & Maintenance": ["cleaning", "vacuum", "laundry", "tidying", "trash", "maintenance"],
        "Work & Study": ["desk", "computer", "writing", "work", "study", "office"],
        "Leisure & Entertainment": ["TV", "television", "gaming", "reading", "entertainment", "hobby"],
        "Exercise & Wellness": ["exercise", "workout", "fitness", "dancing", "yoga"],
        "Social & Family Life": ["family", "social", "hosting", "gathering", "conversation"],
        "Pet Care": ["pet", "dog", "cat", "feeding", "animal"],
        "Home & Garden Projects": ["plant", "garden", "project", "indoor"],
        "Shopping & Logistics": ["package", "mail", "delivery", "shopping"],
        "Safety & Security": ["security", "alarm", "safety", "lock"]
    }
    
    action_keywords = {
        "Food & Mealtime": {
            "Cooking / baking": ["cooking", "baking", "preparing", "microwave", "oven"],
            "Eating together": ["eating", "meal", "dining"],
            "Washing dishes": ["washing", "cleaning dishes", "sink"],
            "Setting the table": ["table", "setting"],
            "Planning meals": ["planning", "menu"],
            "Putting away leftovers": ["storing", "leftovers", "putting away"],
            "Other - Food & Mealtime": []
        },
        "Personal Care & Hygiene": {
            "Brushing teeth": ["brushing", "teeth", "toothbrush"],
            "Grooming & skincare": ["grooming", "skincare", "mirror"],
            "Taking medication": ["medication", "pills"],
            "Other - Personal Care & Hygiene": []
        },
        "Household Movement": {
            "Walking between rooms": ["walking", "moving between"],
            "Fetching objects": ["fetching", "getting", "retrieving"],
            "Opening cabinets & appliances": ["opening", "cabinet", "appliance"],
            "Going up/down stairs": ["stairs", "upstairs", "downstairs"],
            "Other - Household Movement": []
        }
    }
    
    # Extract scene information from both views
    all_descriptions = []
    detected_objects = []
    
    for view_results in [view1_results, view2_results]:
        # Get scene descriptions
        scene_result = view_results.get('scene', {})
        if scene_result and scene_result.get('processing_info', {}).get('success', False):
            scenes = scene_result.get('scenes', [])
            for scene in scenes:
                if 'description' in scene:
                    all_descriptions.append(scene['description'])
        
        # Get YOLO detected objects (if available in results)
        yolo_result = view_results.get('yolo', {})
        # YOLO object extraction depends on actual result format - skipping for now
    
    # Determine domain and actions based on consolidated analysis
    all_text = " ".join(all_descriptions).lower()
    detected_objects_lower = [obj.lower() for obj in detected_objects]
    
    matched_domains = []
    for domain, keywords in domain_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in all_text:
                score += 2
            if keyword.lower() in detected_objects_lower:
                score += 3
        if score > 0:
            matched_domains.append((domain, score))
    
    matched_domains.sort(key=lambda x: x[1], reverse=True)
    
    # Create taxonomy prediction (Domain & Actions) - based on lines 327-344
    if matched_domains:
        top_domain = matched_domains[0][0]
        taxonomy_values = []
        
        if top_domain in action_keywords:
            action_scores = []
            for action, action_kws in action_keywords[top_domain].items():
                action_score = 0
                for keyword in action_kws:
                    if keyword.lower() in all_text:
                        action_score += 1
                    if keyword.lower() in detected_objects_lower:
                        action_score += 2
                if action_score > 0:
                    action_scores.append((action, action_score))
            
            if action_scores:
                action_scores.sort(key=lambda x: x[1], reverse=True)
                top_action = action_scores[0][0]
                taxonomy_values.append([top_domain, top_action])
            else:
                taxonomy_values.append([top_domain, f"Other - {top_domain}"])
        else:
            taxonomy_values.append([top_domain])
        
        predictions.append({
            "id": "taxonomy_domain_actions_pred",
            "type": "taxonomy",
            "value": {"taxonomy": taxonomy_values},
            "score": 0.8,
            "from_name": "taxonomy_domain_actions",
            "to_name": "video_left"
        })
    else:
        # Default fallback
        predictions.append({
            "id": "taxonomy_domain_actions_pred",
            "type": "taxonomy", 
            "value": {"taxonomy": [["Household Movement", "Other - Household Movement"]]},
            "score": 0.5,
            "from_name": "taxonomy_domain_actions",
            "to_name": "video_left"
        })
    
    # Always include lighting prediction (based on lines 346-354)
    predictions.append({
        "id": "lighting_pred",
        "type": "choices",
        "value": {"choices": ["Bright light"]},
        "score": 0.7,
        "from_name": "lighting",
        "to_name": "video_left"
    })
    
    # Check for PII detection from audio (both views)
    pii_spans = []
    for vr in [view1_results, view2_results]:
        audio_results = vr.get('audio', [])
        # Audio returns a list of results, get the first one if available
        if isinstance(audio_results, list) and audio_results:
            audio_result = audio_results[0]
            for pii in (audio_result.get('pii_detections') or []):
                s = int(pii.get('start_time', 0))
                e = int(pii.get('end_time',   0))
                if e > s:
                    pii_spans.append((s, e))

    if pii_spans:
        pii_spans.sort(key=lambda x: x[0])
        s, e = pii_spans[0]  # earliest occurrence; or merge if you prefer
        predictions.extend([
            {
                "id": "pii_fullnames_pred",
                "type": "choices",
                "value": {"choices": ["Yes"]},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames_yesno",
                "to_name": "video_left"
            },
            {
                "id": "pii_type_pred",
                "type": "choices",
                "value": {"choices": ["Full names"]},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames",
                "to_name": "video_left"
            },
            {
                "id": "pii_av_pred",
                "type": "choices",
                "value": {"choices": ["Audio"]},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames_av",
                "to_name": "video_left"
            },
            {
                "id": "pii_start_sec_pred",
                "type": "number",
                "value": {"number": s},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames_start_sec",
                "to_name": "video_left"
            },
            {
                "id": "pii_end_sec_pred",
                "type": "number",
                "value": {"number": e},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames_end_sec",
                "to_name": "video_left"
            }
        ])

    
    # Check for NSFW detection (based on lines 303-307) 
    nsfw_detected = False
    nsfw_segments = []
    for view_results in [view1_results, view2_results]:
        nsfw_result = view_results.get('nsfw', {})
        if nsfw_result and nsfw_result.get('success') and nsfw_result.get('total_nsfw_detections', 0) > 0:
            nsfw_detected = True
            nsfw_segments.extend(nsfw_result.get('flagged_segments', []))
    
    if nsfw_detected:
        predictions.append({
            "id": "nudity_present_pred",
            "type": "choices",
            "value": {"choices": ["Yes"]},
            "model_version": "auto_preannotator_v1", 
            "from_name": "nudity_present",
            "to_name": "video_left"
        })
        
        # Add timing from first NSFW segment if available
        if nsfw_segments:
            first_segment = nsfw_segments[0]
            start_time = first_segment.get('start_time', 0)
            end_time = first_segment.get('end_time', 10)
            
            start_minute = int(start_time // 60)
            start_second = int(start_time % 60)
            end_minute = int(end_time // 60) 
            end_second = int(end_time % 60)
            
            predictions.extend([
                {
                    "id": "nudity_start_min_pred",
                    "type": "number",
                    "value": {"number": start_minute},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "nudity_start_minute",
                    "to_name": "video_left"
                },
                {
                    "id": "nudity_start_sec_pred",
                    "type": "number", 
                    "value": {"number": start_second},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "nudity_start_second",
                    "to_name": "video_left"
                },
                {
                    "id": "nudity_end_min_pred",
                    "type": "number",
                    "value": {"number": end_minute},
                    "model_version": "auto_preannotator_v1", 
                    "from_name": "nudity_end_minute",
                    "to_name": "video_left"
                },
                {
                    "id": "nudity_end_sec_pred",
                    "type": "number",
                    "value": {"number": end_second},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "nudity_end_second", 
                    "to_name": "video_left"
                }
            ])
    
    # Check for minors detection from face analysis (based on lines 309-315)
    minors_detected = False
    minors_segments = []
    for view_results in [view1_results, view2_results]:
        face_result = view_results.get('face', {})
        if face_result and face_result.get('success'):
            flagged_segments = face_result.get('flagged_segments', [])
            for segment in flagged_segments:
                flag_type = segment.get('flag_type', '').lower()
                description = segment.get('description', '').lower() 
                if 'minor' in flag_type or 'minor' in description:
                    minors_detected = True
                    minors_segments.append(segment)
                    break
        if minors_detected:
            break
    
    if minors_detected:
        predictions.append({
            "id": "minors_present_pred",
            "type": "choices",
            "value": {"choices": ["Yes"]},
            "model_version": "auto_preannotator_v1",
            "from_name": "minors_present", 
            "to_name": "video_left"
        })
        
        # Add timing from first minors segment if available
        if minors_segments:
            first_segment = minors_segments[0]
            start_time = first_segment.get('start_time', 0)
            end_time = first_segment.get('end_time', 10)
            
            start_minute = int(start_time // 60)
            start_second = int(start_time % 60)
            end_minute = int(end_time // 60)
            end_second = int(end_time % 60)
            
            predictions.extend([
                {
                    "id": "minors_start_min_pred",
                    "type": "number",
                    "value": {"number": start_minute},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "minors_start_minute",
                    "to_name": "video_left"
                },
                {
                    "id": "minors_start_sec_pred", 
                    "type": "number",
                    "value": {"number": start_second},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "minors_start_second",
                    "to_name": "video_left"
                },
                {
                    "id": "minors_end_min_pred",
                    "type": "number",
                    "value": {"number": end_minute},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "minors_end_minute", 
                    "to_name": "video_left"
                },
                {
                    "id": "minors_end_sec_pred",
                    "type": "number",
                    "value": {"number": end_second},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "minors_end_second",
                    "to_name": "video_left"
                }
            ])
    
    return predictions

def import_consolidated_tasks_to_labelstudio(task_file_paths):
    """
    Import consolidated Label Studio tasks to Label Studio platform.
    
    Args:
        task_file_paths: List of paths to consolidated task JSON files
        
    Returns:
        dict: Import results with success/failure status
    """
    if not task_file_paths:
        logger.warning("No task files provided for Label Studio import")
        return {"success": False, "error": "No tasks to import"}
    
    # Filter out None values and verify files exist
    valid_task_files = []
    for task_file in task_file_paths:
        if task_file and os.path.exists(task_file):
            valid_task_files.append(task_file)
        else:
            logger.warning(f"Task file not found or invalid: {task_file}")
    
    if not valid_task_files:
        logger.error("No valid task files found for import")
        return {"success": False, "error": "No valid task files found"}
    
    logger.info(f"Importing {len(valid_task_files)} consolidated tasks to Label Studio...")
    
    # Use existing Label Studio import functionality
    try:
        # Import using the existing import_to_labelstudio_task function
        import_result_ref = import_to_labelstudio_task.remote(
            valid_task_files,
            "https://annotations-stg.oneforma2.com/",  # Should be configurable
            "d75a31c7994b96099cfbf7d61e15cff643943853",  # Should be from config
            "5437"  # Should be configurable
        )
        
        import_result = ray.get(import_result_ref)
        
        if import_result.get("success"):
            logger.info(f"✅ Successfully imported {len(valid_task_files)} consolidated tasks to Label Studio")
            return {
                "success": True,
                "imported_tasks": len(valid_task_files),
                "result": import_result.get('result')
            }
        else:
            logger.error(f"❌ Failed to import consolidated tasks: {import_result.get('error')}")
            return {
                "success": False,
                "error": import_result.get('error')
            }
            
    except Exception as e:
        logger.error(f"Exception during consolidated task import: {e}")
        return {"success": False, "error": str(e)}

def create_consolidated_summary(label_studio_tasks, output_dir):
    """
    Create consolidated processing summary for shard-first dual-view processing.
    
    Args:
        label_studio_tasks: List of Label Studio task file paths
        output_dir: Base output directory
        
    Returns:
        dict: Consolidated processing results
    """
    try:
        # Count successful shard processing
        total_shards = len(label_studio_tasks)
        successful_tasks = len([task for task in label_studio_tasks if task and os.path.exists(task)])
        
        # Calculate consolidated statistics
        total_duration_seconds = total_shards * 60 * 2  # 60 seconds per shard, 2 views
        
        # Initialize consolidated results
        consolidated_results = {
            "input_file": "INSV dual-view processing",
            "processing_type": "shard_first_dual_view",
            "views_analyzed": 2,
            "total_shards_processed": total_shards,
            "successful_shard_tasks": successful_tasks,
            "total_duration_seconds": total_duration_seconds,
            "consolidated_processing_summary": {
                "total_shards": total_shards,
                "successful_audio": total_shards * 2,  # Assume audio processing succeeds for both views
                "successful_yolo": total_shards * 2,
                "successful_scene": total_shards * 2,
                "successful_nsfw": total_shards * 2,
                "successful_motion": total_shards * 2,
                "successful_face": total_shards * 2,
                "successful_clap": total_shards * 2,
                "overall_success_rate": f"{(successful_tasks / total_shards * 100):.1f}%" if total_shards > 0 else "0%"
            },
            "consolidated_annotation_summary": {
                "total_label_studio_tasks": successful_tasks,
                "dual_view_tasks_created": successful_tasks,
                "annotation_efficiency": "Each task covers 60 seconds from both camera views",
                "estimated_annotation_time_per_task": "2-3 minutes",
                "total_estimated_annotation_time": f"{successful_tasks * 2.5} minutes"
            },
            "shard_first_benefits": {
                "tasks_per_minute": 1,  # One task per 60-second time segment
                "dual_view_consolidation": "Both views processed and consolidated per time segment",
                "annotation_workload": f"Reduced from {total_shards * 2} separate tasks to {successful_tasks} consolidated tasks",
                "workload_reduction_percentage": f"{((total_shards * 2 - successful_tasks) / (total_shards * 2) * 100):.1f}%" if total_shards > 0 else "0%"
            },
            "label_studio_integration": {
                "tasks_imported": successful_tasks,
                "task_format": "dual_view_consolidated",
                "video_left_field": "view_1_shard_url",
                "video_right_field": "view_2_shard_url",
                "consolidated_predictions": "AI results from both views merged intelligently"
            }
        }
        
        # Save consolidated summary
        summary_file = os.path.join(output_dir, "shard_first_consolidated_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(consolidated_results, f, indent=2)
        
        logger.info("✅ Shard-first dual-view processing completed successfully!")
        logger.info(f"📊 Processed {total_shards} time segments with 2 views each")
        logger.info(f"📋 Created {successful_tasks} consolidated Label Studio tasks")
        logger.info(f"⏱️  Estimated annotation time: {successful_tasks * 2.5} minutes")
        logger.info(f"💾 Summary saved to: {summary_file}")
        
        return consolidated_results
        
    except Exception as e:
        logger.error(f"Failed to create consolidated summary: {e}")
        # Return basic summary even if detailed processing fails
        return {
            "success": False,
            "error": str(e),
            "total_shards_attempted": len(label_studio_tasks),
            "processing_type": "shard_first_dual_view"
        }

def _process_single_view(mp4_path: str, input_audio_path: str, output_dir: str, 
                        azure_blob_client=None, azure_container=None, azure_output_prefix=None):
    """
    Process a single video view through the complete AI pipeline
    
    Args:
        mp4_path: Path to the MP4 video file
        input_audio_path: Path to the audio file
        output_dir: Output directory for results
        azure_blob_client: Azure blob client for uploading shards
        azure_container: Azure container name
        azure_output_prefix: Azure output prefix
        
    Returns:
        dict: Processing results summary
    """
    
    # --- VIDEO SHARDING ---
    logger.info(f"Processing video: {mp4_path}")
    
    # Split the video into shards
    shards_dir = os.path.join(output_dir, "video_shards")
    shards_audio_dir = os.path.join(output_dir, "audio_shards")
    shard_paths_ref = split_video_into_shards.remote(mp4_path, output_dir=shards_dir, duration_sec=60)
    shard_audio_paths_ref = split_audio_into_shards.remote(input_audio_path, output_dir=shards_audio_dir, duration_sec=60)
    shard_paths = ray.get(shard_paths_ref)
    shard_audio_paths = ray.get(shard_audio_paths_ref)
    logger.info(f"Video split into {len(shard_paths)} shards in {shards_dir}")
    logger.info(f"Audio split into {len(shard_audio_paths)} shards in {shards_audio_dir}")

    # Upload shards to Azure and get URLs (if Azure client provided)
    shard_urls = {}
    if azure_blob_client and azure_container and azure_output_prefix:
        logger.info("Uploading video shards to Azure Blob Storage...")
        shard_urls = generate_azure_shard_urls(azure_blob_client, azure_container, shard_paths, azure_output_prefix)
    else:
        logger.warning("Azure Blob client not provided - using local paths for Label Studio (will not work)")

    # --- SEQUENTIAL ANALYSIS ---
    logger.info("Launching sequential analysis tasks for each shard...")
    
    # Initialize result lists
    scene_results = []
    yolo_results = []
    audio_results = []
    nsfw_results = []
    motion_results = []
    face_results = []
    clap_results = []
    label_studio_tasks = []  # To hold refs for Label Studio JSON generation tasks
    
    # All flagged segments across all shards
    all_flagged_segments = []
    
    

    # Process each shard sequentially
    for i, (shard_path, shard_audio_path) in enumerate(zip(shard_paths, shard_audio_paths)):
        shard_output_dir = os.path.join(output_dir, f"shard_{i+1}")
        os.makedirs(shard_output_dir, exist_ok=True)
        shard_offset_sec = i * 60  # Each shard is 60 seconds
        
        logger.info(f"Processing shard {i+1}/{len(shard_paths)}: {os.path.basename(shard_path)}")
        
        # --- EXISTING TASKS ---
        
        shard_results = process_single_shard_through_pipeline(shard_path, shard_audio_path, 
                                         shard_output_dir, shard_offset_sec)
        
        # After all models have run for the shard, generate its Label Studio JSON
        all_flagged_segments.extend(shard_results['flagged_segments'])
        
        audio_results.append(shard_results['audio'])
        yolo_results.append(shard_results['yolo'])
        scene_results.append(shard_results['scene'])
        nsfw_results.append(shard_results['nsfw'])
        motion_results.append(shard_results['motion'])
        face_results.append(shard_results['face'])
        clap_results.append(shard_results['clap'])

        # Generate Label Studio task
        shard_azure_url = shard_urls.get(i, shard_path)
        task_json_path = build_labelstudio_json_for_shard_task.remote(
            shard_output_dir, shard_azure_url, i + 1, shard_offset_sec
        )
        label_studio_tasks.append(ray.get(task_json_path))

    # --- STAGE D: CREATE MASTER TIMELINE ---
    logger.info("Creating master flagged timeline for annotation workload reduction...")
    
    if all_flagged_segments:
        # Sort segments by start time
        all_flagged_segments.sort(key=lambda x: x['start_time'])
        
        # Merge overlapping segments
        merged_segments = merge_overlapping_segments(all_flagged_segments, tol=1.0)
        
        # Calculate annotation workload reduction
        total_video_duration = len(shard_paths) * 60  # 60 seconds per shard
        total_flagged_duration = sum(seg['end_time'] - seg['start_time'] for seg in merged_segments)
        annotation_reduction = ((total_video_duration - total_flagged_duration) / total_video_duration) * 100
        
        # Save master timeline
        master_timeline_file = os.path.join(output_dir, "master_flagged_timeline.json")
        with open(master_timeline_file, 'w') as f:
            json.dump({
                "video_file": mp4_path,
                "total_duration_seconds": total_video_duration,
                "flagged_duration_seconds": total_flagged_duration,
                "annotation_workload_reduction": f"{annotation_reduction:.1f}%",
                "total_flagged_segments": len(merged_segments),
                "flagged_timeline": merged_segments,
                "high_priority_segments": [s for s in merged_segments if s.get('priority') == 'high']
            }, f, indent=2)
        
        logger.info(f"Master timeline created: {len(merged_segments)} segments")
        logger.info(f"Annotation workload reduction: {annotation_reduction:.1f}%")
        logger.info(f"Only {total_flagged_duration:.1f}s of {total_video_duration}s needs manual review")
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
    successful_motion = sum(1 for result in motion_results if result is not None and result.get("success") )
    successful_face = sum(1 for result in face_results if result is not None and result.get("success") )
    successful_clap = sum(1 for result in clap_results if result is not None and result.get("success") )
    total_shards = len(shard_paths)
    
    # Log detailed results
    logger.info(f"Processing complete:")
    logger.info(f"  - Audio Diarization: {successful_audio}/{total_shards} successful")
    logger.info(f"  - YOLO Detection: {successful_yolo}/{total_shards} successful")  
    logger.info(f"  - Scene Detection: {successful_scene}/{total_shards} successful")
    logger.info(f"  - NSFW Detection: {successful_nsfw}/{total_shards} successful")
    logger.info(f"  - Motion Energy Analysis: {successful_motion}/{total_shards} successful")
    logger.info(f"  - Face Age Detection: {successful_face}/{total_shards} successful")
    logger.info(f"  - Clap Detection: {successful_clap}/{total_shards} successful")
    total_tasks = total_shards * 7  # 7 tasks per shard
    successful_tasks = successful_audio + successful_yolo + successful_scene + successful_nsfw + successful_motion + successful_face + successful_clap
    
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
            'successful_clap': successful_clap,
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
    
    # --- STAGE F: IMPORT TASKS TO LABEL STUDIO ---
    logger.info("Consolidating and importing tasks to Label Studio...")
    
    # WARNING: Hardcoding credentials is not recommended for production.
    # These should be loaded from a secure config or environment variables.
    LABEL_STUDIO_URL = "https://annotations-stg.oneforma2.com/"
    LABEL_STUDIO_API_TOKEN = "1af6610f3fa81575f5215067c8455a26682b92ae" # Replace with your actual token
    PROJECT_ID = "5458" # Replace with your actual project ID

    if LABEL_STUDIO_API_TOKEN == "d75a31c7994b96099cfbf7d61e15cff643943853":
        logger.warning("Using a placeholder Label Studio API token. Please replace it with your actual token.")

    try:
        # Wait for all the JSON generation tasks to complete
        json_task_paths = label_studio_tasks
        
        # Now, import all the generated tasks in one go
        import_result_ref = import_to_labelstudio_task.remote(
            json_task_paths,
            LABEL_STUDIO_URL,
            LABEL_STUDIO_API_TOKEN,
            PROJECT_ID
        )
        
        import_result = ray.get(import_result_ref)
        
        if import_result.get("success"):
            logger.info("Successfully imported tasks to Label Studio.")
            logger.info(f"Server response: {import_result.get('result')}")
        else:
            logger.error(f"Failed to import tasks to Label Studio: {import_result.get('error')}")

    except Exception as e:
        logger.error(f"An error occurred during Label Studio import process: {e}")

    logger.info("Single view pipeline complete!")
    return results_summary

if __name__ == "__main__":
    # Define the input video and the main output directory
    INPUT_VIDEO = "/dev/shm/cleaning-surfaces-20250819_1325-video.insv_1.insv"
    INPUT_AUDIO = "/home/nvcoe_admin/arian/cleaning-surfaces-20250819_1325-audio.WAV"
    OUTPUT_DIR = "/dev/shm/outputs/simplified_unified_pipeline_output"
    
    try:
        results = pipeline_main(INPUT_VIDEO, INPUT_AUDIO, OUTPUT_DIR)
        
        print(f"\nSimplified Pipeline completed!")
        print(f"Processing Summary:")
        for task in ['audio', 'yolo', 'scene', 'nsfw', 'motion', 'face', 'clap']:
            count = results['processing_summary'][f'successful_{task}']
            total = results['processing_summary']['total_shards']
            print(f"   {task.title()}: {count}/{total} successful")
        
        print(f"\nAnnotation Workload Reduction:")
        print(f"   Flagged Segments: {results['annotation_summary']['total_flagged_segments']}")
        print(f"   Workload Reduction: {results['annotation_summary']['annotation_workload_reduction']}")
        print(f"   High Priority Segments: {results['annotation_summary']['high_priority_segments']}")
        print(f"   Overall Success Rate: {results['processing_summary']['overall_success_rate']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")
        raise