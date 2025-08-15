# main_pipeline_vit_age_detection.py
# 
# This pipeline processes INSV/MP4 videos for ViT age detection with a clean output structure:
# 
# OUTPUT_DIR/
# ├── chunks/                    # Temporary video chunks (cleaned up by default)
# │   ├── view_1/               # Chunks for first view
# │   └── view_2/               # Chunks for second view (if INSV)
# ├── results/                   # Final analysis results
# │   ├── view_1_vit_age_analysis_results.json
# │   ├── view_2_vit_age_analysis_results.json
# │   └── VID_20250720_152154_00_011_vit_age_detection_summary.json
# └── chunk_results/             # Individual chunk results (for debugging)
#     ├── view_1/                # Results for each chunk in view 1
#     └── view_2/                # Results for each chunk in view 2
# 
# Use keep_chunks=True to preserve temporary chunk files for debugging.

import ray
import json
import os
from pathlib import Path

# Import existing Ray jobs
from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
from ray_jobs.vit_video_age_detector import process_video_chunks_for_vit_age_detection

from utils.logger import get_logger
logger = get_logger("vit_age_detection_pipeline")


def pipeline_main_vit_age_detection(input_path: str, output_dir: str = "/tmp/vit_age_detection_results",
                                   frame_interval: int = 30, save_frames: bool = False,
                                   chunk_duration_sec: int = 60, keep_chunks: bool = False):
    """
    Main pipeline that processes INSV/MP4 videos for ViT age detection.
    
    Args:
        input_path: Path to input video (.insv or .mp4)
        output_dir: Directory for output files
        frame_interval: Frame sampling interval for age detection
        save_frames: Whether to save frames with detections
        chunk_duration_sec: Duration of each video chunk
        keep_chunks: Whether to keep temporary chunk files (default: False)
    """
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_path)
    
    try:
        # Stage 1: Convert INSV to MP4 if needed
        if input_path.suffix.lower() == '.insv':
            logger.info(f"Converting INSV file: {input_path}")
            conversion_result = ray.get(convert_insv_to_dual_mp4.remote(str(input_path)))
            
            if not conversion_result or not conversion_result.get("success"):
                logger.error(f"INSV conversion failed: {conversion_result}")
                return None
            
            # Process both views from the INSV file
            mp4_paths = [conversion_result["output_view_1"], conversion_result["output_view_2"]]
            logger.info(f"INSV converted to: {mp4_paths}")
        else:
            # Single MP4 file
            mp4_paths = [str(input_path)]
            logger.info(f"Processing MP4 file: {input_path}")
        
        all_results = []
        
        # Process each MP4 (for INSV, this will be both views)
        for i, mp4_path in enumerate(mp4_paths):
            view_name = f"view_{i+1}" if len(mp4_paths) > 1 else "single_view"
            logger.info(f"Processing {view_name}: {mp4_path}")
            
            # Stage 2: Split video into chunks for THIS view only
            # Use consolidated output directory structure
            view_chunks_dir = Path(output_dir) / "chunks" / view_name
            os.makedirs(view_chunks_dir, exist_ok=True)
            logger.info(f"Splitting video into {chunk_duration_sec}s chunks...")
            view_chunk_paths = ray.get(split_video_into_shards.remote(mp4_path, str(view_chunks_dir), chunk_duration_sec))
            logger.info(f"Created {len(view_chunk_paths)} video chunks for {view_name}")
            
            # Stage 3: Run ViT age detection on chunks in parallel for THIS view only
            logger.info(f"Running ViT age detection on {len(view_chunk_paths)} chunks for {view_name}...")
            vit_age_detection_results = ray.get(process_video_chunks_for_vit_age_detection.remote(
                view_chunk_paths, None, frame_interval, save_frames, chunk_duration_sec
            ))
            
            if vit_age_detection_results["success"]:
                logger.info(f"ViT age detection complete for {view_name}")
                logger.info(f"Total faces detected: {vit_age_detection_results['total_faces_detected']}")
                logger.info(f"Total frames processed: {vit_age_detection_results['total_processed_frames']}")
                logger.info(f"Processing time: {vit_age_detection_results['total_processing_time_seconds']}s")
                
                # Save detailed results for this view in consolidated structure
                output_file = Path(output_dir) / "results" / f"{view_name}_vit_age_analysis_results.json"
                os.makedirs(output_file.parent, exist_ok=True)
                with open(output_file, 'w') as f:
                    json.dump(vit_age_detection_results, f, indent=2)
                
                logger.info(f"Detailed results saved to: {output_file}")
                
                # Save individual chunk results in consolidated structure
                chunk_results_dir = Path(output_dir) / "chunk_results" / view_name
                os.makedirs(chunk_results_dir, exist_ok=True)
                
                for j, chunk_path in enumerate(view_chunk_paths):
                    chunk_name = Path(chunk_path).stem
                    chunk_result_file = chunk_results_dir / f"{chunk_name}_results.json"
                    
                    # Find results for this chunk
                    chunk_results = []
                    for frame_result in vit_age_detection_results["face_analysis_results"]:
                        if frame_result.get("chunk_offset_seconds") == j * chunk_duration_sec:
                            chunk_results.append(frame_result)
                    
                    with open(chunk_result_file, 'w') as f:
                        json.dump({
                            "chunk_path": chunk_path,
                            "chunk_offset_seconds": j * chunk_duration_sec,
                            "frames_analyzed": len(chunk_results),
                            "age_detections": chunk_results
                        }, f, indent=2)
                
                all_results.append({
                    "view": view_name,
                    "mp4_path": mp4_path,
                    "results_file": str(output_file),
                    "total_faces_detected": vit_age_detection_results["total_faces_detected"],
                    "total_processed_frames": vit_age_detection_results["total_processed_frames"],
                    "processing_time": vit_age_detection_results["total_processing_time_seconds"],
                    "chunks_processed": vit_age_detection_results["chunks_processed"]
                })
            else:
                logger.error(f"ViT age detection failed for {view_name}")
                return None
        
        # Stage 4: Create summary report
        summary = {
            "input_file": str(input_path),
            "file_type": "INSV" if input_path.suffix.lower() == '.insv' else "MP4",
            "views_processed": len(mp4_paths),
            "total_faces_detected": sum(r["total_faces_detected"] for r in all_results),
            "total_processed_frames": sum(r["total_processed_frames"] for r in all_results),
            "total_processing_time": sum(r["processing_time"] for r in all_results),
            "chunks_processed": sum(r["chunks_processed"] for r in all_results),
            "frame_interval": frame_interval,
            "save_frames": save_frames,
            "chunk_duration_sec": chunk_duration_sec,
            "view_results": all_results
        }
        
        summary_file = Path(output_dir) / "results" / f"{input_path.stem}_vit_age_detection_summary.json"
        os.makedirs(summary_file.parent, exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("="*60)
        logger.info("VIT AGE DETECTION PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Input: {input_path}")
        logger.info(f"Views processed: {summary['views_processed']}")
        logger.info(f"Total faces detected: {summary['total_faces_detected']}")
        logger.info(f"Total frames processed: {summary['total_processed_frames']}")
        logger.info(f"Total processing time: {summary['total_processing_time']:.2f}s")
        logger.info(f"Chunks processed: {summary['chunks_processed']}")
        logger.info(f"Summary saved to: {summary_file}")
        
        # Clean up temporary chunk files if keep_chunks is False
        if not keep_chunks:
            logger.info("Cleaning up temporary chunk files...")
            try:
                import shutil
                chunks_dir = Path(output_dir) / "chunks"
                if chunks_dir.exists():
                    shutil.rmtree(chunks_dir)
                    logger.info("Removed temporary chunk files")
            except Exception as e:
                logger.warning(f"Could not clean up chunk files: {e}")
        
        # Validate timestamp ordering
        logger.info("Validating timestamp ordering...")
        for view_result in all_results:
            results_file = Path(view_result["results_file"])
            if results_file.exists():
                with open(results_file, 'r') as f:
                    view_data = json.load(f)
                    face_results = view_data.get("face_analysis_results", [])
                    if face_results:
                        # Check first few timestamps
                        logger.info(f"First 5 timestamps for {view_result['view']}:")
                        for i, result in enumerate(face_results[:5]):
                            logger.info(f"  {i+1}. Frame {result.get('frame_num', 'N/A')} at {result.get('timestamp', 'N/A')}")
                        
                        # Check if timestamps are in order
                        timestamps = []
                        for result in face_results:
                            if "timestamp" in result:
                                try:
                                    time_parts = result["timestamp"].split(":")
                                    hours, minutes, seconds = map(int, time_parts)
                                    total_seconds = hours * 3600 + minutes * 60 + seconds
                                    timestamps.append(total_seconds)
                                except:
                                    pass
                        
                        if timestamps:
                            is_ordered = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
                            logger.info(f"Timestamps are {'✅ properly ordered' if is_ordered else '❌ NOT properly ordered'}")
        
        return summary
        
    except Exception as e:
        logger.exception(f"ViT age detection pipeline failed: {e}")
        return None
    
    finally:
        # Clean up temporary chunk files if needed
        # You might want to keep them for debugging
        pass


def pipeline_main_batch_vit_age_detection(input_paths: list, output_base_dir: str = "/tmp/batch_vit_age_detection",
                                        frame_interval: int = 30, save_frames: bool = False,
                                        chunk_duration_sec: int = 60, keep_chunks: bool = False):
    """
    Batch pipeline that processes multiple videos for ViT age detection.
    
    Args:
        input_paths: List of paths to input videos (.insv or .mp4)
        output_base_dir: Base directory for output files
        frame_interval: Frame sampling interval for age detection
        save_frames: Whether to save frames with detections
        chunk_duration_sec: Duration of each video chunk
        keep_chunks: Whether to keep temporary chunk files (default: False)
    """
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    try:
        all_pipeline_results = []
        
        for input_path in input_paths:
            input_path = Path(input_path)
            logger.info(f"Processing video: {input_path}")
            
            # Create unique output directory for each video
            video_output_dir = Path(output_base_dir) / input_path.stem
            
            # Run pipeline for this video
            result = pipeline_main_vit_age_detection(
                input_path=str(input_path),
                output_dir=str(video_output_dir),
                frame_interval=frame_interval,
                save_frames=save_frames,
                chunk_duration_sec=chunk_duration_sec,
                keep_chunks=keep_chunks
            )
            
            if result:
                all_pipeline_results.append({
                    "video_path": str(input_path),
                    "output_dir": str(video_output_dir),
                    "pipeline_result": result
                })
                logger.info(f"✅ Successfully processed: {input_path}")
            else:
                logger.error(f"❌ Failed to process: {input_path}")
        
        # Create batch summary
        batch_summary = {
            "total_videos": len(input_paths),
            "successfully_processed": len(all_pipeline_results),
            "failed": len(input_paths) - len(all_pipeline_results),
            "total_faces_detected": sum(r["pipeline_result"]["total_faces_detected"] for r in all_pipeline_results),
            "total_processed_frames": sum(r["pipeline_result"]["total_processed_frames"] for r in all_pipeline_results),
            "total_processing_time": sum(r["pipeline_result"]["total_processing_time"] for r in all_pipeline_results),
            "frame_interval": frame_interval,
            "save_frames": save_frames,
            "chunk_duration_sec": chunk_duration_sec,
            "video_results": all_pipeline_results
        }
        
        batch_summary_file = Path(output_base_dir) / "batch_vit_age_detection_summary.json"
        with open(batch_summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        logger.info("="*60)
        logger.info("BATCH VIT AGE DETECTION PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Total videos: {batch_summary['total_videos']}")
        logger.info(f"Successfully processed: {batch_summary['successfully_processed']}")
        logger.info(f"Failed: {batch_summary['failed']}")
        logger.info(f"Total faces detected: {batch_summary['total_faces_detected']}")
        logger.info(f"Total processing time: {batch_summary['total_processing_time']:.2f}s")
        logger.info(f"Batch summary saved to: {batch_summary_file}")
        
        return batch_summary
        
    except Exception as e:
        logger.exception(f"Batch ViT age detection pipeline failed: {e}")
        return None


if __name__ == "__main__":
    # Configuration for single video processing
    INPUT_VIDEO = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/VID_20250720_152154_00_011.insv"  # or .mp4
    OUTPUT_DIR = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video-age-detection-pipeline/outputs"
    
    # Import config values instead of hardcoding
    try:
        from video_age_detection_pipeline.utils.config import FRAME_INTERVAL, SAVE_FRAMES
        print(f"✅ Using config values: FRAME_INTERVAL={FRAME_INTERVAL}, SAVE_FRAMES={SAVE_FRAMES}")
    except ImportError:
        try:
            # Try alternative import path
            import sys
            sys.path.insert(0, 'video-age-detection-pipeline')
            from utils.config import FRAME_INTERVAL, SAVE_FRAMES
            print(f"✅ Using config values: FRAME_INTERVAL={FRAME_INTERVAL}, SAVE_FRAMES={SAVE_FRAMES}")
        except ImportError:
            try:
                # Try relative import from current directory
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(current_dir, 'video-age-detection-pipeline', 'utils')
                sys.path.insert(0, config_path)
                from config import FRAME_INTERVAL, SAVE_FRAMES
                print(f"✅ Using config values: FRAME_INTERVAL={FRAME_INTERVAL}, SAVE_FRAMES={SAVE_FRAMES}")
            except ImportError:
                # Fallback values if config import fails
                FRAME_INTERVAL = 30
                SAVE_FRAMES = False
                print(f"⚠️ Config import failed, using fallback: FRAME_INTERVAL={FRAME_INTERVAL}, SAVE_FRAMES={SAVE_FRAMES}")
    
    CHUNK_DURATION_SEC = 20  # 20-second chunks
    
    # Run single video pipeline
    result = pipeline_main_vit_age_detection(
        input_path=INPUT_VIDEO,
        output_dir=OUTPUT_DIR,
        frame_interval=FRAME_INTERVAL,
        save_frames=SAVE_FRAMES,
        chunk_duration_sec=CHUNK_DURATION_SEC,
        keep_chunks=False  # Set to True if you want to keep temporary chunk files
    )
    
    # Example for batch processing
    # input_videos = ["video1.insv", "video2.mp4", "video3.insv"]
    # batch_result = pipeline_main_batch_vit_age_detection(
    #     input_paths=input_videos,
    #     output_base_dir="/tmp/batch_vit_age_detection",
    #     chunk_duration_sec=CHUNK_DURATION_SEC
    # )
