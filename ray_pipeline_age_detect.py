# main_pipeline_age_detection.py
import ray
import json
import os
from pathlib import Path

# Import existing Ray jobs
from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
from ray_jobs.face_age_detector import process_video_chunks_for_face_detection

from utils.logger import get_logger
logger = get_logger("age_detection_pipeline")


def pipeline_main_age_detection(input_path: str, output_dir: str = "/tmp/age_detection_results",
                               frame_interval: int = 30, save_frames: bool = False,
                               chunk_duration_sec: int = 60):
    """
    Main pipeline that processes INSV/MP4 videos for face age detection.
    
    Args:
        input_path: Path to input video (.insv or .mp4)
        output_dir: Directory for output files
        frame_interval: Frame sampling interval for face detection
        save_frames: Whether to save frames with detections
        chunk_duration_sec: Duration of each video chunk
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
                     # Use unique output directory for each view to avoid conflicts
                     view_output_dir = Path(output_dir) / f"chunks_{view_name}"
                     os.makedirs(view_output_dir, exist_ok=True)
                     logger.info(f"Splitting video into {chunk_duration_sec}s chunks...")
                     view_chunk_paths = ray.get(split_video_into_shards.remote(mp4_path, str(view_output_dir), chunk_duration_sec))
                     logger.info(f"Created {len(view_chunk_paths)} video chunks for {view_name}")
                     
                     # Stage 3: Run face age detection on chunks in parallel for THIS view only
                     logger.info(f"Running face age detection on {len(view_chunk_paths)} chunks for {view_name}...")
                     face_detection_results = ray.get(process_video_chunks_for_face_detection.remote(
                         view_chunk_paths, None, frame_interval, save_frames, chunk_duration_sec
                     ))
                     
                     if face_detection_results["success"]:
                         logger.info(f"Face age detection complete for {view_name}")
                         logger.info(f"Total faces detected: {face_detection_results['total_faces_detected']}")
                         logger.info(f"Total frames processed: {face_detection_results['total_processed_frames']}")
                         # Calculate processing time if not available
                         processing_time = face_detection_results.get('total_processing_time_seconds', 0)
                         logger.info(f"Processing time: {processing_time}s")
                         
                         # Save detailed results for this view
                         output_file = Path(output_dir) / f"{input_path.stem}_{view_name}_face_analysis_results.json"
                         with open(output_file, 'w') as f:
                             json.dump(face_detection_results, f, indent=2)
                         
                         logger.info(f"Detailed results saved to: {output_file}")
                         
                         # Save individual chunk results for debugging for THIS view only
                         chunk_results_dir = Path(output_dir) / f"chunk_results_{view_name}"
                         os.makedirs(chunk_results_dir, exist_ok=True)
                         
                         for j, chunk_path in enumerate(view_chunk_paths):
                             chunk_name = Path(chunk_path).stem
                             chunk_result_file = chunk_results_dir / f"{chunk_name}_results.json"
                             
                             # Find results for this chunk
                             chunk_results = []
                             # Use flagged_segments instead of face_analysis_results
                             for segment in face_detection_results.get("flagged_segments", []):
                                 if segment.get("chunk_offset_seconds") == j * chunk_duration_sec:
                                     chunk_results.append(segment)
                             
                             with open(chunk_result_file, 'w') as f:
                                 json.dump({
                                     "chunk_path": chunk_path,
                                     "chunk_offset_seconds": j * chunk_duration_sec,
                                     "frames_analyzed": len(chunk_results),
                                     "face_detections": chunk_results
                                 }, f, indent=2)
                         
                         all_results.append({
                             "view": view_name,
                             "mp4_path": mp4_path,
                             "results_file": str(output_file),
                             "total_faces_detected": face_detection_results["total_faces_detected"],
                             "total_processed_frames": face_detection_results["total_processed_frames"],
                             "processing_time": face_detection_results.get("total_processing_time_seconds", 0),
                             "chunks_processed": face_detection_results["chunks_processed"]
                         })
                     else:
                         logger.error(f"Face age detection failed for {view_name}")
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
        
        summary_file = Path(output_dir) / f"{input_path.stem}_age_detection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("="*60)
        logger.info("AGE DETECTION PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Input: {input_path}")
        logger.info(f"Views processed: {summary['views_processed']}")
        logger.info(f"Total faces detected: {summary['total_faces_detected']}")
        logger.info(f"Total frames processed: {summary['total_processed_frames']}")
        logger.info(f"Total processing time: {summary['total_processing_time']:.2f}s")
        logger.info(f"Chunks processed: {summary['chunks_processed']}")
        logger.info(f"Summary saved to: {summary_file}")
        
        return summary
        
    except Exception as e:
        logger.exception(f"Age detection pipeline failed: {e}")
        return None
    
    finally:
        # Clean up temporary chunk files if needed
        # You might want to keep them for debugging
        pass


def pipeline_main_batch_age_detection(input_paths: list, output_base_dir: str = "/tmp/batch_age_detection",
                                    frame_interval: int = 30, save_frames: bool = False,
                                    chunk_duration_sec: int = 60):
    """
    Batch pipeline that processes multiple videos for face age detection.
    
    Args:
        input_paths: List of paths to input videos (.insv or .mp4)
        output_base_dir: Base directory for output files
        frame_interval: Frame sampling interval for face detection
        save_frames: Whether to save frames with detections
        chunk_duration_sec: Duration of each video chunk
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
            result = pipeline_main_age_detection(
                input_path=str(input_path),
                output_dir=str(video_output_dir),
                frame_interval=frame_interval,
                save_frames=save_frames,
                chunk_duration_sec=chunk_duration_sec
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
        
        batch_summary_file = Path(output_base_dir) / "batch_age_detection_summary.json"
        with open(batch_summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        logger.info("="*60)
        logger.info("BATCH AGE DETECTION PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Total videos: {batch_summary['total_videos']}")
        logger.info(f"Successfully processed: {batch_summary['successfully_processed']}")
        logger.info(f"Failed: {batch_summary['failed']}")
        logger.info(f"Total faces detected: {batch_summary['total_faces_detected']}")
        logger.info(f"Total processing time: {batch_summary['total_processing_time']:.2f}s")
        logger.info(f"Batch summary saved to: {batch_summary_file}")
        
        return batch_summary
        
    except Exception as e:
        logger.exception(f"Batch age detection pipeline failed: {e}")
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
    
    CHUNK_DURATION_SEC = 20  # 1-minute chunks
    
    # Run single video pipeline
    result = pipeline_main_age_detection(
        input_path=INPUT_VIDEO,
        output_dir=OUTPUT_DIR,
        frame_interval=FRAME_INTERVAL,
        save_frames=SAVE_FRAMES,
        chunk_duration_sec=CHUNK_DURATION_SEC
    )
    
    # Example for batch processing
    # input_videos = ["video1.insv", "video2.mp4", "video3.insv"]
    # batch_result = pipeline_main_batch_age_detection(
    #     input_paths=input_videos,
    #     output_base_dir="/tmp/batch_age_detection",
    #     frame_interval=FRAME_INTERVAL,
    #     save_frames=SAVE_FRAMES,
    #     chunk_duration_sec=CHUNK_DURATION_SEC
    # )
