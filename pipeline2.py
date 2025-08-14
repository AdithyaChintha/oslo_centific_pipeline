# main_pipeline.py
import ray
import json
import os
from pathlib import Path

# Import existing Ray jobs
from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
from ray_jobs.nsfw_det import process_video_chunks_for_nsfw

# from ray_jobs.upload_manager import upload_to_azure
# from ray_jobs.scene_change import detect_scene_changes
# from ray_jobs.vad_audio import detect_vad
# from ray_jobs.motion_energy import compute_motion_energy
# from ray_jobs.embeddings_driver import compute_embeddings
# from ray_jobs.yolo_sort_tracker import run_tracking
# from ray_jobs.fusion_gap_merge import fuse_and_merge
# from ray_jobs.quality_flagger import flag_quality_issues
# from ray_jobs.segment_classifier import classify_segments

from utils.logger import get_logger
logger = get_logger("pipeline")

def pipeline_main(input_path: str, model_path: str = "models/model.onnx", 
                 labels_path: str = "labels.json", output_dir: str = "/tmp/results"):
    """
    Main pipeline that processes INSV/MP4 videos for NSFW detection.
    
    Args:
        input_path: Path to input video (.insv or .mp4)
        model_path: Path to ONNX NSFW detection model
        labels_path: Path to labels JSON file
        output_dir: Directory for output files
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
            
            if not conversion_result["success"]:
                logger.error(f"INSV conversion failed: {conversion_result.get('error')}")
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
            
            # Stage 2: Split video into chunks
            logger.info(f"Splitting video into shards...")
            chunk_paths = ray.get(split_video_into_shards.remote(mp4_path))
            logger.info(f"Created {len(chunk_paths)} video chunks")
            
            # Stage 3: Run NSFW detection on chunks in parallel
            logger.info(f"Running NSFW detection on {len(chunk_paths)} chunks...")
            nsfw_results = ray.get(process_video_chunks_for_nsfw.remote(
                chunk_paths, model_path, labels_path, confidence_threshold=0.5
            ))
            
            if nsfw_results["success"]:
                logger.info(f"NSFW detection complete for {view_name}")
                logger.info(f"Total NSFW detections: {nsfw_results['total_nsfw_detections']}")
                logger.info(f"Processing time: {nsfw_results['total_processing_time_seconds']}s")
                
                # Save results for this view
                output_file = Path(output_dir) / f"{input_path.stem}_{view_name}_nsfw_timestamps.json"
                with open(output_file, 'w') as f:
                    json.dump(nsfw_results, f, indent=2)
                
                logger.info(f"Results saved to: {output_file}")
                
                all_results.append({
                    "view": view_name,
                    "mp4_path": mp4_path,
                    "results_file": str(output_file),
                    "nsfw_detections": nsfw_results["total_nsfw_detections"],
                    "processing_time": nsfw_results["total_processing_time_seconds"]
                })
            else:
                logger.error(f"NSFW detection failed for {view_name}")
        
        # Stage 4: Create summary report
        summary = {
            "input_file": str(input_path),
            "file_type": "INSV" if input_path.suffix.lower() == '.insv' else "MP4",
            "views_processed": len(mp4_paths),
            "total_nsfw_detections": sum(r["nsfw_detections"] for r in all_results),
            "total_processing_time": sum(r["processing_time"] for r in all_results),
            "view_results": all_results
        }
        
        summary_file = Path(output_dir) / f"{input_path.stem}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*60)
        logger.info(f"Input: {input_path}")
        logger.info(f"Views processed: {summary['views_processed']}")
        logger.info(f"Total NSFW detections: {summary['total_nsfw_detections']}")
        logger.info(f"Total processing time: {summary['total_processing_time']:.2f}s")
        logger.info(f"Summary saved to: {summary_file}")
        
        return summary
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return None
    
    finally:
        # Clean up temporary chunk files if needed
        # You might want to keep them for debugging
        pass

if __name__ == "__main__":
    # Configuration
    INPUT_VIDEO = "test.insv"  # or .mp4
    MODEL_PATH = "models/model.onnx"
    LABELS_PATH = "labels.json"
    OUTPUT_DIR = "/tmp/nsfw_results"
    
    # Run pipeline
    result = pipeline_main(
        input_path=INPUT_VIDEO,
        model_path=MODEL_PATH,
        labels_path=LABELS_PATH,
        output_dir=OUTPUT_DIR
    )