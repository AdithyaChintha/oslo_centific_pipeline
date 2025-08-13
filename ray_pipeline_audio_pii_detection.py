import ray
import json

from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
from ray_jobs.audio_diarization_pii import process_audio_diarization  # NEW IMPORT
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

def pipeline_main(input_mp4_path: str):
    """Main pipeline for audio diarization and PII detection"""
    if not ray.is_initialized():
        ray.init()

    logger.info(f"Starting pipeline with input: {input_mp4_path}")

    # Stage A: Upload file to Azure (if needed)
    # upload_to_azure(input_mp4_path)

    # Stage B: Split video into shards
    logger.info("Splitting video into shards...")
    shard_paths_future = split_video_into_shards.remote(input_mp4_path)
    shard_paths = ray.get(shard_paths_future)
    logger.info(f"Created {len(shard_paths)} shards")

    # Stage C: Audio processing
    logger.info("Processing audio diarization and PII detection...")
    audio_pii_future = process_audio_diarization.remote(shard_paths)
    audio_pii_results = ray.get(audio_pii_future)
    
    # Log summary only (not detailed results)
    total_speakers = sum([r['summary']['total_speakers'] for r in audio_pii_results])
    total_pii = sum([r['summary']['total_pii_detections'] for r in audio_pii_results])
    
    logger.info(f"Processing complete: {len(audio_pii_results)} shards, {total_speakers} speakers, {total_pii} PII detections")

    return {
        "shard_paths": shard_paths,
        "audio_pii_results": audio_pii_results,
        "total_shards": len(shard_paths),
        "total_speakers": total_speakers,
        "total_pii_detections": total_pii
    }
    
if __name__ == "__main__":
    import sys
    
    # Accept video path as argument or use default
    test_video = sys.argv[1] if len(sys.argv) > 1 else "/home/nvcoe_admin/code/oslo/pii_detection_in_audio/Video_with_PII.mp4"
    
    try:
        logger.info(f"Starting pipeline with: {test_video}")
        results = pipeline_main(test_video)
        
        # Simple summary
        logger.info(f"Pipeline complete: {results['total_shards']} shards, {results['total_speakers']} speakers, {results['total_pii_detections']} PII")
        
        # Save results
        with open("pipeline_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to pipeline_results.json")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
    finally:
        if ray.is_initialized():
            ray.shutdown()