import ray

from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_mp4
from ray_jobs.motion_energy import compute_motion_energy

# from ray_jobs.upload_manager import upload_to_azure
# from ray_jobs.scene_change import detect_scene_changes
# from ray_jobs.vad_audio import detect_vad
# from ray_jobs.embeddings_driver import compute_embeddings
# from ray_jobs.yolo_sort_tracker import run_tracking
# from ray_jobs.fusion_gap_merge import fuse_and_merge
# from ray_jobs.quality_flagger import flag_quality_issues
# from ray_jobs.segment_classifier import classify_segments

from utils.logger import get_logger

logger = get_logger("pipeline")

def pipeline_main(input_mp4_path: str):
    """
    Main Ray pipeline for 360° video processing with motion energy detection.
    
    Args:
        input_mp4_path: Path to input MP4 file or INSV file
    """
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    logger.info(f"Starting Ray pipeline for: {input_mp4_path}")

    try:
        # Stage A: Upload file to Azure (if needed)
        logger.info("Uploading file to Azure...")
        # upload_to_azure(input_mp4_path)

        # Stage A.1: Convert .insv to .mp4 if needed
        if input_mp4_path.lower().endswith('.insv'):
            logger.info("Converting INSV to MP4...")
            converted_mp4 = ray.get(convert_insv_to_mp4.remote(input_mp4_path))
            video_path = converted_mp4
        else:
            video_path = input_mp4_path

        # Stage B: Segment video into 1-minute chunks
        logger.info("Splitting video into shards...")
        shard_paths_future = split_video_into_shards.remote(video_path)
        shard_paths = ray.get(shard_paths_future)
        logger.info(f"Video split into {len(shard_paths)} chunks")

        # Stage C: Parallel Analysis
        logger.info("Starting parallel motion energy analysis...")
        
        # Motion Energy Detection (YOUR IMPLEMENTATION)
        motion_config = {
            "sensitivity_level": "medium",
            "save_detailed_data": True
        }
        motion_tasks = compute_motion_energy.remote(
            shard_paths, 
            config=motion_config
        )
        
        # Placeholder tasks for other team members
        # scene_tasks = detect_scene_changes.remote(shard_paths)
        # vad_tasks = detect_vad.remote(shard_paths)

        # Stage D: Get results
        logger.info("Collecting motion energy results...")
        motion_results = ray.get(motion_tasks)
        
        if motion_results["success"]:
            logger.info(f"✅ Motion Energy Analysis Complete:")
            logger.info(f"   Total segments: {motion_results['total_motion_segments']}")
            logger.info(f"   Processing time: {motion_results['total_processing_time_seconds']:.1f}s")
            logger.info(f"   Video duration: {motion_results['total_video_duration_seconds']:.1f}s")
            logger.info(f"   Speed factor: {motion_results.get('processing_speed_factor', 0):.1f}x real-time")
        else:
            logger.error(f"❌ Motion Energy Analysis Failed: {motion_results.get('error', 'Unknown error')}")

        # Stage D: Feature extraction (placeholders)
        # embeddings = compute_embeddings.remote(scene_tasks, motion_tasks)
        # tracks = run_tracking.remote(vad_tasks)

        # Stage E: Fusion (placeholder)
        # fused = fuse_and_merge.remote(embeddings, tracks)

        # Stage F-G: Quality flag + classification (placeholders)
        # flagged = flag_quality_issues.remote(fused)
        # final_segments = classify_segments.remote(flagged)

        logger.info("="*60)
        logger.info("PIPELINE COMPLETE - Motion Energy Analysis")
        logger.info("="*60)
        logger.info("Motion energy analysis ready for Label Studio integration.")
        
        return motion_results

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return None
    
    finally:
        # Optionally shutdown Ray (comment out if keeping cluster alive)
        # ray.shutdown()
        pass
