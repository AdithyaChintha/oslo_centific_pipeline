import os
import sys
import ray

from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
from ray_jobs.motion_energy import analyze_motion_energy_only

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
            converted_mp4 = ray.get(convert_insv_to_dual_mp4.remote(input_mp4_path))
            if converted_mp4["success"]:
                video_path = converted_mp4["output_view_1"]  # Use first view
                logger.info(f"INSV converted to: {video_path}")
            else:
                logger.error(f"INSV conversion failed: {converted_mp4['error']}")
                return None
        else:
            video_path = input_mp4_path

        # Stage B: Motion Energy Analysis (YOUR IMPLEMENTATION)
        logger.info("Starting motion energy analysis...")
        
        # Import the motion energy function
        from ray_jobs.motion_energy import analyze_motion_energy_only
        
        motion_future = analyze_motion_energy_only.remote(
            video_path=video_path,
            sensitivity_level="medium",
            save_detailed_data=True
        )
        
        logger.info("Motion energy analysis submitted to Ray cluster...")
        motion_results = ray.get(motion_future)
        
        if motion_results["success"]:
            logger.info(f"✅ Motion Energy Analysis Complete:")
            logger.info(f"   Total segments: {motion_results['total_segments']}")
            logger.info(f"   Processing time: {motion_results['processing_time_seconds']:.1f}s")
            logger.info(f"   Video duration: {motion_results['video_duration_seconds']:.1f}s")
            logger.info(f"   Speed factor: {motion_results['video_duration_seconds']/motion_results['processing_time_seconds']:.1f}x real-time")
            
            # Show top segments
            if motion_results['segments']:
                logger.info(f"   Top activity segments:")
                for i, seg in enumerate(motion_results['segments'][:3], 1):
                    logger.info(f"     {i}. {seg['start_time']:.1f}s-{seg['end_time']:.1f}s ({seg['activity_type']})")
            
        else:
            logger.error(f"❌ Motion Energy Analysis Failed: {motion_results['error']}")
            return None

        # Stage C-G: Other analyses (placeholders)
        # embeddings = compute_embeddings.remote(motion_results)
        # tracks = run_tracking.remote(motion_results)
        # fused = fuse_and_merge.remote(embeddings, tracks)
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


if __name__ == "__main__":
    print("🚀 Testing Ray Pipeline with Motion Energy Analysis")
    print("="*60)
    
    # Test video path
    test_video = "input/VID_20250720_152154_00_011.mp4"
    
    # Check if video exists
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        print("Please update the path or place a test video in the input/ directory")
        sys.exit(1)
    
    print(f"📹 Processing video: {test_video}")
    
    try:
        # Run the pipeline
        results = pipeline_main(test_video)
        
        if results:
            print(f"\n🎉 PIPELINE SUCCESS!")
            print(f"📊 Results:")
            print(f"   Segments found: {results['total_segments']}")
            print(f"   Video duration: {results['video_duration_seconds']:.1f}s")
            print(f"   Processing time: {results['processing_time_seconds']:.1f}s")
        else:
            print(f"\n❌ PIPELINE FAILED")
            
    except Exception as e:
        print(f"\n❌ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Pipeline test completed")