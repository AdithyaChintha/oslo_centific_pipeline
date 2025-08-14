import ray

from setup.cosmos_setup import setup_cosmos
from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_mp4
from ray_jobs.scene_detection import detect_scenes
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
    ray.init()

    # Setup Cosmos DB
    setup_cosmos()

    # Stage A: Upload file to Azure (if needed)
    logger.info("Uploading file to Azure...")
    # upload_to_azure(input_mp4_path)

    # Convert .insv to .mp4 before processing
    converted_mp4 = ray.get(convert_insv_to_mp4.remote("data/video_001.insv"))
    shard_paths = split_video_into_shards.remote(converted_mp4)

    # Stage B: Segment video into 1-minute chunks
    shard_paths = split_video_into_shards.remote(input_mp4_path)

    # Define the path to the prompt configuration file
    prompt_path = "config/cosmos_prompt.yaml"

    # Stage C: Parallel Analysis
    scene_detection_tasks = [detect_scenes.remote(shard, prompt_path) for shard in ray.get(shard_paths)]
    scenes = ray.get(scene_detection_tasks)
    # vad_tasks = detect_vad.remote(shard_paths)
    # motion_tasks = compute_motion_energy.remote(shard_paths)

    # # Stage D: Feature extraction
    # embeddings = compute_embeddings.remote(scene_tasks, motion_tasks)
    # tracks = run_tracking.remote(vad_tasks)

    # # Stage E: Fusion
    # fused = fuse_and_merge.remote(embeddings, tracks)

    # # Stage F-G: Quality flag + classification
    # flagged = flag_quality_issues.remote(fused)
    # final_segments = classify_segments.remote(flagged)

    logger.info("Pipeline complete. Final segments ready for Label Studio.")
    # return final_segments

if __name__ == "__main__":
    pipeline_main("sample_data/example_360video.mp4")