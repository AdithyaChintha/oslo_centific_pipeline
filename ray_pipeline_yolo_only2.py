import ray
import os
from utils.logger import get_logger

# Import setup and all necessary Ray tasks
# from setup.cosmos.setup import setup_cosmos            # test
from ray_jobs.video_splitter import split_video_into_shards
# from ray_jobs.insv_to_mp4 import convert_insv_to_mp4   # test
# from ray_jobs.scene_detection import detect_scenes      # test
from ray_jobs.yolo_detection import run_yolo_detection
# from ray_jobs.audio_diarization_pii import audio_diarization_pii  # test

logger = get_logger("UnifiedPipeline")

def pipeline_main(input_video_path: str, output_dir: str):
    """
    YOLO-only Ray pipeline: split video into shards and run YOLO detection per shard.
    """
    ray.init()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- STAGE A: INITIAL SETUP ---
    # logger.info("Starting initial setup...")
    # setup_cosmos()
    # logger.info("Setup complete.")

    # --- STAGE B: VIDEO PREPARATION ---
    logger.info(f"Preparing video: {input_video_path}")
    
    # test, mp4 only
    # if input_video_path.lower().endswith('.insv'):
    #     mp4_path_ref = convert_insv_to_mp4.remote(input_video_path)
    #     mp4_path = ray.get(mp4_path_ref)
    #     logger.info(f"Converted {input_video_path} to {mp4_path}")
    # else:
    #     mp4_path = input_video_path
    mp4_path = input_video_path  # mp4 only

    # Split the video into shards
    shards_dir = os.path.join(output_dir, "video_shards")
    shard_paths_ref = split_video_into_shards.remote(mp4_path, shards_dir, 60)  # time 1 mins
    shard_paths = ray.get(shard_paths_ref)
    logger.info(f"Video split into {len(shard_paths)} shards in {shards_dir}")

    # --- STAGE C: YOLO ONLY ---
    logger.info("Launching YOLO detection tasks for each shard...")
    
    yolo_detection_tasks = []

    for i, shard_path in enumerate(shard_paths):
        shard_output_dir = os.path.join(output_dir, f"shard_{i}")
        os.makedirs(shard_output_dir, exist_ok=True)

        # YOLO ONLY
        yolo_output_dir = os.path.join(shard_output_dir, "yolo_output")
        yolo_task = run_yolo_detection.remote(shard_path, yolo_output_dir)
        yolo_detection_tasks.append(yolo_task)

    # --- STAGE D: GATHER RESULTS ---
    logger.info("Waiting for YOLO tasks to complete...")
    yolo_results = ray.get(yolo_detection_tasks)
    
    logger.info("YOLO-only pipeline complete.")
    logger.info(f"YOLO Detection Results: {yolo_results}")

if __name__ == "__main__":
    # Define the input video and the main output directory
    INPUT_VIDEO = "video_with_minors.mp4"  # input video
    OUTPUT_DIR = "outputs/yolo_only_output"
    
    pipeline_main(INPUT_VIDEO, OUTPUT_DIR)
