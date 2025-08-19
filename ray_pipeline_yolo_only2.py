import ray
import os
from utils.logger import get_logger

# Import setup and all necessary Ray tasks
from setup.cosmos.setup import setup_cosmos
from ray_jobs.video_splitter import split_video_into_shards
#from ray_jobs.insv_to_mp4 import convert_insv_to_mp4
from ray_jobs.scene_detection import detect_scenes
from ray_jobs.yolo_detection import run_yolo_detection
#from ray_jobs.audio_diarization_pii import audio_diarization_pii

logger = get_logger("UnifiedPipeline")

def pipeline_main(input_video_path: str, output_dir: str):
    """
    A unified Ray pipeline for video analysis, including scene detection,
    YOLO object detection, and audio diarization with PII detection.
    """
    ray.init()
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- STAGE A: INITIAL SETUP ---
    logger.info("Starting initial setup...")
    setup_cosmos()
    logger.info("Setup complete.")

    # --- STAGE B: VIDEO PREPARATION ---
    logger.info(f"Preparing video: {input_video_path}")
    
    # Convert .insv to .mp4 if necessary
    if input_video_path.lower().endswith('.insv'):
        mp4_path_ref = convert_insv_to_mp4.remote(input_video_path)
        mp4_path = ray.get(mp4_path_ref)
        logger.info(f"Converted {input_video_path} to {mp4_path}")
    else:
        mp4_path = input_video_path

    # Split the video into shards
    shards_dir = os.path.join(output_dir, "video_shards")
    shard_paths_ref = split_video_into_shards.remote(mp4_path, output_dir=shards_dir, shard_duration_minutes=1)
    shard_paths = ray.get(shard_paths_ref)
    logger.info(f"Video split into {len(shard_paths)} shards in {shards_dir}")

    # --- STAGE C: PARALLEL ANALYSIS ---
    logger.info("Launching parallel analysis tasks for each shard...")
    
    scene_detection_tasks = []
    yolo_detection_tasks = []
    audio_diarization_tasks = []

    # Define the prompt for scene detection
    prompt_path = "config/cosmos_prompt.yaml"

    for i, shard_path in enumerate(shard_paths):
        shard_output_dir = os.path.join(output_dir, f"shard_{i}")
        os.makedirs(shard_output_dir, exist_ok=True)

        # Launch Scene Detection Task
        #scene_task = detect_scenes.remote(shard_path, prompt_path)
        #scene_detection_tasks.append(scene_task)

        # Launch YOLO Detection Task
        yolo_output_dir = os.path.join(shard_output_dir, "yolo_output")
        yolo_task = run_yolo_detection.remote(shard_path, yolo_output_dir)
        yolo_detection_tasks.append(yolo_task)

        # Launch Audio Diarization & PII Task
        #audio_output_dir = os.path.join(shard_output_dir, "audio_output")
        #audio_task = audio_diarization_pii.remote(shard_path, audio_output_dir)
        #audio_diarization_tasks.append(audio_task)

    # --- STAGE D: GATHER RESULTS ---
    logger.info("Waiting for all analysis tasks to complete...")
    
    scene_results = ray.get(scene_detection_tasks)
    yolo_results = ray.get(yolo_detection_tasks)
    audio_results = ray.get(audio_diarization_tasks)
    
    logger.info("All analysis tasks have completed.")
    
    # --- STAGE E: (Optional) MERGE/AGGREGATE RESULTS ---
    # Here you could add logic to merge the results from the shards,
    # for example, merging the YOLO detection JSONL files.
    
    logger.info("Unified pipeline complete.")
    logger.info(f"Scene Detection Results: {scene_results}")
    logger.info(f"YOLO Detection Results: {yolo_results}")
    logger.info(f"Audio Diarization Results: {audio_results}")

if __name__ == "__main__":
    # Define the input video and the main output directory
    INPUT_VIDEO = "sample_data/example_360video.mp4"  # Update this path
    OUTPUT_DIR = "outputs/unified_pipeline_output"
    
    pipeline_main(INPUT_VIDEO, OUTPUT_DIR)
