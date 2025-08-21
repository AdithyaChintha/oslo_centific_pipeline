import os, ray
from utils.logger import get_logger
from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.yolo_detection import run_yolo_detection

log = get_logger("YOLOOnly")

def pipeline_yolo_only(input_video_path: str, output_dir: str, shard_minutes: int = 1):
    ray.init()
    os.makedirs(output_dir, exist_ok=True)

    if input_video_path.lower().endswith(".insv"):
        raise ValueError("Use an MP4 for this quick YOLO-only test.")

    # 1) 切片
    shards_dir = os.path.join(output_dir, "video_shards")
    shard_refs = split_video_into_shards.remote(
        input_video_path, output_dir=shards_dir, shard_duration_minutes=shard_minutes
    )
    shard_paths = ray.get(shard_refs)
    log.info(f"Split into {len(shard_paths)} shards at {shards_dir}")

    # 2) 仅跑 YOLO
    tasks = []
    for i, shard in enumerate(shard_paths):
        yolo_out = os.path.join(output_dir, f"shard_{i}", "yolo_output")
        os.makedirs(yolo_out, exist_ok=True)
        tasks.append(run_yolo_detection.remote(shard, yolo_out))

    results = ray.get(tasks)
    log.info(f"YOLO finished on {len(results)} shards")
    return results

if __name__ == "__main__":
    pipeline_yolo_only("video_with_minors.mp4", "outputs/yolo_only_test", shard_minutes=1)
