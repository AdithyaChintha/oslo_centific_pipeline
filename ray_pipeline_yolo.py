import os
import ray
from typing import List

from utils.logger import get_logger
from ray_jobs.video_splitter import split_video_into_shards
#from ray_jobs.insv_to_mp4 import convert_insv_to_mp4  # alias to convert_insv_to_dual_mp4
from ray_jobs.run_yolodetect_task import run_yolodetect_on_shard

logger = get_logger("pipeline")

def _flatten(x: List[List[str]]) -> List[str]:
    out: List[str] = []
    for lst in x:
        out.extend(lst)
    return out

def pipeline_main(input_path: str, shard_seconds: int = 60) -> dict:
    """
    End-to-end:
      .insv -> (dual) mp4 list -> split into shards -> run yolodetect on each shard
    Returns a dict with lists of mp4s, shards, and per-shard results.
    """
    ray.init(ignore_reinit_error=True)

    # A) Normalize to a list of mp4s
    if input_path.lower().endswith(".insv"):
        mp4_list: List[str] = ray.get(convert_insv_to_mp4.remote(input_path))
    else:
        mp4_list = [input_path]
    logger.info(f"Stage A: got {len(mp4_list)} mp4(s)")

    # B) Split each mp4 into fixed-length shards (in parallel)
    shard_futs = [split_video_into_shards.remote(p, duration_sec=shard_seconds) for p in mp4_list]
    shard_lists: List[List[str]] = ray.get(shard_futs)
    shards: List[str] = _flatten(shard_lists)
    logger.info(f"Stage B: produced {len(shards)} shard(s)")

    # C) Run your original demo on each shard (in parallel)
    one = run_yolodetect_on_shard.remote(
    shards[0],
    out_dir="/tmp/yolo_demo",
    shard_seconds=shard_seconds,
    model="yolov8n.pt",
    conf=0.5,
    iou=0.5,
    frame_stride=5,
    classes=None,
    device="cpu",
    gap_sec=3.0,
    )

    try:
        #print(ray.get(one))
        res = ray.get(one)              # dict
        demo_results = [res] 
    except Exception as e:
        print("REMOTE TASK FAILED:\n", e)
        raise

    return {
        "mp4s": mp4_list,
        "shards": shards,
        #"demo_results": demo_results,
    }

if __name__ == "__main__":
    # .mp4

    print(pipeline_main("test_view2.mp4", shard_seconds=60))
