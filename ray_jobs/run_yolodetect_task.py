# ray_jobs/run_yolodetect_task.py
from __future__ import annotations
import os, re
from typing import Any, Dict, Optional
import ray

from pathlib import Path

# ==== EDIT THESE ====
VIDEO_PATH   = "test_view2.mp4"   # put your video path here
MODEL_NAME   = "yolo11m.pt"       # or your custom weight path
CONF         = 0.5
IOU          = 0.5
FRAME_STRIDE = 5
CLASSES      = [0]                  # e.g., [0] for "person", None for all
DEVICE       = "cuda:0"                 # e.g., "cuda:0" to force GPU
GAP_SEC      = 10.0                  # merge gap for spans
EVENTS_OUT   = "outputs/events.jsonl"
SPANS_OUT    = "outputs/events_spans.jsonl"
# ====================

def _part_index(filename: str) -> int:
    """Extract shard index from names like 'xxx_part3.mp4' -> 3; default to 0."""
    m = re.search(r"_part(\d+)\.mp4$", filename)
    return int(m.group(1)) if m else 0

@ray.remote(num_gpus=0)  # set to 1 if you want GPU per shard
def run_yolodetect_on_shard(
    video_path: str,
    out_dir: str = "/tmp/yolo_demo",
    shard_seconds: int = 60,
    model: str = MODEL_NAME,
    conf: float = CONF,
    iou: float = IOU,
    frame_stride: int = FRAME_STRIDE,
    classes: Optional[list[int]] = CLASSES,
    device: Optional[str] = DEVICE,
    gap_sec: float = GAP_SEC,
) -> Dict[str, Any]:
    """
    Minimal Ray task: call yolodetect(...) for a single shard.
    No changes to detection_module.
    """
    # Import inside the task for robust module resolution on Ray workers
    from yolo_detection_r.run_detection_demo import yolodetect
    from yolo_detection_r.run_detection_demo import merge_events

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(video_path)
    stem, _ = os.path.splitext(base)
    part = _part_index(base)
    base_offset = part * shard_seconds

    events_out = os.path.join(out_dir, f"{stem}.events.jsonl")
    spans_out  = os.path.join(out_dir, f"{stem}.spans.jsonl")

    return yolodetect(
        video=video_path,
        events_out=events_out,
        spans_out=spans_out,
        model=model,
        conf=conf,
        iou=iou,
        frame_stride=frame_stride,
        classes=classes,
        device=device,
        gap_sec=gap_sec,
        base_offset=base_offset,
    )
