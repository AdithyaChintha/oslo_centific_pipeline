import ray
import os
from typing import Any, Dict, List, Optional

# Assuming the script is run from the project root, so yolo_detection_r is in the path
from yolo_detection_r.detection_module import detect_events_raw
from yolo_detection_r.utils.write_jsonl import write_jsonl

@ray.remote(num_cpus=1) # Can be changed to num_gpus=1 if a GPU model is used
def run_yolo_detection(
    video_path: str,
    output_dir: str,
    model: str = "yolo11m.pt",
    conf: float = 0.5,
    iou: float = 0.5,
    frame_stride: int = 5,
    classes: Optional[List[int]] = None,
    device: Optional[str] = None,
    enable_tracking: bool = True,           # NEW
    tracker_backend: str = "botsort",        # NEW
    tracker_cfg: Optional[str] = None,       # NEW
) -> str:
    """
    A Ray task to run YOLO object detection on a video shard.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stem = os.path.splitext(os.path.basename(video_path))[0]
    events_out_path = os.path.join(output_dir, f"{stem}_yolo_events.jsonl")

    # Run raw detection
    events, tracker_used = detect_events_raw(
        video_path=video_path,
        model_name=model,
        conf=conf,
        iou=iou,
        frame_stride=frame_stride,
        classes=classes,
        device=device,
        enable_tracking=enable_tracking,
        tracker_backend=tracker_backend,
        tracker_cfg=tracker_cfg,
    )

    # Write events to JSONL
    meta = {
        "source_video": video_path,
        "model": model,
        "conf": conf,
        "iou": iou,
        "frame_stride": frame_stride,
        "classes": classes,
        "device": device,
        "enable_tracking": enable_tracking,
        "tracker_backend": tracker_backend,
        "tracker_cfg": tracker_cfg,
        "tracker": tracker_used,
    }
    write_jsonl(events_out_path, events, meta=meta)
    
    return events_out_path
