import ray
import os
from typing import Any, Dict, List, Optional
import json, pandas as pd

# Assuming the script is run from the project root, so yolo_detection_r is in the path
from yolo_detection_r.utils.write_jsonl import write_jsonl
from yolo_detection_r.utils.write_jsonl import save_people_count_bins_json
from yolo_detection_r.utils.write_jsonl import save_people_presence_json

from yolo_detection_r.detection_module import detect_events_raw
from yolo_detection_r.detection_module import people_count_bins_from_events
from yolo_detection_r.detection_module import people_presence_spans_from_events

@ray.remote(num_cpus=1, num_gpus=1) # Can be changed to num_gpus=1 if a GPU model is used
def run_yolo_detection(
    video_path: str,
    output_dir: str,
    #detect
    model: str = "yolo11m.pt",
    conf: float = 0.5,
    iou: float = 0.5,
    frame_stride: int = 5,
    classes: Optional[List[int]] = [0],
    device: Optional[str] = None,
    #tracking
    enable_tracking: bool = True,              
    tracker_backend: str = "botsort",        
    tracker_cfg: Optional[str] = None,     
    #counter
    enable_count: bool = True, #new
    bin_size_sec: float = 1.0,  
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
        "tracker": tracker_used, #test
    }
    write_jsonl(events_out_path, events, meta=meta)

    # === NEW: fixed-interval people counting ===
    if enable_count and enable_tracking:
        
        person_classes={"person"}

        counts = people_count_bins_from_events( events=events, bin_size_sec=bin_size_sec, person_classes=person_classes)
        json_path = save_people_count_bins_json(counts, video_path=video_path, output_dir=output_dir)
        print("People count saved to:", json_path)

        presence = people_presence_spans_from_events(events, gap_sec=1.0, person_classes=person_classes)
        json_path = save_people_presence_json(presence, video_path, output_dir)
        print("Person presence saved to:", json_path) 
        
    return events_out_path
