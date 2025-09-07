import ray
import os
import re
from typing import Any, Dict, List, Optional
import json, pandas as pd
from yolo_detection_r.utils.Tstamp import fmt_hhmmss_ms

# Assuming the script is run from the project root, so yolo_detection_r is in the path
from yolo_detection_r.utils.write_jsonl import write_jsonl
from yolo_detection_r.utils.write_jsonl import save_people_count_bins_json
from yolo_detection_r.utils.write_jsonl import save_people_presence_json

from yolo_detection_r.detection_module import detect_events_raw
from yolo_detection_r.detection_module import people_count_bins_from_events
from yolo_detection_r.detection_module import people_presence_spans_from_events
import cv2

@ray.remote(num_cpus=1, num_gpus=1) # Can be changed to num_gpus=1 if a GPU model is used
def run_yolo_detection(
    video_path: str,
    output_dir: str,
    shard_seconds: int = 60,
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
    tracker_cfg: Optional[str] = "config/botsort_reid.yaml",     
    #counter
    enable_count: bool = True, #new
    bin_size_sec: float = 1.0,  
    gap_sec: float = 1.0,
 ) -> Dict[str, Any]:
    """
    A Ray task to run YOLO object detection on a video shard.
    """
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    events_out_path = os.path.join(output_dir, f"{stem}_yolo_events.jsonl")
    spans_out_path = os.path.join(output_dir, f"{stem}_yolo_spans.jsonl")
    # path to saved people count JSON (if produced)
    json_path = None

    try:
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

        # If this shard corresponds to a part of a larger video, offset times
        # by shard_index * shard_seconds so results are in global timeline.
        def _part_index(filename: str) -> int:
            m = re.search(r"_part(\d+)\.mp4$", filename)
            return int(m.group(1)) if m else 0

        shard_idx = _part_index(os.path.basename(video_path))
        time_offset = float(shard_idx * shard_seconds)

        # Apply time offset to events (t and ts)
        if isinstance(events, list) and time_offset:
            for e in events:
                if "t" in e and isinstance(e["t"], (int, float)):
                    e["t"] = round(float(e["t"]) + time_offset, 3)
                    try:
                        e["ts"] = fmt_hhmmss_ms(e["t"])
                    except Exception:
                        pass

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
            "shard_index": shard_idx,
            "shard_seconds": shard_seconds,
        }
        write_jsonl(events_out_path, events, meta=meta)

        # people counting / presence spans
        spans_written = None
        if enable_count and enable_tracking:
            person_classes = {"person"}
            counts = people_count_bins_from_events(events=events, bin_size_sec=bin_size_sec, person_classes=person_classes)
            json_path = save_people_count_bins_json(counts, video_path=video_path, output_dir=output_dir)
            print("People count saved to:", json_path)

            presence = people_presence_spans_from_events(events, gap_sec=gap_sec, person_classes=person_classes)
            spans_written = save_people_presence_json(presence, video_path, output_dir)
            print("Person presence saved to:", spans_written)

        # compute aggregate stats
        num_events = len(events) if isinstance(events, list) else 0
        detections_count = 0
        if isinstance(events, list):
            for e in events:
                dets = e.get("detections") if isinstance(e, dict) else None
                if isinstance(dets, list):
                    detections_count += len(dets)

        # video metadata
        fps = None
        duration = None
        try:
            cap = cv2.VideoCapture(video_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps > 0 and frames > 0:
                duration = frames / fps
            cap.release()
        except Exception:
            fps = None
            duration = None

        result = {
            "video": video_path,
            "events_jsonl": events_out_path,
            "people_count_json": json_path,
            "spans_jsonl": spans_written,
            "num_events": num_events,
            "detections_count": detections_count,
            "video_duration": duration,
            "fps": fps,
        }

        return result

    except Exception as e:
        return {"__error__": f"run_yolo_detection failed: {e}"}
