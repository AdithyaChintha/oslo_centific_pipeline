
# run_detection_demo.py
# Simple, directly-runnable demo. Edit the constants below.
# Requires: ultralytics, opencv-python, utils, and the detection_module.py in the same folder.

"""
Modular YOLO detection pipeline:
0) Frame_generator + preprocess hook for future camera undistortion. (***in progress***)
1) Detect_events_raw: run YOLO on a long video and return per-detection "point" events.
2) Merge_events_to_spans: merge point events into "state" spans (enter/exit with duration).
3) Write_jsonl: write any list of dicts to JSONL with optional meta header.
"""

# run_detection_demo.py
from __future__ import annotations
import os, json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


from .detection_module import detect_events_raw, merge_events_to_spans
from .utils.write_jsonl import write_jsonl

# Reduce noisy output from dependencies
os.environ.setdefault("WANDB_DISABLED", "true")
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("ray").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


# ==== EDIT THESE ====
VIDEO_PATH   = "test_view2.mp4"   # put your video path here
MODEL_NAME   = "yolov8n.pt"        # or your custom weight path
CONF         = 0.5
IOU          = 0.5
FRAME_STRIDE = 5
CLASSES      = [0]                  # e.g., [0] for "person", None for all
DEVICE       = None                 # e.g., "cuda:0" to force GPU
GAP_SEC      = 3.0                  # merge gap for spans
EVENTS_OUT   = "outputs/events.jsonl"
SPANS_OUT    = "outputs/events_spans.jsonl"
# ====================

def yolodetect(
    video: str,
    events_out: Optional[str] = None,
    spans_out: Optional[str] = None,
    model: str = "yolo11m.pt",
    conf: float = 0.5,
    iou: float = 0.5,
    frame_stride: int = 5,
    classes: Optional[List[int]] = None,
    device: Optional[str] = None,
    gap_sec: float = 3.0,
    base_offset: float = 0.0,
    enable_tracking: bool = True,
) -> Dict[str, Any]:
    """
    Run the original demo as a function (extracted from main()).

    NOTE:
      - This keeps detection_module intact.
      - 'base_offset' lets you align shard-local timestamps to a global timeline.
    """
    os.makedirs("/tmp/yolo_demo", exist_ok=True)
    stem = os.path.splitext(os.path.basename(video))[0]
    events_out = events_out or os.path.join("/tmp/yolo_demo", f"{stem}.events.jsonl")
    spans_out  = spans_out  or os.path.join("/tmp/yolo_demo", f"{stem}.spans.jsonl")

    # 1) raw detection
    events, tracker_used = detect_events_raw(
        video_path=video,
        model_name=model,
        conf=conf,
        iou=iou,
        frame_stride=frame_stride,
        classes=classes,
        device=device,
        enable_tracking=enable_tracking,
    )

    # 1.1) optional global offset for shards
    if base_offset and events:
        for e in events:
            if "ts" in e:
                try:
                    e["ts"] = float(e["ts"]) + float(base_offset)
                except Exception:
                    pass

    # 2) write events
    meta = {
        "video": video,
        "model": model,
        "conf": conf,
        "iou": iou,
        "frame_stride": frame_stride,
        "classes": classes,
        "device": device,
        "base_offset": base_offset,
    "enable_tracking": enable_tracking,
    "tracker": tracker_used,
    }
    write_jsonl(events_out, events, meta=meta)

    # 3) merge to spans and write
    #spans = merge_events_to_spans(events, gap_sec=gap_sec, key_fn=None)
    #write_jsonl(spans_out, spans, meta={**meta, "gap_sec": gap_sec})

    return {
        "video": video,
        "events_jsonl": events_out,
        "spans_jsonl": spans_out,
        "num_events": len(events),
        #"num_spans": len(spans),
    }

def merge_events(
    event_jsonl_paths: List[str],
    spans_out: str,
    gap_sec: float = 10.0,
    key_fields: Tuple[str, ...] = ("label", "cls"),
) -> Dict[str, Any]:
    """
    Merge events *across shards* on the global timeline.

    No normalization is performed. Events are grouped by the exact tuple of
    `key_fields` (default: ("label","cls")). Within each group, consecutive
    events are merged into a span if the time gap <= `gap_sec`.

    Args:
        event_jsonl_paths: list of per-shard events.jsonl paths (already with base_offset applied)
        spans_out: output JSONL for merged spans
        gap_sec: max gap (seconds) to keep events in the same span
        key_fields: fields used as the grouping key (exact match)

    Returns:
        Dict with output path and counts.
    """
    # tiny JSONL reader that skips an optional first meta line
    def _read_jsonl_rows(path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            first = True
            for line in f:
                if first and line.lstrip().startswith('{"__meta__"'):
                    first = False
                    continue
                first = False
                rows.append(json.loads(line))
        return rows

    # collect all events
    all_events: List[Dict[str, Any]] = []
    for p in event_jsonl_paths:
        for e in _read_jsonl_rows(p):
            # ensure `ts` is float; skip if not present/convertible
            try:
                e["ts"] = float(e.get("ts", 0.0))
            except Exception:
                continue
            all_events.append(e)

    # sort globally by time
    all_events.sort(key=lambda x: x["ts"])

    # group by exact key_fields (no normalization)
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for e in all_events:
        key = tuple(e.get(k) for k in key_fields)
        groups[key].append(e)

    # merge within each group
    spans: List[Dict[str, Any]] = []
    for key, evs in groups.items():
        if not evs:
            continue
        evs.sort(key=lambda x: x["ts"])
        start = evs[0]["ts"]
        end = evs[0]["ts"]
        count = 1

        # carry through label/cls if present (no transformation)
        first = evs[0]
        span_header = {}
        for k in ("label", "cls"):
            if k in first:
                span_header[k] = first[k]

        for e in evs[1:]:
            ts = e["ts"]
            if ts - end <= gap_sec:
                end = ts
                count += 1
            else:
                spans.append({
                    **span_header,
                    "start": start,
                    "end": end,
                    "duration": max(0.0, end - start),
                    "count": count,
                })
                start = ts
                end = ts
                count = 1

        # flush last span
        spans.append({
            **span_header,
            "start": start,
            "end": end,
            "duration": max(0.0, end - start),
            "count": count,
        })

    # write JSONL (reuse your writer)
    from utils.write_jsonl import write_jsonl
    os.makedirs(os.path.dirname(spans_out) or ".", exist_ok=True)
    meta = {
        "level": "global",
        "gap_sec": gap_sec,
        "num_events": len(all_events),
        "key_fields": list(key_fields),
        "sources": event_jsonl_paths,
    }
    write_jsonl(spans_out, spans, meta=meta)
    return {"spans_jsonl": spans_out, "num_events": len(all_events), "num_spans": len(spans)}




def main():
    # 1) Detect raw point events (no de-dup/cooldown)
    events = detect_events_raw(
        video_path=VIDEO_PATH,
        model_name=MODEL_NAME,
        conf=CONF,
        iou=IOU,
        frame_stride=FRAME_STRIDE,
        classes=CLASSES,
        device=DEVICE,
        preprocess=None,       # plug your undistort function here if needed
        return_frames=False,
    )

    # 2) Save raw events
    meta = {
        "video": VIDEO_PATH,
        "model": MODEL_NAME,
        "conf": CONF,
        "iou": IOU,
        "frame_stride": FRAME_STRIDE,
        "classes": CLASSES,
        "device": DEVICE,
    }
    write_jsonl(EVENTS_OUT, events, meta=meta)
    print(f"[OK] Wrote raw events: {EVENTS_OUT} ({len(events)} rows)")

    # 3) Merge to spans
    spans = merge_events_to_spans(events, gap_sec=GAP_SEC, key_fn=None)
    write_jsonl(SPANS_OUT, spans, meta={**meta, "gap_sec": GAP_SEC})
    print(f"[OK] Wrote spans: {SPANS_OUT} ({len(spans)} rows)")

if __name__ == "__main__":
    main()
