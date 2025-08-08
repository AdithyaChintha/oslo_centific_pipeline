
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

from detection_module import detect_events_raw, merge_events_to_spans
from utils.write_jsonl import write_jsonl

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
