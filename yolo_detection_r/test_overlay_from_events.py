
# run_overlay_from_events.py
# Draws detection boxes from a JSONL events file onto the original video and saves an annotated video.
#
# Requirements: opencv-python
#
# Expected events.jsonl format (one JSON per line; a meta line optional):
#   {"_meta": {...}}           # optional first line
#   {"t": 12.345, "frame": 370, "cls": "person", "conf": 0.91, "bbox": [x1,y1,x2,y2]}
#
# Works with events produced by detection_module.detect_events_raw(...) in detection_module.py.

import json
import os
import cv2
from collections import defaultdict

# ======= EDIT THESE =======
VIDEO_PATH     = "test_view2.mp4"           # input video path
EVENTS_JSONL   = "outputs/events.jsonl"      # detections exported earlier
OUT_VIDEO      = "outputs/overlay.mp4"       # where to save the annotated video
BOX_THICKNESS  = 2
FONT_SCALE     = 0.5
DRAW_CONF      = True                         # draw confidence in the label
FRAME_TOLERANCE = 1                           # allow off-by-one matching: events at frame +/- tolerance
# ==========================

def _color_for_class(cname: str):
    # Stable color from class name (B,G,R)
    h = abs(hash(cname)) % 255
    return (int((37*h) % 255), int((17*h) % 255), int((191*h) % 255))

def _label_on_frame(frame, text, x1, y1, color, font=cv2.FONT_HERSHEY_SIMPLEX):
    # Draw solid label background and then put text
    (tw, th), bl = cv2.getTextSize(text, font, FONT_SCALE, 1)
    x2 = x1 + tw + 6
    y2 = y1 - th - 6
    y1_bg = max(y1 - th - 8, 0)
    x1_bg = max(x1, 0)
    cv2.rectangle(frame, (x1_bg, y1_bg), (x2, y1), color, thickness=-1)
    cv2.putText(frame, text, (x1_bg+3, y1-4), font, FONT_SCALE, (255,255,255), 1, cv2.LINE_AA)

def load_events_map(jsonl_path):
    """Return: dict[int frame] -> list[dict event], ignores meta lines."""
    events_by_frame = defaultdict(list)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Events file not found: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            # prefer explicit 'frame', else compute from 't' later in main()
            events_by_frame[obj.get("frame", -1)].append(obj)
    return events_by_frame

def main():
    os.makedirs(os.path.dirname(OUT_VIDEO) or ".", exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open VideoWriter for: {OUT_VIDEO}")

    # Load events
    events_by_frame = load_events_map(EVENTS_JSONL)

    # Also build a fallback map by rounding t*fps for events missing 'frame'
    events_missing_frame = [e for fr, L in events_by_frame.items() for e in L if fr == -1]
    if events_missing_frame:
        # rebuild map cleanly
        events_by_frame = {k: v for k, v in events_by_frame.items() if k != -1}
        for e in events_missing_frame:
            t = float(e.get("t", 0.0))
            fr = int(round(t * fps))
            events_by_frame.setdefault(fr, []).append(e)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"[INFO] Video fps={fps:.3f}, size=({w}x{h}), frames={total_frames}")
    print(f"[INFO] Loaded frames with events: {len(events_by_frame)} unique frames")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Gather events for this frame with tolerance
        frame_events = []
        for off in range(-FRAME_TOLERANCE, FRAME_TOLERANCE+1):
            L = events_by_frame.get(idx+off)
            if L:
                frame_events.extend(L)

        # Draw
        for ev in frame_events:
            bbox = ev["bbox"]
            cls  = ev.get("cls", "obj")
            conf = ev.get("conf", None)
            x1,y1,x2,y2 = [int(round(v)) for v in bbox]
            color = _color_for_class(cls)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness=BOX_THICKNESS)
            label = cls if not DRAW_CONF or conf is None else f"{cls} {conf:.2f}"
            _label_on_frame(frame, label, x1, max(y1, 15), color)

        writer.write(frame)

        if idx % 200 == 0:
            print(f"[INFO] Processed {idx}/{total_frames} frames...")
        idx += 1

    cap.release()
    writer.release()
    print(f"[OK] Saved overlay video to: {OUT_VIDEO}")

if __name__ == "__main__":
    main()
