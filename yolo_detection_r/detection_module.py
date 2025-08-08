
# detection_module.py

#import os, json, time
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
from utils.Tstamp import fmt_hhmmss_ms

# -------------------------------
# Frame generator
# -------------------------------

def frame_generator(video_path: str, frame_stride: int = 1,
                    preprocess: Optional[Callable[[any], any]] = None
                    ) -> Iterator[Tuple[any, int, float]]:
    """
    Yield frames from a video with a stride. Returns (frame, frame_idx, fps).
    - preprocess: optional callable to transform the frame (e.g., undistort).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            if preprocess is not None:
                frame = preprocess(frame)
            yield frame, idx, fps
        idx += 1

    cap.release()


# -------------------------------
# 1) Detection (point events)
# -------------------------------

def detect_events_raw(
    video_path: str,
    model_name: str = "yolov8n.pt",
    conf: float = 0.5,
    iou: float = 0.5,
    frame_stride: int = 5,
    classes: Optional[List[int]] = None,
    device: Optional[str] = None,
    preprocess: Optional[Callable[[any], any]] = None,
    return_frames: bool = False,
) -> List[Dict]:
    """
    Run YOLO on a video (with frame stride) and return per-detection events.
    Each event has: {t, ts, frame, cls, conf, bbox=[x1,y1,x2,y2]}.
    - No de-dup or cooldown here.
    - If you need accurate timestamps, we rely on cap fps and the actual frame index.
    - preprocess: optional frame transform (e.g., undistortion).
    - return_frames: if True, also include width/height of the processed frame.
    """
    # Lazy import to allow using merge without ultralytics installed
    from ultralytics import YOLO

    model = YOLO(model_name)

    events: List[Dict] = []
    gen = list(frame_generator(video_path, frame_stride=frame_stride, preprocess=preprocess))

    frames_only = [frm for (frm, _, _) in gen]
    if not frames_only:
        return events

    preds = model.predict(
        source=frames_only,
        stream=True,
        conf=conf,
        iou=iou,
        device=device,
        classes=classes,
        verbose=False
    )

    for (r, (frame, idx, fps)) in zip(preds, gen):
        t = idx / fps
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss  = r.boxes.cls.cpu().numpy()

        for box, c, cls_id in zip(boxes, confs, clss):
            cls_id = int(cls_id)
            cls_name = r.names.get(cls_id, str(cls_id)) if hasattr(r, "names") else str(cls_id)
            event = {
                "t": round(float(t), 3),
                "ts": fmt_hhmmss_ms(t),
                "frame": int(idx),
                "cls": cls_name,
                "conf": round(float(c), 4),
                "bbox": [round(float(x), 2) for x in box.tolist()],
            }
            if return_frames:
                h, w = frame.shape[:2]
                event["size"] = [int(w), int(h)]
            events.append(event)

    return events


# -------------------------------
# 2) Merge point events -> state spans
# -------------------------------

def merge_events_to_spans(
    events: List[Dict],
    gap_sec: float = 3.0,
    key_fn: Optional[Callable[[Dict], Tuple]] = None,
) -> List[Dict]:
    """
    Merge point events into temporal spans (enter/exit).

    Parameters
    ----------
    events : list of event dicts, each with at least {"t", "cls", "conf"}.
    gap_sec : maximum gap between consecutive points to keep in the same span.
    key_fn : a function mapping an event to a grouping key. Default groups by class only.
             E.g., for future tracking: key_fn = lambda e: (e["cls"], e["track_id"])

    Returns
    -------
    spans : list of dicts with {key..., cls, start, end, duration, max_conf, count}
    """
    if not events:
        return []

    if key_fn is None:
        key_fn = lambda e: (e.get("cls", None),)

    events_sorted = sorted(events, key=lambda e: e["t"])

    spans: List[Dict] = []
    cur_span: Optional[Dict] = None
    cur_key: Optional[Tuple] = None

    for e in events_sorted:
        k = key_fn(e)
        t = float(e["t"])
        conf = float(e.get("conf", 0.0))

        if (cur_span is None) or (k != cur_key) or (t - cur_span["end"] > gap_sec):
            if cur_span is not None:
                cur_span["duration"] = round(cur_span["end"] - cur_span["start"], 3)
                spans.append(cur_span)
            cur_key = k
            cur_span = {
                "cls": e.get("cls"),
                "start": t,
                "end": t,
                "max_conf": conf,
                "count": 1,
            }
            if isinstance(k, tuple):
                for i, val in enumerate(k):
                    cur_span[f"key{i}"] = val
        else:
            cur_span["end"] = t
            cur_span["max_conf"] = max(cur_span["max_conf"], conf)
            cur_span["count"] += 1

    if cur_span is not None:
        cur_span["duration"] = round(cur_span["end"] - cur_span["start"], 3)
        spans.append(cur_span)

    for s in spans:
        s["start_ts"] = fmt_hhmmss_ms(s["start"])
        s["end_ts"] = fmt_hhmmss_ms(s["end"])

    return spans



