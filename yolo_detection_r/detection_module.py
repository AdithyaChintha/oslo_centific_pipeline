# detection_module.py

#import os, json, time
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import logging

import cv2
import numpy as np
import os
import sys
import builtins
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from ultralytics import YOLO
from typing import Any, Dict, List, Optional

from .utils.Tstamp import fmt_hhmmss_ms

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
        raise ValueError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            out_frame = preprocess(frame) if preprocess is not None else frame
            yield out_frame, idx, fps
        idx += 1

    cap.release()


# -------------------------------
# 1) Detection (point events)
# -------------------------------

def detect_events_raw(
    video_path: str,
    model_name: str = "yolo11m.pt",
    conf: float = 0.5,
    iou: float = 0.5,
    frame_stride: int = 5,
    classes: Optional[List[int]] = None,
    device: Optional[str] = None,
    preprocess: Optional[Callable[[any], any]] = None,
    return_frames: bool = False,
    # TRACKING OPTIONS
    enable_tracking: bool = False,
    tracker_backend: str = "botsort",   # 'botsort' means prefer ultralytics track(tracker='botsort.yaml')
    tracker_cfg: Optional[str] = None,  # if using ultralytics.track, pass this (e.g. 'botsort.yaml')
    track_iou_thresh: float = 0.3,
    track_max_age: int = 30,
 ) -> Tuple[List[Dict], str]:
    """
    Run detection on a video and return a list of event dicts.
    If enable_tracking True, attempt ultralytics.model.track (with tracker_cfg) first;
    if not available, fall back to a local IoU tracker.
    """
    events: List[Dict[str, Any]] = []
    used_tracker: str = "none"

    # Load model
    try:
        model = YOLO(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model '{model_name}': {e}")

    # Helper: convert possible torch tensors or lists to numpy arrays
    def _to_numpy(x):
        try:
            return x.cpu().numpy()
        except Exception:
            return np.array(x) if x is not None else np.array([])

    @contextmanager
    def _suppress_output():
        """Suppress stdout/stderr and builtins.print more aggressively."""
        devnull = open(os.devnull, 'w')
        old_stdout, old_stderr = sys.stdout, sys.stderr
        old_stdout_dup, old_stderr_dup = sys.__stdout__, sys.__stderr__
        old_print = builtins.print
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            sys.__stdout__ = devnull
            sys.__stderr__ = devnull
            builtins.print = lambda *a, **k: None
            yield
        finally:
            builtins.print = old_print
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.__stdout__ = old_stdout_dup
            sys.__stderr__ = old_stderr_dup
            devnull.close()

    # Subfunction 1: parse ultralytics.track results into events
    def _parse_ultralytics_track(track_results) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        # determine fps from video for timestamp computation
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        for frame_counter, r in enumerate(track_results):
            boxes = getattr(getattr(r, "boxes", None), "xyxy", None)
            confs = getattr(getattr(r, "boxes", None), "conf", None)
            clss = getattr(getattr(r, "boxes", None), "cls", None)
            tids = getattr(getattr(r, "boxes", None), "id", None)

            boxes = _to_numpy(boxes) if boxes is not None else np.array([])
            confs = _to_numpy(confs) if confs is not None else np.array([])
            clss = _to_numpy(clss) if clss is not None else np.array([])
            tids = _to_numpy(tids) if tids is not None else None

            # prefer names from the result, else model
            names_map = getattr(r, "names", None) or getattr(model, "names", {})

            frame_idx = int(getattr(r, "orig_frame", frame_counter))
            t = frame_idx / fps if fps and fps > 0 else 0.0
            ts = fmt_hhmmss_ms(t)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = [round(float(v), 2) for v in boxes[i]]
                conf_v = float(confs[i]) if i < len(confs) else None
                cls_v = int(clss[i]) if i < len(clss) else None
                track_id = int(tids[i]) if (tids is not None and i < len(tids)) else None
                if classes is not None and cls_v is not None and cls_v not in classes:
                    continue
                cls_name = names_map.get(cls_v, str(cls_v)) if names_map is not None else str(cls_v)
                ev = {
                    "t": round(t, 3),
                    "ts": ts,
                    "frame": int(frame_idx),
                    "cls": cls_name,
                    "conf": round(float(conf_v), 4) if conf_v is not None else None,
                    "bbox": [x1, y1, x2, y2],
                }
                if track_id is not None:
                    ev["track_id"] = int(track_id)
                parsed.append(ev)
        return parsed

    # If user wants to use ultralytics built-in track API and it's available, try it first.
    if enable_tracking and hasattr(model, "track"):
        try:
            cfg = tracker_cfg if tracker_cfg is not None else f"{tracker_backend}.yaml"
            # suppress noisy ultralytics stdout/stderr/print
            with _suppress_output():
                track_results = model.track(source=video_path, tracker=cfg, device=device, conf=conf, iou=iou, verbose=False, show=False)
            used_tracker = f"ultralytics:{cfg}"
            return _parse_ultralytics_track(track_results), used_tracker
        except Exception:
            # if track fails, fall through to fallback
            pass

    # Subfunction 2: fallback per-frame detection + local IoU tracker
    def _run_fallback_tracker() -> List[Dict[str, Any]]:
        from .trackers.simple_iou import IoUTracker

        tracker = IoUTracker(iou_thresh=track_iou_thresh, max_age=track_max_age)
        out_events: List[Dict[str, Any]] = []

        for frame, idx, fps in frame_generator(video_path, frame_stride=frame_stride, preprocess=preprocess):
            # run prediction on single frame
            try:
                with _suppress_output():
                    results = model.predict(source=frame, conf=conf, iou=iou, device=device, verbose=False, show=False)
            except Exception:
                # fallback without verbose args
                with _suppress_output():
                    results = model(frame)

            if len(results) == 0:
                continue
            r = results[0]

            boxes = getattr(getattr(r, "boxes", None), "xyxy", None)
            confs = getattr(getattr(r, "boxes", None), "conf", None)
            clss = getattr(getattr(r, "boxes", None), "cls", None)

            boxes = _to_numpy(boxes) if boxes is not None else np.array([])
            confs = _to_numpy(confs) if confs is not None else np.array([])
            clss = _to_numpy(clss) if clss is not None else np.array([])

            dets_for_tracker = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(float, boxes[i])
                conf_v = float(confs[i]) if i < len(confs) else 0.0
                cls_v = int(clss[i]) if i < len(clss) else None
                if classes is not None and cls_v is not None and cls_v not in classes:
                    continue
                dets_for_tracker.append([x1, y1, x2, y2, conf_v, cls_v])

            track_ids = tracker.update(dets_for_tracker, idx)

            for det_i, det in enumerate(dets_for_tracker):
                # compute timestamp and readable fields
                t = idx / fps if fps and fps > 0 else 0.0
                ts = fmt_hhmmss_ms(t)
                names_map = getattr(r, "names", None) or getattr(model, "names", {})
                cls_id = det[5]
                cls_name = names_map.get(cls_id, str(cls_id)) if names_map is not None else str(cls_id)
                ev = {
                    "t": round(t, 3),
                    "ts": ts,
                    "frame": int(idx),
                    "cls": cls_name,
                    "conf": round(float(det[4]), 4),
                    "bbox": [round(float(det[0]), 2), round(float(det[1]), 2), round(float(det[2]), 2), round(float(det[3]), 2)],
                }
                tid = track_ids[det_i] if det_i < len(track_ids) else None
                if tid is not None:
                    ev["track_id"] = int(tid)
                out_events.append(ev)

        return out_events

    # Run fallback detection/tracking
    logging.warning("Ultralytics track unavailable or failed — falling back to local IoU tracker")
    used_tracker = "local_iou"
    return _run_fallback_tracker(), used_tracker


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



