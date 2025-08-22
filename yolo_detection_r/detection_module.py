# detection_module.py

from contextlib import contextmanager
import os, sys, builtins, signal, logging
import numpy as np
import cv2
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from ultralytics import YOLO
from .utils.Tstamp import fmt_hhmmss_ms
from .utils.frames import frame_generator


def _to_numpy(x):
    """
    Convert tensor-like or list objects to numpy arrays.
    """
    try:
        return x.cpu().numpy()
    except Exception:
        return np.array(x) if x is not None else np.array([])


@contextmanager
def suppress_output():
    """
    Context manager that suppresses stdout, stderr, and print statements.
    Useful for silencing Ultralytics verbose outputs.
    """
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


def _emit_events_from_result(
    r,
    model_names: Dict[int, str],
    idx: int,
    fps: float,
    classes: Optional[List[int]] = None,
    track_ids: Optional[List[Optional[int]]] = None
) -> List[Dict[str, Any]]:
    """
    Convert YOLO result into event dictionaries.

    Parameters:
        r : YOLO result object
        model_names : mapping of class id to class name
        idx : frame index
        fps : frames per second
        classes : filter for specific class ids
        track_ids : optional list of track IDs

    Returns:
        List of event dictionaries with bbox, confidence, class, timestamp, etc.
    """
    events: List[Dict[str, Any]] = []

    boxes = getattr(getattr(r, "boxes", None), "xyxy", None)
    confs = getattr(getattr(r, "boxes", None), "conf", None)
    clss  = getattr(getattr(r, "boxes", None), "cls", None)

    boxes = _to_numpy(boxes) if boxes is not None else np.array([])
    confs = _to_numpy(confs) if confs is not None else np.array([])
    clss  = _to_numpy(clss)  if clss  is not None else np.array([])

    t  = idx / fps if fps and fps > 0 else 0.0
    ts = fmt_hhmmss_ms(t)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = [round(float(v), 2) for v in boxes[i]]
        conf_v = float(confs[i]) if i < len(confs) else None
        cls_v  = int(clss[i])   if i < len(clss)  else None
        if classes is not None and cls_v is not None and cls_v not in classes:
            continue

        cls_name = model_names.get(cls_v, str(cls_v)) if model_names is not None else str(cls_v)
        ev = {
            "t": round(t, 3),
            "ts": ts,
            "frame": int(idx),
            "cls": cls_name,
            "conf": round(float(conf_v), 4) if conf_v is not None else None,
            "bbox": [x1, y1, x2, y2],
        }
        if track_ids is not None and i < len(track_ids) and track_ids[i] is not None:
            ev["track_id"] = int(track_ids[i])
        events.append(ev)

    return events


def _predict_on_frame(model, frame, conf: float, iou: float, device: Optional[str]):
    """
    Run YOLO prediction on a single frame with error handling.
    Returns the first result or None.
    """
    try:
        with suppress_output():
            results = model.predict(source=frame, conf=conf, iou=iou, device=device, verbose=False, show=False)
    except Exception:
        with suppress_output():
            results = model(frame)
    return results[0] if len(results) else None


def _parse_ultralytics_track(track_results, model, video_path: str, classes: Optional[List[int]]) -> List[Dict[str, Any]]:
    """
    Parse results from Ultralytics track() into events.
    """
    parsed: List[Dict[str, Any]] = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    for frame_counter, r in enumerate(track_results):
        names_map = getattr(r, "names", None) or getattr(model, "names", {})
        frame_idx = int(getattr(r, "orig_frame", frame_counter))

        boxes = getattr(getattr(r, "boxes", None), "xyxy", None)
        boxes_np = _to_numpy(boxes) if boxes is not None else np.array([])
        n_boxes = len(boxes_np)

        tids = getattr(getattr(r, "boxes", None), "id", None)
        tids_np = _to_numpy(tids) if tids is not None else None
        track_ids = [int(tids_np[i]) if (tids_np is not None and i < len(tids_np)) else None
                     for i in range(n_boxes)]

        parsed.extend(_emit_events_from_result(r, names_map, frame_idx, fps, classes=classes, track_ids=track_ids))
    return parsed


def _run_detect_only(
    model,
    video_path: str,
    frame_stride: int,
    preprocess: Optional[Callable[[Any], Any]],
    conf: float,
    iou: float,
    device: Optional[str],
    classes: Optional[List[int]],
) -> List[Dict[str, Any]]:
    """
    Perform detection only (no tracking) on a video.
    """
    print(f"[detect_events_raw] mode=detect-only video={video_path} frame_stride={frame_stride}")
    out_events: List[Dict[str, Any]] = []
    for frame, idx, fps in frame_generator(video_path, frame_stride=frame_stride, preprocess=preprocess):
        r = _predict_on_frame(model, frame, conf, iou, device)
        if r is None:
            continue
        names_map = getattr(r, "names", None) or getattr(model, "names", {})
        out_events.extend(_emit_events_from_result(r, names_map, idx, fps, classes=classes))
    print(f"[detect_events_raw] detect-only complete, events={len(out_events)}")
    return out_events


def _try_ultralytics_track(
    model,
    video_path: str,
    cfg: str,
    conf: float,
    iou: float,
    device: Optional[str],
    timeout_sec: int,
    classes: Optional[List[int]],
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Try running Ultralytics track() with a tracker config.
    Returns (events, used_tracker) on success, or (None, reason) on failure.
    """
    if not hasattr(model, "track"):
        return None, "no-track-attr"

    print(f"[detect_events_raw] attempting ultralytics.track cfg={cfg} timeout={timeout_sec}s")

    def _alarm_handler(signum, frame):
        raise TimeoutError("ultralytics.model.track timed out")

    prev_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(int(timeout_sec))
        with suppress_output():
            track_results = model.track(source=video_path, tracker=cfg, device=device, conf=conf, iou=iou, verbose=False, show=False)
        signal.alarm(0)
        parsed = _parse_ultralytics_track(track_results, model, video_path, classes)
        used = f"ultralytics:{cfg}"
        print(f"[detect_events_raw] ultralytics.track succeeded, events={len(parsed)} tracker={used}")
        return parsed, used
    except TimeoutError:
        logging.warning("ultralytics.model.track timed out after %s seconds — falling back", timeout_sec)
        print(f"[detect_events_raw] ultralytics.track timed out after {timeout_sec}s, falling back to local IoU tracker")
        return None, "timeout"
    except Exception:
        logging.exception("ultralytics.model.track failed, falling back to local tracker")
        print("[detect_events_raw] ultralytics.track failed, falling back to local IoU tracker")
        return None, "exception"
    finally:
        try:
            signal.signal(signal.SIGALRM, prev_handler)
        except Exception:
            pass
        signal.alarm(0)


def _run_fallback_tracker(
    model,
    video_path: str,
    frame_stride: int,
    preprocess: Optional[Callable[[Any], Any]],
    conf: float,
    iou: float,
    device: Optional[str],
    classes: Optional[List[int]],
    iou_thresh: float,
    max_age: int,
) -> List[Dict[str, Any]]:
    """
    Run per-frame detection combined with a simple IoU-based tracker.
    Used when Ultralytics track() is unavailable or fails.
    """
    from .trackers.simple_iou import IoUTracker
    tracker = IoUTracker(iou_thresh=iou_thresh, max_age=max_age)

    out_events: List[Dict[str, Any]] = []
    print(f"[detect_events_raw] entering local IoU tracker fallback video={video_path} frame_stride={frame_stride}")

    for frame, idx, fps in frame_generator(video_path, frame_stride=frame_stride, preprocess=preprocess):
        r = _predict_on_frame(model, frame, conf, iou, device)
        if r is None:
            continue

        boxes = getattr(getattr(r, "boxes", None), "xyxy", None)
        confs = getattr(getattr(r, "boxes", None), "conf", None)
        clss  = getattr(getattr(r, "boxes", None), "cls", None)

        boxes = _to_numpy(boxes) if boxes is not None else np.array([])
        confs = _to_numpy(confs) if confs is not None else np.array([])
        clss  = _to_numpy(clss)  if clss  is not None else np.array([])

        dets_for_tracker = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(float, boxes[i])
            conf_v = float(confs[i]) if i < len(confs) else 0.0
            cls_v  = int(clss[i])    if i < len(clss)  else None
            if classes is not None and cls_v is not None and cls_v not in classes:
                continue
            dets_for_tracker.append([x1, y1, x2, y2, conf_v, cls_v])

        track_ids = tracker.update(dets_for_tracker, idx)

        names_map = getattr(r, "names", None) or getattr(model, "names", {})

        class _PseudoBoxes:
            def __init__(self, arr):
                self.xyxy = np.array([d[:4] for d in arr], dtype=float)
                self.conf = np.array([d[4]    for d in arr], dtype=float)
                self.cls  = np.array([d[5]    for d in arr], dtype=float)

        class _PseudoR:
            def __init__(self, boxes, names):
                self.boxes = boxes
                self.names = names

        pseudo_r = _PseudoR(_PseudoBoxes(dets_for_tracker), names_map)
        out_events.extend(_emit_events_from_result(
            pseudo_r, names_map, idx, fps, classes=None,
            track_ids=[tid for tid in track_ids]
        ))

    print(f"[detect_events_raw] local IoU fallback complete, events={len(out_events)}")
    return out_events


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
    enable_tracking: bool = False,
    tracker_backend: str = "botsort",
    tracker_cfg: Optional[str] = None,
    track_iou_thresh: float = 0.3,
    track_max_age: int = 30,
    track_timeout_sec: int = 120,
) -> Tuple[List[Dict], str]:
    """
    Main entry point for detection and tracking on a video.

    Modes:
        - detect-only
        - Ultralytics track() (preferred if available)
        - fallback IoU tracker

    Returns:
        (events, tracker_used)
    """
    try:
        model = YOLO(model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model '{model_name}': {e}")

    if not enable_tracking:
        events = _run_detect_only(
            model=model,
            video_path=video_path,
            frame_stride=frame_stride,
            preprocess=preprocess,
            conf=conf,
            iou=iou,
            device=device,
            classes=classes,
        )
        return events, "none"

    cfg = tracker_cfg if tracker_cfg is not None else f"{tracker_backend}.yaml"
    used_tracker = "local_iou"
    events_track, reason = _try_ultralytics_track(
        model=model,
        video_path=video_path,
        cfg=cfg,
        conf=conf,
        iou=iou,
        device=device,
        timeout_sec=track_timeout_sec,
        classes=classes,
    )
    if events_track is not None:
        return events_track, reason

    logging.warning("Ultralytics track unavailable or failed — falling back to local IoU tracker")
    print("[detect_events_raw] falling back to local IoU tracker")
    events_fallback = _run_fallback_tracker(
        model=model,
        video_path=video_path,
        frame_stride=frame_stride,
        preprocess=preprocess,
        conf=conf,
        iou=iou,
        device=device,
        classes=classes,
        iou_thresh=track_iou_thresh,
        max_age=track_max_age,
    )
    return events_fallback, used_tracker


def merge_events_to_spans(
    events: List[Dict],
    gap_sec: float = 3.0,
    key_fn: Optional[Callable[[Dict], Tuple]] = None,
) -> List[Dict]:
    """
    Merge point events into temporal spans (enter/exit).

    Parameters:
        events : list of event dicts, each with at least {"t", "cls", "conf"}.
        gap_sec : maximum gap between consecutive points to merge.
        key_fn : grouping function for events (default: group by class only).

    Returns:
        List of span dictionaries with class, start, end, duration, max_conf, and count.
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
