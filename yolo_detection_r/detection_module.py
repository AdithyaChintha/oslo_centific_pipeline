# detection_module.py

from contextlib import contextmanager
import os, sys, builtins, signal, logging
import numpy as np
import cv2
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from ultralytics import YOLO
from .utils.Tstamp import fmt_hhmmss_ms
from .utils.frames import frame_generator
from typing import Iterable, Set, NamedTuple, Union
from collections import defaultdict


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

# ============ People Count: fixed-interval binning (no line/ROI) ============

def people_count_bins_from_events(
    events: List[Dict[str, Any]],
    bin_size_sec: float = 5.0,
    person_classes: Optional[Union[Set[Union[str, int]], List[Union[str, int]]]] = {"person"},
) -> Dict[str, Any]:
    """
    Aggregate per-frame people counts into fixed time bins.

    Workflow:
      1) For each frame: deduplicate by track_id → get the number of people
         simultaneously present (occupancy).
      2) Partition the timeline into fixed bins of length `bin_size_sec`
         (e.g., 5.0 means [0,5), [5,10), ...). For each bin, compute:
           - avg_persons: average occupancy across frames in this bin.
           - max_persons: maximum occupancy observed in this bin.
      3) Return a list of bins (`timeline_bins`) plus overall summary metrics.

    Args:
      events (List[Dict]): Event dicts produced by `detect_events_raw`,
        each containing at least {frame, t, ts, cls, track_id}.
      bin_size_sec (float): Size of each time bin in seconds.
      person_classes (set or list, optional): Which classes to count.
        Defaults to {"person"}. Can also pass {0} depending on whether
        your `cls` field stores names or integer IDs.

    Returns:
      Dict with keys:
        "summary": {
            "bins": number of bins,
            "bin_size_sec": size of each bin (seconds),
            "avg_of_avg": average of all per-bin averages,
            "max_of_max": maximum of all per-bin maxima,
            "unique_people": total distinct track_ids across the whole video
        },
        "timeline_bins": [
           {
             "bin_index": 0,
             "start": 0.0,
             "end": 5.0,
             "start_ts": "...",
             "end_ts": "...",
             "frames": 42,
             "avg_persons": 2.6,
             "max_persons": 3
           },
           ...
        ]
    """
    if not events:
        return {"summary": {"bins": 0, "avg_of_avg": 0.0, "max_of_max": 0, "unique_people": 0},
                "timeline_bins": []}

    # Default to only counting "person" class
    if person_classes is None:
        person_classes = {"person"}
    person_classes = set(person_classes)

    # Filter events: keep only desired classes with valid frame and track_id
    ev_person = []
    for e in events:
        cls_v = e.get("cls")
        # Accept if cls matches either by name or by integer id
        if (cls_v in person_classes) or (isinstance(cls_v, int) and cls_v in person_classes):
            if e.get("track_id") is not None and e.get("frame") is not None:
                ev_person.append(e)

    if not ev_person:
        return {"summary": {"bins": 0, "avg_of_avg": 0.0, "max_of_max": 0, "unique_people": 0},
                "timeline_bins": []}

    from collections import defaultdict
    import numpy as np

    # 1) Build per-frame occupancy
    frame_to_ids = defaultdict(set)
    frame_meta = {}  # store (t, ts) for each frame
    for e in ev_person:
        f = int(e["frame"])
        frame_to_ids[f].add(int(e["track_id"]))
        if f not in frame_meta:
            frame_meta[f] = {"t": float(e.get("t", 0.0)), "ts": e.get("ts")}

    frames_sorted = sorted(frame_to_ids.keys())
    if not frames_sorted:
        return {"summary": {"bins": 0, "avg_of_avg": 0.0, "max_of_max": 0, "unique_people": 0},
                "timeline_bins": []}

    # 2) Aggregate by time bin
    # Use the frame timestamp `t` to determine bin membership
    bins = defaultdict(list)   # bin_index -> list of per-frame person counts
    bin_time = {}              # bin_index -> {"start", "end", "start_ts", "end_ts"}
    for f in frames_sorted:
        meta = frame_meta[f]
        t = float(meta["t"])
        persons = len(frame_to_ids[f])
        b = int(t // bin_size_sec)  # non-overlapping bins: [0,5), [5,10), ...
        bins[b].append(persons)
        # Track bin metadata (numeric boundaries + approx ts using first/last frame)
        if b not in bin_time:
            bin_time[b] = {"start": b * bin_size_sec,
                           "end": (b + 1) * bin_size_sec,
                           "start_ts": None, "end_ts": None}
        if bin_time[b]["start_ts"] is None:
            bin_time[b]["start_ts"] = meta["ts"]
        bin_time[b]["end_ts"] = meta["ts"]

    # 3) Construct timeline_bins
    timeline_bins = []
    for b in sorted(bins.keys()):
        arr = np.array(bins[b], dtype=float)
        avg_persons = float(arr.mean()) if len(arr) else 0.0
        max_persons = int(arr.max()) if len(arr) else 0
        info = bin_time[b]
        timeline_bins.append({
            "bin_index": b,
            "start": round(info["start"], 3),
            "end": round(info["end"], 3),
            "start_ts": info["start_ts"],
            "end_ts": info["end_ts"],
            "frames": int(len(arr)),
            "avg_persons": round(avg_persons, 3),
            "max_persons": max_persons,
        })

    # Compute summary metrics
    unique_people = len({int(e["track_id"]) for e in ev_person})
    avg_of_avg = float(np.mean([b["avg_persons"] for b in timeline_bins])) if timeline_bins else 0.0
    max_of_max = int(max([b["max_persons"] for b in timeline_bins])) if timeline_bins else 0

    return {
        "summary": {
            "bins": int(len(timeline_bins)),
            "bin_size_sec": float(bin_size_sec),
            "avg_of_avg": round(avg_of_avg, 3),   # average of all per-bin averages
            "max_of_max": max_of_max,             # maximum across all per-bin maxima
            "unique_people": int(unique_people),  # distinct individuals in the whole video
        },
        "timeline_bins": timeline_bins
    }


def people_presence_spans_from_events(
    events: List[Dict[str, Any]],
    person_classes: Optional[List[Any]] = {"person"},
    gap_sec: float = 1.0,
    min_duration_sec: float = 3.0,
) -> Dict[str, Any]:
    """
    Build per-person (track_id) presence spans from detection/tracking events.

    Algorithm:
      - Filter to target classes (default: {"person"}).
      - Group by track_id and sort frames by time t.
      - Merge consecutive frames into a span while the gap between adjacent
        frames <= gap_sec; otherwise start a new span.

    Args:
      events: List of event dicts containing at least {t, ts, frame, cls, track_id}.
      person_classes: Which classes to consider as "people". Defaults to {"person"}.
                      Can also pass {0} if cls is integer-coded.
      gap_sec: Max allowed gap (seconds) between consecutive frames to keep within
               the same presence span.
        min_duration_sec: Minimum duration (seconds) for a presence span to be
                            counted towards the "people_number" summary metric.

    Returns:
      {
        "summary": {
           "total_tracks": <num of distinct track_id>,
           "total_spans": <num of spans across all tracks>,
           "min_duration_sec": <as passed in>,
           "people_number": <num of distinct track_id that have at least one span
                             whose duration >= min_duration_sec>
        },
        "tracks": [
           {
             "track_id": 7,
             "spans": [
                {"start": 0.12, "end": 4.96, "start_ts": "00:00:00.120", "end_ts": "00:00:04.960", "frames": 112},
                {"start": 8.00, "end": 10.04, "start_ts": "00:00:08.000", "end_ts": "00:00:10.040", "frames": 61}
             ]
           },
           ...
        ]
      }
    """
    if not events:
        return {"summary": {"total_tracks": 0, "total_spans": 0, "people_number": 0}, "tracks": []}

    if person_classes is None:
        person_classes = {"person"}
    person_classes = set(person_classes)

    # 1) filter valid person events with track_id and time
    by_tid = defaultdict(list)
    for e in events:
        cls_v = e.get("cls")
        if not ((cls_v in person_classes) or (isinstance(cls_v, int) and cls_v in person_classes)):
            continue
        tid = e.get("track_id")
        t = e.get("t")
        if tid is None or t is None:
            continue
        try:
            tid_i = int(tid)
            t_f = float(t)
        except Exception:
            continue
        by_tid[tid_i].append({
            "t": t_f,
            "ts": e.get("ts"),
            "frame": int(e.get("frame", 0))
        })

    tracks_out = []
    total_spans = 0

    # 2) build spans per track, then filter by min_duration_sec
    for tid, rows in by_tid.items():
        rows.sort(key=lambda r: r["t"])
        spans = []
        cur = None  # {"start":.., "end":.., "start_ts":.., "end_ts":.., "frames":..}

        for r in rows:
            t = r["t"]
            ts = r["ts"]
            if cur is None:
                cur = {"start": t, "end": t, "start_ts": ts, "end_ts": ts, "frames": 1}
                continue

            if (t - cur["end"]) <= gap_sec:
                # same span
                cur["end"] = t
                cur["end_ts"] = ts
                cur["frames"] += 1
            else:
                # close previous, start new
                spans.append(cur)
                cur = {"start": t, "end": t, "start_ts": ts, "end_ts": ts, "frames": 1}

        if cur is not None:
            spans.append(cur)

        # Keep all spans in the output for this track
        tracks_out.append({"track_id": tid, "spans": spans})
        total_spans += len(spans)

    # people_number: number of distinct tracks that have at least one span
    # whose duration >= min_duration_sec
    people_number = 0
    for t in tracks_out:
        has_long = False
        for s in t.get('spans', []):
            dur = float(s.get('end', 0.0)) - float(s.get('start', 0.0))
            if dur >= min_duration_sec:
                has_long = True
                break
        if has_long:
            people_number += 1

    return {
        "summary": {
            "total_tracks": len(tracks_out),
            "total_spans": total_spans,
            "min_duration_sec": float(min_duration_sec),
            "people_number": people_number,
        },
        "tracks": tracks_out
    }
