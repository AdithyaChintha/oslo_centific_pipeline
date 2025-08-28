# lighting_by_second.py
import cv2, json, math, os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# ---------- Basic Metrics ----------
def _to_y(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR frame to luminance (Y channel in YUV)."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)[:, :, 0]

def _colorfulness(frame_bgr: np.ndarray) -> float:
    """Compute Hasler & Süsstrunk colorfulness metric."""
    (B, G, R) = cv2.split(frame_bgr.astype(np.float32))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return float(np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * (np.mean(rg) + np.mean(yb)))

def _grayworld_shift(frame_bgr: np.ndarray) -> float:
    """Compute gray-world deviation (distance from equal RGB balance)."""
    m = frame_bgr.reshape(-1, 3).mean(axis=0)
    return float(np.linalg.norm(m - m.mean()))

def _frame_metrics(frame_bgr: np.ndarray,
                   low_sat_cut: int = 5,
                   high_sat_cut: int = 250) -> Dict[str, float]:
    """Compute luminance and color metrics for a single frame."""
    Y = _to_y(frame_bgr)
    flat = Y.reshape(-1)
    p5, p50, p95 = np.percentile(flat, [5, 50, 95])
    return {
        "meanY": float(np.mean(flat)),            # Average luminance
        "stdY": float(np.std(flat)),              # Contrast (RMS)
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "pct_low": float((flat <= low_sat_cut).mean() * 100.0),    # % pixels near black
        "pct_high": float((flat >= high_sat_cut).mean() * 100.0),  # % pixels near white
        "colorfulness": _colorfulness(frame_bgr),
        "grayworld": _grayworld_shift(frame_bgr),
    }

# ---------- Thresholds & Classification ----------
def _derive_thresholds_from_seconds(
    sec_rows,
    *,
    mode: str = "adaptive",                    # "fixed" | "adaptive"
    fixed_bands: Optional[Dict[str, float]] = None,   # {"dark":70,"low":110,"bright":170}
    # explicit overrides
    dark_mean: Optional[float] = None,
    low_mean: Optional[float] = None,
    bright_mean: Optional[float] = None,
    # misc
    merge_min_sec: float = 1.0,
    sat_low_flag: float = 25.0,
    sat_high_flag: float = 25.0,
    # adaptive clamps
    base_bands: Dict[str, float] = {"dark": 70.0, "low": 110.0, "bright": 170.0},
    max_shift: float = 15.0,   # thresholds cannot move more than ±15 from base
) -> Dict[str, float]:
    """
    Derive thresholds for Dark / Low-light / Bright lighting levels.

    Modes:
      - "fixed": use fixed bands (default 70/110/170 or custom).
      - "adaptive": use percentiles (p5, p50, p95), but clamp thresholds so
                    they do not move more than ±max_shift from base_bands.
      - explicit overrides (dark_mean/low_mean/bright_mean) take precedence
        if all three are provided.

    Returns a dict with thresholds and metadata.
    """

    # 1) Explicit overrides always win if all three provided
    if (dark_mean is not None) and (low_mean is not None) and (bright_mean is not None):
        d, l, b = float(dark_mean), float(low_mean), float(bright_mean)
        if l <= d + 1: l = d + 1
        if b <= l + 1: b = l + 1
        return {
            "dark_mean": d, "low_mean": l, "bright_mean": b,
            "merge_min_sec": float(merge_min_sec),
            "sat_low_flag": float(sat_low_flag),
            "sat_high_flag": float(sat_high_flag),
            "mode": "overrides"
        }

    # 2) Fixed thresholds
    if mode == "fixed":
        bands = fixed_bands or base_bands
        d, l, b = float(bands["dark"]), float(bands["low"]), float(bands["bright"])
        if l <= d + 1: l = d + 1
        if b <= l + 1: b = l + 1
        return {
            "dark_mean": d, "low_mean": l, "bright_mean": b,
            "merge_min_sec": float(merge_min_sec),
            "sat_low_flag": float(sat_low_flag),
            "sat_high_flag": float(sat_high_flag),
            "mode": "fixed"
        }

    # 3) Adaptive thresholds from percentiles
    if mode == "adaptive":
        if sec_rows:
            allY = np.array([r["meanY"] for r in sec_rows], dtype=np.float32)
            p5, p50, p95 = np.percentile(allY, [5, 50, 95])
        else:
            p5, p50, p95 = 40.0, 100.0, 180.0

        # clamp each percentile-based threshold around base_bands ± max_shift
        d = float(np.clip(p5 + 5, base_bands["dark"] - max_shift, base_bands["dark"] + max_shift))
        l = float(np.clip(p50,    base_bands["low"]  - max_shift, base_bands["low"]  + max_shift))
        b = float(np.clip(p95 - 10, base_bands["bright"] - max_shift, base_bands["bright"] + max_shift))

        # ensure monotonicity
        if l <= d + 1: l = d + 1
        if b <= l + 1: b = l + 1

        return {
            "dark_mean": d, "low_mean": l, "bright_mean": b,
            "merge_min_sec": float(merge_min_sec),
            "sat_low_flag": float(sat_low_flag),
            "sat_high_flag": float(sat_high_flag),
            "mode": "adaptive",
            "max_shift": float(max_shift)
        }

    # 4) Fallback: use fixed if unknown mode
    bands = base_bands
    return {
        "dark_mean": float(bands["dark"]),
        "low_mean": float(bands["low"]),
        "bright_mean": float(bands["bright"]),
        "merge_min_sec": float(merge_min_sec),
        "sat_low_flag": float(sat_low_flag),
        "sat_high_flag": float(sat_high_flag),
        "mode": "fixed(fallback)"
    }


def _classify_row(row: Dict[str, float], thr: Dict[str, float]) -> str:
    """Assign a lighting label based on thresholds."""
    if row["meanY"] >= thr["bright_mean"] or row["pct_high"] >= 10.0:
        return "Bright"
    if row["meanY"] < thr["dark_mean"]:
        return "Dark"
    if row["meanY"] < thr["low_mean"]:
        return "Low-light"
    return "Normal"

# ---------- Per-second Aggregation ----------
def evaluate_lighting_per_second(
    video_path: str,
    *,
    fps_sample: float = 3.0,
    resize_short: Optional[int] = 480,
    max_frames_per_sec: int = 3,
    low_sat_cut: int = 5,
    high_sat_cut: int = 250,
) -> Dict[str, Any]:
    """
    Sample up to `max_frames_per_sec` frames per second and aggregate them into per-second metrics.

    Args:
        video_path: input video path
        fps_sample: sampling fps (e.g., 3.0)
        resize_short: if set, downscale by making the shorter side = this value (only downscale)
        max_frames_per_sec: at most how many frames to keep per second
        low_sat_cut: pixel value threshold to count as "near black"
        high_sat_cut: pixel value threshold to count as "near white"

    Returns:
        {
          "fps": float,
          "seconds": [
            {"sec": int, "nframes": int, <metrics>...},
            ...
          ]
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sec_buckets: Dict[int, List[Dict[str, float]]] = {}
    step = max(int(round(fps / fps_sample)), 1)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step != 0:
            i += 1
            continue

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        sec = int(np.floor(t))

        if resize_short:
            h, w = frame.shape[:2]
            scale = resize_short / min(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Store limited frames per second
        if sec not in sec_buckets:
            sec_buckets[sec] = []
        if len(sec_buckets[sec]) < max_frames_per_sec:
            sec_buckets[sec].append(_frame_metrics(frame, low_sat_cut, high_sat_cut))

        i += 1
    cap.release()

    seconds_rows: List[Dict[str, Any]] = []
    for sec in sorted(sec_buckets.keys()):
        rows = sec_buckets[sec]
        agg = {k: float(np.median([r[k] for r in rows])) for k in rows[0].keys()}
        seconds_rows.append({"sec": sec, "nframes": len(rows), **agg})

    return {"fps": float(fps), "seconds": seconds_rows}

# ---------- Event Merging ----------
def merge_lighting_events(
    per_second: Dict[str, Any],
    *,
    thresholds: Optional[Dict[str, float]] = None,
    dark_mean: Optional[float] = None,
    low_mean: Optional[float] = None,
    bright_mean: Optional[float] = None,
    merge_min_sec: float = 1.0,
    sat_low_flag: float = 25.0,
    sat_high_flag: float = 25.0,
) -> Dict[str, Any]:
    """
    Merge consecutive seconds with the same lighting label into events.
    If thresholds not provided, derive adaptively (with optional overrides).
    """
    secs = per_second["seconds"]
    if not secs:
        thr = thresholds or _derive_thresholds_from_seconds(
            [], dark_mean=dark_mean, low_mean=low_mean, bright_mean=bright_mean,
            merge_min_sec=merge_min_sec, sat_low_flag=sat_low_flag, sat_high_flag=sat_high_flag
        )
        return {"thresholds": thr, "events": []}

    thr = thresholds or _derive_thresholds_from_seconds(
        secs, dark_mean=dark_mean, low_mean=low_mean, bright_mean=bright_mean,
        merge_min_sec=merge_min_sec, sat_low_flag=sat_low_flag, sat_high_flag=sat_high_flag
    )
    labels = [_classify_row(r, thr) for r in secs]

    events = []
    cur_label = labels[0]
    start_sec = secs[0]["sec"]

    for k in range(1, len(secs)):
        # Break segment if label changes or there is a gap in seconds
        if labels[k] != cur_label or secs[k]["sec"] != secs[k-1]["sec"] + 1:
            end_sec = secs[k-1]["sec"] + 1
            if end_sec - start_sec >= thr["merge_min_sec"]:
                in_rows = [r for r in secs if start_sec <= r["sec"] < end_sec]
                ev = {"start": float(start_sec), "end": float(end_sec), "label": cur_label}
                for key in ["meanY","stdY","p5","p50","p95","pct_low","pct_high","colorfulness","grayworld"]:
                    ev[key + "_med"] = float(np.median([r[key] for r in in_rows]))
                events.append(ev)
            cur_label = labels[k]
            start_sec = secs[k]["sec"]

    # Flush last segment
    end_sec = secs[-1]["sec"] + 1
    if end_sec - start_sec >= thr["merge_min_sec"]:
        in_rows = [r for r in secs if start_sec <= r["sec"] < end_sec]
        ev = {"start": float(start_sec), "end": float(end_sec), "label": cur_label}
        for key in ["meanY","stdY","p5","p50","p95","pct_low","pct_high","colorfulness","grayworld"]:
            ev[key + "_med"] = float(np.median([r[key] for r in in_rows]))
        events.append(ev)

    return {"thresholds": thr, "events": events}

# ---------- Wrapper to Save JSON (now fully configurable) ----------
def run_lighting_two_jsons(
    video_path: str,
    output_dir: Optional[str] = None,   # now optional
    *,
    # sampling & preprocessing
    fps_sample: float = 3.0,
    resize_short: Optional[int] = 480,
    max_frames_per_sec: int = 3,
    low_sat_cut: int = 5,
    high_sat_cut: int = 250,
    # threshold overrides
    thresholds: Optional[Dict[str, float]] = None,
    dark_mean: Optional[float] = None,
    low_mean: Optional[float] = None,
    bright_mean: Optional[float] = None,
    merge_min_sec: float = 1.0,
    sat_low_flag: float = 25.0,
    sat_high_flag: float = 25.0,
) -> Tuple[str, str]:
    """
    Run per-second evaluation and event merging, and save two JSON files:
      - *_lighting_per_second.json
      - *_lighting_events.json

    If output_dir is None, defaults to the same directory as video_path.
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_path) or "."

    os.makedirs(output_dir, exist_ok=True)

    per_sec = evaluate_lighting_per_second(
        video_path,
        fps_sample=fps_sample,
        resize_short=resize_short,
        max_frames_per_sec=max_frames_per_sec,
        low_sat_cut=low_sat_cut,
        high_sat_cut=high_sat_cut,
    )

    merged = merge_lighting_events(
        per_sec,
        thresholds=thresholds,
        dark_mean=dark_mean,
        low_mean=low_mean,
        bright_mean=bright_mean,
        merge_min_sec=merge_min_sec,
        sat_low_flag=sat_low_flag,
        sat_high_flag=sat_high_flag,
    )

    stem = os.path.splitext(os.path.basename(video_path))[0]
    per_second_path = os.path.join(output_dir, f"{stem}_lighting_per_second.json")
    events_path = os.path.join(output_dir, f"{stem}_lighting_events.json")

    with open(per_second_path, "w", encoding="utf-8") as f:
        json.dump(per_sec, f, ensure_ascii=False, indent=2)
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    return per_second_path, events_path
