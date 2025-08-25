#!/usr/bin/env python3
"""
Fisheye -> Multi-View Unwarp (GeoCalib-assisted)

This module exposes a single entry point:

    unwarp_fisheye_views(mp4_path: str,
                         out_dir: Optional[str] = None,
                         out_size: Tuple[int, int] = (1280, 720),
                         model: str = "equisolid",
                         fish_e2e_fov_deg: float = 190.0,
                         border: int = 10,
                         auto_from_geo: bool = True,
                         geo_model: str = "simple_divisional",
                         sample_ts: Sequence[float] = (0.0, 1.0, 2.0),
                         manual_pitch_deg: float = 0.0,
                         manual_roll_deg: float  = 0.0,
                         h_fov_deg: Optional[float] = 120.0,
                         v_fov_deg: Optional[float] = None,
                         views: Optional[Sequence[Tuple[str, float]]] = None) -> Dict[str, str]

It writes three perspective-view videos (default: left60 / front0 / right60) to out_dir
and returns a dict {view_name: output_path}.

Notes
-----
* All comments are ASCII-only to avoid encoding issues.
* If auto_from_geo is True, the function estimates (pitch, roll, v_fov, cx, cy)
  from a few frames using GeoCalib and aggregates via median.
* If GeoCalib is unavailable or fails, it falls back to manual_pitch_deg/roll_deg.
* The function uses yaw about +Y, pitch about +X, roll about +Z, and rotation order Rz@Rx@Ry
  (applied to view rays).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple
from pathlib import Path

import cv2
import numpy as np
from math import tan, atan, radians, degrees


# ------------------------- math / geometry helpers -------------------------

def rot_matrix(yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Build rotation matrix R = Rz(roll) @ Rx(pitch) @ Ry(yaw).

    Axes convention:
      - yaw around +Y
      - pitch around +X
      - roll around +Z
    """
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    Ry = np.array([[ np.cos(y), 0, np.sin(y)],
                   [         0, 1,         0],
                   [-np.sin(y), 0, np.cos(y)]], np.float32)
    Rp = np.array([[1,          0,           0],
                   [0,  np.cos(p), -np.sin(p)],
                   [0,  np.sin(p),  np.cos(p)]], np.float32)
    Rr = np.array([[ np.cos(r), -np.sin(r), 0],
                   [ np.sin(r),  np.cos(r), 0],
                   [         0,          0, 1]], np.float32)
    return Rr @ Rp @ Ry


def v_from_h(hfov_deg: float, W: int, H: int) -> float:
    """Convert horizontal FoV (deg) to vertical FoV (deg) for given output size."""
    return 2.0 * degrees(atan((H / float(W)) * tan(radians(hfov_deg * 0.5))))


def build_map(in_h: int, in_w: int,
              out_w: int, out_h: int,
              v_fov_deg: float,
              fish_total_fov_deg: float,
              model: str = "equisolid",
              yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
              border: int = 8,
              cx: Optional[float] = None, cy: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute mapping grids (map_x, map_y) for cv2.remap.

    - in_h, in_w: size of the circular fisheye frame
    - out_w, out_h: perspective output size
    - v_fov_deg: vertical FoV of the perspective virtual camera
    - fish_total_fov_deg: total fisheye FoV edge-to-edge (e.g., 180..200)
    - model: "equisolid" or "equidistant"
    - yaw/pitch/roll: view orientation in degrees
    - border: shrink the sampling radius to avoid black ring
    - cx, cy: fisheye circle center in pixels (default: image center)
    """
    cx = (in_w * 0.5) if cx is None else float(cx)
    cy = (in_h * 0.5) if cy is None else float(cy)
    R  = min(cx, cy) - float(border)

    theta_max = np.deg2rad(fish_total_fov_deg * 0.5)
    if model == "equidistant":
        f_fish = R / theta_max
    elif model == "equisolid":
        f_fish = R / (2.0 * np.sin(theta_max * 0.5))
    else:
        raise ValueError("model must be 'equisolid' or 'equidistant'")

    fy = (out_h * 0.5) / np.tan(np.deg2rad(v_fov_deg) * 0.5)
    fx = fy  # square pixels

    xs = (np.arange(out_w, dtype=np.float32) + 0.5 - out_w * 0.5) / fx
    ys = (np.arange(out_h, dtype=np.float32) + 0.5 - out_h * 0.5) / fy
    xv, yv = np.meshgrid(xs, ys)  # shape (H,W)

    dz = np.ones_like(xv)
    dirs = np.stack([xv, yv, dz], axis=-1)
    dirs /= (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-9)

    Rmat = rot_matrix(yaw, pitch, roll)
    dirs = dirs @ Rmat.T

    dz = np.clip(dirs[..., 2], -1.0, 1.0)
    theta = np.arccos(dz)
    phi   = np.arctan2(dirs[..., 1], dirs[..., 0])

    if model == "equidistant":
        r = f_fish * theta
    else:  # equisolid
        r = 2.0 * f_fish * np.sin(0.5 * theta)

    map_x = (cx + r * np.cos(phi)).astype(np.float32)
    map_y = (cy + r * np.sin(phi)).astype(np.float32)
    return map_x, map_y


def open_writer(path: str, fps: float, size: Tuple[int, int], prefer_codec: str = "mp4v") -> cv2.VideoWriter:
    """Open a cv2 VideoWriter with fallback to AVI if mp4v is not available."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*prefer_codec)
    out = cv2.VideoWriter(path, fourcc, fps, size)
    if not out.isOpened():
        alt = str(Path(path).with_suffix(".avi"))
        out = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"XVID"), fps, size)
        print("[WARN] mp4v failed; writing AVI instead ->", alt)
    return out


# ----------------------- GeoCalib integration helpers ----------------------

def _read_simple_divisional(cam) -> Tuple[float, float, float, float, float, float, float]:
    """Extract (W, H, fx, fy, cx, cy, k1) from GeoCalib's SimpleDivisional camera."""
    import torch
    data = getattr(cam, "_data", None)
    if data is None:
        raise RuntimeError("GeoCalib camera has no _data")
    if isinstance(data, torch.Tensor):
        vals = data.detach().cpu().numpy().reshape(-1).astype(np.float32)
    else:
        vals = np.array(data, dtype=np.float32).reshape(-1)
    if vals.size < 7:
        raise RuntimeError("Unexpected camera vector length: %d" % vals.size)
    W, H, fx, fy, cx, cy, k1 = map(float, vals[:7])
    return W, H, fx, fy, cx, cy, k1


def geocalib_pose_fov(frame_bgr: np.ndarray,
                      model: str = "simple_divisional",
                      map_to_script_axes: bool = True) -> Dict[str, float]:
    """Estimate pitch/roll/v_fov/cx/cy using GeoCalib from a single frame.

    Behavior:
      1) Prefer gravity.rp (order: roll, pitch) like the official demo.
      2) If gravity.rp is missing, fall back to gravity vector g=(gx,gy,gz):
           roll  = atan2(gx, gy)
           pitch = atan2(-gz, hypot(gx, gy))
      3) v_fov is read from camera.vfov; if unavailable, compute from (H, fy).
      4) If map_to_script_axes is True, apply mapping:
           pitch = 90 - pitch; roll = roll - 90
    """
    try:
        from geocalib import GeoCalib
        from geocalib.utils import rad2deg
        import torch
    except Exception as e:
        raise RuntimeError("GeoCalib is required for auto estimation: %s" % e)

    tmp_dir = Path("outputs/_geocalib_tmp"); tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_img = tmp_dir / "frame0.jpg"
    cv2.imwrite(str(tmp_img), frame_bgr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    geo = GeoCalib(weights="distorted").to(device)
    img = geo.load_image(str(tmp_img)).to(device)
    res = geo.calibrate(img, camera_model=model)

    cam = res["camera"]
    grav = res.get("gravity", None)

    # v_fov
    v_fov_in = None
    try:
        v_fov_in = float(rad2deg(cam.vfov))
    except Exception:
        try:
            Wg, Hg, fx, fy, cx_tmp, cy_tmp, _ = _read_simple_divisional(cam)
            v_fov_in = float(np.degrees(2.0 * np.arctan(Hg / (2.0 * fy))))
        except Exception:
            v_fov_in = None

    # principal point
    try:
        _, _, _, _, cx, cy, _ = _read_simple_divisional(cam)
    except Exception:
        cx, cy = None, None

    # roll/pitch
    pitch, roll = None, None
    if grav is not None and hasattr(grav, "rp"):
        try:
            roll_t, pitch_t = rad2deg(grav.rp).unbind(-1)  # (roll, pitch)
            roll  = float(getattr(roll_t, "item", lambda: roll_t)())
            pitch = float(getattr(pitch_t, "item", lambda: pitch_t)())
        except Exception:
            pass

    if (pitch is None or roll is None) and grav is not None:
        g = grav
        if hasattr(g, "detach"):
            g = g.detach().cpu().numpy()
        gx, gy, gz = map(float, np.array(g).reshape(-1)[:3])
        roll  = np.degrees(np.arctan2(gx, gy))
        pitch = np.degrees(np.arctan2(-gz, (gx * gx + gy * gy) ** 0.5))

    if isinstance(res, dict):
        if res.get("pitch") is not None:
            try: pitch = float(res["pitch"])  # allow explicit override
            except Exception: pass
        if res.get("roll") is not None:
            try: roll = float(res["roll"])   # allow explicit override
            except Exception: pass

    if pitch is None: pitch = 0.0
    if roll  is None: roll  = 0.0

    return {"pitch": pitch, "roll": roll, "v_fov": v_fov_in, "cx": cx, "cy": cy}


def robust_geocalib(cap: cv2.VideoCapture,
                    fps: float,
                    sample_ts: Sequence[float],
                    model: str = "simple_divisional") -> Dict[str, float]:
    """Estimate pose/FoV/cx,cy at several timestamps and return medians."""
    pitch_list: list[float] = []
    roll_list:  list[float] = []
    vfof_list:  list[float] = []
    cx_list:    list[float] = []
    cy_list:    list[float] = []

    for t in sample_ts:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, f = cap.read()
        if not ok:
            print(f"[GEO][WARN][t={t:.1f}s] failed to fetch frame")
            continue
        try:
            est = geocalib_pose_fov(f, model=model, map_to_script_axes=True)
            p, r = est["pitch"], est["roll"]
            pitch_list.append(p)
            roll_list.append(r)
            if est["v_fov"] is not None: vfof_list.append(float(est["v_fov"]))
            if est["cx"] is not None:    cx_list.append(float(est["cx"]))
            if est["cy"] is not None:    cy_list.append(float(est["cy"]))
            print(f"[GEO][t={t:.1f}s] used(pitch={p:.2f}, roll={r:.2f}), vFoV_in={est['v_fov']}, cx={est['cx']}, cy={est['cy']}")
        except Exception as e:
            print(f"[GEO][WARN][t={t:.1f}s] estimation failed: {e}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def med(lst: Sequence[float]) -> Optional[float]:
        return float(np.median(lst)) if len(lst) else None

    return {
        "pitch": med(pitch_list),
        "roll":  med(roll_list),
        "v_fov": med(vfof_list),
        "cx":    med(cx_list),
        "cy":    med(cy_list),
        "n":     len(pitch_list),
    }


# ----------------------------- public API ---------------------------------

def unwarp_fisheye_views(mp4_path: str,
                         out_dir: Optional[str] = None,
                         out_size: Tuple[int, int] = (1280, 720),
                         model: str = "equisolid",
                         fish_e2e_fov_deg: float = 190.0,
                         border: int = 10,
                         auto_from_geo: bool = True,
                         geo_model: str = "simple_divisional",
                         sample_ts: Sequence[float] = (0.0, 1.0, 2.0),
                         manual_pitch_deg: float = 0.0,
                         manual_roll_deg: float  = 0.0,
                         h_fov_deg: Optional[float] = 120.0,
                         v_fov_deg: Optional[float] = None,
                         views: Optional[Sequence[Tuple[str, float]]] = None) -> Dict[str, str]:
    """Unwarp a circular fisheye video into multiple perspective view videos.

    Args:
        mp4_path: input video path (fisheye circle assumed centered in frame).
        out_dir: output directory; default next to input as '<stem>_views'.
        out_size: (W, H) of each perspective view (e.g., (1280, 720)).
        model: 'equisolid' or 'equidistant' fisheye projection model.
        fish_e2e_fov_deg: fisheye edge-to-edge FoV in degrees (e.g., 180..200).
        border: shrink radius to avoid black outer ring.
        auto_from_geo: if True, estimate pose/FoV with GeoCalib using sample_ts.
        geo_model: GeoCalib camera model (e.g., 'simple_divisional').
        sample_ts: seconds to sample for robust estimation.
        manual_pitch_deg, manual_roll_deg: fallback pose if auto fails.
        h_fov_deg / v_fov_deg: choose exactly one to define the perspective FoV.
        views: list of (name, yaw_deg). Default: [('left60', -60), ('front0', 0), ('right60', 60)].

    Returns:
        dict mapping view name -> output video path.
    """
    mp4_path = str(mp4_path)
    if not Path(mp4_path).exists():
        raise FileNotFoundError(mp4_path)

    if views is None:
        views = [("left60", -60.0), ("front0", 0.0), ("right60", 60.0)]

    in_cap = cv2.VideoCapture(mp4_path)
    if not in_cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    fps = in_cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Estimate pose and center
    cx_used = in_w * 0.5
    cy_used = in_h * 0.5
    pitch_used = float(manual_pitch_deg)
    roll_used  = float(manual_roll_deg)
    v_fov_geo  = None

    if auto_from_geo:
        agg = robust_geocalib(in_cap, fps, sample_ts, model=geo_model)
        if agg.get("pitch") is not None and agg.get("roll") is not None:
            pitch_used = float(agg["pitch"])
            roll_used  = float(agg["roll"])
        if agg.get("v_fov") is not None:
            v_fov_geo = float(agg["v_fov"])
        if agg.get("cx") is not None and agg.get("cy") is not None:
            cx_used = float(agg["cx"])
            cy_used = float(agg["cy"])

    # Decide perspective vertical FoV
    out_w, out_h = int(out_size[0]), int(out_size[1])
    if (h_fov_deg is not None) and (v_fov_deg is not None):
        raise ValueError("Set either h_fov_deg or v_fov_deg, not both.")
    if (h_fov_deg is None) and (v_fov_deg is None) and (v_fov_geo is not None):
        v_fov = float(v_fov_geo)
    else:
        v_fov = v_from_h(h_fov_deg, out_w, out_h) if v_fov_deg is None else float(v_fov_deg)

    # Prepare output writers and maps
    if out_dir is None:
        out_dir = str(Path(mp4_path).with_name(Path(mp4_path).stem + "_views"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    writers: Dict[str, cv2.VideoWriter] = {}
    outputs: Dict[str, str] = {}

    for name, yaw in views:
        map_x, map_y = build_map(
            in_h, in_w, out_w, out_h, v_fov,
            fish_e2e_fov_deg, model,
            yaw=float(yaw), pitch=pitch_used, roll=roll_used,
            border=border, cx=cx_used, cy=cy_used,
        )
        maps[name] = (map_x, map_y)

        out_path = str(Path(out_dir) / f"{Path(mp4_path).stem}_{name}_{out_w}x{out_h}.mp4")
        writers[name] = open_writer(out_path, fps, (out_w, out_h))
        outputs[name] = out_path
        print(f"[VIEW] name={name}, yaw={yaw:.1f}, pitch={pitch_used:.2f}, roll={roll_used:.2f}, vFoV={v_fov:.2f}")

    # Process frames
    frames = 0
    ok, frame = in_cap.read()
    while ok:
        for name, (map_x, map_y) in maps.items():
            rect = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            writers[name].write(rect)
        frames += 1
        if frames % 100 == 0:
            print(f"[INFO] written {frames} frames")
        ok, frame = in_cap.read()

    in_cap.release()
    for w in writers.values():
        w.release()

    print(f"[DONE] saved {frames} frames to folder: {Path(out_dir).resolve()}")
    return outputs


# ------------------------------ CLI runner --------------------------------
if __name__ == "__main__":
    # Simple test without CLI. It will read a local file named 'test_view2.mp4'.
    test_mp4 = "test_view2.mp4"


    outputs = unwarp_fisheye_views(mp4_path=test_mp4)

    print("[TEST] Outputs:", outputs)
