#!/usr/bin/env python3
"""
360 Video Toolkit: Fisheye <-> Perspective (GeoCalib-assisted) and ERP -> Perspective

This module provides:

1) Fisheye pipeline (single fisheye circle per frame)
   - GeoCalib-assisted pose estimation (gravity-based, median over multiple timestamps)
   - Functions: geocalib_pose_fov, robust_geocalib, unwarp_fisheye_views

2) Dual-fisheye (360 with two circles per frame)
   - Split into two 180 deg fisheye videos, then call the fisheye pipeline per hemisphere
   - Function: split_dual_fisheye_video_to_two, unwarp_from_dual_fisheye360

3) Equirectangular (ERP, 2:1 panorama) pipeline
   - Direct ERP -> perspective extraction (no GeoCalib needed)
   - Functions: build_map_erp, unwarp_equirectangular_views

Notes
-----
* All comments/docstrings are ASCII only.
* For fisheye, set fish_e2e_fov_deg to the per-fisheye edge-to-edge FoV (typical ~190 deg),
  NOT 360.
* For ERP, do not use fish_e2e_fov_deg; use v_fov_deg for the perspective camera.
* Axes: yaw about +Y, pitch about +X, roll about +Z. Rotation order: Rz @ Rx @ Ry.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, List
from pathlib import Path

import cv2
import numpy as np
from math import tan, atan, radians, degrees

# ------------------------- math / geometry helpers -------------------------

def rot_matrix(yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Build rotation matrix R = Rz(roll) @ Rx(pitch) @ Ry(yaw)."""
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


def build_map_fisheye(in_h: int, in_w: int,
                      out_w: int, out_h: int,
                      v_fov_deg: float,
                      fish_total_fov_deg: float,
                      model: str = "equisolid",
                      yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0,
                      border: int = 8,
                      cx: Optional[float] = None, cy: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute remap grids from fisheye to perspective."""
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
    fx = fy

    xs = (np.arange(out_w, dtype=np.float32) + 0.5 - out_w * 0.5) / fx
    ys = (np.arange(out_h, dtype=np.float32) + 0.5 - out_h * 0.5) / fy
    xv, yv = np.meshgrid(xs, ys)

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
    else:
        r = 2.0 * f_fish * np.sin(0.5 * theta)

    map_x = (cx + r * np.cos(phi)).astype(np.float32)
    map_y = (cy + r * np.sin(phi)).astype(np.float32)
    return map_x, map_y


def open_writer(path: str, fps: float, size: Tuple[int, int], prefer_codec: str = "mp4v") -> cv2.VideoWriter:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*prefer_codec)
    out = cv2.VideoWriter(path, fourcc, fps, size)
    if not out.isOpened():
        alt = str(Path(path).with_suffix(".avi"))
        out = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"XVID"), fps, size)
        print("[WARN] mp4v failed; writing AVI instead ->", alt)
    return out

# ----------------------- GeoCalib integration (fisheye) -------------------

def _read_simple_divisional(cam) -> Tuple[float, float, float, float, float, float, float]:
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
    """Estimate pitch/roll/v_fov/cx/cy from a fisheye frame using GeoCalib.

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
            from geocalib.utils import rad2deg
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

    # Apply mapping to this script's axes (fixes L/C/R becoming U/C/D)
    #if map_to_script_axes and (pitch is not None) and (roll is not None):
    #    pitch = 90.0 - pitch
    #    roll  = roll - 90.0

    if pitch is None: pitch = 0.0
    if roll  is None: roll  = 0.0

    return {"pitch": pitch, "roll": roll, "v_fov": v_fov_in, "cx": cx, "cy": cy}


def robust_geocalib(cap: cv2.VideoCapture,
                    fps: float,
                    sample_ts: Sequence[float],
                    model: str = "simple_divisional") -> Dict[str, float]:
    """Estimate pose/FoV/cx,cy at several timestamps and return medians."""
    pitch_list: List[float] = []
    roll_list:  List[float] = []
    vfof_list:  List[float] = []
    cx_list:    List[float] = []
    cy_list:    List[float] = []

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

# ----------------------------- public API: fisheye ------------------------

def make_even_views(n: int, start_yaw: float = 0.0, clockwise: bool = True, prefix: str = "view") -> List[Tuple[str, float]]:
    """Generate n views evenly spaced in yaw (degrees) around the horizon."""
    if n <= 0:
        raise ValueError("n must be >= 1")
    step = 360.0 / float(n)
    out: List[Tuple[str, float]] = []
    for i in range(n):
        yaw = start_yaw + (-step * i if clockwise else step * i)
        yaw = ((yaw + 180.0) % 360.0) - 180.0  # normalize to [-180, 180)
        name = f"{prefix}{i}_{int(round(yaw))}"
        out.append((name, float(yaw)))
    return out


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
    """Unwarp a circular fisheye video into multiple perspective view videos."""
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

    # Pose and fisheye center
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
            cx_used = float(agg["cx"]); cy_used = float(agg["cy"])

    # Decide perspective vertical FoV
    out_w, out_h = int(out_size[0]), int(out_size[1])
    if (h_fov_deg is not None) and (v_fov_deg is not None):
        raise ValueError("Set either h_fov_deg or v_fov_deg, not both.")
    if (h_fov_deg is None) and (v_fov_deg is None) and (v_fov_geo is not None):
        v_fov = float(v_fov_geo)
    else:
        v_fov = v_from_h(h_fov_deg, out_w, out_h) if v_fov_deg is None else float(v_fov_deg)

    # Writers and maps
    if out_dir is None:
        out_dir = str(Path(mp4_path).with_name(Path(mp4_path).stem + "_views"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    writers: Dict[str, cv2.VideoWriter] = {}
    outputs: Dict[str, str] = {}

    for name, yaw in views:
        map_x, map_y = build_map_fisheye(
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

# ------------------------- dual-fisheye (360) helpers ---------------------


# ----------------------------- ERP pipeline -------------------------------

def build_map_erp(out_w: int, out_h: int,
                  v_fov_deg: float,
                  yaw_deg: float, pitch_deg: float,
                  erp_W: int, erp_H: int,
                  roll_deg: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Build remap grids from equirectangular (ERP) to perspective."""
    v_fov = np.deg2rad(v_fov_deg)
    fy = (out_h * 0.5) / np.tan(v_fov * 0.5)
    fx = fy
    xs = (np.arange(out_w, dtype=np.float32) + 0.5 - out_w * 0.5) / fx
    ys = (np.arange(out_h, dtype=np.float32) + 0.5 - out_h * 0.5) / fy
    xv, yv = np.meshgrid(xs, ys)
    zv = np.ones_like(xv)
    dirs = np.stack([xv, -yv, zv], axis=-1)
    dirs /= (np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-9)

    # Apply yaw(Y), pitch(X), optional roll(Z) for ERP views
    R = rot_matrix(yaw_deg, pitch_deg, roll_deg)
    dirs = dirs @ R.T

    x, y_, z = dirs[...,0], dirs[...,1], dirs[...,2]
    lam = np.arctan2(x, z)                 # [-pi, pi)
    phi = np.arcsin(np.clip(y_, -1.0, 1.0))# [-pi/2, pi/2]
    u = (lam + np.pi) / (2*np.pi) * erp_W
    v = (np.pi/2 - phi) / np.pi * erp_H

    return u.astype(np.float32), v.astype(np.float32)


def unwarp_equirectangular_views(mp4_path: str,
                                 views: Sequence[Tuple[str, float, float]] = [("front",0,0),("right",90,0),("back",180,0),("left",-90,0)],
                                 out_size: Tuple[int,int] = (1280,720),
                                 v_fov_deg: float = 90.0,
                                 out_dir: Optional[str] = None,
                                 roll_deg: float = 0.0) -> Dict[str,str]:
    """Extract perspective views from equirectangular 360 video.
    views: list of (name, yaw_deg, pitch_deg)
    roll_deg rotates the output view around its optical axis.
    """
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = out_size

    if out_dir is None:
        out_dir = str(Path(mp4_path).with_name(Path(mp4_path).stem + "_views_erp"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    maps = {}
    writers = {}
    outputs = {}
    for name, yaw, pitch in views:
        mx, my = build_map_erp(out_w, out_h, v_fov_deg, yaw, pitch, W, H, roll_deg=roll_deg)
        maps[name] = (mx, my)
        out_path = str(Path(out_dir) / f"{Path(mp4_path).stem}_{name}_{out_w}x{out_h}.mp4")
        writers[name] = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        outputs[name] = out_path

    ok, frame = cap.read()
    frames = 0
    while ok:
        for name, (mx, my) in maps.items():
            view = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            writers[name].write(view)
        frames += 1
        if frames % 100 == 0:
            print(f"[ERP] written {frames} frames")
        ok, frame = cap.read()

    cap.release()
    for w in writers.values(): w.release()
    print(f"[ERP] saved {frames} frames to {out_dir}")
    return outputs

# ------------------------------ TEST runner -------------------------------
if __name__ == "__main__":
    # Example: ERP 4-view test (comment out if you prefer fisheye test)
    test_mp4 = "test360.mp4"
    views4 = [("front",0,0),("right",90,0),("back",180,0),("left",-90,0)]
    out = unwarp_equirectangular_views(test_mp4, views4, out_size=(1280,720), v_fov_deg=90.0)
    print("[ERP TEST]", out)

