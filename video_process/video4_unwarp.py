#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INSV -> True ERP (2:1) -> four 90° perspective views (front/right/back/left)

- Step 1 (FFmpeg):  read the INSV, detect one- or two-lens streams.
  * Two streams: hstack them (dual-fisheye) and convert to ERP (equirectangular).
  * One stream: convert the fisheye to ERP (will cover ~180° only).
- Step 2 (OpenCV):  remap ERP to four rectilinear views using spherical sampling.
  This avoids v360's perspective quirks on your machine.

Outputs are written to the current working directory.

Dependencies:
  - ffmpeg available in PATH
  - pip install opencv-python numpy
"""

import math
import os
import json
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np

# ========= Configuration =========
# Change this to your input file if needed:
INSV = Path("/tmp/pipeline_input_y5ps5g9z/VID_20250809_094836_00_045.insv")

# Lens FOV of the Insta360 fisheye (typical 190–200 deg)
LENS_FOV_DEG = 190.0

# Target ERP resolution (must be 2:1)
ERP_W, ERP_H = 5760, 2880

# If your video looks upside-down, set 180.0; otherwise set 0.0
ROLL_DEG = 0.0

# Each perspective view (width, height)
OUT_SIZE = (1440, 1440)

# Vertical field of view for each rectilinear window
V_FOV_DEG = 90.0

# Four yaw directions (left uses -90, not 270; back uses -180)
VIEWS = [(45, "front"), (135, "right"), (225, "back"), (-45, "left")]

# OpenCV writer codec (mp4v is widely supported; switch to avc1 if you have H.264 encoder installed)
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
# =================================


def run(cmd: list) -> None:
    """Run a subprocess and raise on failure."""
    subprocess.run(cmd, check=True)


def ffprobe_video_streams(path: Path):
    """Return FFprobe video stream info as a list of dicts."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=index,codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(out).get("streams", [])


def make_true_erp_from_insv(insv: Path) -> Path:
    """
    Convert an INSV to a true ERP (2:1) MP4 and return its path.

    Two-stream case:   [0:v:0][0:v:1] -> hstack -> v360 dfisheye -> equirect
    One-stream case:   v360 fisheye -> equirect   (covers ~180°)
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    streams = ffprobe_video_streams(insv)
    n_v = len(streams)
    out = Path.cwd() / f"{insv.stem}_equirect_true.mp4"

    if n_v >= 2:
        # Dual-lens: build a dual-fisheye frame and convert to ERP
        vf = (
            f"[0:v:0][0:v:1]hstack=inputs=2[dual];"
            f"[dual]v360=input=dfisheye:output=equirect:"
            f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}[v]"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(insv),
            "-filter_complex",
            vf,
            "-map",
            "[v]",
            "-map",
            "0:a",  # Map audio streams
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-c:a",
            "copy",  # Copy audio
            "-movflags",
            "+faststart",
            str(out),
        ]
    else:
        # Single-lens fisheye to ERP (half sphere coverage)
        vf = (
            f"v360=input=fisheye:output=equirect:"
            f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(insv),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-c:a",
            "copy",  # Copy audio
            "-movflags",
            "+faststart",
            str(out),
        ]

    run(cmd)
    print(f"[OK] ERP written: {out.name}  ({ERP_W}x{ERP_H})")
    return out


# ===== OpenCV: ERP -> perspective views (no v360 used here) =====
def _euler_to_R(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build a rotation matrix Rz(roll) @ Rx(pitch) @ Ry(yaw) from degrees."""
    y, p, r = map(lambda a: math.radians(float(a)), (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Rx @ Ry).astype(np.float32)


def _build_map(
    Wo: int,
    Ho: int,
    v_fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    Wi: int,
    Hi: int,
):
    """
    Construct sampling maps (map_x, map_y) to remap from ERP to a rectilinear view.

    Conventions:
      - Camera looks along +Z.
      - lon = atan2(x, z), lat = asin(y)
      - ERP pixel (u, v):
          u = (lon / (2*pi) + 0.5) * Wi
          v = (0.5 - lat / pi) * Hi
    """
    # Horizontal FOV derived from vertical FOV and aspect ratio
    v = math.radians(v_fov_deg)
    aspect = Wo / Ho
    h = 2.0 * math.atan(math.tan(v / 2.0) * aspect)

    # Build normalized imaging plane in camera coords
    xs = np.linspace(-math.tan(h / 2.0), math.tan(h / 2.0), Wo, dtype=np.float32)
    ys = np.linspace(-math.tan(v / 2.0), math.tan(v / 2.0), Ho, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    zv = np.ones_like(xv, dtype=np.float32)

    denom = np.sqrt(xv * xv + yv * yv + zv * zv)
    xv, yv, zv = xv / denom, yv / denom, zv / denom
    dirs = np.stack([xv, yv, zv], axis=-1)  # (H, W, 3)

    # Rotate to world coords
    R = _euler_to_R(yaw_deg, pitch_deg, roll_deg)
    dirs_world = dirs @ R.T
    dx = dirs_world[..., 0]
    dy = dirs_world[..., 1]
    dz = dirs_world[..., 2]

    # World direction -> ERP lon/lat -> pixel coords
    lon = np.arctan2(dx, dz)  # [-pi, pi]
    lat = np.arcsin(np.clip(dy, -1, 1))  # [-pi/2, pi/2]

    map_x = (lon / (2 * np.pi) + 0.5) * Wi
    map_y = (0.5 - lat / np.pi) * Hi

    # Wrap longitude and clamp latitude
    map_x = np.mod(map_x, Wi).astype(np.float32)
    map_y = np.clip(map_y, 0, Hi - 1).astype(np.float32)
    return map_x, map_y


def split_erp_to_views(erp_path: Path) -> None:
    """Read ERP and write four rectilinear 90° views."""
    cap = cv2.VideoCapture(str(erp_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {erp_path}")
    Wi = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if abs(Wi / Hi - 2.0) > 0.05:
        print(f"[WARN] {erp_path.name} is not 2:1 ({Wi}x{Hi}). Is this a real ERP?")

    Wo, Ho = OUT_SIZE

    # Precompute maps for each view
    maps = []
    for yaw, name in VIEWS:
        yaw_n = ((float(yaw) + 180.0) % 360.0) - 180.0
        mx, my = _build_map(Wo, Ho, V_FOV_DEG, yaw_n, 0.0, ROLL_DEG, Wi, Hi)
        maps.append((name, mx, my))

    # Create writers
    writers = {
        name: cv2.VideoWriter(
            f"{erp_path.stem}_{name}_{Wo}x{Ho}.mp4", FOURCC, fps, (Wo, Ho)
        )
        for name, _, _ in maps
    }

    print(
        f"[INFO] ERP {Wi}x{Hi} -> views {Wo}x{Ho}, vFOV={V_FOV_DEG}, roll={ROLL_DEG}"
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for name, mx, my in maps:
                view = cv2.remap(
                    frame,
                    mx,
                    my,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP,
                )
                writers[name].write(view)
    finally:
        cap.release()
        for w in writers.values():
            w.release()

    print(
        "[DONE] wrote:",
        ", ".join([f"{erp_path.stem}_{n}_{Wo}x{Ho}.mp4" for n, _, _ in maps]),
    )

def video4views_unwarp(
    insv_path: str,
    output_dir: str = None,
    erp_w: int = ERP_W,
    erp_h: int = ERP_H,
    lens_fov_deg: float = LENS_FOV_DEG,
    out_size: tuple = OUT_SIZE,
    v_fov_deg: float = V_FOV_DEG,
    roll_deg: float = ROLL_DEG,
    views: list = None,
) -> dict:
    """
    Convert INSV -> ERP -> four 90° perspective views and return output paths.

    Args:
        insv_path: path to the input .insv file
        output_dir: directory where outputs will be placed (defaults to cwd)

    Returns:
        dict with keys: success (bool), erp (str), views (dict name->path) or error
    """
    try:
        insv = Path(insv_path)
        if not insv.exists():
            return {"success": False, "error": f"Input not found: {insv_path}"}

        out_dir = Path(output_dir) if output_dir else Path.cwd()
        # Apply provided parameters to module-level defaults so existing helpers use them
        global LENS_FOV_DEG, ERP_W, ERP_H, OUT_SIZE, V_FOV_DEG, ROLL_DEG, VIEWS
        LENS_FOV_DEG = float(lens_fov_deg)
        ERP_W = int(erp_w)
        ERP_H = int(erp_h)
        OUT_SIZE = tuple(out_size)
        V_FOV_DEG = float(v_fov_deg)
        ROLL_DEG = float(roll_deg)
        if views is not None:
            # Expect views as list of (yaw, name) tuples or similar; normalize to our VIEWS format
            try:
                VIEWS = [(v[0], v[1]) for v in views]
            except Exception:
                # fallback to default if malformed
                VIEWS = [(45, "front"), (135, "right"), (225, "back"), (-45, "left")]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create ERP (function writes to current working dir). We'll generate then move into out_dir.
        erp = make_true_erp_from_insv(insv)
        dest_erp = out_dir / erp.name
        if erp.resolve() != dest_erp.resolve():
            shutil.move(str(erp), str(dest_erp))
            erp = dest_erp

        # Split ERP into views inside out_dir
        orig_cwd = Path.cwd()
        try:
            os.chdir(out_dir)
            split_erp_to_views(erp)
        finally:
            os.chdir(orig_cwd)

        # Collect produced view files
        produced = {}
        for _, name in VIEWS:
            fname = f"{erp.stem}_{name}_{OUT_SIZE[0]}x{OUT_SIZE[1]}.mp4"
            p = out_dir / fname
            produced[name] = str(p) if p.exists() else None

        return {"success": True, "erp": str(erp), "views": produced}

    except Exception as e:
        return {"success": False, "error": str(e)}


import subprocess
from pathlib import Path
import os

def video4views_unwarpF(
    insv_path: str,
    output_dir: str = None,
    out_size: tuple = (1440, 1440),
    h_fov: int = 90,
    v_fov: int = 90,
    roll: int = 0,
) -> dict:
    """
    Convert a dual-stream INSV file into four 90° perspective views
    using FFmpeg dual-fisheye → rectilinear projection.

    Args:
        insv_path: path to the input .insv file (with 2 fisheye streams).
        output_dir: directory for outputs (default = cwd).
        out_size: (w,h) of output views.
        h_fov: horizontal FOV for v360 rectilinear.
        v_fov: vertical FOV.
        roll: roll correction (deg).

    Returns:
        dict: { "success": bool,
                "dual": path to dual_fisheye.mp4,
                "views": {front, left, right, back},
                "error": str (if failed)}
    """
    try:
        insv = Path(insv_path)
        if not insv.exists():
            return {"success": False, "error": f"Input not found: {insv_path}"}

        out_dir = Path(output_dir) if output_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)

        dual_path = out_dir / "dual_fisheye.mp4"
        w, h = out_size

        # Step 1: combine two streams into SBS dual_fisheye
        # 'libx264'（CPU），'h264_nvenc'（NVIDIA），'hevc_nvenc' 
        cmd_sbs = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", str(insv),
            "-filter_complex",
            "[0:v:0]scale=-1:ih,fps=30[va];"
            "[0:v:1]scale=-1:ih,fps=30[vb];"
            "[va][vb]hstack=inputs=2[sbs]",
            "-map", "[sbs]", "-map", "0:a",  # Map video and audio
            "-c:v", "h264_nvenc", "-crf", "18", "-preset", "veryfast", 
            "-c:a", "copy",  # Copy audio instead of removing it
            str(dual_path)
        ]

        try:
            subprocess.run(cmd_sbs, check=True)
        except subprocess.CalledProcessError as e:
            # NVENC failed (commonly driver/API mismatch). Retry with libx264 as fallback.
            try:
                fallback_cmd = cmd_sbs.copy()
                # replace codec and keep other flags including audio
                for i, v in enumerate(fallback_cmd):
                    if v == 'h264_nvenc':
                        fallback_cmd[i] = 'libx264'
                        break
                subprocess.run(fallback_cmd, check=True)
            except subprocess.CalledProcessError:
                # re-raise original for outer handler
                raise e

        # Step 2: four rectilinear views
        front = out_dir / f"front_{w}x{h}.mp4"
        left  = out_dir / f"left_{w}x{h}.mp4"
        right = out_dir / f"right_{w}x{h}.mp4"
        back  = out_dir / f"back_{w}x{h}.mp4"

        filter_complex = (
            f"[0:v]split=2[l][r];"
            f"[l]crop=iw/2:ih:0:0,split=2[l0][l1];"
            f"[r]crop=iw/2:ih:iw/2:0,split=2[r0][r1];"
            f"[l0]v360=input=fisheye:output=rectilinear:"
            f"h_fov={h_fov}:v_fov={v_fov}:yaw=45:pitch=0:roll={roll}:w={w}:h={h}[front];"
            f"[l1]v360=input=fisheye:output=rectilinear:"
            f"h_fov={h_fov}:v_fov={v_fov}:yaw=-45:pitch=0:roll={roll}:w={w}:h={h}[left];"
            f"[r0]v360=input=fisheye:output=rectilinear:"
            f"h_fov={h_fov}:v_fov={v_fov}:yaw=-45:pitch=0:roll={roll}:w={w}:h={h}[right];"
            f"[r1]v360=input=fisheye:output=rectilinear:"
            f"h_fov={h_fov}:v_fov={v_fov}:yaw=45:pitch=0:roll={roll}:w={w}:h={h}[back]"
        )

        cmd_views = [
            "ffmpeg", "-y", "-hide_banner",
            "-i", str(dual_path),
            "-filter_complex", filter_complex,
            "-map", "[front]", "-map", "0:a", "-c:a", "copy", str(front),
            "-map", "[left]", "-map", "0:a", "-c:a", "copy", str(left),
            "-map", "[right]", "-map", "0:a", "-c:a", "copy", str(right),
            "-map", "[back]", "-map", "0:a", "-c:a", "copy", str(back),
        ]
        subprocess.run(cmd_views, check=True)

        return {
            "success": True,
            "dual": str(dual_path),
            "views": {
                "front": str(front),
                "left": str(left),
                "right": str(right),
                "back": str(back),
            },
        }

    except Exception as e:
        return {"success": False, "error": str(e)}



def main():
    result = video4views_unwarpF(
    "VID_20250809_094836_00_045.insv",
    output_dir="out_4views",
    out_size=(1440,1440),
    h_fov=90, v_fov=90,
    roll=180)
    test = 1

if __name__ == "__main__":
    main()
