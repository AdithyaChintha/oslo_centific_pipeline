#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from typing import List, Optional
import ray

# ---------- helpers ----------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _basename_noext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def _run(cmd: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def _scale_cuda_available() -> bool:
    try:
        out = _run(["ffmpeg", "-hide_banner", "-filters"], timeout=10)
        return out.returncode == 0 and ("scale_cuda" in out.stdout)
    except Exception:
        return False

def _ffprobe_duration(path: str) -> float:
    out = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], timeout=10)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return float(out.stdout.strip())

# ---------- public API ----------
@ray.remote
def split_video_into_shards(
    video_path: str,
    output_dir: str = "/tmp/shards",
    duration_sec: int = 60,
    target_height: int = 480,
    use_gpu: bool = True,
    shards_per_gpu: float = 2.0,   # e.g., 2 shards per GPU => num_gpus=0.5
) -> List[str]:
    """
    Shard a large video; for H100/A100 (no NVENC) we do:
      NVDEC + scale_cuda + hwdownload + format=nv12 -> libx264 (yuv420p)
    """
    p = str(Path(video_path))
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    _ensure_dir(output_dir)

    total_duration = _ffprobe_duration(p)

    # Submit per-shard Ray tasks
    futures = []
    t = 0.0
    idx = 0
    num_gpus_per_task = max(1.0 / max(shards_per_gpu, 0.01), 0.01) if use_gpu else 0.0

    while t < total_duration - 1e-6:
        outp = os.path.join(output_dir, f"{_basename_noext(p)}_part{idx}.mp4")
        dur = min(duration_sec, max(0.0, total_duration - t))
        opts = {"num_gpus": num_gpus_per_task} if use_gpu else {}

        futures.append(_shard_task.options(**opts).remote(
            video_path=p,
            start_time=t,
            duration=dur,
            output_path=outp,
            target_height=target_height,
            use_gpu=use_gpu,
        ))
        t += duration_sec
        idx += 1

    results = ray.get(futures)
    return [r for r in results if r is not None]

# ---------- per-shard task ----------
@ray.remote
def _shard_task(
    video_path: str,
    start_time: float,
    duration: float,
    output_path: str,
    target_height: int,
    use_gpu: bool,
) -> Optional[str]:
    """
    H100-safe path:
      If use_gpu:
        try CUDA scaler:  -vf "scale_cuda=-2:H,hwdownload,format=nv12"
        else CPU scaler:  -vf "scale=-2:H"
      Always CPU encode: libx264 -pix_fmt yuv420p
    """
    REENCODE_AUDIO = os.getenv("SPLITTER_REENCODE_AUDIO", "0") == "1"
    audio_args = ["-c:a", "aac", "-b:a", "128k"] if REENCODE_AUDIO else ["-c:a", "copy"]

    vf_args: List[str] = []
    if target_height and target_height > 0:
        if use_gpu and _scale_cuda_available():
            vf_args = ["-vf", f"scale_cuda=-2:{target_height},hwdownload,format=nv12"]
        else:
            vf_args = ["-vf", f"scale=-2:{target_height}"]

    base_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_time}", "-i", video_path, "-t", f"{duration}",
        *vf_args,
        "-c:v", "libx264", "-preset", os.getenv("SPLITTER_X264_PRESET", "veryfast"),
        "-crf", os.getenv("SPLITTER_X264_CRF", "23"),
        "-pix_fmt", "yuv420p",
        *audio_args, "-movflags", "+faststart",
        output_path,
    ]

    # If we want GPU decode, prepend hwaccel args.
    if use_gpu:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-ss", f"{start_time}", "-i", video_path, "-t", f"{duration}",
            *vf_args,
            "-c:v", "libx264", "-preset", os.getenv("SPLITTER_X264_PRESET", "veryfast"),
            "-crf", os.getenv("SPLITTER_X264_CRF", "23"),
            "-pix_fmt", "yuv420p",
            *audio_args, "-movflags", "+faststart",
            output_path,
        ]
    else:
        cmd = base_cmd

    try:
        print(f"🔧 shard {Path(output_path).name} | GPU={'on' if use_gpu else 'off'} | vf={' '.join(vf_args) or 'none'}")
        res = _run(cmd, timeout=3600)
        if res.returncode == 0:
            return output_path
        print("ffmpeg stderr:", res.stderr[:400])
        return None
    except Exception as e:
        print("shard error:", e)
        return None

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Shard + downscale using H100-safe pipeline")
    ap.add_argument("video", type=str)
    ap.add_argument("out", type=str)
    ap.add_argument("--dur", type=int, default=60)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--gpu", action="store_true", help="Use GPU decode/scale")
    ap.add_argument("--shards-per-gpu", type=float, default=2.0, help="Parallel shards per GPU (default 2)")
    args = ap.parse_args()

    if not ray.is_initialized():
        ray.init()

    fut = split_video_into_shards.remote(
        video_path=args.video,
        output_dir=args.out,
        duration_sec=args.dur,
        target_height=args.h,
        use_gpu=args.gpu,
        shards_per_gpu=args.shards_per_gpu,
    )
    for p in ray.get(fut):
        print(p)
