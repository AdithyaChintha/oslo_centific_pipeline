#!/usr/bin/env python3
"""
stitch_mux.py — Batch stitch Insta360 dual‑fisheye + Zoom H3‑VR FOA audio to H.264 MP4.

Usage examples:
  # Process everything in a folder (auto-groups by activity+timestamp)
  python stitch_mux.py /path/to/sdcard

  # Or pass explicit files (mix of video .insv parts and audio .wav parts)
  python stitch_mux.py cooking-lunch-20250714_1432-video.insv audio-cooking-lunch-20250714_1432_1.wav audio-cooking-lunch-20250714_1432_2.wav

What it does for each (activity, timestamp) group:
  1) Sort multi-part files (video .insv, audio .wav) by their numeric suffix (…_1, …_2, …).
  2) FFmpeg filtergraph (single run) that:
     - For each video part: [v0][v1] -> hstack -> v360 (dfisheye→equirect) at 7680x3840.
     - Concats all equirect parts into one video stream.
     - For each audio part: assumes AmbiX FOA channel order (W,Y,Z,X) @ 4ch;
       downmixes to stereo with a simple L/R = 0.707*W ± 0.5*X (naive, horizontal-only).
       Concats all stereo parts, resamples as needed.
     - setpts=PTS-STARTPTS everywhere to “sync-start” the timelines.
     - -shortest so the mux stops when either audio or video ends.
  3) Writes H.264/AAC MP4 named “[activity]-[timestamp].mp4”.

Assumptions / Notes:
  • Insta360 .insv contains TWO video streams (v:0 and v:1) per file (dual fisheye).
  • Zoom H3‑VR WAV is AmbiX (ACN/SN3D): channels W,Y,Z,X = c0,c1,c2,c3.
    The downmix here is intentionally simple (no HRTF/rotation); swap/scale to taste.
  • If no audio parts are provided, the script will produce a video-only MP4.
  • Requires ffmpeg/ffprobe in PATH.

"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ray

# Optional progress bars
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Global verbosity flag (set in main())
VERBOSE: bool = False
# Optional librosa-based sync (default path)
try:  # pragma: no cover
    import librosa as _librosa  # type: ignore
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _librosa = None  # type: ignore


# ---------- Console Colors ----------


def _supports_color() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def _c(text: str, color: str) -> str:
    if not _supports_color():
        return text
    codes = {
        "red": "\x1b[31m",
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "blue": "\x1b[34m",
        "magenta": "\x1b[35m",
        "cyan": "\x1b[36m",
        "bold": "\x1b[1m",
    }
    reset = "\x1b[0m"
    return f"{codes.get(color, '')}{text}{reset}"


def _log(msg: str):
    """Print a log line without disrupting active tqdm bars."""
    if tqdm is not None and not VERBOSE:
        try:
            tqdm.write(msg)
            return
        except Exception:
            pass
    print(msg)


# ---------- Parsing & Grouping ----------

# Supported filename patterns (we tolerate both "-audio" suffix and "audio-" prefix styles).
_PATTERNS = [
    # Standard: [activity]-[timestamp]-video.insv or -audio.wav, optional _part
    re.compile(
        r"""^(?P<activity>.+)-(?P<ts>\d{8}_\d{4})-(?P<kind>video|audio)"""
        r"""(?:_(?P<part>\d+))?"""
        r"""\.(?P<ext>insv|wav)$""",
        re.IGNORECASE,
    ),
    # Prefixed: audio-[activity]-[timestamp]_part.wav  (video- prefix supported if seen)
    re.compile(
        r"""^(?P<kind>audio|video)-(?P<activity>.+)-(?P<ts>\d{8}_\d{4})"""
        r"""(?:_(?P<part>\d+))?"""
        r"""\.(?P<ext>insv|wav)$""",
        re.IGNORECASE,
    ),
]


def _normalize_activity_for_grouping(raw_activity: str) -> str:
    """
    Normalize the activity prefix used for grouping.

    New capture filenames may include leading IDs before the activity:
      [home_id]-[participant_id]-[device_id]-[activity]-[timestamp]-{video|audio}.*

    In such cases, the device_id (3rd segment) may differ between audio and video.
    To robustly group corresponding audio/video, we drop the 3rd segment if there
    are at least 4 hyphen-separated segments in the prefix before the timestamp.

    Examples:
      raw_activity = "home123-part456-cam789-cooking-lunch" ->
        "home123-part456-cooking-lunch"

      raw_activity = "cooking-lunch" (legacy format) -> unchanged
    """
    try:
        parts = raw_activity.split("-")
        if len(parts) >= 4:
            # Remove the 3rd segment (index 2), preserving the rest
            parts_wo_device = parts[:2] + parts[3:]
            return "-".join(parts_wo_device)
        return raw_activity
    except Exception:
        return raw_activity


@dataclass(frozen=True)
class MediaFile:
    path: Path
    activity: str
    timestamp: str
    kind: str  # 'video' or 'audio'
    part: int
    ext: str  # 'insv' or 'wav'


def parse_media_file(p: Path) -> Optional[MediaFile]:
    name = p.name
    for pat in _PATTERNS:
        m = pat.match(name)
        if m:
            gd = m.groupdict()
            activity = gd["activity"]
            ts = gd["ts"]
            kind = gd["kind"].lower()
            part = int(gd["part"]) if gd.get("part") else 1
            ext = gd["ext"].lower()
            # Normalize: if ext and kind disagree, prefer kind for logical grouping
            if ext == "insv":
                kind = "video"
            elif ext == "wav":
                kind = "audio"
            return MediaFile(
                path=p.resolve(),
                activity=activity,
                timestamp=ts,
                kind=kind,
                part=part,
                ext=ext,
            )
    return None


def group_media(paths: List[Path]) -> Dict[Tuple[str, str], Dict[str, List[MediaFile]]]:
    groups: Dict[Tuple[str, str], Dict[str, List[MediaFile]]] = {}
    for p in paths:
        mf = parse_media_file(p)
        if not mf:
            continue
        norm_activity = _normalize_activity_for_grouping(mf.activity)
        key = (norm_activity, mf.timestamp)
        if key not in groups:
            groups[key] = {"video": [], "audio": []}
        groups[key][mf.kind].append(mf)

    # Sort parts within each kind
    for key in groups:
        for k in ["video", "audio"]:
            groups[key][k].sort(key=lambda mf: mf.part)
    return groups


# ---------- FFmpeg Helpers ----------


def check_ffmpeg():
    for bin_name in ("ffmpeg", "ffprobe"):
        if not shutil.which(bin_name):
            raise RuntimeError(f"Required binary '{bin_name}' not found in PATH.")


def ensure_command(bin_name: str):
    """Raise RuntimeError if command is not available in PATH."""
    if not shutil.which(bin_name):
        raise RuntimeError(f"Required tool '{bin_name}' not found in PATH.")


def run_ffmpeg(cmd: List[str], dry_run: bool = False, description: str = "Encoding"):
    pretty = " \\\n  ".join(shlex.quote(c) for c in cmd)
    if dry_run or VERBOSE:
        print(f"\n[ffmpeg] running:\n{pretty}\n", file=sys.stderr)
    if dry_run:
        return 0
    if tqdm is not None and not VERBOSE:
        pbar = tqdm(total=None, desc=description, unit="step")  # type: ignore[arg-type]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout_buf: List[str] = []
            stderr_buf: List[str] = []
            assert proc.stdout is not None and proc.stderr is not None
            while True:
                line_o = proc.stdout.readline()
                line_e = proc.stderr.readline()
                if line_o:
                    stdout_buf.append(line_o)
                if line_e:
                    stderr_buf.append(line_e)
                if proc.poll() is not None:
                    stdout_buf.extend(proc.stdout.readlines())
                    stderr_buf.extend(proc.stderr.readlines())
                    break
                pbar.update(1)
                time.sleep(0.2)
            ret = proc.returncode or 0
        finally:
            pbar.close()
        if ret != 0:
            sys.stderr.write("".join(stdout_buf))
            sys.stderr.write("".join(stderr_buf))
            raise RuntimeError(f"ffmpeg failed with code {ret}")
        return 0
    else:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if cp.returncode != 0:
            sys.stderr.write(cp.stdout)
            sys.stderr.write(cp.stderr)
            raise RuntimeError(f"ffmpeg failed with code {cp.returncode}")
        return 0


def probe_duration_seconds(path: Path) -> float:
    """Return duration in seconds for the first stream (best-effort)."""
    try:
        cp = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if cp.returncode != 0:
            return 0.0
        return max(0.0, float(cp.stdout.strip()))
    except Exception:
        return 0.0


def probe_fps(path: Path) -> float:
    """Return average FPS for the first video stream (fallback to 30.0)."""
    try:
        cp = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if cp.returncode != 0:
            return 30.0
        rate = cp.stdout.strip()
        if "/" in rate:
            num, den = rate.split("/", 1)
            n = float(num)
            d = float(den) if float(den) != 0 else 1.0
            val = n / d
            return val if val > 0 else 30.0
        val = float(rate)
        return val if val > 0 else 30.0
    except Exception:
        return 30.0


def run_ffmpeg_with_progress(
    cmd: List[str],
    expected_seconds: Optional[float],
    *,
    expected_frames: Optional[int] = None,
    dry_run: bool = False,
    description: str = "Encoding",
):
    pretty = " \\\n+  ".join(shlex.quote(c) for c in cmd)
    if dry_run or VERBOSE:
        print(f"\n[ffmpeg] running:\n{pretty}\n", file=sys.stderr)
    if dry_run:
        return 0

    full_cmd = cmd.copy()
    # Insert -progress pipe:1 and -nostats as global options after executable
    full_cmd[1:1] = ["-progress", "pipe:1", "-nostats"]

    proc = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    pbar = None
    # Prefer frames-based progress if we have a total
    use_frames = expected_frames is not None and expected_frames > 0
    last_val = 0.0
    if use_frames:
        # mypy-safe narrowing
        ef = expected_frames if expected_frames is not None else 0
        total_val: Optional[float] = float(ef)
    else:
        total_val = expected_seconds if expected_seconds and expected_seconds > 0 else None
    unit = "frame" if use_frames else "s"
    if tqdm is not None and not VERBOSE:
        try:
            pbar = tqdm(total=total_val, unit=unit, desc=description, leave=False, colour="yellow")  # type: ignore[arg-type]
        except Exception:
            pbar = None

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if VERBOSE and line:
            print(line)
        if use_frames and line.startswith("frame="):
            try:
                cur_frame = float(line.split("=", 1)[1])
            except Exception:
                continue
            if pbar is not None and total_val is not None:
                inc = max(0.0, min(cur_frame, total_val) - last_val)
                if inc > 0:
                    pbar.update(inc)
                last_val = cur_frame
        elif not use_frames and line.startswith("out_time_ms="):
            try:
                ms = int(line.split("=", 1)[1])
                cur_sec = ms / 1_000_000.0
            except Exception:
                continue
            if pbar is not None:
                if total_val is None:
                    pbar.update(max(0.0, cur_sec - last_val))
                else:
                    inc = max(0.0, min(cur_sec, total_val) - last_val)
                    if inc > 0:
                        pbar.update(inc)
                last_val = cur_sec
        elif line == "progress=end":
            break

    proc.wait()
    if pbar is not None:
        pbar.close()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
    return 0


# ---------- Core Builder ----------


def _estimate_offset_librosa(
    ref_video_with_audio: Path,
    external_audio_wav: Path,
    *,
    analysis_window_seconds: int = 30,
) -> Tuple[float, float]:
    """
    Estimate relative A/V offset using librosa.

    Loads camera audio from the .insv (mono default) and the external .wav as multi-channel,
    uses the first channel (omnidirectional W) from the external audio, then computes a lag
    via cross-correlation of onset-strength envelopes. Returns a tuple of
    (audio_advance_seconds, video_advance_seconds), where only one will be >0.
    """
    if _librosa is None or _np is None:
        raise RuntimeError(
            "librosa not available; install requirements or choose --sync syncstart/none"
        )

    # Load camera audio as mono at native sampling rate
    y_cam, sr_cam = _librosa.load(str(ref_video_with_audio), sr=None, mono=True)

    # Load external audio preserving channels
    y_ext_multi, sr_ext = _librosa.load(str(external_audio_wav), sr=None, mono=False)

    # Extract first channel (omni/W). Handle possible shapes (n, ch) or (ch, n).
    if y_ext_multi.ndim == 1:
        y_ext_first = y_ext_multi
    elif y_ext_multi.ndim == 2:
        n0, n1 = y_ext_multi.shape
        if n0 in (2, 3, 4, 5, 6, 7, 8) and n1 > n0 * 10:
            # Shape (channels, samples)
            y_ext_first = y_ext_multi[0, :]
        else:
            # Shape (samples, channels)
            y_ext_first = y_ext_multi[:, 0]
    else:
        raise RuntimeError("Unexpected external audio array shape for librosa load")

    # Resample external channel to match camera sample rate if needed
    if sr_ext != sr_cam and sr_cam is not None and sr_ext is not None:
        y_ext_first = _librosa.resample(y_ext_first, orig_sr=sr_ext, target_sr=sr_cam)
        sr = sr_cam
    else:
        sr = sr_ext if sr_ext is not None else sr_cam

    if sr is None or sr <= 0:
        raise RuntimeError("Could not determine sampling rate for librosa sync")

    # Limit to analysis window (from start)
    if analysis_window_seconds and analysis_window_seconds > 0:
        max_len = int(analysis_window_seconds * sr)
        if y_cam.shape[0] > max_len:
            y_cam = y_cam[:max_len]
        if y_ext_first.shape[0] > max_len:
            y_ext_first = y_ext_first[:max_len]

    # Compute onset envelopes for robustness
    hop_length = 512
    cam_env = _librosa.onset.onset_strength(y=y_cam, sr=sr, hop_length=hop_length)
    ext_env = _librosa.onset.onset_strength(y=y_ext_first, sr=sr, hop_length=hop_length)

    # Normalize
    cam_env = cam_env - _np.mean(cam_env)
    ext_env = ext_env - _np.mean(ext_env)
    cam_env = cam_env / (_np.std(cam_env) + 1e-9)
    ext_env = ext_env / (_np.std(ext_env) + 1e-9)

    # Cross-correlation (cam vs ext). Positive lag => ext lags (advance audio)
    xc = _np.correlate(cam_env, ext_env, mode="full")
    lag_idx = int(_np.argmax(xc))
    lag = lag_idx - (len(ext_env) - 1)
    lag_seconds = float(lag) * (hop_length / float(sr))

    audio_advance = 0.0
    video_advance = 0.0
    if lag_seconds > 0:
        # External audio lags camera -> advance audio by lag
        audio_advance = lag_seconds
    elif lag_seconds < 0:
        # External audio leads camera -> advance video by |lag|
        video_advance = abs(lag_seconds)

    return audio_advance, video_advance


@dataclass
class RenderSettings:
    width: int = 7680
    height: int = 3840
    ih_fov: int = 190
    iv_fov: int = 190
    pitch: int = 0
    crf: int = 18
    preset: str = "veryfast"
    pix_fmt: str = "yuv420p"
    audio_bitrate: str = "192k"


@ray.remote
def build_and_run_ffmpeg_for_group(
    activity: str,
    ts: str,
    videos: List[str],
    audios: List[str],
    out_dir: Path,
    settings: RenderSettings,
    overwrite: bool = False,
    dry_run: bool = False,
    sync_mode: str = "none",  # 'librosa' | 'syncstart' | 'none'
    sync_window_sec: int = 30,
    manual_audio_offset_seconds: Optional[float] = None,
    # NEW: Per-input user trimming (seconds)
    trim_video_start_seconds: float = 0.0,
    trim_video_end_seconds: float = 0.0,
    trim_audio_start_seconds: float = 0.0,
    trim_audio_end_seconds: float = 0.0,
) -> Path:
    """
    Build a single ffmpeg command that:
      - For each video .insv input: [i:v:0],[i:v:1] -> hstack -> v360 -> [eq{i}]
      - Concat all [eq{i}] into [V] (or just pass through if only one)
      - For each FOA audio input: [i:a:0] -> downmix stereo [S{i}]
      - Concat all [S{i}] into [A] (or pass through if only one)
      - Map [V] + [A], H.264/AAC, -shortest

    Minimal addition: when exactly one video and one audio are passed and any of the
    four trim args are non-zero, create temporary trimmed copies (fast -c copy)
    for both inputs and use those temporary files for the rest of the pipeline.
    """
    if not videos:
        raise ValueError(
            f"No video parts provided for group {activity}-{ts}. At least one .insv is required."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{activity}-{ts}.mp4"
    if out_path.exists() and not overwrite:
        print(f"[skip] Output exists: {out_path}")
        return out_path

    # ------------------------
    # Minimal temp-trim insertion (single .insv + single .wav case)
    # ------------------------
    tmp_video_path: Optional[Path] = None
    tmp_audio_path: Optional[Path] = None
    created_tmp_files: List[Path] = []

    # User trim values (cast to float)
    v_start = float(trim_video_start_seconds or 0.0)
    v_end = float(trim_video_end_seconds or 0.0)
    a_start = float(trim_audio_start_seconds or 0.0)
    a_end = float(trim_audio_end_seconds or 0.0)

    # Only perform trimming when exactly one video and one audio are present AND
    # at least one of the trim values is > 0.
    do_temp_trim = (len(videos) == 1 and len(audios) == 1) and (
        v_start > 0.0 or v_end > 0.0 or a_start > 0.0 or a_end > 0.0
    )

    if do_temp_trim:
        src_video = videos[0]
        src_audio = audios[0]

        # Probe durations for computing -t (best-effort)
        try:
            src_video_dur = probe_duration_seconds(src_video)
        except Exception:
            src_video_dur = None
        try:
            src_audio_dur = probe_duration_seconds(src_audio)
        except Exception:
            src_audio_dur = None

        # Compute kept durations if probe succeeded
        v_keep = None
        a_keep = None
        if src_video_dur is not None:
            v_keep = max(0.0, src_video_dur - v_start - v_end)
        if src_audio_dur is not None:
            a_keep = max(0.0, src_audio_dur - a_start - a_end)

        # Temp filenames placed in out_dir for easy inspection
        tmp_audio_path = out_dir / f"{activity}-{ts}.trimmed.wav"
        tmp_video_path = out_dir / f"{activity}-{ts}.trimmed.mkv"

        trim_video_cmd = ["ffmpeg", "-y", "-fflags", "+genpts"]
        if v_start > 0.0:
            trim_video_cmd += ["-ss", f"{v_start:.6f}"]
            trim_video_cmd += ["-i", str(src_video)]
        if v_keep is not None:
            trim_video_cmd += ["-t", f"{v_keep:.6f}"]
        # Map both video streams and any audio streams; then copy codecs
        trim_video_cmd += ["-map", "0:v:0", "-map", "0:v:1", "-map", "0:a?", "-c", "copy", str(tmp_video_path)]


        trim_audio_cmd = ["ffmpeg", "-y"]
        if a_start > 0.0:
            trim_audio_cmd += ["-ss", f"{a_start:.6f}"]
        trim_audio_cmd += ["-i", str(src_audio)]
        if a_keep is not None:
            trim_audio_cmd += ["-t", f"{a_keep:.6f}"]
        trim_audio_cmd += ["-c", "copy", str(tmp_audio_path)]

        if VERBOSE:
            _log("Trimming input video (clap removal): " + " ".join(trim_video_cmd))
            _log("Trimming input audio (clap removal): " + " ".join(trim_audio_cmd))

        if not dry_run:
            # run the trimming commands and record created files
            rv = subprocess.run(trim_video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if rv.returncode != 0:
                raise RuntimeError(f"Video trimming failed: {rv.stderr.strip()}")
            created_tmp_files.append(tmp_video_path)

            ra = subprocess.run(trim_audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if ra.returncode != 0:
                # cleanup the video temp if audio trim failed
                try:
                    tmp_video_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise RuntimeError(f"Audio trimming failed: {ra.stderr.strip()}")
            created_tmp_files.append(tmp_audio_path)
        else:
            _log("[dry-run] Skipping creating trimmed temp files.")

    # Use tmp inputs if created, else original inputs
    if tmp_video_path is not None and tmp_audio_path is not None and tmp_video_path.exists() and tmp_audio_path.exists():
        __video_inputs = [tmp_video_path]
        __audio_inputs = [tmp_audio_path]
    else:
        __video_inputs = [v.path for v in videos]
        __audio_inputs = [a.path for a in audios]

    # ------------------------
    # Original pipeline begins (unchanged logic, but using __video_inputs / __audio_inputs)
    # ------------------------

    # Assemble inputs
    cmd: List[str] = ["ffmpeg", "-y" if overwrite else "-n"]
    # Better PTS handling for action-cam containers:
    cmd += ["-fflags", "+genpts"]

    # Input ordering: all video parts first, then audio parts — use prepared inputs
    for p in __video_inputs:
        cmd += ["-i", str(p)]
    for p in __audio_inputs:
        cmd += ["-i", str(p)]

    nv = len(__video_inputs)
    na = len(__audio_inputs)

    # Optional: determine alignment via syncstart using first video+audio
    audio_advance_seconds: float = 0.0  # seconds audio should be moved earlier (we trim audio)
    video_advance_seconds: float = 0.0  # seconds video should be moved earlier (we trim video)

    # If a manual offset is provided, it overrides syncstart logic entirely.
    # Convention: positive offset advances external audio earlier; negative delays it (trim video).
    manual_offset_used = False
    if manual_audio_offset_seconds is not None and abs(manual_audio_offset_seconds) > 1e-6:
        if manual_audio_offset_seconds > 0:
            audio_advance_seconds = manual_audio_offset_seconds
        else:
            video_advance_seconds = abs(manual_audio_offset_seconds)
        manual_offset_used = True

    if (sync_mode == "librosa") and (not manual_offset_used) and na > 0:
        ref_video = __video_inputs[0]
        ref_audio = __audio_inputs[0]
        try:
            aa, va = _estimate_offset_librosa(
                ref_video, ref_audio, analysis_window_seconds=sync_window_sec
            )
        except Exception as exc:
            # Cleanup temp files if any before failing
            try:
                for t in created_tmp_files:
                    t.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(f"librosa sync failed for {activity}-{ts}: {exc}")
        audio_advance_seconds = max(0.0, float(aa))
        video_advance_seconds = max(0.0, float(va))
        if audio_advance_seconds > 0:
            _log(
                f"librosa suggests advancing external audio by "
                f"{_c(f'{audio_advance_seconds:.3f}s', 'bold')}"
            )
        elif video_advance_seconds > 0:
            _log(
                f"librosa suggests advancing video by {_c(f'{video_advance_seconds:.3f}s', 'bold')}"
            )
        else:
            _log("librosa suggests no offset.")
    elif (sync_mode == "syncstart") and (not manual_offset_used) and na > 0:
        ensure_command("syncstart")
        ref_video = __video_inputs[0]
        ref_audio = __audio_inputs[0]
        cmd_sync = [
            "syncstart",
            "-sq",
            "-t",
            str(sync_window_sec),
            str(ref_video),
            str(ref_audio),
        ]
        if VERBOSE:
            print(f"Running audio sync: {' '.join(cmd_sync)}")
        result = subprocess.run(
            cmd_sync,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode not in (0, 1):
            # cleanup tmp files before failing
            try:
                for t in created_tmp_files:
                    t.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                f"syncstart failed (exit {result.returncode}).\n{result.stderr.strip()}"
            )
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            # cleanup tmp files before failing
            try:
                for t in created_tmp_files:
                    t.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError("syncstart produced no output")
        last = lines[-1]
        try:
            file_to_adv_str, seconds_str = last.split(",")
            offset_val = abs(float(seconds_str))
            file_to_adv_raw = file_to_adv_str.strip()
            file_to_adv_path = Path(file_to_adv_raw)
            # Robustly detect whether target is the audio file
            is_audio_target = False
            try:
                is_audio_target = (
                    file_to_adv_path.expanduser().resolve() == ref_audio.resolve()
                    or ref_audio.name in file_to_adv_raw
                    or file_to_adv_path.suffix.lower() in (".wav",)
                )
            except Exception:
                # Fall back on extension / basename check
                is_audio_target = (ref_audio.name in file_to_adv_raw) or (
                    file_to_adv_path.suffix.lower() in (".wav",)
                )

            if is_audio_target:
                audio_advance_seconds = offset_val
                _log(
                    f"syncstart suggests advancing external audio by "
                    f"{_c(f'{audio_advance_seconds:.3f}s', 'bold')}"
                )
            else:
                video_advance_seconds = offset_val
                _log(
                    f"syncstart suggests advancing video by "
                    f"{_c(f'{video_advance_seconds:.3f}s', 'bold')}"
                )
        except Exception:
            # cleanup tmp files before failing
            try:
                for t in created_tmp_files:
                    t.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(f"Unrecognized syncstart output: '{last}'")

    # Prefer trimming audio instead of video when possible (keeps video intact)
    # Skip this preference if a manual offset was explicitly provided by the user.
    if (
        not manual_offset_used
        and na > 0
        and video_advance_seconds > 0.0005
        and audio_advance_seconds <= 0.0005
    ):
        audio_advance_seconds = video_advance_seconds
        video_advance_seconds = 0.0
        _log(
            "Preferring to trim external audio instead of video for alignment: "
            f"{_c(f'{audio_advance_seconds:.3f}s', 'bold')}"
        )

    filters: List[str] = []

    # Video chain per part
    eq_labels: List[str] = []
    for i in range(nv):
        # Extract dual fisheye streams, reset PTS, hstack, then v360 to equirect
        v0 = f"{i}:v:0"
        v1 = f"{i}:v:1"
        lbl_v0 = f"v{i}a"
        lbl_v1 = f"v{i}b"
        lbl_sbs = f"sbs{i}"
        lbl_eq = f"eq{i}"

        filters.append(f"[{v0}]setpts=PTS-STARTPTS[{lbl_v0}]")
        filters.append(f"[{v1}]setpts=PTS-STARTPTS[{lbl_v1}]")
        filters.append(f"[{lbl_v0}][{lbl_v1}]hstack=inputs=2[{lbl_sbs}]")
        filters.append(
            f"[{lbl_sbs}]v360=dfisheye:equirect:"
            f"ih_fov={settings.ih_fov}:iv_fov={settings.iv_fov}:"
            f"pitch={settings.pitch}:w={settings.width}:h={settings.height}"
            f"[{lbl_eq}]"
        )
        eq_labels.append(lbl_eq)

    # Concat videos if needed
    if nv == 1:
        v_out = eq_labels[0]
    else:
        v_inputs = "".join(f"[{lbl}]" for lbl in eq_labels)
        v_out = "V"
        filters.append(f"{v_inputs}concat=n={nv}:v=1:a=0[{v_out}]")

    # If sync suggests advancing video, trim the start of the video stream
    if video_advance_seconds > 0.0005:
        sec = video_advance_seconds
        v_shift = "vshift"
        filters.append(f"[{v_out}]trim=start={sec:.6f},setpts=PTS-STARTPTS[{v_shift}]")
        v_out = v_shift

    # Audio chain (optional)
    a_out: Optional[str] = None
    if na > 0:
        stereo_labels: List[str] = []
        for j in range(na):
            idx = nv + j  # Audio inputs start after video inputs
            a_in = f"{idx}:a:0"
            a_lbl = f"a{j}"
            dm_lbl = f"dm{j}"
            sh_lbl = f"sh{j}"
            st_lbl = f"st{j}"
            # Assumes FOA AmbiX (W,Y,Z,X) = c0,c1,c2,c3. Simple L/R from W±X.
            # If we need to advance audio, trim leading samples; otherwise reset PTS.
            if audio_advance_seconds > 0.0005:
                filters.append(
                    f"[{a_in}]atrim=start={audio_advance_seconds:.6f},asetpts=PTS-STARTPTS[{a_lbl}]"
                )
            else:
                filters.append(f"[{a_in}]asetpts=PTS-STARTPTS[{a_lbl}]")
            # Downmix to stereo first
            filters.append(
                f"[{a_lbl}]aformat=channel_layouts=quad,"
                f"pan=stereo|c0=0.707*c0+0.5*c3|c1=0.707*c0-0.5*c3[{dm_lbl}]"
            )
            # No additional delay path; we normalized alignment via trim if needed
            sh_lbl = dm_lbl
            # Resample / normalize timeline after shift
            filters.append(f"[{sh_lbl}]aresample=async=1:first_pts=0[{st_lbl}]")
            stereo_labels.append(st_lbl)

        if na == 1:
            a_out = stereo_labels[0]
        else:
            a_inputs = "".join(f"[{s}]" for s in stereo_labels)
            a_out = "A"
            filters.append(f"{a_inputs}concat=n={na}:v=0:a=1[{a_out}]")

    # Build filter_complex
    filter_complex = ";".join(filters)
    cmd += ["-filter_complex", filter_complex]

    # Maps
    cmd += ["-map", f"[{v_out}]"]
    if a_out:
        cmd += ["-map", f"[{a_out}]"]
    else:
        cmd += ["-an"]  # no audio

    # Codecs & containers
    cmd += [
        "-c:v",
        "libx264",
        "-crf",
        str(settings.crf),
        "-preset",
        settings.preset,
        "-pix_fmt",
        settings.pix_fmt,
    ]
    if a_out:
        cmd += ["-c:a", "aac", "-b:a", settings.audio_bitrate]
    cmd += ["-movflags", "+faststart", "-shortest", str(out_path)]

    # Estimate expected duration for progress bar
    try:
        video_secs = sum(probe_duration_seconds(Path(p)) for p in __video_inputs)
    except Exception:
        video_secs = 0.0
    try:
        audio_secs = sum(probe_duration_seconds(Path(p)) for p in __audio_inputs) if a_out else 0.0
    except Exception:
        audio_secs = 0.0

    # Apply trim effects to estimate output duration
    effective_video = max(
        0.0, video_secs - (video_advance_seconds if video_advance_seconds > 0 else 0.0)
    )
    effective_audio = max(
        0.0,
        (audio_secs - (audio_advance_seconds if audio_advance_seconds > 0 else 0.0))
        if a_out
        else 0.0,
    )
    expected_seconds: Optional[float]
    if a_out:
        expected_seconds = max(
            0.0, min(effective_video or video_secs, effective_audio or audio_secs)
        )
    else:
        expected_seconds = effective_video or video_secs

    # Estimate frames using FPS of first video input
    fps_val = probe_fps(__video_inputs[0]) if __video_inputs else 30.0
    expected_frames = int(round((expected_seconds or 0.0) * fps_val)) if expected_seconds else None

    run_ffmpeg_with_progress(
        cmd,
        expected_seconds,
        expected_frames=expected_frames,
        dry_run=dry_run,
        description=f"Encoding {activity}-{ts}",
    )

    # Cleanup temporary trimmed files if we created them (and not in dry-run)
    if created_tmp_files and not dry_run:
        for tmp in created_tmp_files:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    return out_path




# ---------- CLI ----------


def collect_paths(args: List[str]) -> List[Path]:
    paths: List[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            for ext in ("*.insv", "*.INSV", "*.wav", "*.WAV"):
                paths.extend(p.glob(ext))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[warn] Skipping non-existent path: {a}", file=sys.stderr)
    return paths


def main():
    import argparse
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Stitch dual-fisheye .insv + FOA .wav into 360° equirectangular H.264 MP4 with optional trimming."
    )
    ap.add_argument("video", help="Path to .insv video file (with dual fisheye streams)")
    ap.add_argument("audio", help="Path to .wav audio file")
    ap.add_argument("-o", "--out", default="out", help="Output directory (default: ./out)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="Print ffmpeg commands without running")
    ap.add_argument(
        "--sync",
        choices=["librosa", "syncstart", "none"],
        default="none",
        help="Alignment method: 'librosa' (default), 'syncstart' (external tool), or 'none'",
    )
    ap.add_argument(
        "--res",
        choices=["8k", "4k", "2k"],
        help="Resolution preset: 8k=7680x3840, 4k=3840x1920, 2k=1920x960 (overrides --w/--h)",
    )
    ap.add_argument("--w", type=int, default=7680, help="Equirect width (default: 7680)")
    ap.add_argument("--h", type=int, default=3840, help="Equirect height (default: 3840)")
    ap.add_argument("--ih-fov", type=int, default=190, help="Input horizontal FOV per fisheye (default: 190)")
    ap.add_argument("--iv-fov", type=int, default=190, help="Input vertical FOV per fisheye (default: 190)")
    ap.add_argument("--pitch", type=int, default=0, help="Pitch applied in v360 (default: 180)")
    ap.add_argument("--crf", type=int, default=18, help="x264 CRF (default: 18)")
    ap.add_argument("--preset", default="veryfast", help="x264 preset (default: veryfast)")
    ap.add_argument("--audio-bitrate", default="192k", help="AAC bitrate (default: 192k)")

    # Trimming options
    ap.add_argument("--trim-video-start", type=float, default=0.0, help="Seconds to trim from start of video")
    ap.add_argument("--trim-video-end", type=float, default=0.0, help="Seconds to trim from end of video")
    ap.add_argument("--trim-audio-start", type=float, default=0.0, help="Seconds to trim from start of audio")
    ap.add_argument("--trim-audio-end", type=float, default=0.0, help="Seconds to trim from end of audio")

    ap.add_argument("--audio-offset", type=float, default=None,
        help=("Manual A/V offset in seconds. >0 advances external audio (trim audio); "
              "<0 delays external audio (trim video). Overrides syncstart.")
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Print detailed logs")

    args = ap.parse_args()

    # Verbosity flag
    global VERBOSE
    VERBOSE = args.verbose

    check_ffmpeg()

    # Prepare media objects
    # video = MediaFile(Path(args.video))
    # audio = MediaFile(Path(args.audio))
    video = args.video
    audio = args.audio

    # Resolution preset if provided
    width, height = args.w, args.h
    if args.res:
        if args.res == "8k":
            width, height = 7680, 3840
        elif args.res == "4k":
            width, height = 3840, 1920
        elif args.res == "2k":
            width, height = 1920, 960

    settings = RenderSettings(
        width=width,
        height=height,
        ih_fov=args.ih_fov,
        iv_fov=args.iv_fov,
        pitch=args.pitch,
        crf=args.crf,
        preset=args.preset,
        audio_bitrate=args.audio_bitrate,
    )

    out_dir = Path(args.out)

    if not ray.is_initialized():
        ray.init()

    # try:
    futures = build_and_run_ffmpeg_for_group(
        "capture", "003",
        [video], [audio],
        out_dir,
        settings=settings,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        sync_mode=args.sync,
        sync_window_sec=30,
        manual_audio_offset_seconds=args.audio_offset,
        trim_video_start_seconds=args.trim_video_start,
        trim_video_end_seconds=args.trim_video_end,
        trim_audio_start_seconds=args.trim_audio_start,
        trim_audio_end_seconds=args.trim_audio_end,
    )
        # print(f"\n✅ Output written: {out_path}")
    # except Exception as e:
    #     print(f"❌ Error: {e}", file=sys.stderr)
    #     sys.exit(1)

    results = ray.get(futures)
    print(results)

    ray.shutdown()





if __name__ == "__main__":
    main()
# python stitch_mux.py "skincare-20250819_1359-video.insv" "skincare-20250819_1359-audio.wav" -o stitched_output --trim-video-start 2.39 --trim-video-end 2.63 --trim-audio-start 4.61 --trim-audio-end 4.07 --res 4k --overwrite --verbose