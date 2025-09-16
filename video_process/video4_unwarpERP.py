import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


# =============================
# INSV → ERP(2:1) → 4 rect views (FFmpeg-only)
# =============================
#
# - Step 1: Build a TRUE equirectangular (ERP, 2:1) video from INSV.
#   • If the INSV has TWO fisheye video streams, we first hstack them to a dual-fisheye frame
#     then convert with v360: dfisheye→equirect.
#   • If it has ONE fisheye stream, we use v360: fisheye→equirect.
#
# - Step 2: From ERP, map 4 rectilinear 90° views via v360 (input=equirect → output=rectilinear)
#   • yaw sequence defaults to: FRONT=45, RIGHT=135, BACK=225, LEFT=-45 (avoids seam).
#
# Notes
# -----
# • All heavy lifting is done by FFmpeg/FFprobe; Python only orchestrates commands.
# • Requires FFmpeg built with the `v360` filter.
# • Encoder defaults to NVENC, and auto-falls back to libx264 if NVENC is unavailable.
# • Add ih_fov/iv_fov according to your lens (default 190° typical for action cams).
# • For better compatibility, outputs use yuv420p and +faststart.


class FFmpegError(RuntimeError):
    pass


def _run(cmd: Iterable[str], *, fallback_x264: bool = False) -> None:
    """Run a subprocess command. If NVENC fails and fallback_x264=True, retry with libx264."""
    try:
        subprocess.run(list(cmd), check=True)
    except subprocess.CalledProcessError as e:
        if fallback_x264 and any("h264_nvenc" in tok for tok in cmd):
            # Retry replacing encoder with libx264
            cmd2 = []
            for tok in cmd:
                if tok == "h264_nvenc":
                    cmd2.append("libx264")
                elif tok == "-cq":
                    cmd2.append("-crf")
                else:
                    cmd2.append(tok)
            try:
                subprocess.run(cmd2, check=True)
                return
            except subprocess.CalledProcessError as e2:
                raise FFmpegError(f"FFmpeg failed (fallback to libx264 also failed):\n{e2}") from e2
        raise FFmpegError(f"FFmpeg failed: {e}") from e


def _probe_video_stream_count(path: Path) -> int:
    """Return the number of video streams using ffprobe."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {proc.stderr}")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return len(lines)

def _get_input_video_resolution(path: Path) -> Tuple[int, int]:
    """Get the resolution of the input video using ffprobe."""
    if not path.exists():
        raise FFmpegError(f"Video file does not exist: {path}")

    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed to get video resolution from {path}: {proc.stderr}")

    output = proc.stdout.strip()
    if not output:
        raise FFmpegError(f"ffprobe returned empty output for {path}")

    try:
        width, height = output.split(',')
        return int(width), int(height)
    except (ValueError, IndexError) as e:
        raise FFmpegError(f"Failed to parse video resolution '{output}' from {path}: {e}")

def insv_to_4viewsERP(
    insv_path: str,
    output_dir: str = "out_4views",
    *,
    # Step 1 (ERP)
    erp_size: Tuple[int, int] = None,
    lens_fov_deg: float = 190.0,  # for (d)fisheye→equirect
    encoder: str = "libx264",  # try NVENC first; falls back to x264
    crf_or_cq: int = 18,          # if NVENC: CQ; if x264: CRF
    preset: str = "fast",
    # Step 2 (4 rect views)
    out_size: Optional[Tuple[int, int]] = None,  # Auto-calculate from ERP size if None
    h_fov_deg: float = 90.0,
    v_fov_deg: float = 90.0,
    yaw_front_right_back_left: Tuple[float, float, float, float] = (45.0, 135.0, -135.0, -45.0),
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    flip_erp: bool = True,
) -> Dict[str, str]:
    """
    Convert an INSV file into a TRUE ERP (2:1) video, then map out four 90° rectilinear views
    using FFmpeg's v360 filter. Entire pipeline is FFmpeg-only.

    Returns a dict mapping {view_name: output_path}.
    """
    in_path = Path(insv_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {in_path}")

    # ---------- Step 1: INSV → ERP ----------
    if erp_size is None:
        try:
            resolution_result = _get_input_video_resolution(in_path)
            if resolution_result is None:
                raise FFmpegError(f"Failed to get video resolution from {in_path} - got None")
            input_width, input_height = resolution_result
        except Exception as e:
            raise FFmpegError(f"Error getting input video resolution from {in_path}: {e}")
        # For equirectangular projection, maintain 2:1 aspect ratio
        # Use input width as base and calculate height accordingly
        if input_width >= input_height * 2:
            # Input is already roughly 2:1, use as-is
            W_erp, H_erp = input_width, input_height
        else:
            # Input is not 2:1, create proper ERP dimensions
            # Use the larger dimension to determine scale
            max_dim = max(input_width, input_height)
            W_erp = max_dim if max_dim % 2 == 0 else max_dim + 1  # Ensure even
            H_erp = W_erp // 2
    else:
        W_erp, H_erp = erp_size
    erp_path = out_dir / f"{in_path.stem}_ERP_{W_erp}x{H_erp}.mp4"

    n_streams = _probe_video_stream_count(in_path)

    if n_streams >= 2:
        # Two fisheye streams → hstack → dfisheye→equirect → (optional) vflip
        if flip_erp:
            vf = (
                f"[0:v:0]setpts=PTS-STARTPTS,setsar=1[v0];"
                f"[0:v:1]setpts=PTS-STARTPTS,setsar=1[v1];"
                f"[v0][v1]hstack=inputs=2[dual];"
                f"[dual]v360=input=dfisheye:output=equirect:ih_fov={lens_fov_deg}:iv_fov={lens_fov_deg}:w={W_erp}:h={H_erp}[erp1];"
                f"[erp1]vflip[erp]"
            )
        else:
            vf = (
                f"[0:v:0]setpts=PTS-STARTPTS,setsar=1[v0];"
                f"[0:v:1]setpts=PTS-STARTPTS,setsar=1[v1];"
                f"[v0][v1]hstack=inputs=2[dual];"
                f"[dual]v360=input=dfisheye:output=equirect:ih_fov={lens_fov_deg}:iv_fov={lens_fov_deg}:w={W_erp}:h={H_erp}[erp]"
            )
        cmd_erp = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
            "-i", str(in_path),
            "-filter_complex", vf,
            "-map", "[erp]",
            "-map", "0:a?",
            "-c:v", encoder,
            "-c:a", "copy",
            "-crf" if encoder == "libx264" else "-cq", str(crf_or_cq),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(erp_path),
        ]
        _run(cmd_erp, fallback_x264=True)
    elif n_streams == 1:
        # Single fisheye stream → fisheye→equirect → (optional) vflip
        vf = (
            f"v360=input=fisheye:output=equirect:ih_fov={lens_fov_deg}:iv_fov={lens_fov_deg}:w={W_erp}:h={H_erp}"
        )
        if flip_erp:
            vf += ",vflip"
        cmd_erp = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
            "-i", str(in_path),
            "-vf", vf,
            "-c:v", encoder,
            "-c:a", "copy",
            ("-crf" if encoder == "libx264" else "-cq"), str(crf_or_cq),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(erp_path),
        ]
        _run(cmd_erp, fallback_x264=True)
    else:
        raise FFmpegError("No video streams found in input.")

    # ---------- Step 2: ERP → 4 rect views ----------
    # Calculate output size from ERP dimensions if not specified
    if out_size is None:
        # Use ERP height as the square output dimension (maintains good quality)
        # For 3840x1920 ERP → 1920x1920 square views
        # For 7680x3840 ERP → 3840x3840 square views
        square_size = H_erp
        out_size = (square_size, square_size)

    W, H = out_size
    yaw_f, yaw_r, yaw_b, yaw_l = yaw_front_right_back_left

    # Normalize yaw into [-180, 180] to satisfy v360
    def _norm(a: float) -> float:
        return (a + 180.0) % 360.0 - 180.0
    yaw_f, yaw_r, yaw_b, yaw_l = (_norm(yaw_f), _norm(yaw_r), _norm(yaw_b), _norm(yaw_l))

    # Build one filter_complex that splits ERP into 4 branches and applies v360 per branch
    vf4 = (
        f"[0:v]split=4[f][r][b][l];"
        f"[f]v360=input=equirect:output=rectilinear:h_fov={h_fov_deg}:v_fov={v_fov_deg}:yaw={yaw_f}:pitch={pitch_deg}:roll={roll_deg}:w={W}:h={H}[front];"
        f"[r]v360=input=equirect:output=rectilinear:h_fov={h_fov_deg}:v_fov={v_fov_deg}:yaw={yaw_r}:pitch={pitch_deg}:roll={roll_deg}:w={W}:h={H}[right];"
        f"[b]v360=input=equirect:output=rectilinear:h_fov={h_fov_deg}:v_fov={v_fov_deg}:yaw={yaw_b}:pitch={pitch_deg}:roll={roll_deg}:w={W}:h={H}[back];"
        f"[l]v360=input=equirect:output=rectilinear:h_fov={h_fov_deg}:v_fov={v_fov_deg}:yaw={yaw_l}:pitch={pitch_deg}:roll={roll_deg}:w={W}:h={H}[left]"
    )

    out_front = out_dir / f"front_{W}x{H}.mp4"
    out_right = out_dir / f"right_{W}x{H}.mp4"
    out_back  = out_dir / f"back_{W}x{H}.mp4"
    out_left  = out_dir / f"left_{W}x{H}.mp4"

    cmd_4 = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
        "-i", str(erp_path),
        "-filter_complex", vf4,
        # map 4 labeled outputs
        "-map", "[front]", "-map", "0:a?", "-c:v", encoder, ("-cq" if encoder == "h264_nvenc" else "-crf"), str(crf_or_cq), "-preset", preset, "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_front),
        "-map", "[right]", "-map", "0:a?", "-c:v", encoder, ("-cq" if encoder == "h264_nvenc" else "-crf"), str(crf_or_cq), "-preset", preset, "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_right),
        "-map", "[back]",  "-map", "0:a?", "-c:v", encoder, ("-cq" if encoder == "h264_nvenc" else "-crf"), str(crf_or_cq), "-preset", preset, "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_back),
        "-map", "[left]",  "-map", "0:a?", "-c:v", encoder, ("-cq" if encoder == "h264_nvenc" else "-crf"), str(crf_or_cq), "-preset", preset, "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_left),
    ]

    _run(cmd_4, fallback_x264=True)

    return {
        "erp": str(erp_path),
        "front": str(out_front),
        "right": str(out_right),
        "back":  str(out_back),
        "left":  str(out_left),
    }


if __name__ == "__main__":
    import sys
    import json

    insv = "VID_20250809_094836_00_045.insv"
    out_dir = "out_4viewsERP"

    outputs = insv_to_4views_ffmpeg(
        insv_path=insv,
        output_dir=out_dir,
    )

    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    print("Done.")
