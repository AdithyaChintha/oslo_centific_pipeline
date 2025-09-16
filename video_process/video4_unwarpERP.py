import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple



class FFmpegError(RuntimeError):
    pass


def _run(cmd: Iterable[str], *, fallback_x264: bool = False) -> None:
    """Run a subprocess command. If NVENC fails and fallback_x264=True, retry with libx264."""
    try:
        subprocess.run(list(cmd), check=True, capture_output=True, text=True)
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
                subprocess.run(cmd2, check=True, capture_output=True, text=True)
                return
            except subprocess.CalledProcessError as e2:
                stderr = getattr(e2, "stderr", None)
                raise FFmpegError(f"FFmpeg failed (fallback to libx264 also failed): returncode={e2.returncode}\nstderr={stderr}\ncmd={e2.cmd}") from e2
        stderr = getattr(e, "stderr", None)
        raise FFmpegError(f"FFmpeg failed: returncode={e.returncode}\nstderr={stderr}\ncmd={e.cmd}") from e


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

def insv_to_4viewsERP(
    insv_path: str,
    output_dir: str = "out_4views",
    *,
    # Step 1 (ERP)
    erp_size: Tuple[int, int] = (4096, 2048),
    lens_fov_deg: float = 190.0,  # for (d)fisheye→equirect
    encoder: str = "libx264",  # try NVENC first; falls back to x264
    crf_or_cq: int = 18,          # if NVENC: CQ; if x264: CRF
    preset: str = "fast",
    # Step 2 (4 rect views)
    out_size: Tuple[int, int] = (1440, 1440),
    h_fov_deg: float = 90.0,
    v_fov_deg: float = 90.0,
    yaw_front_right_back_left: Tuple[float, float, float, float] = (45.0, 135.0, -135.0, -45.0),
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    #flip_erp: bool = True,
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
        raise FileNotFoundError(in_path)

    # ---------- Step 1: INSV → ERP ----------
    W_erp, H_erp = erp_size
    erp_path = out_dir / f"{in_path.stem}_ERP_{W_erp}x{H_erp}.mp4"

    n_streams = _probe_video_stream_count(in_path)

    if n_streams == 2:
        # Two fisheye streams → hstack → dfisheye→equirect → vflip
        vf = (
                f"[0:v:0]setpts=PTS-STARTPTS,setsar=1[v0];"
                f"[0:v:1]setpts=PTS-STARTPTS,setsar=1[v1];"
                f"[v0][v1]hstack=inputs=2[dual];"
                f"[dual]v360=input=dfisheye:output=equirect:ih_fov={lens_fov_deg}:iv_fov={lens_fov_deg}:w={W_erp}:h={H_erp}[erp1];"
                f"[erp1]vflip[erp]"
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
    else:
        raise FFmpegError("No suitable video streams found in input.")

    # ---------- Step 2: ERP → 4 rect views ----------
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


from pathlib import Path
from typing import Tuple, Dict

def insv_to_4viewsERP_one_shot(
    insv_path: str,
    output_dir: str = "out_4views",
    *,
    # Step 1 (build ERP from dual-fisheye)
    erp_size: Tuple[int, int] = (4096, 2048),
    lens_fov_deg: float = 190.0,       # fisheye lens FOV used by v360 dfisheye→equirect
    encoder: str = "libx264",          # or "h264_nvenc"
    crf_or_cq: int = 18,               # x264: CRF value; NVENC: CQ value
    preset: str = "fast",              # x264 presets: ultrafast..veryslow; NVENC accepts "slow/medium/fast" or p1..p7
    # Step 2 (extract 4 rectilinear views from ERP)
    out_size: Tuple[int, int] = (1440, 1440),
    h_fov_deg: float = 90.0,
    v_fov_deg: float = 90.0,
    yaw_front_right_back_left: Tuple[float, float, float, float] = (45.0, 135.0, -135.0, -45.0),
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> Dict[str, str]:
    """
    One-shot INSV → ERP → 4 rectilinear views using a single FFmpeg invocation.

    Pipeline (single -filter_complex):
      [0:v:0] + [0:v:1]  --hstack-->  [dual]
      [dual] --v360 dfisheye→equirect--> [erp1] --vflip--> [erp]
      [erp] --split=5--> [erp_out],[f],[r],[b],[l]
      Each of [f],[r],[b],[l] --v360 equirect→rectilinear--> [front],[right],[back],[left]

    Assumptions:
      • The INSV contains two video streams: [0:v:0] and [0:v:1] (dual-fisheye).
      • Audio is optional; we map "0:a?" to copy audio when present.
      • All five outputs are encoded in the same codec/preset/rate-control for simplicity.

    Args:
      insv_path: Path to the input .insv file.
      output_dir: Directory where all outputs will be written/created.
      erp_size: (W,H) of the equirectangular (2:1) frame produced from dual-fisheye.
      lens_fov_deg: FOV parameter for v360 dfisheye→equirect conversion.
      encoder: "libx264" for CPU H.264, or "h264_nvenc" for NVIDIA NVENC H.264.
      crf_or_cq: If encoder=="libx264", interpreted as CRF; if "h264_nvenc", interpreted as CQ.
      preset: Encoder preset. For x264 use standard presets; for NVENC use slow/medium/fast or p1..p7.
      out_size: (W,H) of each rectilinear view (front/right/back/left).
      h_fov_deg, v_fov_deg: Horizontal/vertical FOV for rectilinear projection.
      yaw_front_right_back_left: Yaw angles (deg) for the 4 views: front, right, back, left.
      pitch_deg, roll_deg: Pitch/roll (deg) applied to every rectilinear view.

    Returns:
      Dict with keys:
        "erp", "front", "right", "back", "left"  →  absolute file paths to the five MP4s.

    Notes:
      • Using a single FFmpeg run reduces I/O and avoids recomputing dfisheye→equirect.
      • All outputs are written in one go; compute load rises since 4 encoders run in parallel.
      • If you do NOT want an ERP file, change split=5→4 and remove the [erp_out] mapping block.
      • If the input isn’t dual-stream, you should add a probe/guard before building the graph.
    """
    in_path = Path(insv_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    # ----- Build output paths -----
    W_erp, H_erp = erp_size
    erp_path   = out_dir / f"{in_path.stem}_ERP_{W_erp}x{H_erp}.mp4"

    W, H       = out_size
    out_front  = out_dir / f"front_{W}x{H}.mp4"
    out_right  = out_dir / f"right_{W}x{H}.mp4"
    out_back   = out_dir / f"back_{W}x{H}.mp4"
    out_left   = out_dir / f"left_{W}x{H}.mp4"

    # Normalize yaw to [-180, 180] for v360 stability (wrap-around)
    def _norm(a: float) -> float:
        return (a + 180.0) % 360.0 - 180.0

    yaw_f, yaw_r, yaw_b, yaw_l = map(_norm, yaw_front_right_back_left)

    # ----- Compose the filter graph -----
    # 1) Align PTS & SAR for each fisheye → hstack → dfisheye→equirect into target ERP size → vflip to fix vertical orientation.
    # 2) Split ERP into five branches: one branch is saved as ERP output; four branches are rectified into 90° views.
    vf = (
        # Dual-fisheye → ERP
        f"[0:v:0]setpts=PTS-STARTPTS,setsar=1[v0];"
        f"[0:v:1]setpts=PTS-STARTPTS,setsar=1[v1];"
        f"[v0][v1]hstack=inputs=2[dual];"
        f"[dual]v360=input=dfisheye:output=equirect:ih_fov={lens_fov_deg}:iv_fov={lens_fov_deg}:w={W_erp}:h={H_erp}[erp1];"
        f"[erp1]vflip[erp];"
        # ERP → (erp_out + 4 rect views)
        f"[erp]split=5[erp_out][f][r][b][l];"
        f"[f]v360=input=equirect:output=rectilinear:h_fov={h_fov_deg}:v_fov={v_fov_deg}:"
        f"yaw={yaw_f}:pitch={pitch_deg}:roll={roll_deg}:w={W}:h={H}[front];"
        f"[r]v360=input=equirect:output=rectilinear:h_fov={h_fov_deg}:v_fov={v_fov_deg}:"
        f"yaw={yaw_r}:pitch={pitch_deg}:roll={roll_deg}:w={W}:h={H}[right];"
        f"[b]v360=input=equirect:output=rectilinear:h_fov={h_fov_deg}:v_fov={v_fov_deg}:"
        f"yaw={yaw_b}:pitch={pitch_deg}:roll={roll_deg}:w={W}:h={H}[back];"
        f"[l]v360=input=equirect:output=rectilinear:h_fov={h_fov_deg}:v_fov={v_fov_deg}:"
        f"yaw={yaw_l}:pitch={pitch_deg}:roll={roll_deg}:w={W}:h={H}[left]"
    )

    # Rate-control flag depends on encoder family:
    #   x264 → use -crf; NVENC → use -cq
    rate_flag = "-cq" if encoder == "h264_nvenc" else "-crf"

    # Important: FFmpeg applies options to the *next* output that follows.
    # We therefore repeat mapping/encoding blocks for each of the five outputs.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
        "-i", str(in_path),
        "-filter_complex", vf,

        # Save ERP as an MP4 (with audio if present)
        "-map", "[erp_out]", "-map", "0:a?",
        "-c:v", encoder, rate_flag, str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(erp_path),

        # Front view
        "-map", "[front]", "-map", "0:a?",
        "-c:v", encoder, rate_flag, str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_front),

        # Right view
        "-map", "[right]", "-map", "0:a?",
        "-c:v", encoder, rate_flag, str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_right),

        # Back view
        "-map", "[back]", "-map", "0:a?",
        "-c:v", encoder, rate_flag, str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_back),

        # Left view
        "-map", "[left]", "-map", "0:a?",
        "-c:v", encoder, rate_flag, str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_left),
    ]

    # _run should execute subprocess and (optionally) fall back to x264 if NVENC is unavailable.
    # Example behavior:
    #   try NVENC → if failed, rebuild identical cmd substituting encoder="libx264" and rate_flag="-crf".
    _run(cmd, fallback_x264=True)

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

    outputs = insv_to_4viewsERP_one_shot(
        insv_path=insv,
        output_dir=out_dir,
    )

    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    print("Done.")
