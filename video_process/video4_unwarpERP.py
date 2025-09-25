import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# Import logger
from utils.logger import get_logger

logger = get_logger("video4_unwarpERP")



class FFmpegError(RuntimeError):
    pass


def _run(cmd: Iterable[str], *, fallback_x264: bool = False, operation_name: str = "FFmpeg") -> Dict[str, any]:
    """Run a subprocess command. If NVENC fails and fallback_x264=True, retry with libx264."""
    cmd_list = list(cmd)
    original_encoder = None
    final_encoder = None

    # Detect original encoder
    for i, tok in enumerate(cmd_list):
        if tok in ["h264_nvenc", "libx264"]:
            original_encoder = tok
            break

    start_time = time.time()
    logger.info(f"🎬 Starting {operation_name} with encoder: {original_encoder}")

    try:
        result = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        end_time = time.time()
        duration = end_time - start_time
        final_encoder = original_encoder

        logger.info(f"✅ {operation_name} completed successfully")
        logger.info(f"   📊 Encoder used: {final_encoder}")
        logger.info(f"   ⏱️  Duration: {duration:.2f} seconds")

        return {
            "success": True,
            "encoder_used": final_encoder,
            "duration": duration,
            "fallback_used": False
        }

    except subprocess.CalledProcessError as e:
        if fallback_x264 and any("h264_nvenc" in tok for tok in cmd_list):
            logger.warning(f"⚠️ NVENC failed, attempting fallback to libx264...")

            # Retry replacing encoder with libx264
            cmd2 = []
            for tok in cmd_list:
                if tok == "h264_nvenc":
                    cmd2.append("libx264")
                elif tok == "-cq":
                    cmd2.append("-crf")
                else:
                    cmd2.append(tok)

            try:
                result = subprocess.run(cmd2, check=True, capture_output=True, text=True)
                end_time = time.time()
                duration = end_time - start_time
                final_encoder = "libx264"

                logger.info(f"✅ {operation_name} completed with fallback")
                logger.info(f"   📊 Encoder used: {final_encoder} (fallback from {original_encoder})")
                logger.info(f"   ⏱️  Duration: {duration:.2f} seconds")

                return {
                    "success": True,
                    "encoder_used": final_encoder,
                    "duration": duration,
                    "fallback_used": True,
                    "original_encoder": original_encoder
                }

            except subprocess.CalledProcessError as e2:
                end_time = time.time()
                duration = end_time - start_time
                stderr = getattr(e2, "stderr", None)

                logger.error(f"❌ {operation_name} failed completely")
                logger.error(f"   📊 Original encoder: {original_encoder}")
                logger.error(f"   📊 Fallback encoder: libx264")
                logger.error(f"   ⏱️  Duration before failure: {duration:.2f} seconds")
                logger.error(f"   🔍 Error: {stderr}")

                raise FFmpegError(f"FFmpeg failed (fallback to libx264 also failed): returncode={e2.returncode}\nstderr={stderr}\ncmd={e2.cmd}") from e2

        end_time = time.time()
        duration = end_time - start_time
        stderr = getattr(e, "stderr", None)

        logger.error(f"❌ {operation_name} failed")
        logger.error(f"   📊 Encoder: {original_encoder}")
        logger.error(f"   ⏱️  Duration before failure: {duration:.2f} seconds")
        logger.error(f"   🔍 Error: {stderr}")

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


from pathlib import Path
from typing import Tuple, Dict

def insv_to_4viewsERP_one_shot(
    insv_path: str,
    output_dir: str = "out_4views",
    *,
    # Step 1 (build ERP from dual-fisheye)
    erp_size: Tuple[int, int] = None,
    lens_fov_deg: float = 190.0,       # fisheye lens FOV used by v360 dfisheye→equirect
    encoder="libx264",                 # Use CPU encoder like sharding (reliable)
    crf_or_cq=23,                     # Use CRF like sharding for consistency
    preset="fast",                    # Use x264 preset like sharding
    use_gpu: bool = True,             # Enable GPU acceleration like sharding
    # Step 2 (extract 4 rectilinear views from ERP)
    out_size: Tuple[int, int] = None,
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

    # ----- Build output paths -----
    erp_path   = out_dir / f"{in_path.stem}_ERP_{W_erp}x{H_erp}.mp4"

    # Calculate output size from ERP dimensions if not specified
    if out_size is None:
        # Use ERP height as the square output dimension (maintains good quality)
        # For 3840x1920 ERP → 1920x1920 square views
        # For 7680x3840 ERP → 3840x3840 square views
        square_size = H_erp
        out_size = (square_size, square_size)

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
    # Same filter graph for both GPU and CPU (like sharding approach)
    # GPU acceleration is decode-only, processing is CPU-based
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

    # Follow sharding methodology: GPU decode + CPU encode for reliability
    # Rate-control: Always use -crf for x264 (like sharding)

    # Build base command (CPU-only, like sharding fallback)
    base_cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
        "-i", str(in_path),
        "-filter_complex", vf,

        # Save ERP as an MP4 (with audio if present) - CPU encoding like sharding
        "-map", "[erp_out]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(erp_path),

        # Front view
        "-map", "[front]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_front),

        # Right view
        "-map", "[right]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_right),

        # Back view
        "-map", "[back]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_back),

        # Left view
        "-map", "[left]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_left),
    ]

    # GPU-accelerated command (EXACTLY like sharding methodology)
    if use_gpu:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
            # GPU acceleration EXACTLY like sharding: decode-only acceleration
            "-hwaccel", "cuda",
            "-i", str(in_path),
            "-filter_complex", vf,

            # Save ERP as an MP4 (with audio if present) - CPU encoding like sharding
            "-map", "[erp_out]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(erp_path),

            # Front view
            "-map", "[front]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_front),

            # Right view
            "-map", "[right]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_right),

            # Back view
            "-map", "[back]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_back),

            # Left view
            "-map", "[left]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(crf_or_cq), "-preset", preset,
            "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out_left),
        ]
    else:
        cmd = base_cmd

    # Log input details before processing
    logger.info(f"🚀 Starting one-shot INSV→ERP+4Views processing")
    logger.info(f"   📁 Input: {in_path.name}")
    logger.info(f"   📐 ERP size: {W_erp}x{H_erp}")
    logger.info(f"   📐 View size: {W}x{H}")
    logger.info(f"   🎛️  Target encoder: {encoder}")
    logger.info(f"   🎛️  Quality setting: {crf_or_cq}")
    logger.info(f"   🎛️  Preset: {preset}")

    # Execute with sharding-style fallback logic
    logger.info(f"🔧 unwarp {in_path.name} | GPU={'on' if use_gpu else 'off'} | encoder=libx264")

    processing_metrics = _run(cmd, fallback_x264=False, operation_name="One-shot INSV→ERP+4Views (GPU-accelerated)")

    # If GPU failed, try CPU fallback (like sharding does)
    if not processing_metrics["success"] and use_gpu:
        logger.warning(f"GPU processing failed, falling back to CPU for {in_path.name}")
        logger.info(f"🔧 unwarp {in_path.name} | GPU=fallback-cpu | encoder=libx264")
        processing_metrics = _run(base_cmd, fallback_x264=False, operation_name="One-shot INSV→ERP+4Views (CPU-fallback)")

    # Determine actual GPU usage (following sharding pattern)
    gpu_decode_used = use_gpu and processing_metrics["success"]
    cpu_fallback_used = use_gpu and not processing_metrics["success"]

    # Log final summary (like sharding style)
    logger.info(f"🏁 One-shot processing complete!")
    logger.info(f"   📊 Encoder used: {processing_metrics['encoder_used']} (CPU encoding)")
    logger.info(f"   ⏱️  Total duration: {processing_metrics['duration']:.2f} seconds")
    logger.info(f"   🎯 GPU decode: {'✅' if gpu_decode_used else '❌ fallback to CPU'}")
    logger.info(f"   📂 Outputs: ERP + 4 perspective views")

    if cpu_fallback_used:
        logger.info(f"   ⚠️  GPU decode failed, used CPU fallback (like sharding)")
    elif gpu_decode_used:
        logger.info(f"   ✅ GPU-accelerated decode successful (like sharding)")

    return {
        "erp": str(erp_path),
        "front": str(out_front),
        "right": str(out_right),
        "back":  str(out_back),
        "left":  str(out_left),
        # Add processing metrics (sharding-style)
        "_metrics": {
            "total_duration": processing_metrics["duration"],
            "encoder_used": processing_metrics["encoder_used"],
            "gpu_decode_enabled": use_gpu,
            "gpu_decode_successful": gpu_decode_used,
            "cpu_fallback_used": cpu_fallback_used,
            "processing_method": "gpu_accelerated" if gpu_decode_used else ("cpu_fallback" if cpu_fallback_used else "cpu_only"),
            "input_resolution": f"{W_erp}x{H_erp}",
            "output_resolution": f"{W}x{H}",
            "quality_setting": crf_or_cq,
            "preset": preset,
            "sharding_methodology": True
        }
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
