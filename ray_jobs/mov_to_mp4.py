import subprocess
import os
from utils.logger import get_logger

logger = get_logger("MOV2MP4")
@ray.remote
def convert_mov_to_mp4_task(
    input_mov_path: str,
    output_mp4_path: str,
    video_id: str,
    timer = None
) -> Dict[str, any]:
    """
    Ray remote task to convert MOV file to MP4 format using FFmpeg.

    This task runs as a separate Ray job and can be parallelized across multiple videos.

    Args:
        input_mov_path: Path to input MOV file
        output_mp4_path: Path to save output MP4 file
        video_id: Video identifier for logging
        timer: Optional PipelineTimer instance for tracking conversion time

    Returns:
        Dict containing:
            - success: bool
            - output_path: str (MP4 path if successful)
            - file_size: int (bytes)
            - duration: float (seconds)
            - error: str (if failed)

    Example:
        result = ray.get(convert_mov_to_mp4_task.remote(
            "/tmp/video.mov",
            "/tmp/video.mp4",
            "video_001"
        ))
    """
 

    try:
        logger.info(f"Starting MOV to MP4 conversion: {input_mov_path}")

        # Validate input file exists
        if not os.path.exists(input_mov_path):
            raise FileNotFoundError(f"Input MOV file not found: {input_mov_path}")

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)

        # FFmpeg command with optimized settings
        # - copy video codec if already H.264, otherwise re-encode
        # - copy audio codec if already AAC, otherwise re-encode
        # - fast encoding preset for speed
        # - preserve metadata
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_mov_path,
            '-c:v', 'libx264',           # Video codec: H.264
            '-preset', 'fast',            # Encoding speed vs quality
            '-crf', '23',                 # Quality (18-28 recommended, 23 is default)
            '-c:a', 'aac',               # Audio codec: AAC
            '-b:a', '128k',              # Audio bitrate
            '-movflags', '+faststart',   # Enable streaming (moov atom at start)
            '-y',                         # Overwrite output file
            output_mp4_path
        ]

        # Start timer if provided
        timer_context = None
        if timer:
            timer_context = timer.time_operation(f"video.{video_id}.mov_to_mp4_conversion")
            timer_context.__enter__()

        # Run FFmpeg conversion
        logger.info(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
        start_time = time.time()

        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        conversion_time = time.time() - start_time

        # Stop timer if provided
        if timer_context:
            timer_context.__exit__(None, None, None)

        # Check if conversion was successful
        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            return {
                'success': False,
                'error': f"FFmpeg error: {result.stderr[:500]}",
                'returncode': result.returncode
            }

        # Validate output file
        if not os.path.exists(output_mp4_path):
            raise FileNotFoundError(f"Output MP4 file not created: {output_mp4_path}")

        output_size = os.path.getsize(output_mp4_path)
        if output_size == 0:
            raise ValueError("Output MP4 file is empty")

        logger.info(f"✅ Conversion successful: {output_size / (1024*1024):.2f} MB in {conversion_time:.2f}s")

        return {
            'success': True,
            'output_path': output_mp4_path,
            'file_size': output_size,
            'conversion_time': conversion_time,
            'input_size': os.path.getsize(input_mov_path)
        }

    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg conversion timed out after 1 hour")
        return {
            'success': False,
            'error': 'Conversion timeout (exceeded 1 hour)'
        }

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }