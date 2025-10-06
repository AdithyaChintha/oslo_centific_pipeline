import subprocess
import os
import ray
from typing import Dict
from utils.logger import get_logger
import time

logger = get_logger("MOV2MP4")

@ray.remote
def convert_mov_to_mp4_task(
    input_mov_path: str,
    output_mp4_path: str,
    video_id: str,
    timing_id: str = None
) -> Dict[str, any]:
    """
    Ray remote task to convert MOV file to MP4 format using FFmpeg.

    This task runs as a separate Ray job and can be parallelized across multiple videos.

    Args:
        input_mov_path: Path to input MOV file
        output_mp4_path: Path to save output MP4 file
        video_id: Video identifier for logging
        timing_id: Optional timing identifier for tracking conversion time

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

        # Run FFmpeg conversion
        logger.info(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
        start_time = time.time()

        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3600  # 1 hour timeout
        )

        conversion_time = time.time() - start_time
        end_time = time.time()

        # Check if conversion was successful
        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore') if isinstance(result.stderr, bytes) else result.stderr
            logger.error(f"FFmpeg conversion failed: {error_msg}")
            return {
                'success': False,
                'error': f"FFmpeg error: {error_msg[:500]}",
                'returncode': result.returncode
            }

        # Validate output file
        if not os.path.exists(output_mp4_path):
            raise FileNotFoundError(f"Output MP4 file not created: {output_mp4_path}")

        output_size = os.path.getsize(output_mp4_path)
        if output_size == 0:
            raise ValueError("Output MP4 file is empty")

        logger.info(f"✅ Conversion successful: {output_size / (1024*1024):.2f} MB in {conversion_time:.2f}s")

        # Prepare timing data in format compatible with main timer (only if timing_id provided)
        timing_data = None
        if timing_id:
            timing_data = {
                'execution_time': conversion_time,
                'start_time': start_time,
                'end_time': end_time,
                'status': 'completed',
                'timing_id': timing_id
            }

        return {
            'success': True,
            'output_path': output_mp4_path,
            'file_size': output_size,
            'conversion_time': conversion_time,
            'input_size': os.path.getsize(input_mov_path),
            'timing': timing_data
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