import ray
import subprocess
import os
import time
from pathlib import Path
import logging

logger = logging.getLogger("insv_to_mp4")
logging.basicConfig(level=logging.INFO)


@ray.remote
def convert_insv_to_dual_mp4(insv_path: str, output_dir: str = "/tmp/converted_mp4s") -> dict:
    """
    Converts an INSV file into two separate MP4s (one for each equirectangular view).

    Args:
        insv_path (str): Path to .insv file.
        output_dir (str): Directory where output .mp4s will be stored.

    Returns:
        dict: Conversion status, timing, and paths to the output files.
    """
    insv_path = Path(insv_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = insv_path.stem
    output1 = Path(output_dir) / f"{base_name}_view1.mp4"
    output2 = Path(output_dir) / f"{base_name}_view2.mp4"

    try:
        start_time = time.time()

        # Optional: extract metadata for debugging
        try:
            exiftool_cmd = ["exiftool", str(insv_path)]
            metadata = subprocess.check_output(exiftool_cmd).decode()
            logger.info(f"[EXIF] Metadata for {insv_path.name}:\n{metadata}")
        except Exception as e:
            logger.warning(f"[WARN] ExifTool failed on {insv_path.name}: {e}")

        # Extract each video stream separately with audio
        ffmpeg_cmd1 = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(insv_path),
            "-map", "0:v:0",  # Map first video stream
            "-map", "0:a",    # Map all audio streams
            "-c", "copy",
            str(output1)
        ]
        ffmpeg_cmd2 = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(insv_path),
            "-map", "0:v:1",  # Map second video stream  
            "-map", "0:a",    # Map all audio streams
            "-c", "copy",
            str(output2)
        ]

        subprocess.run(ffmpeg_cmd1, check=True)
        subprocess.run(ffmpeg_cmd2, check=True)

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        logger.info(f"[SUCCESS] {insv_path.name} converted to {output1.name} and {output2.name} in {duration}s")

        return {
            "input": str(insv_path),
            "output_view_1": str(output1),
            "output_view_2": str(output2),
            "success": True,
            "duration_seconds": duration
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] ffmpeg failed for {insv_path.name}: {e}")
        return {
            "input": str(insv_path),
            "error": f"ffmpeg failed: {str(e)}",
            "success": False
        }
    except Exception as e:
        logger.exception(f"[EXCEPTION] {e}")
        return {
            "input": str(insv_path),
            "error": f"Unexpected error: {e}",
            "success": False
        }