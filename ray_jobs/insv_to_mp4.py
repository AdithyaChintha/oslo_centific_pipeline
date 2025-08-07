import ray
import subprocess
import os
from pathlib import Path
import logging

logger = logging.getLogger("insv_to_mp4")

@ray.remote
def convert_insv_to_mp4(insv_path: str, output_dir: str = "/tmp/converted_mp4s") -> str:
    """
    Convert .insv video file to .mp4 using ffmpeg and exiftool (based on insv-to-yt).

    Args:
        insv_path (str): Path to the .insv file.
        output_dir (str): Directory where the .mp4 file will be saved.

    Returns:
        str: Path to the converted .mp4 file or error message.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        insv_path = Path(insv_path)
        base_name = insv_path.stem
        mp4_path = Path(output_dir) / f"{base_name}.mp4"

        # Step 1: Extract metadata using exiftool (optional, for timestamp validation/debugging)
        try:
            exif_cmd = ["exiftool", str(insv_path)]
            exif_output = subprocess.check_output(exif_cmd, stderr=subprocess.DEVNULL).decode()
            logger.info(f"[EXIF] Metadata for {insv_path}:\n{exif_output}")
        except subprocess.CalledProcessError:
            logger.warning(f"ExifTool metadata read failed for {insv_path} — continuing...")

        # Step 2: Convert using ffmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(insv_path),
            "-c", "copy",  # Rewrap the stream without re-encoding
            str(mp4_path)
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        logger.info(f"[SUCCESS] Converted {insv_path.name} to {mp4_path.name}")
        return str(mp4_path)

    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] ffmpeg failed: {e}")
        return f"Conversion failed for {insv_path}: {str(e)}"
    except Exception as e:
        logger.exception(f"[EXCEPTION] {e}")
        return f"Unexpected error during conversion: {e}"