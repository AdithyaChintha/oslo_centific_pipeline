#!/usr/bin/env python3
"""
Video processing utilities for downscaling and format conversion.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from utils.logger import get_logger

logger = get_logger("VideoProcessing")


def downscale_video_to_480p(input_video_path: str, output_video_path: Optional[str] = None, 
                           target_height: int = 480, quality: str = "medium") -> str:
    """
    Downscale a video to 480p (or specified height) while maintaining aspect ratio.
    
    Args:
        input_video_path: Path to the input video file
        output_video_path: Path for the output video (if None, auto-generates)
        target_height: Target height in pixels (default: 480)
        quality: Quality preset ("fast", "medium", "slow")
        
    Returns:
        str: Path to the downscaled video file
        
    Raises:
        RuntimeError: If ffmpeg is not available or processing fails
    """
    input_path = Path(input_video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    
    # Generate output path if not provided
    if output_video_path is None:
        output_path = input_path.parent / f"{input_path.stem}_480p{input_path.suffix}"
    else:
        output_path = Path(output_video_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Quality presets
    quality_settings = {
        "fast": {"crf": 28, "preset": "ultrafast"},
        "medium": {"crf": 23, "preset": "medium"},
        "slow": {"crf": 18, "preset": "slow"}
    }
    
    if quality not in quality_settings:
        quality = "medium"
    
    crf = quality_settings[quality]["crf"]
    preset = quality_settings[quality]["preset"]
    
    # Build ffmpeg command for downscaling
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-vf", f"scale=-2:{target_height}",  # Scale to target height, maintain aspect ratio
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "aac",  # Re-encode audio to ensure compatibility
        "-b:a", "128k",  # Audio bitrate
        "-movflags", "+faststart",  # Optimize for streaming
        "-pix_fmt", "yuv420p",  # Ensure compatibility
        str(output_path)
    ]
    
    logger.info(f"Downscaling video: {input_path.name} -> {output_path.name} ({target_height}p)")
    
    try:
        # Run ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"✅ Successfully downscaled video to {target_height}p: {output_path}")
        return str(output_path)
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg failed to downscale video: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except FileNotFoundError:
        error_msg = "FFmpeg not found in PATH. Please install FFmpeg."
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def get_video_dimensions(video_path: str) -> Tuple[int, int]:
    """
    Get video dimensions using ffprobe.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple[int, int]: (width, height)
        
    Raises:
        RuntimeError: If ffprobe fails or video not found
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "v:0",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        
        if not data.get("streams"):
            raise RuntimeError("No video streams found")
        
        stream = data["streams"][0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        
        if width == 0 or height == 0:
            raise RuntimeError("Invalid video dimensions")
        
        return width, height
        
    except subprocess.CalledProcessError as e:
        error_msg = f"FFprobe failed: {e.stderr}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        error_msg = f"Failed to parse video metadata: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def is_video_already_480p_or_smaller(video_path: str) -> bool:
    """
    Check if video is already 480p or smaller.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        bool: True if video is 480p or smaller
    """
    try:
        width, height = get_video_dimensions(video_path)
        return height <= 480
    except Exception as e:
        logger.warning(f"Could not check video dimensions: {e}")
        return False


def downscale_erp_video_if_needed(erp_video_path: str, target_height: int = 480) -> str:
    """
    Downscale ERP video to 480p if it's larger, otherwise return original path.
    
    Args:
        erp_video_path: Path to the ERP video file
        target_height: Target height in pixels (default: 480)
        
    Returns:
        str: Path to the (possibly downscaled) video file
    """
    if is_video_already_480p_or_smaller(erp_video_path):
        logger.info(f"Video is already {target_height}p or smaller: {Path(erp_video_path).name}")
        return erp_video_path
    
    # Downscale the video
    return downscale_video_to_480p(erp_video_path, target_height=target_height)


if __name__ == "__main__":
    # Test the functions
    import sys
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
        try:
            width, height = get_video_dimensions(test_video)
            print(f"Video dimensions: {width}x{height}")
            
            if not is_video_already_480p_or_smaller(test_video):
                downscaled = downscale_video_to_480p(test_video)
                print(f"Downscaled video: {downscaled}")
            else:
                print("Video is already 480p or smaller")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python video_processing.py <video_path>")
