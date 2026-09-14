#!/usr/bin/env python3
"""
Standalone Dual-Fisheye Detection and ERP Conversion Test Script

This script tests the dual-fisheye video detection and ERP (Equirectangular)
conversion functionality extracted from sharepoint_tar_inspector.py

Usage:
    python erp_dual.py <input_video_path> [output_video_path]

Example:
    python erp_dual.py /home/vision_ai_adm/code/oslo/combined_branch/comparision_movements/P02_scenarioRunner_c007.mp4
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('erp_dual_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def is_dual_fisheye_video(video_path: str) -> bool:
    """
    Detect if a video is a dual-fisheye video (like Insta360 INSV format).

    Detection criteria:
    1. Check if video has 2 video streams (dual-lens cameras)
    2. Check for 2:1 aspect ratio (common for dual-fisheye)
    3. Check file extension (.insv is Insta360 format)

    Args:
        video_path: Path to the video file

    Returns:
        True if the video is detected as dual-fisheye, False otherwise
    """
    try:
        logger.info("=" * 80)
        logger.info("STEP 1: DUAL-FISHEYE DETECTION")
        logger.info("=" * 80)

        # Check file extension first (quick check)
        file_ext = os.path.splitext(video_path)[1].lower()
        logger.debug(f"File extension: {file_ext}")

        if file_ext in ['.insv']:
            logger.info("✅ DETECTED: File has .insv extension (Insta360 format)")
            return True

        # Only check video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.insv']
        if file_ext not in video_extensions:
            logger.warning(f"⚠️  Not a video file: {file_ext}")
            return False

        logger.info(f"Analyzing video file: {os.path.basename(video_path)}")

        # Use ffprobe to get video stream info
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v',
            '-show_entries', 'stream=index,width,height,codec_name',
            '-of', 'json',
            video_path
        ]

        logger.debug(f"Running ffprobe command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            logger.error(f"❌ ffprobe failed: {result.stderr}")
            return False

        data = json.loads(result.stdout)
        streams = data.get('streams', [])

        logger.info(f"Found {len(streams)} video stream(s)")

        # Log detailed stream information
        for i, stream in enumerate(streams):
            width = stream.get('width', 'unknown')
            height = stream.get('height', 'unknown')
            codec = stream.get('codec_name', 'unknown')
            logger.info(f"  Stream {i}: {width}x{height}, codec: {codec}")

        # Check 1: Does it have 2 video streams? (dual-lens indicator)
        if len(streams) >= 2:
            logger.info("✅ DETECTED: Video has 2+ video streams (dual-lens camera)")
            return True

        # Check 2: Does it have 2:1 aspect ratio? (common for dual-fisheye)
        if len(streams) == 1:
            width = streams[0].get('width', 0)
            height = streams[0].get('height', 0)

            if width > 0 and height > 0:
                aspect_ratio = width / height
                logger.debug(f"Aspect ratio: {aspect_ratio:.2f} (width/height = {width}/{height})")

                # Check if aspect ratio is close to 2:1 (allowing some tolerance)
                # Common dual-fisheye resolutions: 3840x1920, 5760x2880, 7680x3840
                if 1.9 <= aspect_ratio <= 2.1:
                    logger.info(f"✅ DETECTED: Video has 2:1 aspect ratio ({width}x{height})")
                    return True
                else:
                    logger.info(f"ℹ️  Not dual-fisheye: Aspect ratio {aspect_ratio:.2f} is not 2:1")

        logger.info("❌ NOT DETECTED: Video is not dual-fisheye format")
        return False

    except subprocess.TimeoutExpired:
        logger.error("❌ ffprobe command timed out")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse ffprobe output: {e}")
        return False
    except FileNotFoundError:
        logger.error("❌ ffprobe not found. Please install ffmpeg.")
        return False
    except Exception as e:
        logger.error(f"❌ Error during detection: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def unwarp_dual_fisheye_to_erp(input_video_path: str, output_video_path: str) -> bool:
    """
    Convert dual-fisheye video to ERP (Equirectangular) view using ffmpeg.

    Args:
        input_video_path: Path to dual-fisheye video
        output_video_path: Path for output ERP video

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: ERP CONVERSION")
        logger.info("=" * 80)
        logger.info(f"Input:  {input_video_path}")
        logger.info(f"Output: {output_video_path}")

        # First, check if video has multiple streams (dual-lens)
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v',
            '-show_entries', 'stream=index,width,height',
            '-of', 'json',
            input_video_path
        ]

        logger.debug(f"Probing video streams: {' '.join(probe_cmd)}")
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if probe_result.returncode != 0:
            logger.error(f"❌ Failed to probe video: {probe_result.stderr}")
            return False

        streams = json.loads(probe_result.stdout).get('streams', [])
        num_video_streams = len(streams)

        logger.info(f"Video analysis: {num_video_streams} video stream(s) found")
        if streams:
            for i, stream in enumerate(streams):
                width = stream.get('width', 'unknown')
                height = stream.get('height', 'unknown')
                logger.info(f"  Stream {i}: {width}x{height}")

        # Lens FOV for Insta360 (typical 190-200 degrees)
        LENS_FOV_DEG = 190.0
        # Target ERP resolution (2:1 aspect ratio)
        ERP_W, ERP_H = 5760, 2880

        logger.info(f"Target ERP resolution: {ERP_W}x{ERP_H}")
        logger.info(f"Lens FOV: {LENS_FOV_DEG} degrees")

        # Build ffmpeg command based on number of video streams
        if num_video_streams >= 2:
            # Dual-lens: hstack two streams and convert to ERP, then rotate 180 degrees
            logger.info("🎬 Mode: DUAL-STREAM conversion (hstack + v360 + rotate 180°)")
            filter_complex = (
                f"[0:v:0][0:v:1]hstack=inputs=2[dual];"
                f"[dual]v360=input=dfisheye:output=equirect:"
                f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}[erp];"
                f"[erp]hflip,vflip"
            )
            logger.debug(f"Filter complex: {filter_complex}")
            logger.info("  ↻ Applying 180° rotation (hflip + vflip) to correct orientation")

            # FFmpeg command for dual-stream (requires -filter_complex)
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-i', input_video_path,
                '-filter_complex', filter_complex,
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                '-c:a', 'copy',  # Copy audio stream
                '-movflags', '+faststart',
                output_video_path
            ]
        else:
            # Single lens fisheye: convert directly to ERP
            logger.info("🎬 Mode: SINGLE-STREAM conversion (v360 only)")
            vf_filter = (
                f"v360=input=fisheye:output=equirect:"
                f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}"
            )
            logger.debug(f"Video filter: {vf_filter}")

            # FFmpeg command for single-stream (can use -vf)
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-i', input_video_path,
                '-vf', vf_filter,
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                '-c:a', 'copy',  # Copy audio stream
                '-movflags', '+faststart',
                output_video_path
            ]

        logger.info(f"\nRunning ffmpeg command:")
        logger.info(f"  {' '.join(ffmpeg_cmd)}")
        logger.info("\n⏳ Converting... This may take a while for large videos...")

        # Run ffmpeg conversion
        result = subprocess.run(
            ffmpeg_cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for large videos
        )

        if result.returncode != 0:
            logger.error(f"\n❌ FFmpeg conversion FAILED!")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"\nSTDERR Output:")
            logger.error(result.stderr)
            if result.stdout:
                logger.debug(f"\nSTDOUT Output:")
                logger.debug(result.stdout)
            return False

        if os.path.exists(output_video_path):
            output_size = os.path.getsize(output_video_path)
            input_size = os.path.getsize(input_video_path)
            logger.info(f"\n✅ ERP CONVERSION SUCCESSFUL!")
            logger.info(f"  Output file: {output_video_path}")
            logger.info(f"  Input size:  {input_size / (1024**2):.2f} MB")
            logger.info(f"  Output size: {output_size / (1024**2):.2f} MB")
            return True
        else:
            logger.error(f"❌ Output file was not created: {output_video_path}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"❌ FFmpeg conversion timed out (exceeded 1 hour)")
        return False
    except Exception as e:
        logger.error(f"❌ Error during ERP conversion: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function."""
    print("\n" + "=" * 80)
    print("DUAL-FISHEYE DETECTION AND ERP CONVERSION TEST")
    print("=" * 80)

    if len(sys.argv) < 2:
        print("\nUsage: python erp_dual.py <input_video_path> [output_video_path]")
        print("\nExample:")
        print("  python erp_dual.py /path/to/video.mp4")
        print("  python erp_dual.py /path/to/video.mp4 /path/to/output_erp.mp4")
        sys.exit(1)

    input_video_path = sys.argv[1]

    # Check if input file exists
    if not os.path.exists(input_video_path):
        logger.error(f"❌ Input file does not exist: {input_video_path}")
        sys.exit(1)

    # Generate output path if not provided
    if len(sys.argv) >= 3:
        output_video_path = sys.argv[2]
    else:
        # Auto-generate output name: input_erpview.mp4
        input_dir = os.path.dirname(input_video_path)
        input_basename = os.path.basename(input_video_path)
        input_name_without_ext = os.path.splitext(input_basename)[0]
        output_video_path = os.path.join(input_dir, f"{input_name_without_ext}_erpview.mp4")

    logger.info(f"Input video:  {input_video_path}")
    logger.info(f"Output video: {output_video_path}")
    logger.info(f"Log file:     erp_dual_test.log")

    # Step 1: Detect if video is dual-fisheye
    is_dual_fisheye = is_dual_fisheye_video(input_video_path)

    if not is_dual_fisheye:
        logger.warning("\n⚠️  Video is NOT detected as dual-fisheye format")
        logger.warning("Do you still want to attempt ERP conversion? (y/n)")
        user_input = input().strip().lower()
        if user_input != 'y':
            logger.info("Conversion cancelled by user")
            sys.exit(0)

    # Step 2: Convert to ERP
    success = unwarp_dual_fisheye_to_erp(input_video_path, output_video_path)

    if success:
        print("\n" + "=" * 80)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"ERP video saved to: {output_video_path}")
        print(f"Detailed logs saved to: erp_dual_test.log")
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print("Check erp_dual_test.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
