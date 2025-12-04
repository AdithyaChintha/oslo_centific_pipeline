#!/usr/bin/env python3
"""
Standalone Video Downscaling Test Script

This script allows you to test video downscaling to 480p resolution.
You can pass a video path and it will downscale it and give you the output video.

Usage:
    python test_video_downscale.py <video_path> [--output OUTPUT_PATH] [--height HEIGHT] [--quality QUALITY]

Examples:
    # Basic usage (auto-generates output path)
    python test_video_downscale.py /path/to/video.mp4

    # Specify output path
    python test_video_downscale.py /path/to/video.mp4 --output /path/to/output_480p.mp4

    # Custom resolution
    python test_video_downscale.py /path/to/video.mp4 --height 720

    # Different quality preset
    python test_video_downscale.py /path/to/video.mp4 --quality slow

Quality presets:
    - fast: Fastest encoding, lower quality (CRF 28)
    - medium: Balanced encoding and quality (CRF 23) [DEFAULT]
    - slow: Slower encoding, higher quality (CRF 18)

Author: Oslo Video Processing Team
"""

import os
import sys
import argparse
from pathlib import Path

# Import the video processing utilities
try:
    from utils.video_processing import (
        downscale_video_to_480p,
        get_video_dimensions,
        is_video_already_480p_or_smaller
    )
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.video_processing import (
        downscale_video_to_480p,
        get_video_dimensions,
        is_video_already_480p_or_smaller
    )


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    """Main function to test video downscaling."""
    parser = argparse.ArgumentParser(
        description='Test video downscaling to 480p (or custom resolution)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('video_path',
                       help='Path to the input video file')
    parser.add_argument('--output', '-o',
                       help='Path for the output downscaled video (auto-generated if not specified)')
    parser.add_argument('--height', '-H', type=int, default=480,
                       help='Target height in pixels (default: 480)')
    parser.add_argument('--quality', '-q', choices=['fast', 'medium', 'slow'], default='medium',
                       help='Quality preset: fast/medium/slow (default: medium)')
    parser.add_argument('--skip-if-smaller', action='store_true',
                       help='Skip downscaling if video is already at target resolution or smaller')

    args = parser.parse_args()

    # Validate input video
    input_path = Path(args.video_path)
    if not input_path.exists():
        print(f"❌ Error: Input video not found: {args.video_path}")
        sys.exit(1)

    if not input_path.is_file():
        print(f"❌ Error: Not a file: {args.video_path}")
        sys.exit(1)

    print("="*80)
    print("🎬 Video Downscaling Test Script")
    print("="*80)

    # Get input video information
    try:
        input_size = input_path.stat().st_size
        print(f"\n📹 Input Video: {input_path.name}")
        print(f"   Path: {input_path}")
        print(f"   Size: {format_file_size(input_size)}")

        # Get video dimensions
        width, height = get_video_dimensions(str(input_path))
        print(f"   Resolution: {width}x{height}")

    except Exception as e:
        print(f"❌ Error reading video metadata: {e}")
        sys.exit(1)

    # Check if downscaling is needed
    if args.skip_if_smaller and height <= args.height:
        print(f"\n✅ Video is already {args.height}p or smaller. Skipping downscaling.")
        print(f"   Current height: {height}px, Target height: {args.height}px")
        sys.exit(0)

    if height <= args.height:
        print(f"\n⚠️  Warning: Video height ({height}px) is already at or below target ({args.height}px)")
        print(f"   Downscaling will still proceed but may not reduce file size significantly.")

    # Downscale the video
    print(f"\n🔄 Downscaling to {args.height}p...")
    print(f"   Quality preset: {args.quality}")

    try:
        output_video_path = downscale_video_to_480p(
            input_video_path=str(input_path),
            output_video_path=args.output,
            target_height=args.height,
            quality=args.quality
        )

        # Get output video information
        output_path = Path(output_video_path)
        output_size = output_path.stat().st_size

        # Get output video dimensions
        out_width, out_height = get_video_dimensions(output_video_path)

        # Calculate compression ratio
        compression_ratio = (1 - output_size / input_size) * 100

        print(f"\n✅ Downscaling completed successfully!")
        print(f"\n📹 Output Video: {output_path.name}")
        print(f"   Path: {output_path}")
        print(f"   Size: {format_file_size(output_size)}")
        print(f"   Resolution: {out_width}x{out_height}")

        print(f"\n📊 Comparison:")
        print(f"   Original: {width}x{height} ({format_file_size(input_size)})")
        print(f"   Downscaled: {out_width}x{out_height} ({format_file_size(output_size)})")
        print(f"   Size reduction: {compression_ratio:.1f}%")

        if compression_ratio > 0:
            print(f"   Saved: {format_file_size(input_size - output_size)}")

        print("="*80)

    except Exception as e:
        print(f"\n❌ Error during downscaling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
