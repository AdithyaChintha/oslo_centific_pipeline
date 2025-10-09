#!/usr/bin/env python3
"""
Check processing progress for multi-server video processing.
Compares input file with state file to show remaining videos.
"""

import json
import argparse
from pathlib import Path


def check_progress(input_file: str, state_file: str):
    """
    Check how many videos are left to process.

    Args:
        input_file: Path to server's input video list
        state_file: Path to server's state JSON file
    """
    # Load input videos
    print(f"\n{'='*70}")
    print(f"Processing Progress Check")
    print(f"{'='*70}\n")

    print(f"📂 Input file: {input_file}")
    print(f"📂 State file: {state_file}")
    print()

    # Read all videos from input file
    input_videos = set()
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                s3_key = line.split('\t')[0]
                input_videos.add(s3_key)

    total_input = len(input_videos)
    print(f"📊 Total videos in input file: {total_input:,}")

    # Load state file
    if not Path(state_file).exists():
        print(f"\n⚠️  State file not found!")
        print(f"   All {total_input:,} videos need to be processed")
        return

    with open(state_file, 'r') as f:
        state_data = json.load(f)

    videos_state = state_data.get('videos', {})

    # Count by status
    status_counts = {
        'completed': 0,
        'failed': 0,
        'processing': 0,
        'downloading': 0,
        'discovered': 0
    }

    completed_videos = set()
    failed_videos = set()

    for s3_key, video_info in videos_state.items():
        if s3_key in input_videos:  # Only count videos that are in our input file
            status = video_info.get('status', 'unknown')
            if status == 'completed':
                status_counts['completed'] += 1
                completed_videos.add(s3_key)
            elif status == 'failed':
                status_counts['failed'] += 1
                failed_videos.add(s3_key)
            elif status == 'processing':
                status_counts['processing'] += 1
            elif status == 'downloading':
                status_counts['downloading'] += 1
            elif status == 'discovered':
                status_counts['discovered'] += 1

    # Calculate remaining
    remaining = input_videos - completed_videos - failed_videos
    remaining_count = len(remaining)

    # Display results
    print(f"\n{'─'*70}")
    print(f"Status Breakdown:")
    print(f"{'─'*70}")
    print(f"  ✅ Completed:   {status_counts['completed']:,}")
    print(f"  ❌ Failed:      {status_counts['failed']:,}")
    print(f"  ⏳ Processing:  {status_counts['processing']:,}")
    print(f"  ⬇️  Downloading: {status_counts['downloading']:,}")
    print(f"  📋 Discovered:  {status_counts['discovered']:,}")
    print(f"{'─'*70}")
    print(f"  🎯 Remaining:   {remaining_count:,}")
    print(f"{'─'*70}")

    # Calculate progress
    processed = status_counts['completed']
    if total_input > 0:
        progress_pct = (processed / total_input) * 100
        print(f"\n📈 Progress: {processed:,} / {total_input:,} ({progress_pct:.2f}%)")

        # Estimate remaining (if we have completed videos)
        if processed > 0:
            # Get timing estimate if available
            print(f"\n💡 Videos to process: {remaining_count:,}")
            print(f"   (Excludes {status_counts['failed']:,} failed videos)")

    # Show some sample remaining videos (first 5)
    if remaining_count > 0:
        print(f"\n📝 Sample of remaining videos (first 5):")
        for i, s3_key in enumerate(list(remaining)[:5], 1):
            print(f"   {i}. {s3_key}")
        if remaining_count > 5:
            print(f"   ... and {remaining_count - 5:,} more")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check video processing progress"
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to input video list file'
    )
    parser.add_argument(
        '--state-file',
        type=str,
        required=True,
        help='Path to state JSON file'
    )

    args = parser.parse_args()

    try:
        check_progress(args.input_file, args.state_file)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
