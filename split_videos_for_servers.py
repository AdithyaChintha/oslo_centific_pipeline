#!/usr/bin/env python3
"""
Split S3 videos into separate input files for multiple servers.

Usage:
    python split_videos_for_servers.py --servers 2

This will create:
    - server_1_videos.txt (50% of unprocessed videos)
    - server_2_videos.txt (50% of unprocessed videos)

Only includes videos that haven't been processed yet (checks state file).
"""

import argparse
import yaml
import os
from utils.s3_utils import create_s3_client, discover_videos_in_s3
from utils.logger import get_logger

logger = get_logger("VideoSplitter")


def load_s3_config(config_path: str = "config/s3_config.yaml") -> dict:
    """Load S3 configuration from YAML file with environment variable expansion."""
    import re

    with open(config_path, 'r') as f:
        content = f.read()

    # Expand environment variables in the format ${VAR_NAME} or $VAR_NAME
    def replace_env_var(match):
        var_name = match.group(1) or match.group(2)
        value = os.environ.get(var_name)
        if value is None:
            logger.warning(f"Environment variable '{var_name}' not found, leaving as-is")
            return match.group(0)
        return value

    # Replace ${VAR_NAME} and $VAR_NAME patterns
    content = re.sub(r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)', replace_env_var, content)

    return yaml.safe_load(content)


def split_videos_into_files(videos: list, num_servers: int, output_dir: str = "."):
    """
    Split videos into separate files for each server.

    Args:
        videos: List of video dictionaries from S3
        num_servers: Number of servers to split across
        output_dir: Directory to save output files
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Calculate videos per server
    total_videos = len(videos)
    videos_per_server = total_videos // num_servers
    remainder = total_videos % num_servers

    logger.info(f"Splitting {total_videos} videos across {num_servers} servers")
    logger.info(f"~{videos_per_server} videos per server")

    start_idx = 0
    for server_id in range(1, num_servers + 1):
        # Calculate how many videos this server gets
        # Distribute remainder videos to first servers
        count = videos_per_server + (1 if server_id <= remainder else 0)
        end_idx = start_idx + count

        # Get videos for this server
        server_videos = videos[start_idx:end_idx]

        # Write to file
        output_file = os.path.join(output_dir, f"server_{server_id}_videos.txt")
        with open(output_file, 'w') as f:
            for video in server_videos:
                # Write S3 key and etag (for deduplication)
                f.write(f"{video['key']}\t{video['etag']}\t{video['size']}\n")

        logger.info(f"Server {server_id}: {len(server_videos)} videos → {output_file}")
        logger.info(f"  Range: video {start_idx + 1} to {end_idx}")

        start_idx = end_idx

    logger.info(f"✅ Successfully split videos into {num_servers} files")
    logger.info(f"📁 Files saved to: {output_dir}/")


def filter_unprocessed_videos(all_videos: list, state_file: str) -> list:
    """
    Filter out videos that have already been completed or failed.

    Args:
        all_videos: All videos from S3
        state_file: Path to state JSON file

    Returns:
        List of videos to process (excludes only completed and failed)
    """
    import json

    # Load state file directly
    with open(state_file, 'r') as f:
        state_data = json.load(f)

    videos_state = state_data.get('videos', {})

    # Count statuses by reading actual video entries
    status_counts = {}
    for video_info in videos_state.values():
        status = video_info.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1

    videos_to_process = []

    for video in all_videos:
        s3_key = video['key']

        # Check if video exists in state
        if s3_key in videos_state:
            status = videos_state[s3_key].get('status')

            # ONLY exclude completed and failed
            # Include: discovered, downloading, processing, and videos not in state
            if status in ['completed', 'failed']:
                continue

        videos_to_process.append(video)

    logger.info(f"Filtered: {len(all_videos)} total → {len(videos_to_process)} to process")

    # Show breakdown by actual status counts
    logger.info(f"State file breakdown (actual counts):")
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status.capitalize()}: {count}")

    logger.info(f"Excluding: {status_counts.get('completed', 0)} completed + {status_counts.get('failed', 0)} failed")

    return videos_to_process


def main():
    parser = argparse.ArgumentParser(
        description="Split S3 videos into input files for multiple servers"
    )
    parser.add_argument(
        '--servers',
        type=int,
        default=2,
        help='Number of servers to split across (default: 2)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/s3_config.yaml',
        help='Path to S3 config file (default: config/s3_config.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./video_lists',
        help='Directory to save output files (default: ./video_lists)'
    )
    parser.add_argument(
        '--state-file',
        type=str,
        default=None,
        help='Path to state file (default: from config)'
    )
    parser.add_argument(
        '--include-all',
        action='store_true',
        help='Include all videos, even if already processed (default: only unprocessed)'
    )

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("S3 Video Splitter for Multi-Server Processing")
    logger.info("="*60)

    # Load S3 configuration
    logger.info(f"Loading S3 config from: {args.config}")
    s3_config = load_s3_config(args.config)

    # Initialize state tracker
    if args.state_file:
        state_file = args.state_file
    else:
        state_file = os.path.join(
            s3_config['processing']['output_dir'],
            s3_config['state_tracking']['state_file']
        )

    logger.info(f"Using state file: {state_file}")

    if not os.path.exists(state_file):
        logger.warning(f"⚠️ State file not found: {state_file}")
        logger.warning("Will treat all videos as unprocessed")
        state_file_exists = False
    else:
        state_file_exists = True
        logger.info(f"✅ State file found")

    # Create S3 client
    s3_client = create_s3_client(s3_config)
    bucket = s3_config['s3']['bucket_name']
    input_prefix = s3_config['s3']['input_prefix']

    # Discover all videos
    logger.info(f"Discovering videos in s3://{bucket}/{input_prefix}")
    all_videos = discover_videos_in_s3(s3_client, bucket, input_prefix)
    logger.info(f"Found {len(all_videos)} total videos in S3")

    if not all_videos:
        logger.error("No videos found! Check your S3 configuration.")
        return

    # Filter to unprocessed videos only (unless --include-all)
    if args.include_all:
        logger.info("📋 Including ALL videos (--include-all flag set)")
        videos_to_split = all_videos
    elif state_file_exists:
        logger.info("🔍 Filtering videos (excluding only completed and failed)...")
        videos_to_split = filter_unprocessed_videos(all_videos, state_file)
    else:
        logger.info("⚠️ No state file - treating all videos as unprocessed")
        videos_to_split = all_videos

    if not videos_to_split:
        logger.error("❌ No unprocessed videos to split!")
        logger.info("All videos have already been processed according to state file.")
        return

    # Split videos
    split_videos_into_files(videos_to_split, args.servers, args.output_dir)

    # Print summary
    logger.info("")
    logger.info("="*60)
    logger.info("NEXT STEPS:")
    logger.info("="*60)
    for server_id in range(1, args.servers + 1):
        logger.info(f"Server {server_id}:")
        logger.info(f"  1. Copy video_lists/server_{server_id}_videos.txt to server")
        logger.info(f"  2. Run: python ray_pipeline_testing_csam_s3.py \\")
        logger.info(f"            --input-file video_lists/server_{server_id}_videos.txt \\")
        logger.info(f"            --server-id {server_id}")
        logger.info("")


if __name__ == "__main__":
    main()
