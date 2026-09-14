#!/usr/bin/env python3
"""
Comprehensive Tracking State Rebuilder

This script rebuilds the processed_sessions_tracking.json file from existing Azure Blob Storage data.
It can work in multiple modes:
1. Scan blobs from specific date(s) and mark sessions as processed
2. Take a list of session names and mark them as processed
3. Scan processed output directory and mark sessions as processed

Usage:
    # Rebuild from specific dates
    python rebuild_tracking_state.py --config config/pipeline_config.yaml --from-date 2025-09-01 --to-date 2025-09-16

    # Rebuild from session names list
    python rebuild_tracking_state.py --config config/pipeline_config.yaml --session-names "session1,session2,session3"

    # Rebuild from processed output directory
    python rebuild_tracking_state.py --config config/pipeline_config.yaml --from-output-dir ./output_test/test

    # Scan all sessions in blob and mark as processed
    python rebuild_tracking_state.py --config config/pipeline_config.yaml --scan-all-blobs
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Set, Dict, Optional
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rebuild_tracking_state.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_config(config_path: str) -> dict:
    """Load pipeline configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Loaded config from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load config from {config_path}: {e}")
        raise

def create_azure_blob_client(azure_config: dict) -> BlobServiceClient:
    """Create Azure Blob Service client"""
    account_url = f"https://{azure_config['account-name']}.blob.core.windows.net"
    return BlobServiceClient(
        account_url=account_url,
        credential=azure_config['account-key']
    )

def get_blob_last_modified_date(blob_client, container_name: str, blob_name: str) -> Optional[datetime]:
    """Get the last modified date of a blob"""
    try:
        container_client = blob_client.get_container_client(container_name)
        blob_properties = container_client.get_blob_client(blob_name).get_blob_properties()
        return blob_properties.last_modified.replace(tzinfo=None)  # Remove timezone for comparison
    except Exception as e:
        logger.debug(f"Could not get blob properties for {blob_name}: {e}")
        return None

def discover_sessions_by_date_range(blob_client, container_name: str, base_prefix: str,
                                  from_date: datetime, to_date: datetime) -> Set[str]:
    """
    Discover sessions that have blobs within the specified date range

    Args:
        blob_client: Azure blob service client
        container_name: Container name
        base_prefix: Base prefix to search under (e.g., "one-data-platform/")
        from_date: Start date (inclusive)
        to_date: End date (inclusive)

    Returns:
        Set of session IDs found within the date range
    """
    logger.info(f"🔍 Scanning blobs from {from_date.date()} to {to_date.date()} under prefix: {base_prefix}")

    sessions_found = set()
    container_client = blob_client.get_container_client(container_name)

    try:
        blobs = container_client.list_blobs(name_starts_with=base_prefix)

        session_dirs = set()
        total_blobs = 0
        date_matched_blobs = 0

        for blob in blobs:
            total_blobs += 1
            if total_blobs % 1000 == 0:
                logger.info(f"   Processed {total_blobs} blobs, found {len(session_dirs)} sessions so far...")

            # Check if blob is within date range
            blob_date = get_blob_last_modified_date(blob_client, container_name, blob.name)
            if blob_date and from_date <= blob_date <= to_date + timedelta(days=1):  # Include full day
                date_matched_blobs += 1

                # Extract session directory from blob path
                relative_path = blob.name[len(base_prefix):] if blob.name.startswith(base_prefix) else blob.name

                if '/' in relative_path:
                    session_dir = relative_path.split('/')[0]
                    if session_dir and session_dir not in session_dirs:
                        session_dirs.add(session_dir)
                        logger.info(f"   📁 Found session: {session_dir} (blob date: {blob_date.date()})")

        logger.info(f"✅ Scanned {total_blobs} total blobs, {date_matched_blobs} within date range")
        logger.info(f"✅ Found {len(session_dirs)} unique sessions in date range")

        return session_dirs

    except Exception as e:
        logger.error(f"❌ Error scanning blobs by date: {e}")
        return set()

def discover_all_sessions_in_blob(blob_client, container_name: str, base_prefix: str) -> Set[str]:
    """
    Discover all sessions in blob storage under the base prefix

    Args:
        blob_client: Azure blob service client
        container_name: Container name
        base_prefix: Base prefix to search under

    Returns:
        Set of all session IDs found
    """
    logger.info(f"🔍 Scanning ALL sessions under prefix: {base_prefix}")

    sessions_found = set()
    container_client = blob_client.get_container_client(container_name)

    try:
        blobs = container_client.list_blobs(name_starts_with=base_prefix)

        session_dirs = set()
        total_blobs = 0

        for blob in blobs:
            total_blobs += 1
            if total_blobs % 1000 == 0:
                logger.info(f"   Processed {total_blobs} blobs, found {len(session_dirs)} sessions so far...")

            # Extract session directory from blob path
            relative_path = blob.name[len(base_prefix):] if blob.name.startswith(base_prefix) else blob.name

            if '/' in relative_path:
                session_dir = relative_path.split('/')[0]
                if session_dir and session_dir not in session_dirs:
                    session_dirs.add(session_dir)
                    logger.debug(f"   📁 Found session: {session_dir}")

        logger.info(f"✅ Scanned {total_blobs} total blobs")
        logger.info(f"✅ Found {len(session_dirs)} unique sessions")

        return session_dirs

    except Exception as e:
        logger.error(f"❌ Error scanning all blobs: {e}")
        return set()

def discover_sessions_from_output_dir(output_dir: str) -> Set[str]:
    """
    Discover sessions from processed output directory

    Args:
        output_dir: Path to output directory containing processed session folders

    Returns:
        Set of session IDs found in output directory
    """
    logger.info(f"🔍 Scanning processed sessions in output directory: {output_dir}")

    sessions_found = set()

    if not os.path.exists(output_dir):
        logger.warning(f"⚠️ Output directory does not exist: {output_dir}")
        return sessions_found

    try:
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                # Check if this looks like a session directory (contains processing artifacts)
                session_files = os.listdir(item_path)
                if any(f.endswith('.json') or f.endswith('.jsonl') for f in session_files):
                    sessions_found.add(item)
                    logger.info(f"   📁 Found processed session: {item}")

        logger.info(f"✅ Found {len(sessions_found)} processed sessions in output directory")
        return sessions_found

    except Exception as e:
        logger.error(f"❌ Error scanning output directory: {e}")
        return set()

def parse_session_names(session_names_str: str) -> Set[str]:
    """
    Parse comma-separated session names into a set

    Args:
        session_names_str: Comma-separated session names

    Returns:
        Set of session names
    """
    if not session_names_str:
        return set()

    sessions = set()
    for name in session_names_str.split(','):
        name = name.strip()
        if name:
            sessions.add(name)

    logger.info(f"✅ Parsed {len(sessions)} session names from input")
    return sessions

def load_existing_tracking_state(tracking_file: str) -> Set[str]:
    """
    Load existing tracking state if it exists

    Args:
        tracking_file: Path to tracking file

    Returns:
        Set of already processed sessions
    """
    if not os.path.exists(tracking_file):
        logger.info(f"📋 No existing tracking file found at: {tracking_file}")
        return set()

    try:
        with open(tracking_file, 'r') as f:
            data = json.load(f)

        existing_sessions = set(data.get('processed_sessions', []))
        logger.info(f"📋 Loaded {len(existing_sessions)} existing processed sessions from: {tracking_file}")
        return existing_sessions

    except Exception as e:
        logger.warning(f"⚠️ Could not load existing tracking file {tracking_file}: {e}")
        return set()

def save_tracking_state(tracking_file: str, processed_sessions: Set[str], backup_existing: bool = True):
    """
    Save tracking state to file

    Args:
        tracking_file: Path to tracking file
        processed_sessions: Set of processed session IDs
        backup_existing: Whether to backup existing file
    """
    try:
        # Backup existing file if it exists
        if backup_existing and os.path.exists(tracking_file):
            backup_file = f"{tracking_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(tracking_file, backup_file)
            logger.info(f"📋 Backed up existing tracking file to: {backup_file}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)

        # Save tracking data in the same format as the pipeline uses
        tracking_data = {
            "processed_sessions": sorted(list(processed_sessions)),  # Sort for consistent output
            "last_updated": datetime.now().isoformat(),
            "total_processed": len(processed_sessions)
        }

        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)

        logger.info(f"💾 Saved tracking state with {len(processed_sessions)} sessions to: {tracking_file}")

    except Exception as e:
        logger.error(f"❌ Error saving tracking state: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Rebuild processed sessions tracking state')
    parser.add_argument('--config', required=True, help='Path to pipeline config file')
    parser.add_argument('--from-date', help='Start date (YYYY-MM-DD) for blob scanning')
    parser.add_argument('--to-date', help='End date (YYYY-MM-DD) for blob scanning')
    parser.add_argument('--session-names', help='Comma-separated list of session names to mark as processed')
    parser.add_argument('--from-output-dir', help='Path to output directory to scan for processed sessions')
    parser.add_argument('--scan-all-blobs', action='store_true', help='Scan all sessions in blob storage')
    parser.add_argument('--tracking-file', help='Path to tracking file (default: from config)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--merge', action='store_true', help='Merge with existing tracking state instead of replacing')

    args = parser.parse_args()

    # Validate arguments
    mode_count = sum([
        bool(args.from_date),
        bool(args.session_names),
        bool(args.from_output_dir),
        bool(args.scan_all_blobs)
    ])

    if mode_count == 0:
        logger.error("❌ Must specify one of: --from-date, --session-names, --from-output-dir, or --scan-all-blobs")
        return 1

    if mode_count > 1:
        logger.error("❌ Can only specify one mode at a time")
        return 1

    if args.from_date and not args.to_date:
        args.to_date = args.from_date  # Use same date for single day

    try:
        # Load configuration
        config = load_config(args.config)

        # Handle both nested and flat config structures
        if 'pipeline' in config:
            pipeline_config = config['pipeline']
            azure_config = config['azstorage']
        else:
            # Flat config structure - use the config directly as pipeline config
            pipeline_config = config

            # Try to load Azure config from blobfuse2_config.yaml
            blobfuse_config_path = os.path.join(current_dir, 'blobfuse2_config.yaml')
            try:
                with open(blobfuse_config_path, 'r') as f:
                    blobfuse_config = yaml.safe_load(f)
                azure_config = blobfuse_config['azstorage']
                logger.info(f"✅ Loaded Azure config from: {blobfuse_config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load blobfuse config: {e}")
                azure_config = {
                    'account-name': config.get('account_name', ''),
                    'account-key': config.get('account_key', ''),
                    'container': config.get('container_name', 'instavideo')
                }

        # Determine tracking file path
        if args.tracking_file:
            tracking_file = args.tracking_file
        else:
            output_base_dir = pipeline_config['local_storage']['output_base_dir']
            output_base_dir = os.path.join(current_dir, output_base_dir.lstrip('./'))
            tracking_file = os.path.join(output_base_dir, "processed_sessions_tracking.json")

        logger.info(f"🎯 Target tracking file: {tracking_file}")

        # Load existing state if merging
        if args.merge:
            existing_sessions = load_existing_tracking_state(tracking_file)
        else:
            existing_sessions = set()

        # Discover sessions based on mode
        new_sessions = set()

        if args.from_date:
            # Date range mode
            from_date = datetime.strptime(args.from_date, '%Y-%m-%d')
            to_date = datetime.strptime(args.to_date, '%Y-%m-%d')

            blob_client = create_azure_blob_client(azure_config)
            container_name = azure_config['container']
            base_prefix = pipeline_config['azure_storage']['input_blob_prefix']

            new_sessions = discover_sessions_by_date_range(
                blob_client, container_name, base_prefix, from_date, to_date
            )

        elif args.session_names:
            # Session names mode
            new_sessions = parse_session_names(args.session_names)

        elif args.from_output_dir:
            # Output directory mode
            new_sessions = discover_sessions_from_output_dir(args.from_output_dir)

        elif args.scan_all_blobs:
            # Scan all blobs mode
            blob_client = create_azure_blob_client(azure_config)
            container_name = azure_config['container']
            base_prefix = pipeline_config['azure_storage']['input_blob_prefix']

            new_sessions = discover_all_sessions_in_blob(
                blob_client, container_name, base_prefix
            )

        # Combine with existing sessions if merging
        all_sessions = existing_sessions | new_sessions

        # Show summary
        logger.info("=" * 80)
        logger.info("📊 TRACKING STATE REBUILD SUMMARY")
        logger.info("=" * 80)
        logger.info(f"   Existing sessions: {len(existing_sessions)}")
        logger.info(f"   New sessions found: {len(new_sessions)}")
        logger.info(f"   Total sessions: {len(all_sessions)}")

        if args.merge:
            added_sessions = new_sessions - existing_sessions
            logger.info(f"   Sessions to be added: {len(added_sessions)}")
            if added_sessions:
                logger.info("   New sessions to add:")
                for session in sorted(added_sessions):
                    logger.info(f"     + {session}")
        else:
            logger.info("   All sessions will be marked as processed")

        logger.info("=" * 80)

        if args.dry_run:
            logger.info("🔍 DRY RUN - No changes will be made")
            return 0

        # Save the updated tracking state
        save_tracking_state(tracking_file, all_sessions, backup_existing=not args.merge)

        logger.info("✅ Tracking state rebuild completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"❌ Error rebuilding tracking state: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)