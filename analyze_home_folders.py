#!/usr/bin/env python3
"""
Azure Blob Folder Analysis Script

This script analyzes Azure blob folders that match the pattern 'home_id-' (e.g., '65-59251661...')
and generates a CSV report with detailed statistics including file counts, sizes, and metadata.

Usage:
    python analyze_home_folders.py --home-id 65 --container one-data-platform --output report.csv

Requirements:
    - blobfuse2_config.yaml with Azure credentials
    - Azure blob container with folders matching home_id-* pattern
"""

import os
import sys
import json
import yaml
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import csv
import re

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Azure SDK verbose logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)


def get_connection_string_from_yaml(config_path: str = "blobfuse2_config.yaml") -> str:
    """
    Parse blobfuse2 config YAML to construct an Azure Storage connection string.

    Args:
        config_path: Path to the blobfuse2 config file

    Returns:
        Azure Storage connection string
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        azstorage_config = config.get('azstorage', {})
        account_name = azstorage_config.get('account-name')
        account_key = azstorage_config.get('account-key')

        if not all([account_name, account_key]):
            raise ValueError("'account-name' or 'account-key' not found in azstorage config.")

        connection_string = (
            f"DefaultEndpointsProtocol=https;AccountName={account_name};"
            f"AccountKey={account_key};EndpointSuffix=core.windows.net"
        )
        logger.info(f"Successfully connected to Azure account: {account_name}")
        return connection_string

    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse YAML or construct connection string: {e}")
        raise


def human_size(num_bytes: int) -> str:
    """
    Convert bytes to human readable format.

    Args:
        num_bytes: Size in bytes

    Returns:
        Human readable size string (e.g., "53.37 GB")
    """
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def parse_folder_name(folder_name: str) -> Dict[str, str]:
    """
    Parse folder name to extract domain, activity, and specific_activity.

    Expected format: "home_id-uuid-activity-details"
    Example: "65-59251661-71d9-445a-b6d9-e61cc9bc18be-cooking-breakfast"

    Args:
        folder_name: The folder name to parse

    Returns:
        Dictionary with parsed components
    """
    # Remove prefix if present
    folder_base = folder_name.split('/')[-1]

    # Try to extract activity information from folder name
    parts = folder_base.split('-')

    # Try to find activity part (after UUID pattern)
    activity_parts = []
    found_uuid_end = False
    uuid_pattern = re.compile(r'^[0-9a-f]{8,}$')

    for i, part in enumerate(parts):
        # Skip home_id (first part) and UUID parts
        if i == 0 or uuid_pattern.match(part):
            continue
        if found_uuid_end or i > 5:  # After UUID
            activity_parts.append(part)
        elif len(part) > 8:  # Likely end of UUID
            found_uuid_end = True

    activity = ' '.join(activity_parts) if activity_parts else ''

    return {
        'activity': activity,
        'specific_activity': activity,
        'domain': ''  # Will be filled from metadata if available
    }


def get_metadata_from_blob(blob_service_client: BlobServiceClient,
                           container_name: str,
                           folder_path: str) -> Optional[Dict]:
    """
    Download and parse metadata.json from a blob folder.

    The metadata file can be in two formats:
    1. Simple: folder_path/metadata.json
    2. Complex: folder_path/{home_id}_{uuid}_activity_{activity}_{timestamp}_metadata.json

    Args:
        blob_service_client: Azure BlobServiceClient instance
        container_name: Name of the container
        folder_path: Path to the folder containing metadata.json

    Returns:
        Parsed metadata dictionary or None if not found
    """
    # First, try the simple format
    metadata_path = f"{folder_path}/metadata.json"

    try:
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=metadata_path
        )

        # Download metadata
        download_stream = blob_client.download_blob()
        metadata_content = download_stream.readall()

        # Parse JSON
        metadata = json.loads(metadata_content)
        logger.debug(f"Successfully loaded metadata from {metadata_path}")
        return metadata

    except ResourceNotFoundError:
        logger.debug(f"No metadata.json found at {metadata_path}, trying pattern-based search...")

        # Try to find metadata file with pattern: *_metadata.json
        try:
            container_client = blob_service_client.get_container_client(container_name)
            blob_list = container_client.list_blobs(name_starts_with=folder_path)

            for blob in blob_list:
                if blob.name.endswith('_metadata.json'):
                    logger.debug(f"Found metadata file: {blob.name}")

                    # Download this metadata file
                    blob_client = blob_service_client.get_blob_client(
                        container=container_name,
                        blob=blob.name
                    )
                    download_stream = blob_client.download_blob()
                    metadata_content = download_stream.readall()

                    # Parse JSON
                    metadata = json.loads(metadata_content)
                    logger.debug(f"Successfully loaded metadata from {blob.name}")
                    return metadata

            logger.debug(f"No metadata file found in folder {folder_path}")
            return None

        except Exception as e:
            logger.warning(f"Error searching for metadata files in {folder_path}: {e}")
            return None

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse metadata.json at {metadata_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error reading metadata from {metadata_path}: {e}")
        return None


def analyze_folder(blob_service_client: BlobServiceClient,
                   container_name: str,
                   folder_path: str,
                   home_id: str) -> Dict:
    """
    Analyze a single folder and collect all statistics.

    Args:
        blob_service_client: Azure BlobServiceClient instance
        container_name: Name of the container
        folder_path: Path to the folder to analyze
        home_id: Home ID for this analysis

    Returns:
        Dictionary containing all folder statistics
    """
    logger.info(f"Analyzing folder: {folder_path}")

    # Initialize statistics
    stats = {
        'home_id': home_id,
        'folder_name': folder_path,
        'total_files': 0,
        'total_size_bytes': 0,
        'video_files_count': 0,
        'audio_files_count': 0,
        'metadata_files_count': 0,
        'other_files_count': 0,
        'has_metadata_json': False,
        'domain': '',
        'activity': '',
        'specific_activity': '',
        'duration_minutes': '',
        'earliest_upload': None,
        'latest_upload': None,
    }

    # File extension categories
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.insv', '.webm', '.flv'}
    audio_extensions = {'.mp3', '.wav', '.aac', '.flac', '.m4a', '.ogg'}
    metadata_extensions = {'.json', '.xml', '.yaml', '.yml', '.txt'}

    # List all blobs in the folder
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs(name_starts_with=folder_path)

    upload_dates = []

    for blob in blob_list:
        # Skip the folder itself
        if blob.name == folder_path or blob.name == f"{folder_path}/":
            continue

        # Count file
        stats['total_files'] += 1
        stats['total_size_bytes'] += blob.size

        # Track upload dates
        if blob.last_modified:
            upload_dates.append(blob.last_modified)

        # Categorize by extension
        _, ext = os.path.splitext(blob.name.lower())

        if ext in video_extensions:
            stats['video_files_count'] += 1
        elif ext in audio_extensions:
            stats['audio_files_count'] += 1
        elif ext in metadata_extensions:
            stats['metadata_files_count'] += 1
            if blob.name.endswith('metadata.json'):
                stats['has_metadata_json'] = True
        else:
            stats['other_files_count'] += 1

    # Calculate upload date statistics
    if upload_dates:
        stats['earliest_upload'] = min(upload_dates)
        stats['latest_upload'] = max(upload_dates)

    # Try to get metadata
    metadata = get_metadata_from_blob(blob_service_client, container_name, folder_path)

    if metadata:
        # Extract information from metadata
        stats['domain'] = metadata.get('domain', '')
        stats['activity'] = metadata.get('activity', '')
        stats['specific_activity'] = metadata.get('specific_activity', '')
        stats['duration_minutes'] = metadata.get('duration_minutes', '')

        # Override dates from metadata if available
        if 'earliest_upload' in metadata and metadata['earliest_upload']:
            try:
                stats['earliest_upload'] = datetime.fromisoformat(
                    metadata['earliest_upload'].replace('Z', '+00:00')
                )
            except:
                pass

        if 'latest_upload' in metadata and metadata['latest_upload']:
            try:
                stats['latest_upload'] = datetime.fromisoformat(
                    metadata['latest_upload'].replace('Z', '+00:00')
                )
            except:
                pass
    else:
        # Parse folder name if no metadata
        parsed = parse_folder_name(folder_path)
        stats.update(parsed)

    return stats


def find_home_folders(blob_service_client: BlobServiceClient,
                      container_name: str,
                      home_id: str,
                      prefix: str = "") -> List[str]:
    """
    Find all folders matching the home_id pattern.

    Args:
        blob_service_client: Azure BlobServiceClient instance
        container_name: Name of the container
        home_id: Home ID to search for (e.g., "65")
        prefix: Optional prefix to narrow search (e.g., "one-data-platform/")

    Returns:
        List of folder paths matching the pattern
    """
    logger.info(f"Searching for folders matching home_id: {home_id}")

    container_client = blob_service_client.get_container_client(container_name)

    # Pattern to match: home_id- followed by UUID and activity
    # Example: "65-59251661-71d9-445a-b6d9-e61cc9bc18be-cooking-breakfast"
    pattern = re.compile(rf'^{re.escape(prefix)}{re.escape(home_id)}-[0-9a-f]{{8}}-')

    folders = set()

    # List all blobs with the prefix
    search_prefix = f"{prefix}{home_id}-" if prefix else f"{home_id}-"
    blob_list = container_client.list_blobs(name_starts_with=search_prefix)

    for blob in blob_list:
        # Extract folder path (everything before the last /)
        parts = blob.name.rsplit('/', 1)
        if len(parts) > 1:
            folder_path = parts[0]

            # Check if this folder matches our pattern
            if pattern.match(folder_path):
                folders.add(folder_path)

    folders_list = sorted(list(folders))
    logger.info(f"Found {len(folders_list)} folders matching home_id {home_id}")

    return folders_list


def generate_csv_report(folder_stats: List[Dict], output_file: str):
    """
    Generate CSV report from folder statistics.

    Args:
        folder_stats: List of folder statistics dictionaries
        output_file: Path to output CSV file
    """
    if not folder_stats:
        logger.warning("No data to write to CSV")
        return

    # Define CSV columns
    fieldnames = [
        'home_id',
        'folder_name',
        'total_files',
        'total_size_bytes',
        'Size in GB',
        'total_size_human',
        'video_files_count',
        'audio_files_count',
        'metadata_files_count',
        'other_files_count',
        'has_metadata_json',
        'domain',
        'activity',
        'specific_activity',
        'duration_minutes',
        'earliest_upload',
        'latest_upload',
        'upload_date_range_days',
        'earliest_upload_date_only',
        'latest_upload_date_only'
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for stats in folder_stats:
            # Prepare row data
            row = {
                'home_id': stats['home_id'],
                'folder_name': stats['folder_name'],
                'total_files': stats['total_files'],
                'total_size_bytes': stats['total_size_bytes'],
                'Size in GB': f"{stats['total_size_bytes'] / (1024**3):.2E}",  # Scientific notation
                'total_size_human': human_size(stats['total_size_bytes']),
                'video_files_count': stats['video_files_count'],
                'audio_files_count': stats['audio_files_count'],
                'metadata_files_count': stats['metadata_files_count'],
                'other_files_count': stats['other_files_count'],
                'has_metadata_json': 'TRUE' if stats['has_metadata_json'] else 'FALSE',
                'domain': stats['domain'],
                'activity': stats['activity'],
                'specific_activity': stats['specific_activity'],
                'duration_minutes': stats['duration_minutes'],
                'earliest_upload': stats['earliest_upload'].isoformat() if stats['earliest_upload'] else '',
                'latest_upload': stats['latest_upload'].isoformat() if stats['latest_upload'] else '',
            }

            # Calculate date range
            if stats['earliest_upload'] and stats['latest_upload']:
                date_range = (stats['latest_upload'] - stats['earliest_upload']).days
                row['upload_date_range_days'] = date_range
                row['earliest_upload_date_only'] = stats['earliest_upload'].strftime('%-m/%-d/%Y')
                row['latest_upload_date_only'] = stats['latest_upload'].strftime('%-m/%-d/%Y')
            else:
                row['upload_date_range_days'] = ''
                row['earliest_upload_date_only'] = ''
                row['latest_upload_date_only'] = ''

            writer.writerow(row)

    logger.info(f"CSV report generated: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze Azure blob folders by home_id and generate CSV report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze folders for home_id 65
  python analyze_home_folders.py --home-ids 65 --output report_65.csv

  # Analyze multiple home IDs
  python analyze_home_folders.py --home-ids 65 68 70 --output report_multi.csv

  # From a file with home IDs (one per line)
  python analyze_home_folders.py --home-ids-file home_ids.txt --output report.csv

  # Use custom config file
  python analyze_home_folders.py --home-ids 65 --config /path/to/config.yaml
        """
    )

    # Accept either --home-ids or --home-ids-file
    home_id_group = parser.add_mutually_exclusive_group(required=True)
    home_id_group.add_argument(
        '--home-ids',
        nargs='+',
        help='Home IDs to search for (e.g., 65 68 70)'
    )
    home_id_group.add_argument(
        '--home-ids-file',
        help='File containing home IDs (one per line)'
    )

    parser.add_argument(
        '--container',
        default='instavideo',
        help='Azure blob container name (default: instavideo)'
    )

    parser.add_argument(
        '--prefix',
        default='one-data-platform/',
        help='Optional prefix to narrow search (e.g., "one-data-platform/")'
    )

    parser.add_argument(
        '--output',
        default='folder_analysis_report.csv',
        help='Output CSV file path (default: folder_analysis_report.csv)'
    )

    parser.add_argument(
        '--config',
        default='blobfuse2_config.yaml',
        help='Path to blobfuse2 config file (default: blobfuse2_config.yaml)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Get list of home IDs
        if args.home_ids:
            home_ids = args.home_ids
        else:
            # Read from file
            with open(args.home_ids_file, 'r') as f:
                home_ids = [line.strip() for line in f if line.strip()]

        logger.info(f"Processing {len(home_ids)} home ID(s): {', '.join(home_ids)}")

        # Get Azure connection
        connection_string = get_connection_string_from_yaml(args.config)
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Collect all folders for all home IDs
        all_stats = []
        total_folders_found = 0

        for home_id in home_ids:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing home_id: {home_id}")
            logger.info(f"{'='*60}")

            # Find folders matching home_id
            folders = find_home_folders(
                blob_service_client,
                args.container,
                home_id,
                args.prefix
            )

            if not folders:
                logger.warning(f"No folders found for home_id: {home_id}")
                continue

            total_folders_found += len(folders)

            # Analyze each folder
            for i, folder in enumerate(folders, 1):
                logger.info(f"Processing folder {i}/{len(folders)}: {folder}")
                try:
                    stats = analyze_folder(
                        blob_service_client,
                        args.container,
                        folder,
                        home_id
                    )
                    all_stats.append(stats)
                except Exception as e:
                    logger.error(f"Failed to analyze folder {folder}: {e}")
                    continue

        # Generate CSV report
        if all_stats:
            generate_csv_report(all_stats, args.output)
            logger.info(f"\n{'='*60}")
            logger.info(f"Analysis complete!")
            logger.info(f"  Home IDs processed: {len(home_ids)}")
            logger.info(f"  Total folders found: {total_folders_found}")
            logger.info(f"  Successfully analyzed: {len(all_stats)}")
            logger.info(f"  Report saved to: {args.output}")
            logger.info(f"{'='*60}")
        else:
            logger.error("No folders were successfully analyzed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
