#!/usr/bin/env python3
"""
Fix Blob Content Headers Script

This script updates existing blobs in Azure to have proper content-type and
content-disposition headers so they can be viewed/played in browser instead of downloading.

Usage:
    # Fix all blobs in a specific folder
    python fix_blob_content_headers.py --prefix "untar_folder_oslo2/" --config config/unified_tar_config.yaml

    # Dry run to see what would be changed
    python fix_blob_content_headers.py --prefix "untar_folder_oslo2/" --dry-run

    # Fix specific file types only
    python fix_blob_content_headers.py --prefix "untar_folder_oslo2/" --extensions .mp4 .json .mp3
"""

import os
import sys
import mimetypes
import logging
from typing import List, Optional
from azure.storage.blob import BlobServiceClient, ContentSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fix_blob_headers.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_content_type_and_disposition(filename: str) -> tuple:
    """
    Get appropriate content-type and content-disposition for browser playback.

    Args:
        filename: Name of the file

    Returns:
        Tuple of (content_type, content_disposition)
    """
    # Get MIME type
    content_type, _ = mimetypes.guess_type(filename)

    if content_type is None:
        content_type = 'application/octet-stream'

    # Files that should be viewable/playable in browser
    viewable_types = [
        'video/',      # All video types (mp4, webm, etc.)
        'audio/',      # All audio types (mp3, wav, etc.)
        'image/',      # All image types (jpg, png, etc.)
        'text/',       # Text files
        'application/json',
        'application/pdf',
        'application/javascript',
        'application/xml'
    ]

    # Check if content type should be inline (viewable in browser)
    should_be_inline = any(content_type.startswith(vt) for vt in viewable_types)

    if should_be_inline:
        content_disposition = 'inline'
    else:
        content_disposition = 'attachment'

    return content_type, content_disposition


def fix_blob_headers(connection_string: str, container_name: str, blob_prefix: str = "",
                    extensions: Optional[List[str]] = None, dry_run: bool = False) -> dict:
    """
    Fix content headers for blobs in Azure Storage.

    Args:
        connection_string: Azure Storage connection string
        container_name: Container name
        blob_prefix: Blob prefix/folder to process
        extensions: Optional list of file extensions to filter (e.g., ['.mp4', '.json'])
        dry_run: If True, show what would be changed without making changes

    Returns:
        Dictionary with statistics
    """
    logger.info("="*100)
    logger.info("FIXING BLOB CONTENT HEADERS FOR BROWSER PLAYBACK")
    logger.info("="*100)
    logger.info(f"Container: {container_name}")
    logger.info(f"Prefix: {blob_prefix or '(root)'}")
    logger.info(f"Extensions filter: {extensions or 'All files'}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("="*100)

    try:
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Statistics
        stats = {
            'total_blobs': 0,
            'processed': 0,
            'updated': 0,
            'skipped': 0,
            'errors': 0
        }

        # List all blobs with the prefix
        logger.info(f"\nListing blobs with prefix: {blob_prefix}")
        blobs = container_client.list_blobs(name_starts_with=blob_prefix)

        for blob in blobs:
            stats['total_blobs'] += 1
            blob_name = blob.name

            # Filter by extension if specified
            if extensions:
                if not any(blob_name.lower().endswith(ext.lower()) for ext in extensions):
                    stats['skipped'] += 1
                    continue

            # Get filename from blob path
            filename = os.path.basename(blob_name)

            # Get current properties
            blob_client = container_client.get_blob_client(blob_name)
            properties = blob_client.get_blob_properties()

            current_content_type = properties.content_settings.content_type
            current_content_disposition = properties.content_settings.content_disposition

            # Determine correct content type and disposition
            correct_content_type, correct_content_disposition = get_content_type_and_disposition(filename)

            # Check if update is needed
            needs_update = (
                current_content_type != correct_content_type or
                current_content_disposition != correct_content_disposition
            )

            if needs_update:
                stats['processed'] += 1

                logger.info(f"\n[{stats['processed']}] Blob: {blob_name}")
                logger.info(f"  Current:  content-type={current_content_type}, disposition={current_content_disposition}")
                logger.info(f"  Correct:  content-type={correct_content_type}, disposition={correct_content_disposition}")

                if not dry_run:
                    try:
                        # Update content settings
                        content_settings = ContentSettings(
                            content_type=correct_content_type,
                            content_disposition=correct_content_disposition
                        )

                        blob_client.set_http_headers(content_settings=content_settings)

                        stats['updated'] += 1
                        logger.info(f"  ✅ UPDATED")
                    except Exception as e:
                        stats['errors'] += 1
                        logger.error(f"  ❌ ERROR: {e}")
                else:
                    logger.info(f"  ⏭️  WOULD UPDATE (dry-run mode)")
            else:
                stats['skipped'] += 1
                logger.debug(f"Skipping {blob_name} (already correct)")

            # Progress every 100 blobs
            if stats['total_blobs'] % 100 == 0:
                logger.info(f"\nProgress: {stats['total_blobs']} blobs scanned, {stats['processed']} need updates...")

        # Final summary
        logger.info("\n" + "="*100)
        logger.info("SUMMARY")
        logger.info("="*100)
        logger.info(f"Total blobs scanned: {stats['total_blobs']}")
        logger.info(f"Blobs needing updates: {stats['processed']}")
        if not dry_run:
            logger.info(f"✅ Successfully updated: {stats['updated']}")
            logger.info(f"❌ Errors: {stats['errors']}")
        else:
            logger.info(f"⏭️  Would update: {stats['processed']} blobs (dry-run mode)")
        logger.info(f"Skipped (already correct or filtered): {stats['skipped']}")
        logger.info("="*100)

        return stats

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {'error': str(e)}


def main():
    """Main function."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description='Fix Azure Blob content headers for browser playback'
    )
    parser.add_argument('--config', '-c', default='config/unified_tar_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--prefix', '-p', default='',
                       help='Blob prefix/folder to process (e.g., "untar_folder_oslo2/")')
    parser.add_argument('--extensions', '-e', nargs='+',
                       help='File extensions to process (e.g., .mp4 .json .mp3)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without making changes')

    args = parser.parse_args()

    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        connection_string = config['azure_blob']['connection_string']
        container_name = config['azure_blob']['container_name']

        # Run the fix
        stats = fix_blob_headers(
            connection_string=connection_string,
            container_name=container_name,
            blob_prefix=args.prefix,
            extensions=args.extensions,
            dry_run=args.dry_run
        )

        if 'error' not in stats:
            logger.info("\n✅ Blob header fix completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Blob header fix completed with errors!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
