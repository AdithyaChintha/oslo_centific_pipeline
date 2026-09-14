#!/usr/bin/env python3
"""
Fix Azure Blob Content Headers from CSV Report

This script reads a CSV report from the TAR inspector and updates blob content headers
so that files can be viewed/played in the browser instead of being downloaded.

It sets:
- content_type: Proper MIME type (video/mp4, image/jpeg, text/plain, etc.)
- content_disposition: 'inline' for viewable files, 'attachment' for others

Usage:
    python fix_blob_headers_from_csv.py --csv REPORT.csv --connection-string CONN_STR [--dry-run]

Examples:
    # Using connection string directly
    python fix_blob_headers_from_csv.py \
        --csv /data/oslo/untar_inspection_reports_/tar_inspection_report_20251118_135049.csv \
        --connection-string "DefaultEndpointsProtocol=https;AccountName=oslotestvideo;AccountKey=...;EndpointSuffix=core.windows.net"

    # Dry run (show what would be changed)
    python fix_blob_headers_from_csv.py --csv report.csv --connection-string "..." --dry-run

    # Using environment variable
    export BLOB_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
    python fix_blob_headers_from_csv.py --csv report.csv
"""

import os
import sys
import csv
import mimetypes
import logging
from typing import Dict, Tuple
from azure.storage.blob import BlobServiceClient, ContentSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fix_blob_headers_csv.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BlobHeaderFixer:
    """Fix blob content headers for browser viewing."""

    def __init__(self, connection_string: str, container_name: str = "instavideo"):
        """
        Initialize the blob header fixer.

        Args:
            connection_string: Azure Blob Storage connection string
            container_name: Container name (default: instavideo)
        """
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.container_name = container_name
        logger.info(f"Initialized BlobHeaderFixer for container: {container_name}")

    def _get_content_type_and_disposition(self, filename: str) -> Tuple[str, str]:
        """
        Get appropriate content-type and content-disposition for browser playback.

        Args:
            filename: Name of the file

        Returns:
            Tuple of (content_type, content_disposition)
        """
        # Explicit extension-to-MIME type mapping for common video/media files
        extension_map = {
            # Video formats
            '.mp4': 'video/mp4',
            '.m4v': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.insv': 'video/mp4',  # Insta360 video format
            '.mpg': 'video/mpeg',
            '.mpeg': 'video/mpeg',
            '.3gp': 'video/3gpp',
            '.ogv': 'video/ogg',

            # Audio formats
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
            '.aac': 'audio/aac',
            '.flac': 'audio/flac',
            '.wma': 'audio/x-ms-wma',

            # Image formats
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.svg': 'image/svg+xml',
            '.webp': 'image/webp',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.ico': 'image/x-icon',

            # Document formats
            '.pdf': 'application/pdf',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.md': 'text/markdown',
        }

        # Get file extension (lowercase)
        ext = os.path.splitext(filename)[1].lower()

        # Try explicit mapping first
        if ext in extension_map:
            content_type = extension_map[ext]
        else:
            # Fallback to mimetypes module
            content_type, _ = mimetypes.guess_type(filename)
            if content_type is None:
                content_type = 'application/octet-stream'

        # Files that should be viewable/playable in browser
        viewable_types = [
            'video/',      # All video types
            'audio/',      # All audio types
            'image/',      # All image types
            'text/',       # Text files
            'application/json',
            'application/pdf',
            'application/javascript',
            'application/xml'
        ]

        # Check if content type should be inline (viewable in browser)
        should_be_inline = any(content_type.startswith(vt) for vt in viewable_types)

        if should_be_inline:
            # inline = viewable/playable in browser
            content_disposition = 'inline'
        else:
            # attachment = download
            content_disposition = 'attachment'

        return content_type, content_disposition

    def update_blob_headers(self, blob_path: str, dry_run: bool = False) -> bool:
        """
        Update content headers for a single blob.

        Args:
            blob_path: Path to blob in container
            dry_run: If True, only show what would be changed

        Returns:
            True if successful (or would be successful in dry run), False otherwise
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_path)

            # Check if blob exists
            if not blob_client.exists():
                logger.error(f"Blob does not exist: {blob_path}")
                return False

            # Get current properties
            properties = blob_client.get_blob_properties()
            current_content_type = properties.content_settings.content_type
            current_disposition = properties.content_settings.content_disposition

            # Calculate new headers
            filename = os.path.basename(blob_path)
            new_content_type, new_disposition = self._get_content_type_and_disposition(filename)

            # Check if update needed
            needs_update = (
                current_content_type != new_content_type or
                current_disposition != new_disposition
            )

            if not needs_update:
                logger.debug(f"No update needed: {blob_path}")
                return True

            # Log the change
            logger.info(f"Blob: {blob_path}")
            logger.info(f"  Current: type={current_content_type}, disposition={current_disposition}")
            logger.info(f"  New:     type={new_content_type}, disposition={new_disposition}")

            if dry_run:
                logger.info(f"  [DRY RUN] Would update blob headers")
                return True

            # Update blob headers
            content_settings = ContentSettings(
                content_type=new_content_type,
                content_disposition=new_disposition
            )

            blob_client.set_http_headers(content_settings=content_settings)
            logger.info(f"  ✅ Updated blob headers")

            return True

        except Exception as e:
            logger.error(f"Failed to update blob {blob_path}: {e}")
            return False

    def process_csv_report(self, csv_path: str, dry_run: bool = False) -> Dict[str, int]:
        """
        Process CSV report and update all blob headers.

        Args:
            csv_path: Path to CSV report file
            dry_run: If True, only show what would be changed

        Returns:
            Dictionary with success/failure counts
        """
        logger.info("="*100)
        logger.info("FIXING BLOB HEADERS FROM CSV REPORT")
        logger.info("="*100)
        logger.info(f"CSV file: {csv_path}")
        logger.info(f"Container: {self.container_name}")
        logger.info(f"Dry run: {dry_run}")
        logger.info("="*100)

        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}

        stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    stats['total'] += 1
                    blob_path = row.get('blob_path', '').strip()

                    if not blob_path:
                        logger.warning(f"Row {stats['total']}: Missing blob_path, skipping")
                        stats['skipped'] += 1
                        continue

                    logger.info(f"\n[{stats['total']}] Processing: {blob_path}")

                    success = self.update_blob_headers(blob_path, dry_run)

                    if success:
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1

            # Summary
            logger.info("\n" + "="*100)
            logger.info("SUMMARY")
            logger.info("="*100)
            logger.info(f"Total blobs in CSV: {stats['total']}")
            logger.info(f"✅ Successfully updated: {stats['success']}")
            logger.info(f"❌ Failed: {stats['failed']}")
            logger.info(f"⏭️  Skipped: {stats['skipped']}")
            logger.info("="*100)

            return stats

        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return stats


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix Azure Blob content headers for browser viewing from CSV report',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using connection string directly
  python fix_blob_headers_from_csv.py \\
      --csv /data/oslo/reports/tar_inspection_report_20251118_135049.csv \\
      --connection-string "DefaultEndpointsProtocol=https;AccountName=oslotestvideo;AccountKey=...;EndpointSuffix=core.windows.net"

  # Using environment variable for connection string
  export BLOB_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
  python fix_blob_headers_from_csv.py --csv /data/oslo/reports/report.csv

  # Dry run to preview changes
  python fix_blob_headers_from_csv.py --csv report.csv --dry-run
        """
    )

    parser.add_argument(
        '--csv',
        required=True,
        help='Path to CSV report file containing blob_path column'
    )

    parser.add_argument(
        '--connection-string',
        help='Azure Blob Storage connection string (or use BLOB_CONNECTION_STRING env var)'
    )

    parser.add_argument(
        '--container',
        default='instavideo',
        help='Azure Blob container name (default: instavideo)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without actually updating'
    )

    args = parser.parse_args()

    # Get connection string
    connection_string = args.connection_string

    # Try environment variable if not provided
    if not connection_string:
        connection_string = os.environ.get('BLOB_CONNECTION_STRING')

    if not connection_string:
        logger.error("="*100)
        logger.error("ERROR: No connection string provided!")
        logger.error("="*100)
        logger.error("Either:")
        logger.error("  1. Use --connection-string argument, OR")
        logger.error("  2. Set BLOB_CONNECTION_STRING environment variable")
        logger.error("")
        logger.error("Example:")
        logger.error('  export BLOB_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"')
        logger.error("="*100)
        sys.exit(1)

    if args.dry_run:
        logger.info("="*100)
        logger.info("DRY RUN MODE - No actual changes will be made")
        logger.info("="*100)

    try:
        # Create fixer and process CSV
        fixer = BlobHeaderFixer(connection_string, args.container)
        stats = fixer.process_csv_report(args.csv, args.dry_run)

        # Exit with appropriate code
        if stats['failed'] > 0:
            logger.error("\n❌ Completed with errors!")
            sys.exit(1)
        else:
            logger.info("\n✅ Completed successfully!")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
