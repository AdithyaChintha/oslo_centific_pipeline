#!/usr/bin/env python3
"""
SharePoint TAR to Azure Blob Transfer Script

This script automatically downloads TAR files from SharePoint and uploads them to Azure Blob Storage.

Key Features:
- Downloads TAR files from configured SharePoint folder
- Uploads downloaded TAR files to Azure Blob Storage
- Tracks processed files to avoid duplicates
- Uses shared SharePoint and Azure Blob utilities
- Provides detailed logging and progress tracking
- All paths configured via tar_transfer_config.yaml

Usage:
    python sharepoint_tar_to_azure_blob.py [--config CONFIG_FILE] [--dry-run]

Author: Auto-generated for TAR file transfer automation
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Import shared SharePoint utilities
try:
    from utils.sharepoint_utils import (
        SharePointConfig, SharePointAuthenticator, SharePointFileManager,
        load_sharepoint_config, create_sharepoint_manager
    )
    from utils.azure_blob_utils import download_blob, list_blobs
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.sharepoint_utils import (
        SharePointConfig, SharePointAuthenticator, SharePointFileManager,
        load_sharepoint_config, create_sharepoint_manager
    )
    from utils.azure_blob_utils import download_blob, list_blobs

# Import Azure Blob Storage SDK
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sharepoint_tar_to_blob.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TarTransferConfig:
    """Configuration class for TAR file transfer settings."""

    def __init__(self, config_dict: Dict):
        """Initialize transfer config from dictionary."""
        # SharePoint settings
        self.sharepoint_tar_folder_path = config_dict['sharepoint']['tar_source_folder_path']

        # Azure Blob settings
        self.azure_connection_string = self._get_connection_string(config_dict['azure_blob'])
        self.azure_container_name = config_dict['azure_blob']['container_name']
        self.azure_target_prefix = config_dict['azure_blob'].get('target_prefix', 'tar_files/')

        # Local paths
        self.temp_download_dir = config_dict['local_paths']['temp_download_dir']
        self.transfer_state_file = config_dict['local_paths']['transfer_state_file']

        # Processing settings
        self.max_files_per_run = config_dict.get('processing', {}).get('max_files_per_run', 50)
        self.cleanup_after_upload = config_dict.get('processing', {}).get('cleanup_after_upload', True)

    def _get_connection_string(self, azure_config: Dict) -> str:
        """Build Azure connection string from config or environment."""
        # Check if connection string is provided directly
        if 'connection_string' in azure_config:
            conn_str = azure_config['connection_string']
            # Support environment variable substitution
            if conn_str.startswith('${') and conn_str.endswith('}'):
                env_var = conn_str[2:-1]
                return os.environ.get(env_var, '')
            return conn_str

        # Build from account name and key
        account_name = azure_config.get('account_name', '')
        account_key = azure_config.get('account_key', '')

        # Support environment variable substitution
        if account_key.startswith('${') and account_key.endswith('}'):
            env_var = account_key[2:-1]
            account_key = os.environ.get(env_var, '')

        if account_name and account_key:
            return f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"

        raise ValueError("Azure Blob connection string or account credentials not configured properly")


class TarTransferManager:
    """Main manager class that orchestrates TAR file transfer from SharePoint to Azure Blob."""

    def __init__(self, sharepoint_config_path: str = "config/sharepoint_config.yaml",
                 transfer_config_path: str = "config/tar_transfer_config.yaml"):
        """
        Initialize transfer manager with configuration.

        Args:
            sharepoint_config_path: Path to SharePoint YAML configuration file
            transfer_config_path: Path to transfer-specific YAML configuration file
        """
        # Load SharePoint configuration
        self.sp_config = load_sharepoint_config(sharepoint_config_path)

        # Create SharePoint authenticator and file manager
        self.authenticator, self.file_manager = create_sharepoint_manager(self.sp_config)

        # Load transfer-specific configuration
        import yaml
        with open(transfer_config_path, 'r') as f:
            transfer_config_dict = yaml.safe_load(f)
        self.transfer_config = TarTransferConfig(transfer_config_dict)

        # Initialize Azure Blob client
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.transfer_config.azure_connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.transfer_config.azure_container_name
        )

        # Ensure local directories exist
        os.makedirs(self.transfer_config.temp_download_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.transfer_config.transfer_state_file), exist_ok=True)

        logger.info(f"TAR Transfer Manager initialized")
        logger.info(f"SharePoint folder: {self.transfer_config.sharepoint_tar_folder_path}")
        logger.info(f"Azure container: {self.transfer_config.azure_container_name}")

    def _load_transfer_state(self) -> Dict[str, Dict]:
        """
        Load transfer state from JSON file.

        Returns:
            Dictionary mapping filenames to transfer metadata
        """
        if os.path.exists(self.transfer_config.transfer_state_file):
            try:
                with open(self.transfer_config.transfer_state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded transfer state: {len(state)} files already processed")
                return state if isinstance(state, dict) else {}
            except Exception as e:
                logger.warning(f"Failed to load transfer state: {e}")
                return {}
        return {}

    def _save_transfer_state(self, state: Dict[str, Dict]):
        """
        Save transfer state to JSON file.

        Args:
            state: Dictionary mapping filenames to transfer metadata
        """
        try:
            with open(self.transfer_config.transfer_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug("Transfer state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save transfer state: {e}")

    def _get_sharepoint_tar_files(self) -> List[Dict]:
        """
        Get list of TAR files from SharePoint folder.

        Returns:
            List of file information dictionaries
        """
        logger.info(f"Fetching TAR files from SharePoint: {self.transfer_config.sharepoint_tar_folder_path}")

        # Get folder ID
        folder_id = self.file_manager.get_folder_id(self.transfer_config.sharepoint_tar_folder_path)
        if not folder_id:
            logger.error(f"Cannot find SharePoint folder: {self.transfer_config.sharepoint_tar_folder_path}")
            return []

        # List files with .tar extension
        tar_files = self.file_manager.list_files_in_folder(folder_id, file_extensions=['.tar', '.tar.gz', '.tgz'])

        logger.info(f"Found {len(tar_files)} TAR files in SharePoint")
        return tar_files

    def _download_from_sharepoint(self, file_info: Dict) -> Optional[str]:
        """
        Download a TAR file from SharePoint to local temporary directory.

        Args:
            file_info: File information dictionary from SharePoint

        Returns:
            Local file path if successful, None otherwise
        """
        filename = file_info['name']
        download_url = file_info.get('download_url')

        if not download_url:
            logger.error(f"No download URL for file: {filename}")
            return None

        local_path = os.path.join(self.transfer_config.temp_download_dir, filename)

        logger.info(f"Downloading from SharePoint: {filename}")
        success = self.file_manager.download_file(download_url, local_path)

        if success:
            logger.info(f"Downloaded to: {local_path}")
            return local_path
        else:
            logger.error(f"Failed to download: {filename}")
            return None

    def _upload_to_azure_blob(self, local_path: str, blob_name: str) -> bool:
        """
        Upload a file to Azure Blob Storage.

        Args:
            local_path: Local file path
            blob_name: Target blob name in Azure

        Returns:
            True if upload successful, False otherwise
        """
        try:
            logger.info(f"Uploading to Azure Blob: {blob_name}")

            # Add prefix if configured
            if self.transfer_config.azure_target_prefix:
                blob_name = os.path.join(self.transfer_config.azure_target_prefix, blob_name)

            # Get blob client
            blob_client = self.container_client.get_blob_client(blob_name)

            # Check if blob already exists
            if blob_client.exists():
                logger.warning(f"Blob already exists: {blob_name} (skipping)")
                return True

            # Upload file
            file_size = os.path.getsize(local_path)
            logger.info(f"Uploading {file_size:,} bytes...")

            with open(local_path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=False)

            logger.info(f"Successfully uploaded to Azure Blob: {blob_name}")
            return True

        except ResourceExistsError:
            logger.warning(f"Blob already exists: {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to Azure Blob {blob_name}: {e}")
            return False

    def _cleanup_local_file(self, local_path: str):
        """
        Clean up local temporary file.

        Args:
            local_path: Path to local file to delete
        """
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                logger.debug(f"Cleaned up local file: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup local file {local_path}: {e}")

    def run_transfer(self, dry_run: bool = False) -> bool:
        """
        Main method to execute the TAR file transfer process.

        Args:
            dry_run: If True, show what would be transferred without actually transferring

        Returns:
            True if successful, False otherwise
        """
        logger.info("============================================================")
        logger.info("Starting TAR File Transfer: SharePoint → Azure Blob")
        logger.info("============================================================")

        try:
            # Load transfer state
            transfer_state = self._load_transfer_state()

            # Get TAR files from SharePoint
            sharepoint_files = self._get_sharepoint_tar_files()

            if not sharepoint_files:
                logger.warning("No TAR files found in SharePoint folder")
                return True

            # Filter files that haven't been transferred yet
            files_to_transfer = []
            for file_info in sharepoint_files:
                filename = file_info['name']
                if filename not in transfer_state:
                    files_to_transfer.append(file_info)
                else:
                    logger.debug(f"Skipping already transferred file: {filename}")

            if not files_to_transfer:
                logger.info("No new files to transfer")
                return True

            # Limit number of files per run
            if len(files_to_transfer) > self.transfer_config.max_files_per_run:
                logger.info(f"Limiting to {self.transfer_config.max_files_per_run} files per run")
                files_to_transfer = files_to_transfer[:self.transfer_config.max_files_per_run]

            logger.info(f"Found {len(files_to_transfer)} files to transfer")

            if dry_run:
                logger.info("DRY RUN MODE - Files that would be transferred:")
                for file_info in files_to_transfer:
                    logger.info(f"  - {file_info['name']} ({file_info['size']:,} bytes)")
                return True

            # Process each file
            successful_transfers = []
            failed_transfers = []

            for i, file_info in enumerate(files_to_transfer, 1):
                filename = file_info['name']
                logger.info(f"\n[{i}/{len(files_to_transfer)}] Processing: {filename}")

                try:
                    # Step 1: Download from SharePoint
                    local_path = self._download_from_sharepoint(file_info)
                    if not local_path:
                        failed_transfers.append(filename)
                        continue

                    # Step 2: Upload to Azure Blob
                    upload_success = self._upload_to_azure_blob(local_path, filename)

                    if upload_success:
                        # Update transfer state
                        transfer_state[filename] = {
                            'transferred_at': datetime.now().isoformat(),
                            'size': file_info['size'],
                            'sharepoint_modified': file_info.get('modified'),
                            'blob_path': os.path.join(self.transfer_config.azure_target_prefix, filename)
                        }
                        successful_transfers.append(filename)

                        # Step 3: Cleanup local file if configured
                        if self.transfer_config.cleanup_after_upload:
                            self._cleanup_local_file(local_path)

                        logger.info(f"✅ Successfully transferred: {filename}")
                    else:
                        failed_transfers.append(filename)
                        logger.error(f"❌ Failed to transfer: {filename}")

                except Exception as e:
                    failed_transfers.append(filename)
                    logger.error(f"❌ Error processing {filename}: {e}")

                # Small delay between transfers
                time.sleep(1)

            # Save updated transfer state
            if successful_transfers:
                self._save_transfer_state(transfer_state)

            # Summary
            logger.info("\n============================================================")
            logger.info("Transfer Summary:")
            logger.info(f"✅ Successful transfers: {len(successful_transfers)}")
            logger.info(f"❌ Failed transfers: {len(failed_transfers)}")
            if failed_transfers:
                logger.info("Failed files:")
                for filename in failed_transfers:
                    logger.info(f"  - {filename}")
            logger.info("============================================================")

            return len(failed_transfers) == 0

        except Exception as e:
            logger.error(f"Transfer process error: {e}")
            return False


def main():
    """Main function to run the TAR file transfer."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Transfer TAR files from SharePoint to Azure Blob Storage'
    )
    parser.add_argument('--sp-config', default='config/sharepoint_config.yaml',
                       help='Path to SharePoint configuration file')
    parser.add_argument('--transfer-config', default='config/tar_transfer_config.yaml',
                       help='Path to transfer configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be transferred without actually transferring')

    args = parser.parse_args()

    try:
        # Create and run transfer manager
        transfer_manager = TarTransferManager(
            sharepoint_config_path=args.sp_config,
            transfer_config_path=args.transfer_config
        )
        success = transfer_manager.run_transfer(dry_run=args.dry_run)

        if success:
            logger.info("TAR transfer completed successfully!")
            sys.exit(0)
        else:
            logger.error("TAR transfer completed with errors!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
