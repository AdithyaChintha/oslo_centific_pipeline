#!/usr/bin/env python3
"""
Unified SharePoint TAR Processing Script
Combines transfer and inspection workflows in a single script.

Workflow:
STAGE 1: Transfer all TAR files from SharePoint to Azure Blob (intact)
  - Downloads/transfers ALL TAR files from SharePoint /Uploads folder
  - Uploads to Azure Blob: instavideo/oslo_stage_2_tar_files/
  - Uses optimized transfer methods (server-to-server, streaming, download+upload)
  - Tracks transferred files to avoid duplicates

STAGE 2: Random inspection with extraction and detailed reporting
  - Randomly selects 1 TAR per every 10 files from SharePoint
  - Downloads selected TAR files
  - Extracts all contents from selected TARs
  - Uploads extracted files to Azure Blob: instavideo/untar_folder_oslo2/
  - Generates SAS tokens for each extracted file
  - Creates comprehensive CSV report with metadata

Usage:
    python sharepoint_unified_tar_processor.py [--config CONFIG_FILE] [--stage {1,2,both}] [--dry-run]

Examples:
    # Run both stages (default)
    python sharepoint_unified_tar_processor.py

    # Run only stage 1 (transfer)
    python sharepoint_unified_tar_processor.py --stage 1

    # Run only stage 2 (inspection)
    python sharepoint_unified_tar_processor.py --stage 2

    # Dry run to see what would be processed
    python sharepoint_unified_tar_processor.py --dry-run

Author: Oslo TAR Processing Unified Workflow
"""

import os
import sys
import json
import csv
import time
import tarfile
import random
import shutil
import requests
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Import shared SharePoint utilities
try:
    from utils.sharepoint_utils import (
        SharePointAuthenticator, SharePointFileManager
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.sharepoint_utils import (
        SharePointAuthenticator, SharePointFileManager
    )

# Import Azure Blob Storage SDK
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
from azure.core.exceptions import ResourceExistsError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sharepoint_unified_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UnifiedProcessorConfig:
    """Configuration class for unified TAR processing."""

    def __init__(self, config_dict: Dict):
        """Initialize unified config from dictionary."""
        # SharePoint settings
        self.sharepoint_tar_folder_path = config_dict['sharepoint']['tar_source_folder_path']

        # Azure Blob settings
        self.azure_connection_string = self._get_connection_string(config_dict['azure_blob'])
        self.azure_container_name = config_dict['azure_blob']['container_name']
        self.azure_account_name = self._extract_account_name(self.azure_connection_string)
        self.azure_account_key = self._extract_account_key(self.azure_connection_string)

        # Stage 1: Transfer settings
        transfer_config = config_dict.get('transfer', {})
        self.transfer_target_prefix = transfer_config.get('target_prefix', 'oslo_stage_2_tar_files/')
        self.transfer_max_files_per_run = transfer_config.get('max_files_per_run', 50)
        self.transfer_cleanup_after_upload = transfer_config.get('cleanup_after_upload', True)

        # Stage 2: Inspection settings
        inspection_config = config_dict.get('inspection', {})
        self.inspection_untar_blob_prefix = inspection_config.get('untar_blob_prefix', 'untar_folder_oslo2/')
        self.inspection_sampling_ratio = inspection_config.get('sampling_ratio', 10)
        self.inspection_sas_token_expiry_days = inspection_config.get('sas_token_expiry_days', 30)

        # Local paths
        local_paths = config_dict.get('local_paths', {})
        self.temp_download_dir = local_paths.get('temp_download_dir', '/data/oslo/tar_temp_downloads')
        self.temp_extract_dir = local_paths.get('temp_extract_dir', '/data/oslo/tar_extractions')
        self.csv_report_dir = local_paths.get('csv_report_dir', '/data/oslo/unified_reports')
        self.transfer_state_file = local_paths.get('transfer_state_file',
                                                    '/data/oslo/tar_temp_downloads/tar_transfer_state.json')
        self.inspection_state_file = local_paths.get('inspection_state_file',
                                                      '/data/oslo/tar_temp_downloads/tar_inspection_state.json')

        # API settings
        api_config = config_dict.get('api', {})
        self.timeout_seconds = api_config.get('timeout_seconds', 300)
        self.retry_attempts = api_config.get('retry_attempts', 3)
        self.retry_delay_seconds = api_config.get('retry_delay_seconds', 5)

    def _get_connection_string(self, azure_config: Dict) -> str:
        """Build Azure connection string from config or environment."""
        if 'connection_string' in azure_config:
            conn_str = azure_config['connection_string']
            if conn_str.startswith('${') and conn_str.endswith('}'):
                env_var = conn_str[2:-1]
                return os.environ.get(env_var, '')
            return conn_str

        account_name = azure_config.get('account_name', '')
        account_key = azure_config.get('account_key', '')

        if account_key.startswith('${') and account_key.endswith('}'):
            env_var = account_key[2:-1]
            account_key = os.environ.get(env_var, '')

        if account_name and account_key:
            return f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"

        raise ValueError("Azure Blob connection string or account credentials not configured properly")

    def _extract_account_name(self, connection_string: str) -> str:
        """Extract account name from connection string."""
        for part in connection_string.split(';'):
            if part.startswith('AccountName='):
                return part.split('=', 1)[1]
        raise ValueError("Cannot extract AccountName from connection string")

    def _extract_account_key(self, connection_string: str) -> str:
        """Extract account key from connection string."""
        for part in connection_string.split(';'):
            if part.startswith('AccountKey='):
                return part.split('=', 1)[1]
        raise ValueError("Cannot extract AccountKey from connection string")


class UnifiedTarProcessor:
    """Unified processor for TAR transfer and inspection workflows."""

    def __init__(self, config_path: str = "config/unified_tar_config.yaml"):
        """
        Initialize unified processor with configuration.

        Args:
            config_path: Path to unified configuration YAML file
        """
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = UnifiedProcessorConfig(config_dict)

        # Extract SharePoint connection details
        azure_ad = config_dict['azure_ad']
        sharepoint_config = config_dict['sharepoint']

        # Create SharePoint authenticator
        self.authenticator = SharePointAuthenticator(
            tenant_id=azure_ad['tenant_id'],
            client_id=azure_ad['client_id'],
            client_secret=azure_ad['client_secret'],
            timeout_seconds=self.config.timeout_seconds
        )

        # Create SharePoint file manager
        self.file_manager = SharePointFileManager(
            authenticator=self.authenticator,
            site_id=sharepoint_config['site_id'],
            drive_id=sharepoint_config['drive_id'],
            timeout=self.config.timeout_seconds,
            max_retries=self.config.retry_attempts,
            retry_delay=self.config.retry_delay_seconds
        )

        # Initialize Azure Blob client
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.config.azure_connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.config.azure_container_name
        )

        # Ensure local directories exist
        os.makedirs(self.config.temp_download_dir, exist_ok=True)
        os.makedirs(self.config.temp_extract_dir, exist_ok=True)
        os.makedirs(self.config.csv_report_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.transfer_state_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.inspection_state_file), exist_ok=True)

        logger.info(f"Unified TAR Processor initialized")
        logger.info(f"SharePoint folder: {self.config.sharepoint_tar_folder_path}")
        logger.info(f"Azure container: {self.config.azure_container_name}")

    # ==================== COMMON UTILITIES ====================

    def _load_state(self, state_file: str) -> Dict[str, Dict]:
        """Load state from JSON file."""
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded state from {state_file}: {len(state)} files tracked")
                return state if isinstance(state, dict) else {}
            except Exception as e:
                logger.warning(f"Failed to load state from {state_file}: {e}")
                return {}
        return {}

    def _save_state(self, state: Dict[str, Dict], state_file: str):
        """Save state to JSON file."""
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"State saved to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save state to {state_file}: {e}")

    def _get_sharepoint_tar_files(self) -> List[Dict]:
        """Get list of TAR files from SharePoint folder."""
        logger.info(f"Fetching TAR files from SharePoint: {self.config.sharepoint_tar_folder_path}")

        folder_id = self.file_manager.get_folder_id(self.config.sharepoint_tar_folder_path)
        if not folder_id:
            logger.error(f"Cannot find SharePoint folder: {self.config.sharepoint_tar_folder_path}")
            return []

        tar_files = self.file_manager.list_files_in_folder(folder_id, file_extensions=['.tar', '.tar.gz', '.tgz'])
        logger.info(f"Found {len(tar_files)} TAR files in SharePoint")
        return tar_files

    def _download_from_sharepoint(self, file_info: Dict) -> Optional[str]:
        """Download a TAR file from SharePoint to local temp directory."""
        filename = file_info['name']
        download_url = file_info.get('download_url')

        if not download_url:
            logger.error(f"No download URL for file: {filename}")
            return None

        local_path = os.path.join(self.config.temp_download_dir, filename)

        logger.info(f"Downloading from SharePoint: {filename}")
        success = self.file_manager.download_file(download_url, local_path)

        if success:
            logger.info(f"Downloaded to: {local_path}")
            return local_path
        else:
            logger.error(f"Failed to download: {filename}")
            return None

    def _cleanup_local_file(self, local_path: str):
        """Clean up local temporary file."""
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                logger.debug(f"Cleaned up local file: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup local file {local_path}: {e}")

    def _get_content_type_and_disposition(self, filename: str) -> tuple:
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
            # inline = viewable/playable in browser
            content_disposition = 'inline'
            logger.debug(f"File {filename} will be viewable in browser (content-type: {content_type})")
        else:
            # attachment = download
            content_disposition = 'attachment'
            logger.debug(f"File {filename} will be downloaded (content-type: {content_type})")

        return content_type, content_disposition

    # ==================== STAGE 1: TAR TRANSFER ====================

    def _copy_url_to_azure_blob(self, download_url: str, blob_name: str, file_size: int = 0,
                                prefix: str = None) -> bool:
        """
        Copy file directly from SharePoint URL to Azure Blob Storage (server-to-server).

        Args:
            download_url: SharePoint direct download URL
            blob_name: Target blob name in Azure
            file_size: File size for logging
            prefix: Blob path prefix

        Returns:
            True if copy successful, False otherwise
        """
        try:
            # Add prefix if provided
            if prefix:
                full_blob_name = os.path.join(prefix, blob_name).replace('\\', '/')
            else:
                full_blob_name = blob_name

            blob_client = self.container_client.get_blob_client(full_blob_name)

            if blob_client.exists():
                logger.warning(f"Blob already exists: {full_blob_name} (skipping)")
                return True

            logger.info(f"Starting server-to-server copy: {blob_name} ({file_size:,} bytes)")

            copy_props = blob_client.start_copy_from_url(download_url)
            copy_id = copy_props['copy_id']
            copy_status = copy_props['copy_status']

            logger.info(f"Copy initiated (ID: {copy_id}), status: {copy_status}")

            # Poll for completion
            max_wait_time = 3600  # 1 hour
            poll_interval = 5
            elapsed_time = 0

            while copy_status == 'pending':
                if elapsed_time >= max_wait_time:
                    logger.error(f"Copy timeout after {max_wait_time}s for {blob_name}")
                    return False

                time.sleep(poll_interval)
                elapsed_time += poll_interval

                props = blob_client.get_blob_properties()
                copy_status = props.copy.status

                if elapsed_time % 30 == 0:
                    logger.info(f"Copy in progress: {blob_name} (elapsed: {elapsed_time}s)")

            if copy_status == 'success':
                logger.info(f"✅ Server-to-server copy completed: {blob_name}")
                return True
            else:
                logger.error(f"Copy failed with status: {copy_status}")
                return False

        except Exception as e:
            logger.warning(f"Server-to-server copy failed for {blob_name}: {e}")
            return False

    def _stream_transfer_to_azure_blob(self, download_url: str, blob_name: str, file_size: int = 0,
                                       prefix: str = None) -> bool:
        """
        Stream file directly from SharePoint to Azure Blob using chunking with proper content headers.

        Args:
            download_url: SharePoint direct download URL
            blob_name: Target blob name in Azure
            file_size: File size for progress tracking
            prefix: Blob path prefix

        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Add prefix if provided
            if prefix:
                full_blob_name = os.path.join(prefix, blob_name).replace('\\', '/')
            else:
                full_blob_name = blob_name

            blob_client = self.container_client.get_blob_client(full_blob_name)

            if blob_client.exists():
                logger.warning(f"Blob already exists: {full_blob_name} (skipping)")
                return True

            logger.info(f"Starting streaming transfer: {blob_name} ({file_size:,} bytes)")

            # Get appropriate content type and disposition for browser playback
            content_type, content_disposition = self._get_content_type_and_disposition(blob_name)

            CHUNK_SIZE = 64 * 1024 * 1024  # 64MB chunks
            response = requests.get(download_url, stream=True, timeout=300)
            response.raise_for_status()

            bytes_transferred = 0
            last_progress_log = 0

            def chunk_generator():
                nonlocal bytes_transferred, last_progress_log
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        bytes_transferred += len(chunk)

                        if file_size > 0:
                            progress = (bytes_transferred / file_size) * 100
                            if progress - last_progress_log >= 10:
                                logger.info(f"Transfer progress: {blob_name} - {progress:.1f}% ({bytes_transferred:,}/{file_size:,} bytes)")
                                last_progress_log = progress

                        yield chunk

            # Create ContentSettings object
            content_settings = ContentSettings(
                content_type=content_type,
                content_disposition=content_disposition
            )

            blob_client.upload_blob(
                chunk_generator(),
                overwrite=False,
                max_concurrency=4,
                content_settings=content_settings
            )

            logger.info(f"✅ Streaming transfer completed: {blob_name} ({bytes_transferred:,} bytes, content-type: {content_type}, disposition: {content_disposition})")
            return True

        except Exception as e:
            logger.error(f"Streaming transfer failed for {blob_name}: {e}")
            return False

    def _upload_to_azure_blob(self, local_path: str, blob_name: str, prefix: str = None) -> bool:
        """
        Upload a file to Azure Blob Storage from local path with proper content headers.

        Args:
            local_path: Local file path
            blob_name: Target blob name in Azure
            prefix: Blob path prefix

        Returns:
            True if upload successful, False otherwise
        """
        try:
            logger.info(f"Uploading from local file to Azure Blob: {blob_name}")

            # Add prefix if provided
            if prefix:
                full_blob_name = os.path.join(prefix, blob_name).replace('\\', '/')
            else:
                full_blob_name = blob_name

            blob_client = self.container_client.get_blob_client(full_blob_name)

            if blob_client.exists():
                logger.warning(f"Blob already exists: {full_blob_name} (skipping)")
                return True

            file_size = os.path.getsize(local_path)
            logger.info(f"Uploading {file_size:,} bytes from local disk...")

            # Get appropriate content type and disposition for browser playback
            filename = os.path.basename(local_path)
            content_type, content_disposition = self._get_content_type_and_disposition(filename)

            # Create ContentSettings object
            content_settings = ContentSettings(
                content_type=content_type,
                content_disposition=content_disposition
            )

            with open(local_path, 'rb') as data:
                blob_client.upload_blob(
                    data,
                    overwrite=False,
                    max_concurrency=4,
                    content_settings=content_settings
                )

            logger.info(f"Successfully uploaded to Azure Blob: {blob_name} (content-type: {content_type}, disposition: {content_disposition})")
            return True

        except ResourceExistsError:
            logger.warning(f"Blob already exists: {blob_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to Azure Blob {blob_name}: {e}")
            return False

    def run_stage1_transfer(self, dry_run: bool = False) -> bool:
        """
        STAGE 1: Transfer all TAR files from SharePoint to Azure Blob (intact).

        Args:
            dry_run: If True, show what would be transferred without processing

        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*100)
        logger.info("STAGE 1: TAR FILE TRANSFER (SharePoint → Azure Blob)")
        logger.info("="*100)

        try:
            # Load transfer state
            transfer_state = self._load_state(self.config.transfer_state_file)

            # Get all TAR files from SharePoint
            all_tar_files = self._get_sharepoint_tar_files()

            if not all_tar_files:
                logger.warning("No TAR files found in SharePoint folder")
                return True

            # Filter files not yet transferred
            files_to_transfer = []
            for file_info in all_tar_files:
                filename = file_info['name']
                if filename not in transfer_state:
                    files_to_transfer.append(file_info)
                else:
                    logger.debug(f"Skipping already transferred file: {filename}")

            if not files_to_transfer:
                logger.info("No new files to transfer")
                return True

            # Limit files per run
            if len(files_to_transfer) > self.config.transfer_max_files_per_run:
                logger.info(f"Limiting to {self.config.transfer_max_files_per_run} files per run")
                files_to_transfer = files_to_transfer[:self.config.transfer_max_files_per_run]

            logger.info(f"\nFound {len(files_to_transfer)} files to transfer:")
            for i, file_info in enumerate(files_to_transfer, 1):
                size_gb = file_info.get('size', 0) / (1024 ** 3)
                logger.info(f"  {i}. {file_info['name']} ({size_gb:.2f} GB)")

            if dry_run:
                logger.info("\nDRY RUN MODE - No actual transfer")
                return True

            # Process each file
            successful_transfers = []
            failed_transfers = []

            for i, file_info in enumerate(files_to_transfer, 1):
                filename = file_info['name']
                file_size = file_info.get('size', 0)
                download_url = file_info.get('download_url')

                logger.info(f"\n[{i}/{len(files_to_transfer)}] Processing: {filename} ({file_size:,} bytes)")

                if not download_url:
                    logger.error(f"No download URL for file: {filename}")
                    failed_transfers.append(filename)
                    continue

                transfer_success = False
                transfer_method = "unknown"

                try:
                    # Method 1: Server-to-server copy (fastest)
                    logger.info(f"[Method 1/3] Attempting server-to-server copy...")
                    transfer_success = self._copy_url_to_azure_blob(
                        download_url, filename, file_size, self.config.transfer_target_prefix
                    )

                    if transfer_success:
                        transfer_method = "server-to-server"
                        logger.info(f"✅ Server-to-server copy succeeded: {filename}")
                    else:
                        # Method 2: Streaming transfer
                        logger.info(f"[Method 2/3] Trying streaming transfer...")
                        transfer_success = self._stream_transfer_to_azure_blob(
                            download_url, filename, file_size, self.config.transfer_target_prefix
                        )

                        if transfer_success:
                            transfer_method = "streaming"
                            logger.info(f"✅ Streaming transfer succeeded: {filename}")
                        else:
                            # Method 3: Download + upload
                            logger.info(f"[Method 3/3] Falling back to download+upload...")
                            local_path = self._download_from_sharepoint(file_info)

                            if local_path:
                                transfer_success = self._upload_to_azure_blob(
                                    local_path, filename, self.config.transfer_target_prefix
                                )

                                if transfer_success:
                                    transfer_method = "download+upload"
                                    logger.info(f"✅ Download+upload succeeded: {filename}")

                                if self.config.transfer_cleanup_after_upload:
                                    self._cleanup_local_file(local_path)

                    # Record success
                    if transfer_success:
                        transfer_state[filename] = {
                            'transferred_at': datetime.now().isoformat(),
                            'size': file_size,
                            'blob_path': os.path.join(self.config.transfer_target_prefix, filename),
                            'transfer_method': transfer_method
                        }
                        successful_transfers.append(filename)
                    else:
                        failed_transfers.append(filename)
                        logger.error(f"❌ All transfer methods failed for: {filename}")

                except Exception as e:
                    failed_transfers.append(filename)
                    logger.error(f"❌ Error processing {filename}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

                time.sleep(1)

            # Save transfer state
            if successful_transfers:
                self._save_state(transfer_state, self.config.transfer_state_file)

            # Summary
            logger.info("\n" + "="*100)
            logger.info("STAGE 1 SUMMARY:")
            logger.info(f"✅ Successful transfers: {len(successful_transfers)}")
            logger.info(f"❌ Failed transfers: {len(failed_transfers)}")
            if failed_transfers:
                logger.info("Failed files:")
                for filename in failed_transfers:
                    logger.info(f"  - {filename}")
            logger.info("="*100)

            return len(failed_transfers) == 0

        except Exception as e:
            logger.error(f"Stage 1 transfer error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    # ==================== STAGE 2: TAR INSPECTION ====================

    def _select_random_samples(self, all_files: List[Dict], already_inspected: set) -> List[Dict]:
        """
        Select random samples: 1 per every N files.

        Args:
            all_files: List of all TAR files
            already_inspected: Set of already inspected filenames

        Returns:
            List of randomly selected files
        """
        uninspected_files = [f for f in all_files if f['name'] not in already_inspected]

        if not uninspected_files:
            logger.info("No uninspected files available for sampling")
            return []

        logger.info(f"Uninspected files: {len(uninspected_files)}")

        num_samples = max(1, len(uninspected_files) // self.config.inspection_sampling_ratio)
        selected_files = random.sample(uninspected_files, min(num_samples, len(uninspected_files)))

        logger.info(f"Selected {len(selected_files)} random samples for inspection")
        return selected_files

    def _extract_tar_file(self, tar_path: str, extract_dir: str) -> Optional[str]:
        """
        Extract TAR file to specified directory.

        Args:
            tar_path: Path to TAR file
            extract_dir: Directory to extract to

        Returns:
            Extraction directory path if successful
        """
        try:
            tar_filename = os.path.basename(tar_path)
            tar_name = tar_filename.rsplit('.tar', 1)[0]
            extraction_path = os.path.join(extract_dir, tar_name)

            if os.path.exists(extraction_path):
                shutil.rmtree(extraction_path)

            os.makedirs(extraction_path, exist_ok=True)

            logger.info(f"Extracting TAR file: {tar_filename}")

            with tarfile.open(tar_path, 'r:*') as tar:
                tar.extractall(path=extraction_path)

            num_files = sum([len(files) for _, _, files in os.walk(extraction_path)])
            logger.info(f"Extracted {num_files} files to: {extraction_path}")

            return extraction_path

        except Exception as e:
            logger.error(f"Failed to extract TAR file {tar_path}: {e}")
            return None

    def _get_all_extracted_files(self, extraction_path: str) -> List[Dict]:
        """Get list of all files from extracted directory with metadata."""
        extracted_files = []

        for root, dirs, files in os.walk(extraction_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, extraction_path)
                file_size = os.path.getsize(file_path)

                extracted_files.append({
                    'path': file_path,
                    'name': filename,
                    'size': file_size,
                    'relative_path': relative_path
                })

        logger.info(f"Found {len(extracted_files)} files in extracted directory")
        return extracted_files

    def _generate_sas_token(self, blob_path: str) -> str:
        """Generate SAS token for a blob with read permissions."""
        try:
            sas_token = generate_blob_sas(
                account_name=self.config.azure_account_name,
                container_name=self.config.azure_container_name,
                blob_name=blob_path,
                account_key=self.config.azure_account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=self.config.inspection_sas_token_expiry_days)
            )
            return sas_token
        except Exception as e:
            logger.error(f"Failed to generate SAS token for {blob_path}: {e}")
            return ""

    def _get_blob_url_with_sas(self, blob_path: str, sas_token: str) -> str:
        """Construct full blob URL with SAS token."""
        blob_client = self.container_client.get_blob_client(blob_path)
        blob_url = blob_client.url

        if sas_token:
            return f"{blob_url}?{sas_token}"
        return blob_url

    def _create_csv_report(self, report_data: List[Dict], report_path: str):
        """Create CSV report with inspection results."""
        try:
            fieldnames = [
                'tar_file_name',
                'individual_file_name',
                'individual_file_full_path',
                'individual_file_type',
                'tar_size_gb',
                'individual_file_size_gb',
                'sas_token',
                'blob_url',
                'blob_path',
                'transfer_date'
            ]

            with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(report_data)

            logger.info(f"CSV report created: {report_path}")
            logger.info(f"Report contains {len(report_data)} file entries")

        except Exception as e:
            logger.error(f"Failed to create CSV report: {e}")

    def _process_tar_for_inspection(self, tar_file_info: Dict) -> List[Dict]:
        """
        Process a single TAR file for inspection: download, extract, upload, generate SAS.

        Args:
            tar_file_info: TAR file info from SharePoint

        Returns:
            List of report data dictionaries
        """
        tar_filename = tar_file_info['name']
        tar_size_bytes = tar_file_info.get('size', 0)
        tar_size_gb = tar_size_bytes / (1024 ** 3)

        report_data = []
        local_tar_path = None
        extraction_path = None

        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing TAR for inspection: {tar_filename} ({tar_size_gb:.2f} GB)")
            logger.info(f"{'='*80}")

            # Download TAR
            local_tar_path = self._download_from_sharepoint(tar_file_info)
            if not local_tar_path:
                logger.error(f"Failed to download TAR file: {tar_filename}")
                return []

            # Extract TAR
            extraction_path = self._extract_tar_file(local_tar_path, self.config.temp_extract_dir)
            if not extraction_path:
                logger.error(f"Failed to extract TAR file: {tar_filename}")
                return []

            # Get all extracted files
            extracted_files = self._get_all_extracted_files(extraction_path)

            if not extracted_files:
                logger.warning(f"No files found in extracted TAR: {tar_filename}")
                return []

            logger.info(f"Processing {len(extracted_files)} extracted files for upload...")

            transfer_date = datetime.now().isoformat()

            for i, file_info in enumerate(extracted_files, 1):
                file_name = file_info['name']
                file_size_bytes = file_info['size']
                file_size_gb = file_size_bytes / (1024 ** 3)
                file_extension = os.path.splitext(file_name)[1] or 'no_extension'

                # Construct blob path
                tar_name = tar_filename.rsplit('.tar', 1)[0]
                blob_path = os.path.join(
                    self.config.inspection_untar_blob_prefix,
                    tar_name,
                    file_info['relative_path']
                ).replace('\\', '/')

                logger.info(f"  [{i}/{len(extracted_files)}] Uploading: {file_info['relative_path']} ({file_size_gb:.4f} GB)")

                # Upload to blob
                upload_success = self._upload_to_azure_blob(file_info['path'], blob_path, prefix=None)

                if not upload_success:
                    logger.warning(f"Failed to upload file: {file_name}")
                    continue

                # Generate SAS token
                sas_token = self._generate_sas_token(blob_path)

                # Generate blob URL with SAS
                blob_url = self._get_blob_url_with_sas(blob_path, sas_token)

                # Add to report data
                report_data.append({
                    'tar_file_name': tar_filename,
                    'individual_file_name': file_name,
                    'individual_file_full_path': file_info['relative_path'],
                    'individual_file_type': file_extension,
                    'tar_size_gb': f"{tar_size_gb:.4f}",
                    'individual_file_size_gb': f"{file_size_gb:.6f}",
                    'sas_token': sas_token,
                    'blob_url': blob_url,
                    'blob_path': blob_path,
                    'transfer_date': transfer_date
                })

            logger.info(f"✅ Successfully processed {len(report_data)} files from TAR: {tar_filename}")

        except Exception as e:
            logger.error(f"Error processing TAR file {tar_filename}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        finally:
            # Cleanup
            logger.info("Cleaning up temporary files...")

            if extraction_path and os.path.exists(extraction_path):
                try:
                    shutil.rmtree(extraction_path)
                    logger.info(f"✓ Removed extraction directory: {extraction_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove extraction directory {extraction_path}: {e}")

            if local_tar_path and os.path.exists(local_tar_path):
                try:
                    os.remove(local_tar_path)
                    logger.info(f"✓ Removed downloaded TAR: {local_tar_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove TAR file {local_tar_path}: {e}")

        return report_data

    def run_stage2_inspection(self, dry_run: bool = False) -> bool:
        """
        STAGE 2: Random inspection with extraction and detailed reporting.

        Args:
            dry_run: If True, show what would be inspected without processing

        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "="*100)
        logger.info("STAGE 2: TAR RANDOM INSPECTION (Extract, Upload, Report)")
        logger.info("="*100)

        try:
            # Load inspection state
            inspection_state = self._load_state(self.config.inspection_state_file)
            already_inspected = set(inspection_state.keys())

            # Get all TAR files from SharePoint
            all_tar_files = self._get_sharepoint_tar_files()

            if not all_tar_files:
                logger.warning("No TAR files found in SharePoint folder")
                return True

            # Select random samples
            selected_files = self._select_random_samples(all_tar_files, already_inspected)

            if not selected_files:
                logger.info("No files to inspect")
                return True

            logger.info(f"\nSelected {len(selected_files)} TAR files for random inspection:")
            for i, file_info in enumerate(selected_files, 1):
                size_gb = file_info.get('size', 0) / (1024 ** 3)
                logger.info(f"  {i}. {file_info['name']} ({size_gb:.2f} GB)")

            if dry_run:
                logger.info("\nDRY RUN MODE - No actual processing")
                return True

            # Process each selected TAR
            all_report_data = []
            successful_inspections = []
            failed_inspections = []

            for i, tar_file_info in enumerate(selected_files, 1):
                tar_filename = tar_file_info['name']

                logger.info(f"\n{'#'*100}")
                logger.info(f"TAR FILE {i}/{len(selected_files)}: {tar_filename}")
                logger.info(f"{'#'*100}")

                report_data = self._process_tar_for_inspection(tar_file_info)

                if report_data:
                    all_report_data.extend(report_data)
                    successful_inspections.append(tar_filename)

                    # Update inspection state
                    inspection_state[tar_filename] = {
                        'inspected_at': datetime.now().isoformat(),
                        'num_files_extracted': len(report_data),
                        'tar_size': tar_file_info.get('size', 0)
                    }

                    # Save state after each TAR
                    self._save_state(inspection_state, self.config.inspection_state_file)
                else:
                    failed_inspections.append(tar_filename)

                time.sleep(2)

            # Create CSV report
            if all_report_data:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_filename = f"unified_inspection_report_{timestamp}.csv"
                report_path = os.path.join(self.config.csv_report_dir, report_filename)

                self._create_csv_report(all_report_data, report_path)

                logger.info(f"\n{'='*100}")
                logger.info(f"📊 CSV Report Generated: {report_path}")
                logger.info(f"{'='*100}")

            # Summary
            logger.info("\n" + "="*100)
            logger.info("STAGE 2 SUMMARY:")
            logger.info(f"✅ Successful inspections: {len(successful_inspections)}")
            logger.info(f"❌ Failed inspections: {len(failed_inspections)}")
            logger.info(f"📁 Total files processed: {len(all_report_data)}")

            if failed_inspections:
                logger.info("\nFailed TAR files:")
                for filename in failed_inspections:
                    logger.info(f"  - {filename}")

            logger.info("="*100)

            return len(failed_inspections) == 0

        except Exception as e:
            logger.error(f"Stage 2 inspection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def run_unified_workflow(self, stage: str = "both", dry_run: bool = False) -> bool:
        """
        Run unified workflow: Stage 1 (transfer), Stage 2 (inspection), or both.

        Args:
            stage: Which stage to run ("1", "2", or "both")
            dry_run: If True, show what would be done without processing

        Returns:
            True if successful, False otherwise
        """
        logger.info("\n" + "#"*100)
        logger.info("UNIFIED TAR PROCESSING WORKFLOW")
        logger.info("#"*100)

        stage1_success = True
        stage2_success = True

        if stage in ["1", "both"]:
            stage1_success = self.run_stage1_transfer(dry_run)
            if not stage1_success:
                logger.error("Stage 1 failed, check logs for details")
                if stage == "both":
                    logger.warning("Continuing to Stage 2 despite Stage 1 failure...")

        if stage in ["2", "both"]:
            stage2_success = self.run_stage2_inspection(dry_run)
            if not stage2_success:
                logger.error("Stage 2 failed, check logs for details")

        # Final summary
        logger.info("\n" + "#"*100)
        logger.info("UNIFIED WORKFLOW COMPLETE")
        logger.info("#"*100)
        if stage in ["1", "both"]:
            logger.info(f"Stage 1 (Transfer): {'✅ SUCCESS' if stage1_success else '❌ FAILED'}")
        if stage in ["2", "both"]:
            logger.info(f"Stage 2 (Inspection): {'✅ SUCCESS' if stage2_success else '❌ FAILED'}")
        logger.info("#"*100)

        return stage1_success and stage2_success


def main():
    """Main function to run the unified TAR processing workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Unified TAR processing: transfer all files, then randomly inspect and report'
    )
    parser.add_argument('--config', '-c', default='config/unified_tar_config.yaml',
                       help='Path to unified configuration file')
    parser.add_argument('--stage', '-s', choices=['1', '2', 'both'], default='both',
                       help='Which stage to run: 1 (transfer), 2 (inspection), or both (default: both)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')

    args = parser.parse_args()

    try:
        # Create and run unified processor
        processor = UnifiedTarProcessor(config_path=args.config)
        success = processor.run_unified_workflow(stage=args.stage, dry_run=args.dry_run)

        if success:
            logger.info("\n✅ Unified TAR processing workflow completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ Unified TAR processing workflow completed with errors!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
