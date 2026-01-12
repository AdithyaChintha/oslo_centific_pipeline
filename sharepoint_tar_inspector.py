#!/usr/bin/env python3
"""
TAR Random Inspection and Azure Blob Upload Script

This script randomly samples TAR files from SharePoint or Azure Blob Storage (1 per 10 files),
untars them, uploads contents to Azure Blob Storage, generates SAS tokens, and creates a
detailed CSV report.

Supported Input Sources:
- SharePoint: Fetch TAR files from SharePoint folder
- Azure Blob: Fetch TAR files from Azure Blob Storage container

Workflow:
1. Lists all TAR files from the configured source (SharePoint or Azure Blob)
2. Randomly selects 1 file per every 10 files (or uses specific list)
3. Downloads selected TAR files
4. Untars/extracts files locally
5. Uploads all extracted files to Azure Blob destination
6. Generates SAS tokens for each uploaded file (for team inspection)
7. Creates CSV report with: tar_name, file_name, file_full_path, file_type, tar_size_gb,
   file_size_gb, sas_token, blob_url, blob_path, transfer_date

Configuration:
- Set `input_source: "sharepoint"` to fetch TAR files from SharePoint
- Set `input_source: "blob"` to fetch TAR files from Azure Blob Storage
- Configure `azure_source` section when using blob input source

Usage:
    python sharepoint_tar_inspector.py [--config CONFIG_FILE] [--dry-run]

Examples:
    # Run with default config (uses sharepoint by default)
    python sharepoint_tar_inspector.py

    # Run with custom config
    python sharepoint_tar_inspector.py --config config/tar_inspection_config.yaml

    # Dry run (show what would be inspected)
    python sharepoint_tar_inspector.py --dry-run

Author: Oslo TAR Inspection Workflow
"""

import os
import sys
import json
import csv
import time
import tarfile
import random
import shutil
import subprocess
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
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.sharepoint_utils import (
        SharePointAuthenticator, SharePointFileManager
    )

# Import Azure Blob Storage SDK
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
from azure.core.exceptions import ResourceExistsError

# Import video processing utilities
from utils.video_processing import downscale_video_to_480p, is_video_already_480p_or_smaller

# Ray import (optional - for parallel processing)
RAY_AVAILABLE = False
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sharepoint_tar_inspector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TarInspectionConfig:
    """Configuration class for TAR inspection settings."""

    def __init__(self, config_dict: Dict):
        """Initialize inspection config from dictionary."""
        # Input source: 'sharepoint' or 'blob'
        self.input_source = config_dict.get('input_source', 'sharepoint').lower()
        if self.input_source not in ['sharepoint', 'blob']:
            raise ValueError(f"Invalid input_source: {self.input_source}. Must be 'sharepoint' or 'blob'")

        # SharePoint settings (only required if input_source is 'sharepoint')
        if self.input_source == 'sharepoint':
            if 'sharepoint' not in config_dict:
                raise ValueError("input_source is 'sharepoint' but sharepoint config is missing")
            self.sharepoint_tar_folder_path = config_dict['sharepoint']['tar_source_folder_path']
        else:
            self.sharepoint_tar_folder_path = config_dict.get('sharepoint', {}).get('tar_source_folder_path', '')

        # Azure Blob source settings (for reading TAR files from blob - only required if input_source is 'blob')
        if self.input_source == 'blob':
            if 'azure_source' not in config_dict:
                raise ValueError("input_source is 'blob' but azure_source config is missing")
            azure_source = config_dict['azure_source']
            self.source_connection_string = self._get_connection_string(azure_source)
            self.source_container_name = azure_source['container_name']
            self.source_prefix = azure_source.get('source_prefix', '')
            self.project_id = azure_source.get('project_id', 'tar-inspection')
        else:
            self.source_connection_string = None
            self.source_container_name = None
            self.source_prefix = None
            self.project_id = config_dict.get('azure_source', {}).get('project_id', 'tar-inspection')

        # Azure Blob destination settings (for uploading extracted files)
        self.azure_connection_string = self._get_connection_string(config_dict['azure_blob'])
        self.azure_container_name = config_dict['azure_blob']['container_name']
        self.azure_account_name = self._extract_account_name(self.azure_connection_string)
        self.azure_account_key = self._extract_account_key(self.azure_connection_string)

        # Inspection-specific settings
        inspection_config = config_dict.get('inspection', {})
        self.untar_blob_prefix = inspection_config.get('untar_blob_prefix', 'untar_folder_oslo2/')
        self.sampling_ratio = inspection_config.get('sampling_ratio', 10)  # 1 per 10 files
        self.sas_token_expiry_days = inspection_config.get('sas_token_expiry_days', 30)
        self.execution_mode = inspection_config.get('execution_mode', 'random').lower()
        if self.execution_mode not in ['random', 'list', 'csv']:
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}. Must be 'random', 'list', or 'csv'")
        self.tar_file_list = inspection_config.get('tar_file_list', [])
        if self.execution_mode == 'list' and not self.tar_file_list:
            raise ValueError("execution_mode is 'list' but tar_file_list is empty")

        # CSV mode settings (for reading TAR files from comparison CSV)
        self.csv_input_file = inspection_config.get('csv_input_file', '')
        self.csv_sample_count = inspection_config.get('csv_sample_count', 4)  # Number of random TAR files to pick from CSV
        if self.execution_mode == 'csv' and not self.csv_input_file:
            raise ValueError("execution_mode is 'csv' but csv_input_file is not specified")

        # Video downscaling settings
        self.enable_video_downscaling = inspection_config.get('enable_video_downscaling', True)
        self.downscale_target_height = inspection_config.get('downscale_target_height', 480)
        self.downscale_quality = inspection_config.get('downscale_quality', 'medium')

        # GPU acceleration for video processing (applies regardless of Ray)
        # Check inspection-level use_gpu first, fall back to ray.use_gpu for backwards compatibility
        self.use_gpu = inspection_config.get('use_gpu', False)

        # Ray parallel processing settings
        ray_config = inspection_config.get('ray', {})
        self.use_ray = ray_config.get('enabled', False)
        self.ray_max_parallel = ray_config.get('max_parallel', 4)
        self.ray_use_gpu = ray_config.get('use_gpu', self.use_gpu)  # Fall back to inspection-level use_gpu

        # Local paths
        self.temp_download_dir = config_dict['local_paths']['temp_download_dir']
        self.temp_extract_dir = inspection_config.get('temp_extract_dir', '/data/oslo/tar_extractions')
        self.csv_report_dir = inspection_config.get('csv_report_dir', '/data/oslo/inspection_reports')
        self.inspection_state_file = inspection_config.get('inspection_state_file',
                                                           '/data/oslo/tar_temp_downloads/tar_inspection_state.json')

        # Report date prefix - extracted from source_prefix (e.g., "2025-12-01" -> "20251201")
        # If not provided, uses today's date
        self.report_date_prefix = inspection_config.get('report_date_prefix', None)

        # azcopy settings for faster downloads
        azcopy_config = inspection_config.get('azcopy', {})
        self.use_azcopy = azcopy_config.get('enabled', True)  # Default to azcopy for speed
        self.azcopy_path = azcopy_config.get('path', '/usr/local/bin/azcopy')
        self.azcopy_concurrency = azcopy_config.get('concurrency', 32)  # Parallel connections per file
        self.azcopy_block_size_mb = azcopy_config.get('block_size_mb', 8)  # Block size in MB

        # Parallel TAR download settings
        parallel_config = inspection_config.get('parallel_downloads', {})
        self.parallel_tar_downloads = parallel_config.get('enabled', False)
        self.max_parallel_tar_downloads = parallel_config.get('max_parallel', 2)

        # Cleanup settings - delete temp files after upload
        cleanup_config = inspection_config.get('cleanup', {})
        self.cleanup_enabled = cleanup_config.get('enabled', False)
        self.delete_tar_after_upload = cleanup_config.get('delete_tar_after_upload', False)
        self.delete_extraction_after_upload = cleanup_config.get('delete_extraction_after_upload', False)
        self.cleanup_on_failure = cleanup_config.get('cleanup_on_failure', False)

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


class TarInspectionManager:
    """Main manager class for random TAR inspection workflow."""

    def __init__(self, config_path: str = "config/tar_inspection_config.yaml"):
        """
        Initialize inspection manager with configuration.

        Args:
            config_path: Path to inspection configuration YAML file
        """
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = TarInspectionConfig(config_dict)

        # Initialize SharePoint only if input_source is 'sharepoint'
        if self.config.input_source == 'sharepoint':
            # Extract SharePoint connection details
            azure_ad = config_dict['azure_ad']
            sharepoint_config = config_dict['sharepoint']

            # Create SharePoint authenticator
            self.authenticator = SharePointAuthenticator(
                tenant_id=azure_ad['tenant_id'],
                client_id=azure_ad['client_id'],
                client_secret=azure_ad['client_secret'],
                timeout_seconds=config_dict.get('api', {}).get('timeout_seconds', 300)
            )

            # Create SharePoint file manager
            self.file_manager = SharePointFileManager(
                authenticator=self.authenticator,
                site_id=sharepoint_config['site_id'],
                drive_id=sharepoint_config['drive_id'],
                timeout=config_dict.get('api', {}).get('timeout_seconds', 300),
                max_retries=config_dict.get('api', {}).get('retry_attempts', 3),
                retry_delay=config_dict.get('api', {}).get('retry_delay_seconds', 5)
            )
        else:
            self.authenticator = None
            self.file_manager = None

        # Initialize Azure Blob source client (for reading TAR files from blob)
        if self.config.input_source == 'blob':
            self.source_blob_service_client = BlobServiceClient.from_connection_string(
                self.config.source_connection_string
            )
            self.source_container_client = self.source_blob_service_client.get_container_client(
                self.config.source_container_name
            )
        else:
            self.source_blob_service_client = None
            self.source_container_client = None

        # Initialize Azure Blob destination client (for uploading extracted files)
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
        os.makedirs(os.path.dirname(self.config.inspection_state_file), exist_ok=True)

        logger.info(f"TAR Inspection Manager initialized")
        logger.info(f"Input source: {self.config.input_source.upper()}")
        if self.config.input_source == 'sharepoint':
            logger.info(f"SharePoint folder: {self.config.sharepoint_tar_folder_path}")
        else:
            logger.info(f"Azure Blob source container: {self.config.source_container_name}")
            logger.info(f"Azure Blob source prefix: {self.config.source_prefix}")
            logger.info(f"Project ID: {self.config.project_id}")
        logger.info(f"Destination Azure container: {self.config.azure_container_name}")
        logger.info(f"Untar blob prefix: {self.config.untar_blob_prefix}")
        logger.info(f"Execution mode: {self.config.execution_mode}")
        if self.config.execution_mode == 'random':
            logger.info(f"Sampling ratio: 1 per {self.config.sampling_ratio} files")
        elif self.config.execution_mode == 'list':
            logger.info(f"Target files: {len(self.config.tar_file_list)} TAR files specified")

    def _load_inspection_state(self) -> Dict[str, Dict]:
        """Load inspection state from JSON file."""
        if os.path.exists(self.config.inspection_state_file):
            try:
                with open(self.config.inspection_state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded inspection state: {len(state)} files already inspected")
                return state if isinstance(state, dict) else {}
            except Exception as e:
                logger.warning(f"Failed to load inspection state: {e}")
                return {}
        return {}

    def _save_inspection_state(self, state: Dict[str, Dict]):
        """Save inspection state to JSON file."""
        try:
            with open(self.config.inspection_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug("Inspection state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save inspection state: {e}")

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

    def _get_blob_tar_files(self) -> List[Dict]:
        """Get list of TAR files from Azure Blob storage."""
        logger.info(f"Fetching TAR files from Azure Blob: {self.config.source_container_name}/{self.config.source_prefix}")

        tar_files = []
        try:
            # List blobs with the specified prefix
            blobs = self.source_container_client.list_blobs(name_starts_with=self.config.source_prefix)

            for blob in blobs:
                blob_name = blob.name
                # Filter for TAR files only
                if blob_name.lower().endswith(('.tar', '.tar.gz', '.tgz')):
                    # Extract just the filename from the full path
                    filename = os.path.basename(blob_name)

                    tar_files.append({
                        'name': filename,
                        'blob_name': blob_name,  # Full blob path for downloading
                        'size': blob.size,
                        'last_modified': blob.last_modified,
                        'source': 'blob'
                    })

            logger.info(f"Found {len(tar_files)} TAR files in Azure Blob")
            return tar_files

        except Exception as e:
            logger.error(f"Error listing TAR files from Azure Blob: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def _generate_blob_sas_url(self, blob_name: str, expiry_hours: int = 24) -> Optional[str]:
        """
        Generate a SAS URL for a specific blob.

        Args:
            blob_name: Full blob path
            expiry_hours: SAS token expiry in hours

        Returns:
            SAS URL string or None if failed
        """
        try:
            # Extract account name and key from connection string
            account_name = None
            account_key = None
            for part in self.config.source_connection_string.split(';'):
                if part.startswith('AccountName='):
                    account_name = part.split('=', 1)[1]
                elif part.startswith('AccountKey='):
                    account_key = part.split('=', 1)[1]

            if not account_name or not account_key:
                logger.error("Could not extract account credentials from connection string")
                return None

            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.config.source_container_name,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now() + timedelta(hours=expiry_hours)
            )

            sas_url = f"https://{account_name}.blob.core.windows.net/{self.config.source_container_name}/{blob_name}?{sas_token}"
            return sas_url

        except Exception as e:
            logger.error(f"Failed to generate SAS URL for {blob_name}: {e}")
            return None

    def _download_from_blob_azcopy(self, file_info: Dict) -> Optional[str]:
        """
        Download a TAR file from Azure Blob using azcopy (faster).

        Uses parallel connections and optimized chunking for 2-4x faster downloads.

        Args:
            file_info: TAR file info dict with 'blob_name' and 'name' keys

        Returns:
            Local file path if successful, None otherwise
        """
        filename = file_info['name']
        blob_name = file_info['blob_name']
        local_path = os.path.join(self.config.temp_download_dir, filename)

        # Generate SAS URL for the blob
        sas_url = self._generate_blob_sas_url(blob_name)
        if not sas_url:
            logger.warning("Failed to generate SAS URL, falling back to SDK download")
            return self._download_from_blob_sdk(file_info)

        logger.info(f"Downloading with azcopy: {blob_name}")
        start_time = time.time()

        try:
            # Build azcopy command with optimization flags
            cmd = [
                self.config.azcopy_path,
                'copy',
                sas_url,
                local_path,
                '--overwrite=true',
                f'--block-size-mb={self.config.azcopy_block_size_mb}',
                '--check-length=false',  # Skip length check for speed
            ]

            # Set environment variable for concurrency
            env = os.environ.copy()
            env['AZCOPY_CONCURRENCY_VALUE'] = str(self.config.azcopy_concurrency)

            # Run azcopy
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout for large files
                env=env
            )

            if result.returncode != 0:
                logger.error(f"azcopy failed: {result.stderr}")
                logger.warning("Falling back to SDK download")
                return self._download_from_blob_sdk(file_info)

            # Verify file exists and log stats
            if os.path.exists(local_path):
                file_size_gb = os.path.getsize(local_path) / (1024 ** 3)
                elapsed = time.time() - start_time
                speed_mbps = (file_size_gb * 1024) / elapsed if elapsed > 0 else 0

                logger.info(f"✅ Downloaded: {local_path}")
                logger.info(f"   Size: {file_size_gb:.2f} GB | Time: {elapsed:.1f}s | Speed: {speed_mbps:.1f} MB/s")
                return local_path
            else:
                logger.error(f"azcopy completed but file not found: {local_path}")
                return self._download_from_blob_sdk(file_info)

        except subprocess.TimeoutExpired:
            logger.error(f"azcopy download timed out for {blob_name}")
            return None
        except FileNotFoundError:
            logger.warning(f"azcopy not found at {self.config.azcopy_path}, falling back to SDK")
            return self._download_from_blob_sdk(file_info)
        except Exception as e:
            logger.error(f"azcopy download failed: {e}")
            logger.warning("Falling back to SDK download")
            return self._download_from_blob_sdk(file_info)

    def _download_from_blob_sdk(self, file_info: Dict) -> Optional[str]:
        """
        Download a TAR file from Azure Blob using SDK (fallback method).

        Uses chunked streaming for better memory efficiency.

        Args:
            file_info: TAR file info dict with 'blob_name' and 'name' keys

        Returns:
            Local file path if successful, None otherwise
        """
        filename = file_info['name']
        blob_name = file_info['blob_name']
        local_path = os.path.join(self.config.temp_download_dir, filename)

        logger.info(f"Downloading with SDK (chunked): {blob_name}")
        start_time = time.time()

        try:
            blob_client = self.source_container_client.get_blob_client(blob_name)

            # Use chunked download for better memory efficiency
            # max_concurrency enables parallel chunk downloads
            with open(local_path, 'wb') as f:
                stream = blob_client.download_blob(max_concurrency=4)
                # Read in 8MB chunks instead of loading entire file
                for chunk in stream.chunks():
                    f.write(chunk)

            file_size_gb = os.path.getsize(local_path) / (1024 ** 3)
            elapsed = time.time() - start_time
            speed_mbps = (file_size_gb * 1024) / elapsed if elapsed > 0 else 0

            logger.info(f"✅ Downloaded: {local_path}")
            logger.info(f"   Size: {file_size_gb:.2f} GB | Time: {elapsed:.1f}s | Speed: {speed_mbps:.1f} MB/s")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download from Azure Blob: {blob_name} - {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _download_from_blob(self, file_info: Dict) -> Optional[str]:
        """
        Download a TAR file from Azure Blob storage to local temp directory.

        Uses azcopy for faster downloads if available, falls back to SDK.

        Args:
            file_info: TAR file info dict with 'blob_name' and 'name' keys

        Returns:
            Local file path if successful, None otherwise
        """
        if self.config.use_azcopy:
            return self._download_from_blob_azcopy(file_info)
        else:
            return self._download_from_blob_sdk(file_info)

    def _select_random_samples(self, all_files: List[Dict], already_inspected: set) -> List[Dict]:
        """
        Select random samples: 1 per every N files (excluding already inspected).

        Args:
            all_files: List of all TAR files
            already_inspected: Set of already inspected filenames

        Returns:
            List of randomly selected files
        """
        # Filter out already inspected files
        uninspected_files = [f for f in all_files if f['name'] not in already_inspected]

        if not uninspected_files:
            logger.info("No uninspected files available for sampling")
            return []

        logger.info(f"Uninspected files: {len(uninspected_files)}")

        # Calculate number of samples
        num_samples = max(1, len(uninspected_files) // self.config.sampling_ratio)

        # Randomly select samples
        selected_files = random.sample(uninspected_files, min(num_samples, len(uninspected_files)))

        logger.info(f"Selected {len(selected_files)} random samples for inspection")
        return selected_files

    def _select_from_list(self, all_files: List[Dict], already_inspected: set) -> List[Dict]:
        """
        Select specific TAR files from the list provided in config.

        Args:
            all_files: List of all TAR files available in SharePoint
            already_inspected: Set of already inspected filenames

        Returns:
            List of selected files matching the tar_file_list from config
        """
        # Create a lookup dictionary for fast matching
        available_files_dict = {f['name']: f for f in all_files}

        selected_files = []
        missing_files = []
        already_processed_files = []

        for tar_filename in self.config.tar_file_list:
            # Check if already inspected
            if tar_filename in already_inspected:
                logger.info(f"Skipping (already inspected): {tar_filename}")
                already_processed_files.append(tar_filename)
                continue

            # Check if file exists in SharePoint
            if tar_filename in available_files_dict:
                selected_files.append(available_files_dict[tar_filename])
                logger.info(f"Found: {tar_filename}")
            else:
                logger.warning(f"NOT FOUND in SharePoint: {tar_filename}")
                missing_files.append(tar_filename)

        # Summary logging
        logger.info(f"\nList Mode Selection Summary:")
        logger.info(f"  Requested files: {len(self.config.tar_file_list)}")
        logger.info(f"  Found and selected: {len(selected_files)}")
        logger.info(f"  Already inspected: {len(already_processed_files)}")
        logger.info(f"  Missing/Not found: {len(missing_files)}")

        if missing_files:
            logger.warning(f"\nMissing files that will be skipped:")
            for filename in missing_files:
                logger.warning(f"{filename}")

        return selected_files

    def _select_from_csv(self, already_inspected: set) -> List[Dict]:
        """
        Select random TAR files from comparison CSV file.

        Reads the comparison CSV (output from tar_state_file_comparison.py),
        filters for successful uploads, and randomly samples N files.

        Args:
            already_inspected: Set of already inspected filenames

        Returns:
            List of selected files with blob info for downloading
        """
        csv_path = self.config.csv_input_file
        sample_count = self.config.csv_sample_count

        logger.info(f"Reading TAR files from comparison CSV: {csv_path}")
        logger.info(f"Random sample count: {sample_count}")

        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return []

        eligible_files = []
        skipped_not_found = 0
        skipped_failed = 0
        skipped_already_inspected = 0

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    tar_filename = row.get('tar_filename', '')
                    mapping_status = row.get('mapping_status', '').upper()
                    state_status = row.get('state_status', '').lower()
                    tar_full_path = row.get('tar_full_path', '')

                    # Skip if no filename
                    if not tar_filename:
                        continue

                    # Skip if already inspected
                    if tar_filename in already_inspected:
                        skipped_already_inspected += 1
                        continue

                    # Skip if not found in state (NOT_FOUND or ORPHANED_IN_STATE)
                    if mapping_status != 'FOUND':
                        skipped_not_found += 1
                        continue

                    # Skip if upload was not successful
                    if state_status not in ['success', 'successful']:
                        skipped_failed += 1
                        continue

                    # This file is eligible for inspection
                    # Build file info dict compatible with blob download
                    file_info = {
                        'name': tar_filename,
                        'blob_name': tar_full_path,  # Full blob path for downloading
                        'size': 0,  # Will be populated during download
                        'source': 'csv',
                        # Additional metadata from CSV
                        'tar_uuid': row.get('tar_uuid', ''),
                        'state_container_id': row.get('state_container_id', ''),
                    }

                    # Try to get size from CSV
                    try:
                        size_gb = float(row.get('tar_file_size_gb', 0))
                        file_info['size'] = int(size_gb * (1024 ** 3))  # Convert GB to bytes
                    except (ValueError, TypeError):
                        pass

                    eligible_files.append(file_info)

            logger.info(f"\nCSV Mode Selection Summary:")
            logger.info(f"  Total eligible files: {len(eligible_files)}")
            logger.info(f"  Skipped (already inspected): {skipped_already_inspected}")
            logger.info(f"  Skipped (not found/orphaned): {skipped_not_found}")
            logger.info(f"  Skipped (upload failed): {skipped_failed}")

            if not eligible_files:
                logger.warning("No eligible files found in CSV for inspection")
                return []

            # Randomly sample from eligible files
            actual_sample_count = min(sample_count, len(eligible_files))
            selected_files = random.sample(eligible_files, actual_sample_count)

            logger.info(f"  Randomly selected: {len(selected_files)} files")

            for i, f in enumerate(selected_files, 1):
                size_gb = f.get('size', 0) / (1024 ** 3)
                logger.info(f"    {i}. {f['name']} ({size_gb:.2f} GB)")

            return selected_files

        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

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

    def _extract_tar_file(self, tar_path: str, extract_dir: str) -> Optional[str]:
        """
        Extract TAR file to specified directory.

        Args:
            tar_path: Path to TAR file
            extract_dir: Directory to extract to

        Returns:
            Extraction directory path if successful, None otherwise
        """
        try:
            tar_filename = os.path.basename(tar_path)
            # Create subdirectory named after tar file (without extension)
            tar_name = tar_filename.rsplit('.tar', 1)[0]
            extraction_path = os.path.join(extract_dir, tar_name)

            # Clean up if exists
            if os.path.exists(extraction_path):
                shutil.rmtree(extraction_path)

            os.makedirs(extraction_path, exist_ok=True)

            logger.info(f"Extracting TAR file: {tar_filename}")

            with tarfile.open(tar_path, 'r:*') as tar:
                tar.extractall(path=extraction_path)

            # Count extracted files
            num_files = sum([len(files) for _, _, files in os.walk(extraction_path)])
            logger.info(f"Extracted {num_files} files to: {extraction_path}")

            return extraction_path

        except Exception as e:
            logger.error(f"Failed to extract TAR file {tar_path}: {e}")
            return None

    def _get_all_extracted_files(self, extraction_path: str) -> List[Dict]:
        """
        Get list of all files from extracted directory with metadata.

        Returns:
            List of dicts with file info: {path, name, size, relative_path}
        """
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

    def _build_scenario_mapping(self, extraction_path: str) -> tuple:
        """
        Parse all capture.json files and build scenario mapping for duplicate detection.

        Each capture.json contains:
        - scenarioId: The scenario identifier (e.g., "E00", "E01", "A01")
        - artifacts: Array of files belonging to this capture (mp4, wav files)

        Args:
            extraction_path: Path to the extracted TAR contents

        Returns:
            tuple: (file_to_scenario, duplicate_scenarios)
            - file_to_scenario: Dict mapping filename to scenarioId
            - duplicate_scenarios: Set of scenarioIds that appear more than once
        """
        file_to_scenario = {}
        scenario_counts = {}

        # Find all capture.json files
        capture_json_count = 0
        for root, dirs, files in os.walk(extraction_path):
            for filename in files:
                if filename == 'capture.json':
                    capture_path = os.path.join(root, filename)
                    try:
                        with open(capture_path, 'r', encoding='utf-8') as f:
                            capture_data = json.load(f)

                        scenario_id = capture_data.get('scenarioId', '')

                        if scenario_id:
                            # Count this scenario occurrence
                            scenario_counts[scenario_id] = scenario_counts.get(scenario_id, 0) + 1
                            capture_json_count += 1

                            # Map artifact files to this scenario
                            artifacts = capture_data.get('artifacts', [])
                            for artifact in artifacts:
                                artifact_file = artifact.get('file', '')
                                if artifact_file:
                                    # Extract just the filename from the path
                                    # e.g., "phases/P02_scenarioRunner/raw/P02_scenarioRunner_c013.mp4" -> "P02_scenarioRunner_c013.mp4"
                                    artifact_filename = os.path.basename(artifact_file)
                                    file_to_scenario[artifact_filename] = scenario_id

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON {capture_path}: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to read {capture_path}: {e}")

        # Find duplicate scenarios (scenarioId appears more than once)
        duplicate_scenarios = {sid for sid, count in scenario_counts.items() if count > 1}

        # Log summary
        logger.info(f"📋 Scenario Mapping Summary:")
        logger.info(f"   - Parsed {capture_json_count} capture.json files")
        logger.info(f"   - Found {len(scenario_counts)} unique scenarios")
        logger.info(f"   - Mapped {len(file_to_scenario)} artifact files")
        if duplicate_scenarios:
            logger.warning(f"   - ⚠️  Found {len(duplicate_scenarios)} DUPLICATE scenarios: {duplicate_scenarios}")
        else:
            logger.info(f"   - ✅ No duplicate scenarios found")

        return file_to_scenario, duplicate_scenarios

    def _unwarp_dual_fisheye_to_erp(self, input_video_path: str, output_video_path: str,
                                     use_gpu: bool = None, downscale_height: int = None) -> bool:
        """
        Convert dual-fisheye video to ERP (Equirectangular) view using ffmpeg.

        Optionally downscales in the same pass to avoid multiple encode/decode cycles.

        GPU Acceleration: When use_gpu=True, uses CUDA hardware decode (-hwaccel cuda)
        for faster processing. Encoding uses CPU (libx264) for reliability.
        Falls back to CPU-only if GPU processing fails.

        Single-Pass Optimization: When downscale_height is specified, the downscaling
        is integrated into the same FFmpeg filter chain, avoiding a separate encode/decode
        cycle. This significantly improves performance and reduces quality loss.

        Args:
            input_video_path: Path to dual-fisheye video
            output_video_path: Path for output ERP video
            use_gpu: Enable GPU acceleration (CUDA decode). If None, uses config setting.
            downscale_height: Target height for downscaling (e.g., 480). If None, no downscaling.

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Use config setting if not explicitly specified
            # Check inspection-level use_gpu first, then ray_use_gpu for backwards compatibility
            if use_gpu is None:
                use_gpu = getattr(self.config, 'use_gpu', False) or getattr(self.config, 'ray_use_gpu', False)

            logger.info(f"Converting dual-fisheye to ERP: {os.path.basename(input_video_path)}")
            logger.info(f"  🎯 GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
            if downscale_height:
                logger.info(f"  📉 Single-pass downscaling to {downscale_height}p enabled")

            # First, check if video has multiple streams (dual-lens)
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v',
                '-show_entries', 'stream=index,width,height',
                '-of', 'json',
                input_video_path
            ]

            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            if probe_result.returncode != 0:
                logger.error(f"Failed to probe video: {input_video_path}")
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
            # If downscaling, use smaller ERP to reduce processing (scale proportionally)
            if downscale_height:
                # For 480p output, use 960x480 ERP (maintains 2:1 ratio)
                ERP_H = downscale_height
                ERP_W = downscale_height * 2
                logger.info(f"Target ERP resolution (downscaled): {ERP_W}x{ERP_H}")
            else:
                ERP_W, ERP_H = 5760, 2880
                logger.info(f"Target ERP resolution: {ERP_W}x{ERP_H}")
            logger.info(f"Lens FOV: {LENS_FOV_DEG} degrees")

            # Build ffmpeg command based on number of video streams
            if num_video_streams >= 2:
                # Dual-lens: hstack two streams and convert to ERP, then rotate 180 degrees
                logger.info("🎬 Detected 2 video streams - converting dual-lens to ERP with 180° rotation")

                # Build filter chain - add scale if downscaling
                if downscale_height:
                    # Combined unwarp + downscale in single pass
                    filter_complex = (
                        f"[0:v:0][0:v:1]hstack=inputs=2[dual];"
                        f"[dual]v360=input=dfisheye:output=equirect:"
                        f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}[erp];"
                        f"[erp]hflip,vflip"
                    )
                    logger.info(f"  🚀 Single-pass: unwarp → rotate → {downscale_height}p output")
                else:
                    filter_complex = (
                        f"[0:v:0][0:v:1]hstack=inputs=2[dual];"
                        f"[dual]v360=input=dfisheye:output=equirect:"
                        f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}[erp];"
                        f"[erp]hflip,vflip"
                    )
                logger.debug(f"Filter complex: {filter_complex}")
                logger.info("  ↻ Applying 180° rotation (hflip + vflip) to correct orientation")

                # FFmpeg command with GPU acceleration (CUDA hardware decode)
                if use_gpu:
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output
                        '-hwaccel', 'cuda',  # GPU-accelerated decode
                        '-i', input_video_path,
                        '-filter_complex', filter_complex,
                        '-c:v', 'libx264',  # CPU encoding for reliability
                        '-crf', '23',
                        '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'copy',  # Copy audio stream
                        '-movflags', '+faststart',
                        output_video_path
                    ]
                else:
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output
                        '-i', input_video_path,
                        '-filter_complex', filter_complex,
                        '-c:v', 'libx264',
                        '-crf', '23',
                        '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'copy',  # Copy audio stream
                        '-movflags', '+faststart',
                        output_video_path
                    ]
            else:
                # Single lens fisheye: convert directly to ERP
                logger.info("🎬 Detected 1 video stream - converting fisheye to ERP")

                # Build filter - add scale if downscaling
                if downscale_height:
                    vf_filter = (
                        f"v360=input=fisheye:output=equirect:"
                        f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}"
                    )
                    logger.info(f"  🚀 Single-pass: unwarp → {downscale_height}p output")
                else:
                    vf_filter = (
                        f"v360=input=fisheye:output=equirect:"
                        f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}"
                    )
                logger.debug(f"Video filter: {vf_filter}")

                # FFmpeg command with GPU acceleration (CUDA hardware decode)
                if use_gpu:
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output
                        '-hwaccel', 'cuda',  # GPU-accelerated decode
                        '-i', input_video_path,
                        '-vf', vf_filter,
                        '-c:v', 'libx264',  # CPU encoding for reliability
                        '-crf', '23',
                        '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'copy',  # Copy audio stream
                        '-movflags', '+faststart',
                        output_video_path
                    ]
                else:
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output
                        '-i', input_video_path,
                        '-vf', vf_filter,
                        '-c:v', 'libx264',
                        '-crf', '23',
                        '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'copy',  # Copy audio stream
                        '-movflags', '+faststart',
                        output_video_path
                    ]

            logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")

            # Run ffmpeg conversion
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for large videos
            )

            if result.returncode != 0:
                # If GPU failed, try CPU fallback
                if use_gpu:
                    logger.warning(f"⚠️ GPU processing failed, falling back to CPU-only...")
                    return self._unwarp_dual_fisheye_to_erp(input_video_path, output_video_path,
                                                            use_gpu=False, downscale_height=downscale_height)
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                return False

            if os.path.exists(output_video_path):
                output_size = os.path.getsize(output_video_path)
                gpu_status = "GPU-accelerated" if use_gpu else "CPU-only"
                downscale_status = f" + {downscale_height}p" if downscale_height else ""
                logger.info(f"✅ ERP conversion successful ({gpu_status}{downscale_status}): {os.path.basename(output_video_path)} ({output_size / (1024**2):.2f} MB)")
                return True
            else:
                logger.error(f"Output file not created: {output_video_path}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg conversion timed out for {input_video_path}")
            return False
        except Exception as e:
            logger.error(f"Error during ERP conversion: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _is_dual_fisheye_video(self, video_path: str) -> bool:
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
            # Check file extension first (quick check)
            file_ext = os.path.splitext(video_path)[1].lower()
            logger.debug(f"Checking video: {os.path.basename(video_path)}, extension: {file_ext}")

            if file_ext in ['.insv']:
                logger.info(f"✅ Dual-fisheye detected (.insv extension): {os.path.basename(video_path)}")
                return True

            # Only check video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.insv']
            if file_ext not in video_extensions:
                logger.debug(f"Not a video file: {file_ext}")
                return False

            # Use ffprobe to get video stream info
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v',
                '-show_entries', 'stream=index,width,height,codec_name',
                '-of', 'json',
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                logger.debug(f"ffprobe failed for {os.path.basename(video_path)}")
                return False

            data = json.loads(result.stdout)
            streams = data.get('streams', [])

            logger.debug(f"Found {len(streams)} video stream(s) in {os.path.basename(video_path)}")
            for i, stream in enumerate(streams):
                width = stream.get('width', 'unknown')
                height = stream.get('height', 'unknown')
                codec = stream.get('codec_name', 'unknown')
                logger.debug(f"  Stream {i}: {width}x{height}, codec: {codec}")

            # Check 1: Does it have 2 video streams? (dual-lens indicator)
            if len(streams) >= 2:
                logger.info(f"✅ Dual-fisheye detected (2+ video streams): {os.path.basename(video_path)}")
                return True

            # Check 2: Does it have 2:1 aspect ratio? (common for dual-fisheye)
            if len(streams) == 1:
                width = streams[0].get('width', 0)
                height = streams[0].get('height', 0)

                if width > 0 and height > 0:
                    aspect_ratio = width / height
                    logger.debug(f"Aspect ratio: {aspect_ratio:.2f} ({width}x{height})")

                    # Check if aspect ratio is close to 2:1 (allowing some tolerance)
                    # Common dual-fisheye resolutions: 3840x1920, 5760x2880, 7680x3840
                    if 1.9 <= aspect_ratio <= 2.1:
                        logger.info(f"✅ Dual-fisheye detected (2:1 aspect ratio): {os.path.basename(video_path)} ({width}x{height})")
                        return True

            logger.debug(f"Not dual-fisheye: {os.path.basename(video_path)}")
            return False

        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            logger.debug(f"Could not check dual-fisheye for {os.path.basename(video_path)}: {e}")
            return False
        except Exception as e:
            logger.debug(f"Error checking dual-fisheye for {os.path.basename(video_path)}: {e}")
            return False

    def _downscale_video_if_needed(self, video_path: str) -> Optional[str]:
        """
        Downscale video to target resolution if enabled and video is larger than target.

        Args:
            video_path: Path to the video file

        Returns:
            Path to the downscaled video if downscaling was performed, None if skipped or failed
        """
        if not self.config.enable_video_downscaling:
            logger.debug(f"Video downscaling disabled in config")
            return None

        try:
            # Check if video is already at target resolution or smaller
            if is_video_already_480p_or_smaller(video_path):
                logger.debug(f"Video already {self.config.downscale_target_height}p or smaller: {os.path.basename(video_path)}")
                return None

            # Downscale the video
            video_name = os.path.basename(video_path)
            logger.info(f"  📉 Downscaling to {self.config.downscale_target_height}p: {video_name}")

            # Generate output path in same directory
            video_dir = os.path.dirname(video_path)
            video_basename = os.path.splitext(video_name)[0]
            downscaled_path = os.path.join(video_dir, f"{video_basename}_480p.mp4")

            # Perform downscaling
            result_path = downscale_video_to_480p(
                input_video_path=video_path,
                output_video_path=downscaled_path,
                target_height=self.config.downscale_target_height,
                quality=self.config.downscale_quality
            )

            if result_path and os.path.exists(result_path):
                # Get file sizes for logging
                original_size = os.path.getsize(video_path) / (1024**2)  # MB
                downscaled_size = os.path.getsize(result_path) / (1024**2)  # MB
                reduction_pct = ((original_size - downscaled_size) / original_size) * 100

                logger.info(f"  ✅ Downscaled: {original_size:.1f}MB → {downscaled_size:.1f}MB ({reduction_pct:.1f}% reduction)")
                return result_path
            else:
                logger.warning(f"  ⚠️  Downscaling failed for: {video_name}")
                return None

        except Exception as e:
            logger.error(f"Error during video downscaling: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _get_content_type(self, file_path: str) -> str:
        """
        Determine the appropriate content type for a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Content type string
        """
        extension = os.path.splitext(file_path)[1].lower()

        # Content type mapping for common file types
        content_type_map = {
            # Video formats
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.wmv': 'video/x-ms-wmv',
            '.flv': 'video/x-flv',
            '.webm': 'video/webm',
            '.mkv': 'video/x-matroska',
            '.m4v': 'video/x-m4v',

            # Audio formats
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
            '.flac': 'audio/flac',
            '.aac': 'audio/aac',
            '.wma': 'audio/x-ms-wma',

            # Text/Document formats
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.csv': 'text/csv',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.md': 'text/markdown',

            # Image formats
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.svg': 'image/svg+xml',

            # PDF
            '.pdf': 'application/pdf',
        }

        return content_type_map.get(extension, 'application/octet-stream')

    def _upload_file_to_blob(self, local_path: str, blob_path: str) -> bool:
        """
        Upload a single file to Azure Blob Storage with appropriate content type.

        Args:
            local_path: Local file path
            blob_path: Target blob path (including prefix)

        Returns:
            True if successful, False otherwise
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_path)

            # Determine content type
            content_type = self._get_content_type(local_path)

            # Set content settings with content type and content disposition
            content_settings = ContentSettings(
                content_type=content_type,
                content_disposition='inline'  # Force browser to display inline
            )

            # Check if already exists
            if blob_client.exists():
                # Update the content type for existing blob
                try:
                    blob_client.set_http_headers(content_settings=content_settings)
                    logger.debug(f"Blob already exists, updated content type: {blob_path} (content_type: {content_type})")
                except Exception as e:
                    logger.warning(f"Failed to update content type for existing blob {blob_path}: {e}")
                return True

            with open(local_path, 'rb') as data:
                blob_client.upload_blob(
                    data,
                    overwrite=False,
                    content_settings=content_settings
                )

            logger.debug(f"Uploaded to blob: {blob_path} (content_type: {content_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to upload {local_path} to blob: {e}")
            return False

    def _generate_sas_token(self, blob_path: str) -> str:
        """
        Generate SAS token for a blob with read permissions.

        Args:
            blob_path: Blob path

        Returns:
            SAS token string
        """
        try:
            sas_token = generate_blob_sas(
                account_name=self.config.azure_account_name,
                container_name=self.config.azure_container_name,
                blob_name=blob_path,
                account_key=self.config.azure_account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(days=self.config.sas_token_expiry_days)
            )
            return sas_token
        except Exception as e:
            logger.error(f"Failed to generate SAS token for {blob_path}: {e}")
            return ""

    def _get_blob_url_with_sas(self, blob_path: str, sas_token: str) -> str:
        """
        Construct full blob URL with SAS token.

        Args:
            blob_path: Blob path
            sas_token: SAS token

        Returns:
            Full URL with SAS token
        """
        blob_client = self.container_client.get_blob_client(blob_path)
        blob_url = blob_client.url

        if sas_token:
            return f"{blob_url}?{sas_token}"
        return blob_url

    def _create_csv_report(self, report_data: List[Dict], report_path: str):
        """
        Create CSV report with inspection results.

        Sorting: Video files (MP4) appear first, then all other files.

        Args:
            report_data: List of dictionaries with report data
            report_path: Path to save CSV report
        """
        try:
            # Sort report data: video files first, then other files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.insv']

            def is_video(entry):
                """Check if file is a video based on extension."""
                file_type = entry.get('individual_file_type', '').lower()
                return file_type in video_extensions

            # Separate videos and non-videos
            video_entries = [entry for entry in report_data if is_video(entry)]
            non_video_entries = [entry for entry in report_data if not is_video(entry)]

            # Combine: videos first, then others
            sorted_report_data = video_entries + non_video_entries

            logger.info(f"CSV sorting: {len(video_entries)} video files, {len(non_video_entries)} other files")

            fieldnames = [
                'tar_file_name',
                'individual_file_name',
                'scenarioID',
                'isDupScn',
                'individual_file_full_path',
                'individual_file_type',
                'is_dual_fisheye',
                'is_erp_version',
                'is_corrupted',
                'corruption_error',
                'tar_size_gb',
                'individual_file_size_gb',
                'sas_token',
                'Video_preview_url',
                'blob_path',
                'transfer_date'
            ]

            with open(report_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sorted_report_data)

            logger.info(f"CSV report created: {report_path}")
            logger.info(f"Report contains {len(sorted_report_data)} file entries (videos first, then other files)")

        except Exception as e:
            logger.error(f"Failed to create CSV report: {e}")

    # =========================================================================
    # MP4 CORRUPTION CHECK
    # =========================================================================

    def _check_mp4_corruption(self, file_path: str) -> tuple:
        """
        Check if an MP4 file is corrupted using ffprobe.

        Args:
            file_path: Path to the MP4 file

        Returns:
            Tuple of (is_corrupted: bool, error_message: Optional[str])
            - (False, None) if file is valid
            - (True, "error description") if file is corrupted
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',           # Only show errors
                '-select_streams', 'v:0', # Check first video stream
                '-show_entries', 'stream=codec_name,duration,nb_frames',
                '-show_entries', 'format=duration',
                '-of', 'json',
                file_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            # Check for errors in stderr
            if result.stderr and result.stderr.strip():
                # ffprobe outputs errors to stderr
                error_msg = result.stderr.strip()
                # Filter out non-critical warnings
                if any(keyword in error_msg.lower() for keyword in [
                    'invalid', 'corrupt', 'error', 'failed', 'moov atom not found',
                    'could not find codec', 'no such file'
                ]):
                    logger.warning(f"Corruption detected in {file_path}: {error_msg[:100]}")
                    return True, error_msg[:200]

            # Check if we got valid output
            if result.returncode != 0:
                return True, f"ffprobe returned exit code {result.returncode}"

            # Parse JSON output
            try:
                probe_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                return True, "Invalid ffprobe output (not valid JSON)"

            # Validate streams exist
            streams = probe_data.get('streams', [])
            if not streams:
                return True, "No video streams found"

            # Validate duration exists and is > 0
            format_info = probe_data.get('format', {})
            duration = format_info.get('duration')
            if duration:
                try:
                    if float(duration) <= 0:
                        return True, "Zero or negative duration"
                except (ValueError, TypeError):
                    return True, f"Invalid duration value: {duration}"

            # File is valid
            return False, None

        except subprocess.TimeoutExpired:
            logger.warning(f"ffprobe timed out for {file_path}")
            return True, "ffprobe timed out (>60s)"

        except FileNotFoundError:
            logger.error("ffprobe not found - ensure ffmpeg is installed")
            return True, "ffprobe not found"

        except Exception as e:
            logger.error(f"Error checking corruption for {file_path}: {e}")
            return True, str(e)[:200]

    # =========================================================================
    # RAY PARALLEL VIDEO PROCESSING METHODS
    # =========================================================================

    def _ray_get_video_streams(self, video_path: str) -> List[Dict]:
        """Get video stream information using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v',
            '-show_entries', 'stream=index,width,height,codec_name',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        return data.get('streams', [])

    def _ray_is_dual_fisheye(self, video_path: str) -> tuple:
        """
        Check if video is dual-fisheye format.

        Returns:
            Tuple of (is_dual_fisheye, num_streams)
        """
        file_ext = os.path.splitext(video_path)[1].lower()

        # Quick check for .insv extension
        if file_ext == '.insv':
            return True, 2

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.insv']
        if file_ext not in video_extensions:
            return False, 0

        try:
            streams = self._ray_get_video_streams(video_path)
            num_streams = len(streams)

            # 2+ video streams = dual-fisheye
            if num_streams >= 2:
                return True, num_streams

            # Check 2:1 aspect ratio (common for dual-fisheye)
            if num_streams == 1:
                width = streams[0].get('width', 0)
                height = streams[0].get('height', 0)
                if width > 0 and height > 0:
                    aspect_ratio = width / height
                    if 1.9 <= aspect_ratio <= 2.1:
                        return True, num_streams

            return False, num_streams
        except Exception:
            return False, 0

    def _ray_is_video_480p_or_smaller(self, video_path: str) -> bool:
        """Check if video is already 480p or smaller."""
        try:
            streams = self._ray_get_video_streams(video_path)
            if not streams:
                return False
            height = streams[0].get('height', 0)
            return height <= 480
        except Exception:
            return False

    def _ray_convert_fisheye_to_erp(self, input_path: str, output_path: str,
                                     lens_fov: int = 190, timeout: int = 3600,
                                     use_gpu: bool = None, downscale_height: int = None) -> Optional[str]:
        """
        Convert dual-fisheye video to equirectangular projection (ERP).

        Optionally downscales in the same pass to avoid multiple encode/decode cycles.

        GPU Acceleration: When use_gpu=True, uses CUDA hardware decode (-hwaccel cuda)
        for faster processing. Encoding uses CPU (libx264) for reliability.
        Falls back to CPU-only if GPU processing fails.

        Single-Pass Optimization: When downscale_height is specified, the ERP output
        resolution is set to match the target height, avoiding a separate downscale step.

        Args:
            input_path: Path to input video
            output_path: Path for output ERP video
            lens_fov: Lens field of view in degrees
            timeout: FFmpeg timeout in seconds
            use_gpu: Enable GPU acceleration (CUDA decode). If None, uses config setting.
            downscale_height: Target height for downscaling (e.g., 480). If None, uses 3840x1920.

        Returns:
            Output path if successful, None if failed
        """
        # Use config setting if not explicitly specified
        # Check inspection-level use_gpu first, then ray_use_gpu for backwards compatibility
        if use_gpu is None:
            use_gpu = getattr(self.config, 'use_gpu', False) or getattr(self.config, 'ray_use_gpu', False)

        # Calculate ERP dimensions - if downscaling, output directly at target resolution
        if downscale_height:
            erp_h = downscale_height
            erp_w = downscale_height * 2  # Maintain 2:1 ERP aspect ratio
            logger.info(f"  🚀 [Ray] Single-pass unwarp+downscale to {erp_w}x{erp_h}")
        else:
            erp_w, erp_h = 3840, 1920

        try:
            is_dual, num_streams = self._ray_is_dual_fisheye(input_path)
            if not is_dual:
                return None

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if num_streams >= 2:
                # Dual-stream: Stack horizontally then convert
                filter_complex = (
                    f"[0:v:0][0:v:1]hstack[dual];"
                    f"[dual]v360=input=dfisheye:output=equirect:"
                    f"ih_fov={lens_fov}:iv_fov={lens_fov}:w={erp_w}:h={erp_h}[erp];"
                    f"[erp]hflip,vflip"
                )
                # GPU-accelerated command (CUDA hardware decode)
                if use_gpu:
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-hwaccel', 'cuda',  # GPU-accelerated decode
                        '-i', input_path,
                        '-filter_complex', filter_complex,
                        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'copy', '-movflags', '+faststart',
                        output_path
                    ]
                else:
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-i', input_path,
                        '-filter_complex', filter_complex,
                        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'copy', '-movflags', '+faststart',
                        output_path
                    ]
            else:
                # Single stream 2:1 aspect - direct conversion
                vf_filter = (
                    f"v360=input=dfisheye:output=equirect:"
                    f"ih_fov={lens_fov}:iv_fov={lens_fov}:w={erp_w}:h={erp_h},"
                    f"hflip,vflip"
                )
                # GPU-accelerated command (CUDA hardware decode)
                if use_gpu:
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-hwaccel', 'cuda',  # GPU-accelerated decode
                        '-i', input_path,
                        '-vf', vf_filter,
                        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'copy', '-movflags', '+faststart',
                        output_path
                    ]
                else:
                    ffmpeg_cmd = [
                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                        '-i', input_path,
                        '-vf', vf_filter,
                        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'copy', '-movflags', '+faststart',
                        output_path
                    ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0 and os.path.exists(output_path):
                return output_path

            # If GPU failed, try CPU fallback
            if use_gpu:
                logger.warning(f"⚠️ GPU processing failed for {os.path.basename(input_path)}, falling back to CPU...")
                return self._ray_convert_fisheye_to_erp(input_path, output_path, lens_fov, timeout,
                                                        use_gpu=False, downscale_height=downscale_height)

            return None
        except Exception as e:
            # If GPU failed with exception, try CPU fallback
            if use_gpu:
                logger.warning(f"⚠️ GPU processing exception for {os.path.basename(input_path)}: {e}, falling back to CPU...")
                return self._ray_convert_fisheye_to_erp(input_path, output_path, lens_fov, timeout,
                                                        use_gpu=False, downscale_height=downscale_height)
            return None

    def _ray_downscale_video(self, input_path: str, output_path: str,
                              target_height: int = 480, quality: str = 'medium',
                              timeout: int = 1800) -> Optional[str]:
        """
        Downscale video to target height.

        Returns:
            Output path if successful, None if failed
        """
        try:
            if self._ray_is_video_480p_or_smaller(input_path):
                return None

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            quality_map = {
                'fast': ('veryfast', '28'),
                'medium': ('medium', '23'),
                'slow': ('slow', '20')
            }
            preset, crf = quality_map.get(quality, ('medium', '23'))

            vf_filter = f"scale=-2:{target_height}"
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', input_path,
                '-vf', vf_filter,
                '-c:v', 'libx264', '-preset', preset, '-crf', crf,
                '-c:a', 'copy', '-movflags', '+faststart',
                output_path
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            return None
        except Exception:
            return None

    def _ray_process_single_video(self, video_path: str, output_dir: str) -> Dict:
        """
        Process a single video: detect type, convert if fisheye, downscale.

        Returns:
            Dict with processing results
        """
        result = {
            'input_path': video_path,
            'output_path': video_path,
            'is_dual_fisheye': False,
            'erp_converted': False,
            'downscaled': False,
            'success': False,
            'error': None
        }

        try:
            video_name = os.path.basename(video_path)
            video_basename = os.path.splitext(video_name)[0]
            os.makedirs(output_dir, exist_ok=True)

            current_path = video_path

            # Step 1: Check if dual-fisheye and convert to ERP
            is_dual, num_streams = self._ray_is_dual_fisheye(current_path)
            result['is_dual_fisheye'] = is_dual

            if is_dual:
                # Determine if we should do single-pass unwarp+downscale
                target_height = self.config.downscale_target_height if self.config.enable_video_downscaling else None

                if target_height:
                    # Single-pass: unwarp + downscale together
                    erp_output = os.path.join(output_dir, f"{video_basename}_erp_{target_height}p.mp4")
                    logger.info(f"  🚀 [Ray] Single-pass unwarp+downscale to {target_height}p: {video_name}")
                else:
                    erp_output = os.path.join(output_dir, f"{video_basename}_erp.mp4")
                    logger.info(f"  🔄 [Ray] Converting fisheye to ERP: {video_name}")

                erp_result = self._ray_convert_fisheye_to_erp(
                    current_path, erp_output,
                    downscale_height=target_height
                )

                if erp_result:
                    result['erp_converted'] = True
                    if target_height:
                        result['downscaled'] = True  # Downscaling was done in same pass
                    current_path = erp_result
                    logger.info(f"  ✅ [Ray] ERP conversion done: {os.path.basename(erp_result)}")
                else:
                    result['error'] = 'ERP conversion failed'
                    logger.warning(f"  ⚠️  [Ray] ERP conversion failed: {video_name}")

            # Step 2: Downscale if needed (only if not already done in Step 1)
            if self.config.enable_video_downscaling and not result.get('downscaled'):
                if not self._ray_is_video_480p_or_smaller(current_path):
                    downscale_output = os.path.join(
                        output_dir,
                        f"{os.path.splitext(os.path.basename(current_path))[0]}_{self.config.downscale_target_height}p.mp4"
                    )
                    logger.info(f"  📉 [Ray] Downscaling: {os.path.basename(current_path)}")

                    downscale_result = self._ray_downscale_video(
                        current_path, downscale_output,
                        target_height=self.config.downscale_target_height,
                        quality=self.config.downscale_quality
                    )

                    if downscale_result:
                        result['downscaled'] = True
                        current_path = downscale_result
                        logger.info(f"  ✅ [Ray] Downscale done: {os.path.basename(downscale_result)}")

            result['output_path'] = current_path
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
            logger.error(f"  ❌ [Ray] Error processing {video_path}: {e}")

        return result

    def _process_videos_with_ray(self, video_files: List[Dict], output_dir: str) -> Dict[str, Dict]:
        """
        Process multiple videos in parallel using Ray.

        Args:
            video_files: List of video file info dicts with 'path', 'name', 'relative_path'
            output_dir: Directory for processed output files

        Returns:
            Dict mapping original file path to processing result:
            {
                '/path/to/video.mp4': {
                    'output_path': '/path/to/processed.mp4',
                    'is_dual_fisheye': bool,
                    'erp_converted': bool,
                    'downscaled': bool,
                    'success': bool
                }
            }
        """
        if not RAY_AVAILABLE:
            logger.warning("Ray not available - falling back to sequential processing")
            return {}

        if not video_files:
            return {}

        # Initialize Ray if needed with proper error handling
        try:
            if not ray.is_initialized():
                # Use local mode to avoid GCS issues, set reasonable resource limits
                ray.init(
                    ignore_reinit_error=True,
                    num_cpus=max(1, self.config.ray_max_parallel),
                    include_dashboard=False,  # Disable dashboard to reduce overhead
                    _temp_dir="/tmp/ray_temp",  # Explicit temp dir
                    logging_level="warning",  # Reduce log verbosity
                )
                logger.info("Ray initialized for parallel video processing")
        except Exception as e:
            logger.warning(f"Failed to initialize Ray: {e}")
            logger.warning("Falling back to sequential processing")
            return {}

        logger.info(f"🚀 Processing {len(video_files)} videos in parallel with Ray")
        logger.info(f"   Max parallel: {self.config.ray_max_parallel}, GPU: {self.config.ray_use_gpu}")

        # Define Ray remote function for processing a single video
        @ray.remote(num_cpus=1)
        def process_video_task(video_path: str, output_dir: str,
                               enable_downscaling: bool, target_height: int,
                               downscale_quality: str, use_gpu: bool = False) -> Dict:
            """Ray task to process a single video with single-pass unwarp+downscale optimization."""
            result = {
                'input_path': video_path,
                'output_path': video_path,
                'is_dual_fisheye': False,
                'erp_converted': False,
                'downscaled': False,
                'success': False,
                'error': None
            }

            try:
                video_name = os.path.basename(video_path)
                video_basename = os.path.splitext(video_name)[0]
                os.makedirs(output_dir, exist_ok=True)

                current_path = video_path

                # Helper: Get video streams
                def get_streams(path):
                    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v',
                           '-show_entries', 'stream=index,width,height',
                           '-of', 'json', path]
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if r.returncode != 0:
                        return []
                    return json.loads(r.stdout).get('streams', [])

                # Helper: Check if dual-fisheye
                def is_dual_fisheye(path):
                    ext = os.path.splitext(path)[1].lower()
                    if ext == '.insv':
                        return True, 2
                    streams = get_streams(path)
                    if len(streams) >= 2:
                        return True, len(streams)
                    if len(streams) == 1:
                        w, h = streams[0].get('width', 0), streams[0].get('height', 0)
                        if w > 0 and h > 0 and 1.9 <= w/h <= 2.1:
                            return True, 1
                    return False, len(streams)

                # Helper: Check if 480p or smaller
                def is_small(path):
                    streams = get_streams(path)
                    return streams and streams[0].get('height', 9999) <= target_height

                # Step 1: Check for dual-fisheye
                is_dual, num_streams = is_dual_fisheye(current_path)
                result['is_dual_fisheye'] = is_dual

                if is_dual:
                    lens_fov = 190

                    # Single-pass optimization: output directly at target resolution if downscaling enabled
                    if enable_downscaling:
                        erp_w, erp_h = target_height * 2, target_height  # e.g., 960x480 for 480p
                        erp_output = os.path.join(output_dir, f"{video_basename}_erp_{target_height}p.mp4")
                    else:
                        erp_w, erp_h = 3840, 1920
                        erp_output = os.path.join(output_dir, f"{video_basename}_erp.mp4")

                    # Build FFmpeg command with optional GPU acceleration
                    base_cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
                    if use_gpu:
                        base_cmd.extend(['-hwaccel', 'cuda'])

                    if num_streams >= 2:
                        fc = (f"[0:v:0][0:v:1]hstack[dual];"
                              f"[dual]v360=input=dfisheye:output=equirect:"
                              f"ih_fov={lens_fov}:iv_fov={lens_fov}:w={erp_w}:h={erp_h}[erp];"
                              f"[erp]hflip,vflip")
                        cmd = base_cmd + ['-i', current_path, '-filter_complex', fc,
                               '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
                               '-pix_fmt', 'yuv420p',
                               '-c:a', 'copy', '-movflags', '+faststart', erp_output]
                    else:
                        vf = (f"v360=input=dfisheye:output=equirect:"
                              f"ih_fov={lens_fov}:iv_fov={lens_fov}:w={erp_w}:h={erp_h},"
                              f"hflip,vflip")
                        cmd = base_cmd + ['-i', current_path, '-vf', vf,
                               '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
                               '-pix_fmt', 'yuv420p',
                               '-c:a', 'copy', '-movflags', '+faststart', erp_output]

                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                    if r.returncode == 0 and os.path.exists(erp_output):
                        result['erp_converted'] = True
                        if enable_downscaling:
                            result['downscaled'] = True  # Downscaling was done in single pass
                        current_path = erp_output

                # Step 2: Downscale if needed (only for non-fisheye videos or if unwarp didn't include downscale)
                if enable_downscaling and not result.get('downscaled') and not is_small(current_path):
                    ds_output = os.path.join(
                        output_dir,
                        f"{os.path.splitext(os.path.basename(current_path))[0]}_{target_height}p.mp4"
                    )
                    quality_map = {'fast': ('veryfast', '28'), 'medium': ('medium', '23'), 'slow': ('slow', '20')}
                    preset, crf = quality_map.get(downscale_quality, ('medium', '23'))

                    base_cmd = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error']
                    if use_gpu:
                        base_cmd.extend(['-hwaccel', 'cuda'])

                    cmd = base_cmd + ['-i', current_path, '-vf', f'scale=-2:{target_height}',
                           '-c:v', 'libx264', '-preset', preset, '-crf', crf,
                           '-pix_fmt', 'yuv420p',
                           '-c:a', 'copy', '-movflags', '+faststart', ds_output]

                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                    if r.returncode == 0 and os.path.exists(ds_output):
                        result['downscaled'] = True
                        current_path = ds_output

                result['output_path'] = current_path
                result['success'] = True

            except Exception as e:
                result['error'] = str(e)

            return result

        # Submit all tasks to Ray with proper cleanup
        result_map = {}
        try:
            futures = []
            use_gpu = self.config.use_gpu or self.config.ray_use_gpu
            for video_file in video_files:
                future = process_video_task.remote(
                    video_file['path'],
                    output_dir,
                    self.config.enable_video_downscaling,
                    self.config.downscale_target_height,
                    self.config.downscale_quality,
                    use_gpu
                )
                futures.append(future)

            # Gather results with timeout
            try:
                results = ray.get(futures, timeout=1800)  # 30 minute timeout for all videos
            except ray.exceptions.GetTimeoutError:
                logger.error("Ray processing timed out after 30 minutes")
                # Cancel any pending tasks
                for future in futures:
                    try:
                        ray.cancel(future, force=True)
                    except Exception:
                        pass
                return {}

            # Summary logging
            successful = sum(1 for r in results if r['success'])
            erp_converted = sum(1 for r in results if r['erp_converted'])
            downscaled = sum(1 for r in results if r['downscaled'])

            logger.info(f"📊 Ray processing complete:")
            logger.info(f"   Total: {len(results)}, Successful: {successful}")
            logger.info(f"   ERP converted: {erp_converted}, Downscaled: {downscaled}")

            # Build result mapping
            for result in results:
                result_map[result['input_path']] = result

        except Exception as e:
            logger.error(f"Ray processing failed: {e}")
            logger.warning("Ray processing error - results may be incomplete")
        finally:
            # Always shutdown Ray to prevent zombie processes
            try:
                if ray.is_initialized():
                    ray.shutdown()
                    logger.info("Ray shutdown complete")
            except Exception as e:
                logger.warning(f"Error during Ray shutdown: {e}")

        return result_map

    def _process_tar_file(self, tar_file_info: Dict) -> List[Dict]:
        """
        Process a single TAR file: download, extract, upload, generate SAS, collect data.

        Args:
            tar_file_info: TAR file info from SharePoint or Azure Blob

        Returns:
            List of report data dictionaries for all extracted files
        """
        tar_filename = tar_file_info['name']
        tar_size_bytes = tar_file_info.get('size', 0)
        tar_size_gb = tar_size_bytes / (1024 ** 3)

        report_data = []
        local_tar_path = None
        extraction_path = None
        upload_successful = False  # Track if uploads completed successfully

        try:
            # Step 1: Download TAR file from source (SharePoint or Azure Blob)
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing TAR file: {tar_filename} ({tar_size_gb:.2f} GB)")
            logger.info(f"Source: {self.config.input_source.upper()}")
            if self.config.use_ray and RAY_AVAILABLE:
                logger.info(f"Ray parallel processing: ENABLED (max {self.config.ray_max_parallel} parallel)")
            logger.info(f"{'='*80}")

            # Download based on input source
            if self.config.input_source == 'blob':
                local_tar_path = self._download_from_blob(tar_file_info)
            else:
                local_tar_path = self._download_from_sharepoint(tar_file_info)

            if not local_tar_path:
                logger.error(f"Failed to download TAR file: {tar_filename}")
                return []

            # Step 2: Extract TAR file
            extraction_path = self._extract_tar_file(local_tar_path, self.config.temp_extract_dir)
            if not extraction_path:
                logger.error(f"Failed to extract TAR file: {tar_filename}")
                return []

            # Step 3: Get all extracted files
            extracted_files = self._get_all_extracted_files(extraction_path)

            if not extracted_files:
                logger.warning(f"No files found in extracted TAR: {tar_filename}")
                return []

            # Step 3.5: Build scenario mapping for duplicate detection
            file_to_scenario, duplicate_scenarios = self._build_scenario_mapping(extraction_path)

            logger.info(f"Processing {len(extracted_files)} extracted files for upload...")

            # Step 4: Upload each file to blob and generate SAS tokens
            transfer_date = datetime.now().isoformat()
            tar_name = tar_filename.rsplit('.tar', 1)[0]

            # Separate video files from non-video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.insv']
            video_files = []
            non_video_files = []

            for file_info in extracted_files:
                file_extension = os.path.splitext(file_info['name'])[1].lower()
                if file_extension in video_extensions:
                    video_files.append(file_info)
                else:
                    non_video_files.append(file_info)

            logger.info(f"Found {len(video_files)} video files and {len(non_video_files)} non-video files")

            # Process videos with Ray if enabled and available
            ray_results = {}
            if self.config.use_ray and RAY_AVAILABLE and video_files:
                logger.info(f"🚀 Processing {len(video_files)} videos with Ray parallel processing...")

                # Create a temp output directory for processed videos
                ray_output_dir = os.path.join(extraction_path, '_ray_processed')
                os.makedirs(ray_output_dir, exist_ok=True)

                ray_results = self._process_videos_with_ray(video_files, ray_output_dir)
                logger.info(f"Ray processing complete. {len(ray_results)} results.")

            # Process all files (videos use Ray results if available, non-videos processed normally)
            all_files = extracted_files
            for i, file_info in enumerate(all_files, 1):
                file_name = file_info['name']
                file_size_bytes = file_info['size']
                file_size_gb = file_size_bytes / (1024 ** 3)
                file_extension = os.path.splitext(file_name)[1].lower() or 'no_extension'

                # Get scenario info for this file (for duplicate detection)
                scenario_id = file_to_scenario.get(file_name, '')
                is_dup_scn = 'Y' if scenario_id in duplicate_scenarios else ('N' if scenario_id else '')

                # Construct blob path: untar_folder_oslo2/tar_name/relative_path
                blob_path = os.path.join(
                    self.config.untar_blob_prefix,
                    tar_name,
                    file_info['relative_path']
                ).replace('\\', '/')  # Ensure forward slashes

                # Check for MP4 corruption BEFORE any processing (unwarp, downscale)
                is_corrupted = False
                corruption_error = ''
                if file_extension in video_extensions:
                    is_corrupted, corruption_error_msg = self._check_mp4_corruption(file_info['path'])
                    corruption_error = corruption_error_msg if corruption_error_msg else ''
                    if is_corrupted:
                        logger.warning(f"  ⚠️  CORRUPTED VIDEO: {file_name} - {corruption_error}")

                # Check if this is a video that was processed by Ray
                ray_result = ray_results.get(file_info['path'])

                if ray_result and ray_result.get('success'):
                    # Video was processed by Ray
                    is_dual_fisheye = ray_result.get('is_dual_fisheye', False)
                    erp_converted = ray_result.get('erp_converted', False)
                    downscaled = ray_result.get('downscaled', False)
                    output_path = ray_result.get('output_path', file_info['path'])

                    if is_dual_fisheye:
                        logger.info(f"  🎥 [{i}/{len(all_files)}] DUAL-FISHEYE (Ray processed): {file_name}")

                        # Skip uploading original dual-fisheye, upload ERP version
                        if erp_converted and os.path.exists(output_path):
                            erp_file_name = os.path.basename(output_path)
                            erp_file_size_bytes = os.path.getsize(output_path)
                            erp_file_size_gb = erp_file_size_bytes / (1024 ** 3)

                            # Construct ERP blob path
                            erp_relative_path = os.path.join(
                                os.path.dirname(file_info['relative_path']),
                                erp_file_name
                            ).replace('\\', '/')

                            erp_blob_path = os.path.join(
                                self.config.untar_blob_prefix,
                                tar_name,
                                erp_relative_path
                            ).replace('\\', '/')

                            logger.info(f"  📤 Uploading ERP video: {erp_file_name} ({erp_file_size_gb:.4f} GB)")

                            upload_success = self._upload_file_to_blob(output_path, erp_blob_path)

                            if upload_success:
                                sas_token = self._generate_sas_token(erp_blob_path)
                                blob_url = self._get_blob_url_with_sas(erp_blob_path, sas_token)

                                report_data.append({
                                    'tar_file_name': tar_filename,
                                    'individual_file_name': erp_file_name,
                                    'scenarioID': scenario_id,
                                    'isDupScn': is_dup_scn,
                                    'individual_file_full_path': erp_relative_path,
                                    'individual_file_type': '.mp4',
                                    'is_dual_fisheye': 'NO',
                                    'is_erp_version': 'YES',
                                    'is_corrupted': 'YES' if is_corrupted else 'NO',
                                    'corruption_error': corruption_error,
                                    'tar_size_gb': f"{tar_size_gb:.4f}",
                                    'individual_file_size_gb': f"{erp_file_size_gb:.6f}",
                                    'sas_token': sas_token,
                                    'Video_preview_url': blob_url,
                                    'blob_path': erp_blob_path,
                                    'transfer_date': transfer_date
                                })
                                logger.info(f"  ✅ ERP video uploaded: {erp_file_name}")
                            else:
                                logger.warning(f"  ⚠️  Failed to upload ERP video: {erp_file_name}")
                        else:
                            logger.warning(f"  ⚠️  ERP conversion failed for: {file_name}")
                    else:
                        # Normal video (possibly downscaled by Ray)
                        final_path = output_path if os.path.exists(output_path) else file_info['path']
                        final_file_name = os.path.basename(final_path)
                        final_size_bytes = os.path.getsize(final_path)
                        final_size_gb = final_size_bytes / (1024 ** 3)

                        # Update blob path if file was downscaled
                        if downscaled and final_path != file_info['path']:
                            blob_path = os.path.join(
                                self.config.untar_blob_prefix,
                                tar_name,
                                os.path.dirname(file_info['relative_path']),
                                final_file_name
                            ).replace('\\', '/')

                        logger.info(f"  [{i}/{len(all_files)}] Uploading (Ray): {final_file_name} ({final_size_gb:.4f} GB)")

                        upload_success = self._upload_file_to_blob(final_path, blob_path)

                        if upload_success:
                            sas_token = self._generate_sas_token(blob_path)
                            blob_url = self._get_blob_url_with_sas(blob_path, sas_token)

                            report_data.append({
                                'tar_file_name': tar_filename,
                                'individual_file_name': final_file_name,
                                'scenarioID': scenario_id,
                                'isDupScn': is_dup_scn,
                                'individual_file_full_path': file_info['relative_path'],
                                'individual_file_type': file_extension,
                                'is_dual_fisheye': 'NO',
                                'is_erp_version': 'NO',
                                'is_corrupted': 'YES' if is_corrupted else 'NO',
                                'corruption_error': corruption_error,
                                'tar_size_gb': f"{tar_size_gb:.4f}",
                                'individual_file_size_gb': f"{final_size_gb:.6f}",
                                'sas_token': sas_token,
                                'Video_preview_url': blob_url,
                                'blob_path': blob_path,
                                'transfer_date': transfer_date
                            })
                        else:
                            logger.warning(f"Failed to upload file: {final_file_name}")

                elif file_extension in video_extensions:
                    # Video file - process without Ray (fallback or Ray disabled)
                    is_dual_fisheye = False
                    erp_video_path = None
                    erp_created = False
                    final_video_path = file_info['path']

                    # STEP 1: Check if dual-fisheye
                    is_dual_fisheye = self._is_dual_fisheye_video(file_info['path'])

                    if is_dual_fisheye:
                        logger.info(f"  🎥 DUAL-FISHEYE detected: {file_name}")

                        # Determine if we should do single-pass unwarp+downscale
                        target_height = self.config.downscale_target_height if self.config.enable_video_downscaling else None

                        # STEP 2: Convert to ERP (optionally with downscaling in single pass)
                        file_name_without_ext = os.path.splitext(file_name)[0]
                        if target_height:
                            erp_file_name = f"{file_name_without_ext}_erp_{target_height}p.mp4"
                            logger.info(f"  🚀 Single-pass unwarp+downscale to {target_height}p: {erp_file_name}")
                        else:
                            erp_file_name = f"{file_name_without_ext}_erpview.mp4"
                            logger.info(f"  🔄 Converting to ERP: {erp_file_name}")

                        erp_video_path = os.path.join(os.path.dirname(file_info['path']), erp_file_name)
                        erp_created = self._unwarp_dual_fisheye_to_erp(
                            file_info['path'], erp_video_path,
                            downscale_height=target_height
                        )

                        if not erp_created:
                            logger.warning(f"  ⚠️  Failed to create ERP video for: {file_name}")

                        # NOTE: Downscaling was already done in single-pass if target_height was set
                    else:
                        # Normal video - just downscale to 480p
                        if self.config.enable_video_downscaling:
                            downscaled_path = self._downscale_video_if_needed(file_info['path'])
                            if downscaled_path:
                                final_video_path = downscaled_path

                    # Skip uploading original dual-fisheye videos
                    if is_dual_fisheye:
                        logger.info(f"  ℹ️  Skipping upload of original dual-fisheye video")
                    else:
                        logger.info(f"  [{i}/{len(all_files)}] Uploading: {file_info['relative_path']} ({file_size_gb:.4f} GB)")

                        upload_success = self._upload_file_to_blob(final_video_path, blob_path)

                        if not upload_success:
                            logger.warning(f"Failed to upload file: {file_name}")
                            continue

                        uploaded_file_size_bytes = os.path.getsize(final_video_path)
                        uploaded_file_size_gb = uploaded_file_size_bytes / (1024 ** 3)
                        uploaded_file_name = os.path.basename(final_video_path)

                        sas_token = self._generate_sas_token(blob_path)
                        blob_url = self._get_blob_url_with_sas(blob_path, sas_token)

                        report_data.append({
                            'tar_file_name': tar_filename,
                            'individual_file_name': uploaded_file_name,
                            'scenarioID': scenario_id,
                            'isDupScn': is_dup_scn,
                            'individual_file_full_path': file_info['relative_path'],
                            'individual_file_type': file_extension,
                            'is_dual_fisheye': 'NO',
                            'is_erp_version': 'NO',
                            'is_corrupted': 'YES' if is_corrupted else 'NO',
                            'corruption_error': corruption_error,
                            'tar_size_gb': f"{tar_size_gb:.4f}",
                            'individual_file_size_gb': f"{uploaded_file_size_gb:.6f}",
                            'sas_token': sas_token,
                            'Video_preview_url': blob_url,
                            'blob_path': blob_path,
                            'transfer_date': transfer_date
                        })

                    # If ERP video was created, upload it too
                    if erp_created and erp_video_path and os.path.exists(erp_video_path):
                        erp_file_size_bytes = os.path.getsize(erp_video_path)
                        erp_file_size_gb = erp_file_size_bytes / (1024 ** 3)
                        erp_file_name = os.path.basename(erp_video_path)

                        erp_relative_path = os.path.join(
                            os.path.dirname(file_info['relative_path']),
                            erp_file_name
                        ).replace('\\', '/')

                        erp_blob_path = os.path.join(
                            self.config.untar_blob_prefix,
                            tar_name,
                            erp_relative_path
                        ).replace('\\', '/')

                        logger.info(f"  📤 Uploading ERP video: {erp_file_name} ({erp_file_size_gb:.4f} GB)")

                        erp_upload_success = self._upload_file_to_blob(erp_video_path, erp_blob_path)

                        if erp_upload_success:
                            erp_sas_token = self._generate_sas_token(erp_blob_path)
                            erp_blob_url = self._get_blob_url_with_sas(erp_blob_path, erp_sas_token)

                            report_data.append({
                                'tar_file_name': tar_filename,
                                'individual_file_name': erp_file_name,
                                'scenarioID': scenario_id,
                                'isDupScn': is_dup_scn,
                                'individual_file_full_path': erp_relative_path,
                                'individual_file_type': '.mp4',
                                'is_dual_fisheye': 'NO',
                                'is_erp_version': 'YES',
                                'is_corrupted': 'YES' if is_corrupted else 'NO',
                                'corruption_error': corruption_error,
                                'tar_size_gb': f"{tar_size_gb:.4f}",
                                'individual_file_size_gb': f"{erp_file_size_gb:.6f}",
                                'sas_token': erp_sas_token,
                                'Video_preview_url': erp_blob_url,
                                'blob_path': erp_blob_path,
                                'transfer_date': transfer_date
                            })

                            logger.info(f"  ✅ ERP video uploaded successfully: {erp_file_name}")
                        else:
                            logger.warning(f"  ⚠️  Failed to upload ERP video: {erp_file_name}")

                        # Clean up local ERP file after upload
                        try:
                            os.remove(erp_video_path)
                            logger.debug(f"  🗑️  Removed local ERP file: {erp_video_path}")
                        except Exception as e:
                            logger.debug(f"Could not remove ERP file {erp_video_path}: {e}")

                else:
                    # Non-video file - upload directly
                    logger.info(f"  [{i}/{len(all_files)}] Uploading: {file_info['relative_path']} ({file_size_gb:.4f} GB)")

                    upload_success = self._upload_file_to_blob(file_info['path'], blob_path)

                    if not upload_success:
                        logger.warning(f"Failed to upload file: {file_name}")
                        continue

                    sas_token = self._generate_sas_token(blob_path)
                    blob_url = self._get_blob_url_with_sas(blob_path, sas_token)

                    report_data.append({
                        'tar_file_name': tar_filename,
                        'individual_file_name': file_name,
                        'scenarioID': scenario_id,
                        'isDupScn': is_dup_scn,
                        'individual_file_full_path': file_info['relative_path'],
                        'individual_file_type': file_extension,
                        'is_dual_fisheye': 'NO',
                        'is_erp_version': 'NO',
                        'is_corrupted': 'NO',
                        'corruption_error': '',
                        'tar_size_gb': f"{tar_size_gb:.4f}",
                        'individual_file_size_gb': f"{file_size_gb:.6f}",
                        'sas_token': sas_token,
                        'Video_preview_url': blob_url,
                        'blob_path': blob_path,
                        'transfer_date': transfer_date
                    })

            logger.info(f"✅ Successfully processed {len(report_data)} files from TAR: {tar_filename}")
            upload_successful = True  # Mark as successful for cleanup decision

        except Exception as e:
            logger.error(f"Error processing TAR file {tar_filename}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            upload_successful = False

        finally:
            # Step 5: Cleanup based on config settings
            should_cleanup = self.config.cleanup_enabled and (upload_successful or self.config.cleanup_on_failure)

            if should_cleanup:
                logger.info("🧹 Cleaning up temporary files (cleanup enabled)...")

                # Remove extracted directory
                if self.config.delete_extraction_after_upload:
                    if extraction_path and os.path.exists(extraction_path):
                        try:
                            shutil.rmtree(extraction_path)
                            logger.info(f"✓ Removed extraction directory: {extraction_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove extraction directory {extraction_path}: {e}")

                # Remove downloaded TAR file
                if self.config.delete_tar_after_upload:
                    if local_tar_path and os.path.exists(local_tar_path):
                        try:
                            os.remove(local_tar_path)
                            logger.info(f"✓ Removed downloaded TAR: {local_tar_path}")
                        except Exception as e:
                            logger.warning(f"Failed to remove TAR file {local_tar_path}: {e}")
            else:
                if not self.config.cleanup_enabled:
                    logger.info("ℹ️  Cleanup disabled - keeping temporary files")
                elif not upload_successful:
                    logger.info("ℹ️  Upload failed and cleanup_on_failure=false - keeping files for debugging")

        return report_data

    def run_inspection(self, dry_run: bool = False) -> bool:
        """
        Main method to execute the TAR inspection workflow.

        Args:
            dry_run: If True, show what would be inspected without actually processing

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 100)
        logger.info("Starting TAR Random Inspection Workflow")
        logger.info(f"Input Source: {self.config.input_source.upper()}")
        logger.info(f"Execution Mode: {self.config.execution_mode.upper()}")
        logger.info("=" * 100)

        try:
            # Load inspection state
            inspection_state = self._load_inspection_state()
            already_inspected = set(inspection_state.keys())

            # Handle CSV mode separately (doesn't need to fetch all files first)
            if self.config.execution_mode == 'csv':
                logger.info(f"Using CSV MODE (random {self.config.csv_sample_count} files from comparison CSV)")
                logger.info(f"CSV Input: {self.config.csv_input_file}")
                selected_files = self._select_from_csv(already_inspected)
            else:
                # Get all TAR files based on input source for random/list modes
                if self.config.input_source == 'blob':
                    all_tar_files = self._get_blob_tar_files()
                    source_name = f"Azure Blob ({self.config.source_container_name}/{self.config.source_prefix})"
                else:
                    all_tar_files = self._get_sharepoint_tar_files()
                    source_name = f"SharePoint ({self.config.sharepoint_tar_folder_path})"

                if not all_tar_files:
                    logger.warning(f"No TAR files found in {source_name}")
                    return True

                # Select files based on execution mode
                if self.config.execution_mode == 'random':
                    logger.info(f"Using RANDOM MODE (1 per {self.config.sampling_ratio} files)")
                    selected_files = self._select_random_samples(all_tar_files, already_inspected)
                elif self.config.execution_mode == 'list':
                    logger.info(f"Using LIST MODE ({len(self.config.tar_file_list)} files specified)")
                    selected_files = self._select_from_list(all_tar_files, already_inspected)
                else:
                    logger.error(f"Invalid execution mode: {self.config.execution_mode}")
                    return False

            if not selected_files:
                logger.info("No files to inspect")
                return True

            logger.info(f"\nSelected {len(selected_files)} TAR files for inspection:")
            for i, file_info in enumerate(selected_files, 1):
                size_gb = file_info.get('size', 0) / (1024 ** 3)
                logger.info(f"  {i}. {file_info['name']} ({size_gb:.2f} GB)")

            if dry_run:
                logger.info("\nDRY RUN MODE - No actual processing")
                return True

            # Process each selected TAR file
            all_report_data = []
            successful_inspections = []
            failed_inspections = []

            for i, tar_file_info in enumerate(selected_files, 1):
                tar_filename = tar_file_info['name']

                logger.info(f"\n{'#'*100}")
                logger.info(f"TAR FILE {i}/{len(selected_files)}: {tar_filename}")
                logger.info(f"{'#'*100}")

                report_data = self._process_tar_file(tar_file_info)

                if report_data:
                    all_report_data.extend(report_data)
                    successful_inspections.append(tar_filename)

                    # Update inspection state
                    inspection_state[tar_filename] = {
                        'inspected_at': datetime.now().isoformat(),
                        'num_files_extracted': len(report_data),
                        'tar_size': tar_file_info.get('size', 0)
                    }

                    # Save state after each TAR (in case of interruption)
                    self._save_inspection_state(inspection_state)
                else:
                    failed_inspections.append(tar_filename)

                # Small delay between TAR files
                time.sleep(2)

            # Create CSV report if we have data
            if all_report_data:
                # Use report_date_prefix from source_prefix if available, otherwise use today's date
                if self.config.report_date_prefix:
                    date_part = self.config.report_date_prefix
                    time_part = datetime.now().strftime('%H%M%S')
                    timestamp = f"{date_part}_{time_part}"
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_filename = f"tar_inspection_report_{timestamp}.csv"
                report_path = os.path.join(self.config.csv_report_dir, report_filename)

                self._create_csv_report(all_report_data, report_path)

                logger.info(f"\n{'='*100}")
                logger.info(f"📊 CSV Report Generated: {report_path}")
                logger.info(f"{'='*100}")

            # Summary
            logger.info("\n" + "="*100)
            logger.info("INSPECTION SUMMARY")
            logger.info("="*100)
            logger.info(f"✅ Successful inspections: {len(successful_inspections)}")
            logger.info(f"❌ Failed inspections: {len(failed_inspections)}")
            logger.info(f"📁 Total files processed: {len(all_report_data)}")

            # Count dual-fisheye videos and ERP conversions
            dual_fisheye_count = sum(1 for item in all_report_data if item.get('is_dual_fisheye') == 'YES')
            erp_version_count = sum(1 for item in all_report_data if item.get('is_erp_version') == 'YES')

            if dual_fisheye_count > 0:
                logger.info(f"🎥 Dual-fisheye videos found: {dual_fisheye_count}")
            if erp_version_count > 0:
                logger.info(f"🔄 ERP videos created: {erp_version_count}")

            if failed_inspections:
                logger.info("\nFailed TAR files:")
                for filename in failed_inspections:
                    logger.info(f"  - {filename}")

            logger.info("="*100)

            return len(failed_inspections) == 0

        except Exception as e:
            logger.error(f"Inspection workflow error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


def _cleanup_ray():
    """Cleanup function to ensure Ray is shutdown properly."""
    if RAY_AVAILABLE:
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray cleanup: shutdown complete")
        except Exception as e:
            logger.warning(f"Ray cleanup error: {e}")


def main():
    """Main function to run the TAR inspection workflow."""
    import argparse
    import signal

    # Register cleanup handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.warning(f"\nReceived signal {signum}, cleaning up...")
        _cleanup_ray()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description='Random TAR inspection: download, extract, upload to blob, generate SAS tokens, create CSV report'
    )
    parser.add_argument('--config', '-c', default='config/tar_inspection_config.yaml',
                       help='Path to inspection configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be inspected without actually processing')

    args = parser.parse_args()

    try:
        # Create and run inspection manager
        inspection_manager = TarInspectionManager(config_path=args.config)
        success = inspection_manager.run_inspection(dry_run=args.dry_run)

        if success:
            logger.info("\n✅ TAR inspection workflow completed successfully!")
            sys.exit(0)
        else:
            logger.error("\n❌ TAR inspection workflow completed with errors!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    finally:
        # Always cleanup Ray to prevent zombie processes
        _cleanup_ray()


if __name__ == "__main__":
    main()
