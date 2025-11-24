#!/usr/bin/env python3
"""
SharePoint TAR Random Inspection and Azure Blob Upload Script

This script randomly samples TAR files from SharePoint (1 per 10 files), untars them,
uploads contents to Azure Blob Storage, generates SAS tokens, and creates a detailed CSV report.

Workflow:
1. Lists all TAR files from SharePoint /Uploads folder
2. Randomly selects 1 file per every 10 files
3. Downloads selected TAR files
4. Untars/extracts files locally
5. Uploads all extracted files to Azure Blob: instavideo/untar_folder_oslo2/
6. Generates SAS tokens for each uploaded file (for team inspection)
7. Creates CSV report with: tar_name, file_name, file_full_path, file_type, tar_size_gb,
   file_size_gb, sas_token, blob_url, blob_path, transfer_date

Usage:
    python sharepoint_tar_inspector.py [--config CONFIG_FILE] [--dry-run]

Examples:
    # Run with default config
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
        # SharePoint settings
        self.sharepoint_tar_folder_path = config_dict['sharepoint']['tar_source_folder_path']

        # Azure Blob settings
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
        if self.execution_mode not in ['random', 'list']:
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}. Must be 'random' or 'list'")
        self.tar_file_list = inspection_config.get('tar_file_list', [])
        if self.execution_mode == 'list' and not self.tar_file_list:
            raise ValueError("execution_mode is 'list' but tar_file_list is empty")

        # Video downscaling settings
        self.enable_video_downscaling = inspection_config.get('enable_video_downscaling', True)
        self.downscale_target_height = inspection_config.get('downscale_target_height', 480)
        self.downscale_quality = inspection_config.get('downscale_quality', 'medium') 

        # Local paths
        self.temp_download_dir = config_dict['local_paths']['temp_download_dir']
        self.temp_extract_dir = inspection_config.get('temp_extract_dir', '/data/oslo/tar_extractions')
        self.csv_report_dir = inspection_config.get('csv_report_dir', '/data/oslo/inspection_reports')
        self.inspection_state_file = inspection_config.get('inspection_state_file',
                                                           '/data/oslo/tar_temp_downloads/tar_inspection_state.json')

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
        os.makedirs(os.path.dirname(self.config.inspection_state_file), exist_ok=True)

        logger.info(f"TAR Inspection Manager initialized")
        logger.info(f"SharePoint folder: {self.config.sharepoint_tar_folder_path}")
        logger.info(f"Azure container: {self.config.azure_container_name}")
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

    def _unwarp_dual_fisheye_to_erp(self, input_video_path: str, output_video_path: str) -> bool:
        """
        Convert dual-fisheye video to ERP (Equirectangular) view using ffmpeg.

        Args:
            input_video_path: Path to dual-fisheye video
            output_video_path: Path for output ERP video

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            logger.info(f"Converting dual-fisheye to ERP: {os.path.basename(input_video_path)}")

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
            ERP_W, ERP_H = 5760, 2880

            logger.info(f"Target ERP resolution: {ERP_W}x{ERP_H}")
            logger.info(f"Lens FOV: {LENS_FOV_DEG} degrees")

            # Build ffmpeg command based on number of video streams
            if num_video_streams >= 2:
                # Dual-lens: hstack two streams and convert to ERP, then rotate 180 degrees
                logger.info("🎬 Detected 2 video streams - converting dual-lens to ERP with 180° rotation")
                filter_complex = (
                    f"[0:v:0][0:v:1]hstack=inputs=2[dual];"
                    f"[dual]v360=input=dfisheye:output=equirect:"
                    f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}[erp];"
                    f"[erp]hflip,vflip"
                )
                logger.debug(f"Filter complex: {filter_complex}")
                logger.info("  ↻ Applying 180° rotation (hflip + vflip) to correct orientation")

                # FFmpeg command for dual-stream (requires -filter_complex)
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output
                    '-i', input_video_path,
                    '-filter_complex', filter_complex,
                    '-c:v', 'libx264',
                    '-crf', '23',
                    '-preset', 'medium',
                    '-c:a', 'copy',  # Copy audio stream
                    '-movflags', '+faststart',
                    output_video_path
                ]
            else:
                # Single lens fisheye: convert directly to ERP
                logger.info("🎬 Detected 1 video stream - converting fisheye to ERP")
                vf_filter = (
                    f"v360=input=fisheye:output=equirect:"
                    f"ih_fov={LENS_FOV_DEG}:iv_fov={LENS_FOV_DEG}:w={ERP_W}:h={ERP_H}"
                )
                logger.debug(f"Video filter: {vf_filter}")

                # FFmpeg command for single-stream (can use -vf)
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output
                    '-i', input_video_path,
                    '-vf', vf_filter,
                    '-c:v', 'libx264',
                    '-crf', '23',
                    '-preset', 'medium',
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
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                return False

            if os.path.exists(output_video_path):
                output_size = os.path.getsize(output_video_path)
                logger.info(f"✅ ERP conversion successful: {os.path.basename(output_video_path)} ({output_size / (1024**2):.2f} MB)")
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
                'individual_file_full_path',
                'individual_file_type',
                'is_dual_fisheye',
                'is_erp_version',
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
                writer.writerows(sorted_report_data)

            logger.info(f"CSV report created: {report_path}")
            logger.info(f"Report contains {len(sorted_report_data)} file entries (videos first, then other files)")

        except Exception as e:
            logger.error(f"Failed to create CSV report: {e}")

    def _process_tar_file(self, tar_file_info: Dict) -> List[Dict]:
        """
        Process a single TAR file: download, extract, upload, generate SAS, collect data.

        Args:
            tar_file_info: TAR file info from SharePoint

        Returns:
            List of report data dictionaries for all extracted files
        """
        tar_filename = tar_file_info['name']
        tar_size_bytes = tar_file_info.get('size', 0)
        tar_size_gb = tar_size_bytes / (1024 ** 3)

        report_data = []
        local_tar_path = None
        extraction_path = None

        try:
            # Step 1: Download TAR file from SharePoint
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing TAR file: {tar_filename} ({tar_size_gb:.2f} GB)")
            logger.info(f"{'='*80}")

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

            logger.info(f"Processing {len(extracted_files)} extracted files for upload...")

            # Step 4: Upload each file to blob and generate SAS tokens
            transfer_date = datetime.now().isoformat()

            for i, file_info in enumerate(extracted_files, 1):
                file_name = file_info['name']
                file_size_bytes = file_info['size']
                file_size_gb = file_size_bytes / (1024 ** 3)
                file_extension = os.path.splitext(file_name)[1] or 'no_extension'

                # Construct blob path: untar_folder_oslo2/tar_name/relative_path
                tar_name = tar_filename.rsplit('.tar', 1)[0]
                blob_path = os.path.join(
                    self.config.untar_blob_prefix,
                    tar_name,
                    file_info['relative_path']
                ).replace('\\', '/')  # Ensure forward slashes

                # Check if video is dual-fisheye and process accordingly
                is_dual_fisheye = False
                erp_video_path = None
                erp_created = False
                final_video_path = file_info['path']  # Path to upload (might be downscaled)
                downscaled_original_path = None
                downscaled_erp_path = None

                if file_extension.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.insv']:
                    # STEP 1: Check if dual-fisheye
                    is_dual_fisheye = self._is_dual_fisheye_video(file_info['path'])

                    if is_dual_fisheye:
                        logger.info(f"  🎥 DUAL-FISHEYE detected: {file_name}")

                        # STEP 2: Convert to ERP (high-resolution)
                        file_name_without_ext = os.path.splitext(file_name)[0]
                        erp_file_name = f"{file_name_without_ext}_erpview.mp4"
                        erp_video_path = os.path.join(os.path.dirname(file_info['path']), erp_file_name)

                        logger.info(f"  🔄 Converting to ERP: {erp_file_name}")
                        erp_created = self._unwarp_dual_fisheye_to_erp(file_info['path'], erp_video_path)

                        if not erp_created:
                            logger.warning(f"  ⚠️  Failed to create ERP video for: {file_name}")

                        # STEP 3: Downscale ONLY the ERP video (not the original fisheye)
                        if self.config.enable_video_downscaling:
                            # Downscale ERP video if created
                            if erp_created and erp_video_path:
                                downscaled_erp_path = self._downscale_video_if_needed(erp_video_path)
                                if downscaled_erp_path:
                                    # Replace ERP path with downscaled version
                                    erp_video_path = downscaled_erp_path
                    else:
                        # STEP 2: Normal video - just downscale to 480p
                        if self.config.enable_video_downscaling:
                            downscaled_path = self._downscale_video_if_needed(file_info['path'])
                            if downscaled_path:
                                final_video_path = downscaled_path

                # Skip uploading original dual-fisheye videos (we'll upload ERP version instead)
                if is_dual_fisheye:
                    logger.info(f"  ℹ️  Skipping upload of original dual-fisheye video (will upload ERP version instead)")
                else:
                    # For non-fisheye videos, upload the final_video_path (which is downscaled if enabled)
                    logger.info(f"  [{i}/{len(extracted_files)}] Uploading: {file_info['relative_path']} ({file_size_gb:.4f} GB)")

                    upload_success = self._upload_file_to_blob(final_video_path, blob_path)

                    if not upload_success:
                        logger.warning(f"Failed to upload file: {file_name}")
                        continue

                    # Get actual uploaded file size (downscaled version)
                    uploaded_file_size_bytes = os.path.getsize(final_video_path)
                    uploaded_file_size_gb = uploaded_file_size_bytes / (1024 ** 3)
                    uploaded_file_name = os.path.basename(final_video_path)

                    # Generate SAS token
                    sas_token = self._generate_sas_token(blob_path)

                    # Generate blob URL with SAS
                    blob_url = self._get_blob_url_with_sas(blob_path, sas_token)

                    # Add to report data with downscaled file info
                    report_data.append({
                        'tar_file_name': tar_filename,
                        'individual_file_name': uploaded_file_name,
                        'individual_file_full_path': file_info['relative_path'],
                        'individual_file_type': file_extension,
                        'is_dual_fisheye': 'NO',
                        'is_erp_version': 'NO',
                        'tar_size_gb': f"{tar_size_gb:.4f}",
                        'individual_file_size_gb': f"{uploaded_file_size_gb:.6f}",
                        'sas_token': sas_token,
                        'blob_url': blob_url,
                        'blob_path': blob_path,
                        'transfer_date': transfer_date
                    })

                # If ERP video was created, upload it too
                if erp_created and erp_video_path and os.path.exists(erp_video_path):
                    erp_file_size_bytes = os.path.getsize(erp_video_path)
                    erp_file_size_gb = erp_file_size_bytes / (1024 ** 3)
                    erp_file_name = os.path.basename(erp_video_path)

                    # Construct ERP blob path (same folder as original)
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

                    # Upload ERP video
                    erp_upload_success = self._upload_file_to_blob(erp_video_path, erp_blob_path)

                    if erp_upload_success:
                        # Generate SAS token for ERP
                        erp_sas_token = self._generate_sas_token(erp_blob_path)
                        erp_blob_url = self._get_blob_url_with_sas(erp_blob_path, erp_sas_token)

                        # Add ERP file to report data
                        report_data.append({
                            'tar_file_name': tar_filename,
                            'individual_file_name': erp_file_name,
                            'individual_file_full_path': erp_relative_path,
                            'individual_file_type': '.mp4',
                            'is_dual_fisheye': 'NO',  # ERP is not dual-fisheye anymore
                            'is_erp_version': 'YES',
                            'tar_size_gb': f"{tar_size_gb:.4f}",
                            'individual_file_size_gb': f"{erp_file_size_gb:.6f}",
                            'sas_token': erp_sas_token,
                            'blob_url': erp_blob_url,
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

            logger.info(f"✅ Successfully processed {len(report_data)} files from TAR: {tar_filename}")

        except Exception as e:
            logger.error(f"Error processing TAR file {tar_filename}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        finally:
            # Step 5: Cleanup (ALWAYS runs, even if errors occur)
            logger.info("Cleaning up temporary files...")

            # Remove extracted directory
            if extraction_path and os.path.exists(extraction_path):
                try:
                    shutil.rmtree(extraction_path)
                    logger.info(f"✓ Removed extraction directory: {extraction_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove extraction directory {extraction_path}: {e}")

            # Remove downloaded TAR file
            if local_tar_path and os.path.exists(local_tar_path):
                try:
                    os.remove(local_tar_path)
                    logger.info(f"✓ Removed downloaded TAR: {local_tar_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove TAR file {local_tar_path}: {e}")

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
        logger.info("=" * 100)

        try:
            # Load inspection state
            inspection_state = self._load_inspection_state()
            already_inspected = set(inspection_state.keys())

            # Get all TAR files from SharePoint
            all_tar_files = self._get_sharepoint_tar_files()

            if not all_tar_files:
                logger.warning("No TAR files found in SharePoint folder")
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

            logger.info(f"\nSelected {len(selected_files)} TAR files for random inspection:")
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


def main():
    """Main function to run the TAR inspection workflow."""
    import argparse

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


if __name__ == "__main__":
    main()
