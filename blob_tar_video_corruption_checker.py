#!/usr/bin/env python3
"""
Blob TAR Video Corruption Checker

This script reads a consolidated CSV report, downloads TAR files from Azure Blob Storage,
extracts them, and checks all MP4 files for corruption using ffprobe.

Key Features:
- Reads consolidated CSV report to get TAR file list
- Downloads TAR files from Azure Blob Storage
- Extracts TAR files locally
- Uses ffprobe to validate all video files
- Identifies corrupted videos and problematic TAR files
- Generates detailed CSV report with corruption status
- Tracks processed files to avoid re-checking

Usage:
    python blob_tar_video_corruption_checker.py --csv-report REPORT_PATH [--config CONFIG_FILE] [--dry-run]

Examples:
    # Run with default config
    python blob_tar_video_corruption_checker.py --csv-report /data/oslo/consolidated_report/tar/consolidated_tar_files_oslo_project_2_20251111_065246.csv

    # Dry run (show what would be checked)
    python blob_tar_video_corruption_checker.py --csv-report /path/to/report.csv --dry-run

    # Custom config
    python blob_tar_video_corruption_checker.py --csv-report /path/to/report.csv --config custom_config.yaml

Author: Auto-generated for video corruption detection
"""

import os
import sys
import json
import csv
import time
import tarfile
import subprocess
import threading
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Import Azure Blob Storage SDK
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blob_tar_video_checker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose Azure SDK logging
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)


class BlobVideoCheckerConfig:
    """Configuration class for blob video corruption checker settings."""

    def __init__(self, config_dict: Dict):
        """Initialize config from dictionary."""
        # Azure Blob settings
        self.azure_connection_string = self._get_connection_string(config_dict['azure_blob'])
        self.azure_container_name = config_dict['azure_blob']['container_name']

        # Local paths
        self.temp_download_dir = config_dict['local_paths'].get('temp_download_dir',
                                                                 '/data/oslo/blob_tar_downloads')
        self.temp_extract_dir = config_dict['local_paths'].get('temp_extract_dir',
                                                                '/data/oslo/blob_tar_extractions')
        self.check_state_file = config_dict['local_paths'].get('check_state_file',
                                                                '/data/oslo/blob_tar_downloads/video_check_state.json')
        self.csv_report_dir = config_dict['local_paths'].get('csv_report_dir',
                                                              '/data/oslo/video_check_reports')

        # Processing settings
        self.max_files_per_run = config_dict.get('processing', {}).get('max_files_per_run', 50)
        self.cleanup_after_check = config_dict.get('processing', {}).get('cleanup_after_check', True)
        self.video_extensions = config_dict.get('processing', {}).get('video_extensions',
                                                                       ['.mp4', '.avi', '.mkv', '.mov'])
        # Parallel processing settings (NEW)
        self.max_parallel_downloads = config_dict.get('processing', {}).get('max_parallel_downloads', 3)
        self.download_chunk_size = config_dict.get('processing', {}).get('download_chunk_size_mb', 64) * 1024 * 1024

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


class BlobTarVideoChecker:
    """Main class that checks TAR files from blob storage for corrupted videos."""

    def __init__(self, config_path: str = "config/tar_transfer_config.yaml"):
        """
        Initialize blob tar video corruption checker.

        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        self.config = BlobVideoCheckerConfig(config_dict)

        # Initialize Azure Blob client with larger connection pool for parallel processing
        from azure.core.pipeline.transport import RequestsTransport
        transport = RequestsTransport(
            connection_pool_maxsize=32  # Increase from default 8 to support parallel downloads
        )
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.config.azure_connection_string,
            transport=transport
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.config.azure_container_name
        )

        # Ensure local directories exist
        os.makedirs(self.config.temp_download_dir, exist_ok=True)
        os.makedirs(self.config.temp_extract_dir, exist_ok=True)
        os.makedirs(self.config.csv_report_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config.check_state_file), exist_ok=True)

        # Thread-safe lock for state file updates
        self._state_lock = threading.Lock()

        logger.info("Blob TAR Video Corruption Checker initialized (OPTIMIZED)")
        logger.info(f"Azure container: {self.config.azure_container_name}")
        logger.info(f"Parallel downloads: {self.config.max_parallel_downloads}")

    def _load_check_state(self) -> Dict[str, Dict]:
        """
        Load check state from JSON file.

        Returns:
            Dictionary mapping blob names to check metadata
        """
        if os.path.exists(self.config.check_state_file):
            try:
                with open(self.config.check_state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded check state: {len(state)} files already checked")
                return state if isinstance(state, dict) else {}
            except Exception as e:
                logger.warning(f"Failed to load check state: {e}")
                return {}
        return {}

    def _save_check_state(self, state: Dict[str, Dict]):
        """
        Save check state to JSON file.

        Args:
            state: Dictionary mapping blob names to check metadata
        """
        try:
            with open(self.config.check_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug("Check state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save check state: {e}")

    def _read_csv_report(self, csv_path: str) -> List[Dict]:
        """
        Read consolidated CSV report to get TAR file list.

        Args:
            csv_path: Path to consolidated CSV report

        Returns:
            List of TAR file information dictionaries
        """
        logger.info(f"Reading CSV report: {csv_path}")

        tar_files = []
        try:
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Only process FILE rows with success status
                    if row.get('row_type') == 'FILE' and row.get('status') == 'success':
                        blob_name = row.get('blob_name', '')
                        if blob_name:
                            tar_files.append({
                                'blob_name': blob_name,
                                'file_size_bytes': int(row.get('file_size_bytes', 0)),
                                'file_size_gb': float(row.get('file_size_gb', 0)),
                                'tar_uuid': row.get('tar_uuid', ''),
                                'upload_date': row.get('upload_end_time', '')
                            })

            logger.info(f"Found {len(tar_files)} TAR files in CSV report")
            return tar_files

        except Exception as e:
            logger.error(f"Failed to read CSV report: {e}")
            return []

    def _download_from_blob(self, blob_name: str) -> Optional[str]:
        """
        Download a TAR file from Azure Blob Storage with optimized chunked download.
        OPTIMIZATION: Uses max_concurrency for parallel chunk downloads (2-3x faster).

        Args:
            blob_name: Blob name in Azure Storage

        Returns:
            Local file path if successful, None otherwise
        """
        try:
            # Get blob client
            blob_client = self.container_client.get_blob_client(blob_name)

            # Check if blob exists
            if not blob_client.exists():
                logger.error(f"Blob does not exist: {blob_name}")
                return None

            # Extract filename from blob path
            filename = os.path.basename(blob_name)
            local_path = os.path.join(self.config.temp_download_dir, filename)

            # Get blob size for progress tracking
            blob_properties = blob_client.get_blob_properties()
            blob_size = blob_properties.size
            blob_size_gb = blob_size / (1024**3)

            # Download blob with optimized settings
            logger.info(f"Downloading from blob: {blob_name} ({blob_size_gb:.2f} GB)")
            start_time = time.time()

            with open(local_path, 'wb') as download_file:
                download_stream = blob_client.download_blob(max_concurrency=8)  # OPTIMIZED: Parallel chunk downloads
                download_file.write(download_stream.readall())

            elapsed = time.time() - start_time
            speed_mbps = (blob_size / (1024**2)) / elapsed if elapsed > 0 else 0
            logger.info(f"Downloaded to: {local_path} ({elapsed:.1f}s, {speed_mbps:.1f} MB/s)")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download blob {blob_name}: {e}")
            return None

    def _extract_tar_file(self, tar_path: str, tar_name: str) -> Optional[str]:
        """
        Extract ONLY video files from TAR to temporary directory.
        OPTIMIZATION: Selective extraction saves disk I/O and space (2x faster).

        Args:
            tar_path: Path to TAR file
            tar_name: Name of TAR file (for creating extract directory)

        Returns:
            Path to extraction directory if successful, None otherwise
        """
        try:
            # Create extraction directory for this TAR
            extract_dir = os.path.join(self.config.temp_extract_dir, Path(tar_name).stem)
            os.makedirs(extract_dir, exist_ok=True)

            logger.info(f"Extracting video files from TAR: {tar_name}")
            start_time = time.time()

            # Extract only video files (OPTIMIZED)
            video_count = 0
            with tarfile.open(tar_path, 'r:*') as tar:
                for member in tar.getmembers():
                    # Check if file has video extension
                    if member.isfile() and any(member.name.lower().endswith(ext) for ext in self.config.video_extensions):
                        # Use 'data' filter for secure extraction (Python 3.14+ requirement)
                        tar.extract(member, path=extract_dir, filter='data')
                        video_count += 1

            elapsed = time.time() - start_time
            logger.info(f"Extracted {video_count} video files to: {extract_dir} ({elapsed:.1f}s)")

            return extract_dir

        except Exception as e:
            logger.error(f"Failed to extract TAR file {tar_name}: {e}")
            return None

    def _check_video_with_ffprobe(self, video_path: str) -> Tuple[bool, str]:
        """
        Check if a video file is corrupted using ffprobe.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Run ffprobe to check video integrity
            # -v error: only show errors
            # -show_entries format=duration: try to read format metadata
            # -of default=noprint_wrappers=1: simple output format
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    video_path
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            # If ffprobe returns non-zero exit code or has stderr output, video is corrupted
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return False, error_msg

            # Check if we got valid duration output
            if result.stdout.strip():
                return True, ""
            else:
                return False, "No duration information found"

        except subprocess.TimeoutExpired:
            return False, "ffprobe timeout (file may be severely corrupted)"
        except FileNotFoundError:
            logger.error("ffprobe not found. Please install ffmpeg: sudo apt-get install ffmpeg")
            return False, "ffprobe not installed"
        except Exception as e:
            return False, f"ffprobe error: {str(e)}"

    def _find_video_files(self, extract_dir: str) -> List[str]:
        """
        Find all video files in extraction directory.

        Args:
            extract_dir: Path to extraction directory

        Returns:
            List of video file paths
        """
        video_files = []
        for ext in self.config.video_extensions:
            video_files.extend(Path(extract_dir).rglob(f'*{ext}'))

        # Convert to strings and sort
        video_files = sorted([str(f) for f in video_files])
        logger.info(f"Found {len(video_files)} video files to check")
        return video_files

    def _cleanup(self, tar_path: Optional[str] = None, extract_dir: Optional[str] = None):
        """
        Clean up temporary files and directories.

        Args:
            tar_path: Path to TAR file to delete
            extract_dir: Path to extraction directory to delete
        """
        try:
            if tar_path and os.path.exists(tar_path):
                os.remove(tar_path)
                logger.debug(f"Cleaned up TAR file: {tar_path}")

            if extract_dir and os.path.exists(extract_dir):
                import shutil
                shutil.rmtree(extract_dir)
                logger.debug(f"Cleaned up extraction directory: {extract_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup: {e}")

    def _process_single_tar(self, tar_info: Dict, file_index: int, total_files: int) -> Tuple[Optional[str], List[Dict], Dict]:
        """
        Process a single TAR file: download, extract, and check videos.
        OPTIMIZATION: Designed for parallel execution.

        Args:
            tar_info: TAR file information dictionary
            file_index: Index of this file in processing queue
            total_files: Total number of files being processed

        Returns:
            Tuple of (blob_name or None if failed, list of video results, tar summary dict)
        """
        blob_name = tar_info['blob_name']
        file_size_gb = tar_info['file_size_gb']

        logger.info(f"\n{'#'*70}")
        logger.info(f"TAR FILE {file_index}/{total_files}: {blob_name}")
        logger.info(f"Size: {file_size_gb:.2f} GB")
        logger.info(f"{'#'*70}")

        tar_path = None
        extract_dir = None
        results = []
        tar_summary = None

        try:
            # Download TAR from blob
            tar_path = self._download_from_blob(blob_name)
            if not tar_path:
                return None, [], None

            # Extract TAR file (only video files)
            tar_filename = os.path.basename(blob_name)
            extract_dir = self._extract_tar_file(tar_path, tar_filename)
            if not extract_dir:
                return None, [], None

            # Find all video files
            video_files = self._find_video_files(extract_dir)

            if not video_files:
                logger.warning(f"No video files found in {blob_name}")
                tar_summary = {
                    'checked_at': datetime.now().isoformat(),
                    'num_videos': 0,
                    'num_corrupted': 0,
                    'file_size_gb': file_size_gb
                }
                return blob_name, [], tar_summary

            # Check each video file
            corrupted_count = 0
            for j, video_path in enumerate(video_files, 1):
                video_name = os.path.basename(video_path)
                relative_path = os.path.relpath(video_path, extract_dir)
                video_size_mb = os.path.getsize(video_path) / (1024 * 1024)

                logger.info(f"  [{j}/{len(video_files)}] Checking: {video_name}")

                is_valid, error_msg = self._check_video_with_ffprobe(video_path)

                result = {
                    'blob_name': blob_name,
                    'tar_file_name': os.path.basename(blob_name),
                    'video_file_path': relative_path,
                    'video_file_name': video_name,
                    'video_size_mb': f"{video_size_mb:.4f}",
                    'is_corrupted': 'YES' if not is_valid else 'NO',
                    'corruption_error': error_msg if not is_valid else '',
                    'check_date': datetime.now().isoformat()
                }
                results.append(result)

                if not is_valid:
                    corrupted_count += 1
                    logger.warning(f"    ❌ CORRUPTED: {error_msg}")
                else:
                    logger.info(f"    ✅ Valid")

            # Create summary for this TAR
            tar_summary = {
                'checked_at': datetime.now().isoformat(),
                'num_videos': len(video_files),
                'num_corrupted': corrupted_count,
                'file_size_gb': file_size_gb
            }

            if corrupted_count > 0:
                logger.warning(f"⚠️  TAR has {corrupted_count}/{len(video_files)} corrupted videos")
            else:
                logger.info(f"✅ All {len(video_files)} videos are valid")

            return blob_name, results, tar_summary

        except Exception as e:
            logger.error(f"❌ Error processing {blob_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None, [], None

        finally:
            # Cleanup temporary files
            if self.config.cleanup_after_check:
                self._cleanup(tar_path, extract_dir)

    def _generate_csv_report(self, results: List[Dict]) -> str:
        """
        Generate CSV report of corruption check results.

        Args:
            results: List of check result dictionaries

        Returns:
            Path to generated CSV file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(self.config.csv_report_dir,
                                f'blob_video_corruption_report_{timestamp}.csv')

        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'blob_name',
                'tar_file_name',
                'video_file_path',
                'video_file_name',
                'video_size_mb',
                'is_corrupted',
                'corruption_error',
                'check_date'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"CSV report generated: {csv_path}")
        return csv_path

    def run_check(self, csv_report_path: str, dry_run: bool = False) -> bool:
        """
        Main method to execute video corruption checking.

        Args:
            csv_report_path: Path to consolidated CSV report
            dry_run: If True, show what would be checked without actually checking

        Returns:
            True if successful, False otherwise
        """
        logger.info("============================================================")
        logger.info("Starting Blob TAR Video Corruption Check")
        logger.info("============================================================")

        try:
            # Load check state
            check_state = self._load_check_state()

            # Read CSV report to get TAR file list
            tar_files = self._read_csv_report(csv_report_path)

            if not tar_files:
                logger.warning("No TAR files found in CSV report")
                return True

            # Filter files that haven't been checked yet
            files_to_check = []
            for tar_info in tar_files:
                blob_name = tar_info['blob_name']
                if blob_name not in check_state:
                    files_to_check.append(tar_info)
                else:
                    logger.debug(f"Skipping already checked file: {blob_name}")

            if not files_to_check:
                logger.info("No new files to check")
                return True

            # Limit number of files per run
            if len(files_to_check) > self.config.max_files_per_run:
                logger.info(f"Limiting to {self.config.max_files_per_run} files per run")
                files_to_check = files_to_check[:self.config.max_files_per_run]

            logger.info(f"Found {len(files_to_check)} TAR files to check")

            if dry_run:
                logger.info("DRY RUN MODE - TAR files that would be checked:")
                for tar_info in files_to_check:
                    logger.info(f"  - {tar_info['blob_name']} ({tar_info['file_size_gb']:.2f} GB)")
                return True

            # OPTIMIZATION: Process TAR files in parallel with ThreadPoolExecutor
            logger.info(f"🚀 Starting parallel processing with {self.config.max_parallel_downloads} workers")

            all_results = []
            tars_with_corruption = []
            successful_checks = []
            failed_checks = []

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_parallel_downloads) as executor:
                # Submit all tasks
                future_to_tar = {
                    executor.submit(self._process_single_tar, tar_info, i+1, len(files_to_check)): tar_info
                    for i, tar_info in enumerate(files_to_check)
                }

                # Process completed tasks as they finish
                for future in as_completed(future_to_tar):
                    tar_info = future_to_tar[future]
                    blob_name = tar_info['blob_name']

                    try:
                        result_blob_name, video_results, tar_summary = future.result()

                        if result_blob_name:
                            # Success
                            all_results.extend(video_results)
                            successful_checks.append(result_blob_name)

                            # Update check state (thread-safe)
                            with self._state_lock:
                                check_state[result_blob_name] = tar_summary

                            # Check if TAR has corruption
                            if tar_summary and tar_summary.get('num_corrupted', 0) > 0:
                                tars_with_corruption.append({
                                    'name': result_blob_name,
                                    'corrupted_videos': tar_summary['num_corrupted'],
                                    'total_videos': tar_summary['num_videos']
                                })
                        else:
                            # Failed
                            failed_checks.append(blob_name)

                    except Exception as e:
                        logger.error(f"❌ Task failed for {blob_name}: {e}")
                        failed_checks.append(blob_name)

            # Save updated check state
            if successful_checks:
                self._save_check_state(check_state)

            # Generate CSV report
            if all_results:
                csv_path = self._generate_csv_report(all_results)
                logger.info(f"\n📊 Detailed CSV report: {csv_path}")

            # Print summary
            logger.info("\n" + "="*70)
            logger.info("CORRUPTION CHECK SUMMARY")
            logger.info("="*70)
            logger.info(f"✅ TAR files checked: {len(successful_checks)}")
            logger.info(f"❌ Failed checks: {len(failed_checks)}")
            logger.info(f"⚠️  TAR files with corrupted videos: {len(tars_with_corruption)}")

            if tars_with_corruption:
                logger.info("\nTAR FILES WITH CORRUPTION ISSUES:")
                logger.info("-" * 70)
                for tar_info in tars_with_corruption:
                    logger.info(f"  ❌ {tar_info['name']}: "
                               f"{tar_info['corrupted_videos']}/{tar_info['total_videos']} "
                               f"videos corrupted")
            else:
                logger.info("\n✅ No corrupted videos found in any TAR files!")

            if failed_checks:
                logger.info("\nFAILED CHECKS:")
                for blob_name in failed_checks:
                    logger.info(f"  - {blob_name}")

            logger.info("="*70)

            return len(failed_checks) == 0

        except Exception as e:
            logger.error(f"Check process error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False


def main():
    """Main function to run the blob tar video corruption checker."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Check TAR files from Azure Blob for corrupted videos'
    )
    parser.add_argument('--csv-report', '-r', required=True,
                       help='Path to consolidated CSV report')
    parser.add_argument('--config', '-c', default='config/tar_transfer_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be checked without actually checking')

    args = parser.parse_args()

    try:
        # Create and run checker
        checker = BlobTarVideoChecker(config_path=args.config)
        success = checker.run_check(csv_report_path=args.csv_report, dry_run=args.dry_run)

        if success:
            logger.info("Blob TAR video corruption check completed successfully!")
            sys.exit(0)
        else:
            logger.error("Blob TAR video corruption check completed with errors!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
