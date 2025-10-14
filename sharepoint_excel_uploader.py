#!/usr/bin/env python3
"""
SharePoint Excel File Uploader Script

This script automatically uploads Excel files from your local batch_3 directory
to SharePoint input folder where the annotation team can access them.

Key Features:
- Uploads new Excel files from csv_files/batch_3 folders to SharePoint
- Two modes: default (skip newest) and total_file (upload all)
- Tracks uploaded files in JSON to avoid duplicates
- Handles authentication via Microsoft Graph API
- Provides detailed logging and progress tracking

Usage:
    python sharepoint_excel_uploader.py [--config CONFIG_FILE] [--dry-run]               # Default mode (skip newest)
    python sharepoint_excel_uploader.py total_file [--config CONFIG_FILE] [--dry-run]   # Total file mode (upload all)

Author: Auto-generated for Excel → SharePoint automation
"""

import os
import sys
import json
import yaml
import glob
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sharepoint_upload.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SharePointConfig:
    """Configuration class for SharePoint connection settings."""
    tenant_id: str
    client_id: str  
    client_secret: str
    site_id: str
    input_folder_path: str
    excel_source_dir: str
    upload_state_file: str
    skip_newest_files: int
    max_files_per_run: int
    timeout_seconds: int
    retry_attempts: int
    retry_delay_seconds: int


class SharePointAuthenticator:
    """Handles Microsoft Graph API authentication for SharePoint access."""
    
    def __init__(self, config: SharePointConfig):
        """
        Initialize authenticator with configuration.
        
        Args:
            config: SharePoint configuration containing credentials
        """
        self.config = config
        self.access_token = None
        self.token_expires_at = None
    
    def get_access_token(self) -> Optional[str]:
        """
        Get a valid access token for Microsoft Graph API.
        Uses OAuth2 Client Credentials Flow for server-to-server authentication.
        
        Returns:
            Valid access token string or None if authentication failed
        """
        # Check if we have a valid cached token
        if (self.access_token and self.token_expires_at and 
            datetime.now() < self.token_expires_at - timedelta(minutes=5)):
            logger.debug("Using cached access token")
            return self.access_token
        
        logger.info("Requesting new access token from Microsoft Graph API")
        
        # OAuth2 Client Credentials Flow endpoint
        token_url = f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token"
        
        # Prepare authentication request
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'scope': 'https://graph.microsoft.com/.default',
            'grant_type': 'client_credentials'
        }
        
        try:
            # Request access token
            response = requests.post(
                token_url,
                headers=headers,
                data=data,
                timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            
            # Parse response
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            expires_in = token_data.get('expires_in', 3600)  # Default 1 hour
            
            # Calculate expiration time
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            logger.info("Successfully obtained access token")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get access token: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid token response format: {e}")
            return None


class SharePointFileUploader:
    """Handles file upload operations to SharePoint via Microsoft Graph API."""
    
    def __init__(self, config: SharePointConfig):
        """
        Initialize uploader with configuration and authenticator.
        
        Args:
            config: SharePoint configuration
        """
        self.config = config
        self.authenticator = SharePointAuthenticator(config)
        
        # Create base API URL for SharePoint site
        self.base_api_url = f"https://graph.microsoft.com/v1.0/sites/{config.site_id}/drive"
    
    def _make_authenticated_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Make an authenticated request to Microsoft Graph API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, etc.)
            url: Full API URL
            **kwargs: Additional arguments passed to requests
            
        Returns:
            Response object or None if all retries failed
        """
        token = self.authenticator.get_access_token()
        if not token:
            logger.error("Cannot make request: no valid access token")
            return None
        
        headers = kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {token}'
        kwargs['headers'] = headers
        
        # Add default timeout
        kwargs.setdefault('timeout', self.config.timeout_seconds)
        
        # Retry logic
        for attempt in range(self.config.retry_attempts):
            try:
                logger.debug(f"Making {method} request to {url} (attempt {attempt + 1})")
                response = requests.request(method, url, **kwargs)
                
                # Check for authentication errors
                if response.status_code == 401:
                    logger.warning("Access token expired, refreshing...")
                    self.authenticator.access_token = None  # Force token refresh
                    token = self.authenticator.get_access_token()
                    if token:
                        headers['Authorization'] = f'Bearer {token}'
                        kwargs['headers'] = headers
                        continue
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    logger.info(f"Retrying in {self.config.retry_delay_seconds} seconds...")
                    time.sleep(self.config.retry_delay_seconds)
                else:
                    logger.error(f"All {self.config.retry_attempts} attempts failed")
        
        return None
    
    def get_folder_id(self, folder_path: str) -> Optional[str]:
        """
        Get SharePoint folder ID by path for use in upload operations.
        
        Args:
            folder_path: SharePoint folder path (e.g., "/Input")
            
        Returns:
            Folder ID string or None if folder not found
        """
        logger.info(f"Looking up folder ID for path: {folder_path}")
        
        # Remove leading/trailing slashes and split path
        clean_path = folder_path.strip('/')
        
        if not clean_path:
            # Root folder
            url = f"{self.base_api_url}/root"
        else:
            # Specific folder path
            url = f"{self.base_api_url}/root:/{clean_path}"
        
        response = self._make_authenticated_request('GET', url)
        if response:
            folder_data = response.json()
            folder_id = folder_data.get('id')
            logger.info(f"Found folder ID: {folder_id}")
            return folder_id
        else:
            logger.error(f"Failed to find folder: {folder_path}")
            return None
    
    def upload_file(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> bool:
        """
        Upload a single file to SharePoint folder.
        
        Args:
            local_file_path: Path to local file to upload
            sharepoint_filename: Desired filename in SharePoint
            folder_id: SharePoint folder ID where file should be uploaded
            
        Returns:
            True if upload successful, False otherwise
        """
        logger.info(f"Uploading {local_file_path} as {sharepoint_filename}")
        
        try:
            # Check if local file exists
            if not os.path.exists(local_file_path):
                logger.error(f"Local file not found: {local_file_path}")
                return False
            
            # Get file size to determine upload method
            file_size = os.path.getsize(local_file_path)
            logger.debug(f"File size: {file_size} bytes")
            
            # For files < 4MB, use simple upload
            if file_size < 4 * 1024 * 1024:
                return self._simple_upload(local_file_path, sharepoint_filename, folder_id)
            else:
                # For larger files, use resumable upload session
                return self._resumable_upload(local_file_path, sharepoint_filename, folder_id)
                
        except Exception as e:
            logger.error(f"Upload failed with exception: {e}")
            return False
    
    def _simple_upload(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> bool:
        """
        Simple upload method for small files (<4MB).
        
        Args:
            local_file_path: Local file path
            sharepoint_filename: SharePoint filename
            folder_id: Target folder ID
            
        Returns:
            True if successful, False otherwise
        """
        # Build upload URL
        url = f"{self.base_api_url}/items/{folder_id}:/{sharepoint_filename}:/content"
        
        try:
            # Read file content
            with open(local_file_path, 'rb') as f:
                file_content = f.read()
            
            # Upload file
            response = self._make_authenticated_request(
                'PUT',
                url,
                data=file_content,
                headers={'Content-Type': 'application/octet-stream'}
            )
            
            if response:
                logger.info(f"Successfully uploaded {sharepoint_filename}")
                return True
            else:
                logger.error(f"Failed to upload {sharepoint_filename}")
                return False
                
        except Exception as e:
            logger.error(f"Simple upload error: {e}")
            return False
    
    def _resumable_upload(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> bool:
        """
        Resumable upload method for large files (>4MB).
        Uses upload session to handle large files in chunks.
        
        Args:
            local_file_path: Local file path
            sharepoint_filename: SharePoint filename  
            folder_id: Target folder ID
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Using resumable upload for large file")
        
        # Create upload session
        session_url = f"{self.base_api_url}/items/{folder_id}:/{sharepoint_filename}:/createUploadSession"
        
        session_data = {
            "item": {
                "@microsoft.graph.conflictBehavior": "replace",
                "name": sharepoint_filename
            }
        }
        
        response = self._make_authenticated_request(
            'POST',
            session_url,
            json=session_data
        )
        
        if not response:
            logger.error("Failed to create upload session")
            return False
        
        upload_url = response.json().get('uploadUrl')
        if not upload_url:
            logger.error("No upload URL in session response")
            return False
        
        # Upload file in chunks
        chunk_size = 320 * 1024  # 320KB chunks
        file_size = os.path.getsize(local_file_path)
        
        try:
            with open(local_file_path, 'rb') as f:
                bytes_uploaded = 0
                
                while bytes_uploaded < file_size:
                    # Read next chunk
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    chunk_size_actual = len(chunk)
                    range_start = bytes_uploaded
                    range_end = bytes_uploaded + chunk_size_actual - 1
                    
                    # Upload chunk
                    headers = {
                        'Content-Length': str(chunk_size_actual),
                        'Content-Range': f'bytes {range_start}-{range_end}/{file_size}'
                    }
                    
                    chunk_response = requests.put(
                        upload_url,
                        data=chunk,
                        headers=headers,
                        timeout=self.config.timeout_seconds
                    )
                    
                    if chunk_response.status_code not in [202, 201, 200]:
                        logger.error(f"Chunk upload failed: {chunk_response.status_code}")
                        return False
                    
                    bytes_uploaded += chunk_size_actual
                    progress = (bytes_uploaded / file_size) * 100
                    logger.debug(f"Upload progress: {progress:.1f}%")
                
                logger.info(f"Successfully uploaded {sharepoint_filename} via resumable upload")
                return True
                
        except Exception as e:
            logger.error(f"Resumable upload error: {e}")
            return False


class ExcelUploadManager:
    """Main manager class that orchestrates the Excel file upload process."""
    
    def __init__(self, config_path: str = "config/sharepoint_config.yaml"):
        """
        Initialize upload manager with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.uploader = SharePointFileUploader(self.config)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.config.upload_state_file), exist_ok=True)
    
    def _load_config(self, config_path: str) -> SharePointConfig:
        """
        Load configuration from YAML file and create SharePointConfig object.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            SharePointConfig object
            
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If required configuration values missing
        """
        logger.info(f"Loading configuration from {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract and validate configuration
        try:
            azure_ad = config_data['azure_ad']
            sharepoint = config_data['sharepoint']
            local_paths = config_data['local_paths']
            processing = config_data['processing']
            api_config = config_data['api']
            
            return SharePointConfig(
                tenant_id=azure_ad['tenant_id'],
                client_id=azure_ad['client_id'],
                client_secret=azure_ad['client_secret'],
                site_id=sharepoint['site_id'],
                input_folder_path=sharepoint['input_folder_path'],
                excel_source_dir=local_paths['excel_source_dir'],
                upload_state_file=local_paths['upload_state_file'],
                skip_newest_files=processing['skip_newest_files'],
                max_files_per_run=processing['max_files_per_run'],
                timeout_seconds=api_config['timeout_seconds'],
                retry_attempts=api_config['retry_attempts'],
                retry_delay_seconds=api_config['retry_delay_seconds']
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")
    
    def _load_upload_state(self) -> List[str]:
        """
        Load upload state from JSON file - simple list of uploaded filenames.
        
        Returns:
            List of filenames that have been uploaded
        """
        if os.path.exists(self.config.upload_state_file):
            try:
                with open(self.config.upload_state_file, 'r') as f:
                    uploaded_files = json.load(f)
                logger.info(f"Loaded upload state: {len(uploaded_files)} files already uploaded")
                return uploaded_files if isinstance(uploaded_files, list) else []
            except Exception as e:
                logger.warning(f"Failed to load upload state: {e}")
                return []
        return []
    
    def _save_upload_state(self, uploaded_files: List[str]):
        """
        Save upload state to JSON file - simple list of uploaded filenames.
        
        Args:
            uploaded_files: List of filenames that have been uploaded
        """
        try:
            with open(self.config.upload_state_file, 'w') as f:
                json.dump(sorted(list(set(uploaded_files))), f, indent=2)
            logger.debug("Upload state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save upload state: {e}")
    
    def _get_local_excel_files(self) -> List[Tuple[str, float]]:
        """
        Get list of Excel files from batch_3 directory structure with their modification times.
        
        Searches in /home/vision_ai_adm/code/oslo/video_convertbranch/csv_files/batch_3/
        and processes both subdirectories (labelstudio_csv_prod_server_1 and labelstudio_csv_prod_server_2).
        
        Returns:
            List of (file_path, modification_time) tuples sorted by modification time
        """
        excel_files = []
        
        # Fixed batch_3 directory path
        batch_3_dir = "/home/vision_ai_adm/code/oslo/video_convertbranch/csv_files/batch_3"
        
        if not os.path.exists(batch_3_dir):
            logger.error(f"Batch 3 directory not found: {batch_3_dir}")
            return []
        
        logger.info(f"Scanning for Excel files in: {batch_3_dir}")
        
        # Get all subdirectories in batch_3
        try:
            subdirs = [d for d in os.listdir(batch_3_dir) 
                      if os.path.isdir(os.path.join(batch_3_dir, d))]
            
            logger.info(f"Found subdirectories in batch_3: {subdirs}")
            
            # Search for .xlsx files in each subdirectory
            for subdir in subdirs:
                subdir_path = os.path.join(batch_3_dir, subdir)
                logger.info(f"Scanning directory: {subdir_path}")
                
                for pattern in ["*.xlsx", "*.xls"]:
                    files = glob.glob(os.path.join(subdir_path, pattern))
                    excel_files.extend(files)
                    logger.debug(f"Found {len(files)} {pattern} files in {subdir}")
                
        except Exception as e:
            logger.error(f"Error scanning batch_3 directory: {e}")
            return []
        
        # Get modification times and sort by modification time (oldest first)
        files_with_times = []
        for file_path in excel_files:
            if os.path.exists(file_path):
                mod_time = os.path.getmtime(file_path)
                files_with_times.append((file_path, mod_time))
        
        # Sort by modification time (oldest first)
        files_with_times.sort(key=lambda x: x[1])
        
        logger.info(f"Found {len(files_with_times)} total Excel files in batch_3 directory")
        return files_with_times
    
    def _filter_files_for_upload(self, all_files: List[Tuple[str, float]], uploaded_files: List[str], skip_newest: bool = True) -> List[str]:
        """
        Filter files to determine which ones need to be uploaded.
        
        Args:
            all_files: List of (file_path, mod_time) tuples
            uploaded_files: List of filenames already uploaded
            skip_newest: If True, skip newest file from each folder (default mode)
            
        Returns:
            List of file paths that need to be uploaded
        """
        if not all_files:
            return []
        
        # Group files by their parent directory (folder)
        files_by_folder = {}
        for file_path, mod_time in all_files:
            folder = os.path.dirname(file_path)
            if folder not in files_by_folder:
                files_by_folder[folder] = []
            files_by_folder[folder].append((file_path, mod_time))
        
        # Sort files within each folder by modification time
        for folder in files_by_folder:
            files_by_folder[folder].sort(key=lambda x: x[1])  # Sort by mod_time
        
        files_to_consider = []
        
        if skip_newest:
            # Default mode: Skip the newest file from each folder
            logger.info("Running in DEFAULT mode - skipping newest file from each folder")
            
            for folder, folder_files in files_by_folder.items():
                folder_name = os.path.basename(folder)
                
                if len(folder_files) > 1:
                    # Skip the newest file (last in sorted list)
                    files_to_consider.extend(folder_files[:-1])
                    skipped_file = folder_files[-1]
                    
                    filename = os.path.basename(skipped_file[0])
                    mod_time_str = datetime.fromtimestamp(skipped_file[1]).strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"  Skipping newest from {folder_name}: {filename} (modified: {mod_time_str})")
                    
                elif len(folder_files) == 1:
                    # Only one file in folder - skip it (it's the newest)
                    skipped_file = folder_files[0]
                    filename = os.path.basename(skipped_file[0])
                    mod_time_str = datetime.fromtimestamp(skipped_file[1]).strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"  Skipping only file from {folder_name}: {filename} (modified: {mod_time_str})")
        else:
            # Total file mode: Include all files
            logger.info("Running in TOTAL_FILE mode - uploading all files")
            for folder_files in files_by_folder.values():
                files_to_consider.extend(folder_files)
        
        # Filter out already uploaded files
        files_to_upload = []
        
        for file_path, mod_time in files_to_consider:
            filename = os.path.basename(file_path)
            
            # Check if file was already uploaded (simple filename check)
            if filename in uploaded_files:
                logger.debug(f"Skipping already uploaded file: {filename}")
                continue
            
            files_to_upload.append(file_path)
        
        # Limit number of files per run
        if len(files_to_upload) > self.config.max_files_per_run:
            logger.info(f"Limiting to {self.config.max_files_per_run} files per run")
            files_to_upload = files_to_upload[:self.config.max_files_per_run]
        
        return files_to_upload
    
    def run_upload(self, dry_run: bool = False, skip_newest: bool = True) -> bool:
        """
        Main method to execute the upload process.
        
        Args:
            dry_run: If True, show what would be uploaded without actually uploading
            skip_newest: If True, skip newest file from each folder (default mode)
            
        Returns:
            True if all uploads successful, False if any failed
        """
        logger.info("=" * 60)
        logger.info("Starting Excel File Upload to SharePoint")
        logger.info("=" * 60)
        
        try:
            # Get SharePoint input folder ID
            input_folder_id = self.uploader.get_folder_id(self.config.input_folder_path)
            if not input_folder_id:
                logger.error("Cannot proceed without valid input folder ID")
                return False
            
            # Load current upload state (simple list of uploaded filenames)
            uploaded_files = self._load_upload_state()
            
            # Get local Excel files
            all_files = self._get_local_excel_files()
            if not all_files:
                logger.info("No Excel files found in source directory")
                return True
            
            # Filter files that need uploading
            files_to_upload = self._filter_files_for_upload(all_files, uploaded_files, skip_newest)
            
            if not files_to_upload:
                logger.info("No new files to upload")
                return True
            
            logger.info(f"Found {len(files_to_upload)} files to upload")
            
            if dry_run:
                logger.info("DRY RUN MODE - Files that would be uploaded:")
                for file_path in files_to_upload:
                    logger.info(f"  - {os.path.basename(file_path)}")
                return True
            
            # Upload files
            successful_uploads = 0
            
            for i, file_path in enumerate(files_to_upload, 1):
                filename = os.path.basename(file_path)
                logger.info(f"Uploading file {i}/{len(files_to_upload)}: {filename}")
                
                # Upload file
                success = self.uploader.upload_file(
                    local_file_path=file_path,
                    sharepoint_filename=filename,
                    folder_id=input_folder_id
                )
                
                if success:
                    successful_uploads += 1
                    logger.info(f"✅ Successfully uploaded: {filename}")
                    # Add to uploaded files list (simple tracking)
                    uploaded_files.append(filename)
                else:
                    logger.error(f"❌ Failed to upload: {filename}")
                
                # Save state after each file (simple list of successful uploads)
                self._save_upload_state(uploaded_files)
            
            # Summary
            logger.info("=" * 60)
            logger.info(f"Upload Summary:")
            logger.info(f"  Total files: {len(files_to_upload)}")
            logger.info(f"  Successful: {successful_uploads}")
            logger.info(f"  Failed: {len(files_to_upload) - successful_uploads}")
            logger.info("=" * 60)
            
            return successful_uploads == len(files_to_upload)
            
        except Exception as e:
            logger.error(f"Upload process failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for the Excel uploader script."""
    import argparse
    
    # Check for positional argument first
    mode = "default"
    if len(sys.argv) > 1 and sys.argv[1] == "total_file":
        mode = "total_file"
        # Remove the mode argument from sys.argv so argparse doesn't see it
        sys.argv.pop(1)
    
    parser = argparse.ArgumentParser(
        description="Upload Excel files from batch_3 folders to SharePoint",
        epilog="""
Modes:
  default      Skip newest file from each folder (prevents uploading files still being processed) 
  total_file   Upload all files including newest ones

Examples:
  python sharepoint_excel_uploader.py                    # Default mode
  python sharepoint_excel_uploader.py total_file         # Total file mode
  python sharepoint_excel_uploader.py --dry-run          # Default mode with dry run
  python sharepoint_excel_uploader.py total_file --dry-run  # Total file mode with dry run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', 
        default='config/sharepoint_config.yaml',
        help='Path to configuration file (default: config/sharepoint_config.yaml)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be uploaded without actually uploading'
    )
    
    args = parser.parse_args()
    
    # Determine skip_newest based on mode
    skip_newest = (mode == "default")
    
    logger.info(f"Running in {mode.upper()} mode")
    
    try:
        # Initialize upload manager
        upload_manager = ExcelUploadManager(args.config)
        
        # Run upload process
        success = upload_manager.run_upload(dry_run=args.dry_run, skip_newest=skip_newest)
        
        if success:
            logger.info("All uploads completed successfully!")
            sys.exit(0)
        else:
            logger.error("Some uploads failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
