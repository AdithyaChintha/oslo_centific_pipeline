#!/usr/bin/env python3
"""
Simplified SharePoint Output Processor Script

This script downloads ALL Excel files from SharePoint output folder,
converts them to JSONL format using batch conversion, and uploads
the JSONL file to SharePoint delivery folder.

Simplified Features:
- Downloads ALL annotated Excel files from SharePoint Output folder
- Saves them to local folder for batch processing
- Converts entire folder to JSONL using existing pyrenees conversion code
- Uploads single JSONL file to SharePoint delivery folder
- No state tracking - processes everything each time
- Handles authentication via Microsoft Graph API
- Provides detailed logging and progress tracking

Usage:
    python sharepoint_output_processor_simplified.py [--config CONFIG_FILE] [--dry-run]

Author: Simplified for batch processing workflow
"""

import os
import sys
import json
import yaml
import requests
import shutil
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the existing Excel to JSONL conversion function
try:
    from pyrenees_post_consolidation_from_xlsx import convert_csv_to_json
except ImportError as e:
    print(f"ERROR: Cannot import conversion module: {e}")
    print("Make sure pyreneus_post_consolidation_from_xlsx.py is in the same directory")
    sys.exit(1)

@dataclass
class SharePointConfig:
    """Configuration for SharePoint processing."""
    tenant_id: str
    client_id: str 
    client_secret: str
    site_id: str
    drive_id: str
    output_folder_path: str
    delivery_folder_path: str
    download_dir: str
    max_files_per_run: int
    timeout_seconds: int
    retry_attempts: int
    retry_delay_seconds: int

class SharePointAuthenticator:
    """Handles SharePoint authentication using Microsoft Graph API."""
    
    def __init__(self, config: SharePointConfig):
        self.config = config
        self.access_token = None
        self.token_expiry = None
    
    def get_access_token(self) -> Optional[str]:
        """
        Get access token using client credentials flow.
        
        Returns:
            Access token string or None if authentication failed
        """
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.access_token
        
        logger.info("Requesting new access token from Microsoft Graph API")
        
        auth_url = f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token"
        
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'scope': 'https://graph.microsoft.com/.default'
        }
        
        try:
            response = requests.post(auth_url, data=data, timeout=30)
            response.raise_for_status()
            
            # Parse response
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            expires_in = token_data.get('expires_in', 3600)  # Default 1 hour
            
            # Set token expiry (subtract 5 minutes for safety buffer)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)
            
            logger.info("Successfully obtained access token")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return None

class SharePointFileManager:
    """Manages SharePoint file operations."""
    
    def __init__(self, config: SharePointConfig):
        self.config = config
        self.authenticator = SharePointAuthenticator(config)
    
    def _make_authenticated_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Make authenticated request to Microsoft Graph API."""
        token = self.authenticator.get_access_token()
        if not token:
            return None
        
        headers = kwargs.get('headers', {})
        headers['Authorization'] = f'Bearer {token}'
        kwargs['headers'] = headers
        
        for attempt in range(self.config.retry_attempts):
            try:
                response = requests.request(
                    method, url, 
                    timeout=self.config.timeout_seconds,
                    **kwargs
                )
                
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', self.config.retry_delay_seconds))
                    logger.warning(f"Rate limited, waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds * (attempt + 1))
                else:
                    logger.error(f"All {self.config.retry_attempts} attempts failed")
        
        return None
    
    def get_folder_id(self, folder_path: str) -> Optional[str]:
        """
        Get SharePoint folder ID from path using drive ID.
        
        Args:
            folder_path: Path to folder (e.g., "/Team Optimization_Delivery/...")
            
        Returns:
            Folder ID string or None if not found
        """
        logger.info(f"Looking up folder ID for path: {folder_path}")
        
        # Clean path - remove leading/trailing slashes
        clean_path = folder_path.strip('/')
        
        # Use drive ID for more reliable access
        if not clean_path:
            url = f"https://graph.microsoft.com/v1.0/sites/{self.config.site_id}/drives/{self.config.drive_id}/root"
        else:
            url = f"https://graph.microsoft.com/v1.0/sites/{self.config.site_id}/drives/{self.config.drive_id}/root:/{clean_path}"
        
        response = self._make_authenticated_request('GET', url)
        
        if response:
            folder_data = response.json()
            folder_id = folder_data.get('id')
            logger.info(f"Found folder ID: {folder_id}")
            return folder_id
        
        logger.error(f"Failed to get folder ID for: {folder_path}")
        return None
    
    def list_files_in_folder(self, folder_id: str) -> List[Dict]:
        """
        List all files in a SharePoint folder.
        
        Args:
            folder_id: SharePoint folder ID
            
        Returns:
            List of file info dictionaries
        """
        logger.info(f"Listing files in folder ID: {folder_id}")
        
        url = f"https://graph.microsoft.com/v1.0/sites/{self.config.site_id}/drives/{self.config.drive_id}/items/{folder_id}/children"
        
        response = self._make_authenticated_request('GET', url)
        
        if response:
            items = response.json().get('value', [])
            
            # Filter to only files (not folders) and Excel files
            files = []
            for item in items:
                if 'file' in item:  # It's a file, not a folder
                    file_name = item['name']
                    if file_name.lower().endswith(('.xlsx', '.xls', '.csv')):
                        files.append({
                            'name': file_name,
                            'id': item['id'],
                            'size': item.get('size', 0),
                            'modified': item.get('lastModifiedDateTime'),
                            'download_url': item.get('@microsoft.graph.downloadUrl')
                        })
            
            logger.info(f"Found {len(files)} Excel/CSV files")
            return files
        
        logger.error("Failed to list files in folder")
        return []
    
    def download_file(self, download_url: str, local_path: str) -> bool:
        """
        Download file from SharePoint to local path.
        
        Args:
            download_url: Direct download URL from Microsoft Graph
            local_path: Local file path to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file (download URLs don't need authentication)
            response = requests.get(download_url, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            
            # Save to local file
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            logger.debug(f"Downloaded: {os.path.basename(local_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {os.path.basename(local_path)}: {e}")
            return False
    
    def upload_file(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> bool:
        """
        Upload file to SharePoint folder.
        
        Args:
            local_file_path: Path to local file
            sharepoint_filename: Name for file in SharePoint
            folder_id: SharePoint folder ID to upload to
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Uploading: {sharepoint_filename}")
        
        # Use simple upload for files < 4MB, resumable upload for larger files
        file_size = os.path.getsize(local_file_path)
        
        if file_size < 4 * 1024 * 1024:  # 4MB
            return self._simple_upload(local_file_path, sharepoint_filename, folder_id)
        else:
            return self._resumable_upload(local_file_path, sharepoint_filename, folder_id)
    
    def _simple_upload(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> bool:
        """Simple upload for small files."""
        url = f"https://graph.microsoft.com/v1.0/sites/{self.config.site_id}/drives/{self.config.drive_id}/items/{folder_id}:/{sharepoint_filename}:/content"
        
        try:
            with open(local_file_path, 'rb') as f:
                file_content = f.read()
            
            headers = {'Content-Type': 'application/octet-stream'}
            response = self._make_authenticated_request('PUT', url, data=file_content, headers=headers)
            
            if response and response.status_code in [200, 201]:
                logger.info(f"✅ Upload successful: {sharepoint_filename}")
                return True
            else:
                logger.error(f"Upload failed with status: {response.status_code if response else 'No response'}")
                return False
                
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def _resumable_upload(self, local_file_path: str, sharepoint_filename: str, folder_id: str) -> bool:
        """Resumable upload for large files."""
        logger.info("Using resumable upload for large file")
        
        # Create upload session
        session_url = f"https://graph.microsoft.com/v1.0/sites/{self.config.site_id}/drives/{self.config.drive_id}/items/{folder_id}:/{sharepoint_filename}:/createUploadSession"
        
        session_response = self._make_authenticated_request('POST', session_url, json={})
        if not session_response:
            return False
        
        upload_url = session_response.json().get('uploadUrl')
        if not upload_url:
            return False
        
        # Upload file in chunks
        chunk_size = 320 * 1024  # 320KB chunks
        file_size = os.path.getsize(local_file_path)
        
        with open(local_file_path, 'rb') as f:
            for start in range(0, file_size, chunk_size):
                end = min(start + chunk_size - 1, file_size - 1)
                chunk_data = f.read(chunk_size)
                
                headers = {
                    'Content-Length': str(len(chunk_data)),
                    'Content-Range': f'bytes {start}-{end}/{file_size}'
                }
                
                response = requests.put(upload_url, data=chunk_data, headers=headers)
                
                if response.status_code not in [202, 201, 200]:
                    logger.error(f"Chunk upload failed: {response.status_code}")
                    return False
        
        logger.info(f"✅ Resumable upload successful: {sharepoint_filename}")
        return True

class SimplifiedOutputProcessor:
    """Simplified SharePoint Output Processor - batch processing without state tracking."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.file_manager = SharePointFileManager(self.config)
        
        # Ensure directories exist
        os.makedirs(self.config.download_dir, exist_ok=True)
    
    def _load_config(self, config_path: str) -> SharePointConfig:
        """Load configuration from YAML file."""
        logger.info(f"Loading configuration from {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Extract configuration sections
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
                drive_id=sharepoint['drive_id'],
                output_folder_path=sharepoint['output_folder_path'],
                delivery_folder_path=sharepoint['delivery_folder_path'],
                download_dir=local_paths['download_dir'],
                max_files_per_run=processing['max_files_per_run'],
                timeout_seconds=api_config['timeout_seconds'],
                retry_attempts=api_config['retry_attempts'],
                retry_delay_seconds=api_config['retry_delay_seconds']
            )
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")
    
    def run_processing(self, dry_run: bool = False) -> bool:
        """
        Main method to execute the simplified download → convert → upload process.
        
        Args:
            dry_run: If True, show what would be processed without actually doing it
            
        Returns:
            True if all processing successful, False if any failed
        """
        logger.info("=" * 70)
        logger.info("Starting Simplified SharePoint Output Processing")
        logger.info("(Batch Download → Folder Convert → JSONL Upload)")
        logger.info("=" * 70)
        
        try:
            # Step 1: Get folder IDs
            output_folder_id = self.file_manager.get_folder_id(self.config.output_folder_path)
            if not output_folder_id:
                logger.error("Cannot proceed without valid output folder ID")
                return False
            
            delivery_folder_id = self.file_manager.get_folder_id(self.config.delivery_folder_path)
            if not delivery_folder_id:
                logger.error("Cannot proceed without valid delivery folder ID")
                return False
            
            # Step 2: List all Excel files in SharePoint output folder
            sharepoint_files = self.file_manager.list_files_in_folder(output_folder_id)
            
            if not sharepoint_files:
                logger.info("No Excel files found in SharePoint output folder")
                return True
            
            logger.info(f"Found {len(sharepoint_files)} Excel files to process")
            
            if dry_run:
                logger.info("DRY RUN MODE - Files that would be processed:")
                for file_info in sharepoint_files:
                    logger.info(f"  - {file_info['name']}")
                logger.info("DRY RUN - No actual processing performed")
                return True
            
            # Step 3: Download all Excel files to local folder
            logger.info(f"📥 Downloading all Excel files to: {self.config.download_dir}")
            
            # Clear download directory first
            if os.path.exists(self.config.download_dir):
                shutil.rmtree(self.config.download_dir)
            os.makedirs(self.config.download_dir, exist_ok=True)
            
            downloaded_count = 0
            for file_info in sharepoint_files:
                file_name = file_info['name']
                download_url = file_info['download_url']
                local_path = os.path.join(self.config.download_dir, file_name)
                
                logger.info(f"  Downloading: {file_name}")
                
                if self.file_manager.download_file(download_url, local_path):
                    downloaded_count += 1
                else:
                    logger.error(f"Failed to download: {file_name}")
            
            logger.info(f"✅ Downloaded {downloaded_count}/{len(sharepoint_files)} files")
            
            if downloaded_count == 0:
                logger.error("No files were downloaded successfully")
                return False
            
            # Step 4: Convert entire folder to JSONL using existing converter
            logger.info(f"🔄 Converting folder to JSONL: {self.config.download_dir}")
            
            try:
                # Use the existing converter in directory mode
                jsonl_path = convert_csv_to_json(self.config.download_dir)
                
                if not jsonl_path or not os.path.exists(jsonl_path):
                    logger.error("JSONL conversion failed - no output file created")
                    return False
                
                logger.info(f"✅ JSONL conversion successful: {os.path.basename(jsonl_path)}")
                
            except Exception as e:
                logger.error(f"JSONL conversion failed: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # Step 5: Upload JSONL to SharePoint delivery folder
            jsonl_filename = os.path.basename(jsonl_path)
            logger.info(f"📤 Uploading JSONL to SharePoint delivery: {jsonl_filename}")
            
            if not self.file_manager.upload_file(
                local_file_path=jsonl_path,
                sharepoint_filename=jsonl_filename,
                folder_id=delivery_folder_id
            ):
                logger.error("Failed to upload JSONL to SharePoint")
                return False
            
            logger.info("=" * 70)
            logger.info("✅ BATCH PROCESSING COMPLETED SUCCESSFULLY!")
            logger.info(f"📊 Processed: {downloaded_count} Excel files → 1 JSONL file")
            logger.info(f"📤 Delivered: {jsonl_filename}")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point for the simplified output processor script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simplified SharePoint Output Processor - Batch Download → Convert → Upload",
        epilog="""
Simplified Processing Flow:
  1. Download ALL Excel files from SharePoint Output folder
  2. Save to local folder for batch processing  
  3. Convert entire folder to single JSONL using existing converter
  4. Upload JSONL to SharePoint Delivery folder
  5. No state tracking - processes everything each time

Examples:
  python sharepoint_output_processor_simplified.py                    # Normal processing
  python sharepoint_output_processor_simplified.py --dry-run          # Show what would be processed
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
        help='Show what would be processed without actually doing it'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = SimplifiedOutputProcessor(args.config)
        
        # Run processing workflow
        success = processor.run_processing(dry_run=args.dry_run)
        
        if success:
            logger.info("All processing completed successfully!")
            sys.exit(0)
        else:
            logger.error("Processing failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

