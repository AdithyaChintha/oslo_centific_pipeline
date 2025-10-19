#!/usr/bin/env python3
"""
JSONL SharePoint to S3 Bucket Transfer Script

This script automates the process of:
1. Downloading JSONL files from a specified SharePoint folder
2. Uploading those JSONL files to an S3 bucket at a specified path
3. Maintaining state to avoid re-processing files

The script uses the same configuration patterns as other scripts in the project
for both SharePoint and S3 credentials.

Author: SharePoint to S3 automation script
Date: October 2025
"""

import os
import sys
import yaml
import json
import shutil
import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add utils to path for S3 operations
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.s3_utils import create_s3_client
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install required packages: pip install boto3")
    sys.exit(1)

# SharePoint operations using shared utilities
try:
    from utils.sharepoint_utils import (
        SharePointAuthenticator, 
        SharePointFileManager, 
        SharePointConfig,
        load_sharepoint_config,
        create_sharepoint_manager
    )
except ImportError:
    # Fallback for direct execution
    sys.path.append('/home/vision_ai_adm/code/oslo/video_convertbranch')
    from utils.sharepoint_utils import (
        SharePointAuthenticator, 
        SharePointFileManager, 
        SharePointConfig,
        load_sharepoint_config,
        create_sharepoint_manager
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/vision_ai_adm/code/oslo/video_convertbranch/jsonl_sharepoint_to_s3.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class JSONLTransferConfig:
    """Configuration class for JSONL transfer from SharePoint to S3."""
    
    # SharePoint Configuration
    tenant_id: str
    client_id: str
    client_secret: str
    site_id: str
    drive_id: str
    jsonl_source_folder_path: str
    
    # S3 Configuration
    aws_access_key_id: str
    aws_secret_access_key: str
    s3_region: str
    s3_bucket_name: str
    s3_target_prefix: str
    
    # Local Configuration
    temp_download_dir: str
    state_file_path: str
    
    # Processing Configuration
    max_files_per_run: int
    timeout_seconds: int
    retry_attempts: int
    retry_delay_seconds: int

class JSONLSharePointToS3Processor:
    """
    Processes JSONL files from SharePoint to S3 bucket.
    
    This class handles the complete workflow:
    - Authenticating with SharePoint
    - Downloading JSONL files from SharePoint folder
    - Uploading JSONL files to S3 bucket
    - State tracking to avoid re-processing
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the JSONL processor with configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.authenticator = None
        self.file_manager = None
        self.s3_client = None
        
    def _load_config(self, config_path: str) -> JSONLTransferConfig:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            JSONLTransferConfig: Validated configuration object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            KeyError: If required configuration keys are missing
        """
        logger.info(f"Loading configuration from: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        try:
            # Extract configuration sections
            azure_ad = config_data['azure_ad']
            sharepoint = config_data['sharepoint']
            s3_config = config_data['s3']
            local_paths = config_data['local_paths']
            processing = config_data['processing']
            api_config = config_data['api']
            
            return JSONLTransferConfig(
                # SharePoint credentials
                tenant_id=azure_ad['tenant_id'],
                client_id=azure_ad['client_id'],
                client_secret=azure_ad['client_secret'],
                site_id=sharepoint['site_id'],
                drive_id=sharepoint['drive_id'],
                jsonl_source_folder_path=sharepoint['jsonl_source_folder_path'],
                
                # S3 credentials and config
                aws_access_key_id=s3_config['aws_access_key_id'],
                aws_secret_access_key=s3_config['aws_secret_access_key'],
                s3_region=s3_config['region'],
                s3_bucket_name=s3_config['bucket_name'],
                s3_target_prefix=s3_config['jsonl_target_prefix'],
                
                # Local paths
                temp_download_dir=local_paths['jsonl_temp_download_dir'],
                state_file_path=local_paths['jsonl_transfer_state_file'],
                
                # Processing settings
                max_files_per_run=processing['max_files_per_run'],
                timeout_seconds=api_config['timeout_seconds'],
                retry_attempts=api_config['retry_attempts'],
                retry_delay_seconds=api_config['retry_delay_seconds']
            )
            
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")
    
    def _initialize_sharepoint_client(self) -> Tuple[SharePointAuthenticator, SharePointFileManager]:
        """
        Initialize SharePoint authentication and file manager using shared utilities.
        
        Returns:
            Tuple[SharePointAuthenticator, SharePointFileManager]: Initialized SharePoint clients
            
        Raises:
            Exception: If SharePoint authentication fails
        """
        logger.info("🔑 Initializing SharePoint authentication...")
        
        try:
            # Create SharePoint config from our config
            sharepoint_config = SharePointConfig(
                tenant_id=self.config.tenant_id,
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                site_id=self.config.site_id,
                drive_id=self.config.drive_id,
                timeout_seconds=self.config.timeout_seconds,
                retry_attempts=self.config.retry_attempts,
                retry_delay_seconds=self.config.retry_delay_seconds
            )
            
            # Create authenticator and file manager using shared utilities
            authenticator, file_manager = create_sharepoint_manager(sharepoint_config)
            
            # Test authentication
            token = authenticator.get_access_token()
            if not token:
                raise Exception("Failed to obtain SharePoint access token")
            
            logger.info("✅ SharePoint authentication successful")
            return authenticator, file_manager
            
        except Exception as e:
            logger.error(f"❌ SharePoint authentication failed: {e}")
            raise
    
    def _initialize_s3_client(self) -> boto3.client:
        """
        Initialize S3 client using existing utility functions.
        
        Returns:
            boto3.client: Configured S3 client
            
        Raises:
            Exception: If S3 client initialization fails
        """
        logger.info("🔑 Initializing S3 client...")
        
        try:
            # Create config dict in the format expected by s3_utils
            s3_config_dict = {
                's3': {
                    'aws_access_key_id': self.config.aws_access_key_id,
                    'aws_secret_access_key': self.config.aws_secret_access_key,
                    'region_name': self.config.s3_region
                }
            }
            
            s3_client = create_s3_client(s3_config_dict)
            
            # Test S3 connection by listing buckets
            s3_client.head_bucket(Bucket=self.config.s3_bucket_name)
            
            logger.info("✅ S3 authentication successful")
            return s3_client
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.error(f"❌ S3 bucket not found: {self.config.s3_bucket_name}")
            else:
                logger.error(f"❌ S3 access error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ S3 client initialization failed: {e}")
            raise
    
    def _load_transfer_state(self) -> List[str]:
        """
        Load the list of already transferred JSONL files.
        
        Returns:
            List[str]: List of filenames that have been already transferred
        """
        if not os.path.exists(self.config.state_file_path):
            logger.info(f"📋 No existing state file found: {self.config.state_file_path}")
            return []
        
        try:
            with open(self.config.state_file_path, 'r') as f:
                transferred_files = json.load(f)
            
            logger.info(f"📋 Loaded state: {len(transferred_files)} files already transferred")
            return transferred_files
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load state file: {e}")
            return []
    
    def _save_transfer_state(self, transferred_files: List[str]):
        """
        Save the list of transferred JSONL files to state file.
        
        Args:
            transferred_files: List of filenames that have been transferred
        """
        try:
            os.makedirs(os.path.dirname(self.config.state_file_path), exist_ok=True)
            
            with open(self.config.state_file_path, 'w') as f:
                json.dump(transferred_files, f, indent=2)
            
            logger.info(f"💾 Saved state: {len(transferred_files)} files in transfer history")
            
        except Exception as e:
            logger.error(f"❌ Failed to save state file: {e}")
    
    def _get_sharepoint_jsonl_files(self, folder_id: str) -> List[Dict]:
        """
        Get list of JSONL files from SharePoint folder using shared utilities.
        
        Args:
            folder_id: SharePoint folder ID to scan
            
        Returns:
            List[Dict]: List of JSONL file information from SharePoint
        """
        logger.info(f"📂 Scanning SharePoint folder for JSONL files...")
        
        try:
            # Use shared utilities with file extension filtering
            jsonl_files = self.file_manager.list_files_in_folder(folder_id, file_extensions=['.jsonl'])
            
            logger.info(f"📄 Found {len(jsonl_files)} JSONL files in SharePoint folder")
            return jsonl_files
            
        except Exception as e:
            logger.error(f"❌ Failed to list SharePoint files: {e}")
            raise
    
    def _filter_new_files(self, all_files: List[Dict], transferred_files: List[str]) -> List[Dict]:
        """
        Filter out files that have already been transferred.
        
        Args:
            all_files: All JSONL files from SharePoint
            transferred_files: List of already transferred filenames
            
        Returns:
            List[Dict]: Files that need to be transferred
        """
        new_files = [
            file for file in all_files 
            if file['name'] not in transferred_files
        ]
        
        # Apply max files per run limit
        if len(new_files) > self.config.max_files_per_run:
            logger.info(f"⚠️ Limiting to {self.config.max_files_per_run} files per run")
            new_files = new_files[:self.config.max_files_per_run]
        
        logger.info(f"🆕 {len(new_files)} new JSONL files to transfer")
        return new_files
    
    def _download_jsonl_files(self, files_to_download: List[Dict]) -> List[str]:
        """
        Download JSONL files from SharePoint to local temp directory.
        
        Args:
            files_to_download: List of file info dictionaries
            
        Returns:
            List[str]: List of local file paths that were successfully downloaded
        """
        logger.info(f"📥 Downloading {len(files_to_download)} JSONL files from SharePoint...")
        
        # Ensure temp download directory exists
        os.makedirs(self.config.temp_download_dir, exist_ok=True)
        
        successfully_downloaded = []
        
        for file_info in files_to_download:
            try:
                filename = file_info['name']
                download_url = file_info.get('download_url')
                local_path = os.path.join(self.config.temp_download_dir, filename)
                
                logger.info(f"  📄 Downloading: {filename}")
                
                if download_url:
                    success = self.file_manager.download_file(download_url, local_path)
                else:
                    # If no direct download URL, try using file ID
                    success = self.file_manager.download_file_by_id(file_info['id'], local_path)
                
                if success and os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    logger.info(f"    ✅ Downloaded: {filename} ({file_size:,} bytes)")
                    successfully_downloaded.append(local_path)
                else:
                    logger.error(f"    ❌ Failed to download: {filename}")
                    
            except Exception as e:
                logger.error(f"    ❌ Error downloading {filename}: {e}")
        
        logger.info(f"✅ Successfully downloaded {len(successfully_downloaded)} JSONL files")
        return successfully_downloaded
    
    def _upload_to_s3(self, local_files: List[str]) -> List[str]:
        """
        Upload JSONL files to S3 bucket.
        
        Args:
            local_files: List of local file paths to upload
            
        Returns:
            List[str]: List of filenames that were successfully uploaded
        """
        logger.info(f"📤 Uploading {len(local_files)} JSONL files to S3...")
        
        successfully_uploaded = []
        
        for local_path in local_files:
            try:
                filename = os.path.basename(local_path)
                s3_key = f"{self.config.s3_target_prefix.rstrip('/')}/{filename}"
                
                logger.info(f"  📄 Uploading: {filename} → s3://{self.config.s3_bucket_name}/{s3_key}")
                
                # Upload file to S3
                self.s3_client.upload_file(
                    local_path,
                    self.config.s3_bucket_name,
                    s3_key
                )
                
                # Verify upload
                response = self.s3_client.head_object(
                    Bucket=self.config.s3_bucket_name,
                    Key=s3_key
                )
                
                file_size = response['ContentLength']
                logger.info(f"    ✅ Uploaded: {filename} ({file_size:,} bytes)")
                successfully_uploaded.append(filename)
                
            except ClientError as e:
                logger.error(f"    ❌ S3 upload failed for {filename}: {e}")
            except Exception as e:
                logger.error(f"    ❌ Error uploading {filename}: {e}")
        
        logger.info(f"✅ Successfully uploaded {len(successfully_uploaded)} files to S3")
        return successfully_uploaded
    
    def _cleanup_temp_files(self):
        """Clean up temporary downloaded files."""
        try:
            if os.path.exists(self.config.temp_download_dir):
                shutil.rmtree(self.config.temp_download_dir)
                logger.info(f"🧹 Cleaned up temporary directory: {self.config.temp_download_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup temp directory: {e}")
    
    def run_transfer(self, dry_run: bool = False) -> bool:
        """
        Execute the complete JSONL transfer workflow.
        
        Args:
            dry_run: If True, simulate the process without actual transfers
            
        Returns:
            bool: True if transfer completed successfully, False otherwise
        """
        logger.info("=" * 80)
        logger.info("🚀 Starting JSONL SharePoint to S3 Transfer")
        logger.info("=" * 80)
        
        if dry_run:
            logger.info("🎭 DRY RUN MODE - No actual transfers will be performed")
        
        try:
            # Initialize clients
            self.authenticator, self.file_manager = self._initialize_sharepoint_client()
            self.s3_client = self._initialize_s3_client()
            
            # Get SharePoint folder ID
            logger.info(f"🔍 Looking up SharePoint folder: {self.config.jsonl_source_folder_path}")
            folder_id = self.file_manager.get_folder_id(self.config.jsonl_source_folder_path)
            
            if not folder_id:
                logger.error(f"❌ SharePoint folder not found: {self.config.jsonl_source_folder_path}")
                return False
            
            # Load transfer state
            transferred_files = self._load_transfer_state()
            
            # Get JSONL files from SharePoint
            all_jsonl_files = self._get_sharepoint_jsonl_files(folder_id)
            
            if not all_jsonl_files:
                logger.info("📭 No JSONL files found in SharePoint folder")
                return True
            
            # Filter for new files
            files_to_transfer = self._filter_new_files(all_jsonl_files, transferred_files)
            
            if not files_to_transfer:
                logger.info("✅ All JSONL files have already been transferred")
                return True
            
            if dry_run:
                logger.info(f"🎭 DRY RUN: Would transfer {len(files_to_transfer)} files:")
                for file_info in files_to_transfer:
                    logger.info(f"  📄 {file_info['name']}")
                return True
            
            # Download files from SharePoint
            downloaded_files = self._download_jsonl_files(files_to_transfer)
            
            if not downloaded_files:
                logger.error("❌ No files were successfully downloaded")
                return False
            
            # Upload files to S3
            uploaded_filenames = self._upload_to_s3(downloaded_files)
            
            # Update transfer state
            if uploaded_filenames:
                transferred_files.extend(uploaded_filenames)
                self._save_transfer_state(transferred_files)
            
            # Cleanup
            self._cleanup_temp_files()
            
            logger.info("=" * 80)
            logger.info(f"✅ TRANSFER COMPLETED SUCCESSFULLY!")
            logger.info(f"📊 Processed: {len(uploaded_filenames)} JSONL files")
            logger.info(f"📤 S3 destination: s3://{self.config.s3_bucket_name}/{self.config.s3_target_prefix}")
            logger.info("=" * 80)
            
            return len(uploaded_filenames) > 0
            
        except Exception as e:
            logger.error(f"❌ Transfer failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Transfer JSONL files from SharePoint to S3 bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal transfer
  python jsonl_sharepoint_to_s3_bucket.py
  
  # Dry run to see what would be transferred
  python jsonl_sharepoint_to_s3_bucket.py --dry-run
  
  # Use custom config file
  python jsonl_sharepoint_to_s3_bucket.py --config /path/to/config.yaml
        """
    )
    
    parser.add_argument(
        '--config',
        default='/home/vision_ai_adm/code/oslo/video_convertbranch/config/sharepoint_config.yaml',
        help='Path to configuration file (default: %(default)s)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate the transfer without actually moving files'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Initialize processor and run transfer
    try:
        processor = JSONLSharePointToS3Processor(args.config)
        success = processor.run_transfer(dry_run=args.dry_run)
        
        if success:
            print("🎉 JSONL transfer completed successfully!")
            sys.exit(0)
        else:
            print("❌ JSONL transfer failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
