#!/usr/bin/env python3
"""
SharePoint Excel File Uploader Script

This script automatically uploads Excel files from your configured source directory
to SharePoint input folder where the annotation team can access them.

Key Features:
- Uploads new Excel files from configurable source directory to SharePoint
- Two modes: default (skip newest) and total_file (upload all)
- Tracks uploaded files in JSON to avoid duplicates
- Uses shared SharePoint utilities for consistency
- Provides detailed logging and progress tracking
- All paths configured via sharepoint_config.yaml

Usage:
    python sharepoint_excel_uploader.py [--config CONFIG_FILE] [--dry-run]               # Default mode (skip newest)
    python sharepoint_excel_uploader.py total_file [--config CONFIG_FILE] [--dry-run]   # Total file mode (upload all)

Author: Auto-generated for Excel → SharePoint automation (Refactored)
"""

import os
import sys
import json
import glob
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

# Import shared SharePoint utilities
try:
    from utils.sharepoint_utils import (
        SharePointConfig, SharePointAuthenticator, SharePointFileManager,
        load_sharepoint_config, create_sharepoint_manager
    )
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.sharepoint_utils import (
        SharePointConfig, SharePointAuthenticator, SharePointFileManager,
        load_sharepoint_config, create_sharepoint_manager
    )

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


class ExcelUploadManager:
    """Main manager class that orchestrates the Excel file upload process."""
    
    def __init__(self, config_path: str = "config/sharepoint_config.yaml"):
        """
        Initialize upload manager with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = load_sharepoint_config(config_path)
        
        # Create SharePoint authenticator and file manager using shared utilities
        self.authenticator, self.file_manager = create_sharepoint_manager(self.config)
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.config.upload_state_file), exist_ok=True)
        
        logger.info(f"Loading configuration from {config_path}")
    
    def _load_upload_state(self) -> List[str]:
        """
        Load list of already uploaded filenames from JSON state file.
        
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
        Save list of uploaded filenames to JSON state file.
        
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
        Get list of Excel files from configured source directory structure with their modification times.
        
        Searches in the configured excel_source_dir from sharepoint_config.yaml
        and processes both subdirectories (labelstudio_csv_prod_server_1 and labelstudio_csv_prod_server_2).
        
        Returns:
            List of (file_path, modification_time) tuples sorted by modification time
        """
        excel_files = []
        
        # Use configured source directory instead of hardcoded path
        batch_3_dir = self.config.excel_source_dir
        
        if not os.path.exists(batch_3_dir):
            logger.error(f"Source directory not found: {batch_3_dir}")
            logger.error("Please verify the excel_source_dir path in your sharepoint_config.yaml")
            return []
        
        logger.info(f"Scanning for Excel files in: {batch_3_dir}")
        
        # Get all subdirectories in batch_3
        try:
            subdirectories = [d for d in os.listdir(batch_3_dir) 
                            if os.path.isdir(os.path.join(batch_3_dir, d))]
            logger.info(f"Found subdirectories in batch_3: {subdirectories}")
            
            # Scan each subdirectory for Excel files
            for subdir in subdirectories:
                subdir_path = os.path.join(batch_3_dir, subdir)
                logger.info(f"Scanning directory: {subdir_path}")
                
                # Find Excel files in this subdirectory
                for extension in ['*.xlsx', '*.xls', '*.csv']:
                    pattern = os.path.join(subdir_path, extension)
                    found_files = glob.glob(pattern)
                    
                    for file_path in found_files:
                        try:
                            mod_time = os.path.getmtime(file_path)
                            excel_files.append((file_path, mod_time))
                        except OSError as e:
                            logger.warning(f"Could not get modification time for {file_path}: {e}")
            
            # Sort files by modification time (oldest first)
            excel_files.sort(key=lambda x: x[1])
            logger.info(f"Found {len(excel_files)} total Excel files in batch_3 directory")
            
            return excel_files
            
        except Exception as e:
            logger.error(f"Error scanning batch_3 directory: {e}")
            return []
    
    def _filter_files_for_upload(self, all_files: List[Tuple[str, float]], 
                                uploaded_files: List[str], skip_newest: bool = True) -> List[str]:
        """
        Filter files based on upload state and skip newest logic.
        
        Args:
            all_files: List of (file_path, mod_time) tuples
            uploaded_files: List of already uploaded filenames
            skip_newest: If True, skip newest file from each folder
            
        Returns:
            List of file paths ready for upload
        """
        if not all_files:
            return []
        
        # Group files by their parent directory (server folder)
        files_by_folder = defaultdict(list)
        for file_path, mod_time in all_files:
            folder_name = os.path.basename(os.path.dirname(file_path))
            files_by_folder[folder_name].append((file_path, mod_time))
        
        files_to_upload = []
        
        if skip_newest:
            logger.info("Running in DEFAULT mode - skipping newest file from each folder")
        else:
            logger.info("Running in TOTAL_FILE mode - processing all files")
        
        for folder_name, folder_files in files_by_folder.items():
            # Sort files in this folder by modification time (oldest first)
            folder_files.sort(key=lambda x: x[1])
            
            if skip_newest and len(folder_files) > 0:
                # Skip the newest file (last in sorted list)
                files_to_consider = folder_files[:-1]
                newest_file = folder_files[-1]
                newest_mod_time = datetime.fromtimestamp(newest_file[1]).strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"  Skipping newest from {folder_name}: {os.path.basename(newest_file[0])} (modified: {newest_mod_time})")
            else:
                files_to_consider = folder_files
            
            # Add files that haven't been uploaded yet
            for file_path, mod_time in files_to_consider:
                filename = os.path.basename(file_path)
                if filename not in uploaded_files:
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
            True if successful, False otherwise
        """
        logger.info("============================================================")
        logger.info("Starting Excel File Upload to SharePoint")
        logger.info("============================================================")
        
        try:
            # Get SharePoint input folder ID using shared utilities
            input_folder_id = self.file_manager.get_folder_id(self.config.input_folder_path)
            if not input_folder_id:
                logger.error("Cannot proceed without valid input folder ID")
                return False
            
            # Load current upload state (simple list of uploaded filenames)
            uploaded_files = self._load_upload_state()
            
            # Get local Excel files
            all_files = self._get_local_excel_files()
            if not all_files:
                logger.warning("No Excel files found in configured source directories")
                return True
            
            # Filter files for upload
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
            
            # Upload files using shared file manager
            successful_uploads = []
            failed_uploads = []
            
            for i, file_path in enumerate(files_to_upload, 1):
                filename = os.path.basename(file_path)
                logger.info(f"Uploading file {i}/{len(files_to_upload)}: {filename}")
                
                try:
                    # Upload file using shared utilities
                    success = self.file_manager.upload_file(
                        local_file_path=file_path,
                        sharepoint_filename=filename,
                        folder_id=input_folder_id
                    )
                    
                    if success:
                        successful_uploads.append(filename)
                        logger.info(f"✅ Successfully uploaded: {filename}")
                    else:
                        failed_uploads.append(filename)
                        logger.error(f"❌ Failed to upload: {filename}")
                        
                except Exception as e:
                    failed_uploads.append(filename)
                    logger.error(f"❌ Error uploading {filename}: {e}")
                
                # Small delay between uploads
                time.sleep(1)
            
            # Update upload state with successful uploads
            if successful_uploads:
                updated_uploaded_files = uploaded_files + successful_uploads
                self._save_upload_state(updated_uploaded_files)
            
            # Summary
            logger.info("============================================================")
            logger.info("Upload Summary:")
            logger.info(f"✅ Successful uploads: {len(successful_uploads)}")
            logger.info(f"❌ Failed uploads: {len(failed_uploads)}")
            logger.info("============================================================")
            
            return len(failed_uploads) == 0  # Success if no failures
            
        except Exception as e:
            logger.error(f"Upload process error: {e}")
            return False


def main():
    """Main function to run the Excel uploader with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload Excel files to SharePoint')
    parser.add_argument('mode', nargs='?', default='default', 
                       choices=['default', 'total_file'],
                       help='Upload mode: default (skip newest) or total_file (upload all)')
    parser.add_argument('--config', default='config/sharepoint_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be uploaded without actually uploading')
    
    args = parser.parse_args()
    
    # Determine skip_newest based on mode
    skip_newest = args.mode == 'default'
    
    try:
        # Create and run uploader
        uploader = ExcelUploadManager(args.config)
        success = uploader.run_upload(dry_run=args.dry_run, skip_newest=skip_newest)
        
        if success:
            logger.info("Excel upload completed successfully!")
            sys.exit(0)
        else:
            logger.error("Excel upload completed with errors!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
