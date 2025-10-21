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
- Uses shared SharePoint utilities for consistency
- Provides detailed logging and progress tracking

Usage:
    python sharepoint_output_processor_simplified.py [--config CONFIG_FILE] [--dry-run]

Author: Simplified for batch processing workflow (Refactored)
"""

import os
import sys
import json
import yaml
import requests
import shutil
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from pathlib import Path

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


class SimplifiedOutputProcessor:
    """
    Simplified processor that downloads all Excel files, converts to JSONL, and uploads result.
    
    This processor uses a simple batch approach:
    1. Download ALL Excel files from SharePoint output folder
    2. Convert entire batch to single JSONL file
    3. Upload JSONL to SharePoint delivery folder
    """
    
    def __init__(self, config_path: str = "config/sharepoint_config.yaml"):
        """
        Initialize processor with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = load_sharepoint_config(config_path)
        
        # Create SharePoint authenticator and file manager using shared utilities
        self.authenticator, self.file_manager = create_sharepoint_manager(self.config)
        
        # Ensure local directories exist
        os.makedirs(self.config.download_dir, exist_ok=True)
        os.makedirs(self.config.jsonl_output_dir, exist_ok=True)
        
        logger.info(f"Loading configuration from {config_path}")
    
    def _download_all_excel_files(self, output_folder_id: str, local_download_dir: str) -> List[str]:
        """
        Download all Excel files from SharePoint output folder.
        
        Args:
            output_folder_id: SharePoint folder ID to download from
            local_download_dir: Local directory to save files
            
        Returns:
            List of local file paths downloaded
        """
        downloaded_files = []
        
        try:
            # List all files in the output folder using shared utilities
            files = self.file_manager.list_files_in_folder(
                folder_id=output_folder_id,
                file_extensions=['.xlsx', '.xls', '.csv']
            )
            
            logger.info(f"Found {len(files)} Excel/CSV files")
            
            if not files:
                logger.info("No Excel files found in output folder")
                return downloaded_files
            
            # Download each file
            for i, file_info in enumerate(files, 1):
                file_name = file_info['name']
                download_url = file_info.get('download_url')
                
                if not download_url:
                    logger.warning(f"No download URL for file: {file_name}")
                    continue
                
                local_file_path = os.path.join(local_download_dir, file_name)
                logger.info(f"Downloading file {i}/{len(files)}: {file_name}")
                
                # Download using shared utilities
                success = self.file_manager.download_file(download_url, local_file_path)
                
                if success:
                    downloaded_files.append(local_file_path)
                    logger.info(f"✅ Downloaded: {file_name}")
                else:
                    logger.error(f"❌ Failed to download: {file_name}")
            
            logger.info(f"Successfully downloaded {len(downloaded_files)} files")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Error downloading files: {e}")
            return downloaded_files
    
    def _convert_excel_to_jsonl_batch(self, excel_folder_path: str) -> Optional[str]:
        """
        Convert entire folder of Excel files to single JSONL file using existing converter.
        
        Args:
            excel_folder_path: Path to folder containing Excel files
            
        Returns:
            Path to generated JSONL file or None if conversion failed
        """
        try:
            logger.info(f"Converting Excel files from folder: {excel_folder_path}")
            
            # Use existing conversion function in directory mode
            # This will process all Excel files in the folder and create a single JSONL output
            jsonl_output_path = convert_csv_to_json(excel_folder_path)
            
            if jsonl_output_path and os.path.exists(jsonl_output_path):
                logger.info(f"✅ Conversion successful: {jsonl_output_path}")
                return jsonl_output_path
            else:
                logger.error("❌ Conversion failed - no output file generated")
                return None
                
        except Exception as e:
            logger.error(f"Error during Excel to JSONL conversion: {e}")
            return None
    
    def run_processing(self, dry_run: bool = False) -> bool:
        """
        Main method to execute the simplified processing workflow.
        
        Args:
            dry_run: If True, show what would be processed without actual processing
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("======================================================================")
        logger.info("Starting Simplified SharePoint Output Processing")
        logger.info("(Batch Download → Folder Convert → JSONL Upload)")
        logger.info("======================================================================")
        
        try:
            # Get SharePoint folder IDs using shared utilities
            output_folder_id = self.file_manager.get_folder_id(self.config.output_folder_path)
            if not output_folder_id:
                logger.error("Cannot find SharePoint output folder")
                return False
            
            delivery_folder_id = self.file_manager.get_folder_id(self.config.delivery_folder_path)
            if not delivery_folder_id:
                logger.error("Cannot find SharePoint delivery folder")
                return False
            
            # Clear and prepare download directory
            if os.path.exists(self.config.download_dir):
                shutil.rmtree(self.config.download_dir)
            os.makedirs(self.config.download_dir, exist_ok=True)
            
            # List files that would be processed
            files = self.file_manager.list_files_in_folder(
                folder_id=output_folder_id,
                file_extensions=['.xlsx', '.xls', '.csv']
            )
            
            logger.info(f"Found {len(files)} Excel files to process")
            
            if dry_run:
                logger.info("DRY RUN MODE - Files that would be processed:")
                for file_info in files:
                    logger.info(f"  - {file_info['name']}")
                logger.info("DRY RUN - No actual processing performed")
                return True
            
            if not files:
                logger.info("No files to process")
                return True
            
            # Step 1: Download all Excel files
            logger.info("Step 1: Downloading all Excel files from SharePoint...")
            downloaded_files = self._download_all_excel_files(output_folder_id, self.config.download_dir)
            
            if not downloaded_files:
                logger.error("No files were downloaded successfully")
                return False
            
            # Step 2: Convert entire folder to JSONL
            logger.info("Step 2: Converting Excel files to JSONL...")
            jsonl_file_path = self._convert_excel_to_jsonl_batch(self.config.download_dir)
            
            if not jsonl_file_path:
                logger.error("JSONL conversion failed")
                return False
            
            # Step 3: Upload JSONL file to SharePoint delivery folder
            logger.info("Step 3: Uploading JSONL file to SharePoint delivery folder...")
            jsonl_filename = os.path.basename(jsonl_file_path)
            
            success = self.file_manager.upload_file(
                local_file_path=jsonl_file_path,
                sharepoint_filename=jsonl_filename,
                folder_id=delivery_folder_id
            )
            
            if success:
                logger.info(f"✅ Successfully uploaded JSONL file: {jsonl_filename}")
            else:
                logger.error(f"❌ Failed to upload JSONL file: {jsonl_filename}")
                return False
            
            # Summary
            logger.info("======================================================================")
            logger.info("Processing Summary:")
            logger.info(f"✅ Excel files downloaded: {len(downloaded_files)}")
            logger.info(f"✅ JSONL file generated: {jsonl_filename}")
            logger.info(f"✅ JSONL file uploaded to SharePoint delivery folder")
            logger.info("======================================================================")
            
            return True
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return False


def main():
    """Main function to run the simplified output processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process annotated Excel files from SharePoint')
    parser.add_argument('--config', default='config/sharepoint_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actual processing')
    
    args = parser.parse_args()
    
    try:
        # Create and run processor
        processor = SimplifiedOutputProcessor(args.config)
        success = processor.run_processing(dry_run=args.dry_run)
        
        if success:
            logger.info("All processing completed successfully!")
            sys.exit(0)
        else:
            logger.error("Processing completed with errors!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
