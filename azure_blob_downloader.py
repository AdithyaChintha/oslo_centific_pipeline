#!/usr/bin/env python3
"""
Azure Blob Storage Downloader
Downloads specific files from Azure Blob Storage using credentials from blobfuse2_config.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import List, Optional
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError, AzureError
import logging
from tqdm import tqdm

class AzureBlobDownloader:
    def __init__(self, config_path: str = "blobfuse2_config.yaml", verbose: bool = False):
        """Initialize the downloader with configuration from blobfuse2_config.yaml"""
        self.config = self._load_config(config_path)
        self.blob_service_client = self._create_blob_client()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Suppress Azure SDK verbose logging unless explicitly requested
        if not verbose:
            logging.getLogger('azure.storage.blob').setLevel(logging.WARNING)
            logging.getLogger('azure.core').setLevel(logging.WARNING)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from blobfuse2_config.yaml"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config['azstorage']
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def _create_blob_client(self) -> BlobServiceClient:
        """Create Azure Blob Service client using configuration"""
        account_name = self.config['account-name']
        account_key = self.config['account-key']
        account_url = f"https://{account_name}.blob.core.windows.net"
        
        return BlobServiceClient(
            account_url=account_url,
            credential=account_key
        )

    def list_blobs(self, prefix: str = "") -> List[str]:
        """List all blobs in the container with optional prefix filter"""
        try:
            container_name = self.config['container']
            container_client = self.blob_service_client.get_container_client(container_name)
            
            blob_list = []
            for blob in container_client.list_blobs(name_starts_with=prefix):
                blob_list.append(blob.name)
            
            return blob_list
        except AzureError as e:
            self.logger.error(f"Error listing blobs: {e}")
            return []

    def download_file(self, blob_name: str, destination_path: str) -> bool:
        """Download a specific file from Azure Blob Storage with progress bar"""
        try:
            container_name = self.config['container']
            
            # Create destination directory if it doesn't exist
            dest_dir = Path(destination_path).parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Get blob client and download
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name, 
                blob=blob_name
            )
            
            # Get blob properties to show file size
            blob_properties = blob_client.get_blob_properties()
            file_size = blob_properties.size
            file_size_mb = file_size / (1024 * 1024)
            
            # Extract just the filename from the blob path for display
            display_name = Path(blob_name).name
            self.logger.info(f"Downloading {display_name} ({file_size_mb:.1f} MB)")
            
            # Download with progress bar
            with open(destination_path, "wb") as download_file:
                with tqdm(
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=display_name[:30] + "..." if len(display_name) > 30 else display_name
                ) as pbar:
                    download_stream = blob_client.download_blob()
                    
                    # Download in chunks to update progress
                    chunk_size = 1024 * 1024  # 1MB chunks
                    for chunk in download_stream.chunks():
                        download_file.write(chunk)
                        pbar.update(len(chunk))
            
            self.logger.info(f"✓ Successfully downloaded {display_name}")
            return True
            
        except ResourceNotFoundError:
            self.logger.error(f"Blob not found: {blob_name}")
            return False
        except AzureError as e:
            self.logger.error(f"Azure error downloading {blob_name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {blob_name}: {e}")
            return False

    def download_files(self, blob_names: List[str], destination_dir: str, preserve_paths: bool = False) -> dict:
        """Download multiple files to a destination directory"""
        results = {
            'success': [],
            'failed': []
        }
        
        destination_path = Path(destination_dir)
        destination_path.mkdir(parents=True, exist_ok=True)
        
        for blob_name in blob_names:
            if preserve_paths:
                # Preserve directory structure in destination
                file_path = destination_path / blob_name
            else:
                # Extract just the filename from the blob path
                filename = Path(blob_name).name
                file_path = destination_path / filename
                
                # Handle filename conflicts by adding a suffix
                counter = 1
                original_file_path = file_path
                while file_path.exists():
                    stem = original_file_path.stem
                    suffix = original_file_path.suffix
                    file_path = destination_path / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            if self.download_file(blob_name, str(file_path)):
                results['success'].append(blob_name)
            else:
                results['failed'].append(blob_name)
        
        return results

    def download_by_pattern(self, pattern: str, destination_dir: str, preserve_paths: bool = False) -> dict:
        """Download files matching a pattern"""
        blobs = self.list_blobs(prefix=pattern)
        
        if not blobs:
            self.logger.warning(f"No blobs found matching pattern: {pattern}")
            return {'success': [], 'failed': []}
        
        self.logger.info(f"Found {len(blobs)} blobs matching pattern '{pattern}'")
        return self.download_files(blobs, destination_dir, preserve_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Download files from Azure Blob Storage using blobfuse2 configuration"
    )
    
    parser.add_argument(
        "--config", 
        default="blobfuse2_config.yaml",
        help="Path to blobfuse2 configuration file (default: blobfuse2_config.yaml)"
    )
    
    parser.add_argument(
        "--destination", "-d",
        required=True,
        help="Destination directory for downloaded files"
    )
    
    # Mutually exclusive group for different download modes
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--files", "-f",
        nargs="+",
        help="Specific blob names to download"
    )
    
    group.add_argument(
        "--pattern", "-p",
        help="Download all files matching this prefix pattern"
    )
    
    group.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available blobs (no download)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--preserve-paths",
        action="store_true",
        help="Preserve blob directory structure in destination (default: download files directly to destination folder)"
    )

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize downloader
        downloader = AzureBlobDownloader(args.config, verbose=args.verbose)
        
        if args.list:
            # List mode
            blobs = downloader.list_blobs()
            print(f"\nFound {len(blobs)} blobs in container '{downloader.config['container']}':")
            for blob in blobs:
                print(f"  {blob}")
            return
        
        if args.files:
            # Download specific files
            print(f"Downloading {len(args.files)} files to {args.destination}")
            results = downloader.download_files(args.files, args.destination, args.preserve_paths)
        
        elif args.pattern:
            # Download by pattern
            print(f"Downloading files matching pattern '{args.pattern}' to {args.destination}")
            results = downloader.download_by_pattern(args.pattern, args.destination, args.preserve_paths)
        
        # Print results
        print(f"\nDownload Results:")
        print(f"  Successful: {len(results['success'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['success']:
            print(f"\nSuccessfully downloaded:")
            for file in results['success']:
                print(f"  ✓ {file}")
        
        if results['failed']:
            print(f"\nFailed to download:")
            for file in results['failed']:
                print(f"  ✗ {file}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
