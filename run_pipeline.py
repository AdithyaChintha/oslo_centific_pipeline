# run_pipeline.py

import os
import sys
import shutil
import tempfile
import logging
import yaml
import time
from pathlib import Path
from datetime import datetime

# Third-party imports
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Import your pipeline's main function
from ray_pipeline_testing import pipeline_main

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AzurePipelineWrapper")
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)  # Hide HTTP request/response details
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)  # Show warnings and errors only
logging.getLogger("azure.core").setLevel(logging.WARNING)  # General Azure core logging


def _human_size(num: int, suffix="B") -> str:
    """Convert bytes to human readable format"""
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


def get_connection_string_from_yaml(config_path: str) -> str:
    """
    Parses a blobfuse2 config YAML to construct an Azure Storage connection string.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        azstorage_config = config.get('azstorage', {})
        account_name = azstorage_config.get('account-name')
        account_key = azstorage_config.get('account-key')

        if not all([account_name, account_key]):
            raise ValueError("'account-name' or 'account-key' not found in azstorage config.")
            
        connection_string = (
            f"DefaultEndpointsProtocol=https;AccountName={account_name};"
            f"AccountKey={account_key};EndpointSuffix=core.windows.net"
        )
        logger.info(f"Successfully constructed connection string for account: {account_name}")
        return connection_string, account_name, account_key

    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse YAML or construct connection string: {e}")
        raise


class AzureBlobPipeline:
    """
    A context manager to handle Azure Blob I/O for a local pipeline.
    """
    def __init__(self, connection_string: str, input_blob_path: str, input_audio_path: str, output_blob_prefix: str):
        if not connection_string:
            raise ValueError("The Azure Storage connection string is required.")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        self.input_container, self.input_blob_name = self._parse_blob_path(input_blob_path)
        self.input_audio_container, self.input_audio_blob_name = self._parse_blob_path(input_audio_path)
        self.output_container, self.output_prefix = self._parse_blob_path(output_blob_prefix)

        self.temp_input_dir = None
        self.temp_output_dir = None
        self.local_input_path = None
        self.local_audio_path = None

        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks for optimal throughput
        self.max_workers = 10  # Parallel download threads
        self.retry_attempts = 3

    @staticmethod
    def _parse_blob_path(path: str) -> tuple[str, str]:
        """Splits a 'container/path/to/blob' string into (container, path/to/blob)."""
        try:
            container, blob_name = path.split('/', 1)
            return container, blob_name
        except ValueError:
            raise ValueError(f"Invalid blob path format: '{path}'. Expected 'container/path/to/blob'.")

    def __enter__(self):
        """Prepares the local environment by creating temp dirs and downloading the input."""
        logger.info("Setting up temporary pipeline environment...")
        self.temp_input_dir = tempfile.mkdtemp(prefix="pipeline_input_")
        self.temp_output_dir = tempfile.mkdtemp(prefix="pipeline_output_")
        logger.info(f"Created temporary input dir: {self.temp_input_dir}")
        logger.info(f"Created temporary output dir: {self.temp_output_dir}")

        self._download_input_blob()
        self._download_audio_blob()

        return self.local_input_path, self.local_audio_path, self.temp_output_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Uploads results on success and always cleans up the local environment."""
        if exc_type is None:
            logger.info("Pipeline finished successfully. Uploading results...")
            self._upload_output_folder()
        else:
            logger.error(f"Pipeline failed with an exception: {exc_val}")
            logger.warning("Skipping results upload due to pipeline failure.")

        self._cleanup()
        logger.info("Cleanup complete.")

    def _download_blob_optimized(self, container: str, blob_name: str, local_path: str, description: str = "file"):
        """Optimized blob download with chunked streaming, retries, and progress"""
        
        for attempt in range(self.retry_attempts):
            try:
                blob_client = self.blob_service_client.get_blob_client(container=container, blob=blob_name)
                
                # Get blob properties for progress tracking
                blob_properties = blob_client.get_blob_properties()
                total_size = blob_properties.size
                total_size_mb = total_size / (1024 * 1024)
                
                logger.info(f"📥 Downloading {description}: {blob_name} ({total_size_mb:.1f} MB)")
                
                start_time = time.time()
                downloaded_bytes = 0
                
                with open(local_path, "wb") as f:
                    # Stream download in chunks
                    stream = blob_client.download_blob()
                    for chunk in stream.chunks():
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        
                        # Progress indication every 100MB
                        if downloaded_bytes % (100 * 1024 * 1024) == 0:
                            progress = (downloaded_bytes / total_size) * 100
                            speed_mbps = (downloaded_bytes / (1024 * 1024)) / (time.time() - start_time)
                            logger.info(f"   Progress: {progress:.1f}% ({speed_mbps:.1f} MB/s)")
                
                download_time = time.time() - start_time
                speed_mbps = total_size_mb / download_time
                logger.info(f"✅ {description} download complete: {total_size_mb:.1f} MB in {download_time:.1f}s ({speed_mbps:.1f} MB/s)")
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {blob_name}: {e}")
                if attempt == self.retry_attempts - 1:
                    logger.error(f"❌ Failed to download {blob_name} after {self.retry_attempts} attempts")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def _download_input_blob(self):
        """Downloads the source blob using optimized streaming"""
        file_name = os.path.basename(self.input_blob_name)
        self.local_input_path = os.path.join(self.temp_input_dir, file_name)
        
        self._download_blob_optimized(
            self.input_container, 
            self.input_blob_name, 
            self.local_input_path,
            "video"
        )

    def _download_audio_blob(self):
        """Downloads the audio blob using optimized streaming"""
        file_name = os.path.basename(self.input_audio_blob_name)
        self.local_audio_path = os.path.join(self.temp_input_dir, file_name)
        
        self._download_blob_optimized(
            self.input_audio_container, 
            self.input_audio_blob_name, 
            self.local_audio_path,
            "audio"
        )

    def _upload_output_folder(self):
        """Uploads all files from the local temp output dir to Azure Blob Storage."""
        if not os.path.isdir(self.temp_output_dir) or not os.listdir(self.temp_output_dir):
            logger.warning("Output directory is empty or does not exist. Nothing to upload.")
            return

        container_client = self.blob_service_client.get_container_client(self.output_container)
        
        # Check if container exists before creating it.
        if not container_client.exists():
            logger.info(f"Output container '{self.output_container}' does not exist. Creating it now.")
            container_client.create_container()

        for root, _, files in os.walk(self.temp_output_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                file_size = os.path.getsize(local_path)
                relative_path = os.path.relpath(local_path, self.temp_output_dir)
                blob_name = os.path.join(self.output_prefix, relative_path).replace("\\", "/")
                
                blob_client = container_client.get_blob_client(blob_name)
                logger.info(f"Uploading '{local_path}' to '{self.output_container}/{blob_name}'...")
                try:
                    with open(local_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                except Exception as e:
                    logger.error(f"Failed to upload '{filename}': {e}")

    def _cleanup(self):
        """Safely removes the temporary input and output directories."""
        logger.info("Cleaning up temporary directories...")
        for dir_path in [self.temp_input_dir, self.temp_output_dir]:
            if dir_path and os.path.isdir(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"Successfully removed {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to remove temporary directory {dir_path}: {e}")

def main():
    """
    The main entry point for running the Ray pipeline with Azure Blob I/O.
    """
    # ==================================================================
    # == 1. EDIT YOUR PATHS HERE                                      ==
    # ==================================================================
    # Input video path in Azure Blob Storage
    # Format: "container-name/full/path/to/video.mp4" test/input_videos/VID_20250809_094836_00_045.insv
    input_path = "instavideo/test/input_videos/VID_20250809_094836_00_045.insv"
    input_audio_path = "instavideo/test/input_videos/VID_20250809_094836_00_045.wav"
    
    # The PARENT directory for your output in Azure Blob Storage
    # Format: "container-name/path/for/all/outputs/"
    output_parent_dir = "instavideo/krishna-test/test1/test_activity/pre-annotation-output/test_08_27"
    # ==================================================================


    # ==================================================================
    # == 2. (AUTOMATIC) CREATE VIDEO-SPECIFIC OUTPUT FOLDER           ==
    # ==================================================================
    # Get the base name of the video file (e.g., "my_cool_video.mp4")
    video_filename = os.path.basename(input_path)
    # Get the video name without the extension (e.g., "my_cool_video")
    video_name_folder = os.path.splitext(video_filename)[0]

    # Create the final, specific output path by joining the parent dir and the video name.
    # The result will be e.g., "results/all-pipeline-runs/my_cool_video"
    final_output_prefix = os.path.join(output_parent_dir, video_name_folder)
    # ==================================================================

    logger.info(f"Input video: {input_path}")
    logger.info(f"Final output will be stored under: {final_output_prefix}")

    config_file = "blobfuse2_config.yaml"
    try:
        connection_string, account_name, account_key = get_connection_string_from_yaml(config_file)


    except (ValueError, FileNotFoundError):
        sys.exit(1)
    
    try:
        # The wrapper is initialized with the final, specific output path
        with AzureBlobPipeline(connection_string, input_path, input_audio_path, final_output_prefix) as (local_input_path, local_audio_path, local_output_dir):
            logger.info("Azure Blob environment is ready. Starting Ray pipeline...")
            
            # Create blob service client to pass to pipeline
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            
            # Extract container name from final output prefix
            output_container = final_output_prefix.split('/', 1)[0]
            
            pipeline_main(
                input_video_path=local_input_path,
                input_audio_path=local_audio_path,  # Now uses downloaded audio
                output_dir=local_output_dir,
                process_dual_views =  True,
                process_unwarped_views = True,
                azure_blob_client=blob_service_client,  # Pass Azure client
                azure_container=output_container,       # Pass container name
                azure_output_prefix=final_output_prefix,  # Pass output prefix
                azure_account_name=account_name,          
                azure_account_key=account_key             
            )

    except Exception as e:
        logger.critical(f"An unhandled error occurred in the pipeline wrapper: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()