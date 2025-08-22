# run_pipeline.py

import os
import sys
import shutil
import tempfile
import logging
import yaml

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
        return connection_string

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
    def __init__(self, connection_string: str, input_blob_path: str, output_blob_prefix: str):
        if not connection_string:
            raise ValueError("The Azure Storage connection string is required.")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        self.input_container, self.input_blob_name = self._parse_blob_path(input_blob_path)
        self.output_container, self.output_prefix = self._parse_blob_path(output_blob_prefix)

        self.temp_input_dir = None
        self.temp_output_dir = None
        self.local_input_path = None

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

        return self.local_input_path, self.temp_output_dir

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

    def _download_input_blob(self):
        """Downloads the source blob to the local temporary input directory."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.input_container, blob=self.input_blob_name
            )
            
            file_name = os.path.basename(self.input_blob_name)
            self.local_input_path = os.path.join(self.temp_input_dir, file_name)
            
            logger.info(f"Downloading '{self.input_container}/{self.input_blob_name}' to '{self.local_input_path}'...")
            with open(self.local_input_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            logger.info("Download complete.")

        except ResourceNotFoundError:
            logger.error(f"Input blob not found: '{self.input_container}/{self.input_blob_name}'")
            raise

    def _upload_output_folder(self):
        """Uploads all files from the local temp output dir to Azure Blob Storage."""
        if not os.path.isdir(self.temp_output_dir) or not os.listdir(self.temp_output_dir):
            logger.warning("Output directory is empty or does not exist. Nothing to upload.")
            return

        container_client = self.blob_service_client.get_container_client(self.output_container)
        
        # CORRECTED CODE: Check if container exists before creating it.
        if not container_client.exists():
            logger.info(f"Output container '{self.output_container}' does not exist. Creating it now.")
            container_client.create_container()

        for root, _, files in os.walk(self.temp_output_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
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
    # Format: "container-name/full/path/to/video.mp4"
    input_path = "instavideo/krishna-test/test1/Making Bed/VID_20250720_152154_00_011.insv"
    
    # The PARENT directory for your output in Azure Blob Storage
    # Format: "container-name/path/for/all/outputs/"
    output_parent_dir = "instavideo/krishna-test/test1/test_activity/pre-annotation-output"
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
        connection_string = get_connection_string_from_yaml(config_file)
    except (ValueError, FileNotFoundError):
        sys.exit(1)
    
    try:
        # The wrapper is initialized with the final, specific output path
        with AzureBlobPipeline(connection_string, input_path, final_output_prefix) as (local_input_path, local_output_dir):
            logger.info("Azure Blob environment is ready. Starting Ray pipeline...")
            
            pipeline_main(
                input_video_path=local_input_path,
                output_dir=local_output_dir
            )

    except Exception as e:
        logger.critical(f"An unhandled error occurred in the pipeline wrapper: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()