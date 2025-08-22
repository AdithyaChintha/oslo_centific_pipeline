import os
import ray
import pytest
import yaml
from azure.storage.blob import BlobServiceClient

from ray_jobs.video_splitter import split_and_downscale_insv

# --- Configuration ---
CONFIG_PATH = "config/azure_blob.yaml"

# Mark the test to be skipped if the config file is not available
pytestmark = pytest.mark.skipif(
    not os.path.exists(CONFIG_PATH),
    reason=f"Azure config file not found at: {CONFIG_PATH}"
)

def load_azure_config(config_path):
    """Loads Azure credentials from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        pytest.fail(f"Config file not found: {config_path}")
    except Exception as e:
        pytest.fail(f"Error reading config file: {e}")

def get_azure_connection_string(config):
    """Constructs the Azure Storage connection string."""
    account_name = config.get("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = config.get("AZURE_STORAGE_ACCOUNT_KEY")
    if not account_name or not account_key:
        pytest.fail("AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY must be in the config.")
    return (
        f"DefaultEndpointsProtocol=https;AccountName={account_name};"
        f"AccountKey={account_key};EndpointSuffix=core.windows.net"
    )

def get_azure_blob_urls(connection_string, container_name, blob_prefix, file_extension=".insv"):
    """Lists blobs in a given container and returns their URLs."""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        blob_urls = []
        print(f"Listing blobs in container '{container_name}' with prefix '{blob_prefix}'...")
        blobs = container_client.list_blobs(name_starts_with=blob_prefix)
        
        for blob in blobs:
            if blob.name.lower().endswith(file_extension):
                blob_url = f"https://{container_client.account_name}.blob.core.windows.net/{container_name}/{blob.name}"
                blob_urls.append(blob_url)
        
        if not blob_urls:
            print(f"Warning: No blobs with extension '{file_extension}' found at prefix '{blob_prefix}'.")
            
        return blob_urls
    except Exception as e:
        pytest.fail(f"Failed to connect to Azure and list blobs: {e}")

@pytest.fixture(scope="module")
def azure_config():
    """Fixture to load Azure config for the test module."""
    return load_azure_config(CONFIG_PATH)

@pytest.fixture(scope="module")
def ray_init():
    """Initialize and shutdown Ray for the test module."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()

def test_split_and_downscale_multiple_videos(ray_init, azure_config):
    """
    Tests the split_and_downscale_insv function on multiple videos from Azure Blob Storage.
    """
    # --- Test Setup ---
    connection_string = get_azure_connection_string(azure_config)
    input_container_name = azure_config.get("AZURE_CONTAINER_NAME", "instavideo")
    
    # Correctly construct the prefix from your request
    input_blob_prefix = "instavideo/azure_directory_path/DCIM/Camera01/"
    
    output_container_name = "instavideo"
    output_blob_prefix = "krishna-test/test1/test_activity/pre-annotation-output"

    # 1. Get the list of .insv video URLs from Azure
    video_urls = get_azure_blob_urls(connection_string, input_container_name, input_blob_prefix)
    
    assert video_urls, f"No .insv videos found in {input_container_name}/{input_blob_prefix}. Test cannot proceed."
    
    print(f"Found {len(video_urls)} videos to process. Starting tasks...")

    # 2. Launch a Ray task for each video
    tasks = []
    for video_url in video_urls:
        task = split_and_downscale_insv.remote(
            azure_blob_url=video_url,
            azure_container_name=input_container_name,
            azure_connection_string=connection_string,
            output_container_name=output_container_name,
            output_blob_prefix=output_blob_prefix,
            duration_sec=60,
            downscale_to_480p=True
        )
        tasks.append(task)

    # 3. Wait for all tasks to complete and get results
    results = ray.get(tasks)
    
    assert len(results) == len(video_urls)
    print("\n--- Task Results ---")
    for i, result in enumerate(results):
        print(f"Video {i+1}: {result.get('input_video')}")
        print(f"  Success: {result.get('success')}")
        if result.get('success'):
            print(f"  Shards Created: {result.get('shard_count')}")
            print(f"  Output Location: {result.get('output_prefix')}")
        else:
            print(f"  Error: {result.get('error')}")

    # 4. Verify the results
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(output_container_name)
    
    for result in results:
        assert result["success"], f"Processing failed for video: {result['input_video']} with error: {result.get('error')}"
        
        video_name_without_ext = os.path.splitext(os.path.basename(result["input_video"]))[0]
        expected_prefix = f"{output_blob_prefix}/{video_name_without_ext}/"
        
        print(f"\nVerifying output for: {video_name_without_ext}")
        print(f"  - Checking for blobs with prefix: {expected_prefix}")
        
        shards = list(container_client.list_blobs(name_starts_with=expected_prefix))
        
        assert len(shards) > 0, f"No shards found in Azure for video {video_name_without_ext} at prefix {expected_prefix}"
        assert len(shards) == result["shard_count"], \
            f"Mismatch in shard count for {video_name_without_ext}. Expected {result['shard_count']}, found {len(shards)} in blob storage."
            
        print(f"  - Verified: Found {len(shards)} shards in blob storage, which matches the expected count.")

    print("\n✅ Test passed successfully for all videos.")

# To run this test:
# 1. Make sure you have pytest and pyyaml installed: pip install pytest pyyaml
# 2. Ensure 'config/azure_blob.yaml' is present.
# 3. Run pytest from the root directory of the project: pytest test/test_video_splitter.py
