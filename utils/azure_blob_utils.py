"""
Azure Blob Storage utilities for video processing pipeline.
Uses the existing credential system from run_pipeline.py
"""
import os
from typing import List
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError


def list_blobs(connection_string: str, container_name: str, blob_prefix: str = "", file_extension: str = None) -> List[str]:
    """
    List blobs in Azure Storage container with optional prefix and extension filtering.
    
    Args:
        connection_string: Azure Storage connection string
        container_name: Name of the Azure Storage container
        blob_prefix: Prefix to filter blobs (e.g., "path/to/folder/")
        file_extension: File extension to filter by (e.g., ".insv", ".mp4")
        
    Returns:
        List of blob names matching the criteria
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        blob_names = []
        blob_list = container_client.list_blobs(name_starts_with=blob_prefix)
        
        for blob in blob_list:
            blob_name = blob.name
            
            # Apply extension filter if specified
            if file_extension and not blob_name.lower().endswith(file_extension.lower()):
                continue
                
            blob_names.append(blob_name)
        
        return blob_names
        
    except Exception as e:
        raise Exception(f"Failed to list blobs in container '{container_name}': {e}")


def download_blob(connection_string: str, container_name: str, blob_name: str, local_path: str) -> str:
    """
    Download a blob from Azure Storage to local file system.
    
    Args:
        connection_string: Azure Storage connection string
        container_name: Name of the Azure Storage container
        blob_name: Name of the blob to download
        local_path: Local file path to save the downloaded blob
        
    Returns:
        Path to the downloaded file
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download the blob
        with open(local_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        
        return local_path
        
    except ResourceNotFoundError:
        raise FileNotFoundError(f"Blob not found: '{container_name}/{blob_name}'")
    except Exception as e:
        raise Exception(f"Failed to download blob '{container_name}/{blob_name}': {e}")
