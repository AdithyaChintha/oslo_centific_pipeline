#!/usr/bin/env python3
"""
Utilities for uploading ERP videos and generating blob URLs for Label Studio tasks.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from utils.logger import get_logger
from utils.video_processing import downscale_erp_video_if_needed

logger = get_logger("ERPUploadUtils")


def upload_erp_video_to_blob(
    erp_video_path: str,
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_base_path: str,
    video_name: str,
    account_key: str,
    downscale_to_480p: bool = True,
    sas_expiry_days: int = 365
) -> Dict[str, Any]:
    """
    Upload ERP video to Azure blob storage and generate SAS URL.
    
    Args:
        erp_video_path: Path to the ERP video file
        blob_service_client: Azure BlobServiceClient instance
        container_name: Azure container name
        blob_base_path: Base path for blob storage
        video_name: Name of the video (used in blob path)
        downscale_to_480p: Whether to downscale to 480p before upload
        sas_expiry_days: Number of days for SAS token expiry
        
    Returns:
        Dict containing upload result with blob URL and metadata
    """
    try:
        # Check if ERP video exists
        erp_path = Path(erp_video_path)
        if not erp_path.exists():
            raise FileNotFoundError(f"ERP video not found: {erp_video_path}")
        
        # Determine the video to upload (original or downscaled)
        if downscale_to_480p:
            logger.info(f"Downscaling ERP video to 480p: {erp_path.name}")
            video_to_upload = downscale_erp_video_if_needed(str(erp_path))
            is_downscaled = video_to_upload != str(erp_path)
        else:
            video_to_upload = str(erp_path)
            is_downscaled = False
        
        # Generate blob name
        video_filename = Path(video_to_upload).name
        blob_name = f"{blob_base_path.strip('/')}/{video_name}/4views/{video_filename}"
        
        # Get file size
        file_size = os.path.getsize(video_to_upload)
        logger.info(f"📤 Uploading ERP video: {video_filename} ({file_size / (1024*1024):.1f} MB) -> {blob_name}")
        
        # Upload file to blob
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        with open(video_to_upload, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        logger.info(f"✅ Successfully uploaded ERP video to blob: {blob_name}")
        
        # Generate SAS URL
        sas_url = generate_erp_video_sas_url(
            blob_service_client, container_name, blob_name, account_key, sas_expiry_days
        )
        
        return {
            "success": True,
            "blob_name": blob_name,
            "blob_url": sas_url,
            "full_video_480P": sas_url,  # Add the expected key for Label Studio integration
            "local_path": video_to_upload,
            "file_size_bytes": file_size,
            "is_downscaled": is_downscaled,
            "original_path": str(erp_path)
        }
        
    except Exception as e:
        error_msg = f"Failed to upload ERP video: {e}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "blob_url": None
        }


def generate_erp_video_sas_url(
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_name: str,
    account_key: str,
    expiry_days: int = 365
) -> str:
    """
    Generate SAS URL for ERP video blob.
    
    Args:
        blob_service_client: Azure BlobServiceClient instance
        container_name: Azure container name
        blob_name: Blob name
        account_key: Azure storage account key
        expiry_days: Number of days for SAS token expiry
        
    Returns:
        str: SAS URL for the blob
    """
    try:
        # Get account name from the client
        account_name = blob_service_client.account_name
        
        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(days=expiry_days)
        )
        
        # Construct full SAS URL
        sas_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        
        logger.debug(f"Generated SAS URL for ERP video: {blob_name}")
        return sas_url
        
    except Exception as e:
        error_msg = f"Failed to generate SAS URL: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def find_erp_video_in_4views_dir(output_dir: str, video_name: str) -> Optional[str]:
    """
    Find the ERP video file in the 4views directory.
    
    Args:
        output_dir: Base output directory
        video_name: Name of the video
        
    Returns:
        Optional[str]: Path to ERP video file if found, None otherwise
    """
    try:
        # Look for ERP video in 4views directory
        views_dir = Path(output_dir) / video_name / "4views"
        
        if not views_dir.exists():
            logger.warning(f"4views directory not found: {views_dir}")
            return None
        
        # Look for ERP video files (various naming patterns)
        erp_patterns = [
            "*_ERP_*.mp4",
            "*erp*.mp4",
            "*ERP*.mp4"
        ]
        
        for pattern in erp_patterns:
            erp_files = list(views_dir.glob(pattern))
            if erp_files:
                # Return the first match
                erp_path = erp_files[0]
                logger.info(f"Found ERP video: {erp_path}")
                return str(erp_path)
        
        logger.warning(f"No ERP video found in {views_dir}")
        return None
        
    except Exception as e:
        logger.error(f"Error finding ERP video: {e}")
        return None


def get_full_audio_url_from_session(
    session_audio_path: str,
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_base_path: str,
    video_name: str,
    sas_expiry_days: int = 365
) -> Optional[str]:
    """
    Get the full audio file URL from the session.
    
    Args:
        session_audio_path: Path to the session audio file
        blob_service_client: Azure BlobServiceClient instance
        container_name: Azure container name
        blob_base_path: Base path for blob storage
        video_name: Name of the video
        sas_expiry_days: Number of days for SAS token expiry
        
    Returns:
        Optional[str]: SAS URL for the audio file if found and uploaded, None otherwise
    """
    try:
        audio_path = Path(session_audio_path)
        if not audio_path.exists():
            logger.warning(f"Session audio file not found: {session_audio_path}")
            return None
        
        # Generate blob name for audio file
        audio_filename = audio_path.name
        blob_name = f"{blob_base_path.strip('/')}/{video_name}/{audio_filename}"
        
        # Get file size
        file_size = os.path.getsize(session_audio_path)
        logger.info(f"📤 Uploading session audio: {audio_filename} ({file_size / (1024*1024):.1f} MB) -> {blob_name}")
        
        # Upload audio file to blob
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        with open(session_audio_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        logger.info(f"✅ Successfully uploaded session audio to blob: {blob_name}")
        
        # Generate SAS URL
        sas_url = generate_erp_video_sas_url(
            blob_service_client, container_name, blob_name, account_key, sas_expiry_days
        )
        
        return sas_url
        
    except Exception as e:
        logger.error(f"Failed to upload session audio: {e}")
        return None


def process_erp_and_audio_for_labelstudio(
    output_dir: str,
    video_name: str,
    session_audio_path: str,
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_base_path: str,
    downscale_erp: bool = True
) -> Dict[str, Any]:
    """
    Process ERP video and session audio for Label Studio tasks.
    
    Args:
        output_dir: Base output directory
        video_name: Name of the video
        session_audio_path: Path to the session audio file
        blob_service_client: Azure BlobServiceClient instance
        container_name: Azure container name
        blob_base_path: Base path for blob storage
        downscale_erp: Whether to downscale ERP video to 480p
        
    Returns:
        Dict containing full_video_480P and full_audio_link URLs
    """
    result = {
        "full_video_480P": None,
        "full_audio_link": None,
        "success": False,
        "errors": []
    }
    
    try:
        # Find and upload ERP video
        erp_video_path = find_erp_video_in_4views_dir(output_dir, video_name)
        if erp_video_path:
            erp_result = upload_erp_video_to_blob(
                erp_video_path=erp_video_path,
                blob_service_client=blob_service_client,
                container_name=container_name,
                blob_base_path=blob_base_path,
                video_name=video_name,
                downscale_to_480p=downscale_erp
            )
            
            if erp_result["success"]:
                result["full_video_480P"] = erp_result["blob_url"]
                logger.info(f"✅ ERP video URL generated: {result['full_video_480P']}")
            else:
                result["errors"].append(f"ERP video upload failed: {erp_result.get('error', 'Unknown error')}")
        else:
            result["errors"].append("ERP video not found in 4views directory")
        
        # Upload session audio
        if session_audio_path:
            audio_url = get_full_audio_url_from_session(
                session_audio_path=session_audio_path,
                blob_service_client=blob_service_client,
                container_name=container_name,
                blob_base_path=blob_base_path,
                video_name=video_name
            )
            
            if audio_url:
                result["full_audio_link"] = audio_url
                logger.info(f"✅ Session audio URL generated: {result['full_audio_link']}")
            else:
                result["errors"].append("Failed to upload session audio")
        else:
            result["errors"].append("Session audio path not provided")
        
        # Determine overall success
        result["success"] = result["full_video_480P"] is not None and result["full_audio_link"] is not None
        
        if result["success"]:
            logger.info("✅ Successfully processed ERP video and audio for Label Studio")
        else:
            logger.warning(f"⚠️ Partial success - errors: {result['errors']}")
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to process ERP and audio: {e}"
        logger.error(error_msg)
        result["errors"].append(error_msg)
        return result


if __name__ == "__main__":
    # Test the functions
    print("ERP Upload Utils - Test functions available:")
    print("- upload_erp_video_to_blob()")
    print("- generate_erp_video_sas_url()")
    print("- find_erp_video_in_4views_dir()")
    print("- get_full_audio_url_from_session()")
    print("- process_erp_and_audio_for_labelstudio()")
