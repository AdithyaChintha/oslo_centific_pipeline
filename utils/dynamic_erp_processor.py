#!/usr/bin/env python3
"""
Dynamic ERP Video and Audio Processor for Pipeline Integration.
This module provides clean integration of ERP video downscaling and audio URL generation
into the main pipeline using the standalone approach.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

from utils.logger import get_logger
from utils.video_processing import downscale_video_to_480p
from utils.erp_upload_utils import upload_erp_video_to_blob

logger = get_logger("DynamicERPProcessor")


def find_erp_video_in_output_dir(output_dir: str, session_id: str) -> Optional[str]:
    """
    Dynamically find the ERP video file in the output directory.
    
    Args:
        output_dir: Base output directory containing processed videos
        session_id: Session ID to match video file
        
    Returns:
        Optional[str]: Path to the ERP video file if found, None otherwise
    """
    try:
        logger.info(f"🔍 Searching for ERP video in output directory: {output_dir}")
        
        # Look for ERP video in 4views directory
        # The structure is: {output_dir}/{session_id}/{session_id}/4views/
        erp_patterns = [
            f"{output_dir}/{session_id}/{session_id}/4views/*ERP*.mp4",
            f"{output_dir}/{session_id}/{session_id}/4views/*erp*.mp4",
            f"{output_dir}/{session_id}/{session_id}/4views/*{session_id}*ERP*.mp4",
            f"{output_dir}/{session_id}/{session_id}/4views/*{session_id}*erp*.mp4",
            # Fallback patterns
            f"{output_dir}/4views/*ERP*.mp4",
            f"{output_dir}/4views/*erp*.mp4",
            f"{output_dir}/4views/*{session_id}*ERP*.mp4",
            f"{output_dir}/4views/*{session_id}*erp*.mp4"
        ]
        
        for pattern in erp_patterns:
            import glob
            erp_files = glob.glob(pattern)
            if erp_files:
                erp_path = erp_files[0]
                logger.info(f"✅ Found ERP video: {erp_path}")
                return erp_path
        
        logger.warning(f"⚠️ No ERP video found in {output_dir}/4views/")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error finding ERP video: {e}")
        return None


def find_audio_file_in_blob_storage(
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_base_path: str,
    session_id: str,
    account_key: str
) -> Optional[str]:
    """
    Dynamically find the audio file in blob storage for a given session.
    
    Args:
        blob_service_client: Azure BlobServiceClient instance
        container_name: Azure container name
        blob_base_path: Base blob path (e.g., "test_mulitsession/")
        session_id: Session ID to match audio file
        account_key: Azure storage account key
        
    Returns:
        Optional[str]: SAS URL to the audio file if found, None otherwise
    """
    try:
        logger.info(f"🔍 Searching for audio file in blob storage for session: {session_id}")
        logger.info(f"🔍 Blob base path: {blob_base_path}")
        logger.info(f"🔍 Container: {container_name}")
        
        # Try multiple search patterns for audio files
        # The audio files are in /instavideo/test_audio/{session_id}/, not in the processed path
        search_patterns = [
            f"test_audio/{session_id}",
            f"test_audio/{session_id}/",
            f"test_audio/{session_id}/{session_id}",
            f"test_audio/{session_id}/audio",
            f"test_audio/{session_id}/4views",
            # Also try the original blob_base_path patterns as fallback
            f"{blob_base_path.rstrip('/')}/{session_id}",
            f"{blob_base_path.rstrip('/')}/{session_id}/{session_id}",
            f"{blob_base_path.rstrip('/')}",
            f"{blob_base_path.rstrip('/')}/{session_id}/audio",
            f"{blob_base_path.rstrip('/')}/{session_id}/4views"
        ]
        
        audio_files = []
        for pattern in search_patterns:
            logger.info(f"🔍 Searching in pattern: {pattern}")
            
            try:
                blobs = blob_service_client.get_container_client(container_name).list_blobs(
                    name_starts_with=pattern
                )
                
                for blob in blobs:
                    blob_name = blob.name.lower()
                    # Look for .wav files that contain the session ID
                    if blob_name.endswith('.wav') and session_id in blob_name:
                        audio_files.append(blob.name)
                        logger.info(f"🎵 Found audio file: {blob.name}")
                        
                        # Also check for any .wav files in the same directory structure
                    elif blob_name.endswith('.wav') and pattern in blob_name:
                        audio_files.append(blob.name)
                        logger.info(f"🎵 Found audio file (pattern match): {blob.name}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Error searching pattern {pattern}: {e}")
                continue
        
        # Remove duplicates
        audio_files = list(set(audio_files))
        
        if not audio_files:
            logger.warning(f"⚠️ No audio files found for session: {session_id}")
            logger.info(f"🔍 Searched patterns: {search_patterns}")
            
            # Debug: List all blobs in the base path to understand the structure
            logger.info(f"🔍 Debug: Listing all blobs in base path: {blob_base_path}")
            try:
                all_blobs = blob_service_client.get_container_client(container_name).list_blobs(
                    name_starts_with=blob_base_path.rstrip('/')
                )
                blob_count = 0
                for blob in all_blobs:
                    if blob_count < 20:  # Limit to first 20 blobs for debugging
                        logger.info(f"🔍 Debug blob: {blob.name}")
                    blob_count += 1
                logger.info(f"🔍 Debug: Found {blob_count} total blobs in base path")
            except Exception as e:
                logger.warning(f"⚠️ Debug listing failed: {e}")
            
            return None
        
        # Use the first audio file found
        audio_blob_name = audio_files[0]
        logger.info(f"✅ Using audio file: {audio_blob_name}")
        
        # Generate SAS URL for the audio file
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=audio_blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(days=365)
        )
        
        audio_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{audio_blob_name}?{sas_token}"
        logger.info(f"✅ Audio URL generated: {audio_url}")
        
        return audio_url
        
    except Exception as e:
        logger.error(f"❌ Error finding audio file: {e}")
        return None


def process_erp_and_audio_dynamically(
    output_dir: str,
    session_id: str,
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_base_path: str,
    account_key: str,
    downscale_erp: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    """
    Dynamically process ERP video and audio files for a session.
    
    This function:
    1. Finds the ERP video file in the output directory
    2. Downscales it to 480p if requested
    3. Uploads the downscaled video to blob storage
    4. Finds the audio file in blob storage
    5. Generates SAS URLs for both files
    
    Args:
        output_dir: Base output directory containing processed videos
        session_id: Session ID
        blob_service_client: Azure BlobServiceClient instance
        container_name: Azure container name
        blob_base_path: Base blob path (e.g., "test_mulitsession/")
        account_key: Azure storage account key
        downscale_erp: Whether to downscale ERP video to 480p
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (full_video_480p_url, full_audio_link)
    """
    try:
        logger.info(f"🎬 Starting dynamic ERP and audio processing for session: {session_id}")
        
        # Step 1: Find ERP video file
        erp_video_path = find_erp_video_in_output_dir(output_dir, session_id)
        if not erp_video_path:
            logger.warning(f"⚠️ No ERP video found for session: {session_id}")
            return None, None
        
        # Step 2: Process ERP video
        full_video_480p_url = None
        if downscale_erp:
            logger.info("🎬 Downscaling ERP video to 480p...")
            downscaled_video_path = downscale_video_to_480p(
                input_video_path=erp_video_path,
                target_height=480,
                quality="medium"
            )
            
            if downscaled_video_path and os.path.exists(downscaled_video_path):
                logger.info("☁️ Uploading downscaled ERP video to blob storage...")
                video_upload_result = upload_erp_video_to_blob(
                    erp_video_path=downscaled_video_path,
                    blob_service_client=blob_service_client,
                    container_name=container_name,
                    blob_base_path=blob_base_path,
                    video_name=session_id,
                    account_key=account_key,
                    downscale_to_480p=False,  # Already downscaled
                    sas_expiry_days=365
                )
                
                if video_upload_result["success"]:
                    full_video_480p_url = video_upload_result["full_video_480P"]
                    logger.info(f"✅ ERP video uploaded: {full_video_480p_url}")
                else:
                    logger.error(f"❌ Failed to upload ERP video: {video_upload_result.get('error', 'Unknown error')}")
                
                # Clean up temporary downscaled video
                try:
                    os.remove(downscaled_video_path)
                    logger.info("🧹 Cleaned up temporary downscaled video")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to clean up temporary file: {e}")
            else:
                logger.error("❌ Failed to downscale ERP video")
        else:
            logger.info("⏭️ Skipping ERP video downscaling")
        
        # Step 3: Find and process audio file
        logger.info("🎵 Finding audio file in blob storage...")
        full_audio_link = find_audio_file_in_blob_storage(
            blob_service_client=blob_service_client,
            container_name=container_name,
            blob_base_path=blob_base_path,
            session_id=session_id,
            account_key=account_key
        )
        
        if not full_audio_link:
            logger.warning(f"⚠️ No audio file found for session: {session_id}")
            full_audio_link = ""  # Use empty string as fallback
        
        logger.info(f"🎬 Dynamic ERP and audio processing completed for session: {session_id}")
        if full_video_480p_url:
            logger.info(f"   ✅ ERP video URL: {full_video_480p_url}")
        if full_audio_link:
            logger.info(f"   ✅ Audio URL: {full_audio_link}")
        
        return full_video_480p_url, full_audio_link
        
    except Exception as e:
        logger.error(f"❌ Dynamic ERP and audio processing failed for {session_id}: {e}")
        return None, None


def integrate_erp_audio_with_labelstudio_tasks(
    output_dir: str,
    session_id: str,
    blob_service_client: BlobServiceClient,
    container_name: str,
    blob_base_path: str,
    account_key: str,
    labelstudio_tasks: list
) -> list:
    """
    Integrate ERP video and audio URLs with existing Label Studio tasks.
    
    Args:
        output_dir: Base output directory containing processed videos
        session_id: Session ID
        blob_service_client: Azure BlobServiceClient instance
        container_name: Azure container name
        blob_base_path: Base blob path (e.g., "test_mulitsession/")
        account_key: Azure storage account key
        labelstudio_tasks: List of Label Studio task file paths
        
    Returns:
        list: Updated list of Label Studio task file paths
    """
    try:
        logger.info(f"🔗 Integrating ERP and audio URLs with {len(labelstudio_tasks)} Label Studio tasks")
        
        # Process ERP video and audio
        full_video_480p_url, full_audio_link = process_erp_and_audio_dynamically(
            output_dir=output_dir,
            session_id=session_id,
            blob_service_client=blob_service_client,
            container_name=container_name,
            blob_base_path=blob_base_path,
            account_key=account_key,
            downscale_erp=True
        )
        
        # Update each Label Studio task with the new URLs
        for task_path in labelstudio_tasks:
            try:
                if os.path.exists(task_path):
                    # Read the existing task
                    with open(task_path, 'r') as f:
                        task_data = json.load(f)
                    
                    # Update the existing keys in the data section
                    if "data" in task_data:
                        task_data["data"]["full_video_480P"] = full_video_480p_url or ""
                        task_data["data"]["full_audio_link"] = full_audio_link or ""
                        
                        # Write the updated task back
                        with open(task_path, 'w') as f:
                            json.dump(task_data, f, indent=2)
                        
                        logger.info(f"✅ Updated Label Studio task: {task_path}")
                    else:
                        logger.warning(f"⚠️ No 'data' section found in task: {task_path}")
                else:
                    logger.warning(f"⚠️ Label Studio task file not found: {task_path}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to update Label Studio task {task_path}: {e}")
        
        logger.info(f"🔗 Integration completed for {len(labelstudio_tasks)} Label Studio tasks")
        return labelstudio_tasks
        
    except Exception as e:
        logger.error(f"❌ Integration failed for session {session_id}: {e}")
        return labelstudio_tasks


if __name__ == "__main__":
    print("Dynamic ERP Processor - Available functions:")
    print("- find_erp_video_in_output_dir()")
    print("- find_audio_file_in_blob_storage()")
    print("- process_erp_and_audio_dynamically()")
    print("- integrate_erp_audio_with_labelstudio_tasks()")
