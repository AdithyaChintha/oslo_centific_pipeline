# Azure Blob Storage Utilities
import os
import json
import time
import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from azure.storage.blob import generate_blob_sas, BlobSasPermissions, BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from utils.logger import get_logger

logger = get_logger("BlobUtils")

# =============================================================================
# AZURE CONFIGURATION FUNCTIONS
# =============================================================================

def load_azure_config(config_path: str) -> dict:
    """Load Azure configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"🔍 Raw config keys: {list(config.keys())}")
            if 'azstorage' not in config:
                raise KeyError(f"'azstorage' key not found in config. Available keys: {list(config.keys())}")
            azstorage_config = config['azstorage']
            logger.info(f"🔍 Azstorage config keys: {list(azstorage_config.keys())}")
            return azstorage_config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")
    except KeyError as e:
        raise KeyError(f"Configuration structure error: {e}")


def create_azure_blob_client(config: dict) -> BlobServiceClient:
    """Create Azure Blob Service client"""
    try:
        # Extract Azure storage config - handle both direct and nested structures
        if 'azstorage' in config:
            # Nested structure from blobfuse2_config.yaml
            az_config = config['azstorage']
        else:
            # Direct structure
            az_config = config
            
        account_name = az_config['account-name']
        account_key = az_config['account-key']
        
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        logger.info("✅ Azure blob service client created")
        return blob_service_client
    except Exception as e:
        logger.error(f"❌ Failed to create Azure blob client: {e}")
        logger.error(f"Config structure: {list(config.keys())}")
        raise


# =============================================================================
# VIDEO CHECKLIST FUNCTIONS
# =============================================================================

def load_video_checklist(checklist_path: str) -> dict:
    """Load or create video processing checklist"""
    if os.path.exists(checklist_path):
        try:
            with open(checklist_path, 'r') as f:
                checklist = json.load(f)
            
            # Validate and fix counters when loading
            validate_and_fix_checklist_counters(checklist)
            
            logger.info(f"📋 Loaded existing checklist: {checklist['total_videos']} total videos")
            return checklist
        except Exception as e:
            logger.warning(f"⚠️ Failed to load checklist: {e}")
    
    # Create new checklist
    checklist = {
        "total_videos": 0,
        "completed_videos": 0,
        "failed_videos": 0,
        "last_updated": datetime.now().isoformat(),
        "videos": {}
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(checklist_path), exist_ok=True)
    
    # Save the new checklist to disk
    try:
        with open(checklist_path, 'w') as f:
            json.dump(checklist, f, indent=2)
        logger.info(f"📋 Created and saved new video checklist to: {checklist_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save new checklist: {e}")
    
    return checklist


def update_video_status(checklist: dict, video_name: str, status: str, **kwargs):
    """Update video status in checklist"""
    if video_name not in checklist["videos"]:
        checklist["videos"][video_name] = {
            "status": "discovered",
            "discovery_date": datetime.now().isoformat()
        }
        checklist["total_videos"] += 1
    
    video_info = checklist["videos"][video_name]
    old_status = video_info.get("status", "unknown")
    video_info["status"] = status
    video_info["last_updated"] = datetime.now().isoformat()
    
    # Add any additional fields
    for key, value in kwargs.items():
        video_info[key] = value
    
    # Update counters properly - only count completed and failed videos
    # Reset counters first
    if old_status == "completed":
        checklist["completed_videos"] = max(0, checklist["completed_videos"] - 1)
    elif old_status == "failed":
        checklist["failed_videos"] = max(0, checklist["failed_videos"] - 1)
    
    # Then add to appropriate counter
    if status == "completed":
        checklist["completed_videos"] += 1
    elif status == "failed":
        checklist["failed_videos"] += 1
    
    checklist["last_updated"] = datetime.now().isoformat()


def validate_and_fix_checklist_counters(checklist: dict):
    """Validate and fix checklist counters to ensure consistency"""
    # Count actual completed and failed videos
    actual_completed = sum(1 for info in checklist["videos"].values() if info.get("status") == "completed")
    actual_failed = sum(1 for info in checklist["videos"].values() if info.get("status") == "failed")
    
    # Fix counters if they're inconsistent
    if checklist["completed_videos"] != actual_completed:
        logger.warning(f"⚠️ Fixed completed_videos counter: {checklist['completed_videos']} -> {actual_completed}")
        checklist["completed_videos"] = actual_completed
    
    if checklist["failed_videos"] != actual_failed:
        logger.warning(f"⚠️ Fixed failed_videos counter: {checklist['failed_videos']} -> {actual_failed}")
        checklist["failed_videos"] = actual_failed


def save_video_checklist(checklist_path: str, checklist: dict):
    """Save video checklist to file"""
    try:
        # Validate and fix counters before saving
        validate_and_fix_checklist_counters(checklist)
        
        with open(checklist_path, 'w') as f:
            json.dump(checklist, f, indent=2)
        logger.debug(f"💾 Saved checklist to: {checklist_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save checklist: {e}")


def get_pending_videos(checklist: Dict) -> List[str]:
    """Get list of video names that need processing"""
    pending_videos = []
    for video_name, info in checklist["videos"].items():
        if info["status"] in ["pending", "failed"]:
            pending_videos.append(video_name)
    return pending_videos


def get_checklist_statistics(checklist: Dict) -> Dict:
    """Get detailed statistics about the checklist"""
    stats = {
        "total_videos": len(checklist["videos"]),
        "completed_videos": 0,
        "failed_videos": 0,
        "pending_videos": 0,
        "processing_videos": 0,
        "discovered_videos": 0,
        "status_breakdown": {}
    }
    
    for video_name, info in checklist["videos"].items():
        status = info.get("status", "unknown")
        stats["status_breakdown"][status] = stats["status_breakdown"].get(status, 0) + 1
        
        if status == "completed":
            stats["completed_videos"] += 1
        elif status == "failed":
            stats["failed_videos"] += 1
        elif status == "pending":
            stats["pending_videos"] += 1
        elif status == "processing":
            stats["processing_videos"] += 1
        elif status == "discovered":
            stats["discovered_videos"] += 1
    
    return stats


def detect_stuck_videos(checklist: Dict, max_processing_hours: int = 2) -> List[str]:
    """Detect videos that have been stuck in 'processing' status for too long"""
    stuck_videos = []
    current_time = datetime.now()
    
    for video_name, info in checklist["videos"].items():
        if info.get("status") == "processing":
            last_updated_str = info.get("last_updated")
            if last_updated_str:
                try:
                    last_updated = datetime.fromisoformat(last_updated_str)
                    hours_elapsed = (current_time - last_updated).total_seconds() / 3600
                    if hours_elapsed > max_processing_hours:
                        stuck_videos.append(video_name)
                        logger.warning(f"⚠️ Video {video_name} has been stuck in 'processing' status for {hours_elapsed:.1f} hours")
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Could not parse last_updated time for {video_name}: {e}")
    
    return stuck_videos


def reset_stuck_videos(checklist: Dict, max_processing_hours: int = 2) -> int:
    """Reset videos that have been stuck in 'processing' status back to 'pending'"""
    stuck_videos = detect_stuck_videos(checklist, max_processing_hours)
    
    for video_name in stuck_videos:
        logger.info(f"🔄 Resetting stuck video {video_name} from 'processing' to 'pending'")
        update_video_status(checklist, video_name, "pending", 
                          reset_reason="stuck_in_processing", 
                          reset_time=datetime.now().isoformat())
    
    return len(stuck_videos)


# =============================================================================
# BLOB POLLING FUNCTIONS
# =============================================================================
def poll_azure_videos_with_checklist(
    blob_service_client: BlobServiceClient, 
    container_name: str, 
    blob_prefix: str, 
    checklist: dict,
    config: dict = None
) -> List[Dict]:
    """Poll Azure Blob Storage for video files and update checklist"""
    try:
        container_client = blob_service_client.get_container_client(container_name)
        
        video_extensions = ['.insv', '.mp4', '.avi', '.mov', '.mkv']
        videos = []
        blob_list = container_client.list_blobs(name_starts_with=blob_prefix)
        
        for blob in blob_list:
            blob_name = blob.name
            
            # Check if it's a video file
            if any(blob_name.lower().endswith(ext) for ext in video_extensions):
                # Look for corresponding audio file
                audio_file = find_corresponding_audio(blob_service_client, container_name, blob_name, blob_prefix, config)
                
                if audio_file:
                    video_name = os.path.splitext(os.path.basename(blob_name))[0]
                    
                    video_info = {
                        "video_name": video_name,
                        "video_blob_name": blob_name,
                        "audio_blob_name": audio_file,
                        "video_size_mb": round(blob.size / (1024 * 1024), 2),
                        "last_modified": blob.last_modified.isoformat(),
                        "processed": False,
                        "processing_date": None,
                        "output_dir": None,
                        "status": "pending"
                    }
                    
                    videos.append(video_info)
                    
                    # Update checklist if this is a new video
                    if video_name not in checklist["videos"]:
                        update_video_status(checklist, video_name, "discovered")
        
        logger.info(f"📋 Found {len(videos)} video files in Azure Blob Storage")
        return videos
        
    except Exception as e:
        logger.error(f"❌ Failed to poll Azure Blob Storage: {e}")
        return []


def poll_azure_videos(
    blob_service_client: BlobServiceClient, 
    container_name: str, 
    blob_prefix: str, 
    checklist: Dict,
    config: dict = None
) -> List[Dict]:
    """Poll Azure blob storage for new videos and update checklist"""
    try:
        logger.info(f"🔍 Polling Azure blob storage: {container_name}/{blob_prefix}")
        
        container_client = blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=blob_prefix)
        
        new_videos = []
        for blob in blobs:
            if blob.name.lower().endswith(('.insv', '.mp4', '.avi', '.mov')):
                video_name = os.path.splitext(os.path.basename(blob.name))[0]
                
                # Check if this video is already in checklist
                if video_name not in checklist["videos"]:
                    # Look for corresponding audio file
                    audio_blob_name = find_corresponding_audio(blob_service_client, container_name, blob.name, blob_prefix, config)
                    
                    if audio_blob_name:
                        video_info = {
                            "video_name": video_name,
                            "video_blob_name": blob.name,
                            "audio_blob_name": audio_blob_name,
                            "discovered_at": datetime.now().isoformat()
                        }
                        new_videos.append(video_info)
                        
                        # Add to checklist
                        update_video_status(checklist, video_name, "pending")
                        logger.info(f"📹 New video found: {video_name}")
                    else:
                        logger.warning(f"⚠️ No corresponding audio found for {video_name}")
        
        if new_videos:
            logger.info(f"🆕 Found {len(new_videos)} new videos")
        else:
            logger.info("✅ No new videos found")
            
        return new_videos
        
    except Exception as e:
        logger.error(f"❌ Failed to poll Azure videos: {e}")
        return []


def find_corresponding_audio(blob_service_client: BlobServiceClient, container_name: str, 
                           video_blob_name: str, blob_prefix: str, config: dict = None) -> Optional[str]:
    """Find audio file corresponding to video file using improved pattern matching"""
    try:
        video_base_name = os.path.splitext(os.path.basename(video_blob_name))[0]
        container_client = blob_service_client.get_container_client(container_name)
        
        # Load audio extensions from config or use defaults
        if config and 'processing' in config and 'audio_extensions' in config['processing']:
            audio_extensions = config['processing']['audio_extensions']
        else:
            # Fallback to default extensions if config not provided or missing
            audio_extensions = ['.wav', '.WAV', '.mp3', '.aac', '.m4a']
            logger.warning("⚠️ Using default audio extensions. Consider providing config with audio_extensions.")
        
        # Strategy 1: Try exact match first (most common case - 80% match rate)
        for ext in audio_extensions:
            # Try exact match with same directory structure
            potential_audio_name = f"{os.path.dirname(video_blob_name)}/{video_base_name}{ext}"
            try:
                blob_client = container_client.get_blob_client(potential_audio_name)
                if blob_client.exists():
                    logger.info(f"✅ Found exact audio match: {potential_audio_name}")
                    return potential_audio_name
            except:
                pass
            
            # Try without directory prefix
            potential_audio_name = f"{blob_prefix}/{video_base_name}{ext}"
            try:
                blob_client = container_client.get_blob_client(potential_audio_name)
                if blob_client.exists():
                    logger.info(f"✅ Found exact audio match: {potential_audio_name}")
                    return potential_audio_name
            except:
                pass
        
        # Strategy 2: Try pattern matching using 'like' operator approach
        # Look for audio files that contain the video base name
        logger.info(f"🔍 Searching for pattern matches for video: {video_base_name}")
        
        # List all blobs in the container with the same prefix
        blobs = container_client.list_blobs(name_starts_with=blob_prefix)
        
        best_match = None
        best_match_score = 0
        
        for blob in blobs:
            if any(blob.name.lower().endswith(ext.lower()) for ext in audio_extensions):
                audio_name = os.path.splitext(os.path.basename(blob.name))[0]
                
                # Calculate similarity score using multiple matching strategies
                score = 0
                
                # Strategy 2a: Check if video base name is contained in audio name
                if video_base_name.lower() in audio_name.lower():
                    score += 50
                
                # Strategy 2b: Check if audio name is contained in video base name
                if audio_name.lower() in video_base_name.lower():
                    score += 40
                
                # Strategy 2c: Check for common patterns (e.g., timestamp patterns)
                # Remove common suffixes like -video, -audio
                clean_video_name = video_base_name.replace('-video', '').replace('-Video', '').replace('_video', '').replace('_Video', '')
                clean_audio_name = audio_name.replace('-audio', '').replace('-Audio', '').replace('_audio', '').replace('_Audio', '')
                
                if clean_video_name.lower() == clean_audio_name.lower():
                    score += 100  # Perfect match after cleaning
                elif clean_video_name.lower() in clean_audio_name.lower():
                    score += 60
                elif clean_audio_name.lower() in clean_video_name.lower():
                    score += 60
                
                # Strategy 2e: Check for exact base name matches (e.g., cooking-20250720_1521)
                # This handles cases like cooking-20250720_1521-video and cooking-20250720_1521-audio
                if video_base_name.endswith('-video') and audio_name.endswith('-audio'):
                    base_video = video_base_name[:-6]  # Remove '-video'
                    base_audio = audio_name[:-6]       # Remove '-audio'
                    if base_video.lower() == base_audio.lower():
                        score += 95  # Very high score for this pattern
                        logger.info(f"🎯 Found video-audio suffix pattern: {base_video}")
                
                # Strategy 2f: Check for timestamp-based patterns
                # Extract timestamp patterns (e.g., 20250720_1521)
                import re
                video_timestamp = re.search(r'\d{8}_\d{4}', video_base_name)
                audio_timestamp = re.search(r'\d{8}_\d{4}', audio_name)
                
                if video_timestamp and audio_timestamp:
                    if video_timestamp.group() == audio_timestamp.group():
                        score += 80  # High score for timestamp match
                        logger.info(f"🕐 Found timestamp match: {video_timestamp.group()}")
                
                # Update best match
                if score > best_match_score:
                    best_match_score = score
                    best_match = blob.name
        
        if best_match and best_match_score >= 50:  # Minimum threshold
            logger.info(f"🎯 Best audio match found: {best_match} (score: {best_match_score})")
            return best_match
        else:
            logger.warning(f"⚠️ No suitable audio file found for {video_base_name}")
            logger.info(f"🔍 Searched in prefix: {blob_prefix}")
        
        return best_match
        
    except Exception as e:
        logger.error(f"❌ Error finding audio for {video_blob_name}: {e}")
        return None


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_video_audio_pair(
    blob_service_client: BlobServiceClient, 
    container_name: str, 
    video_info: Dict, 
    local_download_dir: str
) -> Dict[str, str]:
    """Download video and audio files from Azure"""
    video_blob = video_info['video_blob_name']
    audio_blob = video_info['audio_blob_name']
    
    # Create download paths
    video_filename = os.path.basename(video_blob)
    video_local_path = os.path.join(local_download_dir, video_filename)
    
    audio_local_path = None
    if audio_blob:
        audio_filename = os.path.basename(audio_blob)
        audio_local_path = os.path.join(local_download_dir, audio_filename)
    
    try:
        # Download video
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=video_blob
        )
        
        with open(video_local_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        
        logger.info(f"✅ Video downloaded: {video_local_path}")
        
        # Download audio if exists
        if audio_blob and audio_local_path:
            audio_client = blob_service_client.get_blob_client(
                container=container_name, 
                blob=audio_blob
            )
            
            with open(audio_local_path, "wb") as f:
                download_stream = audio_client.download_blob()
                f.write(download_stream.readall())
            
            logger.info(f"✅ Audio downloaded: {audio_local_path}")
        else:
            logger.warning("⚠️ No corresponding audio file found")
        
        return {
            "video_path": video_local_path,
            "audio_path": audio_local_path
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to download {video_blob}: {e}")
        raise


def download_video_audio_pair_simple(blob_service_client: BlobServiceClient, container_name: str,
                                   video_blob_name: str, audio_blob_name: str, local_download_dir: str) -> tuple:
    """Download video and audio files to local directory"""
    try:
        os.makedirs(local_download_dir, exist_ok=True)
        
        # Download video
        video_local_path = os.path.join(local_download_dir, os.path.basename(video_blob_name))
        container_client = blob_service_client.get_container_client(container_name)
        
        logger.info(f"⬇️ Downloading video: {video_blob_name}")
        with open(video_local_path, "wb") as f:
            blob_client = container_client.get_blob_client(video_blob_name)
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        
        # Download audio
        audio_local_path = os.path.join(local_download_dir, os.path.basename(audio_blob_name))
        logger.info(f"⬇️ Downloading audio: {audio_blob_name}")
        with open(audio_local_path, "wb") as f:
            blob_client = container_client.get_blob_client(audio_blob_name)
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        
        logger.info(f"✅ Downloaded: {os.path.basename(video_blob_name)} + {os.path.basename(audio_blob_name)}")
        return video_local_path, audio_local_path
        
    except Exception as e:
        logger.error(f"❌ Failed to download {video_blob_name}: {e}")
        return None, None


def cleanup_downloaded_files(file_paths: List[str]):
    """Clean up downloaded files"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"🗑️ Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up {file_path}: {e}")


# =============================================================================
# UPLOAD FUNCTIONS
# =============================================================================

def generate_azure_shard_urls(azure_blob_client, container, shard_paths, output_prefix,
                              account_name: str, account_key: str):
    """Generate Azure blob URLs with SAS tokens for shard files"""
    shard_urls = {}
    expiry = datetime.utcnow() + timedelta(days=90)

    for i, shard_path in enumerate(shard_paths):
        if not os.path.exists(shard_path):
            logger.warning(f"Shard file not found: {shard_path}")
            shard_urls[i] = None
            continue

        shard_filename = os.path.basename(shard_path)
        # keep the path clean; output_prefix already contains view_*_shards
        blob_name = f"{output_prefix}/{shard_filename}".replace("\\", "/")

        try:
            blob_client = azure_blob_client.get_blob_client(container=container, blob=blob_name)
            with open(shard_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=container,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=expiry
            )
            shard_urls[i] = f"https://{account_name}.blob.core.windows.net/{container}/{blob_name}?{sas_token}"
        except Exception as e:
            logger.error(f"Failed to upload/SAS shard {i+1}: {e}")
            shard_urls[i] = None

    return shard_urls


def upload_output_directory_to_blob(output_dir: str, azure_blob_client, container_name: str, 
                                   blob_base_path: str, video_name: str):
    """
    Upload entire output directory to Azure blob storage
    
    Args:
        output_dir: Local output directory path
        azure_blob_client: Azure blob service client
        container_name: Azure container name
        blob_base_path: Base blob path (e.g., "/krishna-test/test1/test_activity")
        video_name: Video name for organizing the upload
        
    Returns:
        dict: Upload results with success status and uploaded files list
    """
    from pathlib import Path
    
    logger.info(f"🔄 Uploading output directory to Azure blob storage...")
    logger.info(f"   Local path: {output_dir}")
    logger.info(f"   Blob path: {blob_base_path}/{video_name}")
    
    uploaded_files = []
    failed_files = []
    
    try:
        # Walk through all files in the output directory
        output_path = Path(output_dir)
        if not output_path.exists():
            logger.error(f"Output directory does not exist: {output_dir}")
            return {"success": False, "error": "Output directory not found"}
        
        for file_path in output_path.rglob('*'):
            if file_path.is_file():
                # Calculate relative path from output directory
                relative_path = file_path.relative_to(output_path)
                
                # Create blob path
                blob_name = f"{blob_base_path.strip('/')}/{video_name}/{relative_path}".replace("\\", "/")
                
                try:
                    # Upload file to blob
                    blob_client = azure_blob_client.get_blob_client(
                        container=container_name, 
                        blob=blob_name
                    )
                    
                    with open(file_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                    
                    uploaded_files.append({
                        "local_path": str(file_path),
                        "blob_path": blob_name,
                        "size_bytes": file_path.stat().st_size
                    })
                    
                    logger.debug(f"✅ Uploaded: {relative_path} -> {blob_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to upload {relative_path}: {e}")
                    failed_files.append({
                        "local_path": str(file_path),
                        "blob_path": blob_name,
                        "error": str(e)
                    })
        
        # Summary
        total_files = len(uploaded_files) + len(failed_files)
        success_rate = (len(uploaded_files) / total_files * 100) if total_files > 0 else 0
        
        logger.info(f"📊 Upload Summary:")
        logger.info(f"   ✅ Successful: {len(uploaded_files)}/{total_files} files ({success_rate:.1f}%)")
        logger.info(f"   ❌ Failed: {len(failed_files)} files")
        
        if failed_files:
            logger.warning(f"⚠️ Failed files: {[f['local_path'] for f in failed_files]}")
        
        return {
            "success": len(failed_files) == 0,
            "uploaded_files": uploaded_files,
            "failed_files": failed_files,
            "total_files": total_files,
            "success_rate": success_rate
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to upload output directory: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# PIPELINE CONFIGURATION FUNCTIONS
# =============================================================================

def load_pipeline_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    """Load pipeline configuration from YAML file"""
    try:
        # Check if file exists at given path
        if not os.path.exists(config_path):
            # Try relative to current script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, config_path)
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"✅ Pipeline configuration loaded from: {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"⚠️ Pipeline config file not found: {config_path}. Using defaults.")
        # Return default configuration
        return {
            "azure_storage": {
                "input_blob_prefix": "test/input_videos",
                "output_blob_prefix": "processed_outputs"
            },
            "local_storage": {
                "temp_download_dir": "/tmp/azure_downloads",
                "output_base_dir": "./out",
                "video_list_file": "./azure_video_list.json",
                "checklist_file": "./video_checklist.json"
            },
            "polling": {
                "interval_minutes": 5
            },
            "processing": {
                "max_videos_per_batch": 3
            },
            "cleanup": {
                "cleanup_temp_dir": True,
                "cleanup_files_after_each_video": True
            }
        }
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing pipeline configuration YAML: {e}")


def save_video_list_progress(video_list_file: str, videos: List[Dict]):
    """Save video list progress to file"""
    try:
        progress_data = {
            "last_poll": datetime.now().isoformat(),
            "total_videos": len(videos),
            "videos": videos
        }
        
        with open(video_list_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        logger.info(f"💾 Saved video list progress: {len(videos)} videos")
    except Exception as e:
        logger.error(f"❌ Failed to save video list progress: {e}")
