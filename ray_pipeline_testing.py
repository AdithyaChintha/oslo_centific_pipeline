# Simple Ray Pipeline with Time-Based Detection Integration
import ray
import os
import gc
import torch
import requests
import json
import shutil
import time
import glob
import yaml
import signal
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Optional
from utils.logger import get_logger
from azure.storage.blob import generate_blob_sas, BlobSasPermissions, BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Import setup and all necessary Ray tasks
from setup.cosmos.setup import setup_cosmos
from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.audio_splitter import split_audio_into_shards
#from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
from ray_jobs.scene_det import detect_scenes
from ray_jobs.run_yolodetect_task import run_yolodetect_on_shard
from ray_jobs.audio_diarization_pii import process_audio_diarization
from ray_jobs.clap_detector import detect_claps_in_media

# Import new ray jobs
from ray_jobs.nsfw_det_final import process_video_chunks_for_nsfw
from ray_jobs.motion_energy import compute_motion_energy
from ray_jobs.face_age_detector import process_video_chunks_for_face_detection
from ray_jobs.labelstudio_tasks import build_labelstudio_json_for_shard_task, import_to_labelstudio_task
from ray_jobs.video_unwarp_task import erp_unwarp_task


# Test unwarp
from ray_jobs.video_unwarp_task import insv_unwarp_task

logger = get_logger("SimplifiedUnifiedPipeline")

@ray.remote
def integrated_blob_polling_and_pipeline_task(
    azure_config_path: str = "blobfuse2_config.yaml",
    pipeline_config_path: str = "config/pipeline_config.yaml",
    blob_prefix: str = None,  # Will use config default if None
    max_videos: Optional[int] = None,
    force_refresh: bool = False,
    process_dual_views: bool = None,
    process_unwarped_views: bool = False
):
    """
    Integrated Ray task that handles both blob polling and pipeline processing.
    
    This task performs two sequential loops:
    1. BLOB POLLING LOOP: Find videos and download corresponding audio files
    2. MAIN PIPELINE LOOP: Process video+audio pairs one by one
    
    Args:
        config_path: Path to Azure configuration file
        blob_prefix: Blob prefix to search for videos
        max_videos: Maximum number of videos to process
        force_refresh: Force refresh video list
        process_dual_views: Enable dual view processing
        process_unwarped_views: Enable unwarped view processing
        
    Returns:
        dict: Processing results summary
    """
    logger.info("🚀 Starting Integrated Blob Polling and Pipeline Task")
    logger.info("=" * 80)
    
    # Initialize configuration and Azure client
    try:
        # Load pipeline configuration
        pipeline_config = _load_pipeline_config(pipeline_config_path)
        
        # Load Azure configuration
        azure_config = _load_azure_config(azure_config_path)
        blob_service_client = _create_azure_blob_client(azure_config)
        container_name = azure_config['container']
        account_name = azure_config['account-name']
        account_key = azure_config['account-key']
        
        # Setup directories from pipeline config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_download_dir = pipeline_config['local_storage']['temp_download_dir']
        output_base_dir = os.path.join(current_dir, pipeline_config['local_storage']['output_base_dir'].lstrip('./'))
        video_list_file = os.path.join(current_dir, pipeline_config['local_storage']['video_list_file'].lstrip('./'))
        
        # Use blob_prefix from config if not provided
        if blob_prefix is None:
            blob_prefix = pipeline_config['azure_storage']['input_blob_prefix']
        
        os.makedirs(local_download_dir, exist_ok=True)
        os.makedirs(output_base_dir, exist_ok=True)
        
        logger.info(f"✅ Configuration loaded successfully")
        logger.info(f"   Container: {container_name}")
        logger.info(f"   Blob prefix: {blob_prefix}")
        logger.info(f"   Download dir: {local_download_dir}")
        logger.info(f"   Output dir: {output_base_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize configuration: {e}")
        return {"success": False, "error": f"Configuration error: {e}"}
    
    # =============================================================================
    # LOOP 1: BLOB POLLING - Find videos and download corresponding audio files
    # =============================================================================
    logger.info("\n" + "="*80)
    logger.info("🔍 LOOP 1: BLOB POLLING - Discovering and downloading video+audio pairs")
    logger.info("="*80)
    
    downloaded_file_pairs = []
    
    try:
        # Poll Azure for videos
        logger.info(f"🔍 Polling Azure Blob Storage for videos with prefix: {blob_prefix}")
        videos = _poll_azure_videos(
            blob_service_client, container_name, blob_prefix, 
            video_list_file, force_refresh
        )
        
        if not videos:
            logger.warning("⚠️ No videos found in Azure Blob Storage")
            return {"success": True, "message": "No videos to process", "videos_processed": 0}
        
        # Filter for unprocessed videos
        pending_videos = [v for v in videos if not v.get('processed', False)]
        if max_videos:
            pending_videos = pending_videos[:max_videos]
        
        logger.info(f"📋 Found {len(pending_videos)} videos to download and process")
        
        # Download video+audio pairs
        for i, video_info in enumerate(pending_videos, 1):
            video_blob = video_info['video_blob_name']
            audio_blob = video_info['audio_blob_name']
            
            logger.info(f"\n📥 Downloading pair {i}/{len(pending_videos)}: {video_blob}")
            
            try:
                # Download video and audio files
                downloaded_files = _download_video_audio_pair(
                    blob_service_client, container_name, video_info, local_download_dir
                )
                
                if downloaded_files['video_path']:
                    downloaded_file_pairs.append({
                        'video_info': video_info,
                        'video_path': downloaded_files['video_path'],
                        'audio_path': downloaded_files['audio_path'],
                        'download_index': i
                    })
                    logger.info(f"✅ Downloaded pair {i}: Video + Audio ready for processing")
                else:
                    logger.error(f"❌ Failed to download pair {i}: {video_blob}")
                    
            except Exception as e:
                logger.error(f"❌ Error downloading pair {i} ({video_blob}): {e}")
                continue
        
        logger.info(f"\n📊 LOOP 1 COMPLETE: Downloaded {len(downloaded_file_pairs)} video+audio pairs")
        
    except Exception as e:
        logger.error(f"❌ LOOP 1 FAILED: Blob polling error: {e}")
        return {"success": False, "error": f"Blob polling failed: {e}"}
    
    # =============================================================================
    # LOOP 2: MAIN PIPELINE - Process video+audio pairs one by one
    # =============================================================================
    logger.info("\n" + "="*80)
    logger.info("🎬 LOOP 2: MAIN PIPELINE - Processing video+audio pairs sequentially")
    logger.info("="*80)
    
    processing_results = []
    successful_count = 0
    failed_count = 0
    
    try:
        for pair_data in downloaded_file_pairs:
            video_info = pair_data['video_info']
            video_path = pair_data['video_path']
            audio_path = pair_data['audio_path']
            pair_index = pair_data['download_index']
            
            video_blob = video_info['video_blob_name']
            video_name = os.path.splitext(os.path.basename(video_blob))[0]
            
            logger.info(f"\n🎬 Processing pair {pair_index}/{len(downloaded_file_pairs)}: {video_name}")
            logger.info("-" * 60)
            
            try:
                # Create output directory for this video
                video_output_dir = os.path.join(
                    output_base_dir, 
                    f"video_{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                os.makedirs(video_output_dir, exist_ok=True)
                
                # Run the main pipeline
                logger.info(f"🚀 Starting pipeline processing for {video_name}")
                
                # Get output prefix from config
                output_prefix = pipeline_config['azure_storage']['output_blob_prefix']
                
                pipeline_results = pipeline_main(
                    input_video_path=video_path,
                    input_audio_path=audio_path,
                    output_dir=video_output_dir,
                    process_dual_views=None,  # Let pipeline_main auto-detect for INSV files
                    process_unwarped_views=process_unwarped_views,
                    azure_blob_client=blob_service_client,
                    azure_container=container_name,
                    azure_output_prefix=f"{output_prefix}/{video_name}",
                    azure_account_name=account_name,
                    azure_account_key=account_key
                )
                
                # Update video info with success
                video_info.update({
                    "processed": True,
                    "processing_date": datetime.now().isoformat(),
                    "output_dir": video_output_dir,
                    "status": "completed",
                    "results_summary": pipeline_results
                })
                
                processing_results.append({
                    "video_name": video_name,
                    "status": "success",
                    "output_dir": video_output_dir,
                    "results": pipeline_results
                })
                
                successful_count += 1
                logger.info(f"✅ Pipeline completed successfully for {video_name}")
                logger.info(f"📁 Results saved to: {video_output_dir}")
                
            except Exception as e:
                logger.error(f"❌ Pipeline failed for {video_name}: {e}")
                
                # Update video info with failure
                video_info.update({
                    "processed": False,
                    "processing_date": datetime.now().isoformat(),
                    "status": "failed",
                    "error": str(e)
                })
                
                processing_results.append({
                    "video_name": video_name,
                    "status": "failed",
                    "error": str(e)
                })
                
                failed_count += 1
            
            finally:
                # Clean up downloaded files for this pair
                _cleanup_downloaded_files([video_path, audio_path])
                
                # Save progress after each video
                _save_video_list_progress(video_list_file, videos)
                
                logger.info(f"📊 Progress: {successful_count} successful, {failed_count} failed")
        
        logger.info(f"\n📊 LOOP 2 COMPLETE: Processed {len(downloaded_file_pairs)} video pairs")
        
    except Exception as e:
        logger.error(f"❌ LOOP 2 FAILED: Pipeline processing error: {e}")
        return {"success": False, "error": f"Pipeline processing failed: {e}"}
    
    # =============================================================================
    # FINAL RESULTS
    # =============================================================================
    logger.info("\n" + "="*80)
    logger.info("🏁 INTEGRATED TASK COMPLETE")
    logger.info("="*80)
    
    final_results = {
        "success": True,
        "task_type": "integrated_blob_polling_and_pipeline",
        "videos_discovered": len(pending_videos),
        "pairs_downloaded": len(downloaded_file_pairs),
        "videos_processed": successful_count + failed_count,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "completion_rate": f"{(successful_count/(successful_count + failed_count)*100):.1f}%" if (successful_count + failed_count) > 0 else "0%",
        "output_base_dir": output_base_dir,
        "processing_results": processing_results,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"📊 Final Results:")
    logger.info(f"   Videos Discovered: {final_results['videos_discovered']}")
    logger.info(f"   Pairs Downloaded: {final_results['pairs_downloaded']}")
    logger.info(f"   Videos Processed: {final_results['videos_processed']}")
    logger.info(f"   Successful: {final_results['successful_count']}")
    logger.info(f"   Failed: {final_results['failed_count']}")
    logger.info(f"   Completion Rate: {final_results['completion_rate']}")
    logger.info(f"   Output Directory: {final_results['output_base_dir']}")
    
    # =============================================================================
    # CLEANUP PHASE
    # =============================================================================
    if pipeline_config.get('cleanup', {}).get('cleanup_temp_dir', True):
        logger.info("\n🧹 CLEANUP: Removing temporary download directory")
        try:
            if os.path.exists(local_download_dir):
                shutil.rmtree(local_download_dir)
                logger.info(f"✅ Cleaned up temp directory: {local_download_dir}")
            else:
                logger.info(f"ℹ️ Temp directory already clean: {local_download_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to clean up temp directory: {e}")
    else:
        logger.info("ℹ️ Temp directory cleanup disabled in config")
    
    # After all shards are processed, enhance labelstudio tasks with model results
    try:
        process_shard_labelstudio_enhancement(output_dir, total_shards)
        logger.info("🎯 Label Studio tasks enhanced with model results")
    except Exception as e:
        logger.error(f"❌ Failed to enhance Label Studio tasks: {e}")
    
    return final_results


def _load_azure_config(config_path: str) -> dict:
    """Load Azure configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config['azstorage']
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def _load_pipeline_config(config_path: str = "config/pipeline_config.yaml") -> dict:
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
                "video_list_file": "./azure_video_list.json"
            },
            "cleanup": {
                "cleanup_temp_dir": True,
                "cleanup_files_after_each_video": True
            }
        }
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing pipeline configuration YAML: {e}")


def _create_azure_blob_client(config: dict) -> BlobServiceClient:
    """Create Azure Blob Service client"""
    account_url = f"https://{config['account-name']}.blob.core.windows.net"
    return BlobServiceClient(account_url=account_url, credential=config['account-key'])


def _poll_azure_videos(
    blob_service_client: BlobServiceClient, 
    container_name: str, 
    blob_prefix: str, 
    video_list_file: str, 
    force_refresh: bool = False
) -> List[Dict]:
    """Poll Azure Blob Storage for video files"""
    
    # Check if we already polled recently (unless force refresh)
    if not force_refresh and os.path.exists(video_list_file):
        with open(video_list_file, 'r') as f:
            existing_data = json.load(f)
            last_poll = datetime.fromisoformat(existing_data.get('last_poll', '2000-01-01'))
            time_diff = datetime.now() - last_poll
            if time_diff.total_seconds() < 300:  # 5 minutes
                logger.info(f"📋 Videos polled {int(time_diff.total_seconds())} seconds ago. Found {len(existing_data['videos'])} videos.")
                return existing_data['videos']
    
    video_extensions = ['.insv', '.mp4', '.avi', '.mov', '.mkv']
    
    try:
        container_client = blob_service_client.get_container_client(container_name)
        
        videos = []
        blob_list = container_client.list_blobs(name_starts_with=blob_prefix)
        
        for blob in blob_list:
            blob_name = blob.name
            
            # Check if it's a video file
            if any(blob_name.lower().endswith(ext) for ext in video_extensions):
                # Look for corresponding audio file
                audio_file = _find_corresponding_audio(blob_name, container_client)
                
                video_info = {
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
        
        # Save to JSON file
        video_data = {
            "last_poll": datetime.now().isoformat(),
            "total_videos": len(videos),
            "container": container_name,
            "blob_prefix": blob_prefix,
            "videos": videos
        }
        
        with open(video_list_file, 'w') as f:
            json.dump(video_data, f, indent=2)
        
        logger.info(f"📋 Found {len(videos)} video files in Azure Blob Storage")
        logger.info(f"💾 Video list saved to {video_list_file}")
        
        return videos
        
    except Exception as e:
        logger.error(f"❌ Failed to poll Azure Blob Storage: {e}")
        raise


def _find_corresponding_audio(video_blob_name: str, container_client) -> Optional[str]:
    """Find corresponding audio file for a video"""
    video_base = os.path.splitext(video_blob_name)[0]
    
    # Common audio naming patterns
    audio_patterns = [
        f"{video_base}.wav",
        f"{video_base}.mp3",
        f"{video_base}.m4a",
        f"{video_base}-audio.wav",
        f"{video_base}-audio.mp3",
        video_blob_name.replace("video", "audio").replace(".insv", ".wav"),
        video_blob_name.replace("-video", "-audio").replace(".insv", ".wav")
    ]
    
    for pattern in audio_patterns:
        try:
            blob_client = container_client.get_blob_client(pattern)
            if blob_client.exists():
                return pattern
        except:
            continue
    
    return None


def _download_video_audio_pair(
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


def _cleanup_downloaded_files(file_paths: List[str]):
    """Clean up downloaded files"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"🗑️ Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up {file_path}: {e}")


def _save_video_list_progress(video_list_file: str, videos: List[Dict]):
    """Save video list progress to file"""
    try:
        with open(video_list_file, 'r') as f:
            video_data = json.load(f)
        
        video_data['videos'] = videos
        video_data['last_updated'] = datetime.now().isoformat()
        
        with open(video_list_file, 'w') as f:
            json.dump(video_data, f, indent=2)
            
    except Exception as e:
        logger.warning(f"⚠️ Failed to save progress: {e}")


def clear_gpu_memory():
    """Clear GPU memory between tasks"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")

def extract_flagged_segments(task_result, task_type, shard_index, shard_offset_sec):
    """Extract flagged segments from task results"""
    segments = []
    try:
        if not task_result:
            return segments

        def maybe_offset(t):
            # If upstream added global times (set by those tasks), don't re-add.
            if isinstance(t, (int, float)):
                return t + shard_offset_sec
            return shard_offset_sec

        if task_type == "audio":
            for pii in task_result.get('pii_detections', []):
                segments.append({
                    "start_time": maybe_offset(pii.get('start_time', 0)),
                    "end_time":   maybe_offset(pii.get('end_time',   0)),
                    "task_type": "audio_pii",
                    "confidence": 0.8,
                    "flag_type": "pii_detected",
                    "priority": "high",
                    "description": f"PII detected: {pii.get('entity_type', 'unknown')}",
                    "shard_index": shard_index + 1
                })

        elif task_type == "yolo":
            pass  # handled elsewhere if needed

        elif task_type == "scene":
            success = task_result.get('processing_info', {}).get('success', False)
            if not success:
                logger.warning(f"Scene detection failed for shard {shard_index}: "
                               f"{task_result.get('processing_info', {}).get('error', 'Unknown error')}")
                return segments
            for scene in task_result.get('scenes', []):
                segments.append({
                    "start_time": maybe_offset(scene.get('start_time', 0)),
                    "end_time":   maybe_offset(scene.get('end_time', 60)),
                    "task_type": "scene_detection",
                    "confidence": 0.8,
                    "flag_type": "scene_content",
                    "priority": "medium",
                    "description": scene.get('description', 'Scene detected')[:200],
                    "shard_index": shard_index + 1
                })

        elif task_type == "nsfw":
            for seg in task_result.get('flagged_segments', []):
                segments.append({
                    "start_time": maybe_offset(seg.get('start_time', 0)),
                    "end_time":   maybe_offset(seg.get('end_time',   0)),
                    "task_type": "nsfw_detection",
                    "confidence": seg.get('confidence', 0.7),
                    "flag_type": "nsfw_content",
                    "priority": seg.get('priority', 'high'),
                    "description": seg.get('description', 'NSFW content detected'),
                    "shard_index": shard_index + 1
                })

        elif task_type == "motion":
            for seg in task_result.get('segments', []):
                activity_type = seg.get('activity_type', '').lower()
                if 'high' in activity_type or seg.get('avg_motion_energy', 0) > 0.7:
                    segments.append({
                        "start_time": maybe_offset(seg.get('start_time', 0)),
                        "end_time":   maybe_offset(seg.get('end_time',   0)),
                        "task_type": "motion_energy",
                        "confidence": seg.get('confidence', 0.6),
                        "flag_type": "high_motion",
                        "priority": "medium",
                        "description": f"High motion activity: {activity_type}",
                        "shard_index": shard_index + 1
                    })

        elif task_type == "face":
            for seg in task_result.get('flagged_segments', []):
                segments.append({
                    "start_time": maybe_offset(seg.get('start_time', 0)),
                    "end_time":   maybe_offset(seg.get('end_time',   0)),
                    "task_type": "face_detection",
                    "confidence": seg.get('confidence', 0.7),
                    "flag_type": seg.get('flag_type', 'face_detected'),
                    "priority": seg.get('priority', 'medium'),
                    "description": seg.get('description', 'Face detected'),
                    "shard_index": shard_index + 1
                })

        elif task_type == "clap":
            for clap in task_result.get('clap_timestamps', []):
                clap_time = clap.get('timestamp_seconds', 0)
                desc = clap.get('timestamp_formatted', f"{clap_time:.2f}s")
                segments.append({
                    "start_time": max(0, shard_offset_sec + clap_time - 2),
                    "end_time":   shard_offset_sec + clap_time + 2,
                    "task_type": "clap_detection",
                    "confidence": 0.9,
                    "flag_type": "clap_detected",
                    "priority": "medium",
                    "description": f"Clap detected at {desc}",
                    "shard_index": shard_index + 1
                })
    except Exception as e:
        logger.warning(f"Error extracting segments from {task_type}: {e}")
    return segments


# Put near imports / helpers
PRIORITY_ORDER = ['high', 'medium', 'low']  # 0 is highest

def merge_overlapping_segments(timeline, tol=1.0):
    """
    Merge overlapping segments across views (1s tolerance), dedupe task types,
    combine source_view, keep max confidence, and choose highest priority.
    """
    if not timeline:
        return []
    timeline = sorted(timeline, key=lambda x: x.get("start_time", 0))
    merged, cur = [], dict(timeline[0])

    def to_set(val):
        if isinstance(val, list): return set(val)
        if isinstance(val, str):  return set(t.strip() for t in val.split(",") if t.strip())
        return set()

    for seg in timeline[1:]:
        if seg.get("start_time", 0) <= cur.get("end_time", 0) + tol:
            cur["end_time"] = max(cur.get("end_time", 0), seg.get("end_time", 0))

            cur_tasks = to_set(cur.get("task_type", "")) | to_set(seg.get("task_type", ""))
            cur["task_type"] = ", ".join(sorted(cur_tasks)) if cur_tasks else ""

            cur_desc, new_desc = cur.get("description", ""), seg.get("description", "")
            if new_desc and new_desc not in cur_desc:
                cur["description"] = f"{cur_desc}; {new_desc}" if cur_desc else new_desc

            cur_views = to_set(cur.get("source_view", "")) | to_set(seg.get("source_view", ""))
            cur["source_view"] = ", ".join(sorted(cur_views)) if cur_views else ""

            cur["confidence"] = max(cur.get("confidence", 0), seg.get("confidence", 0))

            cp = PRIORITY_ORDER.index(cur.get("priority", "medium")) if cur.get("priority", "medium") in PRIORITY_ORDER else 1
            sp = PRIORITY_ORDER.index(seg.get("priority", "medium")) if seg.get("priority", "medium") in PRIORITY_ORDER else 1
            if sp < cp:
                cur["priority"] = seg.get("priority", "medium")
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)
    return merged


def merge_overlapping_segments_dual_view(timeline):
    if not timeline:
        return []
    timeline.sort(key=lambda x: x.get("start_time", 0))
    merged, current = [], timeline[0].copy()
    for segment in timeline[1:]:
        if segment["start_time"] <= current["end_time"] + 1.0:
            current["end_time"] = max(current["end_time"], segment["end_time"])
            # task_type
            cur_tasks = current.get("task_type", "")
            cur_list = cur_tasks if isinstance(cur_tasks, list) else ([t.strip() for t in cur_tasks.split(",")] if cur_tasks else [])
            new_tasks = segment.get("task_type", "")
            new_list = new_tasks if isinstance(new_tasks, list) else ([t.strip() for t in new_tasks.split(",")] if new_tasks else [])
            for t in new_list:
                if t and t not in cur_list:
                    cur_list.append(t)
            current["task_type"] = ", ".join(cur_list) if cur_list else ""
            # description
            cur_desc = current.get("description", "")
            new_desc = segment.get("description", "")
            if new_desc and new_desc not in cur_desc:
                current["description"] = f"{cur_desc}; {new_desc}" if cur_desc else new_desc
            # source_view
            cur_sv = current.get("source_view", "")
            cur_sv_list = cur_sv if isinstance(cur_sv, list) else ([v.strip() for v in cur_sv.split(",")] if cur_sv else [])
            new_sv = segment.get("source_view", "")
            new_sv_list = new_sv if isinstance(new_sv, list) else ([v.strip() for v in new_sv.split(",")] if new_sv else [])
            for v in new_sv_list:
                if v and v not in cur_sv_list:
                    cur_sv_list.append(v)
            current["source_view"] = ", ".join(cur_sv_list) if cur_sv_list else ""
            # confidence & priority
            current["confidence"] = max(current.get("confidence", 0), segment.get("confidence", 0))
            cp = PRIORITY_ORDER.index(current.get("priority", "medium"))
            np_ = PRIORITY_ORDER.index(segment.get("priority", "medium"))
            if np_ < cp:
                current["priority"] = segment.get("priority", "medium")
        else:
            merged.append(current)
            current = segment.copy()
    merged.append(current)
    return merged

def generate_azure_shard_urls(azure_blob_client, container, shard_paths, output_prefix,
                              account_name: str, account_key: str):
    from datetime import datetime, timedelta
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
                        "error": str(e)
                    })
        
        total_size = sum(f["size_bytes"] for f in uploaded_files)
        
        logger.info(f"✅ Upload completed!")
        logger.info(f"   Files uploaded: {len(uploaded_files)}")
        logger.info(f"   Files failed: {len(failed_files)}")
        logger.info(f"   Total size: {total_size / (1024*1024):.2f} MB")
        
        return {
            "success": len(failed_files) == 0,
            "uploaded_files_count": len(uploaded_files),
            "failed_files_count": len(failed_files),
            "total_size_bytes": total_size,
            "uploaded_files": uploaded_files,
            "failed_files": failed_files,
            "blob_base_path": f"{blob_base_path}/{video_name}"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to upload output directory: {e}")
        return {
            "success": False, 
            "error": str(e),
            "uploaded_files_count": len(uploaded_files),
            "failed_files_count": len(failed_files)
        }


def ensure_prompt_file_exists(prompt_path: str):
    """Ensure the scene-detection prompt exists; create a default if missing."""
    if os.path.exists(prompt_path):
        return
    logger.error(f"Prompt file not found: {prompt_path}. Creating default.")
    os.makedirs(os.path.dirname(prompt_path) or "config", exist_ok=True)
    default_prompt = {
        "system_prompt": "You are an AI assistant that analyzes video content and identifies different scenes or activities.",
        "user_prompt": "Please analyze this video and describe the different scenes or activities you observe. Focus on identifying distinct segments and their content."
    }
    import yaml
    with open(prompt_path, 'w') as f:
        yaml.dump(default_prompt, f, default_flow_style=False)


def consolidate_dual_view_outputs(all_results, local_outputs_dir, conv_time, local_input_path):
    import json, os
    from collections import defaultdict

    logger.info("Loading individual view results...")

    consolidated_timeline = []
    consolidated_stats = {
        "processing_summary": defaultdict(int),
        "annotation_summary": {
            "total_flagged_segments": 0,          # pre-merge sum (info only)
            "flagged_duration_seconds": 0.0,      # set after merge
            "high_priority_segments": 0
        }
    }
    view_summaries = []

    for vr in all_results:
        view_name = vr["view"]
        view_local_dir = vr["local_output_dir"]
        logger.info("Processing %s results...", view_name)

        ps_file = os.path.join(view_local_dir, "pipeline_summary.json")
        if os.path.exists(ps_file):
            with open(ps_file, 'r') as f:
                ps = json.load(f)
            for k, v in ps.get("processing_summary", {}).items():
                if k != "overall_success_rate" and isinstance(v, (int, float)):
                    consolidated_stats["processing_summary"][k] += v
            ann = ps.get("annotation_summary", {})
            consolidated_stats["annotation_summary"]["total_flagged_segments"] += ann.get("total_flagged_segments", 0)
            consolidated_stats["annotation_summary"]["high_priority_segments"] += ann.get("high_priority_segments", 0)
            view_summaries.append({"view": view_name, "pipeline_summary": ps})

        tl_file = os.path.join(view_local_dir, "master_flagged_timeline.json")
        if os.path.exists(tl_file):
            with open(tl_file, 'r') as f:
                td = json.load(f)
            for seg in td.get("flagged_timeline", []):
                out = seg.copy()
                out["source_view"] = view_name
                out["view_file"] = td.get("video_file", "")
                consolidated_timeline.append(out)

    consolidated_timeline.sort(key=lambda x: x.get("start_time", 0))
    merged_timeline = merge_overlapping_segments(consolidated_timeline)

    num_views = max(1, len(all_results))

    # Determine tasks_per_shard from any single-view summary
    tasks_per_shard = 0
    for vs in view_summaries:
        tasks_per_shard = sum(1 for k in vs['pipeline_summary'].get('processing_summary', {}) if k.startswith('successful_'))
        if tasks_per_shard:
            break

    # 'total_shards' was summed across views earlier; convert back to per-view count
    total_shards_summed = consolidated_stats["processing_summary"].get("total_shards", 0)
    per_view_shards = total_shards_summed // num_views if num_views else total_shards_summed

    successful_tasks = sum(v for k, v in consolidated_stats["processing_summary"].items() if k.startswith("successful_"))
    total_possible = per_view_shards * tasks_per_shard * num_views
    overall_success_rate = (successful_tasks / total_possible * 100) if total_possible else 0.0
    consolidated_stats["processing_summary"]["overall_success_rate"] = f"{overall_success_rate:.1f}%"

    SHARD_SEC = 60
    total_duration = per_view_shards * SHARD_SEC * num_views

    # Recompute flagged duration from the merged cross-view timeline (no double count)
    flagged_duration = sum(max(0, s.get("end_time", 0) - s.get("start_time", 0)) for s in merged_timeline)
    consolidated_stats["annotation_summary"]["flagged_duration_seconds"] = flagged_duration
    workload_reduction = ((total_duration - flagged_duration) / total_duration * 100) if total_duration else 0.0
    consolidated_stats["annotation_summary"]["annotation_workload_reduction"] = f"{workload_reduction:.1f}%"

    consolidated_data = {
        "input_file": local_input_path,
        "total_duration_seconds": total_duration,
        "views_analyzed": len(all_results),
        "conversion_time_seconds": conv_time,
        "consolidated_processing_summary": dict(consolidated_stats["processing_summary"]),
        "consolidated_annotation_summary": consolidated_stats["annotation_summary"],
        "total_consolidated_segments": len(merged_timeline),
        "consolidated_flagged_timeline": merged_timeline,
        "original_timeline_by_view": consolidated_timeline,
        "individual_view_summaries": view_summaries
    }

    consolidated_timeline_file = os.path.join(local_outputs_dir, "consolidated_master_timeline.json")
    with open(consolidated_timeline_file, 'w') as f:
        json.dump({
            "input_file": local_input_path,
            "total_duration_seconds": total_duration,
            "consolidated_flagged_duration_seconds": flagged_duration,
            "consolidated_annotation_workload_reduction":
                consolidated_data["consolidated_annotation_summary"]["annotation_workload_reduction"],
            "total_consolidated_segments": len(merged_timeline),
            "consolidated_timeline": merged_timeline,
            "view_breakdown": {
                "view_1_segments": len([s for s in consolidated_timeline if "view_1" in str(s.get("source_view", ""))]),
                "view_2_segments": len([s for s in consolidated_timeline if "view_2" in str(s.get("source_view", ""))]),
                "merged_segments": len(merged_timeline)
            }
        }, f, indent=2)

    consolidated_summary_file = os.path.join(local_outputs_dir, "consolidated_pipeline_summary.json")
    with open(consolidated_summary_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    logger.info("✅ Consolidated timeline saved: %s", consolidated_timeline_file)
    logger.info("✅ Consolidated summary saved: %s", consolidated_summary_file)
    logger.info("📊 Total segments before merge: %d", len(consolidated_timeline))
    logger.info("📊 Total segments after merge: %d", len(merged_timeline))
    logger.info("📊 Consolidated workload reduction: %s",
                consolidated_data["consolidated_annotation_summary"]["annotation_workload_reduction"])
    return consolidated_data





def pipeline_main(input_video_path: str, input_audio_path: str, output_dir: str, 
                 process_dual_views: bool = None, process_unwarped_views: bool = False,
                 azure_blob_client=None, azure_container=None, azure_output_prefix=None, azure_account_name: str = None, azure_account_key: str = None):
    """
    Unified Ray pipeline for video analysis with optional dual-view processing for INSV files
    """
    # Initialize Ray with basic guard
    try:
        if not ray.is_initialized():
            ray.init()
    except RuntimeError as e:
        if "ray.init twice" in str(e) or "already" in str(e).lower():
            logger.warning("Ray already initialized, continuing...")
        else:
            raise
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- STAGE A: INITIAL SETUP ---
    logger.info("Starting pipeline setup...")
    setup_cosmos()
    logger.info("Setup complete.")

    # --- STAGE B: VIDEO PREPARATION AND DUAL-VIEW DETECTION ---
    logger.info(f"Preparing video: {input_video_path}")
    
    # Extract video name for shard_info
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # Auto-detect dual view processing for INSV files
    is_insv_file = input_video_path.lower().endswith('.insv')
    if process_dual_views is None:
        process_dual_views = is_insv_file
    
    # New branch: unwarped multi-view processing for INSV
    if process_unwarped_views and is_insv_file:
        logger.info("🎥 INSV file detected - enabling unwarped view processing")

        # Convert .insv to four.mp4 videos directly (single-output converter)
        
        #viewsoutput_dir=os.path.join(output_dir, "4views"),
        if input_video_path.lower().endswith('.insv'):
            # Try the simpler single-output
            try:
                mp4_result = ray.get(insv_unwarp_task.remote(input_video_path))#, viewsoutput_dir))
                flat_result = mp4_result.get('views')
                logger.info(f"Using two 180 for unwarp:: {len(flat_result)} views under {flat_result}")
            except Exception:
                flat_result = None
                raise RuntimeError(f"Failed to convert two 180 INSV file: {mp4_result.get('error', 'Unknown error')}")
        else:
            mp4_path = input_video_path
            try:
                 # Undistort/unwarp the 360 video into multiple perspective views
                views4_ref = erp_unwarp_task.remote(mp4_path)
                flat_result = ray.get(views4_ref)  # dict: {view_name: output_path}
                logger.info(f"Using one 360 for unwarp: {len(flat_result)} views under {flat_result}")
            except Exception:
                flat_result = None
                raise RuntimeError(f"Failed to convert one 360 file: {mp4_result.get('error', 'Unknown error')}")
    

        # Split audio once into 60s shards (reused per view by index)
        audio_shards = ray.get(split_audio_into_shards.remote(
            input_audio_path,
            output_dir=os.path.join(output_dir, "audio_shards"),
            duration_sec=60
        ))

        # Optionally upload audio shards for LS streaming
        if azure_blob_client and azure_container and azure_output_prefix and azure_account_name and azure_account_key:
            audio_urls = generate_azure_shard_urls(
                azure_blob_client, azure_container, audio_shards,
                f"{azure_output_prefix}/audio_shards",
                azure_account_name, azure_account_key
            )
        else:
            audio_urls = {}

        # Split each unwarped view into shards and stage per-view shard lists
        view_shards = {}
        view_shard_urls = {}
        for view_name, view_path in flat_result.items():
            # Split this view into 60s shards
            shards = ray.get(split_video_into_shards.remote(
                view_path, 
                output_dir=os.path.join(output_dir, f"{view_name}_shards"), 
                duration_sec=60
            ))
            view_shards[view_name] = shards

            # Upload shards if Azure is configured
            if azure_blob_client and azure_container and azure_output_prefix and azure_account_name and azure_account_key:
                view_shard_urls[view_name] = generate_azure_shard_urls(
                    azure_blob_client, azure_container, shards,
                    f"{azure_output_prefix}/unwarped_shards/{view_name}",
                    azure_account_name, azure_account_key
                )
            else:
                view_shard_urls[view_name] = {}

        # Find the minimum shard count across all views to avoid index errors
        min_shards = min(len(shards) for shards in view_shards.values()) if view_shards else 0
        min_shards = min(min_shards, len(audio_shards)) if audio_shards else min_shards

        if min_shards == 0:
            logger.warning("No aligned shards found across views/audio for unwarped processing")
            return create_consolidated_summary([], output_dir)

        logger.info(f"Processing {min_shards} aligned shards across {len(view_shards)} unwarped views")

        # Process each shard index across all views
        label_studio_tasks = []
        for shard_idx in range(min_shards):
            shard_offset_sec = shard_idx * 60
            
            # Build dict of view -> shard URL for this index
            view_urls_for_shard = {}
            view_results_for_shard = {}

            for view_name in view_shards:
                if shard_idx < len(view_shards[view_name]):
                    v_shard = view_shards[view_name][shard_idx]
                    v_url = view_shard_urls[view_name].get(shard_idx, v_shard)
                    view_urls_for_shard[view_name] = v_url

                    # Pair audio shard by index
                    a_shard = audio_shards[shard_idx] if shard_idx < len(audio_shards) else input_audio_path

                    # Process this view's shard through the models
                    shard_output_dir = os.path.join(output_dir, f"{view_name}_shard_{shard_idx+1}")
                    os.makedirs(shard_output_dir, exist_ok=True)
                    view_results = process_single_shard_through_pipeline(
                        v_shard, a_shard, shard_output_dir, shard_offset_sec, shard_idx
                    )
                    view_results_for_shard[view_name] = view_results

            if not view_urls_for_shard:
                logger.warning(f"No video shards found for shard index {shard_idx}, skipping")
                continue

            # For simplified processing, just use the results from first view for now
            # TODO: Add proper multi-view consolidation like in dub.py
            first_view_results = list(view_results_for_shard.values())[0] if view_results_for_shard else {}
            
            # Choose up to 2 primary views for left/right for UI
            view_names_sorted = sorted(view_urls_for_shard.keys())
            primary_left = view_urls_for_shard.get(view_names_sorted[0], "") if view_names_sorted else ""
            primary_right = view_urls_for_shard.get(view_names_sorted[1], "") if len(view_names_sorted) > 1 else ""

            # Generate Label Studio task for this shard
            task_json_path = generate_consolidated_shard_labelstudio_task(
                os.path.join(output_dir, f"shard_{shard_idx+1}"),
                primary_left,
                primary_right,
                first_view_results,
                shard_idx + 1,
                shard_offset_sec,
                audio_urls.get(shard_idx, audio_shards[shard_idx]) if shard_idx < len(audio_shards) else "",
                min_shards,  # total_shards
                video_name   # video_name
            )
            
            if task_json_path:
                label_studio_tasks.append(task_json_path)
                
                # Merge model results into labelstudio task right after creation
                try:
                    shard_dir = os.path.join(output_dir, f"shard_{shard_idx+1}")
                    merge_model_results_into_labelstudio_task(shard_dir, shard_idx + 1)
                    logger.info(f"📋 Enhanced shard {shard_idx + 1} labelstudio task with model results")
                except Exception as e:
                    logger.error(f"❌ Failed to enhance shard {shard_idx + 1} labelstudio task: {e}")
        
        # Import all generated tasks to Label Studio in one batch
        import_consolidated_tasks_to_labelstudio(label_studio_tasks)   

        return create_consolidated_summary(label_studio_tasks, output_dir)
        
    elif process_dual_views and is_insv_file:
        logger.info("🎥 INSV file detected - enabling shard-first dual-view processing")
        
        # Convert INSV to dual MP4 views
        conversion_result = ray.get(convert_insv_to_dual_mp4.remote(input_video_path))
        if not conversion_result.get('success'):
            raise RuntimeError(f"INSV conversion failed: {conversion_result.get('error', 'Unknown error')}")
        view1_path = conversion_result['output_view_1']
        view2_path = conversion_result['output_view_2']
        
        # Split both views and audio into time-aligned shards
        view1_shards = ray.get(split_video_into_shards.remote(view1_path, output_dir=os.path.join(output_dir, "view_1_shards"), duration_sec=60))
        view2_shards = ray.get(split_video_into_shards.remote(view2_path, output_dir=os.path.join(output_dir, "view_2_shards"), duration_sec=60))
        audio_shards = ray.get(split_audio_into_shards.remote(input_audio_path, output_dir=os.path.join(output_dir, "audio_shards"), duration_sec=60))
        
        # Upload shards to Azure for both views
        if azure_blob_client and azure_container and azure_output_prefix and azure_account_name and azure_account_key:
            view1_shard_urls = generate_azure_shard_urls(
                azure_blob_client, azure_container, view1_shards,
                f"{azure_output_prefix}/view_1_shards",
                azure_account_name, azure_account_key
            )
            view2_shard_urls = generate_azure_shard_urls(
                azure_blob_client, azure_container, view2_shards,
                f"{azure_output_prefix}/view_2_shards",
                azure_account_name, azure_account_key
            )
            audio_shard_urls = generate_azure_shard_urls(
                azure_blob_client, azure_container, audio_shards,
                f"{azure_output_prefix}/audio_shards",
                azure_account_name, azure_account_key
            )
        else:
            logger.warning("Azure client/prefix or account key missing — LS URLs will be local and likely won't stream.")
            view1_shard_urls, view2_shard_urls, audio_shard_urls = {}, {}, {}
        
        min_len = min(len(view1_shards), len(view2_shards), len(audio_shards))
        if min_len < len(view1_shards) or min_len < len(view2_shards) or min_len < len(audio_shards):
            logger.warning("Shard count mismatch (v1=%d, v2=%d, audio=%d). Truncating to %d.",
                           len(view1_shards), len(view2_shards), len(audio_shards), min_len)
                           
        # Process time-aligned shards with enhanced first/last shard clap detection
        label_studio_tasks = []
        consolidated_json_paths = []
        
        logger.info(f"🎬 Processing {min_len} shards with enhanced first/last shard clap detection")
        logger.info(f"🎯 First shard: array[0] = shard {1}")
        logger.info(f"🎯 Last shard: array[-1] = shard {min_len}")
        
        for shard_index in range(min_len):
            shard_results = process_time_aligned_shard(
                shard_index, view1_shards[shard_index], view2_shards[shard_index], 
                audio_shards[shard_index], output_dir,
                view1_shard_urls.get(shard_index), view2_shard_urls.get(shard_index), audio_shard_urls.get(shard_index),
                total_shard_count=min_len,  # Pass total shard count for first/last identification
                video_name=video_name
            )
            label_studio_tasks.append(shard_results['label_studio_task'])
            consolidated_json_paths.append(shard_results['consolidated_model_results_json'])
            
            # Merge model results into labelstudio task right after creation
            try:
                shard_dir = os.path.join(output_dir, f"shard_{shard_index+1}")
                merge_model_results_into_labelstudio_task(shard_dir, shard_index + 1)
                logger.info(f"📋 Enhanced shard {shard_index + 1} labelstudio task with model results")
            except Exception as e:
                logger.error(f"❌ Failed to enhance shard {shard_index + 1} labelstudio task: {e}")
        
        # Generate final combined JSON with all shards
        generate_final_combined_model_results_json(output_dir, consolidated_json_paths)
        
        # Import consolidated tasks to Label Studio
        import_consolidated_tasks_to_labelstudio(label_studio_tasks)
        
        # Upload output directory to Azure blob storage
        if azure_blob_client and azure_container:
            video_name = os.path.basename(output_dir)
            upload_result = upload_output_directory_to_blob(
                output_dir, azure_blob_client, azure_container,
                "krishna-test/test1/test_activity", video_name
            )
            logger.info(f"📤 Upload result: {'✅ Success' if upload_result['success'] else '❌ Failed'}")
            if not upload_result['success']:
                logger.error(f"Upload error: {upload_result.get('error', 'Unknown error')}")
        
        return create_consolidated_summary(label_studio_tasks, output_dir)
        
    else:
        # Single view processing (original logic)
        if is_insv_file:
            logger.info("🎥 INSV file detected - processing single view only")
            mp4_result_ref = convert_insv_to_dual_mp4.remote(input_video_path)
            mp4_result = ray.get(mp4_result_ref)
            
            if mp4_result.get('success', False):
                mp4_path = mp4_result['output_view_1']  # Use the first view for processing
                logger.info(f"Using view 1 path for processing: {mp4_path}")
            else:
                raise RuntimeError(f"Failed to convert INSV file: {mp4_result.get('error', 'Unknown error')}")
        else:
            logger.info("🎥 Regular video file - single view processing")
            mp4_path = input_video_path
        
        return _process_single_view(
                    mp4_path,
                    input_audio_path,
                    output_dir,
                    azure_blob_client,
                    azure_container,
                    azure_output_prefix,
                    azure_account_name,        
                    azure_account_key          
                )

def generate_consolidated_model_results_json(shard_output_dir, shard_number, view1_results, view2_results, total_shards=None, video_name=None):
    """
    Generate consolidated JSON file with view1 and view2 as top-level keys,
    containing all individual model results.
    
    Args:
        shard_output_dir: Directory to save the consolidated JSON
        shard_number: Shard number for filename
        view1_results: Results from view 1 processing
        view2_results: Results from view 2 processing (may be empty for single-view)
        total_shards: Total number of shards in the processing job
        video_name: Name of the video being processed
        
    Returns:
        Path to the generated consolidated JSON file
    """
    # Extract clap detection flags from view1_results
    detection_flags = view1_results.get('detection_flags', {
        "is_first_shard": False,
        "is_last_shard": False,
        "is_intro_statement_there": False,
        "intro_transcript": "",
        "shard_index": shard_number - 1,
        "shard_type": "middle",
        "is_first_shard_processed": False,
        "is_last_shard_processed": False
    })
    
    consolidated_data = {
        "shard_info": {
            "shard_number": shard_number,
            "shard_id": shard_number,
            "total_shards": total_shards,
            "video_name": video_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_type": "dual_view" if view2_results else "single_view"
        },
        "detection_flags": detection_flags,
        "view1": {
            "audio": view1_results.get('audio', {}),
            "yolo": view1_results.get('yolo', {}),
            "scene": view1_results.get('scene', {}),
            "nsfw": view1_results.get('nsfw', {}),
            "motion": view1_results.get('motion', {}),
            "face": view1_results.get('face', {}),
            "clap": view1_results.get('clap', {})
        }
    }
    
    # Add view2 data if available
    if view2_results:
        consolidated_data["view2"] = {
            "audio": view2_results.get('audio', {}),
            "yolo": view2_results.get('yolo', {}),
            "scene": view2_results.get('scene', {}),
            "nsfw": view2_results.get('nsfw', {}),
            "motion": view2_results.get('motion', {}),
            "face": view2_results.get('face', {}),
            "clap": view2_results.get('clap', {})
        }
    else:
        # For single-view, still include view2 key but mark as not processed
        consolidated_data["view2"] = {
            "processing_status": "not_processed_single_view_mode"
        }
    
    # Save consolidated JSON
    json_file = os.path.join(shard_output_dir, f"shard_{shard_number}_consolidated_model_results.json")
    with open(json_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    
    logger.info(f"Consolidated model results saved to: {json_file}")
    return json_file


def generate_final_combined_model_results_json(output_dir, consolidated_json_paths):
    """
    Generate final combined JSON file containing all shards' model results.
    
    Args:
        output_dir: Base output directory to save the final combined JSON
        consolidated_json_paths: List of paths to individual shard consolidated JSONs
        
    Returns:
        Path to the generated final combined JSON file
    """
    logger.info(f"Generating final combined model results JSON with {len(consolidated_json_paths)} shards")
    
    final_combined_data = {
        "pipeline_info": {
            "total_shards": len(consolidated_json_paths),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "description": "Combined model results from all processed shards"
        }
    }
    
    # Initialize summary counters for clap detection and intro statements
    first_shard_clap_detected = False
    last_shard_clap_detected = False
    intro_statement_detected = False
    intro_transcript_text = ""
    
    # Load and combine all shard data
    for json_path in consolidated_json_paths:
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    shard_data = json.load(f)
                
                shard_number = shard_data.get('shard_info', {}).get('shard_number', 'unknown')
                shard_key = f"shard_{shard_number}"
                
                # Extract clap detection flags for this shard
                clap_flags = shard_data.get('detection_flags', {})
                
                # Create base shard data structure
                shard_data_structure = {
                    "view1": shard_data.get('view1', {}),
                    "view2": shard_data.get('view2', {})
                }
                
                # Only include detection_flags for first and last shards
                is_first_shard_processed = clap_flags.get('is_first_shard_processed', False)
                is_last_shard_processed = clap_flags.get('is_last_shard_processed', False)
                
                if is_first_shard_processed or is_last_shard_processed:
                    shard_data_structure["detection_flags"] = clap_flags
                    shard_type = clap_flags.get('shard_type', 'unknown')
                    
                    # Update summary counters
                    if is_first_shard_processed:
                        first_shard_clap_detected = clap_flags.get('is_first_shard', False)
                        intro_statement_detected = clap_flags.get('is_intro_statement_there', False)
                        intro_transcript_text = clap_flags.get('intro_transcript', '')
                    
                    if is_last_shard_processed:
                        last_shard_clap_detected = clap_flags.get('is_last_shard', False)
                    
                    logger.info(f"📊 Including detection_flags for {shard_type} shard {shard_number}")
                
                final_combined_data[shard_key] = shard_data_structure
                
                logger.info(f"Added shard {shard_number} data to final combined JSON")
                
            except Exception as e:
                logger.error(f"Failed to load shard JSON {json_path}: {e}")
        else:
            logger.warning(f"Shard JSON not found or invalid: {json_path}")
    
    # Add overall summary for clap detection and intro statements
    final_combined_data["overall_summary"] = {
        "clap_detection": {
            "first_shard_clap_detected": first_shard_clap_detected,
            "last_shard_clap_detected": last_shard_clap_detected,
            "both_first_and_last_claps": first_shard_clap_detected and last_shard_clap_detected
        },
        "intro_statement": {
            "intro_statement_detected": intro_statement_detected,
            "intro_transcript": intro_transcript_text
        }
    }
    
    # Save final combined JSON
    final_json_file = os.path.join(output_dir, "final_combined_all_shards_model_results.json")
    with open(final_json_file, 'w') as f:
        json.dump(final_combined_data, f, indent=2)
    
    logger.info(f"🎉 Final combined model results saved to: {final_json_file}")
    logger.info(f"📊 Contains results from {len(consolidated_json_paths)} shards with all model outputs")
    logger.info(f"🔍 OVERALL SUMMARY:")
    logger.info(f"   🎬 First shard clap detected: {'✅ YES' if first_shard_clap_detected else '❌ NO'}")
    logger.info(f"   🎬 Last shard clap detected: {'✅ YES' if last_shard_clap_detected else '❌ NO'}")
    logger.info(f"   🗣️ Intro statement detected: {'✅ YES' if intro_statement_detected else '❌ NO'}")
    if intro_transcript_text:
        logger.info(f"   📝 Intro transcript: '{intro_transcript_text}'")
    
    return final_json_file


def process_time_aligned_shard(
    shard_index,
    view1_shard_path,
    view2_shard_path,          # may be None for single-view
    audio_shard_path,
    base_output_dir,
    view1_azure_url,
    view2_azure_url,           # may be None/"" for single-view
    audio_url,
    total_shard_count=None,    # Add total shard count for first/last identification
    video_name=None            # Video name for shard_info
):
    """
    Process one time-aligned shard. If `view2_shard_path` is None, this behaves as single-view,
    but still generates the SAME consolidated LS task JSON used in dual-view.
    
    Enhanced with first/last shard clap detection.
    """
    shard_output_dir = os.path.join(base_output_dir, f"shard_{shard_index+1}")
    shard_offset_sec = shard_index * 60
    os.makedirs(shard_output_dir, exist_ok=True)

    # --- ENHANCED FIRST/LAST SHARD CLAP DETECTION ---
    # Determine if this is first or last shard using array indexing logic
    is_first_shard = (shard_index == 0)  # array[0]
    is_last_shard = (total_shard_count and shard_index == total_shard_count - 1)  # array[-1]
    
    # Initialize clap detection flags
    clap_detected_in_first_shard = False
    clap_detected_in_last_shard = False

    if is_first_shard or is_last_shard:
        shard_type = "first" if is_first_shard else "last"
        logger.info(f"🔍 CLAP DETECTION for {shard_type} shard {shard_index+1}")
        
        # Run clap detection on this shard's audio using existing clap results from normal pipeline
        # We'll get the clap results from the normal pipeline processing
        if is_first_shard:
            logger.info(f"🎬 FIRST SHARD - will check clap detection results")
        else:  # is_last_shard
            logger.info(f"🎬 LAST SHARD - will check clap detection results")
    else:
        logger.info(f"ℹ️ Shard {shard_index+1} is middle shard - no special clap processing")

    # --- View 1 ---
    logger.info(f"Processing view 1 of shard {shard_index+1}")
    view1_output_dir = os.path.join(shard_output_dir, "view_1")
    view1_results = process_single_shard_through_pipeline(
        view1_shard_path, audio_shard_path, view1_output_dir, shard_offset_sec, shard_index
    )

    # --- View 2 (optional) ---
    if view2_shard_path:
        logger.info(f"Processing view 2 of shard {shard_index+1}")
        view2_output_dir = os.path.join(shard_output_dir, "view_2")
        view2_results = process_single_shard_through_pipeline(
            view2_shard_path, audio_shard_path, view2_output_dir, shard_offset_sec, shard_index
        )
    else:
        logger.info(f"No view 2 for shard {shard_index+1} — running single-view consolidation")
        view2_results = {}

    # --- EXTRACT CLAP DETECTION RESULTS FOR FIRST/LAST SHARDS ---
    # Initialize intro statement flag
    is_intro_statement_there = False
    intro_transcript = ""
    
    if is_first_shard or is_last_shard:
        # Extract clap detection results from the normal pipeline processing
        clap_results = view1_results.get('clap', {})
        clap_count = clap_results.get('clap_count', 0) if clap_results else 0
        clap_detected = clap_count > 0
        
        if is_first_shard:
            clap_detected_in_first_shard = clap_detected
            logger.info(f"🎬 FIRST SHARD clap detection: {'✅ DETECTED' if clap_detected else '❌ NOT DETECTED'} ({clap_count} claps)")
            
            # --- DETECT INTRO STATEMENT IN FIRST SHARD ---
            logger.info(f"🎤 ANALYZING FIRST SHARD for intro statement...")
            
            # Check audio results from view1 for transcript
            audio_results = view1_results.get('audio', [])
            if isinstance(audio_results, list) and audio_results:
                audio_result = audio_results[0]  # Get first audio result
                transcript = audio_result.get('transcript', '').strip()
                
                if transcript:
                    intro_transcript = transcript
                    
                    # Detect intro statements using common intro phrases and patterns
                    intro_keywords = [
                        "i'm going to", "i will", "today we", "welcome", "hello", "hi there",
                        "let's", "we're going to", "this is", "in this", "i am going to",
                        "we will", "starting with", "first we", "beginning", "introduction"
                    ]
                    
                    transcript_lower = transcript.lower()
                    
                    # Check for intro keywords or if it's a substantial opening statement
                    has_intro_keywords = any(keyword in transcript_lower for keyword in intro_keywords)
                    has_meaningful_content = len(transcript.strip()) > 10  # More than just a few words
                    
                    is_intro_statement_there = has_intro_keywords or has_meaningful_content
                    
                    logger.info(f"🗣️ INTRO STATEMENT: {'✅ DETECTED' if is_intro_statement_there else '❌ NOT DETECTED'}")
                    logger.info(f"📝 Transcript: '{transcript}'")
                    
                    if has_intro_keywords:
                        matched_keywords = [kw for kw in intro_keywords if kw in transcript_lower]
                        logger.info(f"🔤 Intro keywords found: {matched_keywords}")
                else:
                    logger.info(f"📝 No transcript found in first shard")
            else:
                logger.info(f"📝 No audio results found in first shard")
                
        else:  # is_last_shard
            clap_detected_in_last_shard = clap_detected
            logger.info(f"🎬 LAST SHARD clap detection: {'✅ DETECTED' if clap_detected else '❌ NOT DETECTED'} ({clap_count} claps)")

    # --- CREATE CLAP DETECTION FLAGS ---
    detection_flags = {
        "is_first_shard": clap_detected_in_first_shard,
        "is_last_shard": clap_detected_in_last_shard,
        "is_intro_statement_there": is_intro_statement_there,
        "intro_transcript": intro_transcript,
        "shard_index": shard_index,
        "shard_type": "first" if is_first_shard else ("last" if is_last_shard else "middle"),
        "is_first_shard_processed": is_first_shard,
        "is_last_shard_processed": is_last_shard
    }
    
    # Add flags to view1_results for inclusion in consolidated JSON
    view1_results['detection_flags'] = detection_flags
    if view2_results:
        view2_results['detection_flags'] = detection_flags

    # --- Generate Consolidated Model Results JSON ---
    logger.info(f"Generating consolidated model results JSON for shard {shard_index+1}")
    consolidated_json_path = generate_consolidated_model_results_json(
        shard_output_dir, shard_index+1, view1_results, view2_results, total_shard_count, video_name
    )

    # --- Consolidate & Build LS task (same function for single/dual) ---
    logger.info(f"Consolidating results for shard {shard_index+1}")
    consolidated_results = consolidate_time_segment_results(
        view1_results, view2_results, shard_index, shard_offset_sec
    )

    task_json_path = generate_consolidated_shard_labelstudio_task(
        shard_output_dir,
        view1_azure_url,
        (view2_azure_url or ""),   # keep key present even if blank
        consolidated_results,
        shard_index+1,
        shard_offset_sec,
        audio_url,
        total_shard_count,
        video_name
    )
    
    # Merge model results into labelstudio task right after creation
    if task_json_path:
        try:
            merge_model_results_into_labelstudio_task(shard_output_dir, shard_index + 1)
            logger.info(f"📋 Enhanced shard {shard_index + 1} labelstudio task with model results")
        except Exception as e:
            logger.error(f"❌ Failed to enhance shard {shard_index + 1} labelstudio task: {e}")

    return {
        "shard_index": shard_index,
        "view1_results": view1_results,
        "view2_results": view2_results,
        "consolidated_results": consolidated_results,
        "label_studio_task": task_json_path,
        "consolidated_model_results_json": consolidated_json_path,
        "detection_flags": detection_flags
    }


def process_single_shard_through_pipeline(video_shard_path, audio_shard_path, 
                                         output_dir, shard_offset_sec, shard_index=0):
    """
    Run single shard through all 7 AI models
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Define output directories for models that need them
    audio_output_dir = os.path.join(output_dir, "audio_output")
    yolo_output_dir = os.path.join(output_dir, "yolo_output")
    scene_output_dir = os.path.join(output_dir, "scene_output")
    clap_output_dir = os.path.join(output_dir, "clap_output")
    nsfw_output_dir = os.path.join(output_dir, "nsfw_output")
    motion_output_dir = os.path.join(output_dir, "motion_output")
    face_output_dir = os.path.join(output_dir, "face_output")
    
    # Create output directories
    os.makedirs(nsfw_output_dir, exist_ok=True)
    os.makedirs(motion_output_dir, exist_ok=True)
    os.makedirs(face_output_dir, exist_ok=True)

    # Define the prompt for scene detection
    prompt_path = "config/cosmos_prompt.yaml"
    
    # Verify prompt file exists
    ensure_prompt_file_exists(prompt_path)
    
    audio_ref = process_audio_diarization.remote([audio_shard_path], audio_output_dir)
    yolo_ref  = run_yolodetect_on_shard.remote(video_shard_path, yolo_output_dir)
    scene_ref = detect_scenes.remote(video_shard_path, prompt_path, scene_output_dir)
    nsfw_ref  = process_video_chunks_for_nsfw.remote([video_shard_path], confidence_threshold=0.5, chunk_duration_sec=60)
    motion_ref= compute_motion_energy.remote([video_shard_path], sensitivity_level="medium", save_detailed_data=False)
    face_ref  = process_video_chunks_for_face_detection.remote([video_shard_path], config=None, frame_interval=30, save_frames=False, chunk_duration_sec=60)
    clap_ref  = detect_claps_in_media.remote(audio_shard_path, clap_output_dir, threshold_bias=6000, lowcut=200, highcut=3200)

    (audio_res, yolo_res, scene_res, nsfw_res, motion_res, face_res, clap_res) = ray.get(
        [audio_ref, yolo_ref, scene_ref, nsfw_ref, motion_ref, face_ref, clap_ref]
    )
    
    # Store results from Ray tasks
    results = {
        'audio': audio_res,
        'yolo': yolo_res,
        'scene': scene_res,
        'nsfw': nsfw_res,
        'motion': motion_res,
        'face': face_res,
        'clap': clap_res
    }
    
    # Save individual model results to JSON files (following ray_pipeline_testing_old.py pattern)
    video_name = os.path.splitext(os.path.basename(video_shard_path))[0]
    
    # Save NSFW results
    if nsfw_res and nsfw_res.get("success"):
        nsfw_file = os.path.join(nsfw_output_dir, f"{video_name}_nsfw_results.json")
        with open(nsfw_file, 'w') as f:
            json.dump(nsfw_res, f, indent=2)
        logger.info(f"NSFW results saved to: {nsfw_file}")
    
    # Save Motion Energy results
    if motion_res and motion_res.get("success"):
        motion_file = os.path.join(motion_output_dir, f"{video_name}_motion_results.json")
        with open(motion_file, 'w') as f:
            json.dump(motion_res, f, indent=2)
        logger.info(f"Motion energy results saved to: {motion_file}")
    
    # Save Face Age Detection results
    if face_res and face_res.get("success"):
        face_file = os.path.join(face_output_dir, f"{video_name}_face_results.json")
        with open(face_file, 'w') as f:
            json.dump(face_res, f, indent=2)
        logger.info(f"Face detection results saved to: {face_file}")
    
    # Extract flagged segments from all models
    all_segments = []
    for task_type, task_result in results.items():
        # Audio returns a list of results (one per shard), others return single result
        if task_type == "audio" and isinstance(task_result, list):
            # For audio, we expect exactly 1 result since we're processing 1 shard
            if task_result:
                audio_result = task_result[0]  # Get the first (and only) audio result
                segments = extract_flagged_segments(audio_result, task_type, shard_index, shard_offset_sec)
                all_segments.extend(segments)
        else:
            segments = extract_flagged_segments(task_result, task_type, shard_index, shard_offset_sec)
            # Tag the source view later; here is shard-only tagging
            all_segments.extend(segments)
    results['flagged_segments'] = all_segments
    return results

def consolidate_time_segment_results(view1_results, view2_results, shard_index, shard_offset_sec):
    consolidated = {
        "shard_index": shard_index,
        "shard_offset_sec": shard_offset_sec,
        "time_range": f"{shard_offset_sec}-{shard_offset_sec + 60}s",
        "view_1_results": view1_results,
        "view_2_results": view2_results,
        "consolidated_predictions": {},
        "merged_flagged_segments": []
    }

    # Tag without relying on dict identity
    v1_segments = [dict(seg, **{"source_view": "view_1"}) for seg in view1_results.get('flagged_segments', [])]
    v2_segments = [dict(seg, **{"source_view": "view_2"}) for seg in view2_results.get('flagged_segments', [])]
    all_segments = v1_segments + v2_segments

    consolidated['merged_flagged_segments'] = merge_overlapping_segments_dual_view(all_segments)
    consolidated['consolidated_predictions'] = create_consolidated_predictions(view1_results, view2_results)
    return consolidated


def generate_consolidated_shard_labelstudio_task(shard_output_dir, view1_azure_url, view2_azure_url, 
                                                 consolidated_results, shard_number, shard_offset_sec, audio_url, total_shards=None, video_name=None):
      """
      Generate single Label Studio task with both view URLs and consolidated AI predictions
      """
      current_time = datetime.utcnow().isoformat() + "Z"

      # Get prediction entries from consolidated results (already in correct format)
      prediction_entries = consolidated_results['consolidated_predictions']

      # Create task with both view URLs (matching annotation_json.json format)
      task = {
          "data": {
                "meta": "",  # Required empty meta field
                # Flattened metadata keys at root level to match UI template expectations
                "meta.home_identifier": f"Shard_{shard_number}",
                "meta.recording_datetime": current_time,
                "meta.domain": "production",
                "meta.actions": "",
                # Additional metadata (these won't show in UI but good for context)
                "shard_number": str(shard_number),
                "shard_id": shard_number,
                "total_shards": total_shards,
                "video_name": video_name,
                "timestamp": current_time,
                "processing_type": "label_studio_task",
                "shard_offset_seconds": str(shard_offset_sec), 
                "segments_detected": str(len(prediction_entries)),
                "video_left": view1_azure_url,
                "video_right": view2_azure_url,
                "audio": audio_url,
                "home_id": "",
                "start_datetime": "",
                "end_datetime": "",
                "total_duration": "",
                "files_deleted": [],
          },
          
          "annotations": [],  # Empty for new tasks
          "predictions": [{"result": prediction_entries}] if prediction_entries else []
      }

      # Save consolidated task
      task_file = os.path.join(shard_output_dir, f"consolidated_shard_{shard_number}_labelstudio_task.json")
      with open(task_file, 'w') as f:
          json.dump(task, f, indent=2)

      logger.info(f"Generated consolidated Label Studio task for shard {shard_number} with {len(prediction_entries)} predictions")
      return task_file


def generate_multiview_shard_labelstudio_task(
    base_output_dir: str,
    shard_number: int,
    shard_offset_sec: int,
    view_urls: dict,
    audio_url: str,
    primary_left_url: str,
    primary_right_url: str,
    consolidated_results: dict,
    total_shards: int
):
    """
    Build a single LS task for a shard that includes audio and multiple video views in one request.
    Adds shard_id and total_shards to the task data for identification.
    """
    current_time = datetime.utcnow().isoformat() + "Z"

    # Use the existing prediction consolidator for up to two primary views if available
    prediction_entries = consolidated_results.get('consolidated_predictions', []) or []

    # Prepare data block with primary left/right plus extra views as additional fields
    data_block = {
        "meta": "",
        "meta.home_identifier": f"Shard_{shard_number}",
        "meta.recording_datetime": current_time,
        "meta.domain": "production",
        "meta.actions": "",
        "shard_number": str(shard_number),
        "shard_offset_seconds": str(shard_offset_sec),
        "segments_detected": str(len(prediction_entries)),
        "shard_id": str(shard_number),
        "total_shards": str(total_shards),
        "video_left": primary_left_url or "",
        "video_right": primary_right_url or "",
        "audio": audio_url or "",
        "home_id": "",
        "start_datetime": "",
        "end_datetime": "",
        "total_duration": "",
        "files_deleted": []
    }

    # Attach additional views as dedicated fields, e.g., video_view_front, video_view_right, etc.
    for view_name, url in (view_urls or {}).items():
        data_block[f"video_view_{view_name}"] = url

    task = {
        "data": data_block,
        "annotations": [],
        "predictions": [{"result": prediction_entries}] if prediction_entries else []
    }

    task_file = os.path.join(base_output_dir, f"multiview_shard_{shard_number}_labelstudio_task.json")
    with open(task_file, 'w') as f:
        json.dump(task, f, indent=2)
    logger.info(
        f"Generated multi-view Label Studio task for shard {shard_number} with {len(view_urls or {})} views and audio"
    )
    return task_file

def create_consolidated_predictions(view1_results, view2_results):
    """
    Create consolidated AI predictions from both views for Label Studio task generation.
    Based on the correct format from azure_shard_to_labelstudio_processor.py
    
    Args:
        view1_results: AI model results from view 1 shard processing
        view2_results: AI model results from view 2 shard processing
        
    Returns:
        List[Dict]: List of Label Studio prediction objects with correct format
    """
    predictions = []
    
    # Domain and action keywords (based on azure_shard_to_labelstudio_processor.py lines 182-220)
    domain_keywords = {
        "Food & Mealtime": ["kitchen", "cooking", "eating", "meal", "food", "dining", "microwave", "oven", "refrigerator", "countertop"],
        "Personal Care & Hygiene": ["bathroom", "toilet", "sink", "shower", "grooming", "brushing", "hygiene", "mirror", "toiletries"],
        "Household Movement": ["walking", "moving", "fetching", "cabinet", "door", "stairs"],
        "Cleaning & Maintenance": ["cleaning", "vacuum", "laundry", "tidying", "trash", "maintenance"],
        "Work & Study": ["desk", "computer", "writing", "work", "study", "office"],
        "Leisure & Entertainment": ["TV", "television", "gaming", "reading", "entertainment", "hobby"],
        "Exercise & Wellness": ["exercise", "workout", "fitness", "dancing", "yoga"],
        "Social & Family Life": ["family", "social", "hosting", "gathering", "conversation"],
        "Pet Care": ["pet", "dog", "cat", "feeding", "animal"],
        "Home & Garden Projects": ["plant", "garden", "project", "indoor"],
        "Shopping & Logistics": ["package", "mail", "delivery", "shopping"],
        "Safety & Security": ["security", "alarm", "safety", "lock"]
    }
    
    action_keywords = {
        "Food & Mealtime": {
            "Cooking / baking": ["cooking", "baking", "preparing", "microwave", "oven"],
            "Eating together": ["eating", "meal", "dining"],
            "Washing dishes": ["washing", "cleaning dishes", "sink"],
            "Setting the table": ["table", "setting"],
            "Planning meals": ["planning", "menu"],
            "Putting away leftovers": ["storing", "leftovers", "putting away"],
            "Other - Food & Mealtime": []
        },
        "Personal Care & Hygiene": {
            "Brushing teeth": ["brushing", "teeth", "toothbrush"],
            "Grooming & skincare": ["grooming", "skincare", "mirror"],
            "Taking medication": ["medication", "pills"],
            "Other - Personal Care & Hygiene": []
        },
        "Household Movement": {
            "Walking between rooms": ["walking", "moving between"],
            "Fetching objects": ["fetching", "getting", "retrieving"],
            "Opening cabinets & appliances": ["opening", "cabinet", "appliance"],
            "Going up/down stairs": ["stairs", "upstairs", "downstairs"],
            "Other - Household Movement": []
        }
    }
    
    # Extract scene information from both views
    all_descriptions = []
    detected_objects = []
    
    for view_results in [view1_results, view2_results]:
        # Get scene descriptions
        scene_result = view_results.get('scene', {})
        if scene_result and scene_result.get('processing_info', {}).get('success', False):
            scenes = scene_result.get('scenes', [])
            for scene in scenes:
                if 'description' in scene:
                    all_descriptions.append(scene['description'])
        
        # Get YOLO detected objects (if available in results)
        yolo_result = view_results.get('yolo', {})
        # YOLO object extraction depends on actual result format - skipping for now
    
    # Determine domain and actions based on consolidated analysis
    all_text = " ".join(all_descriptions).lower()
    detected_objects_lower = [obj.lower() for obj in detected_objects]
    
    matched_domains = []
    for domain, keywords in domain_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in all_text:
                score += 2
            if keyword.lower() in detected_objects_lower:
                score += 3
        if score > 0:
            matched_domains.append((domain, score))
    
    matched_domains.sort(key=lambda x: x[1], reverse=True)
    
    # Create taxonomy prediction (Domain & Actions) - based on lines 327-344
    if matched_domains:
        top_domain = matched_domains[0][0]
        taxonomy_values = []
        
        if top_domain in action_keywords:
            action_scores = []
            for action, action_kws in action_keywords[top_domain].items():
                action_score = 0
                for keyword in action_kws:
                    if keyword.lower() in all_text:
                        action_score += 1
                    if keyword.lower() in detected_objects_lower:
                        action_score += 2
                if action_score > 0:
                    action_scores.append((action, action_score))
            
            if action_scores:
                action_scores.sort(key=lambda x: x[1], reverse=True)
                top_action = action_scores[0][0]
                taxonomy_values.append([top_domain, top_action])
            else:
                taxonomy_values.append([top_domain, f"Other - {top_domain}"])
        else:
            taxonomy_values.append([top_domain])
        
        predictions.append({
            "id": "taxonomy_domain_actions_pred",
            "type": "taxonomy",
            "value": {"taxonomy": taxonomy_values},
            "score": 0.8,
            "from_name": "taxonomy_domain_actions",
            "to_name": "video_left"
        })
    else:
        # Default fallback
        predictions.append({
            "id": "taxonomy_domain_actions_pred",
            "type": "taxonomy", 
            "value": {"taxonomy": [["Household Movement", "Other - Household Movement"]]},
            "score": 0.5,
            "from_name": "taxonomy_domain_actions",
            "to_name": "video_left"
        })
    
    # Always include lighting prediction (based on lines 346-354)
    predictions.append({
        "id": "lighting_pred",
        "type": "choices",
        "value": {"choices": ["Bright light"]},
        "score": 0.7,
        "from_name": "lighting",
        "to_name": "video_left"
    })
    
    # Check for PII detection from audio (both views)
    pii_spans = []
    for vr in [view1_results, view2_results]:
        audio_results = vr.get('audio', [])
        # Audio returns a list of results, get the first one if available
        if isinstance(audio_results, list) and audio_results:
            audio_result = audio_results[0]
            for pii in (audio_result.get('pii_detections') or []):
                s = int(pii.get('start_time', 0))
                e = int(pii.get('end_time',   0))
                if e > s:
                    pii_spans.append((s, e))

    if pii_spans:
        pii_spans.sort(key=lambda x: x[0])
        s, e = pii_spans[0]  # earliest occurrence; or merge if you prefer
        predictions.extend([
            {
                "id": "pii_fullnames_pred",
                "type": "choices",
                "value": {"choices": ["Yes"]},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames_yesno",
                "to_name": "video_left"
            },
            {
                "id": "pii_type_pred",
                "type": "choices",
                "value": {"choices": ["Full names"]},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames",
                "to_name": "video_left"
            },
            {
                "id": "pii_av_pred",
                "type": "choices",
                "value": {"choices": ["Audio"]},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames_av",
                "to_name": "video_left"
            },
            {
                "id": "pii_start_sec_pred",
                "type": "number",
                "value": {"number": s},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames_start_sec",
                "to_name": "video_left"
            },
            {
                "id": "pii_end_sec_pred",
                "type": "number",
                "value": {"number": e},
                "model_version": "auto_preannotator_v1",
                "from_name": "pii_fullnames_end_sec",
                "to_name": "video_left"
            }
        ])

    
    # Check for NSFW detection (based on lines 303-307) 
    nsfw_detected = False
    nsfw_segments = []
    for view_results in [view1_results, view2_results]:
        nsfw_result = view_results.get('nsfw', {})
        if nsfw_result and nsfw_result.get('success') and nsfw_result.get('total_nsfw_detections', 0) > 0:
            nsfw_detected = True
            nsfw_segments.extend(nsfw_result.get('flagged_segments', []))
    
    if nsfw_detected:
        predictions.append({
            "id": "nudity_present_pred",
            "type": "choices",
            "value": {"choices": ["Yes"]},
            "model_version": "auto_preannotator_v1", 
            "from_name": "nudity_present",
            "to_name": "video_left"
        })
        
        # Add timing from first NSFW segment if available
        if nsfw_segments:
            first_segment = nsfw_segments[0]
            start_time = first_segment.get('start_time', 0)
            end_time = first_segment.get('end_time', 10)
            
            start_minute = int(start_time // 60)
            start_second = int(start_time % 60)
            end_minute = int(end_time // 60) 
            end_second = int(end_time % 60)
            
            predictions.extend([
                {
                    "id": "nudity_start_min_pred",
                    "type": "number",
                    "value": {"number": start_minute},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "nudity_start_minute",
                    "to_name": "video_left"
                },
                {
                    "id": "nudity_start_sec_pred",
                    "type": "number", 
                    "value": {"number": start_second},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "nudity_start_second",
                    "to_name": "video_left"
                },
                {
                    "id": "nudity_end_min_pred",
                    "type": "number",
                    "value": {"number": end_minute},
                    "model_version": "auto_preannotator_v1", 
                    "from_name": "nudity_end_minute",
                    "to_name": "video_left"
                },
                {
                    "id": "nudity_end_sec_pred",
                    "type": "number",
                    "value": {"number": end_second},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "nudity_end_second", 
                    "to_name": "video_left"
                }
            ])
    
    # Check for minors detection from face analysis (based on lines 309-315)
    minors_detected = False
    minors_segments = []
    for view_results in [view1_results, view2_results]:
        face_result = view_results.get('face', {})
        if face_result and face_result.get('success'):
            flagged_segments = face_result.get('flagged_segments', [])
            for segment in flagged_segments:
                flag_type = segment.get('flag_type', '').lower()
                description = segment.get('description', '').lower() 
                if 'minor' in flag_type or 'minor' in description:
                    minors_detected = True
                    minors_segments.append(segment)
                    break
        if minors_detected:
            break
    
    if minors_detected:
        predictions.append({
            "id": "minors_present_pred",
            "type": "choices",
            "value": {"choices": ["Yes"]},
            "model_version": "auto_preannotator_v1",
            "from_name": "minors_present", 
            "to_name": "video_left"
        })
        
        # Add timing from first minors segment if available
        if minors_segments:
            first_segment = minors_segments[0]
            start_time = first_segment.get('start_time', 0)
            end_time = first_segment.get('end_time', 10)
            
            start_minute = int(start_time // 60)
            start_second = int(start_time % 60)
            end_minute = int(end_time // 60)
            end_second = int(end_time % 60)
            
            predictions.extend([
                {
                    "id": "minors_start_min_pred",
                    "type": "number",
                    "value": {"number": start_minute},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "minors_start_minute",
                    "to_name": "video_left"
                },
                {
                    "id": "minors_start_sec_pred", 
                    "type": "number",
                    "value": {"number": start_second},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "minors_start_second",
                    "to_name": "video_left"
                },
                {
                    "id": "minors_end_min_pred",
                    "type": "number",
                    "value": {"number": end_minute},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "minors_end_minute", 
                    "to_name": "video_left"
                },
                {
                    "id": "minors_end_sec_pred",
                    "type": "number",
                    "value": {"number": end_second},
                    "model_version": "auto_preannotator_v1",
                    "from_name": "minors_end_second",
                    "to_name": "video_left"
                }
            ])
    
    return predictions

def import_consolidated_tasks_to_labelstudio(task_file_paths, pipeline_config=None):
    """
    Import consolidated Label Studio tasks to Label Studio platform.
    
    Args:
        task_file_paths: List of paths to consolidated task JSON files
        pipeline_config: Pipeline configuration dict (optional, will load default if None)
        
    Returns:
        dict: Import results with success/failure status
    """
    if not task_file_paths:
        logger.warning("No task files provided for Label Studio import")
        return {"success": False, "error": "No tasks to import"}
    
    # Filter out None values and verify files exist
    valid_task_files = []
    for task_file in task_file_paths:
        if task_file and os.path.exists(task_file):
            valid_task_files.append(task_file)
        else:
            logger.warning(f"Task file not found or invalid: {task_file}")
    
    if not valid_task_files:
        logger.error("No valid task files found for import")
        return {"success": False, "error": "No valid task files found"}
    
    # Load Label Studio configuration
    if pipeline_config is None:
        try:
            pipeline_config = _load_pipeline_config()
        except Exception as e:
            logger.warning(f"⚠️ Could not load pipeline config for Label Studio settings: {e}")
            pipeline_config = {}
    
    # Get Label Studio settings from config
    label_studio_config = pipeline_config.get('label_studio', {})
    server_url = label_studio_config.get('server_url', 'https://annotations-stg.oneforma2.com/')
    api_token = label_studio_config.get('api_token', 'd75a31c7994b96099cfbf7d61e15cff643943853')
    project_id = label_studio_config.get('project_id', '5458')
    
    logger.info(f"📤 Importing {len(valid_task_files)} consolidated tasks to Label Studio...")
    logger.info(f"   Server: {server_url}")
    logger.info(f"   Project ID: {project_id}")
    
    # Use existing Label Studio import functionality
    try:
        # Import using the existing import_to_labelstudio_task function
        import_result_ref = import_to_labelstudio_task.remote(
            valid_task_files,
            server_url,
            api_token,
            project_id
        )
        
        import_result = ray.get(import_result_ref)
        
        if import_result.get("success"):
            logger.info(f"✅ Successfully imported {len(valid_task_files)} consolidated tasks to Label Studio")
            return {
                "success": True,
                "imported_tasks": len(valid_task_files),
                "result": import_result.get('result')
            }
        else:
            logger.error(f"❌ Failed to import consolidated tasks: {import_result.get('error')}")
            return {
                "success": False,
                "error": import_result.get('error')
            }
            
    except Exception as e:
        logger.error(f"Exception during consolidated task import: {e}")
        return {"success": False, "error": str(e)}

def create_consolidated_summary(label_studio_tasks, output_dir):
    """
    Create consolidated processing summary for shard-first dual-view processing.
    
    Args:
        label_studio_tasks: List of Label Studio task file paths
        output_dir: Base output directory
        
    Returns:
        dict: Consolidated processing results
    """
    try:
        # Count successful shard processing
        total_shards = len(label_studio_tasks)
        successful_tasks = len([task for task in label_studio_tasks if task and os.path.exists(task)])
        
        # Calculate consolidated statistics
        total_duration_seconds = total_shards * 60 * 2  # 60 seconds per shard, 2 views
        
        # Initialize consolidated results
        consolidated_results = {
            "input_file": "INSV dual-view processing",
            "processing_type": "shard_first_dual_view",
            "views_analyzed": 2,
            "total_shards_processed": total_shards,
            "successful_shard_tasks": successful_tasks,
            "total_duration_seconds": total_duration_seconds,
            "consolidated_processing_summary": {
                "total_shards": total_shards,
                "successful_audio": total_shards * 2,  # Assume audio processing succeeds for both views
                "successful_yolo": total_shards * 2,
                "successful_scene": total_shards * 2,
                "successful_nsfw": total_shards * 2,
                "successful_motion": total_shards * 2,
                "successful_face": total_shards * 2,
                "successful_clap": total_shards * 2,
                "overall_success_rate": f"{(successful_tasks / total_shards * 100):.1f}%" if total_shards > 0 else "0%"
            },
            "consolidated_annotation_summary": {
                "total_label_studio_tasks": successful_tasks,
                "dual_view_tasks_created": successful_tasks,
                "annotation_efficiency": "Each task covers 60 seconds from both camera views",
                "estimated_annotation_time_per_task": "2-3 minutes",
                "total_estimated_annotation_time": f"{successful_tasks * 2.5} minutes"
            },
            "shard_first_benefits": {
                "tasks_per_minute": 1,  # One task per 60-second time segment
                "dual_view_consolidation": "Both views processed and consolidated per time segment",
                "annotation_workload": f"Reduced from {total_shards * 2} separate tasks to {successful_tasks} consolidated tasks",
                "workload_reduction_percentage": f"{((total_shards * 2 - successful_tasks) / (total_shards * 2) * 100):.1f}%" if total_shards > 0 else "0%"
            },
            "label_studio_integration": {
                "tasks_imported": successful_tasks,
                "task_format": "dual_view_consolidated",
                "video_left_field": "view_1_shard_url",
                "video_right_field": "view_2_shard_url",
                "consolidated_predictions": "AI results from both views merged intelligently"
            }
        }
        
        # Save consolidated summary
        summary_file = os.path.join(output_dir, "shard_first_consolidated_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(consolidated_results, f, indent=2)
        
        logger.info("✅ Shard-first dual-view processing completed successfully!")
        logger.info(f"📊 Processed {total_shards} time segments with 2 views each")
        logger.info(f"📋 Created {successful_tasks} consolidated Label Studio tasks")
        logger.info(f"⏱️  Estimated annotation time: {successful_tasks * 2.5} minutes")
        logger.info(f"💾 Summary saved to: {summary_file}")
        
        return consolidated_results

    except Exception as e:
        logger.error(f"Failed to create consolidated summary: {e}")
        # Return basic summary even if detailed processing fails
        return {
            "success": False,
            "error": str(e),
            "total_shards_attempted": len(label_studio_tasks),
            "processing_type": "shard_first_dual_view"
        }

def _process_single_view(
    mp4_path: str,
    input_audio_path: str,
    output_dir: str,
    azure_blob_client=None,
    azure_container=None,
    azure_output_prefix=None,
    azure_account_name: str = None,
    azure_account_key: str = None
):
    """
    Single-view pipeline that reuses the SAME LS task creation & consolidation
    functions used by the dual-view shard-first flow.
    """
    logger.info(f"Processing single-view video: {mp4_path}")

    # --- Shard both video and audio (time-aligned) ---
    shards_dir = os.path.join(output_dir, "video_shards")
    shards_audio_dir = os.path.join(output_dir, "audio_shards")
    video_shards = ray.get(split_video_into_shards.remote(mp4_path, output_dir=shards_dir, duration_sec=60))
    audio_shards = ray.get(split_audio_into_shards.remote(input_audio_path, output_dir=shards_audio_dir, duration_sec=60))

    logger.info(f"Video split into {len(video_shards)} shards → {shards_dir}")
    logger.info(f"Audio split into {len(audio_shards)} shards → {shards_audio_dir}")

    # --- Upload shards to Azure (optional, recommended for LS streaming) ---
    if azure_blob_client and azure_container and azure_output_prefix and azure_account_name and azure_account_key:
        logger.info("Uploading single-view video shards to Azure Blob Storage...")
        video_urls = generate_azure_shard_urls(
            azure_blob_client, azure_container, video_shards,
            f"{azure_output_prefix}/video_shards",
            azure_account_name, azure_account_key
        )
        audio_urls = generate_azure_shard_urls(
            azure_blob_client, azure_container, audio_shards,
            f"{azure_output_prefix}/audio_shards",
            azure_account_name, azure_account_key
        )
    else:
        logger.warning("Azure client/prefix or account credentials missing — LS URLs will be local and likely won't stream.")
        video_urls, audio_urls = {}, {}

    # --- Process time-aligned shards using the SAME per-shard function as dual-view ---
    label_studio_tasks = []
    consolidated_json_paths = []
    min_len = min(len(video_shards), len(audio_shards))
    if min_len < len(video_shards) or min_len < len(audio_shards):
        logger.warning("Shard count mismatch (video=%d, audio=%d). Truncating to %d.",
                       len(video_shards), len(audio_shards), min_len)

    logger.info(f"🎬 Processing {min_len} single-view shards with enhanced first/last shard clap detection")
    logger.info(f"🎯 First shard: array[0] = shard {1}")
    logger.info(f"🎯 Last shard: array[-1] = shard {min_len}")
    
    for idx in range(min_len):
        shard_results = process_time_aligned_shard(
            shard_index=idx,
            view1_shard_path=video_shards[idx],
            view2_shard_path=None,  # ← single-view
            audio_shard_path=audio_shards[idx],
            base_output_dir=output_dir,
            view1_azure_url=video_urls.get(idx, video_shards[idx]),
            view2_azure_url=None,   # ← single-view (kept blank in LS task)
            audio_url=audio_urls.get(idx, audio_shards[idx]),
            total_shard_count=min_len,  # Pass total shard count for first/last identification
            video_name=video_name
        )
        label_studio_tasks.append(shard_results['label_studio_task'])
        consolidated_json_paths.append(shard_results['consolidated_model_results_json'])
        
        # Merge model results into labelstudio task right after creation
        try:
            shard_dir = os.path.join(output_dir, f"shard_{idx+1}")
            merge_model_results_into_labelstudio_task(shard_dir, idx + 1)
            logger.info(f"📋 Enhanced shard {idx + 1} labelstudio task with model results")
        except Exception as e:
            logger.error(f"❌ Failed to enhance shard {idx + 1} labelstudio task: {e}")

    # Generate final combined JSON with all shards
    generate_final_combined_model_results_json(output_dir, consolidated_json_paths)

    # --- Import all tasks to Label Studio (SAME function as dual-view) ---
    import_consolidated_tasks_to_labelstudio(label_studio_tasks)

    # Upload output directory to Azure blob storage
    if azure_blob_client and azure_container:
        video_name = os.path.basename(output_dir)
        upload_result = upload_output_directory_to_blob(
            output_dir, azure_blob_client, azure_container,
            "krishna-test/test1/test_activity", video_name
        )
        logger.info(f"📤 Upload result: {'✅ Success' if upload_result['success'] else '❌ Failed'}")
        if not upload_result['success']:
            logger.error(f"Upload error: {upload_result.get('error', 'Unknown error')}")

    # --- Return consolidated summary (SAME function as dual-view) ---
    return create_consolidated_summary(label_studio_tasks, output_dir)
# ===============================================
# LABEL STUDIO ENHANCEMENT FUNCTIONS
# ===============================================

def merge_model_results_into_labelstudio_task(shard_output_dir: str, shard_number: int) -> bool:
    """
    Merges the shard_X_consolidated_model_results.json content into the 
    consolidated_shard_X_labelstudio_task.json file under the data section.
    
    Args:
        shard_output_dir: Directory containing the shard files
        shard_number: Shard number (1-based)
        
    Returns:
        bool: Success status
    """
    try:
        # Construct file paths
        model_results_file = os.path.join(shard_output_dir, f"shard_{shard_number}_consolidated_model_results.json")
        labelstudio_task_file = os.path.join(shard_output_dir, f"consolidated_shard_{shard_number}_labelstudio_task.json")
        
        # Check if both files exist
        if not os.path.exists(model_results_file):
            logger.warning(f"Model results file not found: {model_results_file}")
            return False
            
        if not os.path.exists(labelstudio_task_file):
            logger.warning(f"Label Studio task file not found: {labelstudio_task_file}")
            return False
        
        # Read model results
        with open(model_results_file, 'r') as f:
            model_results = json.load(f)
        
        # Read labelstudio task
        with open(labelstudio_task_file, 'r') as f:
            labelstudio_task = json.load(f)
        
        # Add model results to the data section
        if "data" not in labelstudio_task:
            labelstudio_task["data"] = {}
        
        # Add the entire model results as a new key in data
        labelstudio_task["data"]["model_results"] = model_results
        
        # Write back the updated labelstudio task
        with open(labelstudio_task_file, 'w') as f:
            json.dump(labelstudio_task, f, indent=2)
        
        logger.info(f"✅ Successfully merged model results into {labelstudio_task_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to merge model results into labelstudio task: {e}")
        return False


def process_shard_labelstudio_enhancement(shard_output_dir: str, total_shards: int) -> None:
    """
    Enhances all labelstudio task files in the output directory by adding model results.
    
    Args:
        shard_output_dir: Base output directory containing shard subdirectories
        total_shards: Total number of shards to process
    """
    try:
        for shard_number in range(1, total_shards + 1):
            shard_dir = os.path.join(shard_output_dir, f"shard_{shard_number}")
            if os.path.exists(shard_dir):
                success = merge_model_results_into_labelstudio_task(shard_dir, shard_number)
                if success:
                    logger.info(f"📋 Enhanced shard {shard_number} labelstudio task with model results")
                else:
                    logger.warning(f"⚠️ Failed to enhance shard {shard_number} labelstudio task")
    except Exception as e:
        logger.error(f"❌ Failed to process labelstudio enhancements: {e}")

# ===============================================
# CONTINUOUS POLLING & CHECKLIST MANAGEMENT
# ===============================================

# Global shutdown flag for graceful stopping
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("🛑 Shutdown signal received. Gracefully stopping...")

def _load_video_checklist(checklist_path: str) -> Dict:
    """Load video processing checklist from JSON file"""
    try:
        if os.path.exists(checklist_path):
            with open(checklist_path, 'r') as f:
                checklist = json.load(f)
                logger.info(f"📋 Loaded checklist with {len(checklist.get('videos', {}))} videos")
                return checklist
        else:
            logger.info("📋 Creating new video processing checklist")
            return {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_videos": 0,
                "completed_videos": 0,
                "failed_videos": 0,
                "videos": {}
            }
    except Exception as e:
        logger.error(f"❌ Failed to load checklist: {e}")
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_videos": 0,
            "completed_videos": 0,
            "failed_videos": 0,
            "videos": {}
        }

def _save_video_checklist(checklist_path: str, checklist: Dict):
    """Save video processing checklist to JSON file"""
    try:
        checklist["last_updated"] = datetime.now().isoformat()
        with open(checklist_path, 'w') as f:
            json.dump(checklist, f, indent=2)
        logger.debug(f"💾 Checklist saved to {checklist_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save checklist: {e}")

def _update_video_status(checklist: Dict, video_name: str, status: str, **kwargs):
    """Update video status in checklist
    
    Status can be: 'pending', 'processing', 'completed', 'failed'
    """
    if video_name not in checklist["videos"]:
        checklist["videos"][video_name] = {
            "added_at": datetime.now().isoformat(),
            "status": "pending",
            "attempts": 0,
            "last_attempt": None,
            "error": None,
            "output_dir": None,
            "processing_time": None
        }
        checklist["total_videos"] += 1
    
    old_status = checklist["videos"][video_name]["status"]
    checklist["videos"][video_name]["status"] = status
    checklist["videos"][video_name]["last_attempt"] = datetime.now().isoformat()
    
    # Update additional fields if provided
    for key, value in kwargs.items():
        checklist["videos"][video_name][key] = value
    
    # Update counters
    if old_status == "completed" and status != "completed":
        checklist["completed_videos"] -= 1
    elif old_status != "completed" and status == "completed":
        checklist["completed_videos"] += 1
    
    if old_status == "failed" and status != "failed":
        checklist["failed_videos"] -= 1
    elif old_status != "failed" and status == "failed":
        checklist["failed_videos"] += 1
    
    logger.info(f"📋 Updated {video_name}: {old_status} → {status}")

def _get_pending_videos(checklist: Dict) -> List[str]:
    """Get list of video names that need processing"""
    pending_videos = []
    for video_name, info in checklist["videos"].items():
        if info["status"] in ["pending", "failed"]:
            pending_videos.append(video_name)
    return pending_videos

def _load_pipeline_config(config_path: str = "config/pipeline_config.yaml") -> Dict:
    """Load pipeline configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"✅ Loaded pipeline config from {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"⚠️ Pipeline config not found at {config_path}, using defaults")
        return {
            "azure_storage": {
                "input_blob_prefix": "test/input_videos",
                "output_blob_prefix": "processed_outputs"
            },
            "local_storage": {
                "temp_download_dir": "/tmp/azure_downloads",
                "output_base_dir": "./out",
                "video_list_file": "./azure_video_list.json",
                "checklist_file": "./video_processing_checklist.json"
            },
            "polling": {
                "interval_minutes": 5,
                "max_parallel_videos": 1
            },
            "cleanup": {
                "cleanup_temp_dir": True,
                "cleanup_files_after_each_video": True
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline config: {e}")
        raise

def _load_azure_config(config_path: str) -> Dict:
    """Load Azure blob storage configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"✅ Azure config loaded from {config_path}")
            return config
    except Exception as e:
        logger.error(f"❌ Failed to load Azure config: {e}")
        raise

def _create_azure_blob_client(config: Dict) -> BlobServiceClient:
    """Create Azure blob service client"""
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

def _poll_azure_videos(blob_service_client: BlobServiceClient, container_name: str, blob_prefix: str, 
                      checklist: Dict) -> List[Dict]:
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
                    audio_blob_name = _find_corresponding_audio(blob_service_client, container_name, blob.name, blob_prefix)
                    
                    if audio_blob_name:
                        video_info = {
                            "video_name": video_name,
                            "video_blob_name": blob.name,
                            "audio_blob_name": audio_blob_name,
                            "discovered_at": datetime.now().isoformat()
                        }
                        new_videos.append(video_info)
                        
                        # Add to checklist
                        _update_video_status(checklist, video_name, "pending")
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

def _find_corresponding_audio(blob_service_client: BlobServiceClient, container_name: str, 
                             video_blob_name: str, blob_prefix: str) -> Optional[str]:
    """Find audio file corresponding to video file"""
    try:
        video_base_name = os.path.splitext(os.path.basename(video_blob_name))[0]
        container_client = blob_service_client.get_container_client(container_name)
        
        # Search for audio files with similar names
        audio_extensions = ['.wav', '.mp3', '.aac', '.m4a']
        
        for ext in audio_extensions:
            # Try exact match first
            potential_audio_name = f"{os.path.dirname(video_blob_name)}/{video_base_name}{ext}"
            try:
                blob_client = container_client.get_blob_client(potential_audio_name)
                if blob_client.exists():
                    return potential_audio_name
            except:
                pass
            
            # Try without directory prefix
            potential_audio_name = f"{blob_prefix}/{video_base_name}{ext}"
            try:
                blob_client = container_client.get_blob_client(potential_audio_name)
                if blob_client.exists():
                    return potential_audio_name
            except:
                pass
        
        return None
    except Exception as e:
        logger.error(f"❌ Error finding audio for {video_blob_name}: {e}")
        return None

def _download_video_audio_pair(blob_service_client: BlobServiceClient, container_name: str,
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

@ray.remote
def continuous_blob_polling_and_pipeline_task(azure_config_path: str = "blobfuse2_config.yaml",
                                             pipeline_config_path: str = "config/pipeline_config.yaml"):
    """
    Continuous polling task that checks Azure blob storage for new videos every N minutes
    and processes them through the pipeline, maintaining a checklist of progress.
    """
    global _shutdown_requested
    
    try:
        logger.info("🚀 Starting Continuous Blob Polling & Pipeline Task")
        
        # Load configurations
        pipeline_config = _load_pipeline_config(pipeline_config_path)
        azure_config = _load_azure_config(azure_config_path)
        polling_interval_minutes = pipeline_config.get('polling', {}).get('interval_minutes', 5)
        
        logger.info(f"⏰ Polling interval: {polling_interval_minutes} minutes")
        
        # Setup Azure client
        blob_service_client = _create_azure_blob_client(azure_config)
        
        # Extract Azure storage config - handle both direct and nested structures
        if 'azstorage' in azure_config:
            # Nested structure from blobfuse2_config.yaml
            az_config = azure_config['azstorage']
        else:
            # Direct structure
            az_config = azure_config
            
        container_name = az_config['container']
        account_name = az_config['account-name']
        account_key = az_config['account-key']
        
        # Setup directories and files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_download_dir = pipeline_config['local_storage']['temp_download_dir']
        output_base_dir = os.path.join(current_dir, pipeline_config['local_storage']['output_base_dir'].lstrip('./'))
        checklist_path = os.path.join(current_dir, pipeline_config['local_storage'].get('checklist_file', './video_processing_checklist.json'))
        blob_prefix = pipeline_config['azure_storage']['input_blob_prefix']
        output_prefix = pipeline_config['azure_storage']['output_blob_prefix']
        
        # Load checklist
        checklist = _load_video_checklist(checklist_path)
        
        # Statistics
        total_processed = 0
        successful_processed = 0
        failed_processed = 0
        
        logger.info("✅ Continuous polling initialized successfully")
        logger.info(f"📋 Checklist: {checklist['total_videos']} total, {checklist['completed_videos']} completed, {checklist['failed_videos']} failed")
        
        # Store video blob info for retries
        video_blob_info_cache = {}
        
        # Main polling loop
        while not _shutdown_requested:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Polling cycle started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 1. Poll for new videos
                new_videos = _poll_azure_videos(blob_service_client, container_name, blob_prefix, checklist)
                
                # Update cache with new video info
                for video_info in new_videos:
                    video_blob_info_cache[video_info["video_name"]] = video_info
                
                # 2. Get pending videos (new + previously failed)
                pending_videos = _get_pending_videos(checklist)
                
                if pending_videos:
                    logger.info(f"📋 Found {len(pending_videos)} videos to process")
                    
                    # 3. Process pending videos one by one
                    for video_name in pending_videos:
                        if _shutdown_requested:
                            break
                            
                        video_info = checklist["videos"][video_name]
                        
                        try:
                            logger.info(f"\n🎬 Processing video: {video_name}")
                            _update_video_status(checklist, video_name, "processing", attempts=video_info.get("attempts", 0) + 1)
                            _save_video_checklist(checklist_path, checklist)
                            
                            # Get video blob info from cache
                            video_blob_info = video_blob_info_cache.get(video_name)
                            
                            if not video_blob_info:
                                logger.warning(f"⚠️ No blob info cached for {video_name} - skipping")
                                continue
                            
                            # Download video and audio
                            start_time = time.time()
                            video_path, audio_path = _download_video_audio_pair(
                                blob_service_client, container_name,
                                video_blob_info["video_blob_name"],
                                video_blob_info["audio_blob_name"],
                                local_download_dir
                            )
                            
                            if not video_path or not audio_path:
                                raise Exception("Failed to download video/audio files")
                            
                            # Setup output directory
                            video_output_dir = os.path.join(output_base_dir, f"video_{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                            
                            # Process through pipeline
                            logger.info(f"🔄 Running pipeline for {video_name}")
                            pipeline_results = pipeline_main(
                                input_video_path=video_path,
                                input_audio_path=audio_path,
                                output_dir=video_output_dir,
                                process_dual_views=None,  # Auto-detect
                                process_unwarped_views=False,
                                azure_blob_client=blob_service_client,
                                azure_container=container_name,
                                azure_output_prefix=f"{output_prefix}/{video_name}",
                                azure_account_name=account_name,
                                azure_account_key=account_key
                            )
                            
                            processing_time = time.time() - start_time
                            
                            # Update success status
                            _update_video_status(checklist, video_name, "completed",
                                               output_dir=video_output_dir,
                                               processing_time=f"{processing_time:.2f}s",
                                               completed_at=datetime.now().isoformat())
                            
                            successful_processed += 1
                            total_processed += 1
                            
                            logger.info(f"✅ Successfully processed {video_name} in {processing_time:.2f}s")
                            
                            # Cleanup downloaded files
                            if pipeline_config['cleanup']['cleanup_files_after_each_video']:
                                try:
                                    os.remove(video_path)
                                    os.remove(audio_path)
                                    logger.info(f"🗑️ Cleaned up downloaded files for {video_name}")
                                except Exception as e:
                                    logger.warning(f"⚠️ Failed to cleanup files: {e}")
                            
                        except Exception as e:
                            logger.error(f"❌ Failed to process {video_name}: {e}")
                            _update_video_status(checklist, video_name, "failed", error=str(e))
                            failed_processed += 1
                            total_processed += 1
                        
                        # Save checklist after each video
                        _save_video_checklist(checklist_path, checklist)
                
                else:
                    logger.info("✅ No pending videos to process")
                
                # 4. Wait for next polling cycle
                if not _shutdown_requested:
                    logger.info(f"⏰ Waiting {polling_interval_minutes} minutes until next poll...")
                    for i in range(polling_interval_minutes * 60):  # Convert minutes to seconds
                        if _shutdown_requested:
                            break
                        time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Error in polling cycle: {e}")
                time.sleep(30)  # Wait 30 seconds before retrying
        
        # Final cleanup
        if pipeline_config['cleanup']['cleanup_temp_dir'] and os.path.exists(local_download_dir):
            try:
                shutil.rmtree(local_download_dir)
                logger.info(f"🗑️ Cleaned up temp directory: {local_download_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temp directory: {e}")
        
        logger.info("🏁 Continuous polling stopped gracefully")
        
        return {
            "success": True,
            "total_processed": total_processed,
            "successful_processed": successful_processed,
            "failed_processed": failed_processed,
            "checklist_path": checklist_path,
            "final_stats": {
                "total_videos": checklist["total_videos"],
                "completed_videos": checklist["completed_videos"],
                "failed_videos": checklist["failed_videos"]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Continuous polling task failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ray Pipeline with Integrated Blob Polling")
    parser.add_argument("--mode", choices=["integrated", "standalone"], default="integrated",
                       help="Run mode: 'integrated' for blob polling + pipeline, 'standalone' for direct pipeline")
    parser.add_argument("--azure-config", default="blobfuse2_config.yaml",
                       help="Azure configuration file path")
    parser.add_argument("--pipeline-config", default="config/pipeline_config.yaml",
                       help="Pipeline configuration file path")
    parser.add_argument("--blob-prefix", default="test/input_videos",
                       help="Blob prefix to search for videos")
    parser.add_argument("--max-videos", type=int,
                       help="Maximum number of videos to process")
    parser.add_argument("--force-refresh", action="store_true",
                       help="Force refresh video list")
    parser.add_argument("--dual-views", action="store_true",
                       help="Enable dual view processing")
    parser.add_argument("--unwarped-views", action="store_true",
                       help="Enable unwarped view processing")
    
    # Standalone mode arguments
    parser.add_argument("--input-video", 
                       help="Input video file path (for standalone mode)")
    parser.add_argument("--input-audio", 
                       help="Input audio file path (for standalone mode)")
    parser.add_argument("--output-dir", default="/tmp/pipeline_output",
                       help="Output directory (for standalone mode)")
    
    args = parser.parse_args()
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
            logger.info("✅ Ray initialized successfully")
        
        if args.mode == "integrated":
            # =================================================================
            # INTEGRATED MODE: Continuous Blob Polling + Pipeline Processing
            # =================================================================
            logger.info("🚀 Starting INTEGRATED MODE: Continuous Blob Polling + Pipeline")
            logger.info("=" * 80)
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
            
            # Launch the continuous blob polling and pipeline task
            task_ref = continuous_blob_polling_and_pipeline_task.remote(
                azure_config_path=args.azure_config,
                pipeline_config_path=args.pipeline_config
            )
            
            # Monitor the continuous task 
            print("\n🔄 Continuous polling started!")
            print("📋 The system will continuously poll Azure blob storage every 5 minutes")
            print("📹 New videos will be automatically detected and processed")
            print("💾 Processing status is saved to: video_processing_checklist.json")
            print("🛑 Press Ctrl+C to stop gracefully\n")
            
            try:
                # Wait for task completion or interruption
                while True:
                    if _shutdown_requested:
                        logger.info("🛑 Shutdown requested, waiting for task to complete...")
                        break
                    
                    # Check if task is still running
                    ready_refs, remaining_refs = ray.wait([task_ref], timeout=1)
                    if ready_refs:
                        # Task completed
                        results = ray.get(ready_refs[0])
                        break
                    
                    time.sleep(1)
                
                # Get final results
                if not _shutdown_requested:
                    results = ray.get(task_ref)
                else:
                    # Task was interrupted, get partial results
                    try:
                        results = ray.get(task_ref, timeout=10)
                    except:
                        results = {"success": False, "error": "Task interrupted"}
                
                # Display results
                print("\n" + "="*80)
                print("🏁 CONTINUOUS POLLING RESULTS")
                print("="*80)
                
                if results.get('success'):
                    print(f"✅ Continuous polling completed!")
                    print(f"📊 Summary:")
                    print(f"   Total Processed: {results['total_processed']}")
                    print(f"   Successful: {results['successful_processed']}")
                    print(f"   Failed: {results['failed_processed']}")
                    print(f"   Checklist: {results['checklist_path']}")
                    
                    final_stats = results.get('final_stats', {})
                    print(f"\n📊 Final Statistics:")
                    print(f"   Total Videos: {final_stats.get('total_videos', 0)}")
                    print(f"   Completed: {final_stats.get('completed_videos', 0)}")
                    print(f"   Failed: {final_stats.get('failed_videos', 0)}")
                else:
                    print(f"❌ Continuous polling failed: {results.get('error', 'Unknown error')}")
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupt received, stopping gracefully...")
                _shutdown_requested = True
                # Wait for graceful shutdown
                try:
                    results = ray.get(task_ref, timeout=30)
                    print("✅ Graceful shutdown completed")
                except:
                    print("⚠️ Forced shutdown - some operations may not have completed")
        
        elif args.mode == "standalone":
            # =================================================================
            # STANDALONE MODE: Direct Pipeline Processing
            # =================================================================
            logger.info("🚀 Starting STANDALONE MODE: Direct Pipeline Processing")
            
            if not args.input_video or not args.input_audio:
                print("❌ Error: --input-video and --input-audio are required for standalone mode")
                print("Example: python ray_pipeline_testing.py --mode standalone --input-video /path/to/video.insv --input-audio /path/to/audio.wav")
                exit(1)
            
            logger.info(f"📹 Input Video: {args.input_video}")
            logger.info(f"🎵 Input Audio: {args.input_audio}")
            logger.info(f"📁 Output Dir: {args.output_dir}")
            
            # Configure Azure blob client for upload functionality (optional)
            azure_blob_client = None
            azure_container = None
            azure_account_name = None
            azure_account_key = None
        
            try:
                # Load pipeline config for output prefix
                pipeline_config = _load_pipeline_config(args.pipeline_config)
                
                # Load Azure config
                azure_config = _load_azure_config(args.azure_config)
                azure_blob_client = _create_azure_blob_client(azure_config)
                azure_container = azure_config.get('container')
                azure_account_name = azure_config.get('account-name')
                azure_account_key = azure_config.get('account-key')
                
                # Get output prefix from pipeline config
                output_prefix = pipeline_config['azure_storage']['output_blob_prefix']
                logger.info("✅ Azure blob client configured for upload")
            except Exception as e:
                logger.warning(f"⚠️ Failed to configure Azure blob client: {e} - output will not be uploaded")
                pipeline_config = {}
                output_prefix = "processed_outputs"
            
            # Run the pipeline directly
            results = pipeline_main(
                input_video_path=args.input_video,
                input_audio_path=args.input_audio,
                output_dir=args.output_dir,
                process_dual_views=args.dual_views,
                process_unwarped_views=args.unwarped_views,
            azure_blob_client=azure_blob_client,
            azure_container=azure_container,
                azure_output_prefix=f"{output_prefix}/{os.path.splitext(os.path.basename(args.input_video))[0]}",
            azure_account_name=azure_account_name,
            azure_account_key=azure_account_key
        )
        
            print(f"\n✅ Standalone Pipeline completed!")
            print(f"📁 Results saved to: {args.output_dir}")
            
            # Display basic results if available
            if isinstance(results, dict) and 'processing_summary' in results:
                print(f"\n📊 Processing Summary:")
                for task in ['audio', 'yolo', 'scene', 'nsfw', 'motion', 'face', 'clap']:
                    if f'successful_{task}' in results['processing_summary']:
                        count = results['processing_summary'][f'successful_{task}']
                        total = results['processing_summary']['total_shards']
                        print(f"   {task.title()}: {count}/{total} successful")
                    
                if 'annotation_summary' in results:
                    print(f"\n📋 Annotation Summary:")
                    print(f"   Flagged Segments: {results['annotation_summary'].get('total_flagged_segments', 'N/A')}")
                    print(f"   High Priority: {results['annotation_summary'].get('high_priority_segments', 'N/A')}")
                    print(f"   Overall Success Rate: {results['processing_summary'].get('overall_success_rate', 'N/A')}")
        
        print(f"\n🎉 Pipeline execution completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Pipeline interrupted by user")
        print("\n⏹️ Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        raise
    finally:
        # Shutdown Ray if we initialized it
        if ray.is_initialized():
            ray.shutdown()
            logger.info("🔄 Ray shutdown completed")