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
from utils.blob_utils import (
    load_azure_config, create_azure_blob_client, load_video_checklist, 
    update_video_status, save_video_checklist, get_pending_videos, get_checklist_statistics,
    detect_stuck_videos, reset_stuck_videos,
    poll_azure_videos_with_checklist, poll_azure_videos, find_corresponding_audio,
    download_video_audio_pair, download_video_audio_pair_simple, cleanup_downloaded_files,
    upload_output_directory_to_blob, upload_output_directory_with_sas, upload_output_directory_with_sas_optimized,
    update_labelstudio_tasks_with_new_urls, validate_shard_urls, load_pipeline_config,
    save_video_list_progress
)
from utils.dynamic_erp_processor import integrate_erp_audio_with_labelstudio_tasks
from utils.multi_chunk_blob_utils import download_ready_sessions
from utils.pipeline_timer import PipelineTimer
from utils.pipeline_timer import generate_hierarchical_timing_structure, generate_performance_summary
from utils.performance_csv_exporter import export_timing_data_to_single_csv


from azure.storage.blob import generate_blob_sas, BlobSasPermissions, BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Import setup and all necessary Ray tasks
from setup.cosmos.setup import setup_cosmos
from ray_jobs.video_splitter import split_video_into_shards, split_video_into_shards_with_overlap
from ray_jobs.audio_splitter import split_audio_into_shards, split_audio_into_shards_with_overlap
#from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
# Import scene detection with error handling
try:
    from ray_jobs.scene_det import detect_scenes
    SCENE_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Scene detection not available: {e}")
    SCENE_DETECTION_AVAILABLE = False
    # Create a dummy function
    def detect_scenes(*args, **kwargs):
        return {"success": False, "error": "Scene detection not available"}

# Import domain classification simple for RAG-based domain detection
from domain_classification_simple import DomainClassifier
from ray_jobs.yolo_detection import run_yolo_detection, extract_yolo_people_data, generate_video_people_summary
from ray_jobs.audio_diarization_pii import process_audio_diarization
from ray_jobs.audio_sensitive_info import process_audio_sensitive_info
from ray_jobs.clap_detector import detect_claps_in_media, detect_claps_in_audio_video_pair
from ray_jobs.signal_quality_check_blur_black_screen import detect_blur_and_black_segments
from ray_jobs.video_lighting_task import lighting_by_second_task

# Import comprehensive tracking system
from utils.comprehensive_tracker import initialize_global_tracker, update_tracking, get_global_tracker, check_session_completion_status, is_session_already_processed, cleanup_temp_download_directory, cleanup_session_output_directory, get_incomplete_sessions_for_retry
from utils.chunk_id_extractor import extract_chunk_id_from_path, extract_sequence_number_from_path, extract_chunk_type_from_path

# Import new ray jobs
from ray_jobs.nsfw_det_final import process_video_chunks_for_nsfw
from ray_jobs.motion_energy import compute_motion_energy
from ray_jobs.face_age_detector import process_video_chunks_for_face_detection
from ray_jobs.labelstudio_tasks import (assign_views_to_labelstudio_positions, generate_multiview_4view_labelstudio_task,
    generate_consolidated_shard_labelstudio_task, 
    import_consolidated_tasks_to_labelstudio,
    update_labelstudio_tasks_with_video_domain_and_activity)
   
from ray_jobs.video_unwarp_task import erp_unwarp_task
from ray_jobs.video_unwarp_task import insv_unwarp_task
from ray_jobs.video_lighting_task import lighting_by_second_task

logger = get_logger("SimplifiedUnifiedPipeline")
# Global shutdown flag for graceful stopping
_shutdown_requested = False

# =============================================================================
# BATCH PROCESSING FUNCTIONS (Added from backup file)
# =============================================================================
@ray.remote
def download_video_batch_task(blob_service_client: BlobServiceClient, container_name: str,
                             video_batch: List[Dict], local_download_dir: str, task_id: str) -> List[Dict]:
    """
    Ray remote task to download a batch of video-audio pairs in parallel.
    Each task runs on a separate CPU core.
    
    Args:
        blob_service_client: Azure blob service client
        container_name: Azure container name
        video_batch: List of video info dictionaries with blob names
        local_download_dir: Local directory to download files
        task_id: Unique identifier for this download task
    
    Returns:
        List of download results with paths and status
    """
    try:
        # Enhanced debugging with Ray task information
        import ray
        ray_worker_id = ray.get_runtime_context().get_worker_id()
        ray_node_id = ray.get_runtime_context().get_node_id()
        
        logger.info(f"�� RAY TASK {task_id} STARTED")
        logger.info(f"   📍 Worker ID: {ray_worker_id}")
        logger.info(f"   🖥️  Node ID: {ray_node_id}")
        logger.info(f"   �� Batch size: {len(video_batch)} videos")
        logger.info(f"   📁 Download dir: {local_download_dir}")
        logger.info(f"   �� Videos to download: {[v.get('video_name', 'unknown') for v in video_batch]}")
        
        os.makedirs(local_download_dir, exist_ok=True)
        
        download_results = []
        
        for idx, video_info in enumerate(video_batch, 1):
            try:
                video_name = video_info.get('video_name', 'unknown')
                video_blob = video_info['video_blob_name']
                audio_blob = video_info['audio_blob_name']
                
                logger.info(f"⬇️ RAY TASK {task_id} - Downloading video {idx}/{len(video_batch)}: {video_name}")
                logger.info(f"   �� Video blob: {video_blob}")
                logger.info(f"   �� Audio blob: {audio_blob}")
                
                # Check if session is already completed before downloading
                session_status = check_session_completion_status(video_name, "./outmain2withdn")
                if session_status.get("completion_status") == "completed":
                    logger.info(f"⏭️  RAY TASK {task_id} - Session {video_name} already completed at {session_status.get('completion_time')} - skipping download")
                    download_results.append({
                        'video_name': video_name,
                        'video_path': None,
                        'audio_path': None,
                        'video_info': video_info,
                        'success': False,
                        'error': f"Session already completed at {session_status.get('completion_time')}",
                        'task_id': task_id,
                        'skipped': True,
                        'skip_reason': 'already_completed'
                    })
                    continue
                elif session_status.get("completion_status") == "pending" and session_status.get("exists"):
                    logger.info(f"🔄 RAY TASK {task_id} - Session {video_name} was previously started but not completed - resuming download")
                else:
                    logger.info(f"🆕 RAY TASK {task_id} - Starting new session: {video_name}")
                
                # Fix potential double extension issue
                video_local_filename = os.path.basename(video_blob)
                if video_local_filename.endswith('.insv.insv'):
                    video_local_filename = video_local_filename[:-5]  # Remove extra .insv
                
                audio_local_filename = os.path.basename(audio_blob)
                if audio_local_filename.endswith('.wav.wav'):
                    audio_local_filename = audio_local_filename[:-4]  # Remove extra .wav
                
                video_local_path = os.path.join(local_download_dir, video_local_filename)
                audio_local_path = os.path.join(local_download_dir, audio_local_filename)
                
                # Download video
                start_time = time.time()
                container_client = blob_service_client.get_container_client(container_name)
                
                logger.info(f"   ⬇️ RAY TASK {task_id} - Downloading video to: {video_local_path}")
                with open(video_local_path, "wb") as f:
                    blob_client = container_client.get_blob_client(video_blob)
                    download_stream = blob_client.download_blob()
                    f.write(download_stream.readall())
                
                video_download_time = time.time() - start_time
                video_size = os.path.getsize(video_local_path) / (1024*1024)  # MB
                
                # Download audio
                start_time = time.time()
                logger.info(f"   ⬇️ RAY TASK {task_id} - Downloading audio to: {audio_local_path}")
                with open(audio_local_path, "wb") as f:
                    blob_client = container_client.get_blob_client(audio_blob)
                    download_stream = blob_client.download_blob()
                    f.write(download_stream.readall())
                
                audio_download_time = time.time() - start_time
                audio_size = os.path.getsize(audio_local_path) / (1024*1024)  # MB
                
                download_results.append({
                    'video_name': video_name,
                    'video_path': video_local_path,
                    'audio_path': audio_local_path,
                    'video_info': video_info,
                    'success': True,
                    'error': None,
                    'task_id': task_id,
                    'download_stats': {
                        'video_size_mb': round(video_size, 2),
                        'audio_size_mb': round(audio_size, 2),
                        'video_download_time': round(video_download_time, 2),
                        'audio_download_time': round(audio_download_time, 2)
                    }
                })
                
                logger.info(f"✅ RAY TASK {task_id} - Successfully downloaded {video_name}")
                logger.info(f"   �� Video: {video_size:.1f}MB in {video_download_time:.1f}s")
                logger.info(f"   �� Audio: {audio_size:.1f}MB in {audio_download_time:.1f}s")
                
            except Exception as e:
                video_name = video_info.get('video_name', 'unknown')
                logger.error(f"❌ RAY TASK {task_id} - Error downloading {video_name}: {e}")
                download_results.append({
                    'video_name': video_name,
                    'video_path': None,
                    'audio_path': None,
                    'video_info': video_info,
                    'success': False,
                    'error': str(e),
                    'task_id': task_id
                })
        
        successful_downloads = len([r for r in download_results if r['success']])
        total_download_time = sum([r.get('download_stats', {}).get('video_download_time', 0) + 
                                  r.get('download_stats', {}).get('audio_download_time', 0) 
                                  for r in download_results if r['success']])
        
        logger.info(f"�� RAY TASK {task_id} COMPLETED: {successful_downloads}/{len(video_batch)} successful")
        logger.info(f"   ⏱️ Total download time: {total_download_time:.1f}s")
        logger.info(f"   🖥️ Worker: {ray_worker_id}")
        
        return download_results
        
    except Exception as e:
        logger.error(f"❌ RAY TASK {task_id} FAILED: {e}")
        return []

# =============================================================================
# END OF BATCH PROCESSING FUNCTIONS
# =============================================================================
def read_clap_detection_from_json(shard_output_dir: str, shard_type: str) -> bool:
    """
    Read clap detection results from clap_output JSON files.
    
    Args:
        shard_output_dir: Directory containing the shard (e.g., shard_1, shard_2, etc.)
        shard_type: "first" or "last" for logging purposes
        
    Returns:
        bool: True if clap was detected in video_results, False otherwise
    """
    try:
        logger.info(f"🔍 Reading clap detection from JSON files in {shard_output_dir}")
        
        # Look for clap_output directories in any view subdirectory
        import glob
        clap_output_patterns = [
            f"{shard_output_dir}/*/clap_output/*.json",
            f"{shard_output_dir}/*/clap_output/*_clap_detection.json",
            f"{shard_output_dir}/*/clap_output/*_combined_clap_detection.json"
        ]
        
        clap_json_files = []
        for pattern in clap_output_patterns:
            clap_json_files.extend(glob.glob(pattern))
        
        if not clap_json_files:
            logger.warning(f"⚠️ No clap detection JSON files found in {shard_output_dir}")
            return False
        
        # Use the first JSON file found
        clap_json_file = clap_json_files[0]
        logger.info(f"📄 Reading clap detection from: {clap_json_file}")
        
        # Read and parse the JSON file
        with open(clap_json_file, 'r') as f:
            clap_data = json.load(f)
        
        # Extract video_results success field
        video_results = clap_data.get('video_results', {})
        success = video_results.get('success', False)
        
        logger.info(f"🎬 {shard_type.upper()} SHARD clap detection result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        logger.info(f"   Video results: {video_results}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error reading clap detection from JSON: {e}")
        return False


@ray.remote
def integrated_blob_polling_and_pipeline_task(
    azure_config_path: str = "blobfuse2_config.yaml",
    pipeline_config_path: str = "config/pipeline_config.yaml",
    blob_prefix: str = None,  # Will use config default if None
    max_videos: Optional[int] = None,
    force_refresh: bool = False,
    process_dual_views: bool = None,
    process_unwarped_views: bool = False,
    batch_size: int = 3  # New parameter for batch processing
):
    """
    Integrated Ray task that handles both blob polling and pipeline processing with batch processing.
    
    This task performs two sequential loops:
    1. BLOB POLLING LOOP: Find videos and download corresponding audio files in batches
    2. MAIN PIPELINE LOOP: Process video+audio pairs in batches sequentially
    
    Args:
        config_path: Path to Azure configuration file
        blob_prefix: Blob prefix to search for videos
        max_videos: Maximum number of videos to process
        force_refresh: Force refresh video list
        process_dual_views: Enable dual view processing
        process_unwarped_views: Enable unwarped view processing
        batch_size: Number of videos to process in each batch (default: 3)
        
    Returns:
        dict: Processing results summary
    """
    logger.info("🚀 Starting Integrated Blob Polling and Pipeline Task with Batch Processing")
    logger.info("=" * 80)
    logger.info(f"📦 Batch Size: {batch_size} videos per batch")
    
    # Initialize configuration and Azure client
    try:
        # Load pipeline configuration
        pipeline_config = load_pipeline_config(pipeline_config_path)
        
        # Setup current directory first
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load Azure configuration
        logger.info(f"🔍 Loading Azure config from: {azure_config_path}")
        logger.info(f"🔍 File exists: {os.path.exists(azure_config_path)}")
        
        # Resolve absolute path if needed
        if not os.path.isabs(azure_config_path):
            azure_config_path = os.path.join(current_dir, azure_config_path)
            logger.info(f"🔍 Resolved Azure config path: {azure_config_path}")
            logger.info(f"🔍 Resolved file exists: {os.path.exists(azure_config_path)}")
        
        azure_config = load_azure_config(azure_config_path)
        logger.info(f"🔍 Azure config loaded: {list(azure_config.keys())}")
        blob_service_client = create_azure_blob_client(azure_config)
        
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
        
        # Setup directories from pipeline config
        local_download_dir = pipeline_config['local_storage']['temp_download_dir']
        output_base_dir = os.path.join(current_dir, pipeline_config['local_storage']['output_base_dir'].lstrip('./'))
        video_list_file = os.path.join(current_dir, pipeline_config['local_storage']['video_list_file'].lstrip('./'))
        
        # Get checklist file path
        checklist_path = pipeline_config['local_storage'].get('checklist_file', './video_checklist.json')
        checklist_path = os.path.join(current_dir, checklist_path.lstrip('./'))
        
        # Use blob_prefix from config if not provided
        if blob_prefix is None:
            blob_prefix = pipeline_config['azure_storage']['input_blob_prefix']
        
        # Get batch size from config if available
        batch_size = pipeline_config.get('processing', {}).get('max_videos_per_batch', batch_size)
        
        os.makedirs(local_download_dir, exist_ok=True)
        os.makedirs(output_base_dir, exist_ok=True)
        
        logger.info(f"✅ Configuration loaded successfully")
        logger.info(f"   Container: {container_name}")
        logger.info(f"   Blob prefix: {blob_prefix}")
        logger.info(f"   Batch Size: {batch_size}")
        logger.info(f"   Download dir: {local_download_dir}")
        logger.info(f"   Output dir: {output_base_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize configuration: {e}")
        return {"success": False, "error": f"Configuration error: {e}"}
    
    # =============================================================================
    # CONTINUOUS BATCH PROCESSING WITH POLLING
    # =============================================================================
    logger.info("\n" + "="*80)
    logger.info("🔄 CONTINUOUS BATCH PROCESSING WITH POLLING")
    logger.info("="*80)
    
    # Get polling interval from config
    polling_interval_minutes = pipeline_config.get('polling', {}).get('interval_minutes', 5)
    logger.info(f"⏰ Polling interval: {polling_interval_minutes} minutes")
    
    # Load or create persistent checklist
    checklist = load_video_checklist(checklist_path)
    logger.info(f"📋 Loaded checklist: {checklist['total_videos']} total, {checklist['completed_videos']} completed, {checklist['failed_videos']} failed")
    
    # =============================================================================
    # DOMAIN DETECTION WILL RUN AFTER EACH VIDEO PROCESSING COMPLETES
    # =============================================================================
    logger.info("\n" + "="*80)
    logger.info("🎯 DOMAIN DETECTION ENABLED - Will run after each video processing completes")
    logger.info("="*80)
    
    # =============================================================================
    # CHECK FOR FAILED VIDEOS AND RESET TO PENDING
    # =============================================================================
    failed_videos = [name for name, data in checklist['videos'].items() if data['status'] == 'failed']
    if failed_videos:
        logger.info(f"🔄 Found {len(failed_videos)} failed videos, resetting to pending: {failed_videos}")
        for video_name in failed_videos:
            checklist['videos'][video_name]['status'] = 'pending'
            checklist['videos'][video_name]['last_updated'] = datetime.now().isoformat()
        checklist['failed_videos'] = 0
        checklist['last_updated'] = datetime.now().isoformat()
        save_video_checklist(checklist, checklist_path)
        logger.info("✅ Failed videos reset to pending state")
    else:
        logger.info("✅ No failed videos found")
    
    # =============================================================================
    # CHECK FOR INCOMPLETE SESSIONS AND ADD TO RETRY QUEUE
    # =============================================================================
    incomplete_sessions = get_incomplete_sessions_for_retry(output_base_dir)
    if incomplete_sessions:
        logger.info(f"🔄 Found {len(incomplete_sessions)} incomplete sessions for retry:")
        for session in incomplete_sessions:
            session_id = session['session_id']
            status = session['status']
            last_updated = session['last_updated']
            logger.info(f"   📋 {session_id}: {status} (last updated: {last_updated})")
            
            # Add to checklist if not already present
            if session_id not in checklist['videos']:
                checklist['videos'][session_id] = {
                    'status': 'pending',
                    'last_updated': datetime.now().isoformat(),
                    'retry_reason': f'Incomplete session - {status}',
                    'original_status': status
                }
                checklist['total_videos'] += 1
                logger.info(f"   ➕ Added {session_id} to checklist for retry")
            else:
                # Update existing entry to pending
                checklist['videos'][session_id]['status'] = 'pending'
                checklist['videos'][session_id]['last_updated'] = datetime.now().isoformat()
                checklist['videos'][session_id]['retry_reason'] = f'Incomplete session - {status}'
                logger.info(f"   🔄 Updated {session_id} status to pending for retry")
        
        checklist['last_updated'] = datetime.now().isoformat()
        save_video_checklist(checklist, checklist_path)
        logger.info("✅ Incomplete sessions added to retry queue")
    else:
        logger.info("✅ No incomplete sessions found")
    
    processing_results = []
    successful_count = 0
    failed_count = 0
    total_processed = 0
    
    try:
        while not _shutdown_requested:  # Continuous processing loop with shutdown check
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Polling cycle started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Poll Azure for all videos and update checklist
            all_videos = poll_azure_videos_with_checklist(
                blob_service_client, container_name, blob_prefix, 
                checklist, pipeline_config
            )
            
            if not all_videos:
                logger.info("✅ No videos found in Azure Blob Storage")
                logger.info(f"⏳ Waiting {polling_interval_minutes} minutes before next poll...")
                # Sleep in smaller intervals to check for shutdown
                for i in range(polling_interval_minutes * 60):
                    if _shutdown_requested:
                        break
                    time.sleep(1)
                continue
            
            # Get all videos from checklist and filter for unprocessed ones
            all_video_names = list(checklist["videos"].keys())
            pending_videos = []
            
            for video_name in all_video_names:
                video_info = checklist["videos"][video_name]
                if video_info.get("status") not in ["completed", "processing"]:
                    # Get the video info from the discovered videos
                    for v in all_videos:
                        if v.get("video_name") == video_name:
                            pending_videos.append(v)
                            break
            
            if max_videos:
                pending_videos = pending_videos[:max_videos]
            
            logger.info(f"📋 Found {len(pending_videos)} unprocessed videos out of {len(all_video_names)} total videos")
            
            if not pending_videos:
                logger.info("✅ All videos have been processed")
                logger.info(f"⏳ Waiting {polling_interval_minutes} minutes before next poll...")
                # Sleep in smaller intervals to check for shutdown
                for i in range(polling_interval_minutes * 60):
                    if _shutdown_requested:
                        break
                    time.sleep(1)
                continue
        
            # Process videos in batches
            total_batches = (len(pending_videos) + batch_size - 1) // batch_size  # Ceiling division
        
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(pending_videos))
                current_batch = pending_videos[start_idx:end_idx]
            
                logger.info(f"\n📦 Processing BATCH {batch_num + 1}/{total_batches}")
                logger.info(f"📊 Batch contains {len(current_batch)} videos")
                logger.info("-" * 60)
            
                # Download and process current batch
                batch_video_pairs = []
            
                # Get parallel download batch size from config
                download_batch_size = pipeline_config.get('processing', {}).get('max_videos_per_batch', 3)
                logger.info(f"📦 Using max_videos_per_batch: {download_batch_size} for parallel download in batch {batch_num + 1}")
                logger.info(f"   🎯 This will create {len(current_batch)} parallel Ray tasks for downloading (one per video)")
                logger.info(f"   💡 Expected behavior: All {len(current_batch)} videos download simultaneously, then process sequentially")
                
                # Create one Ray task per video for true parallel downloading
                download_tasks = []
                task_info = []
                logger.info(f"🚀 LAUNCHING {len(current_batch)} PARALLEL RAY TASKS for batch {batch_num + 1}")
                logger.info("🎯 PARALLEL DOWNLOAD PHASE STARTING - Each video gets its own Ray task!")
                logger.info("=" * 80)
                
                for video_idx, video_info in enumerate(current_batch):
                    # Create separate download directory for each video
                    video_download_dir = os.path.join(local_download_dir, f"batch_{batch_num}_video_{video_idx}")
                    task_id = f"BATCH_{batch_num + 1}_VIDEO_{video_idx + 1}"
                    
                    logger.info(f"🎯 LAUNCHING RAY TASK {task_id}")
                    logger.info(f"   📦 Video: {video_info.get('video_name', 'unknown')}")
                    logger.info(f"   📁 Download directory: {video_download_dir}")
                    logger.info(f"   🖥️ Will run on separate CPU core")
                    
                    # Create a single-video batch for this task
                    single_video_batch = [video_info]
                    
                    task_ref = download_video_batch_task.remote(
                        blob_service_client, container_name, single_video_batch, video_download_dir, task_id
                    )
                    download_tasks.append(task_ref)
                    task_info.append({
                        'task_id': task_id,
                        'video_idx': video_idx,
                        'video_name': video_info.get('video_name', 'unknown'),
                        'task_ref': task_ref
                    })
                
                logger.info("=" * 80)
                logger.info(f"⏳ WAITING for {len(download_tasks)} parallel Ray tasks to complete...")
                logger.info("🚀 PARALLEL DOWNLOAD PHASE STARTED - All tasks running simultaneously!")
                
                # Display task status while waiting
                for task in task_info:
                    logger.info(f"   📍 {task['task_id']}: {task['video_name']}")
                
                # Wait for all downloads to complete
                try:
                    logger.info(f"⏳ WAITING for {len(download_tasks)} Ray tasks to finish downloading...")
                    logger.info("💡 This is where the parallel downloading happens - all videos download simultaneously!")
                    download_results_list = ray.get(download_tasks)
                    
                    logger.info("=" * 80)
                    logger.info(f"✅ ALL PARALLEL DOWNLOADS COMPLETED for batch {batch_num + 1}")
                    logger.info("🎯 PARALLEL DOWNLOAD PHASE COMPLETE - Now switching to sequential processing!")
                    logger.info("=" * 80)
                    
                    # Display detailed results from each task
                    total_successful = 0
                    total_failed = 0
                    for i, (download_results, task) in enumerate(zip(download_results_list, task_info)):
                        successful = len([r for r in download_results if r['success']])
                        failed = len([r for r in download_results if not r['success']])
                        total_successful += successful
                        total_failed += failed
                        
                        logger.info(f"📊 {task['task_id']} RESULTS: {successful}/{len(download_results)} successful")
                        for result in download_results:
                            if result['success']:
                                stats = result.get('download_stats', {})
                                logger.info(f"   ✅ {result['video_name']}: {stats.get('video_size_mb', '?')}MB video + {stats.get('audio_size_mb', '?')}MB audio")
                            else:
                                logger.error(f"   ❌ {result['video_name']}: {result['error']}")
                    
                    logger.info(f"📊 BATCH {batch_num + 1} DOWNLOAD SUMMARY: {total_successful} successful, {total_failed} failed")
                    
                except Exception as e:
                    logger.error(f"❌ Parallel download failed for batch {batch_num + 1}: {e}")
                    continue
                
                # Process download results and add to batch_video_pairs
                download_pair_index = 1
                logger.info(f"\n📋 PROCESSING DOWNLOAD RESULTS for batch {batch_num + 1}")
                
                # Store cleanup directories for later
                batch_cleanup_dirs = []
                
                for download_results in download_results_list:
                    for download_result in download_results:
                        if download_result['success']:
                            # Verify files exist before adding to processing queue
                            video_path = download_result['video_path']
                            audio_path = download_result['audio_path']
                            
                            if os.path.exists(video_path) and os.path.exists(audio_path):
                                batch_video_pairs.append({
                                    'video_info': download_result['video_info'],
                                    'video_path': video_path,
                                    'audio_path': audio_path,
                                    'download_index': download_pair_index,
                                    'download_stats': download_result.get('download_stats', {}),
                                    'source_task_id': download_result.get('task_id', 'unknown')
                                })
                                logger.info(f"✅ Downloaded pair {download_pair_index}: {download_result['video_name']} ready for processing (from {download_result.get('task_id', 'unknown')})")
                                logger.info(f"   📁 Video: {video_path}")
                                logger.info(f"   📁 Audio: {audio_path}")
                                download_pair_index += 1
                            else:
                                logger.error(f"❌ Downloaded files missing for {download_result['video_name']}: video={os.path.exists(video_path)}, audio={os.path.exists(audio_path)}")
                        else:
                            logger.error(f"❌ Failed to download pair: {download_result['video_name']} - {download_result['error']} (from {download_result.get('task_id', 'unknown')})")
                
                # Store cleanup directories for later
                for video_idx in range(len(current_batch)):
                    video_download_dir = os.path.join(local_download_dir, f"batch_{batch_num}_video_{video_idx}")
                    if os.path.exists(video_download_dir):
                        batch_cleanup_dirs.append(video_download_dir)
                
                logger.info(f"\n📊 BATCH {batch_num + 1}/{total_batches} DOWNLOAD COMPLETE: Downloaded {len(batch_video_pairs)} video+audio pairs")
                logger.info(f"🎯 All downloads completed successfully! Now starting sequential processing...")
                
                # =============================================================================
                # SEQUENTIAL PROCESSING PHASE - Process all downloaded videos one by one
                # =============================================================================
                logger.info(f"\n🎬 STARTING SEQUENTIAL PROCESSING for batch {batch_num + 1}")
                logger.info(f"📦 Processing {len(batch_video_pairs)} videos sequentially...")
                logger.info("=" * 60)
                
                # Process current batch sequentially (after all downloads are complete)
                for pair_data in batch_video_pairs:
                    video_info = pair_data['video_info']
                    video_path = pair_data['video_path']
                    audio_path = pair_data['audio_path']
                    pair_index = pair_data['download_index']
                    
                    video_blob = video_info['video_blob_name']
                    video_name = os.path.splitext(os.path.basename(video_blob))[0]
                    
                    logger.info(f"\n🎬 Processing pair {pair_index}/{len(batch_video_pairs)}: {video_name}")
                    logger.info("-" * 60)
                    
                    try:
                        # Update checklist status to processing
                        update_video_status(checklist, video_name, "processing")
                        
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
                        timestamped_video_name = os.path.basename(video_output_dir)
                        
                        pipeline_results = pipeline_main(
                            input_video_path=video_path,
                            input_audio_path=audio_path,
                            output_dir=video_output_dir,
                            process_dual_views=False,  # Let pipeline_main auto-detect for INSV files
                            process_unwarped_views=True,
                            azure_blob_client=blob_service_client,
                            azure_container=container_name,
                            azure_output_prefix=f"{output_prefix}/{timestamped_video_name}",
                            azure_account_name=account_name,
                            azure_account_key=account_key,
                            pipeline_config=pipeline_config
                        )
                        
                        # Check if pipeline actually succeeded
                        if pipeline_results and pipeline_results.get("successful_shard_tasks", 0) > 0:
                            # Update video info with success
                            video_info.update({
                                "processed": True,
                                "processing_date": datetime.now().isoformat(),
                                "output_dir": video_output_dir,
                                "status": "completed",
                                "results_summary": pipeline_results
                            })
                        
                            # Update checklist status
                            update_video_status(checklist, video_name, "completed", 
                                               output_dir=video_output_dir,
                                               results_summary=pipeline_results)
                        else:
                            # Pipeline failed - mark as failed
                            error_msg = "Pipeline failed - no successful shard tasks"
                            video_info.update({
                                "processed": False,
                                "processing_date": datetime.now().isoformat(),
                                "status": "failed",
                                "error": error_msg
                            })
                            update_video_status(checklist, video_name, "failed", error=error_msg)
                            raise Exception(error_msg)
                        
                        processing_results.append({
                            "video_name": video_name,
                            "status": "success",
                            "output_dir": video_output_dir,
                            "results": pipeline_results
                        })
                        
                        successful_count += 1
                        total_processed += 1
                        logger.info(f"✅ Pipeline completed successfully for {video_name}")
                        logger.info(f"📁 Results saved to: {video_output_dir}")
                        
                        
                    except Exception as e:
                        error_msg = str(e)
                        
                        # Check for CUDA out of memory errors
                        if "CUDA_ERROR_OUT_OF_MEMORY" in error_msg or "out of memory" in error_msg.lower():
                            error_msg = f"CUDA out of memory error: {error_msg}"
                            logger.error(f"❌ CUDA memory error for {video_name}: {error_msg}")
                        else:
                            logger.error(f"❌ Pipeline failed for {video_name}: {error_msg}")
                        
                        # Update video info with failure
                        video_info.update({
                            "processed": False,
                            "processing_date": datetime.now().isoformat(),
                            "status": "failed",
                            "error": error_msg
                        })
                        
                        # Update checklist status
                        update_video_status(checklist, video_name, "failed", error=error_msg)
                        
                        processing_results.append({
                            "video_name": video_name,
                            "status": "failed",
                            "error": str(e)
                        })
                        
                        failed_count += 1
                        total_processed += 1
                    
                    finally:
                        # Clean up downloaded files for this pair
                        cleanup_downloaded_files([video_path, audio_path])
                        
                        # Save checklist after each video
                        save_video_checklist(checklist_path, checklist)
                        
                        logger.info(f"📊 Progress: {successful_count} successful, {failed_count} failed")
                
                # Cleanup batch download directories after processing is complete
                logger.info(f"🗑️ Cleaning up {len(batch_cleanup_dirs)} batch download directories")
                for chunk_download_dir in batch_cleanup_dirs:
                    try:
                        shutil.rmtree(chunk_download_dir)
                        logger.info(f"🗑️ Cleaned up batch download directory: {chunk_download_dir}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to cleanup batch directory {chunk_download_dir}: {e}")
                
                # Batch completion summary
                logger.info(f"\n📦 BATCH {batch_num + 1}/{total_batches} COMPLETE")
                logger.info(f"📊 Batch Results: {len([r for r in processing_results[-len(batch_video_pairs):] if r['status'] == 'success'])} successful, {len([r for r in processing_results[-len(batch_video_pairs):] if r['status'] == 'failed'])} failed")
                
                # Wait between batches (except after the last batch)
                if batch_num < total_batches - 1:
                    logger.info("⏳ Waiting 10 seconds before starting next batch...")
                    time.sleep(10)
        
        # After processing all batches, wait for next polling cycle
        logger.info(f"\n📊 CYCLE COMPLETE: Processed {total_processed} total videos")
        logger.info(f"⏳ Waiting {polling_interval_minutes} minutes before next polling cycle...")
        time.sleep(polling_interval_minutes * 60)
        
    except Exception as e:
        logger.error(f"❌ CONTINUOUS PROCESSING FAILED: {e}")
        return {"success": False, "error": f"Continuous processing failed: {e}"}
    
    # =============================================================================
    # FINAL RESULTS
    # =============================================================================
    logger.info("\n" + "="*80)
    logger.info("🏁 INTEGRATED TASK COMPLETE")
    logger.info("="*80)
    
    final_results = {
        "success": True,
        "task_type": "integrated_blob_polling_and_pipeline_with_batching",
        "batch_size": batch_size,
        "total_batches": total_batches,
        "videos_discovered": len(pending_videos),
        "pairs_downloaded": len(batch_video_pairs),
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
    logger.info(f"   Total Batches: {final_results['total_batches']}")
    logger.info(f"   Videos Processed: {final_results['videos_processed']}")
    logger.info(f"   Successful: {final_results['successful_count']}")
    logger.info(f"   Failed: {final_results['failed_count']}")
    logger.info(f"   Completion Rate: {final_results['completion_rate']}")
    
    return final_results

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
        elif task_type == "lighting":
            if isinstance(task_result, tuple) and len(task_result) == 2:
                try:
                    per_second_path, events_path = task_result
                    import json
                    with open(events_path, 'r') as f:
                        lighting_data = json.load(f)
                    
                    events = lighting_data.get('events', [])
                    for event in events:
                        # Only flag Dark and Bright lighting as noteworthy
                        if event.get('label') in ['Dark', 'Bright']:
                            segments.append({
                                'start_time': shard_offset_sec + event.get('start', 0),
                                'end_time': shard_offset_sec + event.get('end', 0),
                                'task_type': 'lighting',
                                'confidence': min(event.get('meanY_avg', 0) / 255.0, 1.0),
                                'flag_type': f"lighting_{event.get('label', 'unknown').lower()}",
                                'priority': 'medium',
                                'description': f"Lighting: {event.get('label')} (avg: {event.get('meanY_avg', 0):.1f})",
                                'shard_index': shard_index + 1,
                                'metadata': {
                                    'lighting_label': event.get('label'),
                                    'duration': event.get('duration', 0),
                                    'brightness_avg': event.get('meanY_avg', 0),
                                    'brightness_med': event.get('meanY_med', 0),
                                    'brightness_range': event.get('brightness_range_avg', 0)
                                }
                            })
                except Exception as e:
                    logger.error(f"Failed to extract lighting segments: {e}")
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

        elif task_type == "sensitive":
            # Handle sensitive information analysis results with new structure
            if isinstance(task_result, list) and len(task_result) > 0:
                for sensitive_result in task_result:
                    # Check if sensitive content was detected using the new structure
                    if sensitive_result.get('summary', {}).get('has_sensitive_content', False):
                        # Use sensitive_detections array if available, otherwise create from summary
                        sensitive_detections = sensitive_result.get('sensitive_detections', [])
                        if sensitive_detections:
                            # Use individual detections
                            for detection in sensitive_detections:
                                segments.append({
                                    "start_time": maybe_offset(detection.get('start_time', 0)),
                                    "end_time": maybe_offset(detection.get('end_time', 60)),
                                    "task_type": "sensitive_analysis",
                                    "confidence": detection.get('confidence', 0.7),
                                    "flag_type": "sensitive_content",
                                    "priority": detection.get('priority', 'high'),
                                    "description": f"Sensitive content detected: {detection.get('topic', 'unknown')}",
                                    "shard_index": shard_index + 1,
                                    "sensitive_topic": detection.get('topic', ''),
                                    "analysis": detection.get('analysis', '')
                                })
                        else:
                            # Fallback: create segment from summary data
                            summary = sensitive_result.get('summary', {})
                            segments.append({
                                "start_time": shard_offset_sec,
                                "end_time": shard_offset_sec + 60,  # Assuming 60-second shards
                                "task_type": "sensitive_analysis",
                                "confidence": summary.get('confidence', 0.7),
                                "flag_type": "sensitive_content",
                                "priority": "high",
                                "description": f"Sensitive content detected: {', '.join(summary.get('sensitive_topics', []))}",
                                "shard_index": shard_index + 1,
                                "sensitive_topics": summary.get('sensitive_topics', []),
                                "analysis": sensitive_result.get('sensitive_analysis', {}).get('analysis', '')
                            })
        elif task_type == "signal_quality":
            # Handle blur segments
            for blur_segment in task_result.get('blur_segments', []):
                segments.append({
                    "start_time": maybe_offset(blur_segment.get('start_time', 0)),
                    "end_time": maybe_offset(blur_segment.get('end_time', 0)),
                    "task_type": "signal_quality",
                    "confidence": 0.8,
                    "flag_type": "signal_blur",
                    "priority": "medium",
                    "description": f"Blur detected from {blur_segment.get('start_time', 0):.1f}s to {blur_segment.get('end_time', 0):.1f}s",
                    "shard_index": shard_index + 1
                })
            
            # Handle black screen segments
            for black_segment in task_result.get('black_segments', []):
                segments.append({
                    "start_time": maybe_offset(black_segment.get('start_time', 0)),
                    "end_time": maybe_offset(black_segment.get('end_time', 0)),
                    "task_type": "signal_quality",
                    "confidence": 0.8,
                    "flag_type": "signal_black_screen",
                    "priority": "medium", 
                    "description": f"Black screen detected from {black_segment.get('start_time', 0):.1f}s to {black_segment.get('end_time', 0):.1f}s",
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

    SHARD_SEC = 180
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

def pipeline_main_multichunks_sequential_sessions(session_paths: list, output_base_dir: str, azure_config: dict, pipeline_config: dict):
    """
    Process multiple sessions sequentially - complete one session fully before starting the next.
    
    Args:
        session_paths: List of session paths like ['/one-data-platform/10-ca98a947-4f4b-4903-be80-f3d0fdc553ef-working-in-office', ...]
        output_base_dir: Base output directory for all sessions
        azure_config: Azure configuration
        pipeline_config: Pipeline configuration
    """
    logger.info(f"🚀 Starting sequential multi-session processing for {len(session_paths)} sessions")
    
    # Initialize tracking
    all_sessions_summary = {
        "total_sessions": len(session_paths),
        "completed_sessions": 0,
        "failed_sessions": 0,
        "skipped_sessions": 0,
        "session_results": {},
        "start_time": datetime.now().isoformat(),
        "end_time": None
    }
    
    # Process each session sequentially
    for session_idx, session_info in enumerate(session_paths, 1):
        session_id = session_info["session_id"]
        session_path = session_info["session_path"]
        is_walkthrough = session_info["is_walkthrough"]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎬 PROCESSING SESSION {session_idx}/{len(session_paths)}: {session_id}")
        logger.info(f"📁 Session Path: {session_path}")
        logger.info(f"🔄 Walkthrough Mode: {is_walkthrough}")
        logger.info(f"📋 Domain: {session_info.get('domain', 'Unknown')}")
        logger.info(f"📋 Activity: {session_info.get('activity', 'Unknown')}")
        logger.info(f"{'='*80}")
        
        try:
            # Check if session is already completed
            session_output_dir = os.path.join(output_base_dir, session_id)
            session_status = check_session_completion_status(session_id, output_base_dir)
            
            if session_status.get("completion_status") == "completed":
                logger.info(f"⏭️  Session {session_id} already completed - skipping")
                all_sessions_summary["skipped_sessions"] += 1
                all_sessions_summary["session_results"][session_id] = {
                    "status": "skipped",
                    "reason": "already_completed",
                    "completion_time": session_status.get("completion_time")
                }
                continue
            
            # Download and process this single session
            session_result = process_single_session(
                session_path=session_path,
                session_id=session_id,
                output_base_dir=output_base_dir,
                azure_config=azure_config,
                pipeline_config=pipeline_config,
                is_walkthrough=is_walkthrough
            )
            
            # Track results
            if session_result.get("success"):
                all_sessions_summary["completed_sessions"] += 1
                all_sessions_summary["session_results"][session_id] = {
                    "status": "completed",
                    "processing_time": session_result.get("processing_time"),
                    "shards_processed": session_result.get("shards_processed", 0)
                }
                logger.info(f"✅ Session {session_id} completed successfully")
            else:
                all_sessions_summary["failed_sessions"] += 1
                all_sessions_summary["session_results"][session_id] = {
                    "status": "failed",
                    "error": session_result.get("error", "Unknown error")
                }
                logger.error(f"❌ Session {session_id} failed: {session_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            all_sessions_summary["failed_sessions"] += 1
            all_sessions_summary["session_results"][session_id] = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"❌ Session {session_id} failed with exception: {e}")
        
        # Log progress
        logger.info(f"📊 Progress: {session_idx}/{len(session_paths)} sessions processed")
        logger.info(f"   ✅ Completed: {all_sessions_summary['completed_sessions']}")
        logger.info(f"   ❌ Failed: {all_sessions_summary['failed_sessions']}")
        logger.info(f"   ⏭️  Skipped: {all_sessions_summary['skipped_sessions']}")
    
    # Final summary
    all_sessions_summary["end_time"] = datetime.now().isoformat()
    logger.info(f"\n{'='*80}")
    logger.info(f"🏁 ALL SESSIONS PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Final Summary:")
    logger.info(f"   Total Sessions: {all_sessions_summary['total_sessions']}")
    logger.info(f"   ✅ Completed: {all_sessions_summary['completed_sessions']}")
    logger.info(f"   ❌ Failed: {all_sessions_summary['failed_sessions']}")
    logger.info(f"   ⏭️  Skipped: {all_sessions_summary['skipped_sessions']}")
    
    return all_sessions_summary

def load_processed_sessions_tracking(tracking_file: str) -> set:
    """
    Load the set of processed sessions from a JSON file.
    
    Args:
        tracking_file: Path to the tracking JSON file
        
    Returns:
        Set of processed session IDs
    """
    try:
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r') as f:
                data = json.load(f)
                processed_sessions = set(data.get('processed_sessions', []))
                logger.info(f"📋 Loaded {len(processed_sessions)} previously processed sessions from: {tracking_file}")
                return processed_sessions
        else:
            logger.info(f"📋 No existing tracking file found at: {tracking_file}")
            return set()
    except Exception as e:
        logger.warning(f"⚠️ Could not load tracking file {tracking_file}: {e}")
        return set()

def save_processed_sessions_tracking(tracking_file: str, processed_sessions: set):
    """
    Save the set of processed sessions to a JSON file.
    
    Args:
        tracking_file: Path to the tracking JSON file
        processed_sessions: Set of processed session IDs
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
        
        # Save tracking data
        tracking_data = {
            "processed_sessions": list(processed_sessions),
            "last_updated": datetime.now().isoformat(),
            "total_processed": len(processed_sessions)
        }
        
        with open(tracking_file, 'w') as f:
            json.dump(tracking_data, f, indent=2)
        
        logger.info(f"💾 Saved tracking data for {len(processed_sessions)} sessions to: {tracking_file}")
    except Exception as e:
        logger.warning(f"⚠️ Could not save tracking file {tracking_file}: {e}")

def pipeline_main_with_blob_polling(base_prefix: str, output_base_dir: str, azure_config: dict, pipeline_config: dict, polling_interval_minutes: int = 5):
    """
    Main pipeline function with blob polling - continuously polls for new sessions.
    
    Args:
        base_prefix: Base prefix to search for sessions (e.g., "test_mulitsession/")
        output_base_dir: Base output directory for all sessions
        azure_config: Azure configuration
        pipeline_config: Pipeline configuration
        polling_interval_minutes: How long to wait between polling attempts (default: 5 minutes)
    """
    logger.info("🔄 Starting blob polling mode - Base: %s, Interval: %d minutes, Output: %s", base_prefix, polling_interval_minutes, output_base_dir)
    
    # Initialize blob client
    blob_client = create_azure_blob_client(azure_config)
    container_name = azure_config.get('container')
    
    # Track all processed sessions to avoid reprocessing
    # Create persistent tracking file
    tracking_file = os.path.join(output_base_dir, "processed_sessions_tracking.json")
    all_processed_sessions = load_processed_sessions_tracking(tracking_file)
    polling_cycle = 0
    
    while True:
        polling_cycle += 1
        logger.info("🔍 Polling cycle #%d - Searching sessions under: %s", polling_cycle, base_prefix)
        
        try:
            # Discover all sessions in the directory
            all_sessions = discover_multiple_sessions(blob_client, container_name, base_prefix)
            
            # Filter out already processed sessions
            new_sessions = []
            for session_info in all_sessions:
                session_id = session_info["session_id"]
                if session_id not in all_processed_sessions:
                    new_sessions.append(session_info)
                    all_processed_sessions.add(session_id)
                else:
                    logger.info(f"⏭️  Session {session_id} already processed - skipping")
            
            if new_sessions:
                logger.info("=" * 100)
                logger.info("🎉 Found %d new sessions:", len(new_sessions))
                for i, session in enumerate(new_sessions, 1):
                    walkthrough_status = "🔄 WALKTHROUGH" if session["is_walkthrough"] else "📹 NORMAL"
                    logger.info("   %d. %s - %s", i, session['session_id'], walkthrough_status)
                logger.info("=" * 100)
                
                # Process the new sessions
                logger.info("🚀 Starting processing of new sessions...")
                processing_result = pipeline_main_multichunks_sequential_sessions(
                    new_sessions, 
                    output_base_dir, 
                    azure_config, 
                    pipeline_config
                )
                
                logger.info("✅ Session processing completed - Total: %d, Completed: %d, Failed: %d, Skipped: %d",
                           processing_result['total_sessions'],
                           processing_result['completed_sessions'],
                           processing_result['failed_sessions'],
                           processing_result['skipped_sessions'])
                
                # Save updated tracking data
                save_processed_sessions_tracking(tracking_file, all_processed_sessions)
            else:
                logger.info("😴 No new sessions found - Total: %d, Already processed: %d",
                           len(all_sessions), len(all_processed_sessions))
            
        except Exception as e:
            logger.error(f"❌ Error during polling cycle #{polling_cycle}: {e}")
            logger.error(f"❌ Will retry in {polling_interval_minutes} minutes...")
        
        # Wait before next polling cycle
        logger.info("=" * 100)
        logger.info("⏰ Waiting %d minutes for next poll - Next poll at: %s", polling_interval_minutes, (datetime.now() + timedelta(minutes=polling_interval_minutes)).strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("=" * 100)
        
        # Sleep for the polling interval
        time.sleep(polling_interval_minutes * 60)

def read_session_metadata(blob_client, container_name: str, session_path: str) -> dict:
    """
    Read metadata from Azure Blob Storage for a session to determine walkthrough status.
    Searches for JSON files within the session directory.
    
    Args:
        blob_client: Azure blob service client
        container_name: Container name
        session_path: Session path like "one-data-platform/10-ca98a947-4f4b-4903-be80-f3d0fdc553ef-working-in-office"
    
    Returns:
        Dictionary with metadata and walkthrough status
    """
    try:
        # Get container client
        container_client = blob_client.get_container_client(container_name)
        
        logger.info("🔍 Searching for metadata in session: %s", session_path)
        
        # List all blobs in the session directory
        blobs = container_client.list_blobs(name_starts_with=session_path)
        
        # Find JSON files in the session
        json_files = []
        for blob in blobs:
            if blob.name.endswith('.json'):
                json_files.append(blob.name)
                logger.info(f"🔍 Found JSON file: {blob.name}")
        
        if not json_files:
            logger.warning(f"⚠️ No JSON files found in session: {session_path}")
            return {
                "metadata": {},
                "is_walkthrough": False,
                "domain": "",
                "activity": "",
                "specific_activity": ""
            }
        
        # Try to read each JSON file to find metadata
        for json_file in json_files:
            try:
                logger.info(f"🔍 Attempting to read JSON file: {json_file}")
                
                # Download the JSON file
                blob_data = container_client.download_blob(json_file).readall()
                metadata = json.loads(blob_data.decode('utf-8'))
                
                logger.info("📋 Processing metadata from: %s", json_file)
                
                # Determine walkthrough status based on metadata fields
                domain = metadata.get('domain', '').lower()
                activity = metadata.get('activity', '').lower()
                specific_activity = metadata.get('specific_activity', '').lower()
                
                # Check if any field indicates walkthrough
                is_walkthrough = (
                    'walkthrough' in domain or 
                    'walkthrough' in activity or 
                    'walkthrough' in specific_activity
                )
                
                logger.info("📋 Session %s - Domain: %s, Activity: %s, Walkthrough: %s",
                           os.path.basename(session_path),
                           metadata.get('domain', 'Unknown'),
                           metadata.get('activity', 'Unknown'),
                           is_walkthrough)
                
                if is_walkthrough:
                    logger.info("🔄 Walkthrough video detected - will use 3-minute segments with 60-second overlap")
                else:
                    logger.info("📹 Normal video detected - will use 3-minute segments without overlap")
                
                return {
                    "metadata": metadata,
                    "is_walkthrough": is_walkthrough,
                    "domain": metadata.get('domain', ''),
                    "activity": metadata.get('activity', ''),
                    "specific_activity": metadata.get('specific_activity', ''),
                    "json_file": json_file
                }
                
            except Exception as e:
                logger.warning(f"⚠️ Could not read JSON file {json_file}: {e}")
                continue
        
        # If no JSON file could be read successfully
        logger.warning(f"⚠️ Could not read any JSON files in session: {session_path}")
        return {
            "metadata": {},
            "is_walkthrough": False,
            "domain": "",
            "activity": "",
            "specific_activity": ""
        }
        
    except Exception as e:
        logger.warning(f"⚠️ Could not search for JSON files in {session_path}: {e}")
        return {
            "metadata": {},
            "is_walkthrough": False,
            "domain": "",
            "activity": "",
            "specific_activity": ""
        }

def discover_multiple_sessions(blob_client, container_name: str, base_prefix: str) -> list:
    """
    Discover multiple session directories under a base prefix and read their metadata.
    
    Args:
        blob_client: Azure blob service client
        container_name: Container name
        base_prefix: Base prefix like "one-data-platform/"
    
    Returns:
        List of session info dictionaries with metadata and walkthrough status
    """
    logger.info(f"🔍 Discovering sessions under prefix: {base_prefix}")
    
    session_info_list = []
    try:
        # Get container client and list all blobs under the base prefix
        container_client = blob_client.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=base_prefix)
        
        # Extract unique session directories
        session_dirs = set()
        for blob in blobs:
            # Get the relative path from base_prefix
            relative_path = blob.name[len(base_prefix):] if blob.name.startswith(base_prefix) else blob.name
            
            # Extract session directory (first part before any file)
            if '/' in relative_path:
                session_dir = relative_path.split('/')[0]
                if session_dir:  # Skip empty strings
                    session_dirs.add(session_dir)
        
        # Process each session directory
        for session_dir in sorted(session_dirs):
            session_path = f"{base_prefix.rstrip('/')}/{session_dir}"
            
            # Read metadata for this session
            metadata_info = read_session_metadata(blob_client, container_name, session_path)
            
            session_info = {
                "session_path": session_path,
                "session_id": session_dir,
                "is_walkthrough": metadata_info["is_walkthrough"],
                "metadata": metadata_info["metadata"],
                "domain": metadata_info["domain"],
                "activity": metadata_info["activity"],
                "specific_activity": metadata_info["specific_activity"]
            }
            
            session_info_list.append(session_info)
        
        logger.info(f"📋 Found {len(session_info_list)} session directories:")
        for i, session_info in enumerate(session_info_list, 1):
            walkthrough_status = "🔄 WALKTHROUGH" if session_info["is_walkthrough"] else "📹 NORMAL"
            logger.info(f"   {i}. {session_info['session_path']} - {walkthrough_status}")
            
    except Exception as e:
        logger.error(f"❌ Error discovering sessions: {e}")
    
    return session_info_list

def process_single_session(session_path: str, session_id: str, output_base_dir: str, azure_config: dict, pipeline_config: dict, is_walkthrough: bool = False):
    """
    Process a single session through the complete pipeline.
    """
    start_time = time.time()
    
    timer = PipelineTimer()

    try:
        # Download session
        logger.info(f"📥 Downloading session: {session_id}")
        with timer.time_operation(f"session.download"):
            download_result = download_ready_sessions(
                blob_service_client=create_azure_blob_client(azure_config),
                container_name=azure_config.get('container'),
                input_prefix=session_path,
                local_download_dir=pipeline_config['local_storage']['temp_download_dir'],
                max_sessions=1,
                use_parallel_chunks=True
            )
        
        if not download_result.get("success") or not download_result.get("download_results"):
            return {"success": False, "error": "Failed to download session"}
        
        # Read metadata to determine walkthrough status
        session_metadata_info = read_session_metadata(
            create_azure_blob_client(azure_config), 
            azure_config.get('container'), 
            session_path
        )
        actual_is_walkthrough = session_metadata_info["is_walkthrough"]
        
        logger.info("🔄 Processing session: %s (Walkthrough: %s)", session_id, actual_is_walkthrough)
        
        # Add walkthrough status to download results based on metadata
        for session_result in download_result.get("download_results", {}).values():
            session_result["processing_type"] = "walkthrough" if actual_is_walkthrough else "normal"
        
        session_output_dir = os.path.join(output_base_dir, session_id)
        
        pipeline_result = pipeline_main_multichunks(
            download_results=download_result,
            output_dir=session_output_dir,
            azure_output_prefix=f"{pipeline_config['azure_storage']['output_blob_prefix']}{session_id}",
            blob_client=create_azure_blob_client(azure_config),
            container_name=azure_config.get('container'),
            account_name=azure_config.get('account-name'),
            account_key=azure_config.get('account-key'),
            local_download_dir=pipeline_config['local_storage']['temp_download_dir'],
            timer=timer
        )
        
        processing_time = time.time() - start_time
        
        # Generate hierarchical timing data

        raw_timing_data = timer.get_timing_data()
        hierarchical_timing_data = generate_hierarchical_timing_structure(
            raw_timing_data, session_id, processing_time, start_time, pipeline_result
        )
        hierarchical_timing_data["performance_metrics"] = generate_performance_summary(hierarchical_timing_data)

        # Export to single CSV file - SAVE TO: output_folder/session_id/{session_id}_timing_analysis.csv
        csv_file_path = export_timing_data_to_single_csv(
            hierarchical_timing_data, session_output_dir, session_id
        )

        logger.info(f"📊 Timing analysis exported to: {csv_file_path}")

        # Upload CSV file to blob storage
        try:
            if azure_config and csv_file_path and os.path.exists(csv_file_path):
                blob_client = create_azure_blob_client(azure_config)
                csv_filename = os.path.basename(csv_file_path)
                csv_blob_path = f"{pipeline_config['azure_storage']['output_blob_prefix']}{session_id}/timing_analysis/{csv_filename}"

                logger.info(f"📤 Uploading CSV to blob: {csv_blob_path}")

                with open(csv_file_path, 'rb') as csv_file:
                    blob_client.upload_blob(
                        name=csv_blob_path,
                        data=csv_file,
                        overwrite=True
                    )

                logger.info(f"✅ CSV successfully uploaded to: {csv_blob_path}")
            else:
                logger.warning("⚠️ CSV upload skipped - missing Azure config or CSV file")
        except Exception as upload_error:
            logger.error(f"❌ Failed to upload CSV to blob: {upload_error}")

        return {
            "success": True,
            "processing_time": processing_time,
            "shards_processed": pipeline_result.get("total_shards_processed", 0) if pipeline_result else 0,
            "pipeline_result": pipeline_result
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_video_name_from_path(video_path: str) -> str:
    """Extract clean video name from file path"""
    import os
    filename = os.path.basename(video_path)
    # Remove extension and clean up
    name = os.path.splitext(filename)[0]
    return name

def pipeline_main_multichunks(download_results: dict, output_dir: str, azure_output_prefix: str=None, blob_client:str = None, container_name:str = None, account_name: str = None, account_key: str = None, local_download_dir: str = None, timer=None):
    
    try:
        if not ray.is_initialized():
            ray.init()
    except RuntimeError as e:
        if "ray.init twice" in str(e) or "already" in str(e).lower():
            logger.warning("Ray already initialized, continuing...")
        else:
            raise
    os.makedirs(output_dir, exist_ok=True)

    # --- STAGE A: INITIAL SETUP ---
    logger.info("Starting pipeline setup...")
    # Segregating download results
    for session_id, session_result in download_results.get("download_results", {}).items():
        output_dir = output_dir + f"/{session_id}"
        print(f"\n📦 video_name: {session_id}")
        
        # Check if session is already processed
        session_status = check_session_completion_status(session_id, output_base_dir)
        if session_status.get("completion_status") == "completed":
            logger.info(f"⏭️  Session {session_id} already completed at {session_status.get('completion_time')} - skipping processing")
            print(f"⏭️  Session {session_id} already completed - skipping processing")
            continue
        elif session_status.get("completion_status") == "pending" and session_status.get("exists"):
            logger.info(f"🔄 Session {session_id} was previously started but not completed - resuming processing")
            print(f"🔄 Session {session_id} was previously started but not completed - resuming processing")
        else:
            logger.info(f"🆕 Starting new session: {session_id}")
            print(f"🆕 Starting new session: {session_id}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        #check walkthrough
        is_walkthrough = False  # Initialize to False by default
        if session_result.get('processing_type') == 'walkthrough':
            is_walkthrough = True
        
        # BIG DEBUG: Show what processing_type we got and what is_walkthrough is set to
        logger.info("=" * 100)
        logger.info("🔍 Pipeline walkthrough check - Processing type: %s, is_walkthrough: %s, Equals 'walkthrough': %s", session_result.get('processing_type', 'NOT_SET'), is_walkthrough, session_result.get('processing_type') == 'walkthrough')
        logger.info("=" * 100)

        # Session metadata from JSON
        session_metadata = None
        if session_result.get("success") and session_result.get("session_metadata"):
            session_metadata = session_result["session_metadata"]
            logger.info(f"📋 Session metadata extracted for {session_id}")
            logger.info(f"   Home ID: {session_metadata.get('home_id', 'Unknown')}")
            logger.info(f"   Activity: {session_metadata.get('activity', 'Unknown')}")
            logger.info(f"   Domain: {session_metadata.get('domain', 'Unknown')}")
            logger.info(f"   Duration: {session_metadata.get('duration_minutes', 'Unknown')} minutes")
        else:
            logger.warning(f"⚠️ No session metadata available for session: {session_id}")
            # Create default metadata structure
            session_metadata = {
                "home_id": "",
                "participant_id": "",
                "activity": "Unknown",
                "domain": "production",
                "start_datetime": "",
                "end_datetime": "", 
                "duration_minutes": None,
                "room": "",
                "day_night": ""
            }

        # Initialize comprehensive tracking for this session
        tracker = initialize_global_tracker(session_id, output_dir)
        logger.info(f"🔍 Initialized comprehensive tracking for session: {session_id}")

        if session_result["success"]:
            # Collect all video and audio chunks first, then process with sequential part numbering
            video_downloads = session_result.get('download_result',{}).get('video_downloads',{}).items()
            audio_downloads = session_result.get('download_result',{}).get('audio_downloads',{}).items()
            
            # Process video chunks with sequential part numbering
            view_shards = {}
            view_shard_urls = {}
            global_part_idx = 0  # Track global part number across all chunks
            
            for video_id, video_download_result in video_downloads:
                local_video_path = video_download_result['local_path']
                video_name = extract_video_name_from_path(local_video_path)
                #local_video_path = "VID_20250809_094836_00_045.insv" #temp for debug
                logger.info(f"local_video_path:{local_video_path}")
                
                # Initialize tracking for this video chunk
                chunk_id = extract_chunk_id_from_path(local_video_path) or f"{video_id}-video"
                sequence_number = extract_sequence_number_from_path(local_video_path) or video_id
                chunk_type = extract_chunk_type_from_path(local_video_path) or "video"
                
                tracker.initialize_chunk_tracking(chunk_id, chunk_type, local_video_path, sequence_number)
                update_tracking(chunk_id, "download", "completed")
                if timer:
                    with timer.time_operation(f"video.{video_name}.unwarping"):
                        if local_video_path.lower().endswith('.insv'):
                            # Try the simpler single-output
                            update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "processing")
                            try:
                                mp4_result = ray.get(insv_unwarp_task.remote(local_video_path, out_dir=os.path.join(output_dir, "4views")))

                                # Check if the unwarp task returned an error
                                if "__error__" in mp4_result:
                                    error_msg = mp4_result["__error__"]
                                    logger.error(f"INSV unwarp task failed: {error_msg}")
                                    update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "error",
                                                error_message=error_msg)
                                    continue  # Skip to next chunk

                                flat_result = mp4_result
                                update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "completed",
                                            output_path=mp4_result.get('output_path'))
                                update_tracking(chunk_id, "video_processing.view_unwarping", "completed",
                                            views_created=flat_result)
                            except Exception as e:
                                logger.error(f"Exception raised in video conversion to mp4: {e}")
                                update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "error",
                                            error_message=str(e))
                                continue  # Skip to next chunk
                        else:
                            flat_result = local_video_path
                            update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "skipped")
                            update_tracking(chunk_id, "video_processing.view_unwarping", "skipped")
                else:
                    if local_video_path.lower().endswith('.insv'):
                        # Try the simpler single-output
                        update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "processing")
                        try:
                            mp4_result = ray.get(insv_unwarp_task.remote(local_video_path, out_dir=os.path.join(output_dir, "4views")))

                            # Check if the unwarp task returned an error
                            if "__error__" in mp4_result:
                                error_msg = mp4_result["__error__"]
                                logger.error(f"INSV unwarp task failed: {error_msg}")
                                update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "error",
                                            error_message=error_msg)
                                continue  # Skip to next chunk

                            flat_result = mp4_result
                            update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "completed",
                                        output_path=mp4_result.get('output_path'))
                            update_tracking(chunk_id, "video_processing.view_unwarping", "completed",
                                        views_created=flat_result)
                        except Exception as e:
                            logger.error(f"Exception raised in video conversion to mp4: {e}")
                            update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "error",
                                        error_message=str(e))
                            continue  # Skip to next chunk
                    else:
                        flat_result = local_video_path
                        update_tracking(chunk_id, "video_processing.insv_to_mp4_conversion", "skipped")
                        update_tracking(chunk_id, "video_processing.view_unwarping", "skipped")
                
                # Check if flat_result is valid before iterating
                if not isinstance(flat_result, dict) or "__error__" in flat_result:
                    logger.error(f"Invalid flat_result for sharding: {flat_result}")
                    continue  # Skip to next chunk

                view_names = list(flat_result.keys())
                logger.info(f"Processing sharding for views={view_names} starting from part{global_part_idx}")
                try:
                    update_tracking(chunk_id, "view_sharding", "processing", views=view_names)
                except Exception:
                    logger.debug("Failed to update tracking: view_sharding processing")

                split_tasks = []  # [(vn, ref, out_dir)]
                duration_sec = 180  # keep 180s by default; change if needed
                overlap_sec = 60 if is_walkthrough else 0

                for vn, vp in flat_result.items():
                    out_dir = os.path.join(output_dir, f"{vn}_shards")
                    os.makedirs(out_dir, exist_ok=True)
                    view_shard_urls.setdefault(vn, {})  # keep structure for later upload
                    if timer:
                        with timer.time_operation(f"video.{video_name}.video_sharding"):
                            if is_walkthrough:
                                logger.info("=" * 120)
                                logger.info("🔄 Walkthrough video splitting - View: %s, Path: %s, Duration: %ss, Overlap: %ss, Step: %ss",
                                            vn, vp, duration_sec, overlap_sec, max(0, duration_sec - overlap_sec))
                                logger.info("=" * 120)
                                ref = split_video_into_shards_with_overlap.remote(
                                    vp, output_dir=out_dir, duration_sec=duration_sec, overlap_sec=overlap_sec, start_idx=global_part_idx
                                )
                            else:
                                logger.info("=" * 120)
                                logger.info("📹 Normal video splitting - View: %s, Path: %s, Duration: %ss, Overlap: 0s", vn, vp, duration_sec)
                                logger.info("=" * 120)
                                ref = split_video_into_shards.remote(
                                    vp, output_dir=out_dir, duration_sec=duration_sec, start_idx=global_part_idx
                                )
                            split_tasks.append((vn, ref, out_dir))
                    else:
                        if is_walkthrough:
                            logger.info("=" * 120)
                            logger.info("🔄 Walkthrough video splitting - View: %s, Path: %s, Duration: %ss, Overlap: %ss, Step: %ss",
                                        vn, vp, duration_sec, overlap_sec, max(0, duration_sec - overlap_sec))
                            logger.info("=" * 120)
                            ref = split_video_into_shards_with_overlap.remote(
                                vp, output_dir=out_dir, duration_sec=duration_sec, overlap_sec=overlap_sec, start_idx=global_part_idx
                            )
                        else:
                            logger.info("=" * 120)
                            logger.info("📹 Normal video splitting - View: %s, Path: %s, Duration: %ss, Overlap: 0s", vn, vp, duration_sec)
                            logger.info("=" * 120)
                            ref = split_video_into_shards.remote(
                                vp, output_dir=out_dir, duration_sec=duration_sec, start_idx=global_part_idx
                            )
                        split_tasks.append((vn, ref, out_dir))


                # Prefer parallel; on failure, error tracking
                try:
                    results = ray.get([ref for _, ref, _ in split_tasks])  # List[List[str]]
                except Exception as e:
                    logger.error(f"Parallel split_video_into_shards failed: {e}")


                # Rename shards to ensure global sequential numbering across all views
                for (vn, _, out_dir), shards in zip(split_tasks, results):
                    shards = shards or []
                    renumbered = []
                    for old_path in shards:
                        base, ext = os.path.splitext(old_path)
                        # rename to global_part_idx
                        new_path = os.path.join(out_dir, f"{vn}_part{global_part_idx:04d}{ext or '.mp4'}")
                        if new_path != old_path:
                            try:
                                os.replace(old_path, new_path)  
                            except Exception as rn_ex:
                                logger.warning(f"Rename shard failed ({old_path} -> {new_path}): {rn_ex}; keeping original")
                                new_path = old_path
                        renumbered.append(new_path)
                        global_part_idx += 1
                    
                    
                    # record in view_shards
                    view_shards[vn] = view_shards.get(vn, []) + renumbered
                    try:
                        update_tracking(
                            chunk_id, f"view_sharding.{vn}", "completed",
                            shard_count=len(renumbered), shard_paths=renumbered
                        )
                        global_part_idx = 0  # Reset for next view
                    except Exception:
                        logger.debug(f"Failed to update tracking for {vn}")

                logger.info(f"Sharding done; global_part_idx now: {global_part_idx}")
                # Store the shards for this chunk
                
                # Update global part index based on the number of shards created
                if flat_result:
                    # Get the number of shards from the first view (all views should have same count)
                    first_view = list(flat_result.keys())[0]
                    if first_view in view_shards:
                        # Count only the new shards added in this iteration
                        existing_count = len(view_shards[first_view]) - len(shards) if first_view in view_shards else 0
                        new_shard_count = len(shards)
                        global_part_idx += new_shard_count
                        logger.info(f"Added {new_shard_count} shards for chunk {video_id}, global_part_idx now: {global_part_idx}")
            
            logger.info(view_shards)
                
            # Process audio chunks with sequential part numbering
            audio_shards = []
            global_part_idx = 0  # Reset for audio processing
            
            for audio_id, audio_download_result in audio_downloads:
                local_audio_path = audio_download_result['local_path']
                logger.info(f"local_audio_path:{local_audio_path}")
                
                # Initialize tracking for this audio chunk
                audio_chunk_id = extract_chunk_id_from_path(local_audio_path) or f"{audio_id}-audio"
                audio_sequence_number = extract_sequence_number_from_path(local_audio_path) or audio_id
                audio_chunk_type = extract_chunk_type_from_path(local_audio_path) or "audio"
                
                tracker.initialize_chunk_tracking(audio_chunk_id, audio_chunk_type, local_audio_path, audio_sequence_number)
                update_tracking(audio_chunk_id, "download", "completed")
                
                try:
                    with timer.time_operation(f"video.{video_name}.audio_sharding"):
                        if is_walkthrough:
                            # BIG DEBUG STATEMENT FOR WALKTHROUGH AUDIO SPLITTING
                            logger.info("=" * 120)
                            logger.info("🎵 Walkthrough audio splitting - Path: %s, Duration: 180s, Overlap: 60s, Step: 120s", local_audio_path)
                            logger.info("=" * 120)
                            
                            # Use sliding window overlap for audio as well
                            chunk_audio_shards = ray.get(split_audio_into_shards_with_overlap.remote(
                                local_audio_path,
                                output_dir=os.path.join(output_dir, "audio_shards"),
                                duration_sec=duration_sec,
                                overlap_sec=overlap_sec,  # 60 seconds overlap
                                start_idx=global_part_idx  # Pass the global part index
                            ))
                        else:
                            # BIG DEBUG STATEMENT FOR NORMAL AUDIO SPLITTING
                            logger.info("=" * 120)
                            logger.info("🎵 Normal audio splitting - Path: %s, Duration: 180s, Overlap: 0s", local_audio_path)
                            logger.info("=" * 120)
                            
                            # Normal processing without overlap
                            chunk_audio_shards = ray.get(split_audio_into_shards.remote(
                                local_audio_path,
                                output_dir=os.path.join(output_dir, "audio_shards"),
                                duration_sec=duration_sec,
                                start_idx=global_part_idx  # Pass the global part index
                            ))
                        audio_shards.extend(chunk_audio_shards)
                        global_part_idx += len(chunk_audio_shards)
                        logger.info(f"Added {len(chunk_audio_shards)} audio shards for chunk {audio_id}, global_part_idx now: {global_part_idx}")
                        
                        # Update tracking for audio sharding (using a generic view name for audio)
                        update_tracking(audio_chunk_id, "view_sharding.audio", "completed",
                                    shard_count=len(chunk_audio_shards), shard_paths=chunk_audio_shards)
                except Exception as e:
                    logger.error(f"Error processing audio chunk {audio_id}: {e}")
                    update_tracking(audio_chunk_id, "view_sharding.audio", "error",
                                   error_message=str(e))
            
            logger.info(audio_shards)
            # Compare shard counts
            audio_count = len(audio_shards)
            # Compare counts of audio and video shards
            video_counts = {view: len(shards) for view, shards in view_shards.items()}

            logger.info(f"Audio shard count: {audio_count}")
            for view, count in video_counts.items():
                logger.info(f"{view} shard count: {count}")

            # Consistency check
            all_match = all(count == audio_count for count in video_counts.values())
            if all_match:
                logger.info("✅ Audio and video shard counts match across all views")
                min_shards = audio_count
            else:
                logger.warning("⚠️ Shard count mismatch! processing the minimum or audio and video counts")
                mismatches = {view: count for view, count in video_counts.items() if count != audio_count}
                logger.warning(f"Mismatches: {mismatches}, expected {audio_count}")
                min_shards = min(len(shards) for shards in view_shards.values()) if view_shards else 0
                min_shards = min(min_shards, len(audio_shards)) if audio_shards else min_shards
            logger.info(f"Processing {min_shards} aligned shards across {len(view_shards)} unwarped views")

        # Process each shard index across all views
        label_studio_tasks = []
        consolidated_json_paths = []

        logger.info(f"🎬 Processing {min_shards} unwarped shards with {len(view_shards)} views using equal processing")
        logger.info(f"🎯 All views will be processed equally: {list(view_shards.keys())}")

        for shard_idx in range(min_shards):
            # Prepare view shard paths and URLs for this shard
            shard_view_paths = {}
            # shard_view_urls = {}
            
            for view_name in view_shards:
                if shard_idx < len(view_shards[view_name]):
                    shard_view_paths[view_name] = view_shards[view_name][shard_idx]
                    # shard_view_urls[view_name] = view_shard_urls[view_name].get(shard_idx, shard_view_paths[view_name])
            
            if not shard_view_paths:
                logger.warning(f"No video shards found for shard index {shard_idx}, skipping")
                continue
            
            # Get audio for this shard
            audio_shard = audio_shards[shard_idx] if shard_idx < len(audio_shards) else input_audio_path
            # audio_url = audio_urls.get(shard_idx, audio_shard)
             
            # Process all views equally using new multi-view function
            shard_results = process_time_aligned_shard_multiview(
                shard_index=shard_idx,
                view_shard_paths=shard_view_paths,
                audio_shard_path=audio_shard,
                base_output_dir=output_dir,
                view_azure_urls={},
                audio_url={},
                total_shard_count=min_shards,
                video_name=session_id,
                multi_chunk_process=True,
                session_metadata=session_metadata,
                timer = timer
            )
            
            label_studio_tasks.append(shard_results['label_studio_task'])
            consolidated_json_paths.append(shard_results['consolidated_model_results_json'])
            
            logger.info(f"✅ Completed multi-view processing for shard {shard_idx + 1}: {shard_results['total_views_processed']} views, {shard_results['successful_views']} successful")
        
        
        
        print(label_studio_tasks)
        generate_final_combined_model_results_json(output_dir, consolidated_json_paths)
        
        logger.info(f"🎯 Running video-level domain classification for video: {session_id}")
        if timer:
            with timer.time_operation(f"video.{session_id}.domain_classification"):
                domain_result = process_video_level_domain_and_update_tasks(output_dir, session_id)
        else:
            domain_result = process_video_level_domain_and_update_tasks(output_dir, session_id)

        if domain_result.get("success", False):
            logger.info(f"✅ Video-level domain processing completed: {domain_result.get('video_domain', 'Unknown')}")
        else:
            logger.error(f"❌ Video-level domain processing failed: {domain_result.get('error', 'Unknown error')}")
        
        azure_output_prefix = azure_output_prefix + f"/video_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if blob_client and container_name:
            video_name_for_upload = os.path.basename(output_dir)
            upload_config = pipeline_config.get('blob_upload', {})
            if timer:
                 with timer.time_operation(f"video.{session_id}.upload"):
                    upload_result = upload_output_directory_with_sas_optimized(
                        output_dir, blob_client, container_name,
                        azure_output_prefix, video_name_for_upload,
                        account_name, account_key,
                        365,  # sas_expiry_days
                        upload_config  # config dict with blob_upload settings
                    )
            else:
                upload_result = upload_output_directory_with_sas_optimized(
                        output_dir, blob_client, container_name,
                        azure_output_prefix, video_name_for_upload,
                        account_name, account_key,
                        365,  # sas_expiry_days
                        upload_config  # config dict with blob_upload settings
                    )
            logger.info(f" Multi-view upload result: {' Success' if upload_result['success'] else '❌ Failed'}")
            
            if upload_result['success']:
                # Extract URLs from upload result
                shard_urls = upload_result['shard_urls']
                audio_urls = shard_urls['audio_urls']
                view_shard_urls = shard_urls['view_urls']
                
                logger.info(f" Generated {len(audio_urls)} audio URLs and {sum(len(urls) for urls in view_shard_urls.values())} view URLs with SAS tokens")
                
                # CRITICAL: Update Label Studio tasks with real URLs
                logger.info(" Updating Label Studio tasks with uploaded URLs...")
                updated_count = update_labelstudio_tasks_with_new_urls(label_studio_tasks, shard_urls, "multi_view")
                if updated_count != len(label_studio_tasks):
                    logger.warning(f" Only {updated_count}/{len(label_studio_tasks)} Label Studio tasks were updated with URLs")
                
            else:
                logger.error(f"Upload error: {upload_result.get('error', 'Unknown error')}")
                audio_urls = {}
                view_shard_urls = {}
        else:
            logger.warning("⚠️ Azure configuration missing - Label Studio tasks will have placeholder URLs")
            audio_urls = {}
            view_shard_urls = {}
        # =================================================================
        # DYNAMIC ERP VIDEO AND AUDIO PROCESSING - AFTER LABEL STUDIO TASKS
        # =================================================================
        logger.info(f"🎬 Starting dynamic ERP video and audio processing for session: {session_id}")
        
        # Integrate ERP video and audio URLs with Label Studio tasks
        label_studio_tasks = integrate_erp_audio_with_labelstudio_tasks(
            output_dir=output_dir,
            session_id=session_id,
            blob_service_client=blob_client,
            container_name=container_name,
            blob_base_path=azure_output_prefix,
            account_key=account_key,
            labelstudio_tasks=label_studio_tasks
        )
        
        logger.info(f"🎬 Dynamic ERP and audio processing completed for session: {session_id}")
        # =================================================================
        import_consolidated_tasks_to_labelstudio(label_studio_tasks,pipeline_config, is_walkthrough)
        
        # Mark session as completed
        tracker.mark_session_complete()
        logger.info(f"✅ Session {session_id} processing completed successfully")
        print(f"✅ Session {session_id} processing completed successfully")
        
        # Clean up temporary download directory for this session
        if pipeline_config.get('cleanup', {}).get('cleanup_temp_dir', True):
            logger.info(f"🧹 Cleaning up temporary download directory for session: {session_id}")
            if local_download_dir is not None:
                cleanup_result = cleanup_temp_download_directory(local_download_dir, session_id)
                if cleanup_result.get("success", False):
                    logger.info(f"✅ Temp cleanup completed: {cleanup_result.get('files_removed', 0)} files removed")
                else:
                    logger.warning(f"⚠️ Temp cleanup failed: {cleanup_result.get('error', 'Unknown error')}")
            else:
                logger.warning(f"⚠️ local_download_dir is None - skipping temp cleanup")
        else:
            logger.info(f"ℹ️ Temp cleanup disabled in config - keeping temp files")
        
        # Clean up session output directory after successful upload
        if pipeline_config.get('cleanup', {}).get('cleanup_output_dir', True):
            logger.info(f"🧹 Cleaning up session output directory for session: {session_id}")
            output_cleanup_result = cleanup_session_output_directory(output_dir, session_id)
            if output_cleanup_result.get("success", False):
                logger.info(f"✅ Output cleanup completed: {output_cleanup_result.get('files_removed', 0)} files, {output_cleanup_result.get('dirs_removed', 0)} directories removed")
            else:
                logger.warning(f"⚠️ Output cleanup failed: {output_cleanup_result.get('error', 'Unknown error')}")
        else:
            logger.info(f"ℹ️ Output cleanup disabled in config - keeping output files")

    # Return success result at the end of pipeline processing
    return {
        "success": True,
        "message": f"Pipeline completed successfully for session: {session_id}",
        "total_shards_processed": min_shards,  # Actual number of shards processed
        "session_id": session_id,
        "output_dir": output_dir
    }


def pipeline_main(input_video_path: str, input_audio_path: str, output_dir: str, 
                 process_dual_views: bool = None, process_unwarped_views: bool = True,
                 azure_blob_client=None, azure_container=None, azure_output_prefix=None, azure_account_name: str = None, azure_account_key: str = None, pipeline_config = None):
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
    print("3"*100)
    print(f"is_insv_file: {is_insv_file}")
    if process_dual_views is None:
        process_dual_views = is_insv_file
    
    # New branch: unwarped multi-view processing for INSV
    if process_unwarped_views and is_insv_file:
        logger.info("🎥 INSV file detected - enabling unwarped view processing")

        #test for ERP pipline
        # mp4_result = ray.get(insv_unwarp_task.remote(input_video_path, out_dir=os.path.join(output_dir, "4views")))
        # erp_result = mp4_result.get('erp')
        # print(f"erp_result: {erp_result}")
        # yolo_ref  = run_yolo_detection.remote(erp_result, "out_4viewsERP")

        # Convert .insv to four.mp4 videos directly (single-output converter)
        #viewsoutput_dir=os.path.join(output_dir, "4views"),
        if input_video_path.lower().endswith('.insv'):
            # Try the simpler single-output
            try:
                mp4_result = ray.get(insv_unwarp_task.remote(input_video_path, out_dir=os.path.join(output_dir, "4views")))

                # Check if the unwarp task returned an error
                if "__error__" in mp4_result:
                    error_msg = mp4_result["__error__"]
                    logger.error(f"INSV unwarp task failed: {error_msg}")
                    raise RuntimeError(f"Failed to convert INSV file: {error_msg}")

                flat_result = mp4_result
                logger.info(f"Using two 180 for unwarp:: {len(flat_result)} views under {flat_result}")
                # mp4_result = ray.get(insv_unwarp_task.remote(input_video_path, out_dir=os.path.join(output_dir, "4views")))
                # flat_result = mp4_result.get('erp')
                # print(f"erp_result: {erp_result}")
                # yolo_ref  = run_yolo_detection.remote(erp_result, "out_4viewsERP")
            except Exception as e:
                flat_result = None
                raise RuntimeError(f"Failed to convert two 180 INSV file: {str(e)}")
        else:
            mp4_path = input_video_path
            try:
                 # Undistort/unwarp the 360 video into multiple perspective views
                views4_ref = erp_unwarp_task.remote(mp4_path)
                flat_result = ray.get(views4_ref)  # dict: {view_name: output_path}

                # Check if the unwarp task returned an error
                if "__error__" in flat_result:
                    error_msg = flat_result["__error__"]
                    logger.error(f"ERP unwarp task failed: {error_msg}")
                    raise RuntimeError(f"Failed to convert 360 file: {error_msg}")

                logger.info(f"Using one 360 for unwarp: {len(flat_result)} views under {flat_result}")
            except Exception as e:
                flat_result = None
                raise RuntimeError(f"Failed to convert one 360 file: {str(e)}")

        duration_sec = 5
        # Split audio once into 60s shards (reused per view by index)
        audio_shards = ray.get(split_audio_into_shards.remote(
            input_audio_path,
            output_dir=os.path.join(output_dir, "audio_shards"),
            duration_sec=duration_sec
        ))

        # Optionally upload audio shards for LS streaming
        # if azure_blob_client and azure_container and azure_output_prefix and azure_account_name and azure_account_key:
        #     audio_urls = generate_azure_shard_urls(
        #         azure_blob_client, azure_container, audio_shards,
        #         f"{azure_output_prefix}/audio_shards",
        #         azure_account_name, azure_account_key
        #     )
        # else:
        #     audio_urls = {}
        audio_urls = {}


        # Split each unwarped view into shards and stage per-view shard lists
        view_shards = {}
        view_shard_urls = {}

        # Check if flat_result is valid before iterating
        if not isinstance(flat_result, dict) or "__error__" in flat_result:
            logger.error(f"Invalid flat_result for unwarped view processing: {flat_result}")
            return {"status": "error", "error": "Failed to process unwarped views due to invalid result"}

        # Launch all split jobs in parallel and collect refs
        _split_refs = []
        _split_order = []
        for view_name, view_path in flat_result.items():
            ref = split_video_into_shards.remote(
                view_path,
                output_dir=os.path.join(output_dir, f"{view_name}_shards"),
                duration_sec=duration_sec
            )
            _split_refs.append(ref)
            _split_order.append(view_name)
            # initialize urls dict for this view (filled later after upload)
            view_shard_urls[view_name] = {}

        # Execute all splits concurrently and map results back to view names
        try:
            _split_results = ray.get(_split_refs)
        except Exception as e:
            logger.error(f"Parallel split_video_into_shards failed: {e}")
            # Fallback: try sequentially so we still produce something
            _split_results = []
            for view_name, view_path in flat_result.items():
                try:
                    shards = ray.get(split_video_into_shards.remote(
                        view_path,
                        output_dir=os.path.join(output_dir, f"{view_name}_shards"),
                        duration_sec=duration_sec
                    ))
                except Exception as ex:
                    logger.error(f"Sequential split failed for {view_name}: {ex}")
                    shards = []
                _split_results.append(shards)

        for vn, shards in zip(_split_order, _split_results):
            view_shards[vn] = shards if shards is not None else []

        # Find the minimum shard count across all views to avoid index errors
        min_shards = min(len(shards) for shards in view_shards.values()) if view_shards else 0
        min_shards = min(min_shards, len(audio_shards)) if audio_shards else min_shards

        if min_shards == 0:
            logger.warning("No aligned shards found across views/audio for unwarped processing")
            return create_consolidated_summary([], output_dir)

        logger.info(f"Processing {min_shards} aligned shards across {len(view_shards)} unwarped views")

        # Process each shard index across all views
        label_studio_tasks = []
        consolidated_json_paths = []

        logger.info(f"🎬 Processing {min_shards} unwarped shards with {len(view_shards)} views using equal processing")
        logger.info(f"🎯 All views will be processed equally: {list(view_shards.keys())}")

        for shard_idx in range(min_shards):
            # Prepare view shard paths and URLs for this shard
            shard_view_paths = {}
            shard_view_urls = {}
            
            for view_name in view_shards:
                if shard_idx < len(view_shards[view_name]):
                    shard_view_paths[view_name] = view_shards[view_name][shard_idx]
                    shard_view_urls[view_name] = view_shard_urls[view_name].get(shard_idx, shard_view_paths[view_name])
            
            if not shard_view_paths:
                logger.warning(f"No video shards found for shard index {shard_idx}, skipping")
                continue
            
            # Get audio for this shard
            audio_shard = audio_shards[shard_idx] if shard_idx < len(audio_shards) else input_audio_path
            audio_url = audio_urls.get(shard_idx, audio_shard)
            
            # Process all views equally using new multi-view function
            shard_results = process_time_aligned_shard_multiview(
                shard_index=shard_idx,
                view_shard_paths=shard_view_paths,
                audio_shard_path=audio_shard,
                base_output_dir=output_dir,
                view_azure_urls=shard_view_urls,
                audio_url=audio_url,
                total_shard_count=min_shards,
                video_name=video_name,
                multi_chunk_process=False
            )
            
            label_studio_tasks.append(shard_results['label_studio_task'])
            consolidated_json_paths.append(shard_results['consolidated_model_results_json'])
            
            logger.info(f"✅ Completed multi-view processing for shard {shard_idx + 1}: {shard_results['total_views_processed']} views, {shard_results['successful_views']} successful")

        # =================================================================
        # DYNAMIC ERP VIDEO AND AUDIO PROCESSING - AFTER LABEL STUDIO TASKS
        # =================================================================
        logger.info(f"🎬 Starting dynamic ERP video and audio processing for session: {session_id}")
        
        # Integrate ERP video and audio URLs with Label Studio tasks
        label_studio_tasks = integrate_erp_audio_with_labelstudio_tasks(
            output_dir=output_dir,
            session_id=session_id,
            blob_service_client=blob_client,
            container_name=container_name,
            blob_base_path=azure_output_prefix,
            account_key=account_key,
            labelstudio_tasks=label_studio_tasks
        )
        
        logger.info(f"🎬 Dynamic ERP and audio processing completed for session: {session_id}")
        # =================================================================
        
        # Add missing final processing steps  
        generate_final_combined_model_results_json(output_dir, consolidated_json_paths)
        
        logger.info(f"🎯 Running video-level domain classification for video: {video_name}")
        domain_result = process_video_level_domain_and_update_tasks(output_dir, video_name)
        if domain_result.get("success", False):
            logger.info(f"✅ Video-level domain processing completed: {domain_result.get('video_domain', 'Unknown')}")
        else:
            logger.error(f"❌ Video-level domain processing failed: {domain_result.get('error', 'Unknown error')}")
        
        if azure_blob_client and azure_container:
            video_name_for_upload = os.path.basename(output_dir)
            upload_config = pipeline_config.get('blob_upload', {})

            upload_result = upload_output_directory_with_sas_optimized(
                output_dir, blob_client, container_name,
                azure_output_prefix, video_name_for_upload,
                account_name, account_key,
                365,  # sas_expiry_days
                upload_config  # config dict with blob_upload settings
            )
            logger.info(f" Multi-view upload result: {' Success' if upload_result['success'] else '❌ Failed'}")
            
            if upload_result['success']:
                # Extract URLs from upload result
                shard_urls = upload_result['shard_urls']
                audio_urls = shard_urls['audio_urls']
                view_shard_urls = shard_urls['view_urls']
                
                logger.info(f" Generated {len(audio_urls)} audio URLs and {sum(len(urls) for urls in view_shard_urls.values())} view URLs with SAS tokens")
                
                # CRITICAL: Update Label Studio tasks with real URLs
                logger.info(" Updating Label Studio tasks with uploaded URLs...")
                updated_count = update_labelstudio_tasks_with_new_urls(label_studio_tasks, shard_urls, "multi_view")
                if updated_count != len(label_studio_tasks):
                    logger.warning(f" Only {updated_count}/{len(label_studio_tasks)} Label Studio tasks were updated with URLs")
                
            else:
                logger.error(f"Upload error: {upload_result.get('error', 'Unknown error')}")
                audio_urls = {}
                view_shard_urls = {}
        else:
            logger.warning("⚠️ Azure configuration missing - Label Studio tasks will have placeholder URLs")
            audio_urls = {}
            view_shard_urls = {}

        import_consolidated_tasks_to_labelstudio(label_studio_tasks,pipeline_config, False)
        # if azure_blob_client and azure_container:
        #     video_name_for_upload = os.path.basename(output_dir)
        #     upload_result = upload_output_directory_to_blob(
        #         output_dir, azure_blob_client, azure_container,
        #         azure_output_prefix, video_name_for_upload
        #     )
        #     logger.info(f"📤 Multi-view upload result: {'✅ Success' if upload_result['success'] else '❌ Failed'}")
        #     if not upload_result['success']:
        #         logger.error(f"Upload error: {upload_result.get('error', 'Unknown error')}")
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
        # if azure_blob_client and azure_container and azure_output_prefix and azure_account_name and azure_account_key:
        #     view1_shard_urls = generate_azure_shard_urls(
        #         azure_blob_client, azure_container, view1_shards,
        #         f"{azure_output_prefix}/view_1_shards",
        #         azure_account_name, azure_account_key
        #     )
        #     view2_shard_urls = generate_azure_shard_urls(
        #         azure_blob_client, azure_container, view2_shards,
        #         f"{azure_output_prefix}/view_2_shards",
        #         azure_account_name, azure_account_key
        #     )
        #     audio_shard_urls = generate_azure_shard_urls(
        #         azure_blob_client, azure_container, audio_shards,
        #         f"{azure_output_prefix}/audio_shards",
        #         azure_account_name, azure_account_key
        #     )
        # else:
        #     logger.warning("Azure client/prefix or account key missing — LS URLs will be local and likely won't stream.")
        #     view1_shard_urls, view2_shard_urls, audio_shard_urls = {}, {}, {}
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
        
        # =================================================================
        # DYNAMIC ERP VIDEO AND AUDIO PROCESSING - AFTER LABEL STUDIO TASKS
        # =================================================================
        logger.info(f"🎬 Starting dynamic ERP video and audio processing for session: {session_id}")
        
        # Integrate ERP video and audio URLs with Label Studio tasks
        label_studio_tasks = integrate_erp_audio_with_labelstudio_tasks(
            output_dir=output_dir,
            session_id=session_id,
            blob_service_client=blob_client,
            container_name=container_name,
            blob_base_path=azure_output_prefix,
            account_key=account_key,
            labelstudio_tasks=label_studio_tasks
        )
        
        logger.info(f"🎬 Dynamic ERP and audio processing completed for session: {session_id}")
        # =================================================================
        
        # Generate final combined JSON with all shards
        generate_final_combined_model_results_json(output_dir, consolidated_json_paths)
        
        logger.info(f"🎯 Running video-level domain classification for video: {video_name}")
        domain_result = process_video_level_domain_and_update_tasks(output_dir, video_name)
        if domain_result.get("success", False):
            logger.info(f"✅ Video-level domain processing completed: {domain_result.get('video_domain', 'Unknown')}")
        else:
            logger.error(f"❌ Video-level domain processing failed: {domain_result.get('error', 'Unknown error')}")


        if azure_blob_client and azure_container:
            video_name = os.path.basename(output_dir)
            upload_config = pipeline_config.get('blob_upload', {})

            upload_result = upload_output_directory_with_sas_optimized(
                output_dir, blob_client, container_name,
                azure_output_prefix, video_name_for_upload,
                account_name, account_key,
                365,  # sas_expiry_days
                upload_config  # config dict with blob_upload settings
            )
            logger.info(f"📤 Dual-view upload result: {'✅ Success' if upload_result['success'] else '❌ Failed'}")
            
            if upload_result['success']:
                # Extract URLs from upload result for dual-view
                dual_view_urls = upload_result['shard_urls']['dual_view_urls']
                view1_shard_urls = dual_view_urls['view1_urls']
                view2_shard_urls = dual_view_urls['view2_urls'] 
                audio_shard_urls = dual_view_urls['audio_urls']
                
                logger.info(f"🔗 Generated {len(view1_shard_urls)} view1, {len(view2_shard_urls)} view2, {len(audio_shard_urls)} audio URLs with SAS tokens")
                
                # CRITICAL: Update Label Studio tasks with real URLs
                logger.info("🔄 Updating Label Studio tasks with uploaded URLs...")
                updated_count = update_labelstudio_tasks_with_new_urls(label_studio_tasks, upload_result['shard_urls'], "dual_view")
                if updated_count != len(label_studio_tasks):
                    logger.warning(f"⚠️ Only {updated_count}/{len(label_studio_tasks)} Label Studio tasks were updated with URLs")
                
            else:
                logger.error(f"Upload error: {upload_result.get('error', 'Unknown error')}")
                view1_shard_urls, view2_shard_urls, audio_shard_urls = {}, {}, {}
        else:
            logger.warning("⚠️ Azure configuration missing - Label Studio tasks will have placeholder URLs")
            view1_shard_urls, view2_shard_urls, audio_shard_urls = {}, {}, {}
        # Import consolidated tasks to Label Studio
        import_consolidated_tasks_to_labelstudio(label_studio_tasks,pipeline_config, False)
        
        # Upload output directory to Azure blob storage
        # if azure_blob_client and azure_container:
        #     video_name = os.path.basename(output_dir)
        #     upload_result = upload_output_directory_to_blob(
        #         output_dir, azure_blob_client, azure_container,
        #         azure_output_prefix, video_name
        #     )
        #     logger.info(f"📤 Upload result: {'✅ Success' if upload_result['success'] else '❌ Failed'}")
        #     if not upload_result['success']:
        #         logger.error(f"Upload error: {upload_result.get('error', 'Unknown error')}")
        
        return create_consolidated_summary(label_studio_tasks, output_dir)
        
    else:
        # Single view processing (original logic)
        if is_insv_file:
            logger.info("🎥 INSV file detected - processing single view only")
            mp4_result_ref = convert_insv_to_dual_mp4.remote(input_video_path)
            mp4_result = ray.get(mp4_result_ref)
            
            if mp4_result.get('success', False):
                mp4_path = mp4_result['output_view_1']  # Use the upload_output_directory_to_blob view for processing
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
                    azure_account_key,
                    pipeline_config          
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
            "clap": view1_results.get('clap', {}),
            "signal_quality": view1_results.get('signal_quality', {})
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
            "clap": view2_results.get('clap', {}),
            "signal_quality": view2_results.get('signal_quality', {})
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
                
                # Create base shard data structure - handle both dual-view and multi-view
                if 'view_results' in shard_data:
                    # Multi-view structure (front, back, left, right, etc.)
                    shard_data_structure = shard_data.get('view_results', {})
                else:
                    # Dual-view structure (view1, view2)
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
                        first_shard_clap_detected = clap_flags.get('clap_detected_in_first_shard', False)
                        intro_statement_detected = clap_flags.get('is_intro_statement_there', False)
                        intro_transcript_text = clap_flags.get('intro_transcript', '')
                    
                    if is_last_shard_processed:
                        last_shard_clap_detected = clap_flags.get('clap_detected_in_last_shard', False)
                    
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

def consolidate_multiview_time_segment_results(view_results, shard_index, shard_offset_sec):
    """
    Consolidate results from multiple views (4+) for a single time segment.
    All views are treated equally in the consolidation process.
    
    Args:
        view_results: Dict of {view_name: view_result}
        shard_index: Current shard index
        shard_offset_sec: Time offset in seconds
    
    Returns:
        Dict with consolidated multi-view results
    """
    consolidated = {
        "shard_index": shard_index,
        "shard_offset_sec": shard_offset_sec,
        "time_range": f"{shard_offset_sec}-{shard_offset_sec + 60}s",
        "processing_type": "multi_view_equal_processing",
        "total_views": len(view_results),
        "view_results": view_results,
        "consolidated_predictions_for_ls": {},
        "merged_flagged_segments": [],
        "cross_view_analysis": {}
    }
    
    # --- CONSOLIDATE FLAGGED SEGMENTS ACROSS ALL VIEWS ---
    all_segments = []
    
    for view_name, view_result in view_results.items():
        if view_result.get('success', True) and view_result.get('flagged_segments'):
            # Tag each segment with its source view
            view_segments = [
                dict(seg, **{"source_view": view_name}) 
                for seg in view_result.get('flagged_segments', [])
            ]
            all_segments.extend(view_segments)
    
    # Use existing merge function but with tolerance for multiple views
    consolidated['merged_flagged_segments'] = merge_overlapping_segments(all_segments, tol=1.0)
    
    # --- CONSOLIDATED PREDICTIONS ---
    consolidated['consolidated_predictions_for_ls'] = create_multiview_consolidated_predictions(view_results)
    
    logger.info(f"✅ Consolidated {len(all_segments)} segments from {len(view_results)} views into {len(consolidated['merged_flagged_segments'])} merged segments")
    
    return consolidated

def create_multiview_consolidated_predictions(view_results):
    """
    Build ONLY compliance predictions (PII, NSFW, Minors) in the same shape as the
    manual `annotations.result` from complete-task.txt. All other model outputs (motion, yolo, clap, etc.) are intentionally omitted.
    Lighting, Signal Quality, and Sensitive Information are now included.


    Expected label config (based on complete-task.txt):
      - PII (audio):
          * val_pii_audio (choices Yes/No)    -> to_name="audio_main"
          * pii_type_audio (choices [...])    -> to_name="audio_main"
          * audio_labels_pii (labels spans)   -> to_name="audio_main", labels=["PII Issue"]
      - NSFW (video):
          * val_nudity_video (choices Yes/No) -> to_name="video_left" (or the view you prefer)
          * nudity_start_minute (number)
          * nudity_start_second (number)
          * nudity_end_minute (number)
          * nudity_end_second (number)
      - Minors (video):
          * val_minors_video (choices Yes/No) -> to_name="video_left"
          * minors_start_minute (number)
          * minors_start_second (number)
          * minors_end_minute (number)
          * minors_end_second (number)
      - Optional comments (textarea):
          * compliance_video_comment  -> to_name="md_home_id"
          * compliance_audio_comment  -> to_name="md_home_id"
    """
    def _mk_id(prefix):
        # short stable-ish ids; LS doesn't require UUIDs, only uniqueness in the list
        import random, string
        return f"{prefix}_{''.join(random.choices(string.ascii_letters + string.digits, k=6))}"

    predictions = []

    # -----------------------------
    # 1) PII (Audio-only)
    # -----------------------------
    pii_spans = []        # list of (start_sec, end_sec)
    pii_types = set()     # label choices for pii_type_audio (e.g., "Addresses", "Full names"...)

    # Consolidate across all views (audio is shared but many pipelines attach under each view)
    for view_name, view_result in view_results.items():
        if not view_result.get("success", True):
            continue
        audio_list = view_result.get("audio") or []
        if not isinstance(audio_list, list) or not audio_list:
            continue
        audio_res = audio_list[0]
        for pii in (audio_res.get("pii_detections") or []):
            try:
                s = float(pii.get("start_time", 0))
                e = float(pii.get("end_time", 0))
            except Exception:
                continue
            if e > s:
                pii_spans.append((s, e))
            # Try to extract a human category for the choice list
            # Common keys: 'type', 'pii_type', 'category', 'label'
            for k in ("pii_type", "type", "category", "label"):
                if pii.get(k):
                    pii_types.add(str(pii[k]))
                    break

    # Normalize/Map some common pii types to your project's choice set from the example
    # Example showed: "Addresses", "Financial or Account Numbers", "Full names"
    def _normalize_pii_type(t):
        t_low = t.lower()
        if "address" in t_low:
            return "Addresses"
        if "financial" in t_low or "account" in t_low or "credit" in t_low or "card" in t_low or "ssn" in t_low:
            return "Financial or Account Numbers"
        if "name" in t_low or "fullname" in t_low or "full name" in t_low:
            return "Full names"
        return None

    mapped_types = []
    for t in pii_types:
        mt = _normalize_pii_type(t)
        if mt:
            mapped_types.append(mt)

    if pii_spans:
        # Presence
        predictions.append({
            "id": _mk_id("pii_present"),
            "type": "choices",
            "value": {"choices": ["Yes"]},
            "model_version": "auto_preannotator_v1",
            "from_name": "val_pii_audio",
            "to_name": "audio_main"
        })

        # Types (if any recognized; if not, default to "Full names" as a safe fallback)
        if not mapped_types:
            mapped_types = ["Full names"]
        predictions.append({
            "id": _mk_id("pii_types"),
            "type": "choices",
            "value": {"choices": sorted(set(mapped_types))},
            "model_version": "auto_preannotator_v1",
            "from_name": "pii_type_audio",
            "to_name": "audio_main"
        })

        # Spans (multiple)
        # Label name must exist in your config; example uses "PII Issue"
        for s, e in sorted(pii_spans, key=lambda x: x[0]):
            predictions.append({
                "id": _mk_id("pii_span"),
                "type": "labels",
                "value": {
                    "start": float(s),
                    "end": float(e),
                    "labels": ["PII Issue"],
                    "channel": 0
                },
                "model_version": "auto_preannotator_v1",
                "from_name": "audio_labels_pii",
                "to_name": "audio_main"
            })
    else:
        # Explicitly mark "No" if you want negatives emitted; otherwise omit these three
        predictions.append({
            "id": _mk_id("pii_present"),
            "type": "choices",
            "value": {"choices": ["No"]},
            "model_version": "auto_preannotator_v1",
            "from_name": "val_pii_audio",
            "to_name": "audio_main"
        })

    # -----------------------------
    # 2) NSFW (Video)
    # -----------------------------
    nsfw_segments = []   # list of dicts with start_time/end_time
    for view_name, view_result in view_results.items():
        if not view_result.get("success", True):
            continue
        nsfw = view_result.get("nsfw") or {}
        if nsfw.get("success") and nsfw.get("total_nsfw_detections", 0) > 0:
            for seg in nsfw.get("flagged_segments", []) or []:
                s = float(seg.get("start_time", 0))
                e = float(seg.get("end_time", 0))
                if e > s:
                    nsfw_segments.append((s, e))

    if nsfw_segments:
        nsfw_segments.sort(key=lambda x: x[0])
        s, e = nsfw_segments[0]
        s_min, s_sec = int(s // 60), int(s % 60)
        e_min, e_sec = int(e // 60), int(e % 60)

        predictions.append({
            "id": _mk_id("nudity_yes"),
            "type": "choices",
            "value": {"choices": ["Yes"]},
            "model_version": "auto_preannotator_v1",
            "from_name": "val_nudity_video",
            "to_name": "video_left"
        })
        predictions.extend([
            {
                "id": _mk_id("nudity_smin"),
                "type": "number",
                "value": {"number": s_min},
                "model_version": "auto_preannotator_v1",
                "from_name": "nudity_start_minute",
                "to_name": "video_left"
            },
            {
                "id": _mk_id("nudity_ssec"),
                "type": "number",
                "value": {"number": s_sec},
                "model_version": "auto_preannotator_v1",
                "from_name": "nudity_start_second",
                "to_name": "video_left"
            },
            {
                "id": _mk_id("nudity_emin"),
                "type": "number",
                "value": {"number": e_min},
                "model_version": "auto_preannotator_v1",
                "from_name": "nudity_end_minute",
                "to_name": "video_left"
            },
            {
                "id": _mk_id("nudity_esec"),
                "type": "number",
                "value": {"number": e_sec},
                "model_version": "auto_preannotator_v1",
                "from_name": "nudity_end_second",
                "to_name": "video_left"
            }
        ])
    else:
        predictions.append({
            "id": _mk_id("nudity_no"),
            "type": "choices",
            "value": {"choices": ["No"]},
            "model_version": "auto_preannotator_v1",
            "from_name": "val_nudity_video",
            "to_name": "video_left"
        })

    # -----------------------------
    # 3) Minors (Video)
    # -----------------------------
    minor_segments = []
    for view_name, view_result in view_results.items():
        if not view_result.get("success", True):
            continue
        face = view_result.get("face") or {}
        if not face.get("success"):
            continue
        for seg in face.get("flagged_segments", []) or []:
            desc = str(seg.get("description", "")).lower()
            ftype = str(seg.get("flag_type", "")).lower()
            if "minor" in desc or "minor" in ftype:
                s = float(seg.get("start_time", 0))
                e = float(seg.get("end_time", 0))
                if e > s:
                    minor_segments.append((s, e))

    if minor_segments:
        minor_segments.sort(key=lambda x: x[0])
        s, e = minor_segments[0]
        s_min, s_sec = int(s // 60), int(s % 60)
        e_min, e_sec = int(e // 60), int(e % 60)

        predictions.append({
            "id": _mk_id("minors_yes"),
            "type": "choices",
            "value": {"choices": ["Yes"]},
            "model_version": "auto_preannotator_v1",
            "from_name": "val_minors_video",
            "to_name": "video_left"
        })
        predictions.extend([
            {
                "id": _mk_id("minors_smin"),
                "type": "number",
                "value": {"number": s_min},
                "model_version": "auto_preannotator_v1",
                "from_name": "minors_start_minute",
                "to_name": "video_left"
            },
            {
                "id": _mk_id("minors_ssec"),
                "type": "number",
                "value": {"number": s_sec},
                "model_version": "auto_preannotator_v1",
                "from_name": "minors_start_second",
                "to_name": "video_left"
            },
            {
                "id": _mk_id("minors_emin"),
                "type": "number",
                "value": {"number": e_min},
                "model_version": "auto_preannotator_v1",
                "from_name": "minors_end_minute",
                "to_name": "video_left"
            },
            {
                "id": _mk_id("minors_esec"),
                "type": "number",
                "value": {"number": e_sec},
                "model_version": "auto_preannotator_v1",
                "from_name": "minors_end_second",
                "to_name": "video_left"
            }
        ])
    else:
        predictions.append({
            "id": _mk_id("minors_no"),
            "type": "choices",
            "value": {"choices": ["No"]},
            "model_version": "auto_preannotator_v1",
            "from_name": "val_minors_video",
            "to_name": "video_left"
        })

    # -----------------------------
    # 4) Optional: auto-fill comment textareas like the example
    # -----------------------------
    # If both NSFW and minors true, mirror example comment "nsfw and minor"; if PII true, comment "PII"
    nsfw_yes = any(p.get("from_name") == "val_nudity_video" and "Yes" in p["value"]["choices"] for p in predictions if p["type"] == "choices")
    minors_yes = any(p.get("from_name") == "val_minors_video" and "Yes" in p["value"]["choices"] for p in predictions if p["type"] == "choices")
    pii_yes = any(p.get("from_name") == "val_pii_audio" and "Yes" in p["value"]["choices"] for p in predictions if p["type"] == "choices")

    if nsfw_yes or minors_yes:
        txt = "nsfw and minor" if (nsfw_yes and minors_yes) else ("nsfw" if nsfw_yes else "minor")
        predictions.append({
            "id": _mk_id("comment_video"),
            "type": "textarea",
            "value": {"text": [txt]},
            "model_version": "auto_preannotator_v1",
            "from_name": "compliance_video_comment",
            "to_name": "md_home_id"
        })

    if pii_yes:
        predictions.append({
            "id": _mk_id("comment_audio"),
            "type": "textarea",
            "value": {"text": ["PII"]},
            "model_version": "auto_preannotator_v1",
            "from_name": "compliance_audio_comment",
            "to_name": "md_home_id"
        })

    # -----------------------------
    # 4) LIGHTING PREDICTIONS
    # -----------------------------
    # Extract lighting predictions from all views
    lighting_predictions = []
    dominant_lighting = "Normal"
    lighting_confidence = 0.5
    
    for view_name, view_result in view_results.items():
        if view_result.get('success', True):
            lighting_result = view_result.get('lighting')
            if isinstance(lighting_result, tuple) and len(lighting_result) == 2:
                try:
                    per_second_path, events_path = lighting_result
                    
                    # Read lighting events JSON
                    import json
                    with open(events_path, 'r') as f:
                        lighting_data = json.load(f)
                    
                    events = lighting_data.get('events', [])
                    if events:
                        # Find the most significant lighting event (longest duration)
                        significant_event = max(events, key=lambda x: x.get('duration', 0))
                        lighting_label = significant_event.get('label', 'Normal')
                        lighting_predictions.append(lighting_label)
                        
                        # Calculate confidence based on duration and consistency
                        total_duration = sum(e.get('duration', 0) for e in events)
                        if total_duration > 0:
                            confidence = significant_event.get('duration', 0) / total_duration
                            lighting_confidence = max(lighting_confidence, confidence * 0.9)
                        
                        # Update dominant lighting if higher confidence
                        if confidence * 0.9 > lighting_confidence:
                            dominant_lighting = lighting_label
                            
                except Exception as e:
                    logger.warning(f"Failed to process lighting results for {view_name}: {e}")
    
    # If we have lighting predictions, create consolidated lighting choice
    if lighting_predictions:
        from collections import Counter
        lighting_counts = Counter(lighting_predictions)
        dominant_lighting = lighting_counts.most_common(1)[0][0]
        
        # Create lighting prediction entry
        predictions.append({
            "id": _mk_id("lighting"),
            "type": "choices",
            "value": {
                "choices": [dominant_lighting]
            },
            "from_name": "lighting",
            "to_name": "video_top"  # or whichever view you prefer
        })
    
    # -----------------------------
    # 5) SIGNAL QUALITY PREDICTIONS  
    # -----------------------------
    signal_issues_detected = False
    signal_segments = []
    signal_issue_types = []

    for view_name, view_result in view_results.items():
        if view_result.get('success', True):
            signal_quality = view_result.get('signal_quality', {})
            
            blur_segments = signal_quality.get('blur_segments', [])
            black_segments = signal_quality.get('black_segments', [])
            
            # Collect all signal quality issues
            for segment in blur_segments:
                signal_segments.append({
                    'type': 'blur',
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'view': view_name
                })
                signal_issues_detected = True
                if 'blur' not in signal_issue_types:
                    signal_issue_types.append('blur')
            
            for segment in black_segments:
                signal_segments.append({
                    'type': 'black_screen', 
                    'start_time': segment['start_time'],
                    'end_time': segment['end_time'],
                    'view': view_name
                })
                signal_issues_detected = True
                if 'black_screen' not in signal_issue_types:
                    signal_issue_types.append('black_screen')

    # Create signal quality prediction
    if signal_issues_detected:
        # Find the earliest signal issue for time fields
        earliest_issue = min(signal_segments, key=lambda x: x['start_time'])
        latest_issue = max(signal_segments, key=lambda x: x['end_time'])
        
        start_minutes = int(earliest_issue['start_time'] // 60)
        start_seconds = int(earliest_issue['start_time'] % 60)
        end_minutes = int(latest_issue['end_time'] // 60) 
        end_seconds = int(latest_issue['end_time'] % 60)
        
        # Signal detected - YES choice
        predictions.append({
            "id": _mk_id("signal_detected"),
            "type": "choices", 
            "value": {
                "choices": ["Yes"]
            },
            "from_name": "val_signal",
            "to_name": "video_left"  # Default view
        })
        
        # Start time fields
        predictions.append({
            "id": _mk_id("signal_start_min"),
            "type": "taxonomy",
            "value": {
                "taxonomy": [[str(start_minutes).zfill(2)]]
            },
            "from_name": "signal_start_minute", 
            "to_name": "video_left"
        })
        
        predictions.append({
            "id": _mk_id("signal_start_sec"),
            "type": "taxonomy",
            "value": {
                "taxonomy": [[str(start_seconds).zfill(2)]]
            },
            "from_name": "signal_start_second",
            "to_name": "video_left"
        })
        
        # End time fields
        predictions.append({
            "id": _mk_id("signal_end_min"),
            "type": "taxonomy", 
            "value": {
                "taxonomy": [[str(end_minutes).zfill(2)]]
            },
            "from_name": "signal_end_minute",
            "to_name": "video_left" 
        })
        
        predictions.append({
            "id": _mk_id("signal_end_sec"),
            "type": "taxonomy",
            "value": {
                "taxonomy": [[str(end_seconds).zfill(2)]]
            },
            "from_name": "signal_end_second",
            "to_name": "video_left"
        })

    else:
        # No signal issues detected - NO choice
        predictions.append({
            "id": _mk_id("signal_no_issues"),
            "type": "choices",
            "value": {
                "choices": ["No"]  
            },
            "from_name": "val_signal",
            "to_name": "video_left"
        })

    # -----------------------------
    # 6) SENSITIVE INFORMATION PREDICTIONS  
    # -----------------------------
    sensitive_detected = False
    sensitive_segments = []
    sensitive_topics = set()

    # Process sensitive information from audio processing results
    # Sensitive information is processed per-view like other models
    sensitive_results = None
    for view_name, view_result in view_results.items():
        if view_result.get('success', True):
            view_sensitive = view_result.get('sensitive')
            if view_sensitive:
                sensitive_results = view_sensitive
                break  # Use first available sensitive results (audio is shared across views)

    if sensitive_results:
        
        # Handle list format from audio processing
        if isinstance(sensitive_results, list) and len(sensitive_results) > 0:
            for sensitive_result in sensitive_results:
                # Check if sensitive content was detected
                if sensitive_result.get('summary', {}).get('has_sensitive_content', False):
                    sensitive_detected = True
                    
                    # Extract topics from summary
                    summary_topics = sensitive_result.get('summary', {}).get('sensitive_topics', [])
                    sensitive_topics.update(summary_topics)
                    
                    # Extract individual detections for spectrogram regions
                    sensitive_detections = sensitive_result.get('sensitive_detections', [])
                    for detection in sensitive_detections:
                        sensitive_segments.append({
                            'start_time': detection.get('start_time', 0),
                            'end_time': detection.get('end_time', 0),
                            'topic': detection.get('topic', 'unknown'),
                            'confidence': detection.get('confidence', 0.7),
                            'transcript': detection.get('transcript_segment', '')
                        })

    # Create sensitive information predictions
    if sensitive_detected:
        # Audio sensitive - YES choice
        predictions.append({
            "id": _mk_id("sensitive_audio_yes"),
            "type": "choices",
            "value": {
                "choices": ["Yes"]
            },
            "from_name": "val_sensitive_audio",
            "to_name": "audio_main"
        })
        
        # Video sensitive - NO choice (detection is audio-only)
        predictions.append({
            "id": _mk_id("sensitive_video_no"),
            "type": "choices",
            "value": {
                "choices": ["No"]
            },
            "from_name": "val_sensitive_video", 
            "to_name": "video_left"
        })
        
        # Create spectrogram region labels for each detection
        for i, segment in enumerate(sensitive_segments):
            predictions.append({
                "id": _mk_id(f"sensitive_region_{i}"),
                "type": "labels",
                "value": {
                    "start": segment['start_time'],
                    "end": segment['end_time'],
                    "labels": ["Sensitive Topic Issue"],
                    "channel": 0
                },
                "from_name": "audio_labels_sensitive",
                "to_name": "audio_main"
            })

    else:
        # No sensitive content detected - NO choice for audio
        predictions.append({
            "id": _mk_id("sensitive_audio_no"),
            "type": "choices",
            "value": {
                "choices": ["No"]
            },
            "from_name": "val_sensitive_audio",
            "to_name": "audio_main"
        })
        
        # Video sensitive - NO choice
        predictions.append({
            "id": _mk_id("sensitive_video_no_clean"),
            "type": "choices",
            "value": {
                "choices": ["No"]
            },
            "from_name": "val_sensitive_video",
            "to_name": "video_left"
        })
    # -----------------------------
    # 7) PEOPLE COUNT (YOLO-based)
    # -----------------------------
    people_detected = False
    total_unique_people = 0
    max_simultaneous = 0
    occupancy_timeline = []
    
    # Process YOLO results from any view (typically "erp")
    for view_name, view_result in view_results.items():
        if not view_result.get("success", True):
            continue
        
        yolo_result = view_result.get("yolo", {})
        if not yolo_result or yolo_result.get('__error__'):
            continue
        
        # Extract people count data from JSON files (same logic as extract_yolo_people_data)
        people_analytics = {}
        if 'people_count_json' in yolo_result and yolo_result['people_count_json']:
            try:
                import json
                with open(yolo_result['people_count_json'], 'r') as f:
                    people_count_data = json.load(f)
                people_analytics = {
                    'count_summary': people_count_data.get('summary', {}),
                    'timeline_bins': people_count_data.get('timeline_bins', []),
                    'bin_size_sec': people_count_data.get('summary', {}).get('bin_size_sec', 1.0)
                }
            except Exception as e:
                logger.warning(f"Failed to load people_count_json for {view_name}: {e}")
                continue
        if people_analytics and people_analytics.get('count_summary'):
            people_detected = True
            summary = people_analytics['count_summary']
            total_unique_people = max(total_unique_people, summary.get('unique_people', 0))
            max_simultaneous = max(max_simultaneous, summary.get('max_of_max', 0))
            
            # Extract timeline for visualization (first 60 seconds)
            timeline_bins = people_analytics.get('timeline_bins', [])
            for bin_data in timeline_bins[:60]:
                occupancy_timeline.append({
                    'start': bin_data.get('start', 0),
                    'end': bin_data.get('end', 1),
                    'avg_people': bin_data.get('avg_persons', 0),
                    'max_people': bin_data.get('max_persons', 0)
                })
    
    # Create people count predictions if people detected
    if people_detected and total_unique_people > 0:
        # Add people count summary as metadata prediction
        predictions.append({
            "id": _mk_id("people_count_summary"),
            "type": "people_analytics",
            "value": {
                "unique_people": total_unique_people,
                "max_simultaneous": max_simultaneous,
                "processing_view": "erp",
                "occupancy_timeline": occupancy_timeline[:20],  # Limit for Label Studio
                "detection_approach": "erp_single_view_360"
            },
            "model_version": "yolo_people_counter_v1",
            "from_name": "people_counter",
            "to_name": "video_left"
        })
        
        # Add occupancy timeline as separate prediction for visualization
        if occupancy_timeline:
            predictions.append({
                "id": _mk_id("occupancy_timeline"),
                "type": "occupancy_data",
                "value": {
                    "timeline": occupancy_timeline[:30],  # First 30 seconds for UI
                    "bin_size_sec": 1.0,
                    "total_bins": len(occupancy_timeline)
                },
                "model_version": "yolo_occupancy_tracker_v1", 
                "from_name": "occupancy_timeline",
                "to_name": "video_left"
            })
    
    # -----------------------------
    # 8) ABSENT PEOPLE DETECTION
    # -----------------------------
    absent_segments = []
    video_duration = 60.0  # Default fallback duration
    
    # Extract video duration and analyze YOLO detections across all views
    all_person_timestamps = set()
    
    motion_result = view_results['front'].get("motion", {})
    if motion_result and motion_result.get("video_duration_seconds"):
        video_duration = motion_result["video_duration_seconds"]
    logger.info(f"Video duration:{video_duration}")
    for view_name, view_result in view_results.items():
        if not view_result.get("success", True):
            continue
        
        # Try to get video duration from motion detection or other sources
        
            
        yolo_result = view_result.get("yolo", {})
        if not yolo_result:
            continue
            
        # Parse YOLO events file to get person detection timestamps
        events_file = yolo_result.get("events_jsonl")
        if events_file:
            try:
                import json
                with open(events_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith('{"_meta"'):
                            continue
                        
                        try:
                            event = json.loads(line)
                            if event.get("cls") == "person":
                                timestamp = event.get("t", 0)
                                all_person_timestamps.add(timestamp)
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                logger.warning(f"Failed to parse YOLO events for {view_name}: {e}")
                continue
    
    # Find gaps in person detection (absent segments)
    if all_person_timestamps:
        sorted_timestamps = sorted(all_person_timestamps)
        
        # Check for gaps in person detection (segments where no people detected)
        gap_threshold = 10.0  # Consider 10+ second gaps as "absent" periods
        
        # Check beginning of video
        if sorted_timestamps[0] > gap_threshold:
            absent_segments.append((0.0, sorted_timestamps[0]))
            
        # Check gaps between detections
        for i in range(len(sorted_timestamps) - 1):
            gap_start = sorted_timestamps[i]
            gap_end = sorted_timestamps[i + 1]
            gap_duration = gap_end - gap_start
            
            if gap_duration > gap_threshold:
                absent_segments.append((gap_start, gap_end))
        
        # Check end of video
        if video_duration - sorted_timestamps[-1] > gap_threshold:
            absent_segments.append((sorted_timestamps[-1], video_duration))
    else:
        # No people detected in entire video
        absent_segments.append((0.0, video_duration))
    
    # Create absent people predictions if absent segments found
    if absent_segments:
        # Sort segments and use the first/longest significant segment
        significant_segments = [(s, e) for s, e in absent_segments if e - s >= 1.0]  # At least 1 second
        
        if significant_segments:
            significant_segments.sort(key=lambda x: x[1] - x[0], reverse=True)  # Sort by duration
            start_time, end_time = significant_segments[0]  # Use longest segment
            
            start_min = int(start_time // 60)
            start_sec = int(start_time % 60)
            end_min = int(end_time // 60) 
            end_sec = int(end_time % 60)
            
            # Add absent people detection - YES choice
            predictions.append({
                "id": _mk_id("absent_yes"),
                "type": "choices",
                "value": {
                    "choices": ["Yes"]
                },
                "model_version": "auto_preannotator_v1",
                "from_name": "val_absent",
                "to_name": "video_left"
            })
            
            # Add start time fields
            predictions.extend([
                {
                    "id": _mk_id("absent_start_min"),
                    "type": "taxonomy",
                    "value": {
                        "taxonomy": [[str(start_min).zfill(2)]]
                    },
                    "model_version": "auto_preannotator_v1",
                    "from_name": "absent_start_minute",
                    "to_name": "video_left"
                },
                {
                    "id": _mk_id("absent_start_sec"),
                    "type": "taxonomy",
                    "value": {
                        "taxonomy": [[str(start_sec).zfill(2)]]
                    },
                    "model_version": "auto_preannotator_v1",
                    "from_name": "absent_start_second",
                    "to_name": "video_left"
                },
                {
                    "id": _mk_id("absent_end_min"),
                    "type": "taxonomy",
                    "value": {
                        "taxonomy": [[str(end_min).zfill(2)]]
                    },
                    "model_version": "auto_preannotator_v1",
                    "from_name": "absent_end_minute",
                    "to_name": "video_left"
                },
                {
                    "id": _mk_id("absent_end_sec"),
                    "type": "taxonomy",
                    "value": {
                        "taxonomy": [[str(end_sec).zfill(2)]]
                    },
                    "model_version": "auto_preannotator_v1",
                    "from_name": "absent_end_second",
                    "to_name": "video_left"
                }
            ])
    else:
        # No absent segments found - people detected throughout
        predictions.append({
            "id": _mk_id("absent_no"),
            "type": "choices",
            "value": {
                "choices": ["No"]
            },
            "model_version": "auto_preannotator_v1",
            "from_name": "val_absent",
            "to_name": "video_left"
        })
    
    # -----------------------------
    # 4) CLAP DETECTION (New Addition)
    # -----------------------------
    
    # Extract detection flags - prioritize passed parameter, fallback to view_results
    # if detection_flags is None:
    detection_flags = view_results['front'].get("detection_flags", {})
    # for view_name, view_result in view_results.items():
    #     if view_result.get("success", True) and view_result.get("detection_flags"):
    #         detection_flags = view_result["detection_flags"]
    #         break
    logger.info(f"Detection_flags:{detection_flags}")
    if detection_flags:
        # Determine if this shard is first or last
        is_first_shard = detection_flags.get("is_first_shard", False)
        is_last_shard = detection_flags.get("is_last_shard", False)
        
        # Get clap detection results from the detection flags or view results
        clap_detected_in_first = detection_flags.get("clap_detected_in_first_shard", False) and is_first_shard
        clap_detected_in_last = detection_flags.get("clap_detected_in_last_shard", False) and is_last_shard
        
        # Alternative: Extract clap detection from actual clap results
        # clap_detected_current_shard = False
        # for view_name, view_result in view_results.items():
        #     if view_result.get("success", True):
        #         clap_result = view_result.get("clap", {})
        #         if clap_result.get("success") and clap_result.get("clap_count", 0) > 0:
        #             clap_detected_current_shard = True
        #             break
        
        # 4.1) val_first_video - Is this the first shard of the session?
        predictions.append({
            "id": _mk_id("first_video"),
            "type": "choices",
            "value": {
                "choices": ["Yes" if is_first_shard else "No"]
            },
            "model_version": "auto_preannotator_v1",
            "from_name": "val_first_video",
            "to_name": "video_left"
        })
        
        # 4.2) val_clap_first_video - Was clap detected in first shard?
        # Only add this prediction if this IS the first shard
        if is_first_shard:
            predictions.append({
                "id": _mk_id("clap_first"),
                "type": "choices", 
                "value": {
                    "choices": ["No" if clap_detected_in_first else "Yes"]
                },
                "model_version": "auto_preannotator_v1",
                "from_name": "val_clap_first_video",
                "to_name": "video_left"
            })
        
        # 4.3) val_last_video - Is this the last shard of the session?
        predictions.append({
            "id": _mk_id("last_video"),
            "type": "choices",
            "value": {
                "choices": ["Yes" if is_last_shard else "No"]
            },
            "model_version": "auto_preannotator_v1", 
            "from_name": "val_last_video",
            "to_name": "video_left"
        })
        
        # 4.4) val_clap_last_video - Was clap detected in last shard?
        # Only add this prediction if this IS the last shard
        if is_last_shard:
            predictions.append({
                "id": _mk_id("clap_last"),
                "type": "choices",
                "value": {
                    "choices": ["No" if clap_detected_in_last else "Yes"]
                },
                "model_version": "auto_preannotator_v1",
                "from_name": "val_clap_last_video", 
                "to_name": "video_left"
            })
    
    # If no detection flags available, add default "No" values
    else:
        # Default predictions when detection_flags not available
        predictions.extend([
            {
                "id": _mk_id("first_video_default"),
                "type": "choices",
                "value": {"choices": ["No"]},
                "model_version": "auto_preannotator_v1",
                "from_name": "val_first_video",
                "to_name": "video_left"
            },
            {
                "id": _mk_id("last_video_default"), 
                "type": "choices",
                "value": {"choices": ["No"]},
                "model_version": "auto_preannotator_v1",
                "from_name": "val_last_video",
                "to_name": "video_left"
            }
        ])

    # Done — PII/NSFW/Minors/Lighting/Sensitive/people counter/absent people predictions are returned.
    return predictions

def generate_multiview_consolidated_model_results_json(shard_output_dir, shard_number, view_results, total_shards=None, video_name=None):
    """
    Generate consolidated JSON file with all views as top-level keys.
    
    Args:
        shard_output_dir: Directory to save the consolidated JSON
        shard_number: Shard number for filename
        view_results: Dict of {view_name: view_result} for all views
        total_shards: Total number of shards
        video_name: Name of the video being processed
        
    Returns:
        Path to the generated consolidated JSON file
    """
    # Extract detection flags from any successful view
    detection_flags = {}
    for view_result in view_results.values():
        if view_result.get('success', True) and view_result.get('detection_flags'):
            detection_flags = view_result['detection_flags']
            break
    
    # If no detection flags found, create default
    if not detection_flags:
        detection_flags = {
            "is_first_shard": False,
            "is_last_shard": False,
            "is_intro_statement_there": False,
            "intro_transcript": "",
            "shard_index": shard_number - 1,
            "shard_type": "middle",
            "is_first_shard_processed": False,
            "is_last_shard_processed": False,
            "total_views_processed": len(view_results),
            "processing_type": "multi_view_equal_processing"
        }
    
    consolidated_data = {
        "shard_info": {
            "shard_number": shard_number,
            "shard_id": shard_number,
            "total_shards": total_shards,
            "video_name": video_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "processing_type": "multi_view_unwarped_equal",
            "total_views_processed": len(view_results),
            "successful_views": len([v for v in view_results.values() if v.get('success', True)])
        },
        "detection_flags": detection_flags,
        "view_results": {}
    }
    
    # Add all view results
    for view_name, view_result in view_results.items():
        consolidated_data["view_results"][view_name] = {
            "success": view_result.get('success', True),
            "audio": view_result.get('audio', {}),
            "yolo": extract_yolo_people_data(view_result.get('yolo', {})),
            "scene": view_result.get('scene', {}),
            "nsfw": view_result.get('nsfw', {}),
            "motion": view_result.get('motion', {}),
            "face": view_result.get('face', {}),
            "clap": view_result.get('clap', {}),
            "sensitive": view_result.get('sensitive', {}),
            "signal_quality": view_result.get('signal_quality', {}),
            "flagged_segments": view_result.get('flagged_segments', [])
        }
        # Process lighting results if available
        lighting_result = view_result.get('lighting')
        if isinstance(lighting_result, tuple) and len(lighting_result) == 2:
            try:
                per_second_path, events_path = lighting_result
                consolidated_data["view_results"][view_name]["lighting"] = {
                    "per_second_file": per_second_path,
                    "events_file": events_path
                }
                
                # Extract dominant lighting for quick access
                import json
                with open(events_path, 'r') as f:
                    lighting_data = json.load(f)
                
                events = lighting_data.get('events', [])
                if events:
                    significant_event = max(events, key=lambda x: x.get('duration', 0))
                    consolidated_data["view_results"][view_name]["lighting"]["dominant_lighting"] = significant_event.get('label', 'Normal')
                    consolidated_data["view_results"][view_name]["lighting"]["confidence"] = min(significant_event.get('duration', 0) / 60.0, 1.0) * 0.9
                else:
                    consolidated_data["view_results"][view_name]["lighting"]["dominant_lighting"] = "Normal"
                    consolidated_data["view_results"][view_name]["lighting"]["confidence"] = 0.5
                    
            except Exception as e:
                logger.warning(f"Failed to process lighting results for {view_name}: {e}")
                consolidated_data["view_results"][view_name]["lighting"] = {"error": str(e)}
    
    # Generate video-level people summary from consolidated data (after extract_yolo_people_data enhancement)
    video_people_summary = generate_video_people_summary(consolidated_data)
    consolidated_data["people_summary"] = video_people_summary
    
    # Add sensitive analysis results from audio processing (shared across views)
    sensitive_results = None
    for view_name, view_result in view_results.items():
        if view_result.get('success', True):
            view_sensitive = view_result.get('sensitive')
            if view_sensitive:
                sensitive_results = view_sensitive
                break  # Use first available sensitive results (audio is shared across views)
    
    if sensitive_results:
        consolidated_data["sensitive"] = sensitive_results
    # Save consolidated JSON
    consolidated_json_path = os.path.join(shard_output_dir, f"shard_{shard_number}_consolidated_model_results.json")
    with open(consolidated_json_path, 'w') as f:
        import json as json_module  # Avoid any potential variable shadowing
        json_module.dump(consolidated_data, f, indent=2)
    
    logger.info(f"💾 Saved multi-view consolidated results: {consolidated_json_path}")
    return consolidated_json_path

def process_video_level_domain_and_update_tasks(video_output_dir: str, video_name: str) -> Dict:
    """
    Process video-level domain classification using RAG-based approach and update all Label Studio tasks.
    
    This function runs AFTER all shards have been processed and their Label Studio 
    tasks have been generated. It performs RAG-based domain classification using
    the domain_classification_simple.py approach and updates all tasks with the 
    video-level domain information.
    
    Args:
        video_output_dir: Base output directory containing all shards for this video
        video_name: Name of the video being processed
        
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"🎯 Starting RAG-based domain classification for: {video_name}")
        
        # Step 1: Perform RAG-based domain classification
        logger.info(f"📊 Step 1: Running RAG-based domain classification...")
        
        # Get Groq API key
        groq_api_key = os.getenv("GROQ_API_KEY", "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")
        if not groq_api_key or "your_groq_api_key_here" in groq_api_key:
            logger.error("❌ GROQ_API_KEY not set. Please set it as an environment variable.")
            return {"success": False, "error": "GROQ_API_KEY not configured"}
        
        # Initialize the RAG-based domain classifier
        classifier = DomainClassifier(groq_api_key)
        
        # Run classification on the video output directory
        classification_results = classifier.process_and_classify(video_output_dir)
        
        if "error" in classification_results:
            logger.error(f"❌ RAG-based domain classification failed: {classification_results['error']}")
            return {"success": False, "error": f"Domain classification failed: {classification_results['error']}"}
        
        # Extract results
        best_match = classification_results.get("best_match", {})
        video_domain = best_match.get("domain", "Unknown")
        confidence_score = best_match.get("score", 0)
        reasoning = best_match.get("reasoning", "No reasoning provided")
        
        logger.info(f"✅ Step 1 completed: Video domain='{video_domain}' (confidence: {confidence_score}/10)")
        logger.info(f"📝 Reasoning: {reasoning}")
        
        # Step 2: Save detailed results to JSON file
        logger.info(f"💾 Step 2: Saving detailed classification results...")
        results_file = os.path.join(video_output_dir, "rag_domain_classification_results.json")
        classifier.save_results(classification_results, results_file)
        
        # Step 3: Update all Label Studio tasks with video-level domain
        logger.info(f"🔄 Step 3: Updating Label Studio tasks with video-level domain...")
        update_result_ref = update_labelstudio_tasks_with_video_domain_and_activity.remote(video_output_dir, video_domain, "RAG-classified activity", video_name)
        update_result = ray.get(update_result_ref)
        
        if not update_result.get("success", False):
            logger.error(f"❌ Label Studio task update failed: {update_result.get('error')}")
            return {"success": False, "error": f"Task update failed: {update_result.get('error')}"}
        
        logger.info(f"✅ Step 3 completed: Updated {update_result.get('updated_files', 0)} Label Studio tasks")
        
        # Combined result
        result = {
            "success": True,
            "video_name": video_name,
            "video_domain": video_domain,
            "video_activity": "RAG-classified activity",
            "confidence_score": confidence_score,
            "reasoning": reasoning,
            "updated_files": update_result.get("updated_files", 0),
            "results_file": results_file,
            "total_scenes_indexed": classification_results.get("total_scenes_indexed", 0),
            "processing_time": classification_results.get("processing_time_seconds", 0),
            "domain_classification": classification_results,
            "task_updates": update_result
        }
        
        logger.info(f"🎉 RAG-based domain processing completed for '{video_name}': domain='{video_domain}' (confidence: {confidence_score}/10)")
        return result
        
    except Exception as e:
        logger.error(f"❌ RAG-based domain processing failed for '{video_name}': {e}")
        return {"success": False, "error": str(e)}

def process_time_aligned_shard_multiview(
    shard_index,
    view_shard_paths,          # Dict: {view_name: shard_path}
    audio_shard_path,
    base_output_dir,
    view_azure_urls,           # Dict: {view_name: azure_url}
    audio_url,
    total_shard_count=None,
    video_name=None,
    multi_chunk_process=False,
    session_metadata=None,
    timer = None
):
    """
    Process one time-aligned shard across multiple views (4+).
    All views are processed equally and consolidated intelligently.
    
    Enhanced with first/last shard clap detection and comprehensive multi-view analysis.
    
    Args:
        shard_index: Index of current shard
        view_shard_paths: Dict of {view_name: shard_path} for all views
        audio_shard_path: Path to audio shard
        base_output_dir: Base output directory
        view_azure_urls: Dict of {view_name: azure_url} for all views
        audio_url: Audio Azure URL
        total_shard_count: Total shards for first/last detection
        video_name: Video name for metadata
    
    Returns:
        Dict with consolidated multi-view results
    """
    shard_output_dir = os.path.join(base_output_dir, f"shard_{shard_index+1}")
    shard_offset_sec = shard_index * 60
    os.makedirs(shard_output_dir, exist_ok=True)

    logger.info(f"🎬 Processing shard {shard_index+1} with {len(view_shard_paths)} views: {list(view_shard_paths.keys())}")

    # --- ENHANCED FIRST/LAST SHARD CLAP DETECTION (READ FROM CLAP OUTPUT JSON) ---
    # Always determine first/last shard status regardless of multi_chunk_process
    is_first_shard = (shard_index == 0)
    is_last_shard = (total_shard_count and shard_index == total_shard_count - 1)
    
    clap_detected_in_first_shard = False
    clap_detected_in_last_shard = False

    if is_first_shard or is_last_shard:
        shard_type = "first" if is_first_shard else "last"
        logger.info(f"🔍 CLAP DETECTION for {shard_type} shard {shard_index+1}")
        
        # Read clap detection results from clap_output JSON files
        clap_detected = read_clap_detection_from_json(shard_output_dir, shard_type)
        
        if is_first_shard:
            clap_detected_in_first_shard = clap_detected
            logger.info(f"🎬 FIRST SHARD clap detection: {'✅ DETECTED' if clap_detected else '❌ NOT DETECTED'}")
        else:
            clap_detected_in_last_shard = clap_detected
            logger.info(f"🎬 LAST SHARD clap detection: {'✅ DETECTED' if clap_detected else '❌ NOT DETECTED'}")
    else:
        logger.info(f"ℹ️ Shard {shard_index+1} is middle shard - no special clap processing")

    # --- PROCESS ALL VIEWS EQUALLY ---
    view_results = {}
    
    for view_name, view_shard_path in view_shard_paths.items():
        if not view_shard_path or not os.path.exists(view_shard_path):
            logger.warning(f"⚠️ View {view_name} shard not found: {view_shard_path}")
            continue
            
        logger.info(f"Processing {view_name} of shard {shard_index+1}")
        view_output_dir = os.path.join(shard_output_dir, f"{view_name}")
        os.makedirs(view_output_dir, exist_ok=True)
        
        try:
            # Process each view through the full pipeline
            view_result = process_single_shard_through_pipeline(
                view_shard_path, audio_shard_path, view_output_dir, shard_offset_sec, shard_index, total_shard_count, view_name, multi_chunk_process, timer = timer, video_name = video_name
            )
            view_results[view_name] = view_result
            logger.info(f"✅ Successfully processed {view_name} for shard {shard_index+1}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {view_name} for shard {shard_index+1}: {e}")
            view_results[view_name] = {"error": str(e), "success": False}

    if not view_results:
        raise RuntimeError(f"No views successfully processed for shard {shard_index+1}")

    logger.info(f"✅ Successfully processed {len(view_results)} views for shard {shard_index+1}")

    # =============================================================================
    # SCENE DETECTION OUTPUTS READY FOR VIDEO-LEVEL DOMAIN DETECTION
    # =============================================================================
    logger.info(f"🎯 Scene detection outputs ready for shard {shard_index+1}, proceeding to clap detection...")

    # # --- RELOAD SCENE DATA WITH DOMAIN CLASSIFICATION ---
    # logger.info(f"🔄 Reloading scene data with domain classification for shard {shard_index+1}...")
    # for view_name, view_result in view_results.items():
    #     if view_result.get('success', True):
    #         # Find the scene output directory for this view
    #         view_scene_output_dir = os.path.join(shard_output_dir, view_name, "scene_output")
    #         if os.path.exists(view_scene_output_dir):
    #             # Find the scene detection JSON file
    #             scene_files = [f for f in os.listdir(view_scene_output_dir) if f.endswith('_scene_detection_results.json')]
    #             if scene_files:
    #                 scene_file_path = os.path.join(view_scene_output_dir, scene_files[0])
    #                 try:
    #                     # Reload the scene data with domain classification
    #                     with open(scene_file_path, 'r') as f:
    #                         updated_scene_data = json.load(f)
    #                     # Update the view_results with the fresh scene data
    #                     view_result['scene'] = updated_scene_data
    #                     logger.info(f"✅ Reloaded scene data with domain classification for {view_name}")
    #                 except Exception as e:
    #                     logger.warning(f"⚠️ Failed to reload scene data for {view_name}: {e}")

    # --- ENHANCED CLAP DETECTION ANALYSIS ACROSS ALL VIEWS ---
    is_intro_statement_there = False
    intro_transcript = ""
    
    if is_first_shard or is_last_shard:
        # Analyze clap detection results from all views
        clap_detections = []
        all_transcripts = []
        
        for view_name, view_result in view_results.items():
            if view_result.get('success', True):  # Only process successful views
                clap_results = view_result.get('clap', {})
                if clap_results.get('overall_success'):
                    clap_count = clap_results.get('detected_clap', {}).get('timestamp')
                    clap_detections.append({
                        'view': view_name,
                        'clap_count': clap_count,
                        'clap_detected': clap_timestamp is not None
                    })
                
                # Collect transcripts from all views
                audio_results = view_result.get('audio', [])
                if isinstance(audio_results, list) and audio_results:
                    audio_result = audio_results[0]
                    transcript = audio_result.get('transcript', '').strip()
                    if transcript:
                        all_transcripts.append({
                            'view': view_name,
                            'transcript': transcript
                        })
        
        # Aggregate clap detection across views
        total_claps = sum(detection['clap_count'] for detection in clap_detections)
        views_with_claps = [d['view'] for d in clap_detections if d['clap_detected']]
        clap_detected = total_claps > 0
        
        if is_first_shard:
            clap_detected_in_first_shard = clap_detected
            logger.info(f"🎬 FIRST SHARD multi-view clap detection:")
            logger.info(f"   Total claps across all views: {total_claps}")
            logger.info(f"   Views with claps: {views_with_claps}")
            logger.info(f"   Overall detection: {'✅ DETECTED' if clap_detected else '❌ NOT DETECTED'}")
            
            # --- MULTI-VIEW INTRO STATEMENT DETECTION ---
            logger.info(f"🎤 ANALYZING FIRST SHARD for intro statement across {len(all_transcripts)} views...")
            
            if all_transcripts:
                # Find the longest transcript (likely the best quality)
                best_transcript = max(all_transcripts, key=lambda x: len(x['transcript']))
                intro_transcript = best_transcript['transcript']
                
                # Intro detection logic
                intro_keywords = [
                    "i'm going to", "i will", "today we", "welcome", "hello", "hi there",
                    "let's", "we're going to", "this is", "in this", "i am going to",
                    "we will", "starting with", "first we", "beginning", "introduction"
                ]
                
                transcript_lower = intro_transcript.lower()
                has_intro_keywords = any(keyword in transcript_lower for keyword in intro_keywords)
                has_meaningful_content = len(intro_transcript.strip()) > 10
                
                is_intro_statement_there = has_intro_keywords or has_meaningful_content
                
                logger.info(f"🗣️ INTRO STATEMENT: {'✅ DETECTED' if is_intro_statement_there else '❌ NOT DETECTED'}")
                logger.info(f"📝 Best transcript from {best_transcript['view']}: '{intro_transcript}'")
                
                if has_intro_keywords:
                    matched_keywords = [kw for kw in intro_keywords if kw in transcript_lower]
                    logger.info(f"🔤 Intro keywords found: {matched_keywords}")
                    
                # Log all transcripts for analysis
                for transcript_data in all_transcripts:
                    logger.info(f"📝 {transcript_data['view']}: '{transcript_data['transcript']}'")
            else:
                logger.info(f"📝 No transcripts found across any views in first shard")
                
        else:  # is_last_shard
            clap_detected_in_last_shard = clap_detected
            logger.info(f"🎬 LAST SHARD multi-view clap detection:")
            logger.info(f"   Total claps across all views: {total_claps}")
            logger.info(f"   Views with claps: {views_with_claps}")
            logger.info(f"   Overall detection: {'✅ DETECTED' if clap_detected else '❌ NOT DETECTED'}")

    # --- CREATE ENHANCED DETECTION FLAGS ---
    detection_flags = {
        "is_first_shard": is_first_shard,
        "is_last_shard": is_last_shard,
        "is_intro_statement_there": is_intro_statement_there,
        "intro_transcript": intro_transcript,
        "shard_index": shard_index,
        "shard_type": "first" if is_first_shard else ("last" if is_last_shard else "middle"),
        "is_first_shard_processed": is_first_shard,
        "is_last_shard_processed": is_last_shard,
        "total_views_processed": len(view_results),
        "successful_views": len([v for v in view_results.values() if v.get('success', True)]),
        "processing_type": "multi_view_equal_processing",
        "clap_detected_in_first_shard": clap_detected_in_first_shard,
        "clap_detected_in_last_shard": clap_detected_in_last_shard,
    }
    
    # Add detection flags to all view results
    for view_name, view_result in view_results.items():
        if view_result.get('success', True):
            view_result['detection_flags'] = detection_flags

    # --- GENERATE MULTI-VIEW CONSOLIDATED MODEL RESULTS JSON ---
    logger.info(f"Generating multi-view consolidated model results JSON for shard {shard_index+1}")
    consolidated_json_path = generate_multiview_consolidated_model_results_json(
        shard_output_dir, shard_index+1, view_results, total_shard_count, video_name
    )

    # --- CONSOLIDATE ALL VIEWS ---
    logger.info(f"Consolidating results from {len(view_results)} views for shard {shard_index+1}")
    consolidated_results = consolidate_multiview_time_segment_results(
        view_results, shard_index, shard_offset_sec
    )

    # --- ASSIGN ALL 4 VIEWS TO LABEL STUDIO UI POSITIONS ---
    # Assign all views to specific positions: video_top, video_left, video_right, video_bottom
    assigned_views = assign_views_to_labelstudio_positions(view_results, view_azure_urls)
    
    logger.info(f"🎯 Assigned views for Label Studio 4-view display:")
    logger.info(f"   Top: {assigned_views.get('top', {}).get('view', 'None')}")
    logger.info(f"   Left: {assigned_views.get('left', {}).get('view', 'None')}")
    logger.info(f"   Right: {assigned_views.get('right', {}).get('view', 'None')}")
    logger.info(f"   Bottom: {assigned_views.get('bottom', {}).get('view', 'None')}")

    # --- GENERATE ENHANCED 4-VIEW LABEL STUDIO TASK ---
    task_json_path = generate_multiview_4view_labelstudio_task(
        shard_output_dir,
        assigned_views,
        consolidated_results,
        shard_index+1,
        shard_offset_sec,
        audio_url,
        total_shard_count,
        video_name,
        all_view_urls=view_azure_urls,  # Pass all view URLs for metadata
        session_metadata=session_metadata
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
        "view_results": view_results,
        "consolidated_results": consolidated_results,
        "detection_flags": detection_flags,
        "label_studio_task": task_json_path,
        "consolidated_model_results_json": consolidated_json_path,
        "total_views_processed": len(view_results),
        "successful_views": len([v for v in view_results.values() if v.get('success', True)]),
        "assigned_views": assigned_views
    }

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

    # --- ENHANCED FIRST/LAST SHARD CLAP DETECTION (READ FROM CLAP OUTPUT JSON) ---
    # Determine if this is first or last shard using array indexing logic
    is_first_shard = (shard_index == 0)  # array[0]
    is_last_shard = (total_shard_count and shard_index == total_shard_count - 1)  # array[-1]
    
    # Initialize clap detection flags
    clap_detected_in_first_shard = False
    clap_detected_in_last_shard = False

    if is_first_shard or is_last_shard:
        shard_type = "first" if is_first_shard else "last"
        logger.info(f"🔍 CLAP DETECTION for {shard_type} shard {shard_index+1}")
        
        # Read clap detection results from clap_output JSON files
        clap_detected = read_clap_detection_from_json(shard_output_dir, shard_type)
        
        if is_first_shard:
            clap_detected_in_first_shard = clap_detected
            logger.info(f"🎬 FIRST SHARD clap detection: {'✅ DETECTED' if clap_detected else '❌ NOT DETECTED'}")
        else:
            clap_detected_in_last_shard = clap_detected
            logger.info(f"🎬 LAST SHARD clap detection: {'✅ DETECTED' if clap_detected else '❌ NOT DETECTED'}")
    else:
        logger.info(f"ℹ️ Shard {shard_index+1} is middle shard - no special clap processing")

    # --- View 1 ---
    logger.info(f"Processing view 1 of shard {shard_index+1}")
    view1_output_dir = os.path.join(shard_output_dir, "view_1")
    view1_results = process_single_shard_through_pipeline(
        view1_shard_path, audio_shard_path, view1_output_dir, shard_offset_sec, shard_index, total_shard_count, "view_1", multi_chunk_process
    )

    # --- View 2 (optional) ---
    if view2_shard_path:
        logger.info(f"Processing view 2 of shard {shard_index+1}")
        view2_output_dir = os.path.join(shard_output_dir, "view_2")
        view2_results = process_single_shard_through_pipeline(
            view2_shard_path, audio_shard_path, view2_output_dir, shard_offset_sec, shard_index, total_shard_count, "view_2", multi_chunk_process
        )
    else:
        logger.info(f"No view 2 for shard {shard_index+1} — running single-view consolidation")
        view2_results = {}

    # --- EXTRACT CLAP DETECTION RESULTS FOR FIRST/LAST SHARDS ---
    # Initialize intro statement flag
    is_intro_statement_there = False
    intro_transcript = ""
    
    if is_first_shard:
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

    # --- CREATE CLAP DETECTION FLAGS ---
    detection_flags = {
        "is_first_shard": is_first_shard,
        "is_last_shard": is_last_shard,
        "is_intro_statement_there": is_intro_statement_there,
        "intro_transcript": intro_transcript,
        "shard_index": shard_index,
        "shard_type": "first" if is_first_shard else ("last" if is_last_shard else "middle"),
        "is_first_shard_processed": is_first_shard,
        "is_last_shard_processed": is_last_shard,
        "clap_detected_in_first_shard": clap_detected_in_first_shard,  # ✅ Clap result
        "clap_detected_in_last_shard": clap_detected_in_last_shard,    # ✅ Clap result
    }
    
    # Add flags to view1_results for inclusion in consolidated JSON
    view1_results['detection_flags'] = detection_flags
    if view2_results:
        view2_results['detection_flags'] = detection_flags

    # Domain detection is handled in process_time_aligned_shard_multiview() to avoid duplication
    logger.info(f"🎯 Proceeding to consolidation for shard {shard_index+1}...")

    # --- RELOAD SCENE DATA WITH DOMAIN CLASSIFICATION (if available) ---
    logger.info(f"🔄 Reloading scene data with domain classification for shard {shard_index+1}...")
    
    # Reload scene data for view1
    view1_scene_output_dir = os.path.join(shard_output_dir, "view_1", "scene_output")
    if os.path.exists(view1_scene_output_dir):
        scene_files = [f for f in os.listdir(view1_scene_output_dir) if f.endswith('_scene_detection_results.json')]
        if scene_files:
            scene_file_path = os.path.join(view1_scene_output_dir, scene_files[0])
            try:
                with open(scene_file_path, 'r') as f:
                    updated_scene_data = json.load(f)
                view1_results['scene'] = updated_scene_data
                logger.info(f"✅ Reloaded scene data with domain classification for view_1")
            except Exception as e:
                logger.warning(f"⚠️ Failed to reload scene data for view_1: {e}")
    
    # Reload scene data for view2 (if exists)
    if view2_results:
        view2_scene_output_dir = os.path.join(shard_output_dir, "view_2", "scene_output")
        if os.path.exists(view2_scene_output_dir):
            scene_files = [f for f in os.listdir(view2_scene_output_dir) if f.endswith('_scene_detection_results.json')]
            if scene_files:
                scene_file_path = os.path.join(view2_scene_output_dir, scene_files[0])
                try:
                    with open(scene_file_path, 'r') as f:
                        updated_scene_data = json.load(f)
                    view2_results['scene'] = updated_scene_data
                    logger.info(f"✅ Reloaded scene data with domain classification for view_2")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to reload scene data for view_2: {e}")

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
                                         output_dir, shard_offset_sec, shard_index=0, total_shard_count=None, view_name="view", multi_chunk_process = False, timer = None, video_name = None):
    """
    Run single shard through all 7 AI models
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Extract chunk ID for tracking
    chunk_id = extract_chunk_id_from_path(video_shard_path) or f"shard-{shard_index}-{view_name}"

    if video_name is None:
        video_name = extract_video_name_from_path(video_shard_path)
    base_timing_id = f"video.{video_name}.shard.{shard_index}.{view_name}"

    if view_name == "erp":
        #Yolo people counter only for ERP view
        logger.info(f"ERP view: only run yolo people counter")
        yolo_output_dir = os.path.join(output_dir, "yolo_output")
        update_tracking(chunk_id, "model_processing.yolo_detection.erp", "processing")
        try:
            if timer:
                with timer.time_operation(f"{base_timing_id}.yolo_people_detection"):
                    yolo_ref  = run_yolo_detection.remote(video_shard_path, yolo_output_dir)
                    yolo_res = ray.get(yolo_ref)
            else:
                yolo_ref  = run_yolo_detection.remote(video_shard_path, yolo_output_dir)
                yolo_res = ray.get(yolo_ref)

            results = {'yolo': yolo_res}
            update_tracking(chunk_id, "model_processing.yolo_detection.erp", "completed",
                           results_path=yolo_output_dir)
            logger.info(f"Yolo people counter results saved to yolo_output_dir: {yolo_output_dir}")
        except Exception as e:
            update_tracking(chunk_id, "model_processing.yolo_detection.erp", "error",
                           error_message=str(e))
            logger.error(f"YOLO detection failed: {e}")
        return results

    enable_timing = timer is not None

    # Define output directories for models that need them
    audio_output_dir = os.path.join(output_dir, "audio_output")
    scene_output_dir = os.path.join(output_dir, "scene_output")
    clap_output_dir = os.path.join(output_dir, "clap_output")
    nsfw_output_dir = os.path.join(output_dir, "nsfw_output")
    motion_output_dir = os.path.join(output_dir, "motion_output")
    face_output_dir = os.path.join(output_dir, "face_output")
    signal_quality_output_dir = os.path.join(output_dir, "signal_quality_output")
    lighting_output_dir = os.path.join(output_dir, "lighting_output")

    # Create output directories
    os.makedirs(nsfw_output_dir, exist_ok=True)
    os.makedirs(motion_output_dir, exist_ok=True)
    os.makedirs(face_output_dir, exist_ok=True)
    os.makedirs(lighting_output_dir, exist_ok=True)
    os.makedirs(signal_quality_output_dir, exist_ok=True)

    # Define the prompt for scene detection
    prompt_path = "config/cosmos_prompt.yaml"
    
    # Verify prompt file exists
    ensure_prompt_file_exists(prompt_path)
    
    # Determine if this is first or last shard for clap detection
    # Always determine first/last shard status regardless of multi_chunk_process
    is_first_shard = (shard_index == 0)
    is_last_shard = (total_shard_count and shard_index == total_shard_count - 1)
    
    # Start all model processing with tracking
    update_tracking(chunk_id, "model_processing.audio_diarization", "processing")
    audio_ref = process_audio_diarization.remote([audio_shard_path], audio_output_dir,enable_timing=enable_timing,
        timing_id=f"{base_timing_id}.audio_diarization")
    
    update_tracking(chunk_id, "model_processing.scene_detection.front", "processing")
    scene_ref = detect_scenes.remote(video_shard_path, prompt_path, scene_output_dir, enable_timing=enable_timing,
        timing_id=f"{base_timing_id}.scene_detection")
    
    update_tracking(chunk_id, "model_processing.nsfw_detection.front", "processing")
    nsfw_ref  = process_video_chunks_for_nsfw.remote([video_shard_path], confidence_threshold=0.5, chunk_duration_sec=60, enable_timing=enable_timing,
        timing_id=f"{base_timing_id}.nsfw_detection")
    
    update_tracking(chunk_id, "model_processing.motion_detection.front", "processing")
    motion_ref= compute_motion_energy.remote([video_shard_path], sensitivity_level="medium", save_detailed_data=False, enable_timing=enable_timing,
        timing_id=f"{base_timing_id}.motion_energy")
    
    update_tracking(chunk_id, "model_processing.face_detection.front", "processing")
    face_ref  = process_video_chunks_for_face_detection.remote([video_shard_path], config=None, frame_interval=30, save_frames=False, chunk_duration_sec=60, enable_timing=enable_timing,
        timing_id=f"{base_timing_id}.face_detection")
    
    # Only run clap detection for first and last shards
    clap_ref = None
    if is_first_shard or is_last_shard:
        shard_type = "first" if is_first_shard else "last" 
        logger.info(f"🔍 Running clap detection for {shard_type} shard {shard_index+1}")
        update_tracking(chunk_id, "model_processing.clap_detection", "processing")
        clap_ref = detect_claps_in_audio_video_pair.remote(audio_shard_path, video_shard_path, clap_output_dir, search_window_sec=30.0, enable_timing=enable_timing, timing_id=f"{base_timing_id}.clap_detection")
    else:
        logger.info(f"ℹ️ Skipping clap detection for middle shard {shard_index+1}")
        update_tracking(chunk_id, "model_processing.clap_detection", "skipped")
    
    update_tracking(chunk_id, "model_processing.signal_quality_check", "processing")
    signal_quality_ref = detect_blur_and_black_segments.remote(video_shard_path, enable_timing=enable_timing, timing_id=f"{base_timing_id}.signal_quality")

    # Testing lighting
    update_tracking(chunk_id, "model_processing.lighting_analysis", "processing")
    lighting_ref = lighting_by_second_task.remote(video_shard_path, output_dir=lighting_output_dir, enable_timing=enable_timing, timing_id=f"{base_timing_id}.lighting_analysis")

    scene_res = ray.get(scene_ref)

    # Get results, handling conditional clap detection
    if clap_ref is not None:
        (audio_res, nsfw_res, motion_res, face_res, clap_res, lighting_res, signal_quality_res) = ray.get(
            [audio_ref, nsfw_ref, motion_ref, face_ref, clap_ref, lighting_ref, signal_quality_ref]
        )
        # Update tracking for completed models
        update_tracking(chunk_id, "model_processing.audio_diarization", "completed", results_path=audio_output_dir)
        update_tracking(chunk_id, "model_processing.scene_detection.front", "completed", results_path=scene_output_dir)
        update_tracking(chunk_id, "model_processing.nsfw_detection.front", "completed", results_path=nsfw_output_dir)
        update_tracking(chunk_id, "model_processing.motion_detection.front", "completed", results_path=motion_output_dir)
        update_tracking(chunk_id, "model_processing.face_detection.front", "completed", results_path=face_output_dir)
        update_tracking(chunk_id, "model_processing.clap_detection", "completed", results_path=clap_output_dir)
        update_tracking(chunk_id, "model_processing.signal_quality_check", "completed", results_path=signal_quality_output_dir)
        update_tracking(chunk_id, "model_processing.lighting_analysis", "completed", results_path=lighting_output_dir)
    else:
        # Set empty clap result for middle shards
        (audio_res, nsfw_res, motion_res, face_res, lighting_res, signal_quality_res) = ray.get(
            [audio_ref, nsfw_ref, motion_ref, face_ref, lighting_ref, signal_quality_ref]
        )
        clap_res = {"success": True, "clap_count": 0, "clap_timestamps": [], "message": "Clap detection skipped for middle shard"}
        # Update tracking for completed models (except clap)
        update_tracking(chunk_id, "model_processing.audio_diarization", "completed", results_path=audio_output_dir)
        update_tracking(chunk_id, "model_processing.scene_detection.front", "completed", results_path=scene_output_dir)
        update_tracking(chunk_id, "model_processing.nsfw_detection.front", "completed", results_path=nsfw_output_dir)
        update_tracking(chunk_id, "model_processing.motion_detection.front", "completed", results_path=motion_output_dir)
        update_tracking(chunk_id, "model_processing.face_detection.front", "completed", results_path=face_output_dir)
        update_tracking(chunk_id, "model_processing.signal_quality_check", "completed", results_path=signal_quality_output_dir)
        update_tracking(chunk_id, "model_processing.lighting_analysis", "completed", results_path=lighting_output_dir)
    
    # Run sensitive information analysis after audio_diarization_pii completes
    sensitive_output_dir = os.path.join(output_dir, "sensitive_output")
    os.makedirs(sensitive_output_dir, exist_ok=True)
    update_tracking(chunk_id, "model_processing.sensitive_info_detection", "processing")
    sensitive_ref = process_audio_sensitive_info.remote([audio_shard_path], audio_output_dir, sensitive_output_dir)
    sensitive_res = ray.get(sensitive_ref)
    update_tracking(chunk_id, "model_processing.sensitive_info_detection", "completed", results_path=sensitive_output_dir)

    if timer and enable_timing:
        timing_results = {
            "audio_diarization": audio_res.get("timing"),
            "nsfw_detection": nsfw_res.get("timing"),
            "motion_energy": motion_res.get("timing"),
            "face_detection": face_res.get("timing"),
            "scene_detection": scene_res.get("timing"),
            "clap_detection": clap_res.get("timing") if clap_res and isinstance(clap_res, dict) else None,
            "signal_quality": signal_quality_res.get("timing") if isinstance(signal_quality_res, dict) else None,
            "lighting_analysis": lighting_res.get("timing") if isinstance(lighting_res, dict) and "timing" in lighting_res else None
            # Note: sensitive_info doesn't have timing parameters yet
        }

        # Add individual model timings to main timer
        for model_name, model_timing in timing_results.items():
            if model_timing:
                timer.timings[f"{base_timing_id}.{model_name}"] = {
                    "execution_time": model_timing["execution_time"],
                    "start_time": model_timing["start_time"],
                    "end_time": model_timing["end_time"],
                    "status": model_timing["status"],
                    "timing_id": model_timing["timing_id"]
                }
    # Store results from Ray tasks
    results = {
        'audio': audio_res,
        'scene': scene_res,
        'nsfw': nsfw_res,
        'motion': motion_res,
        'face': face_res,
        'clap': clap_res,
        'sensitive': sensitive_res,
        'lighting': lighting_res.get("result") if isinstance(lighting_res, dict) and "result" in lighting_res else lighting_res,
        'signal_quality': signal_quality_res
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
    
    # Save Sensitive Information Analysis results
    if sensitive_res and isinstance(sensitive_res, list) and len(sensitive_res) > 0:
        sensitive_file = os.path.join(sensitive_output_dir, f"{video_name}_sensitive_results.json")
        with open(sensitive_file, 'w') as f:
            json.dump(sensitive_res, f, indent=2)
        logger.info(f"Sensitive information analysis results saved to: {sensitive_file}")
    
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
                "successful_domain": total_shards * 2,  # Domain classification for both views
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
    azure_account_key: str = None,
    pipeline_config: dict = None
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
    # if azure_blob_client and azure_container and azure_output_prefix and azure_account_name and azure_account_key:
    #     logger.info("Uploading single-view video shards to Azure Blob Storage...")
    #     video_urls = generate_azure_shard_urls(
    #         azure_blob_client, azure_container, video_shards,
    #         f"{azure_output_prefix}/video_shards",
    #         azure_account_name, azure_account_key
    #     )
    #     audio_urls = generate_azure_shard_urls(
    #         azure_blob_client, azure_container, audio_shards,
    #         f"{azure_output_prefix}/audio_shards",
    #         azure_account_name, azure_account_key
    #     )
    # else:
    #     logger.warning("Azure client/prefix or account credentials missing — LS URLs will be local and likely won't stream.")
    #     video_urls, audio_urls = {}, {}
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

    # =================================================================
    # DYNAMIC ERP VIDEO AND AUDIO PROCESSING - AFTER LABEL STUDIO TASKS
    # =================================================================
    logger.info(f"🎬 Starting dynamic ERP video and audio processing for session: {video_name}")
    
    # Integrate ERP video and audio URLs with Label Studio tasks
    label_studio_tasks = integrate_erp_audio_with_labelstudio_tasks(
        output_dir=output_dir,
        session_id=video_name,
        blob_service_client=blob_client,
        container_name=container_name,
        blob_base_path=azure_output_prefix,
        account_key=account_key,
        labelstudio_tasks=label_studio_tasks
    )
    
    logger.info(f"🎬 Dynamic ERP and audio processing completed for session: {video_name}")
    # =================================================================

    # Generate final combined JSON with all shards
    generate_final_combined_model_results_json(output_dir, consolidated_json_paths)

    # --- Import all tasks to Label Studio (SAME function as dual-view) ---
    logger.info(f"🎯 Running video-level domain classification for video: {video_name}")
    domain_result = process_video_level_domain_and_update_tasks(output_dir, video_name)
    if domain_result.get("success", False):
        logger.info(f"✅ Video-level domain processing completed: {domain_result.get('video_domain', 'Unknown')}")
    else:
        logger.error(f"❌ Video-level domain processing failed: {domain_result.get('error', 'Unknown error')}")
    
    if azure_blob_client and azure_container:
        video_name = os.path.basename(output_dir)
        upload_config = pipeline_config.get('blob_upload', {})

        upload_result = upload_output_directory_with_sas_optimized(
            output_dir, blob_client, container_name,
            azure_output_prefix, video_name_for_upload,
            account_name, account_key,
            365,  # sas_expiry_days
            upload_config  # config dict with blob_upload settings
        )
        logger.info(f" Single-view upload result: {' Success' if upload_result['success'] else '❌ Failed'}")
        
        if upload_result['success']:
            # Extract URLs from upload result for single-view
            single_view_urls = upload_result['shard_urls']['single_view_urls']
            video_urls = single_view_urls['video_urls']
            audio_urls = single_view_urls['audio_urls']
            
            logger.info(f" Generated {len(video_urls)} video and {len(audio_urls)} audio URLs with SAS tokens")
            
            # CRITICAL: Update Label Studio tasks with real URLs
            logger.info(" Updating Label Studio tasks with uploaded URLs...")
            updated_count = update_labelstudio_tasks_with_new_urls(label_studio_tasks, upload_result['shard_urls'], "single_view")
            if updated_count != len(label_studio_tasks):
                logger.warning(f"⚠️ Only {updated_count}/{len(label_studio_tasks)} Label Studio tasks were updated with URLs")
            
        else:
            logger.error(f"Upload error: {upload_result.get('error', 'Unknown error')}")
            video_urls, audio_urls = {}, {}
    else:
        logger.warning(" Azure configuration missing - Label Studio tasks will have placeholder URLs")  
        video_urls, audio_urls = {}, {}
        import_consolidated_tasks_to_labelstudio(label_studio_tasks, pipeline_config, False)

    # Upload output directory to Azure blob storage
    # if azure_blob_client and azure_container:
    #     video_name = os.path.basename(output_dir)
    #     upload_result = upload_output_directory_to_blob(
    #         output_dir, azure_blob_client, azure_container,
    #         azure_output_prefix, video_name
    #     )
    #     logger.info(f"📤 Upload result: {'✅ Success' if upload_result['success'] else '❌ Failed'}")
    #     if not upload_result['success']:
    #         logger.error(f"Upload error: {upload_result.get('error', 'Unknown error')}")

    # --- Return consolidated summary (SAME function as dual-view) ---
    return create_consolidated_summary(label_studio_tasks, output_dir)
# ===============================================
# LABEL STUDIO ENHANCEMENT FUNCTIONS
# ===============================================

def merge_model_results_into_labelstudio_task(shard_output_dir: str, shard_number: int) -> bool:
    """
    Merges the shard_X_consolidated_model_results.json content into the 
    shard_X_labelstudio_task.json file under the predictions section.
    
    Args:
        shard_output_dir: Directory containing the shard files
        shard_number: Shard number (1-based)
        
    Returns:
        bool: Success status
    """
    try:
        # Construct file paths
        model_results_file = os.path.join(shard_output_dir, f"shard_{shard_number}_consolidated_model_results.json")
        labelstudio_task_file = os.path.join(shard_output_dir, f"shard_{shard_number}_labelstudio_task.json")
        
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
        
        # Add model results to the predictions section
        if "predictions" not in labelstudio_task:
            labelstudio_task["predictions"] = []
        
        # Get the first prediction entry (or create if none exists)
        if not labelstudio_task["predictions"]:
            labelstudio_task["predictions"].append({"result": []})
        
        # Add model results as a new item in the result array (keeping all existing items)
        labelstudio_task["predictions"][0]["result"].append({
            "id": "consolidated_model_results",
            "type": "model_results",
            "value": model_results,
            "model_version": "consolidated_v1.0",
            "from_name": "model_results",
            "to_name": "data"
        })
        
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


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("🛑 Shutdown signal received. Gracefully stopping...")




# @ray.remote
# def continuous_blob_polling_and_pipeline_task(azure_config_path: str = "blobfuse2_config.yaml",
#                                              pipeline_config_path: str = "config/pipeline_config.yaml"):
#     """
#     Continuous polling task that checks Azure blob storage for new videos every N minutes
#     and processes them through the pipeline, maintaining a checklist of progress.
#     """
#     global _shutdown_requested
    
#     try:
#         logger.info("🚀 Starting Continuous Blob Polling & Pipeline Task")
        
#         # Load configurations
#         pipeline_config = load_pipeline_config(pipeline_config_path)
#         azure_config = load_azure_config(azure_config_path)
#         polling_interval_minutes = pipeline_config.get('polling', {}).get('interval_minutes', 5)
        
#         logger.info(f"⏰ Polling interval: {polling_interval_minutes} minutes")
        
#         # Setup Azure client
#         blob_service_client = create_azure_blob_client(azure_config)
        
#         # Extract Azure storage config - handle both direct and nested structures
#         if 'azstorage' in azure_config:
#             # Nested structure from blobfuse2_config.yaml
#             az_config = azure_config['azstorage']
#         else:
#             # Direct structure
#             az_config = azure_config
            
#         container_name = az_config['container']
#         account_name = az_config['account-name']
#         account_key = az_config['account-key']
        
#         # Setup directories and files
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         local_download_dir = pipeline_config['local_storage']['temp_download_dir']
#         output_base_dir = os.path.join(current_dir, pipeline_config['local_storage']['output_base_dir'].lstrip('./'))
#         checklist_path = os.path.join(current_dir, pipeline_config['local_storage'].get('checklist_file', './video_checklist.json'))
#         blob_prefix = pipeline_config['azure_storage']['input_blob_prefix']
#         output_prefix = pipeline_config['azure_storage']['output_blob_prefix']
        
#         # Load checklist
#         checklist = load_video_checklist(checklist_path)
        
#         # Statistics
#         total_processed = 0
#         successful_processed = 0
#         failed_processed = 0
        
#         logger.info("✅ Continuous polling initialized successfully")
        
#         # Get detailed statistics
#         stats = get_checklist_statistics(checklist)
#         logger.info(f"📋 Checklist Statistics:")
#         logger.info(f"   Total videos: {stats['total_videos']}")
#         logger.info(f"   Completed: {stats['completed_videos']}")
#         logger.info(f"   Failed: {stats['failed_videos']}")
#         logger.info(f"   Pending: {stats['pending_videos']}")
#         logger.info(f"   Processing: {stats['processing_videos']}")
#         logger.info(f"   Discovered: {stats['discovered_videos']}")
#         if stats['status_breakdown']:
#             logger.info(f"   Status breakdown: {stats['status_breakdown']}")
        
#         # Store video blob info for retries
#         video_blob_info_cache = {}
        
#         # Main polling loop
#         while not _shutdown_requested:
#             try:
#                 logger.info(f"\n{'='*60}")
#                 logger.info(f"🔄 Polling cycle started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
#                 # 1. Poll for new videos
#                 new_videos = poll_azure_videos(blob_service_client, container_name, blob_prefix, checklist, pipeline_config)
                
#                 # Update cache with new video info
#                 for video_info in new_videos:
#                     video_blob_info_cache[video_info["video_name"]] = video_info
                
#                 # Save checklist after polling to ensure new videos are recorded
#                 if new_videos:
#                     save_video_checklist(checklist_path, checklist)
                
#                 # 2. Check for and reset stuck videos
#                 stuck_count = reset_stuck_videos(checklist, max_processing_hours=2)
#                 if stuck_count > 0:
#                     logger.info(f"🔄 Reset {stuck_count} stuck videos back to pending status")
#                     save_video_checklist(checklist_path, checklist)
                
#                 # 3. Get pending videos (new + previously failed + reset stuck videos)
#                 pending_videos = get_pending_videos(checklist)
                
#                 if pending_videos:
#                     logger.info(f"📋 Found {len(pending_videos)} videos to process")
                    
#                     # 4. Process pending videos one by one
#                     for video_name in pending_videos:
#                         if _shutdown_requested:
#                             break
                            
#                         video_info = checklist["videos"][video_name]
                        
#                         try:
#                             logger.info(f"\n🎬 Processing video: {video_name}")
#                             update_video_status(checklist, video_name, "processing", attempts=video_info.get("attempts", 0) + 1)
#                             save_video_checklist(checklist_path, checklist)
                            
#                             # Get video blob info from cache
#                             video_blob_info = video_blob_info_cache.get(video_name)
                            
#                             if not video_blob_info:
#                                 logger.warning(f"⚠️ No blob info cached for {video_name} - skipping")
#                                 continue
                            
#                             # Download video and audio
#                             start_time = time.time()
#                             download_result = download_video_audio_pair(
#                                 blob_service_client, container_name,
#                                 video_blob_info,
#                                 local_download_dir
#                             )
#                             video_path = download_result["video_path"]
#                             audio_path = download_result["audio_path"]
                            
#                             if not video_path or not audio_path:
#                                 raise Exception("Failed to download video/audio files")
                            
#                             # Setup output directory
#                             video_output_dir = os.path.join(output_base_dir, f"video_{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                            
#                             # Process through pipeline
#                             logger.info(f"🔄 Running pipeline for {video_name}")
#                             pipeline_results = pipeline_main(
#                                 input_video_path=video_path,
#                                 input_audio_path=audio_path,
#                                 output_dir=video_output_dir,
#                                 process_dual_views=None,  # Auto-detect
#                                 process_unwarped_views=True,
#                                 azure_blob_client=blob_service_client,
#                                 azure_container=container_name,
#                                 azure_output_prefix=f"{output_prefix}/{video_name}",
#                                 azure_account_name=account_name,
#                                 azure_account_key=account_key,
#                                 pipeline_config = pipeline_config
#                             )
                            
#                             processing_time = time.time() - start_time
                            
#                             # Check if pipeline actually succeeded
#                             if pipeline_results and pipeline_results.get("successful_shard_tasks", 0) > 0:
#                             # Update success status
#                                 update_video_status(checklist, video_name, "completed",
#                                                output_dir=video_output_dir,
#                                                processing_time=f"{processing_time:.2f}s",
#                                                completed_at=datetime.now().isoformat())
#                             else:
#                                 # Pipeline failed - mark as failed
#                                 error_msg = "Pipeline failed - no successful shard tasks"
#                                 update_video_status(checklist, video_name, "failed",
#                                                    error=error_msg,
#                                                    processing_time=f"{processing_time:.2f}s")
#                                 raise Exception(error_msg)
                            
#                             successful_processed += 1
#                             total_processed += 1
                            
#                             logger.info(f"✅ Successfully processed {video_name} in {processing_time:.2f}s")
                            
#                             # Cleanup downloaded files
#                             if pipeline_config['cleanup']['cleanup_files_after_each_video']:
#                                 try:
#                                     os.remove(video_path)
#                                     os.remove(audio_path)
#                                     logger.info(f"🗑️ Cleaned up downloaded files for {video_name}")
#                                 except Exception as e:
#                                     logger.warning(f"⚠️ Failed to cleanup files: {e}")
                            
#                         except Exception as e:
#                             error_msg = str(e)
                            
#                             # Check for CUDA out of memory errors
#                             if "CUDA_ERROR_OUT_OF_MEMORY" in error_msg or "out of memory" in error_msg.lower():
#                                 error_msg = f"CUDA out of memory error: {error_msg}"
#                                 logger.error(f"❌ CUDA memory error for {video_name}: {error_msg}")
#                             else:
#                                 logger.error(f"❌ Failed to process {video_name}: {error_msg}")
                            
#                             update_video_status(checklist, video_name, "failed", error=error_msg)
#                             failed_processed += 1
#                             total_processed += 1
                        
#                         # Save checklist after each video
#                         save_video_checklist(checklist_path, checklist)
                
#                 else:
#                     # Get current statistics for logging
#                     stats = get_checklist_statistics(checklist)
#                     logger.info("✅ No pending videos to process")
#                     logger.info(f"📊 Current status: {stats['completed_videos']} completed, {stats['failed_videos']} failed, {stats['processing_videos']} processing")
                
#                 # 5. Wait for next polling cycle
#                 if not _shutdown_requested:
#                     logger.info(f"⏰ Waiting {polling_interval_minutes} minutes until next poll...")
#                     for i in range(polling_interval_minutes * 60):  # Convert minutes to seconds
#                         if _shutdown_requested:
#                             break
#                         time.sleep(1)
                
#             except Exception as e:
#                 logger.error(f"❌ Error in polling cycle: {e}")
#                 time.sleep(30)  # Wait 30 seconds before retrying
        
#         # Final cleanup
#         if pipeline_config['cleanup']['cleanup_temp_dir'] and os.path.exists(local_download_dir):
#             try:
#                 shutil.rmtree(local_download_dir)
#                 logger.info(f"🗑️ Cleaned up temp directory: {local_download_dir}")
#             except Exception as e:
#                 logger.warning(f"⚠️ Failed to cleanup temp directory: {e}")
        
#         logger.info("🏁 Continuous polling stopped gracefully")
        
#         return {
#             "success": True,
#             "total_processed": total_processed,
#             "successful_processed": successful_processed,
#             "failed_processed": failed_processed,
#             "checklist_path": checklist_path,
#             "final_stats": {
#                 "total_videos": checklist["total_videos"],
#                 "completed_videos": checklist["completed_videos"],
#                 "failed_videos": checklist["failed_videos"]
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"❌ Continuous polling task failed: {e}")
#         return {"success": False, "error": str(e)}



if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ray Pipeline with Integrated Blob Polling")
    parser.add_argument("--mode", choices=["integrated", "standalone", "multi_chunks", "multi_sessions", "blob_polling"], default="blob_polling",
                       help="Run mode: 'integrated' for blob polling + pipeline, 'standalone' for direct pipeline, 'multi_chunks' for single session, 'multi_sessions' for multiple sessions, 'blob_polling' for continuous blob polling (default: blob_polling)")
    parser.add_argument("--azure-config", default="blobfuse2_config.yaml",
                       help="Azure configuration file path")
    parser.add_argument("--pipeline-config", default="config/pipeline_config.yaml",
                       help="Pipeline configuration file path")
    parser.add_argument("--blob-prefix", 
                       help="Blob prefix to search for videos (default: from pipeline_config.yaml)")
    parser.add_argument("--max-videos", type=int,
                       help="Maximum number of videos to process")
    parser.add_argument("--force-refresh", action="store_true",
                       help="Force refresh video list")
    parser.add_argument("--dual-views", action="store_true",
                       help="Enable dual view processing")
    parser.add_argument("--unwarped-views", action="store_true",
                       help="Enable unwarped view processing")
    parser.add_argument("--parallel-chunks", action="store_true", default=True,
                       help="Enable parallel chunk downloading using Ray (default: True)")
    parser.add_argument("--sequential-chunks", action="store_true", default=False,
                       help="Use sequential chunk downloading (overrides parallel-chunks)")
    parser.add_argument("--polling-interval", type=int,
                       help="Polling interval in minutes for blob_polling mode (default: from pipeline_config.yaml)")

    # Standalone mode arguments
    parser.add_argument("--input-video", 
                       help="Input video file path (for standalone mode)")
    parser.add_argument("--input-audio", 
                       help="Input audio file path (for standalone mode)")
    parser.add_argument("--output-dir",
                       help="Output directory (for standalone mode, default: from pipeline_config.yaml)")
    
    args = parser.parse_args()
    
    # Load pipeline config to get defaults
    pipeline_config = load_pipeline_config(args.pipeline_config)
    
    # Set defaults from config file if not provided via command line
    if args.blob_prefix is None:
        args.blob_prefix = pipeline_config['azure_storage']['input_blob_prefix']
    
    if args.polling_interval is None:
        args.polling_interval = pipeline_config['polling']['interval_minutes']
    
    if args.output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(current_dir, pipeline_config['local_storage']['output_base_dir'].lstrip('./'))
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
            logger.info("✅ Ray initialized successfully")

        if args.mode == "multi_chunks":
            

            logger.info("Processing multi chunk pipeline")
            pipeline_config = load_pipeline_config(args.pipeline_config)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_base_dir = os.path.join(current_dir, pipeline_config['local_storage']['output_base_dir'].lstrip('./'))
            local_download_dir = pipeline_config['local_storage']['temp_download_dir']
            # video_list_file = os.path.join(current_dir, pipeline_config['local_storage']['video_list_file'].lstrip('./'))
            
            # Get checklist file path
            # checklist_path = pipeline_config['local_storage'].get('checklist_file', './video_checklist.json')
            # checklist_path = os.path.join(current_dir, checklist_path.lstrip('./'))
            config = load_azure_config("blobfuse2_config.yaml")
            blob_client = create_azure_blob_client(config)
            
            # Test discovery
            if 'azstorage' in config:
                # Nested structure from blobfuse2_config.yaml
                az_config = azure_config['azstorage']
            else:
                # Direct structure
                az_config = config
                    
            container_name = az_config['container']
            account_name = az_config['account-name']
            account_key = az_config['account-key']
            input_prefix = pipeline_config['azure_storage']['input_blob_prefix']
            # input_prefix = "one-data-platform/11-515fb09c-f12f-48ee-91e7-b21c9f4ac5ab-watching-tv/"      # Update this
            # local_download_dir = "./mulmti_chunk_test_downloads"

            # print("📥 Testing Phase 2: Basic Download")
            # print("=" * 50)
            
            # result = download_ready_sessions(
            #     blob_service_client=blob_client,
            #     container_name=container_name,
            #     input_prefix=input_prefix,
            #     local_download_dir=local_download_dir,
            #     max_sessions=1  # Test with just 1 session
            # )

            # Determine if we should use parallel chunk downloads
            use_parallel_chunks = args.parallel_chunks and not args.sequential_chunks
            
            if use_parallel_chunks:
                print("📥 Testing Phase 2: PARALLEL Chunk Download with Ray")
                print("=" * 50)
            else:
                print("📥 Testing Phase 2: Sequential Chunk Download")
                print("=" * 50)
            
            result = download_ready_sessions(
                blob_service_client=blob_client,
                container_name=container_name,
                input_prefix=input_prefix,
                local_download_dir=local_download_dir,
                max_sessions=1,  # Test with just 1 session
                use_parallel_chunks=use_parallel_chunks
            )
            print(result)
            print(f"\n📊 DOWNLOAD RESULTS:")
            print(f"Success: {result['success']}")
            print(f"Sessions ready: {result['sessions_ready']}")
            print(f"Sessions downloaded: {result['sessions_downloaded']}")
            print(f"Success rate: {result['success_rate']:.1%}")
            print(f"Total time: {result['total_processing_time']:.1f}s")
            
            # Show details for each session
            for session_id, session_result in result.get("download_results", {}).items():
                print(f"\n📦 SESSION: {session_id}")
                if session_result["success"]:
                    download_stats = session_result["download_result"]["download_stats"]
                    print(f"   ✅ Success: {download_stats['successful_downloads']}/{download_stats['total_downloads']}")
                    print(f"   💾 Data: {download_stats['total_bytes'] / (1024 * 1024):.1f} MB")
                    print(f"   ⏱️ Time: {download_stats['download_time']:.1f}s")
                    print(f"   📈 Speed: {download_stats['mb_per_second']:.1f} MB/s")
                else:
                    print(f"   ❌ Failed: {session_result.get('error', 'Unknown error')}")
            
            # List downloaded files
            if os.path.exists(local_download_dir):
                files = os.listdir(local_download_dir)
                print(f"\n📁 Downloaded {len(files)} files to {local_download_dir}")
                for file in sorted(files):
                    file_path = os.path.join(local_download_dir, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"   📄 {file} ({size_mb:.1f} MB)")
            output_prefix = pipeline_config['azure_storage']['output_blob_prefix']
            pipeline_main_multichunks(result, output_base_dir, f"{output_prefix}", blob_client, container_name, account_name, account_key, local_download_dir)

        elif args.mode == "multi_sessions":
            # =================================================================
            # MULTI-SESSIONS MODE: Process multiple sessions sequentially
            # =================================================================
            logger.info("🚀 Starting MULTI-SESSIONS MODE: Sequential session processing")
            logger.info("=" * 80)
            
            pipeline_config = load_pipeline_config(args.pipeline_config)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_base_dir = os.path.join(current_dir, pipeline_config['local_storage']['output_base_dir'].lstrip('./'))
            
            # Load Azure config
            config = load_azure_config("blobfuse2_config.yaml")
            blob_client = create_azure_blob_client(config)
            
            if 'azstorage' in config:
                az_config = config['azstorage']
            else:
                az_config = config
            
            container_name = az_config['container']
            account_name = az_config['account-name']
            account_key = az_config['account-key']
            
            # Base prefix for discovering sessions
            base_prefix = args.blob_prefix  # e.g., "one-data-platform/"
            
            logger.info(f"🔍 Discovering sessions under: {base_prefix}")
            
            # Discover all session directories
            session_paths = discover_multiple_sessions(blob_client, container_name, base_prefix)
            
            if not session_paths:
                logger.warning(f"⚠️ No sessions found under prefix: {base_prefix}")
                exit(0)
            
            logger.info(f"📋 Found {len(session_paths)} sessions to process")
            
            # Process all sessions sequentially
            all_results = pipeline_main_multichunks_sequential_sessions(
                session_paths=session_paths,
                output_base_dir=output_base_dir,
                azure_config=az_config,
                pipeline_config=pipeline_config
            )
            
            # Save final summary
            summary_file = os.path.join(output_base_dir, "multi_sessions_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"📊 Multi-sessions processing complete! Summary saved to: {summary_file}")

        elif args.mode == "blob_polling":
            # =================================================================
            # BLOB POLLING MODE: Continuous polling for new sessions
            # =================================================================
            logger.info("🚀 Starting BLOB POLLING MODE: Continuous session discovery and processing")
            logger.info("=" * 80)
            
            # Load configurations
            pipeline_config = load_pipeline_config(args.pipeline_config)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_base_dir = os.path.join(current_dir, pipeline_config['local_storage']['output_base_dir'].lstrip('./'))
            
            # Load Azure config
            config = load_azure_config("blobfuse2_config.yaml")
            
            if 'azstorage' in config:
                az_config = config['azstorage']
            else:
                az_config = config
            
            # Base prefix for discovering sessions
            base_prefix = args.blob_prefix  # e.g., "test_mulitsession/"
            polling_interval = args.polling_interval
            
            logger.info(f"🔄 Blob Polling Configuration:")
            logger.info(f"   📁 Base Prefix: {base_prefix}")
            logger.info(f"   ⏰ Polling Interval: {polling_interval} minutes")
            logger.info(f"   📂 Output Directory: {output_base_dir}")
            logger.info("=" * 80)
            
            # Start continuous blob polling
            try:
                pipeline_main_with_blob_polling(
                    base_prefix=base_prefix,
                    output_base_dir=output_base_dir,
                    azure_config=az_config,
                    pipeline_config=pipeline_config,
                    polling_interval_minutes=polling_interval
                )
            except KeyboardInterrupt:
                logger.info("🛑 Blob polling stopped by user (Ctrl+C)")
            except Exception as e:
                logger.error(f"❌ Error in blob polling mode: {e}")
                raise

        if args.mode == "integrated":
            # =================================================================
            # INTEGRATED MODE: Continuous Blob Polling + Pipeline Processing
            # =================================================================
            logger.info("🚀 Starting INTEGRATED MODE: Continuous Blob Polling + Pipeline")
            logger.info("=" * 80)
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
            
            # Launch the integrated blob polling and pipeline task with batch processing
            task_ref = integrated_blob_polling_and_pipeline_task.remote(
                azure_config_path=args.azure_config,
                pipeline_config_path=args.pipeline_config,
                blob_prefix=args.blob_prefix,
                max_videos=args.max_videos,
                force_refresh=args.force_refresh,
                process_dual_views=args.dual_views,
                process_unwarped_views=args.unwarped_views,
                batch_size=pipeline_config.get('processing', {}).get('max_videos_per_batch', 3)
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
                pipeline_config = load_pipeline_config(args.pipeline_config)
                
                # Load Azure config
                azure_config = load_azure_config(args.azure_config)
                azure_blob_client = create_azure_blob_client(azure_config)
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