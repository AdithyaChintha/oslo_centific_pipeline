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

import json
import threading
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import ray
# Third-party imports
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Import your pipeline's main function
from ray_pipeline_testing import pipeline_main

# Import Azure blob utilities (download_blob kept for potential future use)
from utils.azure_blob_utils import download_blob

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AzurePipelineWrapper")
for name in (
    "azure",  # broad hammer
    "azure.storage",
    "azure.storage.blob",
    "azure.core.pipeline.policies.http_logging_policy",
    "urllib3"  
):
    lg = logging.getLogger(name)
    lg.setLevel(logging.WARNING)   # or logging.ERROR
    lg.propagate = False

# GPU and Parallelization Settings
VIDEO_BATCH_SIZE = int(os.environ.get("VIDEO_BATCH_SIZE", "8"))  # Number of concurrent videos
GPU_PER_VIDEO = int(os.environ.get("GPU_PER_VIDEO", "1"))  # GPUs allocated per video
MAX_CONCURRENT_DOWNLOADS = int(os.environ.get("MAX_CONCURRENT_DOWNLOADS", "2"))
STATE_FILE = os.environ.get("PROCESSED_VIDEOS_STATE_FILE", "processed_videos_state.json")

# Blob Storage Configuration  
INPUT_BLOB_PREFIX = os.environ.get("INPUT_BLOB_PREFIX", "instavideo/test/multi_processing_test/input/")
OUTPUT_BLOB_PREFIX = os.environ.get("OUTPUT_BLOB_PREFIX", "instavideo/test/multi_processing_test/output/")
SUPPORTED_EXTENSIONS = [".insv", ".mp4"]

# Processing Configuration
ENABLE_CLEANUP = os.environ.get("ENABLE_CLEANUP", "True").lower() == "true"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "300"))  # seconds
IGNORE_STATE = os.environ.get("IGNORE_STATE", "true").lower() == "true"

def detect_available_gpus() -> int:
    """Detect the number of available GPUs in the system."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Detected {gpu_count} CUDA-capable GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory}GB)")
            return gpu_count
        else:
            logger.warning("No CUDA-capable GPUs detected")
            return 0
    except ImportError:
        logger.warning("PyTorch not available, cannot detect GPUs")
        return 0
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        return 0

def _human_size(num: int, suffix="B") -> str:
    """Convert bytes to human readable format"""
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"

class VideoProcessingState:
    """Manages the state of video processing to track processed videos and avoid reprocessing."""
    
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self._load_state()
        self._lock = threading.Lock()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load processing state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded processing state with {len(state.get('processed_videos', {}))} tracked videos")
                return state
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load state file {self.state_file}: {e}. Starting with empty state.")
        
        # Default state structure
        return {
            "last_scan_timestamp": None,
            "processed_videos": {},
            "processing_stats": {
                "total_videos": 0,
                "completed": 0,
                "failed": 0,
                "in_progress": 0,
                "average_processing_time": 0.0
            }
        }
    
    def _save_state(self):
        tmp = f"{self.state_file}.tmp"
        with open(tmp, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
        os.replace(tmp, self.state_file)

    
    def is_video_processed(self, video_path: str) -> bool:
        """Check if a video has already been successfully processed."""
        with self._lock:
            video_info = self.state["processed_videos"].get(video_path, {})
            return video_info.get("status") == "completed"
    
    def is_video_in_progress(self, video_path: str) -> bool:
        """Check if a video is currently being processed."""
        with self._lock:
            video_info = self.state["processed_videos"].get(video_path, {})
            return video_info.get("status") == "in_progress"
    
    def mark_video_processing(self, video_path: str, gpu_label: str):
        with self._lock:
            self.state["processed_videos"][video_path] = {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "gpu_used": gpu_label,  
                "retry_count": self.state["processed_videos"].get(video_path, {}).get("retry_count", 0)
            }
            self.state["processing_stats"]["in_progress"] += 1
            self._save_state()
    
    def mark_video_completed(self, video_path: str, output_path: str, processing_time: float, views_processed: int = 1):
        """Mark a video as successfully processed."""
        with self._lock:
            if video_path in self.state["processed_videos"]:
                current_info = self.state["processed_videos"][video_path]
                if current_info.get("status") == "in_progress":
                    self.state["processing_stats"]["in_progress"] -= 1
                
            self.state["processed_videos"][video_path] = {
                "status": "completed",
                "processed_at": datetime.now().isoformat(),
                "output_path": output_path,
                "processing_time_seconds": processing_time,
                "views_processed": views_processed,
                "gpu_used": self.state["processed_videos"].get(video_path, {}).get("gpu_used", -1)
            }
            
            # Update stats
            stats = self.state["processing_stats"]
            stats["completed"] += 1
            stats["total_videos"] = len(self.state["processed_videos"])
            
            # Update average processing time
            completed_videos = [v for v in self.state["processed_videos"].values() if v.get("status") == "completed"]
            if completed_videos:
                total_time = sum(v.get("processing_time_seconds", 0) for v in completed_videos)
                stats["average_processing_time"] = total_time / len(completed_videos)
            
            self._save_state()
    
    def mark_video_failed(self, video_path: str, error_message: str):
        """Mark a video as failed to process."""
        with self._lock:
            current_info = self.state["processed_videos"].get(video_path, {})
            retry_count = current_info.get("retry_count", 0) + 1
            
            if current_info.get("status") == "in_progress":
                self.state["processing_stats"]["in_progress"] -= 1
            
            self.state["processed_videos"][video_path] = {
                "status": "failed",
                "processed_at": datetime.now().isoformat(),
                "error": error_message,
                "retry_count": retry_count,
                "gpu_used": current_info.get("gpu_used", -1)
            }
            
            self.state["processing_stats"]["failed"] += 1
            self.state["processing_stats"]["total_videos"] = len(self.state["processed_videos"])
            self._save_state()
    
    def get_failed_videos_for_retry(self, max_retries: int = MAX_RETRIES) -> List[str]:
        """Get list of failed videos that can be retried."""
        with self._lock:
            failed_videos = []
            for video_path, video_info in self.state["processed_videos"].items():
                if (video_info.get("status") == "failed" and 
                    video_info.get("retry_count", 0) < max_retries):
                    failed_videos.append(video_path)
            return failed_videos
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        with self._lock:
            return self.state["processing_stats"].copy()
    
    def update_scan_timestamp(self):
        """Update the last scan timestamp."""
        with self._lock:
            self.state["last_scan_timestamp"] = datetime.now().isoformat()
            self._save_state()

@ray.remote
class StateActor:
    def __init__(self, state_file: str):
        self.manager = VideoProcessingState(state_file)

    # thin wrappers that delegate to VideoProcessingState
    def is_video_processed(self, video_path: str) -> bool:
        return self.manager.is_video_processed(video_path)

    def is_video_in_progress(self, video_path: str) -> bool:
        return self.manager.is_video_in_progress(video_path)

    def mark_video_processing(self, video_path: str, gpu_label: str):
        self.manager.mark_video_processing(video_path, gpu_label)

    def mark_video_completed(self, video_path: str, output_path: str, processing_time: float, views_processed: int = 1):
        self.manager.mark_video_completed(video_path, output_path, processing_time, views_processed)

    def mark_video_failed(self, video_path: str, error_message: str):
        self.manager.mark_video_failed(video_path, error_message)

    def get_failed_videos_for_retry(self, max_retries: int):
        return self.manager.get_failed_videos_for_retry(max_retries)

    def get_stats(self):
        return self.manager.get_stats()

    def update_scan_timestamp(self):
        self.manager.update_scan_timestamp()


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

def consolidate_dual_view_outputs(all_results, local_outputs_dir, conv_time, local_input_path):
    """
    Consolidate outputs from both views into unified analysis results.
    
    Args:
        all_results: List of view processing results
        local_outputs_dir: Base output directory
        conv_time: Conversion time in seconds
        local_input_path: Path to input file
        
    Returns:
        dict: Consolidated summary with merged timeline and statistics
    """
    import json
    import os
    from collections import defaultdict
    
    logger.info("Loading individual view results...")
    
    # Initialize consolidated data structures
    consolidated_timeline = []
    consolidated_stats = {
        "processing_summary": defaultdict(int),
        "annotation_summary": {
            "total_flagged_segments": 0,
            "flagged_duration_seconds": 0.0,
            "high_priority_segments": 0
        }
    }
    
    view_summaries = []
    
    # Process each view's results
    for view_result in all_results:
        view_name = view_result["view"]
        view_local_dir = view_result["local_output_dir"]
        
        logger.info("Processing %s results...", view_name)
        
        # Load pipeline summary
        pipeline_summary_file = os.path.join(view_local_dir, "pipeline_summary.json")
        if os.path.exists(pipeline_summary_file):
            with open(pipeline_summary_file, 'r') as f:
                pipeline_summary = json.load(f)
                
            # Aggregate processing statistics
            proc_summary = pipeline_summary.get("processing_summary", {})
            for key, value in proc_summary.items():
                if key != "overall_success_rate":
                    consolidated_stats["processing_summary"][key] += value
                    
            # Aggregate annotation statistics  
            ann_summary = pipeline_summary.get("annotation_summary", {})
            consolidated_stats["annotation_summary"]["total_flagged_segments"] += ann_summary.get("total_flagged_segments", 0)
            consolidated_stats["annotation_summary"]["flagged_duration_seconds"] += ann_summary.get("flagged_duration_seconds", 0.0)
            consolidated_stats["annotation_summary"]["high_priority_segments"] += ann_summary.get("high_priority_segments", 0)
            
            view_summaries.append({
                "view": view_name,
                "pipeline_summary": pipeline_summary
            })
        
        # Load flagged timeline
        timeline_file = os.path.join(view_local_dir, "master_flagged_timeline.json")
        if os.path.exists(timeline_file):
            with open(timeline_file, 'r') as f:
                timeline_data = json.load(f)
                
            # Add view identifier to each segment and merge
            flagged_timeline = timeline_data.get("flagged_timeline", [])
            for segment in flagged_timeline:
                # Add view information to each segment
                consolidated_segment = segment.copy()
                consolidated_segment["source_view"] = view_name
                consolidated_segment["view_file"] = timeline_data.get("video_file", "")
                consolidated_timeline.append(consolidated_segment)
    
    # Sort consolidated timeline by start time
    consolidated_timeline.sort(key=lambda x: x.get("start_time", 0))
    
    # Calculate overall success rate
    total_tasks = len([k for k in consolidated_stats["processing_summary"].keys() if k.startswith("successful_")])
    if total_tasks > 0:
        successful_tasks = sum(v for k, v in consolidated_stats["processing_summary"].items() if k.startswith("successful_"))
        total_possible = consolidated_stats["processing_summary"].get("total_shards", 1) * total_tasks * 2  # 2 views
        overall_success_rate = (successful_tasks / total_possible) * 100 if total_possible > 0 else 0
        consolidated_stats["processing_summary"]["overall_success_rate"] = f"{overall_success_rate:.1f}%"
    
    # Calculate consolidated workload reduction
    total_duration = 60 * 2  # 60 seconds per view * 2 views = 120 seconds total
    flagged_duration = consolidated_stats["annotation_summary"]["flagged_duration_seconds"]
    workload_reduction = ((total_duration - flagged_duration) / total_duration) * 100 if total_duration > 0 else 0
    consolidated_stats["annotation_summary"]["annotation_workload_reduction"] = f"{workload_reduction:.1f}%"
    
    # Merge overlapping segments from different views
    merged_timeline = merge_overlapping_segments(consolidated_timeline)
    
    # Create consolidated results structure
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
    
    # Save consolidated timeline
    consolidated_timeline_file = os.path.join(local_outputs_dir, "consolidated_master_timeline.json")
    with open(consolidated_timeline_file, 'w') as f:
        json.dump({
            "input_file": local_input_path,
            "total_duration_seconds": total_duration,
            "consolidated_flagged_duration_seconds": flagged_duration,
            "consolidated_annotation_workload_reduction": consolidated_data["consolidated_annotation_summary"]["annotation_workload_reduction"],
            "total_consolidated_segments": len(merged_timeline),
            "consolidated_timeline": merged_timeline,
            "view_breakdown": {
                "view_1_segments": len([s for s in consolidated_timeline if s.get("source_view") == "view_1"]),
                "view_2_segments": len([s for s in consolidated_timeline if s.get("source_view") == "view_2"]),
                "merged_segments": len(merged_timeline)
            }
        }, f, indent=2)
    
    # Save detailed consolidated summary
    consolidated_summary_file = os.path.join(local_outputs_dir, "consolidated_pipeline_summary.json")
    with open(consolidated_summary_file, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    
    logger.info("✅ Consolidated timeline saved: %s", consolidated_timeline_file)
    logger.info("✅ Consolidated summary saved: %s", consolidated_summary_file)
    logger.info("📊 Total segments before merge: %d", len(consolidated_timeline))
    logger.info("📊 Total segments after merge: %d", len(merged_timeline))
    logger.info("📊 Consolidated workload reduction: %s", consolidated_data["consolidated_annotation_summary"]["annotation_workload_reduction"])
    
    return consolidated_data


def consolidate_all_model_outputs(all_results, local_outputs_dir, local_input_path):
    """
    Consolidate all individual model outputs (YOLO, NSFW, Scene, Face, Motion, etc.) 
    from both views into a single comprehensive JSON file.
    
    Args:
        all_results: List of view processing results
        local_outputs_dir: Base output directory
        local_input_path: Path to input file
        
    Returns:
        dict: Consolidated model outputs from all views
    """
    import json
    import os
    import glob
    from collections import defaultdict
    
    logger.info("Loading all model outputs from both views...")
    
    consolidated_models = {
        "input_file": local_input_path,
        "views_processed": len(all_results),
        "yolo_detections": {
            "view_1": [],
            "view_2": [],
            "consolidated": []
        },
        "nsfw_analysis": {
            "view_1": {},
            "view_2": {},
            "consolidated": {}
        },
        "scene_detection": {
            "view_1": [],
            "view_2": [],
            "consolidated": []
        },
        "face_analysis": {
            "view_1": {},
            "view_2": {},
            "consolidated": {}
        },
        "motion_analysis": {
            "view_1": {},
            "view_2": {},
            "consolidated": {}
        },
        "audio_analysis": {
            "view_1": {},
            "view_2": {},
            "consolidated": {}
        },
        "clap_detection": {
            "view_1": {},
            "view_2": {},
            "consolidated": {}
        }
    }
    
    # Process each view's model outputs
    for view_result in all_results:
        view_name = view_result["view"]
        view_local_dir = view_result["local_output_dir"]
        
        logger.info("Processing %s model outputs...", view_name)
        
        # Find all shard directories
        shard_dirs = glob.glob(os.path.join(view_local_dir, "shard_*"))
        
        for shard_dir in sorted(shard_dirs):
            shard_name = os.path.basename(shard_dir)
            logger.info("  Processing %s %s...", view_name, shard_name)
            
            # 1. YOLO Detection Results
            yolo_dir = os.path.join(shard_dir, "yolo_output")
            if os.path.exists(yolo_dir):
                yolo_files = glob.glob(os.path.join(yolo_dir, "*.events.jsonl"))
                for yolo_file in yolo_files:
                    yolo_events = load_jsonl_file(yolo_file)
                    for event in yolo_events:
                        if "_meta" not in event:  # Skip metadata lines
                            event["source_view"] = view_name
                            event["shard"] = shard_name
                    consolidated_models["yolo_detections"][view_name].extend(yolo_events)
            
            # 2. NSFW Analysis Results
            nsfw_dir = os.path.join(shard_dir, "nsfw_output")
            if os.path.exists(nsfw_dir):
                nsfw_files = glob.glob(os.path.join(nsfw_dir, "*_nsfw_results.json"))
                for nsfw_file in nsfw_files:
                    with open(nsfw_file, 'r') as f:
                        nsfw_data = json.load(f)
                        nsfw_data["source_view"] = view_name
                        nsfw_data["shard"] = shard_name
                        if view_name not in consolidated_models["nsfw_analysis"] or not isinstance(consolidated_models["nsfw_analysis"][view_name], list):
                            consolidated_models["nsfw_analysis"][view_name] = []
                        consolidated_models["nsfw_analysis"][view_name].append(nsfw_data)
            
            # 3. Scene Detection Results
            scene_dir = os.path.join(shard_dir, "scene_output")
            if os.path.exists(scene_dir):
                scene_files = glob.glob(os.path.join(scene_dir, "*_scene_detection_results.json"))
                for scene_file in scene_files:
                    with open(scene_file, 'r') as f:
                        scene_data = json.load(f)
                        scene_data["source_view"] = view_name
                        scene_data["shard"] = shard_name
                        # Add view info to each scene
                        for scene in scene_data.get("scenes", []):
                            scene["source_view"] = view_name
                            scene["shard"] = shard_name
                        consolidated_models["scene_detection"][view_name].append(scene_data)
            
            # 4. Face Analysis Results
            face_dir = os.path.join(shard_dir, "face_output")
            if os.path.exists(face_dir):
                face_files = glob.glob(os.path.join(face_dir, "*_face_results.json"))
                for face_file in face_files:
                    with open(face_file, 'r') as f:
                        face_data = json.load(f)
                        face_data["source_view"] = view_name
                        face_data["shard"] = shard_name
                        if view_name not in consolidated_models["face_analysis"] or not isinstance(consolidated_models["face_analysis"][view_name], list):
                            consolidated_models["face_analysis"][view_name] = []
                        consolidated_models["face_analysis"][view_name].append(face_data)
            
            # 5. Motion Analysis Results
            motion_dir = os.path.join(shard_dir, "motion_output")
            if os.path.exists(motion_dir):
                motion_files = glob.glob(os.path.join(motion_dir, "*_motion_results.json"))
                for motion_file in motion_files:
                    with open(motion_file, 'r') as f:
                        motion_data = json.load(f)
                        motion_data["source_view"] = view_name
                        motion_data["shard"] = shard_name
                        # Add view info to each segment
                        for segment in motion_data.get("segments", []):
                            segment["source_view"] = view_name
                            segment["shard"] = shard_name
                        if view_name not in consolidated_models["motion_analysis"] or not isinstance(consolidated_models["motion_analysis"][view_name], list):
                            consolidated_models["motion_analysis"][view_name] = []
                        consolidated_models["motion_analysis"][view_name].append(motion_data)
            
            # 6. Audio Analysis Results (usually in shard_1 only since audio is shared)
            audio_dir = os.path.join(shard_dir, "audio_output")
            if os.path.exists(audio_dir):
                audio_files = glob.glob(os.path.join(audio_dir, "*.json"))
                for audio_file in audio_files:
                    with open(audio_file, 'r') as f:
                        audio_data = json.load(f)
                        audio_data["source_view"] = view_name
                        audio_data["shard"] = shard_name
                        if view_name not in consolidated_models["audio_analysis"] or not isinstance(consolidated_models["audio_analysis"][view_name], list):
                            consolidated_models["audio_analysis"][view_name] = []
                        consolidated_models["audio_analysis"][view_name].append(audio_data)
            
            # 7. Clap Detection Results
            clap_dir = os.path.join(shard_dir, "clap_output")
            if os.path.exists(clap_dir):
                clap_files = glob.glob(os.path.join(clap_dir, "*.json"))
                for clap_file in clap_files:
                    with open(clap_file, 'r') as f:
                        clap_data = json.load(f)
                        clap_data["source_view"] = view_name
                        clap_data["shard"] = shard_name
                        if view_name not in consolidated_models["clap_detection"] or not isinstance(consolidated_models["clap_detection"][view_name], list):
                            consolidated_models["clap_detection"][view_name] = []
                        consolidated_models["clap_detection"][view_name].append(clap_data)
    
    # Create consolidated versions combining both views
    logger.info("Creating consolidated model summaries...")
    
    # Consolidate YOLO detections
    all_yolo = consolidated_models["yolo_detections"]["view_1"] + consolidated_models["yolo_detections"]["view_2"]
    consolidated_models["yolo_detections"]["consolidated"] = sorted(all_yolo, key=lambda x: x.get("t", 0))
    
    # Consolidate scene detection
    all_scenes = []
    for view_scenes in [consolidated_models["scene_detection"]["view_1"], consolidated_models["scene_detection"]["view_2"]]:
        for scene_data in view_scenes:
            for scene in scene_data.get("scenes", []):
                all_scenes.append(scene)
    consolidated_models["scene_detection"]["consolidated"] = sorted(all_scenes, key=lambda x: x.get("start_time", 0))
    
    # Consolidate motion analysis
    all_motion_segments = []
    for view_motion in [consolidated_models["motion_analysis"]["view_1"], consolidated_models["motion_analysis"]["view_2"]]:
        for motion_data in view_motion:
            for segment in motion_data.get("segments", []):
                all_motion_segments.append(segment)
    consolidated_models["motion_analysis"]["consolidated"] = sorted(all_motion_segments, key=lambda x: x.get("start_time", 0))
    
    # Create consolidated statistics
    consolidated_stats = {
        "total_yolo_detections": len([d for d in consolidated_models["yolo_detections"]["consolidated"] if "_meta" not in d]),
        "total_scenes": len(consolidated_models["scene_detection"]["consolidated"]),
        "total_motion_segments": len(consolidated_models["motion_analysis"]["consolidated"]),
        "total_nsfw_detections": sum([
            sum([data.get("total_nsfw_detections", 0) for data in view_data]) 
            for view_data in [consolidated_models["nsfw_analysis"]["view_1"], consolidated_models["nsfw_analysis"]["view_2"]] 
            if isinstance(view_data, list)
        ]),
        "total_faces_detected": sum([
            sum([data.get("total_faces_detected", 0) for data in view_data]) 
            for view_data in [consolidated_models["face_analysis"]["view_1"], consolidated_models["face_analysis"]["view_2"]] 
            if isinstance(view_data, list)
        ])
    }
    
    consolidated_models["consolidated_statistics"] = consolidated_stats
    
    # Save consolidated model outputs
    consolidated_models_file = os.path.join(local_outputs_dir, "consolidated_all_model_outputs.json")
    with open(consolidated_models_file, 'w') as f:
        json.dump(consolidated_models, f, indent=2)
    
    logger.info("✅ Consolidated model outputs saved: %s", consolidated_models_file)
    logger.info("📊 Total YOLO detections: %d", consolidated_stats["total_yolo_detections"])
    logger.info("📊 Total scenes: %d", consolidated_stats["total_scenes"])
    logger.info("📊 Total motion segments: %d", consolidated_stats["total_motion_segments"])
    logger.info("📊 Total NSFW detections: %d", consolidated_stats["total_nsfw_detections"])
    logger.info("📊 Total faces detected: %d", consolidated_stats["total_faces_detected"])
    
    return consolidated_models

def copy_directory_contents(src_dir: str, dst_dir: str):
    """Copy all contents from source directory to destination directory."""
    if os.path.exists(src_dir) and os.listdir(src_dir):
        for item in os.listdir(src_dir):
            src_path = os.path.join(src_dir, item)
            dst_path = os.path.join(dst_dir, item)
            if os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)


def cleanup_temp_directories(directories: List[str]):
    """Safely remove temporary directories."""
    for directory in directories:
        if directory and os.path.isdir(directory):
            try:
                shutil.rmtree(directory)
                logger.debug(f"Cleaned up directory: {directory}")
            except Exception as e:
                logger.warning(f"Failed to cleanup directory {directory}: {e}")

@ray.remote(num_gpus=0)
def discover_videos(connection_string: str, container_name: str, input_prefix: str, state_actor) -> List[str]:
    logger.info(f"Discovering videos in blob prefix: {input_prefix}")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        all_videos = []
        for blob in container_client.list_blobs(name_starts_with=input_prefix):
            blob_name = blob.name
            if any(blob_name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                if not IGNORE_STATE:
                    processed = ray.get(state_actor.is_video_processed.remote(blob_name))
                    in_progress = ray.get(state_actor.is_video_in_progress.remote(blob_name))
                    if processed or in_progress:
                        continue
                all_videos.append(blob_name)

        logger.info(f"Found {len(all_videos)} unprocessed video(s)")
        return all_videos
    except Exception as e:
        logger.error(f"Error discovering videos: {e}")
        return []


def process_insv_video(local_input_path: str, temp_output_dir: str, 
                           local_outputs_dir: str, gpu_label: str, video_name: str) -> int:
    """Process INSV video with dual-view support - outputs organized by video name."""
    logger.info(f"GPU {gpu_label}: Converting INSV to dual MP4 for {video_name}")
    
    # Import the conversion function
    from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
    
    # Convert INSV to dual MP4
    conversion_start = time.time()
    conversion_result = ray.get(convert_insv_to_dual_mp4.remote(local_input_path))
    conv_time = time.time() - conversion_start
    
    if not conversion_result.get('success', False):
        raise RuntimeError(f"INSV conversion failed: {conversion_result.get('error', 'Unknown error')}")
    
    # Get both view paths
    view1_path = conversion_result['output_view_1']
    view2_path = conversion_result['output_view_2']
    logger.info(f"GPU {gpu_label}: INSV converted to dual MP4 in {conv_time:.2f}s")
    
    # Process both views
    mp4_paths = [view1_path, view2_path]
    view_names = ["view_1", "view_2"]
    all_results = []
    
    for i, (mp4_path, view_name) in enumerate(zip(mp4_paths, view_names)):
        logger.info(f"GPU {gpu_label}: Processing {view_name} for {video_name} ({i+1}/2)")
        
        # Create view-specific output directories
        # Temp directories (GPU-specific for processing isolation)
        view_temp_output_dir = os.path.join(temp_output_dir, view_name)
        os.makedirs(view_temp_output_dir, exist_ok=True)
        
        # Final local directories (organized by video name)
        view_local_output_dir = os.path.join(local_outputs_dir, view_name)
        os.makedirs(view_local_output_dir, exist_ok=True)
        
        # Run the pipeline for this view
        pipeline_main(
            input_video_path=mp4_path,
            output_dir=view_temp_output_dir
        )
        
        # Copy view outputs to video-specific local directory
        if os.path.exists(view_temp_output_dir) and os.listdir(view_temp_output_dir):
            copy_directory_contents(view_temp_output_dir, view_local_output_dir)
            
        all_results.append({
            "view": view_name,
            "mp4_path": mp4_path,
            "temp_output_dir": view_temp_output_dir,
            "local_output_dir": view_local_output_dir
        })
    
    # Consolidate outputs from both views in the video-specific directory
    logger.info(f"GPU {gpu_label}: Consolidating outputs from both views for {video_name}")
    consolidate_dual_view_outputs(all_results, local_outputs_dir, conv_time, local_input_path)
    consolidate_all_model_outputs(all_results, local_outputs_dir, local_input_path)
    reorganize_and_consolidate_per_shard(all_results, local_outputs_dir, local_input_path)
    
    return 2  # Two views processed


def process_single_video(local_input_path: str, temp_output_dir: str, 
                             local_outputs_dir: str, gpu_label: str, video_name: str):
    """Process single video file - outputs organized by video name."""
    logger.info(f"GPU {gpu_label}: Processing single video file {video_name}")
    
    # Run the pipeline in temp directory
    pipeline_main(
        input_video_path=local_input_path,
        output_dir=temp_output_dir
    )
    
    # Copy outputs to video-specific local directory
    copy_directory_contents(temp_output_dir, local_outputs_dir)
    
    logger.info(f"GPU {gpu_label}: Single video {video_name} processing complete")

@ray.remote(num_gpus=GPU_PER_VIDEO)
def process_video_parallel(video_blob_path: str, connection_string: str, container_name: str, 
                          output_blob_prefix: str, state_actor) -> Dict[str, Any]:
    """Process a single video on a dedicated GPU."""
    gpu_label = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    logger.info(f"Starting GPU-parallel processing for video: {video_blob_path} (CUDA_VISIBLE_DEVICES={gpu_label})")
    
    # Set GPU device for this process
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    start_time = time.time()
    temp_input_dir = None
    temp_output_dir = None
    local_input_path = None
    
    
    try:
        # Mark video as being processed
        ray.get(state_actor.mark_video_processing.remote(video_blob_path, gpu_label))
        
        # Create temporary directories
        temp_input_dir = tempfile.mkdtemp(prefix=f"gpu{gpu_label}_input_")
        temp_output_dir = tempfile.mkdtemp(prefix=f"gpu{gpu_label}_output_")
        logger.info(f"GPU {gpu_label}: Created temp dirs - input: {temp_input_dir}, output: {temp_output_dir}")
        
        # Download video
        video_filename = os.path.basename(video_blob_path)
        local_input_path = os.path.join(temp_input_dir, video_filename)
        
        logger.info(f"GPU {gpu_label}: Downloading {video_blob_path}")
        download_start = time.time()
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=video_blob_path)
        
        data = blob_client.download_blob(max_concurrency=4).readall()
        with open(local_input_path, "wb") as f:
            f.write(data)
        
        download_time = time.time() - download_start
        logger.info(f"GPU {gpu_label}: Downloaded video in {download_time:.2f} seconds")
        
        # Determine output structure - organized by VIDEO NAME, not GPU
        video_name_folder = os.path.splitext(video_filename)[0]
        final_output_prefix = f"{output_blob_prefix.rstrip('/')}/{video_name_folder}"
        
        # Create local outputs directory organized by video name
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_outputs_dir = os.path.join(script_dir, "outputs", video_name_folder)
        os.makedirs(local_outputs_dir, exist_ok=True)
        
        logger.info(f"GPU {gpu_label}: Processing {video_filename} -> outputs/{video_name_folder}")
        
        # Process based on video type (INSV or regular)
        if local_input_path.lower().endswith('.insv'):
            views_processed = process_insv_video(local_input_path, temp_output_dir, local_outputs_dir, gpu_label, video_name_folder)
        else:
            process_single_video(local_input_path, temp_output_dir, local_outputs_dir, gpu_label, video_name_folder)
            views_processed = 1
        
        # Upload results using existing upload functionality from main pipeline
        logger.info(f"GPU {gpu_label}: Uploading results to {final_output_prefix}")
        upload_result = upload_consolidated_files_to_azure(connection_string, local_outputs_dir, final_output_prefix)
        
        if not upload_result.get('success', False):
            logger.warning(f"GPU {gpu_label}: Upload completed with issues: {upload_result.get('error', 'Unknown error')}")
        else:
            logger.info(f"GPU {gpu_label}: Successfully uploaded {upload_result['total_files']} files")
        
        # Mark as completed
        total_time = time.time() - start_time
        ray.get(state_actor.mark_video_completed.remote(video_blob_path, final_output_prefix, total_time, views_processed))
        
        logger.info(f"GPU {gpu_label}: Successfully completed processing {video_blob_path} in {total_time:.2f} seconds")
        
        return {
            "status": "success",
            "video_path": video_blob_path,
            "video_name": video_name_folder,
            "gpu_label": gpu_label,
            "processing_time": total_time,
            "views_processed": views_processed,
            "output_path": final_output_prefix,
            "local_output_path": local_outputs_dir
        }
        
    except Exception as e:
        error_msg = f"GPU {gpu_label}: Failed to process {video_blob_path}: {str(e)}"
        logger.error(error_msg)
        
        # Mark as failed
        ray.get(state_actor.mark_video_failed.remote(video_blob_path, str(e)))
        
        return {
            "status": "failed",
            "video_path": video_blob_path,
            "gpu_label": gpu_label,
            "error": str(e)
        }
    
    finally:
        # Cleanup temporary directories
        if ENABLE_CLEANUP:
            cleanup_temp_directories([temp_input_dir, temp_output_dir])


def load_jsonl_file(file_path):
    """Load JSONL file and return list of JSON objects."""
    import json
    events = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception as e:
        logger.warning("Failed to load JSONL file %s: %s", file_path, e)
    return events


def reorganize_and_consolidate_per_shard(all_results, local_outputs_dir, local_input_path):
    """
    Reorganize the output structure to have each shard as a top-level folder
    with view_1, view_2 subfolders and consolidated JSON for each shard.
    
    Target structure:
    outputs/
    ├── shard_1/
    │   ├── view_1/          (individual view_1 results for shard_1)
    │   ├── view_2/          (individual view_2 results for shard_1)
    │   └── consolidated_shard_1_output.json  (consolidated results for shard_1)
    ├── shard_2/
    │   ├── view_1/          (individual view_1 results for shard_2)
    │   ├── view_2/          (individual view_2 results for shard_2)  
    │   └── consolidated_shard_2_output.json  (consolidated results for shard_2)
    
    Args:
        all_results: List of view processing results
        local_outputs_dir: Base output directory
        local_input_path: Path to input file
        
    Returns:
        dict: Summary of reorganization and consolidation
    """
    import json
    import os
    import glob
    import shutil
    from datetime import datetime
    
    logger.info("🔄 REORGANIZING OUTPUT STRUCTURE PER SHARD...")
    
    # Find all unique shard names across all views
    all_shards = set()
    for view_result in all_results:
        view_local_dir = view_result["local_output_dir"]
        shard_dirs = glob.glob(os.path.join(view_local_dir, "shard_*"))
        for shard_dir in shard_dirs:
            shard_name = os.path.basename(shard_dir)
            all_shards.add(shard_name)
    
    all_shards = sorted(list(all_shards))
    logger.info("Found %d shards to reorganize: %s", len(all_shards), all_shards)
    
    reorganization_summary = []
    
    # Create the new structure for each shard
    for shard_name in all_shards:
        logger.info("📁 Reorganizing %s...", shard_name)
        
        # Create new shard directory structure
        new_shard_dir = os.path.join(local_outputs_dir, shard_name)
        os.makedirs(new_shard_dir, exist_ok=True)
        
        # Create view subdirectories within shard
        new_view1_dir = os.path.join(new_shard_dir, "view_1")
        new_view2_dir = os.path.join(new_shard_dir, "view_2")
        os.makedirs(new_view1_dir, exist_ok=True)
        os.makedirs(new_view2_dir, exist_ok=True)
        
        moved_views = []
        
        # Move shard content from each view to the new structure
        for view_result in all_results:
            view_name = view_result["view"]
            view_local_dir = view_result["local_output_dir"]
            
            # Source: old view structure (outputs/VIDEO/view_1/shard_N)
            old_shard_path = os.path.join(view_local_dir, shard_name)
            
            # Destination: new shard structure (outputs/VIDEO/shard_N/view_1)
            if view_name == "view_1":
                new_view_path = new_view1_dir
            else:  # view_2
                new_view_path = new_view2_dir
            
            if os.path.exists(old_shard_path):
                logger.info("  Moving %s %s content...", shard_name, view_name)
                
                # Move all content from old shard to new view directory
                for item in os.listdir(old_shard_path):
                    src_item = os.path.join(old_shard_path, item)
                    dst_item = os.path.join(new_view_path, item)
                    
                    if os.path.isdir(src_item):
                        if os.path.exists(dst_item):
                            shutil.rmtree(dst_item)
                        shutil.copytree(src_item, dst_item)
                    else:
                        shutil.copy2(src_item, dst_item)
                
                moved_views.append(view_name)
                logger.info("    ✅ %s %s content moved", shard_name, view_name)
            else:
                logger.warning("    ⚠️  %s not found in %s", shard_name, view_name)
        
        # Generate consolidated JSON for this shard using the reorganized structure
        logger.info("  Creating consolidated JSON for %s...", shard_name)
        shard_consolidated = consolidate_single_shard_from_reorganized(
            shard_name, new_shard_dir, local_input_path
        )
        
        # Save the consolidated JSON in the shard directory
        consolidated_file = os.path.join(new_shard_dir, f"consolidated_{shard_name}_output.json")
        with open(consolidated_file, 'w') as f:
            json.dump(shard_consolidated, f, indent=2)
        
        logger.info("    ✅ Consolidated JSON saved: %s", os.path.basename(consolidated_file))
        
        reorganization_summary.append({
            "shard_name": shard_name,
            "shard_directory": new_shard_dir,
            "views_moved": moved_views,
            "consolidated_file": consolidated_file,
            "statistics": shard_consolidated["shard_statistics"]
        })
    
    # Create overall reorganization summary
    master_summary = {
        "reorganization_info": {
            "input_file": local_input_path,
            "total_shards_reorganized": len(all_shards),
            "reorganization_timestamp": datetime.now().isoformat(),
            "new_structure": "Each shard has own directory with view_1/, view_2/ subdirectories and consolidated JSON"
        },
        "shard_summaries": reorganization_summary,
        "overall_statistics": {
            "total_yolo_detections": sum(s["statistics"]["total_yolo_detections"] for s in reorganization_summary),
            "total_scenes": sum(s["statistics"]["total_scenes"] for s in reorganization_summary),
            "total_motion_segments": sum(s["statistics"]["total_motion_segments"] for s in reorganization_summary),
            "total_nsfw_analyses": sum(s["statistics"]["total_nsfw_analyses"] for s in reorganization_summary),
            "total_face_analyses": sum(s["statistics"]["total_face_analyses"] for s in reorganization_summary),
            "total_audio_analyses": sum(s["statistics"]["total_audio_analyses"] for s in reorganization_summary),
            "total_clap_detections": sum(s["statistics"]["total_clap_detections"] for s in reorganization_summary)
        }
    }
    
    # Save master reorganization summary
    summary_file = os.path.join(local_outputs_dir, "per_shard_reorganization_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(master_summary, f, indent=2)
    
    logger.info("✅ Reorganization complete! Summary saved: %s", summary_file)
    logger.info("📊 REORGANIZATION SUMMARY:")
    logger.info("   Total shards reorganized: %d", len(all_shards))
    for shard_summary in reorganization_summary:
        shard_name = shard_summary["shard_name"]
        stats = shard_summary["statistics"]
        logger.info("   %s: YOLO=%d, Scenes=%d, Motion=%d", 
                   shard_name, stats['total_yolo_detections'], 
                   stats['total_scenes'], stats['total_motion_segments'])
    
    return master_summary


def consolidate_single_shard_from_reorganized(shard_name, shard_dir, local_input_path):
    """
    Create consolidated output for a single shard from the reorganized structure.
    
    Args:
        shard_name: Name of the shard (e.g., 'shard_1')
        shard_dir: Path to the shard directory (contains view_1/, view_2/ subdirs)
        local_input_path: Path to input file
        
    Returns:
        dict: Consolidated output for this specific shard
    """
    import json
    import os
    import glob
    from datetime import datetime
    
    logger.info("    Creating consolidated output for %s...", shard_name)
    
    # Initialize consolidated structure for this shard
    shard_consolidated = {
        "consolidation_info": {
            "shard_name": shard_name,
            "input_file": local_input_path,
            "consolidation_timestamp": datetime.now().isoformat(),
            "structure": "Reorganized per-shard structure"
        },
        "yolo_detections": [],
        "scene_detection": [],
        "nsfw_analysis": [],
        "face_analysis": [],
        "motion_analysis": [],
        "audio_analysis": [],
        "clap_detection": []
    }
    
    # Process each view subdirectory in the shard
    for view_name in ["view_1", "view_2"]:
        view_dir = os.path.join(shard_dir, view_name)
        
        if not os.path.exists(view_dir):
            logger.warning("      Warning: %s not found in %s", view_name, shard_name)
            continue
            
        logger.info("      Processing %s...", view_name)
        
        # 1. YOLO Detection Results
        yolo_dir = os.path.join(view_dir, "yolo_output")
        if os.path.exists(yolo_dir):
            yolo_files = glob.glob(os.path.join(yolo_dir, "*.events.jsonl"))
            for yolo_file in yolo_files:
                yolo_events = load_jsonl_file(yolo_file)
                for event in yolo_events:
                    if "_meta" not in event:  # Skip metadata lines
                        event["source_view"] = view_name
                        event["shard"] = shard_name
                        shard_consolidated["yolo_detections"].append(event)
        
        # 2. Scene Detection Results
        scene_dir = os.path.join(view_dir, "scene_output")
        if os.path.exists(scene_dir):
            scene_files = glob.glob(os.path.join(scene_dir, "*_scene_detection_results.json"))
            for scene_file in scene_files:
                with open(scene_file, 'r') as f:
                    scene_data = json.load(f)
                    for scene in scene_data.get("scenes", []):
                        scene["source_view"] = view_name
                        scene["shard"] = shard_name
                        scene["video_path"] = scene_data.get("video_path", "")
                        shard_consolidated["scene_detection"].append(scene)
        
        # 3. NSFW Analysis Results
        nsfw_dir = os.path.join(view_dir, "nsfw_output")
        if os.path.exists(nsfw_dir):
            nsfw_files = glob.glob(os.path.join(nsfw_dir, "*_nsfw_results.json"))
            for nsfw_file in nsfw_files:
                with open(nsfw_file, 'r') as f:
                    nsfw_data = json.load(f)
                    nsfw_data["source_view"] = view_name
                    nsfw_data["shard"] = shard_name
                    shard_consolidated["nsfw_analysis"].append(nsfw_data)
        
        # 4. Face Analysis Results
        face_dir = os.path.join(view_dir, "face_output")
        if os.path.exists(face_dir):
            face_files = glob.glob(os.path.join(face_dir, "*_face_results.json"))
            for face_file in face_files:
                with open(face_file, 'r') as f:
                    face_data = json.load(f)
                    face_data["source_view"] = view_name
                    face_data["shard"] = shard_name
                    shard_consolidated["face_analysis"].append(face_data)
        
        # 5. Motion Analysis Results
        motion_dir = os.path.join(view_dir, "motion_output")
        if os.path.exists(motion_dir):
            motion_files = glob.glob(os.path.join(motion_dir, "*_motion_results.json"))
            for motion_file in motion_files:
                with open(motion_file, 'r') as f:
                    motion_data = json.load(f)
                    for segment in motion_data.get("segments", []):
                        segment["source_view"] = view_name
                        segment["shard"] = shard_name
                        shard_consolidated["motion_analysis"].append(segment)
        
        # 6. Audio Analysis Results
        audio_dir = os.path.join(view_dir, "audio_output")
        if os.path.exists(audio_dir):
            audio_files = glob.glob(os.path.join(audio_dir, "*.json"))
            for audio_file in audio_files:
                with open(audio_file, 'r') as f:
                    audio_data = json.load(f)
                    audio_data["source_view"] = view_name
                    audio_data["shard"] = shard_name
                    shard_consolidated["audio_analysis"].append(audio_data)
        
        # 7. Clap Detection Results
        clap_dir = os.path.join(view_dir, "clap_output")
        if os.path.exists(clap_dir):
            clap_files = glob.glob(os.path.join(clap_dir, "*.json"))
            for clap_file in clap_files:
                with open(clap_file, 'r') as f:
                    clap_data = json.load(f)
                    clap_data["source_view"] = view_name
                    clap_data["shard"] = shard_name
                    shard_consolidated["clap_detection"].append(clap_data)
    
    # Sort data by timestamp
    shard_consolidated["yolo_detections"].sort(key=lambda x: x.get("t", 0))
    shard_consolidated["scene_detection"].sort(key=lambda x: x.get("start_time", 0))
    shard_consolidated["motion_analysis"].sort(key=lambda x: x.get("start_time", 0))
    
    # Add statistics for this shard
    shard_consolidated["shard_statistics"] = {
        "total_yolo_detections": len(shard_consolidated["yolo_detections"]),
        "total_scenes": len(shard_consolidated["scene_detection"]),
        "total_motion_segments": len(shard_consolidated["motion_analysis"]),
        "total_nsfw_analyses": len(shard_consolidated["nsfw_analysis"]),
        "total_face_analyses": len(shard_consolidated["face_analysis"]),
        "total_audio_analyses": len(shard_consolidated["audio_analysis"]),
        "total_clap_detections": len(shard_consolidated["clap_detection"])
    }
    
    return shard_consolidated


def consolidate_per_shard_outputs(all_results, local_outputs_dir, local_input_path):
    """
    Create consolidated outputs for each shard individually.
    
    Args:
        all_results: List of view processing results
        local_outputs_dir: Base output directory
        local_input_path: Path to input file
        
    Returns:
        dict: Summary of all per-shard consolidations
    """
    import json
    import os
    import glob
    from datetime import datetime
    
    logger.info("Creating per-shard consolidated outputs...")
    
    # Find all unique shard names across all views
    all_shards = set()
    for view_result in all_results:
        view_local_dir = view_result["local_output_dir"]
        shard_dirs = glob.glob(os.path.join(view_local_dir, "shard_*"))
        for shard_dir in shard_dirs:
            shard_name = os.path.basename(shard_dir)
            all_shards.add(shard_name)
    
    all_shards = sorted(list(all_shards))
    logger.info("Found %d shards: %s", len(all_shards), all_shards)
    
    # Create consolidated output for each shard
    per_shard_summaries = []
    
    for shard_name in all_shards:
        # Generate consolidated output for this shard
        shard_consolidated = consolidate_single_shard(
            shard_name, all_results, local_outputs_dir, local_input_path
        )
        
        # Save the per-shard consolidated file
        shard_output_file = os.path.join(local_outputs_dir, f"consolidated_{shard_name}_output.json")
        with open(shard_output_file, 'w') as f:
            json.dump(shard_consolidated, f, indent=2)
        
        logger.info("✅ %s consolidated output saved: %s", shard_name, shard_output_file)
        logger.info("   📊 YOLO detections: %d", shard_consolidated['shard_statistics']['total_yolo_detections'])
        logger.info("   📊 Scene descriptions: %d", shard_consolidated['shard_statistics']['total_scenes'])
        logger.info("   📊 Motion segments: %d", shard_consolidated['shard_statistics']['total_motion_segments'])
        
        per_shard_summaries.append({
            "shard_name": shard_name,
            "output_file": shard_output_file,
            "statistics": shard_consolidated["shard_statistics"]
        })
    
    # Create master summary of all per-shard outputs
    master_summary = {
        "consolidation_info": {
            "input_file": local_input_path,
            "total_shards_processed": len(all_shards),
            "views_processed": len(all_results),
            "consolidation_timestamp": datetime.now().isoformat()
        },
        "per_shard_summaries": per_shard_summaries,
        "overall_statistics": {
            "total_yolo_detections": sum(s["statistics"]["total_yolo_detections"] for s in per_shard_summaries),
            "total_scenes": sum(s["statistics"]["total_scenes"] for s in per_shard_summaries),
            "total_motion_segments": sum(s["statistics"]["total_motion_segments"] for s in per_shard_summaries),
            "total_nsfw_analyses": sum(s["statistics"]["total_nsfw_analyses"] for s in per_shard_summaries),
            "total_face_analyses": sum(s["statistics"]["total_face_analyses"] for s in per_shard_summaries),
            "total_audio_analyses": sum(s["statistics"]["total_audio_analyses"] for s in per_shard_summaries),
            "total_clap_detections": sum(s["statistics"]["total_clap_detections"] for s in per_shard_summaries)
        }
    }
    
    # Save master summary
    master_summary_file = os.path.join(local_outputs_dir, "per_shard_consolidation_summary.json")
    with open(master_summary_file, 'w') as f:
        json.dump(master_summary, f, indent=2)
    
    logger.info("✅ Master per-shard summary saved: %s", master_summary_file)
    logger.info("=" * 60)
    logger.info("PER-SHARD CONSOLIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Total shards processed: %d", len(all_shards))
    logger.info("Total views processed: %d", len(all_results))
    for shard_summary in per_shard_summaries:
        shard_name = shard_summary["shard_name"]
        stats = shard_summary["statistics"]
        logger.info("%s:", shard_name)
        logger.info("  YOLO detections: %d", stats['total_yolo_detections'])
        logger.info("  Scenes: %d", stats['total_scenes'])
        logger.info("  Motion segments: %d", stats['total_motion_segments'])
    logger.info("=" * 60)
    
    return master_summary


def consolidate_single_shard(shard_name, all_results, local_outputs_dir, local_input_path):
    """
    Create a consolidated output for a single shard, combining both views.
    
    Args:
        shard_name: Name of the shard (e.g., 'shard_1')
        all_results: List of view processing results
        local_outputs_dir: Base output directory
        local_input_path: Path to input file
        
    Returns:
        dict: Consolidated output for this specific shard
    """
    import json
    import os
    import glob
    from datetime import datetime
    
    logger.info("Creating consolidated output for %s...", shard_name)
    
    # Initialize consolidated structure for this shard
    shard_consolidated = {
        "consolidation_info": {
            "shard_name": shard_name,
            "input_file": local_input_path,
            "consolidation_timestamp": datetime.now().isoformat(),
            "views_processed": len(all_results)
        },
        "yolo_detections": [],
        "scene_detection": [],
        "nsfw_analysis": [],
        "face_analysis": [],
        "motion_analysis": [],
        "audio_analysis": [],
        "clap_detection": []
    }
    
    # Process each view for this specific shard
    for view_result in all_results:
        view_name = view_result["view"]
        view_local_dir = view_result["local_output_dir"]
        
        logger.info("  Processing %s for %s...", view_name, shard_name)
        
        # Look for this specific shard directory
        shard_dir = os.path.join(view_local_dir, shard_name)
        if not os.path.exists(shard_dir):
            logger.warning("    Warning: %s not found in %s", shard_name, view_name)
            continue
            
        logger.info("    Found %s in %s", shard_name, view_name)
        
        # 1. YOLO Detection Results
        yolo_dir = os.path.join(shard_dir, "yolo_output")
        if os.path.exists(yolo_dir):
            yolo_files = glob.glob(os.path.join(yolo_dir, "*.events.jsonl"))
            for yolo_file in yolo_files:
                yolo_events = load_jsonl_file(yolo_file)
                for event in yolo_events:
                    if "_meta" not in event:  # Skip metadata lines
                        event["source_view"] = view_name
                        event["shard"] = shard_name
                        shard_consolidated["yolo_detections"].append(event)
        
        # 2. Scene Detection Results
        scene_dir = os.path.join(shard_dir, "scene_output")
        if os.path.exists(scene_dir):
            scene_files = glob.glob(os.path.join(scene_dir, "*_scene_detection_results.json"))
            for scene_file in scene_files:
                with open(scene_file, 'r') as f:
                    scene_data = json.load(f)
                    for scene in scene_data.get("scenes", []):
                        scene["source_view"] = view_name
                        scene["shard"] = shard_name
                        scene["video_path"] = scene_data.get("video_path", "")
                        shard_consolidated["scene_detection"].append(scene)
        
        # 3. NSFW Analysis Results
        nsfw_dir = os.path.join(shard_dir, "nsfw_output")
        if os.path.exists(nsfw_dir):
            nsfw_files = glob.glob(os.path.join(nsfw_dir, "*_nsfw_results.json"))
            for nsfw_file in nsfw_files:
                with open(nsfw_file, 'r') as f:
                    nsfw_data = json.load(f)
                    nsfw_data["source_view"] = view_name
                    nsfw_data["shard"] = shard_name
                    shard_consolidated["nsfw_analysis"].append(nsfw_data)
        
        # 4. Face Analysis Results
        face_dir = os.path.join(shard_dir, "face_output")
        if os.path.exists(face_dir):
            face_files = glob.glob(os.path.join(face_dir, "*_face_results.json"))
            for face_file in face_files:
                with open(face_file, 'r') as f:
                    face_data = json.load(f)
                    face_data["source_view"] = view_name
                    face_data["shard"] = shard_name
                    shard_consolidated["face_analysis"].append(face_data)
        
        # 5. Motion Analysis Results
        motion_dir = os.path.join(shard_dir, "motion_output")
        if os.path.exists(motion_dir):
            motion_files = glob.glob(os.path.join(motion_dir, "*_motion_results.json"))
            for motion_file in motion_files:
                with open(motion_file, 'r') as f:
                    motion_data = json.load(f)
                    for segment in motion_data.get("segments", []):
                        segment["source_view"] = view_name
                        segment["shard"] = shard_name
                        shard_consolidated["motion_analysis"].append(segment)
        
        # 6. Audio Analysis Results
        audio_dir = os.path.join(shard_dir, "audio_output")
        if os.path.exists(audio_dir):
            audio_files = glob.glob(os.path.join(audio_dir, "*.json"))
            for audio_file in audio_files:
                with open(audio_file, 'r') as f:
                    audio_data = json.load(f)
                    audio_data["source_view"] = view_name
                    audio_data["shard"] = shard_name
                    shard_consolidated["audio_analysis"].append(audio_data)
        
        # 7. Clap Detection Results
        clap_dir = os.path.join(shard_dir, "clap_output")
        if os.path.exists(clap_dir):
            clap_files = glob.glob(os.path.join(clap_dir, "*.json"))
            for clap_file in clap_files:
                with open(clap_file, 'r') as f:
                    clap_data = json.load(f)
                    clap_data["source_view"] = view_name
                    clap_data["shard"] = shard_name
                    shard_consolidated["clap_detection"].append(clap_data)
    
    # Sort data by timestamp
    shard_consolidated["yolo_detections"].sort(key=lambda x: x.get("t", 0))
    shard_consolidated["scene_detection"].sort(key=lambda x: x.get("start_time", 0))
    shard_consolidated["motion_analysis"].sort(key=lambda x: x.get("start_time", 0))
    
    # Add statistics for this shard
    shard_consolidated["shard_statistics"] = {
        "total_yolo_detections": len(shard_consolidated["yolo_detections"]),
        "total_scenes": len(shard_consolidated["scene_detection"]),
        "total_motion_segments": len(shard_consolidated["motion_analysis"]),
        "total_nsfw_analyses": len(shard_consolidated["nsfw_analysis"]),
        "total_face_analyses": len(shard_consolidated["face_analysis"]),
        "total_audio_analyses": len(shard_consolidated["audio_analysis"]),
        "total_clap_detections": len(shard_consolidated["clap_detection"])
    }
    
    return shard_consolidated


def merge_overlapping_segments(timeline):
    """
    Merge overlapping segments from different views to avoid duplicate annotations.
    
    Args:
        timeline: List of flagged segments from all views
        
    Returns:
        list: Merged timeline with overlapping segments combined
    """
    if not timeline:
        return []
    
    # Sort by start time
    timeline.sort(key=lambda x: x.get("start_time", 0))
    
    merged = []
    current_segment = timeline[0].copy()
    
    for segment in timeline[1:]:
        # Check if segments overlap (allowing 1 second tolerance)
        if segment["start_time"] <= current_segment["end_time"] + 1.0:
            # Merge segments
            current_segment["end_time"] = max(current_segment["end_time"], segment["end_time"])
            
            # Combine task types and descriptions
            current_task_type = current_segment.get("task_type", "")
            if isinstance(current_task_type, list):
                current_tasks = current_task_type
            else:
                current_tasks = current_task_type.split(", ") if current_task_type else []
            
            new_task = segment.get("task_type", "")
            if isinstance(new_task, list):
                new_task = ", ".join(new_task) if new_task else ""
            
            if new_task and new_task not in current_tasks:
                current_tasks.append(new_task)
                current_segment["task_type"] = ", ".join(current_tasks)
            
            # Combine descriptions
            current_desc = current_segment.get("description", "")
            new_desc = segment.get("description", "")
            if new_desc and new_desc not in current_desc:
                current_segment["description"] = f"{current_desc}; {new_desc}"
            
            # Combine source views
            current_source_view = current_segment.get("source_view", "")
            if isinstance(current_source_view, list):
                current_views = current_source_view
            else:
                current_views = current_source_view.split(", ") if current_source_view else []
            
            new_view = segment.get("source_view", "")
            if isinstance(new_view, list):
                new_view = ", ".join(new_view) if new_view else ""
            
            if new_view and new_view not in current_views:
                current_views.append(new_view)
                current_segment["source_view"] = ", ".join(current_views)
                
            # Use higher confidence
            current_segment["confidence"] = max(
                current_segment.get("confidence", 0),
                segment.get("confidence", 0)
            )
            
            # Use higher priority
            priorities = ["low", "medium", "high"]
            current_priority = current_segment.get("priority", "medium")
            new_priority = segment.get("priority", "medium")
            if priorities.index(new_priority) > priorities.index(current_priority):
                current_segment["priority"] = new_priority
                
        else:
            # No overlap, add current segment and start new one
            merged.append(current_segment)
            current_segment = segment.copy()
    
    # Add the last segment
    merged.append(current_segment)
    
    return merged


def upload_consolidated_files_to_azure(connection_string: str, local_outputs_dir: str, final_output_prefix: str):
    """
    Upload all consolidated files from local directory to Azure Blob Storage.
    
    Args:
        connection_string: Azure Storage connection string
        local_outputs_dir: Local directory containing consolidated files
        final_output_prefix: Azure blob prefix for output files
    """
    try:
        logger.info("🚀 UPLOADING CONSOLIDATED FILES TO AZURE...")
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Parse the output container from final_output_prefix
        output_container = final_output_prefix.split('/', 1)[0]
        output_path_prefix = final_output_prefix.split('/', 1)[1] if '/' in final_output_prefix else ''
        
        container_client = blob_service_client.get_container_client(output_container)
        
        # Ensure container exists
        if not container_client.exists():
            logger.info(f"Creating container '{output_container}'...")
            container_client.create_container()
        
        uploaded_files = []
        total_size = 0
        
        # Walk through all files in local outputs directory
        for root, dirs, files in os.walk(local_outputs_dir):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                
                # Get relative path from local_outputs_dir
                relative_path = os.path.relpath(local_file_path, local_outputs_dir)
                
                # Construct blob name
                if output_path_prefix:
                    blob_name = f"{output_path_prefix}/{relative_path}".replace("\\", "/")
                else:
                    blob_name = relative_path.replace("\\", "/")
                
                # Get file size for logging
                file_size = os.path.getsize(local_file_path)
                total_size += file_size
                
                logger.info(f"📤 Uploading: {relative_path} ({_human_size(file_size)}) -> {blob_name}")
                
                # Upload file
                blob_client = container_client.get_blob_client(blob_name)
                with open(local_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                
                uploaded_files.append({
                    "local_path": relative_path,
                    "blob_name": blob_name,
                    "size_bytes": file_size
                })
        
        logger.info("✅ CONSOLIDATED FILES UPLOAD COMPLETE!")
        logger.info(f"📊 Total files uploaded: {len(uploaded_files)}")
        logger.info(f"📊 Total size uploaded: {_human_size(total_size)}")
        
        # Log consolidated files specifically
        consolidated_files = [f for f in uploaded_files if 'consolidated' in f['local_path']]
        if consolidated_files:
            logger.info("📋 CONSOLIDATED FILES UPLOADED:")
            for file_info in consolidated_files:
                logger.info(f"   - {file_info['local_path']}")
        
        return {
            "success": True,
            "uploaded_files": uploaded_files,
            "total_files": len(uploaded_files),
            "total_size_bytes": total_size
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to upload consolidated files to Azure: {e}")
        return {
            "success": False,
            "error": str(e),
            "uploaded_files": [],
            "total_files": 0,
            "total_size_bytes": 0
        }

class GPUParallelPipeline:
    """Main class for GPU-parallelized video processing pipeline."""
    
    def __init__(self, config_file: str):
        self.connection_string = get_connection_string_from_yaml(config_file)
        self.container_name, self.input_prefix = self._parse_container_and_prefix(INPUT_BLOB_PREFIX)
        self.state_actor = None
        self.available_gpus = None
        
        # if self.available_gpus == 0:
        #     raise RuntimeError("No GPUs detected. GPU-parallel processing requires at least one GPU.")
        
        logger.info(f"Initialized GPU-parallel pipeline")
    
    @staticmethod
    def _parse_container_and_prefix(blobish: str) -> tuple[str, str]:
        # Accept either "container/path/…" or just "container"
        if "/" in blobish:
            container, prefix = blobish.split("/", 1)
        else:
            container, prefix = blobish, ""
        return container, prefix
    
    def run_batch_processing(self):
        logger.info("Starting GPU-parallel batch processing")
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Ray initialized for GPU-parallel processing")
        if self.state_actor is None:
            self.state_actor = StateActor.remote(STATE_FILE)

        cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
        self.available_gpus = cluster_gpus
        logger.info(f"Ray reports {self.available_gpus} GPU(s)")
        logger.info(f"Listing blobs in container='{self.container_name}', prefix='{self.input_prefix}'")

        try:
            logger.info("Discovering unprocessed videos...")
            videos_future = discover_videos.remote(
                self.connection_string, 
                self.container_name, 
                self.input_prefix, 
                self.state_actor
            )
            unprocessed_videos = ray.get(videos_future)

            # retries from actor
            retry_videos = ray.get(self.state_actor.get_failed_videos_for_retry.remote(MAX_RETRIES))
            if retry_videos:
                logger.info(f"Adding {len(retry_videos)} failed videos for retry")
                unprocessed_videos.extend(retry_videos)

            if not unprocessed_videos:
                logger.info("No unprocessed videos found")
                return

            logger.info(f"Processing {len(unprocessed_videos)} videos with GPU parallelization")
            self._process_video_batches(unprocessed_videos)

            ray.get(self.state_actor.update_scan_timestamp.remote())
            stats = ray.get(self.state_actor.get_stats.remote())
            logger.info(f"Processing complete. Stats: {stats}")

        finally:
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown complete")

    
    def _process_video_batches(self, videos: List[str]):
        max_concurrent = min(self.available_gpus, VIDEO_BATCH_SIZE)
        for i in range(0, len(videos), max_concurrent):
            batch = videos[i:i + max_concurrent]
            logger.info(f"Processing batch {i//max_concurrent + 1}: {len(batch)} videos")

            batch_futures = []
            for j, video_path in enumerate(batch):
                fut = process_video_parallel.remote(
                    video_path,
                    self.connection_string,
                    self.container_name,
                    OUTPUT_BLOB_PREFIX,
                    self.state_actor   # <--- actor handle
                )
                batch_futures.append((video_path, fut))

            batch_results = []
            for video_path, fut in batch_futures:
                try:
                    result = ray.get(fut)
                    batch_results.append(result)
                    if result["status"] == "success":
                        logger.info(f"✅ {video_path} completed successfully on GPU {result['gpu_label']}")
                    else:
                        logger.error(f"❌ {video_path} failed on GPU {result['gpu_label']}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"❌ {video_path} failed with exception: {e}")
                    ray.get(self.state_actor.mark_video_failed.remote(video_path, str(e)))

            logger.info(
                f"Batch {i//max_concurrent + 1} completed. "
                f"{len([r for r in batch_results if r.get('status') == 'success'])} successful, "
                f"{len([r for r in batch_results if r.get('status') == 'failed'])} failed"
            )
            time.sleep(5)

def main():
    """Enhanced main entry point with GPU parallelization support."""
    start_time = datetime.now()
    logger.info(f"GPU-Parallel Pipeline started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    config_file = "blobfuse2_config.yaml"
    
    # Log configuration
    logger.info("=== GPU-Parallel Pipeline Configuration ===")
    logger.info(f"VIDEO_BATCH_SIZE: {VIDEO_BATCH_SIZE}")
    logger.info(f"GPU_PER_VIDEO: {GPU_PER_VIDEO}")
    logger.info(f"INPUT_BLOB_PREFIX: {INPUT_BLOB_PREFIX}")
    logger.info(f"OUTPUT_BLOB_PREFIX: {OUTPUT_BLOB_PREFIX}")
    logger.info(f"STATE_FILE: {STATE_FILE}")
    logger.info(f"MAX_RETRIES: {MAX_RETRIES}")
    
    try:
        # Initialize and run the pipeline
        pipeline = GPUParallelPipeline(config_file)
        pipeline.run_batch_processing()
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ GPU-Parallel Pipeline completed successfully in {total_time:.2f} seconds")
        
    except Exception as e:
        logger.critical(f"Critical error in GPU-parallel pipeline: {e}")
        sys.exit(1)
if __name__ == "__main__":
    main()