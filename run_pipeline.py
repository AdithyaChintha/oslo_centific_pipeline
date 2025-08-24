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

# Import Azure blob utilities (download_blob kept for potential future use)
from utils.azure_blob_utils import download_blob

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AzurePipelineWrapper")


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


def main():
    """
    The main entry point for running the Ray pipeline with Azure Blob I/O.
    Enhanced with blob listing and selection functionality from test_video_splitter.py
    """
    start_ts = datetime.now()
    logger.info("Pipeline started at %s", start_ts.strftime("%Y-%m-%d %H:%M:%S"))
    
    # ==================================================================
    # == 1. CONFIGURATION - MANUAL VIDEO PATH INPUT                   ==
    # ==================================================================
    # EDIT THIS: Specify your exact video path in blob storage
    # Format: "container-name/full/path/to/video.insv"
    input_path = "instavideo/azure_directory_path/DCIM/Camera01/VID_20250809_094753_00_044.insv"
    
    # The PARENT directory for your output in Azure Blob Storage
    # Format: "container-name/path/for/all/outputs/"
    output_parent_dir = "instavideo/krishna-test/test1/test_activity/pre-annotation-output"
    # ==================================================================


    # ==================================================================
    # == 2. PREPARE PIPELINE WITH MANUAL VIDEO PATH                   ==
    # ==================================================================
    config_file = "blobfuse2_config.yaml"
    try:
        connection_string = get_connection_string_from_yaml(config_file)
    except (ValueError, FileNotFoundError):
        sys.exit(1)
    
    logger.info("Using manually specified video: %s", input_path)
    
    # Get the base name of the video file (e.g., "my_cool_video.mp4")
    video_filename = os.path.basename(input_path.split('/', 1)[1])  # Extract filename from blob path
    # Get the video name without the extension (e.g., "my_cool_video")
    video_name_folder = os.path.splitext(video_filename)[0]

    # Create the final, specific output path by joining the parent dir and the video name.
    # The result will be e.g., "results/all-pipeline-runs/my_cool_video"
    final_output_prefix = os.path.join(output_parent_dir, video_name_folder)
    # ==================================================================

    logger.info(f"Input video: {input_path}")
    logger.info(f"Final output will be stored under: {final_output_prefix}")
    
    try:
        # The wrapper is initialized with the final, specific output path
        with AzureBlobPipeline(connection_string, input_path, final_output_prefix) as (local_input_path, local_output_dir):
            logger.info("Azure Blob environment is ready. Starting Ray pipeline...")
            
            # ==================================================================
            # == 3. DUAL-VIEW PROCESSING FOR INSV FILES                       ==
            # ==================================================================
            # Create local outputs directory in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            local_outputs_dir = os.path.join(script_dir, "outputs", video_name_folder)
            os.makedirs(local_outputs_dir, exist_ok=True)
            logger.info(f"Local outputs will also be saved to: {local_outputs_dir}")
            
            # Check if input is INSV file for dual-view processing
            if local_input_path.lower().endswith('.insv'):
                logger.info("INSV file detected - will process both views")
                
                # Import the conversion function
                from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4
                import ray
                
                # Initialize Ray with proper error handling
                try:
                    if not ray.is_initialized():
                        ray.init()
                except RuntimeError as e:
                    if "ray.init twice" in str(e):
                        logger.warning("Ray already initialized, continuing...")
                    else:
                        raise
                
                # Convert INSV to dual MP4
                logger.info("Converting INSV to dual MP4...")
                t0 = time.perf_counter()
                conversion_result = ray.get(convert_insv_to_dual_mp4.remote(local_input_path))
                conv_time = time.perf_counter() - t0
                
                if not conversion_result.get('success', False):
                    raise RuntimeError(f"INSV conversion failed: {conversion_result.get('error', 'Unknown error')}")
                
                # Get both view paths
                view1_path = conversion_result['output_view_1']
                view2_path = conversion_result['output_view_2']
                logger.info("INSV converted to dual MP4 in %.2fs", conv_time)
                logger.info("View 1: %s", view1_path)
                logger.info("View 2: %s", view2_path)
                
                # Process both views
                mp4_paths = [view1_path, view2_path]
                view_names = ["view_1", "view_2"]
                
                all_results = []
                
                for i, (mp4_path, view_name) in enumerate(zip(mp4_paths, view_names)):
                    logger.info("="*60)
                    logger.info("PROCESSING %s (%d/2)", view_name.upper(), i+1)
                    logger.info("="*60)
                    
                    # Create view-specific output directories
                    view_temp_output_dir = os.path.join(local_output_dir, view_name)
                    view_local_output_dir = os.path.join(local_outputs_dir, view_name)
                    os.makedirs(view_temp_output_dir, exist_ok=True)
                    os.makedirs(view_local_output_dir, exist_ok=True)
                    
                    # Run the pipeline for this view
                    logger.info("Running pipeline for %s...", view_name)
                    pipeline_main(
                        input_video_path=mp4_path,
                        output_dir=view_temp_output_dir
                    )
                    
                    # Copy view outputs to local directory
                    if os.path.exists(view_temp_output_dir) and os.listdir(view_temp_output_dir):
                        logger.info("Copying %s outputs to local directory...", view_name)
                        copy_start = time.perf_counter()
                        for item in os.listdir(view_temp_output_dir):
                            src_path = os.path.join(view_temp_output_dir, item)
                            dst_path = os.path.join(view_local_output_dir, item)
                            if os.path.isdir(src_path):
                                if os.path.exists(dst_path):
                                    shutil.rmtree(dst_path)
                                shutil.copytree(src_path, dst_path)
                            else:
                                shutil.copy2(src_path, dst_path)
                        copy_time = time.perf_counter() - copy_start
                        logger.info("%s local copy completed in %.2fs", view_name, copy_time)
                        
                        all_results.append({
                            "view": view_name,
                            "mp4_path": mp4_path,
                            "temp_output_dir": view_temp_output_dir,
                            "local_output_dir": view_local_output_dir
                        })
                    else:
                        logger.warning("No outputs to copy for %s", view_name)
                
                # ==============================================================
                # == 4. CONSOLIDATE OUTPUTS FROM BOTH VIEWS                  ==
                # ==============================================================
                logger.info("="*60)
                logger.info("CONSOLIDATING OUTPUTS FROM BOTH VIEWS")
                logger.info("="*60)
                
                consolidated_summary = consolidate_dual_view_outputs(
                    all_results, local_outputs_dir, conv_time, local_input_path
                )
                
                # Consolidate all individual model outputs  
                logger.info("Consolidating all model outputs from both views...")
                consolidated_model_outputs = consolidate_all_model_outputs(
                    all_results, local_outputs_dir, local_input_path
                )
                
                # Reorganize and generate per-shard consolidated outputs
                logger.info("Reorganizing output structure and creating per-shard consolidated outputs...")
                per_shard_summary = reorganize_and_consolidate_per_shard(
                    all_results, local_outputs_dir, local_input_path
                )
                
                # Create a summary for dual-view processing
                dual_view_summary = {
                    "input_file": local_input_path,
                    "file_type": "INSV",
                    "views_processed": len(all_results),
                    "conversion_time_seconds": conv_time,
                    "view_results": all_results,
                    "consolidated_summary": consolidated_summary,
                    "consolidated_model_outputs_summary": {
                        "total_yolo_detections": consolidated_model_outputs["consolidated_statistics"]["total_yolo_detections"],
                        "total_scenes": consolidated_model_outputs["consolidated_statistics"]["total_scenes"],
                        "total_motion_segments": consolidated_model_outputs["consolidated_statistics"]["total_motion_segments"],
                        "total_nsfw_detections": consolidated_model_outputs["consolidated_statistics"]["total_nsfw_detections"],
                        "total_faces_detected": consolidated_model_outputs["consolidated_statistics"]["total_faces_detected"]
                    },
                    "per_shard_reorganization_summary": {
                        "total_shards_reorganized": per_shard_summary["reorganization_info"]["total_shards_reorganized"],
                        "per_shard_files_generated": len(per_shard_summary["shard_summaries"]),
                        "overall_statistics": per_shard_summary["overall_statistics"]
                    }
                }
                
                # Save dual-view summary
                summary_file = os.path.join(local_outputs_dir, "dual_view_processing_summary.json")
                with open(summary_file, 'w') as f:
                    import json
                    json.dump(dual_view_summary, f, indent=2)
                
                logger.info("="*60)
                logger.info("DUAL-VIEW PROCESSING COMPLETE")
                logger.info("="*60)
                logger.info("Views processed: %d", dual_view_summary['views_processed'])
                logger.info("Conversion time: %.2fs", dual_view_summary['conversion_time_seconds'])
                logger.info("Consolidated segments: %d", consolidated_summary['total_consolidated_segments'])
                logger.info("🎯 CONSOLIDATED MODEL RESULTS:")
                logger.info("  YOLO detections: %d", consolidated_model_outputs["consolidated_statistics"]["total_yolo_detections"])
                logger.info("  Scene descriptions: %d", consolidated_model_outputs["consolidated_statistics"]["total_scenes"])
                logger.info("  Motion segments: %d", consolidated_model_outputs["consolidated_statistics"]["total_motion_segments"])
                logger.info("  NSFW detections: %d", consolidated_model_outputs["consolidated_statistics"]["total_nsfw_detections"])
                logger.info("  Face detections: %d", consolidated_model_outputs["consolidated_statistics"]["total_faces_detected"])
                logger.info("Summary saved to: %s", summary_file)
                
            else:
                # Single view processing (regular MP4)
                logger.info("Single MP4 file - processing single view")
                
                # Run the pipeline
                pipeline_main(
                    input_video_path=local_input_path,
                    output_dir=local_output_dir
                )
                
                # Copy outputs to local directory as well
                if os.path.exists(local_output_dir) and os.listdir(local_output_dir):
                    logger.info("Copying pipeline outputs to local directory...")
                    copy_start = time.perf_counter()
                    for item in os.listdir(local_output_dir):
                        src_path = os.path.join(local_output_dir, item)
                        dst_path = os.path.join(local_outputs_dir, item)
                        if os.path.isdir(src_path):
                            if os.path.exists(dst_path):
                                shutil.rmtree(dst_path)
                            shutil.copytree(src_path, dst_path)
                        else:
                            shutil.copy2(src_path, dst_path)
                    copy_time = time.perf_counter() - copy_start
                    logger.info("Local copy completed in %.2fs", copy_time)
                else:
                    logger.warning("No outputs to copy to local directory")

    except Exception as e:
        logger.critical(f"An unhandled error occurred in the pipeline wrapper: {e}", exc_info=True)
        sys.exit(1)
    
    total_time = (datetime.now() - start_ts).total_seconds()
    logger.info("✅ Pipeline completed successfully in %.2fs", total_time)

if __name__ == "__main__":
    main()