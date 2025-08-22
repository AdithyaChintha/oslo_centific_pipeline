
import os
import json
import ray
from glob import glob
from typing import List, Dict, Any

# It's better to handle the SDK import gracefully
try:
    from label_studio_sdk import Client
except ImportError:
    Client = None

from utils.logger import get_logger

logger = get_logger("LabelStudioTasks")

@ray.remote
def build_labelstudio_json_for_shard_task(shard_output_dir: str, shard_video_path: str) -> str:
    """
    Generates a Label Studio task JSON for a single shard by consolidating results
    from various models.
    """
    segment_results = []
    
    # Scan all subdirectories in the shard output directory for JSON files
    for model_output_dir in glob(os.path.join(shard_output_dir, "*", "")):
        for file_path in glob(os.path.join(model_output_dir, "*.json")):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Logic adapted from the original generate_label_studio_json.py
                if "flagged_segments" in data:
                    for seg in data["flagged_segments"]:
                        segment_results.append({
                            "start": seg.get("start_time", 0),
                            "end": seg.get("end_time", 0),
                            "labels": [seg.get("flag_type", os.path.basename(os.path.dirname(file_path)))]
                        })
                elif "scenes" in data:
                    for seg in data["scenes"]:
                        segment_results.append({
                            "start": seg.get("start_time", 0),
                            "end": seg.get("end_time", 60), # Defaulting to 60s for scenes
                            "labels": ["scene"]
                        })
                elif "segments" in data: # For motion energy
                    for seg in data["segments"]:
                        segment_results.append({
                            "start": seg.get("start_time", 0),
                            "end": seg.get("end_time", 0),
                            "labels": [seg.get("activity_type", "motion")]
                        })

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

    # Format results into Label Studio's prediction format
    result_entries = [
        {
            "from_name": "label",
            "to_name": "video",
            "type": "labels",
            "value": {
                "start": seg["start"],
                "end": seg["end"],
                "labels": seg["labels"]
            }
        } for seg in segment_results
    ]

    # Create the task structure for this shard
    # NOTE: Using a local file path for 'video'. Label Studio needs access to this path.
    # For a real deployment, this should be a URL (e.g., from a cloud storage presigned URL).
    task = {
        "data": {
            "video": shard_video_path 
        },
        "predictions": [
            {
                "result": result_entries
            }
        ]
    }

    # Save the task to a JSON file within the shard directory
    shard_name = os.path.basename(shard_output_dir)
    save_path = os.path.join(shard_output_dir, f"{shard_name}_labelstudio_task.json")
    with open(save_path, "w") as f:
        json.dump(task, f, indent=2)
        
    logger.info(f"Label Studio task for shard '{shard_name}' saved to {save_path}")
    return save_path


@ray.remote
def import_to_labelstudio_task(
    json_task_paths: List[str], 
    label_studio_url: str, 
    api_token: str, 
    project_id: str
):
    """
    Imports tasks from a list of JSON files into a Label Studio project.
    """
    if Client is None:
        error_msg = "label-studio-sdk is required. Please install it using: pip install label-studio-sdk"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    if not api_token or api_token == "YOUR_LABELSTUDIO_TOKEN":
        error_msg = "A valid Label Studio API token is required."
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    all_tasks = []
    for path in json_task_paths:
        try:
            with open(path, 'r') as f:
                task_data = json.load(f)
                all_tasks.append(task_data)
        except Exception as e:
            logger.warning(f"Could not read or parse task JSON file {path}: {e}")
            continue
    
    if not all_tasks:
        logger.info("No valid tasks found to import.")
        return {"success": True, "message": "No tasks to import."}

    try:
        logger.info(f"Connecting to Label Studio at {label_studio_url}...")
        client = Client(url=label_studio_url, api_key=api_token)
        project = client.get_project(int(project_id))
        
        logger.info(f"Importing {len(all_tasks)} tasks to project ID {project_id}...")
        result = project.import_tasks(all_tasks)
        logger.info(f"Successfully imported tasks. Server response: {result}")
        
        return {"success": True, "result": result}

    except Exception as e:
        logger.error(f"Failed to import tasks to Label Studio: {e}")
        return {"success": False, "error": str(e)}
