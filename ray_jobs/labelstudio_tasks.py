import os
import json
import ray
from glob import glob
from typing import List, Dict, Any, Union
from datetime import datetime

# It's better to handle the SDK import gracefully
try:
    from label_studio_sdk import Client
except ImportError:
    Client = None

from utils.logger import get_logger

logger = get_logger("LabelStudioTasks")


def merge_segments_for_labelstudio(segments: List[Dict[str, Any]], merge_threshold: float = 1.0) -> List[Dict[str, Any]]:
    """
    Merges overlapping or very close segments and combines their labels.
    Adapted for the Label Studio data structure.
    """
    if not segments:
        return []

    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x['start'])
    merged = []

    for segment in sorted_segments:
        if not merged:
            merged.append(segment)
            continue

        last_segment = merged[-1]

        # Check if segments overlap or are very close
        if segment['start'] <= last_segment['end'] + merge_threshold:
            # Merge segments by extending the end time
            last_segment['end'] = max(last_segment['end'], segment['end'])

            # Combine labels, avoiding duplicates
            for label in segment['labels']:
                if label not in last_segment['labels']:
                    last_segment['labels'].append(label)
        else:
            merged.append(segment)

    return merged


@ray.remote
def build_labelstudio_json_for_shard_task(shard_output_dir: str, shard_video_url: str, shard_number: int = 1, shard_offset_sec: int = 0) -> str:
    """
    Generates a Label Studio task JSON for a single shard by consolidating and merging
    results from various models into a common timeline.
    
    Args:
        shard_output_dir: Directory containing analysis results
        shard_video_url: Azure Blob URL to the video shard (not local path)
        shard_number: Shard number (1-based)
        shard_offset_sec: Offset in seconds for this shard
    """
    
    # Initialize prediction data
    predictions = {
        'taxonomy_domain_actions': [],
        'pii': [],
        'minors': [],
        'nudity': [],
        'lighting': ['Bright light'],  # Default
        'no_signal_flag': []
    }
    
    # Activity keyword mapping for taxonomy - map to leaf values only
    activity_mapping = {
        'vacuuming': 'Vacuuming / sweeping',
        'vacuum': 'Vacuuming / sweeping', 
        'cleaning': 'Tidying surfaces',
        'cooking': 'Cooking / baking',
        'kitchen': 'Cooking / baking',
        'eating': 'Eating together',
        'walking': 'Walking between rooms',
        'moving': 'Walking between rooms',
        'hallway': 'Walking between rooms',
        'table': 'Fetching objects',
        'box': 'Fetching objects'
    }
    
    # Scan all subdirectories in the shard output directory for JSON files
    for model_output_dir in glob(os.path.join(shard_output_dir, "*", "")):
        model_type = os.path.basename(model_output_dir.rstrip('/'))
        
        for file_path in glob(os.path.join(model_output_dir, "*.json")):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Process scene detection results
                if model_type == "scene_output" and "scenes" in data:
                    for scene in data["scenes"]:
                        description = scene.get("description", "").lower()
                        for keyword, leaf_action in activity_mapping.items():
                            if keyword in description:
                                if leaf_action not in predictions['taxonomy_domain_actions']:
                                    predictions['taxonomy_domain_actions'].append(leaf_action)
                                break  # Take first match
                
                # Process audio PII results
                elif model_type == "audio_output" and "pii_detections" in data:
                    pii_list = data.get("pii_detections", [])
                    if pii_list:  # If any PII detected
                        # Map based on common PII types in audio
                        if "Full names" not in predictions['pii']:
                            predictions['pii'].append("Full names")
                
                # Process NSFW results
                elif model_type == "nsfw_output" and "total_nsfw_detections" in data:
                    nsfw_count = data.get("total_nsfw_detections", 0)
                    if nsfw_count > 0:
                        if "Nudity present" not in predictions['nudity']:
                            predictions['nudity'].append("Nudity present")
                
                # Process face detection results (check for minors)
                elif model_type == "face_output" and "flagged_segments" in data:
                    flagged_segments = data.get("flagged_segments", [])
                    for segment in flagged_segments:
                        flag_type = segment.get("flag_type", "").lower()
                        if "minor" in flag_type or "child" in flag_type:
                            if "Video" not in predictions['minors']:
                                predictions['minors'].append("Video")
                
                # Process motion energy for lighting assessment
                elif model_type == "motion_output" and "segments" in data:
                    segments = data.get("segments", [])
                    if segments:
                        # Check average motion energy to infer lighting
                        avg_motion = sum(seg.get("avg_motion_energy", 0) for seg in segments) / len(segments)
                        if avg_motion < 0.01:  # Very low motion might indicate low light
                            predictions['lighting'] = ['Low light']
                        else:
                            predictions['lighting'] = ['Bright light']
                    
                    # Check for periods of no activity (potential no signal)
                    total_duration = 60  # Shard duration
                    active_duration = sum(seg.get("duration", 0) for seg in segments)
                    if active_duration < total_duration * 0.1:  # Less than 10% activity
                        if "No Signal present" not in predictions['no_signal_flag']:
                            predictions['no_signal_flag'].append("No Signal present")

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

    # Create Label Studio prediction entries
    result_entries = []
    
    # Taxonomy prediction (if any domain detected)
    if predictions['taxonomy_domain_actions']:
        result_entries.append({
            "from_name": "taxonomy_domain_actions",
            "to_name": "video_left",
            "type": "taxonomy",
            "value": {
                "taxonomy": predictions['taxonomy_domain_actions'][:1]  # Use leaf values directly
            }
        })
    
    # PII predictions
    if predictions['pii']:
        result_entries.append({
            "from_name": "pii",
            "to_name": "video_left",
            "type": "choices",
            "value": {
                "choices": predictions['pii']
            }
        })
    
    # Minors predictions
    if predictions['minors']:
        result_entries.append({
            "from_name": "minors",
            "to_name": "video_left",
            "type": "choices",
            "value": {
                "choices": predictions['minors']
            }
        })
    
    # Nudity predictions
    if predictions['nudity']:
        result_entries.append({
            "from_name": "nudity",
            "to_name": "video_left",
            "type": "choices",
            "value": {
                "choices": predictions['nudity']
            }
        })
    
    # No signal predictions
    if predictions['no_signal_flag']:
        result_entries.append({
            "from_name": "no_signal_flag",
            "to_name": "video_left",
            "type": "choices",
            "value": {
                "choices": predictions['no_signal_flag']
            }
        })
    
    # Lighting prediction (always required)
    result_entries.append({
        "from_name": "lighting",
        "to_name": "video_left",
        "type": "choices",
        "value": {
            "choices": predictions['lighting']
        }
    })

    # Generate current timestamp for recording datetime
    current_time = datetime.utcnow().isoformat() + "Z"

    # Create the task structure for this shard with CORRECT structure for Label Studio
    # Based on successful example: needs both empty "meta" field AND flattened meta.* keys
    task = {
        "data": {
            "video_left": shard_video_url,  # Azure Blob URL
            "video_right": shard_video_url,  # Same video for both views
            "meta": "",  # Required empty meta field
            # Flattened metadata keys at root level to match UI template expectations
            "meta.home_identifier": f"Shard_{shard_number}",
            "meta.recording_datetime": current_time,
            "metadata_domain": "production",  # Required by Label Studio API
            "meta.actions": "",
            "home_id": "",
            "start_datetime": "",
            "end_datetime": "",
            "total_duration": "",
            "files_deleted": [],
            # Additional metadata (these won't show in UI but good for context)
            "shard_number": str(shard_number),
            "shard_offset_seconds": str(shard_offset_sec), 
            "segments_detected": str(len(result_entries)),
            # ERP video and audio URLs
            "full_video_480P": "",  # URL to full ERP video downscaled to 480p
            "full_audio_link": ""   # URL to full session audio file
        },
        "annotations":[],
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
        
    logger.info(f"Label Studio task for shard '{shard_name}' saved to {save_path} with {len(result_entries)} predictions.")
    logger.info(f"Predictions generated: taxonomy={len(predictions['taxonomy_domain_actions'])}, pii={len(predictions['pii'])}, nudity={len(predictions['nudity'])}")
    logger.info(f"Video URL: {shard_video_url}")
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
        
        # Log the first task structure for debugging
        if all_tasks:
            logger.info("Sample task structure:")
            sample_task = all_tasks[0]
            logger.info(f"Data keys: {list(sample_task['data'].keys())}")
            if sample_task.get('predictions') and sample_task['predictions'][0].get('result'):
                logger.info(f"First prediction: {sample_task['predictions'][0]['result'][0] if sample_task['predictions'][0]['result'] else 'No results'}")
        
        result = project.import_tasks(all_tasks)
        logger.info(f"Successfully imported tasks. Server response: {result}")
        
        return {"success": True, "result": result}

    except Exception as e:
        logger.error(f"Failed to import tasks to Label Studio: {e}")
        return {"success": False, "error": str(e)}


def _collect_segments_from_consolidated(consolidated: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a consolidated shard JSON (produced by the unified pipeline) into a common
    list of time segments with labels: [{start, end, labels:[..]}].

    This function is intentionally tolerant to missing sections.
    """
    segments: List[Dict[str, Any]] = []

    subfolders = consolidated.get("subfolders", {})

    # 1) Scene boundaries (semantic scene descriptions)
    scene_sf = subfolders.get("scene_output") or {}
    try:
        for _, blob in (scene_sf.get("data") or {}).items():
            content = blob.get("content") or {}
            for sc in (content.get("scenes") or []):
                start = float(sc.get("start_time", 0))
                end = float(sc.get("end_time", max(0.0, start)))
                segments.append({"start": start, "end": end, "labels": ["scene"]})
    except Exception as e:
        logger.warning(f"scene_output parse error: {e}")

    # 2) Motion-energy segments
    motion_sf = subfolders.get("motion_output") or {}
    try:
        for _, blob in (motion_sf.get("data") or {}).items():
            content = blob.get("content") or {}
            for m in (content.get("segments") or []):
                start = float(m.get("start_time", 0))
                end = float(m.get("end_time", max(0.0, start)))
                label = m.get("activity_type", "motion")
                segments.append({"start": start, "end": end, "labels": [str(label)]})
    except Exception as e:
        logger.warning(f"motion_output parse error: {e}")

    # 3) Face/PII/NSFW flagged ranges (use model-name as label if provided)
    face_sf = subfolders.get("face_output") or {}
    try:
        for _, blob in (face_sf.get("data") or {}).items():
            content = blob.get("content") or {}
            for fs in (content.get("flagged_segments") or []):
                start = float(fs.get("start_time", 0))
                end = float(fs.get("end_time", max(0.0, start)))
                flag = fs.get("flag_type") or "face_flag"
                segments.append({"start": start, "end": end, "labels": [str(flag)]})
    except Exception as e:
        logger.warning(f"face_output parse error: {e}")

    nsfw_sf = subfolders.get("nsfw_output") or {}
    try:
        # Some NSFW pipelines only provide counts; skip if no segment structure is present.
        for _, blob in (nsfw_sf.get("data") or {}).items():
            content = blob.get("content") or {}
            for s in (content.get("flagged_segments") or []):
                start = float(s.get("start_time", 0))
                end = float(s.get("end_time", max(0.0, start)))
                segments.append({"start": start, "end": end, "labels": ["nsfw"]})
    except Exception as e:
        logger.warning(f"nsfw_output parse error: {e}")

    # 4) YOLO detections → quick time grouping into contiguous ~1s windows per class
    # Consolidated format stores YOLO jsonl parsed into a list
    yolo_sf = subfolders.get("yolo_output") or {}
    try:
        for _, blob in (yolo_sf.get("data") or {}).items():
            content = blob.get("content")
            if not isinstance(content, list):
                continue
            # Filter detection rows (those with 't' and 'cls')
            dets = [r for r in content if isinstance(r, dict) and 't' in r and 'cls' in r]
            # Group by class label and form coarse segments by 1.0s adjacency
            by_class: Dict[str, List[float]] = {}
            for r in dets:
                try:
                    by_class.setdefault(str(r['cls']), []).append(float(r['t']))
                except Exception:
                    continue
            for cls, times in by_class.items():
                times.sort()
                if not times:
                    continue
                seg_start = times[0]
                prev = times[0]
                for t in times[1:]:
                    if t - prev > 1.0:  # gap opens a new segment
                        segments.append({"start": seg_start, "end": prev, "labels": [f"object:{cls}"]})
                        seg_start = t
                    prev = t
                # last segment
                segments.append({"start": seg_start, "end": prev, "labels": [f"object:{cls}"]})
    except Exception as e:
        logger.warning(f"yolo_output parse error: {e}")

    # 5) Audio PII spans (if available as list of segments)
    audio_sf = subfolders.get("audio_output") or {}
    try:
        for name, blob in (audio_sf.get("data") or {}).items():
            if not name.endswith("pii_detections.json"):
                continue
            content = blob.get("content")
            # Some runs return [] if none found; otherwise expect {pii_segments:[{start_time,end_time,label}]}
            if isinstance(content, dict):
                for ps in (content.get("pii_segments") or []):
                    start = float(ps.get("start_time", 0))
                    end = float(ps.get("end_time", max(0.0, start)))
                    label = ps.get("label") or "pii"
                    segments.append({"start": start, "end": end, "labels": [str(label)]})
    except Exception as e:
        logger.warning(f"audio_output (pii) parse error: {e}")

    return segments


def _collect_framewise_labels_from_consolidated(consolidated: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Collect frame-wise labels from consolidated data.

    Returns a dict mapping frame number to list of labels.
    """
    frame_labels: Dict[int, List[str]] = {}
    subfolders = consolidated.get("subfolders", {})

    # 1) YOLO detections - add "object:{cls}" per frame
    yolo_sf = subfolders.get("yolo_output") or {}
    try:
        for _, blob in (yolo_sf.get("data") or {}).items():
            content = blob.get("content")
            if not isinstance(content, list):
                continue
            for r in content:
                if not isinstance(r, dict):
                    continue
                frame = r.get("frame")
                cls = r.get("cls")
                if frame is not None and cls is not None:
                    try:
                        frame_int = int(frame)
                        label = f"object:{cls}"
                        frame_labels.setdefault(frame_int, []).append(label)
                    except Exception:
                        continue
    except Exception as e:
        logger.warning(f"framewise yolo_output parse error: {e}")

    # 2) NSFW output - flagged_segments with frames or frame_ids
    nsfw_sf = subfolders.get("nsfw_output") or {}
    try:
        for _, blob in (nsfw_sf.get("data") or {}).items():
            content = blob.get("content") or {}
            flagged_segments = content.get("flagged_segments") or []
            for seg in flagged_segments:
                frames = seg.get("frames") or seg.get("frame_ids")
                if isinstance(frames, list):
                    for f in frames:
                        try:
                            frame_int = int(f)
                            frame_labels.setdefault(frame_int, []).append("nsfw")
                        except Exception:
                            continue
                # else: skip if only start_time/end_time present
    except Exception as e:
        logger.warning(f"framewise nsfw_output parse error: {e}")

    # 3) Face/age/minors outputs
    # Face output
    face_sf = subfolders.get("face_output") or {}
    try:
        for _, blob in (face_sf.get("data") or {}).items():
            content = blob.get("content") or {}
            flagged_segments = content.get("flagged_segments") or []
            for fs in flagged_segments:
                frames = fs.get("frames") or fs.get("frame_ids")
                flag_type = fs.get("flag_type") or "face_flag"
                if isinstance(frames, list):
                    for f in frames:
                        try:
                            frame_int = int(f)
                            frame_labels.setdefault(frame_int, []).append(str(flag_type))
                        except Exception:
                            continue
    except Exception as e:
        logger.warning(f"framewise face_output parse error: {e}")

    # Age output (optional)
    age_sf = subfolders.get("age_output") or subfolders.get("age_detector_output") or {}
    try:
        for _, blob in (age_sf.get("data") or {}).items():
            content = blob.get("content") or {}
            # flagged_segments with frames
            flagged_segments = content.get("flagged_segments") or []
            for fs in flagged_segments:
                frames = fs.get("frames") or fs.get("frame_ids")
                if isinstance(frames, list):
                    for f in frames:
                        try:
                            frame_int = int(f)
                            frame_labels.setdefault(frame_int, []).append("minor")
                        except Exception:
                            continue
            # age_estimates list with 'frame' and 'is_minor'
            age_estimates = content.get("age_estimates") or []
            for est in age_estimates:
                try:
                    frame = est.get("frame")
                    is_minor = est.get("is_minor")
                    if frame is not None and is_minor:
                        frame_int = int(frame)
                        frame_labels.setdefault(frame_int, []).append("age:minor")
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"framewise age_output parse error: {e}")

    # 4) Audio PII framewise detections
    audio_sf = subfolders.get("audio_output") or {}
    try:
        for name, blob in (audio_sf.get("data") or {}).items():
            if not name.endswith("pii_detections.json"):
                continue
            content = blob.get("content")
            if isinstance(content, dict):
                pii_detections = content.get("pii_detections") or []
                for det in pii_detections:
                    frame = det.get("frame")
                    label = det.get("label") or "pii"
                    if frame is not None:
                        try:
                            frame_int = int(frame)
                            frame_labels.setdefault(frame_int, []).append(str(label))
                        except Exception:
                            continue
    except Exception as e:
        logger.warning(f"framewise audio_output (pii) parse error: {e}")

    # Deduplicate labels per frame
    for frame in frame_labels:
        frame_labels[frame] = list(set(frame_labels[frame]))

    return frame_labels


def build_labelstudio_tasks_from_consolidated(
    consolidated_json_path: str,
    azure_video_url: Union[str, Dict[str, str]],
    app_type: str = "scenario",
    metadata: Union[Dict[str, Any], None] = None,
    merge_threshold: float = 1.0,
    save_dir: Union[str, None] = None,
) -> str:
    """
    Build a Label Studio *tasks* JSON file from a consolidated shard JSON.

    Parameters
    ----------
    consolidated_json_path : str
        Path to consolidated json (e.g., `shard_1_consolidated.json`).
    azure_video_url : str or dict
        Azure Blob SAS URL to the video for this shard (Label Studio will read this).
        If dict, expected keys are "left" and "right".
    app_type : str
        One of {"scenario", "geolocation"}. Used only to adjust data keys/metadata.
    metadata : dict
        Any extra fields to expose under `data["meta"]` for the right-pane templates. Examples:
        {
          "home_id": "H001-XYZ",
          "start_datetime": "2025-08-21T06:50:33Z",
          "end_datetime": "2025-08-21T06:51:33Z",
          "total_video_duration": 60.0,
          "files_deleted": "N",
          "room": "Kitchen",
          ...
        }
    merge_threshold : float
        Max gap (seconds) to join adjacent/overlapping segments.
    save_dir : str | None
        Where to write the generated tasks JSON. Default: alongside the consolidated json.

    Returns
    -------
    str : path to the written tasks JSON file.
    """

    # Load consolidated
    with open(consolidated_json_path, "r") as f:
        consolidated = json.load(f)

    # Collect and merge segments
    raw_segments = _collect_segments_from_consolidated(consolidated)
    merged_segments = merge_segments_for_labelstudio(raw_segments, merge_threshold=merge_threshold)

    # Collect framewise labels
    frame_labels = _collect_framewise_labels_from_consolidated(consolidated)

    # Build prediction entries

    # (A) Frame-wise label results for video_left
    framewise_results = [
        {
            "from_name": "label",
            "to_name": "video_left",
            "type": "labels",
            "value": {"frame": int(frame), "labels": sorted(set(labels))}
        }
        for frame, labels in sorted(frame_labels.items())
    ]

    # (B) Time-span label results for video_left (existing structure)
    timespan_results = [
        {
            "from_name": "label",
            "to_name": "video_left",
            "type": "labels",
            "value": {"start": seg["start"], "end": seg["end"], "labels": seg["labels"]},
        }
        for seg in merged_segments
    ]

    prediction_results = framewise_results + timespan_results

    # Data payload: build data_block according to azure_video_url type
    if isinstance(azure_video_url, dict):
        data_block = {
            "video_left": azure_video_url.get("left"),
            "video_right": azure_video_url.get("right"),
        }
    else:
        data_block = {
            "video_left": azure_video_url,
            "video_right": azure_video_url,
        }

    # Always set app_type
    data_block["app_type"] = app_type

    # Metadata under "meta" key if provided
    if metadata:
        data_block["meta"] = metadata

    # One task per consolidated json (shard)
    tasks_payload = [
        {
            "data": data_block,
            "predictions": [{"result": prediction_results}],
        }
    ]

    # Write
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(consolidated_json_path))
    base = os.path.splitext(os.path.basename(consolidated_json_path))[0]
    out_path = os.path.join(save_dir, f"{base}_labelstudio_tasks_{app_type}.json")
    with open(out_path, "w") as f:
        json.dump(tasks_payload, f, indent=2)

    logger.info(
        f"Generated Label Studio tasks: {out_path} | segments_in={len(raw_segments)} merged={len(merged_segments)} framewise_labels={len(frame_labels)}"
    )
    return out_path

def assign_views_to_labelstudio_positions(view_results, view_azure_urls):
    """
    Assign all available views to Label Studio 4-view positions: 
    video_top, video_left, video_right, video_bottom.
    
    Args:
        view_results: Dict of {view_name: view_result}
        view_azure_urls: Dict of {view_name: azure_url}
    
    Returns:
        Dict with view assignments for all 4 positions
    """
    available_views = list(view_results.keys())
    
    # Initialize positions
    positions = {
        'top': None,
        'left': None, 
        'right': None,
        'bottom': None
    }
    
    # Smart assignment based on view names and available views
    assignment_rules = {
        'top': ['front', 'forward', 'top', 'view1', 'view_1'],
        'left': ['left', 'side_left', 'view2', 'view_2'],
        'right': ['right', 'side_right', 'view3', 'view_3'], 
        'bottom': ['back', 'rear', 'bottom', 'view4', 'view_4']
    }
    
    # First pass: Assign based on naming patterns
    used_views = set()
    for position, keywords in assignment_rules.items():
        for view_name in available_views:
            if view_name in used_views:
                continue
            view_lower = view_name.lower()
            if any(keyword in view_lower for keyword in keywords):
                positions[position] = {
                    'view': view_name,
                    'url': view_azure_urls.get(view_name, ""),
                    'result': view_results.get(view_name, {})
                }
                used_views.add(view_name)
                break
    
    # Second pass: Fill remaining positions with available views
    remaining_views = [v for v in available_views if v not in used_views]
    empty_positions = [pos for pos, assignment in positions.items() if assignment is None]
    
    for i, position in enumerate(empty_positions):
        if i < len(remaining_views):
            view_name = remaining_views[i]
            positions[position] = {
                'view': view_name,
                'url': view_azure_urls.get(view_name, ""),
                'result': view_results.get(view_name, {})
            }
    
    # Log the assignments
    logger.info(f"📊 View-to-position assignments:")
    for position, assignment in positions.items():
        view_name = assignment['view'] if assignment else 'None'
        logger.info(f"   {position}: {view_name}")
    
    return positions

def generate_multiview_4view_labelstudio_task(shard_output_dir, assigned_views, 
                                            consolidated_results, shard_number, shard_offset_sec, 
                                            audio_url, total_shards=None, video_name=None, all_view_urls=None, session_metadata=None):
    """
    Generate Label Studio task with 4-view display: video_top, video_left, video_right, video_bottom.
    Follows the format from complete-tast.txt for 4-view UI support.
    
    Args:
        shard_output_dir: Output directory for this shard
        assigned_views: Dict with view assignments for all 4 positions (top, left, right, bottom)
        consolidated_results: Consolidated results from all views
        shard_number: Current shard number
        shard_offset_sec: Time offset in seconds
        audio_url: Audio file URL
        total_shards: Total number of shards
        video_name: Original video name
        all_view_urls: Dict of all view URLs for metadata
        
    Returns:
        Path to generated Label Studio task JSON file
    """
    current_time = datetime.utcnow().isoformat() + "Z"
    
    # Get prediction entries and assign to 4-view positions
    raw_predictions = consolidated_results.get('consolidated_predictions_for_ls', [])
    # prediction_entries = assign_predictions_to_4view_positions(raw_predictions, assigned_views)
    
    # Extract lighting prediction from consolidated results
    lighting_prediction = "Normal light"  # Default fallback
    for pred in raw_predictions:
        if pred.get("from_name") == "lighting" and pred.get("type") == "choices":
            choices = pred.get("value", {}).get("choices", [])
            if choices:
                lighting_prediction = choices[0]
                break

    signal_quality_prediction = extract_signal_quality_prediction_from_consolidated(consolidated_results)
    sensitive_prediction = extract_sensitive_prediction_from_consolidated(consolidated_results)
    pii_prediction = extract_pii_prediction_from_consolidated(consolidated_results)
    nsfw_prediction = extract_nsfw_prediction_from_consolidated(consolidated_results)
    minor_prediction = extract_minor_prediction_from_consolidated(consolidated_results)
    people_count_prediction = extract_people_count_prediction_from_consolidated(consolidated_results)
    absent_participant_prediction = extract_absent_participant_prediction_from_consolidated(consolidated_results)
    # Extract URLs from assigned views, using fallbacks if positions are empty
    video_top = assigned_views.get('top', {}).get('url', '') if assigned_views.get('top') else ''
    video_left = assigned_views.get('left', {}).get('url', '') if assigned_views.get('left') else ''
    video_right = assigned_views.get('right', {}).get('url', '') if assigned_views.get('right') else ''
    video_bottom = assigned_views.get('bottom', {}).get('url', '') if assigned_views.get('bottom') else ''
    
    # Extract metadata fields with fallbacks
    if session_metadata:
        home_id = session_metadata.get("home_id", "")
        moderator_id = session_metadata.get("moderator_id", "")
        start_datetime = session_metadata.get("start_datetime", "")
        end_datetime = session_metadata.get("end_datetime", "")
        
        # Format duration
        duration_minutes = session_metadata.get("duration_minutes")
        if duration_minutes:
            hours, minutes = divmod(int(duration_minutes), 60)
            if hours > 0:
                total_duration = f"{hours}h {minutes}m"
            else:
                total_duration = f"{minutes}m"
        else:
            total_duration = ""
        
        # Combine activity fields
        activity = session_metadata.get("activity", "")
        specific_activity = session_metadata.get("specific_activity", "")
        if activity and specific_activity:
            metadata_activity = specific_activity
        elif activity:
            metadata_activity = activity
        else:
            metadata_activity = ""
        
        # Extract other metadata fields
        metadata_domain = session_metadata.get("domain", "production")
        metadata_participants = len(session_metadata.get("participants", []))
        metadata_room = session_metadata.get("room", "")
        metadata_lighting = session_metadata.get("day_night", "")
        
        logger.info(f"🏷️ Metadata integrated for Label Studio task:")
        logger.info(f"   Home ID: {home_id}")
        logger.info(f"   Activity: {metadata_activity}")
        logger.info(f"   Domain: {metadata_domain}")
        logger.info(f"   Duration: {total_duration}")
    else:
        # Default values when no metadata available
        home_id = ""
        start_datetime = ""
        end_datetime = ""
        total_duration = ""
        metadata_domain = "production"
        metadata_activity = ""
        metadata_participant = ""
        metadata_room = ""
        metadata_lighting = ""
        logger.warning("⚠️ No session metadata available, using default values")
    
    # TODO: Extract files_deleted from delete_files_request if available
    # For now, using empty array as default
    files_deleted = []

    # Create task with 4-view display format

    task = {
        "data": {
            # Core 4-view video URLs for Label Studio UI
            "video_top": video_top,
            "video_left": video_left,
            "video_right": video_right,
            "video_bottom": video_bottom,
            "audio": audio_url,
            
            # Metadata (following complete-tast.txt structure)
            "meta": "",  # Required empty meta field
            
            # ========================
            # METADATA FIELDS - UPDATED
            # ========================
            "home_id": home_id,                           # ← NEW: From metadata.home_id
            "start_datetime": start_datetime,             # ← NEW: From metadata.start_datetime
            "end_datetime": end_datetime,                 # ← NEW: From metadata.end_datetime
            "total_duration": total_duration,             # ← NEW: From metadata.duration_minutes (formatted)
            "files_deleted": files_deleted,               # ← NEW: From delete_files_request (future)
            "metadata_domain": metadata_domain,           # ← UPDATED: From metadata.domain
            "metadata_activity": metadata_activity,       # ← NEW: From metadata.activity + specific_activity
            "metadata_participant": metadata_participants, # ← NEW: From metadata.participant_id
            "metadata_room": metadata_room,               # ← NEW: From metadata.room
            "metadata_lighting": metadata_lighting,       # ← NEW: From metadata.day_night
            
            # Existing fields (keep as-is)
            "meta.actions": "",
            "meta.home_identifier": f"Shard_{shard_number}",
            "meta.recording_datetime": current_time,
            "AI_predicted_domain": "",
            "AI_predicted_activity": "",
            "AI_lighting_prediction": lighting_prediction,
            "AI_signal_prediction": signal_quality_prediction,
            "AI_sensitive_prediction": sensitive_prediction,
            "AI_PII_prediction": pii_prediction,
            "AI_NSFW_prediction": nsfw_prediction,
            "AI_minor_prediction": minor_prediction,
            "AI_prediction_participants_number": people_count_prediction,
            "AI_prediction_absent_participant": str(absent_participant_prediction),
            "shard_number": str(shard_number),
            "shard_offset_seconds": str(shard_offset_sec), 
            "total_shards": total_shards or 1,
            "video_name": video_name or "",
            "segments_detected": str(len(raw_predictions)),
            "processing_type": "multi_view_4view_equal",
            "successful_views": len([v for v in assigned_views.values() if v.get('url')]),
            "total_views_processed": len(assigned_views),
            "full_video_link":"",
            "full_audio_link":"",
            "moderator_id": moderator_id
        },
        "predictions": [{
            "model_version": "multi_view_4view_v1.0",
            "result": raw_predictions
        }] if raw_predictions else []
    }
    

    # Save Label Studio task
    task_file_path = os.path.join(shard_output_dir, f"shard_{shard_number}_labelstudio_task.json")
    with open(task_file_path, 'w') as f:
        json.dump(task, f, indent=2)
    
    logger.info(f"📋 Generated 4-view Label Studio task: {task_file_path}")
    return task_file_path

def import_consolidated_tasks_to_labelstudio(task_file_paths, pipeline_config=None, is_walkthrough = False):
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
    if not is_walkthrough:
        project_id = label_studio_config.get('project_id', '5458')
    else:
        project_id = label_studio_config.get('walkthrough_project_id', '5458')
    
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
                "metadata_domain": "production",  # Required by Label Studio API
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
                "video_top":"",
                "video_bottom":"",
                "video_left": view1_azure_url,
                "video_right": view2_azure_url,
                "audio": audio_url,
                "home_id": "",
                "start_datetime": "",
                "end_datetime": "",
                "total_duration": "",
                "files_deleted": [],
                # ERP video and audio URLs
                "full_video_480P": "",  # URL to full ERP video downscaled to 480p
                "full_audio_link": ""   # URL to full session audio file
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

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("consolidated", help="Path to consolidated shard json")
    p.add_argument("--app", choices=["scenario", "geolocation"], default="scenario")
    p.add_argument("--left", required=True, help="Azure SAS URL for left view")
    p.add_argument("--right", required=True, help="Azure SAS URL for right view")
    p.add_argument("--out", default=None, help="Output directory (defaults to alongside consolidated)")
    p.add_argument("--merge-threshold", type=float, default=1.0)
    args = p.parse_args()

    # Example minimal metadata surface for templates
    meta = {
        # these keys are expected in templates as $meta.*
        "home_identifier": "H-UNKNOWN",
        "recording_datetime": "",
    }

    out_path = build_labelstudio_tasks_from_consolidated(
        consolidated_json_path=args.consolidated,
        azure_video_url={"left": args.left, "right": args.right},
        app_type=args.app,
        metadata=meta,
        merge_threshold=args.merge_threshold,
        save_dir=args.out,
    )
    print(out_path)

@ray.remote  
def update_labelstudio_tasks_with_video_domain_and_activity(video_output_dir: str, video_domain: str, video_activity: str, video_name: str) -> Dict:
    """
    Ray task to update ALL Label Studio task files with the video-level predicted domain and activity.
    
    This function finds all Label Studio task JSON files for a video and updates their
    metadata_domain and metadata_activity fields with the video-level classifications.
    
    Args:
        video_output_dir: Base output directory containing all shards for this video
        video_domain: The predicted domain for the entire video
        video_activity: The predicted activity for the entire video  
        video_name: Name of the video being processed
        
    Returns:
        Dictionary with update results
    """
    try:
        logger.info(f"📝 Updating Label Studio tasks with video classifications for: {video_name}")
        logger.info(f"   🎯 Domain: '{video_domain}'")
        logger.info(f"   🎬 Activity: '{video_activity}'")
        
        # Find all Label Studio task files across all shards
        task_files = []
        for root, dirs, files in os.walk(video_output_dir):
            for file in files:
                if file.endswith('_labelstudio_task.json'):
                    task_file_path = os.path.join(root, file)
                    task_files.append(task_file_path)
        
        if not task_files:
            logger.warning(f"No Label Studio task files found for video: {video_name}")
            return {"success": False, "error": "No task files found"}
        
        logger.info(f"📁 Found {len(task_files)} Label Studio task files to update")
        
        updated_files = []
        failed_files = []
        
        for task_file in task_files:
            try:
                # Read existing task
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                
                # Update metadata_domain and metadata_activity fields
                if 'data' in task_data:
                    old_domain = task_data['data'].get('AI_predicted_domain', 'unknown')
                    old_activity = task_data['data'].get('AI_predicted_activity', 'unknown')
                    
                    task_data['data']['AI_predicted_domain'] = video_domain
                    task_data['data']['AI_predicted_activity'] = video_activity  
                    
                    # Add video-level classification info to metadata for tracking
                    task_data['data']['video_level_domain_applied'] = True
                    task_data['data']['video_level_activity_applied'] = True
                    task_data['data']['video_level_classification_timestamp'] = datetime.utcnow().isoformat() + "Z"
                    
                    # Write updated task back to file
                    with open(task_file, 'w') as f:
                        json.dump(task_data, f, indent=2)
                    
                    updated_files.append(task_file)
                    logger.info(f"✅ Updated {os.path.basename(task_file)}: Domain '{old_domain}' -> '{video_domain}', Activity '{old_activity}' -> '{video_activity}'")
                else:
                    logger.error(f"❌ Invalid task structure in {task_file}: missing 'data' field")
                    failed_files.append(task_file)
                    
            except Exception as e:
                logger.error(f"❌ Error updating task file {task_file}: {e}")
                failed_files.append(task_file)
                continue
        
        result = {
            "success": len(updated_files) > 0,
            "video_name": video_name,
            "video_domain": video_domain,
            "video_activity": video_activity,
            "total_task_files": len(task_files),
            "updated_files": len(updated_files),
            "failed_files": len(failed_files),
            "updated_file_paths": updated_files,
            "failed_file_paths": failed_files
        }
        
        logger.info(f"🎉 Label Studio task update completed for video '{video_name}':")
        logger.info(f"   ✅ Updated: {len(updated_files)}/{len(task_files)} files")
        logger.info(f"   🎯 Applied domain: '{video_domain}'")
        logger.info(f"   🎬 Applied activity: '{video_activity}'")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to update Label Studio tasks for video '{video_name}': {e}")
        return {"success": False, "error": str(e)}
    
def extract_signal_quality_prediction_from_consolidated(consolidated_results):
    """
    Extract signal quality prediction from consolidated results for AI prediction field.
    
    Returns:
        str: Simple prediction of detected signal issues
    """
    view_results = consolidated_results.get('view_results', {})
    issue_types = set()
    
    for view_name, view_result in view_results.items():
        if view_result.get('success', True):
            signal_quality = view_result.get('signal_quality', {})
            
            if signal_quality.get('blur_segments'):
                issue_types.add('Blur')
            if signal_quality.get('black_segments'):
                issue_types.add('Black screen')
    
    if not issue_types:
        return "No signal issues"
    
    return ', '.join(sorted(issue_types))

def extract_sensitive_prediction_from_consolidated(consolidated_results):
    """
    Extract sensitive information prediction from consolidated results for AI prediction field.
    
    Returns:
        str: List of detected sensitive topics or "No sensitive content"
    """
    sensitive_results = consolidated_results.get('sensitive', [])
    sensitive_topics = set()
    
    # Handle list format from audio processing
    if isinstance(sensitive_results, list) and len(sensitive_results) > 0:
        for sensitive_result in sensitive_results:
            if sensitive_result.get('summary', {}).get('has_sensitive_content', False):
                summary_topics = sensitive_result.get('summary', {}).get('sensitive_topics', [])
                sensitive_topics.update(summary_topics)
    
    if not sensitive_topics:
        return "No sensitive content"
    
    return ', '.join(sorted(sensitive_topics))

def extract_pii_prediction_from_consolidated(consolidated_results):
    """
    Extract PII prediction from consolidated results for AI prediction field.
    
    Returns:
        str: PII detection status
    """
    # Check predictions for PII detection
    predictions = consolidated_results.get('consolidated_predictions_for_ls', [])
    for prediction in predictions:
        if (prediction.get('from_name') == 'val_pii_audio' and 
            prediction.get('type') == 'choices' and 
            'Yes' in prediction.get('value', {}).get('choices', [])):
            return "PII detected"
    
    return "No PII detected"

def extract_nsfw_prediction_from_consolidated(consolidated_results):
    """
    Extract NSFW prediction from consolidated results for AI prediction field.
    
    Returns:
        str: NSFW detection status
    """
    # Check predictions for NSFW detection
    predictions = consolidated_results.get('consolidated_predictions_for_ls', [])
    for prediction in predictions:
        if (prediction.get('from_name') == 'val_nudity_video' and 
            prediction.get('type') == 'choices' and 
            'Yes' in prediction.get('value', {}).get('choices', [])):
            return "NSFW content detected"
    
    return "No NSFW content detected"

def extract_minor_prediction_from_consolidated(consolidated_results):
    """
    Extract minor detection prediction from consolidated results for AI prediction field.
    
    Returns:
        str: Minor detection status
    """
    # Check predictions for minor detection
    predictions = consolidated_results.get('consolidated_predictions_for_ls', [])
    for prediction in predictions:
        if (prediction.get('from_name') == 'val_minors_video' and 
            prediction.get('type') == 'choices' and 
            'Yes' in prediction.get('value', {}).get('choices', [])):
            return "Minors detected"
    
    return "No minors detected"

def extract_absent_participant_prediction_from_consolidated(consolidated_results):
    predictions = consolidated_results.get('consolidated_predictions_for_ls', [])
    for prediction in predictions:
        if (prediction.get('from_name') == 'val_absent' and 
            prediction.get('type') == 'choices' and 
            'Yes' in prediction.get('value', {}).get('choices', [])):
            return True
    
    return False

def extract_people_count_prediction_from_consolidated(consolidated_results):
    """
    Extract people count prediction from consolidated results for AI_prediction_participants_number field.
    Updated to match the data structure from create_multiview_consolidated_predictions().
    
    Args:
        consolidated_results: Consolidated results from pipeline processing
        
    Returns:
        str: Number of unique people detected or "0" if none detected
    """
    if not consolidated_results or not isinstance(consolidated_results, dict):
        return "0"
    
    # Check for people data in consolidated predictions (new structure)
    predictions = consolidated_results.get('consolidated_predictions_for_ls', [])
    
    # Look for people_analytics prediction
    for prediction in predictions:
        if (prediction.get('type') == 'people_analytics' and 
            prediction.get('from_name') == 'people_counter'):
            unique_people = prediction.get('value', {}).get('unique_people', 0)
            if unique_people > 0:
                return str(unique_people)
    
    # Fallback: Check video-level people summary (legacy format)
    people_summary = consolidated_results.get('people_summary', {})
    if people_summary and people_summary.get('total_unique_people', 0) > 0:
        return str(people_summary['total_unique_people'])
    
    # Fallback: Check view-level YOLO results directly
    view_results = consolidated_results.get('view_results', {})
    max_unique_people = 0
    
    for view_name, view_result in view_results.items():
        if not view_result.get('success', True):
            continue
            
        yolo_data = view_result.get('yolo', {})
        people_analytics = yolo_data.get('people_analytics', {})
        
        if people_analytics and people_analytics.get('count_summary'):
            unique_people = people_analytics['count_summary'].get('unique_people', 0)
            max_unique_people = max(max_unique_people, unique_people)
    
    return str(max_unique_people) if max_unique_people > 0 else "0"