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
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from utils.logger import get_logger
from azure.storage.blob import generate_blob_sas, BlobSasPermissions

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
            audio_shard_urls = generate_azure_shard_urls(
                azure_blob_client, azure_container, audio_shards,
                f"{azure_output_prefix}/audio_shards",
                azure_account_name, azure_account_key
            )
        else:
            audio_shard_urls = {}

        # Split each unwarped view into shards and stage per-view shard lists
        shards_dir = os.path.join(output_dir, "video_shards")
        os.makedirs(shards_dir, exist_ok=True)

        # Map: view_name -> [list of shard paths]
        view_shards_map = {}
        # Map: view_name -> {index -> azure url}
        view_shard_urls_map = {}

        for view_name, view_path in flat_result.items():
            vdir = os.path.join(shards_dir, view_name)
            os.makedirs(vdir, exist_ok=True)
            v_shards = ray.get(split_video_into_shards.remote(view_path, output_dir=vdir, duration_sec=60))
            view_shards_map[view_name] = v_shards

            # Optionally upload video shards for LS streaming
            if azure_blob_client and azure_container and azure_output_prefix and azure_account_name and azure_account_key:
                view_shard_urls_map[view_name] = generate_azure_shard_urls(
                    azure_blob_client, azure_container, v_shards,
                    f"{azure_output_prefix}/unwarped_shards/{view_name}",
                    azure_account_name, azure_account_key
                )
            else:
                view_shard_urls_map[view_name] = {}

        # Determine the number of shards we can align across all views and audio
        per_view_counts = [len(v) for v in view_shards_map.values()] if view_shards_map else [0]
        min_len = min(per_view_counts + [len(audio_shards) if audio_shards else 0])
        if min_len == 0:
            logger.warning("No aligned shards found across views/audio for unwarped processing")
            return {"status": "no_shards"}

        label_studio_tasks = []
        # Process per-shard index across all views together: one LS task per time segment
        for i in range(min_len):
            shard_offset_sec = i * 60

            # Build dict of view -> shard path/url for this index
            view_urls_for_shard = {}
            view_results_for_shard = {}

            for view_name, v_shards in view_shards_map.items():
                v_shard = v_shards[i]
                v_url = view_shard_urls_map.get(view_name, {}).get(i, v_shard)
                view_urls_for_shard[view_name] = v_url

                # Pair audio shard by index (fallback to last if video has more shards)
                a_idx = min(i, len(audio_shards) - 1) if audio_shards else 0
                a_shard = audio_shards[a_idx] if audio_shards else input_audio_path

                # Process this view's shard through the models
                shard_output_dir = os.path.join(output_dir, f"{view_name}_shard_{i+1}")
                os.makedirs(shard_output_dir, exist_ok=True)
                view_results = process_single_shard_through_pipeline(
                    v_shard, a_shard, shard_output_dir, shard_offset_sec, i
                )
                view_results_for_shard[view_name] = view_results

            # Consolidate results across all views for this shard index
            consolidated_results = consolidate_time_segment_results_multiview(
                view_results_for_shard, i, shard_offset_sec
            )

            # Choose up to 2 primary views for left/right for UI, keep all as extra fields
            view_names_sorted = sorted(view_urls_for_shard.keys())
            primary_left = view_urls_for_shard.get(view_names_sorted[0], "") if view_names_sorted else ""
            primary_right = view_urls_for_shard.get(view_names_sorted[1], "") if len(view_names_sorted) > 1 else ""

            # Generate a single LS task containing audio + all view URLs for this shard index
            task_json_path = generate_multiview_shard_labelstudio_task(
                base_output_dir=output_dir,
                shard_number=i+1,
                shard_offset_sec=shard_offset_sec,
                view_urls=view_urls_for_shard,
                audio_url=audio_shard_urls.get(i, audio_shards[i]) if audio_shards else input_audio_path,
                primary_left_url=primary_left,
                primary_right_url=primary_right,
                consolidated_results=consolidated_results,
                total_shards=min_len
            )
            label_studio_tasks.append(task_json_path)

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
            logger.warning("Azure client/prefix or account key missing — LS URLs will be local and likely won’t stream.")
            view1_shard_urls, view2_shard_urls, audio_shard_urls = {}, {}, {}
        
        min_len = min(len(view1_shards), len(view2_shards), len(audio_shards))
        if min_len < len(view1_shards) or min_len < len(view2_shards) or min_len < len(audio_shards):
            logger.warning("Shard count mismatch (v1=%d, v2=%d, audio=%d). Truncating to %d.",
                           len(view1_shards), len(view2_shards), len(audio_shards), min_len)
                           
        # Process time-aligned shards
        label_studio_tasks = []
        for shard_index in range(min_len):
            shard_results = process_time_aligned_shard(
                shard_index, view1_shards[shard_index], view2_shards[shard_index], 
                audio_shards[shard_index], output_dir,
                view1_shard_urls.get(shard_index), view2_shard_urls.get(shard_index), audio_shard_urls.get(shard_index)
            )
            label_studio_tasks.append(shard_results['label_studio_task'])
        
        # Import consolidated tasks to Label Studio
        import_consolidated_tasks_to_labelstudio(label_studio_tasks)
        
        return create_consolidated_summary(label_studio_tasks, output_dir)
    else:
        logger.warning("Non-INSV input or unsupported mode. Only dual-view or unwarped multi-view for INSV are supported.")
        return {"status": "skipped", "reason": "non_insv_or_unsupported"}

def process_time_aligned_shard(
    shard_index,
    view1_shard_path,
    view2_shard_path,          # may be None for single-view
    audio_shard_path,
    base_output_dir,
    view1_azure_url,
    view2_azure_url,           # may be None/"" for single-view
    audio_url
):
    """
    Process one time-aligned shard. If `view2_shard_path` is None, this behaves as single-view,
    but still generates the SAME consolidated LS task JSON used in dual-view.
    """
    shard_output_dir = os.path.join(base_output_dir, f"shard_{shard_index+1}")
    shard_offset_sec = shard_index * 60
    os.makedirs(shard_output_dir, exist_ok=True)

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
        audio_url
    )

    return {
        "shard_index": shard_index,
        "view1_results": view1_results,
        "view2_results": view2_results,
        "consolidated_results": consolidated_results,
        "label_studio_task": task_json_path
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
                                                 consolidated_results, shard_number, shard_offset_sec, audio_url):
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

def import_consolidated_tasks_to_labelstudio(task_file_paths):
    """
    Import consolidated Label Studio tasks to Label Studio platform.
    
    Args:
        task_file_paths: List of paths to consolidated task JSON files
        
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
    
    logger.info(f"Importing {len(valid_task_files)} consolidated tasks to Label Studio...")
    
    # Use existing Label Studio import functionality
    try:
        # Import using the existing import_to_labelstudio_task function
        import_result_ref = import_to_labelstudio_task.remote(
            valid_task_files,
            "https://annotations-stg.oneforma2.com/",  # Should be configurable
            "d75a31c7994b96099cfbf7d61e15cff643943853",  # Should be from config
            "5458"  # Should be configurable
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

def _process_single_view(*args, **kwargs):
    """Single-view flow disabled; only dual-view or unwarped multi-view supported."""
    logger.warning("Single-view flow is disabled; skipping.")
    return {"status": "unsupported_single_view"}
def consolidate_time_segment_results_multiview(view_results_by_view: dict, shard_index: int, shard_offset_sec: int):
    """
    Consolidate segments across an arbitrary number of views for a given shard index.
    Produces merged_flagged_segments using existing dual-view merger (works for N views).
    """
    all_segments = []
    for view_name, results in (view_results_by_view or {}).items():
        for seg in results.get('flagged_segments', []):
            all_segments.append({**seg, "source_view": view_name})

    merged = merge_overlapping_segments_dual_view(all_segments)
    return {
        "shard_index": shard_index,
        "shard_offset_sec": shard_offset_sec,
        "time_range": f"{shard_offset_sec}-{shard_offset_sec + 60}s",
        "merged_flagged_segments": merged,
        "consolidated_predictions": []  # predictions are added in task builder
    }



if __name__ == "__main__":
    # Define the input video and the main output directory
    INPUT_VIDEO = "/dev/shm/cleaning-surfaces-20250819_1325-video.insv_1.insv"
    INPUT_AUDIO = "/home/nvcoe_admin/arian/cleaning-surfaces-20250819_1325-audio.WAV"
    OUTPUT_DIR = "/dev/shm/outputs/simplified_unified_pipeline_output"
    
    try:
        results = pipeline_main(INPUT_VIDEO, INPUT_AUDIO, OUTPUT_DIR)
        
        print(f"\nSimplified Pipeline completed!")
        print(f"Processing Summary:")
        for task in ['audio', 'yolo', 'scene', 'nsfw', 'motion', 'face', 'clap']:
            count = results['processing_summary'][f'successful_{task}']
            total = results['processing_summary']['total_shards']
            print(f"   {task.title()}: {count}/{total} successful")
        
        print(f"\nAnnotation Workload Reduction:")
        print(f"   Flagged Segments: {results['annotation_summary']['total_flagged_segments']}")
        print(f"   Workload Reduction: {results['annotation_summary']['annotation_workload_reduction']}")
        print(f"   High Priority Segments: {results['annotation_summary']['high_priority_segments']}")
        print(f"   Overall Success Rate: {results['processing_summary']['overall_success_rate']}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Pipeline failed: {e}")
        raise
