import ray
import os
import re
from typing import Any, Dict, List, Optional
import json, pandas as pd
from yolo_detection_r.utils.Tstamp import fmt_hhmmss_ms

# Assuming the script is run from the project root, so yolo_detection_r is in the path
from yolo_detection_r.utils.write_jsonl import write_jsonl
from yolo_detection_r.utils.write_jsonl import save_people_count_bins_json
from yolo_detection_r.utils.write_jsonl import save_people_presence_json

from yolo_detection_r.detection_module import detect_events_raw
from yolo_detection_r.detection_module import people_count_bins_from_events
from yolo_detection_r.detection_module import people_presence_spans_from_events
import cv2

@ray.remote(num_cpus=1, num_gpus=0.06, max_calls=1) # Can be changed to num_gpus=1 if a GPU model is used
def run_yolo_detection(
    video_path: str,
    output_dir: str,
    shard_seconds: int = 60,
    #detect
    model: str = "yolo11m.pt",
    conf: float = 0.5,
    iou: float = 0.5,
    frame_stride: int = 5,
    classes: Optional[List[int]] = [0],
    device: Optional[str] = None,
    #tracking
    enable_tracking: bool = True,              
    tracker_backend: str = "botsort",        
    tracker_cfg: Optional[str] = "config/botsort_reid.yaml",     
    #counter
    enable_count: bool = True, #new
    bin_size_sec: float = 1.0,  
    gap_sec: float = 1.0,
 ) -> Dict[str, Any]:
    """
    A Ray task to run YOLO object detection on a video shard.
    """
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    events_out_path = os.path.join(output_dir, f"{stem}_yolo_events.jsonl")
    spans_out_path = os.path.join(output_dir, f"{stem}_yolo_spans.jsonl")
    # path to saved people count JSON (if produced)
    json_path = None

    try:
        # Run raw detection
        events, tracker_used = detect_events_raw(
            video_path=video_path,
            model_name=model,
            conf=conf,
            iou=iou,
            frame_stride=frame_stride,
            classes=classes,
            device=device,
            enable_tracking=enable_tracking,
            tracker_backend=tracker_backend,
            tracker_cfg=tracker_cfg,
        )

        # If this shard corresponds to a part of a larger video, offset times
        # by shard_index * shard_seconds so results are in global timeline.
        def _part_index(filename: str) -> int:
            m = re.search(r"_part(\d+)\.mp4$", filename)
            return int(m.group(1)) if m else 0

        shard_idx = _part_index(os.path.basename(video_path))
        time_offset = float(shard_idx * shard_seconds)

        # Apply time offset to events (t and ts)
        if isinstance(events, list) and time_offset:
            for e in events:
                if "t" in e and isinstance(e["t"], (int, float)):
                    e["t"] = round(float(e["t"]) + time_offset, 3)
                    try:
                        e["ts"] = fmt_hhmmss_ms(e["t"])
                    except Exception:
                        pass

        # Write events to JSONL
        meta = {
            "source_video": video_path,
            "model": model,
            "conf": conf,
            "iou": iou,
            "frame_stride": frame_stride,
            "classes": classes,
            "device": device,
            "enable_tracking": enable_tracking,
            "tracker_backend": tracker_backend,
            "tracker_cfg": tracker_cfg,
            "tracker": tracker_used,
            "shard_index": shard_idx,
            "shard_seconds": shard_seconds,
        }
        write_jsonl(events_out_path, events, meta=meta)

        # people counting / presence spans
        spans_written = None
        if enable_count and enable_tracking:
            person_classes = {"person"}
            
            presence = people_presence_spans_from_events(events, gap_sec=gap_sec, person_classes=person_classes)
            spans_written = save_people_presence_json(presence, video_path, output_dir)
            print("Person presence saved to:", spans_written)

            counts = people_count_bins_from_events(events=events, bin_size_sec=bin_size_sec, person_classes=person_classes)

            # Override counts summary unique_people with presence-derived people_number
            try:
                if isinstance(counts, dict) and 'summary' in counts and isinstance(presence, dict) and 'summary' in presence:
                    people_num = int(presence['summary'].get('people_number', counts['summary'].get('unique_people', 0)))
                    counts['summary']['unique_id'] =  counts['summary']['unique_people']
                    counts['summary']['unique_people'] = people_num
                    # propagate min_duration_sec from presence summary into counts summary
                    if 'min_duration_sec' in presence['summary']:
                        counts['summary']['min_duration_sec'] = float(presence['summary']['min_duration_sec'])
            except Exception:
                # best-effort; if anything goes wrong, keep original counts
                pass

            json_path = save_people_count_bins_json(counts, video_path=video_path, output_dir=output_dir)
            print("People count saved to:", json_path)


        # compute aggregate stats
        num_events = len(events) if isinstance(events, list) else 0
        detections_count = 0
        if isinstance(events, list):
            for e in events:
                dets = e.get("detections") if isinstance(e, dict) else None
                if isinstance(dets, list):
                    detections_count += len(dets)

        # video metadata
        fps = None
        duration = None
        try:
            cap = cv2.VideoCapture(video_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps > 0 and frames > 0:
                duration = frames / fps
            cap.release()
        except Exception:
            fps = None
            duration = None

        result = {
            "video": video_path,
            "events_jsonl": events_out_path,
            "people_count_json": json_path,
            "spans_jsonl": spans_written,
            "num_events": num_events,
            "detections_count": detections_count,
            "video_duration": duration,
            "fps": fps,
        }

        return result

    except Exception as e:
        return {"__error__": f"run_yolo_detection failed: {e}"}

def extract_yolo_people_data(yolo_result):
    """
    Extract and structure people counting data from YOLO results.
    
    Args:
        yolo_result: Raw YOLO result dict from run_yolo_detection
        
    Returns:
        Enhanced YOLO result with extracted people analytics
    """
    if not yolo_result or yolo_result.get('__error__'):
        return yolo_result
    
    enhanced_result = yolo_result.copy()
    
    # Load people count analytics if available
    if 'people_count_json' in yolo_result and yolo_result['people_count_json']:
        try:
            import json
            with open(yolo_result['people_count_json'], 'r') as f:
                people_count_data = json.load(f)
            
            enhanced_result['people_analytics'] = {
                'count_summary': people_count_data.get('summary', {}),
                'timeline_bins': people_count_data.get('timeline_bins', []),
                'bin_size_sec': people_count_data.get('summary', {}).get('bin_size_sec', 1.0)
            }
            
            print(f"✅ Extracted people analytics: {people_count_data.get('summary', {}).get('unique_people', 0)} unique people detected")
            
        except Exception as e:
            print(f"⚠️ Failed to load people count data from {yolo_result['people_count_json']}: {e}")
    
    # Load individual presence spans if available  
    if 'spans_jsonl' in yolo_result and yolo_result['spans_jsonl']:
        try:
            import json
            with open(yolo_result['spans_jsonl'], 'r') as f:
                presence_data = json.load(f)
            enhanced_result['people_presence'] = presence_data
            
            print(f"✅ Extracted presence data for {len(presence_data)} individuals")
            
        except Exception as e:
            print(f"⚠️ Failed to load people presence data from {yolo_result['spans_jsonl']}: {e}")
    
    return enhanced_result

def generate_video_people_summary(consolidated_data):
    """
    Generate video-level people counting summary from consolidated data.
    
    Args:
        consolidated_data: Consolidated data dict containing view_results with enhanced YOLO data
        
    Returns:
        Dict with video-level people analytics summary
    """
    total_unique_people = 0
    max_simultaneous = 0
    total_avg_occupancy = 0.0
    views_with_people = 0
    processed_views = 0
    
    # Extract view_results from consolidated data
    view_results = consolidated_data.get('view_results', {})
    print(f" Generating people summary from consolidated data with {len(view_results)} view results")
    
    for view_name, view_data in view_results.items():
        print(f" Processing view: {view_name}")
        
        if not view_data.get('success', True):
            print(f"⚠️ View {view_name} not successful, skipping")
            continue
            
        processed_views += 1
        yolo_data = view_data.get('yolo', {})
        print(f"YOLO data keys for {view_name}: {list(yolo_data.keys()) if yolo_data else 'None'}")
        
        people_analytics = yolo_data.get('people_analytics', {})
        print(f" People analytics for {view_name}: {people_analytics.keys() if people_analytics else 'None'}")
        
        if people_analytics and people_analytics.get('count_summary'):
            summary = people_analytics['count_summary']
            print(f" Found people count summary for {view_name}: {summary}")
            
            # Aggregate metrics (for ERP, only one view, but keeping general structure)
            view_unique = summary.get('unique_people', 0)
            view_max_simultaneous = summary.get('max_of_max', 0)
            view_avg = summary.get('avg_of_avg', 0.0)
            
            total_unique_people = max(total_unique_people, view_unique)
            max_simultaneous = max(max_simultaneous, view_max_simultaneous)
            total_avg_occupancy += view_avg
            
            print(f" View {view_name} - Unique: {view_unique}, Max: {view_max_simultaneous}, Avg: {view_avg}")
            
            if view_unique > 0:
                views_with_people += 1
        else:
            print(f" No people analytics found for {view_name}")
    
    result = {
        "total_unique_people": total_unique_people,
        "max_simultaneous_people": max_simultaneous, 
        "avg_occupancy_across_views": round(total_avg_occupancy / max(1, views_with_people if views_with_people > 0 else 1), 3),
        "views_with_people_detected": views_with_people,
        "total_views_processed": processed_views,
        "processing_approach": "erp_single_view" if processed_views == 1 else "multi_view"
    }
    
    print(f" Final people summary: {result}")
    return result