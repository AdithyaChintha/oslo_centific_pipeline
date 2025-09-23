"""
Motion Energy Analysis Only - Minimal Version
"""

import ray
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import time

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "motion_analysis" / "src"))

# Import dependencies
try:
    from utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

try:
    from motion_analysis.src.main import MotionEnergyAnalysisPipeline
except ImportError:
    from main import MotionEnergyAnalysisPipeline

logger = get_logger("motion_energy_ray")

@ray.remote(max_calls=1)
def analyze_motion_energy_only(
    video_path: str,
    sensitivity_level: str = "medium",
    save_detailed_data: bool = False
) -> Dict[str, Any]:
    """
    Motion energy analysis only - no chunking, no conversion.
    
    Args:
        video_path: Path to MP4 video file
        sensitivity_level: "low", "medium", or "high"
        save_detailed_data: Include frame-by-frame data
        
    Returns:
        Motion analysis results
    """
    start_time = time.time()
    
    config = {
        "video_path": video_path,
        "output_directory": tempfile.mkdtemp(prefix="motion_"),
        "sensitivity_level": sensitivity_level,
        "save_annotated_video": False,
        "save_motion_data": save_detailed_data,
        "verbose_logging": False
    }
    
    try:
        # Run motion analysis
        pipeline = MotionEnergyAnalysisPipeline(config)
        pipeline.initialize_components()
        results = pipeline.analyze_single_video(video_path)
        
        # Return clean results
        return {
            "success": True,
            "video_path": video_path,
            "video_duration_seconds": results.duration,
            "processing_time_seconds": time.time() - start_time,
            "total_segments": results.segments,
            "motion_statistics": results.motion_statistics,
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "activity_type": seg.activity_type,
                    "confidence": seg.confidence,
                    "avg_motion_energy": seg.avg_motion_energy
                }
                for seg in results.segments_list
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "video_path": video_path,
            "processing_time_seconds": time.time() - start_time
        }


@ray.remote
def compute_motion_energy(shard_paths, sensitivity_level="medium", save_detailed_data=False, enable_timing=False, timing_id=None):
    """
    Process multiple video shards in parallel and combine motion energy results.
    
    Args:
        shard_paths: List of video shard file paths
        sensitivity_level: "low", "medium", or "high"
        save_detailed_data: Include frame-by-frame data
        
    Returns:
        Combined motion analysis results
    """
    start_time = time.time() if enable_timing else None
    start_time_iso = datetime.utcnow().isoformat() + "Z" if enable_timing else None
    if not shard_paths:
        error_result = {"success": False, "error": "No shard paths provided"}

        # Add timing data for early error case if timing was enabled
        if enable_timing and start_time:
            end_time = time.time()
            end_time_iso = datetime.utcnow().isoformat() + "Z"
            execution_time = end_time - start_time

            error_result["timing"] = {
                "execution_time": execution_time,
                "start_time": start_time_iso,
                "end_time": end_time_iso,
                "timing_id": timing_id,
                "status": "failed"
            }

        return error_result
    
    # Process all shards in parallel
    shard_futures = [
        analyze_motion_energy_only.remote(shard_path, sensitivity_level, save_detailed_data)
        for shard_path in shard_paths
    ]
    
    # Get all results
    shard_results = ray.get(shard_futures)
    
    # Combine results
    total_duration = 0
    total_processing_time = 0
    combined_segments = []
    motion_stats = {"total_motion_energy": 0, "avg_motion_energy": 0}
    
    shard_duration = 60  # Each shard is 60 seconds
    
    for shard_idx, result in enumerate(shard_results):
        if not result["success"]:
            continue
            
        time_offset = shard_idx * shard_duration
        total_duration += result["video_duration_seconds"]
        total_processing_time += result["processing_time_seconds"]
        
        # Adjust segment timestamps for this shard's position in timeline
        for segment in result["segments"]:
            adjusted_segment = segment.copy()
            adjusted_segment["start_time"] += time_offset
            adjusted_segment["end_time"] += time_offset
            combined_segments.append(adjusted_segment)
    
    # Sort segments by start time
    combined_segments.sort(key=lambda x: x["start_time"])

    # Build the base result dictionary
    result = {
        "success": True,
        "video_path": "combined_shards",
        "video_duration_seconds": total_duration,
        "processing_time_seconds": max([r["processing_time_seconds"] for r in shard_results if r["success"]], default=0),
        "total_segments": len(combined_segments),
        "motion_statistics": motion_stats,
        "segments": combined_segments
    }

    # Add timing data only if timing is enabled
    if enable_timing and start_time:
        end_time = time.time()
        end_time_iso = datetime.utcnow().isoformat() + "Z"
        execution_time = end_time - start_time

        result["timing"] = {
            "execution_time": execution_time,
            "start_time": start_time_iso,
            "end_time": end_time_iso,
            "timing_id": timing_id,
            "status": "completed"
        }

    return result



if __name__ == "__main__":
    # Simple test
    test_video = "/home/nvcoe_admin/code/oslo/oslo-motion-energy-analysis/input/VID_20250720_152154_00_011_view2.mp4"
    
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
    
    if not os.path.exists(test_video):
        print(f"❌ Video not found: {test_video}")
        sys.exit(1)
    
    print(f"🧪 Testing motion energy analysis...")
    print(f"📹 Video: {os.path.basename(test_video)}")
    
    ray.init()
    
    future = analyze_motion_energy_only.remote(test_video, "medium")
    result = ray.get(future)
    
    if result["success"]:
        print(f"✅ SUCCESS: {result['total_segments']} segments found")
        print(f"   Duration: {result['video_duration_seconds']:.1f}s")
        print(f"   Processing: {result['processing_time_seconds']:.1f}s")
        
        for i, seg in enumerate(result['segments'], 1):
            print(f"   {i}. {seg['start_time']:.1f}s-{seg['end_time']:.1f}s ({seg['activity_type']})")
    else:
        print(f"❌ FAILED: {result['error']}")
    
    ray.shutdown()