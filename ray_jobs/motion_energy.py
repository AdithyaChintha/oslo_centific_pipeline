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

@ray.remote
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