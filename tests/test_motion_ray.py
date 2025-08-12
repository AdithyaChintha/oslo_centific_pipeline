import ray
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ray_jobs.motion_energy import analyze_motion_energy_only

if __name__ == "__main__":
    
    # Test video path
    test_video = "/home/nvcoe_admin/code/oslo/oslo-motion-energy-analysis/input/VID_20250720_152154_00_011.mp4"
    
    if not Path(test_video).exists():
        print(f"❌ Test video not found: {test_video}")
        sys.exit(1)
    
    print("🧪 Testing Ray Motion Energy Analysis")
    print(f"📹 Video: {Path(test_video).name}")
    
    # Initialize Ray
    ray.init()
    
    try:
        # Submit motion energy analysis
        future = analyze_motion_energy_only.remote(
            video_path=test_video,
            sensitivity_level="medium",
            save_detailed_data=False
        )
        
        print("⏳ Processing...")
        result = ray.get(future)
        
        # Show results
        if result["success"]:
            print(f"✅ SUCCESS: {result['total_segments']} segments found")
            print(f"   Duration: {result['video_duration_seconds']:.1f}s")
            print(f"   Processing: {result['processing_time_seconds']:.1f}s")
            
            for i, seg in enumerate(result['segments'], 1):
                print(f"   {i}. {seg['start_time']:.1f}s-{seg['end_time']:.1f}s ({seg['activity_type']})")
        else:
            print(f"❌ FAILED: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    finally:
        ray.shutdown()
        print("✅ Test complete")