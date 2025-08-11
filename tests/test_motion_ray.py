import ray
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ray_jobs.motion_energy import compute_motion_energy, process_single_video_for_motion_energy

if __name__ == "__main__":
    
    print("🚀 Initializing Ray cluster...")
    ray.init()
    
    # Replace with a valid .mp4 file path for testing
    test_video_file = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/VID_20250720_152154_00_011.mp4"
    
    # Check if INSV file exists, convert to MP4 for testing
    test_insv_file = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/VID_20250720_152154_00_011.insv"
    
    # Determine which file to use
    if Path(test_video_file).exists():
        input_file = test_video_file
        print(f"📹 Using MP4 file: {input_file}")
    elif Path(test_insv_file).exists():
        input_file = test_insv_file
        print(f"📹 Using INSV file: {input_file}")
        print("⚠️  Note: INSV will be converted to MP4 automatically")
    else:
        print(f"❌ Neither test file found:")
        print(f"   MP4: {test_video_file}")
        print(f"   INSV: {test_insv_file}")
        print("📝 Update the file paths in this script to point to your test video")
        ray.shutdown()
        sys.exit(1)

    try:
        print("\n" + "="*60)
        print("🔍 TESTING RAY MOTION ENERGY DETECTION")
        print("="*60)
        
        # Test configuration
        config = {
            "sensitivity_level": "medium",
            "save_detailed_data": True
        }
        
        print(f"🎬 Input file: {Path(input_file).name}")
        print(f"🔧 Sensitivity: {config['sensitivity_level']}")
        print(f"📊 Save detailed data: {config['save_detailed_data']}")
        print(f"⚡ Using GPU acceleration: H100 detected")
        
        # Submit Ray task
        print(f"\n🚀 Submitting motion energy analysis task to Ray cluster...")
        future = process_single_video_for_motion_energy.remote(
            input_file, 
            config, 
            chunk_duration_seconds=30.0  # 30-second chunks for testing
        )
        
        print(f"⏳ Processing video chunks in parallel...")
        print(f"   (This may take a few minutes depending on video length)")
        
        # Get results
        result = ray.get(future)
        
        print("\n" + "="*60)
        print("📊 RAY MOTION ENERGY TEST RESULTS")
        print("="*60)
        
        if result.get("success", False):
            print(f"✅ SUCCESS: Motion energy analysis completed!")
            print(f"")
            print(f"📈 PROCESSING METRICS:")
            print(f"   🎬 Video duration: {result.get('total_video_duration_seconds', 0):.1f} seconds")
            print(f"   ⏱️  Total processing time: {result.get('total_pipeline_time_seconds', 0):.1f} seconds")
            print(f"   🚀 Processing speed: {result.get('processing_speed_factor', 0):.1f}x real-time")
            print(f"   📦 Chunks processed: {result.get('total_chunks_processed', 0)}")
            print(f"   ❌ Chunks failed: {result.get('total_chunks_failed', 0)}")
            print(f"")
            print(f"🎯 MOTION ANALYSIS RESULTS:")
            print(f"   🔍 Motion segments found: {result.get('total_motion_segments', 0)}")
            
            # Display motion statistics if available
            if result.get('overall_motion_statistics'):
                stats = result['overall_motion_statistics']
                print(f"   📊 Total frames analyzed: {stats.get('total_frames', 0)}")
                print(f"   📈 Average motion energy: {stats.get('avg_motion_energy', 0):.4f}")
                print(f"   📈 Peak motion energy: {stats.get('max_motion_energy', 0):.4f}")
                print(f"")
                print(f"🏃 ACTIVITY BREAKDOWN:")
                print(f"   🔴 High activity: {stats.get('high_activity_percentage', 0):.1f}% ({stats.get('high_activity_frames', 0)} frames)")
                print(f"   🟡 Medium activity: {stats.get('medium_activity_percentage', 0):.1f}% ({stats.get('medium_activity_frames', 0)} frames)")
                print(f"   🟢 Low activity: {stats.get('low_activity_percentage', 0):.1f}% ({stats.get('low_activity_frames', 0)} frames)")
            
            # Display top segments
            if result.get('aggregated_segments'):
                segments = result['aggregated_segments']
                print(f"")
                print(f"🎬 TOP ACTIVITY SEGMENTS:")
                # Show top 5 segments by confidence
                top_segments = sorted(segments, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
                
                for i, seg in enumerate(top_segments, 1):
                    start_time = seg.get('start_time', 0)
                    end_time = seg.get('end_time', 0)
                    duration = seg.get('duration', 0)
                    activity_type = seg.get('activity_type', 'UNKNOWN')
                    confidence = seg.get('confidence', 0)
                    
                    print(f"   {i}. {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s) | {activity_type} | Confidence: {confidence:.3f}")
            
            print(f"")
            print(f"💾 RAY TASK METADATA:")
            if result.get('metadata'):
                metadata = result['metadata']
                print(f"   🏷️  Analysis version: {metadata.get('analysis_version', 'unknown')}")
                print(f"   🕐 Timestamp: {metadata.get('timestamp', 'unknown')}")
                print(f"   🖥️  Processing node: {metadata.get('processing_node', 'unknown')}")
                print(f"   🆔 Ray task ID: {metadata.get('ray_task_id', 'unknown')[:12]}...")
            
        else:
            print(f"❌ FAILED: Motion energy analysis failed")
            error_msg = result.get('error', 'Unknown error occurred')
            print(f"🚨 Error details: {error_msg}")
            print(f"📊 Chunks processed before failure: {result.get('total_chunks_processed', 0)}")
            print(f"❌ Total chunks failed: {result.get('total_chunks_failed', 0)}")
        
        print("\n" + "="*60)
        print("🎉 RAY MOTION ENERGY TEST COMPLETED")
        print("="*60)
        
        # Print full result for debugging (optional)
        if len(sys.argv) > 1 and sys.argv[1] == "--debug":
            print(f"\n🔧 DEBUG: Full result dictionary:")
            import json
            print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n🛑 Shutting down Ray cluster...")
        ray.shutdown()
        print(f"✅ Ray cluster shutdown complete")

print(f"\n💡 USAGE TIPS:")
print(f"   • Run with --debug flag to see full results: python test_motion_ray.py --debug") 
print(f"   • Update file paths at the top of this script for your videos")
print(f"   • Check Ray dashboard at http://localhost:8265 during execution")
print(f"   • Adjust chunk_duration_seconds for different performance characteristics")