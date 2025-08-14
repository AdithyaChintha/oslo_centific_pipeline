import ray
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ray_jobs.face_age_detector import (
    process_video_for_face_detection,
    process_video_chunks_for_face_detection
)


if __name__ == "__main__":
    ray.init()

    # Test 1: Test single video processing
    print("🧪 Testing single video processing...")
    
    # Replace with a valid .mp4 file path for testing
    test_video_file = "video-age-detection-pipeline/outputs/chunks_view_1/VID_20250720_152154_00_011_view1_part0.mp4"
    
    if not Path(test_video_file).exists():
        print(f"Test video file not found: {test_video_file}")
        print("Please update the test_video_file path to a valid MP4 file")
        sys.exit(1)
    
    print(f"Testing with video: {test_video_file}")
    
    # Test single video processing
    future1 = process_video_for_face_detection.remote(
        test_video_file, 
        "/tmp/test_face_detection_single", 
        frame_interval=30, 
        save_frames=False
    )
    result1 = ray.get(future1)
    
    print("\n✅ Single video processing result:")
    print(result1)
    
    # Test 2: Test video chunks processing
    print("\n🧪 Testing video chunks processing...")
    
    # Use multiple video files as chunks for testing
    test_chunk_files = [
        "video-age-detection-pipeline/outputs/chunks_view_1/VID_20250720_152154_00_011_view1_part0.mp4",
        "video-age-detection-pipeline/outputs/chunks_view_1/VID_20250720_152154_00_011_view1_part1.mp4"
    ]
    
    # Check if chunk files exist
    existing_chunks = [f for f in test_chunk_files if Path(f).exists()]
    if not existing_chunks:
        print(f"No test chunk files found. Please update the test_chunk_files paths")
        sys.exit(1)
    
    print(f"Testing with {len(existing_chunks)} chunks: {existing_chunks}")
    
    # Test chunks processing
    future2 = process_video_chunks_for_face_detection.remote(
        existing_chunks, 
        frame_interval=30, 
        save_frames=False, 
        chunk_duration_sec=60
    )
    result2 = ray.get(future2)
    
    print("\n✅ Video chunks processing result:")
    print(result2)
    
    print("\n🎉 All tests completed!")
    ray.shutdown()
