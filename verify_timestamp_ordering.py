#!/usr/bin/env python3
"""
Script to verify timestamp ordering in ViT age detection results.
"""

import json
from pathlib import Path

def verify_timestamp_ordering(results_file):
    """Verify that results are properly ordered by timestamp."""
    print(f"🔍 Verifying timestamp ordering in: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    face_results = data.get("face_analysis_results", [])
    if not face_results:
        print("❌ No face analysis results found")
        return
    
    print(f"📊 Total frames analyzed: {len(face_results)}")
    print(f"👥 Total faces detected: {data.get('total_faces_detected', 0)}")
    
    # Extract timestamps and check ordering
    timestamps = []
    for i, result in enumerate(face_results):
        timestamp = result.get("timestamp", "")
        frame_num = result.get("frame_num", 0)
        chunk_offset = result.get("chunk_offset_seconds", 0)
        num_faces = result.get("num_faces", 0)
        
        # Convert timestamp to seconds for comparison
        try:
            time_parts = timestamp.split(":")
            hours, minutes, seconds = map(int, time_parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds
            timestamps.append(total_seconds)
        except:
            print(f"⚠️ Could not parse timestamp: {timestamp}")
            continue
        
        # Show first 10 and last 5 results
        if i < 10 or i >= len(face_results) - 5:
            face_info = f"({num_faces} faces)" if num_faces > 0 else ""
            print(f"  {i+1:3d}. Frame {frame_num:3d} at {timestamp} {face_info}")
    
    # Check if timestamps are in ascending order
    is_ordered = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
    
    print(f"\n📈 Timestamp ordering validation:")
    print(f"   {'✅ Properly ordered' if is_ordered else '❌ NOT properly ordered'}")
    
    if timestamps:
        print(f"   First timestamp: {timestamps[0]}s")
        print(f"   Last timestamp: {timestamps[-1]}s")
        print(f"   Total duration: {timestamps[-1] - timestamps[0]}s")
    
    return is_ordered

def main():
    """Main function to verify all result files."""
    results_dir = Path("video-age-detection-pipeline/outputs/results")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print("🎬 ViT Age Detection Results - Timestamp Ordering Verification")
    print("=" * 70)
    
    # Find all result files
    result_files = list(results_dir.glob("*_vit_age_analysis_results.json"))
    
    if not result_files:
        print("❌ No result files found")
        return
    
    all_ordered = True
    
    for result_file in sorted(result_files):
        print(f"\n{'='*50}")
        is_ordered = verify_timestamp_ordering(result_file)
        if not is_ordered:
            all_ordered = False
    
    print(f"\n{'='*70}")
    if all_ordered:
        print("🎉 All result files have properly ordered timestamps!")
    else:
        print("⚠️ Some result files have timestamp ordering issues.")
    
    print(f"\n📁 Results directory: {results_dir.absolute()}")

if __name__ == "__main__":
    main()
