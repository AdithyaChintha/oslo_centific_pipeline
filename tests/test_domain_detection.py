#!/usr/bin/env python3
"""
Test script for domain detection functionality
"""

import os
import sys
import json
import ray

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ray_jobs.domain_detection import process_scene_domain_classification, test_domain_classification

def test_with_existing_scene_file():
    """Test domain detection with an existing scene detection file"""
    
    # Path to an existing scene detection file
    scene_file_path = "/home/nvcoe_admin/pavan/pr48/insta360-video-activity-segmentation/outtrymain/video_VID_20250809_094836_00_045_20250905_125548/shard_1/front/scene_output/front_1440x1440_part0_scene_detection_results.json"
    
    if not os.path.exists(scene_file_path):
        print(f"❌ Scene file not found: {scene_file_path}")
        return False
    
    print(f"📁 Testing with scene file: {os.path.basename(scene_file_path)}")
    
    # Read the original file
    with open(scene_file_path, 'r') as f:
        original_data = json.load(f)
    
    print(f"📊 Original scenes: {len(original_data.get('scenes', []))}")
    for i, scene in enumerate(original_data.get('scenes', [])):
        print(f"   Scene {i+1}: {scene.get('description', 'No description')[:100]}...")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Test domain classification
    scene_output_dir = os.path.dirname(scene_file_path)
    result = ray.get(process_scene_domain_classification.remote(scene_output_dir))
    
    if result.get('success', False):
        print(f"✅ Domain classification successful!")
        print(f"   📊 Processed files: {result.get('processed_files', 0)}")
        print(f"   📊 Total scenes: {result.get('total_scenes', 0)}")
        print(f"   📊 Classified scenes: {result.get('classified_scenes', 0)}")
        
        # Read the updated file
        with open(scene_file_path, 'r') as f:
            updated_data = json.load(f)
        
        print(f"\n🎯 Updated scenes with domain classification:")
        for i, scene in enumerate(updated_data.get('scenes', [])):
            domain = scene.get('domain', 'No domain')
            description = scene.get('description', 'No description')
            print(f"   Scene {i+1}: [{domain}] {description[:80]}...")
        
        return True
    else:
        print(f"❌ Domain classification failed: {result.get('error', 'Unknown error')}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Domain Detection Functionality")
    print("=" * 50)
    
    # Test 1: Basic classification test
    print("\n1️⃣ Testing basic domain classification...")
    test_domain_classification()
    
    # Test 2: Test with existing scene file
    print("\n2️⃣ Testing with existing scene detection file...")
    success = test_with_existing_scene_file()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    # Shutdown Ray
    if ray.is_initialized():
        ray.shutdown()

if __name__ == "__main__":
    main()
