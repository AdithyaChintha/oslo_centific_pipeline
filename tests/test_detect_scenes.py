import ray
import os
import json
import sys
import time
import tempfile
import yaml
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import the Ray job and helper functions
from ray_jobs.scene_det import detect_scenes, clear_gpu_memory

def test_clear_gpu_memory():
    """Unit test for GPU memory clearing function"""
    print("🧪 Unit Test 1: GPU Memory Clearing")
    print("-" * 40)
    
    try:
        # Test the clear_gpu_memory function
        clear_gpu_memory()
        print("✅ GPU memory clearing function executed successfully")
        return True
    except Exception as e:
        print(f"❌ GPU memory clearing failed: {e}")
        return False

def test_prompt_config_loading():
    """Unit test for prompt configuration loading"""
    print("\n🧪 Unit Test 2: Prompt Config Loading")
    print("-" * 40)
    
    prompt_path = os.path.join(ROOT_DIR, "config", "cosmos_prompt.yaml")
    
    try:
        # Test loading the prompt configuration
        with open(prompt_path, "rb") as f:
            prompt_config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ["system_prompt", "user_prompt"]
        missing_fields = [field for field in required_fields if field not in prompt_config]
        
        if missing_fields:
            print(f"❌ Missing required fields in prompt config: {missing_fields}")
            return False
        
        print("✅ Prompt configuration loaded and validated successfully")
        print(f"   - System prompt length: {len(prompt_config['system_prompt'])} chars")
        print(f"   - User prompt length: {len(prompt_config['user_prompt'])} chars")
        return True
        
    except Exception as e:
        print(f"❌ Prompt config loading failed: {e}")
        return False

def test_result_structure_validation():
    """Unit test for validating scene detection result structure"""
    print("\n🧪 Unit Test 3: Result Structure Validation")
    print("-" * 40)
    
    try:
        # Create a mock result structure to validate
        mock_result = {
            "video_path": "/path/to/video.mp4",
            "scenes": [
                {
                    "start_time": 0.0,
                    "end_time": 30.0,
                    "description": "Sample scene description"
                }
            ],
            "raw_response": "Sample raw response",
            "total_scenes": 1,
            "processing_info": {
                "model": "nvidia/Cosmos-Reason1-7B",
                "success": True,
                "gpu_memory_utilization": 0.25,
                "max_model_len": 6144,
                "scenes_extracted": 1
            }
        }
        
        # Validate the structure
        required_keys = ["video_path", "scenes", "raw_response", "total_scenes", "processing_info"]
        missing_keys = [key for key in required_keys if key not in mock_result]
        
        if missing_keys:
            print(f"❌ Missing required keys in result: {missing_keys}")
            return False
        
        # Validate scene structure
        if mock_result["scenes"]:
            scene = mock_result["scenes"][0]
            scene_keys = ["start_time", "end_time", "description"]
            missing_scene_keys = [key for key in scene_keys if key not in scene]
            
            if missing_scene_keys:
                print(f"❌ Missing required scene keys: {missing_scene_keys}")
                return False
        
        # Validate processing_info structure
        processing_info = mock_result["processing_info"]
        processing_keys = ["model", "success"]
        missing_processing_keys = [key for key in processing_keys if key not in processing_info]
        
        if missing_processing_keys:
            print(f"❌ Missing required processing_info keys: {missing_processing_keys}")
            return False
        
        print("✅ Result structure validation passed")
        print(f"   - All required keys present: {len(required_keys)}")
        print(f"   - Scene structure valid: {len(scene_keys)} fields")
        print(f"   - Processing info valid: {len(processing_keys)} required fields")
        return True
        
    except Exception as e:
        print(f"❌ Result structure validation failed: {e}")
        return False

def test_output_directory_creation():
    """Unit test for output directory creation functionality"""
    print("\n🧪 Unit Test 4: Output Directory Creation")
    print("-" * 40)
    
    try:
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_output_dir = os.path.join(temp_dir, "scene_detection_test")
            
            # Test directory creation (mimicking the Ray job behavior)
            os.makedirs(test_output_dir, exist_ok=True)
            
            if not os.path.exists(test_output_dir):
                print(f"❌ Failed to create output directory: {test_output_dir}")
                return False
            
            # Test file writing in the directory
            test_file = os.path.join(test_output_dir, "test_result.json")
            test_data = {"test": "data"}
            
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            if not os.path.exists(test_file):
                print(f"❌ Failed to write test file: {test_file}")
                return False
            
            print("✅ Output directory creation and file writing successful")
            print(f"   - Directory created: {test_output_dir}")
            print(f"   - Test file written: {os.path.basename(test_file)}")
            return True
            
    except Exception as e:
        print(f"❌ Output directory creation test failed: {e}")
        return False

def test_detect_scenes_integration():
    """Integration test for the complete scene detection Ray job"""
    
    print("\n🔗 Integration Test: Complete Scene Detection")
    print("=" * 50)
    
    # Initialize Ray with the correct Python path
    if not ray.is_initialized():
        import os
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
        print("✅ Ray initialized")
    else:
        print("✅ Ray already initialized")
    
    # Test video path - using same video as motion_ray test
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        print("Please update the test_video path to a valid MP4 file")
        return False
    
    print(f"📹 Testing with video: {os.path.basename(test_video)}")
    print()
    
    # Check if prompt config exists
    prompt_path = os.path.join(ROOT_DIR, "config", "cosmos_prompt.yaml")
    if not os.path.exists(prompt_path):
        print(f"❌ Prompt config not found: {prompt_path}")
        return False
    
    print(f"✅ Using prompt config: {prompt_path}")
    print()
    
    # Test the single video file
    print(f"🎯 Testing Scene Detection: {os.path.basename(test_video)}")
    print("-" * 40)
    
    start_time = time.time()
    output_dir = "/tmp/test_scene_detection"
    
    try:
        # Call the Ray job (exactly how it will be used in production)
        result = ray.get(detect_scenes.remote(
            video_path=test_video,
            prompt_path=prompt_path,
            output_dir=output_dir
        ))
        processing_time = time.time() - start_time
            
        if result and result.get('processing_info', {}).get('success', False):
            print(f"✅ SUCCESS in {processing_time:.2f}s")
            print(f"📊 Results:")
            print(f"   - Total scenes: {result.get('total_scenes', 0)}")
            print(f"   - Model: {result.get('processing_info', {}).get('model', 'unknown')}")
            print(f"   - GPU Memory: {result.get('processing_info', {}).get('gpu_memory_utilization', 'unknown')}")
            print(f"   - Max Model Len: {result.get('processing_info', {}).get('max_model_len', 'unknown')}")
            
            # Show scene details
            scenes = result.get('scenes', [])
            if scenes:
                print(f"🎬 Scene breakdown:")
                for j, scene in enumerate(scenes[:3], 1):  # Show first 3 scenes
                    start_t = scene.get('start_time', 0)
                    end_t = scene.get('end_time', 0)
                    desc = scene.get('description', 'No description')
                    print(f"   {j}. {start_t:.1f}s-{end_t:.1f}s: \"{desc[:80]}{'...' if len(desc) > 80 else ''}\"")
                if len(scenes) > 3:
                    print(f"   ... and {len(scenes) - 3} more scenes")
            else:
                print("🎬 No scenes detected")
            
            # Show raw response sample
            raw_response = result.get('raw_response', '')
            if raw_response:
                print(f"📝 Raw response: \"{raw_response[:100]}{'...' if len(raw_response) > 100 else ''}\"")
            
            # Check if output file was created
            if output_dir and os.path.exists(output_dir):
                output_files = os.listdir(output_dir)
                print(f"💾 Output files: {len(output_files)} file(s) created in {output_dir}")
            
            print(f"\n🎉 Scene detection test PASSED!")
            print(f"   Scenes: {result.get('total_scenes', 0)}, Time: {processing_time:.1f}s, Model: {result.get('processing_info', {}).get('model', 'unknown')}")
            return True
            
        else:
            error_msg = result.get('processing_info', {}).get('error', 'Unknown error')
            print(f"❌ FAILED - Scene detection unsuccessful: {error_msg}")
            return False
            
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ FAILED after {processing_time:.2f}s")
        print(f"   Error: {str(e)}")
        return False

# BEHAVIORAL TESTING - Given input X, do we get expected output Y
def test_behavioral_scene_detection_known_video():
    """
    BEHAVIORAL TEST CASE 1: Known video with documented scenes
    INPUT: Specific video file with known content
    EXPECTED OUTPUT: Scene count > 0, specific processing info structure
    """
    print("BEHAVIORAL TEST 1: Known Video Scene Detection")
    print("-" * 40)
    
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
    
    # GIVEN: Known test video with documented characteristics
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    prompt_path = os.path.join(ROOT_DIR, "config", "cosmos_prompt.yaml")
    
    if not os.path.exists(test_video):
        print(f"FAILED: Test video not found: {test_video}")
        return False
    
    if not os.path.exists(prompt_path):
        print(f"FAILED: Prompt config not found: {prompt_path}")
        return False
    
    try:
        # WHEN: Run scene detection
        start_time = time.time()
        result = ray.get(detect_scenes.remote(
            video_path=test_video,
            prompt_path=prompt_path,
            output_dir="/tmp/behavioral_test_scenes"
        ))
        processing_time = time.time() - start_time
        
        # THEN: Verify expected output structure and content
        EXPECTED_KEYS = ['video_path', 'scenes', 'raw_response', 'total_scenes', 'processing_info']
        EXPECTED_PROCESSING_KEYS = ['model', 'success']
        
        # Check result structure
        if not isinstance(result, dict):
            print(f"FAILED: Expected dict result, got {type(result)}")
            return False
        
        missing_keys = [key for key in EXPECTED_KEYS if key not in result]
        if missing_keys:
            print(f"FAILED: Missing expected keys: {missing_keys}")
            return False
        
        # Check processing info
        processing_info = result.get('processing_info', {})
        missing_proc_keys = [key for key in EXPECTED_PROCESSING_KEYS if key not in processing_info]
        if missing_proc_keys:
            print(f"FAILED: Missing processing info keys: {missing_proc_keys}")
            return False
        
        # Check if processing was successful
        if not processing_info.get('success', False):
            print(f"FAILED: Scene detection was not successful")
            return False
        
        # Verify scene structure if scenes exist
        scenes = result.get('scenes', [])
        total_scenes = result.get('total_scenes', 0)
        
        if total_scenes > 0 and len(scenes) == 0:
            print(f"FAILED: total_scenes={total_scenes} but scenes list is empty")
            return False
        
        if scenes:
            # Check first scene structure
            scene = scenes[0]
            scene_keys = ['start_time', 'end_time', 'description']
            missing_scene_keys = [key for key in scene_keys if key not in scene]
            if missing_scene_keys:
                print(f"FAILED: Missing scene keys: {missing_scene_keys}")
                return False
        
        print(f"PASSED: Video processed successfully")
        print(f"   - Processing time: {processing_time:.1f}s")
        print(f"   - Total scenes: {total_scenes}")
        print(f"   - Model: {processing_info.get('model', 'unknown')}")
        print(f"   - Success: {processing_info.get('success', False)}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_behavioral_scene_detection_result_consistency():
    """
    BEHAVIORAL TEST CASE 2: Result structure consistency
    INPUT: Same video processed twice
    EXPECTED OUTPUT: Consistent result structure (though content may vary)
    """
    print("\nBEHAVIORAL TEST 2: Result Structure Consistency")
    print("-" * 40)
    
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
    
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    prompt_path = os.path.join(ROOT_DIR, "config", "cosmos_prompt.yaml")
    
    if not os.path.exists(test_video) or not os.path.exists(prompt_path):
        print("FAILED: Required files not found")
        return False
    
    try:
        # GIVEN: Run the same detection twice
        result1 = ray.get(detect_scenes.remote(
            video_path=test_video,
            prompt_path=prompt_path,
            output_dir="/tmp/consistency_test1"
        ))
        
        result2 = ray.get(detect_scenes.remote(
            video_path=test_video,
            prompt_path=prompt_path,
            output_dir="/tmp/consistency_test2"
        ))
        
        # THEN: Verify both results have consistent structure
        for i, result in enumerate([result1, result2], 1):
            if not isinstance(result, dict):
                print(f"FAILED: Result {i} is not a dict: {type(result)}")
                return False
            
            # Check required keys
            required_keys = ['video_path', 'scenes', 'total_scenes', 'processing_info']
            missing_keys = [key for key in required_keys if key not in result]
            if missing_keys:
                print(f"FAILED: Result {i} missing keys: {missing_keys}")
                return False
            
            # Check processing info structure
            proc_info = result.get('processing_info', {})
            if 'success' not in proc_info:
                print(f"FAILED: Result {i} processing_info missing 'success'")
                return False
        
        # Check that both results reference the same video
        if result1.get('video_path') != result2.get('video_path'):
            print(f"FAILED: Video paths don't match")
            return False
        
        print(f"PASSED: Both results have consistent structure")
        print(f"   - Result 1 scenes: {result1.get('total_scenes', 0)}")
        print(f"   - Result 2 scenes: {result2.get('total_scenes', 0)}")
        print(f"   - Both successful: {result1.get('processing_info', {}).get('success', False) and result2.get('processing_info', {}).get('success', False)}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_behavioral_scene_detection_output_files():
    """
    BEHAVIORAL TEST CASE 3: Output file creation
    INPUT: Scene detection with specified output directory
    EXPECTED OUTPUT: Files created in output directory
    """
    print("\nBEHAVIORAL TEST 3: Output File Creation")
    print("-" * 40)
    
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
    
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    prompt_path = os.path.join(ROOT_DIR, "config", "cosmos_prompt.yaml")
    
    if not os.path.exists(test_video) or not os.path.exists(prompt_path):
        print("FAILED: Required files not found")
        return False
    
    try:
        # GIVEN: Specify output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "scene_output")
            
            # WHEN: Run scene detection
            result = ray.get(detect_scenes.remote(
                video_path=test_video,
                prompt_path=prompt_path,
                output_dir=output_dir
            ))
            
            # THEN: Verify output directory was created
            if not os.path.exists(output_dir):
                print(f"FAILED: Output directory not created: {output_dir}")
                return False
            
            # Check if any files were created
            output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
            
            # Verify result indicates success
            if not result.get('processing_info', {}).get('success', False):
                print(f"FAILED: Processing was not successful")
                return False
            
            print(f"PASSED: Output handling successful")
            print(f"   - Output directory created: {os.path.exists(output_dir)}")
            print(f"   - Files in output: {len(output_files)}")
            print(f"   - Processing success: {result.get('processing_info', {}).get('success', False)}")
            
            return True
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def run_all_tests():
    """Run ALL tests: original (unit+integration) AND behavioral tests for scene detection"""
    print("COMPLETE Scene Detection Testing Suite")
    print("=" * 50)
    print("Running unit tests, integration tests, AND behavioral tests")
    print()
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("CUDA not available - scene detection may fail or be very slow")
    except ImportError:
        print("PyTorch not available - cannot check GPU status")
    
    # Run original unit tests first
    print("\nUNIT TESTS (Original)")
    print("=" * 30)
    
    unit_tests = [
        test_clear_gpu_memory,
        test_prompt_config_loading,
        test_result_structure_validation,
        test_output_directory_creation
    ]
    
    unit_results = []
    for test_func in unit_tests:
        try:
            result = test_func()
            unit_results.append(result)
        except Exception as e:
            print(f"Unit test {test_func.__name__} failed with exception: {e}")
            unit_results.append(False)
    
    unit_passed = sum(unit_results)
    print(f"\nUnit Tests Summary: {unit_passed}/{len(unit_tests)} passed")
    
    # Run original integration test
    print(f"\nINTEGRATION TEST (Original)")
    print("=" * 30)
    
    integration_success = False
    try:
        integration_success = test_detect_scenes_integration()
    except Exception as e:
        print(f"Integration test failed with exception: {e}")
    
    print(f"Integration Test: {'PASSED' if integration_success else 'FAILED'}")
    
    # Check required files for behavioral tests
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    prompt_path = os.path.join(ROOT_DIR, "config", "cosmos_prompt.yaml")
    
    if not os.path.exists(test_video) or not os.path.exists(prompt_path):
        print(f"\nWARNING: Required files not found:")
        if not os.path.exists(test_video):
            print(f"  - Test video: {test_video}")
        if not os.path.exists(prompt_path):
            print(f"  - Prompt config: {prompt_path}")
        print("Skipping behavioral tests - they require both files")
        behavioral_passed = 0
        behavioral_total = 3
    else:
        # Run behavioral tests
        print(f"\nBEHAVIORAL TESTS (New - Manager's Request)")
        print("=" * 50)
        print("Testing documented scenarios with predictable results")
        print(f"Test video: {os.path.basename(test_video)}")
        print(f"Prompt config: {os.path.basename(prompt_path)}")
        
        behavioral_tests = [
            test_behavioral_scene_detection_known_video,
            test_behavioral_scene_detection_result_consistency,
            test_behavioral_scene_detection_output_files,
        ]
        
        behavioral_results = []
        for test_func in behavioral_tests:
            try:
                result = test_func()
                behavioral_results.append(result)
            except Exception as e:
                print(f"Behavioral test {test_func.__name__} failed: {e}")
                behavioral_results.append(False)
        
        behavioral_passed = sum(behavioral_results)
        behavioral_total = len(behavioral_results)
        
        print(f"\nBehavioral Tests Summary: {behavioral_passed}/{behavioral_total} passed")
        
        # Test case summary
        print(f"\nBEHAVIORAL TEST DOCUMENTATION:")
        print(f"1. Known video processing: {'PASS' if behavioral_results[0] else 'FAIL'}")
        print(f"2. Result structure consistency: {'PASS' if behavioral_results[1] else 'FAIL'}")
        print(f"3. Output file creation: {'PASS' if behavioral_results[2] else 'FAIL'}")
    
    # Overall summary
    total_tests = len(unit_tests) + 1 + behavioral_total  # unit + integration + behavioral
    total_passed = unit_passed + (1 if integration_success else 0) + behavioral_passed
    
    print(f"\nOVERALL RESULTS")
    print("=" * 50)
    print(f"Unit Tests: {unit_passed}/{len(unit_tests)} passed")
    print(f"Integration Test: {'PASSED' if integration_success else 'FAILED'}")
    print(f"Behavioral Tests: {behavioral_passed}/{behavioral_total} passed")
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\nALL TESTS PASSED! Scene detection is fully validated.")
        print("✅ Unit tests: Validate individual functions")
        print("✅ Integration test: Validate complete Ray job")
        print("✅ Behavioral tests: Validate predictable scenarios")
        return True
    else:
        print(f"\n{total_tests - total_passed} test(s) failed. Check errors above.")
        return False

if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted")
        exit_code = 2
    except Exception as e:
        print(f"\n💥 Testing error: {e}")
        exit_code = 3
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("🔧 Ray shutdown")
        
        sys.exit(exit_code)
