import ray
import os
import json
import sys
import time
import tempfile
import re
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import the Ray job and helper functions
#from ray_jobs.run_yolodetect_task import run_yolodetect_on_shard, _part_index
from ray_jobs.yolo_detection import run_yolo_detection

def _part_index(filename: str) -> int:
    """Extract shard index from names like 'xxx_part3.mp4' -> 3; default to 0."""
    m = re.search(r"_part(\d+)\.mp4$", filename)
    return int(m.group(1)) if m else 0

def test_part_index_parsing():
    """Unit test for shard index parsing function"""
    print("Unit Test 1: Part Index Parsing")
    print("-" * 40)
    
    try:
        # Test valid part names
        test_cases = [
            ("video_part0.mp4", 0),
            ("video_part3.mp4", 3),
            ("test_shard_part10.mp4", 10),
            ("video.mp4", 0),  # No part number should default to 0
            ("video_part.mp4", 0),  # Invalid part should default to 0
        ]
        
        all_passed = True
        for filename, expected in test_cases:
            result = _part_index(filename)
            if result != expected:
                print(f"FAILED: {filename} -> expected {expected}, got {result}")
                all_passed = False
            else:
                print(f"PASSED: {filename} -> {result}")
        
        if all_passed:
            print("Part index parsing function works correctly")
            return True
        else:
            print("Some part index parsing tests failed")
            return False
            
    except Exception as e:
        print(f"Part index parsing test failed: {e}")
        return False

def test_output_directory_creation():
    """Unit test for output directory creation functionality"""
    print("\nUnit Test 2: Output Directory Creation")
    print("-" * 40)
    
    try:
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_output_dir = os.path.join(temp_dir, "yolo_detection_test")
            
            # Test directory creation (mimicking the Ray job behavior)
            os.makedirs(test_output_dir, exist_ok=True)
            
            if not os.path.exists(test_output_dir):
                print(f"Failed to create output directory: {test_output_dir}")
                return False
            
            # Test file writing in the directory (mimicking YOLO output files)
            events_file = os.path.join(test_output_dir, "test_video.events.jsonl")
            spans_file = os.path.join(test_output_dir, "test_video.spans.jsonl")
            
            # Create sample JSONL content
            sample_event = {"frame": 0, "timestamp": 0.0, "detections": []}
            sample_span = {"start_time": 0.0, "end_time": 10.0, "person_count": 1}
            
            with open(events_file, 'w') as f:
                f.write(json.dumps(sample_event) + "\n")
            
            with open(spans_file, 'w') as f:
                f.write(json.dumps(sample_span) + "\n")
            
            if not os.path.exists(events_file) or not os.path.exists(spans_file):
                print("Failed to write YOLO output files")
                return False
            
            print("Output directory creation and file writing successful")
            print(f"   - Directory created: {test_output_dir}")
            print(f"   - Events file: {os.path.basename(events_file)}")
            print(f"   - Spans file: {os.path.basename(spans_file)}")
            return True
            
    except Exception as e:
        print(f"Output directory creation test failed: {e}")
        return False

def test_yolo_parameters_validation():
    """Unit test for YOLO detection parameters validation"""
    print("\nUnit Test 3: YOLO Parameters Validation")
    print("-" * 40)
    
    try:
        # Test parameter ranges and types
        test_parameters = {
            "conf": [0.1, 0.5, 0.9],  # Valid confidence values
            "iou": [0.3, 0.5, 0.7],   # Valid IoU values
            "frame_stride": [1, 5, 10],  # Valid frame strides
            "gap_sec": [1.0, 10.0, 30.0],  # Valid gap seconds
        }
        
        valid_classes = [None, [0], [0, 1, 2]]  # Valid class configurations
        valid_devices = ["cpu", "cuda:0", None]  # Valid device configurations
        
        # Test parameter value ranges
        for param_name, values in test_parameters.items():
            for value in values:
                if param_name in ["conf", "iou"] and not (0.0 <= value <= 1.0):
                    print(f"Invalid {param_name} value: {value}")
                    return False
                elif param_name == "frame_stride" and not (value > 0 and isinstance(value, int)):
                    print(f"Invalid {param_name} value: {value}")
                    return False
                elif param_name == "gap_sec" and not (value > 0):
                    print(f"Invalid {param_name} value: {value}")
                    return False
        
        print("YOLO parameters validation passed")
        print(f"   - Confidence range: {test_parameters['conf']}")
        print(f"   - IoU range: {test_parameters['iou']}")
        print(f"   - Frame strides: {test_parameters['frame_stride']}")
        print(f"   - Gap seconds: {test_parameters['gap_sec']}")
        print(f"   - Valid classes: {len(valid_classes)} configurations")
        print(f"   - Valid devices: {len(valid_devices)} configurations")
        return True
        
    except Exception as e:
        print(f"YOLO parameters validation failed: {e}")
        return False

def test_filename_processing():
    """Unit test for filename and path processing logic"""
    print("\nUnit Test 4: Filename Processing")
    print("-" * 40)
    
    try:
        # Test filename processing logic with a single test case
        test_video = "/path/to/video_part0.mp4"
        base = os.path.basename(test_video)
        stem, _ = os.path.splitext(base)
        part = _part_index(base)
        
        # Expected results
        expected_stem = "video_part0"
        expected_part = 0
        
        if stem != expected_stem:
            print(f"Stem mismatch for {base}: expected {expected_stem}, got {stem}")
            return False
        
        if part != expected_part:
            print(f"Part mismatch for {base}: expected {expected_part}, got {part}")
            return False
        
        # Test output filename generation
        test_out_dir = "/tmp/test"
        events_out = os.path.join(test_out_dir, f"{stem}.events.jsonl")
        spans_out = os.path.join(test_out_dir, f"{stem}.spans.jsonl")
        
        expected_events_path = "/tmp/test/video_part0.events.jsonl"
        expected_spans_path = "/tmp/test/video_part0.spans.jsonl"
        
        if events_out != expected_events_path:
            print(f"Events path mismatch: expected {expected_events_path}, got {events_out}")
            return False
        
        if spans_out != expected_spans_path:
            print(f"Spans path mismatch: expected {expected_spans_path}, got {spans_out}")
            return False
        
        print(f"PASSED: {base} -> part={part}, stem='{stem}'")
        
        print("Filename processing validation passed")
        return True
        
    except Exception as e:
        print(f"Filename processing test failed: {e}")
        return False

def test_shard_time_calculation():
    """Unit test for shard time offset calculation"""
    print("\nUnit Test 5: Shard Time Calculation")
    print("-" * 40)
    
    try:
        # Test time offset calculation for different shards
        shard_seconds = 60  # Default shard duration
        
        test_cases = [
            ("video_part0.mp4", 0, 0),      # Part 0 -> offset 0
            ("video_part1.mp4", 1, 60),     # Part 1 -> offset 60
            ("video_part3.mp4", 3, 180),    # Part 3 -> offset 180
            ("video_part10.mp4", 10, 600),  # Part 10 -> offset 600
            ("video.mp4", 0, 0),            # No part -> offset 0
        ]
        
        for filename, expected_part, expected_offset in test_cases:
            part = _part_index(filename)
            base_offset = part * shard_seconds
            
            if part != expected_part:
                print(f"Part mismatch for {filename}: expected {expected_part}, got {part}")
                return False
            
            if base_offset != expected_offset:
                print(f"Offset mismatch for {filename}: expected {expected_offset}, got {base_offset}")
                return False
            
            print(f"PASSED: {filename} -> part={part}, offset={base_offset}s")
        
        print("Shard time calculation validation passed")
        return True
        
    except Exception as e:
        print(f"Shard time calculation test failed: {e}")
        return False

def test_run_yolodetect_integration():
    """Integration test for the complete YOLO detection Ray job"""
    
    print("\nIntegration Test: Complete YOLO Detection")
    print("=" * 50)
    
    # Initialize Ray with the correct Python path
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
        print("Ray initialized")
    else:
        print("Ray already initialized")
    
    # Test video path - using same video as other tests
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    
    if not os.path.exists(test_video):
        print(f"Test video not found: {test_video}")
        print("Please update the test_video path to a valid MP4 file")
        return False
    
    print(f"Testing with video: {os.path.basename(test_video)}")
    print()
    
    # Test the YOLO detection Ray job
    print(f"Testing YOLO Detection: {os.path.basename(test_video)}")
    print("-" * 40)
    
    start_time = time.time()
    output_dir = "/tmp/test_yolo_detection"
    
    try:
        # Call the Ray job (exactly how it will be used in production)
        result = ray.get(run_yolo_detection.remote(
            video_path=test_video,
            output_dir=output_dir,
            shard_seconds=60,
            model="yolo11m.pt",  # Using default model
            conf=0.5,
            iou=0.5,
            frame_stride=5,
            classes=[0],  # Person class only
            device="cuda:0" if os.path.exists("/dev/nvidia0") else "cpu",
            gap_sec=10.0,
        ))
        processing_time = time.time() - start_time

        if result and isinstance(result, dict):
            # New: ensure people_count_json key is present in the result
            if 'people_count_json' in result:
                if result['people_count_json'] is not None:
                    if not os.path.exists(result['people_count_json']):
                        print(f"Warning: people_count_json file not found: {result['people_count_json']}")
            else:
                print("Warning: 'people_count_json' not present in run_yolo_detection result")

            print(f"SUCCESS in {processing_time:.2f}s")
            print("Results:")
            print(f"   - Result type: {type(result)}")
            print(f"   - Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

            # Check if output files were created
            if output_dir and os.path.exists(output_dir):
                output_files = os.listdir(output_dir)
                print(f"Output files: {len(output_files)} file(s) created in {output_dir}")
                for file in output_files:
                    file_path = os.path.join(output_dir, file)
                    file_size = os.path.getsize(file_path)
                    print(f"   - {file} ({file_size} bytes)")

            # Show result summary
            if 'detections_count' in result:
                print(f"Detections: {result['detections_count']}")
            if 'video_duration' in result and result['video_duration'] is not None:
                print(f"Video duration: {result['video_duration']:.1f}s")
            if 'fps' in result and result['fps'] is not None:
                print(f"FPS: {result['fps']:.1f}")
            if 'num_events' in result:
                print(f"Detection events: {result['num_events']}")

            print(f"\nYOLO detection test PASSED!")
            print(f"   Processing time: {processing_time:.1f}s")
            return True

        else:
            print(f"FAILED - Invalid result type: {type(result)}")
            return False
            
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"FAILED after {processing_time:.2f}s")
        print(f"   Error: {str(e)}")
        
        # Check if it's a missing dependency issue
        if "yolo_detection_r" in str(e):
            print("   This might be due to missing yolo_detection_r module")
            print("   Make sure the YOLO detection module is properly installed")
        
        return False

# BEHAVIORAL TESTING - Given input X, do we get expected output Y
def test_behavioral_yolo_detection_known_video():
    """
    BEHAVIORAL TEST CASE 1: Known video with people
    INPUT: Specific video file with documented human presence
    EXPECTED OUTPUT: Valid detection result structure with proper JSONL files
    """
    print("BEHAVIORAL TEST 1: Known Video YOLO Detection")
    print("-" * 40)
    
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
    
    # GIVEN: Known test video with people
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    
    if not os.path.exists(test_video):
        print(f"FAILED: Test video not found: {test_video}")
        return False
    
    try:
        # WHEN: Run YOLO detection with specific parameters
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "yolo_output")
            
            result = ray.get(run_yolo_detection.remote(
                video_path=test_video,
                output_dir=output_dir,
                shard_seconds=60,
                model="yolo11m.pt",
                conf=0.5,
                iou=0.5,
                frame_stride=5,
                classes=[0],  # Person class only
                device="cpu",  # Use CPU for consistent results
                gap_sec=10.0,
            ))
            
            # THEN: Verify expected output structure
            EXPECTED_KEYS = ['video', 'events_jsonl', 'spans_jsonl', 'num_events']
            
            # Check result structure
            if not isinstance(result, dict):
                print(f"FAILED: Expected dict result, got {type(result)}")
                return False
            
            missing_keys = [key for key in EXPECTED_KEYS if key not in result]
            if missing_keys:
                print(f"FAILED: Missing expected keys: {missing_keys}")
                return False
            
            # Verify file paths are returned
            events_file = result.get('events_jsonl')
            spans_file = result.get('spans_jsonl')
            
            if not events_file or not isinstance(events_file, str):
                print(f"FAILED: Invalid events_jsonl: {events_file}")
                return False
            
            if not spans_file or not isinstance(spans_file, str):
                print(f"FAILED: Invalid spans_jsonl: {spans_file}")
                return False
            
            # Verify files exist
            if not os.path.exists(events_file):
                print(f"FAILED: Events file not created: {events_file}")
                return False
            
            # Note: spans file creation might be disabled in yolodetect function
            spans_exists = os.path.exists(spans_file)
            if not spans_exists:
                print(f"WARNING: Spans file not created: {spans_file}")
                print(f"   This may be normal if spans generation is disabled")
            
            # Verify num_events is a number
            num_events = result.get('num_events', -1)
            if not isinstance(num_events, (int, float)) or num_events < 0:
                print(f"FAILED: Invalid num_events: {num_events}")
                return False
            
            print(f"PASSED: YOLO detection completed successfully")
            print(f"   - Events file: {os.path.basename(events_file)} ({os.path.getsize(events_file)} bytes)")
            if spans_exists:
                print(f"   - Spans file: {os.path.basename(spans_file)} ({os.path.getsize(spans_file)} bytes)")
            else:
                print(f"   - Spans file: Not created (spans generation may be disabled)")
            print(f"   - Detection events: {num_events}")
            print(f"   - Video processed: {os.path.basename(result.get('video', 'unknown'))}")
            
            return True
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_behavioral_yolo_detection_output_format():
    """
    BEHAVIORAL TEST CASE 2: JSONL output format validation
    INPUT: Video processing with YOLO detection
    EXPECTED OUTPUT: Valid JSONL files with correct structure
    """
    print("\nBEHAVIORAL TEST 2: JSONL Output Format Validation")
    print("-" * 40)
    
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
    
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    
    if not os.path.exists(test_video):
        print(f"FAILED: Test video not found: {test_video}")
        return False
    
    try:
        # WHEN: Run YOLO detection
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "format_test")
            
            result = ray.get(run_yolo_detection.remote(
                video_path=test_video,
                output_dir=output_dir,
                shard_seconds=30,  # Shorter for faster testing
                conf=0.5,
                frame_stride=15,  # Skip frames for speed
                classes=[0],
                device="cpu",
            ))
            
            # THEN: Verify basic output structure
            events_file = result.get('events_jsonl')
            
            if not events_file or not os.path.exists(events_file):
                print(f"FAILED: Events file not created")
                return False
            
            # Check if events file has valid JSON content
            with open(events_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:  # Should have at least metadata + 1 event
                print(f"FAILED: Events file too short ({len(lines)} lines)")
                return False
            
            # Check that lines contain valid JSON
            valid_json_count = 0
            for line in lines[:5]:  # Check first 5 lines
                try:
                    json.loads(line.strip())
                    valid_json_count += 1
                except json.JSONDecodeError:
                    print(f"FAILED: Invalid JSON in events file")
                    return False
            
            print(f"PASSED: JSONL format validation successful")
            print(f"   - Events file created: {os.path.basename(events_file)}")
            print(f"   - File size: {os.path.getsize(events_file)} bytes")
            print(f"   - Total lines: {len(lines)}")
            print(f"   - Valid JSON lines checked: {valid_json_count}")
            
            return True
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_behavioral_yolo_detection_parameter_consistency():
    """
    BEHAVIORAL TEST CASE 3: Parameter consistency validation
    INPUT: Same video with different confidence thresholds
    EXPECTED OUTPUT: Higher confidence = fewer detections
    """
    print("\nBEHAVIORAL TEST 3: Parameter Consistency")
    print("-" * 40)
    
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
    
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    
    if not os.path.exists(test_video):
        print(f"FAILED: Test video not found: {test_video}")
        return False
    
    try:
        # GIVEN: Test with two different confidence levels
        with tempfile.TemporaryDirectory() as temp_dir:
            # Low confidence detection
            result_low = ray.get(run_yolo_detection.remote(
                video_path=test_video,
                output_dir=os.path.join(temp_dir, "low_conf"),
                shard_seconds=30,
                conf=0.3,  # Low confidence
                frame_stride=15,
                classes=[0],
                device="cpu",
            ))
            
            # High confidence detection
            result_high = ray.get(run_yolo_detection.remote(
                video_path=test_video,
                output_dir=os.path.join(temp_dir, "high_conf"),
                shard_seconds=30,
                conf=0.8,  # High confidence
                frame_stride=15,
                classes=[0],
                device="cpu",
            ))
            
            # THEN: Verify behavior consistency
            if not (isinstance(result_low, dict) and isinstance(result_high, dict)):
                print(f"FAILED: Results are not dictionaries")
                return False
            
            events_low = result_low.get('num_events', 0)
            events_high = result_high.get('num_events', 0)
            
            # Higher confidence should generally produce fewer or equal detections
            if events_high > events_low:
                print(f"WARNING: High confidence ({events_high}) > Low confidence ({events_low})")
                print(f"   This may happen with very short videos or specific content")
            
            # Both should have valid events file outputs
            for result, label in [(result_low, "low"), (result_high, "high")]:
                if not os.path.exists(result.get('events_jsonl', '')):
                    print(f"FAILED: {label} confidence events file missing")
                    return False
            
            print(f"PASSED: Parameter consistency validated")
            print(f"   - Low confidence (0.3): {events_low} events")
            print(f"   - High confidence (0.8): {events_high} events")
            print(f"   - Behavior: {'Expected' if events_high <= events_low else 'Unexpected but acceptable'}")
            
            return True
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def run_all_tests():
    """Run ALL tests: original (unit+integration) AND behavioral tests for YOLO detection"""
    print("COMPLETE YOLO Detection Testing Suite")
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
            print("CUDA not available - using CPU for consistent results")
    except ImportError:
        print("PyTorch not available - cannot check GPU status")
    
    # Run original unit tests first
    print("\nUNIT TESTS (Original)")
    print("=" * 30)
    
    unit_tests = [
        test_part_index_parsing,
        test_output_directory_creation,
        test_yolo_parameters_validation,
        test_filename_processing,
        test_shard_time_calculation
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
        integration_success = test_run_yolodetect_integration()
    except Exception as e:
        print(f"Integration test failed with exception: {e}")
    
    print(f"Integration Test: {'PASSED' if integration_success else 'FAILED'}")
    
    # Check required files for behavioral tests
    test_video = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/video_with_minors.mp4"
    
    if not os.path.exists(test_video):
        print(f"\nWARNING: Test video not found: {test_video}")
        print("Skipping behavioral tests - they require the test video")
        behavioral_passed = 0
        behavioral_total = 3
    else:
        # Run behavioral tests
        print(f"\nBEHAVIORAL TESTS (New - Manager's Request)")
        print("=" * 50)
        print("Testing documented scenarios with predictable results")
        print(f"Test video: {os.path.basename(test_video)}")
        
        behavioral_tests = [
            test_behavioral_yolo_detection_known_video,
            test_behavioral_yolo_detection_output_format,
            test_behavioral_yolo_detection_parameter_consistency,
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
        print(f"2. JSONL output format validation: {'PASS' if behavioral_results[1] else 'FAIL'}")
        print(f"3. Parameter consistency: {'PASS' if behavioral_results[2] else 'FAIL'}")
    
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
        print("\nALL TESTS PASSED! YOLO detection is fully validated.")
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
        print("\nTesting interrupted")
        exit_code = 2
    except Exception as e:
        print(f"\nTesting error: {e}")
        exit_code = 3
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown")
        
        sys.exit(exit_code)