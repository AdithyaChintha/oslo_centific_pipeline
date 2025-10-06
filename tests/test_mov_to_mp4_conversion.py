#!/usr/bin/env python3
"""
MOV to MP4 Conversion pytest Testing Suite

Tests the production ray_jobs/mov_to_mp4.py conversion function.
Each video in the input list becomes a separate test case.

Supports both input formats:
- Simple format: one S3 path per line
- TSV format: path<TAB>hash<TAB>size

Usage:
    # Default input file (failed_list_server2.txt)
    pytest test_mov_to_mp4_conversion.py -v

    # Custom input file using environment variable
    INPUT_FILE=../video_lists/server_1_videos.txt pytest test_mov_to_mp4_conversion.py -v
    INPUT_FILE=../video_lists/failed_list_server2.txt pytest test_mov_to_mp4_conversion.py -v

    # Run only edge case tests
    pytest test_mov_to_mp4_conversion.py::TestConversionEdgeCases -v
"""

import os
import sys
import boto3
import yaml
import ray
import pytest
import tempfile
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ray_jobs.mov_to_mp4 import convert_mov_to_mp4_task

# ==================================================
# CONFIGURATION
# ==================================================
# Get input file from environment variable or use default
DEFAULT_INPUT_FILE = os.environ.get('INPUT_FILE', "../video_lists/server_1_videos.txt")


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def parse_video_list(file_path):
    """
    Parse video list file - supports both simple and TSV formats.

    Returns:
        list of S3 keys (just the path part)
    """
    videos = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Handle both TSV and simple formats
            if '\t' in line:
                s3_path = line.split('\t')[0].strip()
            else:
                s3_path = line.strip()

            videos.append(s3_path)

    return videos


# ==================================================
# FIXTURES
# ==================================================
@pytest.fixture(scope="session", autouse=True)
def setup_ray():
    """Initialize Ray once for all tests"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        print("\n✅ Ray initialized")
    yield
    # Cleanup after all tests
    if ray.is_initialized():
        ray.shutdown()
        print("\n✅ Ray shutdown complete")


@pytest.fixture(scope="session")
def config():
    """Load S3 configuration"""
    config_path = ROOT_DIR / "config" / "s3_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def s3_client(config):
    """Create S3 client for downloading videos"""
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

    if not aws_access_key or not aws_secret_key:
        pytest.skip("AWS credentials not found in environment variables")

    client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=config['s3']['region']
    )

    return client, config['s3']['bucket_name']


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="session")
def video_list():
    """Load and parse video list from input file"""
    input_file = DEFAULT_INPUT_FILE

    # Resolve path relative to tests directory
    if not Path(input_file).is_absolute():
        input_file = ROOT_DIR / "tests" / input_file

    if not Path(input_file).exists():
        pytest.skip(f"Input file not found: {input_file}")

    videos = parse_video_list(input_file)

    print(f"\n{'='*80}")
    print(f"Loaded {len(videos)} videos from: {Path(input_file).name}")
    print(f"{'='*80}")

    return videos


# ==================================================
# PARAMETRIZED TEST - Each video is a test case
# ==================================================
def pytest_generate_tests(metafunc):
    """
    Dynamically generate test cases for each video in the list.
    This creates one test per video instead of using a large range.
    """
    if "video_s3_key" in metafunc.fixturenames:
        # Get video list
        input_file = DEFAULT_INPUT_FILE

        # Resolve path
        if not Path(input_file).is_absolute():
            input_file = ROOT_DIR / "tests" / input_file

        if Path(input_file).exists():
            videos = parse_video_list(input_file)
            # Create test IDs for better output
            ids = [f"{i+1:05d}_{os.path.basename(v)}" for i, v in enumerate(videos)]
            metafunc.parametrize("video_s3_key,video_index",
                               [(v, i) for i, v in enumerate(videos)],
                               ids=ids)


def test_video_conversion(video_s3_key, video_index, video_list, s3_client, temp_dir):
    """
    Test MOV to MP4 conversion for a single video.
    Each video in the list becomes a separate test case.
    """
    s3_client_obj, bucket_name = s3_client
    video_name = os.path.basename(video_s3_key)

    # Create unique local paths
    local_download_path = os.path.join(temp_dir, f"{video_index}_{video_name}")
    output_mp4_path = os.path.join(temp_dir, f"{video_index}_{video_name.replace('.mov', '.mp4')}")

    print(f"\n{'='*80}")
    print(f"Test [{video_index + 1}/{len(video_list)}]: {video_name}")
    print(f"S3 Key: {video_s3_key}")
    print(f"{'='*80}")

    # Step 1: Download from S3
    try:
        s3_client_obj.download_file(bucket_name, video_s3_key, local_download_path)
        size_mb = os.path.getsize(local_download_path) / (1024*1024)
        print(f"✅ Downloaded: {size_mb:.2f} MB")
    except s3_client_obj.exceptions.NoSuchKey:
        pytest.skip(f"Video not found in S3 (404)")
    except Exception as e:
        pytest.fail(f"Download failed: {e}")

    # Step 2: Test conversion using Ray task
    try:
        task_ref = convert_mov_to_mp4_task.remote(
            input_mov_path=local_download_path,
            output_mp4_path=output_mp4_path,
            video_id=video_name,
            timing_id=f"test_{video_index}"
        )

        # Wait for result with timeout
        conversion_result = ray.get(task_ref, timeout=120)

        if not conversion_result['success']:
            error_msg = conversion_result.get('error', 'Unknown error')
            returncode = conversion_result.get('returncode', 'N/A')
            print(f"❌ Conversion failed (return code: {returncode})")
            print(f"Error: {error_msg[:300]}")
            pytest.fail(f"Conversion failed: {error_msg[:200]}")

        # Success
        output_size_mb = conversion_result['file_size'] / (1024*1024)
        conversion_time = conversion_result['conversion_time']
        input_size_mb = os.path.getsize(local_download_path) / (1024*1024)

        print(f"✅ Conversion SUCCESS")
        print(f"   - Input size: {input_size_mb:.2f} MB")
        print(f"   - Output size: {output_size_mb:.2f} MB")
        print(f"   - Conversion time: {conversion_time:.2f}s")
        print(f"   - Speed: {input_size_mb/conversion_time:.2f} MB/s")

        # Verify output file exists and is not empty
        assert os.path.exists(output_mp4_path), "Output MP4 file not created"
        assert os.path.getsize(output_mp4_path) > 0, "Output MP4 file is empty"

    except Exception as e:
        pytest.fail(f"Ray task failed: {e}")

    finally:
        # Cleanup
        if os.path.exists(local_download_path):
            os.remove(local_download_path)
        if os.path.exists(output_mp4_path):
            os.remove(output_mp4_path)


# ==================================================
# EDGE CASE TESTS
# ==================================================
class TestConversionEdgeCases:
    """Edge case tests for the conversion function"""

    def test_missing_input_file(self, temp_dir):
        """Test handling of missing input file"""
        output_path = os.path.join(temp_dir, "output.mp4")

        task_ref = convert_mov_to_mp4_task.remote(
            input_mov_path="/nonexistent/video.mov",
            output_mp4_path=output_path,
            video_id="test_missing"
        )

        result = ray.get(task_ref)
        assert result['success'] is False, "Should fail for missing file"
        print(f"✅ Correctly rejected missing file: {result.get('error', '')[:100]}")

    def test_corrupted_video_file(self, temp_dir):
        """Test handling of corrupted video file"""
        corrupted_path = os.path.join(temp_dir, "corrupted.mov")
        with open(corrupted_path, "wb") as f:
            f.write(b"not a real video file")

        output_path = os.path.join(temp_dir, "output.mp4")

        task_ref = convert_mov_to_mp4_task.remote(
            input_mov_path=corrupted_path,
            output_mp4_path=output_path,
            video_id="test_corrupted"
        )

        result = ray.get(task_ref)
        assert result['success'] is False, "Should fail for corrupted file"
        print(f"✅ Correctly rejected corrupted file: {result.get('error', '')[:100]}")

    def test_empty_video_file(self, temp_dir):
        """Test handling of empty video file"""
        empty_path = os.path.join(temp_dir, "empty.mov")
        Path(empty_path).touch()  # Create empty file

        output_path = os.path.join(temp_dir, "output.mp4")

        task_ref = convert_mov_to_mp4_task.remote(
            input_mov_path=empty_path,
            output_mp4_path=output_path,
            video_id="test_empty"
        )

        result = ray.get(task_ref)
        assert result['success'] is False, "Should fail for empty file"
        print(f"✅ Correctly rejected empty file: {result.get('error', '')[:100]}")


# ==================================================
# CONFIGURATION INFO
# ==================================================
def test_configuration_info(video_list):
    """Display test configuration information"""
    print(f"\n{'='*80}")
    print(f"Test Configuration:")
    print(f"   - Total videos to test: {len(video_list)}")
    print(f"   - Ray initialized: {ray.is_initialized()}")
    print(f"   - Project root: {ROOT_DIR}")
    print(f"   - AWS credentials: {'✓' if os.environ.get('AWS_ACCESS_KEY_ID') else '✗'}")
    print(f"{'='*80}")

    assert len(video_list) > 0, "No videos loaded for testing"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
