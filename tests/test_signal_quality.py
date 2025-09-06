import ray
import os
import pytest
import subprocess
from ray_jobs.signal_quality_check_blur_black_screen import detect_blur_and_black_segments

@pytest.fixture(scope="module")
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()

@pytest.fixture(scope="function")
def noisy_video_file():
    video_path = "tests/test_video_noisy.mp4"
    subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=30:d=10", "-vf", "noise=alls=10:allf=t+u", video_path, "-y"], check=True, capture_output=True)
    yield video_path
    os.remove(video_path)

@pytest.fixture(scope="function")
def non_noisy_video_file():
    video_path = "tests/test_video_nonoise.mp4"
    subprocess.run(["ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=30:d=10", video_path, "-y"], check=True, capture_output=True)
    yield video_path
    os.remove(video_path)

def test_detect_blur_and_black_segments_noisy(ray_init, noisy_video_file):
    video_path = noisy_video_file
    
    # Run the Ray task
    result_ref = detect_blur_and_black_segments.remote(video_path)
    result = ray.get(result_ref)
    
    # Assertions
    assert result["video_path"] == video_path
    assert "blur_segments" in result
    assert "black_segments" in result
    
    # Since the video is black with noise, we expect black segments and blur segments
    assert len(result["black_segments"]) > 0
    assert len(result["blur_segments"]) > 0
    
    # Check the structure of the segments
    for segment in result["black_segments"]:
        assert "start_time" in segment
        assert "end_time" in segment
        assert isinstance(segment["start_time"], float)
        assert isinstance(segment["end_time"], float)

def test_detect_blur_and_black_segments_nonoise(ray_init, non_noisy_video_file):
    video_path = non_noisy_video_file
    
    # Run the Ray task
    result_ref = detect_blur_and_black_segments.remote(video_path, blur_thresh=-1)
    result = ray.get(result_ref)
    
    # Assertions
    assert result["video_path"] == video_path
    assert "blur_segments" in result
    assert "black_segments" in result
    
    # Since the video is black with no noise, we expect black segments and no blur segments
    assert len(result["black_segments"]) > 0
    assert len(result["blur_segments"]) == 0
    
    # Check the structure of the segments
    for segment in result["black_segments"]:
        assert "start_time" in segment
        assert "end_time" in segment
        assert isinstance(segment["start_time"], float)
        assert isinstance(segment["end_time"], float)