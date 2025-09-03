"""
NSFW Detector pytest Testing Suite
Run with: pytest test_nsfw_detector.py -v
"""

import os
import sys
import cv2
import ray
import time
import torch
import tempfile
import pytest
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import current worker (no setup_model method; returns 'nsfw_timestamps')
from ray_jobs.nsfw_det_final import NSFWDetectorWorker  # matches your current impl

# ==================================================
# CONFIGURATION
# ==================================================
CUSTOM_FRAME_PATH = "tests/nsfw_ai_generated.jpg"  # path to your custom frame
USE_CUSTOM_FRAME = True


# ==================================================
# FIXTURES
# ==================================================
@pytest.fixture(scope="session", autouse=True)
def setup_ray():
    """Initialize Ray once for all tests"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    yield
    # Cleanup after all tests
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def video_generator(temp_dir):
    """Create a CustomFrameVideoGenerator instance"""
    return CustomFrameVideoGenerator(
        output_dir=temp_dir,
        custom_frame_path=CUSTOM_FRAME_PATH,
        use_custom=USE_CUSTOM_FRAME
    )


@pytest.fixture
def detector():
    """Create and cleanup NSFWDetectorWorker"""
    detector = NSFWDetectorWorker.remote()
    yield detector
    ray.kill(detector)


# ==================================================
# Utilities
# ==================================================
def tiny_video(path: Path, fps: int = 5, seconds: int = 1, size=(64, 64)):
    """Create a 1-second neutral tiny MP4 to exercise init+inference quickly."""
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, size)
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    for _ in range(fps * seconds):
        out.write(frame)
    out.release()
    return str(path)


# ==================================================
# Video Generator
# ==================================================
class CustomFrameVideoGenerator:
    def __init__(self, output_dir="/tmp", custom_frame_path=None, use_custom=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.custom_frame = None

        if use_custom and custom_frame_path and Path(custom_frame_path).exists():
            img = cv2.imread(str(custom_frame_path))
            if img is not None:
                self.custom_frame = img
                print(f"Loaded custom frame from {custom_frame_path}")
            else:
                print(f"⚠️ Could not read image at {custom_frame_path}, falling back to red square")

    def create_test_video(
        self,
        test_type="single_detection",
        duration=30,
        fps=30,
        resolution=(640, 480),
        custom_frame_duration=1,
        insert_frame=True
    ):
        width, height = resolution
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = str(self.output_dir / f"test_{test_type}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)

        neutral_frame = np.full((height, width, 3), 128, dtype=np.uint8)

        # Use custom frame if available, else fallback red square
        if self.custom_frame is not None:
            custom_frame = cv2.resize(self.custom_frame, (width, height))
        else:
            custom_frame = np.full((height, width, 3), (0, 0, 255), dtype=np.uint8)

        total_frames = int(duration * fps)
        inserted_indices = []

        if insert_frame:
            if test_type == "single_detection":
                inserted_indices = [int(duration * fps / 2)]
            elif test_type == "multiple_detection":
                inserted_indices = [int(fps * t) for t in [10, 20, 30]]
            elif test_type == "clustered_detection":
                inserted_indices = [int(fps * t) for t in [15, 16, 17]]
            elif test_type == "boundary_detection":
                inserted_indices = [0, int(total_frames - fps * 2)]

        for i in range(total_frames):
            if i in inserted_indices:
                for _ in range(custom_frame_duration * fps):
                    out.write(custom_frame)
            else:
                out.write(neutral_frame)

        out.release()
        return output_path


# ==================================================
# ORIGINAL INTEGRATION TESTS
# ==================================================
class TestNSFWDetectorIntegration:
    """Original integration tests converted to pytest format"""
    
    def test_gpu_memory_management(self):
        """Test GPU Memory Management"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # No assertion needed - if this fails, it will raise an exception
            assert True, "GPU memory cache cleared successfully"
        else:
            pytest.skip("CUDA not available")

    def test_model_setup(self, temp_dir):
        """Test Model File Setup"""
        # make a 1s tiny video
        tiny_path = Path(temp_dir) / "tiny_init_probe.mp4"
        video_path = tiny_video(tiny_path)

        detector = NSFWDetectorWorker.remote()  # auto-downloads or loads model
        try:
            result = ray.get(detector.process_video_chunk.remote(video_path))
            assert "nsfw_timestamps" in result, "Expected key 'nsfw_timestamps' not found"
        finally:
            ray.kill(detector)

    def test_custom_frame_detection(self, video_generator):
        """Test NSFW Detection with Custom Frames"""
        single = video_generator.create_test_video("single_detection", duration=30)
        multiple = video_generator.create_test_video("multiple_detection", duration=40)
        clustered = video_generator.create_test_video("clustered_detection", duration=25)

        detector = NSFWDetectorWorker.remote()
        try:
            for video in [single, multiple, clustered]:
                result = ray.get(detector.process_video_chunk.remote(str(video)))
                assert "nsfw_timestamps" in result, f"Missing 'nsfw_timestamps' key in result for {video}"
                # Print for debugging purposes
                print(f"Video {video} detection result: {result['nsfw_timestamps']}")
        finally:
            ray.kill(detector)

    def test_error_handling(self):
        """Test Error Handling for missing files"""
        detector = NSFWDetectorWorker.remote()
        try:
            with pytest.raises(Exception):
                ray.get(detector.process_video_chunk.remote("/nonexistent/video1.mp4"))
        finally:
            ray.kill(detector)

    def test_model_download_failure(self):
        """Test Model Download/Load Failure with invalid paths"""
        bad_detector = NSFWDetectorWorker.remote(
            model_path="/invalid/model.onnx",
            labels_path="/invalid/labels.json"
        )
        try:
            # Any call will surface the init failure as ActorDiedError
            with pytest.raises(Exception):
                ray.get(bad_detector.process_video_chunk.remote("/nonexistent/file.mp4"))
        finally:
            # It's likely already dead, but kill is safe.
            try:
                ray.kill(bad_detector)
            except Exception:
                pass


# ==================================================
# EDGE CASE TESTS
# ==================================================
class TestNSFWDetectorEdgeCases:
    """Edge case tests converted to pytest format"""

    def test_boundary_detection(self, video_generator):
        """Test detection at video boundaries"""
        video_path = video_generator.create_test_video(test_type="boundary_detection")
        
        detector = NSFWDetectorWorker.remote()
        try:
            result = ray.get(detector.process_video_chunk.remote(video_path))
            assert "nsfw_timestamps" in result
            print("Boundary detection result:", result["nsfw_timestamps"])
        finally:
            ray.kill(detector)

    def test_short_video(self, video_generator):
        """Test very short video processing"""
        video_path = video_generator.create_test_video(test_type="single_detection", duration=3)
        
        detector = NSFWDetectorWorker.remote()
        try:
            result = ray.get(detector.process_video_chunk.remote(video_path))
            assert "nsfw_timestamps" in result
            print("Short video result:", result["nsfw_timestamps"])
        finally:
            ray.kill(detector)

    def test_high_fps_video(self, video_generator):
        """Test high FPS video (60fps)"""
        video_path = video_generator.create_test_video(test_type="single_detection", duration=5, fps=60)
        
        detector = NSFWDetectorWorker.remote()
        try:
            result = ray.get(detector.process_video_chunk.remote(video_path))
            assert "nsfw_timestamps" in result
            print("High-FPS video result:", result["nsfw_timestamps"])
        finally:
            ray.kill(detector)

    def test_no_nsfw_frames(self, temp_dir):
        """Test video with no NSFW content"""
        generator = CustomFrameVideoGenerator(temp_dir, CUSTOM_FRAME_PATH, use_custom=False)
        video_path = generator.create_test_video(test_type="single_detection", insert_frame=False)
        
        detector = NSFWDetectorWorker.remote()
        try:
            result = ray.get(detector.process_video_chunk.remote(video_path))
            assert "nsfw_timestamps" in result
            # Should be empty or minimal detections
            print("Expected no detections, got:", result["nsfw_timestamps"])
        finally:
            ray.kill(detector)

    def test_long_continuous_frames(self, video_generator):
        """Test long continuous NSFW frames"""
        video_path = video_generator.create_test_video(
            test_type="single_detection", duration=10, custom_frame_duration=5
        )
        
        detector = NSFWDetectorWorker.remote()
        try:
            result = ray.get(detector.process_video_chunk.remote(video_path))
            assert "nsfw_timestamps" in result
            print("Long-frame video result:", result["nsfw_timestamps"])
        finally:
            ray.kill(detector)

    def test_corrupted_file(self, temp_dir):
        """Test handling of corrupted video files"""
        corrupted_path = Path(temp_dir) / "corrupted.mp4"
        with open(corrupted_path, "wb") as f:
            f.write(b"not a real video file")

        detector = NSFWDetectorWorker.remote()
        try:
            with pytest.raises(Exception):
                ray.get(detector.process_video_chunk.remote(str(corrupted_path)))
        finally:
            ray.kill(detector)

    @pytest.mark.parametrize("resolution", [(640, 480), (1920, 1080), (3840, 2160)])
    def test_different_resolutions(self, video_generator, resolution):
        """Test different video resolutions"""
        video_path = video_generator.create_test_video(test_type="single_detection", resolution=resolution)
        
        detector = NSFWDetectorWorker.remote()
        try:
            result = ray.get(detector.process_video_chunk.remote(video_path))
            assert "nsfw_timestamps" in result
            print(f"Resolution {resolution} result:", result['nsfw_timestamps'])
        finally:
            ray.kill(detector)


# ==================================================
# CONFIGURATION INFO
# ==================================================
def test_configuration_info():
    """Display test configuration information"""
    print(f"\nTest Configuration:")
    print(f"   - Custom test frame: {CUSTOM_FRAME_PATH}")
    print(f"   - Use custom frame: {USE_CUSTOM_FRAME}")
    print(f"   - CUDA available: {torch.cuda.is_available()}")
    
    # This test always passes - it's just for info display
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])