import ray
from ray import serve
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from utils.logger import get_logger
from ray_jobs.face_age_detector_optimized import (
    CachedModelManager,
    configure_tf_for_ray_worker,
    cleanup_tensorflow_memory,
    process_video_for_face_detection_optimized_sync
)

logger = get_logger("FaceDetectionServe")


@serve.deployment(
    name="face-detection",
    num_replicas=2,  # Number of model instances (2 for load balancing)
    ray_actor_options={
        "num_gpus": 0.20  # GPU allocation per replica
    }
)
class FaceDetectionService:
    """
    Ray Serve deployment for face detection inference.

    Models are loaded ONCE when the deployment starts, then serve
    many inference requests without reloading.

    Benefits over Ray Actors:
    - Built-in load balancing across replicas
    - Auto-scaling based on request load
    - Health checks and automatic recovery
    - Request batching for better GPU utilization
    - Easy deployment management
    """

    def __init__(self):
        """Initialize models once when deployment starts."""
        logger.info("Face Detection Service initializing...")
        self.start_time = time.time()
        self.requests_processed = 0
        self.total_processing_time = 0

        # Configure GPU
        logger.info("Configuring TensorFlow GPU...")
        configure_tf_for_ray_worker()

        # Load models ONCE
        logger.info("Loading face detection models...")
        model_load_start = time.time()

        self.model_manager = CachedModelManager()
        self.detector = self.model_manager.get_face_age_detector()

        self.model_load_time = time.time() - model_load_start

        logger.info(f"Face Detection Service ready!")
        logger.info(f"Model load time: {self.model_load_time:.2f}s")
        logger.info(f"Ready to serve requests")

    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle inference request.

        This is called for each video that needs face detection.
        Models are already loaded, so only processing time is incurred.

        Args:
            request: Dictionary with keys:
                - video_path: Path to video file
                - output_dir: Output directory for results
                - frame_interval: Frame sampling interval (default: 30)
                - save_frames: Whether to save frames (default: False)
                - chunk_offset_seconds: Offset for timestamps (default: 0.0)
                - batch_size: Batch size for processing (default: 8)
                - enable_timing: Enable timing metrics (default: False)
                - timing_id: Timing identifier (optional)

        Returns:
            Dictionary with face detection results
        """
        start_time = time.time()
        self.requests_processed += 1

        # Extract request parameters
        video_path = request.get("video_path")
        output_dir = request.get("output_dir", "/tmp/face_analysis")
        frame_interval = request.get("frame_interval", 30)
        save_frames = request.get("save_frames", False)
        chunk_offset_seconds = request.get("chunk_offset_seconds", 0.0)
        batch_size = request.get("batch_size", 8)
        enable_timing = request.get("enable_timing", False)
        timing_id = request.get("timing_id", None)

        logger.info(f"Processing request #{self.requests_processed}: {Path(video_path).name}")

        try:
            # Process using pre-loaded detector
            result = process_video_for_face_detection_optimized_sync(
                video_path=video_path,
                output_dir=output_dir,
                frame_interval=frame_interval,
                save_frames=save_frames,
                chunk_offset_seconds=chunk_offset_seconds,
                batch_size=batch_size
            )

            # Add service metadata
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            result["service_metadata"] = {
                "replica_id": serve.get_replica_context().replica_tag,
                "request_number": self.requests_processed,
                "model_load_time": self.model_load_time,
                "service_uptime": time.time() - self.start_time,
                "avg_processing_time": self.total_processing_time / self.requests_processed
            }

            # Add timing if requested
            if enable_timing:
                result["timing"] = {
                    "execution_time": processing_time,
                    "start_time": datetime.utcnow().isoformat() + "Z",
                    "end_time": datetime.utcnow().isoformat() + "Z",
                    "timing_id": timing_id,
                    "status": "completed"
                }

            logger.info(f"Request #{self.requests_processed} completed in {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Request #{self.requests_processed} failed: {e}")

            error_result = {
                "error": str(e),
                "success": False,
                "total_faces_detected": 0,
                "total_processed_frames": 0,
                "flagged_segments": [],
                "service_metadata": {
                    "replica_id": serve.get_replica_context().replica_tag,
                    "request_number": self.requests_processed,
                    "model_load_time": self.model_load_time
                }
            }

            if enable_timing:
                error_result["timing"] = {
                    "execution_time": time.time() - start_time,
                    "start_time": datetime.utcnow().isoformat() + "Z",
                    "end_time": datetime.utcnow().isoformat() + "Z",
                    "timing_id": timing_id,
                    "status": "failed"
                }

            return error_result


# Deployment configuration
face_detection_deployment = FaceDetectionService.bind()
