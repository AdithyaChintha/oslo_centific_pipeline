import ray
from ray import serve
from typing import Dict, Any, List
from utils.logger import get_logger

logger = get_logger("FaceDetectionServeManager")


class FaceDetectionServeManager:
    """
    Manager for Ray Serve face detection deployment.

    This provides a simple interface for starting/stopping the service
    and making inference requests.

    Usage:
        manager = FaceDetectionServeManager()
        manager.start()

        # Process videos
        for video_path in video_paths:
            result = manager.detect_faces(video_path)

        manager.stop()
    """

    def __init__(self, num_replicas: int = 2, num_gpus_per_replica: float = 0.20):
        """
        Initialize manager.

        Args:
            num_replicas: Number of service replicas (default: 2)
            num_gpus_per_replica: GPU allocation per replica (default: 0.20)
        """
        self.num_replicas = num_replicas
        self.num_gpus_per_replica = num_gpus_per_replica
        self.handle = None
        self.is_running = False

    def start(self):
        """Start the Ray Serve deployment."""
        if self.is_running:
            logger.warning("Service already running")
            return

        logger.info("Starting Face Detection Service...")
        logger.info(f"Replicas: {self.num_replicas}")
        logger.info(f"GPU per replica: {self.num_gpus_per_replica}")

        # Clean up any existing Ray Serve instance first
        try:
            serve.shutdown()
            logger.info("Cleaned up existing Ray Serve instance")
        except Exception:
            pass

        # Initialize Ray Serve with custom port to avoid conflicts
        # Try ports 9000-9003 to avoid conflicts with existing services on 8000-8001
        ports_to_try = [9000, 9001, 9002, 9003]
        serve_started = False

        for port in ports_to_try:
            try:
                serve.start(detached=False, http_options={"host": "127.0.0.1", "port": port})
                logger.info(f"Ray Serve started successfully on port {port}")
                serve_started = True
                break
            except Exception as e:
                logger.debug(f"Port {port} unavailable: {e}")
                continue

        if not serve_started:
            logger.warning("Could not start Ray Serve on ports 9000-9003, may already be running")

        # Import deployment
        from ray_jobs.face_detection_serve import FaceDetectionService

        # Deploy service
        deployment = FaceDetectionService.options(
            num_replicas=self.num_replicas,
            ray_actor_options={"num_gpus": self.num_gpus_per_replica}
        ).bind()

        self.handle = serve.run(deployment, name="face-detection", route_prefix="/face-detection")
        self.is_running = True

        logger.info("Face Detection Service started successfully!")
        logger.info("Service is now ready to accept requests")

    def detect_faces(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        Detect faces in a video.

        Args:
            video_path: Path to video file
            **kwargs: Additional arguments (output_dir, frame_interval, etc.)

        Returns:
            Face detection results
        """
        if not self.is_running:
            raise RuntimeError("Service not running. Call start() first.")

        # Prepare request
        request = {
            "video_path": video_path,
            **kwargs
        }

        # Make synchronous request to service
        result = ray.get(self.handle.remote(request))
        return result

    async def detect_faces_async(self, video_path: str, **kwargs) -> Dict[str, Any]:
        """
        Detect faces in a video (async version).

        Args:
            video_path: Path to video file
            **kwargs: Additional arguments

        Returns:
            Face detection results
        """
        if not self.is_running:
            raise RuntimeError("Service not running. Call start() first.")

        request = {
            "video_path": video_path,
            **kwargs
        }

        result = await self.handle.remote(request)
        return result

    def detect_faces_batch(self, video_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Detect faces in multiple videos in parallel.

        Args:
            video_paths: List of video file paths
            **kwargs: Additional arguments applied to all videos

        Returns:
            List of face detection results
        """
        if not self.is_running:
            raise RuntimeError("Service not running. Call start() first.")

        logger.info(f"Processing {len(video_paths)} videos in parallel...")

        # Submit all requests in parallel
        refs = []
        for video_path in video_paths:
            request = {
                "video_path": video_path,
                **kwargs
            }
            ref = self.handle.remote(request)
            refs.append(ref)

        # Wait for all results
        results = ray.get(refs)

        logger.info(f"Batch processing complete: {len(results)} videos processed")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics from all replicas."""
        if not self.is_running:
            return {"error": "Service not running"}

        # Return basic info
        return {
            "num_replicas": self.num_replicas,
            "num_gpus_per_replica": self.num_gpus_per_replica,
            "is_running": self.is_running
        }

    def stop(self):
        """Stop the Ray Serve deployment."""
        if not self.is_running:
            logger.warning("Service not running")
            return

        logger.info("Stopping Face Detection Service...")

        try:
            serve.delete("face-detection")
        except Exception as e:
            logger.warning(f"Error stopping service: {e}")

        self.handle = None
        self.is_running = False

        logger.info("Face Detection Service stopped")

    def __enter__(self):
        """Context manager support."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.stop()
