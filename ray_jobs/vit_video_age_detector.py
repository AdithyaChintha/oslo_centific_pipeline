import ray
import json
import os
import logging
import sys
from pathlib import Path
from datetime import timedelta

# Logger
logger = logging.getLogger("vit_video_age_detector")


# GPU Configuration - make GPU available when present
def configure_gpu():
    """Configure GPU settings for compatibility on mixed environments."""
    # Clear any restrictive CUDA env vars
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
        logger.info("Clearing restrictive CUDA_VISIBLE_DEVICES setting")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # Default to first GPU when not set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger.info("Set CUDA_VISIBLE_DEVICES=0 to use first GPU (if available)")

    # TensorFlow GPU softness (harmless if TF is absent)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_GPU_THREAD_COUNT"] = "1"


def verify_gpu_access():
    """Log CUDA/TensorFlow availability details (best-effort)."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ PyTorch CUDA available")
        else:
            logger.warning("⚠️ PyTorch CUDA not available; using CPU")
    except Exception as e:
        logger.info(f"PyTorch not available for CUDA check: {e}")

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"✅ TensorFlow sees {len(gpus)} GPU(s)")
        else:
            logger.info("TensorFlow sees no GPUs")
    except Exception:
        logger.info("TensorFlow not available for GPU verification (ok)")


# Apply GPU config at import time
configure_gpu()


@ray.remote
def process_video_for_vit_age_detection(video_path: str, output_dir: str = "/tmp/vit_face_age",
                                        frame_interval: int = 30, save_frames: bool = False):
    """
    Process a single video with ViT age classifier and InsightFace detection.

    Args:
        video_path: Input video file path
        output_dir: Directory to write summary JSON
        frame_interval: Sample every Nth frame
        save_frames: Save annotated frames (honors global config as well)

    Returns:
        Dict with summary and output paths
    """
    try:
        configure_gpu()
        verify_gpu_access()

        # Resolve project structure and import detector
        current_file_dir = Path(__file__).resolve().parent
        project_root = current_file_dir.parent
        video_src_path = project_root / "video-age-detection-pipeline" / "src"

        if video_src_path.exists():
            sys.path.insert(0, str(video_src_path))
            from vit_video_agedetect import ViTVideoAgeDetector  # type: ignore
            logger.info(f"Imported ViTVideoAgeDetector from {video_src_path}")
        else:
            alt_paths = [
                "video-age-detection-pipeline/src",
                "../video-age-detection-pipeline/src",
                "../../video-age-detection-pipeline/src",
            ]
            for alt in alt_paths:
                if Path(alt).exists():
                    sys.path.insert(0, alt)
                    from vit_video_agedetect import ViTVideoAgeDetector  # type: ignore
                    logger.info(f"Imported ViTVideoAgeDetector from {alt}")
                    break
            else:
                raise ImportError("Could not find video-age-detection-pipeline/src for ViTVideoAgeDetector")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create detector
        detector = ViTVideoAgeDetector()
        # Override sampling interval if provided
        if frame_interval and hasattr(detector.config, 'FRAME_INTERVAL'):
            detector.config.FRAME_INTERVAL = frame_interval
        # Control frame saving
        if save_frames is not None and hasattr(detector.config, 'SAVE_FRAMES'):
            detector.config.SAVE_FRAMES = save_frames

        # Process
        logger.info(f"Processing video with ViTVideoAgeDetector: {video_path}")
        results = detector.process_video(video_path)

        # Summary
        summary = detector.get_processing_summary()

        # Save combined summary JSON
        output_file = Path(output_dir) / "vit_face_age_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "video_path": str(video_path),
                "frame_interval": detector.config.FRAME_INTERVAL,
                "total_faces_detected": summary["total_faces_detected"],
                "total_processed_frames": summary["processed_frames"],
                "frames_with_errors": summary["frames_with_errors"],
                "face_analysis_results": detector.frames_json,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"ViT age analysis complete: {output_file}")
        return {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "total_faces_detected": summary["total_faces_detected"],
            "total_processed_frames": summary["processed_frames"],
            "frames_with_errors": summary["frames_with_errors"],
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error in process_video_for_vit_age_detection: {e}")
        return {"error": str(e), "success": False}


@ray.remote
def process_video_chunks_for_vit_age_detection(chunk_paths: list, config=None,
                                               frame_interval: int | None = None, save_frames: bool = False,
                                               chunk_duration_sec: int = 60):
    """
    Process multiple video chunks in parallel and combine results.
    """
    try:
        configure_gpu()
        verify_gpu_access()

        # Import detector path for workers
        current_file_dir = Path(__file__).resolve().parent
        project_root = current_file_dir.parent
        video_src_path = project_root / "video-age-detection-pipeline" / "src"

        if video_src_path.exists():
            sys.path.insert(0, str(video_src_path))
            from vit_video_agedetect import ViTVideoAgeDetector  # type: ignore # noqa: F401 - ensure import works
            logger.info(f"Imported ViTVideoAgeDetector in chunks worker from {video_src_path}")
        else:
            alt_paths = [
                "video-age-detection-pipeline/src",
                "../video-age-detection-pipeline/src",
                "../../video-age-detection-pipeline/src",
            ]
            for alt in alt_paths:
                if Path(alt).exists():
                    sys.path.insert(0, alt)
                    from vit_video_agedetect import ViTVideoAgeDetector  # type: ignore # noqa: F401
                    logger.info(f"Imported ViTVideoAgeDetector in chunks worker from {alt}")
                    break
            else:
                raise ImportError("Could not find video-age-detection-pipeline/src for ViTVideoAgeDetector")

        # Launch per-chunk tasks
        futures = []
        logger.info(f"Processing {len(chunk_paths)} chunks for ViT age detection")
        for i, chunk_path in enumerate(chunk_paths):
            chunk_offset = i * chunk_duration_sec
            chunk_name = Path(chunk_path).stem
            unique_output_dir = f"/tmp/vit_chunk_{chunk_name}_{i}"
            logger.info(f"Submitting chunk {i}: {chunk_path} -> {unique_output_dir}")
            future = process_video_for_vit_age_detection.remote(
                chunk_path, unique_output_dir, frame_interval or 30, save_frames
            )
            futures.append(future)

        results = ray.get(futures)

        # Combine
        all_face_results = []
        total_processing_time = 0
        total_faces_detected = 0
        total_processed_frames = 0

        for i, result in enumerate(results):
            if result.get("success"):
                chunk_offset = i * chunk_duration_sec
                chunk_output_dir = Path(result["output_dir"])
                chunk_results_file = chunk_output_dir / "vit_face_age_results.json"

                logger.info(f"Reading chunk {i} results: {chunk_results_file}")
                if chunk_results_file.exists():
                    with open(chunk_results_file, "r", encoding="utf-8") as f:
                        chunk_data = json.load(f)
                        face_results = chunk_data.get("face_analysis_results", [])
                        logger.info(f"Chunk {i} contains {len(face_results)} results")

                        for frame_result in face_results:
                            if "timestamp" in frame_result:
                                try:
                                    time_parts = frame_result["timestamp"].split(":")
                                    hours, minutes, seconds = map(int, time_parts)
                                    total_seconds = hours * 3600 + minutes * 60 + seconds + chunk_offset
                                    adjusted_time = str(timedelta(seconds=int(total_seconds)))
                                    frame_result["timestamp"] = adjusted_time
                                    frame_result["chunk_offset_seconds"] = chunk_offset
                                    frame_result["chunk_file"] = chunk_paths[i]
                                except Exception:
                                    pass
                            all_face_results.append(frame_result)
                else:
                    logger.warning(f"Results file missing for chunk {i}: {chunk_results_file}")

                total_faces_detected += result.get("total_faces_detected", 0)
                total_processed_frames += result.get("total_processed_frames", 0)
                total_processing_time += result.get("processing_time", 0)

        all_face_results.sort(key=lambda x: x.get("frame_num", 0))

        return {
            "total_faces_detected": total_faces_detected,
            "total_processed_frames": total_processed_frames,
            "total_processing_time_seconds": round(total_processing_time, 2),
            "face_analysis_results": all_face_results,
            "chunks_processed": len(chunk_paths),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error in process_video_chunks_for_vit_age_detection: {e}")
        return {"error": str(e), "success": False}


if __name__ == "__main__":
    """Test with sample video if provided."""
    if len(sys.argv) > 1:
        sample_video = sys.argv[1]
        print(f"🎬 Testing ViT video age detection: {sample_video}")

        if not Path(sample_video).exists():
            print(f"❌ Video file not found: {sample_video}")
            print("Usage: python ray_jobs/vit_video_age_detector.py [path_to_sample_video.mp4]")
            sys.exit(1)

        try:
            print("🔍 Checking GPU access...")
            verify_gpu_access()

            if not ray.is_initialized():
                ray.init()

            result = ray.get(process_video_for_vit_age_detection.remote(
                sample_video,
                "/tmp/test_vit_face_age",
                frame_interval=30,
                save_frames=True,
            ))

            if result.get("success"):
                print("✅ ViT age detection test successful")
                print(f"   - Total faces: {result['total_faces_detected']}")
                print(f"   - Total frames: {result['total_processed_frames']}")
                print(f"   - Output dir: {result['output_dir']}")
            else:
                print(f"❌ Test failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error during test: {e}")
        finally:
            if ray.is_initialized():
                ray.shutdown()
    else:
        print("💡 To test with a sample video, run:")
        print("   python ray_jobs/vit_video_age_detector.py path/to/sample_video.mp4")


