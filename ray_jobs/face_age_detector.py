import ray
import json
import os
import logging
import sys
from pathlib import Path
from datetime import timedelta

# Set up logger first
logger = logging.getLogger("face_age_detector")

# GPU Configuration - Ensure CUDA can access GPU
def configure_gpu():
    """Configure GPU settings to ensure CUDA can access the GPU."""
    # Clear any restrictive CUDA environment variables
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "":
        logger.info("Clearing restrictive CUDA_VISIBLE_DEVICES setting")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    
    # Set CUDA to use the first available GPU
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger.info("Set CUDA_VISIBLE_DEVICES=0 to use first GPU")
    
    # Set other GPU-related environment variables
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TensorFlow logging
    
    # Additional GPU configuration for better performance
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_GPU_THREAD_COUNT"] = "1"
    
    # Try to configure TensorFlow GPU memory growth
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Configured memory growth for {len(gpus)} GPU(s)")
    except ImportError:
        logger.info("TensorFlow not available during GPU configuration")
    except Exception as e:
        logger.warning(f"Could not configure TensorFlow GPU memory growth: {e}")
    
    logger.info(f"GPU Configuration: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

# Configure GPU before any other imports
configure_gpu()

def verify_gpu_access():
    """Verify that GPU is accessible and working."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"✅ GPU access verified: {len(gpus)} GPU(s) available")
            for gpu in gpus:
                logger.info(f"   - {gpu.name}")
            return True
        else:
            logger.warning("⚠️ No GPUs detected by TensorFlow")
            return False
    except ImportError:
        logger.warning("⚠️ TensorFlow not available, cannot verify GPU access")
        return False
    except Exception as e:
        logger.error(f"❌ Error verifying GPU access: {e}")
        return False

# Don't import FaceAgeDetector at module level - import it inside the Ray functions
# This prevents import errors when the module is imported on different machines


@ray.remote
def process_video_for_face_detection(video_path: str, output_dir: str = "/tmp/face_analysis",
                                   frame_interval: int = 30, save_frames: bool = False):
    """
    Process a single video file for face detection without chunking.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory for output files
        frame_interval: Frame sampling interval
        save_frames: Whether to save frames with detections
    
    Returns:
        Face analysis results for the entire video
    """
    try:
        # Configure GPU in Ray worker process
        configure_gpu()
        
        # Verify GPU access in worker process
        gpu_available = verify_gpu_access()
        if not gpu_available:
            logger.warning("GPU not accessible in Ray worker, will use CPU")
        
        # Import FaceAgeDetector inside the Ray function to ensure it works in worker environment
        
        # Try to find and import FaceAgeDetector
        current_file_dir = Path(__file__).resolve().parent
        project_root = current_file_dir.parent
        video_src_path = project_root / "video-age-detection-pipeline" / "src"
        
        if video_src_path.exists():
            sys.path.insert(0, str(video_src_path))
            from deepfacedetect import FaceAgeDetector
            logger.info(f"Successfully imported FaceAgeDetector in Ray worker from {video_src_path}")
        else:
            # Try alternative paths
            alt_paths = [
                "video-age-detection-pipeline/src",
                "../video-age-detection-pipeline/src",
                "../../video-age-detection-pipeline/src"
            ]
            
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    sys.path.insert(0, alt_path)
                    from deepfacedetect import FaceAgeDetector
                    logger.info(f"Successfully imported FaceAgeDetector in Ray worker from {alt_path}")
                    break
            else:
                raise ImportError(f"Could not find video-age-detection-pipeline/src directory")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the entire video using FaceAgeDetector
        logger.info(f"Processing video {video_path} for face detection")
        
        # Create a temporary detector instance for this task
        detector = FaceAgeDetector()
        
        # Override config if frame_interval is provided
        if frame_interval and hasattr(detector.config, 'FRAME_INTERVAL'):
            detector.config.FRAME_INTERVAL = frame_interval
        
        # Override save frames setting if provided
        if save_frames is not None and hasattr(detector.config, 'SAVE_FRAMES'):
            detector.config.SAVE_FRAMES = save_frames
        
        # Process the video
        results = detector.process_video(video_path)
        
        # Get processing summary
        summary = detector.get_processing_summary()
        
        # Save results
        output_file = Path(output_dir) / "face_analysis_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "video_path": str(video_path),
                "frame_interval": detector.config.FRAME_INTERVAL,
                "total_faces_detected": summary["total_faces_detected"],
                "total_processed_frames": summary["processed_frames"],
                "frames_with_errors": summary["frames_with_errors"],
                "face_analysis_results": detector.frames_json
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Face analysis completed. Results saved to {output_file}")
        logger.info(f"Total faces detected: {summary['total_faces_detected']}")
        
        return {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "total_faces_detected": summary["total_faces_detected"],
            "total_processed_frames": summary["processed_frames"],
            "frames_with_errors": summary["frames_with_errors"],
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in process_video_for_face_detection: {e}")
        return {
            "error": str(e),
            "success": False
        }


@ray.remote
def process_video_chunks_for_face_detection(chunk_paths: list, config=None, 
                                          frame_interval: int = None, save_frames: bool = False,
                                          chunk_duration_sec: int = 60):
    """
    Process video chunks in parallel for face age detection.
    
    Args:
        chunk_paths: List of paths to video chunks
        config: Configuration object for FaceAgeDetector
        frame_interval: Frame sampling interval (overrides config if provided)
        save_frames: Whether to save frames with detections
        chunk_duration_sec: Duration of each chunk (for timestamp calculation)
    
    Returns:
        Combined results from all chunks
    """
    try:
        # Configure GPU in Ray worker process
        configure_gpu()
        
        # Verify GPU access in worker process
        gpu_available = verify_gpu_access()
        if not gpu_available:
            logger.warning("GPU not accessible in Ray worker, will use CPU")
        
        # Import FaceAgeDetector inside the Ray function to ensure it works in worker environment
        
        # Try to find and import FaceAgeDetector
        current_file_dir = Path(__file__).resolve().parent
        project_root = current_file_dir.parent
        video_src_path = project_root / "video-age-detection-pipeline" / "src"
        
        if video_src_path.exists():
            sys.path.insert(0, str(video_src_path))
            from deepfacedetect import FaceAgeDetector
            logger.info(f"Successfully imported FaceAgeDetector in Ray worker from {video_src_path}")
        else:
            # Try alternative paths
            alt_paths = [
                "video-age-detection-pipeline/src",
                "../video-age-detection-pipeline/src",
                "../../video-age-detection-pipeline/src"
            ]
            
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    sys.path.insert(0, alt_path)
                    from deepfacedetect import FaceAgeDetector
                    logger.info(f"Successfully imported FaceAgeDetector in Ray worker from {alt_path}")
                    break
            else:
                raise ImportError(f"Could not find video-age-detection-pipeline/src directory")
        
        # Process chunks in parallel
        futures = []
        logger.info(f"Processing {len(chunk_paths)} chunks for face detection")
        for i, chunk_path in enumerate(chunk_paths):
            chunk_offset = i * chunk_duration_sec  # Calculate time offset for this chunk
            # Create unique output directory based on chunk path to avoid conflicts between views
            chunk_name = Path(chunk_path).stem
            unique_output_dir = f"/tmp/chunk_{chunk_name}_{i}"
            logger.info(f"Processing chunk {i}: {chunk_path} -> {unique_output_dir}")
            future = process_video_for_face_detection.remote(
                chunk_path, unique_output_dir, frame_interval, save_frames
            )
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        # Combine all face detection results
        all_face_results = []
        total_processing_time = 0
        total_faces_detected = 0
        total_processed_frames = 0
        
        for i, result in enumerate(results):
            if result["success"]:
                # Adjust timestamps to account for chunk offset
                chunk_offset = i * chunk_duration_sec
                
                # Get the face analysis results from the detector
                chunk_output_dir = Path(result["output_dir"])
                chunk_results_file = chunk_output_dir / "face_analysis_results.json"
                
                logger.info(f"Reading results from chunk {i}: {chunk_results_file}")
                
                if chunk_results_file.exists():
                    with open(chunk_results_file, 'r', encoding="utf-8") as f:
                        chunk_data = json.load(f)
                        face_results = chunk_data.get("face_analysis_results", [])
                        logger.info(f"Found {len(face_results)} face detection results in chunk {i}")
                        
                        # Adjust timestamps for each frame result
                        for frame_result in face_results:
                            if "timestamp" in frame_result:
                                # Parse timestamp and add offset
                                try:
                                    time_parts = frame_result["timestamp"].split(":")
                                    hours, minutes, seconds = map(int, time_parts)
                                    total_seconds = hours * 3600 + minutes * 60 + seconds + chunk_offset
                                    adjusted_time = str(timedelta(seconds=int(total_seconds)))
                                    frame_result["timestamp"] = adjusted_time
                                    frame_result["chunk_offset_seconds"] = chunk_offset
                                    frame_result["chunk_file"] = chunk_paths[i]
                                except:
                                    pass
                            
                            all_face_results.append(frame_result)
                else:
                    logger.warning(f"Results file not found for chunk {i}: {chunk_results_file}")
                
                total_faces_detected += result["total_faces_detected"]
                total_processed_frames += result["total_processed_frames"]
                total_processing_time += result.get("processing_time", 0)
        
        # Sort by frame number (which should correspond to chronological order)
        all_face_results.sort(key=lambda x: x.get("frame_num", 0))
        
        return {
            "total_faces_detected": total_faces_detected,
            "total_processed_frames": total_processed_frames,
            "total_processing_time_seconds": round(total_processing_time, 2),
            "face_analysis_results": all_face_results,
            "chunks_processed": len(chunk_paths),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in process_video_chunks_for_face_detection: {e}")
        return {
            "error": str(e),
            "success": False
        }



if __name__ == "__main__":
    """Test with sample video if provided."""
    if len(sys.argv) > 1:
        sample_video = sys.argv[1]
        print(f"🎬 Testing with sample video: {sample_video}")
        
        if not Path(sample_video).exists():
            print(f"❌ Video file not found: {sample_video}")
            print("Usage: python ray_jobs/face_age_detector.py [path_to_sample_video.mp4]")
            sys.exit(1)
        
        try:
            # Test the Ray function directly
            print("🚀 Testing face detection on sample video...")
            
            # Verify GPU access before running
            print("🔍 Checking GPU access...")
            gpu_available = verify_gpu_access()
            if gpu_available:
                print("✅ GPU is accessible and ready for processing")
            else:
                print("⚠️ GPU access issues detected, will fall back to CPU")
            
            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init()
            
            # Test the process_video_for_face_detection function
            result = ray.get(process_video_for_face_detection.remote(
                sample_video, 
                "/tmp/test_face_detection", 
                frame_interval=30, 
                save_frames=False
            ))
            
            if result["success"]:
                print("✅ Face detection test successful!")
                print(f"   - Total faces detected: {result['total_faces_detected']}")
                print(f"   - Total frames processed: {result['total_processed_frames']}")
                print(f"   - Output directory: {result['output_dir']}")
                print(f"   - Results saved to: {result['output_dir']}/face_analysis_results.json")
            else:
                print(f"❌ Face detection test failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error during face detection test: {e}")
            
        finally:
            # Shutdown Ray
            if ray.is_initialized():
                ray.shutdown()
    else:
        print("💡 To test with a sample video, run:")
        print("   python ray_jobs/face_age_detector.py path/to/sample_video.mp4")

