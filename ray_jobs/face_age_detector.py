import ray
import cv2
import json
import numpy as np
import os
import time
import logging
import sys
from pathlib import Path
from datetime import timedelta

# Set up logger first
logger = logging.getLogger("face_age_detector")

# Try to import FaceAgeDetector with multiple strategies
try:
    # Strategy 1: Direct import if already in path
    from deepfacedetect import FaceAgeDetector
    logger.info("Successfully imported FaceAgeDetector (direct import)")
except ImportError:
    try:
        # Strategy 2: Add relative path to sys.path
        current_file_dir = Path(__file__).resolve().parent
        project_root = current_file_dir.parent
        video_src_path = project_root / "video-age-detection-pipeline" / "src"
        
        if video_src_path.exists():
            sys.path.insert(0, str(video_src_path))
            from deepfacedetect import FaceAgeDetector
            logger.info(f"Successfully imported FaceAgeDetector from {video_src_path}")
        else:
            raise ImportError(f"Path not found: {video_src_path}")
            
    except ImportError:
        try:
            # Strategy 3: Try absolute path from current working directory
            cwd = os.getcwd()
            if "video-age-detection-pipeline" in cwd:
                video_src_path = os.path.join(cwd, "src")
                sys.path.insert(0, video_src_path)
                from deepfacedetect import FaceAgeDetector
                logger.info(f"Successfully imported FaceAgeDetector from {video_src_path}")
            else:
                raise ImportError("video-age-detection-pipeline not found in current working directory")
                
        except ImportError:
            # Strategy 4: Try to find it in parent directories
            current_dir = Path(__file__).resolve().parent
            for i in range(5):  # Look up to 5 levels up
                parent_dir = current_dir.parents[i]
                video_src_path = parent_dir / "video-age-detection-pipeline" / "src"
                if video_src_path.exists():
                    sys.path.insert(0, str(video_src_path))
                    try:
                        from deepfacedetect import FaceAgeDetector
                        logger.info(f"Successfully imported FaceAgeDetector from {video_src_path}")
                        break
                    except ImportError:
                        continue
            else:
                # Final fallback: raise detailed error
                logger.error("All import strategies failed")
                logger.error(f"Current file: {__file__}")
                logger.error(f"Current directory: {os.getcwd()}")
                logger.error(f"Python path: {sys.path}")
                raise ImportError("Could not import FaceAgeDetector. Please ensure deepfacedetect.py is accessible.")


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
        # Import FaceAgeDetector inside the Ray function to ensure it works in worker environment
        import sys
        from pathlib import Path
        
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
        # Import FaceAgeDetector inside the Ray function to ensure it works in worker environment
        import sys
        from pathlib import Path
        
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


@ray.remote
def batch_process_videos_for_face_detection(video_paths: list, output_base_dir: str = "/tmp/face_analysis",
                                          frame_interval: int = 30, save_frames: bool = False):
    """
    Process multiple videos in parallel for face detection.
    
    Args:
        video_paths: List of paths to video files
        output_base_dir: Base directory for output files
        frame_interval: Frame sampling interval
        save_frames: Whether to save frames with detections
    
    Returns:
        Combined results from all videos
    """
    try:
        # Create output base directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Process videos in parallel
        futures = []
        for video_path in video_paths:
            # Create unique output directory for each video
            video_name = Path(video_path).stem
            video_output_dir = Path(output_base_dir) / video_name
            
            future = process_video_for_face_detection.remote(
                video_path, str(video_output_dir), frame_interval, save_frames
            )
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        
        # Aggregate results
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        total_faces_detected = sum(r.get("total_faces_detected", 0) for r in successful_results)
        total_processed_frames = sum(r.get("total_processed_frames", 0) for r in successful_results)
        
        # Save batch summary
        batch_summary_file = Path(output_base_dir) / "batch_summary.json"
        with open(batch_summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "total_videos_processed": len(successful_results),
                "total_videos_failed": len(failed_results),
                "total_faces_detected": total_faces_detected,
                "total_processed_frames": total_processed_frames,
                "individual_results": results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Batch processing completed. {len(successful_results)} videos processed successfully")
        logger.info(f"Total faces detected: {total_faces_detected}")
        
        return {
            "total_videos_processed": len(successful_results),
            "total_videos_failed": len(failed_results),
            "total_faces_detected": total_faces_detected,
            "total_processed_frames": total_processed_frames,
            "success": True,
            "batch_summary_file": str(batch_summary_file)
        }
        
    except Exception as e:
        logger.error(f"Error in batch_process_videos_for_face_detection: {e}")
        return {
            "error": str(e),
            "success": False
        }
