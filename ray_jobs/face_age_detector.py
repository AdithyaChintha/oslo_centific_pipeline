import ray
import json
import os
import logging
import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict, Counter

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

def find_face_detector_module():
    """Find the face detector module from various possible paths"""
    # Get current file directory
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    # Possible paths to check
    possible_paths = [
        # From your provided path
        Path("/home/nvcoe_admin/code/oslo/whole_pipeline_testing/insta360-video-activity-segmentation/video-age-detection-pipeline/src"),
        # Relative to project root
        project_root / "video-age-detection-pipeline" / "src",
        # Relative to current directory
        current_dir / "video-age-detection-pipeline" / "src",
        # Alternative relative paths
        current_dir.parent / "video-age-detection-pipeline" / "src",
        current_dir.parent.parent / "video-age-detection-pipeline" / "src",
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "deepfacedetect.py").exists():
            logger.info(f"Found face detector module at: {path}")
            return str(path)
    
    # If none found, return None
    logger.error("Could not find video-age-detection-pipeline/src directory")
    logger.info("Searched paths:")
    for path in possible_paths:
        logger.info(f"  - {path} (exists: {path.exists()})")
    
    return None

def import_face_detector():
    """Import FaceAgeDetector with proper path handling"""
    face_detector_path = find_face_detector_module()
    
    if not face_detector_path:
        raise ImportError("Could not find video-age-detection-pipeline/src directory")
    
    # Add to sys.path if not already there
    if face_detector_path not in sys.path:
        sys.path.insert(0, face_detector_path)
    
    try:
        from deepfacedetect import FaceAgeDetector
        logger.info("Successfully imported FaceAgeDetector")
        return FaceAgeDetector
    except ImportError as e:
        logger.error(f"Failed to import FaceAgeDetector: {e}")
        raise

def extract_detailed_analysis_summary(frames_json: list):
    """Extract comprehensive analysis from DeepFace results - FIXED for your data format"""
    summary = {
        "gender_distribution": defaultdict(int),
        "age_distribution": {"minors": 0, "adults": 0, "seniors": 0},
        "emotion_distribution": defaultdict(int),
        "race_distribution": defaultdict(int),
        "total_unique_faces": 0,
        "frames_with_faces": 0
    }
    
    if not frames_json:
        return summary
    
    face_count = 0
    frames_with_faces = 0
    
    for frame_data in frames_json:
        faces = frame_data.get('faces', [])
        if faces:
            frames_with_faces += 1
            
        for face in faces:
            face_count += 1
            
            # Gender analysis - FIXED: use 'gender_label' instead of 'gender'
            if 'gender_label' in face:
                gender = face['gender_label']
                summary["gender_distribution"][gender] += 1
            
            # Age analysis - FIXED: use direct 'age' value instead of nested dict
            if 'age' in face:
                age = face['age']  # Direct integer, not nested dict
                if isinstance(age, (int, float)):
                    if age < 18:
                        summary["age_distribution"]["minors"] += 1
                    elif age < 65:
                        summary["age_distribution"]["adults"] += 1
                    else:
                        summary["age_distribution"]["seniors"] += 1
            
            # Note: emotion and race data not present in your current format
            # Add these when available in your DeepFace output
    
    summary["total_unique_faces"] = face_count
    summary["frames_with_faces"] = frames_with_faces
    
    # Convert defaultdicts to regular dicts for JSON serialization
    summary["gender_distribution"] = dict(summary["gender_distribution"])
    summary["emotion_distribution"] = dict(summary["emotion_distribution"])
    summary["race_distribution"] = dict(summary["race_distribution"])
    
    return summary

def process_video_for_face_detection_sync(video_path: str, output_dir: str = "/tmp/face_analysis",
                                        frame_interval: int = 30, save_frames: bool = False,
                                        chunk_offset_seconds: float = 0.0):
    """
    Core face detection processing function (non-remote version)
    """
    try:
        # Configure GPU in worker process
        configure_gpu()
        
        # Verify GPU access
        gpu_available = verify_gpu_access()
        if not gpu_available:
            logger.warning("GPU not accessible, will use CPU")
        
        # Import FaceAgeDetector
        FaceAgeDetector = import_face_detector()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the video using FaceAgeDetector
        logger.info(f"Processing video {video_path} for face detection")
        
        # Create detector instance
        detector = FaceAgeDetector()
        
        # Override config if frame_interval is provided
        if frame_interval and hasattr(detector.config, 'FRAME_INTERVAL'):
            detector.config.FRAME_INTERVAL = frame_interval
        
        # Override save frames setting if provided
        if save_frames is not None and hasattr(detector.config, 'SAVE_FRAMES'):
            detector.config.SAVE_FRAMES = save_frames
        
        # Process the video
        results = detector.process_video(video_path, output_dir)
        
        # Get processing summary
        summary = detector.get_processing_summary()
        
        # Extract detailed analysis from all DeepFace results
        detailed_summary = extract_detailed_analysis_summary(detector.frames_json)
        
        # Convert frame-based results to time-based segments
        flagged_segments = []
        if hasattr(detector, 'frames_json') and detector.frames_json:
            flagged_segments = convert_frames_to_segments(detector.frames_json, chunk_offset_seconds)
        
        # Prepare comprehensive results
        face_result = {
            "video_path": str(video_path),
            "chunk_offset_seconds": chunk_offset_seconds,
            "frame_interval": getattr(detector.config, 'FRAME_INTERVAL', frame_interval),
            "total_faces_detected": summary["total_faces_detected"],
            "total_processed_frames": summary["processed_frames"],
            "frames_with_errors": summary["frames_with_errors"],
            "flagged_segments": flagged_segments,
            
            # Detailed DeepFace analysis results
            "detailed_analysis_summary": detailed_summary,
            "raw_deepface_results": detector.frames_json,
            
            # Processing metadata
            "processing_info": {
                "gpu_available": gpu_available,
                "frame_interval": frame_interval,
                "save_frames": save_frames,
                "chunk_offset_seconds": chunk_offset_seconds
            }
        }
        
        # Save detailed results to file
        output_file = Path(output_dir) / "face_analysis_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(face_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Face analysis completed. Results saved to {output_file}")
        logger.info(f"Total faces detected: {summary['total_faces_detected']}")
        logger.info(f"Flagged segments: {len(flagged_segments)}")
        logger.info(f"Gender distribution: {detailed_summary['gender_distribution']}")
        logger.info(f"Age distribution: {detailed_summary['age_distribution']}")
        
        # Return pipeline-compatible result
        return {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "total_faces_detected": summary["total_faces_detected"],
            "total_processed_frames": summary["processed_frames"],
            "frames_with_errors": summary["frames_with_errors"],
            "flagged_segments": flagged_segments,
            "detailed_analysis_summary": detailed_summary,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in process_video_for_face_detection_sync: {e}")
        return {
            "error": str(e),
            "success": False,
            "total_faces_detected": 0,
            "total_processed_frames": 0,
            "flagged_segments": []
        }

@ray.remote
def process_video_for_face_detection(video_path: str, output_dir: str = "/tmp/face_analysis",
                                   frame_interval: int = 30, save_frames: bool = False,
                                   chunk_offset_seconds: float = 0.0):
    """
    Ray remote wrapper for face detection processing
    """
    return process_video_for_face_detection_sync(
        video_path, output_dir, frame_interval, save_frames, chunk_offset_seconds
    )

def convert_frames_to_segments(frames_json: list, chunk_offset_seconds: float = 0.0, max_gap_seconds: float = 3.0):
    """Convert frame-based face detections to time-based segments with full DeepFace data"""
    if not frames_json:
        return []
    
    flagged_segments = []
    
    # Filter frames that have faces detected
    frames_with_faces = []
    for frame_data in frames_json:
        faces = frame_data.get('faces', [])
        if faces and len(faces) > 0:
            timestamp = parse_timestamp(frame_data.get('timestamp', '0:0:0')) + chunk_offset_seconds
            frames_with_faces.append({
                'timestamp': timestamp,
                'face_count': len(faces),
                'faces': faces,
                'frame_data': frame_data
            })
    
    if not frames_with_faces:
        return []
    
    # Sort by timestamp
    frames_with_faces.sort(key=lambda x: x['timestamp'])
    
    # Group consecutive frames into segments
    current_segment = None
    
    for frame in frames_with_faces:
        timestamp = frame['timestamp']
        face_count = frame['face_count']
        
        if current_segment is None:
            # Start new segment
            current_segment = {
                'start_time': timestamp,
                'end_time': timestamp,
                'face_count': face_count,
                'max_faces': face_count,
                'frames': [frame],
                'all_faces_data': []
            }
            # Collect all face analysis data
            for face in frame['faces']:
                current_segment['all_faces_data'].append({
                    'timestamp': timestamp,
                    'face_analysis': face
                })
        else:
            # Check if this frame belongs to current segment
            gap = timestamp - current_segment['end_time']
            
            if gap <= max_gap_seconds:
                # Extend current segment
                current_segment['end_time'] = timestamp
                current_segment['max_faces'] = max(current_segment['max_faces'], face_count)
                current_segment['frames'].append(frame)
                
                # Add face analysis data
                for face in frame['faces']:
                    current_segment['all_faces_data'].append({
                        'timestamp': timestamp,
                        'face_analysis': face
                    })
            else:
                # Finish current segment and add to results
                if should_flag_segment(current_segment):
                    flagged_segments.append(create_flagged_segment(current_segment))
                
                # Start new segment
                current_segment = {
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'face_count': face_count,
                    'max_faces': face_count,
                    'frames': [frame],
                    'all_faces_data': []
                }
                # Collect face analysis data
                for face in frame['faces']:
                    current_segment['all_faces_data'].append({
                        'timestamp': timestamp,
                        'face_analysis': face
                    })
    
    # Don't forget the last segment
    if current_segment and should_flag_segment(current_segment):
        flagged_segments.append(create_flagged_segment(current_segment))
    
    return flagged_segments

def should_flag_segment(segment):
    """Determine if a face detection segment should be flagged"""
    max_faces = segment['max_faces']
    duration = segment['end_time'] - segment['start_time']
    frame_count = len(segment['frames'])
    
    # Flag criteria
    if max_faces > 1:  # Multiple faces
        return True
    if duration > 10:  # Long duration with faces
        return True
    if frame_count > 5:  # Many frames with faces
        return True
    
    # Check for potential minors or concerning content
    potential_minors = 0
    concerning_emotions = 0
    
    for face_data in segment['all_faces_data']:
        face = face_data['face_analysis']
        
        # Check age
        if 'age' in face and isinstance(face['age'], dict):
            estimated_age = face['age'].get('estimated_age', 25)
            if estimated_age < 18:
                potential_minors += 1
        
        # Check emotions
        if 'emotion' in face and isinstance(face['emotion'], dict):
            dominant_emotion = max(face['emotion'].keys(), key=lambda k: face['emotion'][k])
            if dominant_emotion in ['fear', 'disgust', 'angry', 'sad']:
                concerning_emotions += 1
    
    if potential_minors > 0 or concerning_emotions > 2:
        return True
    
    return False

def create_flagged_segment(segment):
    """Create a flagged segment with full DeepFace analysis data"""
    max_faces = segment['max_faces']
    duration = segment['end_time'] - segment['start_time']
    
    # Analyze segment for detailed metadata
    gender_counts = Counter()
    age_stats = []
    emotion_counts = Counter()
    race_counts = Counter()
    potential_minors = 0
    
    for face_data in segment['all_faces_data']:
        face = face_data['face_analysis']
        
        # Gender analysis
        if 'gender' in face and isinstance(face['gender'], dict):
            dominant_gender = max(face['gender'].keys(), key=lambda k: face['gender'][k])
            gender_counts[dominant_gender] += 1
        
        # Age analysis
        if 'age' in face and isinstance(face['age'], dict):
            estimated_age = face['age'].get('estimated_age', 25)
            age_stats.append(estimated_age)
            if estimated_age < 18:
                potential_minors += 1
        
        # Emotion analysis
        if 'emotion' in face and isinstance(face['emotion'], dict):
            dominant_emotion = max(face['emotion'].keys(), key=lambda k: face['emotion'][k])
            emotion_counts[dominant_emotion] += 1
        
        # Race analysis
        if 'race' in face and isinstance(face['race'], dict):
            dominant_race = max(face['race'].keys(), key=lambda k: face['race'][k])
            race_counts[dominant_race] += 1
    
    # Determine priority and flag type
    priority = 'low'
    flag_type = 'face_detected'
    description = f"Face detection: {max_faces} faces"
    
    if max_faces > 2:
        priority = 'medium'
        flag_type = 'multiple_faces'
        description = f"Multiple faces detected ({max_faces} faces)"
    
    if potential_minors > 0:
        priority = 'high'
        flag_type = 'potential_minor'
        description = f"Potential minor detected ({potential_minors} faces under 18)"
    
    # Calculate confidence based on detection consistency
    confidence = min(0.9, 0.5 + (len(segment['frames']) * 0.1))
    
    return {
        "start_time": segment['start_time'],
        "end_time": segment['end_time'],
        "task_type": "face_detection",
        "confidence": confidence,
        "flag_type": flag_type,
        "priority": priority,
        "description": description,
        "metadata": {
            "max_faces": max_faces,
            "duration": duration,
            "frame_count": len(segment['frames']),
            "potential_minors": potential_minors,
            "gender_distribution": dict(gender_counts),
            "age_stats": {
                "min_age": min(age_stats) if age_stats else None,
                "max_age": max(age_stats) if age_stats else None,
                "avg_age": sum(age_stats) / len(age_stats) if age_stats else None
            },
            "emotion_distribution": dict(emotion_counts),
            "race_distribution": dict(race_counts),
            "detailed_face_data": segment['all_faces_data']
        }
    }

def parse_timestamp(timestamp_str: str) -> float:
    """Parse timestamp string to seconds"""
    try:
        if ':' in timestamp_str:
            parts = timestamp_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
        return float(timestamp_str)
    except (ValueError, AttributeError):
        return 0.0

@ray.remote
def process_video_chunks_for_face_detection(chunk_paths: list, config=None, 
                                          frame_interval: int = None, save_frames: bool = False,
                                          chunk_duration_sec: int = 60):
    """
    Process video chunks for face age detection with proper result aggregation.
    Fixed to avoid nested Ray remote calls.
    """
    try:
        # Process chunks sequentially using the sync version
        all_flagged_segments = []
        total_faces = 0
        total_frames = 0
        combined_detailed_analysis = {
            "gender_distribution": defaultdict(int),
            "age_distribution": {"minors": 0, "adults": 0, "seniors": 0},
            "emotion_distribution": defaultdict(int),
            "race_distribution": defaultdict(int),
            "total_unique_faces": 0,
            "frames_with_faces": 0
        }
        
        logger.info(f"Processing {len(chunk_paths)} chunks for face detection")
        
        for i, chunk_path in enumerate(chunk_paths):
            chunk_offset = i * chunk_duration_sec
            chunk_name = Path(chunk_path).stem
            unique_output_dir = f"/tmp/chunk_{chunk_name}_{i}"
            
            logger.info(f"Processing chunk {i}: {chunk_path} -> {unique_output_dir}")
            
            # Call sync version directly (no nested Ray remote calls)
            result = process_video_for_face_detection_sync(
                chunk_path, unique_output_dir, frame_interval, save_frames, chunk_offset
            )
            
            if result["success"]:
                total_faces += result["total_faces_detected"]
                total_frames += result["total_processed_frames"]
                
                # Add flagged segments (already have correct timestamps from sync function)
                all_flagged_segments.extend(result.get("flagged_segments", []))
                
                # Aggregate detailed analysis
                if "detailed_analysis_summary" in result:
                    detailed = result["detailed_analysis_summary"]
                    
                    # Aggregate gender
                    for gender, count in detailed.get("gender_distribution", {}).items():
                        combined_detailed_analysis["gender_distribution"][gender] += count
                    
                    # Aggregate age
                    age_dist = detailed.get("age_distribution", {})
                    combined_detailed_analysis["age_distribution"]["minors"] += age_dist.get("minors", 0)
                    combined_detailed_analysis["age_distribution"]["adults"] += age_dist.get("adults", 0)
                    combined_detailed_analysis["age_distribution"]["seniors"] += age_dist.get("seniors", 0)
                    
                    # Aggregate emotion
                    for emotion, count in detailed.get("emotion_distribution", {}).items():
                        combined_detailed_analysis["emotion_distribution"][emotion] += count
                    
                    # Aggregate race
                    for race, count in detailed.get("race_distribution", {}).items():
                        combined_detailed_analysis["race_distribution"][race] += count
                    
                    combined_detailed_analysis["total_unique_faces"] += detailed.get("total_unique_faces", 0)
                    combined_detailed_analysis["frames_with_faces"] += detailed.get("frames_with_faces", 0)
        
        # Sort segments by start time
        all_flagged_segments.sort(key=lambda x: x["start_time"])
        
        # Convert defaultdicts to regular dicts
        combined_detailed_analysis["gender_distribution"] = dict(combined_detailed_analysis["gender_distribution"])
        combined_detailed_analysis["emotion_distribution"] = dict(combined_detailed_analysis["emotion_distribution"])
        combined_detailed_analysis["race_distribution"] = dict(combined_detailed_analysis["race_distribution"])
        
        return {
            "total_faces_detected": total_faces,
            "total_processed_frames": total_frames,
            "flagged_segments": all_flagged_segments,
            "detailed_analysis_summary": combined_detailed_analysis,
            "chunks_processed": len(chunk_paths),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in process_video_chunks_for_face_detection: {e}")
        return {
            "error": str(e),
            "success": False,
            "total_faces_detected": 0,
            "total_processed_frames": 0,
            "flagged_segments": []
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
                frame_interval=None,  # Use config default instead of hardcoded 30
                save_frames=False
            ))
            
            if result["success"]:
                print("✅ Face detection test successful!")
                print(f"   - Total faces detected: {result['total_faces_detected']}")
                print(f"   - Total frames processed: {result['total_processed_frames']}")
                print(f"   - Flagged segments: {len(result.get('flagged_segments', []))}")
                
                if "detailed_analysis_summary" in result:
                    summary = result["detailed_analysis_summary"]
                    print(f"   - Gender distribution: {summary.get('gender_distribution', {})}")
                    print(f"   - Age distribution: {summary.get('age_distribution', {})}")
                    print(f"   - Emotion distribution: {summary.get('emotion_distribution', {})}")
                
                print(f"   - Output directory: {result['output_dir']}")
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