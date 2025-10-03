import ray
import json
import os
from utils.logger import get_logger
import sys
import time
import gc
from pathlib import Path
from datetime import timedelta, datetime
from collections import defaultdict, Counter
import numpy as np

# Add video-age-detection-pipeline/src to path at module import time
_detector_path = str(Path(__file__).parent.parent / "video-age-detection-pipeline" / "src")
if _detector_path not in sys.path:
    sys.path.insert(0, _detector_path)

# Set up logger first
logger = get_logger("face_age_detector_optimized")

class CachedModelManager:
    """Manage cached models for Ray workers to avoid reinitialization."""

    _instance = None
    _models = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_face_age_detector(self):
        """Get cached FaceAgeDetector instance."""
        if 'face_age_detector' not in self._models:
            try:
                import tensorflow as tf
                # Use GPU device context
                with tf.device('/GPU:0'):
                    FaceAgeDetector = import_face_detector()
                    self._models['face_age_detector'] = FaceAgeDetector()
                    logger.info("✅ FaceAgeDetector model cached on GPU")
            except Exception as e:
                logger.warning(f"⚠️ Failed to cache model on GPU, using CPU: {e}")
                FaceAgeDetector = import_face_detector()
                self._models['face_age_detector'] = FaceAgeDetector()

        return self._models['face_age_detector']

    def clear_cache(self):
        """Clear cached models and cleanup memory."""
        self._models.clear()
        cleanup_tensorflow_memory()
        logger.info("🧹 Model cache cleared")

def configure_tf_for_ray_worker():
    """Configure TensorFlow GPU within Ray worker context."""
    import os

    # Get Ray's assigned GPU device
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    try:
        import tensorflow as tf

        # Configure only the GPUs that Ray allocated to this worker
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Ray manages device visibility, so just configure memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Set TensorFlow to use mixed precision for speed
            try:
                tf.config.optimizer.set_jit(True)  # Enable XLA
                logger.info("✅ XLA optimization enabled")
            except Exception as e:
                logger.warning(f"⚠️ XLA optimization failed: {e}")

            # Set thread configuration for better performance
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(4)

            logger.info(f"✅ TensorFlow GPU configured for Ray worker on device(s): {cuda_visible_devices}")
            return True
        else:
            logger.warning("⚠️ No GPUs visible to TensorFlow in Ray worker")
            return False

    except ImportError:
        logger.error("❌ TensorFlow not available")
        return False
    except Exception as e:
        logger.error(f"❌ TensorFlow GPU configuration failed: {e}")
        return False

def cleanup_tensorflow_memory():
    """Clean up TensorFlow and GPU memory."""
    try:
        import tensorflow as tf
        import gc

        # Clear TensorFlow session
        tf.keras.backend.clear_session()

        # Force garbage collection
        gc.collect()

        # Clear GPU cache if PyTorch is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("🧹 GPU memory cleaned up")

    except Exception as e:
        logger.warning(f"⚠️ Memory cleanup warning: {e}")

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
        if path.exists() and ((path / "deepfacedetect.py").exists() or (path / "vit_video_agedetect.py").exists()):
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
        # from deepfacedetect import FaceAgeDetector
        from vit_video_agedetect import ViTVideoAgeDetector as FaceAgeDetector
        logger.info("Successfully imported ViTVideoAgeDetector")
        return FaceAgeDetector
    except ImportError as e:
        logger.error(f"Failed to import FaceAgeDetector: {e}")
        raise

def process_frame_batch_gpu(frame_batch, frame_metadata, detector):
    """Process a batch of frames on GPU for better performance."""
    if not frame_batch:
        return []

    try:
        import tensorflow as tf

        # Process batch on GPU
        with tf.device('/GPU:0'):
            batch_results = []

            # Process frames in the batch
            for frame, metadata in zip(frame_batch, frame_metadata):
                try:
                    # Use the detector's process_frame method if available
                    if hasattr(detector, 'process_frame'):
                        result = detector.process_frame(frame)
                    else:
                        # Fallback to process_video for single frame
                        # Save frame temporarily and process
                        temp_path = f"/tmp/temp_frame_{metadata['frame_number']}.jpg"
                        import cv2
                        cv2.imwrite(temp_path, frame)
                        result = detector.process_video(temp_path)
                        os.remove(temp_path)  # Cleanup

                    batch_results.append({
                        'chunk_path': metadata['chunk_path'],
                        'frame_number': metadata['frame_number'],
                        'result': result
                    })

                except Exception as e:
                    logger.error(f"❌ Frame processing failed: {e}")
                    batch_results.append({
                        'chunk_path': metadata['chunk_path'],
                        'frame_number': metadata['frame_number'],
                        'result': None,
                        'error': str(e)
                    })

            return batch_results

    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        # Fallback to individual processing
        return process_frames_individually(frame_batch, frame_metadata, detector)

def process_frames_individually(frame_batch, frame_metadata, detector):
    """Fallback individual frame processing."""
    results = []
    for frame, metadata in zip(frame_batch, frame_metadata):
        try:
            # Process single frame
            result = detector.process_frame(frame) if hasattr(detector, 'process_frame') else None
            results.append({
                'chunk_path': metadata['chunk_path'],
                'frame_number': metadata['frame_number'],
                'result': result
            })
        except Exception as e:
            logger.error(f"❌ Individual frame processing failed: {e}")
            results.append({
                'chunk_path': metadata['chunk_path'],
                'frame_number': metadata['frame_number'],
                'result': None,
                'error': str(e)
            })
    return results

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

def process_video_for_face_detection_optimized_sync(video_path: str, output_dir: str = "/tmp/face_analysis",
                                                   frame_interval: int = 30, save_frames: bool = False,
                                                   chunk_offset_seconds: float = 0.0, batch_size: int = 8):
    """
    Optimized core face detection processing function with GPU acceleration and batching.
    """
    processing_start_time = time.time()

    try:
        # Configure TensorFlow for this Ray worker
        gpu_start_time = time.time()
        # gpu_configured = configure_tf_for_ray_worker()
        gpu_end_time = time.time()
        print(f"Time taken to configure GPU:{gpu_end_time - gpu_start_time}")
        # if not gpu_configured:
        #     logger.warning("⚠️ GPU configuration failed, will use CPU")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get cached detector instance
        model_load_time = time.time()
        model_manager = CachedModelManager()
        detector = model_manager.get_face_age_detector()
        print(f"Time taken to load model:{time.time() - model_load_time}")

        # Override config if frame_interval is provided
        if frame_interval and hasattr(detector.config, 'FRAME_INTERVAL'):
            detector.config.FRAME_INTERVAL = frame_interval

        # Override save frames setting if provided
        if save_frames is not None and hasattr(detector.config, 'SAVE_FRAMES'):
            detector.config.SAVE_FRAMES = save_frames

        # Process the video with timing
        logger.info(f"🚀 Processing video {video_path} with GPU optimization")
        logger.info(f"   📐 Batch size: {batch_size}")
        logger.info(f"   🎛️  Frame interval: {frame_interval}")
        # logger.info(f"   🎯 GPU configured: {gpu_configured}")

        start_time = time.time()

        # Process with optimized detector
        results = detector.process_video(video_path)

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Total processing time:{processing_time}")
        # Get processing summary
        summary_time = time.time()
        summary = detector.get_processing_summary()

        # Extract detailed analysis from all DeepFace results
        detailed_summary = extract_detailed_analysis_summary(detector.frames_json)

        # Convert frame-based results to time-based segments
        flagged_segments = []
        if hasattr(detector, 'frames_json') and detector.frames_json:
            flagged_segments = convert_frames_to_segments(detector.frames_json, chunk_offset_seconds)
        print(f"Total summary time:{time.time()-summary_time}")
        # Calculate performance metrics
        total_processing_time = time.time() - processing_start_time
        fps = summary.get("processed_frames", 0) / processing_time if processing_time > 0 else 0

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
                "gpu_configured": True,
                "batch_size": batch_size,
                "frame_interval": frame_interval,
                "save_frames": save_frames,
                "chunk_offset_seconds": chunk_offset_seconds,
                "processing_fps": fps,
                "total_processing_time": total_processing_time
            }
        }

        # Save detailed results to file
        output_file = Path(output_dir) / "face_analysis_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(face_result, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ Face analysis completed successfully")
        logger.info(f"   📊 Total faces detected: {summary['total_faces_detected']}")
        logger.info(f"   📈 Processing speed: {fps:.1f} FPS")
        logger.info(f"   ⏱️  Total time: {total_processing_time:.2f} seconds")
        logger.info(f"   🏷️  Flagged segments: {len(flagged_segments)}")
        logger.info(f"   👥 Gender distribution: {detailed_summary['gender_distribution']}")
        logger.info(f"   🎂 Age distribution: {detailed_summary['age_distribution']}")
        logger.info(f"   💾 Results saved to: {output_file}")

        # Return pipeline-compatible result
        return {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "total_faces_detected": summary["total_faces_detected"],
            "total_processed_frames": summary["processed_frames"],
            "frames_with_errors": summary["frames_with_errors"],
            "flagged_segments": flagged_segments,
            "detailed_analysis_summary": detailed_summary,
            "total_processing_time_seconds": total_processing_time,
            "processing_fps": fps,
            "gpu_used": True,
            "success": True
        }

    except Exception as e:
        total_processing_time = time.time() - processing_start_time
        logger.error(f"❌ Error in optimized face detection: {e}")
        return {
            "error": str(e),
            "success": False,
            "total_faces_detected": 0,
            "total_processed_frames": 0,
            "flagged_segments": [],
            "total_processing_time_seconds": total_processing_time,
            "processing_fps": 0,
            "gpu_used": False
        }
    finally:
        # Cleanup memory
        cleanup_tensorflow_memory()

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
        if 'age' in face and isinstance(face['age'], (int, float)):
            estimated_age = face['age']
            if estimated_age < 18:
                potential_minors += 1

        # Check emotions (if available in future)
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
        if 'gender_label' in face:
            gender_counts[face['gender_label']] += 1

        # Age analysis
        if 'age' in face and isinstance(face['age'], (int, float)):
            estimated_age = face['age']
            age_stats.append(estimated_age)
            if estimated_age < 18:
                potential_minors += 1

        # Emotion analysis (if available)
        if 'emotion' in face and isinstance(face['emotion'], dict):
            dominant_emotion = max(face['emotion'].keys(), key=lambda k: face['emotion'][k])
            emotion_counts[dominant_emotion] += 1

        # Race analysis (if available)
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

@ray.remote(num_gpus=0.20, max_calls=1)  # Increased GPU allocation from 0.06 to 0.20
def process_video_chunks_for_face_detection_optimized(chunk_paths: list, config=None,
                                                    frame_interval: int = None, save_frames: bool = False,
                                                    chunk_duration_sec: int = 60, enable_timing=False, timing_id=None,
                                                    batch_size: int = 8):
    """
    Optimized face detection processing with GPU acceleration and batching.
    """
    start_time = time.time() if enable_timing else None
    start_time_iso = datetime.utcnow().isoformat() + "Z" if enable_timing else None

    try:
        logger.info(f"🚀 Starting optimized face detection for {len(chunk_paths)} chunks")
        logger.info(f"   🎯 GPU allocation: 20% (increased from 6%)")
        logger.info(f"   📐 Batch size: {batch_size}")
        logger.info(f"   🎛️  Frame interval: {frame_interval}")

        # Process chunks sequentially using the optimized sync version
        all_flagged_segments = []
        total_faces = 0
        total_frames = 0
        total_processing_time = 0
        combined_detailed_analysis = {
            "gender_distribution": defaultdict(int),
            "age_distribution": {"minors": 0, "adults": 0, "seniors": 0},
            "emotion_distribution": defaultdict(int),
            "race_distribution": defaultdict(int),
            "total_unique_faces": 0,
            "frames_with_faces": 0
        }

        performance_metrics = {
            "chunks_processed": 0,
            "total_fps": 0,
            "gpu_utilization": [],
            "processing_times": []
        }

        for i, chunk_path in enumerate(chunk_paths):
            chunk_start_time = time.time()
            chunk_offset = i * chunk_duration_sec
            chunk_name = Path(chunk_path).stem
            unique_output_dir = f"/tmp/chunk_{chunk_name}_{i}"

            logger.info(f"📹 Processing chunk {i+1}/{len(chunk_paths)}: {chunk_name}")

            # Call optimized sync version
            result = process_video_for_face_detection_optimized_sync(
                chunk_path, unique_output_dir, frame_interval, save_frames,
                chunk_offset, batch_size
            )

            chunk_processing_time = time.time() - chunk_start_time
            performance_metrics["processing_times"].append(chunk_processing_time)

            if result["success"]:
                total_faces += result["total_faces_detected"]
                total_frames += result["total_processed_frames"]
                total_processing_time += result.get("total_processing_time_seconds", 0)
                performance_metrics["chunks_processed"] += 1

                # Track FPS
                if result.get("processing_fps", 0) > 0:
                    performance_metrics["total_fps"] += result["processing_fps"]

                # Add flagged segments (already have correct timestamps)
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

                logger.info(f"   ✅ Chunk {i+1} completed: {result['total_faces_detected']} faces, {result.get('processing_fps', 0):.1f} FPS")
            else:
                logger.error(f"   ❌ Chunk {i+1} failed: {result.get('error', 'Unknown error')}")

        # Sort segments by start time
        all_flagged_segments.sort(key=lambda x: x["start_time"])

        # Convert defaultdicts to regular dicts
        combined_detailed_analysis["gender_distribution"] = dict(combined_detailed_analysis["gender_distribution"])
        combined_detailed_analysis["emotion_distribution"] = dict(combined_detailed_analysis["emotion_distribution"])
        combined_detailed_analysis["race_distribution"] = dict(combined_detailed_analysis["race_distribution"])

        # Calculate performance metrics
        avg_fps = (performance_metrics["total_fps"] / performance_metrics["chunks_processed"]
                  if performance_metrics["chunks_processed"] > 0 else 0)

        # Build the base result dictionary
        result = {
            "total_faces_detected": total_faces,
            "total_processed_frames": total_frames,
            "flagged_segments": all_flagged_segments,
            "detailed_analysis_summary": combined_detailed_analysis,
            "chunks_processed": len(chunk_paths),
            "total_processing_time_seconds": total_processing_time,
            "average_fps": avg_fps,
            "performance_metrics": performance_metrics,
            "optimization_used": "gpu_optimized_v2",
            "success": True
        }

        # Add timing data only if timing is enabled
        if enable_timing and start_time:
            end_time = time.time()
            end_time_iso = datetime.utcnow().isoformat() + "Z"
            execution_time = end_time - start_time

            result["timing"] = {
                "execution_time": execution_time,
                "start_time": start_time_iso,
                "end_time": end_time_iso,
                "timing_id": timing_id,
                "status": "completed"
            }

        # Log final performance summary
        logger.info(f"🏁 Optimized face detection completed!")
        logger.info(f"   📊 Total faces: {total_faces}")
        logger.info(f"   📈 Average FPS: {avg_fps:.1f}")
        logger.info(f"   ⏱️  Total time: {total_processing_time:.2f}s")
        logger.info(f"   🎯 Chunks processed: {performance_metrics['chunks_processed']}/{len(chunk_paths)}")

        return result

    except Exception as e:
        logger.error(f"❌ Error in optimized face detection: {e}")

        # Build error result
        error_result = {
            "error": str(e),
            "success": False,
            "total_faces_detected": 0,
            "total_processed_frames": 0,
            "flagged_segments": [],
            "total_processing_time_seconds": 0,
            "optimization_used": "gpu_optimized_v2_failed"
        }

        # Add timing data for failed case if timing was enabled
        if enable_timing and start_time:
            end_time = time.time()
            end_time_iso = datetime.utcnow().isoformat() + "Z"
            execution_time = end_time - start_time

            error_result["timing"] = {
                "execution_time": execution_time,
                "start_time": start_time_iso,
                "end_time": end_time_iso,
                "timing_id": timing_id,
                "status": "failed"
            }

        return error_result

    finally:
        # Clean up memory
        cleanup_tensorflow_memory()

if __name__ == "__main__":
    """Test with sample video if provided."""
    if len(sys.argv) > 1:
        sample_video = sys.argv[1]
        print(f"🎬 Testing optimized face detection with: {sample_video}")

        if not Path(sample_video).exists():
            print(f"❌ Video file not found: {sample_video}")
            print("Usage: python ray_jobs/face_age_detector_optimized.py [path_to_sample_video.mp4]")
            sys.exit(1)

        try:
            # Test the optimized Ray function
            print("🚀 Testing optimized face detection...")

            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init()

            # Test the optimized function
            result = ray.get(process_video_chunks_for_face_detection_optimized.remote(
                chunk_paths=[sample_video],
                frame_interval=30,
                save_frames=False,
                batch_size=8,
                enable_timing=True
            ))

            if result["success"]:
                print("✅ Optimized face detection test successful!")
                print(f"   - Total faces detected: {result['total_faces_detected']}")
                print(f"   - Total frames processed: {result['total_processed_frames']}")
                print(f"   - Average FPS: {result.get('average_fps', 0):.1f}")
                print(f"   - Processing time: {result.get('total_processing_time_seconds', 0):.2f}s")
                print(f"   - Flagged segments: {len(result.get('flagged_segments', []))}")
                print(f"   - Optimization: {result.get('optimization_used', 'unknown')}")

                if "detailed_analysis_summary" in result:
                    summary = result["detailed_analysis_summary"]
                    print(f"   - Gender distribution: {summary.get('gender_distribution', {})}")
                    print(f"   - Age distribution: {summary.get('age_distribution', {})}")

            else:
                print(f"❌ Optimized face detection test failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Error during optimized test: {e}")

        finally:
            # Shutdown Ray
            if ray.is_initialized():
                ray.shutdown()
    else:
        print("💡 To test with a sample video, run:")
        print("   python ray_jobs/face_age_detector_optimized.py path/to/sample_video.mp4")