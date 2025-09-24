# ray_jobs/nsfw_detector.py
import ray
import cv2
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
import time

logger = logging.getLogger("nsfw_detector")

def download_file(url: str, local_path: str) -> bool:
    """Download a file from URL to local path if it doesn't exist."""
    local_path = Path(local_path)
    
    # Create directory if it doesn't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if local_path.exists():
        logger.info(f"File already exists: {local_path}")
        return True
    
    try:
        logger.info(f"Downloading {url} to {local_path}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        if downloaded_size % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Successfully downloaded: {local_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        # Clean up partial download
        if local_path.exists():
            local_path.unlink()
        return False

def setup_nsfw_model_files(model_dir: str = "nsfw_model") -> tuple[str, str]:
    """
    Download and setup NSFW model files if they don't exist.
    
    Args:
        model_dir: Directory to store model files
    
    Returns:
        Tuple of (model_path, labels_path)
    """
    # Use absolute path to ensure consistency across workers
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(os.getcwd(), model_dir)
    
    model_dir = Path(model_dir)
    model_path = model_dir / "model.onnx"
    labels_path = model_dir / "labels.json"
    
    # URLs for the files
    model_url = "https://huggingface.co/onnx-community/nsfw_image_detection-ONNX/resolve/main/onnx/model.onnx"
    labels_url = "https://huggingface.co/Falconsai/nsfw_image_detection/resolve/main/labels.json"
    
    logger.info(f"Setting up NSFW model files in: {model_dir}")
    
    # Download files if they don't exist
    model_success = download_file(model_url, str(model_path))
    labels_success = download_file(labels_url, str(labels_path))
    
    if not model_success:
        raise RuntimeError(f"Failed to download model from {model_url}")
    if not labels_success:
        raise RuntimeError(f"Failed to download labels from {labels_url}")
    
    # Verify files exist and are not empty
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError(f"Model file is missing or empty: {model_path}")
    if not labels_path.exists() or labels_path.stat().st_size == 0:
        raise RuntimeError(f"Labels file is missing or empty: {labels_path}")
    
    logger.info(f"Model files ready: model={model_path}, labels={labels_path}")
    return str(model_path), str(labels_path)

@ray.remote(num_gpus=0.08)
class NSFWDetectorWorker:
    """Ray actor that loads the NSFW model on a GPU and processes video chunks."""
    
    def __init__(self, model_path: str = None, labels_path: str = None, input_size=(224, 224)):
        """
        Initialize the NSFW detector with model loading on GPU.
        
        Args:
            model_path: Path to ONNX model (if None, will auto-download)
            labels_path: Path to labels JSON (if None, will auto-download)
            input_size: Input size for the model
        """
        self.input_size = input_size
        self.session = None
        self.labels = None
        
        # Always ensure model files are available
        logger.info("Setting up NSFW detector worker...")
        if model_path is None or labels_path is None or not os.path.exists(str(model_path)) or not os.path.exists(str(labels_path)):
            logger.info("Model paths not provided or files missing, auto-downloading...")
            self.model_path, self.labels_path = setup_nsfw_model_files()
        else:
            self.model_path = model_path
            self.labels_path = labels_path
        
        self._load_model_and_labels()
    
    def _load_model_and_labels(self):
        """Load ONNX model on CUDA and labels."""
        # Double-check files exist
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        
        logger.info(f"Loading ONNX model from: {self.model_path}")
        
        # Try CUDA first, fallback to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            logger.warning(f"Failed to load with CUDA, trying CPU only: {e}")
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        
        provider_used = self.session.get_providers()[0]
        logger.info(f"Model loaded using: {provider_used}")
        
        # Load labels
        logger.info(f"Loading labels from: {self.labels_path}")
        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)
        
        logger.info(f"Loaded {len(self.labels)} labels: {list(self.labels.values())}")
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for model inference."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_resized = pil_image.resize(self.input_size, Image.Resampling.BILINEAR)
        image_np = np.array(image_resized, dtype=np.float32) / 255.0
        image_np = np.transpose(image_np, (2, 0, 1))  # [C, H, W]
        input_tensor = np.expand_dims(image_np, axis=0).astype(np.float32)
        return input_tensor
    
    def _predict_frame(self, frame):
        """Run inference on a single frame."""
        input_tensor = self._preprocess_frame(frame)
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        outputs = self.session.run([output_name], {input_name: input_tensor})
        predictions = outputs[0]
        
        predicted_index = np.argmax(predictions)
        predicted_label = self.labels.get(str(predicted_index), f"Unknown")
        confidence = float(np.max(predictions))
        
        return predicted_label, confidence
    
    def _is_nsfw_prediction(self, prediction, confidence, threshold):
        """Determine if prediction is NSFW based on label and confidence."""
        return prediction.lower() == "nsfw" and confidence >= threshold
    
    def process_video_chunk(self, video_path: str, confidence_threshold: float = 0.5, 
                           chunk_offset_seconds: float = 0.0):
        """
        Process a video chunk and detect NSFW content with timestamps.
        
        Args:
            video_path: Path to video chunk
            confidence_threshold: Minimum confidence for NSFW detection
            chunk_offset_seconds: Time offset of this chunk in the original video
        
        Returns:
            Dict with NSFW detections and metadata
        """
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Auto-detect frame interval based on FPS (1 frame per second)
        frame_interval = max(1, int(round(fps)))
        
        logger.info(f"Processing {video_path}: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
        
        nsfw_detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every frame_interval frames (1 per second)
            if frame_count % frame_interval == 0:
                timestamp = (frame_count / fps) + chunk_offset_seconds
                
                try:
                    prediction, confidence = self._predict_frame(frame)
                    
                    if self._is_nsfw_prediction(prediction, confidence, confidence_threshold):
                        detection = {
                            "timestamp": round(timestamp, 2),
                            "frame": frame_count,
                            "prediction": prediction,
                            "confidence": round(confidence, 4),
                            "chunk_file": video_path
                        }
                        nsfw_detections.append(detection)
                        logger.info(f"NSFW detected at {timestamp:.2f}s: {prediction} ({confidence:.4f})")
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        processing_time = time.time() - start_time
        
        return {
            "chunk_file": video_path,
            "chunk_offset_seconds": chunk_offset_seconds,
            "chunk_duration_seconds": duration,
            "processing_time_seconds": round(processing_time, 2),
            "total_nsfw_detections": len(nsfw_detections),
            "nsfw_timestamps": nsfw_detections,
            "success": True
        }

def shutdown(self):
    try:
        # Drop ONNX session and labels
        try:
            self.session = None
        except Exception:
            pass
        self.labels = None
        # GC and CUDA cache
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    finally:
        # Exit the actor process to guarantee teardown
        import ray
        ray.actor.exit_actor()

def create_flagged_segments(nsfw_detections: list, segment_buffer: float = 5.0) -> list:
    """
    Create flagged segments from NSFW detections by grouping nearby detections.
    
    Args:
        nsfw_detections: List of NSFW detection dictionaries with timestamps
        segment_buffer: Buffer time in seconds to merge nearby detections
    
    Returns:
        List of flagged segments with start_time, end_time, and max confidence
    """
    if not nsfw_detections:
        return []
    
    # Sort detections by timestamp
    sorted_detections = sorted(nsfw_detections, key=lambda x: x["timestamp"])
    segments = []
    
    current_start = sorted_detections[0]["timestamp"]
    current_end = sorted_detections[0]["timestamp"]
    max_confidence = sorted_detections[0]["confidence"]
    
    for detection in sorted_detections[1:]:
        timestamp = detection["timestamp"]
        confidence = detection["confidence"]
        
        # If this detection is close to the current segment, extend it
        if timestamp <= current_end + segment_buffer:
            current_end = timestamp
            max_confidence = max(max_confidence, confidence)
        else:
            # Save current segment and start a new one
            segments.append({
                "start_time": max(0, current_start - segment_buffer),
                "end_time": current_end + segment_buffer,
                "confidence": max_confidence,
                "detection_count": 1  # Could be improved to count actual detections
            })
            
            current_start = timestamp
            current_end = timestamp
            max_confidence = confidence
    
    # Add the last segment
    segments.append({
        "start_time": max(0, current_start - segment_buffer),
        "end_time": current_end + segment_buffer,
        "confidence": max_confidence,
        "detection_count": 1
    })
    
    return segments


@ray.remote
def process_video_chunks_for_nsfw(chunk_paths: list, model_path: str = None, labels_path: str = None, 
                                 confidence_threshold: float = 0.5, chunk_duration_sec: int = 60, enable_timing=False, timing_id=None):
    """
    Distribute video chunks across multiple GPU workers for NSFW detection.
    
    This function automatically downloads the required model files if they don't exist.
    
    Args:
        chunk_paths: List of paths to video chunks
        model_path: Path to ONNX model (if None, will auto-download)
        labels_path: Path to labels JSON (if None, will auto-download)
        confidence_threshold: Minimum confidence for NSFW detection
        chunk_duration_sec: Duration of each chunk (for timestamp calculation)
    
    Returns:
        Combined results from all chunks with flagged_segments for compatibility
    """
    logger.info(f"Starting NSFW detection for {len(chunk_paths)} chunks")
    

    start_time = time.time() if enable_timing else None
    start_time_iso = datetime.utcnow().isoformat() + "Z" if enable_timing else None
    try:
        # Always ensure model files are available before creating workers
        logger.info("Ensuring NSFW model files are available...")
        if model_path is None or labels_path is None:
            model_path, labels_path = setup_nsfw_model_files()
        else:
            # Verify provided paths exist, download if not
            if not os.path.exists(model_path) or not os.path.exists(labels_path):
                logger.info("Provided model paths don't exist, downloading...")
                model_path, labels_path = setup_nsfw_model_files()
        
        # Create GPU workers (one per available GPU)
        num_gpus = int(ray.available_resources().get("GPU", 1))
        logger.info(f"Creating {num_gpus} NSFW detector workers with model: {model_path}")
        
        workers = []
        for i in range(num_gpus):
            try:
                worker = NSFWDetectorWorker.remote(model_path, labels_path)
                workers.append(worker)
            except Exception as e:
                logger.error(f"Failed to create worker {i}: {e}")
                if i == 0:  # If we can't create any workers, fail
                    raise RuntimeError(f"Failed to create NSFW detector workers: {e}")
        
        if not workers:
            raise RuntimeError("No NSFW detector workers could be created")
        
        logger.info(f"Successfully created {len(workers)} NSFW detector workers")
        
        # Distribute chunks across workers
        futures = []
        for i, chunk_path in enumerate(chunk_paths):
            worker = workers[i % len(workers)]
            chunk_offset = i * chunk_duration_sec  # Calculate time offset for this chunk
            future = worker.process_video_chunk.remote(chunk_path, confidence_threshold, chunk_offset)
            futures.append(future)
        
        logger.info(f"Distributed {len(chunk_paths)} chunks across {len(workers)} workers")
        
        # Collect results
        try:
            results = ray.get(futures)
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_nsfw_detections": 0,
                "flagged_segments": []
            }
        
        # Combine all NSFW detections
        all_nsfw_detections = []
        total_processing_time = 0
        successful_chunks = 0
        
        for result in results:
            if result["success"]:
                all_nsfw_detections.extend(result["nsfw_timestamps"])
                total_processing_time += result["processing_time_seconds"]
                successful_chunks += 1
            else:
                logger.warning(f"Chunk processing failed: {result.get('error', 'Unknown error')}")
        
        # Sort by timestamp
        all_nsfw_detections.sort(key=lambda x: x["timestamp"])
        
        # Create flagged segments from detections
        flagged_segments = create_flagged_segments(all_nsfw_detections)
        
        logger.info(f"NSFW detection complete: {len(all_nsfw_detections)} detections in {len(flagged_segments)} segments")

        # Build the base result dictionary
        result = {
            "total_nsfw_detections": len(all_nsfw_detections),
            "total_processing_time_seconds": round(total_processing_time, 2),
            "nsfw_timestamps": all_nsfw_detections,
            "flagged_segments": flagged_segments,  # Added for compatibility
            "chunks_processed": len(chunk_paths),
            "successful_chunks": successful_chunks,
            "model_path": model_path,
            "labels_path": labels_path,
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

        return result
        
    except Exception as e:
        logger.error(f"Failed to setup model files or process chunks: {e}")

        # Build error result
        error_result = {
            "success": False,
            "error": str(e),
            "total_nsfw_detections": 0,
            "flagged_segments": [],
            "chunks_processed": 0,
            "successful_chunks": 0
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
        # Proactively shut down all workers to release GPU memory
        try:
            ray.get([w.shutdown.remote() for w in workers])
        except Exception:
            pass
        # Backstop: ensure termination even if shutdown fails
        for w in workers:
            try:
                ray.kill(w)
            except Exception:
                pass


def extract_flagged_segments(nsfw_result: dict, content_type: str, shard_index: int, shard_offset_sec: float) -> list:
    """
    Extract flagged segments from NSFW result for compatibility with existing pipeline.
    
    Args:
        nsfw_result: Result dictionary from NSFW detection
        content_type: Type of content (e.g., "nsfw")
        shard_index: Index of the shard being processed
        shard_offset_sec: Time offset of this shard in the original video
    
    Returns:
        List of flagged segment dictionaries
    """
    segments = []
    
    if not nsfw_result or not nsfw_result.get("success"):
        return segments
    
    flagged_segments = nsfw_result.get("flagged_segments", [])
    
    for segment in flagged_segments:
        segments.append({
            "content_type": content_type,
            "shard_index": shard_index,
            "start_time": segment["start_time"] + shard_offset_sec,
            "end_time": segment["end_time"] + shard_offset_sec,
            "confidence": segment["confidence"],
            "detection_count": segment.get("detection_count", 1),
            "original_start_time": segment["start_time"],  # Relative to shard
            "original_end_time": segment["end_time"]       # Relative to shard
        })
    
    return segments


# Convenience function for easy usage
def detect_nsfw_in_video_chunks(chunk_paths: list, confidence_threshold: float = 0.5, 
                               chunk_duration_sec: int = 60, model_dir: str = "nsfw_model"):
    """
    High-level function to detect NSFW content in video chunks with automatic model setup.
    
    Args:
        chunk_paths: List of paths to video chunks
        confidence_threshold: Minimum confidence for NSFW detection (0.0 to 1.0)
        chunk_duration_sec: Duration of each chunk in seconds
        model_dir: Directory to store/find model files
    
    Returns:
        Dict with NSFW detection results including flagged_segments
    """
    if not chunk_paths:
        raise ValueError("No chunk paths provided")
    
    # Ensure Ray is initialized
    if not ray.is_initialized():
        ray.init()
    
    # Setup model files in the specified directory
    model_path, labels_path = setup_nsfw_model_files(model_dir)
    
    # Process chunks
    future = process_video_chunks_for_nsfw.remote(
        chunk_paths=chunk_paths,
        model_path=model_path,
        labels_path=labels_path,
        confidence_threshold=confidence_threshold,
        chunk_duration_sec=chunk_duration_sec
    )
    
    return ray.get(future)


def clear_gpu_memory():
    """
    Clear GPU memory - placeholder function for compatibility.
    In practice, Ray handles GPU memory management automatically.
    """
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass  # PyTorch not available
    
    logger.info("GPU memory cleanup completed")