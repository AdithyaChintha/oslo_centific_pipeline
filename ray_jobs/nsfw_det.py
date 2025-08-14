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

logger = logging.getLogger("nsfw_detector")

@ray.remote(num_gpus=1)
class NSFWDetectorWorker:
    """Ray actor that loads the NSFW model on a GPU and processes video chunks."""
    
    def __init__(self, model_path: str, labels_path: str, input_size=(224, 224)):
        """Initialize the NSFW detector with model loading on GPU."""
        self.model_path = model_path
        self.labels_path = labels_path
        self.input_size = input_size
        self.session = None
        self.labels = None
        self._load_model_and_labels()
    
    def _load_model_and_labels(self):
        """Load ONNX model on CUDA and labels."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        provider_used = self.session.get_providers()[0]
        logger.info(f"Model loaded using: {provider_used}")
        
        with open(self.labels_path, "r") as f:
            self.labels = json.load(f)
    
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
        frame_interval = int(round(fps))
        
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


@ray.remote
def process_video_chunks_for_nsfw(chunk_paths: list, model_path: str, labels_path: str, 
                                 confidence_threshold: float = 0.5, chunk_duration_sec: int = 60):
    """
    Distribute video chunks across multiple GPU workers for NSFW detection.
    
    Args:
        chunk_paths: List of paths to video chunks
        model_path: Path to ONNX model
        labels_path: Path to labels JSON
        confidence_threshold: Minimum confidence for NSFW detection
        chunk_duration_sec: Duration of each chunk (for timestamp calculation)
    
    Returns:
        Combined results from all chunks
    """
    # Create GPU workers (one per available GPU)
    num_gpus = int(ray.available_resources().get("GPU", 1))
    workers = [NSFWDetectorWorker.remote(model_path, labels_path) for _ in range(num_gpus)]
    
    logger.info(f"Created {num_gpus} NSFW detector workers")
    
    # Distribute chunks across workers
    futures = []
    for i, chunk_path in enumerate(chunk_paths):
        worker = workers[i % num_gpus]
        chunk_offset = i * chunk_duration_sec  # Calculate time offset for this chunk
        future = worker.process_video_chunk.remote(chunk_path, confidence_threshold, chunk_offset)
        futures.append(future)
    
    # Collect results
    results = ray.get(futures)
    
    # Combine all NSFW detections
    all_nsfw_detections = []
    total_processing_time = 0
    
    for result in results:
        if result["success"]:
            all_nsfw_detections.extend(result["nsfw_timestamps"])
            total_processing_time += result["processing_time_seconds"]
    
    # Sort by timestamp
    all_nsfw_detections.sort(key=lambda x: x["timestamp"])
    
    return {
        "total_nsfw_detections": len(all_nsfw_detections),
        "total_processing_time_seconds": round(total_processing_time, 2),
        "nsfw_timestamps": all_nsfw_detections,
        "chunks_processed": len(chunk_paths),
        "success": True
    }