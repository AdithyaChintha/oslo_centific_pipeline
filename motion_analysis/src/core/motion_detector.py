"""
Core motion detection using MOG2 background subtraction.
Adapted from public safety frame_differential.py for 360° home activity analysis.
"""
import cv2
import numpy as np
import time
import math

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[GPU] CuPy available - GPU acceleration enabled")
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False
    print("[GPU] CuPy not available - using CPU fallback")

try:
    from numba import cuda
    NUMBA_GPU_AVAILABLE = True
except ImportError:
    NUMBA_GPU_AVAILABLE = False

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from utils.config import *
from utils.logging_utils import get_logger

logger = get_logger(__name__)

@dataclass
class MotionRegion:
    """Data class for detected motion regions."""
    x: int
    y: int
    width: int
    height: int
    area: int
    confidence: float
    timestamp: float
    frame_index: int
    center_x: int = None
    center_y: int = None
    
    def __post_init__(self):
        """Calculate center coordinates after initialization."""
        self.center_x = self.x + self.width // 2
        self.center_y = self.y + self.height // 2

@dataclass
class MotionMetrics:
    """Data class for motion analysis metrics."""
    frame_index: int
    timestamp: float
    total_motion_pixels: int
    motion_regions_count: int
    largest_motion_area: int
    average_motion_confidence: float
    motion_energy_score: float  # Normalized 0.0-1.0
    activity_level: str  # "LOW", "MEDIUM", "HIGH"

class MotionDetector:
    """
    Core motion detection using MOG2 background subtraction.
    
    This class handles the fundamental motion detection algorithm adapted 
    from the public safety system for home activity analysis.
    """
    
    def __init__(self, 
             history: int = BG_HISTORY,
             var_threshold: float = BG_VAR_THRESHOLD,
             detect_shadows: bool = BG_DETECT_SHADOWS):
        """Initialize the motion detector with GPU support."""
        logger.info("Initializing MotionDetector with MOG2 background subtraction")
        
        # Store configuration
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        
        # GPU Setup
        self.use_gpu = self._setup_gpu_backend()
        
        # Initialize MOG2 background subtractor
        if self.use_gpu:
            try:
                # Try GPU version first
                self.bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=history,
                    varThreshold=var_threshold,
                    detectShadows=detect_shadows
                )
                self.gpu_frame = cv2.cuda_GpuMat()
                self.gpu_mask = cv2.cuda_GpuMat()
                self.gpu_gray = cv2.cuda_GpuMat()
                self.stream = cv2.cuda_Stream()
                print("[GPU] Using CUDA-accelerated background subtraction")
            except:
                print("[GPU] CUDA background subtractor failed, falling back to CPU")
                self.use_gpu = False
        
        if not self.use_gpu:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )
        
        # Morphological kernel for noise reduction
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            MORPH_KERNEL_SIZE
        )
        
        # Initialize state tracking
        self.frame_count = 0
        self.motion_history = []
        self.start_time = time.time()
        
        # Motion region tracking (adapted from frame_differential.py)
        self.tracked_regions = {}
        self.next_region_id = 0
        
        logger.info(f"MOG2 initialized with history={history}, "
                   f"varThreshold={var_threshold}, detectShadows={detect_shadows}")
        
        logger.debug(f"MotionDetector initialized")
        logger.debug(f"Configuration: history={history}, threshold={var_threshold}")
    
    def _setup_gpu_backend(self) -> bool:
        """Setup GPU backend and check availability."""
        if not USE_GPU:
            print("[GPU] GPU disabled in config")
            return False
        
        if not GPU_AVAILABLE:
            print("[GPU] CuPy not available")
            return False
        
        try:
            # Check CUDA devices
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices == 0:
                print("[GPU] No CUDA devices found")
                return False
            
            print(f"[GPU] Found {cuda_devices} CUDA device(s)")
            
            # Test basic GPU operation
            test_array = cp.array([1, 2, 3])
            _ = cp.sum(test_array)
            
            print("[GPU] GPU backend initialized successfully")
            return True
            
        except Exception as e:
            print(f"[GPU] GPU setup failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Tuple[MotionMetrics, List[MotionRegion], np.ndarray]:
        """Process a single frame for motion detection with GPU acceleration."""
        if timestamp is None:
            timestamp = time.time() - self.start_time
        
        # Only print every 100 frames or first/last frame
        if self.frame_count % 100 == 0 or self.frame_count == 0:
            logger.debug(f"Processing frame {self.frame_count} at timestamp {timestamp:.2f}s {'[GPU]' if self.use_gpu else '[CPU]'}")
        
        if self.use_gpu:
            return self._process_frame_gpu(frame, timestamp)
        else:
            return self._process_frame_cpu(frame, timestamp)

    def _process_frame_gpu(self, frame: np.ndarray, timestamp: float) -> Tuple[MotionMetrics, List[MotionRegion], np.ndarray]:
        """GPU-accelerated frame processing."""
        try:
            # Upload frame to GPU
            self.gpu_frame.upload(frame)
            
            # Convert to grayscale on GPU
            cv2.cuda.cvtColor(self.gpu_frame, self.gpu_gray, cv2.COLOR_BGR2GRAY, stream=self.stream)
            
            # Apply background subtraction on GPU
            self.bg_subtractor.apply(self.gpu_gray, self.gpu_mask, stream=self.stream)
            
            # Morphological operations on GPU
            gpu_clean_mask = cv2.cuda_GpuMat()
            cv2.cuda.morphologyEx(self.gpu_mask, gpu_clean_mask, cv2.MORPH_OPEN, self.morph_kernel, stream=self.stream)
            cv2.cuda.morphologyEx(gpu_clean_mask, gpu_clean_mask, cv2.MORPH_CLOSE, self.morph_kernel, stream=self.stream)
            
            # Download result from GPU
            clean_mask = gpu_clean_mask.download()
            
            # Only print every 100 frames
            if self.frame_count % 100 == 0:
                logger.debug(f"GPU processing: {np.sum(clean_mask > 0)} foreground pixels")
            
        except Exception as e:
            logger.debug(f"GPU processing failed: {e}, falling back to CPU")
            return self._process_frame_cpu(frame, timestamp)
        
        # Continue with region extraction
        motion_regions = self._extract_motion_regions(clean_mask, timestamp)
        motion_metrics = self._calculate_motion_metrics(clean_mask, motion_regions, timestamp)
        
        self.frame_count += 1
        self.motion_history.append(motion_metrics.motion_energy_score)
        
        return motion_metrics, motion_regions, clean_mask

    def _process_frame_cpu(self, frame: np.ndarray, timestamp: float) -> Tuple[MotionMetrics, List[MotionRegion], np.ndarray]:
        """CPU fallback processing."""
        # Convert to grayscale for background subtraction
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray_frame)
        
        # Only print every 100 frames
        if self.frame_count % 100 == 0:
            logger.debug(f"Raw foreground pixels: {np.sum(fg_mask > 0)}")
        
        # Clean up the mask using morphological operations
        clean_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        if self.frame_count % 100 == 0:
            logger.debug(f"Clean foreground pixels after morphology: {np.sum(clean_mask > 0)}")
        
        # Extract motion regions
        motion_regions = self._extract_motion_regions(clean_mask, timestamp)
        motion_metrics = self._calculate_motion_metrics(clean_mask, motion_regions, timestamp)
        
        self.frame_count += 1
        self.motion_history.append(motion_metrics.motion_energy_score)
        
        return motion_metrics, motion_regions, clean_mask
        
    def _extract_motion_regions(self, motion_mask: np.ndarray, timestamp: float) -> List[MotionRegion]:
        """Extract motion regions with GPU acceleration where possible."""
        
        if self.use_gpu and GPU_AVAILABLE:
            return self._extract_regions_gpu(motion_mask, timestamp)
        else:
            return self._extract_regions_cpu(motion_mask, timestamp)

    def _extract_regions_gpu(self, motion_mask: np.ndarray, timestamp: float) -> List[MotionRegion]:
        """GPU-accelerated region extraction using CuPy."""
        try:
            # Upload mask to GPU
            gpu_mask = cp.asarray(motion_mask)
            
            # GPU-based connected components
            labeled_array, num_features = self._gpu_connected_components(gpu_mask)
            
            motion_regions = []
            
            # Only print every 100 frames
            if self.frame_count % 100 == 0:
                logger.debug(f"GPU found {num_features} connected components")
            
            # Process each connected component
            for label in range(1, num_features + 1):
                # Get component mask
                component_mask = (labeled_array == label)
                
                # Calculate area on GPU
                area = int(cp.sum(component_mask))
                
                if area > MIN_FOREGROUND_AREA:
                    # Get bounding box coordinates
                    coords = cp.where(component_mask)
                    y_coords, x_coords = coords[0], coords[1]
                    
                    if len(y_coords) > 0:
                        y_min, y_max = int(cp.min(y_coords)), int(cp.max(y_coords))
                        x_min, x_max = int(cp.min(x_coords)), int(cp.max(x_coords))
                        
                        width = x_max - x_min + 1
                        height = y_max - y_min + 1
                        
                        # Calculate confidence on GPU
                        bbox_pixels = width * height
                        confidence = float(area / bbox_pixels) if bbox_pixels > 0 else 0.0
                        
                        if confidence > CONFIDENCE_THRESHOLD:
                            motion_region = MotionRegion(
                                x=x_min, y=y_min, width=width, height=height,
                                area=area,
                                confidence=confidence,
                                timestamp=timestamp,
                                frame_index=self.frame_count
                            )
                            motion_regions.append(motion_region)
            
            return motion_regions
            
        except Exception as e:
            if self.frame_count % 100 == 0:
                logger.debug(f"GPU region extraction failed: {e}, falling back to CPU")
            return self._extract_regions_cpu(motion_mask, timestamp)

    def _gpu_connected_components(self, gpu_mask):
        """GPU-based connected components using CuPy."""
        try:
            # Use CuPy's implementation of connected components
            from cupyx.scipy import ndimage as gpu_ndimage
            
            # Structure for 8-connectivity
            structure = cp.ones((3, 3), dtype=bool)
            
            # Label connected components
            labeled_array, num_features = gpu_ndimage.label(gpu_mask, structure=structure)
            
            return labeled_array, num_features
            
        except ImportError:
            # Fallback if cupyx not available
            raise Exception("cupyx.scipy not available for GPU connected components")

    def _extract_regions_cpu(self, motion_mask: np.ndarray, timestamp: float) -> List[MotionRegion]:
        """CPU fallback for region extraction (original method)."""
        # Find contours of motion regions
        contours, _ = cv2.findContours(
            motion_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        motion_regions = []
        
        # Only print every 100 frames
        if self.frame_count % 100 == 0:
            logger.debug(f"CPU found {len(contours)} contours")
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area > MIN_FOREGROUND_AREA:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate confidence based on motion density
                roi_mask = motion_mask[y:y+h, x:x+w]
                motion_pixels = np.sum(roi_mask > 0)
                bbox_pixels = w * h
                confidence = motion_pixels / bbox_pixels if bbox_pixels > 0 else 0.0
                
                if confidence > CONFIDENCE_THRESHOLD:
                    motion_region = MotionRegion(
                        x=x, y=y, width=w, height=h,
                        area=int(area),
                        confidence=confidence,
                        timestamp=timestamp,
                        frame_index=self.frame_count
                    )
                    motion_regions.append(motion_region)
        
        return motion_regions
    
    def _calculate_motion_metrics(self, motion_mask: np.ndarray, 
                            motion_regions: List[MotionRegion], 
                            timestamp: float) -> MotionMetrics:
        """Calculate motion metrics with full GPU acceleration."""
        
        if self.use_gpu and GPU_AVAILABLE:
            return self._calculate_metrics_gpu(motion_mask, motion_regions, timestamp)
        else:
            return self._calculate_metrics_cpu(motion_mask, motion_regions, timestamp)

    def _calculate_metrics_gpu(self, motion_mask: np.ndarray, 
                            motion_regions: List[MotionRegion], 
                            timestamp: float) -> MotionMetrics:
        """GPU-accelerated motion metrics calculation."""
        try:
            # Upload to GPU
            mask_gpu = cp.asarray(motion_mask)
            
            # GPU calculations
            total_motion_pixels = int(cp.sum(mask_gpu > 0))
            frame_pixels = mask_gpu.shape[0] * mask_gpu.shape[1]
            motion_energy_score = float(total_motion_pixels / frame_pixels)
            
            # Additional GPU statistics for history analysis
            if len(self.motion_history) > 0:
                history_gpu = cp.array(self.motion_history + [motion_energy_score])
                motion_variance = float(cp.var(history_gpu))
                motion_std = float(cp.std(history_gpu))
            else:
                motion_variance = 0.0
                motion_std = 0.0
            
        except Exception as e:
            if self.frame_count % 100 == 0:
                logger.debug(f"GPU metrics calculation failed: {e}, falling back to CPU")
            return self._calculate_metrics_cpu(motion_mask, motion_regions, timestamp)
        
        # Rest of calculations (region processing)
        regions_count = len(motion_regions)
        largest_area = 0
        total_confidence = 0.0
        
        if motion_regions:
            largest_area = max(region.area for region in motion_regions)
            total_confidence = sum(region.confidence for region in motion_regions)
            avg_confidence = total_confidence / len(motion_regions)
        else:
            avg_confidence = 0.0
        
        motion_energy_score = min(motion_energy_score, 1.0)
        
        if motion_energy_score > HIGH_MOTION_THRESHOLD:
            activity_level = "HIGH"
        elif motion_energy_score > MEDIUM_MOTION_THRESHOLD:
            activity_level = "MEDIUM"
        else:
            activity_level = "LOW"
        
        if self.frame_count % 100 == 0:
            print(f"[DEBUG] Motion metrics [GPU] - "
                f"Pixels: {total_motion_pixels}, Score: {motion_energy_score:.4f}, Level: {activity_level}")
        
        return MotionMetrics(
            frame_index=self.frame_count,
            timestamp=timestamp,
            total_motion_pixels=total_motion_pixels,
            motion_regions_count=regions_count,
            largest_motion_area=largest_area,
            average_motion_confidence=avg_confidence,
            motion_energy_score=motion_energy_score,
            activity_level=activity_level
        )

    def _calculate_metrics_cpu(self, motion_mask: np.ndarray, 
                            motion_regions: List[MotionRegion], 
                            timestamp: float) -> MotionMetrics:
        """CPU fallback for metrics calculation."""
        # Original CPU implementation
        total_motion_pixels = int(np.sum(motion_mask > 0))
        frame_pixels = motion_mask.shape[0] * motion_mask.shape[1]
        motion_energy_score = total_motion_pixels / frame_pixels
        
        regions_count = len(motion_regions)
        largest_area = 0
        total_confidence = 0.0
        
        if motion_regions:
            largest_area = max(region.area for region in motion_regions)
            total_confidence = sum(region.confidence for region in motion_regions)
            avg_confidence = total_confidence / len(motion_regions)
        else:
            avg_confidence = 0.0
        
        motion_energy_score = min(motion_energy_score, 1.0)
        
        if motion_energy_score > HIGH_MOTION_THRESHOLD:
            activity_level = "HIGH"
        elif motion_energy_score > MEDIUM_MOTION_THRESHOLD:
            activity_level = "MEDIUM"
        else:
            activity_level = "LOW"
        
        return MotionMetrics(
            frame_index=self.frame_count,
            timestamp=timestamp,
            total_motion_pixels=total_motion_pixels,
            motion_regions_count=regions_count,
            largest_motion_area=largest_area,
            average_motion_confidence=avg_confidence,
            motion_energy_score=motion_energy_score,
            activity_level=activity_level
        )
    
    def get_motion_history(self) -> List[float]:
        """
        Get the motion energy history for the entire video.
        
        Returns:
            List of motion energy scores (0.0-1.0) per frame
        """
        return self.motion_history.copy()
    
    def get_motion_statistics(self) -> Dict:
        """
        Get overall motion statistics for the processed video.
        
        Returns:
            Dictionary with motion statistics
        """
        if not self.motion_history:
            return {
                "total_frames": 0,
                "avg_motion_energy": 0.0,
                "max_motion_energy": 0.0,
                "motion_variance": 0.0,
                "high_activity_frames": 0,
                "medium_activity_frames": 0,
                "low_activity_frames": 0
            }
        
        motion_array = np.array(self.motion_history)
        
        # Calculate activity level counts
        high_activity = np.sum(motion_array > HIGH_MOTION_THRESHOLD)
        medium_activity = np.sum((motion_array > MEDIUM_MOTION_THRESHOLD) & 
                               (motion_array <= HIGH_MOTION_THRESHOLD))
        low_activity = len(motion_array) - high_activity - medium_activity
        
        stats = {
            "total_frames": len(self.motion_history),
            "avg_motion_energy": float(np.mean(motion_array)),
            "max_motion_energy": float(np.max(motion_array)),
            "min_motion_energy": float(np.min(motion_array)),
            "motion_variance": float(np.var(motion_array)),
            "motion_std_dev": float(np.std(motion_array)),
            "high_activity_frames": int(high_activity),
            "medium_activity_frames": int(medium_activity),
            "low_activity_frames": int(low_activity),
            "high_activity_percentage": float(high_activity / len(motion_array) * 100),
            "medium_activity_percentage": float(medium_activity / len(motion_array) * 100),
            "low_activity_percentage": float(low_activity / len(motion_array) * 100)
        }
        
        logger.debug(f"Motion statistics calculated: {stats}")
        return stats
    
    def reset(self):
        """
        Reset the detector state for processing a new video.
        """
        logger.info("Resetting MotionDetector state")
        logger.debug(f"Resetting motion detector state")
        
        # Reinitialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        
        # Clear state
        self.frame_count = 0
        self.motion_history.clear()
        self.tracked_regions.clear()
        self.next_region_id = 0
        self.start_time = time.time()
        
        logger.info("MotionDetector reset completed")
        logger.debug(f"Motion detector reset completed")
    
    def smooth_motion_timeline(self, window_size: int = MOTION_SMOOTH_WINDOW) -> List[float]:
        """
        Apply smoothing to the motion timeline to reduce noise.
        
        Args:
            window_size: Size of the smoothing window
            
        Returns:
            Smoothed motion energy timeline
        """
        if len(self.motion_history) < window_size:
            return self.motion_history.copy()
        
        smoothed = []
        motion_array = np.array(self.motion_history)
        
        # Apply moving average
        for i in range(len(motion_array)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(motion_array), i + window_size // 2 + 1)
            window_avg = np.mean(motion_array[start_idx:end_idx])
            smoothed.append(float(window_avg))
        
        logger.debug(f"Smoothed motion timeline with window size {window_size}")
        return smoothed
    
    def detect_activity_peaks(self, smoothed_timeline: Optional[List[float]] = None) -> List[Tuple[int, float, str]]:
        """
        Detect peaks and valleys in motion timeline for activity segmentation.
        
        Args:
            smoothed_timeline: Pre-smoothed timeline (if None, uses raw history)
            
        Returns:
            List of (frame_index, motion_score, peak_type) tuples
        """
        if smoothed_timeline is None:
            smoothed_timeline = self.smooth_motion_timeline()
        
        if len(smoothed_timeline) < 10:
            return []
        
        peaks = []
        motion_array = np.array(smoothed_timeline)
        
        # Find local maxima and minima
        for i in range(1, len(motion_array) - 1):
            prev_val = motion_array[i - 1]
            curr_val = motion_array[i]
            next_val = motion_array[i + 1]
            
            # Local maximum (peak)
            if curr_val > prev_val and curr_val > next_val and curr_val > MEDIUM_MOTION_THRESHOLD:
                peaks.append((i, curr_val, "PEAK"))
            
            # Local minimum (valley)
            elif curr_val < prev_val and curr_val < next_val and curr_val < MEDIUM_MOTION_THRESHOLD:
                peaks.append((i, curr_val, "VALLEY"))
        
        logger.debug(f"Detected {len(peaks)} activity peaks/valleys")
        return peaks
    
    def visualize_motion(self, frame: np.ndarray, motion_mask: np.ndarray, 
                        motion_regions: List[MotionRegion], 
                        motion_metrics: MotionMetrics) -> np.ndarray:
        """
        Create visualization of detected motion on the original frame.
        
        Args:
            frame: Original BGR frame
            motion_mask: Binary motion mask
            motion_regions: List of detected motion regions
            motion_metrics: Motion metrics for the frame
            
        Returns:
            Annotated frame with motion visualization
        """
        # Create overlay
        annotated_frame = frame.copy()
        
        # Create colored motion overlay
        motion_colored = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
        
        # Blend motion overlay with original frame (30% opacity)
        cv2.addWeighted(annotated_frame, 0.7, motion_colored, 0.3, 0, annotated_frame)
        
        # Draw bounding boxes around motion regions
        for region in motion_regions:
            # Color based on confidence: Red (low) -> Green (high)
            color_intensity = int(region.confidence * 255)
            color = (0, color_intensity, 255 - color_intensity)  # BGR format
            
            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (region.x, region.y),
                (region.x + region.width, region.y + region.height),
                color,
                2
            )
            
            # Add confidence and area labels
            label = f"Motion: {region.confidence:.2f} | Area: {region.area}"
            cv2.putText(
                annotated_frame,
                label,
                (region.x, region.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        # Add frame statistics overlay
        stats_y_start = 30
        stats_lines = [
            f"Frame: {motion_metrics.frame_index} | Time: {motion_metrics.timestamp:.1f}s",
            f"Motion Energy: {motion_metrics.motion_energy_score:.4f} | Level: {motion_metrics.activity_level}",
            f"Regions: {motion_metrics.motion_regions_count} | Total Motion Pixels: {motion_metrics.total_motion_pixels}",
            f"Avg Confidence: {motion_metrics.average_motion_confidence:.3f}"
        ]
        
        for i, line in enumerate(stats_lines):
            cv2.putText(
                annotated_frame,
                line,
                (10, stats_y_start + (i * 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return annotated_frame
    
    def _calculate_bbox_iou(self, box1: Tuple[int, int, int, int], 
                           box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.
        Adapted from frame_differential.py
        
        Args:
            box1: First bounding box (x, y, w, h)
            box2: Second bounding box (x, y, w, h)
            
        Returns:
            IoU value between 0.0 and 1.0
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate coordinates of intersection rectangle
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        # Check if there's no intersection
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        # Calculate intersection and union areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0
    
    def export_motion_data(self, output_path: str):
        """
        Export motion detection data to file.
        
        Args:
            output_path: Path to save motion data
        """
        motion_data = {
            "configuration": {
                "history": self.history,
                "var_threshold": self.var_threshold,
                "detect_shadows": self.detect_shadows,
                "min_foreground_area": MIN_FOREGROUND_AREA,
                "confidence_threshold": CONFIDENCE_THRESHOLD
            },
            "processing_info": {
                "total_frames": self.frame_count,
                "processing_duration": time.time() - self.start_time
            },
            "motion_timeline": self.motion_history,
            "statistics": self.get_motion_statistics()
        }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(motion_data, f, indent=2)
        
        logger.info(f"Motion data exported to: {output_path}")
        logger.debug(f"Motion data exported to: {output_path}")