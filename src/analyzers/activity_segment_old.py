"""
High-level activity segmentation using motion analysis for 360° home activities.
"""
import numpy as np
import time
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import signal
from sklearn.cluster import DBSCAN

from core.motion_detector import MotionDetector, MotionMetrics, MotionRegion
from core.video_processor import Video360Processor
from utils.config import *
from utils.logging_utils import get_logger, ProgressLogger

logger = get_logger(__name__)

@dataclass
class ActivitySegment:
    """Data class for detected activity segments."""
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration: float
    activity_type: str  # "HIGH_ACTIVITY", "MEDIUM_ACTIVITY", "LOW_ACTIVITY"
    avg_motion_energy: float
    max_motion_energy: float
    confidence: float
    description: str

@dataclass
class SegmentationResults:
    """Data class for complete segmentation results."""
    video_path: str
    total_duration: float
    total_segments: int
    segments: List[ActivitySegment]
    motion_timeline: List[float]
    motion_statistics: Dict
    processing_time: float
    high_energy_regions: List[Dict] = None

class ActivitySegmenter:
    """
    High-level activity segmentation orchestrator for 360° home activities.
    
    This class combines motion detection with temporal analysis to identify
    meaningful activity segments in long-form home activity videos.
    """
    
    def __init__(self, motion_detector: Optional[MotionDetector] = None,
                 video_processor: Optional[Video360Processor] = None):
        """
        Initialize the activity segmenter.
        
        Args:
            motion_detector: Motion detector instance (creates new if None)
            video_processor: Video processor instance (creates new if None)
        """
        logger.info("Initializing ActivitySegmenter")
        
        # Initialize components
        self.motion_detector = motion_detector or MotionDetector()
        self.video_processor = video_processor or Video360Processor()
        
        # Initialize state
        self.current_video_info = None
        self.processing_start_time = None
        
        logger.info("ActivitySegmenter initialized successfully")
        print(f"[DEBUG] ActivitySegmenter initialized")
    
    def analyze_video(self, video_path: str, save_annotated: bool = SAVE_ANNOTATED_VIDEO,
                     save_motion_data: bool = SAVE_MOTION_DATA,
                     output_dir: str = "output") -> SegmentationResults:
        """
        Complete analysis of a 360° video for activity segmentation.
        
        Args:
            video_path: Path to the 360° video file
            save_annotated: Whether to save annotated video with motion overlay
            save_motion_data: Whether to save detailed motion analysis data
            output_dir: Directory to save outputs
            
        Returns:
            SegmentationResults object with complete analysis
        """
        logger.info(f"Starting video analysis: {video_path}")
        print(f"[DEBUG] Starting analysis of video: {video_path}")
        
        self.processing_start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video information and validate
        self.current_video_info = self.video_processor.get_video_info(video_path)
        validation = self.video_processor.validate_360_video(video_path)
        
        # Log validation results
        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(warning)
                print(f"[DEBUG] WARNING: {warning}")
        
        if validation["recommendations"]:
            for rec in validation["recommendations"]:
                logger.info(f"RECOMMENDATION: {rec}")
                print(f"[DEBUG] RECOMMENDATION: {rec}")
        
        # Estimate processing time
        time_estimate = self.video_processor.estimate_processing_time(self.current_video_info)
        logger.info(f"Estimated processing time: {time_estimate['estimated_minutes']:.1f} minutes")
        print(f"[DEBUG] Estimated processing time: {time_estimate['estimated_minutes']:.1f} minutes")
        
        # Reset motion detector for new video
        self.motion_detector.reset()
        
        # Process video frame by frame
        motion_timeline, annotated_frames = self._process_video_frames(video_path)
        
        print(f"[DEBUG] Completed frame processing. Timeline length: {len(motion_timeline)}")
        
        # Perform activity segmentation
        segments = self._segment_activities(motion_timeline)

        # Always create results, regardless of segments
        results = SegmentationResults(
            video_path=self.current_video_info["path"],
            duration=self.current_video_info["duration"],
            segments=segments,
            processing_time=processing_time,
            motion_statistics=motion_statistics,
        )

        fps = self.current_video_info["fps"]
        high_energy_regions = self._detect_high_energy_regions(motion_timeline, fps)

        # 🧠 Attach to results for saving (optional):
        results.high_energy_regions = high_energy_regions

        print(f"[DEBUG] Generated {len(segments)} activity segments")
        
        # Calculate processing time
        processing_time = time.time() - self.processing_start_time
        
        # Create segmentation results
        results = SegmentationResults(
            video_path=video_path,
            total_duration=self.current_video_info["duration_seconds"],
            total_segments=len(segments),
            segments=segments,
            motion_timeline=motion_timeline,
            motion_statistics=self.motion_detector.get_motion_statistics(),
            processing_time=processing_time
        )
        
        # Save outputs
        self._save_results(results, output_dir, save_annotated, save_motion_data, annotated_frames)
        
        logger.info(f"Video analysis completed in {processing_time:.1f} seconds")
        print(f"[DEBUG] Analysis completed! Processing time: {processing_time:.1f}s")
        
        return results
    
    def _process_video_frames(self, video_path: str) -> Tuple[List[float], List[np.ndarray]]:
        """
        Process all frames in the video for motion detection.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (motion_timeline, annotated_frames)
        """
        logger.info("Processing video frames for motion detection")
        print(f"[DEBUG] Starting frame-by-frame processing")
        
        motion_timeline = []
        annotated_frames = []
        
        # Set up progress tracking
        total_frames = self.current_video_info["total_frames"]
        progress_logger = ProgressLogger(total_frames, logger, "Motion Detection")
        
        # Process frames using generator
        frame_count = 0
        for frame, frame_index, timestamp in self.video_processor.frame_generator(video_path):
            
            # Process frame for motion
            motion_metrics, motion_regions, motion_mask = self.motion_detector.process_frame(
                frame, timestamp
            )
            
            # Store motion energy score
            motion_timeline.append(motion_metrics.motion_energy_score)
            
            # Create annotated frame if visualization is enabled
            if SAVE_ANNOTATED_VIDEO:
                annotated_frame = self.motion_detector.visualize_motion(
                    frame, motion_mask, motion_regions, motion_metrics
                )
                annotated_frames.append(annotated_frame)
            
            # Update progress
            frame_count += 1
            progress_logger.update(frame_count)
        
        progress_logger.complete()
        
        print(f"[DEBUG] Processed {frame_count} frames total")
        print(f"[DEBUG] Motion timeline length: {len(motion_timeline)}")
        
        return motion_timeline, annotated_frames
    
    def _segment_activities(self, motion_timeline: List[float]) -> List[ActivitySegment]:
        """
        Segment the video into meaningful activity periods based on motion analysis.
        
        Args:
            motion_timeline: Motion energy scores per frame
            
        Returns:
            List of ActivitySegment objects
        """
        logger.info("Performing activity segmentation")
        print(f"[DEBUG] Starting activity segmentation on {len(motion_timeline)} frames")
        
        if len(motion_timeline) < 30:  # Less than 1 second at 30fps
            logger.warning("Motion timeline too short for meaningful segmentation")
            return []
        
        # Get video properties
        fps = self.current_video_info["fps"]
        
        # Smooth the motion timeline
        smoothed_timeline = self.motion_detector.smooth_motion_timeline()
        
        print(f"[DEBUG] Smoothed timeline length: {len(smoothed_timeline)}")
        
        # Find activity boundaries using multiple methods
        segments = []
        
        # Method 1: Threshold-based segmentation
        threshold_segments = self._threshold_based_segmentation(smoothed_timeline, fps)
        segments.extend(threshold_segments)
        
        print(f"[DEBUG] Threshold-based segmentation found {len(threshold_segments)} segments")
        
        # Method 2: Peak-valley analysis
        peak_segments = self._peak_valley_segmentation(smoothed_timeline, fps)
        segments.extend(peak_segments)
        
        print(f"[DEBUG] Peak-valley segmentation found {len(peak_segments)} segments")
        
        # Method 3: Change point detection
        changepoint_segments = self._changepoint_segmentation(smoothed_timeline, fps)
        segments.extend(changepoint_segments)
        
        print(f"[DEBUG] Change point segmentation found {len(changepoint_segments)} segments")
        
        # Merge overlapping segments and filter by duration
        merged_segments = self._merge_and_filter_segments(segments)
        
        print(f"[DEBUG] After merging: {len(merged_segments)} final segments")
        
        return merged_segments
    
    def _threshold_based_segmentation(self, motion_timeline: List[float], fps: float) -> List[ActivitySegment]:
        """
        Create segments based on motion energy thresholds.
        
        Args:
            motion_timeline: Smoothed motion energy timeline
            fps: Video frame rate
            
        Returns:
            List of ActivitySegment objects
        """
        segments = []
        motion_array = np.array(motion_timeline)
        
        # Find periods of sustained high activity
        high_activity_mask = motion_array > HIGH_MOTION_THRESHOLD
        medium_activity_mask = (motion_array > MEDIUM_MOTION_THRESHOLD) & (motion_array <= HIGH_MOTION_THRESHOLD)
        
        # Process high activity segments
        segments.extend(self._extract_segments_from_mask(
            high_activity_mask, motion_timeline, fps, "HIGH_ACTIVITY"
        ))
        
        # Process medium activity segments  
        segments.extend(self._extract_segments_from_mask(
            medium_activity_mask, motion_timeline, fps, "MEDIUM_ACTIVITY"
        ))
        
        return segments
    
    def _extract_segments_from_mask(self, activity_mask: np.ndarray, motion_timeline: List[float],
                                   fps: float, activity_type: str) -> List[ActivitySegment]:
        """
        Extract activity segments from a boolean activity mask.
        
        Args:
            activity_mask: Boolean mask indicating active frames
            motion_timeline: Motion energy timeline
            fps: Video frame rate
            activity_type: Type of activity ("HIGH_ACTIVITY", "MEDIUM_ACTIVITY", etc.)
            
        Returns:
            List of ActivitySegment objects
        """
        segments = []
        
        # Find contiguous regions of True values
        diff = np.diff(np.concatenate(([False], activity_mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start_frame, end_frame in zip(starts, ends):
            duration = (end_frame - start_frame) / fps
            
            # Filter by minimum duration
            if duration >= ACTIVITY_MIN_DURATION:
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                # Calculate segment statistics
                segment_motion = motion_timeline[start_frame:end_frame]
                avg_motion = float(np.mean(segment_motion))
                max_motion = float(np.max(segment_motion))
                
                # Simple confidence based on motion consistency
                motion_std = float(np.std(segment_motion))
                confidence = 1.0 - min(motion_std, 0.5) / 0.5  # Higher consistency = higher confidence
                
                # Generate description
                description = self._generate_segment_description(activity_type, duration, avg_motion)
                
                segment = ActivitySegment(
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    duration=duration,
                    activity_type=activity_type,
                    avg_motion_energy=avg_motion,
                    max_motion_energy=max_motion,
                    confidence=confidence,
                    description=description
                )
                
                segments.append(segment)
                print(f"[DEBUG] Found {activity_type} segment: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s)")
        
        return segments
    
    def _peak_valley_segmentation(self, motion_timeline: List[float], fps: float) -> List[ActivitySegment]:
        """
        Create segments based on motion peaks and valleys.
        
        Args:
            motion_timeline: Motion energy timeline
            fps: Video frame rate
            
        Returns:
            List of ActivitySegment objects
        """
        segments = []
        
        # Find peaks and valleys
        peaks_valleys = self.motion_detector.detect_activity_peaks(motion_timeline)
        
        if len(peaks_valleys) < 2:
            return segments
        
        # Create segments between consecutive peaks/valleys
        for i in range(len(peaks_valleys) - 1):
            start_frame, start_motion, start_type = peaks_valleys[i]
            end_frame, end_motion, end_type = peaks_valleys[i + 1]
            
            duration = (end_frame - start_frame) / fps
            
            if duration >= ACTIVITY_MIN_DURATION:
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                # Determine activity type based on motion levels
                segment_motion = motion_timeline[start_frame:end_frame]
                avg_motion = float(np.mean(segment_motion))
                max_motion = float(np.max(segment_motion))
                
                if avg_motion > HIGH_MOTION_THRESHOLD:
                    activity_type = "PEAK_HIGH_ACTIVITY"
                elif avg_motion > MEDIUM_MOTION_THRESHOLD:
                    activity_type = "PEAK_MEDIUM_ACTIVITY"  
                else:
                    activity_type = "PEAK_LOW_ACTIVITY"
                
                confidence = min(max_motion, 1.0)  # Use max motion as confidence proxy
                description = f"Activity period between motion {start_type.lower()} and {end_type.lower()}"
                
                segment = ActivitySegment(
                    start_time=start_time,
                    end_time=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    duration=duration,
                    activity_type=activity_type,
                    avg_motion_energy=avg_motion,
                    max_motion_energy=max_motion,
                    confidence=confidence,
                    description=description
                )
                
                segments.append(segment)
                print(f"[DEBUG] Peak-valley segment: {start_time:.1f}s-{end_time:.1f}s, type: {activity_type}")
        
        return segments
    
    def _changepoint_segmentation(self, motion_timeline: List[float], fps: float) -> List[ActivitySegment]:
        """
        Detect change points in motion timeline for natural activity boundaries.
        
        Args:
            motion_timeline: Motion energy timeline
            fps: Video frame rate
            
        Returns:
            List of ActivitySegment objects
        """
        segments = []
        
        if len(motion_timeline) < 100:  # Need sufficient data for change point detection
            return segments
        
        motion_array = np.array(motion_timeline)
        
        # Simple change point detection using sliding window variance
        window_size = int(fps * 10)  # 10-second windows
        change_points = []
        
        for i in range(window_size, len(motion_array) - window_size, window_size // 2):
            # Calculate variance in current window vs next window
            current_window = motion_array[i - window_size:i]
            next_window = motion_array[i:i + window_size]
            
            current_mean = np.mean(current_window)
            next_mean = np.mean(next_window)
            
            # Detect significant change in motion level
            if abs(current_mean - next_mean) > 0.1:  # Significant change threshold
                change_points.append(i)
                print(f"[DEBUG] Change point detected at frame {i} (time: {i/fps:.1f}s)")
        
        # Create segments between change points
        if change_points:
            # Add start and end points
            all_points = [0] + change_points + [len(motion_timeline) - 1]
            
            for i in range(len(all_points) - 1):
                start_frame = all_points[i]
                end_frame = all_points[i + 1]
                
                duration = (end_frame - start_frame) / fps
                
                if duration >= ACTIVITY_MIN_DURATION:
                    start_time = start_frame / fps
                    end_time = end_frame / fps
                    
                    segment_motion = motion_timeline[start_frame:end_frame]
                    avg_motion = float(np.mean(segment_motion))
                    max_motion = float(np.max(segment_motion))
                    
                    # Classify activity type
                    if avg_motion > HIGH_MOTION_THRESHOLD:
                        activity_type = "CHANGEPOINT_HIGH_ACTIVITY"
                    elif avg_motion > MEDIUM_MOTION_THRESHOLD:
                        activity_type = "CHANGEPOINT_MEDIUM_ACTIVITY"
                    else:
                        activity_type = "CHANGEPOINT_LOW_ACTIVITY"
                    
                    confidence = 1.0 - np.std(segment_motion)  # More consistent = higher confidence
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                    
                    description = f"Activity segment from change point analysis"
                    
                    segment = ActivitySegment(
                        start_time=start_time,
                        end_time=end_time,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        duration=duration,
                        activity_type=activity_type,
                        avg_motion_energy=avg_motion,
                        max_motion_energy=max_motion,
                        confidence=confidence,
                        description=description
                    )
                    
                    segments.append(segment)
                    print(f"[DEBUG] Changepoint segment: {start_time:.1f}s-{end_time:.1f}s, type: {activity_type}")
        
        return segments
    
    def _merge_and_filter_segments(self, segments: List[ActivitySegment]) -> List[ActivitySegment]:
        """
        Merge overlapping segments and filter by quality criteria.
        
        Args:
            segments: List of raw activity segments
            
        Returns:
            List of merged and filtered segments
        """
        if not segments:
            return []
        
        print(f"[DEBUG] Merging {len(segments)} segments")
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start_time)
        
        merged = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # Check for overlap (with small buffer)
            if next_segment.start_time <= current_segment.end_time + 5.0:  # 5-second buffer
                # Merge segments
                current_segment = self._merge_two_segments(current_segment, next_segment)
                print(f"[DEBUG] Merged overlapping segments")
            else:
                # No overlap, add current and move to next
                if current_segment.duration >= ACTIVITY_MIN_DURATION:
                    merged.append(current_segment)
                current_segment = next_segment
        
        # Add the last segment
        if current_segment.duration >= ACTIVITY_MIN_DURATION:
            merged.append(current_segment)
        
        print(f"[DEBUG] After merging: {len(merged)} segments")
        
        # Sort by duration (longest first) and take top segments
        merged.sort(key=lambda x: x.duration, reverse=True)
        
        # Keep only high-confidence, significant segments
        filtered = [seg for seg in merged if seg.confidence > 0.3 and seg.duration > ACTIVITY_MIN_DURATION]
        
        print(f"[DEBUG] After filtering: {len(filtered)} final segments")
        
        return filtered[:10]  # Return top 10 segments to avoid clutter
    
    def _merge_two_segments(self, seg1: ActivitySegment, seg2: ActivitySegment) -> ActivitySegment:
        """
        Merge two overlapping activity segments.
        
        Args:
            seg1: First segment
            seg2: Second segment
            
        Returns:
            Merged ActivitySegment
        """
        # Calculate merged properties
        start_time = min(seg1.start_time, seg2.start_time)
        end_time = max(seg1.end_time, seg2.end_time)
        start_frame = min(seg1.start_frame, seg2.start_frame)
        end_frame = max(seg1.end_frame, seg2.end_frame)
        duration = end_time - start_time
        
        # Weighted average of motion energies
        weight1 = seg1.duration / (seg1.duration + seg2.duration)
        weight2 = seg2.duration / (seg1.duration + seg2.duration)
        
        avg_motion = seg1.avg_motion_energy * weight1 + seg2.avg_motion_energy * weight2
        max_motion = max(seg1.max_motion_energy, seg2.max_motion_energy)
        confidence = (seg1.confidence + seg2.confidence) / 2
        
        # Determine activity type (use higher energy type)
        if seg1.avg_motion_energy > seg2.avg_motion_energy:
            activity_type = seg1.activity_type
        else:
            activity_type = seg2.activity_type
        
        description = f"Merged activity segment: {seg1.activity_type} + {seg2.activity_type}"
        
        return ActivitySegment(
            start_time=start_time,
            end_time=end_time,
            start_frame=start_frame,
            end_frame=end_frame,
            duration=duration,
            activity_type=activity_type,
            avg_motion_energy=avg_motion,
            max_motion_energy=max_motion,
            confidence=confidence,
            description=description
        )
    
    def _generate_segment_description(self, activity_type: str, duration: float, avg_motion: float) -> str:
        """
        Generate human-readable description for an activity segment.
        
        Args:
            activity_type: Type of activity detected
            duration: Duration of the segment
            avg_motion: Average motion energy
            
        Returns:
            Human-readable description string
        """
        duration_desc = f"{duration:.1f} seconds"
        if duration > 60:
            duration_desc = f"{duration/60:.1f} minutes"
        
        motion_desc = f"motion level {avg_motion:.2f}"
        
        activity_descriptions = {
            "HIGH_ACTIVITY": f"High-intensity activity period ({duration_desc}) - likely active cooking, cleaning, or exercise",
            "MEDIUM_ACTIVITY": f"Moderate activity period ({duration_desc}) - likely food prep, organizing, or light tasks",
            "LOW_ACTIVITY": f"Low activity period ({duration_desc}) - likely resting, reading, or passive tasks",
            "PEAK_HIGH_ACTIVITY": f"Activity peak period ({duration_desc}) - intense activity phase",
            "PEAK_MEDIUM_ACTIVITY": f"Moderate activity peak ({duration_desc}) - sustained moderate tasks",
            "CHANGEPOINT_HIGH_ACTIVITY": f"High-activity phase ({duration_desc}) - detected via activity transition",
            "CHANGEPOINT_MEDIUM_ACTIVITY": f"Medium-activity phase ({duration_desc}) - detected via activity transition"
        }
        
        base_desc = activity_descriptions.get(activity_type, f"Activity period ({duration_desc})")
        return f"{base_desc} | {motion_desc}"
    
    def _save_results(self, results: SegmentationResults, output_dir: str,
                     save_annotated: bool, save_motion_data: bool, 
                     annotated_frames: List[np.ndarray]):
        """
        Save analysis results to files.
        
        Args:
            results: Segmentation results
            output_dir: Output directory
            save_annotated: Whether to save annotated video
            save_motion_data: Whether to save motion data
            annotated_frames: List of annotated frames
        """
        logger.info(f"Saving results to {output_dir}")
        print(f"[DEBUG] Saving results to {output_dir}")
        
        # Create base filename
        video_name = os.path.splitext(os.path.basename(results.video_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{video_name}_{timestamp}"
        
        # Save segmentation results as JSON
        results_path = os.path.join(output_dir, f"{base_name}_segments.json")
        self._save_segments_json(results, results_path)
        
        # Save motion timeline data
        if save_motion_data:
            motion_path = os.path.join(output_dir, f"{base_name}_motion_data.json")
            self.motion_detector.export_motion_data(motion_path)
        
        # Save annotated video
        if save_annotated and annotated_frames:
            video_path = os.path.join(output_dir, f"{base_name}_annotated.mp4")
            fps = self.current_video_info["fps"]
            self.video_processor.save_video_with_overlay(
                results.video_path, annotated_frames, video_path, fps
            )
        
        # Save summary report
        summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        self._save_summary_report(results, summary_path)
        
        logger.info("All results saved successfully")
        print(f"[DEBUG] All results saved to {output_dir}")
    
    def _save_segments_json(self, results: SegmentationResults, output_path: str):
        """Save segmentation results as JSON."""
        results_dict = {
            "video_info": {
                "path": results.video_path,
                "duration": results.total_duration,
                "total_segments": results.total_segments,
                "processing_time": results.processing_time,
                "high_energy_regions": results.high_energy_regions
            },
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "activity_type": seg.activity_type,
                    "avg_motion_energy": seg.avg_motion_energy,
                    "max_motion_energy": seg.max_motion_energy,
                    "confidence": seg.confidence,
                    "description": seg.description
                }
                for seg in results.segments
            ],
            "motion_statistics": results.motion_statistics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"[DEBUG] Segments JSON saved: {output_path}")
    
    def _save_summary_report(self, results: SegmentationResults, output_path: str):
        """Save human-readable summary report."""
        with open(output_path, 'w') as f:
            f.write("=== 360° HOME ACTIVITY ANALYSIS REPORT ===\n\n")
            f.write(f"Video: {os.path.basename(results.video_path)}\n")
            f.write(f"Duration: {results.total_duration/60:.1f} minutes\n")
            f.write(f"Processing Time: {results.processing_time:.1f} seconds\n")
            f.write(f"Total Segments Found: {results.total_segments}\n\n")
            
            f.write("=== MOTION STATISTICS ===\n")
            stats = results.motion_statistics
            f.write(f"Average Motion Energy: {stats['avg_motion_energy']:.4f}\n")
            f.write(f"Peak Motion Energy: {stats['max_motion_energy']:.4f}\n")
            f.write(f"High Activity: {stats['high_activity_percentage']:.1f}% of video\n")
            f.write(f"Medium Activity: {stats['medium_activity_percentage']:.1f}% of video\n")
            f.write(f"Low Activity: {stats['low_activity_percentage']:.1f}% of video\n\n")
            
            f.write("=== ACTIVITY SEGMENTS ===\n")
            for i, segment in enumerate(results.segments, 1):
                f.write(f"\nSegment {i}:\n")
                f.write(f"  Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s ({segment.duration:.1f}s)\n")
                f.write(f"  Type: {segment.activity_type}\n")
                f.write(f"  Motion Energy: {segment.avg_motion_energy:.3f} (max: {segment.max_motion_energy:.3f})\n")
                f.write(f"  Confidence: {segment.confidence:.3f}\n")
                f.write(f"  Description: {segment.description}\n")
        
        print(f"[DEBUG] Summary report saved: {output_path}")
    
    def analyze_video_chunk(self, video_path: str, start_time: float, end_time: float) -> Dict:
        """
        Analyze a specific time chunk of a video.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Dictionary with chunk analysis results
        """
        logger.info(f"Analyzing video chunk: {start_time:.1f}s - {end_time:.1f}s")
        print(f"[DEBUG] Analyzing chunk: {start_time:.1f}s - {end_time:.1f}s")
        
        # Extract chunk to temporary file
        temp_chunk_path = self.video_processor.extract_video_chunk(
            video_path, start_time, end_time
        )
        
        try:
            # Analyze the chunk
            chunk_results = self.analyze_video(
                temp_chunk_path, 
                save_annotated=False, 
                save_motion_data=False,
                output_dir="temp_chunk_output"
            )
            
            # Adjust timestamps to original video timeline
            adjusted_segments = []
            for segment in chunk_results.segments:
                adjusted_segment = ActivitySegment(
                    start_time=segment.start_time + start_time,
                    end_time=segment.end_time + start_time,
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame,
                    duration=segment.duration,
                    activity_type=segment.activity_type,
                    avg_motion_energy=segment.avg_motion_energy,
                    max_motion_energy=segment.max_motion_energy,
                    confidence=segment.confidence,
                    description=f"Chunk analysis: {segment.description}"
                )
                adjusted_segments.append(adjusted_segment)
            
            chunk_analysis = {
                "chunk_start": start_time,
                "chunk_end": end_time,
                "chunk_duration": end_time - start_time,
                "segments_found": len(adjusted_segments),
                "segments": adjusted_segments,
                "motion_statistics": chunk_results.motion_statistics,
                "processing_time": chunk_results.processing_time
            }
            
            print(f"[DEBUG] Chunk analysis completed: {len(adjusted_segments)} segments found")
            return chunk_analysis
            
        finally:
            # Clean up temporary chunk file
            self.video_processor.cleanup_temp_files([temp_chunk_path])
    
    def get_activity_summary(self, results: SegmentationResults) -> Dict:
        """
        Generate a comprehensive activity summary.
        
        Args:
            results: Segmentation results
            
        Returns:
            Dictionary with activity summary
        """
        segments = results.segments
        
        if not segments:
            return {"message": "No significant activities detected"}
        
        # Group segments by activity type
        activity_groups = {}
        for segment in segments:
            activity_type = segment.activity_type
            if activity_type not in activity_groups:
                activity_groups[activity_type] = []
            activity_groups[activity_type].append(segment)
        
        # Calculate summary statistics
        total_active_time = sum(seg.duration for seg in segments)
        video_coverage = (total_active_time / results.total_duration) * 100
        
        # Find most dominant activity
        longest_segment = max(segments, key=lambda x: x.duration)
        highest_energy_segment = max(segments, key=lambda x: x.avg_motion_energy)
        
        summary = {
            "total_segments": len(segments),
            "total_active_time": total_active_time,
            "video_coverage_percentage": video_coverage,
            "activity_breakdown": {
                activity_type: {
                    "count": len(segs),
                    "total_duration": sum(s.duration for s in segs),
                    "avg_motion_energy": np.mean([s.avg_motion_energy for s in segs]),
                    "avg_confidence": np.mean([s.confidence for s in segs])
                }
                for activity_type, segs in activity_groups.items()
            },
            "key_insights": {
                "longest_activity": {
                    "type": longest_segment.activity_type,
                    "duration": longest_segment.duration,
                    "time_range": f"{longest_segment.start_time:.1f}s - {longest_segment.end_time:.1f}s"
                },
                "most_intense_activity": {
                    "type": highest_energy_segment.activity_type,
                    "motion_energy": highest_energy_segment.avg_motion_energy,
                    "time_range": f"{highest_energy_segment.start_time:.1f}s - {highest_energy_segment.end_time:.1f}s"
                }
            },
            "recommendations": self._generate_recommendations(results)
        }
        
        print(f"[DEBUG] Activity summary generated: {summary['total_segments']} segments, "
              f"{video_coverage:.1f}% coverage")
        
        return summary
    
    def _generate_recommendations(self, results: SegmentationResults) -> List[str]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            results: Segmentation results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        stats = results.motion_statistics
        
        # Recommendations based on activity patterns
        if stats["high_activity_percentage"] > 50:
            recommendations.append("High activity video - good for studying active behaviors like cooking or cleaning")
        elif stats["low_activity_percentage"] > 70:
            recommendations.append("Low activity video - consider reviewing for sedentary activities or resting periods")
        
        if len(results.segments) < 3:
            recommendations.append("Few activity segments detected - consider adjusting motion sensitivity thresholds")
        elif len(results.segments) > 20:
            recommendations.append("Many segments detected - consider merging similar adjacent activities")
        
        if stats["motion_variance"] < 0.01:
            recommendations.append("Low motion variance - video may have consistent activity throughout")
        elif stats["motion_variance"] > 0.1:
            recommendations.append("High motion variance - video contains diverse activity patterns")
        
        return recommendations
    
    def _detect_high_energy_regions(self, motion_timeline: List[float], fps: float,
                                  threshold: float = 0.6, min_length_sec: float = 1.0):
        """
        Identify time regions where motion energy exceeds a given threshold.

        Args:
            motion_timeline: List of motion energy per frame
            fps: Frames per second
            threshold: Energy threshold to detect high motion
            min_length_sec: Minimum duration (in seconds) to consider a region

        Returns:
            List of dicts with start_time, end_time, duration_sec
        """
        regions = []
        min_frames = int(min_length_sec * fps)
        start = None

        for i, energy in enumerate(motion_timeline):
            if energy >= threshold:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start >= min_frames:
                        regions.append({
                            "start_time": round(start / fps, 2),
                            "end_time": round((i - 1) / fps, 2),
                            "duration_sec": round((i - start) / fps, 2)
                        })
                    start = None

        # Check if video ends with high energy region
        if start is not None and len(motion_timeline) - start >= min_frames:
            regions.append({
                "start_time": round(start / fps, 2),
                "end_time": round((len(motion_timeline) - 1) / fps, 2),
                "duration_sec": round((len(motion_timeline) - start) / fps, 2)
            })

        return regions
