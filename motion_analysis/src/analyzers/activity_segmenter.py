"""
High-level activity segmentation using motion analysis for 360° home activities.
PART 1: Core ActivitySegmenter Class with Basic Functionality

This file contains the main ActivitySegmenter class with basic threshold-based
segmentation. For advanced methods (peak detection, change point analysis), 
use Part 2: activity_segmenter_advanced.py
"""
import numpy as np
import time
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

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
    duration: float
    segments: int
    segments_list: List[ActivitySegment]
    motion_timeline: List[float]
    motion_statistics: Dict
    processing_time: float
    high_energy_regions: List[Dict] = None

class ActivitySegmenter:
    """
    High-level activity segmentation orchestrator for 360° home activities.
    
    This class combines motion detection with temporal analysis to identify
    meaningful activity segments in long-form home activity videos.
    
    PART 1: Contains core functionality and basic threshold-based segmentation.
    For advanced methods, see activity_segmenter_advanced.py (Part 2).
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
        logger.debug(f"ActivitySegmenter.__init__() starting")
        
        # Initialize components
        self.motion_detector = motion_detector or MotionDetector()
        self.video_processor = video_processor or Video360Processor()
        
        logger.debug(f"Motion detector initialized: {type(self.motion_detector).__name__}")
        logger.debug(f"Video processor initialized: {type(self.video_processor).__name__}")
        
        # Initialize state
        self.current_video_info = None
        self.processing_start_time = None
        
        logger.info("ActivitySegmenter initialized successfully")
        logger.debug(f"ActivitySegmenter initialization completed")
    
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
        print(f"[DEBUG] ==" * 10)
        logger.debug(f"analyze_video() starting")
        logger.debug(f"Video path: {video_path}")
        logger.debug(f"Save annotated: {save_annotated}")
        logger.debug(f"Save motion data: {save_motion_data}")
        logger.debug(f"Output directory: {output_dir}")
        print(f"[DEBUG] ==" * 10)
        
        # Start timing
        self.processing_start_time = time.time()
        logger.debug(f"Processing start time recorded: {self.processing_start_time}")
        
        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Output directory created/verified: {output_dir}")
        except Exception as e:
            logger.debug(f"ERROR creating output directory: {e}")
            raise
        
        # Get video information and validate
        logger.debug(f"Getting video information...")
        try:
            self.current_video_info = self.video_processor.get_video_info(video_path)
            logger.debug(f"Video info retrieved successfully:")
            logger.debug(f"  Duration: {self.current_video_info.get('duration_seconds', 'unknown')} seconds")
            logger.debug(f"  Resolution: {self.current_video_info.get('width', 'unknown')}x{self.current_video_info.get('height', 'unknown')}")
            logger.debug(f"  FPS: {self.current_video_info.get('fps', 'unknown')}")
            logger.debug(f"  Total frames: {self.current_video_info.get('total_frames', 'unknown')}")
            logger.debug(f"  File size: {self.current_video_info.get('file_size_mb', 'unknown')} MB")
            logger.debug(f"  360° likely: {self.current_video_info.get('is_360_likely', 'unknown')}")
        except Exception as e:
            logger.debug(f"ERROR getting video info: {e}")
            raise
        
        # Validate 360° video
        logger.debug(f"Validating 360° video format...")
        try:
            validation = self.video_processor.validate_360_video(video_path)
            logger.debug(f"Validation results:")
            logger.debug(f"  Is valid: {validation.get('is_valid', 'unknown')}")
            
            # Log validation results
            if validation.get("warnings"):
                logger.debug(f"Validation warnings found: {len(validation['warnings'])}")
                for i, warning in enumerate(validation["warnings"]):
                    logger.warning(warning)
                    logger.debug(f"  Warning {i+1}: {warning}")
            
            if validation.get("recommendations"):
                logger.debug(f"Validation recommendations: {len(validation['recommendations'])}")
                for i, rec in enumerate(validation["recommendations"]):
                    logger.info(f"RECOMMENDATION: {rec}")
                    logger.debug(f"  Recommendation {i+1}: {rec}")
        except Exception as e:
            logger.debug(f"ERROR during validation: {e}")
            # Continue processing even if validation fails
        
        # Estimate processing time
        logger.debug(f"Estimating processing time...")
        try:
            time_estimate = self.video_processor.estimate_processing_time(self.current_video_info)
            estimated_minutes = time_estimate.get('estimated_minutes', 'unknown')
            logger.info(f"Estimated processing time: {estimated_minutes} minutes")
            logger.debug(f"Estimated processing time: {estimated_minutes} minutes")
            logger.debug(f"Processing factor: {time_estimate.get('processing_factor', 'unknown')}")
            
            if isinstance(estimated_minutes, (int, float)) and estimated_minutes > 30:
                logger.debug(f"WARNING: Long processing time expected!")
        except Exception as e:
            logger.debug(f"ERROR estimating processing time: {e}")
            # Continue processing
        
        # Reset motion detector for new video
        logger.debug(f"Resetting motion detector...")
        try:
            self.motion_detector.reset()
            logger.debug(f"Motion detector reset completed")
        except Exception as e:
            logger.debug(f"ERROR resetting motion detector: {e}")
            raise
        
        # Process video frame by frame
        logger.debug(f"Starting frame-by-frame processing...")
        try:
            motion_timeline, annotated_frames = self._process_video_frames(video_path, save_annotated)
            logger.debug(f"Frame processing completed successfully")
            logger.debug(f"Motion timeline length: {len(motion_timeline)}")
            logger.debug(f"Annotated frames count: {len(annotated_frames)}")
            
            if len(motion_timeline) == 0:
                logger.debug(f"WARNING: Empty motion timeline!")
                
        except Exception as e:
            logger.debug(f"ERROR during frame processing: {e}")
            raise
        
        # Perform activity segmentation
        logger.debug(f"Starting activity segmentation...")
        try:
            segments = self._segment_activities(motion_timeline)
            logger.debug(f"Activity segmentation completed")
            logger.debug(f"Generated {len(segments)} activity segments")
            
            # Debug: Print segment summary
            for i, segment in enumerate(segments[:5]):  # Show first 5 segments
                print(f"[DEBUG] Segment {i+1}: {segment.start_time:.1f}s-{segment.end_time:.1f}s, "
                      f"type: {segment.activity_type}, confidence: {segment.confidence:.3f}")
                
        except Exception as e:
            logger.debug(f"ERROR during segmentation: {e}")
            # Set empty segments to continue processing
            segments = []
            logger.debug(f"Set empty segments list due to error")
        
        # Calculate processing time
        processing_time = time.time() - self.processing_start_time
        logger.debug(f"Total processing time calculated: {processing_time:.2f} seconds")
        
        # Get motion statistics
        logger.debug(f"Getting motion statistics...")
        try:
            motion_stats = self.motion_detector.get_motion_statistics()
            logger.debug(f"Motion statistics retrieved:")
            logger.debug(f"  Total frames: {motion_stats.get('total_frames', 'unknown')}")
            logger.debug(f"  Avg motion energy: {motion_stats.get('avg_motion_energy', 'unknown'):.4f}")
            logger.debug(f"  Max motion energy: {motion_stats.get('max_motion_energy', 'unknown'):.4f}")
            logger.debug(f"  High activity %: {motion_stats.get('high_activity_percentage', 'unknown'):.1f}%")
            logger.debug(f"  Medium activity %: {motion_stats.get('medium_activity_percentage', 'unknown'):.1f}%")
            logger.debug(f"  Low activity %: {motion_stats.get('low_activity_percentage', 'unknown'):.1f}%")
        except Exception as e:
            logger.debug(f"ERROR getting motion statistics: {e}")
            motion_stats = {}  # Set empty dict to continue
        
        # Detect high energy regions
        logger.debug(f"Detecting high energy regions...")
        try:
            fps = self.current_video_info["fps"]
            high_energy_regions = self._detect_high_energy_regions(motion_timeline, fps)
            logger.debug(f"Detected {len(high_energy_regions)} high energy regions")
            
            # Debug: Show high energy regions
            for i, region in enumerate(high_energy_regions[:3]):  # Show first 3
                print(f"[DEBUG] High energy region {i+1}: {region.get('start_time', 'unknown')}s-{region.get('end_time', 'unknown')}s "
                      f"({region.get('duration_sec', 'unknown')}s)")
                
        except Exception as e:
            logger.debug(f"ERROR detecting high energy regions: {e}")
            high_energy_regions = []  # Set empty list to continue
        
        # Create segmentation results (FIXED VERSION)
        logger.debug(f"Creating SegmentationResults object...")
        try:
            results = SegmentationResults(
                video_path=video_path,  # ✅ Use parameter directly (was: self.current_video_info["path"])
                duration=self.current_video_info["duration_seconds"],  # ✅ Correct key
                segments=len(segments),
                segments_list=segments,
                motion_timeline=motion_timeline,
                motion_statistics=motion_stats,  # ✅ Now properly calculated
                processing_time=processing_time,  # ✅ Now properly calculated
                high_energy_regions=high_energy_regions  # ✅ Add this field
            )
            logger.debug(f"SegmentationResults object created successfully")
            logger.debug(f"Results summary:")
            logger.debug(f"  Video path: {results.video_path}")
            logger.debug(f"  Total duration: {results.duration:.1f} seconds")
            logger.debug(f"  Total segments: {results.segments}")
            logger.debug(f"  Processing time: {results.processing_time:.1f} seconds")
            logger.debug(f"  High energy regions: {len(results.high_energy_regions)}")
            
        except Exception as e:
            logger.debug(f"ERROR creating SegmentationResults: {e}")
            raise
        
        # Save outputs
        logger.debug(f"Saving results...")
        try:
            self._save_results(results, output_dir, save_annotated, save_motion_data, annotated_frames)
            logger.debug(f"Results saved successfully")
        except Exception as e:
            logger.debug(f"ERROR saving results: {e}")
            # Continue even if saving fails
        
        # Final summary
        logger.info(f"Video analysis completed in {processing_time:.1f} seconds")
        print(f"[DEBUG] ==" *10)
        logger.debug(f"VIDEO ANALYSIS COMPLETED!")
        logger.debug(f"Total processing time: {processing_time:.1f} seconds")
        logger.debug(f"Segments generated: {len(segments)}")
        logger.debug(f"Motion timeline length: {len(motion_timeline)}")
        logger.debug(f"Results saved to: {output_dir}")
        print(f"[DEBUG] ==" * 10)
        
        return results
    
    def _process_video_frames(self, video_path: str, save_annotated: bool = False) -> Tuple[List[float], List[np.ndarray]]:
        """
        Process all frames in the video for motion detection.
        
        Args:
            video_path: Path to the video file
            save_annotated: Whether to create annotated frames
            
        Returns:
            Tuple of (motion_timeline, annotated_frames)
        """
        logger.info("Processing video frames for motion detection")
        logger.debug(f"_process_video_frames() starting")
        logger.debug(f"Video path: {video_path}")
        logger.debug(f"Save annotated: {save_annotated}")
        
        motion_timeline = []
        annotated_frames = []
        
        # Set up progress tracking
        total_frames = self.current_video_info["total_frames"]
        logger.debug(f"Total frames to process: {total_frames}")
        
        progress_logger = ProgressLogger(total_frames, logger, "Motion Detection")
        logger.debug(f"Progress logger initialized")
        
        # Process frames using generator
        frame_count = 0
        last_progress_report = 0
        
        logger.debug(f"Starting frame generator loop...")
        
        try:
            for frame, frame_index, timestamp in self.video_processor.frame_generator(video_path):
                
                # Debug every 100 frames
                if frame_count % 100 == 0:
                    print(f"[DEBUG] Processing frame {frame_count}/{total_frames} "
                          f"(index: {frame_index}, timestamp: {timestamp:.2f}s)")
                
                # Process frame for motion
                try:
                    motion_metrics, motion_regions, motion_mask = self.motion_detector.process_frame(
                        frame, timestamp
                    )
                    
                    # Store motion energy score
                    motion_timeline.append(motion_metrics.motion_energy_score)
                    
                    # Debug motion metrics every 100 frames
                    if frame_count % 100 == 0:
                        logger.debug(f"Frame {frame_count} motion metrics:")
                        logger.debug(f"  Energy score: {motion_metrics.motion_energy_score:.4f}")
                        logger.debug(f"  Activity level: {motion_metrics.activity_level}")
                        logger.debug(f"  Motion regions: {motion_metrics.motion_regions_count}")
                        logger.debug(f"  Total motion pixels: {motion_metrics.total_motion_pixels}")
                    
                except Exception as e:
                    logger.debug(f"ERROR processing frame {frame_count}: {e}")
                    # Add zero motion for failed frame
                    motion_timeline.append(0.0)
                
                # Create annotated frame if visualization is enabled
                if save_annotated:
                    try:
                        annotated_frame = self.motion_detector.visualize_motion(
                            frame, motion_mask, motion_regions, motion_metrics
                        )
                        annotated_frames.append(annotated_frame)
                        
                        if frame_count % 100 == 0:
                            logger.debug(f"Annotated frame created for frame {frame_count}")
                            
                    except Exception as e:
                        logger.debug(f"ERROR creating annotated frame {frame_count}: {e}")
                        # Add original frame if annotation fails
                        annotated_frames.append(frame)
                
                # Update progress
                frame_count += 1
                if frame_count % PROGRESS_UPDATE_INTERVAL == 0:
                    progress_logger.update(frame_count)
                    logger.debug(f"Progress update: {frame_count}/{total_frames} frames processed")
                
        except Exception as e:
            logger.debug(f"ERROR in frame generator loop: {e}")
            logger.debug(f"Processed {frame_count} frames before error")
        
        progress_logger.complete()
        
        logger.debug(f"Frame processing completed:")
        logger.debug(f"  Total frames processed: {frame_count}")
        logger.debug(f"  Motion timeline length: {len(motion_timeline)}")
        logger.debug(f"  Annotated frames created: {len(annotated_frames)}")
        logger.debug(f"  Expected frames: {total_frames}")
        
        if frame_count != total_frames:
            logger.debug(f"WARNING: Frame count mismatch! Expected {total_frames}, got {frame_count}")
        
        return motion_timeline, annotated_frames
    
    def _segment_activities(self, motion_timeline: List[float]) -> List[ActivitySegment]:
        """
        Segment the video into meaningful activity periods based on motion analysis.
        
        This method uses basic threshold-based segmentation. For advanced methods
        (peak detection, change point analysis), see activity_segmenter_advanced.py
        
        Args:
            motion_timeline: Motion energy scores per frame
            
        Returns:
            List of ActivitySegment objects
        """
        logger.info("Performing activity segmentation")
        logger.debug(f"_segment_activities() starting")
        logger.debug(f"Motion timeline length: {len(motion_timeline)}")
        
        if len(motion_timeline) < 30:  # Less than 1 second at 30fps
            logger.warning("Motion timeline too short for meaningful segmentation")
            logger.debug(f"WARNING: Motion timeline too short ({len(motion_timeline)} frames)")
            return []
        
        # Get video properties
        fps = self.current_video_info["fps"]
        logger.debug(f"Video FPS: {fps}")
        
        # Smooth the motion timeline
        logger.debug(f"Smoothing motion timeline...")
        try:
            smoothed_timeline = self.motion_detector.smooth_motion_timeline()
            logger.debug(f"Motion timeline smoothed successfully")
            logger.debug(f"Smoothed timeline length: {len(smoothed_timeline)}")
            
            # Calculate smoothed timeline statistics
            if smoothed_timeline:
                smoothed_array = np.array(smoothed_timeline)
                logger.debug(f"Smoothed timeline stats:")
                logger.debug(f"  Min: {np.min(smoothed_array):.4f}")
                logger.debug(f"  Max: {np.max(smoothed_array):.4f}")
                logger.debug(f"  Mean: {np.mean(smoothed_array):.4f}")
                logger.debug(f"  Std: {np.std(smoothed_array):.4f}")
            
        except Exception as e:
            logger.debug(f"ERROR smoothing timeline: {e}")
            # Use original timeline if smoothing fails
            smoothed_timeline = motion_timeline
            logger.debug(f"Using original timeline due to smoothing error")
        
        # Basic threshold-based segmentation
        logger.debug(f"Starting threshold-based segmentation...")
        try:
            segments = self._threshold_based_segmentation(smoothed_timeline, fps)
            logger.debug(f"Threshold-based segmentation completed")
            logger.debug(f"Generated {len(segments)} segments")
            
        except Exception as e:
            logger.debug(f"ERROR in threshold-based segmentation: {e}")
            segments = []
        
        # Filter and sort segments
        logger.debug(f"Filtering and sorting segments...")
        try:
            if segments:
                # Filter by minimum duration and confidence
                filtered_segments = [
                    seg for seg in segments 
                    if seg.duration >= ACTIVITY_MIN_DURATION and seg.confidence > 0.1
                ]
                logger.debug(f"After filtering: {len(filtered_segments)} segments")
                
                # Sort by duration (longest first)
                filtered_segments.sort(key=lambda x: x.duration, reverse=True)
                
                # Limit to top 10 segments to avoid clutter
                final_segments = filtered_segments[:10]
                logger.debug(f"Final segments (top 10): {len(final_segments)}")
                
                # Debug: Show final segments
                for i, segment in enumerate(final_segments):
                    print(f"[DEBUG] Final segment {i+1}: {segment.start_time:.1f}s-{segment.end_time:.1f}s, "
                          f"duration: {segment.duration:.1f}s, type: {segment.activity_type}, "
                          f"confidence: {segment.confidence:.3f}")
                
                segments = final_segments
            else:
                logger.debug(f"No segments to filter")
                
        except Exception as e:
            logger.debug(f"ERROR filtering segments: {e}")
            # Keep original segments if filtering fails
        
        # Resolve overlapping segments
        logger.debug(f"Resolving overlapping segments...")
        try:
            segments = self._resolve_overlapping_segments(segments)
            logger.debug(f"Overlap resolution completed")
        except Exception as e:
            logger.debug(f"ERROR in overlap resolution: {e}")
            # Continue with original segments if resolution fails

        logger.debug(f"Activity segmentation completed with {len(segments)} final segments")
        return segments
    
    def _threshold_based_segmentation(self, motion_timeline: List[float], fps: float) -> List[ActivitySegment]:
        """
        Create segments based on motion energy thresholds.
        
        Args:
            motion_timeline: Smoothed motion energy timeline
            fps: Video frame rate
            
        Returns:
            List of ActivitySegment objects
        """
        logger.debug(f"_threshold_based_segmentation() starting")
        logger.debug(f"Timeline length: {len(motion_timeline)}, FPS: {fps}")
        logger.debug(f"Thresholds - High: {HIGH_MOTION_THRESHOLD}, Medium: {MEDIUM_MOTION_THRESHOLD}")
        logger.debug(f"Minimum duration: {ACTIVITY_MIN_DURATION} seconds")
        
        segments = []
        motion_array = np.array(motion_timeline)
        
        # Calculate threshold statistics
        high_activity_count = np.sum(motion_array > HIGH_MOTION_THRESHOLD)
        medium_activity_count = np.sum((motion_array > MEDIUM_MOTION_THRESHOLD) & (motion_array <= HIGH_MOTION_THRESHOLD))
        low_activity_count = len(motion_array) - high_activity_count - medium_activity_count
        
        logger.debug(f"Motion threshold analysis:")
        logger.debug(f"  High activity frames: {high_activity_count} ({high_activity_count/len(motion_array)*100:.1f}%)")
        logger.debug(f"  Medium activity frames: {medium_activity_count} ({medium_activity_count/len(motion_array)*100:.1f}%)")
        logger.debug(f"  Low activity frames: {low_activity_count} ({low_activity_count/len(motion_array)*100:.1f}%)")
        
        # Find periods of sustained high activity
        high_activity_mask = motion_array > HIGH_MOTION_THRESHOLD
        medium_activity_mask = (motion_array > MEDIUM_MOTION_THRESHOLD) & (motion_array <= HIGH_MOTION_THRESHOLD)
        
        logger.debug(f"Processing high activity segments...")
        high_segments = self._extract_segments_from_mask(
            high_activity_mask, motion_timeline, fps, "HIGH_ACTIVITY"
        )
        segments.extend(high_segments)
        logger.debug(f"Found {len(high_segments)} high activity segments")
        
        logger.debug(f"Processing medium activity segments...")
        medium_segments = self._extract_segments_from_mask(
            medium_activity_mask, motion_timeline, fps, "MEDIUM_ACTIVITY"
        )
        segments.extend(medium_segments)
        logger.debug(f"Found {len(medium_segments)} medium activity segments")
        
        logger.debug(f"Threshold-based segmentation completed with {len(segments)} total segments")
        return segments
    
    def _extract_segments_from_mask(self, activity_mask: np.ndarray, motion_timeline: List[float],
                               fps: float, activity_type: str) -> List[ActivitySegment]:
        """
        Extract activity segments from a boolean activity mask with gap tolerance.
        
        Args:
            activity_mask: Boolean mask indicating active frames
            motion_timeline: Motion energy timeline
            fps: Video frame rate
            activity_type: Type of activity ("HIGH_ACTIVITY", "MEDIUM_ACTIVITY", etc.)
            
        Returns:
            List of ActivitySegment objects
        """
        logger.debug(f"_extract_segments_from_mask() for {activity_type}")
        logger.debug(f"Activity mask shape: {activity_mask.shape}")
        logger.debug(f"True values in mask: {np.sum(activity_mask)}")
        
        segments = []
        
        # Find contiguous regions of True values
        diff = np.diff(np.concatenate(([False], activity_mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        logger.debug(f"Found {len(starts)} raw segments before gap merging")
        
        if len(starts) == 0:
            logger.debug(f"No segments found for {activity_type}")
            return segments
        
        # Enhanced gap tolerance for short videos
        video_duration = len(motion_timeline) / fps
        if video_duration < 60:  # For videos under 1 minute
            GAP_TOLERANCE_SECONDS = 5.0  # Much more tolerant
        else:
            GAP_TOLERANCE_SECONDS = 2.0
            
        logger.debug(f"Using gap tolerance: {GAP_TOLERANCE_SECONDS}s for {video_duration:.1f}s video")
        
        merged_starts = []
        merged_ends = []
        
        current_start = starts[0]
        current_end = ends[0]
        
        logger.debug(f"Starting gap merging with tolerance: {GAP_TOLERANCE_SECONDS}s")
        
        for i in range(1, len(starts)):
            gap_frames = starts[i] - current_end
            gap_seconds = gap_frames / fps
            
            logger.debug(f"Gap between segment {i-1} and {i}: {gap_seconds:.1f}s ({gap_frames} frames)")
            
            # If gap is small enough, merge segments
            if gap_seconds <= GAP_TOLERANCE_SECONDS:
                current_end = ends[i]  # Extend current segment to include the gap
                logger.debug(f"✅ Merged segments - gap {gap_seconds:.1f}s <= {GAP_TOLERANCE_SECONDS}s")
            else:
                # Save current merged segment and start new one
                merged_starts.append(current_start)
                merged_ends.append(current_end)
                logger.debug(f"❌ Gap too large - saved segment {current_start}-{current_end}")
                
                current_start = starts[i]
                current_end = ends[i]
        
        # Don't forget the last segment
        merged_starts.append(current_start)
        merged_ends.append(current_end)
        
        logger.debug(f"After gap merging: {len(merged_starts)} segments")
        
        # === Process merged segments ===
        for i, (start_frame, end_frame) in enumerate(zip(merged_starts, merged_ends)):
            duration = (end_frame - start_frame) / fps
            
            logger.debug(f"Merged segment {i+1}: frames {start_frame}-{end_frame}, duration: {duration:.1f}s")
            
            # Filter by minimum duration
            if duration >= ACTIVITY_MIN_DURATION:
                start_time = start_frame / fps
                end_time = end_frame / fps
                
                # Calculate segment statistics
                segment_motion = motion_timeline[start_frame:end_frame]
                avg_motion = float(np.mean(segment_motion))
                max_motion = float(np.max(segment_motion))
                
                # Enhanced confidence calculation
                motion_std = float(np.std(segment_motion))
                motion_consistency = 1.0 - min(motion_std, 0.5) / 0.5
                
                # Bonus confidence for longer segments (more likely to be real activity)
                duration_bonus = min(duration / 30.0, 1.0)  # Max bonus at 30 seconds
                
                confidence = (motion_consistency * 0.7 + duration_bonus * 0.3)
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                
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
                print(f"[DEBUG] ✅ Added {activity_type} segment: {start_time:.1f}s-{end_time:.1f}s "
                    f"({duration:.1f}s), confidence: {confidence:.3f}")
            else:
                logger.debug(f"❌ Rejected merged segment (too short): {duration:.1f}s < {ACTIVITY_MIN_DURATION}s")
        
        logger.debug(f"Extracted {len(segments)} valid segments for {activity_type}")
        return segments
    
    def _resolve_overlapping_segments(self, segments: List[ActivitySegment]) -> List[ActivitySegment]:
        """
        Resolve overlapping segments by merging or removing conflicts.
        
        Args:
            segments: List of potentially overlapping segments
            
        Returns:
            List of non-overlapping segments
        """
        if len(segments) <= 1:
            return segments
        
        logger.debug(f"Resolving overlaps in {len(segments)} segments")
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x.start_time)
        
        resolved_segments = []
        current_segment = sorted_segments[0]
        
        for next_segment in sorted_segments[1:]:
            # Check for overlap
            if current_segment.end_time > next_segment.start_time:
                overlap_duration = current_segment.end_time - next_segment.start_time
                logger.debug(f"Overlap detected: {overlap_duration:.1f}s between segments")
                logger.debug(f"  Current: {current_segment.start_time:.1f}s-{current_segment.end_time:.1f}s ({current_segment.activity_type})")
                logger.debug(f"  Next: {next_segment.start_time:.1f}s-{next_segment.end_time:.1f}s ({next_segment.activity_type})")
                
                # Decide how to resolve overlap
                if overlap_duration < 3.0:  # Small overlap - adjust boundaries
                    # Move the boundary to the midpoint
                    midpoint = (current_segment.end_time + next_segment.start_time) / 2
                    
                    logger.debug(f"  Small overlap - adjusting boundary to {midpoint:.1f}s")
                    
                    # Adjust current segment end time
                    current_segment = ActivitySegment(
                        start_time=current_segment.start_time,
                        end_time=midpoint,
                        start_frame=current_segment.start_frame,
                        end_frame=int(midpoint * self.current_video_info["fps"]),
                        duration=midpoint - current_segment.start_time,
                        activity_type=current_segment.activity_type,
                        avg_motion_energy=current_segment.avg_motion_energy,
                        max_motion_energy=current_segment.max_motion_energy,
                        confidence=current_segment.confidence,
                        description=current_segment.description
                    )
                    
                    # Adjust next segment start time
                    next_segment = ActivitySegment(
                        start_time=midpoint,
                        end_time=next_segment.end_time,
                        start_frame=int(midpoint * self.current_video_info["fps"]),
                        end_frame=next_segment.end_frame,
                        duration=next_segment.end_time - midpoint,
                        activity_type=next_segment.activity_type,
                        avg_motion_energy=next_segment.avg_motion_energy,
                        max_motion_energy=next_segment.max_motion_energy,
                        confidence=next_segment.confidence,
                        description=next_segment.description
                    )
                    
                else:  # Large overlap - keep higher confidence segment
                    if current_segment.confidence >= next_segment.confidence:
                        logger.debug(f"  Large overlap - keeping current (higher confidence: {current_segment.confidence:.3f})")
                        # Keep current, skip next
                        continue
                    else:
                        logger.debug(f"  Large overlap - keeping next (higher confidence: {next_segment.confidence:.3f})")
                        # Replace current with next
                        current_segment = next_segment
                        continue
            
            # No overlap or resolved - add current and move to next
            resolved_segments.append(current_segment)
            current_segment = next_segment
        
        # Don't forget the last segment
        resolved_segments.append(current_segment)
        
        logger.debug(f"Overlap resolution completed: {len(segments)} -> {len(resolved_segments)} segments")
        return resolved_segments
    
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
        
        motion_desc = f"motion level {avg_motion:.3f}"
        
        activity_descriptions = {
            "HIGH_ACTIVITY": f"High-intensity activity period ({duration_desc}) - likely active cooking, cleaning, or exercise",
            "MEDIUM_ACTIVITY": f"Moderate activity period ({duration_desc}) - likely food prep, organizing, or light tasks",
            "LOW_ACTIVITY": f"Low activity period ({duration_desc}) - likely resting, reading, or passive tasks",
        }
        
        base_desc = activity_descriptions.get(activity_type, f"Activity period ({duration_desc})")
        return f"{base_desc} | {motion_desc}"
    
    def _detect_high_energy_regions(self, motion_timeline: List[float], fps: float,
                                  threshold: float = 0.6, min_length_sec: float = 1.0) -> List[Dict]:
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
        logger.debug(f"_detect_high_energy_regions() starting")
        logger.debug(f"Threshold: {threshold}, Min length: {min_length_sec}s")
        logger.debug(f"Timeline length: {len(motion_timeline)}, FPS: {fps}")
        
        regions = []
        min_frames = int(min_length_sec * fps)
        start = None
        
        logger.debug(f"Minimum frames for region: {min_frames}")
        
        for i, energy in enumerate(motion_timeline):
            if energy >= threshold:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start >= min_frames:
                        region = {
                            "start_time": round(start / fps, 2),
                            "end_time": round((i - 1) / fps, 2),
                            "duration_sec": round((i - start) / fps, 2)
                        }
                        regions.append(region)
                        logger.debug(f"High energy region found: {region['start_time']}s-{region['end_time']}s ({region['duration_sec']}s)")
                    start = None

        # Check if video ends with high energy region
        if start is not None and len(motion_timeline) - start >= min_frames:
            region = {
                "start_time": round(start / fps, 2),
                "end_time": round((len(motion_timeline) - 1) / fps, 2),
                "duration_sec": round((len(motion_timeline) - start) / fps, 2)
            }
            regions.append(region)
            logger.debug(f"High energy region at end: {region['start_time']}s-{region['end_time']}s ({region['duration_sec']}s)")

        logger.debug(f"Detected {len(regions)} high energy regions total")
        return regions
    
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
        logger.debug(f"_save_results() starting")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Save annotated: {save_annotated}")
        logger.debug(f"Save motion data: {save_motion_data}")
        logger.debug(f"Annotated frames count: {len(annotated_frames)}")
        
        # Create base filename
        video_name = os.path.splitext(os.path.basename(results.video_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"{video_name}_{timestamp}"
        
        logger.debug(f"Base filename: {base_name}")
        
        # Save segmentation results as JSON
        logger.debug(f"Saving segmentation results to JSON...")
        try:
            results_path = os.path.join(output_dir, f"{base_name}_segments.json")
            self._save_segments_json(results, results_path)
            logger.debug(f"✅ Segments JSON saved: {results_path}")
        except Exception as e:
            logger.debug(f"❌ ERROR saving segments JSON: {e}")
        
        # Save motion timeline data
        if save_motion_data:
            logger.debug(f"Saving motion timeline data...")
            try:
                motion_path = os.path.join(output_dir, f"{base_name}_motion_data.json")
                self.motion_detector.export_motion_data(motion_path)
                logger.debug(f"✅ Motion data saved: {motion_path}")
            except Exception as e:
                logger.debug(f"❌ ERROR saving motion data: {e}")
        else:
            logger.debug(f"Skipping motion data save (save_motion_data=False)")
        
        # Save annotated video
        if save_annotated and annotated_frames:
            logger.debug(f"Saving annotated video...")
            try:
                video_path = os.path.join(output_dir, f"{base_name}_annotated.mp4")
                fps = self.current_video_info["fps"]
                logger.debug(f"Video path: {video_path}, FPS: {fps}")
                
                self.video_processor.save_video_with_overlay(
                    results.video_path, annotated_frames, video_path, fps
                )
                logger.debug(f"✅ Annotated video saved: {video_path}")
            except Exception as e:
                logger.debug(f"❌ ERROR saving annotated video: {e}")
        else:
            logger.debug(f"Skipping annotated video save (save_annotated={save_annotated}, frames={len(annotated_frames)})")
        
        # Save summary report
        logger.debug(f"Saving summary report...")
        try:
            summary_path = os.path.join(output_dir, f"{base_name}_summary.txt")
            self._save_summary_report(results, summary_path)
            logger.debug(f"✅ Summary report saved: {summary_path}")
        except Exception as e:
            logger.debug(f"❌ ERROR saving summary report: {e}")
        
        logger.info("All results saved successfully")
        logger.debug(f"_save_results() completed")
    
    def _save_segments_json(self, results: SegmentationResults, output_path: str):
        """Save segmentation results as JSON."""
        logger.debug(f"_save_segments_json() starting")
        logger.debug(f"Output path: {output_path}")
        logger.debug(f"Results summary:")
        logger.debug(f"  Segments count: {len(results.segments_list)}")
        logger.debug(f"  Total duration: {results.duration}")
        logger.debug(f"  Processing time: {results.processing_time}")
        
        try:
            results_dict = {
                "video_info": {
                    "path": results.video_path,
                    "duration": results.duration,
                    "total_segments": results.segments,
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
                    for seg in results.segments_list
                ],
                "motion_statistics": results.motion_statistics,
                "metadata": {
                    "analysis_version": "1.0",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "configuration": {
                        "high_motion_threshold": HIGH_MOTION_THRESHOLD,
                        "medium_motion_threshold": MEDIUM_MOTION_THRESHOLD,
                        "activity_min_duration": ACTIVITY_MIN_DURATION
                    }
                }
            }
            
            logger.debug(f"Writing JSON data...")
            with open(output_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            logger.debug(f"✅ Segments JSON file written successfully")
            
        except Exception as e:
            logger.debug(f"❌ ERROR in _save_segments_json: {e}")
            raise
    
    def _save_summary_report(self, results: SegmentationResults, output_path: str):
        """Save human-readable summary report."""
        logger.debug(f"_save_summary_report() starting")
        logger.debug(f"Output path: {output_path}")
        
        try:
            with open(output_path, 'w') as f:
                f.write("=== 360° HOME ACTIVITY ANALYSIS REPORT ===\n\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Analysis Version: 1.0 (Core Segmentation)\n\n")
                
                # Video Information
                f.write("=== VIDEO INFORMATION ===\n")
                f.write(f"Video: {os.path.basename(results.video_path)}\n")
                f.write(f"Full Path: {results.video_path}\n")
                f.write(f"Duration: {results.duration:.1f} seconds ({results.duration/60:.1f} minutes)\n")
                f.write(f"Processing Time: {results.processing_time:.1f} seconds\n")
                f.write(f"Processing Speed: {results.duration/results.processing_time:.1f}x real-time\n")
                f.write(f"Total Segments Found: {results.segments}\n\n")
                
                # Motion Statistics
                f.write("=== MOTION STATISTICS ===\n")
                stats = results.motion_statistics
                if stats:
                    f.write(f"Total Frames Analyzed: {stats.get('total_frames', 'unknown')}\n")
                    f.write(f"Average Motion Energy: {stats.get('avg_motion_energy', 0):.4f}\n")
                    f.write(f"Peak Motion Energy: {stats.get('max_motion_energy', 0):.4f}\n")
                    f.write(f"Minimum Motion Energy: {stats.get('min_motion_energy', 0):.4f}\n")
                    f.write(f"Motion Variance: {stats.get('motion_variance', 0):.6f}\n")
                    f.write(f"Motion Standard Deviation: {stats.get('motion_std_dev', 0):.4f}\n\n")
                    
                    f.write("Activity Level Breakdown:\n")
                    f.write(f"  High Activity: {stats.get('high_activity_percentage', 0):.1f}% of video ({stats.get('high_activity_frames', 0)} frames)\n")
                    f.write(f"  Medium Activity: {stats.get('medium_activity_percentage', 0):.1f}% of video ({stats.get('medium_activity_frames', 0)} frames)\n")
                    f.write(f"  Low Activity: {stats.get('low_activity_percentage', 0):.1f}% of video ({stats.get('low_activity_frames', 0)} frames)\n\n")
                else:
                    f.write("Motion statistics not available\n\n")
                
                # High Energy Regions
                if results.high_energy_regions:
                    f.write("=== HIGH ENERGY REGIONS ===\n")
                    f.write(f"Total High Energy Regions: {len(results.high_energy_regions)}\n")
                    total_high_energy_time = sum(region['duration_sec'] for region in results.high_energy_regions)
                    f.write(f"Total High Energy Time: {total_high_energy_time:.1f} seconds ({total_high_energy_time/results.duration*100:.1f}% of video)\n\n")
                    
                    for i, region in enumerate(results.high_energy_regions, 1):
                        f.write(f"  Region {i}: {region['start_time']}s - {region['end_time']}s ({region['duration_sec']}s)\n")
                    f.write("\n")
                
                # Activity Segments
                f.write("=== ACTIVITY SEGMENTS ===\n")
                if results.segments:
                    # Summary statistics
                    total_segment_time = sum(seg.duration for seg in results.segments_list)
                    coverage_percentage = (total_segment_time / results.duration) * 100
                    
                    f.write(f"Total Segments: {len(results.segments_list)}\n")
                    f.write(f"Total Segment Time: {total_segment_time:.1f} seconds ({total_segment_time/60:.1f} minutes)\n")
                    f.write(f"Video Coverage: {coverage_percentage:.1f}%\n\n")
                    
                    # Segment breakdown by type
                    segment_types = {}
                    for seg in results.segments_list:
                        if seg.activity_type not in segment_types:
                            segment_types[seg.activity_type] = []
                        segment_types[seg.activity_type].append(seg)
                    
                    f.write("Segment Breakdown by Type:\n")
                    for seg_type, segs in segment_types.items():
                        total_duration = sum(s.duration for s in segs)
                        avg_confidence = sum(s.confidence for s in segs) / len(segs)
                        f.write(f"  {seg_type}: {len(segs)} segments, {total_duration:.1f}s total, {avg_confidence:.3f} avg confidence\n")
                    f.write("\n")
                    
                    # Individual segments
                    f.write("Individual Segments (sorted by duration):\n")
                    sorted_segments = sorted(results.segments_list, key=lambda x: x.duration, reverse=True)
                    
                    for i, segment in enumerate(sorted_segments, 1):
                        f.write(f"\nSegment {i}:\n")
                        f.write(f"  Time Range: {segment.start_time:.1f}s - {segment.end_time:.1f}s\n")
                        f.write(f"  Duration: {segment.duration:.1f} seconds")
                        if segment.duration > 60:
                            f.write(f" ({segment.duration/60:.1f} minutes)")
                        f.write("\n")
                        f.write(f"  Activity Type: {segment.activity_type}\n")
                        f.write(f"  Motion Energy: {segment.avg_motion_energy:.4f} (avg), {segment.max_motion_energy:.4f} (max)\n")
                        f.write(f"  Confidence: {segment.confidence:.3f}\n")
                        f.write(f"  Description: {segment.description}\n")
                        
                        # Add time references for easy video navigation
                        minutes_start = int(segment.start_time // 60)
                        seconds_start = int(segment.start_time % 60)
                        minutes_end = int(segment.end_time // 60)
                        seconds_end = int(segment.end_time % 60)
                        f.write(f"  Video Time: {minutes_start:02d}:{seconds_start:02d} - {minutes_end:02d}:{seconds_end:02d}\n")
                else:
                    f.write("No activity segments detected.\n\n")
                    f.write("Possible reasons:\n")
                    f.write("- Video contains mostly static scenes\n")
                    f.write("- Motion detection sensitivity too low\n")
                    f.write("- Activities too brief (< 30 seconds)\n")
                    f.write("- Video quality issues\n\n")
                
                # Configuration Information
                f.write("=== ANALYSIS CONFIGURATION ===\n")
                f.write(f"High Motion Threshold: {HIGH_MOTION_THRESHOLD}\n")
                f.write(f"Medium Motion Threshold: {MEDIUM_MOTION_THRESHOLD}\n")
                f.write(f"Minimum Activity Duration: {ACTIVITY_MIN_DURATION} seconds\n")
                f.write(f"Background Subtraction History: {BG_HISTORY} frames\n")
                f.write(f"Foreground Area Threshold: {MIN_FOREGROUND_AREA} pixels\n")
                f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n\n")
                
                # Recommendations
                f.write("=== RECOMMENDATIONS ===\n")
                if results.segments:
                    if len(results.segments_list) < 3:
                        f.write("- Few segments detected: Consider lowering motion sensitivity thresholds\n")
                    if coverage_percentage < 10:
                        f.write("- Low activity coverage: Video may be mostly static\n")
                    if any(seg.confidence < 0.5 for seg in results.segments_list):
                        f.write("- Some low-confidence segments detected: Review video quality\n")
                    
                    # Find most active period
                    if results.segments:
                        most_active = max(results.segments_list, key=lambda x: x.avg_motion_energy)
                        f.write(f"- Most active period: {most_active.start_time:.1f}s-{most_active.end_time:.1f}s ({most_active.activity_type})\n")
                else:
                    f.write("- No segments detected: Try 'high' sensitivity mode\n")
                    f.write("- Verify video contains human activities\n")
                    f.write("- Check video is not corrupted or mostly black frames\n")
                
                f.write("\n=== END OF REPORT ===\n")
            
            logger.debug(f"✅ Summary report written successfully")
            logger.debug(f"Report sections: Video Info, Motion Stats, Segments, Config, Recommendations")
            
        except Exception as e:
            logger.debug(f"❌ ERROR in _save_summary_report: {e}")
            raise