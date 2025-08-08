import os
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)

@dataclass
class SegmentationResults:
    video_path: str
    duration: float
    segments: List[Dict]
    processing_time: float
    motion_statistics: Dict
    high_energy_regions: List[Dict] = None

class ActivitySegmenter:
    def __init__(self, motion_detector=None, video_processor=None):
        self.motion_detector = motion_detector
        self.video_processor = video_processor

    def analyze_video(self, video_path: str, save_annotated: bool = False, output_path: str = None, save_motion_data: bool = False, output_dir: str = None) -> SegmentationResults:
        logger.info(f"Starting video analysis: {video_path}")
        
        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get video information
        video_info = self.video_processor.get_video_info(video_path)
        
        # Reset motion detector
        self.motion_detector.reset()
        
        # Process frames using motion detector
        motion_timeline = []
        start_time = time.time()
        
        for frame, frame_index, timestamp in self.video_processor.frame_generator(video_path):
            motion_metrics, motion_regions, motion_mask = self.motion_detector.process_frame(frame, timestamp)
            motion_timeline.append(motion_metrics.motion_energy_score)
            
            if frame_index % 100 == 0:
                logger.debug(f"Processed frame {frame_index}")
        
        processing_time = time.time() - start_time
        
        # Generate segments
        segments = self._segment_activities(motion_timeline)
        
        # Get motion statistics  
        motion_stats = self.motion_detector.get_motion_statistics()
        
        # Create results
        results = SegmentationResults(
            video_path=video_path,
            duration=video_info["duration_seconds"],
            segments=segments,
            processing_time=processing_time,
            motion_statistics=motion_stats,
        )
        
        # Save outputs
        if output_dir:
            self._save_results(results, output_dir)
        
        return results

    def _save_results(self, results: SegmentationResults, output_dir: str):
        """Save results to JSON files."""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(results.video_path))[0]
        
        # Save segments JSON
        segments_file = os.path.join(output_dir, f"{video_name}_{timestamp}_segments.json")
        segments_data = {
            "video_path": results.video_path,
            "duration": results.duration,
            "processing_time": results.processing_time,
            "segments": results.segments,
            "motion_statistics": results.motion_statistics
        }
        
        with open(segments_file, 'w') as f:
            json.dump(segments_data, f, indent=2)
        
        logger.info(f"Results saved to: {segments_file}")

    def _segment_activities(self, motion_timeline: List[float]) -> List[Dict]:
        smoothed = np.convolve(motion_timeline, np.ones(30)/30, mode='same')
        logger.debug(f"Smoothed motion timeline with window size 30")

        threshold = 0.3
        above_threshold = smoothed > threshold

        segments = []
        start = None
        for i, flag in enumerate(above_threshold):
            if flag:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start >= 10:
                        segments.append({"start_frame": start, "end_frame": i-1})
                    start = None

        if start is not None and len(smoothed) - start >= 10:
            segments.append({"start_frame": start, "end_frame": len(smoothed) - 1})

        logger.debug(f"Threshold-based segmentation found {len(segments)} segments")
        return segments

    def _detect_high_energy_regions(self, motion_timeline: List[float], fps: float,
                                    threshold: float = 0.6, min_length_sec: float = 1.0):
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

        if start is not None and len(motion_timeline) - start >= min_frames:
            regions.append({
                "start_time": round(start / fps, 2),
                "end_time": round((len(motion_timeline) - 1) / fps, 2),
                "duration_sec": round((len(motion_timeline) - start) / fps, 2)
            })

        logger.debug(f"Detected {len(regions)} high energy regions")
        return regions
