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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(f"Video loaded: {video_path}, Duration: {duration:.2f}s, FPS: {fps}, Total frames: {total_frames}")

        motion_timeline = []
        prev_gray = None

        start_time = time.time()

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = np.sum(diff) / diff.size / 255.0
                motion_timeline.append(score)
            else:
                motion_timeline.append(0.0)
            prev_gray = gray

            if frame_idx % 100 == 0 or frame_idx == total_frames - 1:
                logger.debug(f"Processed frame {frame_idx+1}/{total_frames}")

        cap.release()
        processing_time = time.time() - start_time

        self.current_video_info = {
            "path": video_path,
            "duration": duration,
            "fps": fps,
        }

        logger.debug(f"Completed frame processing. Timeline length: {len(motion_timeline)}")

        segments = self._segment_activities(motion_timeline)

        motion_array = np.array(motion_timeline)
        motion_stats = {
            "total_frames": len(motion_array),
            "avg_motion_energy": float(np.mean(motion_array)),
            "max_motion_energy": float(np.max(motion_array)),
            "min_motion_energy": float(np.min(motion_array)),
            "motion_variance": float(np.var(motion_array)),
            "motion_std_dev": float(np.std(motion_array)),
            "high_activity_frames": int(np.sum(motion_array > 0.6)),
            "medium_activity_frames": int(np.sum((motion_array > 0.3) & (motion_array <= 0.6))),
            "low_activity_frames": int(np.sum(motion_array <= 0.3)),
        }

        total = len(motion_array)
        motion_stats.update({
            "high_activity_percentage": 100 * motion_stats["high_activity_frames"] / total,
            "medium_activity_percentage": 100 * motion_stats["medium_activity_frames"] / total,
            "low_activity_percentage": 100 * motion_stats["low_activity_frames"] / total,
        })

        results = SegmentationResults(
            video_path=self.current_video_info.get("path", "unknown"),
            duration=self.current_video_info["duration"],
            segments=segments,
            processing_time=processing_time,
            motion_statistics=motion_stats,
        )

        high_energy_regions = self._detect_high_energy_regions(
            motion_timeline,
            fps=fps,
            threshold=0.2,
            min_length_sec=0.2
        )
        results.high_energy_regions = high_energy_regions

        return results

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
