"""
Video I/O processing utilities for 360° motion analysis.
"""
import cv2
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List,Dict, Optional, Generator

from utils.config import *
from utils.logging_utils import get_logger

import numpy as np

logger = get_logger(__name__)

class Video360Processor:
    """
    Video processing utilities specifically designed for 360° equirectangular videos.
    """
    
    def __init__(self):
        """Initialize the video processor."""
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.insv']
        logger.info("Video360Processor initialized")
    

    def get_video_info(self, video_path: str) -> Dict:
        """
        Get comprehensive video information.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video properties
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Check if it's likely a 360° video based on aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            is_360_likely = abs(aspect_ratio - 2.0) < 0.1  # 360° videos typically have 2:1 ratio
            
            video_info = {
                "file_path": video_path,
                "file_size_mb": os.path.getsize(video_path) / (1024 * 1024),
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "duration_seconds": duration,
                "duration_minutes": duration / 60,
                "aspect_ratio": aspect_ratio,
                "is_360_likely": is_360_likely,
                "format": os.path.splitext(video_path)[1].lower()
            }
            
            logger.info(f"Video info: {width}x{height}, {fps:.1f}fps, "
                       f"{duration:.1f}s, 360°: {is_360_likely}")
            
            return video_info
            
        finally:
            cap.release()
    
    def create_video_chunks(self, video_path: str, chunk_duration: float = 60.0) -> List[Tuple[float, float]]:
        """
        Create time-based chunks for processing large videos.
        
        Args:
            video_path: Path to the video file
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of (start_time, end_time) tuples for each chunk
        """
        video_info = self.get_video_info(video_path)
        total_duration = video_info["duration_seconds"]
        
        chunks = []
        current_time = 0.0
        
        while current_time < total_duration:
            end_time = min(current_time + chunk_duration, total_duration)
            chunks.append((current_time, end_time))
            current_time = end_time
        
        logger.info(f"Created {len(chunks)} chunks of {chunk_duration}s each for "
                   f"{total_duration:.1f}s video")
        
        return chunks
    
    def extract_video_chunk(self, video_path: str, start_time: float, end_time: float, 
                           output_path: Optional[str] = None) -> str:
        """
        Extract a specific time segment from a video.
        
        Args:
            video_path: Source video path
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output path (if None, creates temp file)
            
        Returns:
            Path to the extracted chunk
        """
        if output_path is None:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        # Use ffmpeg to extract the chunk
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c', 'copy',  # Copy without re-encoding for speed
            output_path
        ]
        
        try:
            logger.debug(f"Extracting chunk {start_time:.1f}s-{end_time:.1f}s to {output_path}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            logger.debug(f"Successfully extracted chunk: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract video chunk: {e.stderr}")
    
    def frame_generator(self, video_path: str, start_frame: int = 0, 
                       end_frame: Optional[int] = None, 
                       resize_factor: float = 1) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generate frames from a video with frame indices and timestamps.
        Added resize_factor to handle high-resolution 360° videos.
        
        Args:
            video_path: Path to the video file
            start_frame: Starting frame index
            end_frame: Ending frame index (if None, processes till end)
            resize_factor: Factor to resize frames (0.5 = half size, saves memory)
            
        Yields:
            Tuple of (frame, frame_index, timestamp)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Log memory optimization for high-res videos
            if width > 4000 or height > 2000:
                logger.info(f"High resolution video detected ({width}x{height})")
                logger.info(f"Applying resize factor {resize_factor} to reduce memory usage")
                logger.debug(f"High-res video: {width}x{height} -> {int(width*resize_factor)}x{int(height*resize_factor)}")
            
            # Set starting position
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Determine end frame
            if end_frame is None:
                end_frame = total_frames
            else:
                end_frame = min(end_frame, total_frames)
            
            logger.info(f"Processing frames {start_frame} to {end_frame} "
                       f"({end_frame - start_frame} frames total)")
            
            frame_index = start_frame
            
            while frame_index < end_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_index}")
                    break
                
                # Resize frame if needed to save memory
                if resize_factor != 1.0:
                    new_width = int(frame.shape[1] * resize_factor)
                    new_height = int(frame.shape[0] * resize_factor)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                timestamp = frame_index / fps if fps > 0 else frame_index
                
                yield frame, frame_index, timestamp
                frame_index += 1
                
        finally:
            cap.release()
    
    def save_video_with_overlay(self, video_path: str, annotated_frames: List[np.ndarray], 
                               output_path: str, fps: float = 30.0):
        """
        Save a video with motion detection overlay.
        
        Args:
            video_path: Original video path
            annotated_frames: List of frames with motion overlay
            output_path: Output video path
            fps: Output video FPS
        """
        if not annotated_frames:
            logger.warning("No annotated frames to save")
            return
        
        # Get frame dimensions
        height, width = annotated_frames[0].shape[:2]
        
        # Create temporary AVI file (for compatibility)
        temp_file = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            logger.info(f"Saving {len(annotated_frames)} frames to {output_path}")
            
            # Write all frames
            for frame in annotated_frames:
                writer.write(frame)
            
            writer.release()
            
            # Convert to final format using ffmpeg
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_path,
                '-vcodec', 'libx264',
                '-crf', '23',
                '-preset', 'veryfast',
                output_path
            ]
            
            subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            
            logger.info(f"Video saved successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving video: {str(e)}")
            raise
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def convert_360_format(self, input_path: str, output_path: str, 
                          conversion_type: str = "equirectangular") -> str:
        """
        Convert 360° video formats using ffmpeg.
        
        Args:
            input_path: Source video path (.insv or other 360° format)
            output_path: Output video path
            conversion_type: Type of conversion ("equirectangular", "rectilinear")
            
        Returns:
            Path to converted video
        """
        if conversion_type == "equirectangular":
            # Convert to standard equirectangular MP4
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                '-crf', '23',
                output_path
            ]
        elif conversion_type == "rectilinear":
            # Convert to rectilinear view (reduces distortion)
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', 'v360=input=e:output=rect:out_fov=90:yaw=0',
                '-vcodec', 'libx264',
                '-acodec', 'aac',
                output_path
            ]
        else:
            raise ValueError(f"Unsupported conversion type: {conversion_type}")
        
        try:
            logger.info(f"Converting {input_path} to {conversion_type} format")
            
            result = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            logger.info(f"Successfully converted to: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Video conversion failed: {e.stderr}")
            raise RuntimeError(f"Failed to convert video: {e.stderr}")
    
    def is_video_file(self, file_path: str) -> bool:
        """
        Check if a file is a supported video format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is a supported video format
        """
        extension = os.path.splitext(file_path)[1].lower()
        return extension in self.supported_formats
    
    def get_video_files_in_directory(self, directory: str) -> List[str]:
        """
        Get all video files in a directory.
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of video file paths
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        video_files = []
        
        for file_path in Path(directory).rglob("*"):
            if file_path.is_file() and self.is_video_file(str(file_path)):
                video_files.append(str(file_path))
        
        video_files.sort()  # Sort for consistent processing order
        
        logger.info(f"Found {len(video_files)} video files in {directory}")
        return video_files
    
    def validate_360_video(self, video_path: str) -> Dict:
        """
        Validate if a video appears to be a proper 360° video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Validation results dictionary
        """
        video_info = self.get_video_info(video_path)
        
        validation = {
            "is_valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check aspect ratio (360° videos should be ~2:1)
        if not video_info["is_360_likely"]:
            validation["warnings"].append(
                f"Aspect ratio {video_info['aspect_ratio']:.2f} doesn't match typical 360° format (2:1)"
            )
        
        # Check resolution
        if video_info["width"] < 1920:
            validation["warnings"].append(
                f"Low resolution ({video_info['width']}x{video_info['height']}) may affect motion detection quality"
            )
        
        # Check duration
        if video_info["duration_seconds"] < 30:
            validation["warnings"].append(
                f"Very short video ({video_info['duration_seconds']:.1f}s) may not be sufficient for background learning"
            )
        elif video_info["duration_seconds"] > 7200:  # 2 hours
            validation["recommendations"].append(
                f"Long video ({video_info['duration_minutes']:.1f}m) - consider processing in chunks for better performance"
            )
        
        # Check FPS
        if video_info["fps"] < 15:
            validation["warnings"].append(
                f"Low FPS ({video_info['fps']:.1f}) may affect motion detection accuracy"
            )
        
        # Determine validation status
        validation["is_valid"] = len(validation["warnings"]) == 0
        
        return validation
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """
        Clean up temporary files.
        
        Args:
            file_paths: List of temporary file paths to remove
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
    
    def estimate_processing_time(self, video_info: Dict, processing_factor: float = 2.0) -> Dict:
        """
        Estimate processing time for motion analysis.
        
        Args:
            video_info: Video information dictionary
            processing_factor: Processing speed factor (2.0 = 2x real-time)
            
        Returns:
            Time estimates dictionary
        """
        duration = video_info["duration_seconds"]
        
        estimates = {
            "video_duration": duration,
            "estimated_processing_time": duration * processing_factor,
            "estimated_minutes": (duration * processing_factor) / 60,
            "processing_factor": processing_factor,
            "recommendation": "normal" if duration < 1800 else "consider_chunking"  # 30 minutes
        }
        
        if estimates["estimated_minutes"] > 30:
            estimates["warning"] = "Long processing time expected - consider processing overnight"
        
        return estimates