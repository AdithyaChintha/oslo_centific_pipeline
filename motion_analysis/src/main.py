"""
Main entry point for 360° Home Activity Motion Analysis Pipeline.
Complete implementation with production-ready class structure.
"""
import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.motion_detector import MotionDetector
from core.video_processor import Video360Processor
from analyzers.activity_segmenter import ActivitySegmenter
from utils.logging_utils import setup_logging, get_logger
from utils.config import *

class MotionEnergyAnalysisPipeline:
    """
    Production-ready 360° Video Analysis Pipeline.
    
    This class orchestrates the complete motion analysis workflow including
    motion detection, activity segmentation, and results generation.
    """
    
    def __init__(self, config: Dict):
        # Merge with defaults from config.py FIRST
        self.config = {**DEFAULT_PIPELINE_CONFIG, **config}
        
        # Setup logging IMMEDIATELY
        self.setup_logging()
        
        # Now you can use self.logger
        self.logger.debug(f"VideoAnalysisPipeline.__init__() starting")
        self.logger.debug(f"Pipeline configuration:")
        for key, value in self.config.items():
            self.logger.debug(f"  {key}: {value}")
        
        # Initialize components
        self.motion_detector = None
        self.video_processor = None
        self.activity_segmenter = None
        
        # Initialize state
        self.current_results = None
        self.processing_start_time = None
        
        self.logger.debug(f"VideoAnalysisPipeline initialization completed")
    
    def setup_logging(self):
        """Set up logging for the pipeline."""
        log_level = logging.DEBUG if self.config.get('verbose_logging', True) else logging.INFO
        log_dir = self.config.get('log_directory', 'logs')
        
        setup_logging(log_level, log_dir)
        self.logger = get_logger(__name__)
        
        self.logger.debug(f"Logging configured: level={log_level}, dir={log_dir}")
    
    def initialize_components(self):
        """Initialize motion detection and video processing components."""
        self.logger.debug(f"Initializing analysis components...")
        
        self.motion_detector = MotionDetector()
        self.video_processor = Video360Processor()
        self.activity_segmenter = ActivitySegmenter(self.motion_detector, self.video_processor)
        
        self.logger.debug(f"✅ Components initialized successfully")
        self.logger.debug(f"  Motion detector: {type(self.motion_detector).__name__}")
        self.logger.debug(f"  Video processor: {type(self.video_processor).__name__}")
        self.logger.debug(f"  Activity segmenter: {type(self.activity_segmenter).__name__}")
    
    def validate_setup(self) -> bool:
        """
        Validate that all required components are properly installed.
        
        Returns:
            True if setup is valid, False otherwise
        """
        self.logger.debug(f"Validating setup...")
        
        # Test installation
        if not self.test_installation():
            print("❌ Installation incomplete. Please install missing components.")
            return False
        
        # Test GPU capabilities
        gpu_ready = self.test_gpu_installation()
        if gpu_ready:
            print("🚀 GPU acceleration enabled!")
        else:
            print("⚠️  Running in CPU mode")
        
        # Validate configuration
        if not self.validate_configuration():
            print("❌ Configuration issues found. Please resolve them first.")
            return False
        
        self.logger.debug(f"✅ Setup validation completed successfully")
        return True
    
    def format_time_human_readable(self, seconds):
        """
        Convert seconds to human-readable format (e.g., 1h 15m 23s, 5m 23s, 45s).
        
        Args:
            seconds: Time in seconds (float)
            
        Returns:
            Human-readable time string
        """
        total_seconds = int(seconds)
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:  # Less than 1 hour
            minutes = total_seconds // 60
            remaining_seconds = total_seconds % 60
            
            if remaining_seconds == 0:
                return f"{minutes}m"
            else:
                return f"{minutes}m {remaining_seconds}s"
        else:  # 1 hour or more
            hours = total_seconds // 3600
            remaining_minutes = (total_seconds % 3600) // 60
            remaining_seconds = total_seconds % 60
            
            # Build the string progressively
            result = f"{hours}h"
            
            if remaining_minutes > 0:
                result += f" {remaining_minutes}m"
            
            if remaining_seconds > 0:
                result += f" {remaining_seconds}s"
            
            return result
    
    
    def format_time_range(self, start_seconds, end_seconds):
        """
        Format a time range in human-readable format.
        
        Args:
            start_seconds: Start time in seconds
            end_seconds: End time in seconds
            
        Returns:
            Formatted range string (e.g., "1m 5s-2m 30s")
        """
        start_str = self.format_time_human_readable(start_seconds)
        end_str = self.format_time_human_readable(end_seconds)
        return f"{start_str}-{end_str}"
    
    def test_installation(self) -> bool:
        """Test if all required components are properly installed."""
        print("🔧 Testing installation...")
        
        try:
            import cv2
            self.logger.info(f"OpenCV version: {cv2.__version__}")
        except ImportError:
            print("❌ OpenCV not installed: pip install opencv-python")
            return False
        
        try:
            import numpy as np
            self.logger.info(f"NumPy version: {np.__version__}")
        except ImportError:
            print("❌ NumPy not installed: pip install numpy")
            return False
        
        try:
            import scipy
            self.logger.info(f"SciPy version: {scipy.__version__}")
        except ImportError:
            print("❌ SciPy not installed: pip install scipy")
            return False
        
        try:
            import sklearn
            self.logger.info(f"Scikit-learn version: {sklearn.__version__}")
        except ImportError:
            print("❌ Scikit-learn not installed: pip install scikit-learn")
            return False
        
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ ffmpeg is available")
            else:
                print("⚠️  ffmpeg may not be properly installed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ ffmpeg not found - please install ffmpeg")
            print("   Windows: Download from https://ffmpeg.org/download.html")
            print("   Linux: sudo apt install ffmpeg")  
            print("   Mac: brew install ffmpeg")
            return False
        
        print("✅ All components installed correctly!")
        return True
    
    def test_gpu_installation(self) -> bool:
        """Test GPU acceleration capabilities."""
        print("🔧 Testing GPU acceleration...")
        
        # Test CuPy
        try:
            import cupy as cp
            gpu_count = cp.cuda.runtime.getDeviceCount()
            self.logger.info(f"CuPy available with {gpu_count} GPU(s)")
            
            # Test basic operation
            test = cp.array([1, 2, 3])
            result = cp.sum(test)
            self.logger.info(f"GPU computation test passed")
            
        except ImportError:
            print("❌ CuPy not available: pip install cupy-cuda12x")
            return False
        except Exception as e:
            print(f"⚠️  CuPy error: {e}")
            return False
        
        # Test OpenCV CUDA
        try:
            import cv2
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                self.logger.info(f"OpenCV CUDA support with {cuda_devices} device(s)")
                return True
            else:
                print("❌ OpenCV CUDA not available")
                return False
        except:
            print("❌ OpenCV CUDA not available")
            return False
    
    def validate_configuration(self) -> bool:
        """Validate the configuration before starting analysis."""
        print("🔍 Validating configuration...")
        
        issues = []
        
        # Create output directory if it doesn't exist
        output_dir = self.config.get('output_directory', 'output')
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Output directory ready: {output_dir}")
        except Exception as e:
            issues.append(f"Cannot create output directory: {str(e)}")
        
        # Check ffmpeg availability
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                issues.append("ffmpeg not working properly")
        except:
            issues.append("ffmpeg not found in system PATH")
        
        if issues:
            print("⚠️  Configuration issues found:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        
        print("✅ Configuration validated successfully")
        return True
    
    def configure_sensitivity(self, sensitivity: str):
        """
        Configure motion detection sensitivity.
        
        Args:
            sensitivity: Sensitivity level ('low', 'medium', 'high')
        """
        sensitivity_configs = {
            'low': {
                'min_area': 800,
                'confidence': 0.4,
                'bg_threshold': 80,
                'description': 'Detects only major movements (best for noisy environments)'
            },
            'medium': {
                'min_area': 200,
                'confidence': 0.2,
                'bg_threshold': 50,
                'description': 'Balanced detection (recommended for most home activities)'
            },
            'high': {
                'min_area': 100,
                'confidence': 0.1,
                'bg_threshold': 25,
                'description': 'Detects subtle movements (best for detailed activity analysis)'
            }
        }
        
        config = sensitivity_configs[sensitivity]
        
        print(f"🔧 Motion sensitivity: {sensitivity.upper()}")
        print(f"   {config['description']}")
        self.logger.debug(f"Configuration: {config}")
        
        if self.logger:
            self.logger.info(f"Motion sensitivity configured to: {sensitivity}")
    
    def run_analysis(self):
        """
        Main entry point for running the complete analysis pipeline.
        """
        print("="*60)
        print("🎥 360° HOME ACTIVITY MOTION ANALYSIS PIPELINE")
        print("="*60)
        self.logger.debug(f"Pipeline configuration summary:")
        self.logger.debug(f"  Video: {self.config.get('video_path', 'Not specified')}")
        self.logger.debug(f"  Output: {self.config.get('output_directory', 'output')}")
        self.logger.debug(f"  Sensitivity: {self.config.get('sensitivity_level', 'medium')}")
        
        try:
            # Setup and validation
            self.setup_logging()
            
            if not self.validate_setup():
                print("\n❌ Setup validation failed. Exiting.")
                sys.exit(1)
            
            # Initialize components
            self.initialize_components()
            
            # Configure sensitivity
            sensitivity = self.config.get('sensitivity_level', 'medium')
            self.configure_sensitivity(sensitivity)
            
            # Check for video path
            video_path = self.config.get('video_path')
            video_directory = self.config.get('video_directory')
            
            if video_directory:
                # Batch processing
                print(f"\n📁 Processing directory: {video_directory}")
                batch_results = self.analyze_batch(video_directory)
                self.display_batch_summary(batch_results)
                
            elif video_path:
                # Single video processing
                print(f"\n🎬 Processing single video: {video_path}")
                
                # Check if video file exists
                if not os.path.exists(video_path):
                    self.logger.error(f"Error: Video file not found: {video_path}")
                    print("👉 Please edit the video_path in configuration")
                    return
                
                results = self.analyze_single_video(video_path)
                self.display_results_summary(results)
                
            else:
                print("❌ Error: No video path or directory specified in configuration")
                print("👉 Please set 'video_path' or 'video_directory' in config")
                return
            
            print("\n🎉 Analysis completed successfully!")
            output_dir = self.config.get('output_directory', 'output')
            print(f"📂 Results saved to: {os.path.abspath(output_dir)}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Analysis interrupted by user")
            if self.logger:
                self.logger.info(f"Analysis interrupted by user")
            sys.exit(1)
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            if self.logger:
                self.logger.error(f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def analyze_single_video(self, video_path: str):
        """
        Process a single video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            SegmentationResults object
        """
        print(f"🔍 Analyzing video properties...")
        
        # Validate video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video info
        video_info = self.video_processor.get_video_info(video_path)
        
        print(f"📹 Video Info:")
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        duration_str = self.format_time_human_readable(video_info['duration_seconds'])
        print(f"   Duration: {duration_str}")
        print(f"   FPS: {video_info['fps']:.1f}")
        print(f"   Size: {video_info['file_size_mb']:.1f} MB")
        print(f"   360° Format: {'✅ Yes' if video_info['is_360_likely'] else '❓ Unknown'}")
        
        # Estimate processing time
        time_estimate = self.video_processor.estimate_processing_time(video_info)
        est_time_str = self.format_time_human_readable(time_estimate['estimated_processing_time'])
        print(f"⏱️  Estimated processing time: {est_time_str}")
        
        if time_estimate['estimated_minutes'] > 30:
            print("⚠️  Long processing time expected - consider running overnight")
        
        # Start analysis
        print(f"\n🔄 Starting motion analysis...")
        start_time = time.time()
        
        output_dir = self.config.get('output_directory', 'output')
        save_annotated = self.config.get('save_annotated_video', True)
        save_motion_data = self.config.get('save_motion_data', True)
        
        results = self.activity_segmenter.analyze_video(
            video_path=video_path,
            save_annotated=save_annotated,
            save_motion_data=save_motion_data,
            output_dir=output_dir
        )
        
        processing_time = time.time() - start_time
        processing_str = self.format_time_human_readable(processing_time)
        self.logger.info(f"Analysis completed in {processing_str}")
        
        # Store results for later access
        self.current_results = results
        
        return results
    
    def analyze_batch(self, directory_path: str) -> List:
        """
        Process all videos in a directory.
        
        Args:
            directory_path: Directory containing videos
            
        Returns:
            List of processing results
        """
        print(f"📁 Scanning directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        video_files = self.video_processor.get_video_files_in_directory(directory_path)
        
        if not video_files:
            self.logger.error(f"No video files found in directory: {directory_path}")
            return []
        
        print(f"📁 Found {len(video_files)} video files to process")
        
        batch_results = []
        output_dir = self.config.get('output_directory', 'output')
        save_annotated = self.config.get('save_annotated_video', True)
        save_motion_data = self.config.get('save_motion_data', True)
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n{'='*40}")
            print(f"📹 Processing video {i}/{len(video_files)}: {os.path.basename(video_path)}")
            print(f"{'='*40}")
            
            try:
                # Create unique output directory for each video
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                video_output_dir = os.path.join(output_dir, f"video_{i:03d}_{video_name}")
                
                self.logger.debug(f"Processing video: {video_path}")
                self.logger.debug(f"Output directory: {video_output_dir}")
                
                # Temporarily update config for this video
                original_output_dir = self.config['output_directory']
                self.config['output_directory'] = video_output_dir
                
                results = self.analyze_single_video(video_path)
                
                # Restore original output directory
                self.config['output_directory'] = original_output_dir
                
                batch_results.append({
                    "video_path": video_path,
                    "video_name": os.path.basename(video_path),
                    "results": results,
                    "success": True,
                    "error": None
                })
                
                self.logger.info(f"Successfully processed: {os.path.basename(video_path)}")
                
            except Exception as e:
                error_msg = f"Failed to process {video_path}: {str(e)}"
                self.logger.error(f"{error_msg}")
                if self.logger:
                    self.logger.error(error_msg)
                
                batch_results.append({
                    "video_path": video_path,
                    "video_name": os.path.basename(video_path),
                    "results": None,
                    "success": False,
                    "error": error_msg
                })
        
        successful_count = len([r for r in batch_results if r['success']])
        print(f"\n📊 Batch processing completed: {successful_count}/{len(batch_results)} successful")
        return batch_results
    
    def display_results_summary(self, results):
        """Display a summary of analysis results."""
        print(f"\n📊 ANALYSIS RESULTS SUMMARY")
        print(f"{'='*40}")
        
        print(f"🎬 Video: {os.path.basename(results.video_path)}")
        duration_str = self.format_time_human_readable(results.duration)
        print(f"⏱️  Duration: {duration_str}")
        print(f"🎯 Segments Found: {results.segments}")
        processing_str = self.format_time_human_readable(results.processing_time)
        print(f"🚀 Processing Time: {processing_str}")
        
        stats = results.motion_statistics
        print(f"\n📈 Motion Statistics:")
        print(f"   Average Motion Energy: {stats['avg_motion_energy']:.4f}")
        print(f"   Peak Motion Energy: {stats['max_motion_energy']:.4f}")
        print(f"   Motion Variance: {stats['motion_variance']:.4f}")
        print(f"   High Activity: {stats['high_activity_percentage']:.1f}% of video")
        print(f"   Medium Activity: {stats['medium_activity_percentage']:.1f}% of video")
        print(f"   Low Activity: {stats['low_activity_percentage']:.1f}% of video")
        
        if results.segments:
            print(f"\n🎯 Top Activity Segments:")
            # Show top 5 segments by duration
            top_segments = sorted(results.segments_list, key=lambda x: x.duration, reverse=True)[:5]
            
            for i, segment in enumerate(top_segments, 1):
                duration_str = self.format_time_human_readable(segment.duration)
                time_range = self.format_time_range(segment.start_time, segment.end_time)
                
                print(f"   {i}. {time_range} ({duration_str})")
                print(f"      Type: {segment.activity_type}")
                print(f"      Motion: {segment.avg_motion_energy:.3f} | Confidence: {segment.confidence:.3f}")
                print(f"      Description: {segment.description}")
                print()
        
        else:
            print("\n⚠️  No significant activity segments detected")
            print("   💡 Try adjusting sensitivity or check video content:")
            print("   • Set sensitivity_level = 'high' for more sensitive detection")
            print("   • Ensure video contains actual human activities")
            print("   • Check if video has sufficient motion and isn't mostly static")
            
        
    def display_batch_summary(self, batch_results: List):
        """Display summary of batch processing results."""
        print(f"\n📊 BATCH PROCESSING SUMMARY")
        print(f"{'='*50}")
        
        successful = [r for r in batch_results if r["success"]]
        failed = [r for r in batch_results if not r["success"]]
        
        self.logger.info(f"Successfully processed: {len(successful)}/{len(batch_results)} videos")
        
        if failed:
            self.logger.error(f"Failed: {len(failed)} videos")
            for fail in failed[:5]:  # Show first 5 failures
                print(f"   - {fail['video_name']}: {fail['error']}")
            if len(failed) > 5:
                print(f"   ... and {len(failed) - 5} more failures")
        
        if successful:
            print(f"\n📈 Aggregate Statistics:")
            total_duration = sum(r["results"].duration for r in successful)
            total_segments = sum(r["results"].segments for r in successful)
            total_processing_time = sum(r["results"].processing_time for r in successful)
            
            total_duration_str = self.format_time_human_readable(total_duration)
            total_processing_str = self.format_time_human_readable(total_processing_time)
            
            print(f"   Total Video Duration: {total_duration_str}")
            print(f"   Total Segments Found: {total_segments}")
            print(f"   Total Processing Time: {total_processing_str}")
            print(f"   Average Segments per Video: {total_segments/len(successful):.1f}")
            print(f"   Processing Speed: {total_duration/total_processing_time:.1f}x real-time")
            
            # Show best performing videos
            print(f"\n🏆 Top Videos by Activity Segments:")
            successful_sorted = sorted(successful, key=lambda x: x["results"].segments, reverse=True)
            for i, result in enumerate(successful_sorted[:3], 1):
                video_name = result["video_name"]
                segments_count = result["results"].segments
                duration_str = self.format_time_human_readable(result["results"].duration)
                print(f"   {i}. {video_name}: {segments_count} segments ({duration_str})")
    
    def quick_demo(self):
        """Run a quick demo if no video is specified."""
        print("\n🎬 DEMO MODE")
        print("=" * 40)
        print("To analyze your 360° video:")
        print("1. Set video_path in configuration:")
        print("   video_path = 'path/to/your_video.mp4'")
        print("2. Run the pipeline")
        print("\nExample video characteristics:")
        print("   ✅ Format: .mp4, .insv")
        print("   ✅ Type: 360° equirectangular")
        print("   ✅ Content: Home activities (cooking, cleaning)")
        print("   ✅ Duration: 30 seconds to 2+ hours")
        print("\nSample configuration:")
        print("   video_path = 'videos/cooking_session.mp4'")
        print("   output_directory = 'results'")
        print("   sensitivity_level = 'medium'")
    
    def get_results_summary(self) -> Optional[Dict]:
        """
        Get the last analysis results in a structured format.
        
        Returns:
            Dictionary with analysis summary or None if no analysis performed
        """
        if not self.current_results:
            return None
        
        results = self.current_results
        
        return {
            "video_info": {
                "path": results.video_path,
                "duration_seconds": results.duration,
                "duration_formatted": self.format_time_human_readable(results.duration),
                "total_segments": results.segments,
                "processing_time_seconds": results.processing_time,
                "processing_time_formatted": self.format_time_human_readable(results.processing_time)
            },
            "motion_statistics": results.motion_statistics,
            "segments": [
                {
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "duration_formatted": self.format_time_human_readable(seg.duration),
                    "time_range_formatted": self.format_time_range(seg.start_time, seg.end_time),
                    "activity_type": seg.activity_type,
                    "avg_motion_energy": seg.avg_motion_energy,
                    "confidence": seg.confidence,
                    "description": seg.description
                }
                for seg in results.segments_list
            ],
            "high_energy_regions": results.high_energy_regions
        }


# =============================================================================
# CONFIGURATION SETUP - Add to config.py if not already present
# =============================================================================

# Default configuration - will be merged with custom config
DEFAULT_PIPELINE_CONFIG = {
    "output_directory": "output",
    "sensitivity_level": "medium",
    "save_annotated_video": True,
    "save_motion_data": True,
    "verbose_logging": True,
    "log_directory": "logs",
    "video_path": None,
    "video_directory": None
}


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    print("[DEBUG] Starting main execution")
    
    # =============================================================
    # 📝 VIDEO CONFIGURATION - Edit these for your analysis
    # =============================================================
    
    # Custom configuration for this run (overrides defaults from config.py)
    custom_config = {
        # === VIDEO INPUT CONFIGURATION ===
        "video_path": r"input/VID_20250720_152154_00_011.mp4",  # 👈 Edit this path to your 360° video
        
        # Alternative: Process directory of videos (set video_path to None to use this)
        # "video_directory": "videos/",  # Uncomment to process all videos in directory
        
        # === OUTPUT CONFIGURATION ===
        "output_directory": "output",           # Directory to save results
        "save_annotated_video": False,          # Save video with motion overlay (saves space)
        "save_motion_data": True,              # Save detailed motion timeline data
        
        # === PROCESSING OPTIONS ===
        "sensitivity_level": "medium",         # "low", "medium", or "high"
        
        # === DEBUG AND LOGGING ===
        "verbose_logging": True,               # Enable detailed debug output
        "log_directory": "logs",               # Directory for log files
    }
    
    # =============================================================
    # 🚀 PIPELINE EXECUTION
    # =============================================================
    
    try:
        # Create and configure pipeline
        print(f"🎯 Initializing 360° Motion Analysis Pipeline...")
        pipeline = MotionEnergyAnalysisPipeline(custom_config)
        
        # Run the complete analysis
        pipeline.run_analysis()
        
        # Optionally get structured results for further processing
        summary = pipeline.get_results_summary()
        if summary:
            print(f"\n📋 Analysis completed successfully!")
            print(f"   Video: {summary['video_info']['duration_formatted']}")
            print(f"   Segments: {summary['video_info']['total_segments']}")
            print(f"   Processing: {summary['video_info']['processing_time_formatted']}")
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n🎉 Analysis pipeline completed successfully!")
    print(f"📂 Check the output directory for detailed results")
    
    # =============================================================
    # 💡 USAGE EXAMPLES FOR DIFFERENT SCENARIOS
    # =============================================================
    
    """
    # Example 1: Process single video with high sensitivity
    config = {
        "video_path": "videos/cooking_session.mp4",
        "sensitivity_level": "high",
        "output_directory": "results/cooking"
    }
    pipeline = VideoAnalysisPipeline(config)
    pipeline.run_analysis()
    
    # Example 2: Process directory of videos
    config = {
        "video_directory": "videos/",
        "sensitivity_level": "medium",
        "save_annotated_video": False,  # Save space for batch processing
        "output_directory": "batch_results"
    }
    pipeline = VideoAnalysisPipeline(config)
    pipeline.run_analysis()
    
    # Example 3: Get programmatic results
    config = {"video_path": "test.mp4"}
    pipeline = VideoAnalysisPipeline(config)
    pipeline.run_analysis()
    results = pipeline.get_results_summary()
    
    # Process results programmatically
    for segment in results['segments']:
        print(f"Activity: {segment['time_range_formatted']} - {segment['activity_type']}")
    """