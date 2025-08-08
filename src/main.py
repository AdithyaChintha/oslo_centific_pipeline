"""
Main entry point for 360° Home Activity Motion Analysis Pipeline.
Complete implementation with simple string-based configuration.
"""
import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import List

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.motion_detector import MotionDetector
from core.video_processor import Video360Processor
from analyzers.activity_segmenter import ActivitySegmenter
from utils.logging_utils import setup_logging, get_logger
from utils.config import *

def main():
    """Main application entry point with simple string path configuration."""
    
    # =============================================================
    # 📝 CONFIGURATION - Edit these paths for your setup
    # =============================================================
    
    # Input video configuration
    VIDEO_PATH = r"input\VID_20250720_152154_00_011.mp4"  # 👈 Edit this path to your 360° video
    
    # Alternative: Process directory of videos (set to None to use single video)
    VIDEO_DIRECTORY = None  # Example: "videos/"
    
    # Output configuration
    OUTPUT_DIRECTORY = "output"  # Directory to save results
    
    # Processing options
    SAVE_ANNOTATED_VIDEO = False     # Save video with motion overlay
    SAVE_MOTION_DATA = True         # Save detailed motion timeline data
    SENSITIVITY_LEVEL = "medium"    # "low", "medium", or "high"
    
    # Debug and logging
    VERBOSE_LOGGING = True          # Enable detailed debug output
    LOG_DIRECTORY = "logs"          # Directory for log files
    
    # =============================================================
    # 🚀 PIPELINE EXECUTION
    # =============================================================
    
    # Set up logging
    log_level = logging.DEBUG if VERBOSE_LOGGING else logging.INFO
    setup_logging(log_level, LOG_DIRECTORY)
    logger = get_logger(__name__)
    
    print("="*60)
    print("🎥 360° HOME ACTIVITY MOTION ANALYSIS PIPELINE")
    print("="*60)
    print(f"[DEBUG] Configuration:")
    print(f"[DEBUG]   Video: {VIDEO_PATH}")
    print(f"[DEBUG]   Directory: {VIDEO_DIRECTORY}")
    print(f"[DEBUG]   Output: {OUTPUT_DIRECTORY}")
    print(f"[DEBUG]   Sensitivity: {SENSITIVITY_LEVEL}")
    
    try:
        # Configure sensitivity
        configure_sensitivity(SENSITIVITY_LEVEL)
        
        # Initialize components
        print("\n📊 Initializing motion analysis components...")
        logger.info("Initializing analysis components")
        
        motion_detector = MotionDetector()
        video_processor = Video360Processor()
        activity_segmenter = ActivitySegmenter(motion_detector, video_processor)
        
        print("✅ Components initialized successfully")
        
        # Process video(s)
        if VIDEO_DIRECTORY:
            # Batch processing
            print(f"\n📁 Processing directory: {VIDEO_DIRECTORY}")
            batch_results = process_video_directory(
                VIDEO_DIRECTORY,
                activity_segmenter,
                OUTPUT_DIRECTORY,
                SAVE_ANNOTATED_VIDEO,
                SAVE_MOTION_DATA
            )
            display_batch_summary(batch_results)
            
        else:
            # Single video processing
            print(f"\n🎬 Processing single video: {VIDEO_PATH}")
            
            # Check if video file exists
            if not os.path.exists(VIDEO_PATH):
                print(f"❌ Error: Video file not found: {VIDEO_PATH}")
                print("👉 Please edit the VIDEO_PATH in main.py to point to your 360° video")
                return
            
            results = process_single_video(
                VIDEO_PATH, 
                activity_segmenter, 
                OUTPUT_DIRECTORY,
                SAVE_ANNOTATED_VIDEO,
                SAVE_MOTION_DATA
            )
            display_results_summary(results)
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📂 Results saved to: {os.path.abspath(OUTPUT_DIRECTORY)}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        logger.info("Analysis interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        logger.error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def configure_sensitivity(sensitivity: str):
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
    print(f"[DEBUG] Configuration: {config}")
    
    # Note: In a production system, you'd update the global config here
    logger = get_logger(__name__)
    logger.info(f"Motion sensitivity configured to: {sensitivity}")

def process_single_video(video_path: str, segmenter: ActivitySegmenter, 
                        output_dir: str, save_annotated: bool, 
                        save_motion_data: bool) -> object:
    """
    Process a single video file.
    
    Args:
        video_path: Path to the video file
        segmenter: ActivitySegmenter instance
        output_dir: Output directory
        save_annotated: Whether to save annotated video
        save_motion_data: Whether to save motion data
        
    Returns:
        SegmentationResults object
    """
    print(f"🔍 Analyzing video properties...")
    
    # Validate video file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video info
    video_processor = Video360Processor()
    video_info = video_processor.get_video_info(video_path)
    
    print(f"📹 Video Info:")
    print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    print(f"   Duration: {video_info['duration_minutes']:.1f} minutes")
    print(f"   FPS: {video_info['fps']:.1f}")
    print(f"   Size: {video_info['file_size_mb']:.1f} MB")
    print(f"   360° Format: {'✅ Yes' if video_info['is_360_likely'] else '❓ Unknown'}")
    
    # Estimate processing time
    time_estimate = video_processor.estimate_processing_time(video_info)
    print(f"⏱️  Estimated processing time: {time_estimate['estimated_minutes']:.1f} minutes")
    
    if time_estimate['estimated_minutes'] > 30:
        print("⚠️  Long processing time expected - consider running overnight")
    
    # Start analysis
    print(f"\n🔄 Starting motion analysis...")
    start_time = time.time()
    
    results = segmenter.analyze_video(
        video_path=video_path,
        save_annotated=save_annotated,
        save_motion_data=save_motion_data,
        output_dir=output_dir
    )
    
    processing_time = time.time() - start_time
    print(f"✅ Analysis completed in {processing_time:.1f} seconds")
    
    return results

def process_video_directory(directory_path: str, segmenter: ActivitySegmenter,
                           output_dir: str, save_annotated: bool,
                           save_motion_data: bool) -> List:
    """
    Process all videos in a directory.
    
    Args:
        directory_path: Directory containing videos
        segmenter: ActivitySegmenter instance
        output_dir: Output directory
        save_annotated: Whether to save annotated videos
        save_motion_data: Whether to save motion data
        
    Returns:
        List of processing results
    """
    print(f"📁 Scanning directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    video_processor = Video360Processor()
    video_files = video_processor.get_video_files_in_directory(directory_path)
    
    if not video_files:
        print(f"❌ No video files found in directory: {directory_path}")
        return []
    
    print(f"📁 Found {len(video_files)} video files to process")
    
    batch_results = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*40}")
        print(f"📹 Processing video {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"{'='*40}")
        
        try:
            # Create unique output directory for each video
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(output_dir, f"video_{i:03d}_{video_name}")
            
            print(f"[DEBUG] Processing video: {video_path}")
            print(f"[DEBUG] Output directory: {video_output_dir}")
            
            results = process_single_video(
                video_path, segmenter, video_output_dir, 
                save_annotated, save_motion_data
            )
            
            batch_results.append({
                "video_path": video_path,
                "video_name": os.path.basename(video_path),
                "results": results,
                "success": True,
                "error": None
            })
            
            print(f"✅ Successfully processed: {os.path.basename(video_path)}")
            
        except Exception as e:
            error_msg = f"Failed to process {video_path}: {str(e)}"
            print(f"❌ {error_msg}")
            logger = get_logger(__name__)
            logger.error(error_msg)
            
            batch_results.append({
                "video_path": video_path,
                "video_name": os.path.basename(video_path),
                "results": None,
                "success": False,
                "error": error_msg
            })
    
    print(f"\n📊 Batch processing completed: {len([r for r in batch_results if r['success']])}/{len(batch_results)} successful")
    return batch_results

def display_results_summary(results):
    """Display a summary of analysis results."""
    print(f"\n📊 ANALYSIS RESULTS SUMMARY")
    print(f"{'='*40}")
    
    print(f"🎬 Video: {os.path.basename(results.video_path)}")
    print(f"⏱️  Duration: {results.duration/60:.1f} minutes")
    print(f"🎯 Segments Found: {results.segments}")
    print(f"🚀 Processing Time: {results.processing_time:.1f} seconds")
    
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
        top_segments = sorted(results.segments, key=lambda x: x.duration, reverse=True)[:5]
        
        for i, segment in enumerate(top_segments, 1):
            start_min = segment.start_time / 60
            end_min = segment.end_time / 60
            duration_sec = segment.duration
            
            print(f"   {i}. {start_min:.1f}m-{end_min:.1f}m ({duration_sec:.0f}s)")
            print(f"      Type: {segment.activity_type}")
            print(f"      Motion: {segment.avg_motion_energy:.3f} | Confidence: {segment.confidence:.3f}")
            print(f"      Description: {segment.description}")
            print()
        
        # Activity recommendations
        activity_summary = segment.get_activity_summary(results)
        if "recommendations" in activity_summary:
            print(f"💡 Recommendations:")
            for rec in activity_summary["recommendations"]:
                print(f"   • {rec}")
    else:
        print("\n⚠️  No significant activity segments detected")
        print("   💡 Try adjusting sensitivity or check video content:")
        print("   • Set SENSITIVITY_LEVEL = 'high' for more sensitive detection")
        print("   • Ensure video contains actual human activities")
        print("   • Check if video has sufficient motion and isn't mostly static")

def display_batch_summary(batch_results: List):
    """Display summary of batch processing results."""
    print(f"\n📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*50}")
    
    successful = [r for r in batch_results if r["success"]]
    failed = [r for r in batch_results if not r["success"]]
    
    print(f"✅ Successfully processed: {len(successful)}/{len(batch_results)} videos")
    
    if failed:
        print(f"❌ Failed: {len(failed)} videos")
        for fail in failed[:5]:  # Show first 5 failures
            print(f"   - {fail['video_name']}: {fail['error']}")
        if len(failed) > 5:
            print(f"   ... and {len(failed) - 5} more failures")
    
    if successful:
        print(f"\n📈 Aggregate Statistics:")
        total_duration = sum(r["results"].duration for r in successful)
        total_segments = sum(r["results"].total_segments for r in successful)
        total_processing_time = sum(r["results"].processing_time for r in successful)
        
        print(f"   Total Video Duration: {total_duration/60:.1f} minutes")
        print(f"   Total Segments Found: {total_segments}")
        print(f"   Total Processing Time: {total_processing_time/60:.1f} minutes")
        print(f"   Average Segments per Video: {total_segments/len(successful):.1f}")
        print(f"   Processing Speed: {total_duration/total_processing_time:.1f}x real-time")
        
        # Show best performing videos
        print(f"\n🏆 Top Videos by Activity Segments:")
        successful_sorted = sorted(successful, key=lambda x: x["results"].total_segments, reverse=True)
        for i, result in enumerate(successful_sorted[:3], 1):
            video_name = result["video_name"]
            segments_count = result["results"].total_segments
            duration_min = result["results"].total_duration / 60
            print(f"   {i}. {video_name}: {segments_count} segments ({duration_min:.1f}m)")

def configure_sensitivity(sensitivity: str):
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
    print(f"[DEBUG] Configuration: {config}")
    
    # Note: In a production system, you'd update the global config here
    logger = get_logger(__name__)
    logger.info(f"Motion sensitivity configured to: {sensitivity}")

def process_single_video(video_path: str, segmenter: ActivitySegmenter, 
                        output_dir: str, save_annotated: bool, 
                        save_motion_data: bool) -> object:
    """
    Process a single video file.
    
    Args:
        video_path: Path to the video file
        segmenter: ActivitySegmenter instance
        output_dir: Output directory
        save_annotated: Whether to save annotated video
        save_motion_data: Whether to save motion data
        
    Returns:
        SegmentationResults object
    """
    print(f"🔍 Analyzing video properties...")
    
    # Validate video file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video info
    video_processor = Video360Processor()
    video_info = video_processor.get_video_info(video_path)
    
    print(f"📹 Video Info:")
    print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    print(f"   Duration: {video_info['duration_minutes']:.1f} minutes")
    print(f"   FPS: {video_info['fps']:.1f}")
    print(f"   Size: {video_info['file_size_mb']:.1f} MB")
    print(f"   360° Format: {'✅ Yes' if video_info['is_360_likely'] else '❓ Unknown'}")
    
    # Estimate processing time
    time_estimate = video_processor.estimate_processing_time(video_info)
    print(f"⏱️  Estimated processing time: {time_estimate['estimated_minutes']:.1f} minutes")
    
    if time_estimate['estimated_minutes'] > 30:
        print("⚠️  Long processing time expected - consider running overnight")
    
    # Start analysis
    print(f"\n🔄 Starting motion analysis...")
    start_time = time.time()
    
    results = segmenter.analyze_video(
        video_path=video_path,
        save_annotated=save_annotated,
        save_motion_data=save_motion_data,
        output_dir=output_dir
    )
    
    processing_time = time.time() - start_time
    print(f"✅ Analysis completed in {processing_time:.1f} seconds")
    
    return results

def process_video_directory(directory_path: str, segmenter: ActivitySegmenter,
                           output_dir: str, save_annotated: bool,
                           save_motion_data: bool) -> List:
    """
    Process all videos in a directory.
    
    Args:
        directory_path: Directory containing videos
        segmenter: ActivitySegmenter instance
        output_dir: Output directory
        save_annotated: Whether to save annotated videos
        save_motion_data: Whether to save motion data
        
    Returns:
        List of processing results
    """
    print(f"📁 Scanning directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    video_processor = Video360Processor()
    video_files = video_processor.get_video_files_in_directory(directory_path)
    
    if not video_files:
        print(f"❌ No video files found in directory: {directory_path}")
        return []
    
    print(f"📁 Found {len(video_files)} video files to process")
    
    batch_results = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*40}")
        print(f"📹 Processing video {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"{'='*40}")
        
        try:
            # Create unique output directory for each video
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(output_dir, f"video_{i:03d}_{video_name}")
            
            print(f"[DEBUG] Processing video: {video_path}")
            print(f"[DEBUG] Output directory: {video_output_dir}")
            
            results = process_single_video(
                video_path, segmenter, video_output_dir, 
                save_annotated, save_motion_data
            )
            
            batch_results.append({
                "video_path": video_path,
                "video_name": os.path.basename(video_path),
                "results": results,
                "success": True,
                "error": None
            })
            
            print(f"✅ Successfully processed: {os.path.basename(video_path)}")
            
        except Exception as e:
            error_msg = f"Failed to process {video_path}: {str(e)}"
            print(f"❌ {error_msg}")
            logger = get_logger(__name__)
            logger.error(error_msg)
            
            batch_results.append({
                "video_path": video_path,
                "video_name": os.path.basename(video_path),
                "results": None,
                "success": False,
                "error": error_msg
            })
    
    print(f"\n📊 Batch processing completed: {len([r for r in batch_results if r['success']])}/{len(batch_results)} successful")
    return batch_results

def test_installation():
    """Test if all required components are properly installed."""
    print("🔧 Testing installation...")
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed: pip install opencv-python")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed: pip install numpy")
        return False
    
    try:
        import scipy
        print(f"✅ SciPy version: {scipy.__version__}")
    except ImportError:
        print("❌ SciPy not installed: pip install scipy")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not installed: pip install scikit-learn")
        return False
    
    try:
        import subprocess
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

def quick_demo():
    """Run a quick demo if no video is specified."""
    print("\n🎬 DEMO MODE")
    print("=" * 40)
    print("To analyze your 360° video:")
    print("1. Edit main.py:")
    print("   VIDEO_PATH = 'path/to/your_video.mp4'")
    print("2. Run: python main.py")
    print("\nExample video characteristics:")
    print("   ✅ Format: .mp4, .insv")
    print("   ✅ Type: 360° equirectangular")
    print("   ✅ Content: Home activities (cooking, cleaning)")
    print("   ✅ Duration: 30 seconds to 2+ hours")
    print("\nSample configuration:")
    print("   VIDEO_PATH = 'videos/cooking_session.mp4'")
    print("   OUTPUT_DIRECTORY = 'results'")
    print("   SENSITIVITY_LEVEL = 'medium'")

def validate_configuration():
    """Validate the configuration before starting analysis."""
    print("🔍 Validating configuration...")
    
    # Check if paths exist
    issues = []
    
    # Create output directory if it doesn't exist
    OUTPUT_DIRECTORY = "output"  # This should match main() function
    try:
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        print(f"✅ Output directory ready: {OUTPUT_DIRECTORY}")
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

if __name__ == "__main__":
    print("[DEBUG] Starting main execution")
    
    # Quick installation test on startup
    if not test_installation():
        print("\n❌ Installation incomplete. Please install missing components.")
        print("💡 Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Validate configuration
    if not validate_configuration():
        print("\n❌ Configuration issues found. Please resolve them first.")
        sys.exit(1)
    
    # Check if video path is set in main() function - simplified approach
    print("\n🎯 Starting 360° motion analysis pipeline...")
    
    # Run main analysis
    main()