"""
Ray-based Motion Energy Detection for 360° Home Activity Videos.
This module provides distributed processing of video chunks for motion analysis.
"""
import ray
import os
import sys
import time
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add motion_analysis to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MOTION_ANALYSIS_DIR = PROJECT_ROOT / "motion_analysis"
sys.path.append(str(MOTION_ANALYSIS_DIR))

try:
    from motion_analysis.src.main import MotionEnergyAnalysisPipeline
    from utils.logger import get_logger
except ImportError as e:
    print(f"Error importing motion analysis modules: {e}")
    print(f"Make sure motion_analysis/ directory exists with your code")
    raise

logger = get_logger("motion_energy_ray")

@ray.remote
def compute_motion_energy(
    chunk_paths: List[str], 
    config: Optional[Dict] = None,
    sensitivity_level: str = "medium",
    save_detailed_data: bool = False
) -> Dict[str, Any]:
    """
    Process multiple video chunks for motion energy detection using Ray.
    
    Args:
        chunk_paths: List of video chunk file paths to process
        config: Optional configuration dictionary
        sensitivity_level: Motion detection sensitivity ("low", "medium", "high")
        save_detailed_data: Whether to save detailed motion data
        
    Returns:
        Dictionary with aggregated motion analysis results
    """
    start_time = time.time()
    
    logger.info(f"Starting motion energy processing for {len(chunk_paths)} chunks")
    print(f"[Motion Energy Ray] Processing {len(chunk_paths)} video chunks...")
    
    # Default configuration
    default_config = {
        "sensitivity_level": sensitivity_level,
        "save_annotated_video": False,  # Don't save video in distributed mode
        "save_motion_data": save_detailed_data,
        "verbose_logging": False,  # Reduce log noise in distributed mode
        "output_directory": tempfile.mkdtemp(prefix="motion_ray_")
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    chunk_results = []
    aggregated_timeline = []
    aggregated_segments = []
    total_chunks_processed = 0
    total_chunks_failed = 0
    
    try:
        # Process each chunk
        for i, chunk_path in enumerate(chunk_paths):
            chunk_start_time = time.time()
            
            logger.info(f"Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
            print(f"[Motion Energy Ray] Processing chunk {i+1}/{len(chunk_paths)}")
            
            try:
                # Validate chunk exists
                if not os.path.exists(chunk_path):
                    logger.error(f"Chunk file not found: {chunk_path}")
                    total_chunks_failed += 1
                    continue
                
                # Create unique output directory for this chunk
                chunk_output_dir = os.path.join(
                    default_config["output_directory"], 
                    f"chunk_{i:03d}"
                )
                os.makedirs(chunk_output_dir, exist_ok=True)
                
                # Configure pipeline for this chunk
                chunk_config = default_config.copy()
                chunk_config.update({
                    "video_path": chunk_path,
                    "output_directory": chunk_output_dir
                })
                
                # Initialize and run pipeline
                pipeline = MotionEnergyAnalysisPipeline(chunk_config)
                results = pipeline.analyze_video(
                    video_path=chunk_path,
                    save_annotated=False,
                    save_motion_data=save_detailed_data,
                    output_dir=chunk_output_dir
                )
                
                chunk_processing_time = time.time() - chunk_start_time
                
                # Extract key metrics
                chunk_result = {
                    "chunk_index": i,
                    "chunk_path": chunk_path,
                    "duration_seconds": results.duration,
                    "segments_found": results.segments,
                    "processing_time_seconds": chunk_processing_time,
                    "motion_statistics": results.motion_statistics,
                    "segments_list": [
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
                    ]
                }
                
                chunk_results.append(chunk_result)
                
                # Aggregate timeline data
                aggregated_timeline.extend(results.motion_timeline)
                
                # Aggregate segments (adjust timestamps for chunk offset)
                chunk_offset = sum(r["duration_seconds"] for r in chunk_results[:-1])
                for seg in results.segments_list:
                    adjusted_segment = {
                        "start_time": seg.start_time + chunk_offset,
                        "end_time": seg.end_time + chunk_offset,
                        "duration": seg.duration,
                        "activity_type": seg.activity_type,
                        "avg_motion_energy": seg.avg_motion_energy,
                        "max_motion_energy": seg.max_motion_energy,
                        "confidence": seg.confidence,
                        "description": seg.description,
                        "source_chunk": i
                    }
                    aggregated_segments.append(adjusted_segment)
                
                total_chunks_processed += 1
                
                logger.info(f"Chunk {i+1} completed: {results.segments} segments, "
                           f"{chunk_processing_time:.1f}s processing time")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1} ({chunk_path}): {str(e)}")
                print(f"[Motion Energy Ray] ERROR in chunk {i+1}: {str(e)}")
                total_chunks_failed += 1
                continue
        
        # Calculate aggregated statistics
        total_duration = sum(r["duration_seconds"] for r in chunk_results)
        total_segments = sum(r["segments_found"] for r in chunk_results)
        total_processing_time = time.time() - start_time
        
        # Calculate overall motion statistics
        if aggregated_timeline:
            import numpy as np
            timeline_array = np.array(aggregated_timeline)
            
            overall_motion_stats = {
                "total_frames": len(aggregated_timeline),
                "avg_motion_energy": float(np.mean(timeline_array)),
                "max_motion_energy": float(np.max(timeline_array)),
                "min_motion_energy": float(np.min(timeline_array)),
                "motion_variance": float(np.var(timeline_array)),
                "motion_std_dev": float(np.std(timeline_array))
            }
            
            # Calculate activity level percentages
            HIGH_THRESHOLD = 0.02  # From your config
            MEDIUM_THRESHOLD = 0.005
            
            high_activity = np.sum(timeline_array > HIGH_THRESHOLD)
            medium_activity = np.sum((timeline_array > MEDIUM_THRESHOLD) & 
                                   (timeline_array <= HIGH_THRESHOLD))
            low_activity = len(timeline_array) - high_activity - medium_activity
            
            overall_motion_stats.update({
                "high_activity_frames": int(high_activity),
                "medium_activity_frames": int(medium_activity),
                "low_activity_frames": int(low_activity),
                "high_activity_percentage": float(high_activity / len(timeline_array) * 100),
                "medium_activity_percentage": float(medium_activity / len(timeline_array) * 100),
                "low_activity_percentage": float(low_activity / len(timeline_array) * 100)
            })
        else:
            overall_motion_stats = {}
        
        # Prepare final results
        final_results = {
            "success": True,
            "total_chunks_processed": total_chunks_processed,
            "total_chunks_failed": total_chunks_failed,
            "total_video_duration_seconds": total_duration,
            "total_motion_segments": total_segments,
            "total_processing_time_seconds": total_processing_time,
            "processing_speed_factor": total_duration / total_processing_time if total_processing_time > 0 else 0,
            "configuration": {
                "sensitivity_level": sensitivity_level,
                "save_detailed_data": save_detailed_data,
                "chunks_input": len(chunk_paths)
            },
            "overall_motion_statistics": overall_motion_stats,
            "aggregated_segments": aggregated_segments,
            "chunk_results": chunk_results,
            "motion_timeline": aggregated_timeline if save_detailed_data else [],
            "metadata": {
                "analysis_version": "1.0_ray_distributed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_node": os.getenv("HOSTNAME", "unknown"),
                "ray_task_id": ray.get_runtime_context().task_id.hex()
            }
        }
        
        logger.info(f"Motion energy processing completed successfully")
        logger.info(f"Processed {total_chunks_processed}/{len(chunk_paths)} chunks")
        logger.info(f"Total duration: {total_duration:.1f}s, Total segments: {total_segments}")
        logger.info(f"Processing time: {total_processing_time:.1f}s ({total_duration/total_processing_time:.1f}x real-time)")
        
        print(f"[Motion Energy Ray] ✅ SUCCESS: {total_chunks_processed}/{len(chunk_paths)} chunks processed")
        print(f"[Motion Energy Ray] Found {total_segments} motion segments in {total_duration:.1f}s video")
        print(f"[Motion Energy Ray] Processing speed: {total_duration/total_processing_time:.1f}x real-time")
        
        return final_results
        
    except Exception as e:
        error_msg = f"Critical error in motion energy processing: {str(e)}"
        logger.error(error_msg)
        print(f"[Motion Energy Ray] ❌ CRITICAL ERROR: {str(e)}")
        
        return {
            "success": False,
            "error": error_msg,
            "total_chunks_processed": total_chunks_processed,
            "total_chunks_failed": total_chunks_failed + (len(chunk_paths) - total_chunks_processed),
            "total_processing_time_seconds": time.time() - start_time,
            "configuration": {
                "sensitivity_level": sensitivity_level,
                "save_detailed_data": save_detailed_data,
                "chunks_input": len(chunk_paths)
            }
        }

@ray.remote
def process_single_video_for_motion_energy(
    video_path: str,
    config: Optional[Dict] = None,
    chunk_duration_seconds: float = 60.0
) -> Dict[str, Any]:
    """
    Process a complete video by splitting into chunks and analyzing motion energy.
    
    Args:
        video_path: Path to the input video file
        config: Optional configuration dictionary
        chunk_duration_seconds: Duration of each chunk in seconds
        
    Returns:
        Dictionary with complete motion analysis results
    """
    start_time = time.time()
    
    logger.info(f"Starting single video motion analysis: {video_path}")
    print(f"[Motion Energy Ray] Processing complete video: {os.path.basename(video_path)}")
    
    try:
        # Import video splitter
        from ray_jobs.video_splitter import split_video_into_shards
        
        # Split video into chunks
        logger.info(f"Splitting video into {chunk_duration_seconds}s chunks...")
        chunk_paths = ray.get(split_video_into_shards.remote(
            video_path, 
            shard_duration=chunk_duration_seconds
        ))
        
        logger.info(f"Video split into {len(chunk_paths)} chunks")
        print(f"[Motion Energy Ray] Video split into {len(chunk_paths)} chunks")
        
        # Process chunks for motion energy
        motion_config = config or {}
        motion_results = ray.get(process_video_chunks_for_motion_energy.remote(
            chunk_paths, 
            motion_config,
            sensitivity_level=motion_config.get("sensitivity_level", "medium"),
            save_detailed_data=motion_config.get("save_detailed_data", True)
        ))
        
        total_processing_time = time.time() - start_time
        
        # Add video-level metadata
        motion_results["video_path"] = video_path
        motion_results["total_pipeline_time_seconds"] = total_processing_time
        motion_results["chunk_duration_seconds"] = chunk_duration_seconds
        
        logger.info(f"Complete video processing finished in {total_processing_time:.1f}s")
        print(f"[Motion Energy Ray] ✅ Video processing complete: {total_processing_time:.1f}s")
        
        return motion_results
        
    except Exception as e:
        error_msg = f"Error in single video processing: {str(e)}"
        logger.error(error_msg)
        print(f"[Motion Energy Ray] ❌ Video processing failed: {str(e)}")
        
        return {
            "success": False,
            "error": error_msg,
            "video_path": video_path,
            "total_pipeline_time_seconds": time.time() - start_time
        }

# Convenience function for testing
def test_motion_energy_ray(video_path: str, sensitivity: str = "medium"):
    """
    Test function for motion energy detection with Ray.
    
    Args:
        video_path: Path to test video
        sensitivity: Detection sensitivity level
    """
    if not ray.is_initialized():
        ray.init()
    
    print(f"Testing motion energy detection on: {video_path}")
    
    config = {
        "sensitivity_level": sensitivity,
        "save_detailed_data": True,
        "output_directory": "/tmp/motion_ray_test"
    }
    
    result = ray.get(process_single_video_for_motion_energy.remote(
        video_path, config, chunk_duration_seconds=30.0
    ))
    
    print("\n" + "="*60)
    print("MOTION ENERGY RAY TEST RESULTS")
    print("="*60)
    
    if result["success"]:
        print(f"✅ Success: {result['total_motion_segments']} segments found")
        print(f"📊 Video duration: {result['total_video_duration_seconds']:.1f}s")
        print(f"⚡ Processing time: {result['total_pipeline_time_seconds']:.1f}s")
        print(f"🚀 Speed factor: {result.get('processing_speed_factor', 0):.1f}x real-time")
        
        if result.get('overall_motion_statistics'):
            stats = result['overall_motion_statistics']
            print(f"📈 High activity: {stats.get('high_activity_percentage', 0):.1f}%")
            print(f"📈 Medium activity: {stats.get('medium_activity_percentage', 0):.1f}%")
            print(f"📈 Low activity: {stats.get('low_activity_percentage', 0):.1f}%")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    return result

if __name__ == "__main__":
    # Test with sample video
    test_video = "/path/to/test/video.mp4"
    
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
    
    if os.path.exists(test_video):
        test_motion_energy_ray(test_video, sensitivity="medium")
    else:
        print(f"Test video not found: {test_video}")
        print("Usage: python motion_energy_detector.py [video_path]")