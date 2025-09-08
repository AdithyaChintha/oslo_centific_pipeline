import librosa
import numpy as np
from scipy.signal import find_peaks
import subprocess
import os
import tempfile
from pathlib import Path
import time
import logging
import ray
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_clap_detector")

class UnifiedClapDetector:
    """
    Unified clap detector that handles both audio and video files.
    Uses advanced spectral analysis to detect first and last claps only.
    """
    
    def __init__(self, search_window_sec=30.0):
        self.search_window_sec = search_window_sec
        self.temp_files = []
    
    def detect_claps_in_file(self, file_path):
        """
        Main method to detect first and last claps in audio or video files.
        
        Args:
            file_path (str): Path to audio or video file
            
        Returns:
            dict: Results containing first and last clap timestamps
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = file_path.suffix.lower()
            
            # Determine file type
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            audio_extensions = ['.wav', '.mp3', '.aac', '.flac', '.ogg', '.wma']
            
            print(f"\n{'='*60}")
            print(f"Processing: {file_path.name}")
            print(f"File type: {file_extension}")
            
            if file_extension in video_extensions:
                return self._process_video_file(str(file_path))
            elif file_extension in audio_extensions:
                return self._process_audio_file(str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return {
                "file_path": str(file_path),
                "file_type": "unknown",
                "success": False,
                "error": str(e),
                "clap_timestamp": None
            }
    
    def _process_video_file(self, video_path):
        """Extract audio from video and detect claps."""
        try:
            print("Checking video file for audio streams...")
            
            # First, check if the video has any audio streams
            probe_cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', video_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if probe_result.returncode == 0 and probe_result.stdout.strip():
                print(f"Found audio stream(s) in video: {probe_result.stdout.strip()}")
            else:
                print("No audio streams found in video file")
                return {
                    "file_path": video_path,
                    "file_type": "video",
                    "success": False,
                    "error": "Video file contains no audio streams - cannot detect claps from video-only file",
                    "start_clap": None,
                    "end_clap": None
                }
            
            print("Extracting audio from video...")
            
            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            self.temp_files.append(temp_audio_path)
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '44100',  # Sample rate
                '-ac', '1',  # Mono
                temp_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            print("Audio extracted successfully")
            
            # Process the extracted audio
            detection_result = self._process_audio_file(temp_audio_path)
            
            # Update the result to show original video path
            detection_result["file_path"] = video_path
            detection_result["file_type"] = "video"
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Error processing video file: {e}")
            return {
                "file_path": video_path,
                "file_type": "video",
                "success": False,
                "error": str(e),
                "clap_timestamp": None
            }
    
    def _process_audio_file(self, audio_path):
        """Process audio file to detect first and last claps."""
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            print(f"Audio duration: {duration:.2f} seconds")
            print(f"Sample rate: {sr} Hz")
            print(f"Searching first {self.search_window_sec}s and last {self.search_window_sec}s")

            # Calculate onset strength with higher temporal resolution
            hop_length = 256  # Smaller hop for better temporal resolution
            onset_env = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=hop_length, 
                aggregate=np.median  # Use median instead of mean for robustness
            )
            
            # Calculate spectral features that are characteristic of claps
            # Claps have:
            # - High spectral centroid (bright sound)
            # - High spectral rolloff (energy concentrated in high frequencies)
            # - High zero crossing rate (noisy characteristic)
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(
                y=y, sr=sr, hop_length=hop_length
            )[0]
            
            # Spectral rolloff (85% of energy threshold)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=sr, hop_length=hop_length, roll_percent=0.85
            )[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                y, hop_length=hop_length
            )[0]
            
            # Create a composite clap detection score
            # Normalize features to 0-1 range
            onset_norm = (onset_env - np.min(onset_env)) / (np.max(onset_env) - np.min(onset_env) + 1e-8)
            centroid_norm = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-8)
            rolloff_norm = (spectral_rolloff - np.min(spectral_rolloff)) / (np.max(spectral_rolloff) - np.min(spectral_rolloff) + 1e-8)
            zcr_norm = (zcr - np.min(zcr)) / (np.max(zcr) - np.min(zcr) + 1e-8)
            
            # Weighted combination - emphasizing onset strength and high frequency content
            clap_score = (
                3.0 * onset_norm +      # Strong onset is most important
                2.0 * centroid_norm +   # High frequency content
                1.5 * rolloff_norm +    # Energy in high frequencies  
                1.0 * zcr_norm          # Noisiness
            ) / 7.5  # Normalize by total weights
            
            # Find peaks in the clap score with stricter criteria
            # Calculate dynamic threshold based on audio characteristics
            clap_mean = np.mean(clap_score)
            clap_std = np.std(clap_score)
            
            # Adaptive threshold: mean + 2.5 * std (catches top ~1% of peaks)
            min_height = clap_mean + 2.5 * clap_std
            
            # Minimum distance between peaks (0.3 seconds to avoid double detection)
            min_distance = int(sr * 0.3 / hop_length)
            
            print(f"Clap score - Mean: {clap_mean:.3f}, Std: {clap_std:.3f}")
            print(f"Using threshold: {min_height:.3f}")
            
            peaks, properties = find_peaks(
                clap_score,
                height=min_height,
                distance=min_distance,
                prominence=clap_std * 0.5  # Peak must be prominent
            )
            
            # Convert peak frame indices to time
            times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
            peak_times = times[peaks]
            peak_scores = clap_score[peaks]
            
            print(f"Found {len(peak_times)} potential clap candidates")
            
            if len(peak_times) == 0:
                print("No significant clap-like peaks detected.")
                return {
                    "file_path": audio_path,
                    "file_type": "audio",
                    "success": True,
                    "duration_seconds": duration,
                    "sample_rate": sr,
                    "candidates_found": 0,
                    "clap_timestamp": None
                }

            # Find the single best clap candidate (highest score)
            best_clap_idx = np.argmax(peak_scores)
            clap_timestamp = peak_times[best_clap_idx]
            clap_confidence = peak_scores[best_clap_idx]
            
            print(f"Found {len(peak_times)} clap candidates")
            print(f"Selected best clap: {clap_timestamp:.2f}s (confidence: {clap_confidence:.3f})")
            
            return {
                "file_path": audio_path,
                "file_type": "audio",
                "success": True,
                "duration_seconds": round(duration, 2),
                "sample_rate": sr,
                "candidates_found": len(peak_times),
                "clap_timestamp": round(clap_timestamp, 2),
                "clap_confidence": round(clap_confidence, 3),
                "detection_parameters": {
                    "threshold": round(min_height, 3),
                    "clap_score_mean": round(clap_mean, 3),
                    "clap_score_std": round(clap_std, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {
                "file_path": audio_path,
                "file_type": "audio",
                "success": False,
                "error": str(e),
                "clap_timestamp": None
            }
    
    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Cleaned up temporary file: {Path(temp_file).name}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {e}")
    
    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()


@ray.remote(max_calls=1)
def detect_claps_in_media(media_path: str, 
                         output_dir: str = "/tmp/clap_detection",
                         search_window_sec: float = 30.0) -> dict:
    """
    Detects first and last claps in a video or audio file and returns timestamps in JSON format.

    Args:
        media_path (str): Path to video or audio file.
        output_dir (str): Directory where output JSON will be stored.
        search_window_sec (float): Search window in seconds for first/last clap detection.

    Returns:
        dict: Detection results with first and last clap timestamps and metadata.
    """
    media_path = Path(media_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = media_path.stem
    output_json = Path(output_dir) / f"{base_name}_clap_detection.json"

    try:
        start_time = time.time()

        # Check if media file exists
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        logger.info(f"[START] Processing {media_path.name} for first/last clap detection")

        # Initialize unified clap detector
        detector = UnifiedClapDetector(search_window_sec=search_window_sec)
        
        # Detect first and last claps
        result = detector.detect_claps_in_file(str(media_path))
        
        # Add processing metadata
        end_time = time.time()
        processing_duration = round(end_time - start_time, 2)
        
        result.update({
            "processing_time_seconds": processing_duration,
            "output_file": str(output_json),
            "processed_at": datetime.now().isoformat()
        })

        # Save JSON output
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)

        # Cleanup
        detector.cleanup()

        clap_info = f"Clap: {result.get('clap_timestamp', 'None')}s (confidence: {result.get('clap_confidence', 'N/A')})"
        logger.info(f"[SUCCESS] {media_path.name} processed in {processing_duration}s. {clap_info}")

        return result

    except Exception as e:
        logger.exception(f"[EXCEPTION] Error processing {media_path.name}: {e}")
        return {
            "file_path": str(media_path),
            "error": f"Processing error: {str(e)}",
            "success": False
        }


@ray.remote(max_calls=1)
def detect_claps_in_audio_video_pair(audio_path: str, video_path: str,
                                     output_dir: str = "/tmp/clap_detection",
                                     search_window_sec: float = 30.0) -> dict:
    """
    Detects first and last claps in both audio and video files from the same source
    and combines the results in a single JSON file.
    
    Args:
        audio_path (str): Path to audio shard file
        video_path (str): Path to video shard file  
        output_dir (str): Directory where output JSON will be stored
        search_window_sec (float): Search window in seconds for first/last clap detection
        
    Returns:
        dict: Combined detection results from both audio and video processing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the audio file name as the base for output
    base_name = Path(audio_path).stem
    output_json = Path(output_dir) / f"{base_name}_combined_clap_detection.json"
    
    try:
        start_time = time.time()
        
        logger.info(f"[START] Processing audio-video pair for clap detection")
        logger.info(f"Audio: {Path(audio_path).name}")
        logger.info(f"Video: {Path(video_path).name}")
        
        # Initialize detector
        detector = UnifiedClapDetector(search_window_sec=search_window_sec)
        
        # Process both files
        results = {
            "processing_metadata": {
                "audio_file": str(audio_path),
                "video_file": str(video_path),
                "search_window_seconds": search_window_sec,
                "processed_at": datetime.now().isoformat()
            },
            "audio_results": None,
            "video_results": None,
            "combined_analysis": {}
        }
        
        # Process audio file
        if os.path.exists(audio_path):
            logger.info(f"Processing audio file: {Path(audio_path).name}")
            audio_result = detector.detect_claps_in_file(audio_path)
            results["audio_results"] = audio_result
        else:
            logger.warning(f"Audio file not found: {audio_path}")
            results["audio_results"] = {
                "file_path": str(audio_path),
                "success": False,
                "error": "Audio file not found"
            }
        
        # Process video file  
        if os.path.exists(video_path):
            logger.info(f"Processing video file: {Path(video_path).name}")
            video_result = detector.detect_claps_in_file(video_path)
            results["video_results"] = video_result
        else:
            logger.warning(f"Video file not found: {video_path}")
            results["video_results"] = {
                "file_path": str(video_path),
                "success": False,
                "error": "Video file not found"
            }
        
        # Combine analysis
        audio_success = results["audio_results"].get("success", False)
        video_success = results["video_results"].get("success", False)
        
        if audio_success or video_success:
            # Get clap timestamps from both sources
            audio_clap = results["audio_results"].get("clap_timestamp") if audio_success else None
            audio_confidence = results["audio_results"].get("clap_confidence") if audio_success else None
            video_clap = results["video_results"].get("clap_timestamp") if video_success else None
            video_confidence = results["video_results"].get("clap_confidence") if video_success else None
            
            # Choose the clap with higher confidence
            if audio_clap is not None and video_clap is not None:
                if audio_confidence >= video_confidence:
                    selected_clap = audio_clap
                    selected_confidence = audio_confidence
                    selected_source = "audio"
                else:
                    selected_clap = video_clap
                    selected_confidence = video_confidence
                    selected_source = "video"
            elif audio_clap is not None:
                selected_clap = audio_clap
                selected_confidence = audio_confidence
                selected_source = "audio"
            else:
                selected_clap = video_clap
                selected_confidence = video_confidence
                selected_source = "video"
            
            results["combined_analysis"] = {
                "overall_success": True,
                "sources_processed": {
                    "audio": audio_success,
                    "video": video_success
                },
                "detected_clap": {
                    "timestamp": selected_clap,
                    "confidence": selected_confidence,
                    "source": selected_source
                },
                "audio_clap": {
                    "timestamp": audio_clap,
                    "confidence": audio_confidence
                },
                "video_clap": {
                    "timestamp": video_clap,
                    "confidence": video_confidence
                }
            }
        else:
            results["combined_analysis"] = {
                "overall_success": False,
                "error": "Both audio and video processing failed"
            }
        
        # Add processing time
        end_time = time.time()
        processing_duration = round(end_time - start_time, 2)
        results["processing_metadata"]["processing_time_seconds"] = processing_duration
        results["processing_metadata"]["output_file"] = str(output_json)
        
        # Save combined JSON output
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Cleanup
        detector.cleanup()
        
        # Log summary
        if results["combined_analysis"].get("overall_success", False):
            detected = results["combined_analysis"]["detected_clap"]
            clap_info = f"Clap: {detected.get('timestamp', 'None')}s (confidence: {detected.get('confidence', 'N/A')}, source: {detected.get('source', 'N/A')})"
            logger.info(f"[SUCCESS] Audio-video pair processed in {processing_duration}s. {clap_info}")
        else:
            logger.error(f"[FAILED] Audio-video pair processing failed in {processing_duration}s")
            
        return results
        
    except Exception as e:
        logger.exception(f"[EXCEPTION] Error processing audio-video pair: {e}")
        return {
            "processing_metadata": {
                "audio_file": str(audio_path),
                "video_file": str(video_path),
                "processed_at": datetime.now().isoformat(),
                "error": str(e)
            },
            "audio_results": None,
            "video_results": None,
            "combined_analysis": {
                "overall_success": False,
                "error": f"Processing error: {str(e)}"
            }
        }


# Helper functions for processing multiple files
def process_multiple_files(file_paths, output_dir="/tmp/clap_detection", search_window_sec=30.0):
    """
    Process multiple audio/video files and detect first/last claps in each using Ray.
    
    Args:
        file_paths (list): List of file paths to process
        output_dir (str): Directory where output JSON files will be stored
        search_window_sec (float): Search window in seconds
        
    Returns:
        list: List of detection results for each file
    """
    results = []
    
    try:
        print(f"Processing {len(file_paths)} files...")
        start_time = time.time()
        
        # Submit jobs to Ray
        futures = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                future = detect_claps_in_media.remote(
                    media_path=file_path,
                    output_dir=output_dir,
                    search_window_sec=search_window_sec
                )
                futures.append(future)
                print(f"Submitted: {Path(file_path).name}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        # Get results
        if futures:
            results = ray.get(futures)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print summary
        print_summary(results, total_time)
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
    
    return results


def print_file_results(result):
    """Print results for a single file."""
    print(f"\n{'='*40} RESULTS {'='*40}")
    print(f"File: {Path(result['file_path']).name}")
    print(f"Type: {result.get('file_type', 'unknown')}")
    print(f"Success: {'✅' if result.get('success', False) else '❌'}")
    
    if result.get('success', False):
        if result.get('duration_seconds'):
            print(f"Duration: {result['duration_seconds']} seconds")
        if result.get('candidates_found'):
            print(f"Candidates found: {result['candidates_found']}")
            
        clap_timestamp = result.get('clap_timestamp')
        clap_confidence = result.get('clap_confidence')
        
        if clap_timestamp is not None:
            print(f"🎯 CLAP DETECTED: {clap_timestamp} seconds (confidence: {clap_confidence})")
        else:
            print(f"❌ No clap found in audio")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")


def print_summary(results, total_time):
    """Print summary of all results."""
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\n{'='*20} BATCH SUMMARY {'='*20}")
    print(f"Total files processed: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️ Total processing time: {total_time:.2f} seconds")
    
    if successful:
        print(f"\n📊 SUCCESSFUL DETECTIONS:")
        for result in successful:
            file_name = Path(result['file_path']).name
            clap_timestamp = result.get('clap_timestamp')
            clap_confidence = result.get('clap_confidence')
            
            clap_str = f"{clap_timestamp}s (conf: {clap_confidence})" if clap_timestamp is not None else "None"
            
            print(f"  📁 {file_name}: Clap={clap_str}")
    
    if failed:
        print(f"\n❌ FAILED FILES:")
        for result in failed:
            file_name = Path(result['file_path']).name
            error = result.get('error', 'Unknown error')
            print(f"  📁 {file_name}: {error}")


# Example usage
if __name__ == "__main__":
    # Initialize Ray
    ray.init()
    
    # Define your audio and video files separately for clarity
    audio_files = [
        "/home/nvcoe_admin/code/oslo/whole_pipeline_testing/insta360-video-activity-segmentation/blob_mount/azure_directory_path/Geo_files_test/skincare-20250819_1359-audio.WAV"
    ]
    
    video_files = [
        "/home/nvcoe_admin/insta360-video-activity-segmentation/files/skincare-20250819_1359.mp4"
    ]
    
    # Combine all files for processing
    all_media_files = audio_files + video_files
    
    # Filter to only existing files
    existing_files = []
    missing_files = []
    
    print("Checking file existence...")
    for f in all_media_files:
        if os.path.exists(f):
            existing_files.append(f)
            print(f"Found: {Path(f).name}")
        else:
            missing_files.append(f)
            print(f"Missing: {f}")
    
    if missing_files:
        print(f"\n{len(missing_files)} files not found:")
        for f in missing_files:
            print(f"  {f}")
    
    if existing_files:
        print(f"\nProcessing {len(existing_files)} existing files...")
        
        # Process all files with Ray
        results = process_multiple_files(
            file_paths=existing_files,
            output_dir="/tmp/clap_results",
            search_window_sec=30.0  # Search first and last 30 seconds
        )
        
    else:
        print("No valid files found to process. Please check your file paths.")
    
    ray.shutdown()