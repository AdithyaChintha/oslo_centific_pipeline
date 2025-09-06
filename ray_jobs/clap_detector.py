import ray
import time
import os
import warnings
import logging
import json
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from scipy.io.wavfile import write as writeAudioFile, read as readAudioFile

# Video processing imports
try:
    import subprocess
    FFMPEG_AVAILABLE = True
    # Test if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        FFMPEG_AVAILABLE = False
except ImportError:
    FFMPEG_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logger = logging.getLogger("clap_detector")
logging.basicConfig(level=logging.INFO)

class ClapDetector:
    """
    ClapDetector Class - For Video/Audio File Input with All Clap Detection
    """
    def __init__(self, media_file_path=None, 
                 initialVolumeThreshold=7000,
                 rate=None,
                 bufferLength=2048,
                 debounceTimeFactor=0.15,
                 resetTime=0.35,
                 clapInterval=0.08,
                 secondsPerTimePeriod=10,
                 volumeAverageFactor=0.9,
                 audioBufferLength=3.1,
                 logLevel=logging.INFO):
        
        # Media file setup
        self.media_file_path = media_file_path
        self.file_audio_data = None
        self.file_position = 0
        self.file_sample_rate = None
        self.is_video_file = False
        self.temp_audio_file = None
        
        # All clap detection
        self.clap_timestamps = []
        
        # Parameters
        self.volumeThreshold = initialVolumeThreshold
        self.rate = rate
        self.bufferLength = bufferLength
        self.debounceTimeFactor = debounceTimeFactor
        self.resetTime = resetTime
        self.clapInterval = clapInterval
        self.secondsPerTimePeriod = secondsPerTimePeriod
        self.audioBufferLength = audioBufferLength
        self.volumeAverageFactor = volumeAverageFactor
        self.audioData = np.array([], dtype=np.int16)

        # Clap detection variables
        self._resetClapTimes()
        
        # Initialize logger
        self.logger = logger

    def loadMediaFile(self):
        """Load video or audio file and extract audio for processing."""
        try:
            file_extension = Path(self.media_file_path).suffix.lower()
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            audio_extensions = ['.wav', '.mp3', '.aac', '.flac', '.ogg', '.wma']
            
            if file_extension in video_extensions:
                self.is_video_file = True
                self._extractAudioFromVideo()
            elif file_extension in audio_extensions:
                self.is_video_file = False
                self._loadAudioFile()
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            self.initAudioParams()
            
        except Exception as e:
            self.logger.error(f"Failed to load media file {self.media_file_path}: {e}")
            raise

    def _extractAudioFromVideo(self):
        """Extract audio using ffmpeg command line tool."""
        try:
            self.logger.info(f"Extracting audio from video: {Path(self.media_file_path).name}")
            
            # Create temporary audio file with unique name
            self.temp_audio_file = f"temp_audio_{int(time.time())}_{os.getpid()}.wav"
            
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-i', str(self.media_file_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '44100',  # Sample rate
                '-ac', '1',  # Mono
                self.temp_audio_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            # Load the extracted audio
            self.file_sample_rate, self.file_audio_data = readAudioFile(self.temp_audio_file)
            
            # Handle stereo audio by converting to mono
            if len(self.file_audio_data.shape) > 1:
                self.file_audio_data = np.mean(self.file_audio_data, axis=1).astype(np.int16)
            
            if self.rate is None:
                self.rate = self.file_sample_rate
            
            self.logger.info(f"Audio extracted successfully. Duration: {len(self.file_audio_data)/self.rate:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to extract audio with ffmpeg: {e}")
            raise

    def _loadAudioFile(self):
        """Load audio file directly."""
        try:
            self.file_sample_rate, self.file_audio_data = readAudioFile(self.media_file_path)
            
            # Handle stereo audio by converting to mono
            if len(self.file_audio_data.shape) > 1:
                self.file_audio_data = np.mean(self.file_audio_data, axis=1).astype(np.int16)
            
            if self.rate is None:
                self.rate = self.file_sample_rate
            
            self.logger.info(f"Loaded audio file. Duration: {len(self.file_audio_data)/self.rate:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file: {e}")
            raise

    def initAudioParams(self):
        """Initialize audio processing parameters."""
        self.resetTimeSamples = int(self.resetTime * self.rate)
        self.clapIntervalSamples = int(self.clapInterval * self.rate)
        self.samplesPerTimePeriod = self.secondsPerTimePeriod * self.rate
        self.audioBuffer = deque(maxlen=int((self.rate*self.audioBufferLength)/self.bufferLength))
        self.currentSampleTime = 0 + int(self.debounceTimeFactor * self.rate)
        self.file_position = 0

    def _resetClapTimes(self):
        self.clapTimes = [0]

    def calculateTimeDifference(self, timeA, timeB) -> float:
        """Calculate the time difference between two timestamps."""
        timeDifference = timeA - timeB
        if timeB > timeA:
            timeDifference += self.samplesPerTimePeriod
        return timeDifference

    def convertToCircularTime(self, timestamp) -> float:
        """Convert the timestamp to a circular time scale."""
        return timestamp % self.samplesPerTimePeriod

    def bandpassFilter(self, data, lowcut, highcut, fs, order=5):
        """Apply a bandpass filter to the input data."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def updateDynamicThreshold(self, newValue) -> None:
        """Update the volume threshold using a running average."""
        self.volumeThreshold = (self.volumeAverageFactor * self.volumeThreshold) + \
                            ((1 - self.volumeAverageFactor) * newValue) * 0.5

    def isClap(self, currentSampleTime, thresholdBias=6000, lowcut=100, highcut=4000):
        """Detect the occurrence of a clap in the input audio data."""
        clapDetected = False

        # Apply bandpass filter
        filteredAudio = self.bandpassFilter(self.audioData, lowcut=lowcut, highcut=highcut, fs=self.rate)

        # Calculate dynamic threshold
        dynamicThreshold = np.max(np.abs(self.audioData[-self.rate:]))
        self.updateDynamicThreshold(dynamicThreshold)

        # Find peaks
        peaks, _ = find_peaks(filteredAudio, height=self.volumeThreshold + thresholdBias)

        # Check for clap detection with debounce
        if len(peaks) > 0 and (self.calculateTimeDifference(currentSampleTime, self.clapTimes[-1]) >= int(self.debounceTimeFactor * self.rate)):
            clapDetected = True
            self.clapTimes.append(currentSampleTime)
            
            # Record clap timestamp in seconds
            clap_timestamp = self.file_position / self.rate
            self.clap_timestamps.append(clap_timestamp)

        return clapDetected

    def getAudio(self) -> np.ndarray:
        """Get audio data from file."""
        try:
            if self.file_audio_data is None:
                raise Exception("No audio file loaded")
            
            if self.file_position >= len(self.file_audio_data):
                return np.array([], dtype=np.int16)
            
            # Get the next buffer of audio data
            end_position = min(self.file_position + self.bufferLength, len(self.file_audio_data))
            self.audioData = self.file_audio_data[self.file_position:end_position]
            self.file_position = end_position
            
            # Pad with zeros if buffer is smaller
            if len(self.audioData) < self.bufferLength:
                padding = np.zeros(self.bufferLength - len(self.audioData), dtype=np.int16)
                self.audioData = np.concatenate([self.audioData, padding])
                    
            self.audioBuffer.append(self.audioData)

        except Exception as e:
            self.logger.error(f"Error reading audio data: {e}")
            return np.array([], dtype=np.int16)

        return self.audioData

    def run(self, thresholdBias=6000, lowcut=100, highcut=4000) -> list:
        """Run the clap detection process."""
        self.getAudio()
        
        if len(self.audioData) == 0:
            return []

        self.currentSampleTime = self.convertToCircularTime(self.currentSampleTime + self.bufferLength)
        self.isClap(self.currentSampleTime, thresholdBias=thresholdBias, lowcut=lowcut, highcut=highcut)
        
        return []

    def detectAllClaps(self, thresholdBias=6000, lowcut=100, highcut=4000,
                       audio_threshold_bias=7900, audio_lowcut=780, audio_highcut=4000):
        """Process the entire media file and detect all claps."""
        if self.file_audio_data is None:
            raise Exception("No media file loaded")
        
        self.file_position = 0
        self.clap_timestamps = []
        
        total_duration = len(self.file_audio_data) / self.rate
        
        # Use specific parameters for audio files to reduce sensitivity
        if not self.is_video_file:
            tb = audio_threshold_bias
            lc = audio_lowcut
            hc = audio_highcut
        else:
            tb = thresholdBias
            lc = lowcut
            hc = highcut

        while self.file_position < len(self.file_audio_data):
            # Relax threshold for the first 0.1s to catch initial claps
            if self.file_position < int(0.1 * self.rate):
                current_tb = thresholdBias
            else:
                current_tb = tb
            self.run(thresholdBias=current_tb, lowcut=lc, highcut=hc)
        
        # Create result
        result = {
            "file_path": str(self.media_file_path),
            "file_type": "video" if self.is_video_file else "audio",
            "duration_seconds": round(total_duration, 3),
            "sample_rate": self.rate,
            "clap_count": len(self.clap_timestamps),
            "clap_timestamps": [
                {
                    "timestamp_seconds": round(timestamp, 3),
                    "timestamp_formatted": f"{int(timestamp//60):02d}:{timestamp%60:06.3f}"
                }
                for timestamp in self.clap_timestamps
            ],
            "detection_parameters": {
                "threshold_bias": thresholdBias,
                "frequency_range": {
                    "lowcut": lowcut,
                    "highcut": highcut
                },
                "debounce_time": self.debounceTimeFactor
            }
        }
        
        return result

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            try:
                os.remove(self.temp_audio_file)
                self.logger.debug("Cleaned up temporary audio file")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary file: {e}")


@ray.remote(max_calls=1)
def detect_claps_in_media(media_path: str, 
                         output_dir: str = "/tmp/clap_detection",
                         threshold_bias: int = 6000,
                         lowcut: int = 200,
                         highcut: int = 3200,
                         audio_threshold_bias: int = 7900,
                         audio_lowcut: int = 780,
                         audio_highcut: int = 4000) -> dict:
    """
    Detects all claps in a video or audio file and returns timestamps in JSON format.

    Args:
        media_path (str): Path to video or audio file.
        output_dir (str): Directory where output JSON will be stored.
        threshold_bias (int): Threshold bias for video clap detection.
        lowcut (int): Low frequency cutoff for video bandpass filter.
        highcut (int): High frequency cutoff for video bandpass filter.
        audio_threshold_bias (int): Threshold bias for audio clap detection.
        audio_lowcut (int): Low frequency cutoff for audio bandpass filter.
        audio_highcut (int): High frequency cutoff for audio bandpass filter.

    Returns:
        dict: Detection results with clap timestamps and metadata.
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

        # Check ffmpeg availability for video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        if media_path.suffix.lower() in video_extensions and not FFMPEG_AVAILABLE:
            raise RuntimeError("FFmpeg not available for video processing")

        logger.info(f"[START] Processing {media_path.name} for clap detection")

        # Initialize clap detector
        detector = ClapDetector(
            media_file_path=str(media_path),
            logLevel=logging.INFO
        )
        
        # Load media file
        detector.loadMediaFile()
        
        # Detect all claps
        result = detector.detectAllClaps(
            thresholdBias=threshold_bias,
            lowcut=lowcut,
            highcut=highcut,
            audio_threshold_bias=audio_threshold_bias,
            audio_lowcut=audio_lowcut,
            audio_highcut=audio_highcut
        )
        
        # Add processing metadata
        end_time = time.time()
        processing_duration = round(end_time - start_time, 2)
        
        result.update({
            "processing_time_seconds": processing_duration,
            "output_file": str(output_json),
            "processed_at": datetime.now().isoformat(),
            "success": True
        })

        # Save JSON output
        with open(output_json, 'w') as f:
            json.dump(result, f, indent=2)

        # Cleanup
        detector.cleanup()

        logger.info(f"[SUCCESS] {media_path.name} processed in {processing_duration}s. Found {result['clap_count']} claps")

        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] FFmpeg failed for {media_path.name}: {e}")
        return {
            "input": str(media_path),
            "error": f"FFmpeg failed: {str(e)}",
            "success": False
        }
    except Exception as e:
        logger.exception(f"[EXCEPTION] Error processing {media_path.name}: {e}")
        return {
            "input": str(media_path),
            "error": f"Processing error: {str(e)}",
            "success": False
        }


# Example usage
if __name__ == '__main__':
    # Initialize Ray
    ray.init()
    
    # Example: Process multiple media files
    media_files = [
        "test1.mp4",
        "test2.wav", 
        "test3.mp4"
    ]
    
    # Submit jobs to Ray
    futures = []
    for media_file in media_files:
        if os.path.exists(media_file):
            future = detect_claps_in_media.remote(
                media_path=media_file,
                output_dir="/tmp/clap_results",
                threshold_bias=6000,
                lowcut=200,
                highcut=3200,
                audio_threshold_bias=7900,  # Stricter for audio
                audio_lowcut=780,
                audio_highcut=4000
            )
            futures.append(future)
        else:
            logger.warning(f"File not found: {media_file}")
    
    # Get results
    if futures:
        results = ray.get(futures)
        
        # Print summary
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        print(f"\n🎉 BATCH PROCESSING COMPLETE!")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        for result in successful:
            print(f"📁 {Path(result['file_path']).name}: {result['clap_count']} claps in {result['processing_time_seconds']}s")
        
        for result in failed:
            print(f"❌ {Path(result['input']).name}: {result['error']}")
    else:
        print("No valid media files found to process")
    
    ray.shutdown()