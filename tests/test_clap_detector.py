import ray
import pytest
import numpy as np
import tempfile
import os
import sys
import time
import json
from pathlib import Path
from scipy.io.wavfile import write as write_wav, read as read_wav
from unittest.mock import patch, MagicMock
from collections import deque

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import the Ray job and ClapDetector class
from ray_jobs.clap_detector import detect_claps_in_media, ClapDetector


class TestDataGenerator:
    """Helper class to generate synthetic test audio data"""
    
    @staticmethod
    def generate_clap_sound(sample_rate=44100, duration=0.1, amplitude=20000):
        """Generate a synthetic clap sound"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Create clap-like sound: sharp attack with quick decay
        envelope = np.exp(-t * 50)  # Quick decay
        # Mix of frequencies typical in claps (1-4kHz range)
        noise = np.random.normal(0, 1, len(t))
        filtered_noise = noise * envelope
        clap = (filtered_noise * amplitude).astype(np.int16)
        return clap
    
    @staticmethod
    def generate_silence(sample_rate=44100, duration=0.5):
        """Generate silence"""
        samples = int(sample_rate * duration)
        return np.zeros(samples, dtype=np.int16)
    
    @staticmethod
    def generate_background_noise(sample_rate=44100, duration=1.0, amplitude=1000):
        """Generate low-level background noise"""
        samples = int(sample_rate * duration)
        noise = np.random.normal(0, amplitude, samples)
        return noise.astype(np.int16)
    
    @staticmethod
    def generate_test_audio_with_claps(clap_times, sample_rate=44100, total_duration=5.0):
        """
        Generate test audio with claps at specified times
        
        Args:
            clap_times: List of times (in seconds) where claps should occur
            sample_rate: Audio sample rate
            total_duration: Total audio duration in seconds
        
        Returns:
            numpy array of audio data
        """
        total_samples = int(sample_rate * total_duration)
        audio = np.zeros(total_samples, dtype=np.int16)
        
        for clap_time in clap_times:
            clap_sample = int(clap_time * sample_rate)
            clap_sound = TestDataGenerator.generate_clap_sound(sample_rate)
            
            # Insert clap into audio (ensure we don't exceed bounds)
            end_sample = min(clap_sample + len(clap_sound), total_samples)
            clap_length = end_sample - clap_sample
            
            if clap_sample < total_samples and clap_length > 0:
                audio[clap_sample:end_sample] = clap_sound[:clap_length]
        
        return audio


# UNIT TESTS (from original test_clap_detector.py)
class TestClapDetector:
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_audio_file(self, temp_dir):
        """Create a temporary audio file with known claps"""
        # Create audio with claps at 1.0s, 2.5s, 4.0s
        clap_times = [1.0, 2.5, 4.0]
        audio_data = TestDataGenerator.generate_test_audio_with_claps(
            clap_times, sample_rate=44100, total_duration=5.0
        )
        
        # Save to temporary WAV file
        audio_file = Path(temp_dir) / "test_audio.wav"
        write_wav(str(audio_file), 44100, audio_data)
        
        return str(audio_file), clap_times
    
    @pytest.fixture
    def silent_audio_file(self, temp_dir):
        """Create a silent audio file"""
        silence = TestDataGenerator.generate_silence(duration=2.0)
        audio_file = Path(temp_dir) / "silent_audio.wav"
        write_wav(str(audio_file), 44100, silence)
        return str(audio_file)


    def test_clap_at_start_of_audio(self, detector, temp_dir):
        """Clap exactly at time 0 should be detected"""
        clap_times = [0.0, 1.5, 3.0]
        audio = TestDataGenerator.generate_test_audio_with_claps(clap_times, total_duration=4.0)
        audio_path = Path(temp_dir) / "clap_start.wav"
        write_wav(str(audio_path), 44100, audio)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps()
        assert result['clap_count'] >= 1
        assert any(abs(ts['timestamp_seconds'] - 0.0) < 0.05 for ts in result['clap_timestamps'])

    def test_clap_at_end_of_audio(self, detector, temp_dir):
        """Clap near the very end of audio should still be detected"""
        duration = 3.0
        clap_times = [duration - 0.05]
        audio = TestDataGenerator.generate_test_audio_with_claps(clap_times, total_duration=duration)
        audio_path = Path(temp_dir) / "clap_end.wav"
        write_wav(str(audio_path), 44100, audio)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps()
        assert result['clap_count'] >= 1

    def test_simultaneous_claps(self, detector, temp_dir):
        """Two claps very close together should not trigger false double-detect"""
        clap_times = [1.0, 1.02]  # 20 ms apart
        audio = TestDataGenerator.generate_test_audio_with_claps(clap_times, total_duration=2.0)
        audio_path = Path(temp_dir) / "clap_simultaneous.wav"
        write_wav(str(audio_path), 44100, audio)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps()
        # Depending on debounce, we expect <= 2 claps detected
        assert result['clap_count'] in (1, 2)

    def test_quiet_clap_below_threshold(self, detector, temp_dir):
        """Very quiet clap should not be detected"""
        clap = TestDataGenerator.generate_clap_sound(amplitude=1000)  # much lower
        audio = np.concatenate([clap, TestDataGenerator.generate_silence(duration=1.0)])
        audio_path = Path(temp_dir) / "quiet_clap.wav"
        write_wav(str(audio_path), 44100, audio)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps(thresholdBias=8000)  # stricter
        assert result['clap_count'] == 0

    def test_loud_clap_at_max_amplitude(self, detector, temp_dir):
        """Very loud clap at int16 max should still be detected"""
        clap = np.full(2000, 32000, dtype=np.int16)  # near clipping
        audio = np.concatenate([TestDataGenerator.generate_silence(44100, 0.5), clap])
        audio_path = Path(temp_dir) / "loud_clap.wav"
        write_wav(str(audio_path), 44100, audio)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps()
        assert result['clap_count'] >= 1

    def test_multiple_human_like_claps(self, detector, temp_dir):
        """Several claps spaced like human applause should all be caught"""
        clap_times = [1.0, 1.4, 1.9, 3.0]
        audio = TestDataGenerator.generate_test_audio_with_claps(clap_times, total_duration=4.0)
        audio_path = Path(temp_dir) / "human_applause.wav"
        write_wav(str(audio_path), 44100, audio)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps()
        assert result['clap_count'] >= len(clap_times) * 0.5  # allow slight misses

    def test_noise_without_claps(self, detector, temp_dir):
        """Background noise should not trigger claps"""
        noise = TestDataGenerator.generate_background_noise(amplitude=2000, duration=2.0)
        audio_path = Path(temp_dir) / "noise_only.wav"
        write_wav(str(audio_path), 44100, noise)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps()
        assert result['clap_count'] == 0

    def test_loud_noise_rejection(self, detector, temp_dir):
        """Loud, non-clap sounds should be rejected"""
        # Generate a loud, sustained tone that is not like a clap
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = (np.sin(2 * np.pi * 800 * t) * 15000).astype(np.int16)
        
        audio_path = Path(temp_dir) / "loud_tone.wav"
        write_wav(str(audio_path), 44100, tone)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps()
        assert result['clap_count'] == 0

    def test_long_audio_sparse_claps(self, detector, temp_dir):
        """Long file with only a few claps should be handled efficiently"""
        clap_times = [2.0, 15.0, 28.0]
        audio = TestDataGenerator.generate_test_audio_with_claps(clap_times, total_duration=30.0)
        audio_path = Path(temp_dir) / "long_sparse.wav"
        write_wav(str(audio_path), 44100, audio)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps()
        assert result['clap_count'] >= len(clap_times) * 0.5

    @pytest.fixture
    def detector(self):
        """Create a ClapDetector instance"""
        return ClapDetector()
    
    def test_init_detector(self, detector):
        """Test detector initialization"""
        assert detector.volumeThreshold == 7000
        assert detector.bufferLength == 2048
        assert detector.debounceTimeFactor == 0.15
        assert detector.clap_timestamps == []
        assert detector.file_audio_data is None
    
    def test_load_valid_audio_file(self, detector, sample_audio_file):
        """Test loading a valid audio file"""
        audio_path, expected_claps = sample_audio_file
        detector.media_file_path = audio_path
        
        # Load the file
        detector.loadMediaFile()
        
        # Verify file was loaded correctly
        assert detector.file_audio_data is not None
        assert detector.file_sample_rate == 44100
        assert detector.rate == 44100
        assert len(detector.file_audio_data) > 0
        assert not detector.is_video_file
    
    def test_load_nonexistent_file(self, detector):
        """Test loading a non-existent file raises error"""
        detector.media_file_path = "nonexistent_file.wav"
        
        with pytest.raises(Exception):
            detector.loadMediaFile()
    
    def test_detect_claps_in_known_audio(self, detector, sample_audio_file):
        """Test clap detection with known clap positions"""
        audio_path, expected_clap_times = sample_audio_file
        detector.media_file_path = audio_path
        
        # Load and process
        detector.loadMediaFile()
        result = detector.detectAllClaps()
        
        # Verify results structure
        assert isinstance(result, dict)
        assert 'clap_count' in result
        assert 'clap_timestamps' in result
        assert 'duration_seconds' in result
        assert result['clap_count'] >= 0
        assert len(result['clap_timestamps']) == result['clap_count']
        assert result['duration_seconds'] > 0
    
    def test_detect_claps_in_silent_audio(self, detector, silent_audio_file):
        """Test that no claps are detected in silent audio"""
        detector.media_file_path = silent_audio_file
        
        detector.loadMediaFile()
        result = detector.detectAllClaps()
        
        # Should detect no claps in silence
        assert result['clap_count'] == 0
        assert len(result['clap_timestamps']) == 0
    
    def test_get_audio_buffer_management(self, detector, sample_audio_file):
        """Test audio buffer management"""
        audio_path, _ = sample_audio_file
        detector.media_file_path = audio_path
        detector.loadMediaFile()
        
        # Get first buffer - should return bufferLength sized array
        initial_position = detector.file_position
        audio_buffer = detector.getAudio()
        
        # Basic checks
        assert isinstance(audio_buffer, np.ndarray)
        assert audio_buffer.dtype == np.int16
        assert len(audio_buffer) == detector.bufferLength  # Normal case should be 2048
        
        # Verify file position advances
        assert detector.file_position > initial_position
    
    def test_get_audio_at_end_of_file(self, detector, sample_audio_file):
        """Test audio buffer when reaching end of file"""
        audio_path, _ = sample_audio_file
        detector.media_file_path = audio_path
        detector.loadMediaFile()
        
        # Test near end of file - should still pad to bufferLength
        detector.file_position = len(detector.file_audio_data) - 100
        audio_buffer = detector.getAudio()
        assert len(audio_buffer) == detector.bufferLength
        
        # Test exactly at end of file - based on actual getAudio() implementation
        detector.file_position = len(detector.file_audio_data)
        audio_buffer = detector.getAudio()
        
        # The actual implementation returns empty array when file_position >= file length
        if detector.file_position >= len(detector.file_audio_data):
            assert len(audio_buffer) == 0
        else:
            assert len(audio_buffer) == detector.bufferLength
    
    def test_init_audio_params(self, detector, sample_audio_file):
        """Test audio parameter initialization"""
        audio_path, _ = sample_audio_file
        detector.media_file_path = audio_path
        detector.loadMediaFile()
        
        # Check that parameters are calculated correctly
        expected_reset_samples = int(detector.resetTime * detector.rate)
        assert detector.resetTimeSamples == expected_reset_samples
        
        expected_clap_interval_samples = int(detector.clapInterval * detector.rate)
        assert detector.clapIntervalSamples == expected_clap_interval_samples
        
        assert detector.audioBuffer is not None
        assert detector.currentSampleTime >= 0
    
    def test_cleanup_removes_temp_files(self, detector):
        """Test that cleanup removes temporary files"""
        # Create a fake temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            detector.temp_audio_file = temp_file.name
        
        # Verify file exists
        assert os.path.exists(detector.temp_audio_file)
        
        # Cleanup should remove it
        detector.cleanup()
        
        # File should be gone
        assert not os.path.exists(detector.temp_audio_file)
    
    def test_bandpass_filter(self, detector):
        """Test bandpass filter functionality"""
        # Create test signal with known frequencies
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Signal with 500Hz (should pass) and 5000Hz (should be filtered)
        signal = np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 5000 * t)
        
        # Apply bandpass filter (100-4000 Hz)
        filtered = detector.bandpassFilter(signal, lowcut=100, highcut=4000, fs=sample_rate)
        
        # Filtered signal should have reduced high-frequency content
        assert len(filtered) == len(signal)
        assert isinstance(filtered, np.ndarray)
        
        # The 500Hz component should remain, 5000Hz should be attenuated
        # (This is a basic check - we're not doing detailed frequency analysis)
        assert np.std(filtered) > 0  # Should still have signal content
    
    def test_calculate_time_difference(self, detector):
        """Test time difference calculation"""
        detector.samplesPerTimePeriod = 44100 * 10  # 10 seconds
        
        # Test normal case
        time_a = 5000
        time_b = 3000
        diff = detector.calculateTimeDifference(time_a, time_b)
        assert diff == 2000
        
        # Test wraparound case
        time_a = 1000
        time_b = 400000
        diff = detector.calculateTimeDifference(time_a, time_b)
        assert diff == (1000 + detector.samplesPerTimePeriod - 400000)
    
    def test_convert_to_circular_time(self, detector):
        """Test circular time conversion"""
        detector.samplesPerTimePeriod = 44100 * 10  # 10 seconds
        
        # Test normal case
        timestamp = 50000
        circular = detector.convertToCircularTime(timestamp)
        assert circular == timestamp % detector.samplesPerTimePeriod
        
        # Test wraparound case
        timestamp = detector.samplesPerTimePeriod + 5000
        circular = detector.convertToCircularTime(timestamp)
        assert circular == 5000
    
    @patch('subprocess.run')
    def test_video_processing_ffmpeg_unavailable(self, mock_subprocess, detector, temp_dir):
        """Test video processing when ffmpeg is not available"""
        # Create a fake video file
        video_file = Path(temp_dir) / "test_video.mp4"
        video_file.touch()  # Create empty file
        
        # Mock ffmpeg failure
        mock_subprocess.side_effect = FileNotFoundError("ffmpeg not found")
        
        detector.media_file_path = str(video_file)
        
        # Should raise error when trying to process video without ffmpeg
        with pytest.raises(Exception):
            detector.loadMediaFile()

    def test_ray_job_integration(self, temp_dir):
        """Integration test for the Ray job detect_claps_in_media"""
        if not ray.is_initialized():
            env_vars = {"PYTHONPATH": str(ROOT_DIR)}
            ray.init(runtime_env={"env_vars": env_vars})

        clap_times = [0.5, 2.0, 4.5, 6.0, 8.5, 10.0, 12.5]
        audio_data = TestDataGenerator.generate_test_audio_with_claps(
            clap_times, total_duration=15.0
        )
        
        test_audio_path = os.path.join(temp_dir, "integration_test_audio.wav")
        write_wav(test_audio_path, 44100, audio_data)

        output_dir = os.path.join(temp_dir, "clap_detection_output")
        
        result = ray.get(detect_claps_in_media.remote(
            media_path=test_audio_path,
            output_dir=output_dir,
            threshold_bias=6000,
            lowcut=200,
            highcut=3200,
            audio_threshold_bias=7900,
            audio_lowcut=780,
            audio_highcut=4000
        ))

        assert result and result.get('success', False)
        assert result['clap_count'] >= len(clap_times) * 0.4

    def test_specific_long_audio_claps(self, detector, temp_dir):
        """Test detection of two specific claps in a long audio file."""
        clap_times = [4.5, 311.5]  # 4.5s and 5m 11.5s
        audio_data = TestDataGenerator.generate_test_audio_with_claps(
            clap_times, total_duration=320.0
        )
        
        audio_path = Path(temp_dir) / "long_audio_specific_claps.wav"
        write_wav(str(audio_path), 44100, audio_data)
        detector.media_file_path = str(audio_path)

        detector.loadMediaFile()
        result = detector.detectAllClaps()

        assert result['clap_count'] == 2
        detected_times = [ts['timestamp_seconds'] for ts in result['clap_timestamps']]
        assert any(abs(t - 4.5) < 0.1 for t in detected_times)
        assert any(abs(t - 311.5) < 0.1 for t in detected_times)
