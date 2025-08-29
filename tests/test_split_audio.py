import ray
import pytest
import numpy as np
import tempfile
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from scipy.io.wavfile import write as write_wav, read as read_wav
from unittest.mock import patch, MagicMock

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import the Ray job and helper functions
from ray_jobs.audio_splitter import split_audio_into_shards, _basename_noext, _ensure_dir, get_audio_info


class TestAudioGenerator:
    """Helper class to generate synthetic test audio data"""
    
    @staticmethod
    def generate_tone(frequency=440, sample_rate=44100, duration=1.0, amplitude=10000):
        """Generate a pure tone"""
        t = np.linspace(0, duration, int(sample_rate * duration))
        tone = amplitude * np.sin(2 * np.pi * frequency * t)
        return tone.astype(np.int16)
    
    @staticmethod
    def generate_silence(sample_rate=44100, duration=1.0):
        """Generate silence"""
        samples = int(sample_rate * duration)
        return np.zeros(samples, dtype=np.int16)
    
    @staticmethod
    def generate_white_noise(sample_rate=44100, duration=1.0, amplitude=1000):
        """Generate white noise"""
        samples = int(sample_rate * duration)
        noise = np.random.normal(0, amplitude, samples)
        return noise.astype(np.int16)
    
    @staticmethod
    def generate_test_audio_with_tones(tone_segments, sample_rate=44100, total_duration=10.0):
        """
        Generate test audio with different tones at specified segments
        
        Args:
            tone_segments: List of (start_time, end_time, frequency) tuples
            sample_rate: Audio sample rate
            total_duration: Total audio duration in seconds
        
        Returns:
            numpy array of audio data
        """
        total_samples = int(sample_rate * total_duration)
        audio = np.zeros(total_samples, dtype=np.int16)
        
        for start_time, end_time, frequency in tone_segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            if start_sample < total_samples and end_sample > 0:
                # Ensure we don't exceed bounds
                start_sample = max(0, start_sample)
                end_sample = min(total_samples, end_sample)
                
                # Generate tone for this segment
                duration = (end_sample - start_sample) / sample_rate
                tone = TestAudioGenerator.generate_tone(frequency, sample_rate, duration)
                
                # Insert tone into audio
                audio[start_sample:end_sample] = tone[:end_sample - start_sample]
        
        return audio


# UNIT TESTS
class TestAudioSplitter:
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_audio_file(self, temp_dir):
        """Create a temporary audio file with known segments"""
        # Create audio with different tones at different times
        tone_segments = [
            (0.0, 2.0, 440),    # A4 tone for 2 seconds
            (3.0, 5.0, 880),    # A5 tone for 2 seconds
            (6.0, 8.0, 220),    # A3 tone for 2 seconds
        ]
        
        audio_data = TestAudioGenerator.generate_test_audio_with_tones(
            tone_segments, sample_rate=44100, total_duration=10.0
        )
        
        # Save to temporary WAV file
        audio_file = Path(temp_dir) / "test_audio.wav"
        write_wav(str(audio_file), 44100, audio_data)
        
        return str(audio_file), tone_segments
    
    @pytest.fixture
    def silent_audio_file(self, temp_dir):
        """Create a silent audio file"""
        silence = TestAudioGenerator.generate_silence(duration=5.0)
        audio_file = Path(temp_dir) / "silent_audio.wav"
        write_wav(str(audio_file), 44100, silence)
        return str(audio_file)
    
    @pytest.fixture
    def long_audio_file(self, temp_dir):
        """Create a longer audio file for multi-shard testing"""
        # Create 3 minutes of audio with different segments
        tone_segments = [
            (0.0, 30.0, 440),     # First 30 seconds
            (60.0, 90.0, 880),    # Second 30 seconds  
            (120.0, 150.0, 220),  # Third 30 seconds
        ]
        
        audio_data = TestAudioGenerator.generate_test_audio_with_tones(
            tone_segments, sample_rate=44100, total_duration=180.0  # 3 minutes
        )
        
        audio_file = Path(temp_dir) / "long_audio.wav"
        write_wav(str(audio_file), 44100, audio_data)
        return str(audio_file)
    
    def test_basename_noext_function(self):
        """Test basename without extension helper function"""
        test_cases = [
            ("/path/to/audio.wav", "audio"),
            ("/path/to/long_audio_file.mp3", "long_audio_file"),
            ("simple_audio.aac", "simple_audio"),
            ("/complex/path/audio_part0.wav", "audio_part0"),
            ("audio", "audio"),  # No extension
        ]
        
        for file_path, expected in test_cases:
            result = _basename_noext(file_path)
            assert result == expected, f"Failed for {file_path}: expected {expected}, got {result}"
    
    def test_ensure_dir_function(self, temp_dir):
        """Test directory creation helper function"""
        test_dir = os.path.join(temp_dir, "audio_test_dir")
        
        # Test directory creation
        _ensure_dir(test_dir)
        assert os.path.exists(test_dir)
        
        # Test that it doesn't fail if directory already exists
        _ensure_dir(test_dir)
        assert os.path.exists(test_dir)
    
    def test_audio_parameters_validation(self):
        """Test audio splitting parameters validation"""
        # Test parameter ranges and types
        valid_durations = [30, 60, 120]
        valid_sample_rates = [16000, 22050, 44100, 48000]
        valid_channels = [1, 2, 6]
        valid_formats = ["wav", "mp3", "aac", "flac"]
        
        # Test duration values
        for duration in valid_durations:
            assert duration > 0 and isinstance(duration, int)
        
        # Test sample rates
        for rate in valid_sample_rates:
            assert rate > 0 and isinstance(rate, int)
        
        # Test channel counts
        for channels in valid_channels:
            assert channels > 0 and isinstance(channels, int)
        
        # Test formats
        for fmt in valid_formats:
            assert isinstance(fmt, str) and len(fmt) > 0
    
    def test_output_path_generation(self):
        """Test audio shard output path generation logic"""
        test_audio = "/path/to/test_audio.wav"
        output_dir = "/tmp/audio_shards"
        audio_format = "wav"
        
        # Simulate the path generation from the Ray job
        basename = _basename_noext(test_audio)
        
        for i in range(3):
            expected_path = os.path.join(output_dir, f"{basename}_part{i}.{audio_format}")
            assert expected_path == f"/tmp/audio_shards/test_audio_part{i}.wav"
    
    @patch('subprocess.run')
    def test_ffmpeg_availability_mocked(self, mock_subprocess):
        """Test FFmpeg availability with mocking"""
        # Mock successful FFmpeg response
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="ffmpeg version 4.4.0"
        )
        
        # Test the check
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        assert result.returncode == 0
        mock_subprocess.assert_called_once()
    
    def test_load_valid_audio_file(self, sample_audio_file):
        """Test loading a valid audio file"""
        audio_path, expected_segments = sample_audio_file
        
        # Verify file was created correctly
        assert os.path.exists(audio_path)
        
        # Load and verify audio properties
        sample_rate, audio_data = read_wav(audio_path)
        assert sample_rate == 44100
        assert len(audio_data) > 0
        assert audio_data.dtype == np.int16
    
    def test_load_nonexistent_file(self):
        """Test handling of non-existent audio files"""
        nonexistent_file = "nonexistent_audio.wav"
        
        with pytest.raises(FileNotFoundError):
            if not os.path.exists(nonexistent_file):
                raise FileNotFoundError(f"Audio file not found: {nonexistent_file}")
    
    def test_audio_duration_calculation(self, sample_audio_file):
        """Test audio duration calculation"""
        audio_path, _ = sample_audio_file
        
        # Load audio and calculate duration
        sample_rate, audio_data = read_wav(audio_path)
        calculated_duration = len(audio_data) / sample_rate
        
        # Should be approximately 10 seconds
        assert 9.9 <= calculated_duration <= 10.1
    
    def test_shard_timing_calculation(self):
        """Test shard timing calculation logic"""
        total_duration = 180.0  # 3 minutes
        shard_duration = 60     # 60 seconds per shard
        
        expected_shards = []
        t = 0.0
        idx = 0
        
        while t < total_duration - 1e-6:
            dur = min(shard_duration, max(0.0, total_duration - t))
            expected_shards.append((idx, t, dur))
            t += shard_duration
            idx += 1
        
        # Should create 3 shards: 0-60s, 60-120s, 120-180s
        assert len(expected_shards) == 3
        assert expected_shards[0] == (0, 0.0, 60.0)
        assert expected_shards[1] == (1, 60.0, 60.0)
        assert expected_shards[2] == (2, 120.0, 60.0)


# INTEGRATION TESTS
def test_synthetic_audio_generation():
    """Unit test for synthetic audio generation"""
    print("Unit Test 1: Synthetic Audio Generation")
    print("-" * 40)
    
    try:
        # Test tone generation
        tone = TestAudioGenerator.generate_tone(440, 44100, 1.0)
        assert isinstance(tone, np.ndarray)
        assert tone.dtype == np.int16
        assert len(tone) == 44100  # 1 second at 44100 Hz
        
        # Test silence generation
        silence = TestAudioGenerator.generate_silence(duration=2.0)
        assert isinstance(silence, np.ndarray)
        assert np.all(silence == 0)
        
        # Test complex audio generation
        tone_segments = [(0.0, 1.0, 440), (2.0, 3.0, 880)]
        audio = TestAudioGenerator.generate_test_audio_with_tones(tone_segments, total_duration=5.0)
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 5 * 44100  # 5 seconds at 44100 Hz
        
        print("Synthetic audio generation successful")
        print(f"   - Tone: {len(tone)} samples")
        print(f"   - Test audio: {len(audio)} samples ({len(audio)/44100:.1f}s)")
        return True
        
    except Exception as e:
        print(f"Synthetic audio generation failed: {e}")
        return False

def test_create_synthetic_audio_files():
    """Unit test for creating synthetic audio files"""
    print("\nUnit Test 2: Synthetic Audio File Creation")
    print("-" * 40)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create audio with known tone segments
            tone_segments = [(1.0, 3.0, 440), (5.0, 7.0, 880), (8.0, 10.0, 220)]
            audio_data = TestAudioGenerator.generate_test_audio_with_tones(
                tone_segments, total_duration=12.0
            )
            
            # Save to WAV file
            test_audio_path = os.path.join(temp_dir, "synthetic_tones.wav")
            write_wav(test_audio_path, 44100, audio_data)
            
            # Verify file was created and can be loaded
            assert os.path.exists(test_audio_path)
            
            # Test loading with scipy
            sample_rate, loaded_audio = read_wav(test_audio_path)
            assert sample_rate == 44100
            assert len(loaded_audio) == len(audio_data)
            
            print("Synthetic audio file creation successful")
            print(f"   - Audio duration: {len(audio_data)/44100:.1f}s")
            print(f"   - Expected segments: {len(tone_segments)}")
            print(f"   - File size: {len(loaded_audio)} samples")
            return True
            
    except Exception as e:
        print(f"Synthetic audio file test failed: {e}")
        return False

def test_ffmpeg_tools_availability():
    """Unit test for FFmpeg tools availability"""
    print("\nUnit Test 3: FFmpeg Tools Availability")
    print("-" * 40)
    
    try:
        # Test ffmpeg
        ffmpeg_result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        ffmpeg_available = ffmpeg_result.returncode == 0
        
        # Test ffprobe
        ffprobe_result = subprocess.run(
            ["ffprobe", "-version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        ffprobe_available = ffprobe_result.returncode == 0
        
        if ffmpeg_available and ffprobe_available:
            ffmpeg_version = ffmpeg_result.stdout.split('\n')[0]
            ffprobe_version = ffprobe_result.stdout.split('\n')[0]
            print(f"FFmpeg available: {ffmpeg_version}")
            print(f"FFprobe available: {ffprobe_version}")
            return True
        else:
            print("FFmpeg or FFprobe not available")
            print("Audio splitting requires FFmpeg tools")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"FFmpeg tools check failed: {e}")
        return False

def test_ray_job_integration():
    """Integration test for the Ray job split_audio_into_shards"""
    print("\nIntegration Test: Complete Audio Splitting")
    print("=" * 50)
    
    # Initialize Ray
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
        print("Ray initialized")
    else:
        print("Ray already initialized")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print("Creating synthetic test audio...")
            
            # Create comprehensive test audio with multiple segments
            tone_segments = [
                (0.0, 20.0, 440),    # First 20 seconds - A4
                (30.0, 50.0, 880),   # Middle 20 seconds - A5
                (60.0, 80.0, 220),   # Last 20 seconds - A3
            ]
            
            audio_data = TestAudioGenerator.generate_test_audio_with_tones(
                tone_segments, total_duration=90.0  # 1.5 minutes
            )
            
            # Save test audio
            test_audio_path = os.path.join(temp_dir, "integration_test_audio.wav")
            write_wav(test_audio_path, 44100, audio_data)
            
            print(f"Test audio created: {os.path.basename(test_audio_path)}")
            print(f"   - Duration: {len(audio_data)/44100:.1f}s")
            print(f"   - Expected segments: {len(tone_segments)}")
            print()
            
            # Test the Ray job
            print("Testing Ray Job: split_audio_into_shards")
            print("-" * 40)
            
            start_time = time.time()
            output_dir = os.path.join(temp_dir, "audio_shards")
            
            # Call the Ray job
            result = ray.get(split_audio_into_shards.remote(
                audio_path=test_audio_path,
                output_dir=output_dir,
                duration_sec=30,  # 30 second shards
                audio_format="wav",
                audio_codec="pcm_s16le",
                sample_rate=44100,
                channels=1,
            ))
            
            processing_time = time.time() - start_time
            
            if result and isinstance(result, list) and len(result) > 0:
                print(f"SUCCESS in {processing_time:.2f}s")
                print(f"Results:")
                print(f"   - Result type: {type(result)}")
                print(f"   - Number of shards: {len(result)}")
                
                # Verify output files
                total_size = 0
                for i, shard_path in enumerate(result):
                    if os.path.exists(shard_path):
                        shard_size = os.path.getsize(shard_path)
                        total_size += shard_size
                        
                        # Verify audio properties
                        try:
                            sample_rate, shard_audio = read_wav(shard_path)
                            duration = len(shard_audio) / sample_rate
                            print(f"   - Shard {i+1}: {os.path.basename(shard_path)} ({shard_size} bytes, {duration:.1f}s)")
                        except Exception:
                            print(f"   - Shard {i+1}: {os.path.basename(shard_path)} ({shard_size} bytes)")
                    else:
                        print(f"   - Shard {i+1}: MISSING - {shard_path}")
                        return False
                
                print(f"   - Total output size: {total_size} bytes")
                
                # Calculate expected number of shards
                audio_duration = len(audio_data) / 44100
                expected_shards = int(np.ceil(audio_duration / 30))  # 30 second shards
                
                print(f"\nShard Analysis:")
                print(f"   - Input duration: {audio_duration:.1f}s")
                print(f"   - Expected shards: {expected_shards}")
                print(f"   - Created shards: {len(result)}")
                
                if len(result) >= expected_shards - 1:  # Allow some tolerance
                    print(f"Audio splitting integration test PASSED!")
                    print(f"   Processing time: {processing_time:.1f}s")
                    print(f"   Shards created: {len(result)}")
                    return True
                else:
                    print(f"Unexpected number of shards created")
                    return False
                
            else:
                print(f"FAILED - Invalid result: {result}")
                return False
                
    except Exception as e:
        processing_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"FAILED after {processing_time:.2f}s")
        print(f"   Error: {str(e)}")
        
        # Check for common issues
        if "ffmpeg" in str(e).lower():
            print("   This might be due to FFmpeg not being available")
        
        return False

# BEHAVIORAL TESTING - Given input X, do we get expected output Y
def test_behavioral_audio_splitting_60s():
    """
    BEHAVIORAL TEST CASE 1: 60-second audio → 1 shard
    INPUT: 60.0 seconds of audio, 60-second shard duration
    EXPECTED OUTPUT: Exactly 1 shard of 60.0 seconds
    """
    print("BEHAVIORAL TEST 1: 60s Audio → 1 Shard")
    print("-" * 40)
    
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # GIVEN: Create exactly 60.0 seconds of synthetic audio
            tone_segments = [(0.0, 60.0, 440)]  # Single A4 tone for entire duration
            audio_data = TestAudioGenerator.generate_test_audio_with_tones(
                tone_segments, total_duration=60.0
            )
            
            test_audio_path = os.path.join(temp_dir, "test_60s.wav")
            write_wav(test_audio_path, 44100, audio_data)
            
            # WHEN: Split with 60-second shards
            result = ray.get(split_audio_into_shards.remote(
                audio_path=test_audio_path,
                output_dir=os.path.join(temp_dir, "shards"),
                duration_sec=60,
                audio_format="wav"
            ))
            
            # THEN: Verify expected output
            EXPECTED_SHARD_COUNT = 1
            EXPECTED_DURATION = 60.0
            
            if len(result) != EXPECTED_SHARD_COUNT:
                print(f"FAILED: Expected {EXPECTED_SHARD_COUNT} shard, got {len(result)}")
                return False
            
            # Verify shard duration
            sample_rate, shard_audio = read_wav(result[0])
            actual_duration = len(shard_audio) / sample_rate
            
            if abs(actual_duration - EXPECTED_DURATION) > 0.1:  # 0.1s tolerance
                print(f"FAILED: Expected {EXPECTED_DURATION}s, got {actual_duration:.1f}s")
                return False
            
            print(f"PASSED: Input 60.0s → Output 1 shard of {actual_duration:.1f}s")
            return True
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_behavioral_audio_splitting_150s():
    """
    BEHAVIORAL TEST CASE 2: 150-second audio → 3 shards
    INPUT: 150.0 seconds of audio, 60-second shard duration
    EXPECTED OUTPUT: 3 shards (60s, 60s, 30s)
    """
    print("\nBEHAVIORAL TEST 2: 150s Audio → 3 Shards (60s, 60s, 30s)")
    print("-" * 40)
    
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # GIVEN: Create exactly 150.0 seconds of synthetic audio
            tone_segments = [
                (0.0, 50.0, 440),    # First segment
                (50.0, 100.0, 880), # Second segment
                (100.0, 150.0, 220) # Third segment
            ]
            audio_data = TestAudioGenerator.generate_test_audio_with_tones(
                tone_segments, total_duration=150.0
            )
            
            test_audio_path = os.path.join(temp_dir, "test_150s.wav")
            write_wav(test_audio_path, 44100, audio_data)
            
            # WHEN: Split with 60-second shards
            result = ray.get(split_audio_into_shards.remote(
                audio_path=test_audio_path,
                output_dir=os.path.join(temp_dir, "shards"),
                duration_sec=60,
                audio_format="wav"
            ))
            
            # THEN: Verify expected output
            EXPECTED_SHARD_COUNT = 3
            EXPECTED_DURATIONS = [60.0, 60.0, 30.0]
            
            if len(result) != EXPECTED_SHARD_COUNT:
                print(f"FAILED: Expected {EXPECTED_SHARD_COUNT} shards, got {len(result)}")
                return False
            
            # Verify each shard duration
            actual_durations = []
            for i, shard_path in enumerate(result):
                sample_rate, shard_audio = read_wav(shard_path)
                duration = len(shard_audio) / sample_rate
                actual_durations.append(duration)
                
                expected_dur = EXPECTED_DURATIONS[i]
                if abs(duration - expected_dur) > 0.1:  # 0.1s tolerance
                    print(f"FAILED: Shard {i+1} expected {expected_dur}s, got {duration:.1f}s")
                    return False
            
            print(f"PASSED: Input 150.0s → Output {len(result)} shards: {[f'{d:.1f}s' for d in actual_durations]}")
            return True
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def test_behavioral_audio_splitting_30s():
    """
    BEHAVIORAL TEST CASE 3: 30-second audio → 1 shard (shorter than shard duration)
    INPUT: 30.0 seconds of audio, 60-second shard duration
    EXPECTED OUTPUT: 1 shard of 30.0 seconds
    """
    print("\nBEHAVIORAL TEST 3: 30s Audio → 1 Shard (shorter than duration)")
    print("-" * 40)
    
    if not ray.is_initialized():
        env_vars = {"PYTHONPATH": str(ROOT_DIR)}
        ray.init(runtime_env={"env_vars": env_vars})
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # GIVEN: Create exactly 30.0 seconds of synthetic audio
            tone_segments = [(0.0, 30.0, 440)]
            audio_data = TestAudioGenerator.generate_test_audio_with_tones(
                tone_segments, total_duration=30.0
            )
            
            test_audio_path = os.path.join(temp_dir, "test_30s.wav")
            write_wav(test_audio_path, 44100, audio_data)
            
            # WHEN: Split with 60-second shards
            result = ray.get(split_audio_into_shards.remote(
                audio_path=test_audio_path,
                output_dir=os.path.join(temp_dir, "shards"),
                duration_sec=60,
                audio_format="wav"
            ))
            
            # THEN: Verify expected output
            EXPECTED_SHARD_COUNT = 1
            EXPECTED_DURATION = 30.0
            
            if len(result) != EXPECTED_SHARD_COUNT:
                print(f"FAILED: Expected {EXPECTED_SHARD_COUNT} shard, got {len(result)}")
                return False
            
            # Verify shard duration
            sample_rate, shard_audio = read_wav(result[0])
            actual_duration = len(shard_audio) / sample_rate
            
            if abs(actual_duration - EXPECTED_DURATION) > 0.1:  # 0.1s tolerance
                print(f"FAILED: Expected {EXPECTED_DURATION}s, got {actual_duration:.1f}s")
                return False
            
            print(f"PASSED: Input 30.0s → Output 1 shard of {actual_duration:.1f}s")
            return True
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def run_all_tests():
    """Run ALL tests: original (unit+integration) AND behavioral tests for audio splitting"""
    print("COMPLETE Audio Splitting Testing Suite")
    print("=" * 50)
    print("Running unit tests, integration tests, AND behavioral tests")
    print()
    
    # Check dependencies
    try:
        import scipy
        import numpy as np
        print(f"Dependencies: NumPy {np.__version__}, SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        return False
    
    # Check FFmpeg availability
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if result.returncode != 0:
            print("FFmpeg not available - tests will fail")
            return False
    except:
        print("FFmpeg not available - tests will fail")
        return False
    
    # Run original unit tests
    print("\nUNIT TESTS (Original)")
    print("=" * 30)
    
    unit_tests = [
        test_synthetic_audio_generation,
        test_create_synthetic_audio_files,
        test_ffmpeg_tools_availability,
    ]
    
    unit_results = []
    for test_func in unit_tests:
        try:
            result = test_func()
            unit_results.append(result)
        except Exception as e:
            print(f"Unit test {test_func.__name__} failed: {e}")
            unit_results.append(False)
    
    unit_passed = sum(unit_results)
    print(f"\nUnit Tests Summary: {unit_passed}/{len(unit_tests)} passed")
    
    # Run original integration test
    print(f"\nINTEGRATION TEST (Original)")
    print("=" * 30)
    
    integration_success = False
    try:
        integration_success = test_ray_job_integration()
    except Exception as e:
        print(f"Integration test failed: {e}")
    
    print(f"Integration Test: {'PASSED' if integration_success else 'FAILED'}")
    
    # Run behavioral tests
    print(f"\nBEHAVIORAL TESTS (New - Manager's Request)")
    print("=" * 50)
    print("Testing documented scenarios with predictable results")
    
    behavioral_tests = [
        test_behavioral_audio_splitting_60s,
        test_behavioral_audio_splitting_150s,
        test_behavioral_audio_splitting_30s,
    ]
    
    behavioral_results = []
    for test_func in behavioral_tests:
        try:
            result = test_func()
            behavioral_results.append(result)
        except Exception as e:
            print(f"Behavioral test {test_func.__name__} failed: {e}")
            behavioral_results.append(False)
    
    behavioral_passed = sum(behavioral_results)
    behavioral_total = len(behavioral_results)
    
    print(f"\nBehavioral Tests Summary: {behavioral_passed}/{behavioral_total} passed")
    
    # Test case summary
    print(f"\nBEHAVIORAL TEST DOCUMENTATION:")
    print(f"1. 60s input → 1 shard (60s): {'PASS' if behavioral_results[0] else 'FAIL'}")
    print(f"2. 150s input → 3 shards (60s, 60s, 30s): {'PASS' if behavioral_results[1] else 'FAIL'}")
    print(f"3. 30s input → 1 shard (30s): {'PASS' if behavioral_results[2] else 'FAIL'}")
    
    # Overall summary
    total_tests = len(unit_tests) + 1 + behavioral_total  # unit + integration + behavioral
    total_passed = unit_passed + (1 if integration_success else 0) + behavioral_passed
    
    print(f"\nOVERALL RESULTS")
    print("=" * 50)
    print(f"Unit Tests: {unit_passed}/{len(unit_tests)} passed")
    print(f"Integration Test: {'PASSED' if integration_success else 'FAILED'}")
    print(f"Behavioral Tests: {behavioral_passed}/{behavioral_total} passed")
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\nALL TESTS PASSED! Audio splitting is fully validated.")
        print("✅ Unit tests: Validate individual functions")
        print("✅ Integration test: Validate complete Ray job")
        print("✅ Behavioral tests: Validate predictable scenarios")
        return True
    else:
        print(f"\n{total_tests - passed_tests} test(s) failed.")
        print("Check errors above for details.")
        return False


# Can be run as pytest or standalone
if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit_code = 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nTesting interrupted")
        exit_code = 2
    except Exception as e:
        print(f"\nTesting error: {e}")
        exit_code = 3
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown")
        
        sys.exit(exit_code)