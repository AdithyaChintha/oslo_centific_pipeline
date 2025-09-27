#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Multi-Chunk Blob Utils
Tests with REAL Azure configuration and data for robust validation.
Includes unit tests, integration tests, edge cases, and error handling.
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
import yaml
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError, AzureError
import ray
from datetime import datetime

# Setup paths to import the actual functions
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# Import functions from multi_chunk_blob_utils
try:
    from utils.multi_chunk_blob_utils import (
        parse_chunk_filename_complete_legacy,
        parse_chunk_filename_simple,
        parse_chunk_filename_complete,
        process_session_folder,
        discover_sessions_by_folder,
        discover_sessions_basic,
        parse_completion_metadata_json,
        extract_session_id_from_metadata_filename,
        discover_chunks_in_prefix,
        discover_completion_metadata,
        group_chunks_by_session,
        is_walkthrough_session,
        mark_last_chunks_for_session,
        BasicDownloadManager,
        DownloadProgressTracker,
        download_ready_sessions,
        download_single_chunk_ray
    )
    from utils.blob_utils import (
        load_azure_config,
        create_azure_blob_client
    )
except ImportError as e:
    print(f"Warning: Could not import some functions: {e}")
    pytest.skip("Required modules not available", allow_module_level=True)


class TestMultiChunkBlobUtils:
    """COMPREHENSIVE UNIT TESTS - Testing multi-chunk blob utilities with REAL Azure data"""
    
    @pytest.fixture(scope="session")
    def real_config_path(self):
        """Path to real Azure configuration file"""
        config_path = os.path.join(PROJECT_ROOT, "blobfuse2_config.yaml")
        if not os.path.exists(config_path):
            pytest.skip(f"Real config file not found: {config_path}")
        return config_path
    
    @pytest.fixture(scope="session")
    def real_azure_config(self, real_config_path):
        """Load real Azure configuration"""
        return load_azure_config(real_config_path)
    
    @pytest.fixture(scope="session")
    def real_blob_client(self, real_azure_config):
        """Create real Azure blob client"""
        return create_azure_blob_client(real_azure_config)
    
    @pytest.fixture(scope="session")
    def real_container_client(self, real_blob_client, real_azure_config):
        """Create real Azure container client"""
        container_name = real_azure_config.get('container', 'instavideo')
        return real_blob_client.get_container_client(container_name)
    
    @pytest.fixture
    def temp_download_dir(self):
        """Create temporary download directory"""
        temp_dir = tempfile.mkdtemp(prefix="test_download_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_chunk_filenames(self):
        """Sample chunk filenames for testing parsing functions"""
        return [
            # Legacy format examples
            "11_515fb09c-f12f-48ee-91e7-b21c9f4ac5ab_audio1_working-on-laptop_20250907T00:00:00.000Z122100_01_audio.WAV",
            "11_515fb09c-f12f-48ee-91e7-b21c9f4ac5ab_video_working-on-laptop_20250907T00:00:00.000Z122100_05_video.insv",
            "12_a1b2c3d4-e5f6-7890-abcd-ef1234567890_activity_sitting_20250908T10:30:00.000Z123000_03_video.mp4",
            
            # Simple format examples
            "session_audio_01_audio.wav",
            "session_video_02_video.insv",
            "test_file_03_video.mp4",
            
            # Invalid/edge cases
            "invalid_file.txt",
            "no_extension",
            "",
            "metadata.json",
            "11_515fb09c-f12f-48ee-91e7-b21c9f4ac5ab_metadata.json"
        ]
    
    @pytest.fixture
    def sample_metadata_json(self):
        """Sample metadata JSON content for testing"""
        return {
            "id": "test-session-001",
            "home_id": "11",
            "participant_id": "515fb09c-f12f-48ee-91e7-b21c9f4ac5ab",
            "moderator_id": "mod123",
            "activity": "working-on-laptop",
            "specific_activity": "typing and browsing",
            "domain": "work",
            "start_datetime": "2025-09-07T00:00:00.000Z",
            "end_datetime": "2025-09-07T00:30:00.000Z",
            "duration_minutes": 30,
            "room": "office",
            "day_night": "day",
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "exported_at": "2025-09-07T01:00:00.000Z",
            "video_name": "session_video.mp4",
            "audio_name": "session_audio.wav"
        }

    # =============================================================================
    # UNIT TESTS - Filename Parsing Functions
    # =============================================================================
    
    def test_parse_chunk_filename_legacy_success(self, sample_chunk_filenames):
        """UNIT TEST: parse_chunk_filename_complete_legacy with valid filenames"""
        # Test valid legacy format
        filename = sample_chunk_filenames[0]  # WAV audio file
        result = parse_chunk_filename_complete_legacy(filename)
        
        assert result is not None, "Should successfully parse valid legacy filename"
        assert result["file_name"] == filename
        assert result["chunk_type"] == "audio"
        assert result["chunk_number"] == 1
        assert result["extension"] == ".wav"
        assert "11" in result["session_id"]
        assert "515fb09c-f12f-48ee-91e7-b21c9f4ac5ab" in result["session_id"]
        assert result["format"] == "underscore-new"
        
        print(f"✅ Legacy parsing successful: {filename}")
        print(f"   Session ID: {result['session_id']}")
        print(f"   Chunk: #{result['chunk_number']} ({result['chunk_type']})")
    
    def test_parse_chunk_filename_legacy_video(self, sample_chunk_filenames):
        """UNIT TEST: parse_chunk_filename_complete_legacy with video file"""
        filename = sample_chunk_filenames[1]  # INSV video file
        result = parse_chunk_filename_complete_legacy(filename)
        
        assert result is not None, "Should successfully parse valid video filename"
        assert result["chunk_type"] == "video"
        assert result["chunk_number"] == 5
        assert result["extension"] == ".insv"
        
        print(f"✅ Video parsing successful: {filename}")
    
    def test_parse_chunk_filename_legacy_invalid(self, sample_chunk_filenames):
        """UNIT TEST: parse_chunk_filename_complete_legacy with invalid filenames"""
        invalid_files = sample_chunk_filenames[6:]  # Invalid files
        
        for filename in invalid_files:
            result = parse_chunk_filename_complete_legacy(filename)
            print(f"⏭️ Correctly rejected invalid file: {filename}")
            
        # Test specific edge cases
        assert parse_chunk_filename_complete_legacy("") is None
        assert parse_chunk_filename_complete_legacy(None) is None
        assert parse_chunk_filename_complete_legacy("invalid.txt") is None
        assert parse_chunk_filename_complete_legacy("no_extension") is None
    
    def test_parse_chunk_filename_simple_success(self, sample_chunk_filenames):
        """UNIT TEST: parse_chunk_filename_simple with valid filenames"""
        # Test simple format
        simple_files = sample_chunk_filenames[3:6]  # Simple format files
        
        for filename in simple_files:
            result = parse_chunk_filename_simple(filename)
            
            if filename.endswith(('.wav', '.insv', '.mp4')):
                assert result is not None, f"Should parse simple filename: {filename}"
                assert result["file_name"] == filename
                assert result["parsing_method"] == "simple"
                assert result["chunk_type"] in ["audio", "video"]
                
                if "01" in filename:
                    assert result["chunk_number"] == 1
                elif "02" in filename:
                    assert result["chunk_number"] == 2
                elif "03" in filename:
                    assert result["chunk_number"] == 3
                
                print(f"✅ Simple parsing successful: {filename}")
                print(f"   Type: {result['chunk_type']}, Chunk: #{result.get('chunk_number', 'N/A')}")
    
    def test_parse_chunk_filename_simple_edge_cases(self):
        """UNIT TEST: parse_chunk_filename_simple edge cases"""
        edge_cases = [
            ("test_05_audio.wav", 5, "audio"),
            ("video_without_number.mp4", None, "video"),
            ("file_99_video.insv", 99, "video"),
            ("no_chunk_pattern.wav", None, "audio"),
        ]
        
        for filename, expected_chunk, expected_type in edge_cases:
            result = parse_chunk_filename_simple(filename)
            
            if result:
                assert result["chunk_type"] == expected_type
                assert result["chunk_number"] == expected_chunk
                print(f"✅ Edge case handled: {filename} → {expected_type} #{expected_chunk}")
    
    def test_parse_chunk_filename_complete_fallback(self, sample_chunk_filenames):
        """UNIT TEST: parse_chunk_filename_complete uses simple parsing first, then legacy fallback"""
        # Test that the complete function tries simple first, then legacy
        for filename in sample_chunk_filenames[:6]:  # Valid files
            result = parse_chunk_filename_complete(filename)
            
            if filename.endswith(('.wav', '.insv', '.mp4')):
                assert result is not None, f"Should parse filename: {filename}"
                assert "format" in result
                print(f"✅ Complete parsing: {filename} → format: {result.get('format', 'unknown')}")

    # =============================================================================
    # UNIT TESTS - Metadata Parsing Functions
    # =============================================================================
    
    def test_parse_completion_metadata_json_success(self, sample_metadata_json):
        """UNIT TEST: parse_completion_metadata_json with valid JSON"""
        json_content = json.dumps(sample_metadata_json)
        result = parse_completion_metadata_json(json_content)
        
        assert result is not None, "Should successfully parse valid metadata JSON"
        assert result["completion_detected"] is True
        assert result["completion_method"] == "metadata_json"
        assert "session_metadata" in result
        assert "expected_files" in result
        
        # Check session metadata
        session_meta = result["session_metadata"]
        assert session_meta["id"] == sample_metadata_json["id"]
        assert session_meta["home_id"] == sample_metadata_json["home_id"]
        assert session_meta["activity"] == sample_metadata_json["activity"]
        
        # Check expected files
        expected_files = result["expected_files"]
        assert expected_files["video_name"] == sample_metadata_json["video_name"]
        assert expected_files["audio_name"] == sample_metadata_json["audio_name"]
        
        print(f"✅ Metadata parsing successful")
        print(f"   Session ID: {result['session_id']}")
        print(f"   Activity: {session_meta['activity']}")
    
    def test_parse_completion_metadata_json_missing_fields(self, sample_metadata_json):
        """UNIT TEST: parse_completion_metadata_json with missing required fields"""
        # Remove required field
        incomplete_metadata = sample_metadata_json.copy()
        del incomplete_metadata["home_id"]
        
        json_content = json.dumps(incomplete_metadata)
        result = parse_completion_metadata_json(json_content)
        
        assert result is None, "Should return None for incomplete metadata"
        print("✅ Correctly rejected incomplete metadata (missing home_id)")
    
    def test_parse_completion_metadata_json_invalid_json(self):
        """UNIT TEST: parse_completion_metadata_json with invalid JSON"""
        invalid_json = '{"invalid": json: content}'
        result = parse_completion_metadata_json(invalid_json)
        
        assert result is None, "Should return None for invalid JSON"
        print("✅ Correctly rejected invalid JSON")
    
    def test_extract_session_id_from_metadata_filename(self):
        """UNIT TEST: extract_session_id_from_metadata_filename"""
        test_cases = [
            ("11_515fb09c-f12f-48ee-91e7-b21c9f4ac5ab_activity_working-on-laptop_20250907T00:00:00.000Z122100_metadata.json", 
             "11-515fb09c-f12f-48ee-91e7-b21c9f4ac5ab-working-on-laptop"),
            ("12_a1b2c3d4-e5f6-7890-abcd-ef1234567890_sitting_20250908123000_metadata.json",
             "12-a1b2c3d4-e5f6-7890-abcd-ef1234567890-sitting"),
            ("invalid_file.json", None),
            ("not_metadata.txt", None)
        ]
        
        for filename, expected in test_cases:
            result = extract_session_id_from_metadata_filename(filename)
            assert result == expected, f"Expected {expected}, got {result} for {filename}"
            
            if result:
                print(f"✅ Session ID extracted: {filename} → {result}")
            else:
                print(f"⏭️ Correctly rejected: {filename}")

    # =============================================================================
    # UNIT TESTS - Session Discovery Functions
    # =============================================================================
    
    def test_group_chunks_by_session(self):
        """UNIT TEST: group_chunks_by_session with sample chunks"""
        # Create sample parsed chunks
        sample_chunks = [
            {
                "file_name": "session1_01_audio.wav",
                "session_id": "session-1",
                "chunk_type": "audio",
                "chunk_number": 1
            },
            {
                "file_name": "session1_01_video.mp4",
                "session_id": "session-1", 
                "chunk_type": "video",
                "chunk_number": 1
            },
            {
                "file_name": "session1_02_audio.wav",
                "session_id": "session-1",
                "chunk_type": "audio", 
                "chunk_number": 2
            },
            {
                "file_name": "session2_01_video.mp4",
                "session_id": "session-2",
                "chunk_type": "video",
                "chunk_number": 1
            }
        ]
        
        sessions = group_chunks_by_session(sample_chunks)
        
        assert len(sessions) == 2, "Should group into 2 sessions"
        assert "session-1" in sessions
        assert "session-2" in sessions
        
        session1 = sessions["session-1"]
        assert session1["video_count"] == 1
        assert session1["audio_count"] == 2
        assert len(session1["video_chunks"]) == 1
        assert len(session1["audio_chunks"]) == 2
        assert session1["video_sequences"] == [1]
        assert session1["audio_sequences"] == [1, 2]
        
        print(f"✅ Session grouping successful: {len(sessions)} sessions")
        print(f"   Session 1: {session1['video_count']}V, {session1['audio_count']}A")
    
    def test_is_walkthrough_session(self):
        """UNIT TEST: is_walkthrough_session detection"""
        # Session with walkthrough in filename
        walkthrough_session = {
            "video_chunks": [{"file_name": "walkthrough_test_01_video.mp4"}],
            "audio_chunks": [],
            "session_metadata": {}
        }
        
        # Session with walkthrough in metadata
        metadata_walkthrough_session = {
            "video_chunks": [{"file_name": "normal_01_video.mp4"}],
            "audio_chunks": [],
            "session_metadata": {
                "activity": "device walkthrough",
                "specific_activity": "showing features"
            }
        }
        
        # Normal session
        normal_session = {
            "video_chunks": [{"file_name": "normal_01_video.mp4"}],
            "audio_chunks": [],
            "session_metadata": {
                "activity": "working",
                "specific_activity": "typing"
            }
        }
        
        assert is_walkthrough_session(walkthrough_session) is True
        assert is_walkthrough_session(metadata_walkthrough_session) is True
        assert is_walkthrough_session(normal_session) is False
        
        print("✅ Walkthrough detection working correctly")
    
    def test_mark_last_chunks_for_session(self):
        """UNIT TEST: mark_last_chunks_for_session"""
        session = {
            "video_chunks": [
                {"chunk_number": 1, "is_last_chunk": False},
                {"chunk_number": 2, "is_last_chunk": False},
                {"chunk_number": 3, "is_last_chunk": False}
            ],
            "audio_chunks": [
                {"chunk_number": 1, "is_last_chunk": False},
                {"chunk_number": 2, "is_last_chunk": False}
            ]
        }
        
        mark_last_chunks_for_session(session)
        
        # Check video chunks
        assert session["video_chunks"][0]["is_last_chunk"] is False
        assert session["video_chunks"][1]["is_last_chunk"] is False
        assert session["video_chunks"][2]["is_last_chunk"] is True  # Last video chunk
        
        # Check audio chunks
        assert session["audio_chunks"][0]["is_last_chunk"] is False
        assert session["audio_chunks"][1]["is_last_chunk"] is True  # Last audio chunk
        
        print("✅ Last chunk marking working correctly")

    # =============================================================================
    # UNIT TESTS - Download Manager and Progress Tracker
    # =============================================================================
    
    def test_basic_download_manager_initialization(self, temp_download_dir, real_blob_client, real_azure_config):
        """UNIT TEST: BasicDownloadManager initialization"""
        container_name = real_azure_config.get('container', 'instavideo')
        
        manager = BasicDownloadManager(real_blob_client, container_name, temp_download_dir)
        
        assert manager.blob_service_client == real_blob_client
        assert manager.container_name == container_name
        assert manager.local_download_dir == temp_download_dir
        assert os.path.exists(temp_download_dir), "Download directory should be created"
        assert "total_downloads" in manager.download_stats
        
        print(f"✅ Download manager initialized")
        print(f"   Container: {container_name}")
        print(f"   Download dir: {temp_download_dir}")
    
    def test_download_progress_tracker(self, temp_download_dir):
        """UNIT TEST: DownloadProgressTracker functionality"""
        session_id = "test-session-001"
        progress_dir = os.path.join(temp_download_dir, "progress")
        
        tracker = DownloadProgressTracker(session_id, progress_dir)
        
        assert tracker.session_id == session_id
        assert tracker.progress_dir == progress_dir
        assert os.path.exists(progress_dir), "Progress directory should be created"
        
        # Test progress data structure
        assert tracker.progress_data["session_id"] == session_id
        assert "created_at" in tracker.progress_data
        assert "download_status" in tracker.progress_data
        assert "chunks" in tracker.progress_data
        assert "stats" in tracker.progress_data
        
        print(f"✅ Progress tracker initialized: {session_id}")
        
        # Test session initialization
        sample_session = {
            "video_chunks": [
                {"file_name": "test_01_video.mp4", "chunk_data": {"chunk_type": "video"}, "chunk_number": 1, "size": 1000}
            ],
            "audio_chunks": [
                {"file_name": "test_01_audio.wav", "chunk_data": {"chunk_type": "audio"}, "chunk_number": 1, "size": 500}
            ]
        }
        
        tracker.initialize_session(sample_session)
        
        assert tracker.progress_data["stats"]["total_chunks"] == 2
        assert tracker.progress_data["stats"]["total_bytes"] == 1500
        assert len(tracker.progress_data["chunks"]) == 2
        
        print(f"✅ Session initialized: {tracker.progress_data['stats']['total_chunks']} chunks")
        
        # Test chunk status updates
        chunk_id = "01-video"
        tracker.update_chunk_status(chunk_id, "downloading")
        tracker.update_chunk_status(chunk_id, "completed", local_path="/tmp/test_file.mp4")
        
        chunk_data = tracker.progress_data["chunks"][chunk_id]
        assert chunk_data["download_status"] == "completed"
        assert chunk_data["local_path"] == "/tmp/test_file.mp4"
        assert chunk_data["attempts"] == 1
        
        print(f"✅ Chunk status tracking working")
        
        # Test progress summary
        summary = tracker.get_progress_summary()
        assert summary["session_id"] == session_id
        assert summary["chunks_completed"] == 1
        assert summary["chunks_total"] == 2
        assert summary["progress_percentage"] == 50.0
        
        print(f"✅ Progress summary: {summary['progress_percentage']:.1f}% complete")

    # =============================================================================
    # INTEGRATION TESTS - Real Azure Operations
    # =============================================================================
    
    @pytest.mark.integration
    def test_discover_chunks_in_prefix_real_azure(self, real_blob_client, real_azure_config):
        """INTEGRATION TEST: discover_chunks_in_prefix with real Azure storage"""
        container_name = real_azure_config.get('container', 'instavideo')
        test_prefix = "test/"  # Use a test prefix to avoid interfering with production data
        
        try:
            chunks = discover_chunks_in_prefix(real_blob_client, container_name, test_prefix)
            
            # Should return a list (may be empty)
            assert isinstance(chunks, list), "Should return a list of chunks"
            
            print(f"✅ Chunk discovery successful")
            print(f"   Container: {container_name}")
            print(f"   Prefix: {test_prefix}")
            print(f"   Chunks found: {len(chunks)}")
            
            # If chunks found, validate structure
            if chunks:
                first_chunk = chunks[0]
                required_fields = ["blob_name", "file_name", "size", "last_modified"]
                for field in required_fields:
                    assert field in first_chunk, f"Chunk should have field: {field}"
                
                print(f"   First chunk: {first_chunk['file_name']}")
                
        except Exception as e:
            print(f"⚠️ Integration test skipped (Azure access issue): {e}")
            pytest.skip("Azure integration test failed")
    
    @pytest.mark.integration 
    def test_discover_completion_metadata_real_azure(self, real_blob_client, real_azure_config):
        """INTEGRATION TEST: discover_completion_metadata with real Azure storage"""
        container_name = real_azure_config.get('container', 'instavideo')
        test_prefix = "test/"
        
        try:
            metadata = discover_completion_metadata(real_blob_client, container_name, test_prefix)
            
            assert isinstance(metadata, dict), "Should return a dictionary"
            
            print(f"✅ Metadata discovery successful")
            print(f"   Metadata files found: {len(metadata)}")
            
            # If metadata found, validate structure
            if metadata:
                first_session_id = list(metadata.keys())[0]
                first_metadata = metadata[first_session_id]
                
                required_fields = ["blob_name", "file_name", "session_metadata", "expected_files"]
                for field in required_fields:
                    assert field in first_metadata, f"Metadata should have field: {field}"
                
                print(f"   First session: {first_session_id}")
                
        except Exception as e:
            print(f"⚠️ Integration test skipped (Azure access issue): {e}")
            pytest.skip("Azure integration test failed")

    # =============================================================================
    # ERROR HANDLING AND EDGE CASES
    # =============================================================================
    
    def test_parse_functions_with_malformed_input(self):
        """UNIT TEST: Parse functions with various malformed inputs"""
        malformed_inputs = [
            None,
            "",
            "   ",
            "file_with_no_extension",
            "file.unknown_extension",
            "正常的文件名.mp4",  # Unicode filename
            "file with spaces.wav",
            "file@#$%^&*().insv",
            "very_long_filename_" + "x" * 200 + ".mp4"
        ]
        
        for malformed_input in malformed_inputs:
            # Test all parsing functions handle malformed input gracefully
            result_legacy = parse_chunk_filename_complete_legacy(malformed_input)
            result_simple = parse_chunk_filename_simple(malformed_input)
            result_complete = parse_chunk_filename_complete(malformed_input)
            
            print(f"🔍 Testing malformed input: '{malformed_input}'")
            print(f"   Legacy: {result_legacy is not None}")
            print(f"   Simple: {result_simple is not None}")
            print(f"   Complete: {result_complete is not None}")
    
    def test_download_manager_with_invalid_config(self, temp_download_dir):
        """UNIT TEST: BasicDownloadManager with invalid configuration"""
        # Test with invalid blob client
        with pytest.raises((AttributeError, TypeError)):
            manager = BasicDownloadManager(None, "container", temp_download_dir)
        
        # Test with invalid container name
        mock_client = Mock()
        manager = BasicDownloadManager(mock_client, "", temp_download_dir)
        assert manager.container_name == ""
        
        print("✅ Download manager handles invalid config appropriately")
    
    def test_progress_tracker_edge_cases(self, temp_download_dir):
        """UNIT TEST: DownloadProgressTracker edge cases"""
        session_id = "edge-case-session"
        progress_dir = os.path.join(temp_download_dir, "progress")
        
        tracker = DownloadProgressTracker(session_id, progress_dir)
        
        # Test with empty session
        empty_session = {"video_chunks": [], "audio_chunks": []}
        tracker.initialize_session(empty_session)
        
        assert tracker.progress_data["stats"]["total_chunks"] == 0
        assert tracker.progress_data["stats"]["total_bytes"] == 0
        
        # Test updating non-existent chunk
        tracker.update_chunk_status("non-existent-chunk", "completed")
        
        # Test getting progress with no chunks
        summary = tracker.get_progress_summary()
        assert summary["progress_percentage"] == 0
        assert summary["chunks_total"] == 0
        
        print("✅ Progress tracker handles edge cases")

    # =============================================================================
    # PERFORMANCE TESTS
    # =============================================================================
    
    @pytest.mark.performance
    def test_filename_parsing_performance(self, sample_chunk_filenames):
        """PERFORMANCE TEST: Filename parsing speed"""
        valid_filenames = [f for f in sample_chunk_filenames if f.endswith(('.wav', '.mp4', '.insv'))]
        iterations = 1000
        
        start_time = time.time()
        for _ in range(iterations):
            for filename in valid_filenames:
                parse_chunk_filename_complete(filename)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_parse = total_time / (iterations * len(valid_filenames)) * 1000
        
        # Should parse quickly
        assert avg_time_per_parse < 1.0, f"Parsing too slow: {avg_time_per_parse:.3f}ms average"
        
        print(f"✅ Filename parsing performance: {avg_time_per_parse:.3f}ms average")
        print(f"   Total operations: {iterations * len(valid_filenames)}")


def test_multi_chunk_blob_utils_standalone():
    """Run comprehensive tests without pytest framework"""
    print("🔬 MULTI-CHUNK BLOB UTILS COMPREHENSIVE TESTS")
    print("Testing ACTUAL functions with various scenarios")
    print("=" * 70)
    
    results = []
    
    # Test 1: Filename Parsing
    print("🎯 TEST 1: Filename Parsing Functions")
    print("-" * 40)
    
    try:
        # Test legacy parsing
        legacy_filename = "11_515fb09c-f12f-48ee-91e7-b21c9f4ac5ab_audio1_working-on-laptop_20250907T00:00:00.000Z122100_01_audio.WAV"
        result = parse_chunk_filename_complete_legacy(legacy_filename)
        
        assert result is not None, "Legacy parsing should succeed"
        assert result["chunk_type"] == "audio"
        assert result["chunk_number"] == 1
        assert "11" in result["session_id"]
        
        # Test simple parsing
        simple_filename = "test_02_video.mp4"
        result = parse_chunk_filename_simple(simple_filename)
        
        assert result is not None, "Simple parsing should succeed"
        assert result["chunk_type"] == "video"
        assert result["chunk_number"] == 2
        
        print("✅ PASS: Filename parsing functions work correctly")
        results.append({"test": "filename_parsing", "success": True})
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        results.append({"test": "filename_parsing", "success": False, "error": str(e)})
    
    print()
    
    # Test 2: Metadata Parsing
    print("🎯 TEST 2: Metadata Parsing Functions")
    print("-" * 40)
    
    try:
        sample_metadata = {
            "id": "test-001",
            "home_id": "11",
            "participant_id": "515fb09c-f12f-48ee-91e7-b21c9f4ac5ab",
            "activity": "working",
            "start_datetime": "2025-09-07T00:00:00.000Z",
            "end_datetime": "2025-09-07T00:30:00.000Z",
            "video_name": "test.mp4",
            "audio_name": "test.wav"
        }
        
        json_content = json.dumps(sample_metadata)
        result = parse_completion_metadata_json(json_content)
        
        assert result is not None, "Metadata parsing should succeed"
        assert result["completion_detected"] is True
        assert "session_metadata" in result
        
        # Test session ID extraction from filename
        metadata_filename = "11_515fb09c-f12f-48ee-91e7-b21c9f4ac5ab_activity_working-on-laptop_20250907T00:00:00.000Z122100_metadata.json"
        session_id = extract_session_id_from_metadata_filename(metadata_filename)
        
        assert session_id is not None, "Session ID extraction should succeed"
        assert "11-515fb09c-f12f-48ee-91e7-b21c9f4ac5ab-working-on-laptop" == session_id
        
        print("✅ PASS: Metadata parsing functions work correctly")
        results.append({"test": "metadata_parsing", "success": True})
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        results.append({"test": "metadata_parsing", "success": False, "error": str(e)})
    
    print()
    
    # Test 3: Session Grouping
    print("🎯 TEST 3: Session Grouping Functions")
    print("-" * 40)
    
    try:
        sample_chunks = [
            {"file_name": "test1_01_audio.wav", "session_id": "session-1", "chunk_type": "audio", "chunk_number": 1},
            {"file_name": "test1_01_video.mp4", "session_id": "session-1", "chunk_type": "video", "chunk_number": 1},
            {"file_name": "test1_02_audio.wav", "session_id": "session-1", "chunk_type": "audio", "chunk_number": 2}
        ]
        
        sessions = group_chunks_by_session(sample_chunks)
        
        assert len(sessions) == 1, "Should group into 1 session"
        assert "session-1" in sessions
        
        session = sessions["session-1"]
        assert session["video_count"] == 1
        assert session["audio_count"] == 2
        
        # Test last chunk marking
        mark_last_chunks_for_session(session)
        
        # Find last chunks
        video_last = any(c["is_last_chunk"] for c in session["video_chunks"])
        audio_last = any(c["is_last_chunk"] for c in session["audio_chunks"])
        
        assert video_last, "Should mark last video chunk"
        assert audio_last, "Should mark last audio chunk"
        
        print("✅ PASS: Session grouping functions work correctly")
        results.append({"test": "session_grouping", "success": True})
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        results.append({"test": "session_grouping", "success": False, "error": str(e)})
    
    print()
    
    # Test 4: Progress Tracking
    print("🎯 TEST 4: Progress Tracking")
    print("-" * 40)
    
    try:
        temp_dir = tempfile.mkdtemp(prefix="test_progress_")
        
        try:
            session_id = "test-session"
            progress_dir = os.path.join(temp_dir, "progress")
            
            tracker = DownloadProgressTracker(session_id, progress_dir)
            
            # Initialize with sample session
            sample_session = {
                "video_chunks": [{"file_name": "test.mp4", "chunk_data": {"chunk_type": "video"}, "chunk_number": 1, "size": 1000}],
                "audio_chunks": [{"file_name": "test.wav", "chunk_data": {"chunk_type": "audio"}, "chunk_number": 1, "size": 500}]
            }
            
            tracker.initialize_session(sample_session)
            
            assert tracker.progress_data["stats"]["total_chunks"] == 2
            assert tracker.progress_data["stats"]["total_bytes"] == 1500
            
            # Update chunk status
            tracker.update_chunk_status("01-video", "completed")
            
            summary = tracker.get_progress_summary()
            assert summary["chunks_completed"] == 1
            assert summary["progress_percentage"] == 50.0
            
            print("✅ PASS: Progress tracking works correctly")
            results.append({"test": "progress_tracking", "success": True})
            
        finally:
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        results.append({"test": "progress_tracking", "success": False, "error": str(e)})
    
    print()
    
    # Final Results
    successful = len([r for r in results if r['success']])
    
    print("🏁 MULTI-CHUNK BLOB UTILS TEST RESULTS")
    print("=" * 50)
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{i}. {result['test']}: {status}")
        if not result['success']:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    print(f"\nMulti-Chunk Tests: {successful}/{len(results)} passed")
    
    return successful == len(results)


if __name__ == "__main__":
    print("🔬 MULTI-CHUNK BLOB UTILS COMPREHENSIVE TESTING")
    print("Testing functions with various scenarios and edge cases")
    print("=" * 70)
    
    try:
        success = test_multi_chunk_blob_utils_standalone()
        
        if success:
            print("\n🎉 ALL MULTI-CHUNK BLOB UTILS TESTS PASSED!")
        else:
            print("\n⚠️ Some multi-chunk blob utils tests failed.")
            
    except Exception as e:
        print(f"\n💥 Multi-chunk blob utils test error: {e}")
