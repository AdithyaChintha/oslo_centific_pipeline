"""
Report State JSONs pytest Testing Suite
Run with: pytest test_report_state_jsons.py -v
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Mock azure.storage.blob before importing report_state_jsons
sys.modules['azure'] = Mock()
sys.modules['azure.storage'] = Mock()
sys.modules['azure.storage.blob'] = Mock()
sys.modules['azure.storage.blob'].BlobServiceClient = Mock()

# Import the module under test
from data_handling_to_client import report_state_jsons

# ==================================================
# CONFIGURATION
# ==================================================
TEST_INPUTS_DIR = Path(__file__).parent.parent / "data_handling_to_client" / "test_inputs"
TEST_FILES = ["15_state.json", "23_state.json", "62_state.json", "63_state.json"]
USE_REAL_DATA = True


# ==================================================
# FIXTURES
# ==================================================
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment once for all tests"""
    # Verify test input files exist
    if USE_REAL_DATA:
        for test_file in TEST_FILES:
            file_path = TEST_INPUTS_DIR / test_file
            if not file_path.exists():
                pytest.skip(f"Test input file {test_file} not found")
    yield
    # Cleanup after all tests (if needed)
    pass


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing"""
    return {
        "home_id": "15",
        "version": "2.0",
        "total_data_processed_gb": 559.589,
        "total_data_processed_bytes": 600854574197,
        "total_processing_time_seconds": 451.321483,
        "total_files_processed": 59,
        "total_video_files_processed": 32,
        "total_audio_files_processed": 27
    }


@pytest.fixture
def sample_daily_processing():
    """Sample daily processing data for testing"""
    return {
        "2025-09-28": {
            "files_successful": 59,
            "video_files_successful": 32,
            "audio_files_successful": 27,
            "files_failed": 0,
            "total_upload_time_seconds": 450.9870200000001
        }
    }


@pytest.fixture
def sample_state_json():
    """Sample state JSON structure for testing"""
    return {
        "metadata": {
            "home_id": "15",
            "version": "2.0",
            "total_data_processed_gb": 559.589,
            "total_processing_time_seconds": 451.321483,
            "total_files_processed": 59,
            "total_video_files_processed": 32,
            "total_audio_files_processed": 27
        },
        "daily_processing": {
            "2025-09-28": {
                "processing_sessions": [
                    {
                        "session_id": "sess_123",
                        "container_id": "test_container",
                        "videos": [
                            {
                                "upload_status": "success",
                                "upload_blob_name": "test_video.insv",
                                "file_size": 1024**3,  # 1GB
                                "upload_duration_seconds": 10.5
                            },
                            {
                                "upload_status": "success",
                                "upload_blob_name": "test_audio.wav",
                                "file_size": 1024**3 // 2,  # 0.5GB
                                "upload_duration_seconds": 5.2
                            }
                        ]
                    }
                ]
            }
        }
    }


@pytest.fixture
def mock_blob_service():
    """Mock blob service client"""
    mock_bsc = Mock()
    mock_container = Mock()
    mock_bsc.get_container_client.return_value = mock_container
    return mock_bsc, mock_container


# ==================================================
# Utilities
# ==================================================
def load_test_data(filename):
    """Load test data from JSON file"""
    file_path = TEST_INPUTS_DIR / filename
    with open(file_path, 'r') as f:
        return json.load(f)


def create_mock_blob(name):
    """Create a mock blob object"""
    mock_blob = Mock()
    mock_blob.name = name
    return mock_blob


# ==================================================
# CORE FUNCTION TESTS
# ==================================================
class TestHumanGB:
    """Test cases for the human_gb function"""
    
    def test_valid_byte_conversion(self):
        """Test human_gb with valid byte values"""
        # Test with 1GB
        result = report_state_jsons.human_gb(1024**3)
        assert result == 1.0, "1GB conversion failed"
        
        # Test with 2.5GB
        result = report_state_jsons.human_gb(2.5 * 1024**3)
        assert result == 2.5, "2.5GB conversion failed"
        
        # Test with 0 bytes
        result = report_state_jsons.human_gb(0)
        assert result == 0.0, "0 bytes conversion failed"
    
    def test_none_input(self):
        """Test human_gb with None input"""
        result = report_state_jsons.human_gb(None)
        assert result is None, "None input should return None"
    
    def test_invalid_input_types(self):
        """Test human_gb with invalid input types"""
        # Test with string
        result = report_state_jsons.human_gb("invalid")
        assert result is None, "Invalid string should return None"
        
        # Test with list
        result = report_state_jsons.human_gb([1, 2, 3])
        assert result is None, "Invalid list should return None"
    
    def test_precision_handling(self):
        """Test human_gb precision with large numbers"""
        large_bytes = 1234567890123
        result = report_state_jsons.human_gb(large_bytes)
        expected = round(large_bytes / (1024**3), 3)
        assert result == expected, f"Precision test failed: expected {expected}, got {result}"
    
    @pytest.mark.parametrize("bytes_val,expected", [
        (1024**3, 1.0),
        (1024**3 * 1.5, 1.5),
        (1024**3 * 0.5, 0.5),
        (1024**3 * 10, 10.0),
    ])
    def test_parametrized_conversions(self, bytes_val, expected):
        """Test human_gb with parametrized values"""
        result = report_state_jsons.human_gb(bytes_val)
        assert result == expected, f"Conversion failed for {bytes_val} bytes"


class TestExtractSummary:
    """Test cases for the extract_summary function"""
    
    def test_complete_data_extraction(self, sample_metadata, sample_daily_processing):
        """Test extract_summary with complete data"""
        result = report_state_jsons.extract_summary(
            sample_metadata, sample_daily_processing, "test.json"
        )
        
        # Verify all expected fields are present
        expected_fields = [
            "state_blob_name", "home_id", "version", "total_data_processed_gb",
            "total_processing_time_seconds", "files_processed", "video_files_processed",
            "audio_files_processed", "video_files_successful", "audio_files_successful",
            "files_failed"
        ]
        
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"
        
        # Verify specific values
        assert result["home_id"] == "15", "Home ID mismatch"
        assert result["version"] == "2.0", "Version mismatch"
        assert result["total_data_processed_gb"] == 559.589, "Data processed mismatch"
        assert result["files_processed"] == 59, "Files processed mismatch"
        assert result["video_files_successful"] == 32, "Video files successful mismatch"
        assert result["audio_files_successful"] == 27, "Audio files successful mismatch"
        assert result["files_failed"] == 0, "Files failed mismatch"
    
    def test_missing_metadata_fields(self, sample_daily_processing):
        """Test extract_summary with missing metadata fields"""
        incomplete_meta = {"home_id": "15"}
        result = report_state_jsons.extract_summary(
            incomplete_meta, sample_daily_processing, "test.json"
        )
        
        # Should handle missing fields gracefully
        assert result["home_id"] == "15", "Home ID should be preserved"
        assert result["version"] is None, "Missing version should be None"
        assert result["total_data_processed_gb"] is None, "Missing data should be None"
    
    def test_empty_daily_processing(self, sample_metadata):
        """Test extract_summary with empty daily processing data"""
        empty_daily = {}
        result = report_state_jsons.extract_summary(
            sample_metadata, empty_daily, "test.json"
        )
        
        # Should handle empty daily data
        assert result["video_files_successful"] == 0, "Empty daily should result in 0 video files"
        assert result["audio_files_successful"] == 0, "Empty daily should result in 0 audio files"
        assert result["files_failed"] == 0, "Empty daily should result in 0 failed files"
    
    def test_bytes_to_gb_conversion(self, sample_daily_processing):
        """Test extract_summary with bytes instead of GB"""
        meta_with_bytes = {
            "home_id": "15",
            "total_data_processed_bytes": 600854574197
        }
        
        result = report_state_jsons.extract_summary(
            meta_with_bytes, sample_daily_processing, "test.json"
        )
        
        # Should convert bytes to GB
        expected_gb = round(600854574197 / (1024**3), 3)
        assert result["total_data_processed_gb"] == expected_gb, "Bytes to GB conversion failed"


class TestFlattenFileEntries:
    """Test cases for the flatten_file_entries function"""
    
    def test_video_file_processing(self, sample_state_json):
        """Test flatten_file_entries with video files"""
        result = report_state_jsons.flatten_file_entries(sample_state_json)
        
        assert len(result) == 2, "Expected 2 file entries"
        
        # Check first file (video)
        video_file = result[0]
        assert video_file["day"] == "2025-09-28", "Day mismatch"
        assert video_file["container_id"] == "test_container", "Container ID mismatch"
        assert video_file["status"] == "success", "Status mismatch"
        assert video_file["file_type"] == "video", "File type mismatch"
        assert video_file["blob_name"] == "test_video.insv", "Blob name mismatch"
        assert video_file["file_size_gb"] == 1.0, "File size mismatch"
        assert video_file["upload_duration_seconds"] == 10.5, "Upload duration mismatch"
    
    def test_audio_file_processing(self, sample_state_json):
        """Test flatten_file_entries with audio files"""
        result = report_state_jsons.flatten_file_entries(sample_state_json)
        
        # Check second file (audio)
        audio_file = result[1]
        assert audio_file["file_type"] == "audio", "Audio file type mismatch"
        assert audio_file["blob_name"] == "test_audio.wav", "Audio blob name mismatch"
        assert audio_file["file_size_gb"] == 0.5, "Audio file size mismatch"
    
    @pytest.mark.parametrize("filename,expected_type", [
        ("test.mp4", "video"),
        ("test.mov", "video"),
        ("test.mkv", "video"),
        ("test.insv", "video"),
        ("test.wav", "audio"),
        ("test.mp3", "audio"),
        ("test.flac", "audio"),
        ("test.aac", "audio"),
        ("test.unknown", "unknown")
    ])
    def test_file_type_detection(self, filename, expected_type):
        """Test file type detection for various extensions"""
        state_json = {
            "metadata": {},
            "daily_processing": {
                "2025-09-28": {
                    "processing_sessions": [{
                        "videos": [{
                            "upload_blob_name": filename,
                            "upload_status": "success"
                        }]
                    }]
                }
            }
        }
        
        result = report_state_jsons.flatten_file_entries(state_json)
        assert result[0]["file_type"] == expected_type, f"File type detection failed for {filename}"
    
    def test_empty_data_handling(self):
        """Test flatten_file_entries with empty data"""
        empty_state = {"metadata": {}, "daily_processing": {}}
        result = report_state_jsons.flatten_file_entries(empty_state)
        assert result == [], "Empty data should return empty list"
    
    def test_missing_fields_handling(self):
        """Test flatten_file_entries with missing fields"""
        incomplete_state = {
            "metadata": {},
            "daily_processing": {
                "2025-09-28": {
                    "processing_sessions": [{
                        "videos": [{
                            "upload_status": "success"
                            # Missing blob_name and other fields
                        }]
                    }]
                }
            }
        }
        
        result = report_state_jsons.flatten_file_entries(incomplete_state)
        assert len(result) == 1, "Should handle missing fields gracefully"
        assert result[0]["file_type"] == "unknown", "Missing blob_name should result in unknown type"
        assert result[0]["blob_name"] is None, "Missing blob_name should be None"




# ==================================================
# INTEGRATION TESTS WITH REAL DATA
# ==================================================
class TestRealDataIntegration:
    """Integration tests using actual test input files"""
    
    @pytest.mark.parametrize("test_file", TEST_FILES)
    def test_load_test_input_files(self, test_file):
        """Test that all test input files can be loaded and parsed"""
        data = load_test_data(test_file)
        
        # Verify basic structure
        assert "metadata" in data, f"Missing metadata in {test_file}"
        assert "daily_processing" in data, f"Missing daily_processing in {test_file}"
        
        # Verify metadata structure
        metadata = data["metadata"]
        assert "home_id" in metadata, f"Missing home_id in {test_file}"
        assert "version" in metadata, f"Missing version in {test_file}"
        assert "total_files_processed" in metadata, f"Missing total_files_processed in {test_file}"
    
    @pytest.mark.parametrize("test_file", TEST_FILES)
    def test_extract_summary_with_real_data(self, test_file):
        """Test extract_summary function with actual test input data"""
        data = load_test_data(test_file)
        metadata = data["metadata"]
        daily = data["daily_processing"]
        
        result = report_state_jsons.extract_summary(metadata, daily, test_file)
        
        # Verify all required fields are present
        required_fields = [
            "state_blob_name", "home_id", "version", "total_data_processed_gb",
            "total_processing_time_seconds", "files_processed", "video_files_processed",
            "audio_files_processed", "video_files_successful", "audio_files_successful",
            "files_failed"
        ]
        
        for field in required_fields:
            assert field in result, f"Field {field} missing in {test_file}"
        
        # Verify data types
        assert isinstance(result["home_id"], str), f"home_id should be string in {test_file}"
        assert isinstance(result["files_processed"], int), f"files_processed should be int in {test_file}"
        assert isinstance(result["video_files_successful"], int), f"video_files_successful should be int in {test_file}"
        assert isinstance(result["audio_files_successful"], int), f"audio_files_successful should be int in {test_file}"
    
    @pytest.mark.parametrize("test_file", TEST_FILES)
    def test_flatten_file_entries_with_real_data(self, test_file):
        """Test flatten_file_entries function with actual test input data"""
        data = load_test_data(test_file)
        result = report_state_jsons.flatten_file_entries(data)
        
        # Verify we get a list of file entries
        assert isinstance(result, list), f"Files should be a list in {test_file}"
        
        if result:  # If there are files
            # Verify structure of first file entry
            first_file = result[0]
            required_fields = [
                "day", "container_id", "status", "file_type", 
                "blob_name", "file_size_gb", "upload_duration_seconds"
            ]
            
            for field in required_fields:
                assert field in first_file, f"Field {field} missing in {test_file}"
            
            # Verify file_type is one of the expected values
            assert first_file["file_type"] in ["video", "audio", "unknown"], f"Invalid file_type in {test_file}"
    
    @pytest.mark.parametrize("test_file", TEST_FILES)
    def test_data_consistency_across_functions(self, test_file):
        """Test that data is consistent when processed by different functions"""
        data = load_test_data(test_file)
        metadata = data["metadata"]
        daily = data["daily_processing"]
        
        # Extract summary
        summary = report_state_jsons.extract_summary(metadata, daily, test_file)
        
        # Flatten file entries
        files = report_state_jsons.flatten_file_entries(data)
        
        # Verify consistency
        if files:
            # Count files by type from flattened entries
            video_count = sum(1 for f in files if f["file_type"] == "video")
            audio_count = sum(1 for f in files if f["file_type"] == "audio")
            
            # These should match the summary counts
            assert video_count == summary["video_files_successful"], f"Video count mismatch in {test_file}"
            assert audio_count == summary["audio_files_successful"], f"Audio count mismatch in {test_file}"


# ==================================================
# EDGE CASE TESTS
# ==================================================
class TestEdgeCases:
    """Edge case tests"""
    
    def test_human_gb_edge_cases(self):
        """Test human_gb with edge case values"""
        # Test with very small number
        result = report_state_jsons.human_gb(1)
        assert result == 0.0, "Very small number should round to 0.0"
        
        # Test with negative number
        result = report_state_jsons.human_gb(-1024**3)
        assert result == -1.0, "Negative number should be handled correctly"
        
        # Test with float
        result = report_state_jsons.human_gb(1.5 * 1024**3)
        assert result == 1.5, "Float input should be handled correctly"
    
    def test_extract_summary_edge_cases(self):
        """Test extract_summary with edge case data"""
        # Test with None values
        meta_with_nones = {
            "home_id": None,
            "version": None,
            "total_data_processed_gb": None,
            "total_data_processed_bytes": None
        }
        
        daily_with_zeros = {
            "2025-09-28": {
                "files_successful": 0,
                "video_files_successful": 0,
                "audio_files_successful": 0,
                "files_failed": 0
            }
        }
        
        result = report_state_jsons.extract_summary(meta_with_nones, daily_with_zeros, "test.json")
        
        # Should handle None values gracefully
        assert result["home_id"] is None, "None home_id should be preserved"
        assert result["version"] is None, "None version should be preserved"
        assert result["total_data_processed_gb"] is None, "None data should be preserved"
        assert result["video_files_successful"] == 0, "Zero values should be handled correctly"
    
    def test_flatten_file_entries_malformed_data(self):
        """Test flatten_file_entries with malformed data"""
        # Test with missing processing_sessions
        malformed_data = {
            "metadata": {},
            "daily_processing": {
                "2025-09-28": {
                    # Missing processing_sessions
                }
            }
        }
        
        result = report_state_jsons.flatten_file_entries(malformed_data)
        assert result == [], "Malformed data should return empty list"
        
        # Test with empty processing_sessions
        malformed_data2 = {
            "metadata": {},
            "daily_processing": {
                "2025-09-28": {
                    "processing_sessions": []
                }
            }
        }
        
        result = report_state_jsons.flatten_file_entries(malformed_data2)
        assert result == [], "Empty processing_sessions should return empty list"
    
    def test_file_type_detection_edge_cases(self):
        """Test file type detection with edge cases"""
        # Test uppercase extensions
        state_json = {
            "metadata": {},
            "daily_processing": {
                "2025-09-28": {
                    "processing_sessions": [{
                        "videos": [{
                            "upload_blob_name": "test.INSV",
                            "upload_status": "success"
                        }]
                    }]
                }
            }
        }
        
        result = report_state_jsons.flatten_file_entries(state_json)
        assert result[0]["file_type"] == "video", "Uppercase INSV should be detected as video"


# ==================================================
# END-TO-END INTEGRATION TEST
# ==================================================
class TestEndToEndConsolidation:
    """Test the complete end-to-end consolidation process"""
    
    def test_create_consolidated_json_from_test_files(self, temp_dir):
        """Test creating consolidated JSON from test input files using original functions"""
        # Create output file in the test_inputs directory
        output_file = TEST_INPUTS_DIR / "test_consolidated_report.json"
        
        # Manually create consolidated JSON using the original functions
        consolidated = {"generated_at": datetime.utcnow().isoformat() + "Z", "homes": []}
        
        # Process each test file using the original functions
        for test_file in TEST_FILES:
            file_path = TEST_INPUTS_DIR / test_file
            print(f"Processing {test_file}...")
            
            # Load the test data
            with open(file_path, 'r') as f:
                state_json = json.load(f)
            
            # Use the original functions to process the data
            meta = state_json.get("metadata", {})
            daily = state_json.get("daily_processing", {})
            
            # Create home entry using original functions
            home_entry = {
                "summary": report_state_jsons.extract_summary(meta, daily, test_file),
                "files": report_state_jsons.flatten_file_entries(state_json)
            }
            
            consolidated["homes"].append(home_entry)
        
        # Save the consolidated data to a file
        with open(output_file, 'w') as f:
            json.dump(consolidated, f, indent=2)
        
        # Verify the structure
        assert "generated_at" in consolidated, "Missing generated_at field"
        assert "homes" in consolidated, "Missing homes field"
        assert len(consolidated["homes"]) == len(TEST_FILES), f"Expected {len(TEST_FILES)} homes, got {len(consolidated['homes'])}"
        
        # Verify each home entry
        for i, home_entry in enumerate(consolidated["homes"]):
            assert "summary" in home_entry, f"Missing summary in home {i}"
            assert "files" in home_entry, f"Missing files in home {i}"
            
            summary = home_entry["summary"]
            files = home_entry["files"]
            
            # Verify summary fields
            required_summary_fields = [
                "state_blob_name", "home_id", "version", "total_data_processed_gb",
                "total_processing_time_seconds", "files_processed", "video_files_processed",
                "audio_files_processed", "video_files_successful", "audio_files_successful",
                "files_failed"
            ]
            
            for field in required_summary_fields:
                assert field in summary, f"Missing field {field} in home {i} summary"
            
            # Verify files structure
            if files:  # If there are files
                first_file = files[0]
                required_file_fields = [
                    "day", "container_id", "status", "file_type", 
                    "blob_name", "file_size_gb", "upload_duration_seconds"
                ]
                
                for field in required_file_fields:
                    assert field in first_file, f"Missing field {field} in home {i} files"
        
        # Print summary for verification
        print(f"\n✅ Successfully created consolidated JSON with {len(consolidated['homes'])} homes")
        for i, home in enumerate(consolidated["homes"]):
            summary = home["summary"]
            files_count = len(home["files"])
            print(f"   Home {i+1}: ID={summary['home_id']}, Files={files_count}, Data={summary['total_data_processed_gb']}GB")
        
        print(f"   📄 Consolidated JSON saved to: {output_file}")
    
    def test_consolidated_json_data_consistency(self, temp_dir):
        """Test that consolidated JSON data is consistent with input files"""
        # Manually create consolidated JSON using the original functions
        consolidated = {"generated_at": datetime.utcnow().isoformat() + "Z", "homes": []}
        
        # Process each test file using the original functions
        for test_file in TEST_FILES:
            file_path = TEST_INPUTS_DIR / test_file
            
            # Load the test data
            with open(file_path, 'r') as f:
                state_json = json.load(f)
            
            # Use the original functions to process the data
            meta = state_json.get("metadata", {})
            daily = state_json.get("daily_processing", {})
            
            # Create home entry using original functions
            home_entry = {
                "summary": report_state_jsons.extract_summary(meta, daily, test_file),
                "files": report_state_jsons.flatten_file_entries(state_json)
            }
            
            consolidated["homes"].append(home_entry)
        
        # Verify data consistency for each home
        for home_entry in consolidated["homes"]:
            summary = home_entry["summary"]
            files = home_entry["files"]
            
            # Count files by type from the files list
            video_count = sum(1 for f in files if f["file_type"] == "video")
            audio_count = sum(1 for f in files if f["file_type"] == "audio")
            
            # These should match the summary counts
            assert video_count == summary["video_files_successful"], f"Video count mismatch for home {summary['home_id']}"
            assert audio_count == summary["audio_files_successful"], f"Audio count mismatch for home {summary['home_id']}"
            
            # Verify total files
            total_files = len(files)
            assert total_files == summary["files_processed"], f"Total files mismatch for home {summary['home_id']}"
        
        print(f"\n✅ Data consistency verified for all {len(consolidated['homes'])} homes")




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
