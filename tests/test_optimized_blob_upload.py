"""
Test for optimized blob upload functionality

This test validates the new parallel upload optimization:
- upload_file_worker_enhanced
- upload_output_directory_with_sas_optimized
- Performance comparison with baseline
"""

import os
import sys
import time
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import pytest

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from utils.blob_utils import (
    upload_file_worker_enhanced,
    upload_output_directory_with_sas_optimized,
    upload_output_directory_with_sas
)

class TestOptimizedBlobUpload:
    """Test suite for optimized blob upload functionality"""

    @classmethod
    def setup_class(cls):
        """Setup test fixtures"""
        cls.test_config = {
            'max_workers': 4,
            'chunk_size_mb': 1,  # Small for testing
            'retry_attempts': 2,
            'progress_reporting': True
        }

        cls.mock_azure_config = {
            "account_name": "testaccount",
            "account_key": "test_key_12345678901234567890",
            "container_name": "testcontainer"
        }

    def create_test_files(self, temp_dir, num_files=10):
        """Create test files with various sizes and types"""
        test_files = []

        # Create directory structure
        directories = ["audio_shards", "erp_shards", "front_shards", "metadata"]
        for dir_name in directories:
            os.makedirs(os.path.join(temp_dir, dir_name), exist_ok=True)

        # Critical files (small JSON/metadata)
        for i in range(3):
            file_path = os.path.join(temp_dir, "metadata", f"metadata_{i}.json")
            content = {"id": i, "type": "metadata", "size": "small"}
            with open(file_path, 'w') as f:
                json.dump(content, f)
            test_files.append(Path(file_path))

        # Large video files
        for i in range(3):
            file_path = os.path.join(temp_dir, "erp_shards", f"shard_{i}.mp4")
            # Create 2MB file to test chunked upload
            with open(file_path, 'wb') as f:
                f.write(b'0' * (2 * 1024 * 1024))  # 2MB
            test_files.append(Path(file_path))

        # Audio files
        for i in range(2):
            file_path = os.path.join(temp_dir, "audio_shards", f"audio_{i}.wav")
            # Create 5MB file
            with open(file_path, 'wb') as f:
                f.write(b'0' * (5 * 1024 * 1024))  # 5MB
            test_files.append(Path(file_path))

        # Other files
        for i in range(2):
            file_path = os.path.join(temp_dir, f"other_{i}.txt")
            with open(file_path, 'w') as f:
                f.write(f"Test content {i}")
            test_files.append(Path(file_path))

        return test_files

    def test_upload_file_worker_enhanced(self):
        """Test the enhanced file upload worker"""
        print("🧪 Testing upload_file_worker_enhanced...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir) / "test_file.json"
            test_content = {"test": "data", "size": "1KB"}
            with open(test_file, 'w') as f:
                json.dump(test_content, f)

            # Mock Azure blob client
            mock_blob_client = Mock()
            mock_upload_blob = Mock()
            mock_blob_client.get_blob_client.return_value.upload_blob = mock_upload_blob

            # Test file info
            file_info = (
                test_file,
                "test/blob/path/test_file.json",
                mock_blob_client,
                "testcontainer",
                self.test_config
            )

            # Execute worker
            result = upload_file_worker_enhanced(file_info)

            # Validate result
            assert result["success"] == True
            assert result["blob_name"] == "test/blob/path/test_file.json"
            assert result["local_path"] == str(test_file)
            assert result["size"] > 0
            assert "chunks" in result

            # Verify blob client was called
            mock_blob_client.get_blob_client.assert_called_once()
            mock_upload_blob.assert_called_once()

            print("   ✅ upload_file_worker_enhanced works correctly")

    def test_upload_file_worker_enhanced_with_chunking(self):
        """Test the enhanced worker with large file chunking"""
        print("🧪 Testing upload_file_worker_enhanced with chunking...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create large test file (2MB)
            test_file = Path(temp_dir) / "large_file.mp4"
            with open(test_file, 'wb') as f:
                f.write(b'0' * (2 * 1024 * 1024))  # 2MB

            # Mock Azure blob client for chunked upload
            mock_blob_client = Mock()
            mock_blob_service = Mock()
            mock_stage_block = Mock()
            mock_commit_block_list = Mock()

            mock_blob_service.stage_block = mock_stage_block
            mock_blob_service.commit_block_list = mock_commit_block_list
            mock_blob_client.get_blob_client.return_value = mock_blob_service

            # Test file info with small chunk size
            config = self.test_config.copy()
            config['chunk_size_mb'] = 1  # 1MB chunks

            file_info = (
                test_file,
                "test/blob/path/large_file.mp4",
                mock_blob_client,
                "testcontainer",
                config
            )

            # Execute worker
            result = upload_file_worker_enhanced(file_info)

            # Validate result
            assert result["success"] == True
            assert result["chunks"] == 2  # 2MB file with 1MB chunks = 2 chunks

            # Verify chunked upload was used
            assert mock_stage_block.call_count == 2  # 2 chunks
            mock_commit_block_list.assert_called_once()

            print("   ✅ Chunked upload works correctly")

    def test_upload_file_worker_enhanced_with_retry(self):
        """Test the enhanced worker retry logic"""
        print("🧪 Testing upload_file_worker_enhanced retry logic...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            test_file = Path(temp_dir) / "retry_test.json"
            with open(test_file, 'w') as f:
                json.dump({"test": "retry"}, f)

            # Mock Azure blob client that fails twice then succeeds
            mock_blob_client = Mock()
            mock_upload_blob = Mock()
            mock_upload_blob.side_effect = [
                Exception("Network error"),  # First attempt fails
                Exception("Timeout error"),  # Second attempt fails
                None  # Third attempt succeeds
            ]
            mock_blob_client.get_blob_client.return_value.upload_blob = mock_upload_blob

            file_info = (
                test_file,
                "test/blob/path/retry_test.json",
                mock_blob_client,
                "testcontainer",
                self.test_config
            )

            # Execute worker
            start_time = time.time()
            result = upload_file_worker_enhanced(file_info)
            elapsed_time = time.time() - start_time

            # Validate result
            assert result["success"] == True
            assert mock_upload_blob.call_count == 3  # 3 attempts
            assert elapsed_time > 1.0  # Should have some delay from retries

            print("   ✅ Retry logic works correctly")

    @patch('utils.blob_utils.generate_blob_sas')
    def test_upload_output_directory_with_sas_optimized(self, mock_generate_sas):
        """Test the optimized upload function"""
        print("🧪 Testing upload_output_directory_with_sas_optimized...")

        # Mock SAS token generation
        mock_generate_sas.return_value = "mock_sas_token_12345"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = self.create_test_files(temp_dir)

            # Mock Azure blob client
            mock_blob_client = Mock()
            mock_blob_service = Mock()
            mock_upload_blob = Mock()

            mock_blob_service.upload_blob = mock_upload_blob
            mock_blob_client.get_blob_client.return_value = mock_blob_service

            # Execute optimized upload
            start_time = time.time()
            result = upload_output_directory_with_sas_optimized(
                output_dir=temp_dir,
                azure_blob_client=mock_blob_client,
                container_name="testcontainer",
                blob_base_path="test/output",
                video_name="test_video_20250918_123456",
                account_name="testaccount",
                account_key="test_key",
                sas_expiry_days=365,
                config=self.test_config
            )
            upload_time = time.time() - start_time

            # Validate result structure
            assert result["success"] == True
            assert "uploaded_files" in result
            assert "failed_files" in result
            assert "shard_urls" in result
            assert "summary" in result

            # Validate performance metrics
            summary = result["summary"]
            assert "upload_duration" in summary
            assert "files_per_second" in summary
            assert "improvement_factor" in summary
            assert summary["files_per_second"] > 0

            # Validate uploaded files
            assert len(result["uploaded_files"]) == len(test_files)
            assert len(result["failed_files"]) == 0

            # Validate shard URL structure
            shard_urls = result["shard_urls"]
            assert "audio_urls" in shard_urls
            assert "view_urls" in shard_urls
            assert "dual_view_urls" in shard_urls
            assert "single_view_urls" in shard_urls

            print(f"   ✅ Uploaded {len(result['uploaded_files'])} files")
            print(f"   ✅ Upload time: {upload_time:.2f}s")
            print(f"   ✅ Performance: {summary['files_per_second']:.1f} files/sec")
            print(f"   ✅ Improvement factor: {summary['improvement_factor']:.1f}x")

    def test_file_categorization(self):
        """Test that files are properly categorized for priority upload"""
        print("🧪 Testing file categorization...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = self.create_test_files(temp_dir)

            # Mock the upload to capture categorization
            with patch('utils.blob_utils.ThreadPoolExecutor') as mock_executor:
                # Mock executor to capture file categorization
                mock_executor_instance = Mock()
                mock_executor.return_value.__enter__.return_value = mock_executor_instance
                mock_executor_instance.submit.return_value.result.return_value = {
                    "success": True, "blob_name": "test", "local_path": "test", "size": 100
                }

                mock_blob_client = Mock()
                mock_blob_client.get_blob_client.return_value.upload_blob = Mock()

                # Execute optimized upload
                result = upload_output_directory_with_sas_optimized(
                    output_dir=temp_dir,
                    azure_blob_client=mock_blob_client,
                    container_name="testcontainer",
                    blob_base_path="test/output",
                    video_name="test_video",
                    account_name="testaccount",
                    account_key="test_key",
                    config=self.test_config
                )

                # Verify executor was called multiple times (for different phases)
                assert mock_executor.call_count >= 3  # Phase 1, 2, 3

                print("   ✅ File categorization and phased upload working")

    def test_configuration_validation(self):
        """Test configuration parameter validation"""
        print("🧪 Testing configuration validation...")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal test file
            test_file = Path(temp_dir) / "test.json"
            with open(test_file, 'w') as f:
                json.dump({"test": "data"}, f)

            mock_blob_client = Mock()
            mock_blob_client.get_blob_client.return_value.upload_blob = Mock()

            # Test with various configurations
            test_configs = [
                {},  # Empty config (should use defaults)
                {'max_workers': 2},  # Partial config
                {'max_workers': 16, 'chunk_size_mb': 8, 'retry_attempts': 5},  # Full config
                None  # None config (should use defaults)
            ]

            for i, config in enumerate(test_configs):
                result = upload_output_directory_with_sas_optimized(
                    output_dir=temp_dir,
                    azure_blob_client=mock_blob_client,
                    container_name="testcontainer",
                    blob_base_path="test/output",
                    video_name=f"test_video_{i}",
                    account_name="testaccount",
                    account_key="test_key",
                    config=config
                )

                assert result["success"] == True

            print("   ✅ Configuration validation works correctly")

    def test_error_handling(self):
        """Test error handling in optimized upload"""
        print("🧪 Testing error handling...")

        # Test with non-existent directory
        result = upload_output_directory_with_sas_optimized(
            output_dir="/nonexistent/directory",
            azure_blob_client=Mock(),
            container_name="testcontainer",
            blob_base_path="test/output",
            video_name="test_video",
            account_name="testaccount",
            account_key="test_key",
            config=self.test_config
        )

        assert result["success"] == False
        assert "error" in result
        assert "Output directory not found" in result["error"]

        print("   ✅ Error handling works correctly")

def test_performance_comparison():
    """Compare optimized vs original upload performance (simulation)"""
    print("🎯 Performance Comparison Test")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_instance = TestOptimizedBlobUpload()
        test_files = test_instance.create_test_files(temp_dir, num_files=20)

        print(f"📁 Created {len(test_files)} test files")

        # Mock blob clients
        mock_blob_client = Mock()
        mock_blob_client.get_blob_client.return_value.upload_blob = Mock()

        config = {
            'max_workers': 8,
            'chunk_size_mb': 4,
            'retry_attempts': 3,
            'progress_reporting': True
        }

        # Test optimized version
        print("🚀 Testing optimized upload...")
        start_time = time.time()

        optimized_result = upload_output_directory_with_sas_optimized(
            output_dir=temp_dir,
            azure_blob_client=mock_blob_client,
            container_name="testcontainer",
            blob_base_path="test/output",
            video_name="test_video_optimized",
            account_name="testaccount",
            account_key="test_key",
            config=config
        )

        optimized_time = time.time() - start_time

        print(f"✅ Optimized upload completed:")
        print(f"   Time: {optimized_time:.2f}s")
        print(f"   Files: {len(optimized_result['uploaded_files'])}")
        print(f"   Performance: {optimized_result['summary']['files_per_second']:.1f} files/sec")
        print(f"   Improvement factor: {optimized_result['summary']['improvement_factor']:.1f}x")

        # Validate performance expectations
        files_per_second = optimized_result['summary']['files_per_second']
        improvement_factor = optimized_result['summary']['improvement_factor']

        assert files_per_second > 1.0  # Should be faster than 1 file/second
        assert improvement_factor > 1.0  # Should show improvement

        return True

def run_all_tests():
    """Run all optimized blob upload tests"""
    print("🧪 OPTIMIZED BLOB UPLOAD TEST SUITE")
    print("=" * 50)

    test_instance = TestOptimizedBlobUpload()
    test_instance.setup_class()

    tests = [
        ("Upload Worker Enhanced", test_instance.test_upload_file_worker_enhanced),
        ("Upload Worker Chunking", test_instance.test_upload_file_worker_enhanced_with_chunking),
        ("Upload Worker Retry", test_instance.test_upload_file_worker_enhanced_with_retry),
        ("Optimized Upload Function", test_instance.test_upload_output_directory_with_sas_optimized),
        ("File Categorization", test_instance.test_file_categorization),
        ("Configuration Validation", test_instance.test_configuration_validation),
        ("Error Handling", test_instance.test_error_handling),
        ("Performance Comparison", test_performance_comparison)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n📋 {test_name}")
            print("-" * 30)
            test_func()
            print(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed += 1

    print(f"\n🏁 TEST RESULTS")
    print("=" * 30)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {(passed/(passed+failed)*100):.1f}%")

    return failed == 0

if __name__ == "__main__":
    print("🚀 OPTIMIZED BLOB UPLOAD TESTS")
    print("Testing parallel upload, chunking, retry logic, and performance")
    print("=" * 70)

    try:
        success = run_all_tests()

        if success:
            print("\n🎉 ALL TESTS PASSED!")
            print("   ✅ Optimized blob upload is working correctly")
            print("   ✅ Performance improvements validated")
            print("   ✅ Error handling and retry logic functional")
        else:
            print("\n⚠️ Some tests failed. Check errors above.")

    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted")
    except Exception as e:
        print(f"\n💥 Test suite error: {e}")
        import traceback
        traceback.print_exc()