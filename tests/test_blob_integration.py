#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Azure Blob Pipeline
Tests complete workflows with REAL Azure resources and video files.
Includes integration tests, end-to-end workflows, and error handling.
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

# Setup paths to import the actual functions
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# Import actual functions
try:
    from utils.blob_utils import (
        load_azure_config, create_azure_blob_client, load_video_checklist,
        update_video_status, save_video_checklist, get_pending_videos,
        poll_azure_videos, find_corresponding_audio, download_video_audio_pair,
        cleanup_downloaded_files, upload_output_directory_to_blob
    )
    from ray_pipeline_testing import (
        download_video_batch_task,
        integrated_blob_polling_and_pipeline_task
    )
except ImportError as e:
    print(f"Warning: Could not import some functions: {e}")


class TestBlobIntegrationBulletproof:
    """BULLETPROOF INTEGRATION TESTS - Complete workflows with real Azure"""
    
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
    def temp_workspace(self):
        """Create temporary workspace for integration tests"""
        workspace = tempfile.mkdtemp(prefix="integration_test_")
        
        # Create subdirectories
        os.makedirs(os.path.join(workspace, "downloads"), exist_ok=True)
        os.makedirs(os.path.join(workspace, "outputs"), exist_ok=True)
        os.makedirs(os.path.join(workspace, "configs"), exist_ok=True)
        
        yield workspace
        
        # Cleanup
        if os.path.exists(workspace):
            shutil.rmtree(workspace)
    
    @pytest.fixture
    def sample_video_files(self):
        """Real sample video files for testing"""
        video_files = [
            os.path.join(PROJECT_ROOT, "back_1440x1440.mp4"),
            os.path.join(PROJECT_ROOT, "front_1440x1440.mp4"),
            os.path.join(PROJECT_ROOT, "left_1440x1440.mp4"),
            os.path.join(PROJECT_ROOT, "right_1440x1440.mp4"),
            os.path.join(PROJECT_ROOT, "dual_fisheye.mp4")
        ]
        existing_files = [f for f in video_files if os.path.exists(f)]
        if not existing_files:
            pytest.skip("No sample video files available for integration testing")
        return existing_files

    # =============================================================================
    # INTEGRATION TESTS - Config to Client Workflow
    # =============================================================================
    
    def test_config_to_client_workflow_integration(self, real_config_path, real_azure_config):
        """INTEGRATION TEST: Complete config loading to client creation workflow"""
        print("🔗 INTEGRATION TEST: Config → Client Workflow")
        print("-" * 50)
        
        # Step 1: Load configuration
        config = load_azure_config(real_config_path)
        
        # Validate loaded config
        assert config is not None, "Config must not be None"
        assert isinstance(config, dict), "Config must be dictionary"
        assert "account-name" in config, "Config must have account-name"
        assert "account-key" in config, "Config must have account-key"
        assert config["account-name"] == "oslotestvideo", "Account name must match expected"
        
        print(f"✅ Step 1: Real config loaded successfully")
        print(f"   Config file: {real_config_path}")
        print(f"   Account: {config['account-name']}")
        print(f"   Container: {config.get('container', 'N/A')}")
        
        # Step 2: Create blob client
        client = create_azure_blob_client(config)
        
        # Validate client
        assert client is not None, "Client must not be None"
        assert isinstance(client, BlobServiceClient), f"Expected BlobServiceClient, got {type(client)}"
        assert client.url is not None, "Client URL must not be None"
        assert config["account-name"] in client.url, "URL must contain account name"
        
        print(f"✅ Step 2: Real Azure client created")
        print(f"   Client URL: {client.url}")
        
        # Step 3: Test client functionality
        container_name = config.get('container', 'instavideo')
        container_client = client.get_container_client(container_name)
        
        # Validate container client
        assert container_client is not None, "Container client must not be None"
        assert isinstance(container_client, ContainerClient), "Must be ContainerClient"
        
        # Test container operations
        try:
            container_props = container_client.get_container_properties()
            assert container_props is not None, "Container properties must exist"
            
            print(f"✅ Step 3: Container client created for '{container_name}'")
            print(f"   Container exists and accessible")
            
        except ResourceNotFoundError:
            pytest.fail(f"Container '{container_name}' not found in Azure storage")
        except Exception as e:
            pytest.fail(f"Container operations failed: {e}")
        
        print("✅ INTEGRATION TEST PASSED - Config to client workflow complete")

    # =============================================================================
    # INTEGRATION TESTS - Video Checklist Management
    # =============================================================================
    
    def test_video_checklist_management_integration(self, temp_workspace):
        """INTEGRATION TEST: Complete video checklist lifecycle"""
        print("🔗 INTEGRATION TEST: Video Checklist Management")
        print("-" * 50)
        
        checklist_path = os.path.join(temp_workspace, "test_checklist.json")
        
        try:
            # Step 1: Create new checklist
            checklist = load_video_checklist(checklist_path)
            
            # Validate new checklist
            assert checklist is not None, "Checklist must not be None"
            assert isinstance(checklist, dict), "Checklist must be dictionary"
            assert checklist["total_videos"] == 0, "New checklist should have 0 videos"
            assert checklist["completed_videos"] == 0, "New checklist should have 0 completed"
            assert "videos" in checklist, "Checklist must have videos dict"
            
            print("✅ Step 1: New checklist created")
            
            # Step 2: Add videos to checklist
            test_videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
            
            for video_name in test_videos:
                update_video_status(checklist, video_name, "pending")
            
            # Validate video additions
            assert checklist["total_videos"] == len(test_videos), "Total videos should match"
            for video_name in test_videos:
                assert video_name in checklist["videos"], f"Video {video_name} should be in checklist"
                assert checklist["videos"][video_name]["status"] == "pending", f"Video {video_name} should be pending"
            
            print(f"✅ Step 2: Added {len(test_videos)} videos to checklist")
            
            # Step 3: Update video statuses
            update_video_status(checklist, test_videos[0], "completed")
            update_video_status(checklist, test_videos[1], "processing")
            update_video_status(checklist, test_videos[2], "failed")
            
            # Validate status updates
            assert checklist["videos"][test_videos[0]]["status"] == "completed"
            assert checklist["videos"][test_videos[1]]["status"] == "processing" 
            assert checklist["videos"][test_videos[2]]["status"] == "failed"
            assert checklist["completed_videos"] == 1, "Should have 1 completed video"
            
            print("✅ Step 3: Updated video statuses")
            
            # Step 4: Save and reload checklist
            save_video_checklist(checklist_path, checklist)
            assert os.path.exists(checklist_path), "Checklist file should exist"
            
            # Reload checklist
            reloaded_checklist = load_video_checklist(checklist_path)
            
            # Validate reloaded checklist
            assert reloaded_checklist["total_videos"] == checklist["total_videos"]
            assert reloaded_checklist["completed_videos"] == checklist["completed_videos"]
            assert len(reloaded_checklist["videos"]) == len(checklist["videos"])
            
            for video_name in test_videos:
                assert reloaded_checklist["videos"][video_name]["status"] == checklist["videos"][video_name]["status"]
            
            print("✅ Step 4: Checklist saved and reloaded successfully")
            
            # Step 5: Get pending videos
            pending_videos = get_pending_videos(reloaded_checklist)
            
            # Should have no pending videos (we changed all statuses)
            expected_pending = [v for v in test_videos if reloaded_checklist["videos"][v]["status"] == "pending"]
            assert len(pending_videos) == len(expected_pending), f"Expected {len(expected_pending)} pending videos"
            
            print(f"✅ Step 5: Found {len(pending_videos)} pending videos")
            
        except NameError as e:
            if "load_video_checklist" in str(e):
                pytest.skip("Video checklist functions not available")
            else:
                raise
        
        print("✅ INTEGRATION TEST PASSED - Video checklist management complete")

    # =============================================================================
    # INTEGRATION TESTS - Blob Upload/Download Cycle
    # =============================================================================
    
    @pytest.mark.integration
    def test_blob_upload_download_integration(self, real_container_client, sample_video_files, temp_workspace):
        """INTEGRATION TEST: Complete blob upload/download cycle with real videos"""
        print("🔗 INTEGRATION TEST: Blob Upload/Download Cycle")
        print("-" * 50)
        
        if not sample_video_files:
            pytest.skip("No sample video files available")
        
        # Use first sample video
        source_video = sample_video_files[0]
        video_name = os.path.basename(source_video)
        
        # Create unique blob name for testing
        timestamp = int(time.time())
        test_blob_name = f"integration_test/{timestamp}_{video_name}"
        download_path = os.path.join(temp_workspace, "downloads", video_name)
        
        try:
            # Step 1: Upload real video file
            print(f"📤 Step 1: Uploading {video_name}")
            
            blob_client = real_container_client.get_blob_client(test_blob_name)
            
            with open(source_video, 'rb') as f:
                upload_result = blob_client.upload_blob(f, overwrite=True)
            
            assert upload_result is not None, "Upload should succeed"
            assert blob_client.exists(), "Uploaded blob should exist"
            
            # Get blob properties
            blob_props = blob_client.get_blob_properties()
            source_size = os.path.getsize(source_video)
            
            assert blob_props.size == source_size, f"Blob size {blob_props.size} should match source {source_size}"
            
            print(f"✅ Step 1: Video uploaded successfully")
            print(f"   Blob: {test_blob_name}")
            print(f"   Size: {blob_props.size / (1024*1024):.2f} MB")
            
            # Step 2: Download and verify
            print(f"📥 Step 2: Downloading {video_name}")
            
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            
            with open(download_path, 'wb') as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            
            # Verify downloaded file
            assert os.path.exists(download_path), "Downloaded file should exist"
            downloaded_size = os.path.getsize(download_path)
            assert downloaded_size == source_size, f"Downloaded size {downloaded_size} should match source {source_size}"
            
            print(f"✅ Step 2: Video downloaded successfully")
            print(f"   Downloaded to: {download_path}")
            print(f"   Size verified: {downloaded_size / (1024*1024):.2f} MB")
            
            # Step 3: Verify file integrity (basic check)
            with open(source_video, 'rb') as f1, open(download_path, 'rb') as f2:
                # Check first and last 1024 bytes
                f1_start = f1.read(1024)
                f2_start = f2.read(1024)
                assert f1_start == f2_start, "File headers should match"
                
                f1.seek(-1024, 2)  # Seek to 1024 bytes from end
                f2.seek(-1024, 2)
                f1_end = f1.read(1024)
                f2_end = f2.read(1024)
                assert f1_end == f2_end, "File endings should match"
            
            print("✅ Step 3: File integrity verified")
            
        finally:
            # Cleanup: Delete test blob
            try:
                blob_client.delete_blob()
                print("🧹 Cleanup: Test blob deleted")
            except Exception:
                print("⚠️ Cleanup: Could not delete test blob")
        
        print("✅ INTEGRATION TEST PASSED - Upload/download cycle complete")

    # =============================================================================
    # INTEGRATION TESTS - Ray Pipeline Integration  
    # =============================================================================
    
    @pytest.mark.integration
    def test_ray_pipeline_integration(self, real_blob_client, sample_video_files, temp_workspace):
        """INTEGRATION TEST: Ray pipeline with real Azure integration"""
        print("🔗 INTEGRATION TEST: Ray Pipeline Integration")
        print("-" * 50)
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            print("✅ Ray initialized for integration test")
        
        try:
            # Step 1: Test simple Ray task
            @ray.remote
            def test_azure_connection(blob_service_client, container_name):
                """Ray remote task to test Azure connection"""
                try:
                    container_client = blob_service_client.get_container_client(container_name)
                    props = container_client.get_container_properties()
                    blob_count = len(list(container_client.list_blobs(name_starts_with="test/")))
                    return {
                        "success": True,
                        "container_name": container_name,
                        "blob_count": blob_count,
                        "last_modified": str(props.last_modified)
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Execute Ray task
            result = ray.get(test_azure_connection.remote(real_blob_client, "instavideo"))
            
            assert result["success"], f"Ray Azure task failed: {result.get('error')}"
            
            print("✅ Step 1: Ray Azure connection test passed")
            print(f"   Container: {result['container_name']}")
            print(f"   Test blobs found: {result['blob_count']}")
            
            # Step 2: Test batch processing simulation
            if sample_video_files:
                @ray.remote
                def process_video_info(video_path):
                    """Ray remote task to process video information"""
                    import os
                    return {
                        "video_name": os.path.basename(video_path),
                        "size_mb": os.path.getsize(video_path) / (1024*1024),
                        "exists": os.path.exists(video_path),
                        "extension": os.path.splitext(video_path)[1]
                    }
                
                # Process multiple videos in parallel
                video_futures = [process_video_info.remote(video) for video in sample_video_files[:3]]
                video_results = ray.get(video_futures)
                
                # Validate results
                assert len(video_results) == len(sample_video_files[:3])
                for result in video_results:
                    assert result["exists"], f"Video {result['video_name']} should exist"
                    assert result["size_mb"] > 0, f"Video {result['video_name']} should have size > 0"
                
                total_size = sum(r["size_mb"] for r in video_results)
                
                print("✅ Step 2: Ray batch processing test passed")
                print(f"   Videos processed: {len(video_results)}")
                print(f"   Total size: {total_size:.2f} MB")
                
        except Exception as e:
            pytest.fail(f"Ray pipeline integration failed: {e}")
        
        print("✅ INTEGRATION TEST PASSED - Ray pipeline integration complete")

    # =============================================================================
    # INTEGRATION TESTS - End-to-End Workflow
    # =============================================================================
    
    @pytest.mark.integration
    def test_end_to_end_workflow_simulation(self, real_azure_config, real_container_client, 
                                           sample_video_files, temp_workspace):
        """INTEGRATION TEST: Simulate complete end-to-end workflow"""
        print("🔗 INTEGRATION TEST: End-to-End Workflow Simulation")
        print("-" * 50)
        
        if not sample_video_files:
            pytest.skip("No sample video files available")
        
        # Use first sample video
        test_video = sample_video_files[0]
        video_name = os.path.basename(test_video)
        
        # Create workflow directories
        workflow_dirs = {
            "input": os.path.join(temp_workspace, "input"),
            "processing": os.path.join(temp_workspace, "processing"), 
            "output": os.path.join(temp_workspace, "output"),
            "logs": os.path.join(temp_workspace, "logs")
        }
        
        for dir_path in workflow_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            # Step 1: Simulate video discovery/polling
            print("🔍 Step 1: Video Discovery")
            
            # Copy test video to input directory  
            input_video = os.path.join(workflow_dirs["input"], video_name)
            shutil.copy2(test_video, input_video)
            
            discovered_videos = [f for f in os.listdir(workflow_dirs["input"]) if f.endswith('.mp4')]
            
            assert len(discovered_videos) > 0, "Should discover at least one video"
            assert video_name in discovered_videos, f"Should discover {video_name}"
            
            print(f"✅ Discovered {len(discovered_videos)} videos")
            
            # Step 2: Simulate video processing preparation
            print("⚙️ Step 2: Processing Preparation")
            
            processing_video = os.path.join(workflow_dirs["processing"], video_name)
            shutil.move(input_video, processing_video)
            
            assert os.path.exists(processing_video), "Video should be in processing directory"
            assert not os.path.exists(input_video), "Video should be moved from input"
            
            print(f"✅ Video moved to processing: {processing_video}")
            
            # Step 3: Simulate processing (create mock output)
            print("🔄 Step 3: Processing Simulation")
            
            # Create mock processing results
            output_files = {
                "metadata.json": {"video": video_name, "duration": 30.5, "fps": 24},
                "segments.json": [{"start": 0, "end": 10}, {"start": 10, "end": 20}],
                "results.txt": f"Processing completed for {video_name}"
            }
            
            output_dir = os.path.join(workflow_dirs["output"], os.path.splitext(video_name)[0])
            os.makedirs(output_dir, exist_ok=True)
            
            for filename, content in output_files.items():
                filepath = os.path.join(output_dir, filename)
                if filename.endswith('.json'):
                    with open(filepath, 'w') as f:
                        json.dump(content, f, indent=2)
                else:
                    with open(filepath, 'w') as f:
                        f.write(content)
            
            # Verify outputs created
            output_file_count = len(os.listdir(output_dir))
            assert output_file_count == len(output_files), f"Should have {len(output_files)} output files"
            
            print(f"✅ Created {output_file_count} output files")
            
            # Step 4: Simulate result upload to blob storage
            print("📤 Step 4: Result Upload")
            
            timestamp = int(time.time())
            blob_prefix = f"integration_test_results/{timestamp}_{os.path.splitext(video_name)[0]}"
            
            uploaded_blobs = []
            
            for filename in os.listdir(output_dir):
                local_file = os.path.join(output_dir, filename)
                blob_name = f"{blob_prefix}/{filename}"
                
                blob_client = real_container_client.get_blob_client(blob_name)
                
                with open(local_file, 'rb') as f:
                    blob_client.upload_blob(f, overwrite=True)
                
                uploaded_blobs.append(blob_name)
            
            # Verify uploads
            for blob_name in uploaded_blobs:
                blob_client = real_container_client.get_blob_client(blob_name)
                assert blob_client.exists(), f"Uploaded blob should exist: {blob_name}"
            
            print(f"✅ Uploaded {len(uploaded_blobs)} result files to blob storage")
            
            # Step 5: Cleanup processing files
            print("🧹 Step 5: Cleanup")
            
            os.remove(processing_video)
            
            # Create completion log
            log_file = os.path.join(workflow_dirs["logs"], f"{video_name}.log")
            with open(log_file, 'w') as f:
                f.write(f"Processing completed: {video_name}\n")
                f.write(f"Output files: {len(output_files)}\n")
                f.write(f"Uploaded blobs: {len(uploaded_blobs)}\n")
                f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            assert os.path.exists(log_file), "Completion log should be created"
            assert not os.path.exists(processing_video), "Processing video should be cleaned up"
            
            print("✅ Cleanup completed, log file created")
            
        finally:
            # Cleanup test blobs
            print("🧹 Final Cleanup: Removing test blobs")
            try:
                # List and delete test blobs
                test_blobs = real_container_client.list_blobs(name_starts_with="integration_test_results/")
                deleted_count = 0
                for blob in test_blobs:
                    try:
                        real_container_client.delete_blob(blob.name)
                        deleted_count += 1
                    except Exception:
                        pass
                print(f"🧹 Deleted {deleted_count} test blobs")
            except Exception:
                print("⚠️ Could not cleanup all test blobs")
        
        print("✅ INTEGRATION TEST PASSED - End-to-end workflow simulation complete")

    # =============================================================================
    # STRESS AND PERFORMANCE TESTS
    # =============================================================================
    
    @pytest.mark.performance 
    def test_multiple_blob_operations_performance(self, real_container_client, temp_workspace):
        """PERFORMANCE TEST: Multiple blob operations stress test"""
        print("🚀 PERFORMANCE TEST: Multiple Blob Operations")
        print("-" * 50)
        
        import time
        
        # Create test data
        test_data_sizes = [1024, 10*1024, 100*1024]  # 1KB, 10KB, 100KB
        test_operations = []
        
        start_time = time.time()
        timestamp = int(start_time)
        
        try:
            # Upload multiple test blobs
            for i, size in enumerate(test_data_sizes):
                test_data = b'X' * size
                blob_name = f"performance_test/{timestamp}/test_blob_{i}_{size}.dat"
                
                blob_client = real_container_client.get_blob_client(blob_name)
                
                # Time individual upload
                upload_start = time.time()
                blob_client.upload_blob(test_data, overwrite=True)
                upload_time = time.time() - upload_start
                
                test_operations.append({
                    "operation": "upload",
                    "blob_name": blob_name,
                    "size": size,
                    "time": upload_time
                })
            
            # Download and verify
            for i, size in enumerate(test_data_sizes):
                blob_name = f"performance_test/{timestamp}/test_blob_{i}_{size}.dat"
                blob_client = real_container_client.get_blob_client(blob_name)
                
                # Time individual download
                download_start = time.time()
                downloaded_data = blob_client.download_blob().readall()
                download_time = time.time() - download_start
                
                assert len(downloaded_data) == size, f"Downloaded size should match {size}"
                
                test_operations.append({
                    "operation": "download",
                    "blob_name": blob_name,
                    "size": size,
                    "time": download_time
                })
            
            total_time = time.time() - start_time
            
            # Analyze performance
            upload_times = [op["time"] for op in test_operations if op["operation"] == "upload"]
            download_times = [op["time"] for op in test_operations if op["operation"] == "download"]
            
            avg_upload_time = sum(upload_times) / len(upload_times)
            avg_download_time = sum(download_times) / len(download_times)
            
            print(f"✅ Performance test completed")
            print(f"   Total operations: {len(test_operations)}")
            print(f"   Total time: {total_time:.3f}s")
            print(f"   Average upload time: {avg_upload_time:.3f}s")
            print(f"   Average download time: {avg_download_time:.3f}s")
            
            # Performance assertions
            assert avg_upload_time < 5.0, f"Upload time too slow: {avg_upload_time:.3f}s"
            assert avg_download_time < 5.0, f"Download time too slow: {avg_download_time:.3f}s"
            assert total_time < 30.0, f"Total time too slow: {total_time:.3f}s"
            
        finally:
            # Cleanup performance test blobs
            try:
                test_blobs = real_container_client.list_blobs(name_starts_with=f"performance_test/{timestamp}/")
                for blob in test_blobs:
                    try:
                        real_container_client.delete_blob(blob.name)
                    except Exception:
                        pass
            except Exception:
                pass
        
        print("✅ PERFORMANCE TEST PASSED - Multiple blob operations within acceptable time")


def test_integration_standalone():
    """Run integration tests without pytest framework"""
    print("🔗 INTEGRATION TESTS")
    print("Testing complete workflows with REAL Azure resources")
    print("=" * 70)
    
    results = []
    config_path = os.path.join(PROJECT_ROOT, "blobfuse2_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Test 1: Config to Client Integration  
    print("🎯 INTEGRATION TEST 1: Config → Client Workflow")
    print("-" * 50)
    try:
        # Load config and create client
        config = load_azure_config(config_path)
        client = create_azure_blob_client(config)
        
        # Test container operations
        container_name = config.get('container', 'instavideo')
        container_client = client.get_container_client(container_name)
        props = container_client.get_container_properties()
        
        # Test blob listing with corrected method
        blobs = []
        blob_iter = container_client.list_blobs()
        for i, blob in enumerate(blob_iter):
            if i >= 5:  # Limit to first 5 blobs
                break
            blobs.append(blob)
        
        print("✅ PASS: Config to client integration successful")
        print(f"   Account: {config['account-name']}")
        print(f"   Container: {container_name}")
        print(f"   Sample blobs: {len(blobs)}")
        if blobs:
            print(f"   First blob: {blobs[0].name}")
        
        results.append({"test": "config_client_integration", "success": True, "blobs": len(blobs)})
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        results.append({"test": "config_client_integration", "success": False, "error": str(e)})
    
    print()
    
    # Test 2: Blob Upload/Download Integration
    print("🎯 INTEGRATION TEST 2: Blob Upload/Download Cycle")
    print("-" * 50)
    try:
        # Create test content
        test_content = f"Integration test content - {time.time()}".encode()
        test_blob_name = f"integration_test/standalone_test_{int(time.time())}.txt"
        
        # Upload test blob
        blob_client = container_client.get_blob_client(test_blob_name)
        blob_client.upload_blob(test_content, overwrite=True)
        
        # Verify upload
        assert blob_client.exists(), "Uploaded blob should exist"
        
        # Download and verify
        downloaded_content = blob_client.download_blob().readall()
        assert downloaded_content == test_content, "Downloaded content should match"
        
        # Cleanup
        blob_client.delete_blob()
        
        print("✅ PASS: Upload/download cycle successful")
        print(f"   Blob: {test_blob_name}")
        print(f"   Size: {len(test_content)} bytes")
        
        results.append({"test": "upload_download_cycle", "success": True, "size": len(test_content)})
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        results.append({"test": "upload_download_cycle", "success": False, "error": str(e)})
        
        # Try to cleanup even if test failed
        try:
            blob_client.delete_blob()
        except:
            pass
    
    print()
    
    # Final Results
    successful = len([r for r in results if r['success']])
    
    print("🏁 INTEGRATION TEST RESULTS")
    print("=" * 50)
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{i}. {result['test']}: {status}")
        if result['success']:
            if 'blobs' in result:
                print(f"   Blobs found: {result['blobs']}")
            if 'size' in result:
                print(f"   Bytes processed: {result['size']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    print(f"\nIntegration Tests: {successful}/{len(results)} passed")
    
    return successful == len(results)


if __name__ == "__main__":
    print("🔗 BULLETPROOF INTEGRATION TESTING WITH REAL AZURE")
    print("Testing complete workflows with REAL Azure resources and data")
    print("=" * 70)
    
    try:
        success = test_integration_standalone()
        
        if success:
            print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        else:
            print("\n⚠️ Some integration tests failed.")
            
    except Exception as e:
        print(f"\n💥 Integration test error: {e}")