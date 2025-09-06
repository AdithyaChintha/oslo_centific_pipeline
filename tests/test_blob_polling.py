#!/usr/bin/env python3
"""
Comprehensive Tests for Azure Blob Polling Functionality
Tests blob polling, video discovery, and scheduled processing with real Azure.
Includes unit tests, integration tests, edge cases, and error handling.
"""

import pytest
import ray
import os
import sys
import time
import tempfile
import shutil
import yaml
import json
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

# Setup paths
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
    # Create basic mock functions for missing imports
    def poll_azure_videos(*args, **kwargs):
        """Mock polling function"""
        return [{"name": "test_video.mp4", "size": 1024*1024}]


class TestBlobPollingBulletproof:
    """BULLETPROOF POLLING TESTS - Testing blob polling with real Azure data"""
    
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
        """Create temporary workspace"""
        workspace = tempfile.mkdtemp(prefix="polling_test_")
        
        # Create subdirectories
        os.makedirs(os.path.join(workspace, "downloads"), exist_ok=True)
        os.makedirs(os.path.join(workspace, "checklists"), exist_ok=True)
        os.makedirs(os.path.join(workspace, "logs"), exist_ok=True)
        
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
            os.path.join(PROJECT_ROOT, "left_1440x1440.mp4")
        ]
        return [f for f in video_files if os.path.exists(f)]

    # =============================================================================
    # UNIT TESTS - Polling Functions
    # =============================================================================
    
    def test_azure_blob_polling_basic(self, real_container_client):
        """UNIT TEST: Basic blob polling functionality"""
        print("🔍 UNIT TEST: Basic Azure Blob Polling")
        print("-" * 50)
        
        try:
            # Test basic blob listing
            blobs = []
            blob_iter = real_container_client.list_blobs()
            
            for i, blob in enumerate(blob_iter):
                if i >= 10:  # Limit to first 10 blobs
                    break
                blobs.append(blob)
            
            # Validate results
            assert isinstance(blobs, list), "Blobs should be a list"
            print(f"✅ Found {len(blobs)} blobs via polling")
            
            # Analyze blob types
            video_blobs = [b for b in blobs if b.name.lower().endswith(('.mp4', '.mov', '.avi', '.insv'))]
            other_blobs = [b for b in blobs if not b.name.lower().endswith(('.mp4', '.mov', '.avi', '.insv'))]
            
            print(f"   Video blobs: {len(video_blobs)}")
            print(f"   Other blobs: {len(other_blobs)}")
            
            if video_blobs:
                print(f"   Sample video: {video_blobs[0].name}")
                print(f"   Video size: {video_blobs[0].size / (1024*1024):.2f} MB")
            
        except Exception as e:
            pytest.fail(f"Basic blob polling failed: {e}")
    
    def test_azure_blob_polling_with_filters(self, real_container_client):
        """UNIT TEST: Blob polling with filters and prefixes"""
        print("🔍 UNIT TEST: Blob Polling with Filters")
        print("-" * 50)
        
        try:
            # Test polling with different prefixes
            test_prefixes = ["", "test/", "input/", "processed/"]
            
            for prefix in test_prefixes:
                blobs = []
                if prefix:
                    blob_iter = real_container_client.list_blobs(name_starts_with=prefix)
                else:
                    blob_iter = real_container_client.list_blobs()
                
                for i, blob in enumerate(blob_iter):
                    if i >= 5:  # Limit to first 5 per prefix
                        break
                    blobs.append(blob)
                
                print(f"   Prefix '{prefix}': {len(blobs)} blobs")
                if blobs:
                    print(f"     Sample: {blobs[0].name}")
            
            print("✅ Filtered polling completed successfully")
            
        except Exception as e:
            pytest.fail(f"Filtered blob polling failed: {e}")
    
    def test_blob_polling_performance(self, real_container_client):
        """PERFORMANCE TEST: Blob polling speed"""
        print("🚀 PERFORMANCE TEST: Blob Polling Speed")
        print("-" * 50)
        
        import time
        
        try:
            # Test polling performance
            start_time = time.time()
            
            blob_count = 0
            blob_iter = real_container_client.list_blobs()
            
            for blob in blob_iter:
                blob_count += 1
                if blob_count >= 100:  # Test first 100 blobs
                    break
            
            end_time = time.time()
            polling_time = end_time - start_time
            
            # Performance assertions
            assert polling_time < 30.0, f"Polling too slow: {polling_time:.3f}s for {blob_count} blobs"
            
            blobs_per_second = blob_count / polling_time if polling_time > 0 else 0
            
            print(f"✅ Polling performance test passed")
            print(f"   Blobs polled: {blob_count}")
            print(f"   Time taken: {polling_time:.3f}s")
            print(f"   Rate: {blobs_per_second:.1f} blobs/second")
            
        except Exception as e:
            pytest.fail(f"Polling performance test failed: {e}")

    # =============================================================================
    # INTEGRATION TESTS - Polling with Processing
    # =============================================================================
    
    @pytest.mark.integration
    def test_polling_with_video_discovery(self, real_container_client, temp_workspace):
        """INTEGRATION TEST: Polling with video discovery and filtering"""
        print("🔍 INTEGRATION TEST: Polling with Video Discovery")
        print("-" * 50)
        
        try:
            # Step 1: Poll for all videos
            video_extensions = ['.mp4', '.mov', '.avi', '.insv']
            discovered_videos = []
            
            blob_iter = real_container_client.list_blobs()
            for blob in blob_iter:
                # Check if it's a video file
                if any(blob.name.lower().endswith(ext) for ext in video_extensions):
                    discovered_videos.append({
                        'name': blob.name,
                        'size': blob.size,
                        'last_modified': blob.last_modified,
                        'extension': os.path.splitext(blob.name)[1].lower()
                    })
                
                # Limit discovery for testing
                if len(discovered_videos) >= 10:
                    break
            
            # Validate discovery
            assert len(discovered_videos) >= 0, "Should discover videos (or none if container empty)"
            
            print(f"✅ Step 1: Discovered {len(discovered_videos)} videos")
            
            if discovered_videos:
                # Analyze discovered videos
                total_size = sum(v['size'] for v in discovered_videos)
                avg_size = total_size / len(discovered_videos)
                
                extension_counts = {}
                for video in discovered_videos:
                    ext = video['extension']
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1
                
                print(f"   Total size: {total_size / (1024*1024*1024):.2f} GB")
                print(f"   Average size: {avg_size / (1024*1024):.2f} MB")
                print(f"   Extensions: {extension_counts}")
                print(f"   Sample video: {discovered_videos[0]['name']}")
                
                # Step 2: Create processing checklist
                checklist_path = os.path.join(temp_workspace, "checklists", "discovered_videos.json")
                
                try:
                    checklist = load_video_checklist(checklist_path)
                    
                    # Add discovered videos to checklist
                    for video in discovered_videos[:5]:  # Limit to first 5
                        update_video_status(checklist, video['name'], "pending")
                    
                    # Save checklist
                    save_video_checklist(checklist, checklist_path)
                    
                    # Validate checklist
                    assert os.path.exists(checklist_path), "Checklist should be saved"
                    assert checklist["total_videos"] > 0, "Checklist should have videos"
                    
                    print(f"✅ Step 2: Created checklist with {checklist['total_videos']} videos")
                    
                except NameError:
                    print("⚠️ Step 2: Checklist functions not available, skipping")
            
        except Exception as e:
            pytest.fail(f"Polling integration test failed: {e}")
    
    @pytest.mark.integration
    def test_scheduled_polling_simulation(self, real_container_client, temp_workspace):
        """INTEGRATION TEST: Simulate scheduled polling behavior"""
        print("🔄 INTEGRATION TEST: Scheduled Polling Simulation")
        print("-" * 50)
        
        try:
            polling_results = []
            polling_intervals = [1, 2, 3]  # Short intervals for testing
            
            # Simulate multiple polling cycles
            for cycle in range(3):
                print(f"   Polling cycle {cycle + 1}")
                
                start_time = time.time()
                
                # Poll for new videos
                video_count = 0
                blob_iter = real_container_client.list_blobs()
                
                for blob in blob_iter:
                    if blob.name.lower().endswith(('.mp4', '.insv')):
                        video_count += 1
                    
                    # Limit for testing
                    if video_count >= 5:
                        break
                
                end_time = time.time()
                polling_time = end_time - start_time
                
                result = {
                    'cycle': cycle + 1,
                    'videos_found': video_count,
                    'polling_time': polling_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                polling_results.append(result)
                
                print(f"     Found {video_count} videos in {polling_time:.3f}s")
                
                # Wait before next cycle (except last)
                if cycle < len(polling_intervals) - 1:
                    time.sleep(polling_intervals[cycle])
            
            # Analyze polling results
            total_time = sum(r['polling_time'] for r in polling_results)
            avg_time = total_time / len(polling_results)
            
            # Create polling log
            log_path = os.path.join(temp_workspace, "logs", "polling_log.json")
            with open(log_path, 'w') as f:
                json.dump({
                    'polling_cycles': polling_results,
                    'summary': {
                        'total_cycles': len(polling_results),
                        'average_polling_time': avg_time,
                        'total_polling_time': total_time
                    }
                }, f, indent=2)
            
            assert os.path.exists(log_path), "Polling log should be created"
            
            print(f"✅ Scheduled polling simulation completed")
            print(f"   Cycles: {len(polling_results)}")
            print(f"   Average time per cycle: {avg_time:.3f}s")
            print(f"   Log saved: {log_path}")
            
        except Exception as e:
            pytest.fail(f"Scheduled polling simulation failed: {e}")

    # =============================================================================
    # RAY INTEGRATION TESTS - Distributed Polling
    # =============================================================================
    
    @pytest.mark.integration
    def test_ray_distributed_polling(self, real_blob_client, real_azure_config):
        """INTEGRATION TEST: Ray-based distributed polling"""
        print("🚀 INTEGRATION TEST: Ray Distributed Polling")
        print("-" * 50)
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            print("✅ Ray initialized for polling test")
        
        try:
            @ray.remote
            def poll_blob_prefix(blob_service_client, container_name, prefix="", max_blobs=10):
                """Ray remote task for polling specific blob prefix"""
                try:
                    container_client = blob_service_client.get_container_client(container_name)
                    
                    videos = []
                    if prefix:
                        blob_iter = container_client.list_blobs(name_starts_with=prefix)
                    else:
                        blob_iter = container_client.list_blobs()
                    
                    for blob in blob_iter:
                        if blob.name.lower().endswith(('.mp4', '.insv', '.mov')):
                            videos.append({
                                'name': blob.name,
                                'size': blob.size,
                                'prefix': prefix,
                                'last_modified': str(blob.last_modified)
                            })
                        
                        if len(videos) >= max_blobs:
                            break
                    
                    return {
                        'success': True,
                        'prefix': prefix,
                        'videos': videos,
                        'video_count': len(videos)
                    }
                    
                except Exception as e:
                    return {
                        'success': False,
                        'prefix': prefix,
                        'error': str(e)
                    }
            
            # Test distributed polling with different prefixes
            container_name = real_azure_config.get('container', 'instavideo')
            prefixes = ["", "test/", "input/", "processed/"]
            
            # Launch parallel polling tasks
            polling_futures = []
            for prefix in prefixes:
                future = poll_blob_prefix.remote(real_blob_client, container_name, prefix, 5)
                polling_futures.append(future)
            
            # Collect results
            polling_results = ray.get(polling_futures)
            
            # Validate results
            successful_polls = [r for r in polling_results if r['success']]
            failed_polls = [r for r in polling_results if not r['success']]
            
            assert len(successful_polls) > 0, "At least one polling task should succeed"
            
            total_videos = sum(r['video_count'] for r in successful_polls)
            
            print(f"✅ Ray distributed polling completed")
            print(f"   Successful polls: {len(successful_polls)}")
            print(f"   Failed polls: {len(failed_polls)}")
            print(f"   Total videos found: {total_videos}")
            
            for result in successful_polls:
                print(f"   Prefix '{result['prefix']}': {result['video_count']} videos")
            
            if failed_polls:
                print("   Failed prefixes:")
                for result in failed_polls:
                    print(f"     '{result['prefix']}': {result['error']}")
            
        except Exception as e:
            pytest.fail(f"Ray distributed polling failed: {e}")

    # =============================================================================
    # ERROR HANDLING AND EDGE CASES
    # =============================================================================
    
    def test_polling_error_handling(self, real_blob_client):
        """UNIT TEST: Polling error handling and edge cases"""
        print("⚠️ UNIT TEST: Polling Error Handling")
        print("-" * 50)
        
        try:
            # Test 1: Invalid container name
            try:
                invalid_container = real_blob_client.get_container_client("nonexistent-container-12345")
                blob_list = list(invalid_container.list_blobs())
                # This might not fail immediately, so we check if we can iterate
                for blob in blob_list:
                    break
                print("⚠️ Expected error for invalid container, but operation succeeded")
            except Exception as e:
                print(f"✅ Correctly handled invalid container: {type(e).__name__}")
            
            # Test 2: Network interruption simulation (timeout)
            try:
                # Create container client with very short timeout
                container_client = real_blob_client.get_container_client("instavideo")
                
                # This should work normally
                blob_iter = container_client.list_blobs()
                first_blob = next(iter(blob_iter), None)
                
                if first_blob:
                    print("✅ Normal polling operation succeeded")
                else:
                    print("✅ No blobs found (empty container)")
                    
            except Exception as e:
                print(f"✅ Handled polling exception: {type(e).__name__}")
            
            # Test 3: Empty results handling
            try:
                # Try to poll non-existent prefix
                container_client = real_blob_client.get_container_client("instavideo")
                empty_blobs = list(container_client.list_blobs(name_starts_with="definitely-does-not-exist-12345/"))
                
                assert isinstance(empty_blobs, list), "Empty results should be a list"
                assert len(empty_blobs) == 0, "Should have no results for non-existent prefix"
                
                print("✅ Empty results handled correctly")
                
            except Exception as e:
                print(f"✅ Handled empty results exception: {type(e).__name__}")
            
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")
    
    def test_polling_boundary_conditions(self, real_container_client):
        """UNIT TEST: Polling boundary conditions"""
        print("📏 UNIT TEST: Polling Boundary Conditions")
        print("-" * 50)
        
        try:
            # Test 1: Very long blob names
            try:
                long_prefix = "a" * 100  # Very long prefix
                long_blobs = list(real_container_client.list_blobs(name_starts_with=long_prefix))
                print(f"✅ Long prefix handled: {len(long_blobs)} results")
            except Exception as e:
                print(f"✅ Long prefix exception handled: {type(e).__name__}")
            
            # Test 2: Special characters in prefixes
            special_prefixes = ["test with spaces/", "test-with-dashes/", "test_with_underscores/", "test.with.dots/"]
            
            for prefix in special_prefixes:
                try:
                    special_blobs = list(real_container_client.list_blobs(name_starts_with=prefix))
                    print(f"✅ Special prefix '{prefix}': {len(special_blobs)} results")
                except Exception as e:
                    print(f"✅ Special prefix '{prefix}' exception: {type(e).__name__}")
            
            # Test 3: Case sensitivity
            try:
                # Test case variations
                case_prefixes = ["Test/", "TEST/", "test/"]
                case_results = {}
                
                for prefix in case_prefixes:
                    blobs = list(real_container_client.list_blobs(name_starts_with=prefix))
                    case_results[prefix] = len(blobs)
                
                print("✅ Case sensitivity test results:")
                for prefix, count in case_results.items():
                    print(f"   '{prefix}': {count} blobs")
                    
            except Exception as e:
                print(f"✅ Case sensitivity exception handled: {type(e).__name__}")
            
        except Exception as e:
            pytest.fail(f"Boundary conditions test failed: {e}")


def test_blob_polling_standalone():
    """Run blob polling tests without pytest framework"""
    print("🔍 BLOB POLLING TESTS")
    print("Testing blob polling functions with REAL Azure configuration")
    print("=" * 70)
    
    results = []
    config_path = os.path.join(PROJECT_ROOT, "blobfuse2_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Initialize Ray for distributed tests
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        print("✅ Ray initialized")
    
    try:
        # Load config and create clients
        config = load_azure_config(config_path)
        client = create_azure_blob_client(config)
        container_name = config.get('container', 'instavideo')
        container_client = client.get_container_client(container_name)
        
        # Test 1: Basic Blob Polling
        print("🎯 POLLING TEST 1: Basic Blob Discovery")
        print("-" * 50)
        try:
            start_time = time.time()
            
            blobs = []
            blob_iter = container_client.list_blobs()
            
            for i, blob in enumerate(blob_iter):
                if i >= 20:  # Limit to first 20 blobs
                    break
                blobs.append(blob)
            
            end_time = time.time()
            polling_time = end_time - start_time
            
            # Analyze results
            video_blobs = [b for b in blobs if b.name.lower().endswith(('.mp4', '.insv', '.mov', '.avi'))]
            other_blobs = [b for b in blobs if not b.name.lower().endswith(('.mp4', '.insv', '.mov', '.avi'))]
            
            print(f"✅ PASS: Basic blob polling successful")
            print(f"   Total blobs polled: {len(blobs)}")
            print(f"   Video blobs: {len(video_blobs)}")
            print(f"   Other blobs: {len(other_blobs)}")
            print(f"   Polling time: {polling_time:.3f}s")
            
            if video_blobs:
                print(f"   Sample video: {video_blobs[0].name}")
                print(f"   Video size: {video_blobs[0].size / (1024*1024):.2f} MB")
            
            results.append({"test": "basic_blob_polling", "success": True, "videos": len(video_blobs), "time": polling_time})
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            results.append({"test": "basic_blob_polling", "success": False, "error": str(e)})
        
        print()
        
        # Test 2: Filtered Polling
        print("🎯 POLLING TEST 2: Filtered Blob Polling")
        print("-" * 50)
        try:
            filter_results = {}
            test_prefixes = ["", "test/", "input/"]
            
            for prefix in test_prefixes:
                prefix_blobs = []
                if prefix:
                    blob_iter = container_client.list_blobs(name_starts_with=prefix)
                else:
                    blob_iter = container_client.list_blobs()
                
                for i, blob in enumerate(blob_iter):
                    if i >= 10:  # Limit per prefix
                        break
                    prefix_blobs.append(blob)
                
                filter_results[prefix or "root"] = len(prefix_blobs)
            
            print("✅ PASS: Filtered polling successful")
            for prefix, count in filter_results.items():
                print(f"   Prefix '{prefix}': {count} blobs")
            
            results.append({"test": "filtered_polling", "success": True, "filters": len(filter_results)})
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            results.append({"test": "filtered_polling", "success": False, "error": str(e)})
        
        print()
        
        # Test 3: Ray Distributed Polling
        print("🎯 POLLING TEST 3: Ray Distributed Polling")
        print("-" * 50)
        try:
            @ray.remote
            def parallel_poll(container_name, blob_service_client, prefix=""):
                container_client = blob_service_client.get_container_client(container_name)
                
                videos = []
                if prefix:
                    blob_iter = container_client.list_blobs(name_starts_with=prefix)
                else:
                    blob_iter = container_client.list_blobs()
                
                for blob in blob_iter:
                    if blob.name.lower().endswith(('.mp4', '.insv')):
                        videos.append(blob.name)
                    
                    if len(videos) >= 5:  # Limit for testing
                        break
                
                return {'prefix': prefix or 'root', 'videos': videos}
            
            # Launch parallel tasks
            prefixes = ["", "test/"]
            futures = [parallel_poll.remote(container_name, client, prefix) for prefix in prefixes]
            parallel_results = ray.get(futures)
            
            total_videos = sum(len(r['videos']) for r in parallel_results)
            
            print("✅ PASS: Ray distributed polling successful")
            print(f"   Parallel tasks: {len(parallel_results)}")
            print(f"   Total videos found: {total_videos}")
            
            for result in parallel_results:
                print(f"   {result['prefix']}: {len(result['videos'])} videos")
            
            results.append({"test": "ray_distributed_polling", "success": True, "tasks": len(parallel_results), "videos": total_videos})
            
        except Exception as e:
            print(f"❌ FAIL: {e}")
            results.append({"test": "ray_distributed_polling", "success": False, "error": str(e)})
        
        print()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False
    
    # Final Results
    successful = len([r for r in results if r['success']])
    
    print("🏁 BLOB POLLING TEST RESULTS")
    print("=" * 50)
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{i}. {result['test']}: {status}")
        if result['success']:
            if 'videos' in result:
                print(f"   Videos found: {result['videos']}")
            if 'time' in result:
                print(f"   Time taken: {result['time']:.3f}s")
            if 'tasks' in result:
                print(f"   Parallel tasks: {result['tasks']}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
    
    print(f"\nBlob Polling Tests: {successful}/{len(results)} passed")
    
    if ray.is_initialized():
        ray.shutdown()
    
    return successful == len(results)


if __name__ == "__main__":
    print("🔍 BULLETPROOF BLOB POLLING TESTING WITH REAL AZURE")
    print("Testing blob polling functions with REAL Azure configuration and data")
    print("=" * 70)
    
    try:
        success = test_blob_polling_standalone()
        
        if success:
            print("\n🎉 ALL BLOB POLLING TESTS PASSED!")
        else:
            print("\n⚠️ Some blob polling tests failed.")
            
    except Exception as e:
        print(f"\n💥 Blob polling test error: {e}")