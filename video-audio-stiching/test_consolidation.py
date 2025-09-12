

#!/usr/bin/env python3
"""Test script for consolidate_json.py consolidation pipeline.

This comprehensive test suite covers:
- Helper functions (time formatting, globalisation, blob naming)
- LabelStudio client API interactions
- ConsolidationPipeline core functionality
- Azure Blob Storage integration (mocked)
- Full end-to-end consolidation workflow

Run with: python test_consolidate_json.py
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
import yaml
from unittest.mock import Mock, patch, MagicMock
from datetime import timedelta
import os
from dotenv import load_dotenv

# Load values from .env file into environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

API_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Add the post-annotation-consolidation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "post-annotation-consolidation"))

from consolidate_json import (
    LabelStudioClient,
    ConsolidationPipeline,
    _format_time,
    _globalise_seconds,
    _master_blob_name,
    _final_blob_name
)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "post_processing_config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""

    def test_format_time(self):
        """Test time formatting function."""
        self.assertEqual(_format_time(0), "00:00:00.000")
        self.assertEqual(_format_time(3661.5), "01:01:01.500")
        self.assertEqual(_format_time(125.123), "00:02:05.123")

    def test_globalise_seconds(self):
        """Test time globalisation function."""
        # Shard 1, local time 30 -> global 30
        self.assertEqual(_globalise_seconds(30, 1, 60), 30)
        # Shard 2, local time 30 -> global 90
        self.assertEqual(_globalise_seconds(30, 2, 60), 90)
        # Shard 3, local time 15 -> global 135
        self.assertEqual(_globalise_seconds(15, 3, 60), 135)

    def test_blob_names(self):
        """Test blob name generation."""
        video_id = "test_video_123"
        self.assertEqual(_master_blob_name(video_id), "test_video_123.json")
        self.assertEqual(_final_blob_name(video_id), "test_video_123.json")


class TestLabelStudioClient(unittest.TestCase):
    """Test LabelStudio client functionality."""

    def setUp(self):
        """Set up test client."""
        self.client = LabelStudioClient(
            config["label_studio"]["server_url"],
            API_TOKEN, 
            int(config["label_studio"]["project_id"])
        )

    @patch('requests.get')
    def test_fetch_task_ids_single_page(self, mock_get):
        """Test fetching task IDs from single page."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "tasks": [{"id": 1}, {"id": 2}, {"id": 3}],
            "total": 3
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        task_ids = self.client.fetch_task_ids(1, 1)
        self.assertEqual(task_ids, [1, 2, 3])

    @patch('requests.get')
    def test_fetch_task_ids_multiple_pages(self, mock_get):
        """Test fetching task IDs across multiple pages."""
        responses = [
            {"tasks": [{"id": 1}, {"id": 2}], "total": 4},
            {"tasks": [{"id": 3}, {"id": 4}], "total": 4},
            {"tasks": [], "total": 4}
        ]
        
        mock_response = Mock()
        mock_response.json.side_effect = responses
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        task_ids = self.client.fetch_task_ids(1, 1)
        self.assertEqual(task_ids, [1, 2, 3, 4])

    @patch('requests.get')
    def test_fetch_task_detail(self, mock_get):
        """Test fetching task details."""
        expected_detail = {"id": 1, "data": {"test": "data"}}
        mock_response = Mock()
        mock_response.json.return_value = expected_detail
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        detail = self.client.fetch_task_detail(1)
        self.assertEqual(detail, expected_detail)


class TestConsolidationPipeline(unittest.TestCase):
    """Test consolidation pipeline functionality."""

    def setUp(self):
        """Set up test pipeline with mocked Azure clients."""
        with patch('consolidate_json.BlobServiceClient'):
            self.pipeline = ConsolidationPipeline(
                AZURE_STORAGE_CONNECTION_STRING,
                config["azure_storage"]["intermediate_master_json_container"],
                config["azure_storage"]["final_master_json_container"]
            )
            # Mock the blob clients
            self.pipeline.masters_client = Mock()
            self.pipeline.final_client = Mock()

    def test_validate_shard_json_valid(self):
        """Test validation of valid shard JSON."""
        valid_shard = {
            "data": {
                "video_name": "test_video",
                "shard_number": 1,
                "total_shards": 3
            },
            "predictions": [],
            "annotations": []
        }
        self.assertTrue(self.pipeline.validate_shard_json(valid_shard))

    def test_validate_shard_json_invalid(self):
        """Test validation of invalid shard JSON."""
        invalid_shard = {
            "data": {
                "video_name": "test_video"
                # Missing required fields
            },
            "predictions": "not_a_list",  # Should be list
            "annotations": []
        }
        self.assertFalse(self.pipeline.validate_shard_json(invalid_shard))

    def test_load_master_new(self):
        """Test loading a new master (doesn't exist)."""
        # Mock blob client to simulate non-existing blob
        mock_blob_client = Mock()
        mock_blob_client.exists.return_value = False
        self.pipeline.masters_client.get_blob_client.return_value = mock_blob_client

        master = self.pipeline.load_master("test_video", 5)
        expected = {
            "original_video_id": "test_video",
            "total_shards": 5,
            "consolidated_shards": [],
            "status": "in_progress",
            "shards": {}
        }
        self.assertEqual(master, expected)

    def test_load_master_existing(self):
        """Test loading an existing master."""
        existing_master = {
            "original_video_id": "test_video",
            "total_shards": 3,
            "consolidated_shards": [1],
            "status": "in_progress",
            "shards": {"shard_1": {"data": "test"}}
        }
        
        mock_blob_client = Mock()
        mock_blob_client.exists.return_value = True
        mock_download = Mock()
        mock_download.readall.return_value = json.dumps(existing_master).encode("utf-8")
        mock_blob_client.download_blob.return_value = mock_download
        self.pipeline.masters_client.get_blob_client.return_value = mock_blob_client

        master = self.pipeline.load_master("test_video")
        self.assertEqual(master, existing_master)

    def test_save_master(self):
        """Test saving master JSON."""
        master = {
            "original_video_id": "test_video",
            "total_shards": 2,
            "shards": {"shard_2": {}, "shard_1": {}}  # Unsorted
        }
        
        mock_blob_client = Mock()
        mock_blob_client.url = "http://test.com/blob"
        self.pipeline.masters_client.get_blob_client.return_value = mock_blob_client

        url = self.pipeline.save_master(master)
        self.assertEqual(url, "http://test.com/blob")
        mock_blob_client.upload_blob.assert_called_once()

    def test_globalise_annotations(self):
        """Test globalisation of annotations."""
        annotations = [
            {
                "type": "labels",
                "value": {"start": 10, "end": 20, "labels": ["activity"]}
            },
            {
                "type": "other",
                "value": {"text": "comment"}
            }
        ]
        
        result = self.pipeline._globalise_annotations(annotations, 2, 60)  # Shard 2
        
        # First annotation should have globalised times
        self.assertEqual(result[0]["value"]["start"], 70)  # 10 + 60
        self.assertEqual(result[0]["value"]["end"], 80)    # 20 + 60
        
        # Second annotation should be unchanged
        self.assertEqual(result[1]["value"]["text"], "comment")

    def test_consolidate_shard_new(self):
        """Test consolidating a new shard."""
        shard_json = {
            "data": {
                "video_name": "test_video",
                "shard_number": 1,
                "total_shards": 3
            },
            "predictions": [],
            "annotations": [
                {
                    "type": "labels",
                    "value": {"start": 5, "end": 15, "labels": ["walking"]}
                }
            ]
        }
        
        # Mock load_master to return new master
        with patch.object(self.pipeline, 'load_master') as mock_load:
            with patch.object(self.pipeline, 'save_master') as mock_save:
                mock_load.return_value = {
                    "original_video_id": "test_video",
                    "total_shards": 3,
                    "consolidated_shards": [],
                    "status": "in_progress",
                    "shards": {}
                }
                mock_save.return_value = "http://saved.com"
                
                master = self.pipeline.consolidate_shard(shard_json)
                
                # Check that shard was added
                self.assertIn("shard_1", master["shards"])
                self.assertEqual(master["consolidated_shards"], [1])
                self.assertEqual(master["status"], "in_progress")  # Not all shards yet

    def test_consolidate_shard_complete_video(self):
        """Test consolidating the final shard to complete a video."""
        shard_json = {
            "data": {
                "video_name": "test_video",
                "shard_number": 3,
                "total_shards": 3
            },
            "predictions": [],
            "annotations": []
        }
        
        # Mock load_master to return master with 2 existing shards
        with patch.object(self.pipeline, 'load_master') as mock_load:
            with patch.object(self.pipeline, 'save_master') as mock_save_master:
                with patch.object(self.pipeline, 'save_final_master') as mock_save_final:
                    with patch.object(self.pipeline, 'export_full') as mock_export:
                        mock_load.return_value = {
                            "original_video_id": "test_video",
                            "total_shards": 3,
                            "consolidated_shards": [1, 2],
                            "status": "in_progress",
                            "shards": {"shard_1": {}, "shard_2": {}}
                        }
                        mock_save_master.return_value = "http://master.com"
                        mock_save_final.return_value = "http://final.com"
                        mock_export.return_value = {"final": "data"}
                        
                        master = self.pipeline.consolidate_shard(shard_json)
                        
                        # Check that video is now complete
                        self.assertEqual(master["consolidated_shards"], [1, 2, 3])
                        self.assertEqual(master["status"], "completed")
                        
                        # Check that final master was saved
                        mock_export.assert_called_once()
                        mock_save_final.assert_called_once()

    def test_export_full(self):
        """Test exporting full master."""
        master = {
            "original_video_id": "test_video",
            "total_shards": 3,
            "consolidated_shards": [1, 2, 3],
            "status": "completed",
            "shards": {
                "shard_3": {"data": "third"},
                "shard_1": {"data": "first"},
                "shard_2": {"data": "second"}
            }
        }
        
        result = self.pipeline.prepare_final_json(master)
        
        # Check that shards are sorted
        shard_keys = list(result["shards"].keys())
        self.assertEqual(shard_keys, ["shard_1", "shard_2", "shard_3"])

    def test_report_status(self):
        """Test status reporting."""
        master = {
            "original_video_id": "test_video",
            "total_shards": 5,
            "consolidated_shards": [1, 3, 5],
            "status": "in_progress"
        }
        
        with patch.object(self.pipeline, 'load_master') as mock_load:
            mock_load.return_value = master
            
            report = self.pipeline.report_status("test_video")
            
            expected = {
                "video_id": "test_video",
                "status": "in_progress",
                "total_shards_expected": 5,
                "shards_consolidated": [1, 3, 5],
                "shards_missing": [2, 4]
            }
            
            self.assertEqual(report, expected)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_full_consolidation_workflow(self):
        """Test complete workflow from shard to final master."""
        # Create sample shard data
        shard_data = [
            {
                "data": {
                    "video_name": "integration_test",
                    "shard_number": 1,
                    "total_shards": 2
                },
                "predictions": [],
                "annotations": [
                    {
                        "type": "labels",
                        "value": {"start": 10, "end": 20, "labels": ["walking"]}
                    }
                ]
            },
            {
                "data": {
                    "video_name": "integration_test",
                    "shard_number": 2,
                    "total_shards": 2
                },
                "predictions": [],
                "annotations": [
                    {
                        "type": "labels", 
                        "value": {"start": 5, "end": 15, "labels": ["running"]}
                    }
                ]
            }
        ]
        
        with patch('consolidate_json.BlobServiceClient'):
            pipeline = ConsolidationPipeline(
                config["azure_storage"]["AZURE_STORAGE_CONNECTION_STRING"],
                config["azure_storage"]["intermediate_master_json_container"],
                config["azure_storage"]["final_master_json_container"]
            )
            pipeline.masters_client = Mock()
            pipeline.final_client = Mock()
            
            # Mock blob operations
            mock_master_blob = Mock()
            mock_master_blob.url = "http://master.url"
            pipeline.masters_client.get_blob_client.return_value = mock_master_blob
            
            mock_final_blob = Mock()
            mock_final_blob.url = "http://final.url"
            pipeline.final_client.get_blob_client.return_value = mock_final_blob
            
            # Keep track of saved master state
            saved_master = None
            
            def mock_load_master(video_id, total_shards=None):
                if saved_master is None:
                    # First time - return new master
                    return {
                        "original_video_id": video_id,
                        "total_shards": total_shards or 0,
                        "consolidated_shards": [],
                        "status": "in_progress",
                        "shards": {}
                    }
                else:
                    # Return the saved master
                    return saved_master
            
            def mock_save_master(master):
                nonlocal saved_master
                saved_master = master.copy()
                saved_master["shards"] = {k: v for k, v in master["shards"].items()}
                return "http://master.url"
            
            # Mock the load and save methods
            with patch.object(pipeline, 'load_master', side_effect=mock_load_master):
                with patch.object(pipeline, 'save_master', side_effect=mock_save_master):
                    with patch.object(pipeline, 'save_final_master', return_value="http://final.url"):
                        # Process shards one by one to simulate real workflow
                        master1 = pipeline.consolidate_shard(shard_data[0])
                        master2 = pipeline.consolidate_shard(shard_data[1])
                        
                        # Final master should be complete after processing both shards
                        self.assertEqual(master2["status"], "completed")
                        self.assertEqual(master2["consolidated_shards"], [1, 2])
                        master = master2
            
            # Check that times were globalised correctly
            shard1_ann = master["shards"]["shard_1"]["annotations"][0]
            shard2_ann = master["shards"]["shard_2"]["annotations"][0]
            
            # Shard 1 times should be unchanged (shard 1 has no offset)
            self.assertEqual(shard1_ann["value"]["start"], 10)
            self.assertEqual(shard1_ann["value"]["end"], 20)
            
            # Shard 2 times should be offset by 60 seconds
            self.assertEqual(shard2_ann["value"]["start"], 65)  # 5 + 60
            self.assertEqual(shard2_ann["value"]["end"], 75)    # 15 + 60


def create_sample_data():
    """Create sample test data files."""
    sample_shard = {
        "data": {
            "video_name": "sample_video",
            "shard_number": 1,
            "total_shards": 2,
            "model_results": {
                "view1": {
                    "audio": [
                        {
                            "diarization": [
                                {"start_time": 5.0, "end_time": 15.0, "speaker": "A"}
                            ]
                        }
                    ],
                    "scene": {
                        "scenes": [
                            {"start_time": 0.0, "end_time": 30.0, "description": "indoor"}
                        ]
                    }
                }
            }
        },
        "predictions": [],
        "annotations": [
            {
                "type": "labels",
                "value": {"start": 10, "end": 25, "labels": ["activity1"]}
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_shard, f, indent=2)
        return f.name


def main():
    """Run the test suite."""
    print("🧪 Testing consolidate_json.py consolidation pipeline...")
    print("=" * 60)
    
    # Create sample data for manual testing
    sample_file = create_sample_data()
    print(f"📄 Created sample data file: {sample_file}")
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Clean up
    try:
        os.unlink(sample_file)
        print(f"🧹 Cleaned up sample file: {sample_file}")
    except OSError:
        pass
    
    print("\n✅ Test suite completed!")


if __name__ == "__main__":
    main()