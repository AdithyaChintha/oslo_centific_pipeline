"""
Pytest fixtures and configuration for all tests
"""
import pytest
from moto import mock_aws
import boto3
import tempfile
import shutil
import os
import json
from datetime import datetime


@pytest.fixture
def mock_s3_client():
    """Mock S3 client using moto - NO REAL AWS OPERATIONS"""
    with mock_aws():
        client = boto3.client('s3', region_name='us-east-1')
        yield client


@pytest.fixture
def mock_s3_bucket(mock_s3_client):
    """Mock S3 bucket with test data"""
    bucket = 'test-bucket'
    mock_s3_client.create_bucket(Bucket=bucket)

    # Add test videos
    mock_s3_client.put_object(
        Bucket=bucket,
        Key='clips/00eEzUmL_9360653015/1000-1008.mov',
        Body=b'fake video content',
        Metadata={'etag': 'abc123'}
    )

    mock_s3_client.put_object(
        Bucket=bucket,
        Key='clips/00eEzUmL_9360653015/1008-1016.mov',
        Body=b'fake video content 2',
        Metadata={'etag': 'def456'}
    )

    # Add source video
    mock_s3_client.put_object(
        Bucket=bucket,
        Key='input-videos/9360653015/00eEzUmL_9360653015.mp4',
        Body=b'fake source video content',
        Metadata={'etag': 'source123'}
    )

    yield mock_s3_client, bucket


@pytest.fixture
def temp_output_dir():
    """Temporary output directory"""
    temp_dir = tempfile.mkdtemp(prefix="test_output_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_download_dir():
    """Temporary download directory"""
    temp_dir = tempfile.mkdtemp(prefix="test_download_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_s3_config():
    """Sample S3 configuration"""
    return {
        's3': {
            'bucket_name': 'test-bucket',
            'input_prefix': 'clips/',
            'output_prefix': 'results/',
            'region_name': 'us-east-1',
            'movement_metadata': {
                'enabled': False
            }
        },
        'processing': {
            'output_dir': './test_output'
        },
        'state_tracking': {
            'state_file': 's3_state.json',
            'max_retry_attempts': 3
        },
        'polling': {
            'interval_minutes': 5,
            'max_videos_per_cycle': 10
        },
        'labelstudio': {
            'auto_create_tasks': True
        }
    }


@pytest.fixture
def mock_state_tracker(temp_output_dir, sample_s3_config):
    """Mock state tracker"""
    from utils.s3_video_state_tracker import S3VideoStateTracker
    state_file = os.path.join(temp_output_dir, "test_state.json")
    tracker = S3VideoStateTracker(state_file, sample_s3_config)
    return tracker


@pytest.fixture
def sample_video_metadata():
    """Sample video metadata"""
    return {
        'filename': '1000-1008.mov',
        'key': 'clips/00eEzUmL_9360653015/1000-1008.mov',
        'size': 1024000,
        'etag': 'abc123',
        'last_modified': datetime(2025, 1, 1, 0, 0, 0),
        'folder': 'clips/00eEzUmL_9360653015'
    }


@pytest.fixture
def sample_labelstudio_task():
    """Sample Label Studio task"""
    return {
        'data': {
            'video': 'https://example.com/video.mp4',
            'video_id': '00eEzUmL_9360653015_1000-1008',
            's3_key': 'clips/00eEzUmL_9360653015/1000-1008.mov',
            'source_video_id': '9360653015',
            'clip_id': '1000-1008.mov',
            'start_ms': 1000000,
            'end_ms': 1008000,
            'duration_ms': 8000
        },
        'predictions': [],
        'meta': {}
    }


@pytest.fixture
def sample_consolidated_results():
    """Sample consolidated model results"""
    return {
        'audio': {'detected': False},
        'yolo': {'people_detected': True, 'count': 2},
        'scene': {'scene_changes': [10, 20, 30]},
        'nsfw': {'detected': False, 'confidence': 0.1},
        'motion': {'energy': 0.5},
        'face': {'faces_detected': 1, 'minors': False},
        'clap': {'detected': False},
        'signal_quality': {'blur_score': 0.2, 'black_screen': False}
    }


# Configure pytest markers
def pytest_configure(config):
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
