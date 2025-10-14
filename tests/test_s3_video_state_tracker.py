#!/usr/bin/env python3
"""
Comprehensive Tests for S3VideoStateTracker (utils/s3_video_state_tracker.py)
Tests state tracking, retry logic, persistence, and thread safety

Run with: python3 tests/test_s3_video_state_tracker.py
Or: pytest tests/test_s3_video_state_tracker.py -v
"""
import pytest
import os
import sys
import json
import tempfile
import shutil
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from utils.s3_video_state_tracker import (
    S3VideoStateTracker,
    initialize_s3_tracker,
    get_s3_tracker
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for state files"""
    temp_path = tempfile.mkdtemp(prefix="test_tracker_")
    yield temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def state_config():
    """Sample configuration for state tracker"""
    return {
        's3': {
            'bucket_name': 'test-bucket',
            'input_prefix': 'videos/',
            'output_prefix': 'results/',
            'region_name': 'us-east-1'
        },
        'state_tracking': {
            'max_retry_attempts': 3
        }
    }


@pytest.fixture
def tracker(temp_dir, state_config):
    """Create S3VideoStateTracker instance"""
    state_file = os.path.join(temp_dir, 'test_state.json')
    return S3VideoStateTracker(state_file, state_config)


# ============================================================================
# UNIT TESTS - INITIALIZATION
# ============================================================================

def test_state_tracker_init_new(temp_dir, state_config):
    """Create new state file"""
    state_file = os.path.join(temp_dir, 'state.json')
    tracker = S3VideoStateTracker(state_file, state_config)

    assert tracker.state is not None
    assert tracker.state['bucket_name'] == 'test-bucket'
    assert tracker.state['videos'] == {}


def test_state_tracker_init_load_existing(temp_dir, state_config):
    """Load existing state file"""
    state_file = os.path.join(temp_dir, 'state.json')

    # Create first tracker and add video
    tracker1 = S3VideoStateTracker(state_file, state_config)
    tracker1.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker1.save()

    # Load in second tracker
    tracker2 = S3VideoStateTracker(state_file, state_config)
    assert 'test.mov' in tracker2.state['videos']


def test_state_tracker_init_directory_creation(temp_dir, state_config):
    """Auto-create state directory"""
    state_file = os.path.join(temp_dir, 'nested', 'dir', 'state.json')
    tracker = S3VideoStateTracker(state_file, state_config)

    assert os.path.exists(os.path.dirname(state_file))


def test_state_tracker_init_default_statistics(temp_dir, state_config):
    """Initialize statistics to 0"""
    state_file = os.path.join(temp_dir, 'state.json')
    tracker = S3VideoStateTracker(state_file, state_config)

    stats = tracker.state['statistics']
    assert stats['total_discovered'] == 0
    assert stats['completed'] == 0
    assert stats['processing'] == 0
    assert stats['failed'] == 0
    assert stats['skipped'] == 0


# ============================================================================
# UNIT TESTS - VIDEO PROCESSING STATUS
# ============================================================================

def test_is_video_processed_new_video(tracker):
    """Return False for new video"""
    result = tracker.is_video_processed('new.mov', 'abc123')
    assert result is False


def test_is_video_processed_completed_etag_match(tracker):
    """Return True for completed with matching ETag"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_completed('test.mov', {})

    result = tracker.is_video_processed('test.mov', 'abc123')
    assert result is True


def test_is_video_processed_etag_changed(tracker):
    """Return False when ETag differs"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_completed('test.mov', {})

    # Different ETag means video changed
    result = tracker.is_video_processed('test.mov', 'xyz789')
    assert result is False


def test_is_video_processed_failed_status(tracker):
    """Return False for failed video"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')
    # Fail max times
    for _ in range(3):
        tracker.mark_failed('test.mov', 'Test error')

    result = tracker.is_video_processed('test.mov', 'abc123')
    assert result is False


# ============================================================================
# UNIT TESTS - STATE TRANSITIONS
# ============================================================================

def test_mark_discovered_new_video(tracker):
    """Mark video as discovered with metadata"""
    metadata = {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000,
        'last_modified': datetime.now()
    }
    tracker.mark_discovered('test.mov', metadata)

    assert 'test.mov' in tracker.state['videos']
    video_data = tracker.state['videos']['test.mov']
    assert video_data['status'] == 'discovered'
    assert video_data['s3_etag'] == 'abc123'


def test_mark_discovered_update_existing(tracker):
    """Update metadata for existing video"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })

    # Update with new ETag
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'xyz789',
        'size': 1100000
    })

    video_data = tracker.state['videos']['test.mov']
    assert video_data['s3_etag'] == 'xyz789'
    assert video_data['size_bytes'] == 1100000


def test_mark_downloading(tracker):
    """Transition to downloading state"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_downloading('test.mov')

    video_data = tracker.state['videos']['test.mov']
    assert video_data['status'] == 'downloading'
    assert 'download_start' in video_data


def test_mark_downloaded_with_path(tracker):
    """Transition to downloaded with local path"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_downloading('test.mov')
    tracker.mark_downloaded('test.mov', '/tmp/test.mov')

    video_data = tracker.state['videos']['test.mov']
    assert video_data['status'] == 'downloaded'
    assert video_data['local_path'] == '/tmp/test.mov'


def test_mark_processing(tracker):
    """Transition to processing state"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')

    video_data = tracker.state['videos']['test.mov']
    assert video_data['status'] == 'processing'
    assert tracker.state['statistics']['processing'] == 1


def test_mark_completed_with_results(tracker):
    """Mark completed with results dict"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')

    results = {
        's3_video_url': 'https://s3.amazonaws.com/test',
        'nsfw_detected': True,
        'minors_detected': False
    }
    tracker.mark_completed('test.mov', results)

    video_data = tracker.state['videos']['test.mov']
    assert video_data['status'] == 'completed'
    assert video_data['nsfw_detected'] is True
    assert tracker.state['statistics']['completed'] == 1


def test_mark_failed_with_error(tracker):
    """Mark as failed with error message"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')
    tracker.mark_failed('test.mov', 'Download failed')

    video_data = tracker.state['videos']['test.mov']
    assert video_data['error'] == 'Download failed'
    assert video_data['retry_count'] == 1


def test_mark_skipped_with_reason(tracker):
    """Mark as skipped with reason"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_skipped('test.mov', 'Already processed')

    video_data = tracker.state['videos']['test.mov']
    assert video_data['status'] == 'skipped'
    assert video_data['skip_reason'] == 'Already processed'
    assert tracker.state['statistics']['skipped'] == 1


# ============================================================================
# UNIT TESTS - RETRY LOGIC
# ============================================================================

def test_retry_increment_count_on_failure(tracker):
    """Increment retry_count on mark_failed"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')
    tracker.mark_failed('test.mov', 'Error 1')

    video_data = tracker.state['videos']['test.mov']
    assert video_data['retry_count'] == 1

    tracker.mark_failed('test.mov', 'Error 2')
    assert video_data['retry_count'] == 2


def test_retry_max_attempts_mark_failed(tracker):
    """Mark as failed after max retries"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')

    # Fail 3 times (max_retry_attempts = 3)
    for i in range(3):
        tracker.mark_failed('test.mov', f'Error {i+1}')

    video_data = tracker.state['videos']['test.mov']
    assert video_data['status'] == 'failed'
    assert tracker.state['statistics']['failed'] == 1


def test_retry_status_before_max(tracker):
    """Set status='retry' before max attempts"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')
    tracker.mark_failed('test.mov', 'Error')

    video_data = tracker.state['videos']['test.mov']
    assert video_data['status'] == 'retry'


def test_get_retry_videos_list(tracker):
    """Get list of videos with status='retry'"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')
    tracker.mark_failed('test.mov', 'Error')

    retry_videos = tracker.get_retry_videos()
    assert 'test.mov' in retry_videos


# ============================================================================
# UNIT TESTS - STATISTICS TRACKING
# ============================================================================

def test_statistics_discovered_increment(tracker):
    """Increment total_discovered on mark_discovered"""
    tracker.mark_discovered('test1.mov', {
        'key': 'videos/test1.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_discovered('test2.mov', {
        'key': 'videos/test2.mov',
        'etag': 'xyz789',
        'size': 2000000
    })

    assert tracker.state['statistics']['total_discovered'] == 2


def test_statistics_completed_increment(tracker):
    """Increment completed on mark_completed"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')
    tracker.mark_completed('test.mov', {})

    assert tracker.state['statistics']['completed'] == 1


def test_statistics_processing_decrement(tracker):
    """Decrement processing on completion/failure"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')
    assert tracker.state['statistics']['processing'] == 1

    tracker.mark_completed('test.mov', {})
    assert tracker.state['statistics']['processing'] == 0


# ============================================================================
# UNIT TESTS - PERSISTENCE
# ============================================================================

def test_save_state_atomic_write(temp_dir, state_config):
    """Use temp file + atomic rename"""
    state_file = os.path.join(temp_dir, 'state.json')
    tracker = S3VideoStateTracker(state_file, state_config)

    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.save()

    # Verify temp file is cleaned up
    temp_file = f"{state_file}.tmp"
    assert not os.path.exists(temp_file)


def test_save_state_json_format(temp_dir, state_config):
    """Save as formatted JSON (indent=2)"""
    state_file = os.path.join(temp_dir, 'state.json')
    tracker = S3VideoStateTracker(state_file, state_config)

    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.save()

    # Read and verify JSON formatting
    with open(state_file, 'r') as f:
        content = f.read()
        # JSON with indent=2 will have newlines
        assert '\n' in content


# ============================================================================
# UNIT TESTS - STATE MANAGEMENT
# ============================================================================

def test_reset_video_state(tracker):
    """Reset video to 'discovered' status"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')
    tracker.mark_completed('test.mov', {})

    tracker.reset_video('test.mov')

    video_data = tracker.state['videos']['test.mov']
    assert video_data['status'] == 'discovered'
    assert video_data['retry_count'] == 0


def test_cleanup_old_entries(tracker):
    """Remove old completed entries (>30 days)"""
    # Add old video
    tracker.mark_discovered('old.mov', {
        'key': 'videos/old.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_completed('old.mov', {})

    # Manually set old timestamp
    old_time = (datetime.now() - timedelta(days=40)).isoformat()
    tracker.state['videos']['old.mov']['processed_at'] = old_time

    # Add recent video
    tracker.mark_discovered('recent.mov', {
        'key': 'videos/recent.mov',
        'etag': 'xyz789',
        'size': 1000000
    })
    tracker.mark_completed('recent.mov', {})

    # Cleanup
    tracker.cleanup_old_entries(days=30)

    assert 'old.mov' not in tracker.state['videos']
    assert 'recent.mov' in tracker.state['videos']


def test_get_video_status(tracker):
    """Get status of specific video"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })

    status = tracker.get_video_status('test.mov')
    assert status == 'discovered'


def test_get_video_status_not_found(tracker):
    """Return None for non-existent video"""
    status = tracker.get_video_status('nonexistent.mov')
    assert status is None


def test_get_failed_videos(tracker):
    """Get list of failed video filenames"""
    # Add failed videos
    for i in range(3):
        filename = f'test{i}.mov'
        tracker.mark_discovered(filename, {
            'key': f'videos/{filename}',
            'etag': f'etag{i}',
            'size': 1000000
        })
        tracker.mark_processing(filename)
        # Fail max times
        for _ in range(3):
            tracker.mark_failed(filename, 'Error')

    failed_videos = tracker.get_failed_videos()
    assert len(failed_videos) == 3


def test_get_statistics(tracker):
    """Get statistics snapshot"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')
    tracker.mark_completed('test.mov', {})

    stats = tracker.get_statistics()
    assert stats['total_discovered'] == 1
    assert stats['completed'] == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_state_tracker_full_video_lifecycle(tracker):
    """Discovered → Downloading → Downloaded → Processing → Completed"""
    # 1. Discovered
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    assert tracker.get_video_status('test.mov') == 'discovered'

    # 2. Downloading
    tracker.mark_downloading('test.mov')
    assert tracker.get_video_status('test.mov') == 'downloading'

    # 3. Downloaded
    tracker.mark_downloaded('test.mov', '/tmp/test.mov')
    assert tracker.get_video_status('test.mov') == 'downloaded'

    # 4. Processing
    tracker.mark_processing('test.mov')
    assert tracker.get_video_status('test.mov') == 'processing'

    # 5. Completed
    tracker.mark_completed('test.mov', {'result': 'success'})
    assert tracker.get_video_status('test.mov') == 'completed'


def test_state_tracker_failure_and_retry(tracker):
    """Processing → Failed → Retry → Completed"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })

    # 1. Fail first attempt
    tracker.mark_processing('test.mov')
    tracker.mark_failed('test.mov', 'Network error')
    assert tracker.get_video_status('test.mov') == 'retry'

    # 2. Succeed on retry
    tracker.mark_processing('test.mov')
    tracker.mark_completed('test.mov', {})
    assert tracker.get_video_status('test.mov') == 'completed'


def test_state_tracker_max_retries_exhausted(tracker):
    """Retry → Retry → Retry → Failed (permanent)"""
    tracker.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker.mark_processing('test.mov')

    # Fail 3 times
    for i in range(3):
        tracker.mark_failed('test.mov', f'Error {i+1}')

    assert tracker.get_video_status('test.mov') == 'failed'
    assert tracker.state['statistics']['failed'] == 1


def test_state_tracker_concurrent_access(tracker):
    """Multiple threads updating different videos"""
    def add_video(i):
        filename = f'test{i}.mov'
        tracker.mark_discovered(filename, {
            'key': f'videos/{filename}',
            'etag': f'etag{i}',
            'size': 1000000
        })
        tracker.mark_processing(filename)
        tracker.mark_completed(filename, {})

    threads = [threading.Thread(target=add_video, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 10 videos should be completed
    assert tracker.state['statistics']['completed'] == 10


def test_state_tracker_save_and_reload(temp_dir, state_config):
    """Save → Reload → Verify state preserved"""
    state_file = os.path.join(temp_dir, 'state.json')

    # Create and save
    tracker1 = S3VideoStateTracker(state_file, state_config)
    tracker1.mark_discovered('test.mov', {
        'key': 'videos/test.mov',
        'etag': 'abc123',
        'size': 1000000
    })
    tracker1.mark_processing('test.mov')
    tracker1.mark_completed('test.mov', {'result': 'success'})
    tracker1.save()

    # Reload
    tracker2 = S3VideoStateTracker(state_file, state_config)

    # Verify
    assert 'test.mov' in tracker2.state['videos']
    assert tracker2.get_video_status('test.mov') == 'completed'
    assert tracker2.state['videos']['test.mov']['result'] == 'success'


def test_initialize_global_tracker(temp_dir, state_config):
    """Initialize and get global tracker instance"""
    state_file = os.path.join(temp_dir, 'global_state.json')
    tracker = initialize_s3_tracker(state_file, state_config)

    global_tracker = get_s3_tracker()
    assert tracker is global_tracker


# ============================================================================
# MAIN - DIRECT PYTHON EXECUTION
# ============================================================================

if __name__ == "__main__":
    import pytest
    print("Running S3VideoStateTracker Tests...")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short"])
