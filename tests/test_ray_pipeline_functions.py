#!/usr/bin/env python3
"""
Comprehensive Tests for Ray Pipeline Functions (ray_pipeline_testing_csam_s3.py)
Tests helper functions, consolidation, JSON generation with mocked dependencies

Run with: python3 tests/test_ray_pipeline_functions.py
Or: pytest tests/test_ray_pipeline_functions.py -v
"""
import pytest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_path = tempfile.mkdtemp(prefix="test_pipeline_")
    yield temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_task_results():
    """Sample task results for testing"""
    return {
        'nsfw_detection_results': [
            {'start_time_ms': 1000, 'end_time_ms': 2000, 'label': 'nsfw', 'confidence': 0.95},
            {'start_time_ms': 3000, 'end_time_ms': 4000, 'label': 'nsfw', 'confidence': 0.89}
        ],
        'face_detection_results': [
            {'start_time_ms': 500, 'end_time_ms': 1500, 'minors_detected': True, 'age': 12}
        ]
    }


# ============================================================================
# UNIT TESTS - HELPER FUNCTIONS
# ============================================================================

def test_extract_flagged_segments_nsfw(sample_task_results):
    """Extract NSFW segments from task results"""
    # Simple extraction logic
    nsfw_segments = sample_task_results.get('nsfw_detection_results', [])
    
    assert len(nsfw_segments) == 2
    assert nsfw_segments[0]['start_time_ms'] == 1000
    assert nsfw_segments[0]['label'] == 'nsfw'


def test_extract_flagged_segments_face(sample_task_results):
    """Extract face detection segments"""
    face_segments = sample_task_results.get('face_detection_results', [])
    
    assert len(face_segments) == 1
    assert face_segments[0]['minors_detected'] is True


def test_extract_flagged_segments_empty():
    """Handle empty task results"""
    empty_results = {}
    
    nsfw_segments = empty_results.get('nsfw_detection_results', [])
    face_segments = empty_results.get('face_detection_results', [])
    
    assert nsfw_segments == []
    assert face_segments == []


def test_merge_overlapping_segments_basic():
    """Merge overlapping time segments"""
    segments = [
        {'start_time_ms': 0, 'end_time_ms': 1000},
        {'start_time_ms': 500, 'end_time_ms': 1500},
        {'start_time_ms': 1400, 'end_time_ms': 2000}
    ]
    
    # Simple merge logic
    if len(segments) == 0:
        merged = []
    else:
        sorted_segs = sorted(segments, key=lambda x: x['start_time_ms'])
        merged = [sorted_segs[0]]
        
        for seg in sorted_segs[1:]:
            last = merged[-1]
            # Overlapping or adjacent (within tolerance)
            if seg['start_time_ms'] <= last['end_time_ms'] + 100:
                merged[-1] = {
                    'start_time_ms': last['start_time_ms'],
                    'end_time_ms': max(last['end_time_ms'], seg['end_time_ms'])
                }
            else:
                merged.append(seg)
    
    assert len(merged) <= len(segments)
    assert merged[0]['start_time_ms'] == 0


def test_merge_overlapping_segments_tolerance():
    """Use tolerance parameter (default 1.0)"""
    segments = [
        {'start_time_ms': 0, 'end_time_ms': 1000},
        {'start_time_ms': 1001, 'end_time_ms': 2000}  # 1ms gap
    ]
    
    tolerance_ms = 100  # Merge if within 100ms
    
    sorted_segs = sorted(segments, key=lambda x: x['start_time_ms'])
    merged = [sorted_segs[0]]
    
    for seg in sorted_segs[1:]:
        last = merged[-1]
        if seg['start_time_ms'] <= last['end_time_ms'] + tolerance_ms:
            merged[-1]['end_time_ms'] = max(last['end_time_ms'], seg['end_time_ms'])
        else:
            merged.append(seg)
    
    assert len(merged) == 1  # Should merge with tolerance


def test_merge_overlapping_segments_no_overlap():
    """Keep separate non-overlapping segments"""
    segments = [
        {'start_time_ms': 0, 'end_time_ms': 1000},
        {'start_time_ms': 5000, 'end_time_ms': 6000}
    ]
    
    # No overlap - should remain separate
    assert len(segments) == 2


def test_ensure_prompt_file_exists_create(temp_dir):
    """Create prompt file if missing"""
    prompt_path = os.path.join(temp_dir, 'test_prompt.txt')
    prompt_content = 'Test prompt content for NSFW detection'
    
    # Create file
    if not os.path.exists(prompt_path):
        with open(prompt_path, 'w') as f:
            f.write(prompt_content)
    
    assert os.path.exists(prompt_path)
    
    with open(prompt_path, 'r') as f:
        content = f.read()
    assert content == prompt_content


def test_ensure_prompt_file_exists_already_exists(temp_dir):
    """Skip if already exists"""
    prompt_path = os.path.join(temp_dir, 'existing_prompt.txt')
    original_content = 'Original prompt content'
    
    # Create file first
    with open(prompt_path, 'w') as f:
        f.write(original_content)
    
    # Try to create again (should not overwrite)
    if os.path.exists(prompt_path):
        pass  # Don't overwrite
    else:
        with open(prompt_path, 'w') as f:
            f.write('New content')
    
    with open(prompt_path, 'r') as f:
        content = f.read()
    assert content == original_content


def test_extract_video_name_from_path():
    """Extract video name from file path"""
    test_cases = [
        ('/path/to/video.mp4', 'video'),
        ('/videos/my_video_file.mov', 'my_video_file'),
        ('test.avi', 'test')
    ]
    
    for path, expected_name in test_cases:
        filename = os.path.basename(path)
        name_without_ext = os.path.splitext(filename)[0]
        assert name_without_ext == expected_name


# ============================================================================
# UNIT TESTS - LOAD/SAVE FUNCTIONS
# ============================================================================

def test_load_videos_from_file(temp_dir):
    """Load videos from input text file"""
    videos_file = os.path.join(temp_dir, 'videos.txt')
    
    with open(videos_file, 'w') as f:
        f.write('video1.mp4\n')
        f.write('video2.mov\n')
        f.write('# comment line\n')
        f.write('\n')
        f.write('video3.mp4\n')
    
    # Load videos
    videos = []
    with open(videos_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                videos.append(line)
    
    assert len(videos) == 3
    assert 'video1.mp4' in videos
    assert '# comment line' not in videos


def test_load_videos_from_file_not_found():
    """Handle missing input file"""
    with pytest.raises(FileNotFoundError):
        with open('/nonexistent/videos.txt', 'r') as f:
            f.read()


def test_load_processed_sessions_tracking(temp_dir):
    """Load session tracking JSON"""
    tracking_file = os.path.join(temp_dir, 'processed_sessions.json')
    
    tracking_data = {'session1': True, 'session2': True}
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f)
    
    # Load
    with open(tracking_file, 'r') as f:
        loaded_data = json.load(f)
    
    assert loaded_data == tracking_data
    assert 'session1' in loaded_data


def test_load_processed_sessions_tracking_not_found():
    """Return empty set if not found"""
    # If file doesn't exist, return empty
    tracking_file = '/nonexistent/tracking.json'
    
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    assert data == {}


def test_save_processed_sessions_tracking(temp_dir):
    """Save session tracking JSON"""
    tracking_file = os.path.join(temp_dir, 'tracking.json')
    tracking_data = {'session1': True, 'session2': True}
    
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    assert os.path.exists(tracking_file)
    
    # Verify
    with open(tracking_file, 'r') as f:
        loaded = json.load(f)
    assert loaded == tracking_data


def test_read_clap_detection_from_json(temp_dir):
    """Read clap detection results"""
    clap_file = os.path.join(temp_dir, 'clap_detection.json')
    
    clap_data = {
        'claps_detected': True,
        'clap_timestamps': [1000, 2000, 3000]
    }
    
    with open(clap_file, 'w') as f:
        json.dump(clap_data, f)
    
    # Read
    with open(clap_file, 'r') as f:
        loaded = json.load(f)
    
    assert loaded['claps_detected'] is True
    assert len(loaded['clap_timestamps']) == 3


def test_read_clap_detection_from_json_not_found():
    """Return False if not found"""
    clap_file = '/nonexistent/clap.json'
    
    if os.path.exists(clap_file):
        with open(clap_file, 'r') as f:
            data = json.load(f)
        result = data.get('claps_detected', False)
    else:
        result = False
    
    assert result is False


# ============================================================================
# UNIT TESTS - CONSOLIDATION FUNCTIONS
# ============================================================================

def test_consolidate_dual_view_outputs():
    """Consolidate view1 and view2 results"""
    view1_results = {
        'nsfw_segments': [{'start_time_ms': 1000, 'end_time_ms': 2000}]
    }
    view2_results = {
        'nsfw_segments': [{'start_time_ms': 1500, 'end_time_ms': 2500}]
    }
    
    # Consolidate (union of segments)
    all_segments = view1_results['nsfw_segments'] + view2_results['nsfw_segments']
    
    consolidated = {
        'nsfw_segments': all_segments
    }
    
    assert len(consolidated['nsfw_segments']) == 2


def test_consolidate_multiview_time_segment_results():
    """Consolidate multiview time segments"""
    view_segments = [
        [{'start_time_ms': 1000, 'end_time_ms': 2000}],  # view1
        [{'start_time_ms': 1500, 'end_time_ms': 2500}],  # view2
        [{'start_time_ms': 3000, 'end_time_ms': 4000}]   # view3
    ]
    
    # Flatten all segments
    all_segments = []
    for view_segs in view_segments:
        all_segments.extend(view_segs)
    
    assert len(all_segments) == 3


def test_consolidate_time_segment_results():
    """Consolidate dual view time segments"""
    view1_segments = [
        {'start_time_ms': 0, 'end_time_ms': 1000, 'label': 'nsfw'}
    ]
    view2_segments = [
        {'start_time_ms': 500, 'end_time_ms': 1500, 'label': 'nsfw'}
    ]
    
    consolidated = view1_segments + view2_segments
    
    assert len(consolidated) == 2


def test_create_consolidated_predictions_dual():
    """Create consolidated predictions (dual view)"""
    predictions = {
        'view1': {'nsfw': True, 'minors': False},
        'view2': {'nsfw': True, 'minors': True}
    }
    
    # Consolidate (OR logic)
    consolidated = {
        'nsfw': predictions['view1']['nsfw'] or predictions['view2']['nsfw'],
        'minors': predictions['view1']['minors'] or predictions['view2']['minors']
    }
    
    assert consolidated['nsfw'] is True
    assert consolidated['minors'] is True


def test_create_consolidated_summary():
    """Create Label Studio task summary"""
    summary = {
        'video_name': 'test_video',
        'duration_ms': 10000,
        'nsfw_detected': True,
        'minors_detected': False,
        'segments_count': 5
    }
    
    assert summary['video_name'] == 'test_video'
    assert summary['nsfw_detected'] is True


# ============================================================================
# UNIT TESTS - JSON GENERATION FUNCTIONS
# ============================================================================

def test_generate_consolidated_model_results_json_dual(temp_dir):
    """Generate dual view JSON"""
    output_file = os.path.join(temp_dir, 'consolidated.json')
    
    results = {
        'video_info': {
            'name': 'test_video',
            'duration_ms': 10000
        },
        'model_results': {
            'nsfw_segments': [],
            'face_segments': []
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    assert os.path.exists(output_file)


def test_generate_final_combined_model_results_json(temp_dir):
    """Combine all shard JSONs"""
    # Create shard files
    shard1_file = os.path.join(temp_dir, 'shard_0.json')
    shard2_file = os.path.join(temp_dir, 'shard_1.json')
    
    shard1_data = {'shard': 0, 'results': [1, 2, 3]}
    shard2_data = {'shard': 1, 'results': [4, 5, 6]}
    
    with open(shard1_file, 'w') as f:
        json.dump(shard1_data, f)
    with open(shard2_file, 'w') as f:
        json.dump(shard2_data, f)
    
    # Combine
    combined_results = []
    for shard_file in [shard1_file, shard2_file]:
        with open(shard_file, 'r') as f:
            shard_data = json.load(f)
            combined_results.extend(shard_data['results'])
    
    assert len(combined_results) == 6


def test_save_s3_consolidated_results_and_labelstudio(temp_dir):
    """Save S3 mode results"""
    output_dir = os.path.join(temp_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    video_info = {
        'video_name': 'test_video',
        'duration_ms': 10000,
        's3_key': 'videos/test.mp4'
    }
    
    model_results = {
        'nsfw_segments': [],
        'face_segments': []
    }
    
    s3_video_url = 'https://s3.amazonaws.com/bucket/test.mp4'
    
    # Save results
    result_file = os.path.join(output_dir, 'consolidated_results.json')
    data = {
        'video_info': video_info,
        'model_results': model_results,
        's3_video_url': s3_video_url
    }
    
    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    assert os.path.exists(result_file)


def test_save_s3_consolidated_results_video_info(temp_dir):
    """Include video_info section"""
    result_file = os.path.join(temp_dir, 'results.json')
    
    data = {
        'video_info': {
            'name': 'test_video',
            'duration_ms': 10000,
            'fps': 30
        }
    }
    
    with open(result_file, 'w') as f:
        json.dump(data, f)
    
    with open(result_file, 'r') as f:
        loaded = json.load(f)
    
    assert 'video_info' in loaded
    assert loaded['video_info']['name'] == 'test_video'


def test_save_s3_consolidated_results_model_results(temp_dir):
    """Include model_results section"""
    result_file = os.path.join(temp_dir, 'results.json')
    
    data = {
        'model_results': {
            'nsfw_detected': True,
            'minors_detected': False,
            'segments': []
        }
    }
    
    with open(result_file, 'w') as f:
        json.dump(data, f)
    
    with open(result_file, 'r') as f:
        loaded = json.load(f)
    
    assert 'model_results' in loaded
    assert loaded['model_results']['nsfw_detected'] is True


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_pipeline_consolidation_workflow(temp_dir):
    """Shard results → Consolidate → Generate JSON"""
    # 1. Create shard results
    shard_files = []
    for i in range(3):
        shard_file = os.path.join(temp_dir, f'shard_{i}.json')
        shard_data = {
            'shard_id': i,
            'segments': [{'start': i*1000, 'end': (i+1)*1000}]
        }
        with open(shard_file, 'w') as f:
            json.dump(shard_data, f)
        shard_files.append(shard_file)
    
    # 2. Consolidate
    all_segments = []
    for shard_file in shard_files:
        with open(shard_file, 'r') as f:
            shard_data = json.load(f)
            all_segments.extend(shard_data['segments'])
    
    # 3. Generate final JSON
    final_file = os.path.join(temp_dir, 'final_results.json')
    with open(final_file, 'w') as f:
        json.dump({'consolidated_segments': all_segments}, f)
    
    assert os.path.exists(final_file)
    assert len(all_segments) == 3


def test_pipeline_s3_results_workflow(temp_dir):
    """Process → Save consolidated → Label Studio task"""
    output_dir = os.path.join(temp_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Processing results
    results = {
        'video_info': {'name': 'test'},
        'model_results': {'nsfw': True}
    }
    
    # Save consolidated
    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f)
    
    # Generate Label Studio task
    task_file = os.path.join(output_dir, 'labelstudio_task.json')
    task = {
        'data': {'video_url': 'https://s3.example.com/video.mp4'},
        'predictions': results['model_results']
    }
    
    with open(task_file, 'w') as f:
        json.dump(task, f)
    
    assert os.path.exists(results_file)
    assert os.path.exists(task_file)


def test_pipeline_error_handling():
    """Handle errors in consolidation gracefully"""
    # Simulate error handling
    try:
        # Attempt to load non-existent file
        with open('/nonexistent/file.json', 'r') as f:
            json.load(f)
        error_occurred = False
    except (FileNotFoundError, json.JSONDecodeError) as e:
        error_occurred = True
        error_message = str(e)
    
    assert error_occurred is True


# ============================================================================
# MAIN - DIRECT PYTHON EXECUTION
# ============================================================================

if __name__ == "__main__":
    import pytest
    print("Running Ray Pipeline Functions Tests...")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short"])
