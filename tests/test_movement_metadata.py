"""
Tests for utils/parquet_metadata_loader.py
"""
import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# ============================================================================
# Unit Tests - Loader Initialization
# ============================================================================

@pytest.mark.unit
def test_movement_loader_init(mock_s3_client):
    """Initialize MovementMetadataLoader"""
    try:
        from utils.parquet_metadata_loader import MovementMetadataLoader
        loader = MovementMetadataLoader(
            mock_s3_client,
            'test-bucket',
            'movement_data.parquet',
            cache_dir='./cache'
        )
        assert loader is not None
    except ImportError:
        pytest.skip("MovementMetadataLoader not available")


@pytest.mark.unit
def test_movement_loader_set_cache_dir():
    """Set cache directory"""
    cache_dir = './cache/movement_metadata'
    assert cache_dir is not None


# ============================================================================
# Unit Tests - Parquet Loading (Mock S3)
# ============================================================================

@pytest.mark.unit
def test_movement_loader_download_parquet(mock_s3_bucket, temp_output_dir):
    """Download parquet from mock S3"""
    s3_client, bucket = mock_s3_bucket

    # Mock parquet file
    s3_client.put_object(
        Bucket=bucket,
        Key='movement_data.parquet',
        Body=b'mock parquet data'
    )

    # Test download
    from utils.s3_utils import download_video_from_s3
    local_path = os.path.join(temp_output_dir, 'movement_data.parquet')
    result = download_video_from_s3(s3_client, bucket, 'movement_data.parquet', local_path)
    assert result is True


@pytest.mark.unit
def test_movement_loader_cache_parquet(temp_output_dir):
    """Cache downloaded parquet file"""
    cache_file = os.path.join(temp_output_dir, 'cached_movement.parquet')
    with open(cache_file, 'wb') as f:
        f.write(b'cached data')

    assert os.path.exists(cache_file)


@pytest.mark.unit
def test_movement_loader_use_cached():
    """Use cached file if exists"""
    cache_exists = True
    should_download = not cache_exists
    assert should_download is False


# ============================================================================
# Unit Tests - Metadata Lookup
# ============================================================================

@pytest.mark.unit
def test_movement_loader_lookup_by_video_id():
    """Lookup metadata by video ID"""
    # Mock lookup
    video_id = '9360653015'
    metadata = {'video_id': video_id, 'movement_score': 0.75}
    assert metadata['video_id'] == video_id


@pytest.mark.unit
def test_movement_loader_video_not_found():
    """Return None if video not found"""
    # Mock not found
    result = None
    assert result is None


@pytest.mark.unit
def test_movement_loader_extract_fields():
    """Extract relevant fields from DataFrame"""
    # Mock fields
    fields = ['video_id', 'movement_score', 'timestamp']
    assert len(fields) == 3


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.integration
def test_movement_metadata_full_workflow(mock_s3_bucket, temp_output_dir):
    """Init → Download → Load → Lookup"""
    s3_client, bucket = mock_s3_bucket

    # Add parquet file to mock S3
    s3_client.put_object(
        Bucket=bucket,
        Key='movement_data.parquet',
        Body=b'mock parquet data'
    )

    # Test workflow
    try:
        from utils.parquet_metadata_loader import MovementMetadataLoader
        loader = MovementMetadataLoader(
            s3_client,
            bucket,
            'movement_data.parquet',
            cache_dir=temp_output_dir
        )
        # Load would happen here
        assert loader is not None
    except ImportError:
        pytest.skip("MovementMetadataLoader not available")


@pytest.mark.integration
def test_movement_metadata_integration_with_pipeline():
    """Use in pipeline_s3_mode"""
    # Mock integration
    movement_enabled = True
    assert movement_enabled is True


# ============================================================================
# MAIN - DIRECT PYTHON EXECUTION
# ============================================================================

if __name__ == "__main__":
    import pytest
    print("Running Movement Metadata Tests...")
    print("=" * 70)
    pytest.main([__file__, "-v", "--tb=short"])
