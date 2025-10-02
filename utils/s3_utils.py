import os
import time
from utils.logger import get_logger
from typing import List, Dict, Optional, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    from botocore.config import Config
except ImportError:
    raise ImportError(
        "boto3 is required for S3 operations. Install with: pip install boto3"
    )

logger = get_logger("S3 utils")

def create_s3_client(config: dict):
    """
    Create and configure S3 client.

    Args:
        config: Configuration dict with s3 section containing:
            - aws_access_key_id (optional, uses env/IAM if not provided)
            - aws_secret_access_key (optional)
            - region_name (optional, defaults to us-east-1)
            - endpoint_url (optional, for S3-compatible storage)

    Returns:
        boto3.client: Configured S3 client

    Raises:
        NoCredentialsError: If credentials not found

    Example:
        s3_config = {
            's3': {
                'aws_access_key_id': 'AKIAIOSFODNN7EXAMPLE',
                'aws_secret_access_key': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
                'region_name': 'us-west-2'
            }
        }
        client = create_s3_client(s3_config)
    """
    s3_config = config.get('s3', {})

    # Configure with retries and timeout
    boto_config = Config(
        retries={'max_attempts': 3, 'mode': 'adaptive'},
        connect_timeout=10,
        read_timeout=300
    )

    client_kwargs = {'config': boto_config}

    # Add credentials if provided
    if s3_config.get('aws_access_key_id'):
        client_kwargs['aws_access_key_id'] = s3_config['aws_access_key_id']
    if s3_config.get('aws_secret_access_key'):
        client_kwargs['aws_secret_access_key'] = s3_config['aws_secret_access_key']

    # Add region
    client_kwargs['region_name'] = s3_config.get('region_name', 'us-east-1')

    # Add endpoint URL for S3-compatible storage
    if s3_config.get('endpoint_url'):
        client_kwargs['endpoint_url'] = s3_config['endpoint_url']

    try:
        s3_client = boto3.client('s3', **client_kwargs)
        logger.info(f"S3 client created for region: {client_kwargs['region_name']}")
        return s3_client
    except NoCredentialsError:
        logger.error("AWS credentials not found. Set them in config or environment variables.")
        raise

def discover_videos_in_s3(
    s3_client,
    bucket: str,
    prefix: str = "",
    file_patterns: List[str] = None,
    min_size_bytes: int = 0
) -> List[Dict]:
    """
    Discover videos in S3 bucket with filtering.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: Prefix to filter objects (e.g., "videos/")
        file_patterns: List of patterns to match (e.g., ["*.mp4", "*.insv"])
        min_size_bytes: Minimum file size in bytes (default: 0)

    Returns:
        List of dicts with video metadata:
        [{
            'filename': 'video_001.mp4',
            'key': 'videos/video_001.mp4',
            'size': 15728640,
            'etag': 'abc123...',
            'last_modified': datetime object
        }, ...]

    Example:
        videos = discover_videos_in_s3(
            s3_client,
            'my-bucket',
            prefix='videos/',
            file_patterns=['*.mp4', '*.insv'],
            min_size_bytes=1048576  # 1MB minimum
        )
    """
    if file_patterns is None:
        file_patterns = ['*.mp4', '*.insv', '*.MP4', '*.INSV', '*.mov', '*.MOV']

    logger.info(f"Discovering videos in s3://{bucket}/{prefix}")

    videos = []
    continuation_token = None

    try:
        while True:
            # List objects in bucket
            list_kwargs = {
                'Bucket': bucket,
                'Prefix': prefix
            }
            if continuation_token:
                list_kwargs['ContinuationToken'] = continuation_token

            response = s3_client.list_objects_v2(**list_kwargs)

            if 'Contents' not in response:
                logger.info(f"No objects found in s3://{bucket}/{prefix}")
                break

            # Filter videos
            for obj in response['Contents']:
                key = obj['Key']
                filename = os.path.basename(key)
                size = obj['Size']
                etag = obj['ETag'].strip('"')

                # Skip if too small
                if size < min_size_bytes:
                    continue

                # Check file pattern
                if file_patterns:
                    if not any(_match_pattern(filename, pattern) for pattern in file_patterns):
                        continue

                videos.append({
                    'filename': filename,
                    'key': key,
                    'size': size,
                    'etag': etag,
                    'last_modified': obj['LastModified'],
                    'folder': os.path.dirname(key)  # Add folder path for metadata extraction
                })

            # Check if more results
            if response.get('IsTruncated'):
                continuation_token = response.get('NextContinuationToken')
            else:
                break

        logger.info(f"Discovered {len(videos)} videos in S3")
        return videos

    except ClientError as e:
        logger.error(f"Error discovering videos in S3: {e}")
        raise

def _match_pattern(filename: str, pattern: str) -> bool:
    """Simple wildcard pattern matching for file extensions."""
    if pattern.startswith('*.'):
        ext = pattern[2:]
        return filename.lower().endswith(ext.lower())
    return filename == pattern

def download_video_from_s3(
    s3_client,
    bucket: str,
    s3_key: str,
    local_path: str,
    retry_attempts: int = 3,
    timeout: int = 300
) -> bool:
    """
    Download video from S3 with retry logic.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        s3_key: S3 object key
        local_path: Local file path to save to
        retry_attempts: Number of retry attempts (default: 3)
        timeout: Download timeout in seconds (default: 300)

    Returns:
        bool: True if download successful, False otherwise

    Example:
        success = download_video_from_s3(
            s3_client,
            'my-bucket',
            'videos/video_001.mp4',
            '/tmp/video_001.mp4'
        )
    """
    # Create parent directory if needed
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    for attempt in range(1, retry_attempts + 1):
        try:
            logger.info(f"Downloading s3://{bucket}/{s3_key} to {local_path} (attempt {attempt}/{retry_attempts})")

            # Download file
            s3_client.download_file(
                bucket,
                s3_key,
                local_path
            )

            # Verify file exists and has size > 0
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                logger.info(f"Downloaded successfully: {os.path.getsize(local_path)} bytes")
                return True
            else:
                logger.warning(f"Downloaded file is empty or missing: {local_path}")

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"S3 download error (attempt {attempt}/{retry_attempts}): {error_code} - {e}")

            if attempt < retry_attempts:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download after {retry_attempts} attempts")
                return False

        except Exception as e:
            logger.error(f"Unexpected error downloading from S3: {e}")
            if attempt < retry_attempts:
                time.sleep(2 ** attempt)
            else:
                return False

    return False

def generate_presigned_url(
    s3_client,
    bucket: str,
    s3_key: str,
    expiry_days: int = 7
) -> Optional[str]:
    """
    Generate presigned URL for S3 object (for Label Studio).

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        s3_key: S3 object key
        expiry_days: URL expiration in days (default: 7)

    Returns:
        str: Presigned URL, or None if error

    Example:
        url = generate_presigned_url(
            s3_client,
            'my-bucket',
            'videos/video_001.mp4',
            expiry_days=7
        )
        # Returns: https://my-bucket.s3.amazonaws.com/videos/video_001.mp4?...
    """
    try:
        expiration_seconds = expiry_days * 24 * 60 * 60

        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': s3_key},
            ExpiresIn=expiration_seconds
        )

        logger.debug(f"Generated presigned URL for s3://{bucket}/{s3_key} (expires in {expiry_days} days)")
        return url

    except ClientError as e:
        logger.error(f"Error generating presigned URL: {e}")
        return None

def upload_file_to_s3(
    s3_client,
    local_path: str,
    bucket: str,
    s3_key: str,
    content_type: str = None,
    retry_attempts: int = 3
) -> bool:
    """
    Upload file to S3 with retry logic.

    Args:
        s3_client: boto3 S3 client
        local_path: Local file path
        bucket: S3 bucket name
        s3_key: S3 object key
        content_type: MIME type (auto-detected if None)
        retry_attempts: Number of retry attempts (default: 3)

    Returns:
        bool: True if upload successful, False otherwise

    Example:
        success = upload_file_to_s3(
            s3_client,
            '/tmp/results.json',
            'my-bucket',
            'results/video_001/results.json'
        )
    """
    if not os.path.exists(local_path):
        logger.error(f"Local file not found: {local_path}")
        return False

    # Auto-detect content type if not provided
    if content_type is None:
        content_type = _get_content_type(local_path)

    for attempt in range(1, retry_attempts + 1):
        try:
            logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key} (attempt {attempt}/{retry_attempts})")

            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type

            s3_client.upload_file(
                local_path,
                bucket,
                s3_key,
                ExtraArgs=extra_args
            )

            logger.info(f"Uploaded successfully to s3://{bucket}/{s3_key}")
            return True

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"S3 upload error (attempt {attempt}/{retry_attempts}): {error_code} - {e}")

            if attempt < retry_attempts:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to upload after {retry_attempts} attempts")
                return False

        except Exception as e:
            logger.error(f"Unexpected error uploading to S3: {e}")
            if attempt < retry_attempts:
                time.sleep(2 ** attempt)
            else:
                return False

    return False


def upload_directory_to_s3(
    s3_client,
    local_dir: str,
    bucket: str,
    s3_prefix: str,
    retry_attempts: int = 3
) -> Tuple[int, int]:
    """
    Upload entire directory to S3 recursively.

    Args:
        s3_client: boto3 S3 client
        local_dir: Local directory path
        bucket: S3 bucket name
        s3_prefix: S3 key prefix (e.g., "results/video_001/")
        retry_attempts: Number of retry attempts per file (default: 3)

    Returns:
        Tuple of (successful_uploads, failed_uploads)

    Example:
        successful, failed = upload_directory_to_s3(
            s3_client,
            '/tmp/output/video_001',
            'my-bucket',
            'results/video_001/'
        )
        print(f"Uploaded {successful} files, {failed} failures")
    """
    if not os.path.isdir(local_dir):
        logger.error(f"Local directory not found: {local_dir}")
        return 0, 0

    successful = 0
    failed = 0

    # Walk directory tree
    for root, dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)

            # Calculate relative path
            rel_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, rel_path).replace('\\', '/')

            # Upload file
            if upload_file_to_s3(s3_client, local_path, bucket, s3_key, retry_attempts=retry_attempts):
                successful += 1
            else:
                failed += 1

    logger.info(f"Directory upload complete: {successful} successful, {failed} failed")
    return successful, failed


def _get_content_type(filename: str) -> str:
    """Auto-detect MIME type from file extension."""
    ext = os.path.splitext(filename)[1].lower()

    content_types = {
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.insv': 'application/octet-stream',  
        '.json': 'application/json',
        '.jsonl': 'application/x-ndjson',
        '.txt': 'text/plain',
        '.csv': 'text/csv',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg'
    }

    return content_types.get(ext, 'application/octet-stream')



def upload_results_to_s3(
    s3_client,
    bucket: str,
    local_dir: str,
    s3_prefix: str,
    retry_attempts: int = 3
) -> bool:
    """
    Upload processing results to S3.

    Wrapper function for compatibility with pipeline_s3_mode().
    Calls upload_directory_to_s3() internally.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        local_dir: Local directory path with processing results
        s3_prefix: S3 key prefix (e.g., "results/video_001/")
        retry_attempts: Number of retry attempts per file (default: 3)

    Returns:
        bool: True if all uploads successful, False if any failures

    Example:
        success = upload_results_to_s3(
            s3_client,
            'my-bucket',
            '/tmp/output/video_001',
            'results/video_001/'
        )
    """
    successful, failed = upload_directory_to_s3(
        s3_client, local_dir, bucket, s3_prefix, retry_attempts
    )

    if failed > 0:
        logger.warning(f"Upload completed with {failed} failures out of {successful + failed} files")
        return False

    logger.info(f"All {successful} files uploaded successfully")
    return True

def parse_clip_metadata_from_s3_key(s3_key: str, input_prefix: str) -> dict:
    """
    Parse clip metadata from S3 key format: video_name/1000-1008.mov

    Args:
        s3_key: Full S3 key (e.g., "input-videos/video_name/1000-1008.mov")
        input_prefix: S3 prefix to strip (e.g., "input-videos/")

    Returns:
        dict with:
            - source_video_id: video folder name
            - clip_id: "1000-1008"
            - start_ms: 1000
            - end_ms: 1008
            - duration_ms: 8
            - display_filename: "video_name_1000-1008.mov"

    Raises:
        ValueError: If S3 key format is invalid

    Example:
        >>> parse_clip_metadata_from_s3_key(
        ...     "input-videos/beach_video/1000-1008.mov",
        ...     "input-videos/"
        ... )
        {
            'source_video_id': 'beach_video',
            'clip_id': '1000-1008',
            'start_ms': 1000,
            'end_ms': 1008,
            'duration_ms': 8,
            'display_filename': 'beach_video_1000-1008.mov'
        }
    """
    # Remove input prefix to get relative path
    relative_key = s3_key[len(input_prefix):] if s3_key.startswith(input_prefix) else s3_key

    # Strip leading/trailing slashes to normalize the path
    relative_key = relative_key.strip('/')

    # Split into folder and filename
    # Expected format: video_name/1000-1008.mov
    parts = relative_key.split('/')

    if parts and parts[0].lower() == 'clips':
          parts = parts[1:]

    if len(parts) < 2:
        raise ValueError(f"Invalid S3 key format. Expected 'video_name/clip.mov', got: {s3_key}")

    # Extract source video ID (folder name)
    source_video_id = parts[0]  # e.g., "beach_video"

    # Extract filename
    filename = parts[-1]  # e.g., "1000-1008.mov"

    # Parse clip_id from filename (remove extension)
    clip_id = os.path.splitext(filename)[0]  # e.g., "1000-1008"
    file_ext = os.path.splitext(filename)[1]  # e.g., ".mov"

    # Parse start_ms and end_ms from clip_id
    # Expected format: start-end (e.g., "1000-1008")
    clip_parts = clip_id.split('-')
    if len(clip_parts) != 2:
        raise ValueError(f"Invalid clip_id format. Expected 'start-end', got: {clip_id}")

    try:
        start_ms = int(clip_parts[0]) * 1000
        end_ms = int(clip_parts[1]) * 1000
    except ValueError as e:
        raise ValueError(f"Failed to parse start/end times from clip_id '{clip_id}': {e}")

    # Calculate duration
    duration_ms = end_ms - start_ms

    # Generate display filename: video_name_1000-1008.mov
    display_filename = f"{source_video_id}_{clip_id}{file_ext}"

    return {
        'source_video_id': source_video_id,
        'clip_id': clip_id,
        'start_ms': start_ms,
        'end_ms': end_ms,
        'duration_ms': duration_ms,
        'display_filename': display_filename
    }

def find_source_video_in_s3(s3_client, bucket: str, source_prefix: str,
                           source_video_id: str, video_filename: str) -> dict:
    """
    Find the original source video file in S3.

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        source_prefix: Prefix where source videos are stored (e.g., "input-videos")
        source_video_id: Full video ID with hash (e.g., "00eEzUmL_9360653015")
        video_filename: Expected filename (e.g., "00eEzUmL_9360653015.mp4")

    Returns:
        dict with:
            - 'found': bool - Whether source video was found
            - 'source_s3_key': str - Full S3 key to source video (if found)
            - 'folder_name': str - Folder name searched (video ID without hash)
            - 'error': str - Error message (if not found)

    Logic:
        1. Extract folder name by removing hash prefix from video ID
           - "00eEzUmL_9360653015" → "9360653015"
        2. Construct expected S3 path:
           - "{source_prefix}/{folder_name}/{video_filename}"
        3. Check if file exists in S3
    """
    import logging
    logger = logging.getLogger(__name__)

    # Extract folder name - remove hash prefix (first underscore and everything before it)
    # "00eEzUmL_9360653015" → "9360653015"
    if '_' in source_video_id:
        folder_name = source_video_id.split('_', 1)[1]  # Take everything after first underscore
    else:
        # Fallback if no underscore found
        folder_name = source_video_id

    # Construct expected S3 key
    source_prefix = source_prefix.rstrip('/')
    expected_key = f"{source_prefix}/{folder_name}/{video_filename}"

    logger.info(f"Looking for source video: s3://{bucket}/{expected_key}")

    try:
        # Check if object exists
        s3_client.head_object(Bucket=bucket, Key=expected_key)

        logger.info(f"✅ Found source video: {expected_key}")
        return {
            'found': True,
            'source_s3_key': expected_key,
            'folder_name': folder_name,
            'error': None
        }

    except s3_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            error_msg = f"Source video not found: {expected_key}"
        else:
            error_msg = f"Error checking source video: {e}"

        logger.warning(f"⚠️ {error_msg}")
        return {
            'found': False,
            'source_s3_key': None,
            'folder_name': folder_name,
            'error': error_msg
        }