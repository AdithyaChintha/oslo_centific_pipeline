"""
Lightweight state tracker for S3 video processing.

Tracks individual videos (not sessions or chunks) with ETag-based deduplication.
Simple flat structure optimized for single-video processing workflow.
"""

import json
import os
import fcntl
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("S3VideoStateTracker")


class S3VideoStateTracker:
    """
    Track processing state for videos in S3 bucket.

    Features:
    - Video-level tracking (not session/chunk based)
    - ETag-based deduplication
    - Thread-safe with file locking
    - Atomic writes to prevent corruption
    - Automatic retry logic
    """

    def __init__(self, state_file: str, config: dict):
        """
        Initialize state tracker.

        Args:
            state_file: Path to state JSON file
            config: Configuration dictionary
        """
        self.state_file = os.path.abspath(state_file)
        self.config = config
        self.lock = threading.Lock()

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        # Load or initialize state
        self.state = self._load_or_initialize()

        logger.info(f"✅ S3 state tracker initialized: {self.state_file}")
        logger.info(f"   Total videos tracked: {len(self.state['videos'])}")
        logger.info(f"   Completed: {self.state['statistics']['completed']}")
        logger.info(f"   Failed: {self.state['statistics']['failed']}")

    def _load_or_initialize(self) -> dict:
        """Load existing state or create new."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    # Acquire shared lock for reading
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        state = json.load(f)
                        logger.info(f"📋 Loaded existing state: {len(state.get('videos', {}))} videos")
                        return state
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except json.JSONDecodeError as e:
                logger.error(f"❌ Corrupted state file: {e}")
                # Backup corrupted file
                backup_path = f"{self.state_file}.backup.{int(datetime.now().timestamp())}"
                os.rename(self.state_file, backup_path)
                logger.info(f"💾 Backed up corrupted file to: {backup_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load state: {e}")

        # Initialize new state
        s3_config = self.config.get('s3', {})
        return {
            "bucket_name": s3_config.get('bucket_name'),
            "input_prefix": s3_config.get('input_prefix', 'videos/'),
            "output_prefix": s3_config.get('output_prefix', 'results/'),
            "last_poll_time": None,
            "poll_count": 0,
            "videos": {},
            "statistics": {
                "total_discovered": 0,
                "completed": 0,
                "processing": 0,
                "failed": 0,
                "skipped": 0
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": None
        }

    def is_video_processed(self, filename: str, etag: str) -> bool:
        """
        Check if video already processed.

        Uses ETag for deduplication - if ETag changed, re-process.

        Args:
            filename: Video filename
            etag: S3 ETag for video

        Returns:
            True if video already processed with same ETag
        """
        with self.lock:
            if filename not in self.state['videos']:
                logger.debug(f"🆕 New video: {filename}")
                return False

            video_data = self.state['videos'][filename]

            # Check ETag match (if ETag changed, video changed)
            stored_etag = video_data.get('s3_etag')
            if stored_etag != etag:
                logger.info(f"🔄 Video changed (ETag mismatch): {filename}")
                logger.debug(f"   Old ETag: {stored_etag}")
                logger.debug(f"   New ETag: {etag}")
                return False

            # Check completion status
            status = video_data.get('status')
            if status == 'completed':
                logger.debug(f"✅ Already processed: {filename}")
                return True
            elif status == 'failed':
                logger.debug(f"❌ Previously failed: {filename} (retry_count: {video_data.get('retry_count', 0)})")
                return False
            elif status in ['processing', 'downloading', 'downloaded']:
                # In-progress videos - consider as not processed
                logger.debug(f"🔄 In progress: {filename} (status: {status})")
                return False

            return False

    def mark_discovered(self, filename: str, video_metadata: dict):
        """
        Mark video as discovered in S3.

        Args:
            filename: Video filename
            video_metadata: Dict with 'key', 'etag', 'size', 'last_modified'
        """
        with self.lock:
            if filename not in self.state['videos']:
                # Convert last_modified datetime to ISO string for JSON serialization
                last_modified = video_metadata.get('last_modified')
                if last_modified and hasattr(last_modified, 'isoformat'):
                    last_modified = last_modified.isoformat()

                self.state['videos'][filename] = {
                    "s3_key": video_metadata.get('key'),
                    "s3_etag": video_metadata.get('etag'),
                    "size_bytes": video_metadata.get('size', 0),
                    "s3_last_modified": last_modified,
                    "status": "discovered",
                    "discovered_at": datetime.now().isoformat(),
                    "downloaded_at": None,
                    "processing_start": None,
                    "processed_at": None,
                    "processing_time_seconds": None,
                    "local_path": None,
                    "output_path": None,
                    "s3_video_url": None,
                    "s3_results_path": None,
                    "labelstudio_task_id": None,
                    "nsfw_detected": None,
                    "minors_detected": None,
                    "error": None,
                    "retry_count": 0
                }
                self.state['statistics']['total_discovered'] += 1
                logger.info(f"📋 Discovered: {filename} ({video_metadata.get('size', 0) / 1024 / 1024:.1f} MB)")
            else:
                # Update metadata if video already exists (ETag might have changed)
                video_data = self.state['videos'][filename]
                video_data['s3_etag'] = video_metadata.get('etag')
                video_data['size_bytes'] = video_metadata.get('size', 0)

                # Convert last_modified datetime to ISO string for JSON serialization
                last_modified = video_metadata.get('last_modified')
                if last_modified and hasattr(last_modified, 'isoformat'):
                    last_modified = last_modified.isoformat()
                video_data['s3_last_modified'] = last_modified

                logger.debug(f"🔄 Updated metadata: {filename}")

    def mark_downloading(self, filename: str):
        """Mark video as currently downloading."""
        with self.lock:
            if filename in self.state['videos']:
                self.state['videos'][filename]['status'] = 'downloading'
                self.state['videos'][filename]['download_start'] = datetime.now().isoformat()
                logger.info(f"⬇️ Downloading: {filename}")

    def mark_downloaded(self, filename: str, local_path: str):
        """
        Mark video as downloaded.

        Args:
            filename: Video filename
            local_path: Local path where video was saved
        """
        with self.lock:
            if filename in self.state['videos']:
                video_data = self.state['videos'][filename]
                video_data['status'] = 'downloaded'
                video_data['downloaded_at'] = datetime.now().isoformat()
                video_data['local_path'] = local_path

                # Calculate download time
                if video_data.get('download_start'):
                    start = datetime.fromisoformat(video_data['download_start'])
                    end = datetime.now()
                    download_time = (end - start).total_seconds()
                    video_data['download_time_seconds'] = download_time
                    size_mb = video_data.get('size_bytes', 0) / 1024 / 1024
                    speed_mbps = size_mb / download_time if download_time > 0 else 0
                    logger.info(f"✅ Downloaded: {filename} ({size_mb:.1f} MB in {download_time:.1f}s, {speed_mbps:.1f} MB/s)")
                else:
                    logger.info(f"✅ Downloaded: {filename}")

    def mark_processing(self, filename: str):
        """Mark video as currently processing."""
        with self.lock:
            if filename in self.state['videos']:
                video_data = self.state['videos'][filename]
                old_status = video_data['status']
                video_data['status'] = 'processing'
                video_data['processing_start'] = datetime.now().isoformat()

                # Update statistics
                if old_status != 'processing':
                    self.state['statistics']['processing'] += 1

                logger.info(f"⚙️ Processing: {filename}")

    def mark_completed(self, filename: str, results: dict):
        """
        Mark video as completed.

        Args:
            filename: Video filename
            results: Processing results dict with:
                - s3_video_url: Pre-signed S3 URL
                - s3_results_path: S3 path to results
                - output_path: Local output directory
                - labelstudio_task_id: Label Studio task ID
                - nsfw_detected: Boolean
                - minors_detected: Boolean
                - (any other custom fields)
        """
        with self.lock:
            if filename in self.state['videos']:
                video_data = self.state['videos'][filename]
                old_status = video_data['status']
                video_data['status'] = 'completed'
                video_data['processed_at'] = datetime.now().isoformat()

                # Calculate processing time
                if video_data.get('processing_start'):
                    start = datetime.fromisoformat(video_data['processing_start'])
                    end = datetime.now()
                    video_data['processing_time_seconds'] = (end - start).total_seconds()
                    logger.info(f"✅ Completed: {filename} (processing time: {video_data['processing_time_seconds']:.1f}s)")
                else:
                    logger.info(f"✅ Completed: {filename}")

                # Add results
                for key, value in results.items():
                    video_data[key] = value

                # Update statistics
                self.state['statistics']['completed'] += 1
                if old_status == 'processing' and self.state['statistics']['processing'] > 0:
                    self.state['statistics']['processing'] -= 1

    def mark_failed(self, filename: str, error: str):
        """
        Mark video as failed.

        Implements retry logic based on max_retry_attempts from config.

        Args:
            filename: Video filename
            error: Error message
        """
        with self.lock:
            if filename in self.state['videos']:
                video_data = self.state['videos'][filename]
                old_status = video_data['status']
                video_data['retry_count'] = video_data.get('retry_count', 0) + 1

                max_retries = self.config.get('state_tracking', {}).get('max_retry_attempts', 3)

                if video_data['retry_count'] >= max_retries:
                    video_data['status'] = 'failed'
                    video_data['error'] = error
                    video_data['failed_at'] = datetime.now().isoformat()

                    # Update statistics
                    self.state['statistics']['failed'] += 1
                    if old_status == 'processing' and self.state['statistics']['processing'] > 0:
                        self.state['statistics']['processing'] -= 1

                    logger.error(f"❌ Failed (max retries): {filename} - {error}")
                else:
                    # Mark for retry
                    video_data['status'] = 'retry'
                    video_data['error'] = error
                    video_data['last_error_at'] = datetime.now().isoformat()

                    logger.warning(f"⚠️ Failed (retry {video_data['retry_count']}/{max_retries}): {filename} - {error}")

    def mark_skipped(self, filename: str, reason: str):
        """
        Mark video as skipped (e.g., already processed, too small, etc.).

        Args:
            filename: Video filename
            reason: Reason for skipping
        """
        with self.lock:
            if filename in self.state['videos']:
                video_data = self.state['videos'][filename]
                video_data['status'] = 'skipped'
                video_data['skip_reason'] = reason
                video_data['skipped_at'] = datetime.now().isoformat()
                self.state['statistics']['skipped'] += 1
                logger.info(f"⏭️ Skipped: {filename} - {reason}")

    def get_video_status(self, filename: str) -> Optional[str]:
        """
        Get current status of a video.

        Args:
            filename: Video filename

        Returns:
            Status string or None if not found
        """
        with self.lock:
            if filename in self.state['videos']:
                return self.state['videos'][filename].get('status')
            return None

    def get_failed_videos(self) -> List[str]:
        """
        Get list of failed video filenames.

        Returns:
            List of filenames with status='failed'
        """
        with self.lock:
            return [
                filename for filename, data in self.state['videos'].items()
                if data.get('status') == 'failed'
            ]

    def get_retry_videos(self) -> List[str]:
        """
        Get list of videos marked for retry.

        Returns:
            List of filenames with status='retry'
        """
        with self.lock:
            return [
                filename for filename, data in self.state['videos'].items()
                if data.get('status') == 'retry'
            ]

    def get_statistics(self) -> Dict:
        """
        Get current statistics.

        Returns:
            Statistics dictionary
        """
        with self.lock:
            return self.state['statistics'].copy()

    def reset_video(self, filename: str):
        """
        Reset a video's state to allow reprocessing.

        Args:
            filename: Video filename
        """
        with self.lock:
            if filename in self.state['videos']:
                video_data = self.state['videos'][filename]
                old_status = video_data.get('status')

                # Reset to discovered state
                video_data['status'] = 'discovered'
                video_data['retry_count'] = 0
                video_data['error'] = None
                video_data['processing_start'] = None
                video_data['processed_at'] = None

                # Update statistics
                if old_status == 'completed':
                    self.state['statistics']['completed'] -= 1
                elif old_status == 'failed':
                    self.state['statistics']['failed'] -= 1

                logger.info(f"🔄 Reset video: {filename}")

    def cleanup_old_entries(self, days: int = 30):
        """
        Remove completed entries older than specified days.

        Args:
            days: Age threshold in days
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=days)

        with self.lock:
            to_remove = []

            for filename, data in self.state['videos'].items():
                if data.get('status') != 'completed':
                    continue

                processed_at = data.get('processed_at')
                if not processed_at:
                    continue

                try:
                    processed_time = datetime.fromisoformat(processed_at)
                    if processed_time < cutoff_time:
                        to_remove.append(filename)
                except ValueError:
                    continue

            # Remove old entries
            for filename in to_remove:
                del self.state['videos'][filename]
                self.state['statistics']['completed'] -= 1

            if to_remove:
                logger.info(f"🗑️ Cleaned up {len(to_remove)} old entries (>{days} days)")

    def save(self):
        """
        Save state to file with atomic write and locking.

        Uses temporary file + atomic rename to prevent corruption.
        """
        with self.lock:
            try:
                # Update timestamps
                self.state['last_updated'] = datetime.now().isoformat()
                self.state['last_poll_time'] = datetime.now().isoformat()
                self.state['poll_count'] += 1

                # Atomic write with temp file
                temp_file = f"{self.state_file}.tmp"

                with open(temp_file, 'w') as f:
                    # Acquire exclusive lock for writing
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(self.state, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                # Atomic rename
                os.rename(temp_file, self.state_file)

                logger.debug(f"💾 State saved: {len(self.state['videos'])} videos")

            except Exception as e:
                logger.error(f"❌ Failed to save state: {e}")
                # Clean up temp file if exists
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

    def print_summary(self):
        """Print a summary of current state."""
        stats = self.state['statistics']

        logger.info("=" * 80)
        logger.info("📊 S3 VIDEO PROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Bucket: {self.state['bucket_name']}")
        logger.info(f"Prefix: {self.state['input_prefix']}")
        logger.info(f"Poll Count: {self.state['poll_count']}")
        logger.info(f"Last Poll: {self.state['last_poll_time']}")
        logger.info("-" * 80)
        logger.info(f"Total Discovered: {stats['total_discovered']}")
        logger.info(f"✅ Completed: {stats['completed']}")
        logger.info(f"⚙️ Processing: {stats['processing']}")
        logger.info(f"❌ Failed: {stats['failed']}")
        logger.info(f"⏭️ Skipped: {stats['skipped']}")
        logger.info("=" * 80)

        # Show failed videos if any
        failed = self.get_failed_videos()
        if failed:
            logger.info("Failed videos:")
            for filename in failed[:10]:  # Show max 10
                error = self.state['videos'][filename].get('error', 'Unknown error')
                logger.info(f"  - {filename}: {error}")
            if len(failed) > 10:
                logger.info(f"  ... and {len(failed) - 10} more")


# Convenience functions for single global instance
_global_tracker: Optional[S3VideoStateTracker] = None


def initialize_s3_tracker(state_file: str, config: dict) -> S3VideoStateTracker:
    """
    Initialize global S3 tracker instance.

    Args:
        state_file: Path to state JSON file
        config: Configuration dictionary

    Returns:
        S3VideoStateTracker instance
    """
    global _global_tracker
    _global_tracker = S3VideoStateTracker(state_file, config)
    return _global_tracker


def get_s3_tracker() -> Optional[S3VideoStateTracker]:
    """
    Get global S3 tracker instance.

    Returns:
        S3VideoStateTracker instance or None if not initialized
    """
    return _global_tracker
