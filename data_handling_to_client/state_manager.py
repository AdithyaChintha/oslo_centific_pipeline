#!/usr/bin/env python3
"""
Enhanced State Tracking Implementation

This module provides the enhanced state tracking system to replace the simple
watermark-based approach with comprehensive daily state tracking.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict

# Azure imports
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError

# Import utilities from shared utils module
from utils import (
    now_utc, now_iso, atomic_write_json, ensure_dir,
    normalize_content_md5, parse_iso_to_utc, read_json_if_exists
)

logger = logging.getLogger("enhanced_state_tracker")


@dataclass
class VideoProcessingResult:
    """Data class for video processing results"""
    blob_name: str
    upload_blob_name: str
    fingerprint: dict
    file_size: Optional[int]
    last_modified: str
    processed_at: str
    upload_status: str  # 'success', 'failed', 'pending'
    container_id: str
    metadata_uploaded: bool
    retry_count: int
    error_details: Optional[str]
    # Timing information
    upload_start_time: Optional[str] = None
    upload_end_time: Optional[str] = None
    upload_duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class StateStorageManager:
    """Manages state storage in your own Azure blob container"""

    def __init__(self, connection_string: str, container_name: str, prefix: str = ""):
        self.conn_str = connection_string
        self.container_name = container_name
        self.prefix = prefix.rstrip("/")
        self._client = None

        if connection_string:
            svc = BlobServiceClient.from_connection_string(connection_string)
            self._client = svc.get_container_client(container_name)
            try:
                self._client.create_container()
            except Exception:
                pass

    def _get_state_blob_name(self, home_id: str) -> str:
        """Get full blob path for state file"""
        filename = f"{home_id}_state.json"
        if self.prefix:
            return f"{self.prefix}/{filename}"
        return filename

    def upload_state(self, home_id: str, state_data: dict) -> None:
        """Upload state to your blob storage"""
        if not self._client:
            return

        blob_name = self._get_state_blob_name(home_id)
        blob_client = self._client.get_blob_client(blob_name)
        data = json.dumps(state_data, indent=2, ensure_ascii=False).encode("utf-8")
        blob_client.upload_blob(data, overwrite=True)
        logger.info("Uploaded state for %s to %s", home_id, blob_name)

    def download_state(self, home_id: str) -> Optional[dict]:
        """Download state from your blob storage"""
        if not self._client:
            return None

        blob_name = self._get_state_blob_name(home_id)
        try:
            blob_client = self._client.get_blob_client(blob_name)
            stream = blob_client.download_blob()
            content = stream.content_as_text(encoding="utf-8")
            return json.loads(content)
        except ResourceNotFoundError:
            return None
        except Exception as e:
            logger.warning("Failed to download state %s: %s", blob_name, e)
            return None

class EnhancedStateStore:
    """
    Enhanced state tracking system replacing simple watermark.
    Uses your own blob storage instead of client manifests.
    """

    def __init__(self, home_id: str, local_dir: str, state_storage: Optional[StateStorageManager] = None):
        self.home_id = home_id
        self.local_dir = Path(local_dir)
        self.state_file = self.local_dir / f"{home_id}_state.json"
        self.state_storage = state_storage
        self.current_state = self.load_state()
        ensure_dir(str(self.local_dir))

    def _get_state_template(self) -> dict:
        """Create empty state structure"""
        return {
            "metadata": {
                "home_id": self.home_id,
                "created_at": now_iso(),
                "last_updated": now_iso(),
                "version": "2.0",
                "total_videos_processed": 0,
                "schema_version": 1,
                "first_processing_start": None,
                "last_processing_end": None,
                "total_processing_time_seconds": 0
            },
            "daily_processing": {},
            "processing_statistics": {
                "total_sessions": 0,
                "total_containers_created": 0,
                "average_videos_per_session": 0,
                "total_upload_time_seconds": 0,
                "average_upload_time_per_file_seconds": 0,
                "last_30_days": {
                    "videos_processed": 0,
                    "success_rate": 0,
                    "average_session_duration_minutes": 0,
                    "total_upload_time_seconds": 0
                }
            },
            "error_summary": {
                "authentication_failures": 0,
                "network_timeouts": 0,
                "file_corruption": 0,
                "last_error_date": None
            }
        }

    def load_state(self) -> dict:
        """Load state from your blob storage first, fallback to local"""
        # Try your blob storage first
        if self.state_storage:
            try:
                state = self.state_storage.download_state(self.home_id)
                if state:
                    logger.info("Loaded state from your blob storage for home_id: %s", self.home_id)
                    return state
            except Exception as e:
                logger.warning("Could not load state from your blob storage for %s: %s", self.home_id, e)

        # Try local
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    logger.info("Loaded state from local file for home_id: %s", self.home_id)
                    return state
        except Exception as e:
            logger.warning("Could not load local state for %s: %s", self.home_id, e)

        # Create new state
        logger.info("Creating new state for home_id: %s", self.home_id)
        return self._get_state_template()

    def save_state(self, state: dict) -> None:
        """Save state to both local and your blob storage"""
        state["metadata"]["last_updated"] = now_iso()

        # Save local
        atomic_write_json(str(self.state_file), state)

        # Save to your blob storage (best effort)
        if self.state_storage:
            try:
                self.state_storage.upload_state(self.home_id, state)
            except Exception as e:
                logger.warning("Failed to upload state to your blob storage for %s: %s", self.home_id, e)

    def get_processed_videos(self, date: Optional[str] = None) -> set:
        """Get set of all processed video blob names"""
        processed_videos = set()

        if date:
            # Get videos for specific date
            daily_data = self.current_state["daily_processing"].get(date, {})
            for session in daily_data.get("processing_sessions", []):
                for video in session.get("videos", []):
                    if video.get("upload_status") == "success":
                        processed_videos.add(video["blob_name"])
        else:
            # Get all processed videos
            for date_data in self.current_state["daily_processing"].values():
                for session in date_data.get("processing_sessions", []):
                    for video in session.get("videos", []):
                        if video.get("upload_status") == "success":
                            processed_videos.add(video["blob_name"])

        return processed_videos

    def is_video_processed(self, blob_name: str, fingerprint: dict) -> bool:
        """Check if video is already successfully processed"""
        processed_videos = self.get_processed_videos()

        if blob_name in processed_videos:
            return True

        # Check by fingerprint for renamed files
        for date_data in self.current_state["daily_processing"].values():
            for session in date_data.get("processing_sessions", []):
                for video in session.get("videos", []):
                    if (video.get("upload_status") == "success" and
                        VideoFingerprintManager.fingerprints_match(video.get("fingerprint", {}), fingerprint)):
                        return True

        return False

    def record_processing_session(self, session_info: dict) -> None:
        """Record completed processing session"""
        today = now_utc().strftime("%Y-%m-%d")

        # Initialize daily data if needed
        if today not in self.current_state["daily_processing"]:
            self.current_state["daily_processing"][today] = {
                "date": today,
                "videos_processed": 0,
                "videos_successful": 0,
                "videos_failed": 0,
                "total_upload_time_seconds": 0,
                "processing_sessions": []
            }

        daily_data = self.current_state["daily_processing"][today]
        daily_data["processing_sessions"].append(session_info)

        # Update daily statistics
        videos_in_session = len(session_info.get("videos", []))
        successful_videos = len([v for v in session_info.get("videos", []) if v.get("upload_status") == "success"])
        failed_videos = videos_in_session - successful_videos
        session_upload_time = session_info.get("total_upload_time_seconds", 0)

        daily_data["videos_processed"] += videos_in_session
        daily_data["videos_successful"] += successful_videos
        daily_data["videos_failed"] += failed_videos
        daily_data["total_upload_time_seconds"] += session_upload_time

        # Update metadata
        self.current_state["metadata"]["total_videos_processed"] += videos_in_session
        
        # Update home_id level timing
        if not self.current_state["metadata"]["first_processing_start"]:
            self.current_state["metadata"]["first_processing_start"] = session_info.get("started_at")
        
        self.current_state["metadata"]["last_processing_end"] = session_info.get("completed_at")
        
        # Calculate total processing time for this home_id
        if (self.current_state["metadata"]["first_processing_start"] and 
            self.current_state["metadata"]["last_processing_end"]):
            try:
                from dateutil import parser as dtparser
                start_dt = dtparser.parse(self.current_state["metadata"]["first_processing_start"])
                end_dt = dtparser.parse(self.current_state["metadata"]["last_processing_end"])
                self.current_state["metadata"]["total_processing_time_seconds"] = (end_dt - start_dt).total_seconds()
            except Exception as e:
                logger.warning("Could not calculate total processing time: %s", e)

        # Update statistics
        stats = self.current_state["processing_statistics"]
        stats["total_sessions"] += 1
        stats["total_containers_created"] += 1
        stats["total_upload_time_seconds"] += session_upload_time
        
        # Calculate average upload time per file
        total_successful_videos = sum(
            len([v for v in session.get("videos", []) if v.get("upload_status") == "success"])
            for date_data in self.current_state["daily_processing"].values()
            for session in date_data.get("processing_sessions", [])
        )
        
        if total_successful_videos > 0:
            stats["average_upload_time_per_file_seconds"] = stats["total_upload_time_seconds"] / total_successful_videos
        
        # Update last 30 days statistics
        stats["last_30_days"]["total_upload_time_seconds"] += session_upload_time

        # Save updated state
        self.save_state(self.current_state)

    def get_videos_to_process(self, all_videos: List[Tuple[str, dict, str]]) -> List[Tuple[str, dict, str]]:
        """Filter videos to only include unprocessed ones"""
        videos_to_process = []
        processed_videos = self.get_processed_videos()

        for blob_name, fingerprint, last_modified in all_videos:
            if not self.is_video_processed(blob_name, fingerprint):
                videos_to_process.append((blob_name, fingerprint, last_modified))
                logger.info("SELECTED for processing: %s", blob_name)
            else:
                logger.info("SKIPPED (already processed): %s", blob_name)

        return videos_to_process


class ProcessingSession:
    """Manages individual processing session data"""

    def __init__(self, home_id: str, container_id: str):
        self.session_id = f"sess_{now_utc().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.home_id = home_id
        self.container_id = container_id
        self.started_at = now_iso()
        self.completed_at = None
        self.videos = []

    def add_video_result(self, video_info: dict) -> None:
        """Add video processing result to session"""
        self.videos.append(video_info)

    def mark_completed(self) -> None:
        """Mark session as completed"""
        self.completed_at = now_iso()

    def get_session_info(self) -> dict:
        """Get complete session information"""
        # Calculate session duration
        session_duration_seconds = None
        if self.completed_at and self.started_at:
            try:
                from dateutil import parser as dtparser
                start_dt = dtparser.parse(self.started_at)
                end_dt = dtparser.parse(self.completed_at)
                session_duration_seconds = (end_dt - start_dt).total_seconds()
            except Exception as e:
                logger.warning("Could not calculate session duration: %s", e)
        
        # Calculate total upload time for all videos
        total_upload_time_seconds = 0
        successful_uploads = 0
        for video in self.videos:
            if video.get("upload_duration_seconds") and video.get("upload_status") == "success":
                total_upload_time_seconds += video["upload_duration_seconds"]
                successful_uploads += 1
        
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "container_id": self.container_id,
            "session_duration_seconds": session_duration_seconds,
            "total_upload_time_seconds": total_upload_time_seconds,
            "successful_uploads": successful_uploads,
            "videos": self.videos
        }


class VideoFingerprintManager:
    """Handles video fingerprinting and duplicate detection"""

    @staticmethod
    def create_fingerprint(blob_props: dict) -> dict:
        """Create fingerprint from blob properties"""
        md5 = blob_props.get("content_md5")
        if md5:
            return {
                "type": "content_md5",
                "md5": normalize_content_md5(md5),
                "size": blob_props.get("size"),
                "last_modified": blob_props.get("last_modified")
            }
        return {
            "type": "size_etag_timestamp",
            "size": blob_props.get("size"),
            "etag": blob_props.get("etag"),
            "last_modified": blob_props.get("last_modified")
        }

    @staticmethod
    def fingerprints_match(fp1: dict, fp2: dict) -> bool:
        """Check if two fingerprints represent the same file"""
        if not fp1 or not fp2:
            return False

        # If both have MD5, compare MD5
        if fp1.get("type") == "content_md5" and fp2.get("type") == "content_md5":
            return (fp1.get("md5") == fp2.get("md5") and
                    fp1.get("size") == fp2.get("size"))

        # Fallback to size + etag comparison
        return (fp1.get("size") == fp2.get("size") and
                fp1.get("etag") == fp2.get("etag") and
                fp1.get("last_modified") == fp2.get("last_modified"))