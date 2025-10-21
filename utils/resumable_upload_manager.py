#!/usr/bin/env python3
"""
Resumable Upload Manager for SharePoint

This module provides true resumable upload functionality with session persistence
and failure recovery. It can resume interrupted uploads from the exact point
where they failed, without re-uploading already completed chunks.

Key Features:
- Session persistence to disk (JSON)
- Chunk-level progress tracking
- Session validation and recovery
- Automatic cleanup of expired sessions
- Smart resume logic

Author: True resumable upload implementation
Date: October 2025
"""

import os
import json
import time
import hashlib
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class UploadChunk:
    """Represents a single upload chunk with its metadata."""
    chunk_index: int
    start_byte: int
    end_byte: int
    size: int
    uploaded: bool = False
    upload_time: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class UploadSession:
    """Represents a complete upload session with all metadata."""
    session_id: str
    file_path: str
    sharepoint_filename: str
    folder_id: str
    file_size: int
    chunk_size: int
    total_chunks: int
    upload_url: str
    created_time: str
    last_activity: str
    completed: bool = False
    chunks: List[UploadChunk] = None
    
    def __post_init__(self):
        """Initialize chunks list if not provided."""
        if self.chunks is None:
            self.chunks = []

class ResumableUploadManager:
    """
    Manages resumable uploads with session persistence and recovery.
    
    This class handles the complete lifecycle of resumable uploads:
    - Creating and managing upload sessions
    - Tracking chunk-level progress
    - Persisting state to disk
    - Recovering from interruptions
    - Validating and cleaning up sessions
    """
    
    def __init__(self, state_dir: str = "/tmp/sharepoint_upload_sessions", 
                 chunk_size: int = 320 * 1024, session_timeout_hours: int = 24):
        """
        Initialize the resumable upload manager.
        
        Args:
            state_dir: Directory to store upload session state files
            chunk_size: Size of each upload chunk in bytes (must be multiple of 320KB)
            session_timeout_hours: Hours after which sessions are considered expired
        """
        self.state_dir = Path(state_dir)
        self.chunk_size = chunk_size
        self.session_timeout = timedelta(hours=session_timeout_hours)
        
        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate chunk size (SharePoint requires multiples of 320KB)
        if chunk_size % (320 * 1024) != 0:
            raise ValueError("Chunk size must be a multiple of 320KB for SharePoint")
        
        logger.debug(f"ResumableUploadManager initialized: state_dir={state_dir}, chunk_size={chunk_size}")
    
    def _generate_session_id(self, file_path: str, sharepoint_filename: str, folder_id: str) -> str:
        """
        Generate a unique session ID based on file and destination.
        
        Args:
            file_path: Local file path
            sharepoint_filename: SharePoint filename
            folder_id: SharePoint folder ID
            
        Returns:
            Unique session ID string
        """
        # Create hash from file path, destination, and file stats
        file_stat = os.stat(file_path)
        hash_input = f"{file_path}:{sharepoint_filename}:{folder_id}:{file_stat.st_size}:{file_stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_session_file_path(self, session_id: str) -> Path:
        """Get the file path for storing session state."""
        return self.state_dir / f"session_{session_id}.json"
    
    def _save_session(self, session: UploadSession) -> bool:
        """
        Save upload session state to disk.
        
        Args:
            session: Upload session to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            session_file = self._get_session_file_path(session.session_id)
            session_data = asdict(session)
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.debug(f"Saved session state: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
    
    def _load_session(self, session_id: str) -> Optional[UploadSession]:
        """
        Load upload session state from disk.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            UploadSession object or None if not found/invalid
        """
        try:
            session_file = self._get_session_file_path(session_id)
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Convert chunks back to UploadChunk objects
            chunks = []
            for chunk_data in session_data.get('chunks', []):
                chunks.append(UploadChunk(**chunk_data))
            session_data['chunks'] = chunks
            
            session = UploadSession(**session_data)
            logger.debug(f"Loaded session state: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def _delete_session(self, session_id: str) -> bool:
        """
        Delete upload session state file.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            session_file = self._get_session_file_path(session_id)
            if session_file.exists():
                session_file.unlink()
                logger.debug(f"Deleted session state: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def _is_session_expired(self, session: UploadSession) -> bool:
        """
        Check if upload session has expired.
        
        Args:
            session: Upload session to check
            
        Returns:
            True if session is expired, False otherwise
        """
        try:
            last_activity = datetime.fromisoformat(session.last_activity)
            return datetime.now() - last_activity > self.session_timeout
        except Exception:
            return True  # Consider invalid timestamps as expired
    
    def _validate_session(self, session: UploadSession, file_manager) -> bool:
        """
        Validate that upload session is still valid on SharePoint.
        
        Args:
            session: Upload session to validate
            file_manager: SharePoint file manager for API calls
            
        Returns:
            True if session is valid, False otherwise
        """
        try:
            # Check if session is expired locally
            if self._is_session_expired(session):
                logger.info(f"Session {session.session_id} expired locally")
                return False
            
            # Verify the file still exists and hasn't changed
            if not os.path.exists(session.file_path):
                logger.info(f"Source file no longer exists: {session.file_path}")
                return False
            
            current_file_size = os.path.getsize(session.file_path)
            if current_file_size != session.file_size:
                logger.info(f"Source file size changed: {session.file_size} -> {current_file_size}")
                return False
            
            # Test the upload URL (SharePoint sessions can expire)
            try:
                test_response = requests.get(session.upload_url, timeout=10)
                # A valid session URL should return 400 (bad request) for GET
                # An expired session URL typically returns 404 or 410
                if test_response.status_code in [404, 410]:
                    logger.info(f"Upload session expired on SharePoint: {session.session_id}")
                    return False
            except requests.RequestException:
                # Network issues shouldn't invalidate the session
                pass
            
            logger.debug(f"Session validation passed: {session.session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False
    
    def _create_upload_chunks(self, file_size: int) -> List[UploadChunk]:
        """
        Create list of upload chunks for the file.
        
        Args:
            file_size: Total size of file to upload
            
        Returns:
            List of UploadChunk objects
        """
        chunks = []
        chunk_index = 0
        bytes_processed = 0
        
        while bytes_processed < file_size:
            start_byte = bytes_processed
            end_byte = min(bytes_processed + self.chunk_size - 1, file_size - 1)
            chunk_size = end_byte - start_byte + 1
            
            chunk = UploadChunk(
                chunk_index=chunk_index,
                start_byte=start_byte,
                end_byte=end_byte,
                size=chunk_size
            )
            chunks.append(chunk)
            
            bytes_processed = end_byte + 1
            chunk_index += 1
        
        return chunks
    
    def _query_upload_progress(self, session: UploadSession, file_manager) -> Optional[int]:
        """
        Query SharePoint for current upload progress.
        
        Args:
            session: Upload session to query
            file_manager: SharePoint file manager for API calls
            
        Returns:
            Number of bytes already uploaded, or None if query failed
        """
        try:
            # Use a zero-byte PUT request to query progress
            headers = {
                'Content-Range': f'bytes */{session.file_size}',
                'Content-Length': '0'
            }
            
            response = requests.put(
                session.upload_url,
                data=b'',
                headers=headers,
                timeout=30
            )
            
            # Response code 308 indicates resumable upload in progress
            if response.status_code == 308:
                # Parse Range header to get uploaded bytes
                range_header = response.headers.get('Range', '')
                if range_header.startswith('bytes='):
                    range_part = range_header[6:]  # Remove 'bytes='
                    if '-' in range_part:
                        end_byte = int(range_part.split('-')[1])
                        return end_byte + 1  # Convert to bytes uploaded
            
            # Response code 200/201 means upload is complete
            elif response.status_code in [200, 201]:
                return session.file_size
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to query upload progress: {e}")
            return None
    
    def _update_chunk_status_from_progress(self, session: UploadSession, bytes_uploaded: int):
        """
        Update chunk upload status based on SharePoint progress.
        
        Args:
            session: Upload session to update
            bytes_uploaded: Number of bytes confirmed uploaded by SharePoint
        """
        for chunk in session.chunks:
            if chunk.end_byte < bytes_uploaded:
                if not chunk.uploaded:
                    chunk.uploaded = True
                    chunk.upload_time = datetime.now().isoformat()
                    logger.debug(f"Marked chunk {chunk.chunk_index} as uploaded (recovered)")
    
    def create_or_resume_session(self, file_path: str, sharepoint_filename: str, 
                                folder_id: str, file_manager) -> Optional[UploadSession]:
        """
        Create new upload session or resume existing one.
        
        Args:
            file_path: Local file path to upload
            sharepoint_filename: Desired filename in SharePoint
            folder_id: SharePoint folder ID
            file_manager: SharePoint file manager for API calls
            
        Returns:
            UploadSession object or None if creation failed
        """
        # Generate session ID
        session_id = self._generate_session_id(file_path, sharepoint_filename, folder_id)
        
        # Try to load existing session
        existing_session = self._load_session(session_id)
        
        if existing_session:
            logger.info(f"Found existing upload session: {session_id}")
            
            # Validate existing session
            if self._validate_session(existing_session, file_manager):
                # Query current progress from SharePoint
                bytes_uploaded = self._query_upload_progress(existing_session, file_manager)
                
                if bytes_uploaded is not None:
                    # Update chunk status based on SharePoint progress
                    self._update_chunk_status_from_progress(existing_session, bytes_uploaded)
                    
                    # Update last activity
                    existing_session.last_activity = datetime.now().isoformat()
                    self._save_session(existing_session)
                    
                    logger.info(f"Resuming upload from byte {bytes_uploaded}/{existing_session.file_size}")
                    return existing_session
                else:
                    logger.warning("Could not query upload progress, creating new session")
            else:
                logger.info("Existing session invalid, creating new session")
                self._delete_session(session_id)
        
        # Create new session
        logger.info(f"Creating new upload session: {session_id}")
        
        try:
            # Get file information
            file_size = os.path.getsize(file_path)
            
            # Create upload session on SharePoint
            import urllib.parse
            encoded_filename = urllib.parse.quote(sharepoint_filename)
            session_url = f"{file_manager.drive_api_url}/items/{folder_id}:/{encoded_filename}:/createUploadSession"
            
            session_data = {
                "item": {
                    "@microsoft.graph.conflictBehavior": "replace",
                    "name": sharepoint_filename
                }
            }
            
            response = file_manager._make_authenticated_request(
                'POST', 
                session_url, 
                json=session_data
            )
            
            if not response:
                logger.error("Failed to create SharePoint upload session")
                return None
            
            upload_url = response.json().get('uploadUrl')
            if not upload_url:
                logger.error("No upload URL in SharePoint session response")
                return None
            
            # Create session object
            now = datetime.now().isoformat()
            chunks = self._create_upload_chunks(file_size)
            
            session = UploadSession(
                session_id=session_id,
                file_path=file_path,
                sharepoint_filename=sharepoint_filename,
                folder_id=folder_id,
                file_size=file_size,
                chunk_size=self.chunk_size,
                total_chunks=len(chunks),
                upload_url=upload_url,
                created_time=now,
                last_activity=now,
                chunks=chunks
            )
            
            # Save session state
            if self._save_session(session):
                logger.info(f"Created new upload session with {len(chunks)} chunks")
                return session
            else:
                logger.error("Failed to save new session state")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create upload session: {e}")
            return None
    
    def upload_next_chunk(self, session: UploadSession) -> Tuple[bool, bool]:
        """
        Upload the next pending chunk in the session.
        
        Args:
            session: Upload session
            
        Returns:
            Tuple of (chunk_success, upload_complete)
        """
        try:
            # Find next unuploaded chunk
            next_chunk = None
            for chunk in session.chunks:
                if not chunk.uploaded:
                    next_chunk = chunk
                    break
            
            if not next_chunk:
                # All chunks uploaded
                session.completed = True
                session.last_activity = datetime.now().isoformat()
                self._save_session(session)
                return True, True
            
            # Read chunk data from file
            with open(session.file_path, 'rb') as f:
                f.seek(next_chunk.start_byte)
                chunk_data = f.read(next_chunk.size)
            
            if len(chunk_data) != next_chunk.size:
                logger.error(f"Chunk data size mismatch: expected {next_chunk.size}, got {len(chunk_data)}")
                return False, False
            
            # Upload chunk
            headers = {
                'Content-Range': f'bytes {next_chunk.start_byte}-{next_chunk.end_byte}/{session.file_size}',
                'Content-Type': 'application/octet-stream',
                'Content-Length': str(next_chunk.size)
            }
            
            response = requests.put(
                session.upload_url,
                data=chunk_data,
                headers=headers,
                timeout=300  # 5 minute timeout for chunk upload
            )
            
            if response.status_code in [202, 200, 201]:
                # Chunk uploaded successfully
                next_chunk.uploaded = True
                next_chunk.upload_time = datetime.now().isoformat()
                
                # Update session
                session.last_activity = datetime.now().isoformat()
                self._save_session(session)
                
                # Check if upload is complete
                uploaded_chunks = sum(1 for chunk in session.chunks if chunk.uploaded)
                upload_complete = uploaded_chunks == session.total_chunks
                
                if upload_complete:
                    session.completed = True
                    self._save_session(session)
                
                logger.debug(f"Uploaded chunk {next_chunk.chunk_index}/{session.total_chunks - 1}")
                return True, upload_complete
            else:
                logger.error(f"Chunk upload failed: HTTP {response.status_code} - {response.text}")
                return False, False
                
        except Exception as e:
            logger.error(f"Error uploading chunk: {e}")
            return False, False
    
    def get_upload_progress(self, session: UploadSession) -> Dict[str, Any]:
        """
        Get detailed upload progress information.
        
        Args:
            session: Upload session
            
        Returns:
            Dictionary with progress information
        """
        uploaded_chunks = sum(1 for chunk in session.chunks if chunk.uploaded)
        uploaded_bytes = sum(chunk.size for chunk in session.chunks if chunk.uploaded)
        
        progress = {
            'session_id': session.session_id,
            'file_path': session.file_path,
            'sharepoint_filename': session.sharepoint_filename,
            'file_size': session.file_size,
            'total_chunks': session.total_chunks,
            'uploaded_chunks': uploaded_chunks,
            'uploaded_bytes': uploaded_bytes,
            'progress_percent': (uploaded_bytes / session.file_size) * 100,
            'completed': session.completed,
            'created_time': session.created_time,
            'last_activity': session.last_activity
        }
        
        return progress
    
    def cleanup_completed_sessions(self) -> int:
        """
        Clean up completed and expired session files.
        
        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0
        
        try:
            for session_file in self.state_dir.glob("session_*.json"):
                try:
                    session_id = session_file.stem.replace("session_", "")
                    session = self._load_session(session_id)
                    
                    if session and (session.completed or self._is_session_expired(session)):
                        self._delete_session(session_id)
                        cleaned_count += 1
                        logger.debug(f"Cleaned up session: {session_id}")
                        
                except Exception as e:
                    logger.warning(f"Error cleaning up session file {session_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} upload sessions")
        
        return cleaned_count
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        List all active upload sessions.
        
        Returns:
            List of session progress dictionaries
        """
        active_sessions = []
        
        try:
            for session_file in self.state_dir.glob("session_*.json"):
                try:
                    session_id = session_file.stem.replace("session_", "")
                    session = self._load_session(session_id)
                    
                    if session and not session.completed and not self._is_session_expired(session):
                        progress = self.get_upload_progress(session)
                        active_sessions.append(progress)
                        
                except Exception as e:
                    logger.warning(f"Error loading session {session_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error listing active sessions: {e}")
        
        return active_sessions
