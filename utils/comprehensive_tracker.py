#!/usr/bin/env python3
"""
Comprehensive Pipeline Tracking System

This module provides detailed tracking of each chunk and its processing stages
throughout the entire pipeline, from download to final upload.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ComprehensiveTracker:
    """
    Tracks the complete state of each chunk through all pipeline stages.
    """
    
    def __init__(self, session_id: str, output_dir: str):
        self.session_id = session_id
        self.output_dir = output_dir
        
        # Save tracking files in the same base directory as output_dir
        # Extract base directory from output_dir (e.g., ./outmain2withdoamain/session_id -> ./outmain2withdoamain)
        base_tracking_dir = os.path.dirname(output_dir)
        self.tracking_dir = os.path.join(base_tracking_dir, "comprehensive_tracking")
        os.makedirs(self.tracking_dir, exist_ok=True)
        
        # Initialize session-level tracking
        self.session_tracking_file = os.path.join(self.tracking_dir, f"{session_id}_session_tracking.json")
        self.session_data = self._load_or_create_session_data()
    
    def _load_or_create_session_data(self) -> Dict:
        """Load existing session data or create new structure."""
        if os.path.exists(self.session_tracking_file):
            try:
                with open(self.session_tracking_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load session tracking data: {e}")
        
        return {
            "session_id": self.session_id,
            "session_start_time": datetime.now().isoformat(),
            "session_status": "initialized",
            "session_completion_status": "pending",
            "session_completion_time": None,
            "total_chunks": 0,
            "chunks": {},
            "overall_progress": {
                "download_completed": 0,
                "conversion_completed": 0,
                "sharding_completed": 0,
                "model_processing_completed": 0,
                "upload_completed": 0,
                "labelstudio_completed": 0
            },
            "error_summary": [],
            "last_updated": datetime.now().isoformat()
        }
    
    def _save_session_data(self):
        """Save session data to file."""
        self.session_data["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.session_tracking_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session tracking data: {e}")
    
    def initialize_chunk_tracking(self, chunk_id: str, chunk_type: str, file_path: str, sequence_number: str) -> Dict:
        """
        Initialize tracking for a new chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk (e.g., "01-video", "02-audio")
            chunk_type: Type of chunk ("video" or "audio")
            file_path: Path to the chunk file
            sequence_number: Sequence number of the chunk
            
        Returns:
            Dictionary with initial chunk tracking data
        """
        chunk_data = {
            "chunk_id": chunk_id,
            "sequence_number": sequence_number,
            "chunk_type": chunk_type,
            "file_path": file_path,
            "download_status": "pending",
            "download_start_time": None,
            "download_end_time": None,
            "video_processing": {
                "insv_to_mp4_conversion": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "output_path": None,
                    "error_message": None
                },
                "view_unwarping": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "views_created": {},
                    "error_message": None
                }
            },
            "view_sharding": {
                "front": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "shard_count": 0,
                    "shard_paths": [],
                    "error_message": None
                },
                "back": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "shard_count": 0,
                    "shard_paths": [],
                    "error_message": None
                },
                "left": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "shard_count": 0,
                    "shard_paths": [],
                    "error_message": None
                },
                "right": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "shard_count": 0,
                    "shard_paths": [],
                    "error_message": None
                },
                "erp": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "shard_count": 0,
                    "shard_paths": [],
                    "error_message": None
                }
            },
            "model_processing": {
                "yolo_detection": {
                    "erp": {
                        "status": "pending",
                        "start_time": None,
                        "end_time": None,
                        "results_path": None,
                        "error_message": None
                    }
                },
                "face_detection": {
                    "front": {
                        "status": "pending",
                        "start_time": None,
                        "end_time": None,
                        "results_path": None,
                        "error_message": None
                    }
                },
                "scene_detection": {
                    "front": {
                        "status": "pending",
                        "start_time": None,
                        "end_time": None,
                        "results_path": None,
                        "error_message": None
                    }
                },
                "nsfw_detection": {
                    "front": {
                        "status": "pending",
                        "start_time": None,
                        "end_time": None,
                        "results_path": None,
                        "error_message": None
                    }
                },
                "motion_detection": {
                    "front": {
                        "status": "pending",
                        "start_time": None,
                        "end_time": None,
                        "results_path": None,
                        "error_message": None
                    }
                },
                "audio_diarization": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "results_path": None,
                    "error_message": None
                },
                "clap_detection": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "results_path": None,
                    "error_message": None,
                    "note": "Only runs on first/last chunks, uses both video and audio"
                },
                "sensitive_info_detection": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "results_path": None,
                    "error_message": None
                },
                "signal_quality_check": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "results_path": None,
                    "error_message": None
                },
                "lighting_analysis": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "results_path": None,
                    "error_message": None
                }
            },
            "upload_status": {
                "blob_upload": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "uploaded_shards": {},
                    "error_message": None
                },
                "labelstudio_tasks": {
                    "status": "pending",
                    "start_time": None,
                    "end_time": None,
                    "tasks_created": 0,
                    "tasks_uploaded": 0,
                    "task_ids": [],
                    "error_message": None
                }
            },
            "error_details": {
                "stage": None,
                "model": None,
                "error_message": None,
                "timestamp": None,
                "retry_count": 0
            },
            "chunk_completion_status": "pending",
            "chunk_completion_time": None
        }
        
        # Save chunk data
        self.session_data["chunks"][chunk_id] = chunk_data
        self.session_data["total_chunks"] = len(self.session_data["chunks"])
        self._save_session_data()
        
        # Save individual chunk file
        chunk_file = os.path.join(self.tracking_dir, f"{chunk_id}_tracking.json")
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        logger.info(f"Initialized tracking for chunk: {chunk_id}")
        return chunk_data
    
    def update_chunk_status(self, chunk_id: str, stage: str, status: str, **kwargs):
        """
        Update the status of a specific stage for a chunk.
        
        Args:
            chunk_id: Chunk identifier
            stage: Stage name (e.g., "download", "video_processing.insv_to_mp4_conversion")
            status: New status ("pending", "processing", "completed", "error")
            **kwargs: Additional data to update
        """
        if chunk_id not in self.session_data["chunks"]:
            logger.warning(f"Chunk {chunk_id} not found in tracking data")
            return
        
        chunk_data = self.session_data["chunks"][chunk_id]
        
        # Navigate to the correct nested location
        keys = stage.split('.')
        current = chunk_data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Update the final key
        final_key = keys[-1]
        if final_key not in current:
            current[final_key] = {}
        
        # Update status and timestamp
        current[final_key]["status"] = status
        current[final_key]["last_updated"] = datetime.now().isoformat()
        
        # Add start/end times based on status
        if status == "processing":
            current[final_key]["start_time"] = datetime.now().isoformat()
        elif status in ["completed", "error"]:
            current[final_key]["end_time"] = datetime.now().isoformat()
        
        # Update additional fields
        for key, value in kwargs.items():
            current[final_key][key] = value
        
        # Handle errors
        if status == "error":
            chunk_data["error_details"] = {
                "stage": stage,
                "model": kwargs.get("model", None),
                "error_message": kwargs.get("error_message", "Unknown error"),
                "timestamp": datetime.now().isoformat(),
                "retry_count": chunk_data["error_details"].get("retry_count", 0) + 1
            }
            
            # Add to session error summary
            self.session_data["error_summary"].append({
                "chunk_id": chunk_id,
                "stage": stage,
                "error_message": kwargs.get("error_message", "Unknown error"),
                "timestamp": datetime.now().isoformat()
            })
        
        # Update overall progress
        self._update_overall_progress(chunk_data)
        
        # Save data
        self._save_session_data()
        
        # Save individual chunk file
        chunk_file = os.path.join(self.tracking_dir, f"{chunk_id}_tracking.json")
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        logger.info(f"Updated {chunk_id} - {stage}: {status}")
    
    def _update_overall_progress(self, chunk_data: Dict):
        """Update overall session progress based on chunk completion."""
        progress = self.session_data["overall_progress"]
        
        # Check download completion
        if chunk_data["download_status"] == "completed":
            progress["download_completed"] += 1
        
        # Check conversion completion
        if chunk_data["video_processing"]["insv_to_mp4_conversion"]["status"] == "completed":
            progress["conversion_completed"] += 1
        
        # Check sharding completion (check if all views are completed)
        sharding_completed = all(
            view_data["status"] == "completed" 
            for view_data in chunk_data["view_sharding"].values()
        )
        if sharding_completed:
            progress["sharding_completed"] += 1
        
        # Check model processing completion
        model_completed = True
        for model_group in chunk_data["model_processing"].values():
            if isinstance(model_group, dict):
                for model_data in model_group.values():
                    if isinstance(model_data, dict) and model_data.get("status") not in ["completed", "skipped"]:
                        model_completed = False
                        break
            elif isinstance(model_group, dict) and model_group.get("status") not in ["completed", "skipped"]:
                model_completed = False
                break
            if not model_completed:
                break
        if model_completed:
            progress["model_processing_completed"] += 1
        
        # Check upload completion
        if chunk_data["upload_status"]["blob_upload"]["status"] == "completed":
            progress["upload_completed"] += 1
        
        # Check label studio completion
        if chunk_data["upload_status"]["labelstudio_tasks"]["status"] == "completed":
            progress["labelstudio_completed"] += 1
    
    def get_chunk_status(self, chunk_id: str) -> Optional[Dict]:
        """Get current status of a specific chunk."""
        return self.session_data["chunks"].get(chunk_id)
    
    def get_session_summary(self) -> Dict:
        """Get overall session summary."""
        total_chunks = self.session_data["total_chunks"]
        progress = self.session_data["overall_progress"]
        
        summary = {
            "session_id": self.session_id,
            "total_chunks": total_chunks,
            "progress_percentage": {
                "download": (progress["download_completed"] / total_chunks * 100) if total_chunks > 0 else 0,
                "conversion": (progress["conversion_completed"] / total_chunks * 100) if total_chunks > 0 else 0,
                "sharding": (progress["sharding_completed"] / total_chunks * 100) if total_chunks > 0 else 0,
                "model_processing": (progress["model_processing_completed"] / total_chunks * 100) if total_chunks > 0 else 0,
                "upload": (progress["upload_completed"] / total_chunks * 100) if total_chunks > 0 else 0,
                "labelstudio": (progress["labelstudio_completed"] / total_chunks * 100) if total_chunks > 0 else 0
            },
            "error_count": len(self.session_data["error_summary"]),
            "session_status": self.session_data["session_status"],
            "last_updated": self.session_data["last_updated"]
        }
        
        return summary
    
    def mark_chunk_complete(self, chunk_id: str):
        """Mark a chunk as completely processed."""
        if chunk_id in self.session_data["chunks"]:
            chunk_data = self.session_data["chunks"][chunk_id]
            chunk_data["chunk_completion_status"] = "completed"
            chunk_data["chunk_completion_time"] = datetime.now().isoformat()
            self._save_session_data()
            
            # Save individual chunk file
            chunk_file = os.path.join(self.tracking_dir, f"{chunk_id}_tracking.json")
            with open(chunk_file, 'w') as f:
                json.dump(chunk_data, f, indent=2)
            
            logger.info(f"Marked chunk {chunk_id} as completed")
    
    def get_tracking_files(self) -> List[str]:
        """Get list of all tracking files created."""
        tracking_files = []
        for file in os.listdir(self.tracking_dir):
            if file.endswith('_tracking.json'):
                tracking_files.append(os.path.join(self.tracking_dir, file))
        return tracking_files
    
    def mark_session_complete(self):
        """Mark the entire session as completed."""
        self.session_data["session_completion_status"] = "completed"
        self.session_data["session_completion_time"] = datetime.now().isoformat()
        self.session_data["session_status"] = "completed"
        self._save_session_data()
        logger.info(f"Marked session {self.session_id} as completed")
    
    def is_session_completed(self) -> bool:
        """Check if the session has been completed."""
        return self.session_data.get("session_completion_status") == "completed"
    
    def get_session_completion_info(self) -> Dict:
        """Get session completion information."""
        return {
            "session_id": self.session_id,
            "completion_status": self.session_data.get("session_completion_status", "pending"),
            "completion_time": self.session_data.get("session_completion_time"),
            "session_status": self.session_data.get("session_status", "unknown")
        }


# Global tracker instance
_global_tracker = None

def get_global_tracker() -> Optional[ComprehensiveTracker]:
    """Get the global tracker instance."""
    return _global_tracker

def initialize_global_tracker(session_id: str, output_dir: str) -> ComprehensiveTracker:
    """Initialize the global tracker instance."""
    global _global_tracker
    _global_tracker = ComprehensiveTracker(session_id, output_dir)
    return _global_tracker

def update_tracking(chunk_id: str, stage: str, status: str, **kwargs):
    """Convenience function to update tracking using global tracker."""
    if _global_tracker:
        _global_tracker.update_chunk_status(chunk_id, stage, status, **kwargs)
    else:
        logger.warning("Global tracker not initialized")

def check_session_completion_status(session_id: str, output_base_dir: str = "./outmain2withdoamain") -> Dict:
    """
    Check if a session has already been completed by looking at its tracking file.
    
    Args:
        session_id: The session ID to check
        output_base_dir: Base output directory from config (default: ./outmain2withdoamain)
        
    Returns:
        Dictionary with completion status information
    """
    tracking_dir = os.path.join(output_base_dir, "comprehensive_tracking")
    session_tracking_file = os.path.join(tracking_dir, f"{session_id}_session_tracking.json")
    
    if not os.path.exists(session_tracking_file):
        return {
            "session_id": session_id,
            "exists": False,
            "completion_status": "not_found",
            "completion_time": None,
            "session_status": "not_found"
        }
    
    try:
        with open(session_tracking_file, 'r') as f:
            session_data = json.load(f)
        
        return {
            "session_id": session_id,
            "exists": True,
            "completion_status": session_data.get("session_completion_status", "pending"),
            "completion_time": session_data.get("session_completion_time"),
            "session_status": session_data.get("session_status", "unknown"),
            "total_chunks": session_data.get("total_chunks", 0),
            "error_count": len(session_data.get("error_summary", []))
        }
    except Exception as e:
        logger.error(f"Failed to read session tracking file for {session_id}: {e}")
        return {
            "session_id": session_id,
            "exists": True,
            "completion_status": "error",
            "completion_time": None,
            "session_status": "error",
            "error": str(e)
        }

def is_session_already_processed(session_id: str, output_base_dir: str = "./outmain2withdoamain") -> bool:
    """
    Simple check to see if a session has already been completed.
    
    Args:
        session_id: The session ID to check
        output_base_dir: Base output directory from config (default: ./outmain2withdoamain)
        
    Returns:
        True if session is completed, False otherwise
    """
    status = check_session_completion_status(session_id, output_base_dir)
    return status.get("completion_status") == "completed"

def cleanup_temp_download_directory(temp_download_dir: str, session_id: str = None) -> Dict:
    """
    Clean up temporary download directory after session processing is complete.
    
    Args:
        temp_download_dir: Path to the temporary download directory
        session_id: Optional session ID to clean up specific session files
        
    Returns:
        Dictionary with cleanup results
    """
    try:
        if not os.path.exists(temp_download_dir):
            logger.info(f"📁 Temp download directory does not exist: {temp_download_dir}")
            return {"success": True, "message": "Directory does not exist", "files_removed": 0}
        
        files_removed = 0
        dirs_removed = 0
        
        if session_id:
            # Clean up specific session files
            logger.info(f"🧹 Cleaning up temp files for session: {session_id}")
            for root, dirs, files in os.walk(temp_download_dir):
                for file in files:
                    if session_id in file:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            files_removed += 1
                            logger.debug(f"   🗑️ Removed file: {file}")
                        except Exception as e:
                            logger.warning(f"   ⚠️ Failed to remove {file}: {e}")
        else:
            # Clean up entire temp directory
            logger.info(f"🧹 Cleaning up entire temp download directory: {temp_download_dir}")
            for root, dirs, files in os.walk(temp_download_dir, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        files_removed += 1
                        logger.debug(f"   🗑️ Removed file: {file}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to remove {file}: {e}")
                
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        os.rmdir(dir_path)
                        dirs_removed += 1
                        logger.debug(f"   🗑️ Removed directory: {dir_name}")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to remove directory {dir_name}: {e}")
        
        result = {
            "success": True,
            "files_removed": files_removed,
            "dirs_removed": dirs_removed,
            "temp_download_dir": temp_download_dir
        }
        
        logger.info(f"✅ Cleanup completed: {files_removed} files, {dirs_removed} directories removed")
        return result
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return {"success": False, "error": str(e), "files_removed": 0, "dirs_removed": 0}

def cleanup_session_output_directory(session_output_dir: str, session_id: str) -> Dict:
    """
    Clean up session output directory after successful upload to blob storage.
    
    Args:
        session_output_dir: Path to the session output directory
        session_id: Session ID for logging purposes
        
    Returns:
        Dictionary with cleanup results
    """
    try:
        if not os.path.exists(session_output_dir):
            logger.info(f"📁 Session output directory does not exist: {session_output_dir}")
            return {"success": True, "message": "Directory does not exist", "files_removed": 0}
        
        files_removed = 0
        dirs_removed = 0
        
        logger.info(f"🧹 Cleaning up session output directory for session: {session_id}")
        logger.info(f"   📂 Directory: {session_output_dir}")
        
        # Walk through the directory and remove all files and subdirectories
        for root, dirs, files in os.walk(session_output_dir, topdown=False):
            # Remove all files
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    files_removed += 1
                    logger.debug(f"   🗑️ Removed file: {file}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to remove {file}: {e}")
            
            # Remove all subdirectories
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)
                    dirs_removed += 1
                    logger.debug(f"   🗑️ Removed directory: {dir_name}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to remove directory {dir_name}: {e}")
        
        # Finally, remove the main session directory itself
        try:
            os.rmdir(session_output_dir)
            dirs_removed += 1
            logger.debug(f"   🗑️ Removed main session directory: {session_output_dir}")
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to remove main session directory: {e}")
        
        result = {
            "success": True,
            "files_removed": files_removed,
            "dirs_removed": dirs_removed,
            "session_output_dir": session_output_dir
        }
        
        logger.info(f"✅ Session output cleanup completed: {files_removed} files, {dirs_removed} directories removed")
        return result
        
    except Exception as e:
        logger.error(f"❌ Session output cleanup failed: {e}")
        return {"success": False, "error": str(e), "files_removed": 0, "dirs_removed": 0}

def get_incomplete_sessions_for_retry(output_base_dir: str = "./outmain2withdn") -> List[Dict]:
    """
    Get list of sessions that were started but not completed (failed or interrupted).
    
    Args:
        output_base_dir: Base directory containing session outputs
        
    Returns:
        List of dictionaries containing session info for retry
    """
    incomplete_sessions = []
    
    try:
        if not os.path.exists(output_base_dir):
            logger.info(f"📁 Output base directory does not exist: {output_base_dir}")
            return incomplete_sessions
        
        # Look for session directories
        for item in os.listdir(output_base_dir):
            session_path = os.path.join(output_base_dir, item)
            if os.path.isdir(session_path):
                session_id = item
                
                # Check session completion status
                session_status = check_session_completion_status(session_id, output_base_dir)
                
                # Include sessions that are pending (started but not completed)
                if (session_status.get("completion_status") == "pending" or 
                    session_status.get("completion_status") == "failed"):
                    
                    incomplete_sessions.append({
                        "session_id": session_id,
                        "session_path": session_path,
                        "status": session_status.get("completion_status"),
                        "last_updated": session_status.get("last_updated"),
                        "error_details": session_status.get("error_details")
                    })
        
        logger.info(f"🔍 Found {len(incomplete_sessions)} incomplete sessions for potential retry")
        for session in incomplete_sessions:
            logger.info(f"   📋 {session['session_id']}: {session['status']} (last updated: {session['last_updated']})")
        
        return incomplete_sessions
        
    except Exception as e:
        logger.error(f"❌ Error getting incomplete sessions: {e}")
        return incomplete_sessions
