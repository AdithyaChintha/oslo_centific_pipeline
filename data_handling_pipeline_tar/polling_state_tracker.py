#!/usr/bin/env python3
"""
Polling State Tracker

Tracks pending and processed TAR files across polling cycles.
Supports resume capability - if the pipeline is interrupted, it can resume from the last state.

State file structure:
{
    "pipeline_run_id": "20231201_063000",
    "project_id": "oslo_project_2",
    "started_at": "2023-12-01T06:30:00Z",
    "last_updated_at": "2023-12-01T07:15:00Z",
    "status": "polling",  # "initial" | "polling" | "completed" | "interrupted"

    "initial_scan": {
        "total_tar_files": 100,
        "tar_with_json": 80,
        "tar_without_json": 20,
        "scanned_at": "2023-12-01T06:30:00Z"
    },

    "polling_progress": {
        "current_cycle": 5,
        "max_cycles": 12,
        "interval_minutes": 5,
        "cycles_completed": [
            {"cycle": 1, "completed_at": "...", "new_jsons_found": 0, "pending_count": 20},
            {"cycle": 2, "completed_at": "...", "new_jsons_found": 3, "pending_count": 17},
            ...
        ]
    },

    "pending_files": [
        {"tar_uuid": "abc123", "filename": "abc123.tar", "full_blob_path": "...", "added_at": "..."},
        ...
    ],

    "processed_files": [
        {"tar_uuid": "def456", "filename": "def456.tar", "processed_at": "...", "cycle_number": 2},
        ...
    ],

    "uploaded_during_polling": [
        {"tar_uuid": "ghi789", "filename": "ghi789.tar", "uploaded_at": "...", "cycle_number": 3},
        ...
    ]
}
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class PollingStateTracker:
    """Tracks pending and processed TAR files across polling cycles with resume support."""

    def __init__(self, state_file_path: str, project_id: str):
        """
        Initialize the polling state tracker.

        Args:
            state_file_path: Path to the state JSON file
            project_id: Project identifier
        """
        self.state_file = state_file_path
        self.project_id = project_id

        # Ensure directory exists
        state_dir = os.path.dirname(state_file_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

        # Load existing state or create new
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or create new state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info(f"Loaded existing polling state from: {self.state_file}")
                logger.info(f"  Status: {state.get('status', 'unknown')}")
                logger.info(f"  Pending files: {len(state.get('pending_files', []))}")
                logger.info(f"  Processed files: {len(state.get('processed_files', []))}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load polling state: {e}. Creating new state.")
                return self._create_new_state()
        return self._create_new_state()

    def _create_new_state(self) -> Dict[str, Any]:
        """Create a new state structure."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return {
            "pipeline_run_id": run_id,
            "project_id": self.project_id,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "last_updated_at": datetime.utcnow().isoformat() + "Z",
            "status": "initial",
            "initial_scan": {
                "total_tar_files": 0,
                "tar_with_json": 0,
                "tar_without_json": 0,
                "scanned_at": None
            },
            "polling_progress": {
                "current_cycle": 0,
                "max_cycles": 0,
                "interval_minutes": 0,
                "cycles_completed": []
            },
            "pending_files": [],
            "processed_files": [],
            "uploaded_during_polling": []
        }

    def save_state(self):
        """Save current state to file."""
        try:
            self.state["last_updated_at"] = datetime.utcnow().isoformat() + "Z"
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            logger.debug(f"Polling state saved to: {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save polling state: {e}")

    def is_resumable(self) -> bool:
        """Check if there's a previous run that can be resumed."""
        if not os.path.exists(self.state_file):
            return False

        status = self.state.get("status", "")
        # Can resume if status is "polling" or "interrupted"
        return status in ["polling", "interrupted"]

    def get_resume_info(self) -> Dict[str, Any]:
        """Get information about the previous run for resume."""
        return {
            "run_id": self.state.get("pipeline_run_id"),
            "started_at": self.state.get("started_at"),
            "status": self.state.get("status"),
            "current_cycle": self.state.get("polling_progress", {}).get("current_cycle", 0),
            "max_cycles": self.state.get("polling_progress", {}).get("max_cycles", 0),
            "pending_count": len(self.state.get("pending_files", [])),
            "processed_count": len(self.state.get("processed_files", []))
        }

    def start_new_run(self, max_cycles: int, interval_minutes: int):
        """Start a new polling run (clears previous state)."""
        self.state = self._create_new_state()
        self.state["polling_progress"]["max_cycles"] = max_cycles
        self.state["polling_progress"]["interval_minutes"] = interval_minutes
        self.save_state()
        logger.info(f"Started new polling run: {self.state['pipeline_run_id']}")

    def record_initial_scan(self, total_files: int, files_with_json: int, files_without_json: int):
        """Record the results of the initial TAR file scan."""
        self.state["initial_scan"] = {
            "total_tar_files": total_files,
            "tar_with_json": files_with_json,
            "tar_without_json": files_without_json,
            "scanned_at": datetime.utcnow().isoformat() + "Z"
        }
        self.save_state()

    def add_pending_files(self, files: List[Dict[str, Any]]):
        """
        Add files to the pending list (TAR files without JSON).

        Args:
            files: List of file info dicts with keys: tar_uuid, filename, full_blob_path, etc.
        """
        for file_info in files:
            pending_entry = {
                "tar_uuid": file_info.get("tar_uuid", ""),
                "filename": file_info.get("filename", ""),
                "full_blob_path": file_info.get("full_blob_path", ""),
                "file_size_bytes": file_info.get("file_size_bytes", 0),
                "file_size_gb": file_info.get("file_size_gb", 0),
                "added_at": datetime.utcnow().isoformat() + "Z"
            }
            self.state["pending_files"].append(pending_entry)

        self.state["status"] = "polling"
        self.save_state()
        logger.info(f"Added {len(files)} files to pending list. Total pending: {len(self.state['pending_files'])}")

    def add_processed_files(self, files: List[Dict[str, Any]]):
        """
        Add files to the processed list (TAR files uploaded in initial batch).

        Args:
            files: List of file info dicts
        """
        for file_info in files:
            processed_entry = {
                "tar_uuid": file_info.get("tar_uuid", ""),
                "filename": file_info.get("filename", ""),
                "full_blob_path": file_info.get("full_blob_path", ""),
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "cycle_number": 0  # 0 = initial batch, not during polling
            }
            self.state["processed_files"].append(processed_entry)

        self.save_state()

    def get_pending_files(self) -> List[Dict[str, Any]]:
        """Get list of pending files (TAR files waiting for JSON)."""
        return self.state.get("pending_files", [])

    def get_pending_count(self) -> int:
        """Get count of pending files."""
        return len(self.state.get("pending_files", []))

    def mark_files_as_ready(self, tar_uuids: List[str], cycle_number: int) -> List[Dict[str, Any]]:
        """
        Mark files as ready (JSON found) and move from pending to uploaded_during_polling.

        Args:
            tar_uuids: List of TAR UUIDs that are now ready
            cycle_number: Current polling cycle number

        Returns:
            List of file entries that were marked as ready
        """
        ready_files = []
        remaining_pending = []

        uuid_set = set(tar_uuids)

        for pending_file in self.state.get("pending_files", []):
            if pending_file.get("tar_uuid") in uuid_set:
                # Move to uploaded_during_polling
                uploaded_entry = {
                    **pending_file,
                    "json_found_at": datetime.utcnow().isoformat() + "Z",
                    "cycle_number": cycle_number
                }
                self.state["uploaded_during_polling"].append(uploaded_entry)
                ready_files.append(uploaded_entry)
            else:
                remaining_pending.append(pending_file)

        self.state["pending_files"] = remaining_pending
        self.save_state()

        logger.info(f"Marked {len(ready_files)} files as ready in cycle {cycle_number}. "
                   f"Remaining pending: {len(remaining_pending)}")

        return ready_files

    def record_cycle_completion(self, cycle_number: int, new_jsons_found: int):
        """
        Record completion of a polling cycle.

        Args:
            cycle_number: The cycle number that completed
            new_jsons_found: Number of new JSON files found in this cycle
        """
        cycle_record = {
            "cycle": cycle_number,
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "new_jsons_found": new_jsons_found,
            "pending_count": len(self.state.get("pending_files", []))
        }

        self.state["polling_progress"]["current_cycle"] = cycle_number
        self.state["polling_progress"]["cycles_completed"].append(cycle_record)
        self.save_state()

        logger.info(f"Recorded cycle {cycle_number} completion: "
                   f"{new_jsons_found} new JSONs, {cycle_record['pending_count']} still pending")

    def mark_polling_completed(self):
        """Mark polling as completed."""
        self.state["status"] = "completed"
        self.state["completed_at"] = datetime.utcnow().isoformat() + "Z"
        self.save_state()
        logger.info("Polling marked as completed")

    def mark_polling_interrupted(self):
        """Mark polling as interrupted (for graceful shutdown)."""
        self.state["status"] = "interrupted"
        self.state["interrupted_at"] = datetime.utcnow().isoformat() + "Z"
        self.save_state()
        logger.info("Polling marked as interrupted")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current polling state."""
        initial = self.state.get("initial_scan", {})
        progress = self.state.get("polling_progress", {})

        total_uploaded_during_polling = len(self.state.get("uploaded_during_polling", []))
        total_processed_initial = len(self.state.get("processed_files", []))

        return {
            "status": self.state.get("status", "unknown"),
            "started_at": self.state.get("started_at"),
            "last_updated_at": self.state.get("last_updated_at"),
            "initial_scan": {
                "total_tar_files": initial.get("total_tar_files", 0),
                "tar_with_json": initial.get("tar_with_json", 0),
                "tar_without_json": initial.get("tar_without_json", 0)
            },
            "polling": {
                "current_cycle": progress.get("current_cycle", 0),
                "max_cycles": progress.get("max_cycles", 0),
                "cycles_completed": len(progress.get("cycles_completed", [])),
                "total_jsons_found_during_polling": sum(
                    c.get("new_jsons_found", 0)
                    for c in progress.get("cycles_completed", [])
                )
            },
            "files": {
                "uploaded_initial": total_processed_initial,
                "uploaded_during_polling": total_uploaded_during_polling,
                "total_uploaded": total_processed_initial + total_uploaded_during_polling,
                "still_pending": len(self.state.get("pending_files", []))
            }
        }

    def get_all_uploaded_tar_uuids(self) -> set:
        """Get set of all TAR UUIDs that have been uploaded (initial + during polling)."""
        uploaded_uuids = set()

        # From initial batch
        for f in self.state.get("processed_files", []):
            if f.get("tar_uuid"):
                uploaded_uuids.add(f["tar_uuid"])

        # From polling
        for f in self.state.get("uploaded_during_polling", []):
            if f.get("tar_uuid"):
                uploaded_uuids.add(f["tar_uuid"])

        return uploaded_uuids

    def clear_state(self):
        """Clear the state file (for fresh start)."""
        if os.path.exists(self.state_file):
            try:
                os.remove(self.state_file)
                logger.info(f"Cleared polling state file: {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to clear state file: {e}")

        self.state = self._create_new_state()
