#!/usr/bin/env python3
"""
Blob Polling Manager

Core polling logic for monitoring Azure Blob Storage for new JSON metadata files.
Periodically checks pending TAR files to see if their corresponding JSON files
have been uploaded, then triggers upload of newly ready TAR files.

Features:
- Configurable polling interval and max cycles
- Early exit when all pending files are processed
- Retry logic for Azure connectivity issues
- Callbacks for handling newly ready files
- Progress logging
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple, Any

from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)


class BlobPollingManager:
    """Manages blob polling for new JSON metadata files."""

    def __init__(
        self,
        connection_string: str,
        container_name: str,
        interval_minutes: int = 5,
        max_cycles: int = 12,
        exit_when_all_processed: bool = True,
        retry_on_error: bool = True,
        max_retries_per_cycle: int = 3,
        retry_delay_seconds: int = 10
    ):
        """
        Initialize the blob polling manager.

        Args:
            connection_string: Azure Blob Storage connection string
            container_name: Container name where TAR/JSON files are stored
            interval_minutes: Time between polling cycles
            max_cycles: Maximum number of polling cycles
            exit_when_all_processed: Exit early if all pending files are processed
            retry_on_error: Retry on Azure connectivity errors
            max_retries_per_cycle: Max retries per cycle on error
            retry_delay_seconds: Delay between retries
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.interval_minutes = interval_minutes
        self.max_cycles = max_cycles
        self.exit_when_all_processed = exit_when_all_processed
        self.retry_on_error = retry_on_error
        self.max_retries_per_cycle = max_retries_per_cycle
        self.retry_delay_seconds = retry_delay_seconds

        # Initialize Azure client
        self._blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self._container_client = self._blob_service_client.get_container_client(container_name)

        # Polling state
        self._is_running = False
        self._should_stop = False

        logger.info(f"BlobPollingManager initialized:")
        logger.info(f"  Container: {container_name}")
        logger.info(f"  Interval: {interval_minutes} minutes")
        logger.info(f"  Max cycles: {max_cycles}")
        logger.info(f"  Total polling time: {interval_minutes * max_cycles} minutes")

    def get_json_path_for_tar(self, tar_blob_path: str) -> str:
        """
        Get the expected JSON file path for a TAR file.

        Args:
            tar_blob_path: Full blob path to TAR file

        Returns:
            Expected JSON blob path
        """
        if tar_blob_path.lower().endswith('.tar'):
            return tar_blob_path[:-4] + '.json'
        return tar_blob_path + '.json'

    def check_json_exists(self, tar_blob_path: str) -> bool:
        """
        Check if JSON metadata exists for a TAR file.

        Args:
            tar_blob_path: Full blob path to TAR file

        Returns:
            True if corresponding JSON exists
        """
        json_path = self.get_json_path_for_tar(tar_blob_path)
        try:
            self._container_client.get_blob_client(json_path).get_blob_properties()
            return True
        except Exception:
            return False

    def check_pending_files(
        self,
        pending_files: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Check pending files for new JSON metadata.

        Args:
            pending_files: List of pending TAR file info dicts
                          Each dict should have 'full_blob_path' key

        Returns:
            Tuple of (newly_ready_files, still_pending_files)
        """
        newly_ready = []
        still_pending = []

        for file_info in pending_files:
            blob_path = file_info.get('full_blob_path', '')
            if not blob_path:
                logger.warning(f"File missing full_blob_path: {file_info}")
                still_pending.append(file_info)
                continue

            try:
                if self.check_json_exists(blob_path):
                    # JSON found - mark as ready
                    updated_file_info = file_info.copy()
                    updated_file_info['has_metadata_json'] = True
                    updated_file_info['json_found_at'] = datetime.utcnow().isoformat() + "Z"
                    newly_ready.append(updated_file_info)
                    logger.info(f"  JSON found for: {file_info.get('filename', blob_path)}")
                else:
                    still_pending.append(file_info)
            except AzureError as e:
                logger.warning(f"Azure error checking {blob_path}: {e}")
                still_pending.append(file_info)

        return newly_ready, still_pending

    def _check_pending_with_retry(
        self,
        pending_files: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Check pending files with retry logic.

        Args:
            pending_files: List of pending TAR file info dicts

        Returns:
            Tuple of (newly_ready_files, still_pending_files)
        """
        last_error = None

        for attempt in range(1, self.max_retries_per_cycle + 1):
            try:
                return self.check_pending_files(pending_files)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries_per_cycle:
                    logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {self.retry_delay_seconds}s...")
                    time.sleep(self.retry_delay_seconds)
                else:
                    logger.error(f"All {self.max_retries_per_cycle} attempts failed: {e}")

        # If all retries failed, return original pending list
        if self.retry_on_error:
            logger.warning("Returning original pending list due to errors")
            return [], pending_files
        else:
            raise last_error

    def run_polling_loop(
        self,
        pending_files: List[Dict[str, Any]],
        on_new_files_ready: Optional[Callable[[List[Dict[str, Any]], int], None]] = None,
        on_cycle_complete: Optional[Callable[[int, int, int], None]] = None,
        state_tracker: Optional[Any] = None,
        start_cycle: int = 1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run the polling loop until max cycles or all files processed.

        Args:
            pending_files: List of TAR files waiting for JSON
            on_new_files_ready: Callback when new files are ready
                               Signature: (newly_ready_files, cycle_number) -> None
            on_cycle_complete: Callback after each cycle
                              Signature: (cycle_number, new_count, pending_count) -> None
            state_tracker: Optional PollingStateTracker for state persistence
            start_cycle: Starting cycle number (for resume support)

        Returns:
            Tuple of (all_newly_ready_files, remaining_pending_files)
        """
        self._is_running = True
        self._should_stop = False

        all_newly_ready = []
        current_pending = pending_files.copy()

        logger.info(f"\n{'='*80}")
        logger.info(f"Starting polling loop")
        logger.info(f"  Pending files: {len(current_pending)}")
        logger.info(f"  Start cycle: {start_cycle}")
        logger.info(f"  Max cycles: {self.max_cycles}")
        logger.info(f"  Interval: {self.interval_minutes} minutes")
        logger.info(f"{'='*80}\n")

        for cycle in range(start_cycle, self.max_cycles + 1):
            if self._should_stop:
                logger.info("Polling stopped by external signal")
                break

            # Wait for interval (skip wait on first cycle if resuming)
            if cycle > start_cycle or start_cycle == 1:
                logger.info(f"\n--- Polling cycle {cycle}/{self.max_cycles} ---")
                logger.info(f"Waiting {self.interval_minutes} minutes before checking...")

                # Sleep in smaller chunks to allow interruption
                wait_seconds = self.interval_minutes * 60
                sleep_chunk = 10  # Check for stop signal every 10 seconds
                waited = 0
                while waited < wait_seconds and not self._should_stop:
                    time.sleep(min(sleep_chunk, wait_seconds - waited))
                    waited += sleep_chunk

                if self._should_stop:
                    logger.info("Polling stopped during wait")
                    break

            logger.info(f"Checking {len(current_pending)} pending files for new JSONs...")

            # Check for new JSONs
            newly_ready, current_pending = self._check_pending_with_retry(current_pending)

            # Process newly ready files
            if newly_ready:
                logger.info(f"Found {len(newly_ready)} new JSON files!")
                all_newly_ready.extend(newly_ready)

                # Update state tracker
                if state_tracker:
                    ready_uuids = [f.get('tar_uuid') for f in newly_ready if f.get('tar_uuid')]
                    state_tracker.mark_files_as_ready(ready_uuids, cycle)

                # Callback for processing
                if on_new_files_ready:
                    try:
                        on_new_files_ready(newly_ready, cycle)
                    except Exception as e:
                        logger.error(f"Error in on_new_files_ready callback: {e}")
            else:
                logger.info("No new JSON files found this cycle")

            # Record cycle completion
            if state_tracker:
                state_tracker.record_cycle_completion(cycle, len(newly_ready))

            # Callback for cycle completion
            if on_cycle_complete:
                try:
                    on_cycle_complete(cycle, len(newly_ready), len(current_pending))
                except Exception as e:
                    logger.error(f"Error in on_cycle_complete callback: {e}")

            # Log progress
            logger.info(f"Cycle {cycle}/{self.max_cycles} complete:")
            logger.info(f"  New JSONs found: {len(newly_ready)}")
            logger.info(f"  Still pending: {len(current_pending)}")
            logger.info(f"  Total uploaded during polling: {len(all_newly_ready)}")

            # Early exit if all processed
            if not current_pending and self.exit_when_all_processed:
                logger.info("\nAll pending files processed - exiting polling early")
                break

        self._is_running = False

        logger.info(f"\n{'='*80}")
        logger.info(f"Polling loop completed")
        logger.info(f"  Total cycles: {cycle}")
        logger.info(f"  Total new JSONs found: {len(all_newly_ready)}")
        logger.info(f"  Remaining pending: {len(current_pending)}")
        logger.info(f"{'='*80}\n")

        return all_newly_ready, current_pending

    def stop(self):
        """Signal the polling loop to stop gracefully."""
        logger.info("Polling stop requested")
        self._should_stop = True

    def is_running(self) -> bool:
        """Check if polling loop is currently running."""
        return self._is_running

    def get_estimated_completion_time(self, current_cycle: int = 0) -> str:
        """
        Get estimated completion time based on remaining cycles.

        Args:
            current_cycle: Current cycle number

        Returns:
            Estimated completion time as ISO string
        """
        remaining_cycles = self.max_cycles - current_cycle
        remaining_minutes = remaining_cycles * self.interval_minutes
        from datetime import timedelta
        completion_time = datetime.utcnow() + timedelta(minutes=remaining_minutes)
        return completion_time.isoformat() + "Z"


def create_polling_manager_from_config(config) -> BlobPollingManager:
    """
    Create a BlobPollingManager from configuration.

    Args:
        config: ConfigLoader instance with polling configuration

    Returns:
        Configured BlobPollingManager instance
    """
    # Get Azure source configuration
    connection_string = config.get("azure_source", "connection_string")
    container_name = config.get("azure_source", "container_name")

    # Get polling configuration
    interval_minutes = config.get("polling", "interval_minutes", default=5)
    max_cycles = config.get("polling", "max_cycles", default=12)
    exit_when_all_processed = config.get("polling", "exit_when_all_processed", default=True)
    retry_on_error = config.get("polling", "retry_on_error", default=True)
    max_retries = config.get("polling", "max_retries_per_cycle", default=3)
    retry_delay = config.get("polling", "retry_delay_seconds", default=10)

    return BlobPollingManager(
        connection_string=connection_string,
        container_name=container_name,
        interval_minutes=interval_minutes,
        max_cycles=max_cycles,
        exit_when_all_processed=exit_when_all_processed,
        retry_on_error=retry_on_error,
        max_retries_per_cycle=max_retries,
        retry_delay_seconds=retry_delay
    )
