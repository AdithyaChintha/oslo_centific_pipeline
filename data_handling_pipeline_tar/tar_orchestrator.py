#!/usr/bin/env python3
"""
Tar Orchestrator with Polling Support
Runs all tar file processing tools in sequence with optional polling for new JSON metadata files.

WORKFLOW MODES:
===============

1. FULL WORKFLOW (default):
   PHASE 1: INITIAL PROCESSING
     - Step 1: List all TAR files and categorize (with/without JSON)
     - Step 2: Upload TAR files WITH JSON to Client Container
   PHASE 2: POLLING (if enabled and pending files exist)
     - Poll for new JSON metadata files at configurable intervals
     - When new JSONs found: upload corresponding TAR files
   PHASE 3: FINAL REPORTING
     - Step 3: Generate consolidated state reports
     - Step 4: Compare TAR files with state files
     - Step 5: TAR inspection (random sampling)
     - Step 6: Upload all reports to Azure Blob AND SharePoint

2. INSPECTION ONLY MODE (--skip-upload):
   - Skips all upload, polling, and comparison steps
   - Runs TAR inspection using execution_mode from config (random/list)
   - Reads TAR files from azure_source.source_prefix path
   - Uses inspection.sampling_ratio for random sampling (1 per N files, min 1)
   - Uploads inspection report to Azure Blob AND SharePoint

3. UPLOAD ONLY MODE (--skip-inspection):
   - Runs listing, upload, polling, state report, comparison
   - Skips TAR inspection step
   - Uploads comparison report to Azure Blob AND SharePoint

Usage:
    # Full workflow
    python tar_orchestrator.py --config config/tar_pipeline_config.yaml

    # INSPECTION ONLY: Random sample TAR files from blob path
    python tar_orchestrator.py --config config/tar_pipeline_config.yaml --skip-upload

    # UPLOAD ONLY: Upload and generate comparison report
    python tar_orchestrator.py --config config/tar_pipeline_config.yaml --skip-inspection --no-polling

    # Override polling settings
    python tar_orchestrator.py --config config/tar_pipeline_config.yaml --polling-interval 10 --polling-cycles 24

    # Resume interrupted polling
    python tar_orchestrator.py --config config/tar_pipeline_config.yaml --resume
"""

import os
import sys
import signal
import argparse
import subprocess
import yaml
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from azure.storage.blob import BlobServiceClient

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import polling modules
from polling_state_tracker import PollingStateTracker
from blob_polling_manager import BlobPollingManager, create_polling_manager_from_config
from tar_file_lister import (
    list_tar_files_categorized,
    recheck_files_for_json,
    update_csv_with_new_json_status,
    save_to_csv
)

# Import SharePoint utilities for report upload
try:
    from utils.sharepoint_utils import SharePointAuthenticator, SharePointFileManager
    SHAREPOINT_AVAILABLE = True
except ImportError:
    SHAREPOINT_AVAILABLE = False


class ConfigLoader:
    """Load and parse YAML configuration"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load(config_path)

    def _load(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, *keys: str, default=None):
        node = self.config
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


class TarOrchestrator:
    """Orchestrates all tar file processing and reporting tools with polling support"""

    def __init__(
        self,
        config_path: str,
        skip_upload: bool = False,
        skip_inspection: bool = False,
        inspection_sample_count: int = None,
        no_polling: bool = False,
        polling_interval: int = None,
        polling_cycles: int = None,
        resume: bool = False
    ):
        self.config_path = os.path.abspath(config_path)
        self.script_dir = Path(__file__).parent
        self.skip_upload = skip_upload
        self.skip_inspection = skip_inspection
        self.resume = resume

        # Load configuration (unified config for all steps)
        self.cfg = ConfigLoader(self.config_path)
        self.project_id = self.cfg.get("azure_source", "project_id", default="tar-upload")

        # Extract date from source_prefix for report naming
        # e.g., "one-data-platform/hummus_prod/2025-12-01/tar/" -> "20251201"
        self.report_date_prefix = self._extract_date_from_source_prefix()

        # Get inspection sample count from config or command line override
        if inspection_sample_count is not None:
            self.inspection_sample_count = inspection_sample_count
        else:
            self.inspection_sample_count = self.cfg.get("inspection", "csv_sample_count", default=4)

        # Polling configuration
        self.polling_enabled = not no_polling and self.cfg.get("polling", "enabled", default=False)
        self.polling_interval = polling_interval or self.cfg.get("polling", "interval_minutes", default=5)
        self.polling_max_cycles = polling_cycles or self.cfg.get("polling", "max_cycles", default=12)
        self.polling_state_file = self.cfg.get(
            "polling", "state_file",
            default="/data/oslo/polling_state/tar_polling_state.json"
        )

        # Output directories
        self.tar_files_dir = Path("/data/oslo/tar_files")
        self.consolidated_dir = Path("/data/oslo/consolidated_report/tar")
        self.comparison_dir = Path("/data/oslo/comparison_results/tar")
        self.inspection_dir = Path("/data/oslo/inspection_reports/tar")
        self.polling_state_dir = Path(self.polling_state_file).parent

        # Create output directories
        self.tar_files_dir.mkdir(parents=True, exist_ok=True)
        self.consolidated_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        self.inspection_dir.mkdir(parents=True, exist_ok=True)
        self.polling_state_dir.mkdir(parents=True, exist_ok=True)

        # Track generated files
        self.generated_files = {
            'tar_listing': None,
            'upload_manifest': None,
            'state_file': None,
            'state_report_json': None,
            'state_report_csv': None,
            'comparison_csv': None,
            'comparison_json': None,
            'inspection_report': None
        }

        # Timestamp for consistent naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Polling state tracker
        self.state_tracker = PollingStateTracker(self.polling_state_file, self.project_id)

        # Polling manager (initialized later if needed)
        self.polling_manager = None

        # Track categorized files
        self.all_tar_files = []
        self.tar_with_json = []
        self.tar_without_json = []

        # Signal handling for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown during polling"""
        def handle_interrupt(signum, frame):
            print("\n\n⚠️  Interrupt received - saving state and shutting down gracefully...")
            if self.polling_manager:
                self.polling_manager.stop()
            self.state_tracker.mark_polling_interrupted()
            sys.exit(130)

        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)

    def _extract_date_from_source_prefix(self) -> str:
        """Extract date from source_prefix for report naming.

        Looks for date patterns like:
        - 2025-12-01 -> 20251201
        - 2025/12/01 -> 20251201
        - 20251201 -> 20251201

        Returns:
            Date string in YYYYMMDD format, or None if not found
        """
        import re
        source_prefix = self.cfg.get("azure_source", "source_prefix", default="")

        # Try to find date pattern YYYY-MM-DD or YYYY/MM/DD
        date_pattern = r'(\d{4})[-/](\d{2})[-/](\d{2})'
        match = re.search(date_pattern, source_prefix)
        if match:
            year, month, day = match.groups()
            date_str = f"{year}{month}{day}"
            print(f"   Extracted date from source_prefix: {year}-{month}-{day} -> {date_str}")
            return date_str

        # Try to find YYYYMMDD pattern
        date_pattern2 = r'(\d{8})'
        match2 = re.search(date_pattern2, source_prefix)
        if match2:
            date_str = match2.group(1)
            print(f"   Extracted date from source_prefix: {date_str}")
            return date_str

        # No date found, return None (will use today's date)
        print(f"   No date found in source_prefix, will use today's date")
        return None

    def print_header(self, title: str):
        """Print formatted section header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80 + "\n")

    def run_command(self, cmd: list, description: str, stream_output: bool = False) -> tuple:
        """Run a command and return success status and output"""
        print(f"  {description}")
        print(f"   Command: {' '.join(cmd)}")

        if stream_output:
            try:
                print()
                result = subprocess.run(cmd, cwd=str(self.script_dir), check=True)
                print(f"\n  {description} - SUCCESS")
                return True, ""
            except subprocess.CalledProcessError as e:
                print(f"\n  {description} - FAILED (exit code: {e.returncode})")
                return False, ""
        else:
            try:
                result = subprocess.run(
                    cmd, cwd=str(self.script_dir),
                    capture_output=True, text=True, check=True
                )
                print(f"  {description} - SUCCESS")
                return True, result.stdout
            except subprocess.CalledProcessError as e:
                print(f"  {description} - FAILED")
                print(f"   Error: {e.stderr}")
                return False, e.stderr

    def _copy_state_file_to_data(self):
        """Copy state file from Azure to /data folder"""
        try:
            state_conn = self.cfg.get("state_storage", "connection_string")
            state_container = self.cfg.get("state_storage", "container_name")
            state_prefix = self.cfg.get("state_storage", "prefix", default="")

            if not state_conn or not state_container:
                print("  State storage not configured, skipping state file copy")
                return

            state_blob_name = f"{state_prefix}{self.project_id}_state.json"
            blob_service_client = BlobServiceClient.from_connection_string(state_conn)
            container_client = blob_service_client.get_container_client(state_container)
            blob_client = container_client.get_blob_client(state_blob_name)

            dest_dir = Path("/data/oslo/upload_state/tar")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"{self.project_id}_state.json"

            with open(dest_file, "wb") as f:
                stream = blob_client.download_blob()
                f.write(stream.readall())

            self.generated_files['state_file'] = str(dest_file)
            print(f"   State file saved to: {dest_file}")

        except Exception as e:
            print(f"   Warning: Could not copy state file: {e}")

    def _upload_csv_report_to_blob(self, local_file_path: str, report_type: str) -> bool:
        """Upload a CSV report file to Azure Blob Storage"""
        try:
            csv_storage_enabled = self.cfg.get("csv_report_storage", "enabled", default=False)
            if not csv_storage_enabled:
                return False

            csv_conn = self.cfg.get("csv_report_storage", "connection_string")
            csv_container = self.cfg.get("csv_report_storage", "container_name")
            csv_prefix = self.cfg.get("csv_report_storage", "prefix", default="")

            if not csv_conn or not csv_container or not os.path.exists(local_file_path):
                return False

            filename = os.path.basename(local_file_path)
            blob_name = f"{csv_prefix.rstrip('/')}/{filename}" if csv_prefix else filename

            blob_service_client = BlobServiceClient.from_connection_string(csv_conn)
            container_client = blob_service_client.get_container_client(csv_container)
            blob_client = container_client.get_blob_client(blob_name)

            with open(local_file_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)

            print(f"   {report_type.capitalize()} report uploaded: {csv_container}/{blob_name}")
            return True

        except Exception as e:
            print(f"   Warning: Could not upload {report_type} report: {e}")
            return False

    def _upload_csv_report_to_sharepoint(self, local_file_path: str, report_type: str) -> bool:
        """Upload a CSV report file to SharePoint

        Uses separate paths for different report types:
        - inspection: sharepoint_report_upload.inspection_report_path
        - comparison: sharepoint_report_upload.comparison_report_path
        """
        try:
            if not SHAREPOINT_AVAILABLE:
                print(f"   SharePoint utilities not available, skipping SharePoint upload")
                return False

            sp_upload_enabled = self.cfg.get("sharepoint_report_upload", "enabled", default=False)
            if not sp_upload_enabled:
                return False

            # Get the appropriate SharePoint path based on report type
            if report_type == "inspection":
                sp_folder_path = self.cfg.get("sharepoint_report_upload", "inspection_report_path")
            elif report_type == "comparison":
                sp_folder_path = self.cfg.get("sharepoint_report_upload", "comparison_report_path")
            else:
                # Fallback to inspection path for other types
                sp_folder_path = self.cfg.get("sharepoint_report_upload", "inspection_report_path")

            if not sp_folder_path or not os.path.exists(local_file_path):
                print(f"   SharePoint path not configured for {report_type} reports")
                return False

            # Get SharePoint credentials from config
            tenant_id = self.cfg.get("azure_ad", "tenant_id")
            client_id = self.cfg.get("azure_ad", "client_id")
            client_secret = self.cfg.get("azure_ad", "client_secret")
            site_id = self.cfg.get("sharepoint", "site_id")
            drive_id = self.cfg.get("sharepoint", "drive_id")

            if not all([tenant_id, client_id, client_secret, site_id, drive_id]):
                print(f"   SharePoint credentials not fully configured, skipping upload")
                return False

            # Initialize SharePoint authenticator and file manager
            authenticator = SharePointAuthenticator(tenant_id, client_id, client_secret)
            access_token = authenticator.get_access_token()
            if not access_token:
                print(f"   SharePoint authentication failed")
                return False

            file_manager = SharePointFileManager(authenticator, site_id, drive_id)

            # Get folder ID from path
            folder_id = file_manager.get_folder_id(sp_folder_path)
            if not folder_id:
                print(f"   SharePoint folder not found: {sp_folder_path}")
                return False

            # Upload file
            filename = os.path.basename(local_file_path)
            success = file_manager.upload_file(local_file_path, filename, folder_id)

            if success:
                print(f"   {report_type.capitalize()} report uploaded to SharePoint: {sp_folder_path}/{filename}")
                return True
            else:
                print(f"   Warning: SharePoint upload failed for {report_type} report")
                return False

        except Exception as e:
            print(f"   Warning: Could not upload {report_type} report to SharePoint: {e}")
            return False

    def _upload_all_csv_reports(self):
        """Upload all generated CSV reports to Azure Blob Storage and SharePoint"""
        self.print_header("UPLOADING CSV REPORTS TO AZURE BLOB AND SHAREPOINT")

        blob_upload_results = []
        sharepoint_upload_results = []

        # Check if blob storage is enabled
        csv_storage_enabled = self.cfg.get("csv_report_storage", "enabled", default=False)
        sp_upload_enabled = self.cfg.get("sharepoint_report_upload", "enabled", default=False)

        if not csv_storage_enabled and not sp_upload_enabled:
            print("  CSV report storage not enabled (blob and SharePoint), skipping uploads")
            return

        # Upload comparison CSV
        comparison_csv = self.generated_files.get('comparison_csv')
        if comparison_csv and os.path.exists(comparison_csv):
            if csv_storage_enabled:
                success = self._upload_csv_report_to_blob(comparison_csv, "comparison")
                blob_upload_results.append(("Comparison CSV (Blob)", success))
            if sp_upload_enabled:
                success = self._upload_csv_report_to_sharepoint(comparison_csv, "comparison")
                sharepoint_upload_results.append(("Comparison CSV (SharePoint)", success))

        # Upload inspection CSV
        inspection_csv = self.generated_files.get('inspection_report')
        if inspection_csv and os.path.exists(inspection_csv):
            if csv_storage_enabled:
                success = self._upload_csv_report_to_blob(inspection_csv, "inspection")
                blob_upload_results.append(("Inspection Report CSV (Blob)", success))
            if sp_upload_enabled:
                success = self._upload_csv_report_to_sharepoint(inspection_csv, "inspection")
                sharepoint_upload_results.append(("Inspection Report CSV (SharePoint)", success))

        # Print upload summary
        all_results = blob_upload_results + sharepoint_upload_results
        if all_results:
            print("\n   CSV Upload Summary:")
            for report_name, success in all_results:
                status = "Uploaded" if success else "Failed"
                print(f"     {status}: {report_name}")

    # =========================================================================
    # PHASE 1: INITIAL PROCESSING
    # =========================================================================

    def step1_list_and_categorize_tar_files(self) -> bool:
        """Step 1: List all TAR files and categorize by JSON availability"""
        self.print_header("STEP 1: List and Categorize TAR Files")

        try:
            # Use categorization function
            self.all_tar_files, self.tar_with_json, self.tar_without_json, stats = \
                list_tar_files_categorized(self.cfg)

            # Save the listing CSV
            output_file = self.tar_files_dir / f"tar_files_{self.project_id}_{self.timestamp}.csv"
            save_to_csv(self.all_tar_files, str(output_file), self.project_id)
            self.generated_files['tar_listing'] = str(output_file)

            print(f"\n   TAR listing saved to: {output_file}")
            print(f"\n   Summary:")
            print(f"     Total TAR files: {len(self.all_tar_files)}")
            print(f"     Ready (with JSON): {len(self.tar_with_json)}")
            print(f"     Pending (without JSON): {len(self.tar_without_json)}")

            # Record initial scan in state tracker
            self.state_tracker.record_initial_scan(
                total_files=len(self.all_tar_files),
                files_with_json=len(self.tar_with_json),
                files_without_json=len(self.tar_without_json)
            )

            # Add pending files to state tracker
            if self.tar_without_json:
                self.state_tracker.add_pending_files(self.tar_without_json)

            # Add processed files (with JSON) to state tracker
            if self.tar_with_json:
                self.state_tracker.add_processed_files(self.tar_with_json)

            return True

        except Exception as e:
            print(f"   Error listing TAR files: {e}")
            import traceback
            traceback.print_exc()
            return False

    def step2_upload_tar_files(self, files_to_upload: List[Dict] = None) -> bool:
        """Step 2: Upload TAR files to OpenAI Partner API"""
        if files_to_upload is None:
            self.print_header("STEP 2: Upload TAR Files (Initial Batch)")
            files_to_upload = self.tar_with_json
        else:
            print(f"\n   Uploading {len(files_to_upload)} newly ready TAR files...")

        if self.skip_upload:
            print("   Skipping upload step (--skip-upload flag set)")
            return True

        if not files_to_upload:
            print("   No TAR files to upload")
            return True

        cmd = [
            sys.executable,
            "tar_upload_pipeline.py",
            "--config", self.config_path
        ]

        success, output = self.run_command(cmd, "Uploading tar files", stream_output=True)

        if success:
            self._copy_state_file_to_data()
            print("   Upload completed successfully")
            return True
        else:
            print("   Upload failed")
            return False

    # =========================================================================
    # PHASE 2: POLLING
    # =========================================================================

    def _on_new_files_ready(self, newly_ready_files: List[Dict], cycle_number: int):
        """Callback when polling finds new JSON files"""
        print(f"\n   Cycle {cycle_number}: Found {len(newly_ready_files)} new JSON files!")

        # Update TAR listing CSV
        if self.generated_files.get('tar_listing'):
            update_csv_with_new_json_status(
                self.generated_files['tar_listing'],
                newly_ready_files
            )
            print(f"   Updated TAR listing CSV")

        # Upload newly ready files
        self.step2_upload_tar_files(newly_ready_files)

    def _on_cycle_complete(self, cycle_number: int, new_count: int, pending_count: int):
        """Callback after each polling cycle"""
        print(f"\n   Cycle {cycle_number} complete: {new_count} new, {pending_count} pending")

    def run_polling_phase(self) -> bool:
        """Run the polling phase to monitor for new JSON files"""
        if not self.polling_enabled:
            print("\n   Polling disabled")
            return True

        pending_files = self.state_tracker.get_pending_files()

        if not pending_files:
            print("\n   No pending files - skipping polling phase")
            return True

        self.print_header("PHASE 2: POLLING FOR NEW JSON METADATA FILES")

        print(f"   Polling Configuration:")
        print(f"     Pending files: {len(pending_files)}")
        print(f"     Interval: {self.polling_interval} minutes")
        print(f"     Max cycles: {self.polling_max_cycles}")
        print(f"     Total max time: {self.polling_interval * self.polling_max_cycles} minutes")
        print()

        # Initialize polling manager
        self.polling_manager = create_polling_manager_from_config(self.cfg)
        self.polling_manager.interval_minutes = self.polling_interval
        self.polling_manager.max_cycles = self.polling_max_cycles

        # Determine start cycle (for resume support)
        start_cycle = 1
        if self.resume:
            resume_info = self.state_tracker.get_resume_info()
            if resume_info.get('status') in ['polling', 'interrupted']:
                start_cycle = resume_info.get('current_cycle', 0) + 1
                print(f"   Resuming from cycle {start_cycle}")
                pending_files = self.state_tracker.get_pending_files()

        # Start new run if not resuming
        if not self.resume or start_cycle == 1:
            self.state_tracker.start_new_run(self.polling_max_cycles, self.polling_interval)

        # Run polling loop
        all_newly_ready, remaining_pending = self.polling_manager.run_polling_loop(
            pending_files=pending_files,
            on_new_files_ready=self._on_new_files_ready,
            on_cycle_complete=self._on_cycle_complete,
            state_tracker=self.state_tracker,
            start_cycle=start_cycle
        )

        # Mark polling as completed
        self.state_tracker.mark_polling_completed()

        # Print polling summary
        print(f"\n   Polling Summary:")
        print(f"     Total new JSONs found: {len(all_newly_ready)}")
        print(f"     Files still pending: {len(remaining_pending)}")

        if remaining_pending:
            print(f"\n   Files still missing JSON metadata (will be marked NOT_FOUND):")
            for f in remaining_pending[:5]:
                print(f"     - {f.get('filename', 'unknown')}")
            if len(remaining_pending) > 5:
                print(f"     ... and {len(remaining_pending) - 5} more")

        return True

    # =========================================================================
    # PHASE 3: FINAL REPORTING
    # =========================================================================

    def step3_generate_state_report(self) -> bool:
        """Step 3: Generate consolidated state reports"""
        self.print_header("STEP 3: Generate Consolidated State Reports (FINAL)")

        cmd = [
            sys.executable,
            "report_tar_state.py",
            "--config", self.config_path
        ]

        success, output = self.run_command(cmd, "Generating state reports")

        if success:
            # Sort by modification time to get the most recently created files
            json_files = sorted(
                self.consolidated_dir.glob(f"consolidated_tar_state_report_{self.project_id}_*.json"),
                key=lambda p: p.stat().st_mtime
            )
            csv_files = sorted(
                self.consolidated_dir.glob(f"consolidated_tar_files_{self.project_id}_*.csv"),
                key=lambda p: p.stat().st_mtime
            )

            if json_files:
                self.generated_files['state_report_json'] = str(json_files[-1])
                print(f"   State report JSON: {json_files[-1]}")

            if csv_files:
                self.generated_files['state_report_csv'] = str(csv_files[-1])
                print(f"   State report CSV: {csv_files[-1]}")

            return True
        else:
            print("   Warning: State report generation failed")
            return False

    def step4_compare_files(self) -> bool:
        """Step 4: Compare TAR files with state files"""
        self.print_header("STEP 4: Compare TAR Files with State Files (FINAL)")

        tar_listing = self.generated_files.get('tar_listing')
        state_csv = self.generated_files.get('state_report_csv')

        if not tar_listing:
            print("   Tar listing file not available - cannot perform comparison")
            return False

        if not state_csv:
            print("   State report CSV not available - cannot perform comparison")
            return False

        cmd = [
            sys.executable,
            "tar_state_file_comparison.py",
            "--tar", tar_listing,
            "--state", state_csv,
            "--output", str(self.comparison_dir),
            "--project-id", self.project_id
        ]

        # Add report date if extracted from source_prefix
        if self.report_date_prefix:
            cmd.extend(["--report-date", self.report_date_prefix])

        success, output = self.run_command(cmd, "Comparing files")

        if success:
            # Sort by modification time to get the most recently created files
            csv_files = sorted(
                self.comparison_dir.glob(f"*_tar_client_report.csv"),
                key=lambda p: p.stat().st_mtime
            )
            json_files = sorted(
                self.comparison_dir.glob(f"*_tar_client_report.json"),
                key=lambda p: p.stat().st_mtime
            )

            if csv_files:
                self.generated_files['comparison_csv'] = str(csv_files[-1])
                print(f"   Comparison CSV: {csv_files[-1]}")

            if json_files:
                self.generated_files['comparison_json'] = str(json_files[-1])
                print(f"   Comparison JSON: {json_files[-1]}")

            return True
        else:
            print("   Warning: File comparison failed")
            return False

    def step5_run_inspection(self) -> bool:
        """Step 5: Run TAR inspection on random samples

        When --skip-upload is set: Uses execution_mode from config (random/list mode)
        When --skip-upload is NOT set: Uses csv mode with comparison CSV
        """
        self.print_header("STEP 5: TAR Inspection (FINAL)")

        if self.skip_inspection:
            print("   Skipping inspection step (--skip-inspection flag set)")
            return True

        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'inspection' not in config:
                config['inspection'] = {}

            # Determine execution mode based on --skip-upload flag
            if self.skip_upload:
                # When skip_upload is set, use execution_mode from config (random/list)
                execution_mode = config.get('inspection', {}).get('execution_mode', 'random')
                print(f"   Running inspection in '{execution_mode}' mode (--skip-upload set)")
                print(f"   Reading TAR files from: {self.cfg.get('azure_source', 'source_prefix')}")

                if execution_mode == 'random':
                    sampling_ratio = config.get('inspection', {}).get('sampling_ratio', 10)
                    print(f"   Sampling ratio: 1 per {sampling_ratio} files (min 1)")
                elif execution_mode == 'list':
                    tar_file_list = config.get('inspection', {}).get('tar_file_list', [])
                    print(f"   Processing {len(tar_file_list)} specific TAR files from list")

                # Keep execution_mode from config (don't override to csv)
                config['inspection']['csv_report_dir'] = str(self.inspection_dir)
            else:
                # Normal flow: use csv mode with comparison CSV
                comparison_csv = self.generated_files.get('comparison_csv')
                if not comparison_csv or not os.path.exists(comparison_csv):
                    print("   Comparison CSV not available - cannot run inspection")
                    return False

                print(f"   Using comparison CSV: {comparison_csv}")
                print(f"   Random sample count: {self.inspection_sample_count}")

                config['inspection']['execution_mode'] = 'csv'
                config['inspection']['csv_input_file'] = comparison_csv
                config['inspection']['csv_sample_count'] = self.inspection_sample_count
                config['inspection']['csv_report_dir'] = str(self.inspection_dir)

            # Pass the extracted date from source_prefix for report naming
            if self.report_date_prefix:
                config['inspection']['report_date_prefix'] = self.report_date_prefix
                print(f"   Report date prefix: {self.report_date_prefix}")

            # Use /tmp for temp config file to avoid cluttering inspection_dir
            import tempfile
            temp_config_path = Path(tempfile.gettempdir()) / f"temp_inspection_config_{self.timestamp}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            inspector_script = self.script_dir.parent / "sharepoint_tar_inspector.py"
            if not inspector_script.exists():
                print(f"   Inspector script not found: {inspector_script}")
                # Cleanup temp file before returning
                try:
                    os.remove(temp_config_path)
                except Exception:
                    pass
                return False

            cmd = [
                sys.executable,
                str(inspector_script),
                "--config", str(temp_config_path)
            ]

            try:
                success, output = self.run_command(cmd, "Running TAR inspection", stream_output=True)

                # Always check for CSV report - it may be generated even if some TARs failed
                # Find the inspection report that matches this run's date prefix
                # or fall back to the most recently modified file
                inspection_reports = []
                if self.report_date_prefix:
                    # Look for files matching this run's date prefix
                    pattern = f"tar_inspection_report_{self.report_date_prefix}_*.csv"
                    inspection_reports = list(self.inspection_dir.glob(pattern))
                    if inspection_reports:
                        # Sort by modification time to get the latest one with this prefix
                        inspection_reports = sorted(inspection_reports, key=lambda p: p.stat().st_mtime)

                if not inspection_reports:
                    # Fallback: get all reports and sort by modification time (most recent last)
                    inspection_reports = sorted(
                        self.inspection_dir.glob("tar_inspection_report_*.csv"),
                        key=lambda p: p.stat().st_mtime
                    )

                if inspection_reports:
                    self.generated_files['inspection_report'] = str(inspection_reports[-1])
                    print(f"   Inspection report: {inspection_reports[-1]}")

                if success:
                    return True
                else:
                    print("   Warning: TAR inspection completed with some errors (CSV still generated)")
                    # Return True if CSV was generated, so uploads proceed
                    return bool(inspection_reports)
            finally:
                # Always cleanup temp config file
                try:
                    os.remove(temp_config_path)
                except Exception:
                    pass

        except Exception as e:
            print(f"   Error running inspection: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_summary(self):
        """Print final summary"""
        self.print_header("ORCHESTRATION COMPLETE - SUMMARY")

        # Get polling summary
        polling_summary = self.state_tracker.get_summary()

        print("   Pipeline Summary:")
        print()

        # Initial scan
        initial = polling_summary.get('initial_scan', {})
        print(f"   Initial Scan:")
        print(f"     Total TAR files: {initial.get('total_tar_files', 0)}")
        print(f"     With JSON: {initial.get('tar_with_json', 0)}")
        print(f"     Without JSON: {initial.get('tar_without_json', 0)}")
        print()

        # Polling results
        if self.polling_enabled:
            polling = polling_summary.get('polling', {})
            files = polling_summary.get('files', {})
            print(f"   Polling Results:")
            print(f"     Cycles completed: {polling.get('cycles_completed', 0)}/{polling.get('max_cycles', 0)}")
            print(f"     New JSONs found during polling: {polling.get('total_jsons_found_during_polling', 0)}")
            print(f"     Files uploaded (initial): {files.get('uploaded_initial', 0)}")
            print(f"     Files uploaded (polling): {files.get('uploaded_during_polling', 0)}")
            print(f"     Files still pending: {files.get('still_pending', 0)}")
            print()

        # Generated files
        print(f"   Generated Files:")
        if self.generated_files['tar_listing']:
            print(f"     TAR Listing: {self.generated_files['tar_listing']}")
        if self.generated_files['state_report_csv']:
            print(f"     State Report: {self.generated_files['state_report_csv']}")
        if self.generated_files['comparison_csv']:
            print(f"     Client Report: {self.generated_files['comparison_csv']}")
        if self.generated_files['inspection_report']:
            print(f"     Inspection Report: {self.generated_files['inspection_report']}")

        print()
        print("="*80)
        print(f"   All reports generated for project: {self.project_id}")
        print("="*80)

    def run(self):
        """Execute the complete pipeline"""
        print("\n" + "="*80)
        print("  TAR FILE ORCHESTRATOR WITH POLLING")
        print("="*80)
        print(f"Project ID: {self.project_id}")
        print(f"Config: {self.config_path}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Skip Upload: {self.skip_upload}")
        print(f"Skip Inspection: {self.skip_inspection}")
        print(f"Polling Enabled: {self.polling_enabled}")
        if self.polling_enabled:
            print(f"Polling Interval: {self.polling_interval} minutes")
            print(f"Polling Max Cycles: {self.polling_max_cycles}")
        if self.resume:
            print(f"Resume Mode: ENABLED")
        print("="*80)

        # Check for resume
        if self.resume and self.state_tracker.is_resumable():
            resume_info = self.state_tracker.get_resume_info()
            print(f"\n   Resuming from previous run:")
            print(f"     Run ID: {resume_info.get('run_id')}")
            print(f"     Status: {resume_info.get('status')}")
            print(f"     Cycle: {resume_info.get('current_cycle')}/{resume_info.get('max_cycles')}")
            print(f"     Pending: {resume_info.get('pending_count')}")

        results = {
            'phase1_step1': False,
            'phase1_step2': False,
            'phase2_polling': False,
            'phase3_step3': False,
            'phase3_step4': False,
            'phase3_step5': False
        }

        # ===== INSPECTION ONLY MODE (--skip-upload) =====
        if self.skip_upload and not self.skip_inspection:
            # When --skip-upload is set, run ONLY inspection using execution_mode from config
            self.print_header("INSPECTION ONLY MODE (--skip-upload)")
            print("   Skipping: Listing, Upload, Polling, State Report, Comparison")
            print("   Running: TAR Inspection using execution_mode from config")

            # Run inspection directly with random/list mode from config
            results['phase3_step5'] = self.step5_run_inspection()

            # Upload inspection report to blob and SharePoint
            self._upload_all_csv_reports()

            # Print simple summary for inspection-only mode
            print("\n" + "="*80)
            print("   INSPECTION ONLY - COMPLETE")
            print("="*80)
            if self.generated_files.get('inspection_report'):
                print(f"   Inspection Report: {self.generated_files['inspection_report']}")
            print("="*80)

            return 0 if results['phase3_step5'] else 1

        # ===== UPLOAD ONLY MODE (--skip-inspection) =====
        if self.skip_inspection and not self.skip_upload:
            self.print_header("UPLOAD MODE (--skip-inspection)")
            print("   Running: Listing, Upload, Polling, State Report, Comparison")
            print("   Skipping: TAR Inspection")

        # ===== PHASE 1: INITIAL PROCESSING =====
        self.print_header("PHASE 1: INITIAL PROCESSING")

        # Step 1: List and categorize TAR files
        results['phase1_step1'] = self.step1_list_and_categorize_tar_files()

        # Step 2: Upload TAR files with JSON (initial batch)
        if results['phase1_step1']:
            results['phase1_step2'] = self.step2_upload_tar_files()

        # ===== PHASE 2: POLLING =====
        if self.polling_enabled and self.tar_without_json:
            results['phase2_polling'] = self.run_polling_phase()
        else:
            results['phase2_polling'] = True
            if not self.polling_enabled:
                print("\n   Polling disabled - proceeding to final reports")
            elif not self.tar_without_json:
                print("\n   No pending files - proceeding to final reports")

        # ===== PHASE 3: FINAL REPORTING =====
        self.print_header("PHASE 3: FINAL REPORTING")

        # Step 3: Generate consolidated state report
        results['phase3_step3'] = self.step3_generate_state_report()

        # Step 4: Compare files
        results['phase3_step4'] = self.step4_compare_files()

        # Step 5: Run inspection (skipped if --skip-inspection)
        results['phase3_step5'] = self.step5_run_inspection()

        # Upload CSV reports (comparison + inspection) to blob and SharePoint
        self._upload_all_csv_reports()

        # Print summary
        self.print_summary()

        # Return overall success
        if self.skip_inspection:
            # For upload-only mode, critical steps are listing, state report, comparison
            critical_steps = ['phase1_step1', 'phase3_step3', 'phase3_step4']
        else:
            critical_steps = ['phase1_step1', 'phase3_step3', 'phase3_step4']

        all_critical_passed = all(results.get(step, False) for step in critical_steps)

        if all_critical_passed:
            print("\n   All critical steps completed successfully!")
            return 0
        else:
            print("\n   Some steps failed. Check the output above for details.")
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='TAR File Orchestrator with Polling Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW MODES:

  FULL WORKFLOW (default):
    PHASE 1: Initial Processing
      - Step 1: List TAR files and categorize (with/without JSON)
      - Step 2: Upload TAR files WITH JSON to client container
    PHASE 2: Polling (if enabled)
      - Poll for new JSON metadata files at configurable intervals
      - Upload newly ready TAR files as JSONs appear
    PHASE 3: Final Reporting
      - Step 3: Generate consolidated state reports
      - Step 4: Generate client comparison report
      - Step 5: Run TAR inspection (random sampling)
      - Step 6: Upload all reports to Azure Blob AND SharePoint

  INSPECTION ONLY MODE (--skip-upload):
    - Skips: Listing, Upload, Polling, State Report, Comparison
    - Runs: TAR Inspection using execution_mode from config (random/list)
    - Uses: azure_source.source_prefix path, inspection.sampling_ratio
    - Uploads: Inspection report to Azure Blob AND SharePoint

  UPLOAD ONLY MODE (--skip-inspection):
    - Runs: Listing, Upload, Polling, State Report, Comparison
    - Skips: TAR Inspection
    - Uploads: Comparison report to Azure Blob AND SharePoint

Examples:
  # Full workflow with polling
  python tar_orchestrator.py --config config/tar_pipeline_config.yaml

  # INSPECTION ONLY: Random sample from blob path (uses config execution_mode)
  python tar_orchestrator.py -c config/tar_pipeline_config.yaml --skip-upload

  # UPLOAD ONLY: Upload to client, generate comparison report
  python tar_orchestrator.py -c config/tar_pipeline_config.yaml --skip-inspection --no-polling

  # Override polling settings
  python tar_orchestrator.py -c config/tar_pipeline_config.yaml --polling-interval 10 --polling-cycles 24

  # Resume interrupted polling
  python tar_orchestrator.py -c config/tar_pipeline_config.yaml --resume
        """
    )

    parser.add_argument(
        '--config', '-c',
        default='config/tar_pipeline_config.yaml',
        help='Path to config YAML (default: config/tar_pipeline_config.yaml)'
    )

    parser.add_argument(
        '--skip-upload', '-s',
        action='store_true',
        help='INSPECTION ONLY MODE: Skip upload steps, run inspection using execution_mode from config (random/list)'
    )

    parser.add_argument(
        '--skip-inspection',
        action='store_true',
        help='UPLOAD ONLY MODE: Run upload and comparison, skip inspection step'
    )

    parser.add_argument(
        '--inspection-count', '-n',
        type=int,
        default=None,
        help='Number of random TAR files to inspect (default: from config or 4)'
    )

    parser.add_argument(
        '--no-polling',
        action='store_true',
        help='Disable polling (process only files with JSON at start)'
    )

    parser.add_argument(
        '--polling-interval',
        type=int,
        default=None,
        help='Polling interval in minutes (overrides config)'
    )

    parser.add_argument(
        '--polling-cycles',
        type=int,
        default=None,
        help='Maximum polling cycles (overrides config)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous interrupted polling run'
    )

    args = parser.parse_args()

    try:
        config_path = args.config
        if not os.path.isabs(config_path):
            script_dir = Path(__file__).parent
            config_path = str(script_dir / config_path)

        if not os.path.exists(config_path):
            print(f"   Error: Config file not found: {config_path}")
            return 1

        orchestrator = TarOrchestrator(
            config_path,
            skip_upload=args.skip_upload,
            skip_inspection=args.skip_inspection,
            inspection_sample_count=args.inspection_count,
            no_polling=args.no_polling,
            polling_interval=args.polling_interval,
            polling_cycles=args.polling_cycles,
            resume=args.resume
        )
        return orchestrator.run()

    except KeyboardInterrupt:
        print("\n\n   Orchestration interrupted by user")
        return 130
    except Exception as e:
        print(f"\n   Orchestration failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
