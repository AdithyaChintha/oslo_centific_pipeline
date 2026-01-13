#!/usr/bin/env python3
"""
Verification script for inspection pipeline reports.

This script validates:
1. Scenario IDs in the inspection report CSV match capture.json files in Azure Blob
2. Deduplication flags (isDupScn) are correctly set
3. Video file to scenario mapping is correct (videos belong to correct scenarios)
4. Duplicate scenarios in CSV match actual duplicates from capture.json

Usage:
    # Read CSV from local file:
    python verify_inspection_report.py --csv-report /path/to/report.csv --config config/tar_pipeline_config.yaml

    # Read CSV from Azure Blob (auto-discover latest inspection report):
    python verify_inspection_report.py --csv-from-blob --config config/tar_pipeline_config.yaml

    # Read specific CSV from Azure Blob:
    python verify_inspection_report.py --csv-blob-path "one-data-platform/hummus_prod/2026-01-12/inspection_report.csv" --config config/tar_pipeline_config.yaml

    # Limit to N TAR files:
    python verify_inspection_report.py --csv-from-blob --config config/tar_pipeline_config.yaml --limit 5
"""

import os
import io
import json
import csv
import argparse
import logging
import yaml
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

try:
    from azure.storage.blob import BlobServiceClient, ContainerClient
except ImportError:
    print("ERROR: azure-storage-blob package not installed.")
    print("Install with: pip install azure-storage-blob")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose Azure SDK HTTP logging
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)
logging.getLogger('azure.storage.blob').setLevel(logging.WARNING)
logging.getLogger('azure.core').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

# Video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.insv'}

# Suffixes added by the pipeline during processing
# Order matters: check longer suffixes first
PROCESSING_SUFFIXES = [
    '_erp_480p',    # Fisheye converted to ERP + downscaled
    '_erpview',     # Alternative ERP naming
    '_erp',         # Fisheye converted to ERP only
    '_480p',        # Downscaled only
]


@dataclass
class ScenarioMapping:
    """Stores scenario to video mapping from capture.json."""
    scenario_id: str
    videos: List[str] = field(default_factory=list)
    all_artifacts: List[str] = field(default_factory=list)
    capture_json_path: str = ""


@dataclass
class VerificationResult:
    """Stores verification results for a single TAR file."""
    tar_name: str

    # Scenario ID verification
    scenario_id_matches: int = 0
    scenario_id_mismatches: int = 0

    # Deduplication verification
    dedup_correct: int = 0
    dedup_incorrect: int = 0

    # Video mapping verification
    video_mapping_correct: int = 0
    video_mapping_incorrect: int = 0

    # Missing files
    missing_in_csv: List[str] = field(default_factory=list)
    missing_in_folder: List[str] = field(default_factory=list)

    # Detailed errors
    errors: List[str] = field(default_factory=list)
    scenario_id_errors: List[str] = field(default_factory=list)
    dedup_errors: List[str] = field(default_factory=list)
    video_mapping_errors: List[str] = field(default_factory=list)

    # Capture.json stats
    capture_json_count: int = 0
    unique_scenarios: int = 0

    # Duplicate scenario tracking
    actual_duplicate_scenarios: Set[str] = field(default_factory=set)  # From capture.json
    csv_duplicate_scenarios: Set[str] = field(default_factory=set)     # From CSV (isDupScn='Y')
    duplicate_mismatch: bool = False

    # Scenario to video mapping (from capture.json)
    scenario_to_videos: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return (self.scenario_id_mismatches == 0 and
                self.dedup_incorrect == 0 and
                self.video_mapping_incorrect == 0 and
                not self.duplicate_mismatch and
                len(self.errors) == 0)


class InspectionReportVerifier:
    """Verifies inspection report CSV against actual untarred folder contents in Azure Blob."""

    def __init__(self, config_path: str, csv_report_path: str = None,
                 csv_from_blob: bool = False, csv_blob_path: str = None):
        """
        Initialize the verifier.

        Args:
            config_path: Path to the pipeline config YAML
            csv_report_path: Path to local inspection report CSV (optional)
            csv_from_blob: If True, auto-discover inspection report from blob
            csv_blob_path: Specific blob path to inspection report CSV (optional)
        """
        self.config = self._load_config(config_path)
        self.container_client = self._init_blob_client()
        self.untar_blob_prefix = self.config['inspection']['untar_blob_prefix']
        self.results: Dict[str, VerificationResult] = {}

        # Determine CSV source
        self.csv_report_path = csv_report_path
        self.csv_from_blob = csv_from_blob
        self.csv_blob_path = csv_blob_path
        self.csv_source_description = None

    def _load_config(self, config_path: str) -> dict:
        """Load the pipeline configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _init_blob_client(self) -> ContainerClient:
        """Initialize Azure Blob container client."""
        connection_string = self.config['azure_blob']['connection_string']
        container_name = self.config['azure_blob']['container_name']

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        return blob_service_client.get_container_client(container_name)

    def _find_latest_inspection_report(self) -> Optional[str]:
        """
        Find the latest inspection report CSV in blob storage.

        Searches in the csv_report_storage prefix for files matching
        'tar_inspection_report*.csv' pattern.

        Returns:
            Blob path to the latest inspection report, or None if not found
        """
        csv_prefix = self.config.get('csv_report_storage', {}).get('prefix', '')
        if not csv_prefix:
            logger.error("csv_report_storage.prefix not configured")
            return None

        logger.info(f"Searching for inspection reports in: {csv_prefix}")

        inspection_reports = []
        try:
            blobs = self.container_client.list_blobs(name_starts_with=csv_prefix)
            for blob in blobs:
                # Match inspection report files
                if 'inspection' in blob.name.lower() and blob.name.endswith('.csv'):
                    inspection_reports.append({
                        'name': blob.name,
                        'last_modified': blob.last_modified
                    })
        except Exception as e:
            logger.error(f"Error listing blobs: {e}")
            return None

        if not inspection_reports:
            logger.warning(f"No inspection report CSVs found in {csv_prefix}")
            return None

        # Sort by last modified, get latest
        inspection_reports.sort(key=lambda x: x['last_modified'], reverse=True)
        latest = inspection_reports[0]

        logger.info(f"Found {len(inspection_reports)} inspection reports")
        logger.info(f"Using latest: {latest['name']} (modified: {latest['last_modified']})")

        return latest['name']

    def _download_csv_from_blob(self, blob_path: str) -> str:
        """
        Download CSV content from blob storage.

        Args:
            blob_path: Full blob path to the CSV file

        Returns:
            CSV content as string
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            data = blob_client.download_blob().readall()
            return data.decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to download CSV from blob: {blob_path}. Error: {e}")

    def _is_video_file(self, filename: str) -> bool:
        """Check if a filename is a video file based on extension."""
        ext = os.path.splitext(filename)[1].lower()
        return ext in VIDEO_EXTENSIONS

    def _get_original_filename(self, processed_filename: str) -> str:
        """
        Strip processing suffixes to get the original filename.

        The pipeline adds suffixes like _erp, _480p, _erp_480p during processing.
        capture.json contains original filenames, so we need to strip these
        to look up the scenario ID.

        Example:
            P02_scenarioRunner_c019_erp_480p.mp4 -> P02_scenarioRunner_c019.mp4
            P02_scenarioRunner_c019_480p.mp4 -> P02_scenarioRunner_c019.mp4
            P02_scenarioRunner_c019.mp4 -> P02_scenarioRunner_c019.mp4
        """
        name, ext = os.path.splitext(processed_filename)

        # Try to strip each suffix (longer ones first)
        for suffix in PROCESSING_SUFFIXES:
            if name.endswith(suffix):
                original_name = name[:-len(suffix)]
                return original_name + ext

        # No suffix found, return as-is
        return processed_filename

    def load_csv_report(self) -> Dict[str, List[dict]]:
        """
        Load the inspection report CSV and group by TAR file name.

        Supports loading from:
        - Local file (csv_report_path)
        - Specific blob path (csv_blob_path)
        - Auto-discovered latest inspection report (csv_from_blob)

        Returns:
            Dictionary mapping tar_file_name to list of row dicts
        """
        csv_content = None

        # Determine source and load CSV content
        if self.csv_report_path:
            # Load from local file
            local_path = Path(self.csv_report_path)
            if not local_path.exists():
                raise FileNotFoundError(f"CSV report not found: {local_path}")
            with open(local_path, 'r', newline='', encoding='utf-8') as f:
                csv_content = f.read()
            self.csv_source_description = f"Local file: {local_path}"

        elif self.csv_blob_path:
            # Load from specific blob path
            csv_content = self._download_csv_from_blob(self.csv_blob_path)
            self.csv_source_description = f"Blob: {self.csv_blob_path}"

        elif self.csv_from_blob:
            # Auto-discover latest inspection report
            blob_path = self._find_latest_inspection_report()
            if not blob_path:
                raise RuntimeError("Could not find inspection report in blob storage")
            csv_content = self._download_csv_from_blob(blob_path)
            self.csv_source_description = f"Blob (auto-discovered): {blob_path}"

        else:
            raise ValueError("No CSV source specified. Use --csv-report, --csv-blob-path, or --csv-from-blob")

        logger.info(f"Loading CSV from: {self.csv_source_description}")

        # Parse CSV content
        csv_data = defaultdict(list)
        reader = csv.DictReader(io.StringIO(csv_content))
        for row in reader:
            tar_name = row.get('tar_file_name', '')
            if tar_name:
                csv_data[tar_name].append(row)

        logger.info(f"Loaded CSV report with {len(csv_data)} TAR files")
        return dict(csv_data)

    def find_capture_json_blobs(self, tar_folder_prefix: str) -> List[str]:
        """
        Find all capture.json files in a TAR folder in blob storage.

        Args:
            tar_folder_prefix: Blob prefix for the untarred folder

        Returns:
            List of blob paths to capture.json files
        """
        capture_files = []
        try:
            blobs = self.container_client.list_blobs(name_starts_with=tar_folder_prefix)
            for blob in blobs:
                if blob.name.endswith('capture.json'):
                    capture_files.append(blob.name)
        except Exception as e:
            logger.warning(f"Error listing blobs under {tar_folder_prefix}: {e}")

        return capture_files

    def read_capture_json(self, blob_path: str) -> Optional[dict]:
        """
        Read and parse a capture.json file from blob storage.

        Args:
            blob_path: Full blob path to the capture.json

        Returns:
            Parsed JSON content or None if failed
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            data = blob_client.download_blob().readall()
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to read {blob_path}: {e}")
            return None

    def build_detailed_scenario_mapping(self, tar_folder_prefix: str) -> Tuple[
            Dict[str, str],           # file_to_scenario
            Set[str],                 # duplicate_scenarios
            int,                      # capture_json_count
            Dict[str, ScenarioMapping]  # scenario_details
        ]:
        """
        Build detailed scenario mapping from capture.json files in blob storage.

        Args:
            tar_folder_prefix: Blob prefix for the untarred folder

        Returns:
            Tuple of:
            - file_to_scenario: mapping filename -> scenarioId
            - duplicate_scenarios: set of scenarioIds appearing >1 time
            - capture_json_count: number of capture.json files found
            - scenario_details: detailed mapping scenarioId -> ScenarioMapping
        """
        file_to_scenario = {}
        scenario_counts = defaultdict(int)
        scenario_details: Dict[str, ScenarioMapping] = {}

        capture_files = self.find_capture_json_blobs(tar_folder_prefix)

        for capture_blob_path in capture_files:
            capture_data = self.read_capture_json(capture_blob_path)
            if capture_data is None:
                continue

            scenario_id = capture_data.get('scenarioId', '') or ''  # Treat None as empty string
            if scenario_id:
                scenario_counts[scenario_id] += 1

                # Create or update scenario mapping
                if scenario_id not in scenario_details:
                    scenario_details[scenario_id] = ScenarioMapping(
                        scenario_id=scenario_id,
                        capture_json_path=capture_blob_path
                    )

            # Map artifact files to scenario
            for artifact in capture_data.get('artifacts', []):
                artifact_file = artifact.get('file', '')
                if artifact_file:
                    artifact_filename = os.path.basename(artifact_file)
                    file_to_scenario[artifact_filename] = scenario_id  # Will be '' if no scenarioId

                    # Track in scenario details
                    if scenario_id and scenario_id in scenario_details:
                        scenario_details[scenario_id].all_artifacts.append(artifact_filename)
                        if self._is_video_file(artifact_filename):
                            scenario_details[scenario_id].videos.append(artifact_filename)

        # Find duplicate scenarios (appear more than once)
        duplicate_scenarios = {sid for sid, count in scenario_counts.items() if count > 1}

        logger.debug(f"Found {len(capture_files)} capture.json files, "
                     f"{len(file_to_scenario)} file mappings, "
                     f"{len(duplicate_scenarios)} duplicate scenarios")

        return file_to_scenario, duplicate_scenarios, len(capture_files), scenario_details

    def check_tar_folder_exists(self, tar_folder_prefix: str) -> bool:
        """
        Check if the untarred folder exists in blob storage.

        Args:
            tar_folder_prefix: Blob prefix to check

        Returns:
            True if folder exists (has at least one blob)
        """
        try:
            blobs = self.container_client.list_blobs(name_starts_with=tar_folder_prefix)
            # Check if there's at least one blob
            for _ in blobs:
                return True
            return False
        except Exception as e:
            logger.warning(f"Error checking folder existence: {e}")
            return False

    def verify_tar_folder(self, tar_name: str, csv_rows: List[dict]) -> VerificationResult:
        """
        Verify a single TAR folder against CSV entries.

        Args:
            tar_name: Name of the TAR file
            csv_rows: List of CSV rows for this TAR

        Returns:
            VerificationResult with detailed findings
        """
        result = VerificationResult(tar_name=tar_name)

        # Build the blob prefix for this TAR's untarred folder
        folder_name = tar_name.replace('.tar', '')
        tar_folder_prefix = f"{self.untar_blob_prefix}/{folder_name}/"

        # Check if folder exists
        if not self.check_tar_folder_exists(tar_folder_prefix):
            # Try with .tar in folder name
            tar_folder_prefix = f"{self.untar_blob_prefix}/{tar_name}/"
            if not self.check_tar_folder_exists(tar_folder_prefix):
                result.errors.append(f"Untarred folder not found in blob: {folder_name}")
                return result

        # Build detailed scenario mapping from capture.json files
        file_to_scenario, actual_duplicates, capture_count, scenario_details = \
            self.build_detailed_scenario_mapping(tar_folder_prefix)

        result.capture_json_count = capture_count
        result.unique_scenarios = len(scenario_details)
        result.actual_duplicate_scenarios = actual_duplicates
        result.scenario_to_videos = {sid: sm.videos for sid, sm in scenario_details.items()}

        # Track CSV data for comparison
        csv_files = {row.get('individual_file_name', '') for row in csv_rows}
        csv_scenario_to_files: Dict[str, List[str]] = defaultdict(list)
        csv_dup_scenarios: Set[str] = set()

        # Verify each CSV row
        for row in csv_rows:
            filename = row.get('individual_file_name', '')
            csv_scenario_id = row.get('scenarioID', '')
            csv_is_dup = row.get('isDupScn', '')

            # Track CSV scenario to files mapping
            if csv_scenario_id:
                csv_scenario_to_files[csv_scenario_id].append(filename)
                if csv_is_dup == 'Y':
                    csv_dup_scenarios.add(csv_scenario_id)

            # Get actual scenario from capture.json mapping
            # First try the processed filename, then try the original filename
            # (capture.json has original names like P02_scenarioRunner_c019.mp4
            #  but CSV has processed names like P02_scenarioRunner_c019_erp_480p.mp4)
            actual_scenario_id = file_to_scenario.get(filename, '')
            original_filename = None
            if not actual_scenario_id:
                original_filename = self._get_original_filename(filename)
                if original_filename != filename:
                    actual_scenario_id = file_to_scenario.get(original_filename, '')

            # 1. Verify scenario ID matches
            if csv_scenario_id == actual_scenario_id:
                result.scenario_id_matches += 1
            else:
                result.scenario_id_mismatches += 1
                error_msg = (f"Scenario ID mismatch for '{filename}': "
                           f"CSV='{csv_scenario_id}', Actual='{actual_scenario_id}'")
                result.scenario_id_errors.append(error_msg)
                result.errors.append(error_msg)

            # 2. Verify deduplication flag
            expected_is_dup = ''
            if actual_scenario_id:
                expected_is_dup = 'Y' if actual_scenario_id in actual_duplicates else 'N'

            if csv_is_dup == expected_is_dup:
                result.dedup_correct += 1
            else:
                result.dedup_incorrect += 1
                error_msg = (f"Dedup flag mismatch for '{filename}' (scenario={actual_scenario_id}): "
                           f"CSV='{csv_is_dup}', Expected='{expected_is_dup}'")
                result.dedup_errors.append(error_msg)
                result.errors.append(error_msg)

            # 3. Verify video mapping (for video files)
            if self._is_video_file(filename) and actual_scenario_id:
                # Check if this video is in the correct scenario's artifact list
                # Use original filename for lookup since capture.json has original names
                scenario_info = scenario_details.get(actual_scenario_id)
                lookup_filename = original_filename if original_filename else filename
                if scenario_info:
                    if lookup_filename in scenario_info.all_artifacts:
                        result.video_mapping_correct += 1
                    else:
                        result.video_mapping_incorrect += 1
                        error_msg = (f"Video '{filename}' (original: '{lookup_filename}') not in scenario "
                                   f"'{actual_scenario_id}' artifacts. Expected: {scenario_info.videos[:3]}")
                        result.video_mapping_errors.append(error_msg)
                        result.errors.append(error_msg)

        # Store CSV duplicate scenarios for comparison
        result.csv_duplicate_scenarios = csv_dup_scenarios

        # 4. Cross-check duplicate scenarios between CSV and capture.json
        if actual_duplicates != csv_dup_scenarios:
            result.duplicate_mismatch = True

            # Find discrepancies
            only_in_actual = actual_duplicates - csv_dup_scenarios
            only_in_csv = csv_dup_scenarios - actual_duplicates

            if only_in_actual:
                error_msg = f"Duplicate scenarios in capture.json but not marked in CSV: {only_in_actual}"
                result.errors.append(error_msg)

            if only_in_csv:
                error_msg = f"Scenarios marked as duplicate in CSV but not in capture.json: {only_in_csv}"
                result.errors.append(error_msg)

        # Check for files in capture.json but missing from CSV
        for filename in file_to_scenario:
            if filename not in csv_files:
                result.missing_in_csv.append(filename)

        return result

    def verify_all(self, limit: Optional[int] = None) -> Dict[str, VerificationResult]:
        """
        Verify all TAR folders against the CSV report.

        Args:
            limit: Optional limit on number of TAR files to verify

        Returns:
            Dictionary mapping TAR names to their verification results
        """
        csv_data = self.load_csv_report()

        tar_names = list(csv_data.keys())
        if limit:
            tar_names = tar_names[:limit]

        logger.info(f"Verifying {len(tar_names)} TAR files against blob storage...")
        logger.info(f"Blob prefix: {self.untar_blob_prefix}")

        for i, tar_name in enumerate(tar_names, 1):
            logger.info(f"[{i}/{len(tar_names)}] Verifying {tar_name}")
            result = self.verify_tar_folder(tar_name, csv_data[tar_name])
            self.results[tar_name] = result

            if result.is_valid:
                logger.info(f"  ✅ Valid - {result.scenario_id_matches} scenario IDs, "
                            f"{result.dedup_correct} dedup, {result.video_mapping_correct} video mappings")
                if result.actual_duplicate_scenarios:
                    logger.info(f"     Duplicate scenarios: {result.actual_duplicate_scenarios}")
            else:
                logger.warning(f"  ❌ Issues found - {len(result.errors)} errors")
                for error in result.errors[:3]:
                    logger.warning(f"     - {error}")
                if len(result.errors) > 3:
                    logger.warning(f"     ... and {len(result.errors) - 3} more")

        return self.results

    def print_summary(self):
        """Print a summary of all verification results."""
        if not self.results:
            logger.info("No results to summarize")
            return

        total_valid = sum(1 for r in self.results.values() if r.is_valid)
        total_invalid = len(self.results) - total_valid
        total_not_found = sum(1 for r in self.results.values()
                             if any("not found" in e.lower() for e in r.errors))

        # Scenario ID stats
        total_scenario_matches = sum(r.scenario_id_matches for r in self.results.values())
        total_scenario_mismatches = sum(r.scenario_id_mismatches for r in self.results.values())

        # Dedup stats
        total_dedup_correct = sum(r.dedup_correct for r in self.results.values())
        total_dedup_incorrect = sum(r.dedup_incorrect for r in self.results.values())

        # Video mapping stats
        total_video_correct = sum(r.video_mapping_correct for r in self.results.values())
        total_video_incorrect = sum(r.video_mapping_incorrect for r in self.results.values())

        # Duplicate mismatch count
        total_dup_mismatch = sum(1 for r in self.results.values() if r.duplicate_mismatch)

        # Count TAR files with duplicates
        tars_with_duplicates = sum(1 for r in self.results.values() if r.actual_duplicate_scenarios)

        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        print(f"\nCSV Source: {self.csv_source_description}")
        print(f"Untar Blob Prefix: {self.untar_blob_prefix}")

        print(f"\nTAR Files Verified: {len(self.results)}")
        print(f"  ✅ Valid: {total_valid}")
        print(f"  ❌ Invalid: {total_invalid}")
        print(f"  📂 Not found in blob: {total_not_found}")

        print(f"\n{'─' * 40}")
        print("1. SCENARIO ID VERIFICATION")
        print(f"{'─' * 40}")
        print(f"  Matches: {total_scenario_matches}")
        print(f"  Mismatches: {total_scenario_mismatches}")
        accuracy = (total_scenario_matches / (total_scenario_matches + total_scenario_mismatches) * 100
                    if (total_scenario_matches + total_scenario_mismatches) > 0 else 0)
        print(f"  Accuracy: {accuracy:.2f}%")

        print(f"\n{'─' * 40}")
        print("2. DEDUPLICATION FLAG VERIFICATION")
        print(f"{'─' * 40}")
        print(f"  Correct: {total_dedup_correct}")
        print(f"  Incorrect: {total_dedup_incorrect}")
        dedup_accuracy = (total_dedup_correct / (total_dedup_correct + total_dedup_incorrect) * 100
                         if (total_dedup_correct + total_dedup_incorrect) > 0 else 0)
        print(f"  Accuracy: {dedup_accuracy:.2f}%")

        print(f"\n{'─' * 40}")
        print("3. VIDEO-TO-SCENARIO MAPPING VERIFICATION")
        print(f"{'─' * 40}")
        print(f"  Correct mappings: {total_video_correct}")
        print(f"  Incorrect mappings: {total_video_incorrect}")
        video_accuracy = (total_video_correct / (total_video_correct + total_video_incorrect) * 100
                         if (total_video_correct + total_video_incorrect) > 0 else 0)
        print(f"  Accuracy: {video_accuracy:.2f}%")

        print(f"\n{'─' * 40}")
        print("4. DUPLICATE SCENARIO CROSS-CHECK")
        print(f"{'─' * 40}")
        print(f"  TAR files with duplicate scenarios: {tars_with_duplicates}")
        print(f"  TAR files with CSV/capture.json mismatch: {total_dup_mismatch}")

        # Print details for invalid TAR files
        invalid_results = [(name, r) for name, r in self.results.items() if not r.is_valid]
        if invalid_results:
            print("\n" + "=" * 80)
            print("INVALID TAR FILES DETAILS")
            print("=" * 80)

            for tar_name, result in invalid_results[:20]:  # Limit to 20
                print(f"\n📁 {tar_name}")
                print(f"   capture.json files: {result.capture_json_count}")
                print(f"   Unique scenarios: {result.unique_scenarios}")

                if result.actual_duplicate_scenarios:
                    print(f"   Actual duplicate scenarios (capture.json): {result.actual_duplicate_scenarios}")
                if result.csv_duplicate_scenarios:
                    print(f"   CSV duplicate scenarios (isDupScn='Y'): {result.csv_duplicate_scenarios}")
                if result.duplicate_mismatch:
                    print(f"   ⚠️  DUPLICATE SCENARIO MISMATCH between CSV and capture.json!")

                # Show scenario to video mappings
                if result.scenario_to_videos:
                    print(f"   Scenario -> Video mappings from capture.json:")
                    for sid, videos in list(result.scenario_to_videos.items())[:5]:
                        print(f"      {sid}: {videos[:3]}{'...' if len(videos) > 3 else ''}")
                    if len(result.scenario_to_videos) > 5:
                        print(f"      ... and {len(result.scenario_to_videos) - 5} more scenarios")

                # Show errors by category
                if result.scenario_id_errors:
                    print(f"   Scenario ID errors ({len(result.scenario_id_errors)}):")
                    for err in result.scenario_id_errors[:2]:
                        print(f"      ❌ {err}")
                    if len(result.scenario_id_errors) > 2:
                        print(f"      ... and {len(result.scenario_id_errors) - 2} more")

                if result.dedup_errors:
                    print(f"   Dedup errors ({len(result.dedup_errors)}):")
                    for err in result.dedup_errors[:2]:
                        print(f"      ❌ {err}")
                    if len(result.dedup_errors) > 2:
                        print(f"      ... and {len(result.dedup_errors) - 2} more")

                if result.video_mapping_errors:
                    print(f"   Video mapping errors ({len(result.video_mapping_errors)}):")
                    for err in result.video_mapping_errors[:2]:
                        print(f"      ❌ {err}")
                    if len(result.video_mapping_errors) > 2:
                        print(f"      ... and {len(result.video_mapping_errors) - 2} more")

            if len(invalid_results) > 20:
                print(f"\n... and {len(invalid_results) - 20} more invalid TAR files")

        print("\n" + "=" * 80)

    def export_results(self, output_path: str):
        """
        Export verification results to a CSV file.

        Args:
            output_path: Path for the output CSV
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'tar_name', 'is_valid',
                'scenario_id_matches', 'scenario_id_mismatches', 'scenario_id_accuracy',
                'dedup_correct', 'dedup_incorrect', 'dedup_accuracy',
                'video_mapping_correct', 'video_mapping_incorrect', 'video_mapping_accuracy',
                'capture_json_count', 'unique_scenarios',
                'actual_duplicate_scenarios', 'csv_duplicate_scenarios', 'duplicate_mismatch',
                'missing_in_csv_count', 'error_count',
                'scenario_id_error_count', 'dedup_error_count', 'video_mapping_error_count',
                'first_error'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for tar_name, result in self.results.items():
                # Calculate accuracies
                total_scenario = result.scenario_id_matches + result.scenario_id_mismatches
                scenario_accuracy = (result.scenario_id_matches / total_scenario * 100) if total_scenario > 0 else 0

                total_dedup = result.dedup_correct + result.dedup_incorrect
                dedup_accuracy = (result.dedup_correct / total_dedup * 100) if total_dedup > 0 else 0

                total_video = result.video_mapping_correct + result.video_mapping_incorrect
                video_accuracy = (result.video_mapping_correct / total_video * 100) if total_video > 0 else 0

                writer.writerow({
                    'tar_name': tar_name,
                    'is_valid': 'YES' if result.is_valid else 'NO',
                    'scenario_id_matches': result.scenario_id_matches,
                    'scenario_id_mismatches': result.scenario_id_mismatches,
                    'scenario_id_accuracy': f"{scenario_accuracy:.2f}%",
                    'dedup_correct': result.dedup_correct,
                    'dedup_incorrect': result.dedup_incorrect,
                    'dedup_accuracy': f"{dedup_accuracy:.2f}%",
                    'video_mapping_correct': result.video_mapping_correct,
                    'video_mapping_incorrect': result.video_mapping_incorrect,
                    'video_mapping_accuracy': f"{video_accuracy:.2f}%",
                    'capture_json_count': result.capture_json_count,
                    'unique_scenarios': result.unique_scenarios,
                    'actual_duplicate_scenarios': ','.join(sorted(result.actual_duplicate_scenarios)),
                    'csv_duplicate_scenarios': ','.join(sorted(result.csv_duplicate_scenarios)),
                    'duplicate_mismatch': 'YES' if result.duplicate_mismatch else 'NO',
                    'missing_in_csv_count': len(result.missing_in_csv),
                    'error_count': len(result.errors),
                    'scenario_id_error_count': len(result.scenario_id_errors),
                    'dedup_error_count': len(result.dedup_errors),
                    'video_mapping_error_count': len(result.video_mapping_errors),
                    'first_error': result.errors[0] if result.errors else ''
                })

        logger.info(f"Results exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify inspection report CSV against untarred folder contents in Azure Blob"
    )

    # CSV source options (mutually exclusive)
    csv_group = parser.add_mutually_exclusive_group(required=True)
    csv_group.add_argument(
        '--csv-report', '-c',
        help="Path to local inspection report CSV"
    )
    csv_group.add_argument(
        '--csv-from-blob', '-b',
        action='store_true',
        help="Auto-discover and use latest inspection report from blob storage"
    )
    csv_group.add_argument(
        '--csv-blob-path',
        help="Specific blob path to inspection report CSV"
    )

    # Config and other options
    parser.add_argument(
        '--config', '-f',
        default='config/tar_pipeline_config.yaml',
        help="Path to the pipeline config YAML (default: config/tar_pipeline_config.yaml)"
    )
    parser.add_argument(
        '--output', '-o',
        help="Path to export verification results CSV"
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help="Limit verification to N TAR files"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    verifier = InspectionReportVerifier(
        config_path=args.config,
        csv_report_path=args.csv_report,
        csv_from_blob=args.csv_from_blob,
        csv_blob_path=args.csv_blob_path
    )

    verifier.verify_all(limit=args.limit)
    verifier.print_summary()

    if args.output:
        verifier.export_results(args.output)


if __name__ == '__main__':
    main()
