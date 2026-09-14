#!/usr/bin/env python3
"""
Fetches the tar upload state JSON from Azure blob storage and produces consolidated reports
in multiple formats.

Since there is only ONE state file for all tar files (not per home like video/audio),
this script is simpler than the home-based video reporting.

Outputs:
  consolidated_tar_state_report.json  - Complete JSON report
  consolidated_tar_files.csv          - Detailed file-level data with summary

All files are also uploaded to Azure Blob Storage.
"""

import os
import json
import csv
import argparse
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from azure.storage.blob import BlobServiceClient


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


def human_gb(bytes_val: Optional[float]) -> Optional[float]:
    """Convert bytes to GB with 3 decimal precision"""
    if bytes_val is None:
        return None
    try:
        return round(float(bytes_val) / (1024 ** 3), 3)
    except Exception:
        return None


def list_tar_state_blobs(bsc: BlobServiceClient, container_name: str, state_prefix: str) -> List[str]:
    """List all tar state JSON files from blob storage"""
    container = bsc.get_container_client(container_name)
    return [
        b.name for b in container.list_blobs(name_starts_with=state_prefix)
        if b.name.lower().endswith("_state.json")
    ]


def download_json(bsc: BlobServiceClient, container_name: str, blob_name: str) -> Dict[str, Any]:
    """Download and parse JSON from blob storage"""
    container = bsc.get_container_client(container_name)
    data = container.download_blob(blob_name).readall()
    return json.loads(data)


def extract_summary(meta: Dict[str, Any], daily: Dict[str, Any], blob_name: str) -> Dict[str, Any]:
    """Extract summary information from state JSON metadata"""
    files_successful = sum(day.get("files_successful", 0) for day in daily.values())
    files_failed = sum(day.get("files_failed", 0) for day in daily.values())
    total_upload_time = round(sum(day.get("total_upload_time_seconds", 0.0) for day in daily.values()), 3)

    return {
        "state_blob_name": blob_name,
        "project_id": meta.get("home_id") or meta.get("project_id"),  # home_id field used for project_id
        "version": meta.get("version"),
        "created_at": meta.get("created_at"),
        "last_updated": meta.get("last_updated"),
        "total_data_processed_gb": meta.get("total_data_processed_gb")
            or human_gb(meta.get("total_data_processed_bytes")),
        "total_processing_time_seconds": meta.get("total_processing_time_seconds"),
        "files_processed": meta.get("total_files_processed"),
        "files_successful": files_successful,
        "files_failed": files_failed,
        "total_upload_time_seconds": total_upload_time
    }


def flatten_file_entries(state_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all file entries from the state JSON"""
    meta = state_json.get("metadata", {})
    daily = state_json.get("daily_processing", {})
    out = []

    for day, day_block in daily.items():
        for session in day_block.get("processing_sessions", []):
            session_id = session.get("session_id")
            container_id = session.get("container_id")
            session_start = session.get("session_start_time")
            session_end = session.get("session_end_time")

            # Tar files are stored in "videos" field (reusing existing structure)
            files = session.get("videos") or session.get("files") or []

            for item in files:
                status = item.get("upload_status", "unknown")
                blob_name = item.get("blob_name", "")

                # Extract tar UUID from filename
                tar_uuid = None
                if blob_name:
                    filename = blob_name.split('/')[-1]
                    if filename.lower().endswith('.tar'):
                        tar_uuid = filename[:-4]

                file_size_bytes = item.get("file_size") or item.get("fingerprint", {}).get("size")

                out.append({
                    "day": day,
                    "session_id": session_id,
                    "container_id": container_id,
                    "session_start_time": session_start,
                    "session_end_time": session_end,
                    "status": status,
                    "tar_uuid": tar_uuid,
                    "blob_name": blob_name,
                    "file_size_bytes": file_size_bytes,
                    "file_size_gb": human_gb(file_size_bytes),
                    "upload_duration_seconds": item.get("upload_duration_seconds"),
                    "upload_start_time": item.get("upload_start_time"),
                    "upload_end_time": item.get("upload_end_time"),
                    "last_modified": item.get("last_modified"),
                    "metadata_uploaded": item.get("metadata_uploaded"),
                    "error_details": item.get("error_details")
                })
    return out


def upload_to_blob(bsc: BlobServiceClient, container_name: str, local_path: str, target_blob_name: str):
    """Upload a local file to Azure Blob Storage"""
    container_client = bsc.get_container_client(container_name)
    print(f"Uploading {local_path} to {container_name}/{target_blob_name} ...")
    with open(local_path, "rb") as data:
        container_client.upload_blob(
            name=target_blob_name,
            data=data,
            overwrite=True
        )
    print("✅ Upload completed.")


def export_files_to_csv(consolidated_data: Dict[str, Any], output_file: str) -> None:
    """Export detailed file data to CSV format with summary information"""
    print(f"📁 Exporting tar files data with summary to {output_file}...")

    # Ensure output directory exists
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'project_id', 'row_type', 'day', 'session_id', 'container_id',
            'session_start_time', 'session_end_time', 'status', 'tar_uuid',
            'blob_name', 'file_size_bytes', 'file_size_gb',
            'upload_duration_seconds', 'upload_start_time', 'upload_end_time',
            'last_modified', 'metadata_uploaded', 'error_details',
            'state_blob_name', 'total_data_processed_gb',
            'total_processing_time_seconds', 'files_processed',
            'files_successful', 'files_failed', 'total_upload_time_seconds'
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Since there's only one state file for all tar files
        summary = consolidated_data.get("summary", {})
        files = consolidated_data.get("files", [])

        # File rows
        for file_entry in files:
            row = {
                'project_id': summary.get("project_id"),
                'row_type': 'FILE',
                'day': file_entry.get("day"),
                'session_id': file_entry.get("session_id"),
                'container_id': file_entry.get("container_id"),
                'session_start_time': file_entry.get("session_start_time"),
                'session_end_time': file_entry.get("session_end_time"),
                'status': file_entry.get("status"),
                'tar_uuid': file_entry.get("tar_uuid"),
                'blob_name': file_entry.get("blob_name"),
                'file_size_bytes': file_entry.get("file_size_bytes"),
                'file_size_gb': file_entry.get("file_size_gb"),
                'upload_duration_seconds': file_entry.get("upload_duration_seconds"),
                'upload_start_time': file_entry.get("upload_start_time"),
                'upload_end_time': file_entry.get("upload_end_time"),
                'last_modified': file_entry.get("last_modified"),
                'metadata_uploaded': file_entry.get("metadata_uploaded"),
                'error_details': file_entry.get("error_details"),
                'state_blob_name': '',
                'total_data_processed_gb': '',
                'total_processing_time_seconds': '',
                'files_processed': '',
                'files_successful': '',
                'files_failed': '',
                'total_upload_time_seconds': ''
            }
            writer.writerow(row)

        # Summary row at the end
        summary_row = {
            'project_id': summary.get("project_id"),
            'row_type': 'SUMMARY',
            'day': '',
            'session_id': '',
            'container_id': '',
            'session_start_time': '',
            'session_end_time': '',
            'status': '',
            'tar_uuid': '',
            'blob_name': '',
            'file_size_bytes': '',
            'file_size_gb': '',
            'upload_duration_seconds': '',
            'upload_start_time': '',
            'upload_end_time': '',
            'last_modified': '',
            'metadata_uploaded': '',
            'error_details': '',
            'state_blob_name': summary.get("state_blob_name"),
            'total_data_processed_gb': summary.get("total_data_processed_gb"),
            'total_processing_time_seconds': summary.get("total_processing_time_seconds"),
            'files_processed': summary.get("files_processed"),
            'files_successful': summary.get("files_successful"),
            'files_failed': summary.get("files_failed"),
            'total_upload_time_seconds': summary.get("total_upload_time_seconds")
        }
        writer.writerow(summary_row)

    print(f"✅ Files CSV with summary saved to {output_file}")


def build_consolidated_report(cfg: ConfigLoader):
    """Main function to build consolidated tar state report"""

    # Load configuration
    connection_string = cfg.get("state_storage", "connection_string")
    container_name = cfg.get("state_storage", "container_name")
    state_prefix = cfg.get("state_storage", "prefix", default="upload_state_files/production/tar-bulk-")
    project_id = cfg.get("azure_source", "project_id", default="tar-upload")

    # Output paths
    output_dir = Path("/data/oslo/consolidated_report/tar")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_output_file = str(output_dir / f"consolidated_tar_state_report_{project_id}_{timestamp}.json")
    csv_output_file = str(output_dir / f"consolidated_tar_files_{project_id}_{timestamp}.csv")

    if not connection_string or not container_name:
        raise ValueError("Azure connection string or container name missing in config")

    print("=== Tar State Report Generator ===")
    print(f"Project ID: {project_id}")
    print(f"Container: {container_name}")
    print(f"State Prefix: {state_prefix}")
    print()

    bsc = BlobServiceClient.from_connection_string(connection_string)
    state_blobs = list_tar_state_blobs(bsc, container_name, state_prefix)

    if not state_blobs:
        print("❌ No tar state JSON files found.")
        return

    print(f"Found {len(state_blobs)} tar state file(s):")
    for blob in state_blobs:
        print(f"  - {blob}")
    print()

    # Since there's only one state file, process it directly
    if len(state_blobs) > 1:
        print(f"⚠️  Warning: Found multiple state files. Using the first one: {state_blobs[0]}")

    blob_name = state_blobs[0]
    print(f"Processing {blob_name} ...")
    state_json = download_json(bsc, container_name, blob_name)

    meta = state_json.get("metadata", {})
    daily = state_json.get("daily_processing", {})

    consolidated = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "project_id": project_id,
        "summary": extract_summary(meta, daily, blob_name),
        "files": flatten_file_entries(state_json)
    }

    # Save JSON
    with open(json_output_file, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\n✅ Consolidated JSON saved to {json_output_file}")

    # Export to CSV
    export_files_to_csv(consolidated, csv_output_file)

    # Upload to Azure
    print("\n📤 Uploading reports to Azure Blob Storage...")

    # Upload JSON
    json_target_blob = f"{state_prefix.rstrip('-')}/consolidated_tar_state_report.json"
    upload_to_blob(bsc, container_name, json_output_file, json_target_blob)
    print(f"✅ JSON uploaded to Azure as: {container_name}/{json_target_blob}")

    # Upload CSV file
    csv_target_blob = f"{state_prefix.rstrip('-')}/consolidated_tar_files.csv"
    upload_to_blob(bsc, container_name, csv_output_file, csv_target_blob)
    print(f"✅ CSV uploaded to Azure as: {container_name}/{csv_target_blob}")

    # Print summary statistics
    print("\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    summary = consolidated.get("summary", {})
    print(f"Project ID: {summary.get('project_id')}")
    print(f"Total files processed: {summary.get('files_processed', 0)}")
    print(f"Files successful: {summary.get('files_successful', 0)}")
    print(f"Files failed: {summary.get('files_failed', 0)}")
    print(f"Total data processed: {summary.get('total_data_processed_gb', 0)} GB")
    print(f"Total upload time: {summary.get('total_upload_time_seconds', 0)} seconds")
    print("="*60)

    print(f"\n🎉 All files generated and uploaded successfully!")
    print(f"   📄 JSON: {json_output_file}")
    print(f"   📁 CSV: {csv_output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate consolidated tar state report")
    parser.add_argument(
        "--config", "-c",
        default="config/tar_upload_config.yaml",
        help="Path to tar_upload_config.yaml (default: config/tar_upload_config.yaml)"
    )
    args = parser.parse_args()

    try:
        # Load configuration
        config_path = args.config
        if not os.path.isabs(config_path):
            # Make relative to script directory
            script_dir = Path(__file__).parent
            config_path = str(script_dir / config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"Loading config from: {config_path}\n")
        cfg = ConfigLoader(config_path)

        build_consolidated_report(cfg)

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
