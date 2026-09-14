#!/usr/bin/env python3
"""
Tar File Lister
Lists all tar files from Azure Blob Storage based on configuration in tar_upload_config.yaml

Enhanced with:
- Categorization: separates TAR files with/without JSON metadata
- Re-check functionality: can re-check specific files for new JSON files
- Support for polling workflow
"""

import os
import csv
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
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


def get_file_type(filename):
    """Determine file type based on extension"""
    filename_lower = filename.lower()

    if filename_lower.endswith(".tar"):
        return "tar"
    elif filename_lower.endswith(".tar.gz") or filename_lower.endswith(".tgz"):
        return "compressed_tar"
    elif filename_lower.endswith(".json"):
        return "metadata"
    else:
        return "other"


def extract_tar_info(blob_name, source_prefix):
    """Extract tar file information from blob path

    Example blob_name: oslo_stage_2_tar_files/a8fac332-3cee-43d9-b062-b1fc89c9b50c.tar
    All tar files are stored flat in the same directory (no home ID subdirectories)
    """
    # Extract just the filename (UUID.tar)
    filename = blob_name.split('/')[-1]

    # Extract UUID from filename (without .tar extension)
    tar_uuid = None
    if filename.lower().endswith('.tar'):
        tar_uuid = filename[:-4]  # Remove .tar extension
    elif filename.lower().endswith('.tar.gz'):
        tar_uuid = filename[:-7]  # Remove .tar.gz extension

    return {
        "tar_uuid": tar_uuid,
        "filename": filename,
        "full_blob_path": blob_name,
        "source_prefix": source_prefix
    }


def get_json_path_for_tar(tar_blob_name: str) -> str:
    """Get the expected JSON file path for a tar file.

    Example:
        one-data-platform/.../test-1gb.tar -> one-data-platform/.../test-1gb.json
    """
    if tar_blob_name.lower().endswith('.tar'):
        return tar_blob_name[:-4] + '.json'
    return tar_blob_name + '.json'


def check_json_exists(container_client, json_blob_name: str) -> bool:
    """Check if a JSON metadata file exists in the container"""
    try:
        container_client.get_blob_client(json_blob_name).get_blob_properties()
        return True
    except Exception:
        return False


def list_tar_files(cfg: ConfigLoader):
    """List all tar files from source container based on config"""

    # Load configuration
    connection_string = cfg.get("azure_source", "connection_string")
    container_name = cfg.get("azure_source", "container_name")
    source_prefix = cfg.get("azure_source", "source_prefix", default="")
    project_id = cfg.get("azure_source", "project_id", default="tar-upload")

    if not connection_string or not container_name:
        raise ValueError("Azure connection string or container name missing in config")

    print("=== Tar File Lister ===")
    print(f"Project ID: {project_id}")
    print(f"Container: {container_name}")
    print(f"Source Prefix: {source_prefix or '(root)'}")
    print()

    # Initialize Azure Blob Service Client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    all_files = []
    tar_count = 0
    metadata_count = 0
    other_count = 0

    # List all blobs with the specified prefix
    print(f"Scanning container for files...")
    if source_prefix:
        blobs = container_client.list_blobs(name_starts_with=source_prefix)
    else:
        blobs = container_client.list_blobs()

    for blob in blobs:
        # Skip if it's a directory marker (ends with /)
        if blob.name.endswith('/'):
            continue

        # Extract tar file information
        tar_info = extract_tar_info(blob.name, source_prefix)

        # Get file type
        file_type = get_file_type(blob.name)

        # Count by type
        if file_type == "tar":
            tar_count += 1
        elif file_type == "compressed_tar":
            tar_count += 1
        elif file_type == "metadata":
            metadata_count += 1
        else:
            other_count += 1

        # Check if corresponding JSON metadata exists for tar files
        has_metadata_json = False
        expected_json_path = ""
        if file_type in ["tar", "compressed_tar"]:
            expected_json_path = get_json_path_for_tar(blob.name)
            has_metadata_json = check_json_exists(container_client, expected_json_path)

        file_info = {
            "project_id": project_id,
            "tar_uuid": tar_info["tar_uuid"],
            "filename": tar_info["filename"],
            "file_type": file_type,
            "file_size_bytes": blob.size,
            "file_size_mb": round(blob.size / (1024 * 1024), 2) if blob.size else 0,
            "file_size_gb": round(blob.size / (1024 * 1024 * 1024), 3) if blob.size else 0,
            "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
            "etag": blob.etag,
            "full_blob_path": blob.name,
            "blob_url": f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob.name}",
            "has_metadata_json": has_metadata_json,
            "expected_json_path": expected_json_path
        }

        all_files.append(file_info)

    # Count tar files with/without metadata JSON
    tar_with_json = sum(1 for f in all_files if f["file_type"] in ["tar", "compressed_tar"] and f["has_metadata_json"])
    tar_without_json = sum(1 for f in all_files if f["file_type"] in ["tar", "compressed_tar"] and not f["has_metadata_json"])

    print(f"  Found {tar_count} tar files")
    print(f"    - With metadata JSON: {tar_with_json}")
    print(f"    - Without metadata JSON: {tar_without_json}")
    print(f"  Found {metadata_count} metadata files")
    print(f"  Found {other_count} other files")
    print(f"  Total files: {len(all_files)}")

    return all_files, {
        "tar_files": tar_count,
        "tar_with_json": tar_with_json,
        "tar_without_json": tar_without_json,
        "metadata_files": metadata_count,
        "other_files": other_count,
        "total_files": len(all_files)
    }


def save_to_csv(files_data, filename, project_id):
    """Save file data to CSV"""

    if not files_data:
        print("No files found to save!")
        return

    # Ensure output directory exists
    output_dir = Path(filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define CSV columns
    fieldnames = [
        "project_id",
        "tar_uuid",
        "filename",
        "file_type",
        "file_size_bytes",
        "file_size_mb",
        "file_size_gb",
        "last_modified",
        "etag",
        "full_blob_path",
        "blob_url",
        "has_metadata_json",
        "expected_json_path"
    ]

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(files_data)

    print(f"CSV file saved: {filename}")


def print_summary(files_data, stats, project_id):
    """Print summary statistics"""

    print("\n=== SUMMARY ===")
    print(f"Project ID: {project_id}")
    print(f"Total files found: {stats['total_files']}")
    print(f"  Tar files: {stats['tar_files']}")
    print(f"    - With metadata JSON: {stats.get('tar_with_json', 0)}")
    print(f"    - Without metadata JSON: {stats.get('tar_without_json', 0)}")
    print(f"  Metadata files: {stats['metadata_files']}")
    print(f"  Other files: {stats['other_files']}")

    # Calculate total size
    total_size_bytes = sum(f["file_size_bytes"] for f in files_data)
    total_size_gb = round(total_size_bytes / (1024 * 1024 * 1024), 2)
    print(f"\nTotal size: {total_size_gb} GB")

    # Count tar files only
    tar_files = [f for f in files_data if f["file_type"] in ["tar", "compressed_tar"]]
    if tar_files:
        tar_size_bytes = sum(f["file_size_bytes"] for f in tar_files)
        tar_size_gb = round(tar_size_bytes / (1024 * 1024 * 1024), 2)
        print(f"Tar files total size: {tar_size_gb} GB")

        # Show largest tar files
        print("\nLargest tar files:")
        sorted_tars = sorted(tar_files, key=lambda x: x["file_size_bytes"], reverse=True)[:10]
        for i, f in enumerate(sorted_tars, 1):
            print(f"  {i}. {f['filename']} - {f['file_size_gb']} GB")


# =============================================================================
# CATEGORIZATION FUNCTIONS (for polling workflow)
# =============================================================================

def list_tar_files_categorized(cfg: ConfigLoader) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    List all tar files and categorize them by JSON metadata availability.

    This is the main function used by the polling workflow.

    Args:
        cfg: ConfigLoader instance

    Returns:
        Tuple of:
        - all_tar_files: All TAR files found
        - tar_with_json: TAR files that have corresponding JSON metadata
        - tar_without_json: TAR files missing JSON metadata (pending)
        - stats: Statistics dictionary
    """
    # Get all files using existing function
    all_files, stats = list_tar_files(cfg)

    # Filter only TAR files
    all_tar_files = [
        f for f in all_files
        if f.get("file_type") in ["tar", "compressed_tar"]
    ]

    # Categorize by JSON availability
    tar_with_json = [f for f in all_tar_files if f.get("has_metadata_json", False)]
    tar_without_json = [f for f in all_tar_files if not f.get("has_metadata_json", False)]

    print(f"\n=== CATEGORIZATION ===")
    print(f"Total TAR files: {len(all_tar_files)}")
    print(f"  Ready (with JSON): {len(tar_with_json)}")
    print(f"  Pending (without JSON): {len(tar_without_json)}")

    return all_tar_files, tar_with_json, tar_without_json, stats


def recheck_files_for_json(
    cfg: ConfigLoader,
    files_to_check: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Re-check a list of TAR files to see if their JSON metadata now exists.

    Used during polling to check if pending files have received their JSON.

    Args:
        cfg: ConfigLoader instance
        files_to_check: List of TAR file info dicts to re-check

    Returns:
        Tuple of:
        - newly_ready: Files that now have JSON metadata
        - still_pending: Files still missing JSON metadata
    """
    connection_string = cfg.get("azure_source", "connection_string")
    container_name = cfg.get("azure_source", "container_name")

    if not connection_string or not container_name:
        raise ValueError("Azure connection string or container name missing in config")

    # Initialize Azure client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    newly_ready = []
    still_pending = []

    for file_info in files_to_check:
        blob_path = file_info.get("full_blob_path", "")
        if not blob_path:
            still_pending.append(file_info)
            continue

        # Get expected JSON path
        json_path = get_json_path_for_tar(blob_path)

        # Check if JSON now exists
        if check_json_exists(container_client, json_path):
            # JSON found - update file info
            updated_file = file_info.copy()
            updated_file["has_metadata_json"] = True
            updated_file["json_found_at"] = datetime.utcnow().isoformat() + "Z"
            newly_ready.append(updated_file)
        else:
            still_pending.append(file_info)

    return newly_ready, still_pending


def update_csv_with_new_json_status(
    csv_path: str,
    updated_files: List[Dict],
    output_path: Optional[str] = None
) -> str:
    """
    Update a TAR listing CSV with new JSON status for specific files.

    Args:
        csv_path: Path to existing TAR listing CSV
        updated_files: List of file dicts with updated has_metadata_json=True
        output_path: Output path (defaults to overwriting input)

    Returns:
        Path to updated CSV
    """
    if output_path is None:
        output_path = csv_path

    # Build lookup of updated files by tar_uuid
    updated_lookup = {f.get("tar_uuid"): f for f in updated_files if f.get("tar_uuid")}

    # Read existing CSV
    rows = []
    fieldnames = None

    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames
        for row in reader:
            tar_uuid = row.get("tar_uuid")
            if tar_uuid in updated_lookup:
                # Update this row with new JSON status
                row["has_metadata_json"] = "True"
            rows.append(row)

    # Write updated CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated CSV saved: {output_path}")
    return output_path


class TarFileListerWithPolling:
    """
    TAR file lister with support for polling workflow.

    Provides methods for:
    - Initial scan with categorization
    - Re-checking specific files
    - Updating CSV files
    """

    def __init__(self, cfg: ConfigLoader):
        """
        Initialize the lister.

        Args:
            cfg: ConfigLoader instance
        """
        self.cfg = cfg
        self.connection_string = cfg.get("azure_source", "connection_string")
        self.container_name = cfg.get("azure_source", "container_name")
        self.source_prefix = cfg.get("azure_source", "source_prefix", default="")
        self.project_id = cfg.get("azure_source", "project_id", default="tar-upload")

        if not self.connection_string or not self.container_name:
            raise ValueError("Azure connection string or container name missing in config")

        self._blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self._container_client = self._blob_service_client.get_container_client(self.container_name)

    def scan_all_files(self) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
        """
        Scan all TAR files and categorize by JSON availability.

        Returns:
            Tuple of (all_tar_files, tar_with_json, tar_without_json, stats)
        """
        return list_tar_files_categorized(self.cfg)

    def check_json_for_file(self, tar_blob_path: str) -> bool:
        """
        Check if JSON metadata exists for a specific TAR file.

        Args:
            tar_blob_path: Full blob path to TAR file

        Returns:
            True if JSON exists
        """
        json_path = get_json_path_for_tar(tar_blob_path)
        return check_json_exists(self._container_client, json_path)

    def recheck_pending_files(self, pending_files: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Re-check pending files for new JSON metadata.

        Args:
            pending_files: List of TAR file info dicts

        Returns:
            Tuple of (newly_ready, still_pending)
        """
        return recheck_files_for_json(self.cfg, pending_files)

    def save_categorized_csv(
        self,
        all_files: List[Dict],
        output_dir: str,
        timestamp: Optional[str] = None
    ) -> str:
        """
        Save categorized TAR files to CSV.

        Args:
            all_files: List of all TAR file info dicts
            output_dir: Output directory
            timestamp: Optional timestamp string

        Returns:
            Path to saved CSV
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        csv_filename = str(output_path / f"tar_files_{self.project_id}_{timestamp}.csv")
        save_to_csv(all_files, csv_filename, self.project_id)

        return csv_filename


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="List tar files from Azure Blob Storage")
    parser.add_argument(
        "--config", "-c",
        default="config/tar_upload_config.yaml",
        help="Path to tar_upload_config.yaml (default: config/tar_upload_config.yaml)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV file path (default: /data/oslo/tar_files/tar_files_{project_id}_{timestamp}.csv)"
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

        # Get project_id from config
        project_id = cfg.get("azure_source", "project_id", default="tar-upload")

        # List all tar files
        files_data, stats = list_tar_files(cfg)

        if not files_data:
            print("No files found!")
            return

        # Generate output filename with timestamp
        if args.output:
            csv_filename = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("/data/oslo/tar_files")
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_filename = str(output_dir / f"tar_files_{project_id}_{timestamp}.csv")

        # Save to CSV
        save_to_csv(files_data, csv_filename, project_id)

        # Print summary
        print_summary(files_data, stats, project_id)

        print(f"\nDetailed file list saved to: {csv_filename}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
