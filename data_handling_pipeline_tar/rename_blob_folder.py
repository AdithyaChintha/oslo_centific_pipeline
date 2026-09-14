#!/usr/bin/env python3
"""
Rename a folder (prefix) in Azure Blob Storage.

Since Azure Blob Storage doesn't have actual folders, this script:
1. Lists all blobs with the old prefix
2. Copies each blob to the new prefix
3. Deletes the original blobs

Usage:
    python rename_blob_folder.py -c config/tar_pipeline_config.yaml \
        --old-prefix "path/to/old_folder" \
        --new-prefix "path/to/new_folder" \ 2026-02-03
        [--container CONTAINER_NAME] \
        [--dry-run]
"""

import argparse
import yaml
from azure.storage.blob import BlobServiceClient
from urllib.parse import urlparse


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def rename_blob_folder(
    connection_string: str,
    container_name: str,
    old_prefix: str,
    new_prefix: str,
    dry_run: bool = False
) -> tuple[int, int]:
    """
    Rename a folder in blob storage by copying blobs to new prefix and deleting originals.

    Args:
        connection_string: Azure Blob Storage connection string
        container_name: Container name
        old_prefix: The old folder prefix (e.g., "folder/subfolder")
        new_prefix: The new folder prefix (e.g., "folder/renamed_subfolder")
        dry_run: If True, only print what would be done without making changes

    Returns:
        Tuple of (copied_count, failed_count)
    """
    # Ensure prefixes don't have trailing slashes for consistency
    old_prefix = old_prefix.rstrip('/')
    new_prefix = new_prefix.rstrip('/')

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # List all blobs with the old prefix
    blobs_to_rename = list(container_client.list_blobs(name_starts_with=old_prefix))

    if not blobs_to_rename:
        print(f"No blobs found with prefix: {old_prefix}")
        return 0, 0

    print(f"Found {len(blobs_to_rename)} blob(s) to rename")
    print(f"  From: {old_prefix}")
    print(f"  To:   {new_prefix}")
    print()

    copied_count = 0
    failed_count = 0

    for blob in blobs_to_rename:
        old_name = blob.name
        # Replace the old prefix with new prefix
        new_name = new_prefix + old_name[len(old_prefix):]

        print(f"  {old_name}")
        print(f"    -> {new_name}")

        if dry_run:
            copied_count += 1
            continue

        try:
            # Get source and destination blob clients
            source_blob = container_client.get_blob_client(old_name)
            dest_blob = container_client.get_blob_client(new_name)

            # Copy blob to new location
            dest_blob.start_copy_from_url(source_blob.url)

            # Wait for copy to complete and delete source
            copy_props = dest_blob.get_blob_properties()
            if copy_props.copy.status == 'success':
                source_blob.delete_blob()
                copied_count += 1
                print(f"    ✓ Done")
            else:
                failed_count += 1
                print(f"    ✗ Copy failed: {copy_props.copy.status}")

        except Exception as e:
            failed_count += 1
            print(f"    ✗ Error: {e}")

    return copied_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="Rename a folder (prefix) in Azure Blob Storage"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Path to YAML config file with connection string"
    )
    parser.add_argument(
        "--old-prefix",
        required=True,
        help="Old folder prefix to rename (e.g., 'folder/old_name')"
    )
    parser.add_argument(
        "--new-prefix",
        required=True,
        help="New folder prefix (e.g., 'folder/new_name')"
    )
    parser.add_argument(
        "--container",
        help="Container name (defaults to azure_source.container_name from config)"
    )
    parser.add_argument(
        "--config-section",
        default="azure_source",
        choices=["azure_source", "azure_blob", "state_storage", "csv_report_storage", "remote_manifests"],
        help="Which config section to use for connection string (default: azure_source)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without making changes"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Get connection string and container from specified config section
    section = config.get(args.config_section, {})
    connection_string = section.get("connection_string")
    container_name = args.container or section.get("container_name")

    if not connection_string:
        print(f"Error: No connection_string found in config section '{args.config_section}'")
        return 1

    if not container_name:
        print(f"Error: No container name specified and none found in config section '{args.config_section}'")
        return 1

    print(f"Container: {container_name}")
    print(f"Config section: {args.config_section}")
    if args.dry_run:
        print("Mode: DRY RUN (no changes will be made)")
    print()

    copied, failed = rename_blob_folder(
        connection_string=connection_string,
        container_name=container_name,
        old_prefix=args.old_prefix,
        new_prefix=args.new_prefix,
        dry_run=args.dry_run
    )

    print()
    print(f"Summary: {copied} renamed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
