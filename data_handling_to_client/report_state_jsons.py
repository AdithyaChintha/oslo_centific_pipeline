#!/usr/bin/env python3
"""
Fetches all *_state.json from Azure blob storage and produces a
single consolidated JSON file for all homes.

Output:
  consolidated_state_report.json
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from azure.storage.blob import BlobServiceClient

# ---------------------- CONFIG ---------------------- #
AZURE_STORAGE_CONNECTION_STRING ="DefaultEndpointsProtocol=https;AccountName=oslotestvideo;AccountKey=zOevIegkZjld6ciTY+alA+YkzZ2gdAWVP7rkuhty5NAZ67AtiBB3fRTaa+eE3UbqhgwHZvWkOM0L+ASt4zCx6g==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = os.getenv("CONTAINER_NAME", "instavideo")
STATE_PREFIX = os.getenv("STATE_PREFIX", "upload_state_files/customer/")
INCLUDE_SUFFIX = "_state.json"
OUTPUT_FILE = "consolidated_state_report.json"
# ----------------------------------------------------- #


def human_gb(bytes_val: Optional[float]) -> Optional[float]:
    if bytes_val is None:
        return None
    try:
        return round(float(bytes_val) / (1024 ** 3), 3)
    except Exception:
        return None


def list_state_blobs(bsc: BlobServiceClient) -> List[str]:
    container = bsc.get_container_client(CONTAINER_NAME)
    return [
        b.name for b in container.list_blobs(name_starts_with=STATE_PREFIX)
        if b.name.lower().endswith(INCLUDE_SUFFIX)
    ]


def download_json(bsc: BlobServiceClient, blob_name: str) -> Dict[str, Any]:
    container = bsc.get_container_client(CONTAINER_NAME)
    data = container.download_blob(blob_name).readall()
    return json.loads(data)


def extract_summary(meta: Dict[str, Any], daily: Dict[str, Any], blob_name: str) -> Dict[str, Any]:
    files_successful = sum(day.get("files_successful", 0) for day in daily.values())
    video_files_successful = sum(day.get("video_files_successful", 0) for day in daily.values())
    audio_files_successful = sum(day.get("audio_files_successful", 0) for day in daily.values())
    files_failed = sum(day.get("files_failed", 0) for day in daily.values())
    total_upload_time = round(sum(day.get("total_upload_time_seconds", 0.0) for day in daily.values()), 3)

    return {
        "state_blob_name": blob_name,
        "home_id": meta.get("home_id"),
        "version": meta.get("version"),
        # "created_at": meta.get("created_at"),
        # "last_updated": meta.get("last_updated"),
        "total_data_processed_gb": meta.get("total_data_processed_gb")
            or human_gb(meta.get("total_data_processed_bytes")),
        "total_processing_time_seconds": meta.get("total_processing_time_seconds"),
        "files_processed": meta.get("total_files_processed"),
        "video_files_processed": meta.get("total_video_files_processed"),
        "audio_files_processed": meta.get("total_audio_files_processed"),
        "video_files_successful": video_files_successful,
        "audio_files_successful": audio_files_successful,
        "files_failed": files_failed,
        # "sum_upload_time_seconds": total_upload_time
    }


def flatten_file_entries(state_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = state_json.get("metadata", {})
    daily = state_json.get("daily_processing", {})
    out = []

    for day, day_block in daily.items():
        for session in day_block.get("processing_sessions", []):
            session_id = session.get("session_id")
            container_id = session.get("container_id")
            files = session.get("videos") or session.get("files") or []

            for item in files:
                status = item.get("upload_status", "unknown")
                blob_name = item.get("upload_blob_name") #or item.get("blob_name")
                file_type = "video" if blob_name and blob_name.lower().endswith(
                    (".insv", ".mp4", ".mov", ".mkv")) else (
                        "audio" if blob_name and blob_name.lower().endswith(
                            (".wav", ".mp3", ".flac", ".aac")) else "unknown"
                )

                out.append({
                    "day": day,
                    # "session_id": session_id,
                    "container_id": container_id,
                    "status": status,
                    "file_type": file_type,
                    "blob_name": blob_name,
                    # "file_size_bytes": item.get("file_size")
                    #     or item.get("fingerprint", {}).get("size"),
                    "file_size_gb": human_gb(
                        item.get("file_size")
                        or item.get("fingerprint", {}).get("size")
                    ),
                    "upload_duration_seconds": item.get("upload_duration_seconds"),
                    # "last_modified": item.get("last_modified"),
                    # "error_details": item.get("error_details")
                })
    return out

def upload_to_blob(bsc: BlobServiceClient, local_path: str, target_blob_name: str):
    """
    Upload a local file to Azure Blob Storage.
    """
    container_client = bsc.get_container_client(CONTAINER_NAME)
    print(f"Uploading {local_path} to {CONTAINER_NAME}/{target_blob_name} ...")
    with open(local_path, "rb") as data:
        container_client.upload_blob(
            name=target_blob_name,
            data=data,
            overwrite=True   # so we can replace older versions
        )
    print("✅ Upload completed.")



def build_consolidated_json():
    bsc = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    state_blobs = list_state_blobs(bsc)

    if not state_blobs:
        print("No state JSON files found.")
        return

    consolidated = {"generated_at": datetime.utcnow().isoformat() + "Z", "homes": []}

    for blob_name in state_blobs:
        print(f"Processing {blob_name} ...")
        state_json = download_json(bsc, blob_name)

        meta = state_json.get("metadata", {})
        daily = state_json.get("daily_processing", {})

        home_entry = {
            "summary": extract_summary(meta, daily, blob_name),
            "files": flatten_file_entries(state_json)
        }

        consolidated["homes"].append(home_entry)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\n✅ Consolidated JSON saved to {OUTPUT_FILE}")


        # === Upload to Azure ===
    target_blob_name = f"{STATE_PREFIX.rstrip('/')}/consolidated_state_report.json"
    upload_to_blob(bsc, OUTPUT_FILE, target_blob_name)

    print(f"✅ Also uploaded to Azure as: {CONTAINER_NAME}/{target_blob_name}")


if __name__ == "__main__":
    build_consolidated_json()
