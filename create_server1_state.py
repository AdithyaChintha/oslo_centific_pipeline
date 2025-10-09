#!/usr/bin/env python3
"""
Create state file for server 1 by reading all Excel sheets and marking videos as processed.
"""
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from glob import glob

def read_excel_files(excel_dir):
    """Read all Excel files and extract video information."""
    videos = {}
    excel_files = sorted(glob(os.path.join(excel_dir, "*.xlsx")))

    print(f"Found {len(excel_files)} Excel files to process")

    for excel_file in excel_files:
        print(f"Processing: {os.path.basename(excel_file)}")
        try:
            df = pd.read_excel(excel_file)

            for idx, row in df.iterrows():
                # Extract s3_key - it has s3:// prefix, remove it
                s3_key_raw = row.get('s3_key')
                if not s3_key_raw or pd.isna(s3_key_raw):
                    continue

                # Remove s3:// prefix and bucket name to get the actual key
                # Format: s3://troveo-videodb-shared/outputs/batch_2_tier_1/clips/...
                # We want: outputs/batch_2_tier_1/clips/...
                s3_key = s3_key_raw.replace('s3://troveo-videodb-shared/', '')

                # Get source_s3_key and clean it
                source_s3_key_raw = row.get('source_s3_key')
                source_s3_key = source_s3_key_raw.replace('s3://troveo-videodb-shared/', '') if source_s3_key_raw and not pd.isna(source_s3_key_raw) else None

                # Extract source_video_id and clip_id for display filename
                source_video_id = row.get('source_video_id') if not pd.isna(row.get('source_video_id')) else None
                clip_id = row.get('clip_id') if not pd.isna(row.get('clip_id')) else None

                # Construct output_path similar to server2
                # Format: /data/pyrenees_output/batch2/server1/{source_video_id}_{clip_id_without_ext}
                output_path = None
                if source_video_id and clip_id:
                    clip_id_no_ext = clip_id.replace('.mov', '')
                    output_path = f"/data/pyrenees_output/batch2/server1/{source_video_id}_{clip_id_no_ext}"

                # Construct display_filename
                display_filename = row.get('filename') if not pd.isna(row.get('filename')) else None

                # Get NSFW and minor detection status
                nsfw_detected = False
                minors_detected = False
                if 'nsfw_segments_count' in df.columns and not pd.isna(row.get('nsfw_segments_count')):
                    nsfw_detected = int(row['nsfw_segments_count']) > 0
                if 'minor_segments_count' in df.columns and not pd.isna(row.get('minor_segments_count')):
                    minors_detected = int(row['minor_segments_count']) > 0

                # Create video entry with processed status
                video_entry = {
                    "s3_key": s3_key,
                    "s3_etag": None,
                    "size_bytes": None,
                    "s3_last_modified": None,
                    "status": "completed",
                    "discovered_at": datetime.now().isoformat(),
                    "downloaded_at": datetime.now().isoformat(),
                    "processing_start": datetime.now().isoformat(),
                    "processed_at": datetime.now().isoformat(),
                    "processing_time_seconds": 0,
                    "local_path": None,
                    "output_path": output_path,
                    "s3_video_url": None,
                    "s3_results_path": None,
                    "labelstudio_task_id": None,
                    "nsfw_detected": nsfw_detected,
                    "minors_detected": minors_detected,
                    "error": None,
                    "retry_count": 0,
                    "download_start": datetime.now().isoformat(),
                    "download_time_seconds": 0,
                    "azure_video_url": row.get('azure_url') if not pd.isna(row.get('azure_url')) else None,
                    "azure_results_path": "batch_1/pyranee_prod_server_1/",
                    "source_s3_key": source_s3_key,
                    "source_video_id": source_video_id,
                    "clip_id": clip_id,
                    "start_ms": int(row['start_ms']) if not pd.isna(row.get('start_ms')) else None,
                    "end_ms": int(row['end_ms']) if not pd.isna(row.get('end_ms')) else None,
                    "duration_ms": int(row['duration_ms']) if not pd.isna(row.get('duration_ms')) else None,
                    "display_filename": display_filename
                }

                videos[s3_key] = video_entry

        except Exception as e:
            print(f"Error processing {excel_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nTotal videos extracted: {len(videos)}")
    return videos

def create_state_file(videos, output_path):
    """Create the state file with proper structure."""
    # Calculate statistics
    total_videos = len(videos)
    completed = sum(1 for v in videos.values() if v['status'] == 'completed')
    processing = sum(1 for v in videos.values() if v['status'] == 'processing')
    failed = sum(1 for v in videos.values() if v['status'] == 'failed')
    skipped = sum(1 for v in videos.values() if v['status'] == 'skipped')

    now = datetime.now().isoformat()

    state = {
        "bucket_name": "troveo-videodb-shared",
        "input_prefix": "outputs/batch_2_tier_1",
        "output_prefix": "results/",
        "last_poll_time": now,
        "poll_count": 0,
        "videos": videos,
        "statistics": {
            "total_discovered": total_videos,
            "completed": completed,
            "processing": processing,
            "failed": failed,
            "skipped": skipped
        },
        "created_at": now,
        "last_updated": now
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write state file
    with open(output_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"\nState file created: {output_path}")
    print(f"Total videos: {len(videos)}")
    print(f"\nStatistics:")
    print(f"  Total discovered: {total_videos}")
    print(f"  Completed: {completed}")
    print(f"  Processing: {processing}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")

def main():
    excel_dir = "/home/vision_ai_adm/code/oslo/video_convertbranch/csv_files/batch_2/server_1"
    output_path = "/data/pyrenees_output/batch2/server1/s3_state_prod_batch_2_server1.json"

    print("=" * 80)
    print("Creating Server 1 State File")
    print("=" * 80)
    print(f"Excel directory: {excel_dir}")
    print(f"Output path: {output_path}")
    print("=" * 80)
    print()

    # Read all Excel files
    videos = read_excel_files(excel_dir)

    if not videos:
        print("\nWARNING: No videos found in Excel files!")
        return

    # Create state file
    create_state_file(videos, output_path)

    print("\nDone!")

if __name__ == "__main__":
    main()
