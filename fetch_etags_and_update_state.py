#!/usr/bin/env python3
"""
Fetch ETags from S3 for all videos in the state file and update the state file.
"""
import json
import boto3
from datetime import datetime
from tqdm import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def load_state_file(state_file_path):
    """Load the state JSON file."""
    print(f"Loading state file: {state_file_path}")
    with open(state_file_path, 'r') as f:
        state = json.load(f)
    print(f"Loaded {len(state['videos'])} videos from state file")
    return state

def load_s3_config(config_path):
    """Load S3 configuration."""
    print(f"Loading S3 config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['s3']

def fetch_single_etag(s3_config, bucket_name, s3_key):
    """Fetch ETag for a single S3 object."""
    try:
        # Create S3 client for this request
        session = boto3.Session(
            aws_access_key_id=s3_config['aws_access_key_id'],
            aws_secret_access_key=s3_config['aws_secret_access_key'],
            region_name=s3_config.get('region_name', 'us-east-1')
        )
        s3_client = session.client('s3')

        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        etag = response['ETag'].strip('"')
        size_bytes = response['ContentLength']
        last_modified = response['LastModified'].isoformat() if 'LastModified' in response else None

        return {
            'status': 'success',
            's3_key': s3_key,
            'etag': etag,
            'size_bytes': size_bytes,
            'last_modified': last_modified
        }
    except Exception as e:
        # Check if it's a NoSuchKey error
        if 'NoSuchKey' in str(e) or '404' in str(e):
            return {
                'status': 'not_found',
                's3_key': s3_key
            }
        return {
            'status': 'error',
            's3_key': s3_key,
            'error': str(e)
        }

def fetch_etags_from_s3(state, s3_config, max_workers=50):
    """Fetch ETags from S3 for all videos in the state file using parallel threads."""
    import sys

    bucket_name = s3_config['bucket_name']

    print(f"\nFetching ETags from S3 bucket: {bucket_name}", flush=True)
    print(f"Using {max_workers} parallel workers", flush=True)
    print("=" * 80, flush=True)

    updated_count = 0
    not_found_count = 0
    error_count = 0
    processed_count = 0

    # Thread-safe counters
    lock = threading.Lock()

    # Get all S3 keys to process
    s3_keys = list(state['videos'].keys())
    total = len(s3_keys)

    print(f"Total videos to process: {total}", flush=True)
    print("Starting parallel fetch...\n", flush=True)

    # Process with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        print(f"Submitting {total} tasks to thread pool...", flush=True)
        future_to_key = {
            executor.submit(fetch_single_etag, s3_config, bucket_name, s3_key): s3_key
            for s3_key in s3_keys
        }
        print(f"All tasks submitted. Processing...\n", flush=True)

        # Process results - print progress every 100 items
        for future in as_completed(future_to_key):
            result = future.result()

            with lock:
                processed_count += 1

                if result['status'] == 'success':
                    # Update state
                    video_data = state['videos'][result['s3_key']]
                    video_data['s3_etag'] = result['etag']
                    video_data['size_bytes'] = result['size_bytes']
                    video_data['s3_last_modified'] = result['last_modified']
                    updated_count += 1

                elif result['status'] == 'not_found':
                    not_found_count += 1
                    print(f"⚠️  Not found: {result['s3_key']}", flush=True)

                elif result['status'] == 'error':
                    error_count += 1
                    # Only log non-403 errors
                    if '403' not in result['error']:
                        print(f"❌ Error: {result['s3_key']}: {result['error']}", flush=True)

                # Print progress every 100 items
                if processed_count % 100 == 0:
                    percent = (processed_count / total) * 100
                    print(f"Progress: {processed_count}/{total} ({percent:.1f}%) | "
                          f"✅ OK: {updated_count} | ⚠️  NotFound: {not_found_count} | "
                          f"❌ Errors: {error_count}", flush=True)

        # Final update
        percent = (processed_count / total) * 100
        print(f"\nFinal: {processed_count}/{total} ({percent:.1f}%) | "
              f"✅ OK: {updated_count} | ⚠️  NotFound: {not_found_count} | "
              f"❌ Errors: {error_count}", flush=True)

    print("\n" + "=" * 80)
    print(f"✅ Updated: {updated_count}")
    print(f"⚠️  Not found: {not_found_count}")
    print(f"❌ Errors: {error_count}")
    print("=" * 80)

    return updated_count, not_found_count, error_count

def save_state_file(state, state_file_path):
    """Save the updated state file."""
    print(f"\nSaving updated state file: {state_file_path}")

    # Update timestamp
    state['last_updated'] = datetime.now().isoformat()

    # Create backup of original
    backup_path = f"{state_file_path}.backup.{int(datetime.now().timestamp())}"
    print(f"Creating backup: {backup_path}")

    import shutil
    shutil.copy2(state_file_path, backup_path)

    # Write updated state
    with open(state_file_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"✅ State file updated successfully!")

def main():
    state_file_path = "/data/pyrenees_output/batch2/server1/s3_state_prod_batch_2_server1.json"
    s3_config_path = "/home/vision_ai_adm/code/oslo/video_convertbranch/config/s3_config.yaml"

    print("=" * 80)
    print("FETCH ETAGS FROM S3 AND UPDATE STATE FILE")
    print("=" * 80)
    print(f"State file: {state_file_path}")
    print(f"S3 config: {s3_config_path}")
    print("=" * 80)
    print()

    # Load state file
    state = load_state_file(state_file_path)

    # Load S3 config
    s3_config = load_s3_config(s3_config_path)

    # Fetch ETags from S3
    updated, not_found, errors = fetch_etags_from_s3(state, s3_config)

    if updated == 0:
        print("\n⚠️  No videos were updated. Aborting save.")
        return

    # Save updated state
    save_state_file(state, state_file_path)

    print("\n✅ Done!")
    print(f"Total videos in state: {len(state['videos'])}")
    print(f"Videos updated with ETags: {updated}")

if __name__ == "__main__":
    main()
