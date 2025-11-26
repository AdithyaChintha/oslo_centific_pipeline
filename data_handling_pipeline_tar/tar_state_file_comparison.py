#!/usr/bin/env python3
"""
Tar State File Comparison
Compares tar files (from tar_file_lister.py) with consolidated tar state files
to identify discrepancies and mapping between the two datasets.
"""

import os
import csv
import json
import argparse
from datetime import datetime
from collections import defaultdict

def load_tar_files(tar_csv_path):
    """Load tar files CSV and return as list of dictionaries"""
    tar_files = []

    try:
        with open(tar_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tar_files.append(row)

        print(f"Loaded {len(tar_files)} records from tar files CSV")
        return tar_files

    except Exception as e:
        print(f"Error loading tar files CSV: {e}")
        return []

def load_state_files(state_csv_path):
    """Load consolidated tar state files CSV and return as list of dictionaries"""
    state_files = []

    try:
        with open(state_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip summary rows
                if row.get('row_type') == 'SUMMARY':
                    continue
                state_files.append(row)

        print(f"Loaded {len(state_files)} records from state files CSV (excluding summary rows)")
        return state_files

    except Exception as e:
        print(f"Error loading state files CSV: {e}")
        return []

def extract_filename_from_blob_path(blob_path):
    """Extract just the filename from a full blob path"""
    if not blob_path:
        return ""
    return blob_path.split('/')[-1]

def normalize_filename(filename):
    """Normalize filename for comparison (remove any extra spaces, convert to lowercase)"""
    if not filename:
        return ""
    return filename.strip().lower()

def compare_files(tar_files, state_files, project_id):
    """Compare tar files with state files and identify matches/discrepancies"""

    # Create lookup dictionaries for faster comparison
    tar_lookup = {}
    state_lookup = {}

    # Build tar files lookup by tar_uuid
    for tar_file in tar_files:
        tar_uuid = tar_file.get('tar_uuid', '')
        if tar_uuid:
            tar_lookup[tar_uuid] = tar_file

    # Build state files lookup by tar_uuid
    for state_file in state_files:
        tar_uuid = state_file.get('tar_uuid', '')
        if tar_uuid:
            state_lookup[tar_uuid] = state_file

    # Perform comparison
    comparison_results = []

    # Check all tar files against state files
    for tar_file in tar_files:
        tar_uuid = tar_file.get('tar_uuid', '')

        # Skip non-tar files
        if tar_file.get('file_type', '').lower() not in ['tar', 'compressed_tar']:
            continue

        if not tar_uuid:
            continue

        state_file = state_lookup.get(tar_uuid)

        if state_file:
            # Found a match
            status = "FOUND"
            discrepancy = ""

            # Check for discrepancies
            discrepancies = []

            # Compare file sizes (convert to comparable units)
            try:
                tar_size_gb = float(tar_file.get('file_size_gb', 0))
                state_size_gb = float(state_file.get('file_size_gb', 0))

                # Allow for small differences due to rounding
                size_diff = abs(tar_size_gb - state_size_gb)
                if size_diff > 0.001:  # More than 1MB difference
                    discrepancies.append(f"Size mismatch: tar={tar_size_gb:.3f}GB, state={state_size_gb:.3f}GB")
            except (ValueError, TypeError):
                discrepancies.append("Size comparison failed - invalid size values")

            # Check upload status
            upload_status = state_file.get('status', '').lower()
            if upload_status not in ['success', 'successful']:
                discrepancies.append(f"Upload status: {upload_status}")

            if discrepancies:
                discrepancy = "; ".join(discrepancies)
            else:
                discrepancy = "None"

        else:
            # No match found
            status = "NOT_FOUND"
            discrepancy = "File exists in blob but not in upload_state"
            state_file = {}  # Empty dict for consistent structure

        comparison_results.append({
            'project_id': project_id,
            'tar_uuid': tar_uuid,
            'tar_filename': tar_file.get('filename', ''),
            'tar_file_size_gb': tar_file.get('file_size_gb', ''),
            'tar_last_modified': tar_file.get('last_modified', ''),
            'tar_full_path': tar_file.get('full_blob_path', ''),
            'tar_blob_url': tar_file.get('blob_url', ''),
            'state_blob_name': state_file.get('blob_name', ''),
            'state_file_size_gb': state_file.get('file_size_gb', ''),
            'state_status': state_file.get('status', ''),
            'state_container_id': state_file.get('container_id', ''),
            'state_upload_duration_seconds': state_file.get('upload_duration_seconds', ''),
            'state_upload_start_time': state_file.get('upload_start_time', ''),
            'state_upload_end_time': state_file.get('upload_end_time', ''),
            'state_metadata_uploaded': state_file.get('metadata_uploaded', ''),
            'state_error_details': state_file.get('error_details', ''),
            'mapping_status': status,
            'discrepancy': discrepancy
        })

    # Check for files in state that are not in tar files
    for state_file in state_files:
        tar_uuid = state_file.get('tar_uuid', '')

        if not tar_uuid:
            continue

        if tar_uuid not in tar_lookup:
            comparison_results.append({
                'project_id': project_id,
                'tar_uuid': tar_uuid,
                'tar_filename': '',  # Empty because file doesn't exist in tar files
                'tar_file_size_gb': '',
                'tar_last_modified': '',
                'tar_full_path': '',
                'tar_blob_url': '',
                'state_blob_name': state_file.get('blob_name', ''),
                'state_file_size_gb': state_file.get('file_size_gb', ''),
                'state_status': state_file.get('status', ''),
                'state_container_id': state_file.get('container_id', ''),
                'state_upload_duration_seconds': state_file.get('upload_duration_seconds', ''),
                'state_upload_start_time': state_file.get('upload_start_time', ''),
                'state_upload_end_time': state_file.get('upload_end_time', ''),
                'state_metadata_uploaded': state_file.get('metadata_uploaded', ''),
                'state_error_details': state_file.get('error_details', ''),
                'mapping_status': 'ORPHANED_IN_STATE',
                'discrepancy': f'File exists in upload_state but not in blob storage'
            })

    # Sort results to put all discrepancies at the top
    def sort_key(result):
        mapping_status = result['mapping_status']
        has_discrepancy = result['discrepancy'] and result['discrepancy'] != 'None'

        # Priority order: discrepancies first, then by status
        if has_discrepancy:
            return (0, mapping_status)  # Discrepancies first
        elif mapping_status == 'NOT_FOUND':
            return (1, mapping_status)  # Not found second
        elif mapping_status == 'ORPHANED_IN_STATE':
            return (2, mapping_status)  # Orphaned third
        else:
            return (3, mapping_status)  # Found last

    comparison_results.sort(key=sort_key)

    return comparison_results

def save_comparison_results(comparison_results, output_dir, project_id):
    """Save comparison results to CSV and JSON files"""

    if not comparison_results:
        print("No comparison results to save!")
        return None, None

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV filename
    csv_filename = f"{timestamp}_{project_id}_tar_state_comparison.csv"
    csv_path = os.path.join(output_dir, csv_filename)

    # JSON filename
    json_filename = f"{timestamp}_{project_id}_tar_state_comparison.json"
    json_path = os.path.join(output_dir, json_filename)

    # Define CSV columns
    fieldnames = [
        'project_id',
        'tar_uuid',
        'tar_filename',
        'tar_file_size_gb',
        'tar_last_modified',
        'tar_full_path',
        'tar_blob_url',
        'state_blob_name',
        'state_file_size_gb',
        'state_status',
        'state_container_id',
        'state_upload_duration_seconds',
        'state_upload_start_time',
        'state_upload_end_time',
        'state_metadata_uploaded',
        'state_error_details',
        'mapping_status',
        'discrepancy'
    ]

    # Save CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_results)

    # Save JSON
    json_data = {
        'generated_at': datetime.now().isoformat(),
        'project_id': project_id,
        'total_comparisons': len(comparison_results),
        'summary': {
            'found': sum(1 for r in comparison_results if r['mapping_status'] == 'FOUND'),
            'not_found': sum(1 for r in comparison_results if r['mapping_status'] == 'NOT_FOUND'),
            'orphaned_in_state': sum(1 for r in comparison_results if r['mapping_status'] == 'ORPHANED_IN_STATE'),
            'with_discrepancies': sum(1 for r in comparison_results if r['discrepancy'] and r['discrepancy'] != 'None'),
            'successful_uploads': sum(1 for r in comparison_results if r['state_status'] and r['state_status'].lower() in ['success', 'successful']),
            'failed_uploads': sum(1 for r in comparison_results if r['state_status'] and r['state_status'].lower() in ['failed', 'error'])
        },
        'comparisons': comparison_results
    }

    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_data, jsonfile, indent=2)

    return csv_path, json_path

def print_summary(comparison_results, project_id):
    """Print summary statistics"""

    if not comparison_results:
        print("No comparison results to summarize!")
        return

    total = len(comparison_results)
    found = sum(1 for r in comparison_results if r['mapping_status'] == 'FOUND')
    not_found = sum(1 for r in comparison_results if r['mapping_status'] == 'NOT_FOUND')
    orphaned = sum(1 for r in comparison_results if r['mapping_status'] == 'ORPHANED_IN_STATE')
    with_discrepancies = sum(1 for r in comparison_results if r['discrepancy'] and r['discrepancy'] != 'None')

    # Upload status counts
    successful = sum(1 for r in comparison_results if r['state_status'] and r['state_status'].lower() in ['success', 'successful'])
    failed = sum(1 for r in comparison_results if r['state_status'] and r['state_status'].lower() in ['failed', 'error'])

    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Project ID: {project_id}")
    print(f"Total tar files compared: {total}")
    print(f"Files found (matched): {found}")
    print(f"Files not found in upload_state: {not_found}")
    print(f"Files orphaned in upload_state: {orphaned}")
    print(f"Files with discrepancies: {with_discrepancies}")
    print(f"\nUpload Status:")
    print(f"  Successful uploads: {successful}")
    print(f"  Failed uploads: {failed}")

    # Calculate total sizes
    tar_total_size = sum(float(r['tar_file_size_gb'] or 0) for r in comparison_results if r['tar_file_size_gb'])
    state_total_size = sum(float(r['state_file_size_gb'] or 0) for r in comparison_results if r['state_file_size_gb'])

    print(f"\nTotal Data:")
    print(f"  Tar files total size: {tar_total_size:.2f} GB")
    print(f"  State files total size: {state_total_size:.2f} GB")

def main():
    """Main function"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compare tar files with consolidated tar state files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using explicit file paths
  python tar_state_file_comparison.py --tar tar_files_oslo_project_2_20250103.csv --state consolidated_tar_files_oslo_project_2_20250103.csv

  # Using short options
  python tar_state_file_comparison.py -t tar_files.csv -s state_files.csv

  # Specify output directory and project ID
  python tar_state_file_comparison.py -t tar_files.csv -s state.csv -o ./results -p oslo_project_2
        """
    )

    parser.add_argument(
        '--tar', '-t',
        required=True,
        help='Path to tar files CSV (from tar_file_lister.py)'
    )

    parser.add_argument(
        '--state', '-s',
        required=True,
        help='Path to consolidated tar state files CSV (from report_tar_state.py)'
    )

    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory for comparison results (default: same directory as script)'
    )

    parser.add_argument(
        '--project-id', '-p',
        default='tar-upload',
        help='Project ID for the comparison (default: tar-upload)'
    )

    args = parser.parse_args()

    # Get absolute paths
    tar_csv_path = os.path.abspath(args.tar)
    state_csv_path = os.path.abspath(args.state)
    project_id = args.project_id

    # Determine output directory
    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== Tar State File Comparison ===")
    print(f"Project ID: {project_id}")
    print(f"Tar files CSV: {tar_csv_path}")
    print(f"State files CSV: {state_csv_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Check if files exist
    if not os.path.exists(tar_csv_path):
        print(f"Error: Tar files CSV not found: {tar_csv_path}")
        return 1

    if not os.path.exists(state_csv_path):
        print(f"Error: State files CSV not found: {state_csv_path}")
        return 1

    # Load data
    tar_files = load_tar_files(tar_csv_path)
    state_files = load_state_files(state_csv_path)

    if not tar_files:
        print("Error: No tar files loaded!")
        return 1

    if not state_files:
        print("Error: No state files loaded!")
        return 1

    # Perform comparison
    print("Performing file comparison...")
    comparison_results = compare_files(tar_files, state_files, project_id)

    # Save results
    csv_path, json_path = save_comparison_results(comparison_results, output_dir, project_id)

    # Print summary
    print_summary(comparison_results, project_id)

    if csv_path and json_path:
        print(f"\nComparison results saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")

    return 0

if __name__ == "__main__":
    exit(main())
