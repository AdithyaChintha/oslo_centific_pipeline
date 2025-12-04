#!/usr/bin/env python3
"""
Filter Dual-Fisheye Videos from TAR Inspection CSV Report

This script reads a TAR inspection CSV report and creates a filtered report
containing only dual-fisheye videos.

Usage:
    python filter_dual_fisheye_videos.py <input_csv> [output_csv]

Examples:
    # Filter and create new CSV with dual-fisheye videos only
    python filter_dual_fisheye_videos.py tar_inspection_report_20250119_123456.csv

    # Specify custom output filename
    python filter_dual_fisheye_videos.py input.csv dual_fisheye_only.csv

    # Print summary only (no output file)
    python filter_dual_fisheye_videos.py input.csv --summary-only
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict


def filter_dual_fisheye_videos(input_csv: str, output_csv: str = None, summary_only: bool = False) -> List[Dict]:
    """
    Filter dual-fisheye videos from CSV report.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (optional)
        summary_only: If True, only print summary without creating output file

    Returns:
        List of dual-fisheye video entries
    """
    input_path = Path(input_csv)

    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_csv}")
        sys.exit(1)

    # Read CSV
    dual_fisheye_rows = []
    total_rows = 0
    fieldnames = []

    try:
        with open(input_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = reader.fieldnames

            for row in reader:
                total_rows += 1
                if row.get('is_dual_fisheye', '').upper() == 'YES':
                    dual_fisheye_rows.append(row)

    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        sys.exit(1)

    # Print summary
    print("\n" + "="*80)
    print("DUAL-FISHEYE VIDEO FILTER RESULTS")
    print("="*80)
    print(f"📄 Input file: {input_csv}")
    print(f"📊 Total entries: {total_rows}")
    print(f"🎥 Dual-fisheye videos: {len(dual_fisheye_rows)}")
    print(f"📈 Percentage: {(len(dual_fisheye_rows) / total_rows * 100):.1f}%" if total_rows > 0 else "0%")
    print("="*80)

    if len(dual_fisheye_rows) == 0:
        print("\n✅ No dual-fisheye videos found in the report.")
        return []

    # List dual-fisheye videos
    print("\n🎥 Dual-Fisheye Videos Found:")
    print("-"*80)
    for i, row in enumerate(dual_fisheye_rows, 1):
        tar_name = row.get('tar_file_name', 'Unknown')
        file_name = row.get('individual_file_name', 'Unknown')
        file_type = row.get('individual_file_type', '')
        is_erp = row.get('is_erp_version', 'NO')
        erp_label = " [ERP]" if is_erp.upper() == 'YES' else ""
        print(f"{i:3d}. {file_name:<40} (TAR: {tar_name}){erp_label}")

    # Create output CSV if requested
    if not summary_only:
        if output_csv is None:
            # Auto-generate output filename
            output_csv = input_path.parent / f"{input_path.stem}_dual_fisheye_only.csv"
        else:
            output_csv = Path(output_csv)

        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dual_fisheye_rows)

            print(f"\n✅ Filtered CSV created: {output_csv}")
            print(f"   Contains {len(dual_fisheye_rows)} dual-fisheye video entries")

        except Exception as e:
            print(f"\n❌ Error writing output CSV: {e}")
            sys.exit(1)

    print("="*80 + "\n")
    return dual_fisheye_rows


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = None
    summary_only = False

    if len(sys.argv) >= 3:
        if sys.argv[2] == '--summary-only':
            summary_only = True
        else:
            output_csv = sys.argv[2]

    filter_dual_fisheye_videos(input_csv, output_csv, summary_only)


if __name__ == "__main__":
    main()
