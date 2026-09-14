#!/usr/bin/env python3
"""
Convert Label Studio CSV/Excel to Movement Metadata JSONL format.

Input: CSV/Excel file or folder with CSV/Excel files (with columns including movement_metadata JSON string)
Output: Single JSONL file with movement types and confidences from all files
"""

import csv
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import glob
from datetime import datetime
import pandas as pd


def extract_numeric_video_id(source_video_id: str) -> str:
    """
    Extract numeric video ID from source_video_id.

    Examples:
        "00NfzFc8_6066938804" -> "6066938804"
        "03qJIR4a_542721916" -> "542721916"
        "6066938804" -> "6066938804" (already numeric)

    Args:
        source_video_id: Full source video ID

    Returns:
        Numeric portion of the video ID
    """
    if not source_video_id:
        return ""

    # If it contains underscore, take the part after underscore
    if '_' in source_video_id:
        parts = source_video_id.split('_')
        # Return the last part which should be numeric
        return parts[-1]

    # If no underscore, return as is
    return source_video_id


def parse_movement_metadata(movement_str: str) -> List[Dict[str, Any]]:
    """
    Parse movement metadata JSON string and extract movement types with confidences.

    Args:
        movement_str: JSON string containing movement metadata

    Returns:
        List of movement dictionaries sorted by confidence (descending)
    """
    if not movement_str or movement_str == "":
        return []

    try:
        # Parse JSON string
        movement_data = json.loads(movement_str)

        # Extract movements if present
        if isinstance(movement_data, dict) and 'movements' in movement_data:
            movements = movement_data['movements']
            # Sort by confidence descending
            movements_sorted = sorted(movements, key=lambda x: x.get('confidence', 0), reverse=True)
            return movements_sorted[:5]  # Top 5 movements

        return []
    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Warning: Failed to parse movement metadata: {e}", file=sys.stderr)
        return []


def process_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Process a single CSV or Excel file and return records.

    Args:
        file_path: Path to CSV or Excel file

    Returns:
        List of record dictionaries
    """
    records = []

    # Read file based on extension
    file_ext = file_path.suffix.lower()

    if file_ext in ['.xlsx', '.xls']:
        # Read Excel file
        df = pd.read_excel(file_path)
        rows = df.to_dict('records')
    elif file_ext == '.csv':
        # Read CSV file
        df = pd.read_csv(file_path)
        rows = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

    for row in rows:
        # Extract base fields - handle both string keys and NaN values
        full_source_video_id = str(row.get('source_video_id', '')) if pd.notna(row.get('source_video_id')) else ''
        numeric_video_id = extract_numeric_video_id(full_source_video_id)

        record = {
            "source_s3_key": str(row.get('source_s3_key', '')) if pd.notna(row.get('source_s3_key')) else '',
            "source_video_id": numeric_video_id,  # Use numeric video ID
            "s3_key": str(row.get('s3_key', '')) if pd.notna(row.get('s3_key')) else '',
            "clip_id": str(row.get('clip_id', '')) if pd.notna(row.get('clip_id')) else '',
            "start_ms": int(row.get('start_ms', 0)) if pd.notna(row.get('start_ms')) else None,
            "end_ms": int(row.get('end_ms', 0)) if pd.notna(row.get('end_ms')) else None,
            "duration_ms": int(row.get('duration_ms', 0)) if pd.notna(row.get('duration_ms')) else None,
        }

        # Check if file has Movement1-5 columns (new format) or movement_metadata (old format)
        has_movement_columns = ' Movement1' in row or 'Movement1' in row
        has_any_movement = False

        if has_movement_columns:
            # Read from Movement1-5 columns directly
            for i in range(1, 6):
                # Try both with and without space prefix
                movement_col = row.get(f' Movement{i}') or row.get(f'Movement{i}')

                # Handle NaN and convert to string
                if pd.notna(movement_col) and movement_col:
                    movement_col = str(movement_col).strip()
                    record[f"movement_type_{i}"] = movement_col
                    record[f"confidence_{i}"] = 1.0  # Default confidence when not provided
                    has_any_movement = True
                else:
                    record[f"movement_type_{i}"] = None
                    record[f"confidence_{i}"] = None
        else:
            # Parse movement metadata JSON (old format)
            movement_metadata_str = row.get('movement_metadata', '')
            if pd.notna(movement_metadata_str):
                movements = parse_movement_metadata(str(movement_metadata_str))
            else:
                movements = []

            # Add up to 5 movement types and confidences
            for i in range(1, 6):
                if i <= len(movements):
                    record[f"movement_type_{i}"] = movements[i-1].get('type')
                    record[f"confidence_{i}"] = movements[i-1].get('confidence')
                    has_any_movement = True
                else:
                    record[f"movement_type_{i}"] = None
                    record[f"confidence_{i}"] = None

        # Add has_movement flag
        record["has_movement"] = "Yes" if has_any_movement else "No"

        records.append(record)

    return records


def convert_csv_to_json(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert Label Studio CSV/Excel file(s) to movement metadata JSONL format.

    Args:
        input_path: Path to input CSV/Excel file or folder containing files
        output_path: Path to output JSONL file (optional)

    Returns:
        Path to output JSONL file
    """
    input_path = Path(input_path)
    all_results = []

    # Check if input is a directory or file
    if input_path.is_dir():
        # Find all CSV and Excel files in directory
        csv_files = sorted(input_path.glob('*.csv'))
        excel_files = sorted(list(input_path.glob('*.xlsx')) + list(input_path.glob('*.xls')))
        all_files = csv_files + excel_files

        if not all_files:
            print(f"⚠️  No CSV or Excel files found in {input_path}", file=sys.stderr)
            return None

        print(f"📂 Found {len(all_files)} files in {input_path}")

        # Process each file
        for file_path in all_files:
            print(f"   Processing: {file_path.name}...", end=' ')
            records = process_file(file_path)
            all_results.extend(records)
            print(f"✅ {len(records)} records")

        # Default output path for directory - use Troveo naming convention
        if output_path is None:
            today = datetime.now().strftime('%m%d%Y')
            num_tasks = len(all_results)
            output_filename = f"Troveo_Delivery1_{today}_{num_tasks}"
            output_path = input_path / f"{output_filename}.jsonl"
        else:
            output_path = Path(output_path)

    elif input_path.is_file():
        # Single file mode
        print(f"📄 Processing single file: {input_path}")
        all_results = process_file(input_path)

        # Default output path for single file - use Troveo naming convention
        if output_path is None:
            today = datetime.now().strftime('%m%d%Y')
            num_tasks = len(all_results)
            output_filename = f"Troveo_Delivery1_{today}_{num_tasks}"
            output_path = input_path.parent / f"{output_filename}.jsonl"
        else:
            output_path = Path(output_path)

    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Write JSONL output (one JSON object per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in all_results:
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + '\n')

    print(f"\n✅ Successfully converted {len(all_results)} total records to JSONL")
    print(f"📄 Output: {output_path}")

    return str(output_path)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python pyrenees_post_consolidation_from_xlsx.py <csv_excel_file_or_folder> [output_jsonl]")
        print("\nExamples:")
        print("  # Single CSV file:")
        print("  python pyrenees_post_consolidation_from_xlsx.py labelstudio_tasks_cycle_1.csv")
        print("  # Output: Troveo_Delivery1_10032025_25.jsonl (today's date + 25 tasks)")
        print("\n  # Single Excel file:")
        print("  python pyrenees_post_consolidation_from_xlsx.py data.xlsx")
        print("  # Output: Troveo_Delivery1_10032025_100.jsonl")
        print("\n  # Folder with multiple CSV/Excel files:")
        print("  python pyrenees_post_consolidation_from_xlsx.py ./csv_files/labelstudio_csv_prod/")
        print("  # Output: Troveo_Delivery1_10032025_500.jsonl")
        print("\n  # Custom output name:")
        print("  python pyrenees_post_consolidation_from_xlsx.py ./csv_files/labelstudio_csv_prod/ custom_output.jsonl")
        print("\nNote: source_video_id is automatically converted to numeric format")
        print("  e.g., '00NfzFc8_6066938804' -> '6066938804'")
        sys.exit(1)

    input_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(input_path).exists():
        print(f"❌ Error: Path not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        convert_csv_to_json(input_path, output_file)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
