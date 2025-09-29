# Metadata JSON Validator

This tool validates metadata JSON files for individual home IDs and identifies problematic session IDs and files that are being skipped due to missing or invalid metadata.

## Features

- **Individual Home ID Validation**: Validate metadata for a specific home ID
- **Session Analysis**: Identify session IDs with improper JSON
- **File Tracking**: List files being skipped due to metadata issues
- **Detailed Reporting**: Generate comprehensive JSON reports
- **Azure Integration**: Works directly with Azure Blob Storage

## Files

- `metadata_validator.py` - Main validator class and CLI
- `run_metadata_validation.py` - Example script to run validation
- `README_METADATA_VALIDATOR.md` - This documentation

## Usage

### Method 1: Using the CLI directly

```bash
# Using config file
python metadata_validator.py --home-id 63 --config config/data_handling_config.yaml

# Using direct parameters
python metadata_validator.py --home-id 63 \
    --connection-string "your_connection_string" \
    --container-name "instavideo" \
    --source-prefix "one-data-platform/"

# Save results to specific file
python metadata_validator.py --home-id 63 --config config.yaml --output results_63.json

# Enable verbose logging
python metadata_validator.py --home-id 63 --config config.yaml --verbose
```

### Method 2: Using the example script

```bash
# Edit run_metadata_validation.py to set your HOME_ID
python run_metadata_validation.py
```

### Method 3: Using as a Python module

```python
from metadata_validator import MetadataValidator

validator = MetadataValidator(
    home_id="63",
    connection_string="your_connection_string",
    container_name="instavideo",
    source_prefix="one-data-platform/"
)

results = validator.validate_home_id_metadata()
validator.print_summary()
validator.save_results_to_file("results.json")
```

## Configuration

The validator can be configured using:

1. **YAML Config File**: Use the same config file as your data handling pipeline
2. **Command Line Arguments**: Pass parameters directly
3. **Environment Variables**: Set `AZURE_CONNECTION_STRING`, `AZURE_CONTAINER_NAME`, etc.

## Output Format

The validator generates a comprehensive JSON report with the following structure:

```json
{
  "home_id": "63",
  "validation_timestamp": "2025-01-27T10:30:00Z",
  "total_media_files": 150,
  "files_with_metadata": 120,
  "files_without_metadata": 25,
  "invalid_metadata_files": 5,
  "session_analysis": {
    "20250925094000": {
      "session_id": "20250925094000",
      "issues": ["Missing metadata file", "Invalid metadata structure"],
      "affected_files": ["file1.insv", "file2.wav"]
    }
  },
  "problematic_sessions": [
    {
      "session_id": "20250925094000",
      "issue_count": 2,
      "affected_files_count": 2,
      "issues": ["Missing metadata file", "Invalid metadata structure"],
      "affected_files": ["file1.insv", "file2.wav"]
    }
  ],
  "skipped_files": [
    {
      "media_file": "one-data-platform/63-folder/file1.insv",
      "metadata_path": "one-data-platform/63-folder/63_activity_metadata.json",
      "reason": "Metadata file does not exist",
      "session_id": "20250925094000"
    }
  ],
  "validation_summary": {
    "total_sessions_analyzed": 10,
    "problematic_sessions_count": 2,
    "success_rate_percentage": 80.0,
    "most_common_issue": "Missing metadata file"
  }
}
```

## What the Validator Checks

### 1. Metadata File Existence
- Checks if expected metadata files exist for each media file
- Uses the naming pattern: `{id}_activity_{activity}_{timestamp}_metadata.json`

### 2. Metadata JSON Structure
- Validates required fields: `session_id`, `home_id`, `activity`, `timestamp`, `device_id`, `file_info`, `processing_info`
- Checks data types and structure integrity
- Validates nested objects like `file_info` and `processing_info`

### 3. Session Analysis
- Groups files by session ID
- Identifies sessions with multiple issues
- Tracks affected files per session

### 4. File Classification
- **Files with valid metadata**: Successfully processed
- **Files without metadata**: Missing metadata files
- **Files with invalid metadata**: Corrupted or malformed JSON

## Common Issues Identified

1. **Missing Metadata Files**: Media files without corresponding metadata
2. **Invalid JSON Structure**: Malformed or incomplete metadata JSON
3. **Missing Required Fields**: Metadata missing essential fields
4. **Data Type Mismatches**: Incorrect data types in metadata fields
5. **Session ID Inconsistencies**: Mismatched session IDs between files

## Integration with Data Handling Pipeline

This validator uses the same logic as the main data handling pipeline:

- **Pattern Matching**: Uses the same regex patterns to identify media files and metadata paths
- **Azure Integration**: Uses the same Azure Blob Storage client
- **Validation Logic**: Implements the same metadata validation checks

## Troubleshooting

### Common Errors

1. **Connection Issues**: Check your Azure connection string and container name
2. **Permission Issues**: Ensure your Azure credentials have read access to the container
3. **Network Issues**: Check your internet connection and Azure service status

### Debug Mode

Use the `--verbose` flag to enable detailed logging:

```bash
python metadata_validator.py --home-id 63 --config config.yaml --verbose
```

## Example Output

```
============================================================
METADATA VALIDATION SUMMARY FOR HOME ID: 63
============================================================
Total media files: 150
Files with valid metadata: 120
Files without metadata: 25
Files with invalid metadata: 5
Success rate: 80.0%
Total sessions analyzed: 10
Problematic sessions: 2
Most common issue: Missing metadata file

PROBLEMATIC SESSIONS:
  Session ID: 20250925094000
    Issues: 2
    Affected files: 2
    Issues: Missing metadata file, Invalid metadata structure

  Session ID: 20250925120000
    Issues: 1
    Affected files: 1
    Issues: Missing required field: session_id

============================================================
```

## Requirements

- Python 3.7+
- azure-storage-blob
- PyYAML
- dateutil

Install dependencies:
```bash
pip install azure-storage-blob pyyaml python-dateutil
```
