# TAR Random Inspection Workflow Documentation

## Overview

This workflow randomly samples TAR files from SharePoint (1 per 10 files), extracts them, uploads contents to Azure Blob Storage, generates SAS tokens, and creates a detailed CSV report for team inspection.

---

## Files Created

### 1. `sharepoint_tar_inspector.py`
Main Python script that handles the entire inspection workflow

### 2. `config/tar_inspection_config.yaml`
Configuration file (reuses credentials from `tar_transfer_config.yaml`)

---

## How It Works

### Workflow Steps:

```
SharePoint /Uploads folder
        ↓
1. List all TAR files (e.g., 100 files found)
        ↓
2. Filter already-inspected files (state tracking)
        ↓
3. Random sampling: Pick 1 per 10 files (e.g., 10 random files selected)
        ↓
4. For each selected TAR file:
   a. Download TAR from SharePoint
   b. Extract/untar locally to /data/oslo/tar_extractions/
   c. Get list of all extracted files with metadata
   d. Upload each file to Azure Blob:
      → Container: instavideo
      → Path: untar_folder_oslo2/<tar_name>/<file_path>
   e. Generate SAS token (30-day expiry, read-only)
   f. Construct blob URL with SAS token
   g. Add record to CSV report
        ↓
5. Create CSV report with columns:
   - tar_file_name
   - individual_file_name
   - individual_file_full_path
   - individual_file_type
   - tar_size_gb
   - individual_file_size_gb
   - sas_token
   - blob_url
   - blob_path
   - transfer_date
        ↓
6. Save report to: /data/oslo/inspection_reports/tar_inspection_report_<timestamp>.csv
        ↓
7. Update state file (marks TAR as inspected)
        ↓
8. Cleanup: Delete downloaded TAR and extracted files
```

---

## Key Features

### ✅ Random Sampling
- Picks **1 random TAR file per every 10 files** (configurable)
- Avoids re-inspection using state tracking

### ✅ State Tracking
- **File:** `/data/oslo/tar_temp_downloads/tar_inspection_state.json`
- **Purpose:** Prevents re-inspecting already-processed TAR files
- **Format:**
```json
{
  "file1.tar": {
    "inspected_at": "2025-11-07T10:30:00",
    "num_files_extracted": 1245,
    "tar_size": 5368709120
  }
}
```

### ✅ Blob Upload Structure
```
Container: instavideo
    └── untar_folder_oslo2/
        ├── archive1/           (extracted from archive1.tar)
        │   ├── video1.mp4
        │   ├── video2.mp4
        │   └── metadata.json
        └── archive2/           (extracted from archive2.tar)
            ├── video3.mp4
            └── data.xml
```

### ✅ SAS Token Generation
- **Expiry:** 30 days (configurable via `sas_token_expiry_days`)
- **Permissions:** Read-only
- **Purpose:** Team can inspect files via URL without Azure credentials

### ✅ CSV Report Format
| tar_file_name | individual_file_name | individual_file_full_path | individual_file_type | tar_size_gb | individual_file_size_gb | sas_token | blob_url | blob_path | transfer_date |
|---------------|---------------------|---------------------------|---------------------|-------------|------------------------|-----------|----------|-----------|---------------|
| archive1.tar | video1.mp4 | videos/batch1/video1.mp4 | .mp4 | 5.0000 | 1.234500 | sv=2022-11-02&... | https://oslotestvideo... | untar_folder_oslo2/archive1/videos/batch1/video1.mp4 | 2025-11-07T10:30:00 |

---

## Configuration

### Important Settings in `tar_inspection_config.yaml`:

```yaml
inspection:
  # Where extracted files go in blob storage
  untar_blob_prefix: "untar_folder_oslo2/"

  # Sampling: 1 random file per N files
  sampling_ratio: 10

  # SAS token validity period
  sas_token_expiry_days: 30

  # Local directories
  temp_extract_dir: "/data/oslo/tar_extractions"
  csv_report_dir: "/data/oslo/inspection_reports"
  inspection_state_file: "/data/oslo/tar_temp_downloads/tar_inspection_state.json"
```

---

## Usage

### 1. Basic Run (Default Config)
```bash
cd /home/vision_ai_adm/code/oslo/combined_branch
python sharepoint_tar_inspector.py
```

### 2. Custom Config
```bash
python sharepoint_tar_inspector.py --config /path/to/custom_config.yaml
```

### 3. Dry Run (Preview)
```bash
python sharepoint_tar_inspector.py --dry-run
```
**Output:**
```
Selected 10 TAR files for random inspection:
  1. archive_001.tar (4.52 GB)
  2. archive_025.tar (3.81 GB)
  ...
DRY RUN MODE - No actual processing
```

---

## Example Output

### Console Output:
```
============================================================
Starting TAR Random Inspection Workflow
============================================================
Fetching TAR files from SharePoint: /Uploads
Found 150 TAR files in SharePoint
Uninspected files: 145
Selected 14 random samples for inspection

Selected 14 TAR files for random inspection:
  1. archive_003.tar (4.52 GB)
  2. archive_017.tar (3.81 GB)
  ...

####################################################################
TAR FILE 1/14: archive_003.tar
####################################################################
Downloading from SharePoint: archive_003.tar
Extracting TAR file: archive_003.tar
Extracted 1847 files to: /data/oslo/tar_extractions/archive_003
Processing 1847 extracted files for upload...
  [1/1847] Uploading: video_001.mp4 (1.2345 GB)
  [2/1847] Uploading: video_002.mp4 (0.9876 GB)
  ...
✅ Successfully processed 1847 files from TAR: archive_003.tar

============================================================
📊 CSV Report Generated: /data/oslo/inspection_reports/tar_inspection_report_20251107_103045.csv
============================================================

INSPECTION SUMMARY
============================================================
✅ Successful inspections: 14
❌ Failed inspections: 0
📁 Total files processed: 23,458
============================================================
```

---

## CSV Report Example

The generated CSV will look like this:

```csv
tar_file_name,individual_file_name,individual_file_full_path,individual_file_type,tar_size_gb,individual_file_size_gb,sas_token,blob_url,blob_path,transfer_date
archive_003.tar,video_001.mp4,videos/batch1/video_001.mp4,.mp4,4.5200,1.2345,sv=2022-11-02&ss=b&srt=o&sp=r&...,https://oslotestvideo.blob.core.windows.net/instavideo/untar_folder_oslo2/archive_003/videos/batch1/video_001.mp4?sv=...,untar_folder_oslo2/archive_003/videos/batch1/video_001.mp4,2025-11-07T10:30:45
archive_003.tar,video_002.mp4,videos/batch1/video_002.mp4,.mp4,4.5200,0.9876,sv=2022-11-02&ss=b&srt=o&sp=r&...,https://oslotestvideo.blob.core.windows.net/instavideo/untar_folder_oslo2/archive_003/videos/batch1/video_002.mp4?sv=...,untar_folder_oslo2/archive_003/videos/batch1/video_002.mp4,2025-11-07T10:30:47
```

**Team members can:**
1. Open CSV in Excel/Google Sheets
2. Copy `blob_url` (includes SAS token)
3. Paste in browser to inspect file directly
4. No Azure credentials needed!

---

## Differences from `sharepoint_tar_to_azure_blob.py`

| Feature | sharepoint_tar_to_azure_blob.py | sharepoint_tar_inspector.py |
|---------|--------------------------------|----------------------------|
| **Purpose** | Transfer ALL TAR files as-is | Randomly sample, extract, inspect |
| **Sampling** | All files | 1 per 10 files (random) |
| **Extraction** | No (keeps TAR format) | Yes (extracts contents) |
| **Blob Path** | `oslo_stage_2_tar_files/` | `untar_folder_oslo2/` |
| **SAS Tokens** | No | Yes (for inspection) |
| **CSV Report** | No | Yes (detailed report) |
| **Transfer Method** | Server-to-server, streaming, download+upload | Download → Extract → Upload |
| **Use Case** | Bulk TAR archival | Quality inspection & validation |

---

## State Tracking Benefits

### Without State Tracking:
- Re-runs would re-inspect the same TAR files
- Wastes time and resources
- Duplicate entries in reports

### With State Tracking:
- ✅ Skips already-inspected TAR files
- ✅ Only processes new files
- ✅ Safe to run multiple times
- ✅ Can resume after interruption

---

## Customization Options

### Change Sampling Ratio
```yaml
inspection:
  sampling_ratio: 20  # Pick 1 per 20 files (less sampling)
  sampling_ratio: 5   # Pick 1 per 5 files (more sampling)
```

### Change SAS Token Validity
```yaml
inspection:
  sas_token_expiry_days: 7   # 1 week
  sas_token_expiry_days: 90  # 3 months
```

### Change Blob Upload Location
```yaml
inspection:
  untar_blob_prefix: "inspections/batch_001/"  # Different folder
```

---

## Logging

### Log File
- **Location:** `sharepoint_tar_inspector.log` (in current directory)
- **Format:** Timestamp, level, message
- **Levels:** INFO, WARNING, ERROR, DEBUG

### Console Output
- Real-time progress updates
- File-by-file processing status
- Summary statistics

---

## Troubleshooting

### Issue: "No TAR files found in SharePoint folder"
**Solution:** Check `tar_source_folder_path` in config matches SharePoint folder

### Issue: "Failed to extract TAR file"
**Solution:** Check TAR file is not corrupted, ensure sufficient disk space

### Issue: "Failed to generate SAS token"
**Solution:** Verify Azure connection string and account key are correct

### Issue: "No uninspected files available"
**Solution:** All files already inspected. To re-inspect, delete or modify `tar_inspection_state.json`

---

## Clean Up Old State (Force Re-inspection)

If you want to re-inspect all TAR files:

```bash
# Remove state file
rm /data/oslo/tar_temp_downloads/tar_inspection_state.json

# Or remove specific entries
# Edit the JSON file and delete specific tar file entries
```

---

## Integration with Existing Workflow

You can run both scripts together:

```bash
# Step 1: Transfer all TAR files to Azure (for archival)
python sharepoint_tar_to_azure_blob.py

# Step 2: Randomly inspect some TAR files (for quality check)
python sharepoint_tar_inspector.py
```

Both scripts:
- ✅ Share same credentials (reuse config)
- ✅ Have independent state tracking
- ✅ Can run simultaneously or separately
- ✅ Won't interfere with each other

---

## Performance Considerations

### Disk Space Requirements
- **TAR files:** Temporary (deleted after extraction)
- **Extracted files:** Temporary (deleted after upload)
- **Estimate:** 2x size of largest TAR file

### Time Estimates
- **Download:** Depends on SharePoint speed
- **Extraction:** 1-2 minutes per GB
- **Upload:** Depends on Azure Blob speed
- **Overall:** ~5-10 minutes per TAR file (varies by size/network)

### Optimization Tips
1. Increase `sampling_ratio` if you want fewer inspections
2. Use fast local SSD for temp directories
3. Ensure good network connectivity to SharePoint & Azure

---

## Questions?

### How many files will be inspected?
- **Formula:** `Total TAR files / sampling_ratio`
- **Example:** 150 files ÷ 10 = 15 random TAR files inspected

### Can I inspect specific TAR files?
Not with this script (designed for random sampling). For specific files, manually edit the code or use a custom config.

### What happens if script crashes mid-run?
- ✅ State file saves after each TAR
- ✅ Re-run will skip successfully-inspected files
- ✅ Only continues with remaining files

### Can I change blob container?
Yes, edit `container_name` in config. But ensure container exists first!

---

## Support

For issues or questions:
1. Check log file: `sharepoint_tar_inspector.log`
2. Run in dry-run mode first: `--dry-run`
3. Verify config paths and credentials
4. Check disk space and permissions

---

## Summary

This inspection workflow provides:
- ✅ **Random quality checks** on TAR files
- ✅ **Team-friendly access** via SAS tokens
- ✅ **Detailed CSV reports** for tracking
- ✅ **State tracking** to avoid duplicates
- ✅ **Automatic cleanup** of temp files
- ✅ **Safe resumption** after interruptions

Perfect for validating TAR file contents before full-scale processing!
