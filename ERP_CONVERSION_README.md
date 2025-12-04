# TAR Inspector with ERP Conversion

## Overview

This enhanced version of `sharepoint_tar_inspector.py` now automatically:
1. **Detects dual-fisheye videos** (Insta360 format)
2. **Converts them to ERP (Equirectangular) view**
3. **Uploads both original AND ERP versions** to Azure Blob
4. **Generates CSV with all files** including ERP conversions

---

## Complete End-to-End Flow

### Step 1: Download & Extract TAR
```
SharePoint TAR → Download → Extract to local disk
```

### Step 2: Scan for Dual-Fisheye Videos
For each video file, checks:
- **2 video streams** (dual-lens camera)
- **2:1 aspect ratio** (3840x1920, 5760x2880, etc.)
- **.insv extension** (Insta360 format)

### Step 3: Convert to ERP (if dual-fisheye detected)
```
original_video.mp4 → FFmpeg conversion → original_video_erpview.mp4
```

Uses FFmpeg v360 filter:
- **Input**: `dfisheye` (dual fisheye) or `fisheye` (single)
- **Output**: `equirect` (ERP 2:1 format)
- **Resolution**: 5760x2880 (can be configured)
- **Codec**: H.264 (libx264) with CRF 23

### Step 4: Upload to Azure Blob
Uploads **BOTH** files:
- `original_video.mp4` → Azure Blob (original)
- `original_video_erpview.mp4` → Azure Blob (ERP version)

Both files get:
- ✅ Correct `Content-Type` (video/mp4)
- ✅ `Content-Disposition: inline` (plays in browser)
- ✅ SAS token with 365-day expiry

### Step 5: Generate CSV Report
Each row contains:
- `tar_file_name`: Source TAR file
- `individual_file_name`: File name (original or _erpview)
- `is_dual_fisheye`: YES/NO
- `is_erp_version`: YES/NO (identifies ERP conversions)
- `blob_url`: Full URL with SAS token
- All other metadata

---

## Usage

### Run the Script
```bash
python sharepoint_tar_inspector.py --config config/tar_inspection_config.yaml
```

### Configuration (tar_inspection_config.yaml)
```yaml
inspection:
  execution_mode: "list"  # or "random"

  # List specific TAR files to process
  tar_file_list:
    - "83fe2ca2-cf27-4bd6-b02e-a0c434f67748.tar"
    - "e85b5221-0201-4df9-b91b-e3542e7f39d5.tar"
    - "cad90f83-504b-40f0-bd1c-43e2ed79f3f2.tar"

  sas_token_expiry_days: 365
```

---

## CSV Report Structure

| Column | Description | Example |
|--------|-------------|---------|
| `tar_file_name` | Source TAR file | `83fe2ca2...tar` |
| `individual_file_name` | File name | `video1.mp4` or `video1_erpview.mp4` |
| `individual_file_full_path` | Path inside TAR | `subfolder/video1.mp4` |
| `individual_file_type` | Extension | `.mp4` |
| `is_dual_fisheye` | Dual-fisheye detected? | `YES` / `NO` |
| `is_erp_version` | ERP converted version? | `YES` / `NO` |
| `tar_size_gb` | TAR file size | `2.5000` |
| `individual_file_size_gb` | File size | `0.123456` |
| `sas_token` | Azure SAS token | `sv=2021-06-08&...` |
| `blob_url` | **Full playable URL** | `https://...?sv=...` |
| `blob_path` | Blob storage path | `untar_folder_oslo2/...` |
| `transfer_date` | Upload timestamp | `2025-01-19T12:34:56` |

---

## Example Workflow Output

```
Processing TAR file: 83fe2ca2-cf27-4bd6-b02e-a0c434f67748.tar (2.50 GB)
================================================================================

  [1/15] Analyzing: video1.mp4
  🎥 DUAL-FISHEYE detected: video1.mp4
  🔄 Converting to ERP: video1_erpview.mp4
  Converting dual-fisheye to ERP: video1.mp4
  Detected 2 video streams - converting dual-lens to ERP
  ✅ ERP conversion successful: video1_erpview.mp4 (2.35 GB)

  📤 Uploading: video1.mp4 (2.50 GB)
  ✅ Uploaded to blob: untar_folder_oslo2/.../video1.mp4

  📤 Uploading ERP video: video1_erpview.mp4 (2.35 GB)
  ✅ ERP video uploaded successfully: video1_erpview.mp4

  [2/15] Analyzing: audio1.wav
  📤 Uploading: audio1.wav (0.05 GB)

  ...

✅ Successfully processed 17 files from TAR (15 original + 2 ERP conversions)
```

---

## Summary Report

```
INSPECTION SUMMARY
================================================================================
✅ Successful inspections: 3
❌ Failed inspections: 0
📁 Total files processed: 47
🎥 Dual-fisheye videos found: 5
🔄 ERP videos created: 5
================================================================================
```

---

## Filtering Results

### Filter Dual-Fisheye Videos Only
```bash
python filter_dual_fisheye_videos.py tar_inspection_report_20250119_123456.csv
```

Output:
```
DUAL-FISHEYE VIDEO FILTER RESULTS
================================================================================
📄 Input file: tar_inspection_report_20250119_123456.csv
📊 Total entries: 47
🎥 Dual-fisheye videos: 5
📈 Percentage: 10.6%
================================================================================

🎥 Dual-Fisheye Videos Found:
--------------------------------------------------------------------------------
  1. video1.mp4                              (TAR: 83fe2ca2...tar)
  2. video2.insv                             (TAR: 83fe2ca2...tar)
  3. video3.mp4                              (TAR: e85b5221...tar)
  4. video4.mov                              (TAR: cad90f83...tar)
  5. video5.mp4                              (TAR: cad90f83...tar)

✅ Filtered CSV created: tar_inspection_report_20250119_123456_dual_fisheye_only.csv
   Contains 5 dual-fisheye video entries
```

---

## File Naming Convention

| Original File | ERP Converted File |
|--------------|-------------------|
| `video.mp4` | `video_erpview.mp4` |
| `clip.insv` | `clip_erpview.mp4` |
| `VID_20250119.mov` | `VID_20250119_erpview.mp4` |

---

## Browser Playback

### Original Dual-Fisheye Video
```
https://oslotestvideo.blob.core.windows.net/.../video1.mp4?sv=...
```
→ **Shows dual circular fisheye view** (side-by-side circles)

### ERP Converted Video
```
https://oslotestvideo.blob.core.windows.net/.../video1_erpview.mp4?sv=...
```
→ **Shows equirectangular panoramic view** (360° stretched view)

Both URLs:
- ✅ **Play directly in browser** (no download!)
- ✅ Support HTML5 video player
- ✅ Can be shared with team
- ✅ Valid for 365 days

---

## Technical Details

### Dual-Fisheye Detection Logic

```python
def _is_dual_fisheye_video(video_path):
    # Method 1: Check file extension
    if extension == '.insv':
        return True

    # Method 2: Check video streams (ffprobe)
    if num_video_streams >= 2:
        return True

    # Method 3: Check aspect ratio
    if width/height between 1.9 and 2.1:
        return True

    return False
```

### ERP Conversion Command

**Dual-lens (2 streams):**
```bash
ffmpeg -i input.insv \
  -vf "[0:v:0][0:v:1]hstack=inputs=2[dual]; \
       [dual]v360=input=dfisheye:output=equirect: \
       ih_fov=190:iv_fov=190:w=5760:h=2880" \
  -c:v libx264 -crf 23 -preset medium \
  -c:a copy -movflags +faststart \
  output_erpview.mp4
```

**Single-lens (1 stream):**
```bash
ffmpeg -i input.mp4 \
  -vf "v360=input=fisheye:output=equirect: \
       ih_fov=190:iv_fov=190:w=5760:h=2880" \
  -c:v libx264 -crf 23 -preset medium \
  -c:a copy -movflags +faststart \
  output_erpview.mp4
```

---

## Requirements

### System Dependencies
- **ffmpeg** (with v360 filter support)
- **ffprobe** (part of ffmpeg)

Check installation:
```bash
ffmpeg -version | grep v360
```

### Python Packages
Already included in the project:
- `azure-storage-blob`
- `pyyaml`
- `requests`

---

## Disk Space Considerations

During processing:
- Original TAR file downloaded (e.g., 2.5 GB)
- Extracted files (e.g., 2.5 GB)
- **ERP conversions created** (similar size to originals)

**Total temporary space**: ~3x TAR size

**After processing**:
- Local files cleaned up automatically
- Only Azure Blob storage used

---

## Performance

### ERP Conversion Time
| Video Size | Resolution | Conversion Time |
|-----------|-----------|----------------|
| 100 MB | 3840x1920 | ~30 seconds |
| 500 MB | 5760x2880 | ~2 minutes |
| 2 GB | 5760x2880 | ~8 minutes |
| 5 GB | 7680x3840 | ~20 minutes |

*Times are approximate, depends on CPU*

### Timeout Settings
- FFmpeg conversion: **1 hour** per video
- Blob upload: No timeout (handles large files)

---

## Troubleshooting

### Issue: "FFmpeg conversion failed"
**Solution**: Check ffmpeg has v360 filter:
```bash
ffmpeg -filters | grep v360
```

### Issue: "Failed to probe video"
**Solution**: Check ffprobe is installed:
```bash
which ffprobe
```

### Issue: "ERP file not created"
**Possible causes**:
- Corrupted video file
- Unsupported codec
- Insufficient disk space

**Check logs**: `sharepoint_tar_inspector.log`

---

## Next Steps

1. Run the script with your config
2. Check the CSV report for both original and ERP videos
3. Open blob URLs in browser to verify playback
4. Use filter script to analyze dual-fisheye distribution
5. Share URLs with your team for inspection!
