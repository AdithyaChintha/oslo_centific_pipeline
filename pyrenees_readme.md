# Pyrenees Video Processing Pipeline

## Overview

The Pyrenees pipeline is a mode that is added to our comprehensive video processing system that analyzes videos using multiple AI models and generates annotations for Label Studio. It supports one mode:

1. **S3 Mode**: Processes video clips from S3 buckets with movement metadata integration

## Architecture

### Core Components

- **Ray Pipeline** (`ray_pipeline_testing_csam_s3.py`): Main orchestration pipeline
- **AI Models**:
  - NSFW content detection
  - Face detection & age estimation

### Output Structure

```
output_dir/
├── {video_id}_consolidated_model_results.json    # All model results
├── {video_id}_labelstudio_task.json             # Label Studio task
├── nsfw_output/                                 # NSFW detection results
├── face_output/                                 # Face detection results
```

## Prerequisites

### System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster processing)
- Ray cluster (recommended for distributed processing)

### Python Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `ray` - Distributed computing
- `boto3` - AWS S3 integration
- `azure-storage-blob` - Azure Blob Storage
- `torch` - PyTorch for AI models
- `opencv-python` - Video processing
- `pandas` - Data processing
- `pyarrow` - Parquet file handling

## Configuration

### 1. Environment Variables

Create a `.env` file or export these variables:

```bash
# AWS S3 Configuration (for S3 mode)
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"

```

### 2. S3 Configuration File

Modify `config/s3_config.yaml` if any changes are needed:

```yaml
input_file: "video_lists/server_1_videos.txt" # for splitting processing between two servers. NOTE: Using input file will diable s3 polling in pipeline, we will only process files present in the input file, check section 5 for creating input files
s3:
  # AWS Credentials (optional if using env vars)
  aws_access_key_id: "your_key"
  aws_secret_access_key: "your_secret"
  region_name: "us-east-1"

  # S3 Bucket Configuration
  bucket_name: "your-bucket-name"
  input_prefix: "outputs/batch_1_tier_1/clips/"     # Where clips are stored
  # output_prefix: "results/batch_1_tier_1/"          # Where to save results
  source_video_prefix: "transcoding_profile_id=5"              # Source video location

  # Movement Metadata (optional)
  movement_metadata:
    enabled: true
    s3_key: "outputs/batch_1_tier_1/batch_1_tier_1_metadata.parquet"

# Processing Configuration
processing:
  output_dir: "./output_s3"
  cleanup_after_processing: false  # Set true to delete local files after processing
  convert_mov_to_mp4: true

# Polling Configuration (for continuous processing)
polling:
  interval_minutes: 5
  max_videos_per_cycle: 10

# State Tracking
state_tracking:
  state_file: "s3_video_state.json"

# Label Studio Integration (Excel Export - NEW)
labelstudio:
  auto_create_tasks: true
  export_mode: "csv"  # Options: "api", "csv", "both"

  # Excel Export Configuration (NEW)
  csv_export:
    enabled: true
    local_output_dir: "./csv_files/batch_1/labelstudio_csv_prod"
    azure_upload: true
    azure_blob_prefix: "batch_1/csv_files_prod/"
    filename_format: "labelstudio_tasks_cycle_{cycle}_{timestamp}.xlsx"  # Excel format
    per_cycle_file: true  # One Excel file per polling cycle

  # Primary Label Studio project
  primary:
    api_url: "http://20.55.226.129:8082/api/projects/"
    api_key: "your_api_key"
    project_id: 1

  # Secondary Label Studio project (optional)
  secondary:
    enabled: true
    api_url: "http://20.55.226.129:8081/api/projects/"
    api_key: "your_api_key"
    project_id: 1
```


### 4. Pipeline Configuration

Modify `config/pipeline_config.yaml` to change the output pblob path to store the results and blob upload configuration:

```yaml
# Azure Storage Upload Settings
azure_storage:
  output_blob_prefix: "output_test"

# Blob Upload Configuration
blob_upload:
  max_workers: 8
  timeout_seconds: 300
  retry_attempts: 3
```
### 5. Input File Mode (Multi-Server Processing)

The pipeline supports two modes of operation:
1. **Polling Mode** (default): Continuously discovers new videos from S3
2. **Input File Mode**: Processes a pre-defined list of videos from a file

#### When to Use Input File Mode

- Processing videos across multiple servers in parallel
- Processing a specific subset of videos
- Resuming failed batches
- Avoiding S3 discovery overhead for large batches

#### How Input File Mode Works

When `input_file` is specified in `s3_config.yaml`:

1. **S3 polling is disabled** - The pipeline will NOT discover new videos from S3
2. **Videos are loaded from file** - Only videos listed in the input file are processed
3. **State tracking still works** - Already-completed videos are skipped based on state file
4. **Cycles continue** - Pipeline processes `max_videos_per_cycle` videos per cycle until all are done

#### Input File Format

The input file is a tab-separated text file with three columns:

```
s3_key	etag	size
outputs/batch_1_tier_1/clips/video_001.mov	"abc123def456"	2048576
outputs/batch_1_tier_1/clips/video_002.mov	"def789ghi012"	3145728
```

**Columns:**
1. `s3_key` - Full S3 path to the video clip
2. `etag` - S3 ETag for deduplication
3. `size` - File size in bytes

#### Generating Input Files for Multi-Server Processing

**Step 1: Generate split files**

Change all the required parameters in `s3_config.yaml` specifically related to the input S3 path and state path for the batch you are processing, then run:

```bash
python split_videos_for_servers.py \
  --servers 2 \
  --config config/s3_config.yaml \
  --output-dir video_lists_batch_2
```

**What this does:**
- Discovers all videos from S3 bucket (based on `input_prefix` in config)
- Loads existing state file to check what's already processed
- Excludes completed and failed videos
- Splits remaining videos evenly across the specified number of servers

**Output files:**
- `video_lists_batch_2/server_1_videos.txt` - First half of unprocessed videos
- `video_lists_batch_2/server_2_videos.txt` - Second half of unprocessed videos

**Step 2: Configure each server**

Create separate config files for each server:

**Server 1** - `config/s3_config_server1.yaml`:
```yaml
input_file: "video_lists_batch_2/server_1_videos.txt"

state_tracking:
  state_file: "s3_state_batch_2_server1.json"

labelstudio:
  csv_export:
    local_output_dir: "./csv_files/batch_2/server_1"
    filename_format: "labelstudio_tasks_server_1_cycle_{cycle}_{timestamp}.xlsx"
```

**Server 2** - `config/s3_config_server2.yaml`:
```yaml
input_file: "video_lists_batch_2/server_2_videos.txt"

state_tracking:
  state_file: "s3_state_batch_2_server2.json"

labelstudio:
  csv_export:
    local_output_dir: "./csv_files/batch_2/server_2"
    filename_format: "labelstudio_tasks_server_2_cycle_{cycle}_{timestamp}.xlsx"
```

**Step 3: Run on both servers**

```bash
# Server 1
python ray_pipeline_testing_csam_s3.py --mode s3 --s3-config config/s3_config_server1.yaml > logs/server1.log 2>&1

# Server 2
python ray_pipeline_testing_csam_s3.py --mode s3 --s3-config config/s3_config_server2.yaml > logs/server2.log 2>&1
```

#### Key Differences: Polling Mode vs Input File Mode

| Feature | Polling Mode | Input File Mode |
|---------|-------------|-----------------|
| **Video Discovery** | Continuous S3 polling | One-time load from file |
| **New Videos** | Automatically detected | Not detected (fixed list) |
| **State Tracking** | Tracks all discovered videos | Tracks only input file videos |
| **Multi-Server** | Requires coordination | Independent servers |
| **Use Case** | Real-time processing | Batch processing |
| **Completion** | Runs indefinitely | Exits when input file complete |

#### State File Behavior with Input Files

1. **On startup**: Pipeline loads all videos from input file
2. **Filtering**: Videos already marked "completed" or "failed" in state are skipped
3. **Processing**: Only unprocessed videos are queued (respects `max_videos_per_cycle`)
4. **Completion**: Pipeline exits when all input file videos are processed

**Example workflow:**
- Input file has 22,681 videos
- State file shows 5,000 already completed
- Pipeline will process the remaining 17,681 videos
- With `max_videos_per_cycle: 500`, it will take ~36 cycles

#### Monitoring Multi-Server Progress

Check progress on each server using the helper script:

```bash
# Server 1 progress
python check_progress.py \
  --input-file video_lists_batch_2/server_1_videos.txt \
  --state-file pyrenees_output/batch2/s3_state_batch_2_server1.json

# Server 2 progress
python check_progress.py \
  --input-file video_lists_batch_2/server_2_videos.txt \
  --state-file pyrenees_output/batch2/s3_state_batch_2_server2.json
```

**Output:**
```
======================================================================
Processing Progress Check
======================================================================
📊 Total videos in input file: 22,681
──────────────────────────────────────────────────────────────────────
Status Breakdown:
  ✅ Completed:   15,000
  ❌ Failed:      0
  ⏳ Processing:  0
  📋 Discovered:  50
──────────────────────────────────────────────────────────────────────
  🎯 Remaining:   7,681
──────────────────────────────────────────────────────────────────────
📈 Progress: 15,000 / 22,681 (66.13%)
```

#### Important Notes

- **No overlap**: The split script ensures each video appears in only ONE server's input file
- **Separate state files**: Each server maintains its own state to avoid conflicts
- **Separate outputs**: Each server writes to its own CSV/Excel output directory
- **Independent operation**: Servers can run at different speeds without coordination
- **Failure handling**: Failed videos are marked in state and won't be retried (excluded from splits)

## Running the Pipeline

### Mode 1: S3 Polling Mode (Recommended)

Process videos from S3 bucket continuously:

```bash
python ray_pipeline_testing_csam_s3.py \
  --mode s3 \
  --s3-config config/s3_config.yaml > log/run.log 2>&1
```


## Output Files

### 1. Consolidated Model Results JSON

`{video_id}_consolidated_model_results.json`:

```json
{
  "video_info": {
    "video_id": "00eEzUmL_9360653015_1000-1008",
    "s3_key": "outputs/batch_1_tier_1/clips/00eEzUmL_9360653015/1000-1008.mov",
    "source_video_id": "9360653015",
    "clip_id": "1000-1008.mov",
    "start_ms": 1000,
    "end_ms": 1008,
    "duration_ms": 8000
  },
  "model_results": {
    "nsfw": { ... },
    "face": { ... },
  }
}
```

### 2. Label Studio Task JSON

`{video_id}_labelstudio_task.json`:

```json
{
  "data": {
    "video": "https://azure_url_with_sas_token",
    "filename": "9360653015_1000-1008.mov",
    "s3_key": "s3://troveo-videodb-shared/outputs/batch_1_tier_1/clips/00eEzUmL_9360653015/1000-1008.mov",
    "source_s3_key": "s3://troveo-videodb-shared/transcoding_profile_id=5/00eEzUmL_9360653015.mp4",
    "source_video_id": "9360653015",
    "clip_id": "1000-1008.mov",
    "start_ms": 1000000,
    "end_ms": 1008000,
    "duration_ms": 8000000,
    "azure_url": "https://azure_url_with_sas_token",
    "model_status": {
      "nsfw_detection": true,
      "minor_detection": 2
    },
    "preannotations": {
      "has_movement": "Yes",
      "movement_type_primary": "Walking",
      "movement_type_secondary": null,
      "movement_confidence_primary": 0.95,
      "movement_confidence_secondary": null
    }
  },
  "predictions": [
    {
      "id": "nsfw_Xy12Ab",
      "type": "videoregion",
      "value": {
        "start": 2.5,
        "end": 5.8,
        "labels": ["nsfw"]
      },
      "score": 0.8
    },
    {
      "id": "minor_Cd34Ef",
      "type": "videoregion",
      "value": {
        "start": 1.0,
        "end": 7.5,
        "labels": ["minor"]
      },
      "score": 0.8
    }
  ]
}
```

**Key Fields:**
- `data.video`: Azure SAS URL for video playback
- `data.filename`: Display filename for the clip
- `data.s3_key`: Full S3 URI of the clip (s3://bucket/path format)
- `data.source_s3_key`: Full S3 URI of the source video
- `data.source_video_id`: Source video identifier (numeric ID only)
- `data.clip_id`: Clip filename with extension
- `data.start_ms/end_ms/duration_ms`: Clip temporal metadata in milliseconds
- `data.azure_url`: Azure SAS URL (same as video field)
- `data.model_status`: Model detection results
  - `nsfw_detection`: `true` if NSFW detected, `null` if not detected or model failed
  - `minor_detection`: count of minors if detected, `null` if not detected or model failed
- `data.preannotations`: Movement metadata from parquet file (optional)
  - `has_movement`: "Yes"/"No"
  - `movement_type_primary/secondary`: Movement classification
  - `movement_confidence_primary/secondary`: Confidence scores (0.0-1.0)
- `predictions[]`: Array of video region annotations
  - `id`: Unique identifier for each prediction
  - `type`: "videoregion" for temporal segments
  - `value.start/end`: Start and end time in seconds (float)
  - `value.labels`: Array with single label - `["nsfw"]` or `["minor"]`
  - `score`: Confidence score (0.0-1.0)

## State Tracking

The pipeline maintains state in `s3_video_state.json`:

```json
{
  "outputs/batch_1_tier_1/clips/video_001.mov": {
    "status": "completed",
    "etag": "abc123...",
    "discovered_at": "2025-10-02T10:00:00Z",
    "processing_started_at": "2025-10-02T10:01:00Z",
    "completed_at": "2025-10-02T10:05:00Z",
    "output_path": "/output_s3/video_001",
    "azure_video_url": "https://...",
    "labelstudio_task_id": 123
  }
}
```

## Monitoring and Debugging

### View Processing Logs

```bash
# Real-time logs
tail -f log/run.log

# Search for errors
grep "ERROR" logs/pipeline.log

# View specific video processing
grep "video_id" logs/pipeline.log
```

### Check State

```python
import json

with open('s3_video_state.json') as f:
    state = json.load(f)

# Count by status
from collections import Counter
statuses = Counter(v['status'] for v in state.values())
print(statuses)
```

### Performance Metrics

The pipeline generates timing data in CSV format:

```bash
# View timing summary
cat output_s3/timing_summary.csv

# Analyze bottlenecks
python -c "import pandas as pd; df = pd.read_csv('output_s3/timing_summary.csv'); print(df.sort_values('duration_seconds', ascending=False).head(10))"
```

## Troubleshooting

### Common Issues

1. **AWS Credentials Error**
   ```
   Solution: Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set
   ```

2. **Azure Upload Failed**
   ```
   Solution: Check Azure credentials in blobfuse2_config.yaml
   ```

3. **CUDA Out of Memory**
   ```
   Solution: Reduce batch size or number of parallel workers
   ```

4. **Ray Connection Failed**
   ```
   Solution: Start Ray cluster: ray start --head
   ```

5. **Movement Metadata Not Found**
   ```
   Solution: Verify s3_key in movement_metadata config points to correct parquet file
   ```

## Advanced Features

### Movement Metadata Integration

The pipeline can enrich clips with preannotations movement metadata from parquet files:

```yaml
movement_metadata:
  enabled: true
  s3_key: "outputs/batch_1_tier_1/batch_1_tier_1_metadata.parquet"
```

Metadata includes:
- Movement type
- Has movement flag
- Confidence scores

### Excel Export for Label Studio (NEW)

The pipeline now supports exporting task data to **Excel files (.xlsx)** instead of or in addition to API calls:

#### Export Modes

1. **CSV Mode** (Excel Export Only):
```yaml
labelstudio:
  export_mode: "csv"
  csv_export:
    enabled: true
    local_output_dir: "./csv_files/batch_1/labelstudio_csv_prod"
    azure_upload: true
```

2. **API Mode** (Traditional Label Studio API):
```yaml
labelstudio:
  export_mode: "api"
  primary:
    api_url: "http://20.55.226.129:8082/api/projects/"
    api_key: "your_api_key"
    project_id: 1
```

3. **Both Mode** (Excel + API):
```yaml
labelstudio:
  export_mode: "both"
  csv_export:
    enabled: true
  primary:
    api_url: "..."
```

#### Excel File Structure

Files are created per polling cycle:
- `labelstudio_tasks_cycle_1_20251003_183233.xlsx` - First 500 videos
- `labelstudio_tasks_cycle_2_20251003_190000.xlsx` - Next 500 videos
- etc.

**Columns include:**
- `video`, `filename`, `azure_url`
- `s3_key`, `source_s3_key`, `bucket`
- `source_video_id`, `clip_id`
- `start_ms`, `end_ms`, `duration_ms`
- `nsfw_detection_status`, `minor_detection_status`
- `nsfw_segments`, `minor_segments` (JSON)
- `movement_metadata` (JSON)
- `Movement1` - `Movement5` (if available)

#### Multi-Project Support

Send tasks to **two Label Studio projects** simultaneously:

```yaml
labelstudio:
  primary:
    api_url: "http://20.55.226.129:8082/api/projects/"
    api_key: "primary_key"
    project_id: 1

  secondary:
    enabled: true
    api_url: "http://20.55.226.129:8081/api/projects/"
    api_key: "secondary_key"
    project_id: 1
```

#### Azure Blob Upload

Excel files are automatically uploaded to Azure:

```yaml
csv_export:
  azure_upload: true
  azure_blob_prefix: "batch_1/csv_files_prod/"
```

Files appear in blob storage:
```
batch_1/csv_files_prod/labelstudio_tasks_cycle_1_20251003_183233.xlsx
batch_1/csv_files_prod/labelstudio_tasks_cycle_2_20251003_190000.xlsx
```

### Post-Processing: Excel to JSONL Conversion

Convert Excel files to JSONL format with numeric video IDs:

```bash
python pyrenees_post_consolidation_from_xlsx.py data.xlsx
# Output: Troveo_Delivery1_10032025_100.jsonl
```

**Key Features:**
- Converts to JSONL format (one JSON object per line)
- Extracts numeric video IDs: `"00NfzFc8_6066938804"` → `"6066938804"`
- Auto-generated filename: `Troveo_Delivery1_MMDDYYYY_NumTasks.jsonl`
- Supports batch processing of multiple files

**Usage Examples:**

```bash
# Single Excel file
python pyrenees_post_consolidation_from_xlsx.py labelstudio_tasks.xlsx
# Output: Troveo_Delivery1_10032025_25.jsonl

# Folder with multiple files
python pyrenees_post_consolidation_from_xlsx.py ./csv_files/batch_1/
# Output: Troveo_Delivery1_10032025_1500.jsonl

# Custom output name
python pyrenees_post_consolidation_from_xlsx.py data.xlsx custom_output.jsonl
```

**JSONL Output Format:**

```jsonl
{"source_s3_key": "s3://bucket/video.mp4", "source_video_id": "6066938804", "clip_id": "1112-1120.mov", "start_ms": 1112000, "end_ms": 1120000, "duration_ms": 8000, "movement_type_1": "shake", "confidence_1": 1.0, "has_movement": "Yes"}
{"source_s3_key": "s3://bucket/video.mp4", "source_video_id": "542721916", "clip_id": "668-676.mov", "start_ms": 668000, "end_ms": 676000, "duration_ms": 8000, "movement_type_1": null, "confidence_1": null, "has_movement": "No"}
```


## API Reference

### Main Functions

#### `pipeline_s3_mode(config, s3_config)`
Main S3 polling pipeline

#### `save_s3_consolidated_results_and_labelstudio(...)`
Save consolidated results and Label Studio JSON

#### `process_single_shard_through_pipeline(...)`
Process single video through all AI models

#### `create_labelstudio_task_for_s3_mode(...)`
Create Label Studio task with S3 metadata

### Cleanup

```yaml
# Auto-cleanup local files after upload
processing:
  cleanup_after_processing: true
```

## Support

For issues and questions:
- Check logs in `logs/pipeline.log`
- Review state in `s3_video_state.json`
- Contact: Leela Krishna, Sai Charith Pasula, Gouti Pavan, Adithya Chintha, Mangesh Damre

