# Azure Blob Storage Downloader

A Python script to download specific files from Azure Blob Storage using your blobfuse2 configuration.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make the script executable:
```bash
chmod +x azure_blob_downloader.py
```

## Usage

### List all available files
```bash
python azure_blob_downloader.py --list --destination /tmp
```

### Download specific files
```bash
python azure_blob_downloader.py --files video1.mp4 video2.mp4 --destination /path/to/download
```

### Download files by pattern (prefix)
```bash
python azure_blob_downloader.py --pattern "videos/" --destination /path/to/download
```

### Download with custom config file
```bash
python azure_blob_downloader.py --config /path/to/config.yaml --files video.mp4 --destination /path/to/download
```

## Features

- Uses your existing blobfuse2_config.yaml credentials
- Downloads single files or multiple files at once
- Pattern-based downloads (by prefix)
- Preserves directory structure from blob storage
- Progress logging and error handling
- Lists available files without downloading

## Examples

```bash
# List all files in the container
python azure_blob_downloader.py --list -d /tmp

# Download specific video files
python azure_blob_downloader.py -f "path/to/video1.mp4" "path/to/video2.mp4" -d ./downloads

# Download all files starting with "2024/"
python azure_blob_downloader.py -p "2024/" -d ./downloads/2024

# Verbose output
python azure_blob_downloader.py -f "video.mp4" -d ./downloads --verbose
```
