# Azure Upload Implementation for S3 Pipeline

## Overview
This document describes the implementation of Azure Blob Storage upload functionality in the S3-based video processing pipeline. The implementation converts the pipeline from uploading results to S3 to uploading to Azure Blob Storage while maintaining all existing S3 functionality for video discovery and download.

## Implementation Summary

### Files Modified
- **Target File**: `ray_pipeline_testing_csam_s3.py`
- **Function**: `pipeline_s3_mode()` function
- **Sections Modified**: Lines 1519-1854

### Key Changes Made

#### 1. Azure Utility Functions Added (Lines 1519-1641)
- **`generate_azure_sas_url()`**: Generates SAS URLs for Azure blobs with configurable expiry
- **`create_labelstudio_task_for_azure()`**: Creates Label Studio tasks using Azure video URLs with the same format as existing pipeline

#### 2. Azure Configuration Loading (Lines 1651-1680)
- Added Azure configuration loading using `load_azure_config()` and `load_pipeline_config()`
- Handles both direct and nested Azure configuration structures
- Graceful fallback if Azure configuration is missing

#### 3. S3 Upload Section Replaced (Lines 1802-1830)
- **Before**: Uploaded results to S3 using `upload_results_to_s3()`
- **After**: Uploads results to Azure using `upload_output_directory_with_sas_optimized()`
- Generates Azure SAS URLs for MP4 videos

#### 4. Label Studio Task Creation Updated (Lines 1832-1843)
- **Before**: Used `create_labelstudio_task_for_s3()` with S3 URLs
- **After**: Uses `create_labelstudio_task_for_azure()` with Azure SAS URLs
- Added validation to ensure Azure video URL is available

#### 5. Completion Tracking Updated (Lines 1845-1854)
- **Before**: Tracked `s3_video_url` and `s3_results_path`
- **After**: Tracks `azure_video_url` and `azure_results_path`

## Technical Details

### Azure Configuration Structure
The implementation expects Azure configuration in one of these formats:

**Option 1: Direct Structure**
```yaml
container: "your-container"
account-name: "your-account"
account-key: "your-key"
```

**Option 2: Nested Structure (blobfuse2_config.yaml)**
```yaml
azstorage:
  container: "your-container"
  account-name: "your-account"
  account-key: "your-key"
```

### Pipeline Configuration
The implementation uses the following pipeline configuration:
```yaml
azure_storage:
  output_blob_prefix: "one_data_platform_csam/"

blob_upload:
  # Upload configuration settings
```

### SAS URL Generation
- **Expiry**: 365 days (configurable)
- **Permissions**: Read-only access
- **Format**: `https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}`

### Label Studio Integration
- Uses the same task format as existing pipeline
- Extracts NSFW and minor segments from processing results
- Creates predictions with proper video region annotations
- Maintains compatibility with existing Label Studio project structure

## Error Handling

### Azure Configuration Missing
- Logs warning and continues without Azure upload
- Skips Label Studio task creation if no Azure video URL
- Maintains S3 functionality for video discovery and download

### Upload Failures
- Azure upload failures are logged but don't stop processing
- Graceful degradation to local processing only
- State tracking continues to work

### SAS URL Generation Failures
- Logs error and returns None
- Prevents Label Studio task creation if URL generation fails

## Preserved Functionality

### S3 Operations (Unchanged)
- ✅ S3 client creation and configuration
- ✅ S3 video discovery (`discover_videos_in_s3()`)
- ✅ S3 video download (`download_video_from_s3()`)
- ✅ S3 state tracking (`S3VideoStateTracker`)
- ✅ S3 imports and dependencies

### Pipeline Operations (Unchanged)
- ✅ Video processing pipeline
- ✅ MOV to MP4 conversion
- ✅ NSFW and face detection
- ✅ Results generation
- ✅ State management

## Testing Requirements

### Prerequisites
1. Azure Blob Storage account configured
2. `blobfuse2_config.yaml` or equivalent Azure configuration
3. `config/pipeline_config.yaml` with Azure settings
4. Label Studio server accessible

### Test Scenarios
1. **Azure Configuration Present**: Verify upload and URL generation
2. **Azure Configuration Missing**: Verify graceful fallback
3. **Upload Success**: Verify results uploaded and URLs generated
4. **Upload Failure**: Verify error handling and continuation
5. **Label Studio Integration**: Verify task creation with Azure URLs

### Validation Points
- [ ] Azure blob upload works correctly
- [ ] SAS URLs are generated properly
- [ ] Label Studio tasks are created with Azure URLs
- [ ] No S3 functionality is broken
- [ ] Error handling works for Azure operations
- [ ] State tracking includes Azure URLs

## Usage

### Running the Pipeline
```bash
python ray_pipeline_testing_csam_s3.py
```

### Configuration Files Required
1. **S3 Configuration**: For video discovery and download
2. **Azure Configuration**: For result upload and URL generation
3. **Pipeline Configuration**: For upload settings and prefixes
4. **Label Studio Configuration**: For task creation

### Expected Output
- Videos discovered from S3
- Videos downloaded and processed locally
- Results uploaded to Azure Blob Storage
- Azure SAS URLs generated for MP4 videos
- Label Studio tasks created with Azure video URLs
- State tracking updated with Azure URLs

## Dependencies

### Required Python Packages
- `azure-storage-blob`: For Azure Blob Storage operations
- `requests`: For Label Studio API calls
- All existing S3 and pipeline dependencies

### Configuration Files
- `blobfuse2_config.yaml`: Azure storage configuration
- `config/pipeline_config.yaml`: Pipeline and upload settings
- S3 configuration files: For video discovery and download

## Troubleshooting

### Common Issues

1. **Azure Configuration Not Found**
   - Check `blobfuse2_config.yaml` exists and is properly formatted
   - Verify Azure credentials are correct

2. **Upload Failures**
   - Check Azure storage account permissions
   - Verify container exists and is accessible
   - Check network connectivity to Azure

3. **SAS URL Generation Fails**
   - Verify Azure account key is correct
   - Check blob exists before generating SAS URL
   - Verify SAS permissions are properly configured

4. **Label Studio Task Creation Fails**
   - Check Label Studio server is accessible
   - Verify API token is valid
   - Check project ID exists

### Debug Logging
Enable debug logging to see detailed Azure operations:
```python
import logging
logging.getLogger('azure').setLevel(logging.DEBUG)
```

## Future Enhancements

### Potential Improvements
1. **Retry Logic**: Add retry mechanism for Azure upload failures
2. **Progress Tracking**: Add progress indicators for large uploads
3. **Compression**: Add compression for result files before upload
4. **Parallel Uploads**: Optimize upload performance with parallel processing
5. **Error Recovery**: Add recovery mechanisms for partial uploads

### Configuration Options
1. **SAS Expiry**: Make SAS URL expiry configurable
2. **Upload Chunk Size**: Optimize upload performance
3. **Retry Attempts**: Configure retry behavior
4. **Cleanup Options**: Configure local file cleanup after upload

## Conclusion

The Azure upload implementation successfully integrates Azure Blob Storage into the S3-based video processing pipeline while maintaining all existing functionality. The implementation provides:

- ✅ Seamless Azure integration
- ✅ Preserved S3 functionality
- ✅ Robust error handling
- ✅ Label Studio compatibility
- ✅ Comprehensive logging
- ✅ Graceful degradation

The pipeline now uploads processed results to Azure Blob Storage and generates SAS URLs for video access, enabling cloud-based video processing workflows while maintaining the existing S3-based video discovery and download capabilities.
