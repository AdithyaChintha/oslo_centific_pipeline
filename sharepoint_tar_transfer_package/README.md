# SharePoint TAR to Azure Blob Transfer Package

This package provides a complete workflow for automatically transferring TAR files from SharePoint to Azure Blob Storage.

## Package Contents

```
sharepoint_tar_transfer_package/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── sharepoint_tar_to_azure_blob.py       # Main script
├── config/
│   ├── tar_transfer_config.yaml          # Configuration file (template)
│   └── tar_transfer_config.yaml.example  # Example configuration
├── utils/
│   ├── __init__.py                       # Python package marker
│   ├── sharepoint_utils.py               # SharePoint authentication and operations
│   ├── azure_blob_utils.py               # Azure Blob Storage utilities
│   └── resumable_upload_manager.py       # Large file upload manager
└── docs/
    └── SharePoint_TAR_to_Azure_Blob_Documentation.docx  # Complete documentation
```

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Azure Storage Account with Blob Storage
- SharePoint site with appropriate access permissions
- Azure AD App Registration with client credentials

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install azure-storage-blob>=12.19.0 PyYAML>=6.0 requests
```

### 3. Configure the Script

1. Edit `config/tar_transfer_config.yaml` with your credentials:
   - Azure AD credentials (tenant_id, client_id, client_secret)
   - SharePoint site details (site_id, drive_id)
   - Azure Blob Storage connection details
   - Local paths and processing options


### 4. Run the Script

Basic usage with default configuration:
```bash
python sharepoint_tar_to_azure_blob.py
```

With custom configuration:
```bash
python sharepoint_tar_to_azure_blob.py --config config/tar_transfer_config.yaml
```

Dry run (preview without transferring):
```bash
python sharepoint_tar_to_azure_blob.py --dry-run
```

## Key Features

- **Multiple Transfer Methods**: Server-to-server copy, streaming transfer, and download+upload fallback
- **Duplicate Prevention**: Tracks transferred files to avoid re-processing
- **Progress Tracking**: Detailed logging and progress updates
- **Automatic Retry**: Falls back to alternative methods on failure
- **Batch Processing**: Configurable limits to prevent overwhelming systems
- **State Persistence**: JSON state file tracks all transfers

## Configuration Guide

### Azure AD Settings

You need an Azure AD App Registration with these permissions:
- `Sites.ReadWrite.All` - For SharePoint access
- `Files.ReadWrite.All` - For file operations

```yaml
azure_ad:
  tenant_id: "your-tenant-id"
  client_id: "your-client-id"
  client_secret: "your-client-secret"
```

### SharePoint Settings

Get your SharePoint site_id and drive_id from your IT administrator:

```yaml
sharepoint:
  site_id: "your-site-id"
  drive_id: "your-drive-id"
  tar_source_folder_path: "/Uploads"  # Folder containing TAR files
```

### Azure Blob Storage Settings

```yaml
azure_blob:
  connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
  container_name: "tar-files"
  target_prefix: "tar_files/"  # Optional prefix within container
```

### Processing Options

```yaml
processing:
  max_files_per_run: 50          # Limit files per execution
  cleanup_after_upload: true     # Remove temporary files after upload

api:
  timeout_seconds: 300           # API timeout
  retry_attempts: 3              # Number of retries
  retry_delay_seconds: 5         # Delay between retries
```

## Transfer Methods

The script tries three methods in order:

1. **Server-to-Server Copy** (fastest)
   - Azure copies directly from SharePoint URL
   - Zero client bandwidth usage
   - Best for large files

2. **Streaming Transfer** (low memory)
   - Streams data through client
   - Minimal memory footprint (64MB chunks)
   - No disk I/O required

3. **Download + Upload** (fallback)
   - Downloads to local disk
   - Uploads from local disk
   - Most reliable but slowest

## Logging and Monitoring

Logs are written to:
- Console (stdout)
- File: `sharepoint_tar_to_blob.log`

Transfer state is tracked in:
- JSON file specified in config (default: `tar_transfer_state.json`)

Example state file:
```json
{
  "file1.tar": {
    "transferred_at": "2025-11-11T10:30:00",
    "size": 1048576000,
    "sharepoint_modified": "2025-11-10T15:22:33Z",
    "blob_path": "tar_files/file1.tar",
    "transfer_method": "server-to-server"
  }
}
```

## Troubleshooting

### No files found in SharePoint
- Verify `tar_source_folder_path` in config
- Check SharePoint permissions
- Ensure files have `.tar`, `.tar.gz`, or `.tgz` extension

### Authentication failures
- Verify Azure AD credentials
- Check credential expiration
- Ensure app has proper permissions in Azure AD

### Transfer timeouts
- Increase `timeout_seconds` in config
- Check network connectivity
- Verify SharePoint URLs are accessible

### All transfer methods fail
- Review detailed error messages in log file
- Verify network connectivity to both SharePoint and Azure
- Check disk space for fallback method

## Security Best Practices

1. **Never commit credentials to version control**
   - Use environment variables
   - Use Azure Key Vault for production
   - Add config files with credentials to `.gitignore`

2. **Use least-privilege permissions**
   - Grant only necessary SharePoint permissions
   - Use separate app registrations for different environments

3. **Secure the state file**
   - Contains metadata about transferred files
   - Store in secure location
   - Back up regularly

## Support and Documentation

For detailed documentation, see:
- `docs/SharePoint_TAR_to_Azure_Blob_Documentation.docx`

For issues or questions:
- Review the troubleshooting section
- Check the log files for detailed error messages
- Contact your IT administrator for credential issues

## Example Workflow

1. **Initial Setup** (one-time)
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Configure credentials
   cp config/tar_transfer_config.yaml.example config/tar_transfer_config.yaml
   nano config/tar_transfer_config.yaml

   # Set environment variables
   export AZURE_CLIENT_SECRET="your-secret"
   ```

2. **Test Run** (verify configuration)
   ```bash
   # Dry run to preview transfers
   python sharepoint_tar_to_azure_blob.py --dry-run
   ```

3. **Production Run**
   ```bash
   # Execute transfers
   python sharepoint_tar_to_azure_blob.py
   ```

4. **Scheduled Execution** (optional)
   ```bash
   # Add to crontab for daily execution
   0 2 * * * cd /path/to/package && python sharepoint_tar_to_azure_blob.py
   ```

## License

This package is provided for internal use within your organization.

## Version

Version: 1.0.0
Last Updated: November 2025
