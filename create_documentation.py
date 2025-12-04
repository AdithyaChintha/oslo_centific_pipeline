#!/usr/bin/env python3
"""
Script to generate comprehensive documentation for sharepoint_tar_to_azure_blob.py
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE

def add_heading_with_style(doc, text, level=1):
    """Add a styled heading to the document."""
    heading = doc.add_heading(text, level=level)
    if level == 1:
        heading.runs[0].font.color.rgb = RGBColor(0, 51, 102)
    return heading

def add_code_block(doc, code_text):
    """Add a code block with monospace font and gray background."""
    para = doc.add_paragraph()
    run = para.add_run(code_text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    para.paragraph_format.left_indent = Inches(0.5)
    para.paragraph_format.space_before = Pt(6)
    para.paragraph_format.space_after = Pt(6)
    return para

def create_documentation():
    """Create comprehensive documentation for the SharePoint TAR to Azure Blob script."""

    doc = Document()

    # Title
    title = doc.add_heading('SharePoint TAR to Azure Blob Transfer Script', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Add subtitle with date
    subtitle = doc.add_paragraph('Technical Documentation')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(14)
    subtitle.runs[0].font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph()  # Spacing

    # Table of Contents (manual)
    add_heading_with_style(doc, 'Table of Contents', 1)
    toc_items = [
        '1. Overview',
        '2. Key Features',
        '3. Architecture & Design',
        '4. Configuration',
        '5. Main Components',
        '6. Transfer Methods',
        '7. Usage Instructions',
        '8. Logging & Monitoring',
        '9. Error Handling',
        '10. Best Practices',
    ]
    for item in toc_items:
        doc.add_paragraph(item, style='List Number')

    doc.add_page_break()

    # 1. Overview
    add_heading_with_style(doc, '1. Overview', 1)
    doc.add_paragraph(
        'The sharepoint_tar_to_azure_blob.py script is an automated tool designed to transfer TAR files '
        'from SharePoint to Azure Blob Storage. It implements a sophisticated multi-method transfer approach '
        'with automatic fallback mechanisms to ensure reliable data migration.'
    )

    doc.add_paragraph(
        'File Path: /home/vision_ai_adm/code/oslo/combined_branch/sharepoint_tar_to_azure_blob.py'
    ).runs[0].font.italic = True

    # 2. Key Features
    add_heading_with_style(doc, '2. Key Features', 1)
    features = [
        'Automated TAR file discovery and transfer from SharePoint',
        'Multiple transfer methods with intelligent fallback strategy',
        'Duplicate detection and transfer state tracking',
        'Server-to-server copy for maximum efficiency',
        'Streaming transfers to minimize memory usage',
        'Comprehensive logging and progress tracking',
        'Dry-run mode for testing',
        'Configurable batch processing limits',
        'Automatic cleanup of temporary files',
        'Unified YAML configuration for all settings',
    ]
    for feature in features:
        doc.add_paragraph(feature, style='List Bullet')

    # 3. Architecture & Design
    add_heading_with_style(doc, '3. Architecture & Design', 1)

    add_heading_with_style(doc, '3.1 High-Level Architecture', 2)
    doc.add_paragraph(
        'The script follows a modular architecture with clear separation of concerns:'
    )

    architecture_points = [
        'TarTransferConfig: Configuration management and validation',
        'TarTransferManager: Core orchestration and business logic',
        'SharePointAuthenticator: Azure AD authentication (from utils)',
        'SharePointFileManager: SharePoint file operations (from utils)',
        'BlobServiceClient: Azure Blob Storage operations',
    ]
    for point in architecture_points:
        doc.add_paragraph(point, style='List Bullet')

    add_heading_with_style(doc, '3.2 Data Flow', 2)
    doc.add_paragraph('The transfer process follows this sequence:')
    flow_steps = [
        '1. Load configuration from YAML file',
        '2. Initialize SharePoint and Azure connections',
        '3. Load transfer state (track previously transferred files)',
        '4. Query SharePoint for TAR files',
        '5. Filter out already-transferred files',
        '6. For each new file, attempt transfer using methods 1-3',
        '7. Update transfer state on success',
        '8. Generate summary report',
    ]
    for step in flow_steps:
        doc.add_paragraph(step, style='List Number')

    doc.add_page_break()

    # 4. Configuration
    add_heading_with_style(doc, '4. Configuration', 1)

    doc.add_paragraph(
        'All settings are stored in a single unified YAML configuration file: '
        'config/tar_transfer_config.yaml'
    )

    add_heading_with_style(doc, '4.1 Configuration Structure', 2)

    config_sections = [
        ('azure_ad', 'Azure Active Directory credentials (tenant_id, client_id, client_secret)'),
        ('sharepoint', 'SharePoint connection details (site_id, drive_id, tar_source_folder_path)'),
        ('azure_blob', 'Azure Blob Storage settings (connection_string, container_name, target_prefix)'),
        ('local_paths', 'Local file paths (temp_download_dir, transfer_state_file)'),
        ('processing', 'Processing options (max_files_per_run, cleanup_after_upload)'),
        ('api', 'API settings (timeout_seconds, retry_attempts, retry_delay_seconds)'),
    ]

    for section, description in config_sections:
        para = doc.add_paragraph()
        para.add_run(f'{section}: ').bold = True
        para.add_run(description)

    add_heading_with_style(doc, '4.2 Example Configuration', 2)
    example_config = '''azure_ad:
  tenant_id: "your-tenant-id"
  client_id: "your-client-id"
  client_secret: "your-client-secret"

sharepoint:
  site_id: "your-site-id"
  drive_id: "your-drive-id"
  tar_source_folder_path: "/path/to/tar/files"

azure_blob:
  connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
  container_name: "tar-files"
  target_prefix: "tar_files/"

local_paths:
  temp_download_dir: "/tmp/tar_downloads"
  transfer_state_file: "config/tar_transfer_state.json"

processing:
  max_files_per_run: 50
  cleanup_after_upload: true'''

    add_code_block(doc, example_config)

    doc.add_page_break()

    # 5. Main Components
    add_heading_with_style(doc, '5. Main Components', 1)

    add_heading_with_style(doc, '5.1 TarTransferConfig Class', 2)
    doc.add_paragraph(
        'Manages all configuration settings for the transfer process. Key responsibilities:'
    )
    config_responsibilities = [
        'Parse and validate YAML configuration',
        'Handle environment variable substitution (${VAR_NAME})',
        'Build Azure connection strings from credentials',
        'Provide easy access to all settings',
    ]
    for resp in config_responsibilities:
        doc.add_paragraph(resp, style='List Bullet')

    add_heading_with_style(doc, '5.2 TarTransferManager Class', 2)
    doc.add_paragraph(
        'Core orchestrator that manages the entire transfer workflow. Key methods:'
    )

    methods = [
        ('__init__', 'Initialize connections to SharePoint and Azure'),
        ('run_transfer', 'Main execution method coordinating the entire process'),
        ('_load_transfer_state', 'Load JSON file tracking previously transferred files'),
        ('_save_transfer_state', 'Persist transfer state to prevent duplicates'),
        ('_get_sharepoint_tar_files', 'Query SharePoint for TAR files'),
        ('_copy_url_to_azure_blob', 'Server-to-server copy (Method 1)'),
        ('_stream_transfer_to_azure_blob', 'Streaming transfer (Method 2)'),
        ('_download_from_sharepoint', 'Download to local disk'),
        ('_upload_to_azure_blob', 'Upload from local disk (Method 3)'),
        ('_cleanup_local_file', 'Remove temporary files'),
    ]

    for method_name, description in methods:
        para = doc.add_paragraph()
        para.add_run(f'{method_name}(): ').bold = True
        para.add_run(description)

    doc.add_page_break()

    # 6. Transfer Methods
    add_heading_with_style(doc, '6. Transfer Methods', 1)

    doc.add_paragraph(
        'The script implements three transfer methods with automatic fallback, '
        'attempting the most efficient method first:'
    )

    add_heading_with_style(doc, 'Method 1: Server-to-Server Copy (Fastest)', 2)
    method1_points = [
        'Uses Azure Blob Storage "Copy from URL" feature',
        'Azure directly copies from SharePoint URL',
        'Zero client resources (no download/upload)',
        'Best for large files and bandwidth-constrained environments',
        'Async operation with polling for completion',
        'Timeout: 1 hour maximum wait time',
    ]
    for point in method1_points:
        doc.add_paragraph(point, style='List Bullet')

    add_code_block(doc, 'blob_client.start_copy_from_url(download_url)')

    add_heading_with_style(doc, 'Method 2: Streaming Transfer (Low Memory)', 2)
    method2_points = [
        'Streams data directly from SharePoint to Azure',
        'Minimal memory footprint (64MB chunks)',
        'No disk I/O required',
        'Progress tracking every 10%',
        'Supports concurrent chunk uploads (4 parallel)',
        'Used when server-to-server copy fails',
    ]
    for point in method2_points:
        doc.add_paragraph(point, style='List Bullet')

    add_code_block(doc, 'response = requests.get(download_url, stream=True)\nblob_client.upload_blob(chunk_generator(), max_concurrency=4)')

    add_heading_with_style(doc, 'Method 3: Download + Upload (Fallback)', 2)
    method3_points = [
        'Downloads file to local temporary directory',
        'Uploads from local disk to Azure Blob',
        'Most reliable but slowest method',
        'Requires disk space for temporary storage',
        'Automatic cleanup after successful upload',
        'Last resort when other methods fail',
    ]
    for point in method3_points:
        doc.add_paragraph(point, style='List Bullet')

    doc.add_page_break()

    # 7. Usage Instructions
    add_heading_with_style(doc, '7. Usage Instructions', 1)

    add_heading_with_style(doc, '7.1 Basic Usage', 2)
    doc.add_paragraph('Run with default configuration:')
    add_code_block(doc, 'python sharepoint_tar_to_azure_blob.py')

    add_heading_with_style(doc, '7.2 Custom Configuration', 2)
    doc.add_paragraph('Specify a custom configuration file:')
    add_code_block(doc, 'python sharepoint_tar_to_azure_blob.py --config /path/to/config.yaml')

    add_heading_with_style(doc, '7.3 Dry Run Mode', 2)
    doc.add_paragraph('Preview transfers without actually executing them:')
    add_code_block(doc, 'python sharepoint_tar_to_azure_blob.py --dry-run')

    add_heading_with_style(doc, '7.4 Command-Line Arguments', 2)
    args = [
        ('--config, -c', 'Path to configuration YAML file (default: config/tar_transfer_config.yaml)'),
        ('--dry-run', 'Show what would be transferred without executing'),
    ]
    for arg, desc in args:
        para = doc.add_paragraph()
        para.add_run(f'{arg}: ').bold = True
        para.add_run(desc)

    # 8. Logging & Monitoring
    add_heading_with_style(doc, '8. Logging & Monitoring', 1)

    doc.add_paragraph(
        'The script provides comprehensive logging to both console and file '
        '(sharepoint_tar_to_blob.log).'
    )

    add_heading_with_style(doc, '8.1 Log Levels', 2)
    log_levels = [
        'INFO: General progress and status updates',
        'WARNING: Non-critical issues (e.g., already-existing blobs)',
        'ERROR: Transfer failures and errors',
        'DEBUG: Detailed diagnostic information',
    ]
    for level in log_levels:
        doc.add_paragraph(level, style='List Bullet')

    add_heading_with_style(doc, '8.2 Key Log Events', 2)
    log_events = [
        'Transfer initiation and configuration summary',
        'File discovery results from SharePoint',
        'Transfer method attempts and results',
        'Progress updates (10% increments for streaming)',
        'Server-to-server copy status polling',
        'Success/failure for each file',
        'Final summary with counts',
    ]
    for event in log_events:
        doc.add_paragraph(event, style='List Bullet')

    add_heading_with_style(doc, '8.3 Transfer State File', 2)
    doc.add_paragraph(
        'The script maintains a JSON state file (config/tar_transfer_state.json) '
        'to track transferred files:'
    )
    state_example = '''{
  "file1.tar": {
    "transferred_at": "2025-11-11T10:30:00",
    "size": 1048576000,
    "sharepoint_modified": "2025-11-10T15:22:33Z",
    "blob_path": "tar_files/file1.tar",
    "transfer_method": "server-to-server"
  }
}'''
    add_code_block(doc, state_example)

    doc.add_page_break()

    # 9. Error Handling
    add_heading_with_style(doc, '9. Error Handling', 1)

    doc.add_paragraph('The script implements robust error handling at multiple levels:')

    add_heading_with_style(doc, '9.1 Connection Errors', 2)
    conn_errors = [
        'SharePoint authentication failures',
        'Azure Blob Storage connection issues',
        'Network timeouts and retries',
    ]
    for error in conn_errors:
        doc.add_paragraph(error, style='List Bullet')

    add_heading_with_style(doc, '9.2 Transfer Errors', 2)
    transfer_errors = [
        'Missing download URLs',
        'Server-to-server copy timeouts',
        'Streaming transfer failures',
        'Disk space issues during download',
    ]
    for error in transfer_errors:
        doc.add_paragraph(error, style='List Bullet')

    add_heading_with_style(doc, '9.3 Retry Logic', 2)
    doc.add_paragraph(
        'The script uses cascading retry mechanisms:'
    )
    retry_logic = [
        'Primary: Three transfer methods with automatic fallback',
        'SharePoint API: Configurable retries (max_retries in config)',
        'Azure operations: Built-in SDK retry policies',
    ]
    for logic in retry_logic:
        doc.add_paragraph(logic, style='List Bullet')

    add_heading_with_style(doc, '9.4 Exit Codes', 2)
    exit_codes = [
        ('0', 'Successful transfer of all files'),
        ('1', 'Transfer completed with errors or failures'),
    ]
    for code, desc in exit_codes:
        para = doc.add_paragraph()
        para.add_run(f'Exit {code}: ').bold = True
        para.add_run(desc)

    # 10. Best Practices
    add_heading_with_style(doc, '10. Best Practices', 1)

    best_practices = [
        ('Test with dry-run first', 'Always validate configuration and file selection before actual transfer'),
        ('Monitor logs', 'Review both console and log file output for issues'),
        ('Start small', 'Use max_files_per_run to limit batch size initially'),
        ('Verify credentials', 'Ensure Azure AD and Blob Storage credentials are current'),
        ('Check disk space', 'Ensure sufficient space in temp_download_dir for Method 3 fallback'),
        ('Schedule appropriately', 'Consider network load and business hours for large transfers'),
        ('Back up state file', 'Preserve tar_transfer_state.json to avoid re-transfers'),
        ('Use environment variables', 'Store sensitive credentials in environment variables, not config files'),
    ]

    for title, desc in best_practices:
        para = doc.add_paragraph()
        para.add_run(f'{title}: ').bold = True
        para.add_run(desc)

    doc.add_page_break()

    # Dependencies
    add_heading_with_style(doc, 'Dependencies', 1)

    doc.add_paragraph('Required Python packages:')
    dependencies = [
        'azure-storage-blob: Azure Blob Storage SDK',
        'azure-core: Azure core libraries',
        'requests: HTTP library for streaming transfers',
        'pyyaml: YAML configuration parsing',
    ]
    for dep in dependencies:
        doc.add_paragraph(dep, style='List Bullet')

    doc.add_paragraph('Required utility modules:')
    utils = [
        'utils.sharepoint_utils: SharePoint authentication and file management',
        'utils.azure_blob_utils: Azure Blob helper functions',
    ]
    for util in utils:
        doc.add_paragraph(util, style='List Bullet')

    # Troubleshooting
    add_heading_with_style(doc, 'Troubleshooting', 1)

    issues = [
        ('No files found in SharePoint', 'Verify tar_source_folder_path in config\nCheck SharePoint permissions\nEnsure files have .tar, .tar.gz, or .tgz extension'),
        ('Authentication failures', 'Verify Azure AD credentials (tenant_id, client_id, client_secret)\nCheck credential expiration\nEnsure app has proper permissions'),
        ('Transfer timeouts', 'Increase timeout_seconds in config\nCheck network connectivity\nVerify SharePoint URLs are accessible'),
        ('Blob already exists errors', 'These are warnings, not errors - file was previously transferred\nCheck transfer_state.json for details'),
        ('All transfer methods fail', 'Review detailed error messages in log file\nVerify network connectivity to both SharePoint and Azure\nCheck disk space for Method 3 fallback'),
    ]

    for issue, solution in issues:
        para = doc.add_paragraph()
        para.add_run(f'{issue}:\n').bold = True
        para.add_run(solution)
        para.paragraph_format.space_after = Pt(12)

    doc.add_page_break()

    # Footer
    footer = doc.sections[0].footer
    footer_para = footer.paragraphs[0]
    footer_para.text = "SharePoint TAR to Azure Blob Transfer - Technical Documentation"
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_para.runs[0].font.size = Pt(9)
    footer_para.runs[0].font.color.rgb = RGBColor(128, 128, 128)

    # Save document
    output_path = '/home/vision_ai_adm/code/oslo/combined_branch/SharePoint_TAR_to_Azure_Blob_Documentation.docx'
    doc.save(output_path)
    print(f"Documentation created successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    create_documentation()
