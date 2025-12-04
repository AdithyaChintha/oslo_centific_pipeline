# TAR Inspection Pipeline Enhancement Proposal
## Adding List-Based Execution Mode

**Date:** 2025-11-13
**Author:** Oslo TAR Inspection Team
**Target File:** `sharepoint_tar_inspector.py`

---

## Executive Summary

This document proposes enhancements to the TAR inspection pipeline to support two execution modes:
1. **Random Mode** (existing): Randomly sample 1 TAR file per N files
2. **List Mode** (new): Process specific TAR files provided via configuration

---

## Current Behavior

The pipeline currently operates in **random sampling mode only**:
- Lists all TAR files from SharePoint `/Uploads` folder
- Randomly selects 1 file per 10 files (configurable via `sampling_ratio`)
- Downloads, extracts, uploads to blob, generates SAS tokens, creates CSV report

---

## Proposed Changes

### 1. Configuration Changes

#### File: `config/tar_inspection_config.yaml`

**Add new section under `inspection:`**

```yaml
inspection:
  # Execution mode: 'random' or 'list'
  # - random: Randomly sample TAR files (uses sampling_ratio)
  # - list: Process specific TAR files (uses tar_file_list)
  execution_mode: "random"  # Options: "random" | "list"

  # Target blob prefix for extracted/untarred files
  untar_blob_prefix: "untar_folder_oslo2/"

  # --- RANDOM MODE SETTINGS ---
  # Sampling ratio: Pick 1 random TAR file per every N files
  sampling_ratio: 10

  # --- LIST MODE SETTINGS ---
  # List of specific TAR filenames to process (only used when execution_mode: "list")
  # Filenames should match exactly as they appear in SharePoint
  tar_file_list:
    - "example_video_batch_01.tar"
    - "example_video_batch_02.tar.gz"
    - "user_uploads_2024_11_12.tar"

  # SAS token expiry in days
  sas_token_expiry_days: 30

  # Local temporary directory for extracting TAR files
  temp_extract_dir: "/data/oslo/tar_extractions"

  # Directory where CSV inspection reports will be saved
  csv_report_dir: "/data/oslo/untar_inspection_reports_"

  # State file to track already-inspected TAR files
  inspection_state_file: "/data/oslo/tar_temp_downloads/tar_inspection_state.json"
```

**Key Changes:**
- Add `execution_mode` field (values: `"random"` or `"list"`)
- Add `tar_file_list` array for list mode
- Organize settings by mode for clarity

---

### 2. Code Changes

#### File: `sharepoint_tar_inspector.py`

---

#### Change 2.1: Update `TarInspectionConfig` Class

**Location:** Lines 75-135

**Modifications:**

```python
class TarInspectionConfig:
    """Configuration class for TAR inspection settings."""

    def __init__(self, config_dict: Dict):
        """Initialize inspection config from dictionary."""
        # SharePoint settings
        self.sharepoint_tar_folder_path = config_dict['sharepoint']['tar_source_folder_path']

        # Azure Blob settings
        self.azure_connection_string = self._get_connection_string(config_dict['azure_blob'])
        self.azure_container_name = config_dict['azure_blob']['container_name']
        self.azure_account_name = self._extract_account_name(self.azure_connection_string)
        self.azure_account_key = self._extract_account_key(self.azure_connection_string)

        # Inspection-specific settings
        inspection_config = config_dict.get('inspection', {})
        self.untar_blob_prefix = inspection_config.get('untar_blob_prefix', 'untar_folder_oslo2/')

        # ✨ NEW: Execution mode
        self.execution_mode = inspection_config.get('execution_mode', 'random').lower()
        if self.execution_mode not in ['random', 'list']:
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}. Must be 'random' or 'list'")

        # Random mode settings
        self.sampling_ratio = inspection_config.get('sampling_ratio', 10)

        # ✨ NEW: List mode settings
        self.tar_file_list = inspection_config.get('tar_file_list', [])
        if self.execution_mode == 'list' and not self.tar_file_list:
            raise ValueError("execution_mode is 'list' but tar_file_list is empty")

        self.sas_token_expiry_days = inspection_config.get('sas_token_expiry_days', 30)

        # Local paths
        self.temp_download_dir = config_dict['local_paths']['temp_download_dir']
        self.temp_extract_dir = inspection_config.get('temp_extract_dir', '/data/oslo/tar_extractions')
        self.csv_report_dir = inspection_config.get('csv_report_dir', '/data/oslo/inspection_reports')
        self.inspection_state_file = inspection_config.get('inspection_state_file',
                                                           '/data/oslo/tar_temp_downloads/tar_inspection_state.json')

    # ... rest of the methods remain unchanged ...
```

**What Changed:**
- Added `execution_mode` validation
- Added `tar_file_list` loading
- Added validation to ensure list mode has filenames
- Kept all existing settings for backward compatibility

---

#### Change 2.2: Add New Method `_select_from_list()`

**Location:** After `_select_random_samples()` method (after line 258)

**Add new method:**

```python
def _select_from_list(self, all_files: List[Dict], already_inspected: set) -> List[Dict]:
    """
    Select specific TAR files from the list provided in config.

    Args:
        all_files: List of all TAR files available in SharePoint
        already_inspected: Set of already inspected filenames

    Returns:
        List of selected files matching the tar_file_list from config
    """
    # Create a lookup dictionary for fast matching
    available_files_dict = {f['name']: f for f in all_files}

    selected_files = []
    missing_files = []
    already_processed_files = []

    for tar_filename in self.config.tar_file_list:
        # Check if already inspected
        if tar_filename in already_inspected:
            logger.info(f"  ⏭️  Skipping (already inspected): {tar_filename}")
            already_processed_files.append(tar_filename)
            continue

        # Check if file exists in SharePoint
        if tar_filename in available_files_dict:
            selected_files.append(available_files_dict[tar_filename])
            logger.info(f"  ✓ Found: {tar_filename}")
        else:
            logger.warning(f"  ✗ NOT FOUND in SharePoint: {tar_filename}")
            missing_files.append(tar_filename)

    # Summary logging
    logger.info(f"\nList Mode Selection Summary:")
    logger.info(f"  - Requested files: {len(self.config.tar_file_list)}")
    logger.info(f"  - Found and selected: {len(selected_files)}")
    logger.info(f"  - Already inspected: {len(already_processed_files)}")
    logger.info(f"  - Missing/Not found: {len(missing_files)}")

    if missing_files:
        logger.warning(f"\nMissing files that will be skipped:")
        for filename in missing_files:
            logger.warning(f"  - {filename}")

    return selected_files
```

**What This Does:**
- Matches filenames from config against available SharePoint files
- Filters out already-inspected files
- Logs missing files that don't exist in SharePoint
- Returns only valid, uninspected files

---

#### Change 2.3: Update `__init__()` Logging

**Location:** Lines 190-194

**Modify:**

```python
logger.info(f"TAR Inspection Manager initialized")
logger.info(f"SharePoint folder: {self.config.sharepoint_tar_folder_path}")
logger.info(f"Azure container: {self.config.azure_container_name}")
logger.info(f"Untar blob prefix: {self.config.untar_blob_prefix}")

# ✨ NEW: Log execution mode
logger.info(f"Execution mode: {self.config.execution_mode.upper()}")
if self.config.execution_mode == 'random':
    logger.info(f"Sampling ratio: 1 per {self.config.sampling_ratio} files")
elif self.config.execution_mode == 'list':
    logger.info(f"Target files: {len(self.config.tar_file_list)} TAR files specified")
```

---

#### Change 2.4: Update `run_inspection()` Method

**Location:** Lines 567-676

**Modify the file selection logic (around lines 593-598):**

```python
# Select random samples
# ✨ MODIFIED: Select files based on execution mode
if self.config.execution_mode == 'random':
    logger.info(f"\n📊 Using RANDOM MODE (1 per {self.config.sampling_ratio} files)")
    selected_files = self._select_random_samples(all_tar_files, already_inspected)
elif self.config.execution_mode == 'list':
    logger.info(f"\n📋 Using LIST MODE ({len(self.config.tar_file_list)} files specified)")
    selected_files = self._select_from_list(all_tar_files, already_inspected)
else:
    logger.error(f"Invalid execution mode: {self.config.execution_mode}")
    return False

if not selected_files:
    logger.info("No files to inspect")
    return True

logger.info(f"\nSelected {len(selected_files)} TAR files for inspection:")
for i, file_info in enumerate(selected_files, 1):
    size_gb = file_info.get('size', 0) / (1024 ** 3)
    logger.info(f"  {i}. {file_info['name']} ({size_gb:.2f} GB)")
```

**What Changed:**
- Added conditional logic to choose between random and list mode
- Kept all existing downstream processing unchanged
- Added clearer logging for mode selection

---

### 3. Usage Examples

#### Example 1: Random Mode (Existing Behavior)

**Config:**
```yaml
inspection:
  execution_mode: "random"
  sampling_ratio: 10
  # tar_file_list is ignored in random mode
```

**Command:**
```bash
python sharepoint_tar_inspector.py --config config/tar_inspection_config.yaml
```

**Behavior:** Randomly samples 1 file per 10 TAR files from SharePoint

---

#### Example 2: List Mode (New Feature)

**Config:**
```yaml
inspection:
  execution_mode: "list"
  tar_file_list:
    - "video_batch_20241101.tar"
    - "video_batch_20241102.tar"
    - "video_batch_20241103.tar.gz"
  # sampling_ratio is ignored in list mode
```

**Command:**
```bash
python sharepoint_tar_inspector.py --config config/tar_inspection_config.yaml
```

**Behavior:** Processes only the 3 specified TAR files

---

#### Example 3: Dry Run with List Mode

**Config:** Same as Example 2

**Command:**
```bash
python sharepoint_tar_inspector.py --config config/tar_inspection_config.yaml --dry-run
```

**Behavior:** Shows which files would be processed without actually downloading/extracting

---

### 4. Backward Compatibility

The changes are **fully backward compatible**:

✅ **Existing configs without `execution_mode`:** Default to `"random"` mode
✅ **Existing random mode behavior:** Unchanged
✅ **Existing command-line arguments:** Still work (`--config`, `--dry-run`)
✅ **State tracking:** Works with both modes
✅ **CSV reports:** Format unchanged

---

### 5. Error Handling

The implementation includes robust error handling:

| Scenario | Behavior |
|----------|----------|
| Invalid `execution_mode` | Raise `ValueError` with helpful message |
| List mode with empty `tar_file_list` | Raise `ValueError` at config load time |
| TAR file in list not found in SharePoint | Log warning, skip file, continue processing others |
| All files in list already inspected | Log message, exit gracefully |
| Mixed mode (some files exist, some don't) | Process available files, log missing ones |

---

### 6. Implementation Checklist

- [ ] Update `config/tar_inspection_config.yaml` with new fields
- [ ] Add `execution_mode` and `tar_file_list` to `TarInspectionConfig.__init__()`
- [ ] Add validation for execution mode and list mode requirements
- [ ] Implement `_select_from_list()` method
- [ ] Update `__init__()` logging to show execution mode
- [ ] Update `run_inspection()` to route to correct selection method
- [ ] Test random mode still works (regression test)
- [ ] Test list mode with valid files
- [ ] Test list mode with missing files
- [ ] Test list mode with already-inspected files
- [ ] Test dry-run with both modes
- [ ] Update documentation/docstrings

---

### 7. Testing Strategy

#### Test Case 1: Random Mode (Regression)
**Config:** `execution_mode: "random"`, `sampling_ratio: 10`
**Expected:** Selects random samples as before

#### Test Case 2: List Mode - All Files Exist
**Config:** `execution_mode: "list"`, list of 3 valid TAR files
**Expected:** Processes exactly those 3 files

#### Test Case 3: List Mode - Some Files Missing
**Config:** `execution_mode: "list"`, list includes 2 valid + 1 non-existent file
**Expected:** Processes 2 valid files, logs warning for missing file

#### Test Case 4: List Mode - All Already Inspected
**Config:** `execution_mode: "list"`, all files in state file
**Expected:** Skips all, exits gracefully

#### Test Case 5: Invalid Mode
**Config:** `execution_mode: "invalid"`
**Expected:** Raises `ValueError` at initialization

#### Test Case 6: List Mode with Empty List
**Config:** `execution_mode: "list"`, `tar_file_list: []`
**Expected:** Raises `ValueError` at initialization

---

### 8. Benefits

✅ **Targeted Inspection:** Process specific problematic TAR files on demand
✅ **Reproducibility:** Re-run inspection on exact same files
✅ **Debugging:** Inspect specific files reported as problematic
✅ **Flexibility:** Support both exploratory (random) and targeted (list) workflows
✅ **No Breaking Changes:** Existing deployments continue working
✅ **Minimal Code Changes:** Reuses 95% of existing pipeline code

---

### 9. Example Workflow Scenarios

#### Scenario A: Weekly Random Quality Check
```yaml
execution_mode: "random"
sampling_ratio: 10
```
Use for ongoing quality monitoring

#### Scenario B: Investigate Specific Issues
```yaml
execution_mode: "list"
tar_file_list:
  - "corrupted_batch_01.tar"
  - "corrupted_batch_02.tar"
```
Use when users report problems with specific files

#### Scenario C: Batch Processing Specific Uploads
```yaml
execution_mode: "list"
tar_file_list:
  - "november_01_uploads.tar"
  - "november_02_uploads.tar"
  - "november_03_uploads.tar"
  - "november_04_uploads.tar"
  - "november_05_uploads.tar"
```
Use for systematic processing of date-range uploads

---

### 10. Code Reuse Summary

| Component | Reuse % | Notes |
|-----------|---------|-------|
| Download logic | 100% | No changes needed |
| Extraction logic | 100% | No changes needed |
| Blob upload logic | 100% | No changes needed |
| SAS token generation | 100% | No changes needed |
| CSV report creation | 100% | No changes needed |
| State tracking | 100% | No changes needed |
| Cleanup logic | 100% | No changes needed |
| File selection | 50% | Add new method, keep existing |
| Configuration | 90% | Add 2 new fields |

**Overall Code Reuse: ~95%**

---

### 11. Summary of Changes

**Configuration File Changes:**
- Add 2 new fields to `config/tar_inspection_config.yaml`

**Python Code Changes:**
- Update 1 class (`TarInspectionConfig.__init__()`) - add ~10 lines
- Add 1 new method (`_select_from_list()`) - ~45 lines
- Update 1 method (`run_inspection()`) - modify ~10 lines
- Update logging in `__init__()` - add ~6 lines

**Total New/Modified Lines:** ~70 lines
**Total Existing Lines:** ~710 lines
**Change Impact:** ~10% of codebase

---

## Conclusion

This proposal adds significant flexibility to the TAR inspection pipeline with minimal code changes and zero breaking changes. The implementation leverages existing infrastructure and maintains the same robust error handling, state tracking, and cleanup mechanisms already in place.

The dual-mode approach supports both exploratory quality checks (random mode) and targeted debugging/processing (list mode), making the pipeline more versatile for production use.
