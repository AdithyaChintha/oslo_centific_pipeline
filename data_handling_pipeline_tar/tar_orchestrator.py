#!/usr/bin/env python3
"""
Tar Orchestrator
Runs all tar file processing tools in sequence to generate comprehensive reports.

This orchestrator executes:
1. tar_file_lister.py           - List all tar files from blob storage
2. tar_upload_pipeline.py       - Upload tar files to OpenAI Partner API
3. report_tar_state.py          - Generate consolidated state reports
4. tar_state_file_comparison.py - Compare tar files with state files
5. sharepoint_tar_inspector.py  - Random inspection of TAR files (extract, upload, generate SAS tokens)

Usage:
    python tar_orchestrator.py --config config/tar_upload_config.yaml
    python tar_orchestrator.py --config config/tar_upload_config.yaml --skip-upload
    python tar_orchestrator.py --config config/tar_upload_config.yaml --skip-inspection
    python tar_orchestrator.py --config config/tar_upload_config.yaml --inspection-count 10
"""

import os
import sys
import argparse
import subprocess
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from azure.storage.blob import BlobServiceClient


class ConfigLoader:
    """Load and parse YAML configuration"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load(config_path)

    def _load(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, *keys: str, default=None):
        node = self.config
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


class TarOrchestrator:
    """Orchestrates all tar file processing and reporting tools"""

    def __init__(self, config_path: str, skip_upload: bool = False,
                 skip_inspection: bool = False, inspection_sample_count: int = None):
        self.config_path = os.path.abspath(config_path)
        self.script_dir = Path(__file__).parent
        self.skip_upload = skip_upload
        self.skip_inspection = skip_inspection

        # Load configuration (unified config for all steps)
        self.cfg = ConfigLoader(self.config_path)
        self.project_id = self.cfg.get("azure_source", "project_id", default="tar-upload")

        # Get inspection sample count from config or command line override
        if inspection_sample_count is not None:
            self.inspection_sample_count = inspection_sample_count
        else:
            self.inspection_sample_count = self.cfg.get("inspection", "csv_sample_count", default=4)

        # Output directories
        self.tar_files_dir = Path("/data/oslo/tar_files")
        self.consolidated_dir = Path("/data/oslo/consolidated_report/tar")
        self.comparison_dir = Path("/data/oslo/comparison_results/tar")
        self.inspection_dir = Path("/data/oslo/inspection_reports/tar")

        # Create output directories
        self.tar_files_dir.mkdir(parents=True, exist_ok=True)
        self.consolidated_dir.mkdir(parents=True, exist_ok=True)
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        self.inspection_dir.mkdir(parents=True, exist_ok=True)

        # Track generated files
        self.generated_files = {
            'tar_listing': None,
            'upload_manifest': None,
            'state_file': None,
            'state_report_json': None,
            'state_report_csv': None,
            'comparison_csv': None,
            'comparison_json': None,
            'inspection_report': None
        }

        # Timestamp for consistent naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def print_header(self, title: str):
        """Print formatted section header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80 + "\n")

    def run_command(self, cmd: list, description: str, stream_output: bool = False) -> tuple:
        """Run a command and return success status and output

        Args:
            cmd: Command to run as list
            description: Description of the command
            stream_output: If True, stream output in real-time instead of capturing
        """
        print(f"🔧 {description}")
        print(f"   Command: {' '.join(cmd)}")

        if stream_output:
            # Stream output in real-time (for long-running commands like inspection)
            try:
                print()  # Add newline before streaming output
                result = subprocess.run(
                    cmd,
                    cwd=str(self.script_dir),
                    check=True
                    # No capture_output - output goes directly to console
                )
                print(f"\n✅ {description} - SUCCESS")
                return True, ""
            except subprocess.CalledProcessError as e:
                print(f"\n❌ {description} - FAILED (exit code: {e.returncode})")
                return False, ""
        else:
            # Capture output (for quick commands)
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.script_dir),
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✅ {description} - SUCCESS")
                return True, result.stdout
            except subprocess.CalledProcessError as e:
                print(f"❌ {description} - FAILED")
                print(f"   Error: {e.stderr}")
                return False, e.stderr

    def _copy_state_file_to_data(self):
        """Copy state file from Azure to /data folder"""
        try:
            # Get state storage configuration
            state_conn = self.cfg.get("state_storage", "connection_string")
            state_container = self.cfg.get("state_storage", "container_name")
            state_prefix = self.cfg.get("state_storage", "prefix", default="")

            if not state_conn or not state_container:
                print("⚠️  State storage not configured, skipping state file copy")
                return

            # Build state file blob name
            state_blob_name = f"{state_prefix}{self.project_id}_state.json"

            # Download from Azure
            blob_service_client = BlobServiceClient.from_connection_string(state_conn)
            container_client = blob_service_client.get_container_client(state_container)
            blob_client = container_client.get_blob_client(state_blob_name)

            # Create destination directory
            dest_dir = Path("/data/oslo/upload_state/tar")
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Download state file
            dest_file = dest_dir / f"{self.project_id}_state.json"
            with open(dest_file, "wb") as f:
                stream = blob_client.download_blob()
                f.write(stream.readall())

            self.generated_files['state_file'] = str(dest_file)
            print(f"📄 State file saved to: {dest_file}")

        except Exception as e:
            print(f"⚠️  Warning: Could not copy state file: {e}")

    def step1_list_tar_files(self) -> bool:
        """Step 1: List all tar files from blob storage"""
        self.print_header("STEP 1: List Tar Files from Blob Storage")

        output_file = self.tar_files_dir / f"tar_files_{self.project_id}_{self.timestamp}.csv"

        cmd = [
            sys.executable,
            "tar_file_lister.py",
            "--config", self.config_path,
            "--output", str(output_file)
        ]

        success, output = self.run_command(cmd, "Listing tar files")

        if success and output_file.exists():
            self.generated_files['tar_listing'] = str(output_file)
            print(f"📄 Tar listing saved to: {output_file}")
            return True
        else:
            print(f"⚠️  Warning: Tar listing file not found at {output_file}")
            return False

    def step2_upload_tar_files(self) -> bool:
        """Step 2: Upload tar files to OpenAI Partner API"""
        self.print_header("STEP 2: Upload Tar Files to OpenAI Partner API")

        if self.skip_upload:
            print("⏭️  Skipping upload step (--skip-upload flag set)")
            return True

        cmd = [
            sys.executable,
            "tar_upload_pipeline.py",
            "--config", self.config_path
        ]

        # Use stream_output=True to show real-time logs during upload (azcopy, etc.)
        success, output = self.run_command(cmd, "Uploading tar files", stream_output=True)

        if success:
            # Print upload output (contains summary)
            print("\n" + "="*60)
            print("UPLOAD SUMMARY:")
            print("="*60)
            print(output)
            print("="*60 + "\n")

            # Copy state file to /data folder
            self._copy_state_file_to_data()

            print("✅ Upload completed successfully")
            return True
        else:
            print("❌ Upload failed - continuing with existing state data")
            return False

    def step3_generate_state_report(self) -> bool:
        """Step 3: Generate consolidated state reports"""
        self.print_header("STEP 3: Generate Consolidated State Reports")

        cmd = [
            sys.executable,
            "report_tar_state.py",
            "--config", self.config_path
        ]

        success, output = self.run_command(cmd, "Generating state reports")

        if success:
            # Find the generated files (most recent in the directory)
            json_files = sorted(self.consolidated_dir.glob(f"consolidated_tar_state_report_{self.project_id}_*.json"))
            csv_files = sorted(self.consolidated_dir.glob(f"consolidated_tar_files_{self.project_id}_*.csv"))

            if json_files:
                self.generated_files['state_report_json'] = str(json_files[-1])
                print(f"📄 State report JSON: {json_files[-1]}")

            if csv_files:
                self.generated_files['state_report_csv'] = str(csv_files[-1])
                print(f"📄 State report CSV: {csv_files[-1]}")

            return True
        else:
            print("⚠️  Warning: State report generation failed")
            return False

    def step4_compare_files(self) -> bool:
        """Step 4: Compare tar files with state files"""
        self.print_header("STEP 4: Compare Tar Files with State Files")

        # Check if we have the required input files
        tar_listing = self.generated_files.get('tar_listing')
        state_csv = self.generated_files.get('state_report_csv')

        if not tar_listing:
            print("❌ Tar listing file not available - cannot perform comparison")
            return False

        if not state_csv:
            print("❌ State report CSV not available - cannot perform comparison")
            return False

        cmd = [
            sys.executable,
            "tar_state_file_comparison.py",
            "--tar", tar_listing,
            "--state", state_csv,
            "--output", str(self.comparison_dir),
            "--project-id", self.project_id
        ]

        success, output = self.run_command(cmd, "Comparing files")

        if success:
            # Find the generated comparison files
            csv_files = sorted(self.comparison_dir.glob(f"*_{self.project_id}_tar_state_comparison.csv"))
            json_files = sorted(self.comparison_dir.glob(f"*_{self.project_id}_tar_state_comparison.json"))

            if csv_files:
                self.generated_files['comparison_csv'] = str(csv_files[-1])
                print(f"📄 Comparison CSV: {csv_files[-1]}")

            if json_files:
                self.generated_files['comparison_json'] = str(json_files[-1])
                print(f"📄 Comparison JSON: {json_files[-1]}")

            return True
        else:
            print("⚠️  Warning: File comparison failed")
            return False

    def step5_run_inspection(self) -> bool:
        """Step 5: Run TAR inspection on random samples from comparison CSV"""
        self.print_header("STEP 5: TAR Inspection (Random Sampling from Comparison CSV)")

        if self.skip_inspection:
            print("⏭️  Skipping inspection step (--skip-inspection flag set)")
            return True

        # Check if we have the required comparison CSV
        comparison_csv = self.generated_files.get('comparison_csv')
        if not comparison_csv:
            print("❌ Comparison CSV not available - cannot run inspection")
            print("   (Step 4 must complete successfully first)")
            return False

        if not os.path.exists(comparison_csv):
            print(f"❌ Comparison CSV file not found: {comparison_csv}")
            return False

        print(f"📄 Using comparison CSV: {comparison_csv}")
        print(f"📄 Using config: {self.config_path}")
        print(f"🎲 Random sample count: {self.inspection_sample_count}")

        # Build temporary config with CSV mode settings
        # We use the same unified config file and just override inspection settings
        try:
            import yaml

            # Load the unified config (same config used for all steps)
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Override settings for CSV mode inspection
            if 'inspection' not in config:
                config['inspection'] = {}
            config['inspection']['execution_mode'] = 'csv'
            config['inspection']['csv_input_file'] = comparison_csv
            config['inspection']['csv_sample_count'] = self.inspection_sample_count
            config['inspection']['csv_report_dir'] = str(self.inspection_dir)

            # Write temporary config file
            temp_config_path = self.inspection_dir / f"temp_inspection_config_{self.timestamp}.yaml"
            with open(temp_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            print(f"📝 Generated temporary config: {temp_config_path}")

            # Path to the inspector script (in combined_branch folder)
            inspector_script = self.script_dir.parent / "sharepoint_tar_inspector.py"

            if not inspector_script.exists():
                print(f"❌ Inspector script not found: {inspector_script}")
                return False

            cmd = [
                sys.executable,
                str(inspector_script),
                "--config", str(temp_config_path)
            ]

            # Use stream_output=True to show real-time logs during inspection
            success, output = self.run_command(cmd, "Running TAR inspection", stream_output=True)

            if success:
                # Find the generated inspection report (most recent in the directory)
                inspection_reports = sorted(self.inspection_dir.glob("tar_inspection_report_*.csv"))

                if inspection_reports:
                    self.generated_files['inspection_report'] = str(inspection_reports[-1])
                    print(f"📄 Inspection report: {inspection_reports[-1]}")

                # Cleanup temporary config
                try:
                    os.remove(temp_config_path)
                except Exception:
                    pass

                return True
            else:
                print("⚠️  Warning: TAR inspection failed")
                return False

        except Exception as e:
            print(f"❌ Error running inspection: {e}")
            import traceback
            traceback.print_exc()
            return False

    def print_summary(self):
        """Print final summary of all generated files"""
        self.print_header("ORCHESTRATION COMPLETE - SUMMARY")

        print("📊 Generated Files:")
        print()

        if self.generated_files['tar_listing']:
            print(f"1️⃣  Tar File Listing:")
            print(f"   📄 {self.generated_files['tar_listing']}")
            print()

        if not self.skip_upload:
            print(f"2️⃣  Upload Pipeline:")
            print(f"   ✅ Upload step completed")
            if self.generated_files['state_file']:
                print(f"   📄 State file: {self.generated_files['state_file']}")
            print()
        else:
            print(f"2️⃣  Upload Pipeline:")
            print(f"   ⏭️  Skipped")
            print()

        if self.generated_files['state_report_json'] or self.generated_files['state_report_csv']:
            print(f"3️⃣  State Reports:")
            if self.generated_files['state_report_json']:
                print(f"   📄 JSON: {self.generated_files['state_report_json']}")
            if self.generated_files['state_report_csv']:
                print(f"   📄 CSV: {self.generated_files['state_report_csv']}")
            print()

        if self.generated_files['comparison_csv'] or self.generated_files['comparison_json']:
            print(f"4️⃣  Comparison Reports:")
            if self.generated_files['comparison_csv']:
                print(f"   📄 CSV: {self.generated_files['comparison_csv']}")
            if self.generated_files['comparison_json']:
                print(f"   📄 JSON: {self.generated_files['comparison_json']}")
            print()

        if not self.skip_inspection:
            print(f"5️⃣  TAR Inspection:")
            if self.generated_files['inspection_report']:
                print(f"   ✅ Inspection completed")
                print(f"   📄 Report: {self.generated_files['inspection_report']}")
            else:
                print(f"   ⚠️  No inspection report generated")
            print()
        else:
            print(f"5️⃣  TAR Inspection:")
            print(f"   ⏭️  Skipped")
            print()

        print("="*80)
        print(f"🎉 All reports generated successfully for project: {self.project_id}")
        print("="*80)

    def run(self):
        """Execute all steps in sequence"""
        print("\n" + "="*80)
        print("  TAR FILE ORCHESTRATOR")
        print("="*80)
        print(f"Project ID: {self.project_id}")
        print(f"Config: {self.config_path}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Skip Upload: {self.skip_upload}")
        print(f"Skip Inspection: {self.skip_inspection}")
        if not self.skip_inspection:
            print(f"Inspection Sample Count: {self.inspection_sample_count}")
        print("="*80)

        # Track success of each step
        results = {
            'step1': False,
            'step2': False,
            'step3': False,
            'step4': False,
            'step5': False
        }

        # Step 1: List tar files
        results['step1'] = self.step1_list_tar_files()

        # Step 2: Upload tar files (optional)
        results['step2'] = self.step2_upload_tar_files()

        # Step 3: Generate state report
        results['step3'] = self.step3_generate_state_report()

        # Step 4: Compare files
        results['step4'] = self.step4_compare_files()

        # Step 5: Run TAR inspection (optional)
        results['step5'] = self.step5_run_inspection()

        # Print summary
        self.print_summary()

        # Return overall success
        critical_steps = ['step1', 'step3', 'step4']
        if self.skip_upload:
            critical_steps.remove('step3')  # State report might not be updated

        all_critical_passed = all(results.get(step, False) for step in critical_steps)

        if all_critical_passed:
            print("\n✅ All critical steps completed successfully!")
            return 0
        else:
            print("\n⚠️  Some steps failed. Check the output above for details.")
            return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Orchestrate all tar file processing and reporting tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This orchestrator runs the following tools in sequence:
  1. tar_file_lister.py           - List all tar files from blob storage
  2. tar_upload_pipeline.py       - Upload tar files to OpenAI Partner API
  3. report_tar_state.py          - Generate consolidated state reports
  4. tar_state_file_comparison.py - Compare tar files with state files
  5. sharepoint_tar_inspector.py  - Random inspection of TAR files (extract, upload, generate SAS)

Examples:
  # Run all steps with unified config
  python tar_orchestrator.py --config config/tar_pipeline_config.yaml

  # Skip upload step (use existing state data)
  python tar_orchestrator.py --config config/tar_pipeline_config.yaml --skip-upload

  # Skip inspection step
  python tar_orchestrator.py --config config/tar_pipeline_config.yaml --skip-inspection

  # Custom inspection sample count (inspect 10 random TAR files)
  python tar_orchestrator.py --config config/tar_pipeline_config.yaml --inspection-count 10

  # Skip both upload and inspection
  python tar_orchestrator.py --config config/tar_pipeline_config.yaml --skip-upload --skip-inspection
        """
    )

    parser.add_argument(
        '--config', '-c',
        default='config/tar_pipeline_config.yaml',
        help='Path to unified config YAML (default: config/tar_pipeline_config.yaml)'
    )

    parser.add_argument(
        '--skip-upload', '-s',
        action='store_true',
        help='Skip the upload step (step 2) and only generate reports'
    )

    parser.add_argument(
        '--skip-inspection',
        action='store_true',
        help='Skip the TAR inspection step (step 5)'
    )

    parser.add_argument(
        '--inspection-count', '-n',
        type=int,
        default=None,
        help='Number of random TAR files to inspect (default: from config or 4)'
    )

    args = parser.parse_args()

    try:
        # Resolve config path
        config_path = args.config
        if not os.path.isabs(config_path):
            script_dir = Path(__file__).parent
            config_path = str(script_dir / config_path)

        if not os.path.exists(config_path):
            print(f"❌ Error: Config file not found: {config_path}")
            return 1

        # Create and run orchestrator
        orchestrator = TarOrchestrator(
            config_path,
            skip_upload=args.skip_upload,
            skip_inspection=args.skip_inspection,
            inspection_sample_count=args.inspection_count
        )
        return orchestrator.run()

    except KeyboardInterrupt:
        print("\n\n⚠️  Orchestration interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Orchestration failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
