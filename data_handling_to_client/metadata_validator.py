#!/usr/bin/env python3
"""
Metadata JSON Validator for Individual Home IDs

This script validates metadata JSON files for a specific home ID and identifies:
1. Session IDs with improper JSON
2. Files being skipped due to missing or invalid metadata
3. Detailed validation results in JSON format

Usage:
    python metadata_validator.py --home-id 63 --config config.yaml
    python metadata_validator.py --home-id 63 --input-dir /path/to/state/files
"""

import os
import sys
import json
import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import quote

import yaml
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, AzureError

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("metadata_validator")

class MetadataValidator:
    """Validates metadata JSON files for a specific home ID"""
    
    def __init__(self, home_id: str, connection_string: str, container_name: str, source_prefix: str = ""):
        self.home_id = home_id
        self.connection_string = connection_string
        self.container_name = container_name
        self.source_prefix = source_prefix
        self._client = BlobServiceClient.from_connection_string(connection_string)
        self._container_client = self._client.get_container_client(container_name)
        
        # Validation results
        self.validation_results = {
            "home_id": home_id,
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_media_files": 0,
            "files_with_metadata": 0,
            "files_without_metadata": 0,
            "invalid_metadata_files": 0,
            "session_analysis": {},
            "problematic_sessions": [],
            "skipped_files": [],
            "validation_summary": {}
        }
    
    def extract_metadata_pattern_info(self, media_blob_name: str) -> Optional[dict]:
        """
        Extract pattern info from media file to find corresponding metadata file.
        
        Media: {id}_{type}{num}_{activity}_{timestamp}_{seq}_{type}.{ext}
        Metadata: {id}_activity_{activity}_{timestamp}_metadata.json
        """
        # Extract filename from path
        path_parts = media_blob_name.split('/')
        if len(path_parts) < 2:
            return None

        filename = path_parts[-1]
        folder_path = '/'.join(path_parts[:-1])

        # Pattern for media files - handles various formats
        media_pattern = r'^([^_]+_[^_]+)_([^_]+)_([^_]+)_([^_]+)_\d+_(video|audio)\.(insv|wav|WAV|INSV)$'

        match = re.match(media_pattern, filename)
        if not match:
            return None

        id_part = match.group(1)  # "67_3638d12d-e041-4514-adaa-3a3055068f3c"
        device_id = match.group(2)  # "video1", "audio1", "C30024373"
        activity = match.group(3)  # "locking-doors-and-windows"
        timestamp = match.group(4)  # "20250925094000"

        return {
            'folder_path': folder_path,
            'id_part': id_part,
            'activity': activity,
            'timestamp': timestamp,
            'device_id': device_id
        }

    def get_metadata_path_for_media(self, media_blob_name: str) -> Optional[str]:
        """Get the expected metadata file path for a given media file."""
        info = self.extract_metadata_pattern_info(media_blob_name)
        if not info:
            return None

        # Build metadata filename: {id}_activity_{activity}_{timestamp}_metadata.json
        metadata_filename = f"{info['id_part']}_activity_{info['activity']}_{info['timestamp']}_metadata.json"
        metadata_path = f"{info['folder_path']}/{metadata_filename}"

        return metadata_path

    def metadata_blob_exists(self, metadata_blob_name: str) -> bool:
        """Check if a metadata blob exists in the container."""
        try:
            self._container_client.get_blob_client(metadata_blob_name).get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except AzureError:
            return False

    def download_metadata_json(self, metadata_blob_name: str) -> Optional[dict]:
        """Download and parse metadata JSON from blob."""
        try:
            blob_client = self._container_client.get_blob_client(metadata_blob_name)
            stream = blob_client.download_blob()
            content = stream.content_as_text(encoding="utf-8")
            return json.loads(content)
        except (ResourceNotFoundError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to download/parse metadata {metadata_blob_name}: {e}")
            return None

    def validate_metadata_json(self, metadata_content: dict) -> Tuple[bool, List[str]]:
        """
        Validate metadata JSON structure and content.
        Returns (is_valid, list_of_issues)
        """
        issues = []
        
        # Check required fields
        required_fields = [
            "session_id", "home_id", "activity", "timestamp", 
            "device_id", "file_info", "processing_info"
        ]
        
        for field in required_fields:
            if field not in metadata_content:
                issues.append(f"Missing required field: {field}")
        
        # Validate file_info structure
        if "file_info" in metadata_content:
            file_info = metadata_content["file_info"]
            if not isinstance(file_info, dict):
                issues.append("file_info must be a dictionary")
            else:
                file_required = ["file_size", "duration", "format", "codec"]
                for field in file_required:
                    if field not in file_info:
                        issues.append(f"Missing file_info field: {field}")
        
        # Validate processing_info structure
        if "processing_info" in metadata_content:
            proc_info = metadata_content["processing_info"]
            if not isinstance(proc_info, dict):
                issues.append("processing_info must be a dictionary")
            else:
                proc_required = ["processed_at", "processing_duration", "status"]
                for field in proc_required:
                    if field not in proc_info:
                        issues.append(f"Missing processing_info field: {field}")
        
        # Validate data types
        if "session_id" in metadata_content and not isinstance(metadata_content["session_id"], str):
            issues.append("session_id must be a string")
        
        if "home_id" in metadata_content and not isinstance(metadata_content["home_id"], str):
            issues.append("home_id must be a string")
        
        if "timestamp" in metadata_content and not isinstance(metadata_content["timestamp"], str):
            issues.append("timestamp must be a string")
        
        return len(issues) == 0, issues

    def extract_session_id_from_metadata_path(self, metadata_path: str) -> Optional[str]:
        """Extract session ID from metadata file path."""
        # Pattern: {id}_activity_{activity}_{timestamp}_metadata.json
        # Session ID is typically the timestamp part
        match = re.search(r'_activity_[^_]+_([^_]+)_metadata\.json$', metadata_path)
        if match:
            return match.group(1)
        return None

    def get_media_files_for_home_id(self) -> List[dict]:
        """Get all media files for the specified home ID."""
        media_files = []
        
        try:
            # List all blobs with the source prefix
            blobs = self._container_client.list_blobs(name_starts_with=self.source_prefix)
            
            for blob in blobs:
                blob_name = blob.name
                
                # Check if this is a media file for our home ID
                if self.is_target_media_file(blob_name):
                    media_files.append({
                        "blob_name": blob_name,
                        "size": blob.size,
                        "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                        "etag": blob.etag
                    })
        
        except AzureError as e:
            logger.error(f"Failed to list blobs: {e}")
            raise
        
        return media_files

    def is_target_media_file(self, blob_path: str) -> bool:
        """Check if blob is a target media file for our home ID."""
        parts = blob_path.split("/")
        if len(parts) < 2:
            return False
        
        filename = parts[-1]
        parent_folders = parts[:-1]
        
        # Check if any parent folder starts with our home ID
        has_home = any(p.startswith(self.home_id) for p in parent_folders)
        
        # Check if it's a media file
        is_media = filename.lower().endswith((".insv", ".wav"))
        
        return has_home and is_media

    def validate_home_id_metadata(self) -> dict:
        """Main validation function for the home ID."""
        logger.info(f"Starting metadata validation for home ID: {self.home_id}")
        
        # Get all media files for this home ID
        media_files = self.get_media_files_for_home_id()
        self.validation_results["total_media_files"] = len(media_files)
        
        logger.info(f"Found {len(media_files)} media files for home ID {self.home_id}")
        
        # Track session IDs and their issues
        session_issues = {}
        files_with_metadata = 0
        files_without_metadata = 0
        invalid_metadata_count = 0
        skipped_files = []
        
        for media_file in media_files:
            blob_name = media_file["blob_name"]
            
            # Get expected metadata path
            metadata_path = self.get_metadata_path_for_media(blob_name)
            
            if not metadata_path:
                logger.warning(f"Could not determine metadata path for {blob_name}")
                skipped_files.append({
                    "media_file": blob_name,
                    "reason": "Could not determine metadata path",
                    "session_id": None
                })
                continue
            
            # Check if metadata file exists
            if not self.metadata_blob_exists(metadata_path):
                logger.warning(f"Missing metadata for {blob_name}")
                files_without_metadata += 1
                session_id = self.extract_session_id_from_metadata_path(metadata_path)
                skipped_files.append({
                    "media_file": blob_name,
                    "metadata_path": metadata_path,
                    "reason": "Metadata file does not exist",
                    "session_id": session_id
                })
                
                # Track session issues
                if session_id:
                    if session_id not in session_issues:
                        session_issues[session_id] = {
                            "session_id": session_id,
                            "issues": [],
                            "affected_files": []
                        }
                    session_issues[session_id]["issues"].append("Missing metadata file")
                    session_issues[session_id]["affected_files"].append(blob_name)
                continue
            
            # Download and validate metadata JSON
            metadata_content = self.download_metadata_json(metadata_path)
            
            if metadata_content is None:
                logger.error(f"Failed to download/parse metadata for {blob_name}")
                invalid_metadata_count += 1
                session_id = self.extract_session_id_from_metadata_path(metadata_path)
                skipped_files.append({
                    "media_file": blob_name,
                    "metadata_path": metadata_path,
                    "reason": "Failed to download/parse metadata JSON",
                    "session_id": session_id
                })
                
                # Track session issues
                if session_id:
                    if session_id not in session_issues:
                        session_issues[session_id] = {
                            "session_id": session_id,
                            "issues": [],
                            "affected_files": []
                        }
                    session_issues[session_id]["issues"].append("Invalid metadata JSON")
                    session_issues[session_id]["affected_files"].append(blob_name)
                continue
            
            # Validate metadata structure
            is_valid, issues = self.validate_metadata_json(metadata_content)
            
            if not is_valid:
                logger.error(f"Invalid metadata structure for {blob_name}: {issues}")
                invalid_metadata_count += 1
                session_id = metadata_content.get("session_id") or self.extract_session_id_from_metadata_path(metadata_path)
                skipped_files.append({
                    "media_file": blob_name,
                    "metadata_path": metadata_path,
                    "reason": f"Invalid metadata structure: {', '.join(issues)}",
                    "session_id": session_id
                })
                
                # Track session issues
                if session_id:
                    if session_id not in session_issues:
                        session_issues[session_id] = {
                            "session_id": session_id,
                            "issues": [],
                            "affected_files": []
                        }
                    session_issues[session_id]["issues"].extend(issues)
                    session_issues[session_id]["affected_files"].append(blob_name)
            else:
                files_with_metadata += 1
                logger.debug(f"Valid metadata for {blob_name}")
        
        # Update validation results
        self.validation_results["files_with_metadata"] = files_with_metadata
        self.validation_results["files_without_metadata"] = files_without_metadata
        self.validation_results["invalid_metadata_files"] = invalid_metadata_count
        self.validation_results["session_analysis"] = session_issues
        self.validation_results["skipped_files"] = skipped_files
        
        # Identify problematic sessions
        problematic_sessions = []
        for session_id, session_data in session_issues.items():
            if session_data["issues"]:
                problematic_sessions.append({
                    "session_id": session_id,
                    "issue_count": len(session_data["issues"]),
                    "affected_files_count": len(session_data["affected_files"]),
                    "issues": session_data["issues"],
                    "affected_files": session_data["affected_files"]
                })
        
        self.validation_results["problematic_sessions"] = problematic_sessions
        
        # Create validation summary
        self.validation_results["validation_summary"] = {
            "total_sessions_analyzed": len(session_issues),
            "problematic_sessions_count": len(problematic_sessions),
            "success_rate_percentage": round((files_with_metadata / len(media_files)) * 100, 2) if media_files else 0,
            "most_common_issue": self._get_most_common_issue(session_issues)
        }
        
        logger.info(f"Validation complete for home ID {self.home_id}")
        logger.info(f"Total files: {len(media_files)}")
        logger.info(f"Files with valid metadata: {files_with_metadata}")
        logger.info(f"Files without metadata: {files_without_metadata}")
        logger.info(f"Files with invalid metadata: {invalid_metadata_count}")
        logger.info(f"Problematic sessions: {len(problematic_sessions)}")
        
        return self.validation_results

    def _get_most_common_issue(self, session_issues: dict) -> str:
        """Get the most common issue across all sessions."""
        issue_counts = {}
        for session_data in session_issues.values():
            for issue in session_data["issues"]:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        if not issue_counts:
            return "No issues found"
        
        return max(issue_counts.items(), key=lambda x: x[1])[0]

    def save_results_to_file(self, output_path: str) -> None:
        """Save validation results to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Validation results saved to: {output_path}")

    def print_summary(self) -> None:
        """Print a summary of validation results."""
        summary = self.validation_results["validation_summary"]
        
        print(f"\n{'='*60}")
        print(f"METADATA VALIDATION SUMMARY FOR HOME ID: {self.home_id}")
        print(f"{'='*60}")
        print(f"Total media files: {self.validation_results['total_media_files']}")
        print(f"Files with valid metadata: {self.validation_results['files_with_metadata']}")
        print(f"Files without metadata: {self.validation_results['files_without_metadata']}")
        print(f"Files with invalid metadata: {self.validation_results['invalid_metadata_files']}")
        print(f"Success rate: {summary['success_rate_percentage']}%")
        print(f"Total sessions analyzed: {summary['total_sessions_analyzed']}")
        print(f"Problematic sessions: {summary['problematic_sessions_count']}")
        print(f"Most common issue: {summary['most_common_issue']}")
        
        if self.validation_results["problematic_sessions"]:
            print(f"\nPROBLEMATIC SESSIONS:")
            for session in self.validation_results["problematic_sessions"]:
                print(f"  Session ID: {session['session_id']}")
                print(f"    Issues: {session['issue_count']}")
                print(f"    Affected files: {session['affected_files_count']}")
                print(f"    Issues: {', '.join(session['issues'])}")
                print()
        
        print(f"{'='*60}\n")

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Validate metadata JSON files for a specific home ID")
    parser.add_argument("--home-id", required=True, help="Home ID to validate")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--connection-string", help="Azure connection string")
    parser.add_argument("--container-name", help="Azure container name")
    parser.add_argument("--source-prefix", default="", help="Source prefix for blob names")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        connection_string = args.connection_string or config.get("azure_source", {}).get("connection_string")
        container_name = args.container_name or config.get("azure_source", {}).get("container_name")
        source_prefix = args.source_prefix or config.get("azure_source", {}).get("source_prefix", "")
    else:
        connection_string = args.connection_string
        container_name = args.container_name
        source_prefix = args.source_prefix or ""
    
    if not connection_string or not container_name:
        logger.error("Connection string and container name are required")
        sys.exit(1)
    
    # Create validator and run validation
    validator = MetadataValidator(
        home_id=args.home_id,
        connection_string=connection_string,
        container_name=container_name,
        source_prefix=source_prefix
    )
    
    try:
        results = validator.validate_home_id_metadata()
        
        # Print summary
        validator.print_summary()
        
        # Save results if output path specified
        if args.output:
            validator.save_results_to_file(args.output)
        else:
            # Default output path
            output_path = f"metadata_validation_{args.home_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            validator.save_results_to_file(output_path)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
