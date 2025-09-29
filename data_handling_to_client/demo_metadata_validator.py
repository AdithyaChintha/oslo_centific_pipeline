#!/usr/bin/env python3
"""
Demo script showing how the metadata validator works
This version works without Azure credentials for demonstration purposes
"""

import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

class DemoMetadataValidator:
    """Demo version of the metadata validator that works without Azure"""
    
    def __init__(self, home_id: str):
        self.home_id = home_id
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
        """Extract pattern info from media file to find corresponding metadata file."""
        path_parts = media_blob_name.split('/')
        if len(path_parts) < 2:
            return None

        filename = path_parts[-1]
        folder_path = '/'.join(path_parts[:-1])

        # Pattern for media files
        media_pattern = r'^([^_]+_[^_]+)_([^_]+)_([^_]+)_([^_]+)_\d+_(video|audio)\.(insv|wav|WAV|INSV)$'

        match = re.match(media_pattern, filename)
        if not match:
            return None

        id_part = match.group(1)
        device_id = match.group(2)
        activity = match.group(3)
        timestamp = match.group(4)

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

        metadata_filename = f"{info['id_part']}_activity_{info['activity']}_{info['timestamp']}_metadata.json"
        metadata_path = f"{info['folder_path']}/{metadata_filename}"
        return metadata_path

    def validate_metadata_json(self, metadata_content: dict) -> Tuple[bool, List[str]]:
        """Validate metadata JSON structure and content."""
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
        
        return len(issues) == 0, issues

    def extract_session_id_from_metadata_path(self, metadata_path: str) -> Optional[str]:
        """Extract session ID from metadata file path."""
        match = re.search(r'_activity_[^_]+_([^_]+)_metadata\.json$', metadata_path)
        if match:
            return match.group(1)
        return None

    def demo_validation(self, sample_media_files: List[str], sample_metadata: Dict[str, dict]) -> dict:
        """Demo validation with sample data"""
        print(f"🔍 Starting metadata validation demo for Home ID: {self.home_id}")
        print(f"📁 Processing {len(sample_media_files)} sample media files...")
        print("-" * 60)
        
        self.validation_results["total_media_files"] = len(sample_media_files)
        
        session_issues = {}
        files_with_metadata = 0
        files_without_metadata = 0
        invalid_metadata_count = 0
        skipped_files = []
        
        for media_file in sample_media_files:
            print(f"📄 Processing: {media_file}")
            
            # Get expected metadata path
            metadata_path = self.get_metadata_path_for_media(media_file)
            
            if not metadata_path:
                print(f"   ❌ Could not determine metadata path")
                skipped_files.append({
                    "media_file": media_file,
                    "reason": "Could not determine metadata path",
                    "session_id": None
                })
                continue
            
            print(f"   📋 Expected metadata: {metadata_path}")
            
            # Check if metadata exists in our sample data
            if metadata_path not in sample_metadata:
                print(f"   ❌ Missing metadata file")
                files_without_metadata += 1
                session_id = self.extract_session_id_from_metadata_path(metadata_path)
                skipped_files.append({
                    "media_file": media_file,
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
                    session_issues[session_id]["affected_files"].append(media_file)
                continue
            
            # Validate metadata JSON
            metadata_content = sample_metadata[metadata_path]
            is_valid, issues = self.validate_metadata_json(metadata_content)
            
            if not is_valid:
                print(f"   ❌ Invalid metadata structure: {', '.join(issues)}")
                invalid_metadata_count += 1
                session_id = metadata_content.get("session_id") or self.extract_session_id_from_metadata_path(metadata_path)
                skipped_files.append({
                    "media_file": media_file,
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
                    session_issues[session_id]["affected_files"].append(media_file)
            else:
                print(f"   ✅ Valid metadata")
                files_with_metadata += 1
        
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
            "success_rate_percentage": round((files_with_metadata / len(sample_media_files)) * 100, 2) if sample_media_files else 0,
            "most_common_issue": self._get_most_common_issue(session_issues)
        }
        
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
            print(f"\n🚨 PROBLEMATIC SESSIONS:")
            for session in self.validation_results["problematic_sessions"]:
                print(f"  Session ID: {session['session_id']}")
                print(f"    Issues: {session['issue_count']}")
                print(f"    Affected files: {session['affected_files_count']}")
                print(f"    Issues: {', '.join(session['issues'])}")
                print()
        
        print(f"{'='*60}\n")

def main():
    """Demo the metadata validator with sample data"""
    
    # Sample media files (based on real data patterns)
    sample_media_files = [
        "one-data-platform/63-0347e6fd-ade1-46c2-ba07-bb0419c99a90-bedroom-movement/63_0347e6fd-ade1-46c2-ba07-bb0419c99a90_audio1_bedroom-movement_20250925115500_1_audio.wav",
        "one-data-platform/63-0347e6fd-ade1-46c2-ba07-bb0419c99a90-bedroom-movement/63_0347e6fd-ade1-46c2-ba07-bb0419c99a90_C30024454_bedroom-movement_20250925115500_1_video.insv",
        "one-data-platform/63-0347e6fd-ade1-46c2-ba07-bb0419c99a90-bedroom-movement/63_0347e6fd-ade1-46c2-ba07-bb0419c99a90_audio2_bedroom-movement_20250925120000_1_audio.wav",
        "one-data-platform/63-0347e6fd-ade1-46c2-ba07-bb0419c99a90-bedroom-movement/63_0347e6fd-ade1-46c2-ba07-bb0419c99a90_video2_bedroom-movement_20250925120000_1_video.insv",
        "one-data-platform/63-0347e6fd-ade1-46c2-ba07-bb0419c99a90-bedroom-movement/63_0347e6fd-ade1-46c2-ba07-bb0419c99a90_audio3_bedroom-movement_20250925130000_1_audio.wav"
    ]
    
    # Sample metadata (some valid, some invalid, some missing)
    sample_metadata = {
        # Valid metadata
        "one-data-platform/63-0347e6fd-ade1-46c2-ba07-bb0419c99a90-bedroom-movement/63_0347e6fd-ade1-46c2-ba07-bb0419c99a90_activity_bedroom-movement_20250925115500_metadata.json": {
            "session_id": "sess_20250925115500",
            "home_id": "63",
            "activity": "bedroom-movement",
            "timestamp": "20250925115500",
            "device_id": "audio1",
            "file_info": {
                "file_size": 77457920,
                "duration": 120.5,
                "format": "wav",
                "codec": "pcm"
            },
            "processing_info": {
                "processed_at": "2025-09-25T11:55:00Z",
                "processing_duration": 2.5,
                "status": "completed"
            }
        },
        
        # Invalid metadata (missing required fields)
        "one-data-platform/63-0347e6fd-ade1-46c2-ba07-bb0419c99a90-bedroom-movement/63_0347e6fd-ade1-46c2-ba07-bb0419c99a90_activity_bedroom-movement_20250925120000_metadata.json": {
            "session_id": "sess_20250925120000",
            "home_id": "63",
            "activity": "bedroom-movement",
            "timestamp": "20250925120000",
            "device_id": "audio2",
            # Missing file_info and processing_info
        },
        
        # Invalid metadata (wrong data types)
        "one-data-platform/63-0347e6fd-ade1-46c2-ba07-bb0419c99a90-bedroom-movement/63_0347e6fd-ade1-46c2-ba07-bb0419c99a90_activity_bedroom-movement_20250925130000_metadata.json": {
            "session_id": 12345,  # Should be string
            "home_id": "63",
            "activity": "bedroom-movement",
            "timestamp": "20250925130000",
            "device_id": "audio3",
            "file_info": {
                "file_size": "invalid_size",  # Should be number
                "duration": 90.0,
                "format": "wav",
                "codec": "pcm"
            },
            "processing_info": {
                "processed_at": "2025-09-25T13:00:00Z",
                "processing_duration": 1.8,
                "status": "completed"
            }
        }
        
        # Note: Some metadata files are intentionally missing to demonstrate the validation
    }
    
    # Create validator and run demo
    validator = DemoMetadataValidator("63")
    results = validator.demo_validation(sample_media_files, sample_metadata)
    
    # Print summary
    validator.print_summary()
    
    # Save results
    output_file = "demo_metadata_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Detailed results saved to: {output_file}")
    
    # Show some key findings
    if results["problematic_sessions"]:
        print(f"\n🚨 FOUND {len(results['problematic_sessions'])} PROBLEMATIC SESSIONS:")
        for session in results["problematic_sessions"]:
            print(f"  • Session {session['session_id']}: {session['issue_count']} issues, {session['affected_files_count']} affected files")
            print(f"    Issues: {', '.join(session['issues'][:3])}")
    else:
        print(f"\n✅ No problematic sessions found!")
    
    if results["skipped_files"]:
        print(f"\n⚠️  {len(results['skipped_files'])} FILES WERE SKIPPED:")
        for file_info in results["skipped_files"][:3]:
            print(f"  • {file_info['media_file']}")
            print(f"    Reason: {file_info['reason']}")

if __name__ == "__main__":
    main()
