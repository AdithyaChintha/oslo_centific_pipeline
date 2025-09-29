#!/usr/bin/env python3
"""
Example script to run metadata validation for a specific home ID
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from metadata_validator import MetadataValidator

def main():
    # Configuration - you can modify these values
    HOME_ID = "63"  # Change this to the home ID you want to validate
    
    # Azure configuration - you can either use environment variables or hardcode
    CONNECTION_STRING = os.getenv(
        "AZURE_CONNECTION_STRING", 
        "DefaultEndpointsProtocol=https;AccountName=oslotestvideo;AccountKey=zOevIegkZjld6ciTY+alA+YkzZ2gdAWVP7rkuhty5NAZ67AtiBB3fRTaa+eE3UbqhgwHZvWkOM0L+ASt4zCx6g==;EndpointSuffix=core.windows.net"
    )
    
    CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "instavideo")
    SOURCE_PREFIX = os.getenv("AZURE_SOURCE_PREFIX", "one-data-platform/")
    
    print(f"Starting metadata validation for Home ID: {HOME_ID}")
    print(f"Container: {CONTAINER_NAME}")
    print(f"Source Prefix: {SOURCE_PREFIX}")
    print("-" * 60)
    
    try:
        # Create validator
        validator = MetadataValidator(
            home_id=HOME_ID,
            connection_string=CONNECTION_STRING,
            container_name=CONTAINER_NAME,
            source_prefix=SOURCE_PREFIX
        )
        
        # Run validation
        results = validator.validate_home_id_metadata()
        
        # Print summary
        validator.print_summary()
        
        # Save results
        output_file = f"metadata_validation_{HOME_ID}.json"
        validator.save_results_to_file(output_file)
        print(f"Detailed results saved to: {output_file}")
        
        # Print some key findings
        if results["problematic_sessions"]:
            print(f"\n🚨 FOUND {len(results['problematic_sessions'])} PROBLEMATIC SESSIONS:")
            for session in results["problematic_sessions"][:5]:  # Show first 5
                print(f"  • Session {session['session_id']}: {session['issue_count']} issues, {session['affected_files_count']} affected files")
                print(f"    Issues: {', '.join(session['issues'][:3])}")  # Show first 3 issues
        else:
            print(f"\n✅ No problematic sessions found!")
        
        if results["skipped_files"]:
            print(f"\n⚠️  {len(results['skipped_files'])} FILES WERE SKIPPED:")
            for file_info in results["skipped_files"][:5]:  # Show first 5
                print(f"  • {file_info['media_file']}")
                print(f"    Reason: {file_info['reason']}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
