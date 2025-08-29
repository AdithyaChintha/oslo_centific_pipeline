#!/usr/bin/env python3
"""
Azure Video Polling and Processing System
Polls Azure Blob Storage for videos, processes them through Ray pipeline one by one
"""

import os
import sys
import json
import time
import yaml
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# Get the correct directory paths (file is now at insta360-video-activity-segmentation level)
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)  # This is insta360-video-activity-segmentation
blob_poll_dir = os.path.join(current_dir, "blob_poll")  # blob_poll subdirectory

# Add current directory to Python path (for ray_jobs, setup, etc.)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Store original working directory
original_cwd = os.getcwd()

# Change to parent directory for imports to work correctly
os.chdir(current_dir)

# Note: pipeline_main will be imported when needed to avoid early dependency loading
# Restore original working directory
os.chdir(original_cwd)

class AzureVideoProcessor:
    def __init__(self, config_path: str = None):
        """Initialize with Azure configuration"""
        # Set default config path relative to parent directory
        if config_path is None:
            # config_path = os.path.join(parent_dir, "blobfuse2_config.yaml")
            config_path = "blobfuse2_config.yaml"
        
        self.config = self._load_config(config_path)
        self.blob_service_client = self._create_blob_client()
        self.container_name = self.config['container']
        self.account_name = self.config['account-name']
        self.account_key = self.config['account-key']
        
        # Setup logging
        log_file = os.path.join(current_dir, 'azure_video_processor.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.video_extensions = ['.insv', '.mp4', '.avi', '.mov', '.mkv']
        self.audio_extensions = ['.wav', '.mp3', '.m4a', '.aac']
        self.local_download_dir = "/tmp/azure_downloads"
        self.output_base_dir = os.path.join(current_dir, "out")
        self.video_list_file = os.path.join(current_dir, "azure_video_list.json")
        self.processing_status_file = os.path.join(current_dir, "processing_status.json")
        
        # Create directories
        os.makedirs(self.local_download_dir, exist_ok=True)
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        self.logger.info(f"🔧 Initialized Azure Video Processor")
        self.logger.info(f"   Current dir: {current_dir}")
        self.logger.info(f"   Config: {config_path}")
        self.logger.info(f"   Container: {self.container_name}")
        self.logger.info(f"   Output dir: {self.output_base_dir}")

    def _load_config(self, config_path: str) -> dict:
        """Load Azure configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config['azstorage']
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def _create_blob_client(self) -> BlobServiceClient:
        """Create Azure Blob Service client"""
        account_url = f"https://{self.config['account-name']}.blob.core.windows.net"
        return BlobServiceClient(
            account_url=account_url,
            credential=self.config['account-key']
        )

    def poll_azure_videos(self, blob_prefix: str = "", force_refresh: bool = False) -> List[Dict]:
        """
        Poll Azure Blob Storage for video files and save to JSON
        
        Args:
            blob_prefix: Optional prefix to filter blobs (e.g., "test/input_videos/")
            force_refresh: Force refresh even if polling was done today
            
        Returns:
            List of video file information
        """
        self.logger.info("🔍 Polling Azure Blob Storage for videos...")
        
        # Check if we already polled within the last 5 minutes (unless force refresh)
        if not force_refresh and os.path.exists(self.video_list_file):
            with open(self.video_list_file, 'r') as f:
                existing_data = json.load(f)
                last_poll = datetime.fromisoformat(existing_data.get('last_poll', '2000-01-01'))
                time_diff = datetime.now() - last_poll
                if time_diff.total_seconds() < 300:  # 300 seconds = 5 minutes
                    self.logger.info(f"📋 Videos polled {int(time_diff.total_seconds())} seconds ago. Found {len(existing_data['videos'])} videos.")
                    return existing_data['videos']
        
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            videos = []
            blob_list = container_client.list_blobs(name_starts_with=blob_prefix)
            
            for blob in blob_list:
                blob_name = blob.name
                
                # Check if it's a video file
                if any(blob_name.lower().endswith(ext) for ext in self.video_extensions):
                    # Look for corresponding audio file
                    audio_file = self._find_corresponding_audio(blob_name, container_client)
                    
                    video_info = {
                        "video_blob_name": blob_name,
                        "audio_blob_name": audio_file,
                        "video_size_mb": round(blob.size / (1024 * 1024), 2),
                        "last_modified": blob.last_modified.isoformat(),
                        "processed": False,
                        "processing_date": None,
                        "output_dir": None,
                        "status": "pending"
                    }
                    videos.append(video_info)
            
            # Save to JSON file
            video_data = {
                "last_poll": datetime.now().isoformat(),
                "total_videos": len(videos),
                "container": self.container_name,
                "blob_prefix": blob_prefix,
                "videos": videos
            }
            
            with open(self.video_list_file, 'w') as f:
                json.dump(video_data, f, indent=2)
            
            self.logger.info(f"📋 Found {len(videos)} video files in Azure Blob Storage")
            self.logger.info(f"💾 Video list saved to {self.video_list_file}")
            
            return videos
            
        except Exception as e:
            self.logger.error(f"❌ Failed to poll Azure Blob Storage: {e}")
            raise

    def _find_corresponding_audio(self, video_blob_name: str, container_client) -> Optional[str]:
        """Find corresponding audio file for a video"""
        video_base = os.path.splitext(video_blob_name)[0]
        
        # Common audio naming patterns
        audio_patterns = [
            f"{video_base}.wav",
            f"{video_base}.mp3",
            f"{video_base}.m4a",
            f"{video_base}-audio.wav",
            f"{video_base}-audio.mp3",
            video_blob_name.replace("video", "audio").replace(".insv", ".wav"),
            video_blob_name.replace("-video", "-audio").replace(".insv", ".wav")
        ]
        
        for pattern in audio_patterns:
            try:
                blob_client = container_client.get_blob_client(pattern)
                if blob_client.exists():
                    return pattern
            except:
                continue
        
        return None

    def download_video_files(self, video_info: Dict) -> Dict[str, str]:
        """Download video and audio files from Azure"""
        video_blob = video_info['video_blob_name']
        audio_blob = video_info['audio_blob_name']
        
        self.logger.info(f"⬇️ Downloading {video_blob}...")
        
        # Create download paths
        video_filename = os.path.basename(video_blob)
        video_local_path = os.path.join(self.local_download_dir, video_filename)
        
        audio_local_path = None
        if audio_blob:
            audio_filename = os.path.basename(audio_blob)
            audio_local_path = os.path.join(self.local_download_dir, audio_filename)
        
        try:
            # Download video
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=video_blob
            )
            
            with open(video_local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            
            self.logger.info(f"✅ Video downloaded: {video_local_path}")
            
            # Download audio if exists
            if audio_blob and audio_local_path:
                audio_client = self.blob_service_client.get_blob_client(
                    container=self.container_name, 
                    blob=audio_blob
                )
                
                with open(audio_local_path, "wb") as f:
                    download_stream = audio_client.download_blob()
                    f.write(download_stream.readall())
                
                self.logger.info(f"✅ Audio downloaded: {audio_local_path}")
            else:
                self.logger.warning("⚠️ No corresponding audio file found")
            
            return {
                "video_path": video_local_path,
                "audio_path": audio_local_path
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to download {video_blob}: {e}")
            raise

    def process_single_video(self, video_info: Dict) -> bool:
        """Process a single video through the Ray pipeline"""
        video_blob = video_info['video_blob_name']
        self.logger.info(f"🎬 Processing video: {video_blob}")
        
        try:
            # Download files
            downloaded_files = self.download_video_files(video_info)
            video_path = downloaded_files['video_path']
            audio_path = downloaded_files['audio_path']
            
            # Create output directory for this video
            video_name = os.path.splitext(os.path.basename(video_blob))[0]
            video_output_dir = os.path.join(self.output_base_dir, f"video_{video_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Run the Ray pipeline
            self.logger.info(f"🚀 Starting Ray pipeline for {video_name}")
            
            # Change to parent directory for pipeline execution
            original_cwd = os.getcwd()
            os.chdir(current_dir)
            
            try:
                # Import pipeline_main here to avoid early dependency loading
                try:
                    from ray_pipeline_testing import pipeline_main
                    self.logger.info("✅ Successfully imported pipeline_main")
                except ImportError as e:
                    self.logger.error(f"❌ Failed to import pipeline_main: {e}")
                    raise
                
                # Create Azure blob client for the pipeline
                azure_blob_client = self.blob_service_client
                azure_output_prefix = f"processed_outputs/{video_name}"
                
                results = pipeline_main(
                    input_video_path=video_path,
                    input_audio_path=audio_path,
                    output_dir=video_output_dir,
                    process_dual_views=None,  # Auto-detect
                    process_unwarped_views=False,  # Default to False, can be made configurable
                    azure_blob_client=azure_blob_client,
                    azure_container=self.container_name,
                    azure_output_prefix=azure_output_prefix,
                    azure_account_name=self.account_name,
                    azure_account_key=self.account_key
                )
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
            
            # Update video info
            video_info.update({
                "processed": True,
                "processing_date": datetime.now().isoformat(),
                "output_dir": video_output_dir,
                "status": "completed",
                "results_summary": results
            })
            
            self.logger.info(f"✅ Video processing completed: {video_name}")
            self.logger.info(f"📊 Results saved to: {video_output_dir}")
            
            # Clean up downloaded files
            self._cleanup_downloaded_files([video_path, audio_path])
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process video {video_blob}: {e}")
            video_info.update({
                "processed": False,
                "processing_date": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            # Clean up downloaded files even on failure
            try:
                if 'downloaded_files' in locals():
                    self._cleanup_downloaded_files([downloaded_files['video_path'], downloaded_files['audio_path']])
            except:
                pass
            
            return False

    def _cleanup_downloaded_files(self, file_paths: List[str]):
        """Clean up downloaded files"""
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.info(f"🗑️ Cleaned up: {file_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to clean up {file_path}: {e}")

    def process_all_videos(self, max_videos: Optional[int] = None):
        """Process all pending videos one by one"""
        self.logger.info("🎯 Starting batch video processing...")
        
        # Load video list
        if not os.path.exists(self.video_list_file):
            self.logger.error(f"❌ Video list file not found: {self.video_list_file}")
            self.logger.info("🔍 Run poll_azure_videos() first to create the video list")
            return
        
        with open(self.video_list_file, 'r') as f:
            video_data = json.load(f)
        
        videos = video_data['videos']
        pending_videos = [v for v in videos if not v.get('processed', False)]
        
        if not pending_videos:
            self.logger.info("✅ No pending videos to process")
            return
        
        if max_videos:
            pending_videos = pending_videos[:max_videos]
        
        self.logger.info(f"📋 Found {len(pending_videos)} videos to process")
        
        successful = 0
        failed = 0
        
        for i, video_info in enumerate(pending_videos, 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎬 Processing video {i}/{len(pending_videos)}: {video_info['video_blob_name']}")
            self.logger.info(f"{'='*60}")
            
            try:
                if self.process_single_video(video_info):
                    successful += 1
                else:
                    failed += 1
                    
                # Save progress after each video
                with open(self.video_list_file, 'w') as f:
                    json.dump(video_data, f, indent=2)
                
                self.logger.info(f"📊 Progress: {successful} successful, {failed} failed")
                
            except KeyboardInterrupt:
                self.logger.info("\n⏹️ Processing interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Unexpected error processing video: {e}")
                failed += 1
        
        self.logger.info(f"\n🏁 Batch processing completed!")
        self.logger.info(f"📊 Final results: {successful} successful, {failed} failed")

    def get_processing_status(self) -> Dict:
        """Get current processing status"""
        if not os.path.exists(self.video_list_file):
            return {"error": "Video list not found. Run poll first."}
        
        with open(self.video_list_file, 'r') as f:
            video_data = json.load(f)
        
        videos = video_data['videos']
        total = len(videos)
        processed = len([v for v in videos if v.get('processed', False)])
        failed = len([v for v in videos if v.get('status') == 'failed'])
        pending = total - processed
        
        return {
            "total_videos": total,
            "processed": processed,
            "failed": failed,
            "pending": pending,
            "last_poll": video_data.get('last_poll'),
            "completion_rate": f"{(processed/total*100):.1f}%" if total > 0 else "0%"
        }


def main():
    """Main function - Automatic poll, status, and process workflow"""
    import argparse
    
    # Parse arguments but provide defaults for automated workflow
    parser = argparse.ArgumentParser(description="Azure Video Processing System - Automated Workflow")
    parser.add_argument("--poll", action="store_true", help="Poll Azure for videos (legacy)")
    parser.add_argument("--process", action="store_true", help="Process all pending videos (legacy)")
    parser.add_argument("--status", action="store_true", help="Show processing status (legacy)")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh video list")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos to process")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = AzureVideoProcessor()
    
    # Legacy support for individual commands
    if args.poll:
        print("🔍 Polling Azure Blob Storage for videos...")
        videos = processor.poll_azure_videos(
            blob_prefix="test/input_videos",
            force_refresh=args.force_refresh
        )
        print(f"📋 Found {len(videos)} videos")
        return
        
    elif args.process:
        print("🎬 Starting video processing...")
        processor.process_all_videos(max_videos=args.max_videos)
        return
        
    elif args.status:
        status = processor.get_processing_status()
        print("\n📊 Processing Status:")
        print(f"   Total Videos: {status.get('total_videos', 0)}")
        print(f"   Processed: {status.get('processed', 0)}")
        print(f"   Failed: {status.get('failed', 0)}")
        print(f"   Pending: {status.get('pending', 0)}")
        print(f"   Completion Rate: {status.get('completion_rate', '0%')}")
        print(f"   Last Poll: {status.get('last_poll', 'Never')}")
        return
    
    # DEFAULT AUTOMATED WORKFLOW
    print("🚀 Azure Video Processing System - Automated Workflow")
    print("=" * 60)
    
    # Step 1: Poll Azure for videos
    print("🔍 Step 1: Polling Azure Blob Storage for videos...")
    videos = processor.poll_azure_videos(
        blob_prefix="test/input_videos",
        force_refresh=args.force_refresh
    )
    print(f"📋 Found {len(videos)} videos from test/input_videos/")
    
    # Step 2: Show current status
    print("\n📊 Step 2: Current Processing Status")
    status = processor.get_processing_status()
    print(f"   Total Videos: {status.get('total_videos', 0)}")
    print(f"   Processed: {status.get('processed', 0)}")
    print(f"   Failed: {status.get('failed', 0)}")
    print(f"   Pending: {status.get('pending', 0)}")
    print(f"   Completion Rate: {status.get('completion_rate', '0%')}")
    print(f"   Last Poll: {status.get('last_poll', 'Never')}")
    
    # Step 3: Process pending videos
    if status.get('pending', 0) > 0:
        print(f"\n🎬 Step 3: Processing {status['pending']} pending videos...")
        print("=" * 60)
        processor.process_all_videos(max_videos=args.max_videos)
        
        # Step 4: Show final results
        print("\n🏁 Final Results:")
        print("=" * 60)
        final_status = processor.get_processing_status()
        print(f"   Total Videos: {final_status.get('total_videos', 0)}")
        print(f"   Processed: {final_status.get('processed', 0)}")
        print(f"   Failed: {final_status.get('failed', 0)}")
        print(f"   Pending: {final_status.get('pending', 0)}")
        print(f"   Completion Rate: {final_status.get('completion_rate', '0%')}")
        
        if final_status.get('pending', 0) == 0:
            print("\n🎉 All videos have been processed successfully!")
        else:
            print(f"\n⚠️ {final_status.get('pending', 0)} videos still pending. Run again to continue processing.")
            
    else:
        print("\n✅ No pending videos to process!")
        print("🎉 All videos have been processed successfully!")
    
    print(f"\n📁 Results saved to: {processor.output_base_dir}")
    print("🚀 Automated workflow completed!")


if __name__ == "__main__":
    main()