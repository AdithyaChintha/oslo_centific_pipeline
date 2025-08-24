#!/usr/bin/env python3

"""
End-to-End Shard Processing Pipeline for OSLO NORA Label Studio Tasks

This script handles the complete workflow:
1. Discover video shards and generate Azure SAS URLs
2. Download shard-level model analysis JSON files
3. Convert model data to OSLO predictions
4. Build Label Studio tasks with video URLs + predictions
5. Import tasks to Label Studio

Usage:
    python shard_processor.py --video_id VID_20250809_094753_00_044 --shards shard_1,shard_2
"""

import json
import os
import sys
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import quote

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

try:
    from label_studio_sdk import Client as LabelStudioClient
except ImportError:
    LabelStudioClient = None
    print("Warning: label-studio-sdk not installed. Install with: pip install label-studio-sdk")


class AzureBlobManager:
    """Handles Azure Blob Storage connections and operations"""
    
    def __init__(self, account_name: str, account_key: str, container_name: str):
        self.account_name = account_name
        self.account_key = account_key
        self.container_name = container_name
        self.account_url = f"https://{account_name}.blob.core.windows.net"
        self.client = BlobServiceClient(account_url=self.account_url, credential=account_key)
    
    def generate_sas_url(self, blob_path: str, expiry_days: int = 90) -> str:
        """Generate SAS URL for a blob with read permissions"""
        expiry = datetime.now(timezone.utc) + timedelta(days=expiry_days)
        sas_token = generate_blob_sas(
            account_name=self.account_name,
            container_name=self.container_name,
            blob_name=blob_path,
            account_key=self.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        base_url = f"{self.account_url}/{self.container_name}/{quote(blob_path, safe='/')}"
        return f"{base_url}?{sas_token}"
    
    def download_blob_json(self, blob_path: str) -> Dict[str, Any]:
        """Download and parse JSON blob"""
        try:
            blob_client = self.client.get_blob_client(container=self.container_name, blob=blob_path)
            json_content = blob_client.download_blob().readall()
            return json.loads(json_content.decode('utf-8'))
        except Exception as e:
            print(f"Error downloading blob {blob_path}: {e}")
            raise
    
    def list_blobs_with_prefix(self, prefix: str) -> List[str]:
        """List all blobs with given prefix"""
        container_client = self.client.get_container_client(self.container_name)
        return [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]


class ShardVideoPathResolver:
    """Resolves video shard paths and generates SAS URLs"""
    
    def __init__(self, blob_manager: AzureBlobManager):
        self.blob_manager = blob_manager
    
    def find_shard_video_paths(self, video_id: str, base_path: str) -> Dict[str, Dict[str, str]]:
        """
        Find video shard paths for all shards and views
        
        Returns:
        {
            "shard_1": {
                "view_1_videos": ["path/to/view1_part0.mp4", ...],
                "view_2_videos": ["path/to/view2_part0.mp4", ...]
            }
        }
        """
        video_base_path = f"{base_path}/{video_id}".lstrip("/")
        all_blobs = self.blob_manager.list_blobs_with_prefix(video_base_path)
        
        shard_videos = {}
        
        for blob_path in all_blobs:
            # Look for video_shards directories in view folders
            if "/video_shards/" in blob_path and blob_path.endswith(".mp4"):
                # Extract shard and view info from path
                # Expected: base_path/video_id/view_1/video_shards/file.mp4 or base_path/video_id/shard_N/view_1/video_shards/file.mp4
                
                parts = blob_path.split("/")
                
                # Try to find shard pattern
                shard_name = None
                view_name = None
                
                for i, part in enumerate(parts):
                    if part.startswith("shard_"):
                        shard_name = part
                        if i + 1 < len(parts) and parts[i + 1].startswith("view_"):
                            view_name = parts[i + 1]
                    elif part.startswith("view_") and not shard_name:
                        # Direct view structure (no shard folder)
                        view_name = part
                        shard_name = "shard_1"  # Default shard name
                
                if shard_name and view_name:
                    if shard_name not in shard_videos:
                        shard_videos[shard_name] = {"view_1_videos": [], "view_2_videos": []}
                    
                    if view_name == "view_1":
                        shard_videos[shard_name]["view_1_videos"].append(blob_path)
                    elif view_name == "view_2":
                        shard_videos[shard_name]["view_2_videos"].append(blob_path)
        
        return shard_videos
    
    def get_shard_video_urls(self, video_id: str, shard_name: str, base_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Get SAS URLs for a specific shard's videos"""
        shard_videos = self.find_shard_video_paths(video_id, base_path)
        
        if shard_name not in shard_videos:
            print(f"Warning: {shard_name} not found in video structure")
            return None, None
        
        shard_data = shard_videos[shard_name]
        
        # Get first video from each view (assuming single video per shard/view)
        view_1_url = None
        view_2_url = None
        
        if shard_data["view_1_videos"]:
            view_1_url = self.blob_manager.generate_sas_url(shard_data["view_1_videos"][0])
        
        if shard_data["view_2_videos"]:
            view_2_url = self.blob_manager.generate_sas_url(shard_data["view_2_videos"][0])
        
        return view_1_url, view_2_url


class ShardModelDataFetcher:
    """Downloads and parses shard-level model analysis data"""
    
    def __init__(self, blob_manager: AzureBlobManager):
        self.blob_manager = blob_manager
    
    def get_shard_model_data(self, video_id: str, shard_name: str, base_path: str) -> Optional[Dict[str, Any]]:
        """Download shard-level consolidated model data"""
        # Try different possible paths for shard model data
        possible_paths = [
            f"{base_path}/{video_id}/{shard_name}/consolidated_{shard_name}_output.json",
            f"{base_path}/{video_id}/consolidated_{shard_name}_output.json"
        ]
        
        for blob_path in possible_paths:
            try:
                print(f"Trying to download: {blob_path}")
                return self.blob_manager.download_blob_json(blob_path.lstrip("/"))
            except Exception:
                continue
        
        print(f"Error: Could not find shard model data for {video_id}/{shard_name}")
        return None


class OSLOModelConverter:
    """Converts shard-level model outputs to OSLO NORA UI prediction format"""
    
    def __init__(self):
        self.domain_keywords = {
            "Food & Mealtime": ["kitchen", "cooking", "eating", "meal", "food", "dining", "microwave", "oven", "refrigerator", "countertop"],
            "Personal Care & Hygiene": ["bathroom", "toilet", "sink", "shower", "grooming", "brushing", "hygiene", "mirror", "toiletries"],
            "Household Movement": ["walking", "moving", "fetching", "cabinet", "door", "stairs"],
            "Cleaning & Maintenance": ["cleaning", "vacuum", "laundry", "tidying", "trash", "maintenance"],
            "Work & Study": ["desk", "computer", "writing", "work", "study", "office"],
            "Leisure & Entertainment": ["TV", "television", "gaming", "reading", "entertainment", "hobby"],
            "Exercise & Wellness": ["exercise", "workout", "fitness", "dancing", "yoga"],
            "Social & Family Life": ["family", "social", "hosting", "gathering", "conversation"],
            "Pet Care": ["pet", "dog", "cat", "feeding", "animal"],
            "Home & Garden Projects": ["plant", "garden", "project", "indoor"],
            "Shopping & Logistics": ["package", "mail", "delivery", "shopping"],
            "Safety & Security": ["security", "alarm", "safety", "lock"]
        }
        
        self.action_keywords = {
            "Food & Mealtime": {
                "Cooking / baking": ["cooking", "baking", "preparing", "microwave", "oven"],
                "Eating together": ["eating", "meal", "dining"],
                "Washing dishes": ["washing", "cleaning dishes", "sink"],
                "Setting the table": ["table", "setting"],
                "Planning meals": ["planning", "menu"],
                "Putting away leftovers": ["storing", "leftovers", "putting away"],
                "Other - Food & Mealtime": []
            },
            "Personal Care & Hygiene": {
                "Brushing teeth": ["brushing", "teeth", "toothbrush"],
                "Grooming & skincare": ["grooming", "skincare", "mirror"],
                "Taking medication": ["medication", "pills"],
                "Other - Personal Care & Hygiene": []
            },
            "Household Movement": {
                "Walking between rooms": ["walking", "moving between"],
                "Fetching objects": ["fetching", "getting", "retrieving"],
                "Opening cabinets & appliances": ["opening", "cabinet", "appliance"],
                "Going up/down stairs": ["stairs", "upstairs", "downstairs"],
                "Other - Household Movement": []
            }
        }
    
    def extract_scene_info(self, model_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scene information from shard-level model outputs"""
        scene_info = {
            "descriptions": [],
            "detected_objects": [],
            "motion_segments": [],
            "yolo_detections": []
        }
        
        if "scene_detection" in model_output:
            for scene in model_output["scene_detection"]:
                if "description" in scene:
                    scene_info["descriptions"].append(scene["description"])
        
        if "yolo_detections" in model_output:
            for detection in model_output["yolo_detections"]:
                if isinstance(detection, dict) and "cls" in detection:
                    scene_info["detected_objects"].append(detection["cls"])
                    scene_info["yolo_detections"].append(detection)
        
        if "motion_analysis" in model_output:
            scene_info["motion_segments"] = model_output["motion_analysis"]
        
        return scene_info
    
    def determine_domain_and_actions(self, scene_info: Dict[str, Any]) -> List[str]:
        """Determine domain and actions based on scene analysis"""
        all_text = " ".join(scene_info["descriptions"]).lower()
        detected_objects = [obj.lower() for obj in scene_info["detected_objects"]]
        
        matched_domains = []
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in all_text:
                    score += 2
                if keyword.lower() in detected_objects:
                    score += 3
            
            if score > 0:
                matched_domains.append((domain, score))
        
        matched_domains.sort(key=lambda x: x[1], reverse=True)
        
        result_paths = []
        if matched_domains:
            top_domain = matched_domains[0][0]
            
            if top_domain in self.action_keywords:
                action_scores = []
                for action, action_keywords in self.action_keywords[top_domain].items():
                    action_score = 0
                    for keyword in action_keywords:
                        if keyword.lower() in all_text:
                            action_score += 1
                        if keyword.lower() in detected_objects:
                            action_score += 2
                    
                    if action_score > 0:
                        action_scores.append((action, action_score))
                
                if action_scores:
                    action_scores.sort(key=lambda x: x[1], reverse=True)
                    top_action = action_scores[0][0]
                    result_paths.append(f"{top_domain} > {top_action}")
                else:
                    result_paths.append(f"{top_domain} > Other - {top_domain}")
            else:
                result_paths.append(top_domain)
        
        return result_paths if result_paths else ["Household Movement > Other - Household Movement"]
    
    def make_model_based_predictions(self, model_output: Dict[str, Any]) -> Dict[str, List[str]]:
        """Make predictions from actual model outputs"""
        predictions = {
            "nudity": [],
            "minors": [],
            "non_consenting": []
        }
        
        if "nsfw_analysis" in model_output:
            for result in model_output["nsfw_analysis"]:
                if isinstance(result, dict) and result.get("total_nsfw_detections", 0) > 0:
                    predictions["nudity"].append("Nudity present")
                    break
        
        if "face_analysis" in model_output:
            for result in model_output["face_analysis"]:
                if isinstance(result, dict):
                    age_dist = result.get("detailed_analysis_summary", {}).get("age_distribution", {})
                    if age_dist.get("minors", 0) > 0:
                        predictions["minors"].extend(["Audio", "Video"])
                        break
        
        return predictions
    
    def convert_to_predictions(self, model_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert shard model output to Label Studio predictions"""
        scene_info = self.extract_scene_info(model_output)
        domain_actions = self.determine_domain_and_actions(scene_info)
        model_predictions = self.make_model_based_predictions(model_output)
        
        predictions = []
        
        # Domain & Actions prediction
        if domain_actions:
            taxonomy_values = []
            for domain_action in domain_actions:
                if " > " in domain_action:
                    parts = domain_action.split(" > ")
                    taxonomy_values.append(parts)
                else:
                    taxonomy_values.append([domain_action])
            
            predictions.append({
                "id": "taxonomy_domain_actions_pred",
                "type": "taxonomy",
                "value": {"taxonomy": taxonomy_values},
                "score": 0.8,
                "from_name": "taxonomy_domain_actions",
                "to_name": "video_left"
            })
        
        # Lighting prediction (always include)
        predictions.append({
            "id": "lighting_pred",
            "type": "choices",
            "value": {"choices": ["Bright light"]},
            "score": 0.7,
            "from_name": "lighting",
            "to_name": "video_left"
        })
        
        # Nudity prediction
        if model_predictions["nudity"]:
            predictions.append({
                "id": "nudity_pred",
                "type": "choices",
                "value": {"choices": model_predictions["nudity"]},
                "score": 0.9,
                "from_name": "nudity",
                "to_name": "video_left"
            })
        
        # Minors prediction
        if model_predictions["minors"]:
            predictions.append({
                "id": "minors_pred",
                "type": "choices",
                "value": {"choices": model_predictions["minors"]},
                "score": 0.9,
                "from_name": "minors",
                "to_name": "video_left"
            })
        
        return predictions


class LabelStudioTaskBuilder:
    """Builds Label Studio task format from video URLs and predictions"""
    
    def build_task(self, video_left_url: str, video_right_url: str, 
                   predictions: List[Dict[str, Any]], video_id: str, 
                   shard_name: str) -> Dict[str, Any]:
        """Build a single Label Studio task"""
        return {
            "data": {
                "video_left": video_left_url,
                "video_right": video_right_url,
                "meta": "",
                "meta.home_identifier": f"{video_id}_{shard_name}",
                "meta.recording_datetime": "",
                "meta.domain": "production",
                "meta.actions": ""
            },
            "annotations": [],
            "predictions": [{"result": predictions}] if predictions else []
        }


class LabelStudioImporter:
    """Imports tasks to Label Studio"""
    
    def __init__(self, url: str, api_token: str, project_id: str):
        self.url = url
        self.api_token = api_token
        self.project_id = project_id
        
        if LabelStudioClient is None:
            raise ImportError("label-studio-sdk is required. Install with: pip install label-studio-sdk")
        
        self.client = LabelStudioClient(url=url, api_key=api_token)
    
    def import_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import tasks to Label Studio project"""
        try:
            project = self.client.get_project(int(self.project_id))
            print(f"Connected to Label Studio project: {project.title}")
            print(f"Importing {len(tasks)} tasks...")
            
            result = project.import_tasks(tasks)
            
            # Get detailed result information
            if isinstance(result, dict):
                print(f"Import successful! Details: {result}")
                
                # Try to get task IDs from the imported tasks
                if "task_count" in result:
                    print(f"Successfully imported {result['task_count']} tasks")
                
                # Get the actual task IDs by querying the project
                try:
                    all_tasks = project.get_tasks()
                    # Get the most recent tasks (assuming they are the ones just imported)
                    recent_tasks = sorted(all_tasks, key=lambda x: x.get('id', 0), reverse=True)[:len(tasks)]
                    
                    print("\nImported Task IDs:")
                    for i, task in enumerate(recent_tasks):
                        task_id = task.get('id')
                        home_id = task.get('data', {}).get('meta.home_identifier', 'unknown')
                        print(f"  Task {i+1}: ID={task_id}, Home={home_id}")
                        print(f"    URL: {self.url}/projects/{self.project_id}/data?tab=0&task={task_id}")
                
                except Exception as e:
                    print(f"Could not retrieve task IDs: {e}")
                
            else:
                print(f"Import result: {result}")
            
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            print(f"Error importing tasks to Label Studio: {e}")
            import traceback
            traceback.print_exc()
            raise


class EndToEndShardProcessor:
    """Main orchestrator for the end-to-end shard processing pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Azure components
        self.blob_manager = AzureBlobManager(
            config["azure"]["account_name"],
            config["azure"]["account_key"],
            config["azure"]["container_name"]
        )
        
        self.video_resolver = ShardVideoPathResolver(self.blob_manager)
        self.model_fetcher = ShardModelDataFetcher(self.blob_manager)
        
        # Initialize processing components
        self.converter = OSLOModelConverter()
        self.task_builder = LabelStudioTaskBuilder()
        
        # Initialize Label Studio importer if configured
        if config.get("label_studio"):
            self.importer = LabelStudioImporter(
                config["label_studio"]["url"],
                config["label_studio"]["api_token"],
                config["label_studio"]["project_id"]
            )
        else:
            self.importer = None
    
    def discover_available_shards(self, video_id: str) -> List[str]:
        """Auto-discover all available shards for a video"""
        base_path = self.config["azure"]["base_path"]
        video_path = f"{base_path}/{video_id}".lstrip("/")
        
        print(f"Discovering shards in: {video_path}")
        
        # Get all blobs under the video path
        all_blobs = self.blob_manager.list_blobs_with_prefix(video_path)
        
        # Find shard directories
        shard_names = set()
        
        for blob_path in all_blobs:
            # Look for shard patterns in the path
            parts = blob_path.split("/")
            
            for part in parts:
                if part.startswith("shard_") and part not in shard_names:
                    # Validate this shard has required files
                    if self._validate_shard(video_id, part):
                        shard_names.add(part)
        
        discovered_shards = sorted(list(shard_names))
        print(f"Discovered {len(discovered_shards)} valid shards: {discovered_shards}")
        
        return discovered_shards
    
    def _validate_shard(self, video_id: str, shard_name: str) -> bool:
        """Validate that a shard has both model data and video files"""
        base_path = self.config["azure"]["base_path"]
        
        # Check for model JSON file
        json_paths = [
            f"{base_path}/{video_id}/{shard_name}/consolidated_{shard_name}_output.json",
            f"{base_path}/{video_id}/consolidated_{shard_name}_output.json"
        ]
        
        has_json = False
        for json_path in json_paths:
            try:
                self.blob_manager.download_blob_json(json_path.lstrip("/"))
                has_json = True
                break
            except:
                continue
        
        if not has_json:
            print(f"   Warning: {shard_name} missing model JSON file")
            return False
        
        # Check for video files
        view_1_url, view_2_url = self.video_resolver.get_shard_video_urls(video_id, shard_name, base_path)
        
        if not view_1_url or not view_2_url:
            print(f"   Warning: {shard_name} missing video files")
            return False
        
        return True
        """Process a single shard into a Label Studio task"""
        print(f"\n=== Processing {video_id}/{shard_name} ===")
        
        base_path = self.config["azure"]["base_path"]
        
        # 1. Get video URLs
        print("1. Getting video URLs...")
        view_1_url, view_2_url = self.video_resolver.get_shard_video_urls(video_id, shard_name, base_path)
        
        if not view_1_url or not view_2_url:
            print(f"Warning: Missing video URLs for {shard_name}")
            return None
        
        print(f"   View 1: {view_1_url[:100]}...")
        print(f"   View 2: {view_2_url[:100]}...")
        
        # 2. Get model data
        print("2. Downloading model data...")
        model_data = self.model_fetcher.get_shard_model_data(video_id, shard_name, base_path)
        
        if not model_data:
            print(f"Error: No model data found for {shard_name}")
            return None
        
        print(f"   Found {len(model_data.get('yolo_detections', []))} YOLO detections")
        print(f"   Found {len(model_data.get('scene_detection', []))} scene descriptions")
        
        # 3. Convert to predictions
        print("3. Converting to predictions...")
        predictions = self.converter.convert_to_predictions(model_data)
        print(f"   Generated {len(predictions)} predictions")
        
        # 4. Build task
        print("4. Building Label Studio task...")
        task = self.task_builder.build_task(view_1_url, view_2_url, predictions, video_id, shard_name)
        
        return task
    
    def process_multiple_shards(self, video_id: str, shard_names: List[str]) -> List[Dict[str, Any]]:
        """Process multiple shards"""
        tasks = []
        
        for shard_name in shard_names:
            task = self.process_single_shard(video_id, shard_name)
            if task:
                tasks.append(task)
        
        return tasks
    
    def save_tasks_to_file(self, tasks: List[Dict[str, Any]], output_path: str):
        """Save tasks to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(tasks)} tasks to {output_path}")
    
    def import_to_label_studio(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import tasks to Label Studio"""
        if not self.importer:
            raise ValueError("Label Studio configuration not provided")
        
        print(f"\n=== IMPORTING TO LABEL STUDIO ===")
        print(f"URL: {self.config['label_studio']['url']}")
        print(f"Project ID: {self.config['label_studio']['project_id']}")
        print(f"Tasks to import: {len(tasks)}")
        
        result = self.importer.import_tasks(tasks)
        return result
    
    def run(self, video_id: str, shard_names: Optional[List[str]] = None, 
            output_file: Optional[str] = None, import_to_ls: bool = False) -> List[Dict[str, Any]]:
        """Run the complete pipeline"""
        print(f"Starting end-to-end processing for {video_id}")
        
        # Auto-discover shards if not provided
        if shard_names is None or len(shard_names) == 0:
            print("No shards specified - auto-discovering available shards...")
            shard_names = self.discover_available_shards(video_id)
            
            if not shard_names:
                print("No valid shards found for auto-discovery. Exiting.")
                return []
        
        print(f"Processing shards: {shard_names}")
        
        # Process all shards
        tasks = self.process_multiple_shards(video_id, shard_names)
        
        if not tasks:
            print("No tasks generated. Exiting.")
            return []
        
        print(f"\nSuccessfully generated {len(tasks)} tasks from {len(shard_names)} shards")
        
        # Save to file if requested
        if output_file:
            self.save_tasks_to_file(tasks, output_file)
        
        # Import to Label Studio if requested
        if import_to_ls:
            self.import_to_label_studio(tasks)
        
        return tasks


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Process video shards for Label Studio")
    parser.add_argument("--video_id", required=True, help="Video ID (e.g., VID_20250809_094753_00_044)")
    parser.add_argument("--shards", help="Comma-separated shard names (e.g., shard_1,shard_2). If not provided, auto-discovers all available shards")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--import-to-ls", action="store_true", help="Import to Label Studio")
    parser.add_argument("--config", help="Configuration file path", default="shard_config.json")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "azure": {
                "account_name": "oslotestvideo",
                "account_key": "zOevIegkZjld6ciTY+alA+YkzZ2gdAWVP7rkuhty5NAZ67AtiBB3fRTaa+eE3UbqhgwHZvWkOM0L+ASt4zCx6g==",
                "container_name": "instavideo",
                "base_path": "krishna-test/test1/test_activity/pre-annotation-output"
            },
            "label_studio": {
                "url": "https://annotations-stg.oneforma2.com/",
                "api_token": "d75a31c7994b96099cfbf7d61e15cff643943853",
                "project_id": "5445"
            }
        }
    
    # Parse shard names (optional now)
    shard_names = []
    if args.shards:
        shard_names = [s.strip() for s in args.shards.split(",")]
    
    # Initialize processor
    processor = EndToEndShardProcessor(config)
    
    # Set default output file if not provided
    output_file = args.output or f"tasks_{args.video_id}_all_shards.json"
    
    # Run the pipeline
    try:
        tasks = processor.run(
            video_id=args.video_id,
            shard_names=shard_names,
            output_file=output_file,
            import_to_ls=args.import_to_ls
        )
        
        print(f"\nPipeline completed successfully!")
        print(f"Generated {len(tasks)} tasks")
        
        if getattr(args, 'import_to_ls', False):
            print(f"Tasks have been imported to Label Studio project {config['label_studio']['project_id']}")
            print(f"View them at: {config['label_studio']['url']}/projects/{config['label_studio']['project_id']}/data")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Example usage:
"""
# working command 
python azure_shard_to_labelstudio_processor.py --video_id VID_20250809_094753_00_044 --import-to-ls

# Process single shard
python shard_processor.py --video_id VID_20250809_094753_00_044 --shards shard_1

# Process multiple shards
python shard_processor.py --video_id VID_20250809_094753_00_044 --shards shard_1,shard_2

# Process and import to Label Studio
python shard_processor.py --video_id VID_20250809_094753_00_044 --shards shard_1 --import

# Use custom output file
python shard_processor.py --video_id VID_20250809_094753_00_044 --shards shard_1 --output my_tasks.json
"""