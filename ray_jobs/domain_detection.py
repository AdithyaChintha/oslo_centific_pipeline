#!/usr/bin/env python3
"""
Domain Detection Ray Task

This module provides a Ray task for domain classification of video scenes.
It takes scene detection outputs and classifies them into predefined domains
using the Groq API with GPT-OSS-20B model.
"""

import os
import json
import yaml
import ray
from typing import Dict, List, Optional
from groq import Groq
from utils.logger import get_logger
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = get_logger("DomainDetection")

# Load configuration from YAML file
def load_groq_config(config_path: str = "config/domain_detection_groqconfig.yaml") -> Dict:
    """Load Groq configuration from YAML file."""
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to get the project root
        project_root = os.path.dirname(script_dir)
        # Construct the full path to the config file
        full_config_path = os.path.join(project_root, config_path)
        
        with open(full_config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load Groq config from {config_path}: {e}")
        # Return default configuration as fallback
        return {
            "groq": {
                "api_key": "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",
                "model": "openai/gpt-oss-20b",
                "temperature": 0.3,
                "max_tokens": 50,
                "top_p": 1,
                "stream": False
            },
            "domain_activity_detection": {
                "domains": [
                    "Food & Mealtime",
                    "Personal Care & Hygiene", 
                    "Household Movement",
                    "Cleaning & Maintenance",
                    "Work & Study",
                    "Leisure & Entertainment",
                    "Exercise & Wellness",
                    "Social & Family Life",
                    "Pet Care",
                    "Home & Garden Projects",
                    "Shopping & Logistics",
                    "Safety & Security"
                ],
                "default_domain": "Unkonwn",
                "prompt_template": """You are a video scene domain classifier. Given a scene description, classify it into one of the following domains:

{domains_list}

Scene Description: "{description}"

Please respond with ONLY the exact domain name from the list above that best matches this scene. Do not include any additional text, explanations, or formatting."""
            }
        }

# Load configuration
CONFIG = load_groq_config()
DOMAINS = CONFIG["domain_activity_detection"]["domains"]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")

def classify_scene_domain(description: str, domains: List[str]) -> str:
    """
    Classify a scene description into one of the predefined domains using Groq API.
    
    Args:
        description: Scene description text
        domains: List of available domains
        
    Returns:
        Classified domain name
    """
    try:
        # Get configuration from loaded config
        groq_config = CONFIG["groq"]
        domain_config = CONFIG["domain_activity_detection"]
        
        client = Groq(api_key=groq_config["api_key"])
        
        # Create the prompt for domain classification using template from config
        domains_text = "\n".join([f"- {domain}" for domain in domains])
        
        prompt = domain_config["prompt_template"].format(
            domains_list=domains_text,
            description=description
        )

        completion = client.chat.completions.create(
            model=groq_config["model"],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=groq_config["temperature"],
            max_tokens=groq_config["max_tokens"],
            top_p=groq_config["top_p"],
            stream=groq_config["stream"],
            stop=None
        )
        
        # Extract the response
        response = completion.choices[0].message.content.strip()
        
        # Validate that the response is one of our domains
        if response in domains:
            return response
        else:
            # If response doesn't match exactly, try to find the closest match
            for domain in domains:
                if domain.lower() in response.lower() or response.lower() in domain.lower():
                    logger.warning(f"Approximate match found: '{response}' -> '{domain}'")
                    return domain
            
            # Default fallback
            logger.warning(f"Could not classify scene, using default domain. Response: '{response}'")
            return CONFIG["domain_activity_detection"]["default_domain"]
            
    except Exception as e:
        logger.error(f"Error classifying scene domain: {e}")
        return CONFIG["domain_activity_detection"]["default_domain"]

@ray.remote
def process_scene_domain_classification(scene_output_dir: str) -> Dict:
    """
    Ray task to process scene detection outputs and add domain classification.
    
    Args:
        scene_output_dir: Directory containing scene detection JSON files
        
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"🎯 Starting domain classification for scene outputs in: {scene_output_dir}")
        
        if not os.path.exists(scene_output_dir):
            logger.error(f"Scene output directory does not exist: {scene_output_dir}")
            return {"success": False, "error": "Directory not found"}
        
        # Find all scene detection JSON files
        json_files = []
        for file in os.listdir(scene_output_dir):
            if file.endswith('_scene_detection_results.json'):
                json_files.append(os.path.join(scene_output_dir, file))
        
        if not json_files:
            logger.warning(f"No scene detection JSON files found in: {scene_output_dir}")
            return {"success": False, "error": "No scene files found"}
        
        logger.info(f"📁 Found {len(json_files)} scene detection files to process")
        
        processed_files = []
        total_scenes = 0
        classified_scenes = 0
        
        for json_file in json_files:
            try:
                logger.info(f"🔄 Processing: {os.path.basename(json_file)}")
                
                # Load the scene detection results
                with open(json_file, 'r') as f:
                    scene_data = json.load(f)
                
                # Process each scene in the file
                if 'scenes' in scene_data:
                    file_scenes = len(scene_data['scenes'])
                    total_scenes += file_scenes
                    
                    for scene in scene_data['scenes']:
                        if 'description' in scene:
                            description = scene['description']
                            
                            # Classify the scene domain
                            domain = classify_scene_domain(description, DOMAINS)
                            scene['domain'] = domain
                            classified_scenes += 1
                            
                            logger.debug(f"   📝 Scene classified as: {domain}")
                
                # Save the updated JSON file
                with open(json_file, 'w') as f:
                    json.dump(scene_data, f, indent=2)
                
                processed_files.append(json_file)
                logger.info(f"✅ Updated: {os.path.basename(json_file)}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {json_file}: {e}")
                continue
        
        result = {
            "success": True,
            "processed_files": len(processed_files),
            "total_scenes": total_scenes,
            "classified_scenes": classified_scenes,
            "files": processed_files
        }
        
        logger.info(f"🎉 Domain classification completed: {classified_scenes}/{total_scenes} scenes classified")
        return result
        
    except Exception as e:
        logger.error(f"❌ Domain classification failed: {e}")
        return {"success": False, "error": str(e)}

@ray.remote
def process_multiple_scene_directories(scene_directories: List[str]) -> Dict:
    """
    Ray task to process multiple scene output directories in parallel.
    
    Args:
        scene_directories: List of directories containing scene detection JSON files
        
    Returns:
        Dictionary with combined processing results
    """
    try:
        logger.info(f"🎯 Starting batch domain classification for {len(scene_directories)} directories")
        
        # Process all directories in parallel
        tasks = []
        for scene_dir in scene_directories:
            task = process_scene_domain_classification.remote(scene_dir)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = ray.get(tasks)
        
        # Combine results
        total_processed_files = 0
        total_scenes = 0
        total_classified_scenes = 0
        successful_dirs = 0
        failed_dirs = 0
        
        for result in results:
            if result.get("success", False):
                successful_dirs += 1
                total_processed_files += result.get("processed_files", 0)
                total_scenes += result.get("total_scenes", 0)
                total_classified_scenes += result.get("classified_scenes", 0)
            else:
                failed_dirs += 1
                logger.error(f"Directory processing failed: {result.get('error', 'Unknown error')}")
        
        combined_result = {
            "success": successful_dirs > 0,
            "successful_directories": successful_dirs,
            "failed_directories": failed_dirs,
            "total_processed_files": total_processed_files,
            "total_scenes": total_scenes,
            "total_classified_scenes": total_classified_scenes,
            "individual_results": results
        }
        
        logger.info(f"🎉 Batch domain classification completed: {successful_dirs}/{len(scene_directories)} directories successful")
        logger.info(f"📊 Total: {total_classified_scenes}/{total_scenes} scenes classified across {total_processed_files} files")
        
        return combined_result
        
    except Exception as e:
        logger.error(f"❌ Batch domain classification failed: {e}")
        return {"success": False, "error": str(e)}

# Test function for standalone usage
def test_domain_classification():
    """Test function to verify domain classification works correctly."""
    test_descriptions = [
        "A person is cooking pasta in a kitchen with various ingredients on the counter",
        "Someone is brushing their teeth in the bathroom mirror",
        "A person is vacuuming the living room carpet",
        "Someone is working on a laptop at a desk with papers scattered around",
        "A family is watching TV together on the couch",
        "A person is doing yoga exercises in the living room"
    ]
    
    print("🧪 Testing domain classification...")
    for desc in test_descriptions:
        domain = classify_scene_domain(desc, DOMAINS)
        print(f"Description: {desc[:50]}...")
        print(f"Classified as: {domain}")
        print("-" * 50)


# domain classification at video level
@ray.remote
def process_video_level_domain_classification(video_output_dir: str, video_name: str) -> Dict:
    """
    Ray task to process ALL shards of a video and perform single video-level domain classification.
    
    This replaces the shard-level domain classification approach.
    
    Args:
        video_output_dir: Base output directory containing all shards for this video
        video_name: Name of the video being processed
        
    Returns:
        Dictionary with video-level domain classification results
    """
    try:
        logger.info(f"🎯 Starting VIDEO-LEVEL domain classification for: {video_name}")
        logger.info(f"📁 Video output directory: {video_output_dir}")
        
        if not os.path.exists(video_output_dir):
            logger.error(f"Video output directory does not exist: {video_output_dir}")
            return {"success": False, "error": "Directory not found"}
        
        # Find all scene detection JSON files across ALL shards
        all_scene_files = []
        all_scene_descriptions = []
        
        # Walk through all shard directories
        for root, dirs, files in os.walk(video_output_dir):
            for file in files:
                if file.endswith('_scene_detection_results.json'):
                    scene_file_path = os.path.join(root, file)
                    all_scene_files.append(scene_file_path)
        
        if not all_scene_files:
            logger.warning(f"No scene detection JSON files found in video directory: {video_output_dir}")
            return {"success": False, "error": "No scene files found"}
        
        logger.info(f"📁 Found {len(all_scene_files)} scene detection files across all shards")
        
        # Collect ALL scene descriptions from ALL shards
        total_scenes = 0
        for scene_file in all_scene_files:
            try:
                with open(scene_file, 'r') as f:
                    scene_data = json.load(f)
                
                scenes = scene_data.get('scenes', [])
                total_scenes += len(scenes)
                
                # Collect scene descriptions with timing info for context
                for scene in scenes:
                    if 'description' in scene:
                        description = scene['description']
                        start_time = scene.get('start_time', 0)
                        end_time = scene.get('end_time', 0)
                        
                        # Add contextual information to scene description
                        contextual_description = f"[{start_time:.1f}s-{end_time:.1f}s]: {description}"
                        all_scene_descriptions.append(contextual_description)
                        
            except Exception as e:
                logger.error(f"Error reading scene file {scene_file}: {e}")
                continue
        
        if not all_scene_descriptions:
            logger.warning(f"No scene descriptions found across all shards")
            return {"success": False, "error": "No scene descriptions found"}
        
        logger.info(f"📝 Collected {len(all_scene_descriptions)} scene descriptions from {total_scenes} total scenes")
        
        # Create consolidated scene description for video-level classification
        consolidated_description = "\n".join(all_scene_descriptions)
        
        # Limit description length for API call (openai/gpt-oss-20b supports 128K tokens ≈ 512K chars)
        max_chars = 400000  # Conservative limit using ~80% of 128K token context window
        if len(consolidated_description) > max_chars:
            logger.warning(f"Scene descriptions too long ({len(consolidated_description)} chars), truncating to {max_chars}")
            consolidated_description = consolidated_description[:max_chars] + "..."
        
        # Classify the entire video using aggregated scene descriptions
        logger.info(f"🤖 Calling Groq API for video-level domain and activity classification with prompt:{consolidated_description}...")
        classification_result = classify_video_level_domain_and_activity(consolidated_description, DOMAINS, video_name)
        
        video_domain = classification_result["domain"]
        video_activity = classification_result["activity"]
        
        # Save video-level domain and activity result
        domain_result_file = os.path.join(video_output_dir, "video_level_domain_activity_classification.json")
        domain_result = {
            "video_name": video_name,
            "predicted_domain": video_domain,
            "predicted_activity": video_activity,
            "total_scene_files": len(all_scene_files),
            "total_scenes": total_scenes,
            "total_descriptions": len(all_scene_descriptions),
            "classification_timestamp": datetime.utcnow().isoformat() + "Z",
            "consolidated_description_preview": consolidated_description[:500] + "..." if len(consolidated_description) > 500 else consolidated_description
        }
        
        with open(domain_result_file, 'w') as f:
            json.dump(domain_result, f, indent=2)
        
        result = {
            "success": True,
            "video_name": video_name,
            "predicted_domain": video_domain,
            "predicted_activity": video_activity,
            "total_scene_files": len(all_scene_files),
            "total_scenes": total_scenes,
            "domain_result_file": domain_result_file
        }
        
        logger.info(f"🎉 Video-level classification completed for '{video_name}': Domain='{video_domain}', Activity='{video_activity}'")
        logger.info(f"📊 Classification summary: {len(all_scene_descriptions)} descriptions from {len(all_scene_files)} files across all shards")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Video-level domain classification failed for {video_name}: {e}")
        return {"success": False, "error": str(e)}


def classify_video_level_domain_and_activity(consolidated_description: str, domains: List[str], video_name: str) -> Dict[str, str]:
    """
    Classify an entire video using consolidated scene descriptions from all shards.
    Extracts both domain and activity from the video content.
    
    Args:
        consolidated_description: All scene descriptions from all shards combined
        domains: List of available domains
        video_name: Name of the video for logging
        
    Returns:
        Dictionary with classified domain and activity: {"domain": str, "activity": str}
    """
    try:
        # Get configuration from loaded config
        groq_config = CONFIG["groq"]
        domain_activity_config = CONFIG["domain_activity_detection"]  # Updated config section
        
        client = Groq(api_key=groq_config["api_key"])
        
        # Create enhanced prompt for video-level classification
        domains_text = "\n".join([f"- {domain}" for domain in domains])
        
        # Use the new video-level prompt template from config
        video_level_prompt = domain_activity_config["video_level_prompt_template"].format(
            domains_list=domains_text,
            video_name=video_name,
            consolidated_description=consolidated_description
        )

        completion = client.chat.completions.create(
            model=groq_config["model"],
            messages=[
                {
                    "role": "user",
                    "content": video_level_prompt
                }
            ],
            temperature=groq_config["temperature"],
            max_tokens=groq_config["max_tokens"],
            top_p=groq_config["top_p"],
            stream=groq_config["stream"],
            stop=None
        )
        
        # Extract the response
        response = completion.choices[0].message.content.strip()
        
        # Parse structured response: "DOMAIN: ... \n ACTIVITY: ..."
        parsed_result = parse_domain_activity_response(response, domains, domain_activity_config)
        
        logger.info(f"✅ Video-level classification successful for '{video_name}': Domain='{parsed_result['domain']}', Activity='{parsed_result['activity']}'")
        return parsed_result
            
    except Exception as e:
        logger.error(f"Error classifying video-level domain and activity for '{video_name}': {e}")
        return {
            "domain": domain_activity_config["default_domain"],
            "activity": domain_activity_config["default_activity"]
        }


def parse_domain_activity_response(response: str, domains: List[str], config: Dict) -> Dict[str, str]:
    """
    Parse the structured response from Groq API to extract domain and activity.
    
    Expected format:
    DOMAIN: Food & Mealtime
    ACTIVITY: Cooking pasta dinner
    
    Args:
        response: Raw response from Groq API
        domains: List of valid domains for validation
        config: Configuration with default values
        
    Returns:
        Dictionary with parsed domain and activity
    """
    try:
        # Initialize with defaults
        result = {
            "domain": config["default_domain"],
            "activity": config["default_activity"]
        }
        
        # Split response into lines
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Parse domain line
            if line.startswith('DOMAIN:'):
                domain_text = line.replace('DOMAIN:', '').strip()
                
                # Validate domain
                if domain_text in domains:
                    result["domain"] = domain_text
                else:
                    # Try approximate match
                    for domain in domains:
                        if domain.lower() in domain_text.lower() or domain_text.lower() in domain.lower():
                            result["domain"] = domain
                            logger.warning(f"Approximate domain match: '{domain_text}' -> '{domain}'")
                            break
            
            # Parse activity line
            elif line.startswith('ACTIVITY:'):
                activity_text = line.replace('ACTIVITY:', '').strip()
                if activity_text:  # Only use if not empty
                    result["activity"] = activity_text
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing domain/activity response: {e}")
        return {
            "domain": config["default_domain"],
            "activity": config["default_activity"]
        }

if __name__ == "__main__":
    # Initialize Ray for testing
    if not ray.is_initialized():
        ray.init()
    
    # Run test
    test_domain_classification()
