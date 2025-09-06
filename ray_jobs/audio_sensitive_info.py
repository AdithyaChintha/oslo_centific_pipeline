import warnings
warnings.filterwarnings("ignore")  # Catch-all for all warnings

import ray
import os
import sys
import json
import requests
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
from utils.logger import get_logger

# Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

def load_config():
    """Load configuration from pipeline_config.yaml"""
    config_path = os.path.join(PROJECT_ROOT, "config", "pipeline_config.yaml")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return None

def get_groq_api_key():
    """Get Groq API key from config file"""
    config = load_config()
    if config and 'api_keys' in config and 'groq_api_key' in config['api_keys']:
        return config['api_keys']['groq_api_key']
    else:
        logger.error("Groq API key not found in config file")
        return None

logger = get_logger("audio_sensitive_info")

def analyze_sensitive_content(transcript: str) -> Dict[str, Any]:
    """
    Analyze transcript for sensitive information using Groq API
    
    Args:
        transcript: The text transcript to analyze
        
    Returns:
        Dict containing analysis results
    """
    if not transcript or not transcript.strip():
        return {
            "has_sensitive_content": False,
            "sensitive_topics": [],
            "analysis": "No sensitive topics",
            "confidence": 1.0,
            "error": None
        }
    
    # Get API key from config
    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        return {
            "has_sensitive_content": False,
            "sensitive_topics": [],
            "analysis": "No sensitive topics",
            "confidence": 0.0,
            "error": "Groq API key not found in config"
        }
    
    # Prepare the prompt for sensitive content detection
    prompt = f"""
Analyze the following audio transcript for sensitive information. Look for content related to:

1. Political opinions
2. Religious beliefs  
3. Gender identity
4. Sexual orientation

Transcript: "{transcript}"

Please respond with a JSON object in this exact format:
{{
    "has_sensitive_content": true/false,
    "sensitive_topics": ["topic1", "topic2", ...],
    "analysis": "Brief explanation of findings",
    "confidence": 0.0-1.0
}}

If no sensitive topics are found, return:
{{
    "has_sensitive_content": false,
    "sensitive_topics": [],
    "analysis": "No sensitive topics",
    "confidence": 1.0
}}

Only return the JSON object, no additional text.
"""

    try:
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        logger.info(f"Sending request to Groq API for sensitive content analysis")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Try to parse JSON response
            try:
                analysis_result = json.loads(content)
                logger.info(f"Successfully analyzed transcript for sensitive content")
                return analysis_result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Groq API response as JSON: {e}")
                logger.error(f"Raw response: {content}")
                return {
                    "has_sensitive_content": False,
                    "sensitive_topics": [],
                    "analysis": "No sensitive topics",
                    "confidence": 0.0,
                    "error": f"JSON parsing error: {str(e)}"
                }
        else:
            logger.error(f"Groq API request failed with status {response.status_code}: {response.text}")
            return {
                "has_sensitive_content": False,
                "sensitive_topics": [],
                "analysis": "No sensitive topics", 
                "confidence": 0.0,
                "error": f"API request failed: {response.status_code}"
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error calling Groq API: {e}")
        return {
            "has_sensitive_content": False,
            "sensitive_topics": [],
            "analysis": "No sensitive topics",
            "confidence": 0.0,
            "error": f"Network error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Unexpected error in sensitive content analysis: {e}")
        return {
            "has_sensitive_content": False,
            "sensitive_topics": [],
            "analysis": "No sensitive topics",
            "confidence": 0.0,
            "error": f"Unexpected error: {str(e)}"
        }

def load_transcript_from_audio_output(audio_output_dir: str, shard_name: str) -> Optional[str]:
    """
    Load transcript from audio_diarization_pii output
    
    Args:
        audio_output_dir: Directory where audio_diarization_pii saved its results
        shard_name: Name of the shard (without extension)
        
    Returns:
        Transcript text or None if not found
    """
    try:
        # Try to load from transcript file first
        transcript_file = os.path.join(audio_output_dir, f"{shard_name}_transcript.txt")
        if os.path.exists(transcript_file):
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
                logger.info(f"Loaded transcript from {transcript_file}")
                return transcript
        
        # Try to load from complete results file
        complete_file = os.path.join(audio_output_dir, f"{shard_name}_complete_results.json")
        if os.path.exists(complete_file):
            with open(complete_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                transcript = data.get("transcript", "").strip()
                if transcript:
                    logger.info(f"Loaded transcript from {complete_file}")
                    return transcript
        
        logger.warning(f"No transcript found for {shard_name} in {audio_output_dir}")
        return None
        
    except Exception as e:
        logger.error(f"Error loading transcript for {shard_name}: {e}")
        return None

@ray.remote
def process_audio_sensitive_info(shard_paths: List[str], audio_output_dir: str, output_dir: str = None):
    """
    Process audio shards for sensitive information detection
    
    Args:
        shard_paths: List of audio/video shard paths
        audio_output_dir: Directory where audio_diarization_pii saved its results
        output_dir: Directory to save results (optional)
        
    Returns:
        List of analysis results
    """
    logger.info(f"Starting sensitive information analysis for {len(shard_paths)} shards")
    results = []
    
    for shard_path in shard_paths:
        try:
            shard_name = os.path.splitext(os.path.basename(shard_path))[0]
            logger.info(f"Processing sensitive info analysis for: {shard_name}")
            start_time = datetime.now()
            
            # Load transcript from audio_diarization_pii output
            transcript = load_transcript_from_audio_output(audio_output_dir, shard_name)
            
            if not transcript:
                logger.warning(f"No transcript available for {shard_name}, skipping sensitive info analysis")
                result = {
                    "shard_path": shard_path,
                    "shard_name": shard_name,
                    "transcript": "",  # Empty transcript when not available
                    "transcript_length": 0,
                    
                    # === MAIN OUTPUTS (matching audio_res structure) ===
                    "sensitive_analysis": {
                        "has_sensitive_content": False,
                        "sensitive_topics": [],
                        "analysis": "No transcript available",
                        "confidence": 0.0,
                        "error": "Transcript not found"
                    },
                    "sensitive_detections": [],
                    
                    # === SUMMARY STATISTICS (matching audio_res structure) ===
                    "summary": {
                        "has_sensitive_content": False,
                        "sensitive_topics_count": 0,
                        "sensitive_topics": [],
                        "confidence": 0.0
                    },
                    
                    # === PROCESSING METADATA (matching audio_res structure) ===
                    "processing_stats": {
                        "total_processing_time": 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "device_used": "cpu"
                    }
                }
                results.append(result)
                continue
            
            # Analyze for sensitive content
            sensitive_analysis = analyze_sensitive_content(transcript)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Compile result with structure matching audio_res
            result = {
                "shard_path": shard_path,
                "shard_name": shard_name,
                "transcript": transcript,  # Include full transcript like audio_res
                "transcript_length": len(transcript),
                
                # === MAIN OUTPUTS (matching audio_res structure) ===
                "sensitive_analysis": sensitive_analysis,
                "sensitive_detections": [],  # Will be populated if sensitive content found
                
                # === SUMMARY STATISTICS (matching audio_res structure) ===
                "summary": {
                    "has_sensitive_content": sensitive_analysis.get("has_sensitive_content", False),
                    "sensitive_topics_count": len(sensitive_analysis.get("sensitive_topics", [])),
                    "sensitive_topics": sensitive_analysis.get("sensitive_topics", []),
                    "confidence": sensitive_analysis.get("confidence", 0.0)
                },
                
                # === PROCESSING METADATA (matching audio_res structure) ===
                "processing_stats": {
                    "total_processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "device_used": "cpu"  # Sensitive analysis runs on CPU via API
                }
            }
            
            # Populate sensitive_detections if sensitive content found
            if sensitive_analysis.get("has_sensitive_content", False):
                result["sensitive_detections"] = [
                    {
                        "topic": topic,
                        "confidence": sensitive_analysis.get("confidence", 0.0),
                        "analysis": sensitive_analysis.get("analysis", ""),
                        "start_time": 0.0,  # Sensitive analysis covers entire transcript
                        "end_time": len(transcript) / 10.0,  # Rough estimate based on transcript length
                        "priority": "high"
                    }
                    for topic in sensitive_analysis.get("sensitive_topics", [])
                ]
            
            results.append(result)
            logger.info(f"Completed sensitive info analysis for {shard_name} in {processing_time:.2f}s")
            
            # Save individual result if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                result_file = os.path.join(output_dir, f"{shard_name}_sensitive_analysis.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Saved result to {result_file}")
                
        except Exception as e:
            logger.error(f"Error processing {shard_path} for sensitive info analysis: {e}")
            # Add error result with structure matching audio_res
            error_result = {
                "shard_path": shard_path,
                "shard_name": os.path.splitext(os.path.basename(shard_path))[0],
                "transcript": "",  # Empty transcript for error case
                "transcript_length": 0,
                
                # === MAIN OUTPUTS (matching audio_res structure) ===
                "sensitive_analysis": {
                    "has_sensitive_content": False,
                    "sensitive_topics": [],
                    "analysis": "Processing error",
                    "confidence": 0.0,
                    "error": str(e)
                },
                "sensitive_detections": [],
                
                # === SUMMARY STATISTICS (matching audio_res structure) ===
                "summary": {
                    "has_sensitive_content": False,
                    "sensitive_topics_count": 0,
                    "sensitive_topics": [],
                    "confidence": 0.0
                },
                
                # === PROCESSING METADATA (matching audio_res structure) ===
                "processing_stats": {
                    "total_processing_time": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "device_used": "cpu"
                }
            }
            results.append(error_result)
            continue
    
    logger.info(f"Completed sensitive information analysis for {len(results)} shards")
    return results

if __name__ == "__main__":
    # Test the sensitive information analysis
    test_transcript = "I believe in democracy and support the current government. My religious views are personal and I identify as non-binary."
    
    print("Testing sensitive information analysis...")
    result = analyze_sensitive_content(test_transcript)
    print("Analysis result:")
    print(json.dumps(result, indent=2))

