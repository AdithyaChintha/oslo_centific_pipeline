import ray
import gc
import json
import os
import torch
import cv2
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import yaml
import pathlib
import logging
from typing import Optional, List, Dict, Any

ROOT = pathlib.Path(__file__).parents[2].resolve()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_gpu_memory():
    """Aggressive GPU memory cleanup"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            gc.collect()
            logger.info("GPU memory cleared")
    except Exception as e:
        logger.warning(f"GPU memory cleanup failed: {e}")

def check_gpu_memory(required_gb: float = 15.0) -> tuple[bool, float]:
    """Check available GPU memory"""
    try:
        if not torch.cuda.is_available():
            logger.info("CUDA not available, will use CPU fallback")
            return False, 0.0
        
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - allocated_memory
        free_gb = free_memory / (1024**3)
        
        logger.info(f"GPU Memory: {free_gb:.1f}GB free / {total_memory/(1024**3):.1f}GB total")
        return free_gb >= required_gb, free_gb
    except Exception as e:
        logger.warning(f"GPU memory check failed: {e}")
        return False, 0.0

def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get comprehensive video information"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "is_8k": width >= 7680 and height >= 4320,
            "is_4k": width >= 3840 and height >= 2160,
            "is_hd": width >= 1280 and height >= 720
        }
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {
            "duration": 60.0,
            "fps": 30.0,
            "frame_count": 1800,
            "width": 1920,
            "height": 1080,
            "resolution": "1920x1080",
            "is_8k": False,
            "is_4k": False,
            "is_hd": True
        }

def analyze_scene_content(scene_text: str) -> Dict[str, Any]:
    """Analyze scene content for flagging criteria"""
    if not scene_text or len(scene_text.strip()) < 10:
        return {
            'should_flag': True,
            'priority': 'low',
            'flag_type': 'insufficient_analysis',
            'matched_keywords': [],
            'confidence': 0.2,
            'reason': "Content requires manual review - insufficient analysis"
        }
    
    scene_lower = scene_text.lower()
    
    # Enhanced keywords for better flagging
    high_priority_keywords = [
        'violence', 'fight', 'fighting', 'weapon', 'blood', 'inappropriate', 
        'nude', 'naked', 'sexual', 'drugs', 'alcohol', 'smoking', 'dangerous',
        'accident', 'injury', 'emergency', 'fire', 'explosion', 'threat'
    ]
    
    medium_priority_keywords = [
        'crowd', 'party', 'gathering', 'meeting', 'activity', 'movement',
        'multiple people', 'group', 'interaction', 'conversation', 'emotional',
        'celebration', 'event', 'performance', 'dancing', 'singing', 'playing',
        'cleaning', 'vacuuming', 'working', 'exercise', 'cooking'
    ]
    
    low_priority_keywords = [
        'person', 'people', 'walking', 'talking', 'sitting', 'standing',
        'room', 'house', 'indoor', 'outdoor', 'furniture', 'objects'
    ]
    
    # Check for matches
    high_matches = [kw for kw in high_priority_keywords if kw in scene_lower]
    medium_matches = [kw for kw in medium_priority_keywords if kw in scene_lower]
    low_matches = [kw for kw in low_priority_keywords if kw in scene_lower]
    
    if high_matches:
        return {
            'should_flag': True,
            'priority': 'high',
            'flag_type': 'concerning_content',
            'matched_keywords': high_matches,
            'confidence': 0.9,
            'reason': f"High-priority content detected: {', '.join(high_matches)}"
        }
    elif medium_matches:
        return {
            'should_flag': True,
            'priority': 'medium',
            'flag_type': 'notable_activity',
            'matched_keywords': medium_matches,
            'confidence': 0.7,
            'reason': f"Notable activity detected: {', '.join(medium_matches)}"
        }
    elif low_matches:
        return {
            'should_flag': True,
            'priority': 'low',
            'flag_type': 'general_activity',
            'matched_keywords': low_matches,
            'confidence': 0.5,
            'reason': f"General activity detected: {', '.join(low_matches)}"
        }
    else:
        return {
            'should_flag': True,
            'priority': 'low',
            'flag_type': 'unknown_content',
            'matched_keywords': [],
            'confidence': 0.3,
            'reason': "Content requires manual review"
        }

@ray.remote(num_gpus=1, max_retries=0)
def detect_scenes(video_path: str, prompt_path: str, output_dir: str = None) -> Optional[Dict[str, Any]]:
    """
    GPU-based scene detection with optimized vLLM configuration
    """
    llm = None
    try:
        logger.info(f"Starting scene detection for: {os.path.basename(video_path)}")
        
        # Get video information
        video_info = get_video_info(video_path)
        logger.info(f"Video info: {video_info['resolution']}, {video_info['duration']:.1f}s, {video_info['fps']:.1f} FPS")
        
        # Memory check
        can_run_gpu, available_gb = check_gpu_memory(required_gb=15.0)
        if not can_run_gpu:
            raise RuntimeError(f"Insufficient GPU memory: {available_gb:.1f}GB < 15GB required")
        
        # Calculate optimal settings based on video resolution
        if video_info.get('is_8k'):
            # 8K video - very conservative settings
            gpu_memory_util = 0.15
            total_pixels = 512 * 256  # ~131K pixels
            max_model_len = 512
            max_num_seqs = 16  # Very small for 8K
            processing_fps = 0.5  # Very slow for 8K
        elif video_info.get('is_4k'):
            # 4K video - conservative settings  
            gpu_memory_util = 0.25
            total_pixels = 640 * 320  # ~205K pixels
            max_model_len = 768
            max_num_seqs = 32
            processing_fps = 1
        else:
            # HD video - standard settings
            gpu_memory_util = 0.35
            total_pixels = 768 * 432  # ~332K pixels
            max_model_len = 1024
            max_num_seqs = 64
            processing_fps = 2
        
        logger.info(f"Optimized settings: GPU {int(gpu_memory_util*100)}%, Pixels: {total_pixels:,}, FPS: {processing_fps}")
        
        clear_gpu_memory()
        
        # CRITICAL FIX: Ensure max_num_batched_tokens >= max_num_seqs
        max_num_batched_tokens = max(max_num_seqs * 4, 1024)
        
        logger.info(f"vLLM config: max_num_seqs={max_num_seqs}, max_num_batched_tokens={max_num_batched_tokens}")
        
        llm = LLM(
            model="nvidia/Cosmos-Reason1-7B",
            limit_mm_per_prompt={"image": 0, "video": 1},
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_util,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_num_batched_tokens=max_num_batched_tokens,  # >= max_num_seqs
            max_num_seqs=max_num_seqs,
            disable_log_stats=True,
            swap_space=0,
        )

        # Load prompt configuration with fallback
        try:
            with open(prompt_path, "rb") as f:
                prompt_config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load prompt config: {e}, using defaults")
            prompt_config = {
                "system_prompt": "You are an expert video analyst. Describe the video content in detail.",
                "user_prompt": "Analyze this video and describe what you see, including people, activities, objects, and setting."
            }

        # Enhanced sampling parameters for better descriptions
        sampling_params = SamplingParams(
            n=1,
            temperature=0.7,
            top_k=30,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=800,  # Increased for detailed descriptions
            seed=42,
        )

        # Create detailed prompt messages
        system_prompt = prompt_config.get("system_prompt", 
            "You are an expert video content analyst. Your task is to provide detailed, accurate descriptions "
            "of video content. Focus on identifying people, their activities, objects, settings, interactions, "
            "and any notable events. Be specific and descriptive.")
        
        user_prompt = prompt_config.get("user_prompt",
            "Please analyze this video carefully and provide a detailed description including: "
            "1. The setting/environment (indoor/outdoor, type of room, etc.) "
            "2. People present (number, activities, interactions) "
            "3. Objects and items visible "
            "4. Actions and activities taking place "
            "5. Overall mood or atmosphere "
            "Be specific and detailed in your observations.")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": processing_fps,
                        "total_pixels": total_pixels,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # Process video with the model
        logger.info("Processing video with Cosmos model...")
        processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason1-7B")
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        mm_data = {}
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        # Generate scene description
        outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = [output.text for output in outputs[0].outputs]
        
        # Analyze scene content for flagging
        full_scene_text = ' '.join(generated_text).strip()
        analysis = analyze_scene_content(full_scene_text)
        
        logger.info(f"Generated description: {len(full_scene_text)} characters")
        if full_scene_text:
            preview = full_scene_text[:200] + ("..." if len(full_scene_text) > 200 else "")
            logger.info(f"Preview: {preview}")
        
        # Create comprehensive result
        scene_result = {
            "video_path": video_path,
            "video_info": video_info,
            "scene_description": generated_text,
            "full_scene_text": full_scene_text,
            "analysis": analysis,
            "flagged_segments": [],
            "processing_info": {
                "model": "nvidia/Cosmos-Reason1-7B",
                "success": True,
                "gpu_memory_utilization": gpu_memory_util,
                "processing_fps": processing_fps,
                "total_pixels": total_pixels,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_num_batched_tokens
            }
        }
        
        # Create flagged segment if content should be flagged
        if analysis['should_flag']:
            flagged_segment = {
                "start_time": 0.0,
                "end_time": video_info['duration'],
                "task_type": "scene_detection",
                "confidence": analysis['confidence'],
                "flag_type": analysis['flag_type'],
                "priority": analysis['priority'],
                "description": analysis['reason'],
                "metadata": {
                    "scene_description": generated_text,
                    "matched_keywords": analysis['matched_keywords'],
                    "full_analysis": analysis
                }
            }
            scene_result["flagged_segments"].append(flagged_segment)
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            scene_file = os.path.join(output_dir, "scene_detections.json")
            with open(scene_file, 'w', encoding='utf-8') as f:
                json.dump(scene_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {scene_file}")

        return scene_result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Scene detection failed: {error_msg}")
        
        # Return error result for debugging
        return {
            "video_path": video_path,
            "video_info": get_video_info(video_path),
            "scene_description": [f"Scene detection failed: {error_msg}"],
            "full_scene_text": f"Processing error: {error_msg}",
            "analysis": {
                'should_flag': True,
                'priority': 'error',
                'flag_type': 'processing_error',
                'matched_keywords': [],
                'confidence': 0.0,
                'reason': f"Processing error: {error_msg}"
            },
            "flagged_segments": [],
            "processing_info": {
                "model": "nvidia/Cosmos-Reason1-7B",
                "success": False,
                "error": error_msg
            }
        }
        
    finally:
        # Cleanup
        if llm is not None:
            try:
                del llm
            except Exception:
                pass
        clear_gpu_memory()

@ray.remote(num_cpus=4, max_retries=0)
def detect_scenes_cpu_fallback(video_path: str, prompt_path: str, output_dir: str = None) -> Optional[Dict[str, Any]]:
    """
    Enhanced CPU fallback with basic video frame analysis
    """
    try:
        logger.info(f"CPU fallback processing: {os.path.basename(video_path)}")
        
        video_info = get_video_info(video_path)
        
        # Enhanced frame-based analysis
        scene_descriptions = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Sample more frames for better analysis
                sample_indices = [
                    int(frame_count * 0.1),   # 10%
                    int(frame_count * 0.3),   # 30%
                    int(frame_count * 0.5),   # 50%
                    int(frame_count * 0.7),   # 70% 
                    int(frame_count * 0.9)    # 90%
                ]
                
                frame_analyses = []
                for i, frame_idx in enumerate(sample_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # Basic frame analysis
                        height, width = frame.shape[:2]
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        brightness = gray.mean()
                        
                        # Simple color analysis
                        b, g, r = cv2.split(frame)
                        color_balance = {
                            'blue_avg': b.mean(),
                            'green_avg': g.mean(), 
                            'red_avg': r.mean()
                        }
                        
                        # Determine scene characteristics
                        if brightness > 120:
                            lighting = "bright, well-lit"
                        elif brightness > 80:
                            lighting = "moderately lit"
                        elif brightness > 40:
                            lighting = "dim"
                        else:
                            lighting = "dark or low-light"
                            
                        # Simple motion estimation (edge density)
                        edges = cv2.Canny(gray, 50, 150)
                        edge_density = (edges > 0).sum() / edges.size
                        
                        if edge_density > 0.1:
                            complexity = "complex scene with many details"
                        elif edge_density > 0.05:
                            complexity = "moderate detail"
                        else:
                            complexity = "simple scene"
                        
                        timestamp = frame_idx / video_info['fps']
                        frame_analyses.append(
                            f"At {timestamp:.1f}s: {lighting} environment with {complexity}. "
                            f"Resolution {width}x{height}."
                        )
                
                cap.release()
                
                if frame_analyses:
                    scene_descriptions = [
                        f"CPU-based analysis of {os.path.basename(video_path)} "
                        f"({video_info['duration']:.1f}s duration, {video_info['resolution']} resolution):",
                        "",
                        "Frame-by-frame analysis:",
                        *frame_analyses,
                        "",
                        "Note: This is basic frame analysis. For detailed content recognition "
                        "including object/person detection and activity analysis, GPU processing is required."
                    ]
                else:
                    scene_descriptions = [
                        f"Basic analysis of {os.path.basename(video_path)}: "
                        f"{video_info['resolution']} resolution video, {video_info['duration']:.1f} seconds duration. "
                        f"Frame sampling failed - manual review recommended."
                    ]
            else:
                scene_descriptions = [
                    f"Video file analysis: {os.path.basename(video_path)} "
                    f"({video_info['resolution']}, {video_info['duration']:.1f}s). "
                    f"Could not access video frames for detailed analysis."
                ]
                
        except Exception as frame_error:
            logger.warning(f"Frame analysis failed: {frame_error}")
            scene_descriptions = [
                f"Limited CPU analysis of {os.path.basename(video_path)}. "
                f"Video properties: {video_info['resolution']}, {video_info['duration']:.1f}s duration. "
                f"Detailed content analysis requires GPU processing."
            ]
        
        full_scene_text = '\n'.join(scene_descriptions)
        analysis = analyze_scene_content(full_scene_text)
        
        # Always flag CPU results for manual review
        analysis.update({
            'should_flag': True,
            'priority': 'low',
            'flag_type': 'cpu_analysis',
            'confidence': 0.3,
            'reason': 'CPU fallback analysis - limited accuracy, manual review recommended'
        })
        
        scene_result = {
            "video_path": video_path,
            "video_info": video_info,
            "scene_description": scene_descriptions,
            "full_scene_text": full_scene_text,
            "analysis": analysis,
            "flagged_segments": [{
                "start_time": 0.0,
                "end_time": video_info['duration'],
                "task_type": "scene_detection",
                "confidence": 0.3,
                "flag_type": "cpu_analysis",
                "priority": "low",
                "description": "CPU fallback analysis - limited accuracy",
                "metadata": {
                    "cpu_analysis": True,
                    "frame_samples": len([d for d in scene_descriptions if "At " in d])
                }
            }],
            "processing_info": {
                "model": "CPU_FALLBACK",
                "success": True,
                "note": "Basic frame analysis - GPU recommended for detailed content recognition"
            }
        }
        
        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            scene_file = os.path.join(output_dir, "scene_detections.json")
            with open(scene_file, 'w', encoding='utf-8') as f:
                json.dump(scene_result, f, indent=2, ensure_ascii=False)
            logger.info(f"CPU results saved to: {scene_file}")
        
        return scene_result
        
    except Exception as e:
        logger.error(f"CPU scene detection failed: {e}")
        return None

@ray.remote(num_gpus=1)
def detect_scenes_with_fallback(video_path: str, prompt_path: str, output_dir: str = None) -> Optional[Dict[str, Any]]:
    """
    Scene detection with automatic GPU->CPU fallback
    """
    try:
        # Check GPU availability first
        can_run_gpu, available_gb = check_gpu_memory(required_gb=15.0)
        
        if can_run_gpu:
            logger.info("Attempting GPU scene detection...")
            try:
                result = ray.get(detect_scenes.remote(video_path, prompt_path, output_dir))
                if result and result.get('processing_info', {}).get('success', False):
                    logger.info("GPU scene detection successful")
                    return result
                else:
                    logger.warning("GPU detection failed, using CPU fallback")
            except Exception as gpu_error:
                logger.warning(f"GPU detection failed: {gpu_error}, using CPU fallback")
        else:
            logger.info(f"Insufficient GPU memory ({available_gb:.1f}GB < 15GB), using CPU fallback")

        # Fallback to CPU version
        logger.info("Using CPU fallback...")
        result = ray.get(detect_scenes_cpu_fallback.remote(video_path, prompt_path, output_dir))
        if result:
            logger.info("CPU fallback completed")
        return result

    except Exception as e:
        logger.error(f"All scene detection methods failed: {e}")
        return None