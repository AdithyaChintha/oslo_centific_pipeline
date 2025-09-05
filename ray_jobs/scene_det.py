import ray
import torch
import gc
import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import yaml
import pathlib

ROOT = pathlib.Path(__file__).parents[2].resolve()

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

@ray.remote(num_gpus=1, max_calls=1, max_retries=0)
def detect_scenes(video_path, prompt_path, output_dir=None):
    """
    Detects scenes in a video file using the nvidia/Cosmos-Reason1-7B model.
    Fixed to properly save results to output directory.
    
    Args:
        video_path: Path to the video file
        prompt_path: Path to the prompt configuration YAML file  
        output_dir: Output directory to save results
    """
    llm = None
    
    try:
        # Clear GPU memory before starting
        clear_gpu_memory()
        
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # FIXED: Use working vLLM configuration with larger context
        llm = LLM(
            model="nvidia/Cosmos-Reason1-7B",
            limit_mm_per_prompt={"image": 0, "video": 1},
            enforce_eager=True,
            gpu_memory_utilization=0.25,     # REDUCED: Lower GPU memory for larger context
            max_model_len=6144,              # INCREASED: 6K tokens for video shards (was 4096)
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_num_batched_tokens=3072,     # INCREASED: Match larger context
            max_num_seqs=16,                 # REDUCED: Fewer sequences for larger context
            disable_log_stats=True,          # ADDED: Reduce overhead
            swap_space=0,                    # ADDED: Disable CPU offloading
        )

        # Load prompt configuration
        with open(prompt_path, "rb") as f:
            prompt_config = yaml.safe_load(f)

        # FIXED: Reasonable sampling parameters
        sampling_params = SamplingParams(
            n=1,
            temperature=0.7,                 # FIXED: Slightly higher for better variety
            top_k=20,                        # FIXED: Reduced from 50 for better quality
            top_p=0.9,                       # FIXED: Reduced from 0.95 for better quality
            repetition_penalty=1.1,          # FIXED: Slightly higher to reduce repetition
            max_tokens=800,                  # FIXED: Reduced from 4096 to save context space
            seed=42,                         # FIXED: Use consistent seed
        )

        # FIXED: Use reasonable video processing parameters
        messages = [
            {"role": "system", "content": prompt_config.get("system_prompt", "")},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": 0.5,                  # REDUCED: Even lower FPS for shards
                        "total_pixels": 480 * 270,  # REDUCED: Smaller resolution for shards
                    },
                    {"type": "text", "text": prompt_config.get("user_prompt", "")},
                ],
            },
        ]

        # Process with AutoProcessor
        processor = AutoProcessor.from_pretrained("nvidia/Cosmos-Reason1-7B")
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process vision info
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )

        # Prepare multimodal data
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
        
        # FIXED: Parse and extract structured data from JSON response
        full_scene_text = ' '.join(generated_text).strip()
        
        # Extract and parse JSON if present
        parsed_scenes = []
        try:
            # Remove markdown backticks and extract JSON
            import re
            
            # Find JSON content between ```json and ```
            json_match = re.search(r'```json\s*\n(.*?)\n```', full_scene_text, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                parsed_data = json.loads(json_content)
                
                # Extract scenes with start/end times and descriptions
                if isinstance(parsed_data, list):
                    for item in parsed_data:
                        scene_entry = {
                            "start_time": float(item.get("start_time", 0.0)),
                            "end_time": float(item.get("end_time", 60.0)),
                            "description": str(item.get("scene", item.get("caption", "No description available")))
                        }
                        parsed_scenes.append(scene_entry)
                
                print(f"Successfully parsed {len(parsed_scenes)} scenes from JSON response")
            else:
                # Fallback: treat as plain text description
                parsed_scenes = [{
                    "start_time": 0.0,
                    "end_time": 60.0,  # Default duration for 60-second shard
                    "description": full_scene_text
                }]
                print("No JSON found, using full text as single scene")
                
        except Exception as e:
            print(f"Failed to parse JSON response: {e}")
            # Fallback: treat as plain text
            parsed_scenes = [{
                "start_time": 0.0,
                "end_time": 60.0,
                "description": full_scene_text
            }]
        
        # Create clean result with extracted scenes
        result = {
            "video_path": video_path,
            "scenes": parsed_scenes,  # CLEAN: Extracted scenes with times
            "raw_response": full_scene_text,  # Keep original for debugging
            "total_scenes": len(parsed_scenes),
            "processing_info": {
                "model": "nvidia/Cosmos-Reason1-7B",
                "success": True,
                "gpu_memory_utilization": 0.25,
                "max_model_len": 6144,
                "scenes_extracted": len(parsed_scenes)
            }
        }

        # FIXED: Save results to output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base name from video path (without extension)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            result_file = os.path.join(output_dir, f"{video_name}_scene_detection_results.json")
            
            try:
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Scene detection results saved to {result_file}")
                
            except Exception as e:
                print(f"Warning: Failed to save scene detection results: {e}")

        return result

    except Exception as e:
        print(f"Scene detection failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error result
        error_result = {
            "video_path": video_path,
            "scenes": [{
                "start_time": 0.0,
                "end_time": 60.0,
                "description": f"Error: Scene detection failed - {str(e)}"
            }],
            "raw_response": f"Error: Scene detection failed - {str(e)}",
            "total_scenes": 0,
            "processing_info": {
                "model": "nvidia/Cosmos-Reason1-7B", 
                "success": False,
                "error": str(e)
            }
        }
        
        # FIXED: Save error result to output directory if provided
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                error_file = os.path.join(output_dir, f"{video_name}_scene_detection_error.json")
                with open(error_file, 'w') as f:
                    json.dump(error_result, f, indent=2)
                print(f"Scene detection error saved to {error_file}")
            except Exception as save_error:
                print(f"Failed to save error result: {save_error}")
        
        return error_result
        
    finally:
        # ADDED: Proper cleanup
        if llm is not None:
            try:
                del llm
            except Exception:
                pass
        clear_gpu_memory()