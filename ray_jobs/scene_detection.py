
import ray
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import yaml
import pathlib

ROOT = pathlib.Path(__file__).parents[2].resolve()

@ray.remote(num_gpus=1)
def detect_scenes(video_path, prompt_path):
    """
    Detects scenes in a video file using the nvidia/Cosmos-Reason1-7B model.
    """
    llm = LLM(
        model="nvidia/Cosmos-Reason1-7B",
        limit_mm_per_prompt={"image": 0, "video": 1},
        enforce_eager=True,
    )

    with open(prompt_path, "rb") as f:
        prompt_config = yaml.safe_load(f)

    sampling_params = SamplingParams(
        n=1,
        temperature=0.6,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
        max_tokens=4096,
        seed=1,
    )

    messages = [
        {"role": "system", "content": prompt_config.get("system_prompt", "")},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 16,
                    "total_pixels": 8192 * 28 * 28,
                },
                {"type": "text", "text": prompt_config.get("user_prompt", "")},
            ],
        },
    ]
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

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = [output.text for output in outputs[0].outputs]

    return generated_text
