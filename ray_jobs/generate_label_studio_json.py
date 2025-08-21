import os
import json
import ray
from glob import glob

@ray.remote
def generate_label_studio_json(output_dir: str, azure_video_url: str):
    task_list = []

    shard_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith("shard_")])

    for shard in shard_dirs:
        shard_path = os.path.join(output_dir, shard)
        shard_index = int(shard.split("_")[-1])
        segment_results = []

        for model_name in ["scene_output", "audio_output", "yolo_output", "nsfw_output", "motion_output", "face_output"]:
            model_path = os.path.join(shard_path, model_name)
            if not os.path.exists(model_path):
                continue
            for file in glob(os.path.join(model_path, "*.json")):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)

                        if "flagged_segments" in data:
                            for seg in data["flagged_segments"]:
                                segment_results.append({
                                    "start": seg.get("start_time", 0),
                                    "end": seg.get("end_time", 0),
                                    "labels": [seg.get("flag_type", model_name)]
                                })

                        elif "scenes" in data:
                            for seg in data["scenes"]:
                                segment_results.append({
                                    "start": seg.get("start_time", 0),
                                    "end": seg.get("end_time", 60),
                                    "labels": ["scene"]
                                })

                        elif "segments" in data:
                            for seg in data["segments"]:
                                label = seg.get("activity_type", "motion")
                                segment_results.append({
                                    "start": seg.get("start_time", 0),
                                    "end": seg.get("end_time", 0),
                                    "labels": [label]
                                })

                except Exception as e:
                    print(f"Error loading {file}: {e}")

        result_entries = [
            {
                "from_name": "label",
                "to_name": "video",
                "type": "labels",
                "value": {
                    "start": seg["start"],
                    "end": seg["end"],
                    "labels": seg["labels"]
                }
            } for seg in segment_results
        ]

        task_list.append({
            "data": {
                "video_url": azure_video_url,
                "metadata": {
                    "home_id": "TBD",
                    "datetime": "TBD",
                    "domain": "TBD",
                    "actions": ["TBD"]
                }
            },
            "annotations": [],
            "predictions": [
                {
                    "result": result_entries
                }
            ]
        })

    save_path = os.path.join(output_dir, "label_studio_tasks.json")
    with open(save_path, "w") as f:
        json.dump(task_list, f, indent=2)
    return save_path
