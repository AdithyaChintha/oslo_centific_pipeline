import time
import threading
from datetime import datetime
from typing import Dict, Optional, Any
from contextlib import contextmanager

class PipelineTimer:
    """Central timing manager for pipeline operations"""

    def __init__(self):
        self.timings: Dict[str, Any] = {}
        self.active_timers: Dict[str, float] = {}
        self.lock = threading.Lock()

    def start_timer(self, operation_id: str, metadata: Optional[Dict] = None):
        """Start timing an operation"""
        with self.lock:
            start_time = time.time()
            self.active_timers[operation_id] = start_time

            # Initialize timing record
            self.timings[operation_id] = {
                "start_time": datetime.utcnow().isoformat() + "Z",
                "start_timestamp": start_time,
                "metadata": metadata or {},
                "status": "running"
            }

    def end_timer(self, operation_id: str, status: str = "completed",
                  result_metadata: Optional[Dict] = None):
        """End timing an operation"""
        with self.lock:
            if operation_id not in self.active_timers:
                raise ValueError(f"Timer {operation_id} not started")

            end_time = time.time()
            start_time = self.active_timers.pop(operation_id)
            execution_time = end_time - start_time

            self.timings[operation_id].update({
                "end_time": datetime.utcnow().isoformat() + "Z",
                "end_timestamp": end_time,
                "execution_time": execution_time,
                "status": status
            })

            if result_metadata:
                self.timings[operation_id]["result_metadata"] = result_metadata

    @contextmanager
    def time_operation(self, operation_id: str, metadata: Optional[Dict] = None):
        """Context manager for timing operations"""
        self.start_timer(operation_id, metadata)
        try:
            yield
            self.end_timer(operation_id, "completed")
        except Exception as e:
            self.end_timer(operation_id, "failed", {"error": str(e)})
            raise

    def get_timing_data(self) -> Dict:
        """Get all timing data"""
        with self.lock:
            return self.timings.copy()



def generate_hierarchical_timing_structure(raw_timing_data: Dict, session_id: str,
                                         total_session_time: float, session_start_timestamp: float,
                                         pipeline_result: Dict) -> Dict:
    """
    Convert flat timing data to hierarchical structure for blob_polling mode:
    Session → Videos → Video Processing + Shards → Views → Models
    """
    from datetime import datetime
    import time

    session_start_time = datetime.utcfromtimestamp(session_start_timestamp).isoformat() + "Z"
    session_end_time = datetime.utcnow().isoformat() + "Z"

    # Initialize hierarchical structure for blob_polling
    hierarchical_data = {
        "session_id": session_id,
        "total_session_time": round(total_session_time, 2),
        "session_start_time": session_start_time,
        "session_end_time": session_end_time,
        "processing_mode": "blob_polling",
        "video_count": 0,
        "infrastructure_times": {},
        "videos": []
    }

    # Group timing data by video
    video_data = {}  # video_name -> video_info

    for operation_id, timing in raw_timing_data.items():
        parts = operation_id.split('.')

        # Parse operation_id patterns for blob_polling:
        # - "session.download" for session-level download
        # - "video.{session_id}.{video_id}.{operation}" for video-level operations (NEW)
        # - "video.{session_id}.{video_id}.shard.{index}.{view}.{model}" for model operations (NEW)
        # - "video.{video_name}.{operation}" for video-level operations (LEGACY)
        # - "video.{video_name}.shard.{index}.{view}.{model}" for model operations (LEGACY)

        if operation_id == "session.download":
            # Session-level download
            hierarchical_data["infrastructure_times"]["session_download"] = {
                "execution_time": round(timing.get("execution_time", 0), 2),
                "start_time": timing.get("start_time"),
                "end_time": timing.get("end_time"),
                "timing_id": operation_id,
                "status": timing.get("status", "completed")
            }

        elif len(parts) >= 3 and parts[0] == "video":
            # Determine if this is the new 4-part pattern or legacy 3-part pattern
            if (len(parts) >= 4 and parts[3] in ["unwarping", "video_sharding", "audio_sharding", "domain_classification", "upload"]) or (len(parts) >= 7 and parts[3] == "shard"):
                # NEW PATTERN: video.{session_id}.{video_id}.{operation} or video.{session_id}.{video_id}.shard.{index}.{view}.{model}
                session_id_part = parts[1]
                video_id = parts[2]
                video_name = f"{session_id_part}-{video_id}"  # Combine for unique identifier

                # Initialize video data if not exists
                if video_name not in video_data:
                    video_data[video_name] = {
                        "video_name": video_name,
                        "video_id": video_id,
                        "session_id": session_id_part,
                        "total_video_time": 0,
                        "video_start_time": None,
                        "video_end_time": None,
                        "shard_count": 0,
                        "video_processing": {},
                        "shards": {}
                    }

                if len(parts) == 4:
                    # Video-level operation: "video.{session_id}.{video_id}.{operation}"
                    operation = parts[3]
                    video_data[video_name]["video_processing"][operation] = {
                        "execution_time": round(timing.get("execution_time", 0), 2),
                        "start_time": timing.get("start_time"),
                        "end_time": timing.get("end_time"),
                        "timing_id": operation_id,
                        "status": timing.get("status", "completed")
                    }

                elif len(parts) >= 7 and parts[3] == "shard":
                    # Model-level timing: "video.{session_id}.{video_id}.shard.{index}.{view}.{model}"
                    shard_index = int(parts[4])
                    view_name = parts[5]
                    model_name = parts[6]

                    # Initialize shard data
                    if shard_index not in video_data[video_name]["shards"]:
                        video_data[video_name]["shards"][shard_index] = {
                            "shard_index": shard_index,
                            "shard_duration": 60.0,  # Default, can be extracted from config
                            "total_shard_time": 0,
                            "shard_start_time": timing.get("start_time"),
                            "shard_end_time": timing.get("end_time"),
                            "views": {}
                        }

                    # Initialize view data
                    if view_name not in video_data[video_name]["shards"][shard_index]["views"]:
                        video_data[video_name]["shards"][shard_index]["views"][view_name] = {
                            "total_view_time": 0,
                            "models": {}
                        }

                    # Add model data
                    video_data[video_name]["shards"][shard_index]["views"][view_name]["models"][model_name] = {
                        "execution_time": round(timing.get("execution_time", 0), 2),
                        "start_time": timing.get("start_time"),
                        "end_time": timing.get("end_time"),
                        "timing_id": operation_id,
                        "status": timing.get("status", "completed")
                    }

            else:
                # LEGACY PATTERN: video.{video_name}.{operation} or video.{video_name}.shard.{index}.{view}.{model}
                video_name = parts[1]

                # Initialize video data if not exists
                if video_name not in video_data:
                    video_data[video_name] = {
                        "video_name": video_name,
                        "total_video_time": 0,
                        "video_start_time": None,
                        "video_end_time": None,
                        "shard_count": 0,
                        "video_processing": {},
                        "shards": {}
                    }

                if len(parts) == 3:
                    # Video-level operation: "video.{video_name}.{operation}"
                    operation = parts[2]
                    video_data[video_name]["video_processing"][operation] = {
                        "execution_time": round(timing.get("execution_time", 0), 2),
                        "start_time": timing.get("start_time"),
                        "end_time": timing.get("end_time"),
                        "timing_id": operation_id,
                        "status": timing.get("status", "completed")
                    }

                elif len(parts) >= 6 and parts[2] == "shard":
                    # Model-level timing: "video.{video_name}.shard.{index}.{view}.{model}"
                    shard_index = int(parts[3])
                    view_name = parts[4]
                    model_name = parts[5]

                    # Initialize shard data
                    if shard_index not in video_data[video_name]["shards"]:
                        video_data[video_name]["shards"][shard_index] = {
                            "shard_index": shard_index,
                            "shard_duration": 60.0,  # Default, can be extracted from config
                            "total_shard_time": 0,
                            "shard_start_time": timing.get("start_time"),
                            "shard_end_time": timing.get("end_time"),
                            "views": {}
                        }

                    # Initialize view data
                    if view_name not in video_data[video_name]["shards"][shard_index]["views"]:
                        video_data[video_name]["shards"][shard_index]["views"][view_name] = {
                            "total_view_time": 0,
                            "models": {}
                        }

                    # Add model timing data
                    video_data[video_name]["shards"][shard_index]["views"][view_name]["models"][model_name] = {
                        "execution_time": round(timing.get("execution_time", 0), 2),
                        "start_time": timing.get("start_time"),
                        "end_time": timing.get("end_time"),
                        "timing_id": operation_id,
                        "status": timing.get("status", "completed")
                    }

    # Calculate aggregated times for each video
    for video_name, video_info in video_data.items():
        video_start_times = []
        video_end_times = []
        video_processing_time = 0
        shard_processing_time = 0

        # Calculate video processing times (unwarping, sharding, consolidation, upload)
        for operation, timing_data in video_info["video_processing"].items():
            video_processing_time += timing_data.get("execution_time", 0)
            if timing_data.get("start_time"):
                video_start_times.append(timing_data["start_time"])
            if timing_data.get("end_time"):
                video_end_times.append(timing_data["end_time"])

        # Process each shard in the video
        shard_list = []
        for shard_index in sorted(video_info["shards"].keys()):
            shard_data = video_info["shards"][shard_index]

            # Calculate view times for this shard
            for view_name, view_data in shard_data["views"].items():
                model_times = []

                for model_name, model_data in view_data["models"].items():
                    model_time = model_data["execution_time"]
                    model_times.append(model_time)

                    # Collect start/end times for video aggregation
                    if model_data["start_time"]:
                        video_start_times.append(model_data["start_time"])
                    if model_data["end_time"]:
                        video_end_times.append(model_data["end_time"])

                # For parallel execution, view time = max(model_times)
                view_data["total_view_time"] = round(max(model_times) if model_times else 0, 2)

            # For parallel execution across views, shard time = max(view_times)
            view_times = [view_data["total_view_time"] for view_data in shard_data["views"].values()]
            shard_data["total_shard_time"] = round(max(view_times) if view_times else 0, 2)

            # Add to shard processing time (shards can run in parallel, but we track max)
            shard_processing_time = max(shard_processing_time, shard_data["total_shard_time"])

            shard_list.append(shard_data)

        # Update video metadata
        video_info["shards"] = shard_list
        video_info["shard_count"] = len(shard_list)

        # Total video time = video processing time + shard processing time
        video_info["total_video_time"] = round(video_processing_time + shard_processing_time, 2)

        # Set video start/end times from earliest/latest operation times
        if video_start_times:
            video_info["video_start_time"] = min(video_start_times)
        if video_end_times:
            video_info["video_end_time"] = max(video_end_times)

        hierarchical_data["videos"].append(video_info)

    # Update session metadata
    hierarchical_data["video_count"] = len(video_data)

    return hierarchical_data

def generate_performance_summary(hierarchical_timing_data: Dict) -> Dict:
    """
    Calculate performance metrics from hierarchical timing data for blob_polling mode
    """

    # Extract model timings from hierarchical structure
    model_times = {}
    shard_times = []
    video_times = []
    all_execution_times = []

    for video in hierarchical_timing_data.get("videos", []):
        video_time = video.get("total_video_time", 0)
        video_times.append(video_time)

        for shard in video.get("shards", []):
            shard_time = shard.get("total_shard_time", 0)
            shard_times.append(shard_time)

            for view_name, view_data in shard.get("views", {}).items():
                for model_name, model_data in view_data.get("models", {}).items():
                    execution_time = model_data.get("execution_time", 0)
                    all_execution_times.append(execution_time)

                    # Collect model times
                    if model_name not in model_times:
                        model_times[model_name] = []
                    model_times[model_name].append(execution_time)

    # Calculate video-level metrics
    avg_video_time = sum(video_times) / len(video_times) if video_times else 0

    # Calculate shard-level metrics
    avg_shard_time = sum(shard_times) / len(shard_times) if shard_times else 0

    # Calculate model-level metrics
    model_stats = {}
    for model_name, times in model_times.items():
        model_stats[model_name] = {
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "execution_count": len(times),
            "total_time": sum(times)
        }

    avg_model_time = sum(all_execution_times) / len(all_execution_times) if all_execution_times else 0

    # Find slowest and fastest models
    slowest_model = None
    fastest_model = None

    if model_stats:
        slowest_model = max(model_stats.keys(), key=lambda k: model_stats[k]["average_time"])
        fastest_model = min(model_stats.keys(), key=lambda k: model_stats[k]["average_time"])

    # Calculate parallelization efficiency for blob_polling
    total_sequential_time = sum(all_execution_times)
    total_parallel_time = sum(video_times)

    parallelization_efficiency = 0.0
    if total_sequential_time > 0 and total_parallel_time > 0:
        parallelization_efficiency = (total_sequential_time - total_parallel_time) / total_sequential_time

    # Note: GPU/CPU time separation based on model types
    gpu_models = ["nsfw_detection", "face_detection", "yolo_detection", "scene_detection"]
    total_gpu_time = sum(model_stats[model]["total_time"]
                        for model in gpu_models
                        if model in model_stats)

    total_cpu_time = sum(model_stats[model]["total_time"]
                        for model in model_stats.keys()
                        if model not in gpu_models)

    return {
        "average_video_time": round(avg_video_time, 2),
        "average_shard_time": round(avg_shard_time, 2),
        "average_model_time": round(avg_model_time, 2),
        "slowest_model": {
            "name": slowest_model,
            "average_time": round(model_stats[slowest_model]["average_time"], 2) if slowest_model else 0,
            "max_time": round(model_stats[slowest_model]["max_time"], 2) if slowest_model else 0
        } if slowest_model else None,
        "fastest_model": {
            "name": fastest_model,
            "average_time": round(model_stats[fastest_model]["average_time"], 2) if fastest_model else 0,
            "min_time": round(model_stats[fastest_model]["min_time"], 2) if fastest_model else 0
        } if fastest_model else None,
        "total_gpu_time": round(total_gpu_time, 2),
        "total_cpu_time": round(total_cpu_time, 2),
        "parallelization_efficiency": round(parallelization_efficiency, 3),
        "model_breakdown": {
            model: {
                "average_time": round(stats["average_time"], 2),
                "min_time": round(stats["min_time"], 2),
                "max_time": round(stats["max_time"], 2),
                "execution_count": stats["execution_count"]
            }
            for model, stats in model_stats.items()
        },
        "total_models_executed": len(model_stats),
        "total_execution_time": round(sum(all_execution_times), 2)
    }