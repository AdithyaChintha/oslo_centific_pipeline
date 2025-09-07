# ray_jobs/lighting_by_second_job.py
import ray
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Ray tasks
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1, num_gpus=0.04, max_retries=1, max_calls =1)
def lighting_by_second_task(
    video_path: str,
    *,
    output_dir: Optional[str] = None,
    fps_sample: float = 3.0,
    resize_short: Optional[int] = 480,
    max_frames_per_sec: int = 3,
    low_sat_cut: int = 5,
    high_sat_cut: int = 250,
    thresholds: Optional[Dict] = None,
    dark_mean: Optional[float] = None,
    low_mean: Optional[float] = None,
    bright_mean: Optional[float] = None,
    merge_min_sec: float = 1.0,
    sat_low_flag: float = 25.0,
    sat_high_flag: float = 25.0,
) -> Tuple[str, str]:
    """
    Ray task wrapper for Lighting Detection with rich configuration.
    If output_dir is None, defaults to the video's directory.
    """
    from video_process.lighting import run_lighting_two_jsons
    return run_lighting_two_jsons(
        video_path,
        output_dir=output_dir,
        fps_sample=fps_sample,
        resize_short=resize_short,
        max_frames_per_sec=max_frames_per_sec,
        low_sat_cut=low_sat_cut,
        high_sat_cut=high_sat_cut,
        thresholds=thresholds,
        dark_mean=dark_mean,
        low_mean=low_mean,
        bright_mean=bright_mean,
        merge_min_sec=merge_min_sec,
        sat_low_flag=sat_low_flag,
        sat_high_flag=sat_high_flag,
    )

# ---------------------------------------------------------------------------
# Demo driver
# ---------------------------------------------------------------------------

def _demo_print(msg: str):
    print(f"[lighting_by_second_job demo] {msg}")

def simple_test():
    """
    Minimal local test (no cluster config).
    If output_dir is not provided, results will be saved
    in the same folder as the input video.
    """
    _demo_print("Initializing Ray...")
    ray.init(ignore_reinit_error=True)

    video_path = "outputs/simplified_unified_pipeline_output/video_shards/back/vaccum_floor_1GB_back_1280x720_part0.mp4"   # TODO: replace with your test video path

    _demo_print("Submitting lighting detection task...")
    ref = lighting_by_second_task.remote(
        video_path,
        # output_dir=None → defaults to same folder as video
        fps_sample=3.0,
        resize_short=480,
        max_frames_per_sec=3,
        merge_min_sec=1.0,
    )

    per_second_json, events_json = ray.get(ref)
    _demo_print("Done.")
    _demo_print(f"Per-second JSON: {per_second_json}")
    _demo_print(f"Events JSON: {events_json}")


if __name__ == "__main__":
    simple_test()
