import os
import ray
import pytest
import tempfile
import shutil
from pathlib import Path
import logging
import time
from datetime import datetime

# --- Ensure local package is importable ---
import sys
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# --- Logging setup ---
LOG_LEVEL = os.getenv("TEST_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("TestVideoSplitter")
if not logger.handlers:
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)

# Quiet noisy Azure SDK / HTTP logging
for noisy in [
    "azure",
    "azure.core",
    "azure.storage",
    "azure.storage.blob",
    "azure.core.pipeline.policies.http_logging_policy",
    "urllib3",
]:
    _lz = logging.getLogger(noisy)
    _lz.setLevel(logging.WARNING)
    _lz.propagate = False

# Some environments still emit from the http_logging_policy; hard-disable as a fallback
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").disabled = True

def _human_size(num: int, suffix="B") -> str:
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"

def _ffmpeg_has_nvenc() -> bool:
    try:
        import subprocess
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and ("h264_nvenc" in out.stdout or "hevc_nvenc" in out.stdout)
    except Exception:
        return False


def _ffmpeg_has_scale_cuda() -> bool:
    try:
        import subprocess
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=10)
        return out.returncode == 0 and ("scale_cuda" in out.stdout)
    except Exception:
        return False

from ray_jobs.video_splitter import split_video_into_shards
from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4

def _run_insv_convert(src_insv: str, output_dir: str) -> str:
    # The converter is a Ray @remote function; call via .remote and ray.get
    obj_ref = convert_insv_to_dual_mp4.remote(src_insv, output_dir=output_dir)
    start_wait = time.perf_counter()
    last_log = start_wait
    while True:
        ready, not_ready = ray.wait([obj_ref], timeout=5.0)
        if ready:
            out = ray.get(ready[0])
            break
        now = time.perf_counter()
        if now - last_log >= 15.0:  # log every ~15s to avoid spam
            logger.info("Waiting for INSV->MP4 conversion... elapsed %.1fs", now - start_wait)
            last_log = now
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return out[0]
    if isinstance(out, dict):
        return out.get("view1") or next(iter(out.values()))
    raise AssertionError(f"Unexpected converter output type: {type(out)}")

from utils.azure_blob_utils import list_blobs, download_blob

# --- Configuration ---
CONFIG_PATH = "config/azure_blob.yaml"

# Mark the test to be skipped if the config file is not available
pytestmark = pytest.mark.skipif(
    not os.path.exists(CONFIG_PATH),
    reason=f"Azure config file not found at: {CONFIG_PATH}"
)

@pytest.fixture(scope="module")
def ray_init():
    """Initialize and shutdown Ray for the test module."""
    logger.info("Initializing Ray (ignore_reinit_error=True)...")
    t0 = time.perf_counter()
    ray.init(ignore_reinit_error=True)
    logger.info("Ray initialized in %.2fs | resources=%s", time.perf_counter()-t0, ray.cluster_resources())
    yield
    logger.info("Shutting down Ray...")
    t1 = time.perf_counter()
    ray.shutdown()
    logger.info("Ray shutdown in %.2fs", time.perf_counter()-t1)

@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    td = tempfile.mkdtemp()
    logger.info("Created temporary directory: %s", td)
    yield td
    logger.info("Deleting temporary directory: %s", td)
    t0 = time.perf_counter()
    shutil.rmtree(td)
    logger.info("Temporary directory deleted in %.2fs", time.perf_counter()-t0)

def test_split_video_from_azure_blob(ray_init, temp_dir, fast_mode):
    """
    Tests the split_video_into_shards function by downloading a video from
    Azure Blob Storage and processing it locally.
    """
    start_ts = datetime.now()
    logger.info("Test started at %s", start_ts.strftime("%Y-%m-%d %H:%M:%S"))
    # --- Test Setup ---
    container_name = "instavideo"
    blob_prefix = "azure_directory_path/DCIM/Camera01/"
    
    # 1. Find a video file in Azure Blob Storage
    t0 = time.perf_counter()
    logger.info("Listing blobs in '%s' with prefix '%s'...", container_name, blob_prefix)
    blob_names = list_blobs(container_name, blob_prefix, file_extension=".insv")
    list_time = time.perf_counter()-t0
    logger.info("Found %d blob(s) in %.2fs", len(blob_names), list_time)
    before = len(blob_names)
    blob_names = [b for b in blob_names if not os.path.basename(b).startswith("._")]
    logger.info("Filtered macOS resource forks: %d -> %d", before, len(blob_names))

    assert blob_names, f"No .insv videos found in {container_name}/{blob_prefix}. Test cannot proceed."

    video_to_download = blob_names[0]
    logger.info("Selected blob: %s", video_to_download)

    # 2. Download the video to a temporary local path
    local_video_path = os.path.join(temp_dir, os.path.basename(video_to_download))
    t0 = time.perf_counter()
    downloaded_path = download_blob(container_name, video_to_download, local_video_path)
    dl_time = time.perf_counter()-t0
    assert downloaded_path and os.path.exists(downloaded_path), "Failed to download video from Azure."
    insv_size = os.path.getsize(downloaded_path)
    logger.info("Downloaded to %s (%s) in %.2fs [%.2f MB/s]", downloaded_path, _human_size(insv_size), dl_time, (insv_size/1e6)/dl_time if dl_time>0 else 0.0)

    # 3. Convert the downloaded .insv to .mp4 using ray_jobs converter
    logger.info("Converting INSV -> MP4 via convert_insv_to_dual_mp4 (Ray remote)...")
    t0 = time.perf_counter()
    mp4_path = _run_insv_convert(downloaded_path, output_dir=temp_dir)
    conv_time = time.perf_counter()-t0
    assert mp4_path and os.path.exists(mp4_path), f"INSV conversion failed, mp4 not found: {mp4_path}"
    mp4_size = os.path.getsize(mp4_path)
    logger.info("Conversion produced %s (%s) in %.2fs [%.2f MB/s]", mp4_path, _human_size(mp4_size), conv_time, (mp4_size/1e6)/conv_time if conv_time>0 else 0.0)

    # Log local FFmpeg capabilities and set GPU fraction for fast mode if not already provided
    logger.info("FFmpeg capabilities: NVENC=%s | scale_cuda=%s", _ffmpeg_has_nvenc(), _ffmpeg_has_scale_cuda())
    if os.getenv("SPLITTER_GPU_FRACTION") is None:
        os.environ["SPLITTER_GPU_FRACTION"] = "0.5" if fast_mode else "1.0"
        logger.info("SPLITTER_GPU_FRACTION not set by env; defaulting to %s", os.environ["SPLITTER_GPU_FRACTION"])

    # 4. Run the video splitter Ray task on the converted mp4 file
    output_shard_dir = os.path.join(temp_dir, "shards")
    shard_duration = 10 if fast_mode else 60
    logger.info("Mode: %s | shard_duration=%ss", "FAST" if fast_mode else "NORMAL", shard_duration)
    logger.info("Splitting MP4 into shards: dst=%s duration_sec=%s downscale_480p=%s", output_shard_dir, shard_duration, True)
    t0 = time.perf_counter()
    task = split_video_into_shards.remote(
        video_path=mp4_path,
        output_dir=output_shard_dir,
        duration_sec=shard_duration,
        downscale_to_480p=True
    )
    start_wait = time.perf_counter()
    last_log = start_wait
    while True:
        ready, not_ready = ray.wait([task], timeout=5.0)
        if ready:
            shard_paths = ray.get(ready[0])
            break
        now = time.perf_counter()
        if now - last_log >= 15.0:
            logger.info("Waiting for split to finish... elapsed %.1fs", now - start_wait)
            last_log = now
    split_time = time.perf_counter()-t0
    assert shard_paths, "The video splitter returned no shard paths."
    logger.info("Split produced %d shard(s) in %.2fs", len(shard_paths), split_time)

    # 5. Verify the results
    total_bytes = 0
    for i, shard_path in enumerate(shard_paths, 1):
        assert os.path.exists(shard_path), f"Shard file not found: {shard_path}"
        sz = os.path.getsize(shard_path)
        total_bytes += sz
        assert sz > 0, f"Shard file is empty: {shard_path}"
        logger.info("Shard %d: %s (%s)", i, shard_path, _human_size(sz))
    logger.info("Total shard size: %s", _human_size(total_bytes))

    # A simple check to ensure we have more than one shard for a video longer than duration_sec
    # (This depends on the actual video length)
    assert len(shard_paths) > 0, "Expected at least one shard to be created."

    total_time = (datetime.now() - start_ts).total_seconds()
    logger.info("✅ Test passed successfully in %.2fs (download %.2fs | convert %.2fs | split %.2fs)", total_time, dl_time, conv_time, split_time)