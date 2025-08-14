import ray
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import ray
from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4


# Add current directory to Python path to allow import of local modules
sys.path.append(str(Path(__file__).resolve().parent))

from ray_jobs.insv_to_mp4 import convert_insv_to_dual_mp4

if __name__ == "__main__":
    ray.init()

    # Replace with a valid .insv file path for testing
    test_insv_file = "/home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/VID_20250720_152154_00_011.insv"

    if not Path(test_insv_file).exists():
        print(f"Test INSV file not found: {test_insv_file}")
        sys.exit(1)

    future = convert_insv_to_dual_mp4.remote(test_insv_file)
    result = ray.get(future)

    print("Ray job result:")
    print(result)

    ray.shutdown()