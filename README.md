# insta360-video-activity-segmentation

A pre-annotation video segmentation pipeline for Insta360 videos using Ray and the `nvidia/Cosmos-Reason1-7B` model for scene analysis.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd insta360-video-activity-segmentation
    ```

2.  **Model Setup:**
    This project utilizes the `nvidia/Cosmos-Reason1-7B` model. Ensure you have the model's repository cloned into the `setup/cosmos-reason1` directory.

3.  **Install Dependencies:**
    Install all the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

```
insta360-video-activity-segmentation/
├── ray_jobs/
│   ├── scene_detection.py
│   ├── ... (other ray jobs)
├── setup/
│   ├── cosmos/
│   │   └── setup.py
│   └── cosmos-reason1/
│       └── (nvidia/Cosmos-Reason1-7B model repository)
├── ray_pipeline.py
├── requirements.txt
└── README.md
```

## Running the Pipeline

The main pipeline can be executed by running the `ray_pipeline.py` script.

```bash
python ray_pipeline.py
```

The input video path is currently hardcoded in the main execution block of the script. You can modify the `pipeline_main("path/to/your/video.mp4")` line to point to your video file.

The prompt used for the scene detection model can be configured by changing the `prompt_path` variable within `ray_pipeline.py`.

## Tests

Tests can be run as individual jobs using the command below:

```shell
/home/nvcoe_admin/miniconda3/envs/py311/bin/python /home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/tests/ray_job_test
```

## Merges and PRs

Ensure you merge with a PR to develop.