# insta360-video-activity-segmentation

A pre-annotation video segmentation pipeline for Insta360 videos using Ray for scene analysis, audio diarization, and PII detection.

## Features

*   **Scene Detection:** Utilizes the `nvidia/Cosmos-Reason1-7B` model to analyze video content and identify distinct scenes.
*   **Audio Diarization:** Identifies different speakers in the audio track.
*   **PII Detection:** Detects and redacts personally identifiable information from the audio.
*   **YOLO Object Detection:** A long-video object detection tool based on [Ultralytics YOLOv8](https://docs.ultralytics.com/).

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
├── config/
│   ├── cosmos_prompt.yaml
│   └── azure_blob_config.yaml
│   ├── upload_manager.py
|   ├── insv_to_mp4.py
│   ├── download_manager.py
│   ├── video_splitter.py
│   ├── scene_detection.py
│   ├── audio_diarization_pii.py
│   ├── vad_audio.py
│   ├── motion_energy.py
│   ├── embeddings_driver.py
│   ├── yolo_sort_tracker.py
│   ├── fusion_gap_merge.py
│   ├── quality_flagger.py
│   ├── segment_classifier.py
│   └── ray_driver.py
├── tests/
│   ├── ray_job_test.py
├── utils/
│   ├── blob_utils.py
│   └── logger.py
├── setup/
│   ├── cosmos/
│   │   └── setup.py
│   └── cosmos-reason1/
│       └── (nvidia/Cosmos-Reason1-7B model repository)
├── yolo_detection_r/
│   ├── ... (YOLO detection module)
├── ray_pipeline.py
├── ray_pipeline_audio_pii_detection.py
├── requirements.txt
└── README.md
```

## Running the Pipelines

### Scene Analysis Pipeline

The main pipeline can be executed by running the `ray_pipeline.py` script.

```bash
python ray_pipeline.py
```

The input video path is currently hardcoded in the main execution block of the script. You can modify the `pipeline_main("path/to/your/video.mp4")` line to point to your video file.

The prompt used for the scene detection model can be configured by changing the `prompt_path` variable within `ray_pipeline.py`.

### Audio Diarization and PII Detection Pipeline

The audio analysis pipeline can be run using the `ray_pipeline_audio_pii_detection.py` script.

```bash
python ray_pipeline_audio_pii_detection.py
```

### YOLO Detection Module

A long-video object detection tool based on [Ultralytics YOLOv8](https://docs.ultralytics.com/).
Supports saving detection results as JSONL, merging them into “state-style” events, and visualizing bounding boxes on the original video.

#### How to Run YOLO Detection

1.  **Run Detection:**
    Edit the parameters at the top of `ray_pipeline_yolo.py` (video path, model path, detection classes, etc.), then:
    ```bash
    python ray_pipeline_yolo.py --input ~/videos/video.mp4
    ```
    This will generate raw frame-by-frame detection events.

2.  **Visualize Detection Results:**
    Edit the parameters at the top of `test_overlay_from_events.py` (original video path, events.jsonl path, output video path, etc.), then run the script. The resulting video will have bounding boxes and labels overlaid on detected frames.

## Tests

Tests can be run as individual jobs using the command below:

```shell
/home/nvcoe_admin/miniconda3/envs/py311/bin/python /home/nvcoe_admin/code/oslo/insta360-video-activity-segmentation/tests/ray_job_test
```

## Merges and PRs

Ensure you merge with a PR to develop.