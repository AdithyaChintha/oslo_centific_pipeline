
# 🎥 Insta360 Video Activity Segmentation with Face Age Detection

A distributed video processing pipeline built with [Ray](https://ray.io/) for detecting **age** and **gender** of individuals in Insta360 videos. The pipeline processes `.insv` files, splits them into chunks, and runs face detection in parallel using [DeepFace](https://github.com/serengil/deepface).

---

## 🏗️ Project Structure

```
insta360-video-activity-segmentation/
├── ray_jobs/                          # Ray job modules
│   ├── insv_to_mp4.py                # Convert INSV to dual MP4 views
│   ├── video_splitter.py             # Split videos into chunks
│   └── face_age_detector.py          # Face age/gender detection
├── video-age-detection-pipeline/      # Original single-machine pipeline
│   ├── src/
│   │   ├── deepfacedetect.py         # FaceAgeDetector class
│   │   └── insightface.py            # InsightFace implementation
│   └── outputs/                       # Processing results
├── ray_pipeline_age_detect.py         # Main Ray pipeline orchestrator
├── tests/                             # Test files
│   └── test_face_age_detector.py     # Test Ray job functions
└── requirements.txt                    # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install External Tools

The pipeline requires these external tools:

- **FFmpeg**: Video processing and conversion
  ```bash
  # macOS
  brew install ffmpeg
  
  # Ubuntu/Debian
  sudo apt update && sudo apt install ffmpeg
  
  # Windows
  # Download from https://ffmpeg.org/download.html
  ```

- **ExifTool**: Metadata extraction
  ```bash
  # macOS
  brew install exiftool
  
  # Ubuntu/Debian
  sudo apt install exiftool
  
  # Windows
  # Download from https://exiftool.org/
  ```

---

## 🚀 How to Run

### Option 1: Full Ray Pipeline (Recommended)

Process INSV files through the complete distributed pipeline:

```bash
python ray_pipeline_age_detect.py /path/to/video.insv
```

**What this does:**
1. **Converts** `.insv` → dual `.mp4` views (front/back)
2. **Splits** each view into 60-second chunks
3. **Processes** chunks in parallel for face detection
4. **Aggregates** results with proper timestamps
5. **Saves** results to `outputs/<video_name>/`

### Option 2: Individual Ray Jobs

Test specific components independently:

```bash
# Test INSV to MP4 conversion
python ray_jobs/insv_to_mp4.py

# Test video splitting
python ray_jobs/video_splitter.py

# Test face detection with sample video
python ray_jobs/face_age_detector.py path/to/sample_video.mp4
```

### Option 3: Original Single-Machine Pipeline

Run the original non-distributed version:

```bash
python video-age-detection-pipeline/src/deepfacedetect.py /path/to/video.mp4
```

---

## 🧪 Testing

Run the test suite to verify Ray job functionality:

```bash
python tests/test_face_age_detector.py
```

This will:
- Test single video processing
- Test video chunks processing
- Use actual MP4 files from your outputs directory
- Show real face detection results

---

## 📊 Output Structure

```
outputs/
├── <video_name>/
│   ├── <video_name>_view1.mp4        # Front view MP4
│   ├── <video_name>_view2.mp4        # Back view MP4
│   ├── chunks_view_1/                 # Chunks for view 1
│   │   ├── <video_name>_view1_part0.mp4
│   │   ├── <video_name>_view1_part1.mp4
│   │   └── ...
│   ├── chunks_view_2/                 # Chunks for view 2
│   ├── chunk_results_view_1/          # Face detection results for view 1
│   ├── chunk_results_view_2/          # Face detection results for view 2
│   ├── <video_name>_view_1_face_analysis_results.json
│   ├── <video_name>_view_2_face_analysis_results.json
│   └── <video_name>_age_detection_summary.json
```

---

## 🔧 Configuration

### Frame Sampling

Control how often frames are analyzed:

```python
# In ray_pipeline_age_detect.py
frame_interval = 30  # Process every 30th frame
```

### Chunk Duration

Adjust video chunk size for processing:

```python
# In ray_pipeline_age_detect.py
chunk_duration_sec = 60  # 60-second chunks
```

### Output Directories

Customize where results are saved:

```python
# In ray_pipeline_age_detect.py
output_dir = "/path/to/your/outputs"
```

---

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg/ExifTool not found**
   - Ensure external tools are installed and in PATH
   - Verify with: `ffmpeg -version` and `exiftool -ver`

2. **Module import errors**
   - Check Python path: `python -c "import sys; print(sys.path)"`
   - Ensure you're running from project root directory

3. **Ray initialization issues**
   - Verify Ray installation: `pip install ray`
   - Check Ray status: `ray status`

### Debug Mode

Enable verbose logging in Ray jobs:

```python
# In ray_jobs/face_age_detector.py
logger.setLevel(logging.DEBUG)
```

---

## 🚀 Performance

- **Parallel Processing**: Multiple video chunks processed simultaneously
- **Distributed Computing**: Ray handles worker management and task distribution
- **Memory Efficient**: Processes videos in chunks rather than loading entire files
- **Scalable**: Can run on multiple machines in a Ray cluster

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Test** your changes with the test suite
4. **Submit** a pull request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) for face analysis
- [Ray](https://ray.io/) for distributed computing
- [FFmpeg](https://ffmpeg.org/) for video processing
- [ExifTool](https://exiftool.org/) for metadata extraction
=======
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
### feature/nsfw_detection

model weights 
https://huggingface.co/onnx-community/nsfw_image_detection-ONNX/tree/main/onnx?not-for-all-audiences=true
 
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
