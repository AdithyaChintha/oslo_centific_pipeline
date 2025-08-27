
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
### Output JSON files:
#### Audio and diarization model:
```
{
  "shard_path": "/tmp/pipeline_output_yeeju1dh/audio_shards/VID_20250809_094836_00_045_part0.wav",
  "shard_name": "VID_20250809_094836_00_045_part0.wav",
  "audio_duration": 9.4506875,
  "transcript": "It's 9.45am and we're going to buy water to make coffee.",
  "diarization": [
    {
      "speaker": "Speaker 1",
      "start_time": 0.03096875,
      "end_time": 3.4734687500000003,
      "duration": 3.4425000000000003
    }
  ],
  "pii_detections": [],
  "summary": {
    "total_speakers": 1,
    "total_segments": 1,
    "segments_with_pii": 0,
    "total_pii_detections": 0
  },
  "processing_stats": {
    "total_processing_time": 7.484316,
    "diarization_time": 0.348663,
    "transcription_time": 5.873095,
    "device_used": "cpu"
  }
}
```

#### clap model:
```
{
  "file_path": "/tmp/pipeline_output_yeeju1dh/audio_shards/VID_20250809_094836_00_045_part0.wav",
  "file_type": "audio",
  "duration_seconds": 9.451,
  "sample_rate": 16000,
  "clap_count": 1,
  "clap_timestamps": [
    {
      "timestamp_seconds": 1.408,
      "timestamp_formatted": "00:01.408"
    }
  ],
  "detection_parameters": {
    "threshold_bias": 6000,
    "frequency_range": {
      "lowcut": 200,
      "highcut": 3200
    },
    "debounce_time": 0.15
  },
  "processing_time_seconds": 0.04,
  "output_file": "/tmp/pipeline_output_yeeju1dh/shard_1/view_1/clap_output/VID_20250809_094836_00_045_part0_clap_detection.json",
  "processed_at": "2025-08-27T18:18:23.131071",
  "success": true
}
```

#### Scene detection:
```
{
  "video_path": "/tmp/pipeline_output_yeeju1dh/view_1_shards/VID_20250809_094836_00_045_view1_part0.mp4",
  "scenes": [
    {
      "start_time": 0.0,
      "end_time": 2.7,
      "description": "The frame captures a person's hand reaching out towards the sink faucet handle."
    },
    {
      "start_time": 3.6,
      "end_time": 4.5,
      "description": "The hand grasps the faucet handle firmly and turns it clockwise, initiating the flow of water into the sink."
    }
  ],
  "raw_response": "```json\n[\n  {\n    \"start_time\": 0.0,\n    \"end_time\": 2.7,\n    \"caption\": \"The frame captures a person's hand reaching out towards the sink faucet handle.\"\n  },\n  {\n    \"start_time\": 3.6,\n    \"end_time\": 4.5,\n    \"caption\": \"The hand grasps the faucet handle firmly and turns it clockwise, initiating the flow of water into the sink.\"\n  }\n]\n```",
  "total_scenes": 2,
  "processing_info": {
    "model": "nvidia/Cosmos-Reason1-7B",
    "success": true,
    "gpu_memory_utilization": 0.25,
    "max_model_len": 6144,
    "scenes_extracted": 2
  }
}
```

#### yolo detection:
```
{"_meta": {"video": "/tmp/pipeline_output_yeeju1dh/view_1_shards/VID_20250809_094836_00_045_view1_part0.mp4", "model": "yolov8n.pt", "conf": 0.5, "iou": 0.5, "frame_stride": 5, "classes": null, "device": null, "base_offset": 0, "enable_tracking": true, "tracker": "ultralytics:botsort.yaml"}}
{"t": 8.809, "ts": "00:00:08.809", "frame": 264, "cls": "person", "conf": 0.5021, "bbox": [191.04, 335.25, 449.74, 462.56]}
{"t": 8.842, "ts": "00:00:08.842", "frame": 265, "cls": "person", "conf": 0.6589, "bbox": [175.52, 333.3, 452.91, 463.62], "track_id": 1}
{"t": 8.876, "ts": "00:00:08.876", "frame": 266, "cls": "person", "conf": 0.5956, "bbox": [169.02, 330.69, 450.93, 463.73], "track_id": 1}
{"t": 9.343, "ts": "00:00:09.343", "frame": 280, "cls": "pizza", "conf": 0.5208, "bbox": [28.14, 0.08, 477.39, 238.9]}
```

#### Label studio integration:
```
{
  "data": {
    "meta": "",
    "meta.home_identifier": "Shard_1",
    "meta.recording_datetime": "2025-08-27T18:21:11.736694Z",
    "meta.domain": "production",
    "meta.actions": "",
    "shard_number": "1",
    "shard_offset_seconds": "0",
    "segments_detected": "2",
    "video_left": "https://oslotestvideo.blob.core.windows.net/instavideo/instavideo/krishna-test/test1/test_activity/pre-annotation-output/test_08_27/VID_20250809_094836_00_045/view_1_shards/VID_20250809_094836_00_045_view1_part0.mp4?se=2025-11-25T18%3A18%3A20Z&sp=r&sv=2025-07-05&sr=b&sig=tdwVCT9XIRsLoy%2BLiBnOfmN1/K1bt%2Bi9ZwOFd9I6D7k%3D",
    "video_right": "https://oslotestvideo.blob.core.windows.net/instavideo/instavideo/krishna-test/test1/test_activity/pre-annotation-output/test_08_27/VID_20250809_094836_00_045/view_2_shards/VID_20250809_094836_00_045_view2_part0.mp4?se=2025-11-25T18%3A18%3A21Z&sp=r&sv=2025-07-05&sr=b&sig=Dr7ijCVNEslO33xle0oAnWUl2wZ96kyXhoGdV6fiy5g%3D",
    "audio": "https://oslotestvideo.blob.core.windows.net/instavideo/instavideo/krishna-test/test1/test_activity/pre-annotation-output/test_08_27/VID_20250809_094836_00_045/audio_shards/VID_20250809_094836_00_045_part0.wav?se=2025-11-25T18%3A18%3A21Z&sp=r&sv=2025-07-05&sr=b&sig=JzHf9pLTt/SQylhYYDxVXzD9HL6LFW1OAd67uaF7wo0%3D",
    "home_id": "",
    "start_datetime": "",
    "end_datetime": "",
    "total_duration": "",
    "files_deleted": []
  },
  "annotations": [],
  "predictions": [
    {
      "result": [
        {
          "id": "taxonomy_domain_actions_pred",
          "type": "taxonomy",
          "value": {
            "taxonomy": [
              [
                "Food & Mealtime",
                "Washing dishes"
              ]
            ]
          },
          "score": 0.8,
          "from_name": "taxonomy_domain_actions",
          "to_name": "video_left"
        },
        {
          "id": "lighting_pred",
          "type": "choices",
          "value": {
            "choices": [
              "Bright light"
            ]
          },
          "score": 0.7,
          "from_name": "lighting",
          "to_name": "video_left"
        }
      ]
    }
  ]
}
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
