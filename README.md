# YOLO Detection Module

A long-video object detection tool based on [Ultralytics YOLOv8](https://docs.ultralytics.com/).  
Supports saving detection results as JSONL, merging them into “state-style” events, and visualizing bounding boxes on the original video.

## 📂 Project Structure
```
yolo_detection_r/
├── detection_module.py          # Core detection / merging / JSON writing functions
├── run_detection_demo.py        # Test script (run detection + merge)
├── test_overlay_from_events.py  # Test script (overlay events.jsonl on the video)
├── utils/
│   ├── Tstamp.py                 # Timestamp formatting utility (fmt_hhmmss_ms)
│   └── write_jsonl.py            # JSONL writing utility
```

## ⚙️ Setup
1. Python 3.8+
2. Install dependencies(See requirements.txt)


## 🚀 How to Run

### 1. Run Detection
Edit the parameters at the top of `run_detection_demo.py` (video path, model path, detection classes, etc.), then:
```bash
python yolo_detection_r/run_detection_demo.py
```
This will generate:
- `yolo_detection_r/outputs/events.jsonl`: raw frame-by-frame detection events (`t`, `frame`, `cls`, `conf`, `bbox`)

### 2. Visualize Detection Results
Edit the parameters at the top of `test_overlay_from_events.py` (original video path, events.jsonl path, output video path, etc.), then:
```bash
python ray_pipeline_yolo.py
```
The resulting video will have bounding boxes and labels overlaid on detected frames.

## 📄 JSONL Format

**Raw events file** (`events.jsonl`), one JSON object per line:
```json
{"t": 0.167, "frame": 5, "cls": "person", "conf": 0.91, "bbox": [100.5, 50.2, 200.1, 300.4]}
```
- `t`: timestamp in seconds  
- `frame`: frame index  
- `cls`: detected class  
- `conf`: detection confidence  
- `bbox`: bounding box `[x1, y1, x2, y2]`


