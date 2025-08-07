# 🎥 Video Age Detection Pipeline

A modular pipeline for detecting **age** and **gender** of individuals in a video using [DeepFace](https://github.com/serengil/deepface). The script samples frames from the input video, analyzes each detected face, and outputs structured results in JSON format.

---

## ⚙️ Setup Instructions



###
Install required packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run

Run DeepFace-based age and gender detection:

```bash
python src/deepfacedetect.py /path/to/video.mp4
```

This will:

- Sample frames at intervals (e.g., every 30 frames)
- Detect faces
- Estimate age and gender
- Save output to `outputs/<video_name>/predictions.json`
