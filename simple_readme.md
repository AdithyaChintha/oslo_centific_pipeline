# Audio Diarization & PII Detection Pipeline

Ray-based pipeline for processing 360° video files to extract audio, identify speakers, transcribe speech, and detect PII.

## Setup

### 1. Environment
```bash
conda activate py311
pip install -r requirements.txt
```

### 2. Hugging Face Access
```bash
export HF_TOKEN=your_huggingface_token
```

**Required Model Access:**
- Request access to `pyannote/speaker-diarization-3.1` on Hugging Face
- Accept terms for `pyannote/segmentation-3.0` 
- These are gated models requiring manual approval

### 3. Dependencies
- `ffmpeg` (for audio extraction)
- CUDA support recommended for faster processing

## Pipeline Components

### Core Ray Jobs

**`ray_jobs/video_splitter.py`**
- Splits video into 60-second shards
- Input: Video file path → Output: List of shard paths

**`ray_jobs/audio_diarization_pii.py`**
- Main processing: speaker diarization, transcription, PII detection
- Input: List of video shards → Output: Transcript, speaker timeline, PII detections

**`ray_jobs/insv_to_mp4.py`**
- Converts Insta360 .insv files to standard .mp4
- Input: .insv file → Output: Dual .mp4 views

### Pipeline Runners

**`ray_pipeline_audio_pii_detection.py`**
- Complete pipeline orchestration
- Input: Video file → Output: JSON with all results

## Usage

### Quick Test
```bash
python ray_jobs/audio_diarization_pii.py
```

### Full Pipeline
```bash
python ray_pipeline_audio_pii_detection.py
# Or with custom video:
python ray_pipeline_audio_pii_detection.py /path/to/your/video.mp4
```

### Run Tests
```bash
python test/test_audio_diarization.py
```

## Output Structure

Pipeline produces three main outputs:
- **Transcript**: Full speech transcription
- **Diarization**: Speaker timeline (who spoke when)
- **PII Detections**: Personal information found with confidence scores

Results saved to `pipeline_results.json`

## File Structure
```
├── ray_jobs/
│   ├── audio_diarization_pii.py    # Main audio processing
│   ├── video_splitter.py           # Video segmentation  
│   └── insv_to_mp4.py              # Insta360 conversion
├── test/
│   └── test_audio_diarization.py   # Test runner
├── utils/
│   └── logger.py                   # Logging config
└── ray_pipeline_audio_pii_detection.py  # Main pipeline
```

## Troubleshooting

**"ffmpeg error":** Install ffmpeg or check video file integrity  
**"HF_TOKEN not found":** Export your Hugging Face token  
**"Model access denied":** Request access to gated pyannote models on HF  
**No speech detected:** Video may not contain clear audio