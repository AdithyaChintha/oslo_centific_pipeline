# 360° Home Activity Motion Analysis Pipeline

A production-ready Python-based motion analysis system for detecting and segmenting activities in 360° home recordings using MOG2 background subtraction and temporal analysis with GPU acceleration support.

## 🎯 Overview

This pipeline automatically processes long-form 360° home activity videos (cooking, cleaning, exercising) and segments them into meaningful activity periods. It uses computer vision techniques adapted from public safety surveillance systems for home activity analysis, now packaged in a production-ready class-based architecture.

### Key Features

- **🎥 360° Video Support**: Processes equirectangular format videos from Insta360 and similar cameras
- **🔍 Motion Energy Analysis**: Uses MOG2 background subtraction for robust motion detection
- **⚡ Activity Segmentation**: Automatically identifies high/medium/low activity periods with overlap resolution
- **📊 Comprehensive Analytics**: Detailed motion statistics and activity summaries
- **🎨 Visual Output**: Generates annotated videos with motion overlay
- **🚀 GPU Acceleration**: CUDA support with CuPy for faster processing
- **🏗️ Production-Ready**: Class-based architecture with comprehensive logging
- **🔧 Configurable**: Adjustable sensitivity and processing parameters
- **⏰ Human-Readable Times**: Displays times as "1h 15m 23s" instead of "75.4 minutes"

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
git clone <repository-url>
cd 360_motion_analyzer

# Install dependencies
pip install -r requirements.txt

# Optional: Install GPU acceleration (for NVIDIA GPUs)
pip install cupy-cuda12x

# Verify ffmpeg installation (required for video processing)
ffmpeg -version
```

### 2. Basic Usage

**Edit the configuration in `main.py`:**

```python
# Custom configuration for this run
custom_config = {
    # === VIDEO INPUT CONFIGURATION ===
    "video_path": r"input/your_360_video.mp4",  # 👈 Your video path
    
    # === OUTPUT CONFIGURATION ===
    "output_directory": "output",               # Results folder
    "save_annotated_video": False,              # Save video with overlay (saves space)
    "save_motion_data": True,                   # Save detailed motion data
    
    # === PROCESSING OPTIONS ===
    "sensitivity_level": "medium",              # low/medium/high
    
    # === DEBUG AND LOGGING ===
    "verbose_logging": True,                    # Enable detailed debug output
}
```

**Run the analysis:**

```bash
python src/main.py
```

### 3. Results

The pipeline generates:
- **📄 `segments.json`**: Detailed activity segments with timestamps
- **🎥 `annotated.mp4`**: Video with motion detection overlay (optional)
- **📊 `motion_data.json`**: Complete motion timeline data
- **📝 `summary.txt`**: Human-readable analysis report

## 📁 Project Structure

```
360_motion_analyzer/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── src/
│   ├── main.py                        # Production-ready pipeline class (edit VIDEO_PATH here)
│   ├── core/
│   │   ├── motion_detector.py         # MOG2 motion detection with GPU acceleration
│   │   └── video_processor.py         # Video I/O and 360° handling
│   ├── utils/
│   │   ├── config.py                  # Configuration constants
│   │   └── logging_utils.py           # Comprehensive logging setup
│   ├── analyzers/
│   │   └── activity_segmenter.py      # Activity segmentation with overlap resolution
│   └── __init__.py files
├── output/                             # Generated results
├── logs/                              # Detailed application logs
└── tests/                             # Unit tests
```

## 🏗️ Production Architecture

### Class-Based Design

The new `MotionEnergyAnalysisPipeline` class provides:

```python
# Initialize pipeline with configuration
pipeline = MotionEnergyAnalysisPipeline(config)

# Run complete analysis
pipeline.run_analysis()

# Get structured results for further processing
results = pipeline.get_results_summary()
```

### Configuration Management

**Flexible configuration system:**
- **Defaults**: Defined in `config.py`
- **Runtime overrides**: Specified in `main.py`
- **No argument parsing**: Simple dictionary-based configuration

```python
# Override defaults for specific analysis
custom_config = {
    "video_path": "videos/cooking_session.mp4",
    "sensitivity_level": "high",
    "output_directory": "results/cooking"
}
```

### Enhanced Logging

**Unified logging system:**
- **Console output**: Progress updates with emoji formatting
- **File logging**: Detailed debug information for analysis
- **Human-readable times**: All durations shown as "1h 15m 23s"
- **GPU status**: Automatic detection and reporting

## 🔧 Configuration Options

### Motion Detection Sensitivity

**Low Sensitivity:**
- Best for noisy environments
- Detects only major movements
- Fewer false positives

**Medium Sensitivity (Recommended):**
- Balanced for most home activities
- Good for cooking, cleaning, exercise
- Default configuration

**High Sensitivity:**
- Detects subtle movements
- Best for detailed activity analysis
- May be noisy in some environments

### Advanced Parameters (in `src/utils/config.py`):

```python
# Motion Detection
MIN_FOREGROUND_AREA = 200        # Minimum motion region size
CONFIDENCE_THRESHOLD = 0.02      # Motion confidence threshold (lowered for home activities)
BG_VAR_THRESHOLD = 25           # Background learning sensitivity

# Activity Segmentation  
ACTIVITY_MIN_DURATION = 1.0     # Minimum activity duration (reduced to 1 second)
HIGH_MOTION_THRESHOLD = 0.02    # High activity threshold (more sensitive)
MEDIUM_MOTION_THRESHOLD = 0.005 # Medium activity threshold (much more sensitive)

# Video Processing
MOTION_SMOOTH_WINDOW = 15       # Smoothing window (frames)
PROGRESS_UPDATE_INTERVAL = 100  # Progress logging frequency
```

## 🎥 Supported Video Formats

- **Primary**: `.mp4` (H.264 encoded)
- **360° Native**: `.insv` (Insta360 format)
- **Additional**: `.avi`, `.mov`, `.mkv`

### 360° Video Requirements

- **Format**: Equirectangular projection
- **Aspect Ratio**: ~2:1 (e.g., 7680x3840, 5760x2880)
- **Content**: Indoor home activities
- **Duration**: 30 seconds to 2+ hours
- **Resolution**: Automatically downscaled for memory efficiency

## 📊 Understanding Results

### Activity Segment Types

- **HIGH_ACTIVITY**: Intense movements (active cooking, vigorous cleaning)
- **MEDIUM_ACTIVITY**: Moderate tasks (food prep, organizing)
- **LOW_ACTIVITY**: Minimal movement (resting, reading)

### Motion Energy Scores

- **0.000 - 0.005**: Low activity (sitting, standing, minimal movement)
- **0.005 - 0.020**: Medium activity (cooking prep, light cleaning)
- **0.020 - 1.000**: High activity (vigorous cooking, active cleaning)

### Confidence Scores

- **0.0 - 0.5**: Low confidence (may be noise or brief movements)
- **0.5 - 0.8**: Medium confidence (likely genuine activity)
- **0.8 - 1.0**: High confidence (clear, sustained activity)

### Human-Readable Time Display

All times are displayed in intuitive formats:
- **45 seconds** → "45s"
- **90 seconds** → "1m 30s"
- **3600 seconds** → "1h"
- **4500 seconds** → "1h 15m"
- **7323 seconds** → "2h 2m 3s"

## 🔍 Example Output

### Terminal Output
```
📊 ANALYSIS RESULTS SUMMARY
========================================
🎬 Video: cooking_session.mp4
⏱️  Duration: 1h 15m 23s
🎯 Segments Found: 3
🚀 Processing Time: 2m 45s

🎯 Top Activity Segments:
   1. 5m 12s-8m 45s (3m 33s)
      Type: HIGH_ACTIVITY
      Motion: 0.087 | Confidence: 0.892
      Description: High-intensity activity period - likely active cooking
```

### JSON Output
```json
{
  "segments": [
    {
      "start_time": 312.5,
      "end_time": 525.3,
      "duration": 212.8,
      "activity_type": "HIGH_ACTIVITY",
      "avg_motion_energy": 0.087,
      "confidence": 0.892,
      "description": "High-intensity activity period (3m 33s) - likely active cooking"
    }
  ]
}
```

## 🚀 GPU Acceleration

### Automatic GPU Detection

The pipeline automatically detects and uses GPU acceleration when available:

```
[GPU] CuPy available - GPU acceleration enabled
🚀 GPU acceleration enabled!
✅ OpenCV CUDA support with 1 device(s)
```

### GPU Requirements

- **NVIDIA GPU** with CUDA support
- **CuPy**: `pip install cupy-cuda12x`
- **OpenCV with CUDA**: For advanced acceleration

### CPU Fallback

The system gracefully falls back to CPU processing if GPU is unavailable:
```
⚠️  Running in CPU mode
```

## 🛠️ Advanced Usage

### Batch Processing

```python
config = {
    "video_directory": "videos/",           # Process entire directory
    "sensitivity_level": "high",
    "save_annotated_video": False,         # Save space for batch
    "output_directory": "batch_results"
}

pipeline = MotionEnergyAnalysisPipeline(config)
pipeline.run_analysis()
```

### Programmatic Access

```python
# Get structured results for further processing
pipeline = MotionEnergyAnalysisPipeline(config)
pipeline.run_analysis()

results = pipeline.get_results_summary()
for segment in results['segments']:
    print(f"Activity: {segment['time_range_formatted']} - {segment['activity_type']}")
```

### Integration with Other Systems

```python
from src.main import MotionEnergyAnalysisPipeline

# Easy integration into larger systems
def analyze_home_video(video_path: str) -> dict:
    config = {"video_path": video_path, "save_annotated_video": False}
    pipeline = MotionEnergyAnalysisPipeline(config)
    pipeline.run_analysis()
    return pipeline.get_results_summary()
```

## 🛠️ Troubleshooting

### Common Issues

**"No video file found":**
- Check that `video_path` in the configuration points to your actual video file
- Ensure the file extension is supported

**"No significant activities detected":**
- Try increasing sensitivity: `"sensitivity_level": "high"`
- Check if video contains actual human activities
- Verify video isn't mostly static scenes

**"Processing very slow":**
- Normal for long videos (1-2x real-time on CPU, faster with GPU)
- Consider setting `"save_annotated_video": False` to save processing time
- GPU acceleration significantly improves performance

**"ffmpeg not found":**
```bash
# Install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
```

### GPU Troubleshooting

**CuPy installation issues:**
```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

**GPU memory issues:**
- Reduce `GPU_MEMORY_FRACTION` in config.py
- Set `FORCE_CPU_FALLBACK = True` to disable GPU

### Debug Mode

Enable comprehensive logging:
```python
custom_config = {
    "verbose_logging": True,    # Detailed console + file logging
    "log_directory": "logs"     # Check logs/ directory for detailed analysis
}
```

## 📈 Performance Expectations

### Processing Speed
- **CPU-only**: 1-2x real-time (1 hour video → 1-2 hours processing)
- **GPU-accelerated**: 2-4x real-time (1 hour video → 15-30 minutes processing)
- **Factors**: Video resolution, motion complexity, hardware specs

### Memory Usage
- **Typical**: 2-4 GB RAM for most videos
- **High-res videos**: Automatically downscaled (7680x3840 → 1920x960)
- **Adaptive**: MOG2 uses constant memory regardless of video length
- **Storage**: ~50MB output data per hour of video

### System Requirements

**Minimum:**
- 8GB RAM
- 4-core CPU
- Python 3.8+

**Recommended:**
- 16GB+ RAM
- 8+ core CPU or NVIDIA GPU
- SSD storage for faster I/O

## 🔮 Key Improvements in This Version

### ✅ Production-Ready Architecture
- **Class-based design**: `MotionEnergyAnalysisPipeline` for clean organization
- **Configuration management**: Flexible config system with defaults and overrides
- **Error handling**: Comprehensive exception handling throughout
- **Logging**: Unified console and file logging with debug information

### ✅ Enhanced User Experience
- **Human-readable times**: "1h 15m 23s" instead of "75.4 minutes"
- **Overlap resolution**: Automatically resolves conflicting activity segments
- **Clear progress tracking**: Real-time processing updates with time estimates
- **GPU detection**: Automatic GPU/CPU detection and optimization

### ✅ Improved Accuracy
- **Optimized thresholds**: Tuned for home activities (more sensitive detection)
- **Gap merging**: Intelligent merging of nearby activity segments
- **Confidence scoring**: Enhanced confidence calculation for segment reliability
- **Noise reduction**: Better morphological operations and filtering

### ✅ Better Output
- **Structured results**: Easy programmatic access to analysis results
- **Comprehensive reports**: Detailed text summaries with recommendations
- **Time navigation**: Video timestamps for easy segment location
- **Memory efficient**: Optional annotated video generation

## 🎛️ Configuration Examples

### High-Sensitivity Analysis
```python
custom_config = {
    "video_path": "videos/subtle_activities.mp4",
    "sensitivity_level": "high",
    "save_annotated_video": True,
    "output_directory": "detailed_analysis"
}
```

### Batch Processing
```python
custom_config = {
    "video_directory": "videos/",
    "sensitivity_level": "medium", 
    "save_annotated_video": False,  # Save space and time
    "output_directory": "batch_results"
}
```

### Memory-Optimized Processing
```python
custom_config = {
    "video_path": "large_video.mp4",
    "save_annotated_video": False,
    "save_motion_data": False,       # Minimal output
    "verbose_logging": False         # Reduce log size
}
```

## 📊 Sample Analysis Results

### For a 39-second cooking video:
```
📊 ANALYSIS RESULTS SUMMARY
========================================
🎬 Video: cooking_demo.mp4
⏱️  Duration: 39s
🎯 Segments Found: 2
🚀 Processing Time: 41s

📈 Motion Statistics:
   Average Motion Energy: 0.0121
   Peak Motion Energy: 1.0000
   High Activity: 21.0% of video
   Medium Activity: 20.6% of video
   Low Activity: 58.4% of video

🎯 Top Activity Segments:
   1. 3s-17s (14s)
      Type: MEDIUM_ACTIVITY
      Motion: 0.015 | Confidence: 0.812
      Description: Moderate activity - likely food prep

   2. 24s-37s (13s)
      Type: MEDIUM_ACTIVITY  
      Motion: 0.008 | Confidence: 0.814
      Description: Moderate activity - likely organizing
```

## 🔬 Technical Details

### Motion Detection Algorithm
- **MOG2 Background Subtraction**: Adaptive background learning
- **Morphological Operations**: Noise reduction and hole filling
- **Connected Components**: Region-based motion analysis
- **Temporal Smoothing**: 15-frame window for stability

### Activity Segmentation
- **Threshold-based**: High/medium/low activity classification
- **Gap merging**: Combines nearby segments (5-second tolerance for short videos)
- **Overlap resolution**: Resolves conflicts by confidence or boundary adjustment
- **Quality filtering**: Minimum duration and confidence requirements

### GPU Acceleration
- **CuPy integration**: GPU-accelerated array operations
- **CUDA background subtraction**: Hardware-accelerated motion detection
- **Automatic fallback**: Seamless CPU fallback if GPU unavailable
- **Memory management**: Efficient GPU memory usage

## 🔍 Analysis Capabilities

### Activity Detection
- **Cooking activities**: Chopping, stirring, pot handling
- **Cleaning activities**: Sweeping, wiping, organizing
- **Exercise activities**: Stretching, walking, equipment use
- **General movement**: Walking, reaching, object manipulation

### Temporal Analysis
- **Short activities**: 1-second minimum detection
- **Long sessions**: Multi-hour video support
- **Activity transitions**: Automatic boundary detection
- **Rest periods**: Low-activity segment identification

## 🚀 Performance Optimization

### For Large Videos (>1 hour):
```python
custom_config = {
    "save_annotated_video": False,    # Skip video generation
    "sensitivity_level": "medium",    # Balanced performance
}
```

### For High-Resolution Videos:
- **Automatic downscaling**: 7680x3840 → 1920x960 for processing
- **Memory efficiency**: 25% resize factor reduces memory usage by ~94%
- **Quality preservation**: Motion detection accuracy maintained

### For Batch Processing:
```python
# Process entire directories efficiently
custom_config = {
    "video_directory": "surveillance_footage/",
    "save_motion_data": False,        # Skip detailed data for space
    "verbose_logging": False          # Reduce log volume
}
```

## 🔮 Future Enhancements

### Planned Features
- **🚀 Distributed Processing**: Multi-GPU support for large-scale analysis
- **🎵 Audio Analysis**: Voice Activity Detection integration
- **🧠 Semantic Classification**: Activity type recognition using deep learning
- **☁️ Cloud Storage**: Azure Blob/AWS S3 integration
- **📱 Real-time Processing**: Live camera feed analysis

## 📞 Support

### Getting Help
1. **Check logs**: Review detailed logs in `logs/` directory
2. **Enable debug mode**: Set `"verbose_logging": True`
3. **Validate setup**: Pipeline automatically tests installation
4. **Test with short video**: Try 30-60 second clips first

### System Requirements Validation
The pipeline automatically validates:
- **Dependencies**: OpenCV, NumPy, SciPy, scikit-learn
- **ffmpeg**: Video processing capability
- **GPU**: CUDA and CuPy availability
- **Configuration**: Output directories and file permissions

## 📄 License

This project is developed for research and educational purposes. Please ensure compliance with your organization's policies when processing video data.

---

**🏠📹 Ready to analyze your 360° home activities with production-ready performance!**

### Recent Updates (v2.0)
- ✅ **Class-based architecture** for production use
- ✅ **Human-readable time formatting** (1h 15m 23s)
- ✅ **Overlap resolution** eliminates conflicting segments
- ✅ **GPU acceleration** with automatic CPU fallback
- ✅ **Enhanced logging** with detailed debug information
- ✅ **Optimized thresholds** for better home activity detection