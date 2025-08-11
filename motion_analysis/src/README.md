# insta360-video-activity-segmentation

Ray-based distributed motion energy detection pipeline for 360° home activity videos with production-ready integration.

## 🎯 Overview

This pipeline automatically processes long-form 360° home activity videos (cooking, cleaning, exercising) and segments them into meaningful activity periods using distributed Ray processing for scalable performance.

### Key Features

- **🚀 Ray Distributed Processing**: Scalable motion analysis using Ray clusters
- **🎥 360° Video Support**: Processes equirectangular format videos from Insta360 cameras  
- **🔍 Motion Energy Detection**: MOG2 background subtraction for robust motion detection
- **⚡ Activity Segmentation**: Automatically identifies high/medium/low activity periods
- **📊 Production Ready**: Clean integration APIs for existing pipelines
- **🎨 Minimal Dependencies**: Focused on essential motion analysis functionality
- **🖥️ GPU Acceleration**: CUDA support for faster processing

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/nvidia-coe/insta360-video-activity-segmentation.git
cd insta360-video-activity-segmentation

git checkout feature/motion-energy-analysis

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

#### Standalone Motion Analysis
```python
import ray
from ray_jobs.motion_energy import analyze_motion_energy_only

# Initialize Ray
ray.init()

# Analyze motion energy
future = analyze_motion_energy_only.remote(
    video_path="input/your_video.mp4",
    sensitivity_level="medium",
    save_detailed_data=False
)

result = ray.get(future)

if result["success"]:
    print(f"Found {result['total_segments']} activity segments")
    for seg in result['segments']:
        print(f"{seg['start_time']:.1f}s-{seg['end_time']:.1f}s: {seg['activity_type']}")

ray.shutdown()
```

#### Pipeline Integration
```python
# In your existing pipeline
from ray_jobs.motion_energy import analyze_motion_energy_only

def motion_analysis_stage(video_path: str):
    motion_results = ray.get(analyze_motion_energy_only.remote(
        video_path=video_path,
        sensitivity_level="medium"
    ))
    return motion_results
```

### 3. Test the System

```bash
# Test motion energy analysis
python ray_jobs/motion_energy.py

# Test Ray integration  
python tests/test_motion_ray.py

# Test full pipeline
python my_pipeline.py
```

## 📁 Project Structure

```
insta360-video-activity-segmentation/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── my_pipeline.py                      # Main pipeline with motion energy integration
├── ray_jobs/
│   ├── motion_energy.py               # Core motion energy analysis (Ray remote)
│   ├── video_splitter.py             # Video chunking utilities
│   ├── insv_to_mp4.py                # INSV format conversion
│   └── __init__.py
├── motion_analysis/                    # Motion analysis pipeline
│   └── src/
│       ├── main.py                    # Motion analysis pipeline class
│       ├── core/
│       │   ├── motion_detector.py     # MOG2 motion detection
│       │   └── video_processor.py     # Video I/O and processing
│       ├── analyzers/
│       │   └── activity_segmenter.py  # Activity segmentation logic
│       └── utils/
│           ├── config.py              # Configuration constants
│           └── logging_utils.py       # Logging utilities
├── utils/
│   └── logger.py                      # Ray pipeline logging
├── tests/
│   └── test_motion_ray.py            # Motion energy testing
└── input/                             # Test video directory
```

## 🔧 Core Functions

### Motion Energy Analysis

```python
@ray.remote
def analyze_motion_energy_only(
    video_path: str,
    sensitivity_level: str = "medium",  # "low", "medium", "high"
    save_detailed_data: bool = False
) -> Dict[str, Any]:
    """
    Analyzes motion energy in video without chunking or conversion.
    Perfect for pipeline integration.
    
    Returns:
        {
            "success": bool,
            "total_segments": int,
            "video_duration_seconds": float,
            "processing_time_seconds": float,
            "motion_statistics": {...},
            "segments": [
                {
                    "start_time": float,
                    "end_time": float, 
                    "duration": float,
                    "activity_type": str,  # "HIGH_ACTIVITY", "MEDIUM_ACTIVITY", "LOW_ACTIVITY"
                    "confidence": float,
                    "avg_motion_energy": float
                }
            ]
        }
    """
```

## 📊 Understanding Results

### Activity Segment Types

- **HIGH_ACTIVITY**: Intense movements (active cooking, vigorous cleaning, exercise)
- **MEDIUM_ACTIVITY**: Moderate tasks (food prep, organizing, light cleaning)  
- **LOW_ACTIVITY**: Minimal movement (resting, reading, passive activities)

### Motion Energy Scores

- **0.000 - 0.005**: Low activity (sitting, standing, minimal movement)
- **0.005 - 0.020**: Medium activity (cooking prep, light cleaning)
- **0.020 - 1.000**: High activity (vigorous cooking, active cleaning)

### Confidence Scores

- **0.0 - 0.5**: Low confidence (may be noise or brief movements)
- **0.5 - 0.8**: Medium confidence (likely genuine activity)
- **0.8 - 1.0**: High confidence (clear, sustained activity)

## 🎥 Supported Video Formats

- **Primary**: `.mp4` (H.264 encoded)
- **360° Native**: `.insv` (Insta360 format, auto-converted)
- **Additional**: `.avi`, `.mov`, `.mkv`

### 360° Video Requirements

- **Format**: Equirectangular projection preferred
- **Aspect Ratio**: ~2:1 (e.g., 7680x3840, 5760x2880)
- **Content**: Indoor home activities
- **Duration**: 30 seconds to 2+ hours
- **Resolution**: Automatically optimized for processing

## ⚙️ Configuration

### Sensitivity Levels

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

### Advanced Configuration

Edit `motion_analysis/src/utils/config.py`:

```python
# Motion Detection Thresholds
HIGH_MOTION_THRESHOLD = 0.02        # High activity threshold
MEDIUM_MOTION_THRESHOLD = 0.005     # Medium activity threshold
ACTIVITY_MIN_DURATION = 1.0         # Minimum segment duration (seconds)

# Processing Parameters
BG_VAR_THRESHOLD = 25               # Background sensitivity
MIN_FOREGROUND_AREA = 50            # Minimum motion area
```

## 🚀 Performance & Scaling

### Processing Speed
- **CPU**: 1-2x real-time (1 hour video → 1-2 hours processing)
- **GPU**: 2-4x real-time (1 hour video → 15-30 minutes processing)
- **Ray Cluster**: Scales linearly with additional nodes

### Memory Usage
- **Typical**: 2-4 GB RAM for standard videos
- **High-res**: Automatically downscaled for efficiency
- **Ray workers**: ~1-2 GB per worker

### Ray Cluster Setup

```python
# Single machine
ray.init()

# Multi-machine cluster
ray.init(address='ray://head-node-ip:10001')

# With resource constraints
ray.init(num_cpus=8, num_gpus=1)
```

## 🛠️ Advanced Usage

### Batch Processing

```python
import ray
from ray_jobs.motion_energy import analyze_motion_energy_only

ray.init()

video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]

# Process multiple videos in parallel
futures = [
    analyze_motion_energy_only.remote(video, "medium")
    for video in video_files
]

results = ray.get(futures)

for i, result in enumerate(results):
    if result["success"]:
        print(f"Video {i+1}: {result['total_segments']} segments")
    else:
        print(f"Video {i+1}: Failed - {result['error']}")
```

### Integration with Existing Pipelines

```python
# my_pipeline.py integration example
def enhanced_pipeline(input_video: str):
    
    # Stage 1: Video conversion (if needed)
    if input_video.endswith('.insv'):
        converted = ray.get(convert_insv_to_dual_mp4.remote(input_video))
        video_path = converted["output_view_1"]
    else:
        video_path = input_video
    
    # Stage 2: Motion energy analysis
    motion_results = ray.get(analyze_motion_energy_only.remote(
        video_path=video_path,
        sensitivity_level="medium"
    ))
    
    # Stage 3: Use motion results for downstream processing
    if motion_results["success"]:
        segments = motion_results["segments"]
        # Process segments for labeling, classification, etc.
        return process_segments(segments)
    
    return None
```

## 🧪 Testing

### Run Individual Tests

```bash
# Test motion energy analysis
python ray_jobs/motion_energy.py

# Test Ray integration
python tests/test_motion_ray.py  

# Test full pipeline
python my_pipeline.py
```

### Custom Video Testing

```bash
# Test with custom video
python ray_jobs/motion_energy.py /path/to/your/video.mp4
python tests/test_motion_ray.py /path/to/your/video.mp4
```

## 🛠️ Troubleshooting

### Common Issues

**"No motion segments detected":**
- Try increasing sensitivity: `sensitivity_level="high"`
- Verify video contains actual human activities
- Check video quality and lighting conditions

**"Processing very slow":**
- Normal for CPU-only processing (1-2x real-time)
- Install GPU acceleration: `pip install cupy-cuda12x`
- Consider using Ray cluster for parallel processing

**Import errors:**
- Ensure all `__init__.py` files exist in directories
- Check Python path includes project root
- Verify motion_analysis directory structure

### GPU Troubleshooting

```bash
# Check GPU availability
python -c "import cupy; print('GPU available:', cupy.cuda.runtime.getDeviceCount())"

# Force CPU mode if GPU issues
export CUDA_VISIBLE_DEVICES=""
```

## 📈 Example Results

### Sample Output for 39-second cooking video:
```
✅ SUCCESS: 3 segments found
   Duration: 39.6s
   Processing: 67.5s
   
   Activity Segments:
   1. 0.0s-4.4s (HIGH_ACTIVITY) - Confidence: 0.841
   2. 4.4s-17.5s (MEDIUM_ACTIVITY) - Confidence: 0.820  
   3. 24.5s-39.6s (MEDIUM_ACTIVITY) - Confidence: 0.818
   
   Motion Statistics:
   - High activity: 21.8% (259 frames)
   - Medium activity: 25.8% (306 frames)  
   - Low activity: 52.4% (621 frames)
```

## 🔮 System Requirements

**Minimum:**
- Python 3.8+
- 8GB RAM
- 4-core CPU

**Recommended:**
- Python 3.9+
- 16GB+ RAM  
- 8+ core CPU or NVIDIA GPU
- SSD storage for video processing

**For Ray Clusters:**
- Network connectivity between nodes
- Shared storage or distributed file system
- Consistent Python environments

## 📄 License

This project is developed for research and educational purposes. Please ensure compliance with your organization's policies when processing video data.

---

**🏠📹 Ready to analyze your 360° home activities with scalable Ray processing!**

### Recent Updates
- ✅ **Ray Integration**: Distributed processing with Ray clusters
- ✅ **Clean API**: Simple `analyze_motion_energy_only()` function
- ✅ **Production Ready**: Minimal dependencies and clean integration
- ✅ **GPU Acceleration**: CUDA support for faster processing
- ✅ **Pipeline Integration**: Easy integration into existing workflows