# 360° Home Activity Motion Analysis Pipeline

A Python-based motion analysis system for detecting and segmenting activities in 360° home recordings using MOG2 background subtraction and temporal analysis.

## 🚧 Current Status: Work in Progress

This project is currently **under development** with basic motion detection working and segmentation being refined. See [Current Issues](#-current-issues-being-resolved) section for known limitations.

## 🎯 Overview

This pipeline processes 360° home activity videos (walking, bed-making, cooking, cleaning) and attempts to segment them into meaningful activity periods. It uses computer vision techniques adapted from surveillance systems for home activity analysis.

### Key Features ✅ Working

- **🎥 360° Video Support**: Processes equirectangular format videos from Insta360 and similar cameras
- **🔍 Motion Detection**: Uses MOG2 background subtraction for motion detection
- **📊 Motion Statistics**: Detailed motion energy analysis and timeline data  
- **📝 Analysis Reports**: Generates comprehensive analysis reports
- **🔧 Configurable**: Adjustable sensitivity and processing parameters

### Features 🚧 In Development

- **⚡ Activity Segmentation**: Currently struggles with scattered activity detection
- **🎨 Video Overlay**: Annotated video output (implemented but optional)
- **📈 Advanced Analytics**: Peak detection and change point analysis (Part 2)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the project
git clone https://github.com/nvidia-coe/insta360-video-activity-segmentation.git
cd feature/motion_energy_analysis

# Install dependencies
pip install -r requirements.txt

# Verify ffmpeg installation (required for video processing)
ffmpeg -version
```

### 2. Basic Usage

**Edit the configuration in `main.py`:**

```python
# 📝 CONFIGURATION - Edit these paths
VIDEO_PATH = "path/to/your_360_video.mp4"  # 👈 Your video path
OUTPUT_DIRECTORY = "output"                 # Results folder
SAVE_ANNOTATED_VIDEO = False               # Video overlay (optional)
SAVE_MOTION_DATA = True                    # Motion timeline (recommended)
SENSITIVITY_LEVEL = "medium"               # low/medium/high
```

**Run the analysis:**

```bash
python main.py
```

### 3. What You'll Get

The pipeline generates:
- **📄 `segments.json`**: Activity segments (may be empty - see current issues)
- **📊 `motion_data.json`**: Complete frame-by-frame motion timeline ✅
- **📝 `summary.txt`**: Human-readable analysis report ✅
- **🎥 `annotated.mp4`**: Video with motion overlay (optional)

## 🔧 Configuration for Short Videos

**Important**: Default settings are for long videos. For short videos (< 2 minutes), modify `src/utils/config.py`:

```python
# For short videos, change these values:
BG_HISTORY = 200                # Down from 5000 (important!)
ACTIVITY_MIN_DURATION = 3.0     # Down from 30.0 seconds
HIGH_MOTION_THRESHOLD = 0.05    # Down from 0.7 (more sensitive)
MEDIUM_MOTION_THRESHOLD = 0.01  # Down from 0.3 (more sensitive)
MIN_FOREGROUND_AREA = 50        # Down from 200 pixels
CONFIDENCE_THRESHOLD = 0.05     # Down from 0.2
```

## 📊 Understanding Motion Results

### What Works Well ✅

**Motion Detection Statistics:**
```
Average Motion Energy: 0.0121      # Baseline activity level
Peak Motion Energy: 1.0000         # Highest motion detected
High Activity: 5.5% of video       # Frames with significant motion
Medium Activity: 22.6% of video    # Frames with moderate motion
Low Activity: 71.9% of video       # Quiet/static frames
```

**Motion Timeline:** The `motion_data.json` contains frame-by-frame motion energy values (0.0-1.0 scale) showing exactly when motion occurs.

### Current Limitations ⚠️

**Activity Segmentation:** Motion is detected successfully, but converting scattered motion frames into coherent activity segments is challenging. The algorithm expects continuous activity but real-world activities have natural pauses and variations.

## 🔍 Current Issues Being Resolved

### Issue 1: Zero Segments Despite Motion Detection
**Status**: Motion detection working (detects 28% activity) but segmentation fails
**Cause**: Real activities have gaps/pauses that break segmentation logic
**Workaround**: Check motion_data.json for actual activity patterns

### Issue 2: Scattered vs Continuous Activity  
**Problem**: Algorithm expects continuous motion, but walking/bed-making has natural pauses
**Example**: Person detected in frames 100-180, but with dips that break segments
**Solution**: Advanced gap-tolerant segmentation (in development)

### Issue 3: Threshold Calibration
**Challenge**: Default thresholds too high for subtle home activities
**Current**: Requires manual config.py tuning per video type
**Goal**: Auto-calibration based on video characteristics

## 🛠️ Development Notes

### Architecture Status
- **Part 1**: Core segmentation (basic threshold method) ✅ Complete
<!-- - **Part 2**: Advanced segmentation (gap tolerance, peak detection) 🚧 In Progress -->
- **Integration**: Combining both approaches 📋 Planned

### Testing Approach
Currently tested on:
- 39-second home activity video (walking + bed arrangement)
- Motion detection: Working perfectly
- Segmentation: Needs refinement

## 🎥 Video Requirements

- **Format**: .mp4, .avi, .mov, .mkv (360° equirectangular preferred)
- **Content**: Indoor human activities
- **Duration**: 30 seconds to 2+ hours (shorter videos need config changes)
- **Quality**: Any resolution (high-res videos are automatically downscaled)

## 📈 Performance

### What to Expect
- **Processing Speed**: ~0.1x real-time (39s video = 5 minutes processing)
- **Memory Usage**: 1-2 GB RAM
- **Motion Detection**: Very reliable
- **Segmentation**: Currently inconsistent (being improved)

## 🔮 Roadmap

### Immediate Priorities
1. **Fix segmentation gaps** - Handle natural activity pauses
2. **Better debug output** - Help users understand what's detected


## 🚨 Known Workarounds

If you get "0 segments detected":

1. **Check motion detection first**: Look at motion statistics in summary report
2. **Lower thresholds**: Edit config.py values (see Configuration section)
3. **Analyze motion timeline**: Check motion_data.json for actual patterns
4. **Verify video content**: Ensure it contains actual human movement

## 📞 Getting Help

### Debug Steps
1. **Enable verbose logging**: Set `VERBOSE_LOGGING = True` in main.py
2. **Check motion statistics**: Look for non-zero activity percentages  
3. **Examine motion timeline**: Review motion_data.json values
4. **Try different thresholds**: Start with high sensitivity settings

### Current Development
This is an active research project. The motion detection foundation is solid, but activity segmentation is being refined. Contributions and feedback welcome!

---

**Current Status**: Motion detection working excellently, segmentation logic being improved. 🔧🎯