"""
Configuration constants for 360° Motion Analysis Pipeline.
"""

# Background Subtraction Parameters
BG_HISTORY = 200               # Number of frames to learn background
BG_VAR_THRESHOLD = 25           # Sensitivity to background changes
BG_DETECT_SHADOWS = True        # Whether to detect and ignore shadows

# Motion Detection Parameters
MIN_FOREGROUND_AREA = 50       # Minimum area for motion detection (reduced for home activities)
CONFIDENCE_THRESHOLD = 0.05      # Motion confidence threshold (more sensitive for home)
DISTANCE_THRESHOLD = 50         # Distance threshold for object matching

# Temporal Analysis Parameters
ACTIVITY_TIME_THRESHOLD = 60.0  # Time threshold for activity detection (1 minute)
UNMATCHED_FRAME_TOLERANCE = 45  # Frames before considering object "gone" (~1.5s at 30fps)

# Video Processing Parameters
DEFAULT_OUTPUT_FORMAT = "mp4"
PROGRESS_UPDATE_INTERVAL = 100  # Print progress every N frames

# Morphological Operations
MORPH_KERNEL_SIZE = (5, 5)      # Kernel size for noise removal

# 360° Video Specific
EQUIRECTANGULAR_WIDTH = 5760    # Typical 360° video width
EQUIRECTANGULAR_HEIGHT = 2880   # Typical 360° video height

# Segmentation Parameters
MOTION_SMOOTH_WINDOW = 15       # Frames to smooth motion data (1 second at 30fps)
# ACTIVITY_MIN_DURATION = 3.0    # Minimum activity duration (seconds)
# HIGH_MOTION_THRESHOLD = 0.05    # Threshold for high activity classification
# MEDIUM_MOTION_THRESHOLD = 0.01   # Threshold for medium activity classification
# New settings (optimized for short home activities)
ACTIVITY_MIN_DURATION = 1.0     # 1 second minimum
HIGH_MOTION_THRESHOLD = 0.02    # More sensitive
MEDIUM_MOTION_THRESHOLD = 0.005  # Much more sensitive  
CONFIDENCE_THRESHOLD = 0.02     # Lower confidence requirement

# Output Parameters
SAVE_ANNOTATED_VIDEO = True     # Whether to save video with motion overlay
SAVE_MOTION_DATA = True         # Whether to save motion timeline data
DEBUG_MODE = True               # Enable debug logging and visualizations

# GPU Configuration
USE_GPU = True                      # Enable GPU acceleration
GPU_MEMORY_FRACTION = 0.8          # Use 80% of GPU memory
FORCE_CPU_FALLBACK = False         # Force CPU if GPU fails

# Video Analysis Pipeline Configuration
DEFAULT_PIPELINE_CONFIG = {
    "output_directory": "output",
    "sensitivity_level": "medium",
    "save_annotated_video": True,
    "save_motion_data": True,
    "verbose_logging": True,
    "log_directory": "logs"
}