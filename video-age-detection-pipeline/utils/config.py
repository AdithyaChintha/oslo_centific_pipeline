# utils/config.py
FRAME_INTERVAL = 30
ENFORCE_DETECTION = False
ACTIONS = ["age", "gender"]

OUTPUT_DIR = "outputs"
SAVE_FRAMES = True
SAVE_ONLY_DETECTIONS = False

WRITE_JSON = True      # <-- turn on nested JSON

SEED = 42

# Age classification thresholds
MINOR_AGE_THRESHOLD = 18      # Under 18 is considered minor
SENIOR_AGE_THRESHOLD = 65     # 65 and above is considered senior
# Ages 18-64 are considered adult
