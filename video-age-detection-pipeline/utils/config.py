# utils/config.py
FRAME_INTERVAL = 30
ENFORCE_DETECTION = False
ACTIONS = ["age", "gender"]

OUTPUT_DIR = "outputs"
SAVE_FRAMES = True
SAVE_ONLY_DETECTIONS = False

WRITE_JSON = True      # <-- turn on nested JSON
WRITE_JSONL = False    # optional line-delimited JSON
WRITE_CSV = False      # disable CSV since JSON is your target

SEED = 42
