import sys
import json
import cv2
from datetime import timedelta
from pathlib import Path

# make ../ importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from deepface import DeepFace
from utils import config as C


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_gender(g):
    """
    DeepFace can return:
      - a string label ("Man"/"Woman"), or
      - a dict with probabilities {"Woman": np.float32(...), "Man": np.float32(...)}
    Return (label, scores_dict_as_float) where label is "Man"/"Woman".
    """
    if isinstance(g, str):
        return g, None
    if isinstance(g, dict):
        # convert np.float32 to float and pick max label
        scores = {k: float(v) for k, v in g.items()}
        label = max(scores, key=scores.get)
        return label, scores
    return None, None


def analyze_frame(frame):
    """
    Returns list[dict] with per-face results:
      [{
        "age": number or null,
        "gender_label": "Man"/"Woman"/null,
        "gender_scores": {"Woman": float, "Man": float} or null
      }, ...]
    """
    result = DeepFace.analyze(
        frame,
        actions=C.ACTIONS,
        enforce_detection=C.ENFORCE_DETECTION
    )

    # DeepFace sometimes returns dict or list
    if isinstance(result, dict):
        result = [result]

    faces_out = []
    for r in result:
        age = r.get("age", None)
        gender_raw = r.get("gender", None)
        gender_label, gender_scores = normalize_gender(gender_raw)
        faces_out.append({
            "age": age if (age is None or isinstance(age, (int, float))) else float(age),
            "gender_label": gender_label,
            "gender_scores": gender_scores
        })
    return faces_out


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/video_age.py /path/to/video.mp4")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"ERROR: Video path not found: {video_path}")
        sys.exit(1)

    video_stem = video_path.stem
    base_out = ensure_dir(Path(C.OUTPUT_DIR) / video_stem)
    frames_out = ensure_dir(base_out / "frames")
    json_path = base_out / "predictions.json"
    jsonl_path = base_out / "predictions.jsonl"

    print(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Collect nested JSON in-memory (switch to JSONL if video is huge)
    frames_json = []

    frame_num = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % C.FRAME_INTERVAL == 0:
                timestamp = str(timedelta(seconds=int(frame_num / fps)))
                frame_filename = f"frame_{frame_num:06d}.jpg"
                frame_path = frames_out / frame_filename

                try:
                    faces = analyze_frame(frame)  # list of face dicts
                    num_faces = len(faces)

                    should_save = C.SAVE_FRAMES and (num_faces > 0 if C.SAVE_ONLY_DETECTIONS else True)
                    if should_save:
                        cv2.imwrite(str(frame_path), frame)

                    frame_record = {
                        "frame_num": frame_num,
                        "timestamp": timestamp,
                        "frame_path": str(frame_path) if should_save else "",
                        "num_faces": num_faces,
                        "faces": faces,
                        "error": ""
                    }

                except Exception as e:
                    # optional save on error
                    if C.SAVE_FRAMES and not C.SAVE_ONLY_DETECTIONS:
                        cv2.imwrite(str(frame_path), frame)

                    frame_record = {
                        "frame_num": frame_num,
                        "timestamp": timestamp,
                        "frame_path": str(frame_path) if C.SAVE_FRAMES and not C.SAVE_ONLY_DETECTIONS else "",
                        "num_faces": 0,
                        "faces": [],
                        "error": str(e)
                    }

                if C.WRITE_JSON:
                    frames_json.append(frame_record)

                if C.WRITE_JSONL:
                    with open(jsonl_path, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(frame_record, ensure_ascii=False) + "\n")

                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed} sampled frames (last={frame_num})")

            frame_num += 1

    finally:
        cap.release()

    # Final JSON dump
    if C.WRITE_JSON:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_path": str(video_path),
                "frame_interval": C.FRAME_INTERVAL,
                "results": frames_json
            }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Outputs saved to: {base_out}")
    if C.WRITE_JSON:
        print(f"- JSON:  {json_path}")
    if C.WRITE_JSONL:
        print(f"- JSONL: {jsonl_path}")
    if C.SAVE_FRAMES:
        print(f"- Frames: {frames_out}")


if __name__ == "__main__":
    main()
import sys
import json
import cv2
from datetime import timedelta
from pathlib import Path

# make ../ importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

from insightface.app import FaceAnalysis
from utils import config as C


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_gender_from_sex(sex_value):
    """
    InsightFace Face.sex: 1 => Male, 0 => Female
    Returns (label, scores_dict_or_none)
    (InsightFace does not provide gender probabilities; scores will be None.)
    """
    if sex_value is None:
        return None, None
    return ("Male" if int(sex_value) == 1 else "Female"), None


# Lazy/global init so we only load the model once
_APP = None
def get_insightface_app():
    global _APP
    if _APP is None:
        # Optional overrides from config (safe defaults provided)
        model_name = getattr(C, "INSIGHTFACE_MODEL_NAME", "buffalo_l")
        ctx_id = getattr(C, "INSIGHTFACE_CTX_ID", 0)   # 0 = GPU if available, -1 = CPU
        _APP = FaceAnalysis(name=model_name)
        _APP.prepare(ctx_id=ctx_id)
    return _APP


def analyze_frame(frame):
    """
    Returns list[dict] with per-face results:
      [{
        "age": float|int|null,
        "gender_label": "Male"/"Female"/null,
        "gender_scores": null  # InsightFace doesn't expose gender probabilities
      }, ...]
    """
    app = get_insightface_app()
    faces = app.get(frame)  # list of Face objects
    out = []
    for f in faces:
        age_val = f.age if (f.age is None or isinstance(f.age, (int, float))) else float(f.age)
        gender_label, gender_scores = normalize_gender_from_sex(getattr(f, "sex", None))
        out.append({
            "age": age_val,
            "gender_label": gender_label,
            "gender_scores": gender_scores
        })
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/video_age_insightface.py /path/to/video.mp4")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"ERROR: Video path not found: {video_path}")
        sys.exit(1)

    video_stem = video_path.stem
    base_out = ensure_dir(Path(C.OUTPUT_DIR) / video_stem)
    frames_out = ensure_dir(base_out / "frames")
    json_path = base_out / "predictions_insightface.json"
    jsonl_path = base_out / "predictions_insightface.jsonl"

    print(f"Processing video (InsightFace): {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames_json = []
    frame_num = 0
    processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % C.FRAME_INTERVAL == 0:
                timestamp = str(timedelta(seconds=int(frame_num / fps)))
                frame_filename = f"frame_{frame_num:06d}.jpg"
                frame_path = frames_out / frame_filename

                try:
                    faces = analyze_frame(frame)
                    num_faces = len(faces)

                    should_save = C.SAVE_FRAMES and (num_faces > 0 if C.SAVE_ONLY_DETECTIONS else True)
                    if should_save:
                        cv2.imwrite(str(frame_path), frame)

                    frame_record = {
                        "frame_num": frame_num,
                        "timestamp": timestamp,
                        "frame_path": str(frame_path) if should_save else "",
                        "num_faces": num_faces,
                        "faces": faces,
                        "error": ""
                    }

                except Exception as e:
                    if C.SAVE_FRAMES and not C.SAVE_ONLY_DETECTIONS:
                        cv2.imwrite(str(frame_path), frame)

                    frame_record = {
                        "frame_num": frame_num,
                        "timestamp": timestamp,
                        "frame_path": str(frame_path) if C.SAVE_FRAMES and not C.SAVE_ONLY_DETECTIONS else "",
                        "num_faces": 0,
                        "faces": [],
                        "error": str(e)
                    }

                if getattr(C, "WRITE_JSON", True):
                    frames_json.append(frame_record)

                if getattr(C, "WRITE_JSONL", False):
                    with open(jsonl_path, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(frame_record, ensure_ascii=False) + "\n")

                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed} sampled frames (last={frame_num})")

            frame_num += 1

    finally:
        cap.release()

    if getattr(C, "WRITE_JSON", True):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_path": str(video_path),
                "frame_interval": C.FRAME_INTERVAL,
                "insightface_model": getattr(C, "INSIGHTFACE_MODEL_NAME", "buffalo_l"),
                "results": frames_json
            }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done (InsightFace). Outputs saved to: {base_out}")
    if getattr(C, "WRITE_JSON", True):
        print(f"- JSON:  {json_path}")
    if getattr(C, "WRITE_JSONL", False):
        print(f"- JSONL: {jsonl_path}")
    if C.SAVE_FRAMES:
        print(f"- Frames: {frames_out}")


if __name__ == "__main__":
    main()
