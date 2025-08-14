import sys
import json
from datetime import timedelta
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple

# make project root (video-age-detection-pipeline) importable and preferred
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
import importlib.util as _importlib_util

# Robustly load config.py from this pipeline's utils to avoid clashes with top-level `utils`
_cfg_path = Path(__file__).resolve().parent.parent / "utils" / "config.py"
_spec = _importlib_util.spec_from_file_location("va_config", str(_cfg_path))
_cfg_module = _importlib_util.module_from_spec(_spec) if _spec and _spec.loader else None
if _cfg_module and _spec and _spec.loader:
    _spec.loader.exec_module(_cfg_module)
    C = _cfg_module  # type: ignore
else:
    raise ImportError(f"Could not load config from {_cfg_path}")

# InsightFace imports for face detection
try:
    from insightface.app import FaceAnalysis
except Exception as import_error:  # pragma: no cover
    FaceAnalysis = None
    _INSIGHTFACE_IMPORT_ERROR = import_error
else:
    _INSIGHTFACE_IMPORT_ERROR = None


class ViTVideoAgeDetector:
    """
    Detect faces per frame using InsightFace, then classify age on face crops
    with 'nateraw/vit-age-classifier'. Outputs JSON in the same schema as the
    other detectors (deepfacedetect/insightfacedetect).
    """

    AGE_LABEL_TO_REP_AGE: Dict[str, float] = {
        "0-2": 1.0,
        "3-9": 6.0,
        "10-19": 15.0,
        "20-29": 25.0,
        "30-39": 35.0,
        "40-49": 45.0,
        "50-59": 55.0,
        "60-69": 65.0,
        "70+": 75.0,
    }

    def __init__(self, config=None, model_name: str = "buffalo_l", ctx_id: int = -1, det_size: Tuple[int, int] = (640, 640)):
        if _INSIGHTFACE_IMPORT_ERROR is not None:
            raise RuntimeError(f"Failed to import InsightFace: {_INSIGHTFACE_IMPORT_ERROR}")

        self.config = config or C
        self.processed_frames = 0
        self.frames_json: List[Dict[str, Any]] = []

        # Face detector
        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])  # CPU by default
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # ViT age classifier
        self.model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        self.processor = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

    def ensure_dir(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def classify_age_band_to_numeric(self, label: str) -> float | None:
        return self.AGE_LABEL_TO_REP_AGE.get(label)

    def classify_age_group(self, age_value: float | None) -> str:
        if age_value is None:
            return "unknown"
        age_value = float(age_value)
        if age_value < self.config.MINOR_AGE_THRESHOLD:
            return "minor"
        if age_value >= self.config.SENIOR_AGE_THRESHOLD:
            return "senior"
        return "adult"

    def _crop_face(self, frame: np.ndarray, bbox: np.ndarray, margin: float = 0.2) -> Image.Image:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)
        bw = x2 - x1
        bh = y2 - y1
        # add margin around bbox
        mx = int(bw * margin)
        my = int(bh * margin)
        xx1 = max(0, x1 - mx)
        yy1 = max(0, y1 - my)
        xx2 = min(w, x2 + mx)
        yy2 = min(h, y2 + my)
        crop = frame[yy1:yy2, xx1:xx2]
        if crop.size == 0:
            # fallback to entire frame if bbox invalid
            crop = frame
        # Convert to PIL Image
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    def _predict_age_for_crop(self, image: Image.Image) -> tuple[float | None, str | None, float | None]:
        inputs = self.processor(image, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
            proba = outputs.logits.softmax(dim=1)
            pred_id = int(proba.argmax(dim=1).item())
            pred_label = self.model.config.id2label[pred_id]
            conf = float(proba[0, pred_id])
        age_value = self.classify_age_band_to_numeric(pred_label)
        return age_value, pred_label, conf

    def analyze_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        faces = self.app.get(frame)
        faces_out: List[Dict[str, Any]] = []

        for f in faces:
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            pil_face = self._crop_face(frame, np.asarray(bbox))
            age_value, age_band, conf = self._predict_age_for_crop(pil_face)

            faces_out.append({
                "age": age_value,
                "age_classification": self.classify_age_group(age_value),
                "gender_label": None,
                "gender_scores": None,
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "age_band": age_band,
                "age_confidence": conf,
                # Optional: could include age_band/conf if needed later
            })

        return faces_out

    def _draw_overlays(self, frame: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        annotated = frame.copy()
        
        def _age_color(age_cls: str) -> tuple[int, int, int]:
            # BGR colors
            if age_cls == "minor":
                return (0, 0, 255)    # red
            if age_cls == "senior":
                return (0, 255, 255)  # yellow
            return (0, 200, 0)        # green-ish for adult

        for face in faces:
            bbox = face.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            age_val = face.get("age")
            age_cls = face.get("age_classification", "")
            age_text = "?" if not isinstance(age_val, (int, float)) else str(int(round(age_val)))
            label = f"{age_text} ({age_cls})"

            color = _age_color(age_cls)

            # dynamic scale by face size
            box_h = max(1, y2 - y1)
            scale = max(0.7, min(1.6, box_h / 120.0))
            thickness = max(2, int(round(scale * 2)))

            # box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # text with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
            pad = max(4, int(4 * scale))
            tx1, ty1 = x1, max(0, y1 - th - 2 * pad)
            tx2, ty2 = x1 + tw + 2 * pad, y1
            cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x1 + pad, y1 - pad), font, scale, color, thickness, cv2.LINE_AA)

        return annotated

    def process_video(self, video_path: Union[str, Path]):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video path not found: {video_path}")

        video_stem = video_path.stem
        base_out = self.ensure_dir(Path(self.config.OUTPUT_DIR) / video_stem)
        frames_out = self.ensure_dir(base_out / "frames")
        json_path = base_out / "predictions.json"

        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Reset state
        self.processed_frames = 0
        self.frames_json = []

        try:
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % self.config.FRAME_INTERVAL == 0:
                    timestamp = str(timedelta(seconds=int(frame_num / fps)))
                    frame_filename = f"frame_{frame_num:06d}.jpg"
                    frame_path = frames_out / frame_filename

                    try:
                        faces = self.analyze_frame(frame)
                        num_faces = len(faces)
                        should_save = self.config.SAVE_FRAMES and (num_faces > 0 if self.config.SAVE_ONLY_DETECTIONS else True)
                        if should_save:
                            to_save = self._draw_overlays(frame, faces) if num_faces > 0 else frame
                            cv2.imwrite(str(frame_path), to_save)
                        frame_record = {
                            "frame_num": frame_num,
                            "timestamp": timestamp,
                            "frame_path": str(frame_path) if should_save else "",
                            "num_faces": num_faces,
                            "faces": faces,
                            "error": "",
                        }
                    except Exception as e:
                        if self.config.SAVE_FRAMES and not self.config.SAVE_ONLY_DETECTIONS:
                            cv2.imwrite(str(frame_path), frame)
                        frame_record = {
                            "frame_num": frame_num,
                            "timestamp": timestamp,
                            "frame_path": str(frame_path) if self.config.SAVE_FRAMES and not self.config.SAVE_ONLY_DETECTIONS else "",
                            "num_faces": 0,
                            "faces": [],
                            "error": str(e),
                        }

                    if self.config.WRITE_JSON:
                        self.frames_json.append(frame_record)
                    self.processed_frames += 1

                    if self.processed_frames % 50 == 0:
                        print(f"Processed {self.processed_frames} sampled frames (last={frame_num})")

                frame_num += 1
        finally:
            cap.release()

        # Save results
        if self.config.WRITE_JSON:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "video_path": str(video_path),
                    "frame_interval": self.config.FRAME_INTERVAL,
                    "results": self.frames_json,
                }, f, ensure_ascii=False, indent=2)

        return {
            "video_path": str(video_path),
            "output_dir": str(base_out),
            "processed_frames": self.processed_frames,
            "total_frames": len(self.frames_json),
        }

    def get_processing_summary(self):
        age_counts = {"minor": 0, "adult": 0, "senior": 0, "unknown": 0}
        for frame in self.frames_json:
            for face in frame.get("faces", []):
                age_class = face.get("age_classification", "unknown")
                age_counts[age_class] += 1
        return {
            "processed_frames": self.processed_frames,
            "total_faces_detected": sum(len(frame.get("faces", [])) for frame in self.frames_json),
            "frames_with_errors": sum(1 for frame in self.frames_json if frame.get("error")),
            "age_classifications": age_counts,
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/vit_video_agedetect.py /path/to/video.mp4")
        sys.exit(1)

    try:
        detector = ViTVideoAgeDetector()
        results = detector.process_video(sys.argv[1])

        print(f"\n✅ Done. Outputs saved to: {results['output_dir']}")
        print(f"- Processed {results['processed_frames']} frames")

        summary = detector.get_processing_summary()
        print(f"- Total faces detected: {summary['total_faces_detected']}")
        if summary['frames_with_errors'] > 0:
            print(f"- Frames with errors: {summary['frames_with_errors']}")

        age_stats = summary.get('age_classifications', {})
        if age_stats:
            print("\nAge Classification Summary:")
            for age_class, count in age_stats.items():
                if count > 0:
                    print(f"  - {age_class.capitalize()}: {count}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


