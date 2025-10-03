import sys
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional

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

    def __init__(self, config=None, model_name: str = "buffalo_l", ctx_id: int = -1, det_size: Tuple[int, int] = (480, 480)):
        if _INSIGHTFACE_IMPORT_ERROR is not None:
            raise RuntimeError(f"Failed to import InsightFace: {_INSIGHTFACE_IMPORT_ERROR}")

        self.config = config or C
        self.processed_frames = 0
        self.frames_json: List[Dict[str, Any]] = []

        # Determine available providers for InsightFace
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"✅ Using GPU providers: {providers}")
            else:
                providers = ['CPUExecutionProvider']
                print(f"⚠️ GPU not available, using CPU providers: {providers}")
        except ImportError:
            providers = ['CPUExecutionProvider']
            print("⚠️ ONNX Runtime not available, using CPU providers")

        # Face detector
        self.app = FaceAnalysis(name=model_name, providers=providers, allowed_modules=['detection'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # ViT age classifier - move to GPU if available
        self.model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        self.processor = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            print(f"✅ ViT model moved to GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("⚠️ GPU not available, using CPU for ViT model")

    def ensure_dir(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def classify_age_band_to_numeric(self, label: str) -> Optional[float]:
        return self.AGE_LABEL_TO_REP_AGE.get(label)

    def classify_age_group(self, age_value: Optional[float]) -> str:
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

    def _predict_age_for_crop(self, image: Image.Image) -> Tuple[Optional[float], Optional[str], Optional[float]]:
        inputs = self.processor(image, return_tensors='pt')
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            proba = outputs.logits.softmax(dim=1)
            pred_id = int(proba.argmax(dim=1).item())
            pred_label = self.model.config.id2label[pred_id]
            conf = float(proba[0, pred_id])
        age_value = self.classify_age_band_to_numeric(pred_label)
        return age_value, pred_label, conf

    def analyze_frame_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Analyze multiple frames in a single batch with PARALLEL face detection.

        OPTIMIZATION STRATEGY:
        1. Detect faces in parallel across all frames (CPU-bound, parallelized with ThreadPool)
        2. Collect all faces from all frames
        3. Run ViT age classification on ALL faces in ONE GPU batch

        Args:
            frames: List of numpy arrays (frames to process)

        Returns:
            List of face detection results (one list per frame)
        """
        if not frames:
            return []

        import time
        batch_start = time.time()
        from concurrent.futures import ThreadPoolExecutor

        # Step 1: PARALLEL face detection across all frames using ThreadPool
        # Note: We use ThreadPool instead of Ray here because:
        #   - Face detection is CPU-bound (InsightFace ONNX on CPU)
        #   - ThreadPool has lower overhead for small tasks
        #   - We're already inside a Ray task, so nested Ray calls would add overhead
        #   - ThreadPool allows better CPU utilization for I/O-bound operations

        def detect_faces_in_frame(frame_data):
            """Detect faces in a single frame (runs in parallel thread)."""
            frame_idx, frame = frame_data

            # InsightFace face detection (CPU-bound)
            faces = self.app.get(frame)

            frame_faces = []
            face_images = []

            for face in faces:
                bbox = getattr(face, "bbox", None)
                if bbox is None:
                    continue

                try:
                    # Crop face from frame
                    pil_face = self._crop_face(frame, np.asarray(bbox))
                    face_images.append(pil_face)
                    frame_faces.append(face)
                except:
                    continue

            return frame_idx, frame_faces, face_images

        # Use ThreadPool to parallelize face detection across frames
        # num_workers: 4-8 threads work well for CPU-bound face detection
        num_workers = min(8, len(frames), os.cpu_count() or 4)

        detection_start = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Map each frame to a thread for parallel processing
            detection_results = list(executor.map(detect_faces_in_frame, enumerate(frames)))
        detection_time = time.time() - detection_start

        # Step 2: Collect ALL faces from ALL frames
        collection_start = time.time()
        all_face_images = []
        all_frame_faces = [[] for _ in frames]
        face_to_frame_mapping = []  # Track (frame_idx, local_face_idx) for each face

        for frame_idx, frame_faces, face_images in detection_results:
            all_frame_faces[frame_idx] = frame_faces

            for local_idx, face_img in enumerate(face_images):
                all_face_images.append(face_img)
                face_to_frame_mapping.append((frame_idx, local_idx))

        collection_time = time.time() - collection_start

        # Step 3: Process ALL faces from ALL frames in ONE GPU batch
        if not all_face_images:
            # No faces detected in any frame
            print(f"⏱️ BATCH TIMING: detection={detection_time:.3f}s | collection={collection_time:.3f}s | total={time.time()-batch_start:.3f}s | frames={len(frames)} | faces=0")
            return [[] for _ in frames]

        try:
            # Batch process ALL face images at once on GPU
            preprocessing_start = time.time()
            inputs = self.processor(all_face_images, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            preprocessing_time = time.time() - preprocessing_start

            gpu_inference_start = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
                proba = outputs.logits.softmax(dim=1)
                pred_ids = proba.argmax(dim=1)  # Shape: [total_faces_across_all_frames]
                confs = proba.max(dim=1).values
            gpu_inference_time = time.time() - gpu_inference_start

            # Step 4: Distribute results back to respective frames
            postprocessing_start = time.time()
            results_per_frame = [[] for _ in frames]

            for face_idx, (frame_idx, local_face_idx) in enumerate(face_to_frame_mapping):
                pred_id = int(pred_ids[face_idx].item())
                pred_label = self.model.config.id2label[pred_id]
                age_value = self.classify_age_band_to_numeric(pred_label)

                face_obj = all_frame_faces[frame_idx][local_face_idx]

                face_result = {
                    "age": age_value,
                    "age_classification": self.classify_age_group(age_value),
                    "gender_label": None,
                    "gender_scores": None,
                    "bbox": [int(face_obj.bbox[0]), int(face_obj.bbox[1]),
                            int(face_obj.bbox[2]), int(face_obj.bbox[3])],
                    "age_band": pred_label,
                    "age_confidence": float(confs[face_idx].item()),
                }

                results_per_frame[frame_idx].append(face_result)

            postprocessing_time = time.time() - postprocessing_start
            total_time = time.time() - batch_start

            print(f"⏱️ BATCH TIMING: detection={detection_time:.3f}s ({detection_time/total_time*100:.1f}%) | "
                  f"collection={collection_time:.3f}s ({collection_time/total_time*100:.1f}%) | "
                  f"preprocess={preprocessing_time:.3f}s ({preprocessing_time/total_time*100:.1f}%) | "
                  f"gpu_inference={gpu_inference_time:.3f}s ({gpu_inference_time/total_time*100:.1f}%) | "
                  f"postprocess={postprocessing_time:.3f}s ({postprocessing_time/total_time*100:.1f}%) | "
                  f"total={total_time:.3f}s | frames={len(frames)} | faces={len(all_face_images)} | workers={num_workers}")

        except Exception as e:
            print(f"⚠️ Batch processing failed: {e}, falling back to per-frame")
            # Fallback: process each frame individually
            results_per_frame = [self.analyze_frame(frame) for frame in frames]

        return results_per_frame

    def analyze_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        faces = self.app.get(frame)
        faces_out: List[Dict[str, Any]] = []

        face_images = []
        valid_faces = []
        for f in faces:
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            try:
                pil_face = self._crop_face(frame, np.asarray(bbox))
                face_images.append(pil_face)
                valid_faces.append(f)
            except Exception as e:
                # Skip invalid face crops
                continue
        if not face_images:
            return []

        # OPTIMIZATION : Batch process ALL faces with ViT in ONE forward pass
        try:
            # Processor can handle list of images - adds padding automatically
            inputs = self.processor(face_images, return_tensors='pt', padding=True)

            # Single transfer to GPU for all faces
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                proba = outputs.logits.softmax(dim=1)
                pred_ids = proba.argmax(dim=1)  # Shape: [num_faces]
                confs = proba.max(dim=1).values  # Shape: [num_faces]

            # Build results for all faces
            for i, f in enumerate(valid_faces):
                pred_id = int(pred_ids[i].item())
                pred_label = self.model.config.id2label[pred_id]
                age_value = self.classify_age_band_to_numeric(pred_label)

                faces_out.append({
                    "age": age_value,
                    "age_classification": self.classify_age_group(age_value),
                    "gender_label": None,
                    "gender_scores": None,
                    "bbox": [int(f.bbox[0]), int(f.bbox[1]), int(f.bbox[2]), int(f.bbox[3])],
                    "age_band": pred_label,
                    "age_confidence": float(confs[i].item()),
                })

        except Exception as e:
            # Fallback to per-face processing if batch fails
            print(f"⚠️ Batch processing failed, falling back to per-face: {e}")
            for f in valid_faces:
                try:
                    pil_face = self._crop_face(frame, np.asarray(f.bbox))
                    age_value, age_band, conf = self._predict_age_for_crop(pil_face)

                    faces_out.append({
                        "age": age_value,
                        "age_classification": self.classify_age_group(age_value),
                        "gender_label": None,
                        "gender_scores": None,
                        "bbox": [int(f.bbox[0]), int(f.bbox[1]), int(f.bbox[2]), int(f.bbox[3])],
                        "age_band": age_band,
                        "age_confidence": conf,
                    })
                except:
                    continue
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

    def process_video(self, video_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        import time
        total_start = time.time()
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video path not found: {video_path}")

        video_stem = video_path.stem
        # Use provided output_dir if specified, otherwise fall back to config default
        if output_dir is not None:
            base_out = self.ensure_dir(Path(output_dir))
        else:
            base_out = self.ensure_dir(Path(self.config.OUTPUT_DIR) / video_stem)
        
        # Only create frames directory if frames are being saved
        frames_out = None
        if self.config.SAVE_FRAMES:
            frames_out = self.ensure_dir(base_out / "frames")
        
        json_path = base_out / "predictions.json"

        
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Reset state
        self.processed_frames = 0
        self.frames_json = []

        FRAME_BATCH_SIZE = 32  # Process 16 frames at once (adjust based on GPU memory)
        video_read_time = 0
        batch_process_time = 0
        try:
            frame_num = 0
            frames_batch = []
            frames_metadata_batch = []

            while True:
                read_start = time.time()
                ret, frame = cap.read()
                video_read_time += time.time() - read_start

                if not ret:
                    # Process remaining frames in batch
                    if frames_batch:
                        batch_start = time.time()
                        self._process_and_save_batch(
                            frames_batch, frames_metadata_batch,
                            frames_out, fps
                        )
                        batch_process_time += time.time() - batch_start
                    break

                # Collect frames for batch processing
                if frame_num % self.config.FRAME_INTERVAL == 0:
                    timestamp = str(timedelta(seconds=int(frame_num / fps)))

                    frame_path = None
                    if frames_out is not None:
                        frame_filename = f"frame_{frame_num:06d}.jpg"
                        frame_path = frames_out / frame_filename

                    # Add to batch
                    frames_batch.append(frame.copy())
                    frames_metadata_batch.append({
                        'frame_num': frame_num,
                        'timestamp': timestamp,
                        'frame_path': frame_path,
                    })

                    # Process batch when full
                    if len(frames_batch) >= FRAME_BATCH_SIZE:
                        batch_start = time.time()
                        self._process_and_save_batch(
                            frames_batch, frames_metadata_batch,
                            frames_out, fps
                        )
                        batch_process_time += time.time() - batch_start
                        frames_batch = []
                        frames_metadata_batch = []

                frame_num += 1

        finally:
            cap.release()
            total_time = time.time() - total_start
            print(f"📊 VIDEO PROCESSING BREAKDOWN:")
            print(f"   ⏱️  Total time: {total_time:.3f}s")
            print(f"   📹 Video reading: {video_read_time:.3f}s ({video_read_time/total_time*100:.1f}%)")
            print(f"   🔄 Batch processing: {batch_process_time:.3f}s ({batch_process_time/total_time*100:.1f}%)")
            print(f"   ⚙️  Other overhead: {total_time - video_read_time - batch_process_time:.3f}s ({(total_time - video_read_time - batch_process_time)/total_time*100:.1f}%)")
            print(f"   📊 Total frames read: {frame_num}, Frames processed: {len(self.frames_json)}")

        # Save results
        if self.config.WRITE_JSON:
            # Only save predictions.json if using default config output directory
            # If custom output_dir is provided, let the calling code handle output
            if output_dir is None:
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

    def _process_and_save_batch(self, frames_batch: List[np.ndarray],
                           metadata_batch: List[Dict],
                           frames_out: Optional[Path],
                           fps: float):
        """
        Process a batch of frames and save results.
        """
        # Process all frames in batch
        batch_results = self.analyze_frame_batch(frames_batch)

        # Save results for each frame
        for i, (frame, metadata, faces) in enumerate(zip(frames_batch, metadata_batch, batch_results)):
            frame_num = metadata['frame_num']
            timestamp = metadata['timestamp']
            frame_path = metadata['frame_path']
            num_faces = len(faces)

            # Save frame if needed
            should_save = self.config.SAVE_FRAMES and (
                num_faces > 0 if self.config.SAVE_ONLY_DETECTIONS else True
            )

            if should_save and frame_path is not None:
                to_save = self._draw_overlays(frame, faces) if num_faces > 0 else frame
                cv2.imwrite(str(frame_path), to_save)

            # Build frame record
            frame_record = {
                "frame_num": frame_num,
                "timestamp": timestamp,
                "frame_path": str(frame_path) if should_save and frame_path is not None else "",
                "num_faces": num_faces,
                "faces": faces,
                "error": "",
            }

            if self.config.WRITE_JSON:
                self.frames_json.append(frame_record)

            self.processed_frames += 1

        # Progress logging
        if self.processed_frames % 50 == 0:
            print(f"Processed {self.processed_frames} sampled frames (batch size: {len(frames_batch)})")

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


