# ray_jobs/nsfw_det.py
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

import ray
import cv2
import numpy as np

# Optional torch import for nice GPU logging (NudeNet will use it internally if installed with CUDA)
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from nudenet import NudeDetector  # NudeNet manages model + inference


# --------- Logging ----------
logger = logging.getLogger("nsfw_detector")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _ensure_cv2_fast():
    # Keep OpenCV lean; avoid oversubscribing CPU threads
    try:
        cv2.setNumThreads(max(1, min(4, os.cpu_count() or 4)))
    except Exception:
        pass


@ray.remote(num_gpus=1)
class NSFWDetectorWorker:
    """
    Ray actor that loads NudeNet once (on GPU if available) and processes video chunks.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        batch_size: int = 16,
        model_path: str = None,
        inference_resolution: int = 320,
        use_classification_fallback: bool = True,
    ):
        """
        Args:
            confidence_threshold: min confidence to count a frame as NSFW.
            batch_size: frames per batch for detect_batch().
            model_path: optional ONNX path for NudeNet 640m model (see PyPI docs).
            inference_resolution: 320 (default) or 640 when using 640m model.
            use_classification_fallback: if True, treat classification dict {'unsafe': p}
                                         as NSFW when p >= threshold.
        """
        _ensure_cv2_fast()

        # Log GPU info
        if TORCH_OK:
            gpu = torch.cuda.is_available()
            num = torch.cuda.device_count() if gpu else 0
            name = torch.cuda.get_device_name(0) if gpu else "CPU"
            logger.info(f"PyTorch CUDA available: {gpu} | GPUs: {num} | Using: {name}")
        else:
            logger.info("PyTorch not importable; NudeNet will still run (CPU fallback).")

        # Initialize NudeNet (auto-downloads default 320n; or use provided 640m path)
        init_kwargs = {}
        if model_path:
            init_kwargs["model_path"] = model_path
            logger.info(f"Using NudeNet model from: {model_path} @ {inference_resolution}")

        self.detector = NudeDetector(**init_kwargs)
        self.inference_resolution = inference_resolution
        self.confidence_threshold = float(confidence_threshold)
        self.batch_size = max(1, int(batch_size))
        self.use_classification_fallback = use_classification_fallback

        logger.info("✅ NudeNet model loaded")
        # Note: NudeNet 3.4+ returns detections like:
        # [{'class': 'FEMALE_BREAST_EXPOSED', 'score': 0.87, 'box': [x, y, w, h]}, ...]
        # Or classification dict: {'safe': 0.91, 'unsafe': 0.09}. :contentReference[oaicite:1]{index=1}

    # ---------- Inference helpers ----------

    def _predict_batch(self, frames_bgr: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Run batched detection. Returns per-frame summaries:
          {'is_nsfw': bool, 'label': str or None, 'score': float}
        """
        if not frames_bgr:
            return []

        # NudeNet accepts OpenCV images; convert BGR->RGB improves consistency
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

        try:
            raw = self.detector.detect_batch(frames_rgb, inference_resolution=self.inference_resolution)
            # detect_batch returns List[List[Det]]  (one list per frame) in detection mode.
            # Some builds can return classification dicts; handle both.
        except TypeError:
            # Older signatures may not accept inference_resolution; try without.
            raw = self.detector.detect_batch(frames_rgb)

        results = []
        for item in raw:
            if isinstance(item, list):
                # Detection list -> choose highest-score detection as representative
                if len(item) == 0:
                    results.append({"is_nsfw": False, "label": None, "score": 0.0})
                else:
                    best = max(item, key=lambda x: x.get("score", 0.0))
                    label = best.get("class") or best.get("label")  # robust to older keys
                    score = float(best.get("score", 0.0))
                    results.append({"is_nsfw": score >= self.confidence_threshold, "label": label, "score": score})
            elif isinstance(item, dict) and self.use_classification_fallback:
                # Classification dict: {'safe': p1, 'unsafe': p2}
                unsafe = float(item.get("unsafe", 0.0))
                results.append({"is_nsfw": unsafe >= self.confidence_threshold, "label": "unsafe", "score": unsafe})
            else:
                results.append({"is_nsfw": False, "label": None, "score": 0.0})

        return results

    # ---------- Public method ----------

    def process_video_chunk(self, video_path: str, chunk_offset_seconds: float = 0.0):
        """
        Process a single (possibly whole) video: sample 1 FPS, run batched GPU detection,
        and return timestamps + merged flagged segments.
        """
        t0 = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (total_frames / fps) if fps > 0 else 0.0
        # Sample 1 frame per second:
        step = max(1, int(round(fps))) if fps > 0 else 30

        logger.info(f"Processing {Path(video_path).name}: fps={fps:.2f}, frames={total_frames}, dur={duration:.2f}s")

        nsfw_detections = []
        batch_frames, batch_indices = [], []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                batch_frames.append(frame)
                batch_indices.append(frame_idx)

                if len(batch_frames) >= self.batch_size:
                    nsfw_detections.extend(
                        self._consume_batch(batch_frames, batch_indices, fps, chunk_offset_seconds, video_path)
                    )
                    batch_frames, batch_indices = [], []

            frame_idx += 1

        # flush remainder
        if batch_frames:
            nsfw_detections.extend(
                self._consume_batch(batch_frames, batch_indices, fps, chunk_offset_seconds, video_path)
            )

        cap.release()

        flagged_segments = self._convert_detections_to_segments(nsfw_detections)
        elapsed = round(time.time() - t0, 2)

        return {
            "chunk_file": video_path,
            "chunk_offset_seconds": chunk_offset_seconds,
            "chunk_duration_seconds": duration,
            "processing_time_seconds": elapsed,
            "total_nsfw_detections": len(nsfw_detections),
            "nsfw_timestamps": nsfw_detections,
            "flagged_segments": flagged_segments,
            "success": True,
        }

    def _consume_batch(self, frames, indices, fps, offset, src_path):
        out = []
        preds = self._predict_batch(frames)
        for idx, pred in zip(indices, preds):
            if fps > 0:
                ts = (idx / fps) + offset
            else:
                ts = float(idx) + offset

            if pred["is_nsfw"]:
                out.append(
                    {
                        "timestamp": round(ts, 2),
                        "frame": int(idx),
                        "prediction": pred["label"],
                        "confidence": round(float(pred["score"]), 4),
                        "chunk_file": src_path,
                    }
                )
        return out

    # ---------- Segmenting ----------

    def _convert_detections_to_segments(self, detections: List[Dict[str, Any]], max_gap_seconds: float = 2.0):
        if not detections:
            return []

        dets = sorted(detections, key=lambda x: x["timestamp"])
        segments = []
        cur = None

        for d in dets:
            t = d["timestamp"]
            c = d["confidence"]
            if cur is None:
                cur = {
                    "start_time": t,
                    "end_time": t,
                    "detections": [d],
                    "max_confidence": c,
                    "avg_confidence": c,
                }
            else:
                gap = t - cur["end_time"]
                if gap <= max_gap_seconds:
                    cur["end_time"] = t
                    cur["detections"].append(d)
                    cur["max_confidence"] = max(cur["max_confidence"], c)
                    confs = [x["confidence"] for x in cur["detections"]]
                    cur["avg_confidence"] = sum(confs) / len(confs)
                else:
                    segments.append(self._finalize_segment(cur))
                    cur = {
                        "start_time": t,
                        "end_time": t,
                        "detections": [d],
                        "max_confidence": c,
                        "avg_confidence": c,
                    }

        if cur:
            segments.append(self._finalize_segment(cur))
        return segments

    def _finalize_segment(self, s):
        n = len(s["detections"])
        avg_c = s["avg_confidence"]
        max_c = s["max_confidence"]
        dur = s["end_time"] - s["start_time"]

        if max_c > 0.8 or n > 3:
            priority = "high"
        elif max_c > 0.6 or n > 1:
            priority = "medium"
        else:
            priority = "low"

        return {
            "start_time": s["start_time"],
            "end_time": s["end_time"],
            "task_type": "nsfw_detection",
            "confidence": avg_c,
            "flag_type": "nsfw_content",
            "priority": priority,
            "description": f"NSFW content detected ({n} frames, {dur:.1f}s)",
            "metadata": {
                "detection_count": n,
                "avg_confidence": round(avg_c, 4),
                "max_confidence": round(max_c, 4),
                "duration": round(dur, 2),
            },
        }


@ray.remote
def process_video_chunks_for_nsfw(
    chunk_paths: List[str],
    confidence_threshold: float = 0.5,
    chunk_duration_sec: int = 60,
    batch_size: int = 16,
    model_path: str = None,
    inference_resolution: int = 320,
):
    """
    Distribute chunks across GPU workers; returns merged detections + segments.
    """
    try:
        available_gpus = int(ray.available_resources().get("GPU", 1))
        num_workers = max(1, min(available_gpus, len(chunk_paths)))
        workers = [
            NSFWDetectorWorker.remote(
                confidence_threshold=confidence_threshold,
                batch_size=batch_size,
                model_path=model_path,
                inference_resolution=inference_resolution,
            )
            for _ in range(num_workers)
        ]
        logger.info(f"Created {num_workers} NSFW detector workers (GPU-aware)")

        futures = []
        for i, chunk_path in enumerate(chunk_paths):
            worker = workers[i % num_workers]
            chunk_offset = i * chunk_duration_sec
            futures.append(worker.process_video_chunk.remote(chunk_path, chunk_offset_seconds=chunk_offset))

        results = ray.get(futures)

        all_dets, all_segments, total_time = [], [], 0.0
        for r in results:
            if r.get("success"):
                all_dets.extend(r["nsfw_timestamps"])
                all_segments.extend(r.get("flagged_segments", []))
                total_time += float(r["processing_time_seconds"])

        all_dets.sort(key=lambda x: x["timestamp"])
        all_segments.sort(key=lambda x: x["start_time"])

        return {
            "total_nsfw_detections": len(all_dets),
            "total_processing_time_seconds": round(total_time, 2),
            "nsfw_timestamps": all_dets,
            "flagged_segments": all_segments,
            "chunks_processed": len(chunk_paths),
            "success": True,
        }
    except Exception as e:
        logger.exception("Error in process_video_chunks_for_nsfw")
        return {"error": str(e), "success": False}


if __name__ == "__main__":
    # Hardcoded single input path for your test
    test_video = "/home/nvcoe_admin/code/oslo/whole_pipeline_testing/kamwai_chan_input_videos/vaccum_floor_1GB.mp4"
    if not os.path.exists(test_video):
        print(f"❌ Video not found: {test_video}")
        raise SystemExit(1)

    # Tuning knobs (GPU-optimal defaults)
    CONF_THRESH = 0.5
    BATCH = 16                   # Increase (e.g., 32) if your GPU has headroom
    USE_640M = False             # Set True if you downloaded 640m.onnx
    MODEL_PATH = None
    INFER_RES = 320

    # If you have 640m.onnx locally, point to it and use 640 resolution:
    # (Per NudeNet docs: pass model_path + set inference_resolution=640) :contentReference[oaicite:2]{index=2}
    if USE_640M:
        MODEL_PATH = "/path/to/640m.onnx"
        INFER_RES = 640

    try:
        ray.init()
        print("🚀 Testing NSFW detection with NudeNet (GPU-optimized)...")
        result = ray.get(
            process_video_chunks_for_nsfw.remote(
                [test_video],
                confidence_threshold=CONF_THRESH,
                chunk_duration_sec=60,
                batch_size=BATCH,
                model_path=MODEL_PATH,
                inference_resolution=INFER_RES,
            )
        )
        if result.get("success"):
            print("✅ NSFW detection test successful!")
            print(f"   - Total detections: {result['total_nsfw_detections']}")
            print(f"   - Flagged segments: {len(result.get('flagged_segments', []))}")
            print(f"   - Processing time: {result['total_processing_time_seconds']}s")
            for i, seg in enumerate(result.get("flagged_segments", [])[:5], 1):
                print(f"   - Segment {i}: {seg['start_time']:.1f}-{seg['end_time']:.1f}s ({seg['priority']}) "
                      f"[avg {seg['metadata']['avg_confidence']:.2f}]")
        else:
            print(f"❌ NSFW detection failed: {result.get('error', 'Unknown error')}")
    finally:
        ray.shutdown()
