#!/usr/bin/env python3
"""
scoreboard_detection_cosmos_reasonb_vllm.py

Scoreboard presence detector using on-VM Cosmos-Reason1 (via vLLM).
- Samples every Nth frame from the input video (ffmpeg).
- For each sampled frame, sends the full-frame vision tokens + JSON-only prompt
  to the Cosmos-Reason1 model via vLLM.
- Expects the model to RETURN EXACTLY ONE JSON OBJECT per prompt:
    {"detected": bool, "type": str, "text": str, "confidence": float, "notes": (str|null)}
- Aggregates per-frame detections and marks video as containing a scoreboard
  if presence_ratio >= presence_thresh.

IMPORTANT:
- This script requires on-VM packages: vllm, transformers (AutoProcessor), and qwen_vl_utils.
- It performs no mock fallback; if these packages are missing, it raises an informative error.

Usage example:
    python scoreboard_detection_cosmos_reasonb_vllm.py \
      --video /path/to/video.mp4 \
      --out_dir ./results \
      --frame_interval 30 \
      --presence_thresh 0.6 \
      --use_cosmos \
      --prompt_path /data/deepseek-ocr/scoreboard_detection_cosmos_prompt.yaml
"""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
import re
import sys
import yaml

# ---------------------------------------------------------------------
# Required on-VM imports (fail-fast if missing)
# ---------------------------------------------------------------------
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
except Exception as e:
    print("ERROR: Required on-VM packages are not available (vllm, transformers.AutoProcessor, qwen_vl_utils).")
    print("Install or activate the environment that has those packages and re-run.")
    print("Underlying import error:", repr(e))
    raise SystemExit(1)

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULTS = {
    "FRAME_INTERVAL": 30,
    "TMP_BASE": "/tmp/scoreboard_cosmos",
    "PRESENCE_THRESH": 0.6,
    "COSMOS_MODEL": "nvidia/Cosmos-Reason1-7B",
    "TEMPERATURE": 0.0,
    "MAX_TOKENS": 256,
    "GPU_MEMORY_FRACTION": 0.25,
    "MAX_MODEL_LEN": 6144,
}

# ---------------------------------------------------------------------
# Small parser (regex fallback if model returns non-JSON) — kept for logging,
# but by default we enforce JSON-only output and will treat non-JSON as an error.
# ---------------------------------------------------------------------
REGEXES = [
    ("HH:MM:SS", re.compile(r'\b([01]?\d|2[0-3]):[0-5]\d:[0-5]\d\b')),
    ("HH:MM", re.compile(r'\b([01]?\d|2[0-3]):[0-5]\d\b')),
    ("MM:SS", re.compile(r'\b([0-5]?\d):[0-5]\d\b')),
    ("CRICKET_RUNS_WKTS", re.compile(r'\b\d{1,3}/\d{1,2}\b')),
    ("SCORE_DASH", re.compile(r'\b\d{1,3}\s*[-:]\s*\d{1,3}\b')),
]
class SimpleParser:
    def find_any(self, text: str) -> Optional[Tuple[str,str]]:
        if not text:
            return None
        for name, cre in REGEXES:
            m = cre.search(text)
            if m:
                return name, m.group(0)
        return None

# ---------------------------------------------------------------------
# Frame extractor
# ---------------------------------------------------------------------
class FrameExtractor:
    def __init__(self, video_path: str, tmp_base: str = DEFAULTS["TMP_BASE"]):
        self.video_path = Path(video_path)
        self.tmp_base = Path(tmp_base)

    def extract_every_nth(self, interval: int, tmp_dir: Optional[Path] = None) -> Tuple[List[Path], Dict[str,Any]]:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        tmp_dir = tmp_dir or Path(tempfile.mkdtemp(prefix="frames_", dir=str(self.tmp_base)))
        tmp_dir.mkdir(parents=True, exist_ok=True)
        pattern = tmp_dir / "frame_%06d.jpg"
        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", str(self.video_path),
            "-vf", f"select=not(mod(n\\,{interval}))",
            "-vsync", "0",
            str(pattern)
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        frames = sorted(tmp_dir.glob("frame_*.jpg"))
        probe = {"frame_interval": interval}
        return frames, probe

# ---------------------------------------------------------------------
# Cosmos Reason1 client using vLLM + qwen_vl_utils
# ---------------------------------------------------------------------
class CosmosReasonClient:
    """
    Client wrapper that runs an on-VM vLLM instance (Cosmos-Reason1) using the
    vllm.LLM object and qwen_vl_utils.process_vision_info to convert images to
    model-correct vision tokens.

    The prompt is loaded from the YAML file provided via --prompt_path and is
    expected to instruct the model to RETURN ONLY VALID JSON that matches
    the schema:
      {"detected": bool, "type": str, "text": str, "confidence": float, "notes": (str|null)}
    """
    def __init__(self, model_name: str = DEFAULTS["COSMOS_MODEL"], prompt_path: Optional[str] = None,
                 device_gpu_fraction: float = DEFAULTS["GPU_MEMORY_FRACTION"],
                 max_model_len: int = DEFAULTS["MAX_MODEL_LEN"],
                 temperature: float = DEFAULTS["TEMPERATURE"],
                 max_tokens: int = DEFAULTS["MAX_TOKENS"]):
        self.model_name = model_name
        self.prompt_path = Path(prompt_path) if prompt_path else None
        self.temperature = temperature
        self.max_tokens = max_tokens

        # instantiate vLLM LLM with conservative VM-friendly settings (from your snippet)
        self.llm = LLM(
            model=self.model_name,
            limit_mm_per_prompt={"image": 0, "video": 1},
            enforce_eager=True,
            gpu_memory_utilization=device_gpu_fraction,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_num_batched_tokens=3072,
            max_num_seqs=16,
            disable_log_stats=True,
            swap_space=0,
        )

        # optional processor (may not be necessary in all builds)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
        except Exception:
            self.processor = None

        # load prompt YAML if provided, else use a safe default skeleton
        if self.prompt_path:
            if not self.prompt_path.exists():
                raise FileNotFoundError(f"Prompt YAML not found: {self.prompt_path}")
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.prompt_config = yaml.safe_load(f)
        else:
            # Hard-coded skeleton if user omitted prompt file (still JSON-only)
            self.prompt_config = {
                "system": "You are an expert vision reasoning assistant for sports broadcast frames.",
                "user": (
                    "Look at the provided image (a sports broadcast frame).\n"
                    "1) Answer if a scoreboard/score overlay or timer is visible.\n"
                    "2) If yes, extract the contents (scores, time, overs, team abbreviations) into JSON.\n"
                    "Return ONLY valid JSON with keys: {\"detected\": bool, \"type\": str, \"text\": str, \"confidence\": float, \"notes\": null}."
                )
            }

    def _build_messages(self, extra: Optional[str] = None) -> List[Dict[str,Any]]:
        sys_msg = {"role": "system", "content": self.prompt_config.get("system", "")}
        user_text = self.prompt_config.get("user", "")
        if extra:
            user_text = user_text + "\n" + extra
        user_msg = {"role": "user", "content": user_text}
        return [sys_msg, user_msg]

    def infer_on_image(self, image_bgr: np.ndarray) -> Dict[str,Any]:
        """
        Run a single inference for an image. Expects the model to return a single
        JSON object — if parsing fails we raise an error (no fallback).
        """
        # build vision info via qwen_vl_utils helper
        vision_info = process_vision_info(image_bgr)

        # messages / prompt
        messages = self._build_messages()

        # sampling params
        sampling = SamplingParams(temperature=self.temperature, top_p=0.95, max_tokens=self.max_tokens, top_k=0)

        # prepare inputs — many vLLM wrappers accept a dict { "messages": ..., "vision": ... }
        gen_inputs = {"messages": messages, "vision": vision_info}

        try:
            outputs = self.llm.generate(gen_inputs, sampling_params=sampling)
        except Exception as e:
            raise RuntimeError(f"vLLM generation failed: {e}")

        # Extract textual output robustly across vLLM versions
        text_out = None
        try:
            for out in outputs:
                # common shapes:
                # - out.text
                # - out.generations[0].text
                # - out.outputs[0].text or out.choices
                if hasattr(out, "text") and out.text:
                    text_out = out.text
                elif hasattr(out, "generations") and len(out.generations) > 0:
                    cand = out.generations[0]
                    text_out = getattr(cand, "text", None)
                else:
                    # last-resort: try common dict-like patterns
                    cand = getattr(out, "outputs", None) or getattr(out, "choices", None)
                    if isinstance(cand, (list, tuple)) and cand:
                        first = cand[0]
                        if isinstance(first, dict):
                            text_out = first.get("text") or first.get("message") or first.get("content")
                        else:
                            text_out = getattr(first, "text", None)
                if text_out:
                    break
        except Exception as e:
            raise RuntimeError(f"Error while extracting generation text: {e}")

        if not text_out:
            raise RuntimeError("vLLM returned no textual output (empty).")

        text_out = text_out.strip()

        # Expect strict JSON-only; parse
        try:
            parsed = json.loads(text_out)
        except Exception as e:
            # fail-fast and give raw output for debugging
            raise RuntimeError(f"Failed to parse model output as JSON. Raw output:\n{text_out!r}\nParse error: {e}")

        # sanitize and return
        return {
            "detected": bool(parsed.get("detected", False)),
            "type": str(parsed.get("type", "scoreboard")),
            "text": str(parsed.get("text", "")),
            "confidence": float(parsed.get("confidence", 0.0)),
            "notes": parsed.get("notes", None)
        }

# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
class CosmosScoreboardDetector:
    def __init__(self, video_path: str, out_dir: str, frame_interval: int = DEFAULTS["FRAME_INTERVAL"],
                 presence_thresh: float = DEFAULTS["PRESENCE_THRESH"], prompt_path: Optional[str] = None,
                 model_name: str = DEFAULTS["COSMOS_MODEL"], tmp_base: str = DEFAULTS["TMP_BASE"],
                 temperature: float = DEFAULTS["TEMPERATURE"], max_tokens: int = DEFAULTS["MAX_TOKENS"]):
        self.video_path = Path(video_path)
        self.out_dir = Path(out_dir)
        self.frame_interval = frame_interval
        self.presence_thresh = presence_thresh
        self.prompt_path = Path(prompt_path) if prompt_path else None
        self.tmp_base = tmp_base

        # client: use strict CosmosReasonClient (no mock)
        self.client = CosmosReasonClient(model_name=model_name, prompt_path=str(self.prompt_path) if self.prompt_path else None,
                                         device_gpu_fraction=DEFAULTS["GPU_MEMORY_FRACTION"],
                                         max_model_len=DEFAULTS["MAX_MODEL_LEN"],
                                         temperature=temperature, max_tokens=max_tokens)
        self.extractor = FrameExtractor(str(self.video_path), tmp_base=self.tmp_base)
        self.parser = SimpleParser()

    def prepare(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        Path(self.tmp_base).mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str,Any]:
        self.prepare()
        tmp_dir = Path(tempfile.mkdtemp(prefix="frames_", dir=str(self.tmp_base)))
        frames, probe = self.extractor.extract_every_nth(self.frame_interval, tmp_dir=tmp_dir)
        if not frames:
            raise RuntimeError("No frames extracted.")
        detections = []
        detected_count = 0
        total = 0

        for idx, fp in enumerate(tqdm(frames, desc="Frames", unit="frame")):
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            total += 1
            # call Cosmos Reason1 via vLLM (expects JSON-only)
            try:
                res = self.client.infer_on_image(img)
            except Exception as e:
                # fail fast — we are using direct Cosmos calls only
                raise RuntimeError(f"Cosmos inference failed on frame {fp}: {e}")

            # Optional parser match for logging (but we trust model JSON)
            parsed_token = self.parser.find_any(res.get("text", "")) if res.get("text") else None

            final_detected = bool(res.get("detected", False))
            if final_detected:
                detected_count += 1

            detections.append({
                "sample_index": idx,
                "frame_path": str(fp),
                "detected": bool(final_detected),
                "reason_type": res.get("type"),
                "text": res.get("text"),
                "confidence": float(res.get("confidence", 0.0)),
                "parser_match": parsed_token[1] if parsed_token else None,
                "notes": res.get("notes", None)
            })

        presence_ratio = detected_count / total if total else 0.0
        scoreboard_present = presence_ratio >= self.presence_thresh

        out = {
            "video": str(self.video_path),
            "num_samples": total,
            "detected_samples": detected_count,
            "presence_ratio": presence_ratio,
            "presence_thresh": self.presence_thresh,
            "scoreboard_present": bool(scoreboard_present),
            "detections": detections,
            "probe": probe,
        }

        out_path = self.out_dir / f"scoreboard_presence_{self.video_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        # cleanup extracted frames
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

        return {"result_json": str(out_path), "summary": out}

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Cosmos-Reason1 (vLLM) scoreboard presence detector (frame-sampling)")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--out_dir", required=True, help="Directory to write output JSON")
    p.add_argument("--frame_interval", type=int, default=DEFAULTS["FRAME_INTERVAL"], help="Extract every Nth frame")
    p.add_argument("--presence_thresh", type=float, default=DEFAULTS["PRESENCE_THRESH"], help="Fraction of frames that must contain scoreboard to flag video")
    p.add_argument("--prompt_path", default="/data/deepseek-ocr/scoreboard_detection_cosmos_prompt.yaml", help="Path to YAML prompt for Cosmos Reason1 (JSON-only prompt recommended)")
    p.add_argument("--model_name", default=DEFAULTS["COSMOS_MODEL"], help="Cosmos model name (vLLM)")
    p.add_argument("--tmp_base", default=DEFAULTS["TMP_BASE"], help="Base temp dir for extracted frames")
    p.add_argument("--temperature", type=float, default=DEFAULTS["TEMPERATURE"], help="Generation temperature (use 0.0 for deterministic)")
    p.add_argument("--max_tokens", type=int, default=DEFAULTS["MAX_TOKENS"], help="Max tokens for generation")
    return p.parse_args()

def main():
    args = parse_args()
    detector = CosmosScoreboardDetector(
        video_path=args.video,
        out_dir=args.out_dir,
        frame_interval=args.frame_interval,
        presence_thresh=args.presence_thresh,
        prompt_path=args.prompt_path,
        model_name=args.model_name,
        tmp_base=args.tmp_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    start = time.time()
    res = detector.run()
    elapsed = time.time() - start
    print(f"Done. Results: {res['result_json']} (elapsed {elapsed:.1f}s)")

if __name__ == "__main__":
    main()
