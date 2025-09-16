"""Consolidation pipeline for Label Studio video shard annotations.

This script fetches reviewed tasks from Label Studio, consolidates per-shard
JSONs into per-video master JSONs, and saves both intermediate and final
results into Azure Blob Storage.
"""

import os
import json
import copy
import requests
import time
import yaml
from datetime import timedelta,datetime, timezone
from azure.storage.blob import BlobServiceClient
from typing import Dict, Any, List, Optional, Callable, Tuple
import re
from stitch_mux import RenderSettings,build_and_run_ffmpeg_for_group
import shutil
import requests
from pathlib import Path
import ray
import copy
import logging
from azure.storage.blob import BlobServiceClient, ContentSettings
import mimetypes
from dotenv import load_dotenv
import signal
import sys
import threading
import zoneinfo
from typing import Any, Dict, Iterable, List, Set, Union
from collections import defaultdict
import math

_NEGATIVE_KEYWORDS = {"no", "none", "no pii", "no minors", "no nsfw", "no sensitive", "unknown"}

# mapping signals to keywords to match in from_name/to_name or text
_SIGNAL_KEYWORDS = {
    "pii": ["pii", "personal", "ssn", "full name", "address", "pii_audio", "pii_video"],
    "minors": ["minor", "minors", "child", "juvenile"],
    "nsfw": ["nsfw", "nudity", "adult"],
    "domain": ["domain", "domain_issue", "domain_prediction", "video_domain", "audio_domain"],
    "signal": ["signal issue" "signal_issue", "signal"],
    "people": ["person", "people", "face", "participant"],
    "lighting": ["light", "lighting","low light","low-light", "low_light", "underexposed"],
    "sensitive_audio": ["sensitive_audio", "sensitive_audio_comment", "sensitive_audio", "sensitive"],
    "scene": ["scene", "scenes", "shot"]
}

_FROMNAME_RE = re.compile(r"(?P<prefix>.+?)_(?P<pos>start|end)_(?P<unit>minute|second)$", re.IGNORECASE)

# Load values from .env file into environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

API_TOKEN = os.getenv("LABEL_STUDIO_API_TOKEN")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

print("Token loaded:", API_TOKEN is not None)

# ==================================
# CONFIG
# ==================================
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "post_processing_config.yaml"
)


with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

BASE_URL = config["label_studio"]["server_url"]
# API_TOKEN = config["label_studio"]["api_token"]
PROJECT_ID = int(config["label_studio"]["project_id"])
VIEW_ID = int(config["label_studio"]["view_id"])
PAGE_SIZE = int(config["label_studio"]["page_size"])

# AZURE_STORAGE_CONNECTION_STRING = config["azure_storage"]["AZURE_STORAGE_CONNECTION_STRING"]
MASTER_JSONS_CONTAINER = config["azure_storage"]["intermediate_master_json_container"]
FINAL_MASTER_JSONS_CONTAINER = config["azure_storage"]["final_master_json_container"]

POLL_INTERVAL_SEC = int(config["polling"]["interval_seconds"])
SUMMARY_CONTAINER="summary-jsons"
CONTAINER_CONTAINING_VIDEOS="instavideo"
CONTAINER_CONTAINING_AUDIOS="instavideo"


# ==================================
# Helpers
# ==================================
def _master_blob_name(video_identifier: str) -> str:
    """Return blob name for master JSON file."""
    return f"{video_identifier}.json"


def _final_blob_name(video_identifier: str) -> str:
    """Return blob name for final JSON file."""
    return f"{video_identifier}.json"


def _globalise_seconds(local_time: float, shard_id: int,
                       default_duration: int = 60) -> float:
    """Shift local time by shard offset to global timeline."""
    offset = (shard_id - 1) * default_duration
    return offset + float(local_time)

def _safe_get(d: Dict, path: List[str]):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def _is_negative_text(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    for neg in _NEGATIVE_KEYWORDS:
        if neg in t:
            return True
    return False

def _matches_any_keyword(name_or_text: str, keywords: Iterable[str]) -> bool:
    if not name_or_text:
        return False
    s = str(name_or_text).lower()
    for kw in keywords:
        if kw in s:
            return True
    return False

def compute_shard_offset(shard_index: int, shard_minutes: float = 3.0, overlap_minutes: float = 1.0) -> int:
    """
    Return offset seconds for given shard.
    - shard_index: integer index (0-based by default). If one_indexed=True, pass 1-based index.
    - shard_minutes: duration of each shard in minutes (default 3).
    - overlap_minutes: overlap between consecutive shards in minutes (default 1).
    Returns integer seconds (rounded down).
    """
    if shard_index < 1:
        raise ValueError("shard_index must be >= 1")
    

    shard_len_s = int(round(shard_minutes * 60))
    overlap_s = int(round(overlap_minutes * 60))
    step_s = shard_len_s - overlap_s
    return (shard_index -1)* step_s

Numeric = Union[int, float]

def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm (milliseconds precision)."""
    if seconds is None:
        return None
    seconds = max(0.0, float(seconds))  # clamp negatives
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{millis:03d}"

def _clamp_and_warn(value: float, context: str) -> float:
    """Clamp negative to 0 and log a warning."""
    if value is None:
        return value
    try:
        if value < 0:
            logger.warning(f"[GLOBALISE] {context}: value {value} < 0 after adding offset — clamped to 0")
            return 0.0
        return float(value)
    except Exception:
        return value

def _is_iso_datetime_string(s: Any) -> bool:
    """Detect ISO datetime strings to avoid shifting them."""
    return isinstance(s, str) and ("T" in s and "-" in s and ":" in s)

def _format_count_summary(label: str, total: int, adjusted: int) -> str:
    """Helper to format logging summary lines like before."""
    return f"[GLOBALISE] {label}: total={total}, adjusted={adjusted}"


# 🔹 Create log file path in current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(current_dir, "globalise.log")

# 🔹 Configure logger
logger = logging.getLogger("globalise_logger")
logger.setLevel(logging.INFO)

# Prevent adding duplicate handlers if this file is imported multiple times
if not logger.handlers:
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Format for both handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)









def _format_count_summary(name: str, total: int, adjusted: int) -> str:
    return f"[GLOBALISE] {name}: processed={total}, adjusted_negative={adjusted}"





# ==================================
# Label Studio Client
# ==================================
class LabelStudioClient:
    """Client to fetch tasks and details from Label Studio API."""

    def __init__(self, base_url: str, api_token: str, page_size: int = 100) -> None:
        self.base_url = base_url
        self.headers = {"Authorization": f"Token {api_token}", "Content-Type": "application/json"}
        self.page_size = page_size

    def fetch_task_ids(self, project_id: int, view_id: int) -> List[int]:
        """Fetch all task IDs for a given project and view."""
        task_ids = []
        page = 1
        while True:
            url = (f"{self.base_url}/tasks?project={project_id}&view={view_id}"
                   f"&page={page}&page_size={self.page_size}")
            resp = requests.get(url, headers=self.headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            tasks = data.get("tasks", [])
            if not tasks:
                break
            task_ids.extend([t["id"] for t in tasks])
            if len(task_ids) >= data.get("total", 0):
                break
            page += 1
        return task_ids

    def fetch_task_detail(self, task_id: int) -> Dict[str, Any]:
        """Fetch full detail for a single Label Studio task."""
        url = f"{self.base_url}/tasks/{task_id}"
        resp = requests.get(url, headers=self.headers, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def fetch_all_tasks(self, project_id: int, view_id: int) -> List[Dict[str, Any]]:
        """Fetch all task JSONs for a project and view."""
        task_ids = self.fetch_task_ids(project_id, view_id)
        print(f"Found {len(task_ids)} tasks in project {project_id}, view {view_id}")
        all_tasks: List[Dict[str, Any]] = []
        for i, tid in enumerate(task_ids, start=1):
            try:
                detail = self.fetch_task_detail(tid)
                all_tasks.append(detail)
                print(f"[{i}/{len(task_ids)}] fetched task {tid}")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f"⚠️ Failed to fetch task {tid}: {exc}")
        return all_tasks

# ==================================
# Consolidation Pipeline
# ==================================
@ray.remote
class ConsolidationPipeline:
    """Pipeline to consolidate Label Studio shard JSONs into master and final JSONs stored in Azure."""
    def __init__(self, connection_string: str, masters_container: str, final_container: str, do_stitching: bool = True):

        """
        Initialize Azure Blob Storage clients for intermediate and final JSON containers.

        Args:
            connection_string (str): Azure storage connection string.
            masters_container (str): Name of the intermediate masters container.
            final_container (str): Name of the final masters container.
        """
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.masters_client = self.blob_service.get_container_client(masters_container)
        self.final_client = self.blob_service.get_container_client(final_container)
        self.do_stitching = bool(do_stitching)
        for c in [self.masters_client, self.final_client]:
            try:
                c.create_container()
            except Exception:
                pass
        self._run_summary_lock = threading.Lock()
        self._current_run_summary = None   # dict or None
        self._current_run_filename = None

    # ------------------------------
    # Validation
    # ------------------------------
    def validate_shard_json(self, shard_json: Dict[str, Any]) -> bool:
        """
        Validate that a shard JSON contains required fields and correct types.

        Args:
            shard_json (dict): Shard JSON object.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            data = shard_json.get("data", {})
            preds = shard_json.get("predictions", [])
            anns = shard_json.get("annotations", [])
            required_data_keys = ["video_name", "shard_number", "total_shards"]
            for key in required_data_keys:
                if key not in data:
                    raise ValueError(f"Missing required key in data: {key}")
            if not isinstance(preds, list):
                raise ValueError("predictions must be a list")
            if not isinstance(anns, list):
                raise ValueError("annotations must be a list")
            return True
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return False

    # ------------------------------
    # Azure I/O
    # ------------------------------
    def load_master(self, video_id: str, total_shards: Optional[int] = None) -> Dict[str, Any]:
        """
        Load an existing master JSON from Azure, or create a new skeleton if none exists.

        Args:
            video_id (str): Unique video identifier.
            total_shards (int, optional): Total expected shards. Defaults to None.

        Returns:
            dict: Master JSON object.
        """
        blob_client = self.masters_client.get_blob_client(_master_blob_name(video_id))
        if blob_client.exists():
            try:
                blob_bytes = blob_client.download_blob().readall()
                return json.loads(blob_bytes.decode("utf-8"))
            except Exception as e:
                print(f"⚠️ Failed to load master: {e}")
        return {
            "original_video_id": video_id,
            "total_shards": total_shards or 0,
            "consolidated_shards": [],
            "status": "in_progress",
            "shards": {}
        }

    def save_master(self, master: Dict[str, Any]) -> str:
        """
        Save a master JSON back to Azure blob storage.

        Args:
            master (dict): Master JSON object.

        Returns:
            str: Blob URL of the saved master JSON.
        """
        try:
            blob_client = self.masters_client.get_blob_client(_master_blob_name(master["original_video_id"]))
            master["shards"] = {
                k: master["shards"][k]
                for k in sorted(master["shards"].keys(), key=lambda x: int(x.replace("shard_", "")))
            }
            data_bytes = json.dumps(master, indent=2).encode("utf-8")
            blob_client.upload_blob(data_bytes, overwrite=True)
            return blob_client.url
        except Exception as e:
            print(f"❌ Failed to save master: {e}")
            return ""

    def save_final_master(self, flat: Dict[str, Any]) -> str:
        """
        Save the fully consolidated final master JSON to Azure blob storage.

        Args:
            flat (dict): Final master JSON object.

        Returns:
            str: Blob URL of the saved final master JSON.
        """
        try:
            blob_client = self.final_client.get_blob_client(_final_blob_name(flat["original_video_id"]))
            data_bytes = json.dumps(flat, indent=2).encode("utf-8")
            blob_client.upload_blob(data_bytes, overwrite=True)
            return blob_client.url
        except Exception as e:
            print(f"❌ Failed to save final master: {e}")
            return ""


    def globalise_predictions(self,predictions: List[Dict[str, Any]], offset_seconds: float) -> Dict[str, Dict[str,int]]:
        """
        Globalise numeric time fields under predictions[].result[].value by adding offset_seconds.
        Mutates `predictions` in-place. Returns a summary dict of counters.
        """
        logger.info(f"[GLOBALISE] Starting globalisation with offset_seconds={offset_seconds}")

        numeric_time_keys = {
            "start", "end", "start_time", "end_time",
            "clap_timestamp", "timestamp_seconds", "timestamp"
        }
        array_time_keys = {"nsfw_timestamps"}
        ts_string_keys = {"start_ts", "end_ts"}

        counters = defaultdict(lambda: {"total": 0, "adjusted": 0})

        def _shift_numeric(key: str, value: Numeric, ctx_path: str) -> Numeric:
            counters[key]["total"] += 1
            try:
                orig = float(value)
            except Exception:
                return value
            new = orig + offset_seconds
            new = _clamp_and_warn(new, f"{ctx_path}.{key}")
            if abs(new - orig) > 1e-6:
                counters[key]["adjusted"] += 1
            return new

        def _process_dict(d: Dict[str, Any], path: str = ""):
            for k, v in list(d.items()):
                current_path = f"{path}.{k}" if path else k

                if k in array_time_keys and isinstance(v, list):
                    counters[k]["total"] += len(v)
                    changed = 0
                    new_list = []
                    for i, item in enumerate(v):
                        if isinstance(item, (int, float)):
                            orig = float(item)
                            new_item = orig + offset_seconds
                            new_item = _clamp_and_warn(new_item, f"{current_path}[{i}]")
                            new_list.append(new_item)
                            if abs(new_item - orig) > 1e-6:
                                changed += 1
                        else:
                            new_list.append(item)
                    d[k] = new_list
                    counters[k]["adjusted"] += changed
                    continue

                if k in numeric_time_keys:
                    if k == "timestamp" and _is_iso_datetime_string(v):
                        continue
                    if isinstance(v, (int, float)):
                        new_val = _shift_numeric(k, v, path or "predictions")
                        d[k] = new_val
                        if k in ("start", "start_time"):
                            d["start_ts"] = _format_time(new_val)
                        elif k in ("end", "end_time"):
                            d["end_ts"] = _format_time(new_val)
                        continue
                    else:
                        continue

                if k in ts_string_keys:
                    numeric_counterpart = "start" if k == "start_ts" else "end"
                    alt_counterpart = numeric_counterpart + "_time"
                    if numeric_counterpart in d and isinstance(d[numeric_counterpart], (int, float)):
                        d[k] = _format_time(d[numeric_counterpart])
                    elif alt_counterpart in d and isinstance(d[alt_counterpart], (int, float)):
                        d[k] = _format_time(d[alt_counterpart])
                    continue

                if isinstance(v, dict):
                    _process_dict(v, current_path)
                elif isinstance(v, list):
                    _process_list(v, current_path)

        def _process_list(lst: List[Any], path: str = ""):
            for idx, item in enumerate(lst):
                item_path = f"{path}[{idx}]"
                if isinstance(item, dict):
                    _process_dict(item, item_path)
                elif isinstance(item, list):
                    _process_list(item, item_path)

        for p_idx, pred in enumerate(predictions):
            base_path = f"predictions[{p_idx}]"
            results = pred.get("result", []) or []
            for r_idx, res in enumerate(results):
                value = res.get("value")
                if isinstance(value, dict):
                    _process_dict(value, f"{base_path}.result[{r_idx}].value")
                elif isinstance(value, list):
                    _process_list(value, f"{base_path}.result[{r_idx}].value")

        # Log summary
        for key, counts in counters.items():
            logger.info(_format_count_summary(key, counts["total"], counts["adjusted"]))

        logger.info("[GLOBALISE] Done.")
        return counters


    def _globalise_annotations(self,records: List[Dict[str, Any]], offset_seconds: float) -> List[Dict[str, Any]]:
        """
        Globalise annotations in-place-style but return a deep-copied updated list.

        Args:
        records: sample_json['annotations'] (list of record dicts; each record must contain "result": list)
        offset_seconds: seconds to add to the minute+second pair (and numeric label start/end)

        Returns:
        new_records: deep-copied list of records with updated times:
            - numeric value.start/value.end shifted (float)
            - taxonomy minute annotations updated to global minute string in taxonomy nested array
            - taxonomy second annotations updated to global second string in taxonomy nested array
        """
        def _parse_primitive_as_int(x: Any) -> Optional[int]:
            if x is None:
                return None
            val = x
            # unwrap nested single-element lists like [["06"]] -> "06"
            while isinstance(val, list) and len(val) > 0:
                val = val[0]
            if isinstance(val, (int, float)):
                try:
                    return int(val)
                except Exception:
                    return None
            if isinstance(val, str):
                s = val.strip()
                if s == "":
                    return None
                try:
                    return int(float(s))
                except Exception:
                    return None
            return None

        def _clamp_nonneg(v: Optional[float]) -> Optional[float]:
            if v is None:
                return None
            try:
                fv = float(v)
            except Exception:
                return None
            if fv < 0:
                logger.warning("[GLOBALISE] clamping negative time to 0: %s", fv)
                return 0.0
            return fv

        if not isinstance(records, list):
            raise TypeError("Expected `records` to be a list (sample_json['annotations']).")

        out_records = copy.deepcopy(records)

        stats = {
            "records": 0,
            "annotations_total": 0,
            "labels_seen": 0,
            "labels_shifted": 0,
            "taxonomy_groups": 0,
            "taxonomy_updated": 0,
        }

        for rec_idx, rec in enumerate(out_records):
            stats["records"] += 1
            if not isinstance(rec, dict):
                logger.warning("[GLOBALISE] skipping non-dict record at index %d", rec_idx)
                continue
            results = rec.get("result", [])
            if not isinstance(results, list):
                logger.warning("[GLOBALISE] record %d has no 'result' list; skipping", rec_idx)
                continue

            # 1) Shift numeric label start/end in each annotation's value (or top-level)
            for ann in results:
                stats["annotations_total"] += 1
                val_container = ann.get("value") if isinstance(ann.get("value"), dict) else ann
                for key in ("start", "end"):
                    if key in val_container and isinstance(val_container[key], (int, float)):
                        old = float(val_container[key])
                        stats["labels_seen"] += 1
                        new = _clamp_nonneg(old + float(offset_seconds))
                        if new is None:
                            continue
                        if abs(new - old) > 1e-12:
                            stats["labels_shifted"] += 1
                        val_container[key] = new
                if isinstance(ann.get("value"), dict):
                    ann["value"] = val_container

            # 2) Collect taxonomy minute/second annotations grouped by (prefix, pos)
            # groups[(prefix,pos)] = {"minutes": [(idx,ann),...], "seconds": [(idx,ann),...]}
            groups = {}

            for idx_ann, ann in enumerate(results):
                if not isinstance(ann, dict):
                    continue
                if ann.get("type") != "taxonomy":
                    continue
                from_name = (ann.get("from_name") or "")
                m = _FROMNAME_RE.match(from_name)
                if not m:
                    continue
                prefix = m.group("prefix")
                pos = m.group("pos").lower()
                unit = m.group("unit").lower()

                key = (prefix, pos)
                if key not in groups:
                    groups[key] = {"minutes": [], "seconds": []}

                # retrieve numeric from ann["value"]["taxonomy"] nested-array
                num = None
                if isinstance(ann.get("value"), dict) and "taxonomy" in ann["value"]:
                    # ann["value"]["taxonomy"] typically looks like [["06"]]
                    num = _parse_primitive_as_int(ann["value"]["taxonomy"])

                if unit.startswith("min"):
                    groups[key]["minutes"].append((idx_ann, ann, num))
                else:
                    groups[key]["seconds"].append((idx_ann, ann, num))

            # 3) For each group compute global total seconds and write back minute/second split
            for (prefix, pos), info in groups.items():
                stats["taxonomy_groups"] += 1
                # Determine minute and second sources (use first available value if multiple; missing -> 0)
                minute_val = None
                second_val = None
                if info["minutes"]:
                    # first minute annotation's parsed numeric value
                    minute_val = info["minutes"][0][2]
                if info["seconds"]:
                    second_val = info["seconds"][0][2]

                if minute_val is None:
                    minute_val = 0
                if second_val is None:
                    second_val = 0

                try:
                    minute_int = int(minute_val)
                except Exception:
                    minute_int = 0
                try:
                    second_int = int(second_val)
                except Exception:
                    second_int = 0

                total_seconds = float(minute_int * 60 + second_int) + float(offset_seconds)
                total_seconds = _clamp_nonneg(total_seconds)
                if total_seconds is None:
                    total_seconds = 0.0

                # compute global minute and second components
                global_minute = int(total_seconds) // 60
                global_second = int(total_seconds) % 60

                # format strings: minutes with at least 2 digits, seconds always 2 digits
                minute_str = f"{global_minute:02d}"
                second_str = f"{global_second:02d}"

                # update all minute annotations in group to contain minute_str in nested taxonomy array
                for idx_ann, ann_obj, _num in info["minutes"]:
                    # ensure value dict exists
                    if not isinstance(results[idx_ann].get("value"), dict):
                        results[idx_ann]["value"] = {}
                    results[idx_ann]["value"]["taxonomy"] = [[minute_str]]
                    stats["taxonomy_updated"] += 1

                # update all second annotations in group to contain second_str in nested taxonomy array
                for idx_ann, ann_obj, _num in info["seconds"]:
                    if not isinstance(results[idx_ann].get("value"), dict):
                        results[idx_ann]["value"] = {}
                    results[idx_ann]["value"]["taxonomy"] = [[second_str]]
                    stats["taxonomy_updated"] += 1

        logger.info("[GLOBALISE] records=%d annotations=%d labels_seen=%d labels_shifted=%d taxonomy_groups=%d taxonomy_updated=%d",
                    stats["records"], stats["annotations_total"], stats["labels_seen"], stats["labels_shifted"],
                    stats["taxonomy_groups"], stats["taxonomy_updated"])

        return out_records

    # ------------------------------
    # Consolidation
    # ------------------------------
    def consolidate_shard(self, shard_json: Dict[str, Any], *, default_duration: int = 60, overwrite_existing: bool = False) -> Dict[str, Any]:
        """
        Consolidate a single shard JSON into its video master.

        Args:
            shard_json (dict): Shard JSON object.
            default_duration (int): Default shard duration in seconds.
            overwrite_existing (bool): Whether to overwrite existing shard in master.

        Returns:
            dict: Updated master JSON.
        """
        try:
            video_id = shard_json["data"]["video_name"]
            shard_number = int(shard_json["data"]["shard_number"])
            total_shards = int(shard_json["data"].get("total_shards", 0))

            master = self.load_master(video_id, total_shards=total_shards)
            if total_shards:
                master["total_shards"] = total_shards

            if shard_number in master["consolidated_shards"] and not overwrite_existing:
                print(f"ℹ️ No new merge needed for {video_id} (shard {shard_number})")
                return master
            
            offset_seconds=compute_shard_offset(shard_number)
            shard_key = f"shard_{shard_number}"
            shard_entry = {
                "shard_number": shard_number,
                "data": copy.deepcopy(shard_json.get("data", {})),
                "predictions": shard_json.get("predictions", []),
                "annotations": self._globalise_annotations(shard_json.get("annotations", []), offset_seconds)
            }

            try:
                counters = self.globalise_predictions(shard_entry["predictions"], offset_seconds)
                logger.info(f"[GLOBALISE] Globalised predictions for {shard_key} with offset_seconds={offset_seconds}")
                # Log a compact summary (mirrors earlier summary style)
                for key, counts in counters.items():
                    logger.info(f"[GLOBALISE] {key}: total={counts['total']}, adjusted={counts['adjusted']}")
            except Exception as e:
                logger.exception(f"⚠️ Error globalising predictions for shard={shard_number}: {e}")

            master["shards"][shard_key] = shard_entry

            if shard_number not in master["consolidated_shards"]:
                master["consolidated_shards"].append(shard_number)
                master["consolidated_shards"].sort()

            try:
                self._append_shard_event_to_run_summary(master, shard_key, shard_entry)
            except Exception as e:
                print(f"[warn] failed to append shard event to run-summary: {e}")


            total = master.get("total_shards", 0)
            master["status"] = "completed" if total and len(master["consolidated_shards"]) >= total else "in_progress"

            master_url = self.save_master(master)
            final_url = ""
            if master["status"] == "completed":
                flat = self.prepare_final_json(master)
                final_url = self.save_final_master(flat)
                if self.do_stitching:
                    ref = self.prepare_and_run_stitcher(flat, out_dir=Path("/tmp/out_mp4"))
                    # To wait for completion:
                    result_path = ray.get(ref)
                    print("Completed, output:", result_path)

                    # --- upload result_path to Azure blob ---
                    try:
                        local_out = Path(result_path)
                        # place results under folder named by vid
                        dest_blob_name = f"final-video-mp4s/{local_out.name}"

                        # choose container name (adjust if needed)
                        container_name = FINAL_MASTER_JSONS_CONTAINER

                        uploaded_url = self.upload_file_to_blob(
                            local_file=local_out,
                            connection_string=AZURE_STORAGE_CONNECTION_STRING,
                            container_name=container_name,
                            dest_blob_path=dest_blob_name,
                            overwrite=True,
                        )
                        print(f"[info] Uploaded stitched mp4 to: {uploaded_url}")
                    except Exception as e:
                        print(f"[error] Failed to upload stitched result to blob: {e}")


            print(f"✅ Master URL: {master_url}")
            if final_url:
                print(f"✅ Final Master URL: {final_url}")

            return master
        except Exception as e:
            print(f"❌ Error consolidating shard: {e}")
            return {}

    def consolidate_multiple_shard_jsons(self, chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Consolidate multiple shard JSONs at once.

        Args:
            chunks (list): List of shard JSON objects.

        Returns:
            dict: Mapping of video_id → updated master JSON.
        """
        updated: Dict[str, Dict[str, Any]] = {}
        for c in chunks:
            m = self.consolidate_shard(c)
            if m:
                updated[m["original_video_id"]] = m
        return updated

    def prepare_final_json(self, master: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export a master JSON into its final structured form (sorted shards, etc.).

        Args:
            master (dict): Master JSON object.

        Returns:
            dict: Final export JSON.
        """
        try:
            out: Dict[str, Any] = {
                "original_video_id": master["original_video_id"],
                "total_shards": master.get("total_shards", 0),
                "consolidated_shards": master.get("consolidated_shards", []),
                "status": master.get("status", "in_progress"),
                "shards": {}
            }
            for shard_key in sorted(master["shards"].keys(), key=lambda x: int(x.replace("shard_", ""))):
                out["shards"][shard_key] = copy.deepcopy(master["shards"][shard_key])
            return out
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return master


    def report_status(self, video_id: str) -> Dict[str, Any]:
        """
        Report consolidation status for a given video.
        """
        master = self.load_master(video_id)

        consolidated = set(master.get("consolidated_shards", []))
        total = master.get("total_shards", 0)

        report = {
            "video_id": video_id,
            "status": master.get("status", "unknown"),
            "total_shards_expected": total,
            "shards_consolidated": sorted(list(consolidated)),
            "shards_missing": [],
        }

        # If we know total shards, we can compute missing
        if total:
            all_expected = set(range(1, total + 1))
            missing = all_expected - consolidated
            report["shards_missing"] = sorted(list(missing))

        return report
    
    def poll_labelstudio(
        self,
        fetch_reviewed_chunks: Callable[[], List[Dict[str, Any]]],
        project_id: int,
        view_id: int,
        interval_seconds: int = POLL_INTERVAL_SEC,
        run_forever: bool = True,
    ) -> None:
        """
        Continuously poll Label Studio for reviewed tasks,
        consolidate shards, and print per-video reports.

        Args:
            fetch_reviewed_chunks (Callable): Function to fetch reviewed LS tasks.
            project_id (int): Label Studio project ID.
            view_id (int): Label Studio view ID.
            interval_seconds (int): Polling interval in seconds.
        """
        while True:
            try:
                chunks = fetch_reviewed_chunks(project_id, view_id)
                if chunks:
                    updated = self.consolidate_multiple_shard_jsons(chunks)

                    # ✅ Print a report for each updated video
                    for video_id, master in updated.items():
                        report = self.report_status(video_id)
                        print(f"📊 Report for {video_id}:")
                        print(json.dumps(report, indent=2))

            except Exception as e:
                print(f"[poll] error: {e}")

            if not run_forever:
                return

            # keep the existing behaviour when run_forever True
            time.sleep(interval_seconds)


    def upload_file_to_blob(self,local_file: Path,
                            connection_string: str,
                            container_name: str,
                            dest_blob_path: str = None,
                            overwrite: bool = True) -> str:
        """
        Upload local_file to Azure Blob Storage.
        - local_file: Path to the local file to upload
        - connection_string: AZURE_STORAGE_CONNECTION_STRING
        - container_name: name of the target container
        - dest_blob_path: path (including file name) in the container. If None, uses local_file.name
        - overwrite: whether to overwrite existing blob
        
        Returns: blob_client.url (string)
        """
        local_file = Path(local_file)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_file}")

        if dest_blob_path is None:
            dest_blob_path = local_file.name

        # Normalize dest path (no leading '/')
        dest_blob_path = str(dest_blob_path).lstrip("/")

        # Init client
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service.get_container_client(container_name)
        try:
            container_client.create_container()
            print(f"[info] Created container: {container_name}")
        except Exception:
            # container may already exist; ignore
            pass

        blob_client = container_client.get_blob_client(dest_blob_path)

        # Determine content type
        content_type, _ = mimetypes.guess_type(local_file.name)
        if content_type is None:
            # fallback for MP4/INSV etc.
            if local_file.suffix.lower() in (".mp4", ".insv"):
                content_type = "video/mp4"
            elif local_file.suffix.lower() in (".wav", ".mp3"):
                content_type = "audio/wav"
            else:
                content_type = "application/octet-stream"

        # Upload with streaming
        with open(local_file, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=overwrite,
                content_settings=ContentSettings(content_type=content_type)
            )

        # Return the blob url (includes account and container)
        return blob_client.url

    def download_blob_to_local(self,blob_path: str,
                            connection_string: str,
                            container_name: str,
                            dest_dir: Path,
                            filename: str = None,
                            overwrite: bool = False) -> Path:
        """
        Downloads blob from Azure Blob Storage to dest_dir/filename using blob path and connection string.

        Args:
            blob_path: Path of the blob inside the container (e.g. "azure_directory_path/Oslo/test2/file.insv")
            connection_string: Azure storage connection string
            container_name: Name of the blob container
            dest_dir: Local directory where file will be saved
            filename: Optional filename for saving locally. If None, inferred from blob_path.
            overwrite: If False and file already exists, returns existing path.

        Returns:
            Path to local downloaded file.
        """
        if not blob_path:
            raise ValueError("No blob_path provided")

        dest_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = Path(blob_path).name
        dest_path = dest_dir / filename

        if dest_path.exists() and not overwrite:
            return dest_path

        # Initialize BlobServiceClient
        blob_service = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service.get_blob_client(container=container_name, blob=blob_path)

        # Download the blob
        with open(dest_path, "wb") as f:
            data = blob_client.download_blob()
            data.readinto(f)

        return dest_path
    
    def _find_consolidated_result(self,predictions: List[Dict]) -> Optional[List[Dict]]:
        """Return the 'result' list for the consolidated_model_results entry (or similar)."""
        if not isinstance(predictions, list):
            return None
        for p in predictions:
            # common key in your snippet: id == "consolidated_model_results"
            if p.get("id") == "consolidated_model_results":
                return p.get("result")
            # fallback: maybe the entry is the first one and contains view_results
            res = p.get("result")
            if isinstance(res, list):
                for r in res:
                    # if we find view_results -> back -> clap -> combined_analysis, assume this is the one
                    val = r.get("value") if isinstance(r, dict) else None
                    if isinstance(val, dict) and _safe_get(val, ["view_results", "back", "clap", "combined_analysis"]) is not None:
                        return res
        return None

    def _extract_claps_from_result_list(self,result_list: List[Dict]) -> Dict[str, Any]:
        """
        From a result list (like data['predictions'][0]['result']), return clap dicts:
        {'audio_clap': {...} or None, 'video_clap': {...} or None, 'detected_clap': {...} or None}
        """
        if not isinstance(result_list, list):
            return {"audio_clap": None, "video_clap": None, "detected_clap": None}

        # Try direct path used in your snippet:
        for item in result_list:
            if item.get("id") == "consolidated_model_results":
                # value -> view_results -> back -> clap -> combined_analysis
                combined = _safe_get(item, ["value", "view_results", "back", "clap", "combined_analysis"])
                if combined:
                    audio = combined.get("audio_clap")
                    video = combined.get("video_clap")
                    detected = combined.get("detected_clap") or combined.get("detected")
                    return {"audio_clap": audio, "video_clap": video, "detected_clap": detected}

        # Generic search fallback: scan for any dict containing combined_analysis
        for item in result_list:
            val = item.get("value")
            if isinstance(val, dict):
                # recursively search for 'combined_analysis' key in this subtree
                stack = [val]
                while stack:
                    node = stack.pop()
                    if not isinstance(node, dict):
                        continue
                    if "combined_analysis" in node and isinstance(node["combined_analysis"], dict):
                        comb = node["combined_analysis"]
                        return {
                            "audio_clap": comb.get("audio_clap"),
                            "video_clap": comb.get("video_clap"),
                            "detected_clap": comb.get("detected_clap") or comb.get("detected"),
                        }
                    # push children
                    for v in node.values():
                        if isinstance(v, dict):
                            stack.append(v)
                        elif isinstance(v, list):
                            for e in v:
                                if isinstance(e, dict):
                                    stack.append(e)
        return {"audio_clap": None, "video_clap": None, "detected_clap": None}

    def _shard_key_index(self,key: str) -> int:
        """Return numeric index for keys like 'shard_1' or plain '1'. If not possible, return large."""
        if key is None:
            return 10**9
        m = re.search(r"(\d+)$", str(key))
        if m:
            return int(m.group(1))
        try:
            return int(key)
        except Exception:
            return 10**9

    def _get_field(self,d: Optional[Dict], *path, default=None):
        """Safe nested dict getter."""
        cur = d
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def extract_claps_from_master(self,master_json: Dict) -> Dict[str, Any]:
        """
        Extract claps and compute useful artifacts (timestamps, global timestamps, blob paths,
        original durations, and suggested trim start/end seconds).

        Returns dict:
        {
            "per_shard": [ ... ],
            "first_clap": { ... },   # dict with many convenience fields (see below)
            "last_clap": { ... },
            "trim_suggestions": {
                "trim_video_start_seconds": float or 0.0,
                "trim_video_end_seconds": float or None,
                "trim_audio_start_seconds": float or 0.0,
                "trim_audio_end_seconds": float or None,
                "manual_audio_offset_seconds": float or None  # audio_global - video_global if both exist in same shard or cross-shard fallback
            }
        }

        First/Last clap dict fields (if available):
        - audio_clap_raw_ts (float)      : timestamp reported inside shard (seconds)
        - audio_clap_global_ts (float)   : audio_clap_raw_ts + shard_offset_seconds
        - audio_full_length (float)      : audio original_full_audio_length (seconds) if present
        - audio_blob_path (str)          : original_audio_blob_path if present
        - video_clap_raw_ts (float)
        - video_clap_global_ts (float)
        - video_full_length (float)
        - video_blob_path (str)
        - shard_key / shard_idx / shard_offset_seconds
        """
        shards = master_json.get("shards") or {}
        per_shard = []

        # parameters for trim suggestion
        # pre_pad = 2.0   # seconds before first clap to keep
        # post_pad = 2.0  # seconds after last clap to keep

        for shard_key, shard_obj in shards.items():
            preds = shard_obj.get("predictions") or shard_obj.get("prediction") or []
            result_list = self._find_consolidated_result(preds)
            claps = self._extract_claps_from_result_list(result_list) if result_list is not None else {"audio_clap": None, "video_clap": None, "detected_clap": None}
            
            # compute shard_offset_seconds heuristically (you used default_seconds = 60 previously)
            shard_offset = None
            try:
                ds = shard_obj.get("data", {}) or {}
                # try reading explicit shard_offset if present, else fallback to (shard_number-1)*default_seconds
                if "shard_offset_seconds" in ds and ds["shard_offset_seconds"] not in (None, "", {}):
                    shard_offset = float(ds["shard_offset_seconds"])
                else:
                    shard_number = int(ds.get("shard_number", 1))
                    default_seconds = 60
                    shard_offset = float((shard_number - 1) * default_seconds)
                    video_id=ds["video_name"]
            except Exception:
                shard_offset = None

            per_shard.append({
                "shard_key": shard_key,
                "shard_idx": self._shard_key_index(shard_key),
                "audio_clap": claps.get("audio_clap"),
                "video_clap": claps.get("video_clap"),
                "detected_clap": claps.get("detected_clap"),
                "shard_offset_seconds": shard_offset,
            })

        # sort by shard index
        per_shard = sorted(per_shard, key=lambda x: x["shard_idx"])

        # helper to extract fields from clap dict (which may be dict or None)
        def _ts_from_clap(c):
            if c is None:
                return None
            if isinstance(c, dict):
                return self._get_field(c, "timestamp", default=None) or self._get_field(c, "time", default=None) or self._get_field(c, "ts", default=None)
            # if c is numeric already
            try:
                return float(c)
            except Exception:
                return None

        def _orig_len_from_clap(c):
            if not isinstance(c, dict):
                return None
            # common names in your samples:
            return self._get_field(c, "original_full_audio_length", default=None) or self._get_field(c, "original_audio_length", default=None) or self._get_field(c, "duration_seconds", default=None)

        def _orig_blob_path_from_clap(c):
            if not isinstance(c, dict):
                return None
            return self._get_field(c, "original_audio_blob_path", default=None) or self._get_field(c, "original_audio_blob_url", default=None) \
                or self._get_field(c, "original_video_blob_path", default=None) or self._get_field(c, "original_video_blob_url", default=None)

        # find first/last shards that contain any clap info
        first_item = next((s for s in per_shard if any(s[k] for k in ("audio_clap", "video_clap", "detected_clap"))), None)
        last_item = next((s for s in reversed(per_shard) if any(s[k] for k in ("audio_clap", "video_clap", "detected_clap"))), None)

        def enrich_item(item):
            """Return enriched dict with raw/global timestamps and blob paths/durations."""
            if not item:
                return None
            a_clap = item.get("audio_clap")
            v_clap = item.get("video_clap")
            d_clap = item.get("detected_clap")
            shard_offset = item.get("shard_offset_seconds") or 0.0

            a_raw = _ts_from_clap(a_clap) or _ts_from_clap(d_clap)
            v_raw = _ts_from_clap(v_clap) or _ts_from_clap(d_clap)
            a_global = float(a_raw) + float(shard_offset) if (a_raw is not None) else None
            v_global = float(v_raw) + float(shard_offset) if (v_raw is not None) else None

            a_full_len = _orig_len_from_clap(a_clap)
            v_full_len = None
            if isinstance(v_clap, dict):
                v_full_len = self._get_field(v_clap, "original_video_length", default=None) or self._get_field(v_clap, "duration_seconds", default=None)

            # blob paths
            a_blob_path = None
            v_blob_path = None
            if isinstance(a_clap, dict):
                a_blob_path = self._get_field(a_clap, "original_audio_blob_path", default=None) or self._get_field(a_clap, "original_audio_blob_url", default=None)
            if isinstance(v_clap, dict):
                v_blob_path = self._get_field(v_clap, "original_video_blob_path", default=None) or self._get_field(v_clap, "original_video_blob_url", default=None)
            # detected_clap may carry original urls too
            if not a_blob_path and isinstance(d_clap, dict):
                a_blob_path = self._get_field(d_clap, "original_audio_blob_path", default=None) or a_blob_path
                v_blob_path = self._get_field(d_clap, "original_video_blob_path", default=None) or v_blob_path

            return {
                "shard_key": item.get("shard_key"),
                "shard_idx": item.get("shard_idx"),
                "shard_offset_seconds": shard_offset,
                "audio_clap_raw_ts": float(a_raw) if a_raw is not None else None,
                "audio_clap_global_ts": float(a_global) if a_global is not None else None,
                "audio_full_length": float(a_full_len) if a_full_len is not None else None,
                "audio_blob_path": a_blob_path,
                "video_clap_raw_ts": float(v_raw) if v_raw is not None else None,
                "video_clap_global_ts": float(v_global) if v_global is not None else None,
                "video_full_length": float(v_full_len) if v_full_len is not None else None,
                "video_blob_path": v_blob_path
            }

        first_enriched = enrich_item(first_item)
        last_enriched = enrich_item(last_item)

        trim_video_start_seconds = None
        trim_video_end_seconds = None
        trim_audio_start_seconds = None
        trim_audio_end_seconds = None

        # video trims: need global timestamps and total video length
        if first_enriched and first_enriched.get("video_clap_global_ts") is not None:
            trim_video_start_seconds = max(0.0, first_enriched["video_clap_global_ts"])
        if last_enriched and last_enriched.get("video_clap_global_ts") is not None and last_enriched.get("video_full_length") is not None:
            # how many seconds to cut from end = total_length - (last_clap_global + post_pad)
            # ensure not negative
            end_trim_video = float(last_enriched["video_full_length"]) - (float(last_enriched["video_clap_global_ts"]))
            trim_video_end_seconds = max(0.0, end_trim_video) if last_enriched["video_full_length"] is not None else None

        # audio trims:
        if first_enriched and first_enriched.get("audio_clap_global_ts") is not None:
            trim_audio_start_seconds = max(0.0, first_enriched["audio_clap_global_ts"])
        if last_enriched and last_enriched.get("audio_clap_global_ts") is not None and last_enriched.get("audio_full_length") is not None:
            end_trim_audio = float(last_enriched["audio_full_length"]) - (float(last_enriched["audio_clap_global_ts"]))
            trim_audio_end_seconds = max(0.0, end_trim_audio) if last_enriched["audio_full_length"] is not None else None

        # If we couldn't compute end trims because durations missing, leave them as None (caller can fallback to not trimming end)
        trim_suggestions = {
            "trim_video_start_seconds": trim_video_start_seconds,
            "trim_video_end_seconds": trim_video_end_seconds,
            "trim_audio_start_seconds": trim_audio_start_seconds,
            "trim_audio_end_seconds": trim_audio_end_seconds
        }

        return {
            "first_clap": first_enriched,
            "last_clap": last_enriched,
            "trim_suggestions": trim_suggestions,
            "video_id":video_id,
            "video_blob_path":first_enriched.get("video_blob_path"),
            "audio_blob_path":first_enriched.get("audio_blob_path")
        }




    # ---- main wrapper to prepare args and call Ray stitcher ----
    def prepare_and_run_stitcher(self,master_json: Dict[str, Any],
                                out_dir: Path,
                                downloads_dir: Path = Path("/tmp/clap_downloads"),
                                overwrite_downloads: bool = False,
                                overwrite_output: bool = False,
                                dry_run: bool = False):
        """
        - master_json: consolidated JSON (dict)
        - out_dir: output directory Path for final mp4
        - downloads_dir: where to save downloaded original audio/video blobs
        """

        # 1) extract clap info & trim suggestions
        result = self.extract_claps_from_master(master_json)
        first_clap = result.get("first_clap") or {}
        trim = result.get("trim_suggestions", {})
        activity = result.get("video_id") or "unknown_activity"

        video_blob_url = result.get("video_blob_path") or first_clap.get("video_blob_path")
        audio_blob_url = result.get("audio_blob_path") or first_clap.get("audio_blob_path")

        if not video_blob_url and not audio_blob_url:
            raise RuntimeError("No audio or video blob URLs found in consolidated master JSON.")

        # 2) download blobs to local files (if present)
        downloads_dir = Path(downloads_dir)
        downloads_dir.mkdir(parents=True, exist_ok=True)

        local_video_path = None
        local_audio_path = None
        try:
            if video_blob_url:
                print(f"[info] Downloading video blob: {video_blob_url}")
                local_video_path = self.download_blob_to_local(
                    blob_path=video_blob_url,
                    connection_string=AZURE_STORAGE_CONNECTION_STRING,
                    container_name=CONTAINER_CONTAINING_VIDEOS,              # 👈 adjust container name if needed
                    dest_dir=downloads_dir,
                    overwrite=overwrite_downloads
                )
                print(f"[info] Saved video to {local_video_path}")

            if audio_blob_url:
                print(f"[info] Downloading audio blob: {audio_blob_url}")
                local_audio_path = self.download_blob_to_local(
                    blob_path=audio_blob_url,
                    connection_string=AZURE_STORAGE_CONNECTION_STRING,
                    container_name=CONTAINER_CONTAINING_AUDIOS,              # 👈 adjust container name if needed
                    dest_dir=downloads_dir,
                    overwrite=overwrite_downloads
                )
                print(f"[info] Saved audio to {local_audio_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download blobs: {e}")


        # 3) prepare lists for stitcher (expecting lists of paths)
        videos = [str(local_video_path)] if local_video_path else []
        audios = [str(local_audio_path)] if local_audio_path else []

        # 4) prepare RenderSettings and trim args
        settings = RenderSettings()  # use defaults; change if needed

        # use trim suggestions but ensure numeric defaults (stitcher expects floats)
        def _to_float_or_default(x, default=0.0):
            try:
                return float(x) if x is not None else default
            except Exception:
                return default

        trim_video_start_seconds = _to_float_or_default(trim.get("trim_video_start_seconds"), 0.0)
        # stitcher expects trim end seconds too; if None we pass 0.0 (no end-trim)
        trim_video_end_seconds = _to_float_or_default(trim.get("trim_video_end_seconds"), 0.0)
        trim_audio_start_seconds = _to_float_or_default(trim.get("trim_audio_start_seconds"), 0.0)
        trim_audio_end_seconds = _to_float_or_default(trim.get("trim_audio_end_seconds"), 0.0)

        # 5) call the Ray remote stitcher (no manual offset per your request)
        # Note: build_and_run_ffmpeg_for_group is ray.remote - use .remote(...)
        try:
            objref = build_and_run_ffmpeg_for_group.remote(
                activity=activity,
                ts="",
                videos=videos,
                audios=audios,
                out_dir=out_dir,
                settings=settings,
                overwrite=overwrite_output,
                dry_run=dry_run,
                sync_mode="none",  # you said you don't need manual seconds; choose desired fallback
                sync_window_sec=30,
                manual_audio_offset_seconds=None,
                trim_video_start_seconds=trim_video_start_seconds,
                trim_video_end_seconds=trim_video_end_seconds,
                trim_audio_start_seconds=trim_audio_start_seconds,
                trim_audio_end_seconds=trim_audio_end_seconds,
            )
            print("[info] Stitcher invoked, returned Ray ObjectRef.")
            return objref
        except Exception as e:
            raise RuntimeError(f"Failed to invoke stitcher: {e}")
    
    def _ensure_run_summary_started(self):
        """
        Ensure an in-memory run summary exists for the current scheduled window.
        Called lazily when the first shard is consolidated during a run.
        """
        with self._run_summary_lock:
            if self._current_run_summary is not None:
                return self._current_run_summary
            # create a new run summary with PST timestamp (run_time)
            pst = zoneinfo.ZoneInfo("America/Los_Angeles")
            run_time = datetime.now(timezone.utc).astimezone(pst)
            run_time_iso = run_time.isoformat()
            filename_ts = run_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_ts}_summary.json"
            self._current_run_summary = {
                "run_time": run_time_iso,
                "no_of_shards_processed_in_run": 0,
                "per_video": {},  # keyed by video_id
            }
            self._current_run_filename = filename
            return self._current_run_summary

    def _append_shard_event_to_run_summary(self, master: Dict[str, Any], shard_key: str, shard_entry: Dict[str, Any]):
        """
        Append info about this consolidated shard to the open run-summary.
        Must be called after master['shards'][shard_key] assignment and consolidated_shards update.
        """
        summary = self._ensure_run_summary_started()
        with self._run_summary_lock:
            # derive video id & total_shards
            video_id = master.get("original_video_id") or master.get("video_id") or shard_entry.get("data", {}).get("video_name", "unknown_video")
            total_shards = master.get("total_shards") or shard_entry.get("data", {}).get("total_shards") or None

            # ensure per_video entry exists
            per_video = summary["per_video"].setdefault(video_id, {
                "video_id": video_id,
                "total_shards": total_shards,
                "consolidated_shards": [],
                "shards_consolidated_count": 0,
                "shards": []  # list of per-shard objects
            })

            # current consolidated list from master (post-merge)
            consolidated_list = master.get("consolidated_shards", [])
            per_video["consolidated_shards"] = consolidated_list
            per_video["shards_consolidated_count"] = len(consolidated_list)
            per_video["total_shards"] = total_shards

            # derive shard index
            try:
                shard_idx = int(str(shard_key).split("_")[-1])
            except Exception:
                shard_idx = None

            # gather per-shard model signals if present (predictions) - optional small extraction
            model_signals = []
            try:
                preds = shard_entry.get("predictions") or []
                # quick scan: gather keys or id markers
                for p in preds:
                    # look for consolidated_model_results item
                    if isinstance(p, dict) and p.get("id") == "consolidated_model_results":
                        # record that model produced consolidated results for this shard
                        model_signals.append("consolidated_model_results")
                        break
                # You can extend model signal extraction here if you need detailed signals
            except Exception:
                pass

            # summarize annotations for this shard (your function returns the desired shape)
            shard_annotations = shard_entry.get("annotations") or []
            try:
                annotation_signals = self.summarize_annotations(shard_annotations)
            except Exception as e:
                # fallback: record error and continue
                annotation_signals = {"error": f"summarize_annotations failed: {e}"}

            # optional: include clap details if present in predictions/combined_analysis
            audio_clap = None
            video_clap = None
            try:
                # try to find combined_analysis inside predictions
                for p in shard_entry.get("predictions", []):
                    if isinstance(p, dict):
                        for res in p.get("result", []) if isinstance(p.get("result", []), list) else []:
                            val = res.get("value") if isinstance(res, dict) else None
                            if val and isinstance(val, dict):
                                # view_results -> back -> clap -> combined_analysis
                                comb = val.get("view_results", {}).get("back", {}).get("clap", {}).get("combined_analysis")
                                if comb:
                                    audio_clap = comb.get("audio_clap")
                                    video_clap = comb.get("video_clap")
                                    break
                    if audio_clap or video_clap:
                        break
            except Exception:
                pass

            # --- NEW: extract metadata and AI predictions ---
            try:
                raw_data = dict(shard_entry.get("data") or {})
                meta_pred_keys = [k for k in raw_data.keys() if k.lower().startswith("meta")]
                meta_summary = {k: raw_data[k] for k in meta_pred_keys}

                # Collect AI_* keys into a compact dict
                ai_pred_keys = [k for k in raw_data.keys() if k.upper().startswith("AI_")]
                ai_predictions_summary = {k: raw_data[k] for k in ai_pred_keys}

            except Exception:
                metadata = {}
                ai_predictions_summary = {"error": "failed to extract ai predictions safely"}

            shard_obj = {
                "shard_key": shard_key,
                "shard_idx": shard_idx,
                "consolidated_at": datetime.now(timezone.utc).astimezone(zoneinfo.ZoneInfo("America/Los_Angeles")).isoformat(),
                "consolidated_shards_after_merge": consolidated_list,
                "shards_consolidated_count_for_video": len(consolidated_list),
                "annotation_signals": annotation_signals,
                # "model_signals": model_signals,
                "audio_clap": audio_clap,
                "video_clap": video_clap,
                # NEW fields
                "metadata": meta_summary,
                "ai_predictions": ai_predictions_summary,
            }


            # replace any previous entry for this shard (dedupe) — remove old then append
            per_video["shards"] = [s for s in per_video["shards"] if s.get("shard_key") != shard_key]
            per_video["shards"].append(shard_obj)

            # increment run-level shard counter (only count new shard events)
            summary["no_of_shards_processed_in_run"] = summary.get("no_of_shards_processed_in_run", 0) + 1

            # done — keep summary in memory; will be uploaded at finish
            return shard_obj

    def finish_run_and_upload_current_summary(self, summary_container_name: str, overwrite: bool = False) -> str:
        """
        Serialize and upload the in-memory run-summary to the given container.
        Returns blob URL. Resets the in-memory run summary to None (new run will start next time).
        """
        with self._run_summary_lock:
            if not self._current_run_summary:
                raise RuntimeError("No run-summary in progress to finish.")
            data = json.dumps(self._current_run_summary, indent=2).encode("utf-8")
            blob_name = self._current_run_filename or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_summary.json"

            # upload using blob_service (self.blob_service exists)
            container_client = self.blob_service.get_container_client(summary_container_name)
            try:
                container_client.create_container()
            except Exception:
                pass
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=overwrite)

            url = blob_client.url

            # reset in-memory summary for next scheduled run
            self._current_run_summary = None
            self._current_run_filename = None

            return url
    def _is_negative_text(self,text: str) -> bool:
        if not text:
            return False
        t = text.strip().lower()
        for neg in _NEGATIVE_KEYWORDS:
            if neg in t:
                return True
        return False

    def _matches_any_keyword(self,name_or_text: str, keywords: Iterable[str]) -> bool:
        if not name_or_text:
            return False
        s = str(name_or_text).lower()
        for kw in keywords:
            if kw in s:
                return True
        return False

    def summarize_annotations(self,annotations, run_time_utc: datetime = None) -> Dict[str, Any]:
        """
        Summarize signals from Label Studio `annotations` list (or single annotation entry).
        Only inspects annotation.result entries (choices/labels/taxonomy/textarea).
        Returns dict with run_time (PST ISO), aggregated flags (y/n), domain (yes/no),
        model_level_signals_detected (list) and counts of unique videos/audios found.
        """
        # normalize input: if user passed a single annotation dict (with "result"), wrap into list
        ann_list = []
        if annotations is None:
            ann_list = []
        elif isinstance(annotations, dict) and "result" in annotations:
            ann_list = [annotations]
        elif isinstance(annotations, list):
            ann_list = annotations
        else:
            # unexpected shape: try to coerce
            ann_list = list(annotations)

        # PST run_time
        now_utc = (run_time_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        pst_tz = zoneinfo.ZoneInfo("America/Los_Angeles")
        run_time_pst = now_utc.astimezone(pst_tz).isoformat()

        # accumulators
        videos_seen: Set[str] = set()
        audios_seen: Set[str] = set()

        # per-channel booleans (sets); presence in set means detected
        audio_signals = set()
        video_signals = set()
        detected_signals = set()

        # keep track of domain choices / comments per channel (two-pass style)
        domain_choice_by_channel = {"audio": False, "video": False, "unknown": False}
        domain_comment_by_channel = {"audio": None, "video": None, "unknown": None}

        # helpers for identifying channel: check to_name/from_name and value context
        def classify_channel(item: Dict[str, Any]) -> str:
            """Return 'audio', 'video', or 'unknown'"""
            to_name = (item.get("to_name") or "") or ""
            from_name = (item.get("from_name") or "") or ""
            tn = str(to_name).lower()
            fn = str(from_name).lower()
            if "audio" in tn or "audio" in fn or "audio_main" in tn or "audio_main" in fn:
                return "audio"
            if any(v in tn for v in ("video", "video_left", "video_right", "video_top", "video_front", "video_back")) or "video" in fn:
                return "video"
            if fn.startswith("audio_") or fn.startswith("audio") or "audio" in fn:
                return "audio"
            if fn.startswith("video_") or "video" in fn:
                return "video"
            return "unknown"

        # helper to test "meaningful" domain comments
        def _is_comment_meaningful(text: str) -> bool:
            if not text:
                return False
            t = text.strip().lower()
            if t == "" or t in ("unknown", "n/a", "na"):
                return False
            return True

        # iterate all annotation entries (first pass: collect data & non-domain signals)
        for ann in ann_list:
            results = ann.get("result", []) if isinstance(ann, dict) else []
            for item in results:
                typ = (item.get("type") or "").lower()
                channel = classify_channel(item)

                value = item.get("value", {})
                choices = value.get("choices") if isinstance(value, dict) else None
                labels = value.get("labels") if isinstance(value, dict) else None
                taxonomy = value.get("taxonomy") if isinstance(value, dict) else None

                textarea_text = None
                if typ == "textarea" and isinstance(value, dict):
                    txt_list = value.get("text")
                    if isinstance(txt_list, list) and txt_list:
                        textarea_text = " ".join([str(t) for t in txt_list]).strip()
                    elif isinstance(value.get("text"), str):
                        textarea_text = value.get("text").strip()

                # --- collect domain choice info (do not immediately add 'domain' as a signal here) ---
                fn = (item.get("from_name") or "") or ""
                tn = (item.get("to_name") or "") or ""
                fn_lower = str(fn).lower()
                tn_lower = str(tn).lower()

                if typ == "choices" and ("domain" in fn_lower or "domain" in tn_lower):
                    # if any choice contains 'yes', treat as positive domain choice
                    if choices and any(isinstance(c, str) and "yes" == c.strip().lower() for c in choices):
                        domain_choice_by_channel[channel] = True
                    # explicit "no" can set to False (keeps default False otherwise)
                    elif choices and any(isinstance(c, str) and c.strip().lower().startswith("no") for c in choices):
                        domain_choice_by_channel[channel] = False

                # --- collect domain textarea comments ---
                if typ == "textarea" and ("domain" in fn_lower or "domain" in tn_lower):
                    # store last seen comment for that channel (if multiple, last one wins)
                    if textarea_text is not None:
                        domain_comment_by_channel[channel] = textarea_text

                # --- derive normalized positive flag for this item (same as before) ---
                positive = False
                if choices:
                    for c in choices:
                        if isinstance(c, str) and "yes" == c.strip().lower():
                            positive = True
                            break
                        if isinstance(c, str) and "yes" in c.strip().lower():
                            positive = True
                            break
                if labels:
                    positive = True
                if taxonomy:
                    positive = True
                if choices and any(isinstance(c, str) and c.strip().lower().startswith("no") for c in choices):
                    positive = False
                if textarea_text:
                    # assume existence of explicit negative phrase implies negative
                    if _is_negative_text(textarea_text):
                        positive = False
                    else:
                        # if contains meaningful words that match our signals, consider positive for matching keywords
                        positive = positive or _matches_any_keyword(textarea_text, sum(_SIGNAL_KEYWORDS.values(), []))

                # build combined text for keyword scanning (exclude domain decision here)
                combined = " ".join([str(fn), str(tn), str(item.get("id", ""))]).lower()
                if textarea_text:
                    combined += " " + textarea_text.lower()
                if choices:
                    combined += " " + " ".join([str(c).lower() for c in choices])

                # --- check each signal keyword group, BUT skip adding domain here ---
                for signal, kws in _SIGNAL_KEYWORDS.items():
                    if signal == "domain":
                        # skip domain additions in first pass; we'll decide after collecting comment + choice.
                        continue
                    if _matches_any_keyword(combined, kws):
                        if positive:
                            detected_signals.add(signal)
                            if channel == "audio":
                                audio_signals.add(signal)
                            elif channel == "video":
                                video_signals.add(signal)
                            else:
                                # unknown channel - add to both to be conservative
                                audio_signals.add(signal)
                                video_signals.add(signal)

        # ---- After processing all items: decide domain detection per channel (second pass) ----
        for ch in ("audio", "video", "unknown"):
            comment = domain_comment_by_channel.get(ch)
            choice_flag = domain_choice_by_channel.get(ch, False)
            # Add domain only when there is a meaningful comment.
            # If comment is not meaningful (e.g., "unknown", empty) we DO NOT add domain,
            # even if a choice said "Yes".
            if _is_comment_meaningful(comment):
                # add domain signal for this channel
                if ch == "audio":
                    audio_signals.add("domain")
                elif ch == "video":
                    video_signals.add("domain")
                else:
                    audio_signals.add("domain")
                    video_signals.add("domain")
                detected_signals.add("domain")
            else:
                # comment not meaningful: do NOT add domain even if choice_flag is True
                # (exactly the behaviour you requested)
                pass

        # build model_level_signals_detected as sorted list
        model_level_signals_detected = sorted(detected_signals)

        # convert booleans to requested y/n or yes/no
        def y_n(flag: bool) -> str:
            return "yes" if flag else "no"

        summary = {
            "run_time": run_time_pst,
            "audio_pii_detected": y_n("pii" in audio_signals),
            "minor_detected": y_n("minors" in audio_signals or "minors" in video_signals),
            "nsfw_content": y_n("nsfw" in audio_signals or "nsfw" in video_signals),
            "domain_detected": "yes" if ("domain" in audio_signals or "domain" in video_signals) else "no",
            "signal_detected": y_n("signal" in audio_signals or "signal" in video_signals),
            "people_detected": y_n("people" in audio_signals or "people" in video_signals),
            "lighting_detected": y_n("lighting" in audio_signals or "lighting" in video_signals),
            "audio_sensitive_info_detected": y_n("sensitive_audio" in audio_signals or "sensitive_audio" in video_signals or "sensitive" in audio_signals),
            "scene_detected": "yes",
        }

        return summary
    def get_current_run_summary(self):
        """Debug helper - returns the in-memory run summary (or None)."""
        return self._current_run_summary


def _graceful_shutdown(actor_handle):
    try:
        print("[main] killing actor...")
        ray.kill(actor_handle)
    except Exception:
        pass
    try:
        ray.shutdown()
    except Exception:
        pass
    print("[main] shutdown complete.")
    sys.exit(0)


if __name__ == "__main__":
    
    ray.init(ignore_reinit_error=True)

    # If LS_CLIENT.fetch_all_tasks isn't picklable, make actor create its own LS client.
    LS_CLIENT = LabelStudioClient(BASE_URL, API_TOKEN, PAGE_SIZE)

    pipeline_actor = ConsolidationPipeline.remote(
        AZURE_STORAGE_CONNECTION_STRING,
        MASTER_JSONS_CONTAINER,
        FINAL_MASTER_JSONS_CONTAINER,
        do_stitching=False
    )

    def _handle_signal(signum, frame):
        print("signal, shutting down...")
        _graceful_shutdown(pipeline_actor)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while True:
            # 1) Start run-summary inside actor (create PST timestamped summary)
            try:
                # This should create the in-memory summary for this run in the actor
                ray.get(pipeline_actor._ensure_run_summary_started.remote(), timeout=30)
            except Exception as e:
                print("failed to ensure run summary started:", e)
                # decide to continue or break; we continue to attempt poll

            # 2) Trigger a single poll run (no sleep in poll_labelstudio)
            try:
                # If you pass fetch function, be cautious of pickling; alternatively let actor build LS client inside.
                poll_ref = pipeline_actor.poll_labelstudio.remote(LS_CLIENT.fetch_all_tasks, PROJECT_ID, VIEW_ID, POLL_INTERVAL_SEC, run_forever=False)
                ray.get(poll_ref, timeout=180) 
            except Exception as e:
                print("poll failed or timed out:", e)
                # continue — we will still attempt to finish/upload whatever summary was accumulated

            # 3) Finish & upload the run summary
            try:
                upload_ref = pipeline_actor.finish_run_and_upload_current_summary.remote(SUMMARY_CONTAINER, overwrite=False)
                uploaded_url = ray.get(upload_ref, timeout=60)
                print("Uploaded run summary:", uploaded_url)
            except Exception as e:
                print("failed to upload run summary:", e)
                # keep going; next iteration will create a new summary
            
            summary = ray.get(pipeline_actor.get_current_run_summary.remote())
            print("CURRENT RUN SUMMARY (actor):", json.dumps(summary, indent=2) if summary else "None")


            # 4) Sleep until next run window
            # (Optionally align to wall clock here instead of simple sleep)
            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        _graceful_shutdown(pipeline_actor)
