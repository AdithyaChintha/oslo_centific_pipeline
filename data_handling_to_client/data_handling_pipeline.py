#!/usr/bin/env python3
"""
complete_pipeline4.py

Watermark + tie-break pipeline that pushes new blobs from an Azure source container
to an OpenAI container, and stores per-submission manifests into a separate Azure
"submission-manifests" container (if configured). Falls back to local manifest storage.

Usage:
    python complete_pipeline4.py --config config.yaml

Requirements:
    pip install azure-storage-blob requests pyyaml python-dateutil
"""
import os
import sys
import json
import argparse
import uuid
import time
import base64
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import mimetypes
import time
import json
import requests

import requests
import yaml
from dateutil import parser as dtparser
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, AzureError

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_pusher")

# ---------- Utilities ----------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def now_iso() -> str:
    return now_utc().isoformat()

def atomic_write_json(path: str, obj: object) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def read_json_if_exists(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_content_md5(md5) -> Optional[str]:
    if md5 is None:
        return None
    if isinstance(md5, (bytes, bytearray)):
        return md5.hex()
    s = str(md5)
    try:
        dec = base64.b64decode(s)
        return dec.hex()
    except Exception:
        return s

def parse_iso_to_utc(iso_str: str) -> datetime:
    dt = dtparser.parse(iso_str)
    return dt.astimezone(timezone.utc)

# ---------- Config ----------
class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load(config_path)

    def _load(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get(self, *keys: str, default=None):
        node = self.config
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

# ---------- Watermark Store ----------
class WatermarkStore:
    """
    Stores watermark JSON locally:
      {"watermark_ts": "<ISO UTC>", "watermark_last_blob": "<blob-name>"}
    """
    def __init__(self, watermark_file: str):
        self.watermark_file = watermark_file
        ensure_dir(str(Path(self.watermark_file).parent))
        self._wm = read_json_if_exists(self.watermark_file) or {}

    def get_watermark(self) -> Tuple[Optional[str], Optional[str]]:
        ts = self._wm.get("watermark_ts")
        lb = self._wm.get("watermark_last_blob")
        return ts, lb

    def set_watermark(self, watermark_ts_iso: str, watermark_last_blob: Optional[str]) -> None:
        obj = {"watermark_ts": watermark_ts_iso, "watermark_last_blob": watermark_last_blob or ""}
        atomic_write_json(self.watermark_file, obj)
        self._wm = obj

# ---------- Azure source client ----------
class AzureSourceClient:
    def __init__(self, connection_string: str, container_name: str):
        self.conn_str = connection_string
        self.container_name = container_name
        self._svc = BlobServiceClient.from_connection_string(self.conn_str)
        self._container_client = self._svc.get_container_client(self.container_name)

    def list_blobs_with_props(self) -> List[dict]:
        blobs = []
        try:
            for b in self._container_client.list_blobs():
                props = {
                    "name": b.name,
                    "size": b.size,
                    "etag": b.etag,
                    "last_modified": b.last_modified.isoformat() if b.last_modified else None,
                    "content_md5": getattr(b, "content_settings", None) and getattr(b.content_settings, "content_md5", None)
                }
                blobs.append(props)
        except AzureError as e:
            raise RuntimeError(f"Failed to list blobs in source container '{self.container_name}': {e}")
        return blobs

    def download_blob_to_path(self, blob_name: str, dest_path: str) -> None:
        blob_client = self._container_client.get_blob_client(blob_name)
        with open(dest_path, "wb") as f:
            stream = blob_client.download_blob()
            stream.readinto(f)

    def metadata_blob_exists(self, metadata_blob_name: str) -> bool:
        try:
            self._container_client.get_blob_client(metadata_blob_name).get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except AzureError:
            return False

    def download_metadata_to_path(self, metadata_blob_name: str, dest_path: str) -> None:
        self.download_blob_to_path(metadata_blob_name, dest_path)

# ---------- Partner uploader (OpenAI Containers + Container-Files + Responses) ---------


import os
import requests
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger("data_pusher")

class PartnerUploader:
    """
    PartnerUploader that:
      - create_partner_container -> POST /v1/containers
      - upload_file_to_container -> tries POST /v1/containers/{id}/files (multipart) for small files;
          falls back to Uploads API (create -> parts -> complete) and then attaches file_id via POST /v1/containers/{id}/files (JSON {"file_id":...})
      - post_submission -> creates a Responses run using a tools entry that references the container (avoids top-level 'container' param)
    """
    OPENAI_BASE = "https://api.openai.com/v1"

    def __init__(
        self,
        openai_api_key: str,
        containers_api: Optional[str] = None,
        container_files_api: Optional[str] = None,
        submissions_api: Optional[str] = None,
        single_request_max_bytes: int = 50 * 1024 * 1024,  # threshold for single-request upload (tuneable)
        upload_part_size: int = 64 * 1024 * 1024,         # 32 MiB part size for Uploads API
    ):
        self.openai_api_key = openai_api_key
        self.containers_api = containers_api or f"{self.OPENAI_BASE}/containers"
        self.container_files_api = container_files_api or f"{self.OPENAI_BASE}/container_files"
        self.responses_api = submissions_api or f"{self.OPENAI_BASE}/responses"
        self.single_request_max_bytes = int(single_request_max_bytes)
        self.upload_part_size = int(upload_part_size)

    def _auth_headers(self, content_type: Optional[str] = None) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {self.openai_api_key}"}
        if content_type:
            h["Content-Type"] = content_type
        return h

    # -----------------------
    # Container creation
    # -----------------------
    def create_partner_container(self, name: Optional[str] = None, purpose: str = "ingestion", extra_body: Optional[dict] = None) -> Dict[str, Any]:
        """
        Create a container in OpenAI. Returns parsed JSON (expects 'id').
        """
        body = {"name": name}
        # if name:
        #     body["name"] = name
        # if extra_body:
        #     body.update(extra_body)

        url = self.containers_api
        logger.info("Creating container via %s", url)
        r = requests.post(url, headers=self._auth_headers("application/json"), json=body, timeout=120)
        if not r.ok:
            logger.error("Failed creating container: status=%s body=%s", r.status_code, r.text)
            r.raise_for_status()
        return r.json()

    # -----------------------
    # Uploads API helpers (create -> parts -> complete)
    # -----------------------
    def _create_upload(self, filename: str, total_bytes: int, purpose: str = "responses", mime_type: str = "application/octet-stream") -> Dict[str, Any]:
        url = f"{self.OPENAI_BASE}/uploads"
        body = {"filename": filename, "bytes": total_bytes, "purpose": purpose, "mime_type": mime_type}
        r = requests.post(url, headers=self._auth_headers("application/json"), json=body, timeout=30)
        r.raise_for_status()
        return r.json()

    def _add_upload_part(self, upload_id: str, part_bytes: bytes) -> Dict[str, Any]:
        url = f"{self.OPENAI_BASE}/uploads/{upload_id}/parts"
        files = {"data": ("chunk", part_bytes)}
        r = requests.post(url, headers=self._auth_headers(), files=files, timeout=300)
        r.raise_for_status()
        return r.json()

    def _complete_upload(self, upload_id: str, part_ids: List[str]) -> Dict[str, Any]:
        url = f"{self.OPENAI_BASE}/uploads/{upload_id}/complete"
        r = requests.post(url, headers=self._auth_headers("application/json"), json={"part_ids": part_ids}, timeout=30)
        r.raise_for_status()
        return r.json()

    def _upload_file_in_parts(self, local_file_path: str, purpose: str = "responses") -> Dict[str, Any]:
        total_bytes = os.path.getsize(local_file_path)
        filename = os.path.basename(local_file_path)
        create_resp = self._create_upload(filename, total_bytes, purpose)
        upload_id = create_resp.get("id") or create_resp.get("upload", {}).get("id")
        if not upload_id:
            raise RuntimeError(f"Upload create did not return upload id: {create_resp}")

        part_ids: List[str] = []
        with open(local_file_path, "rb") as fh:
            while True:
                chunk = fh.read(self.upload_part_size)
                if not chunk:
                    break
                resp = self._add_upload_part(upload_id, chunk)
                pid = resp.get("id") or resp.get("part_id") or (resp.get("part") or {}).get("id")
                if not pid:
                    raise RuntimeError(f"Upload part did not return a part id: {resp}")
                part_ids.append(pid)

        complete_resp = self._complete_upload(upload_id, part_ids)
        return complete_resp


    def _multipart_post_to_containers_files(self, container_id: str, blob_name: str, local_file_path: str, mime_type_override: Optional[str] = None) -> requests.Response:
        """
        POST multipart to /v1/containers/{container_id}/files with a single 'file' part.
        Pass explicit content-type for the file part when possible.
        Returns requests.Response (caller checks .ok/.status_code).
        """
        url = f"{self.containers_api.rstrip('/')}/{container_id}/files"
        headers = {"Authorization": f"Bearer {self.openai_api_key}"}  # let requests set boundary

        mime_type = mime_type_override
        if not mime_type:
            _, ext = os.path.splitext(blob_name)
            if ext.lower() == ".insv":
                mime_type = "application/octet-stream"
            else:
                mime_type = "application/json"


        # Send file with explicit content-type for the part: (filename, fileobj, content_type)
        with open(local_file_path, "rb") as fh:
            files = {"file": (blob_name, fh, mime_type)}
            resp = requests.post(url, headers=headers, files=files, timeout=300)
        return resp

    def _attach_file_id_to_container(self, container_id: str, file_id: str) -> Dict[str, Any]:
        url = f"{self.containers_api.rstrip('/')}/{container_id}/files"
        r = requests.post(url, headers=self._auth_headers("application/json"), json={"file_id": file_id}, timeout=30)
        r.raise_for_status()
        return r.json()

    # -----------------------
    # Public upload method
    # -----------------------
    def upload_file_to_container(self, container_id: str, blob_name: str, local_file_path: str) -> Dict[str, Any]:
        """
        High-level upload:
          - Try direct multipart to /v1/containers/{id}/files first if size <= threshold.
          - If that fails due to 413 or size > threshold, use Uploads API then attach file_id.
        Returns JSON response from successful attach (container-file object).
        """
        size = os.path.getsize(local_file_path)

        # Try direct multipart when small
        if size <= self.single_request_max_bytes:
            try:
                logger.info("Attempting direct multipart upload to container %s for %s (size=%d)", container_id, blob_name, size)
                r = self._multipart_post_to_containers_files(container_id, blob_name, local_file_path)
                if r.ok:
                    logger.info("Direct multipart upload succeeded")
                    return r.json()
                if r.status_code == 413:
                    logger.warning("Direct multipart upload rejected with 413 (too large). Falling back.")
                    # fall through to parts flow
                else:
                    logger.warning("Direct multipart returned status %s: %s", r.status_code, r.text[:1000])
                    # For non-413 errors, attempt parts flow as fallback
            except requests.RequestException as e:
                logger.warning("Direct multipart attempt failed: %s; will try parts flow", e)

        # Upload via Uploads API parts flow
        logger.info("Uploading in parts via Uploads API: %s (size=%d)", blob_name, size)
        completed_upload_resp = self._upload_file_in_parts(local_file_path, purpose="responses")
        # extract file id
        file_id = None
        if isinstance(completed_upload_resp, dict):
            file_obj = completed_upload_resp.get("file") or completed_upload_resp.get("file_object")
            if isinstance(file_obj, dict):
                file_id = file_obj.get("id") or file_obj.get("file_id")
            if not file_id:
                file_id = completed_upload_resp.get("id") or completed_upload_resp.get("file_id")
        if not file_id:
            raise RuntimeError(f"Could not find file id in completed upload response: {completed_upload_resp}")

        # attach file id to container
        logger.info("Attaching uploaded file id %s to container %s", file_id, container_id)
        attach_resp = self._attach_file_id_to_container(container_id, file_id)
        return attach_resp

    # -----------------------
    # Post submission (Responses)
    # -----------------------
    def post_submission(self, container_id: str, model: str = "gpt-4o-mini", prompt: Optional[str] = None, extra_body: Optional[dict] = None) -> Dict[str, Any]:
        """
        Create a Responses run referencing the container via tools config.
        """
        if prompt is None:
            prompt = "Please process the uploaded files in the provided container."

        tools_payload = [
            {
                "type": "code_interpreter",
                "container": {"type": "existing", "id": container_id}
            }
        ]

        body: Dict[str, Any] = {
            "model": model,
            "input": prompt,
            "tools": tools_payload
        }
        if extra_body:
            body.update(extra_body)

        r = requests.post(self.responses_api, headers=self._auth_headers("application/json"), json=body, timeout=120)
        if not r.ok:
            logger.error("Failed creating response (post_submission): status=%s body=%s", r.status_code, r.text)
            r.raise_for_status()
        return r.json()


# ----------------------------------------------------------------------------------------------

# ---------- Remote manifests helper ----------
class RemoteManifestsUploader:
    """
    Uploads manifests to a configured Azure container (creates container if missing).
    """
    def __init__(self, enabled: bool, connection_string: Optional[str], container_name: Optional[str], prefix: Optional[str] = None):
        self.enabled = bool(enabled)
        self.conn_str = connection_string
        self.container_name = container_name
        self.prefix = prefix or ""
        self._client: Optional[ContainerClient] = None
        if self.enabled:
            if not self.conn_str or not self.container_name:
                raise ValueError("remote_manifests.enabled true but connection_string/container_name missing")
            svc = BlobServiceClient.from_connection_string(self.conn_str)
            self._client = svc.get_container_client(self.container_name)
            try:
                # create container if not exists
                self._client.create_container()
            except Exception:
                pass

    def upload_manifest_dict(self, manifest: dict, blob_name: Optional[str] = None) -> str:
        if not self.enabled or not self._client:
            raise RuntimeError("Remote manifests not enabled/configured")
        blob_name = blob_name or f"{self.prefix}{now_utc().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex}.json"
        blob_client = self._client.get_blob_client(blob_name)
        data = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        blob_client.upload_blob(data, overwrite=True, content_settings=None)
        return blob_name

# ---------- Orchestrator ----------
class DataPushOrchestrator:
    def __init__(self, cfg: ConfigLoader):
        self.cfg = cfg

        # OpenAI API key
        self.openai_key = cfg.get("openai", "api_key") or os.getenv(cfg.get("openai", "api_key_env", default="OPENAI_API_KEY"))
        if not self.openai_key:
            raise SystemExit("OpenAI API key not found in config or environment")

        # Azure source
        self.source_conn = cfg.get("azure_source", "connection_string", default="AZURE_SOURCE_CONN")
        self.source_container = cfg.get("azure_source", "container_name")
        if not self.source_conn or not self.source_container:
            raise SystemExit("Azure source connection string or container name missing in config or env")

        self.source_sas = cfg.get("azure_source", "container_sas") or os.getenv(cfg.get("azure_source", "container_sas_env", default=""))
        self.source_prefix = cfg.get("azure_source", "source_prefix", default=None)

        # partner / OpenAI endpoints config (you can override in config)
        self.containers_api = cfg.get("openai", "containers_api", default=f"https://api.openai.com/v1/containers")
        self.container_files_api = cfg.get("openai", "container_files_api", default=f"https://api.openai.com/v1/container_files")
        self.submissions_api = cfg.get("openai", "submissions_api", default=f"https://api.openai.com/v1/responses")
        self.purpose = cfg.get("openai", "purpose", default="ingestion")

        # dedupe safety
        self.safety_window_minutes = int(cfg.get("deduplication", "safety_window_minutes", default=2))

        # local paths
        self.manifests_dir = cfg.get("local_paths", "manifests_dir", default="./local_manifests")
        self.watermark_file = cfg.get("local_paths", "watermark_file", default=str(Path(self.manifests_dir) / "watermark.json"))
        self.work_dir = cfg.get("local_paths", "work_dir", default="./work")
        ensure_dir(self.manifests_dir)
        ensure_dir(self.work_dir)

        # azcopy flag kept but we won't use azcopy in this variant
        self.use_azcopy = False

        # submission behavior
        self.auto_submit = bool(cfg.get("submission", "auto_submit", default=True))
        self.retry_attempts = int(cfg.get("submission", "retry_attempts", default=3))
        self.retry_delay_seconds = int(cfg.get("submission", "retry_delay_seconds", default=60))

        # remote manifests config
        rconf = cfg.get("remote_manifests", default={})
        self.remote_manifests_enabled = bool(rconf.get("enabled", False))
        if self.remote_manifests_enabled:
            rm_conn = rconf.get("connection_string")
            rm_container = rconf.get("container_name")
            rm_prefix = rconf.get("prefix", "")
            self.remote_manifests = RemoteManifestsUploader(True, rm_conn, rm_container, prefix=rm_prefix)
        else:
            self.remote_manifests = RemoteManifestsUploader(False, None, None)

        # helpers
        self.watermark_store = WatermarkStore(self.watermark_file)
        self.source_client = AzureSourceClient(self.source_conn, self.source_container)
        # instantiate uploader with configured endpoints
        self.uploader = PartnerUploader(self.openai_key, containers_api=self.containers_api, submissions_api=self.submissions_api)

    # fingerprint & safety
    def fingerprint_from_blob_props(self, blob_props: dict) -> dict:
        md5 = blob_props.get("content_md5")
        if md5:
            return {"type": "content_md5", "md5": normalize_content_md5(md5)}
        return {"type": "size_etag", "size": blob_props.get("size"), "etag": blob_props.get("etag"), "last_modified": blob_props.get("last_modified")}

    def is_blob_older_than_safety_window(self, last_modified_iso: Optional[str]) -> bool:
        if not last_modified_iso:
            return False
        try:
            lm = parse_iso_to_utc(last_modified_iso)
        except Exception:
            return False
        cutoff = now_utc() - timedelta(minutes=self.safety_window_minutes)
        return lm <= cutoff

    def _watermark_tuple(self) -> Tuple[Optional[datetime], str]:
        ts_iso, last_blob = self.watermark_store.get_watermark()
        if not ts_iso:
            return None, ""
        try:
            wm_dt = parse_iso_to_utc(ts_iso)
        except Exception:
            return None, ""
        return wm_dt, (last_blob or "")

    def blobs_to_send(self) -> List[Tuple[str, dict, str]]:
        all_blobs = self.source_client.list_blobs_with_props()
        if self.source_prefix:
            all_blobs = [b for b in all_blobs if b["name"].startswith(self.source_prefix)]

        wm_dt, wm_last_blob = self._watermark_tuple()
        candidates: List[Tuple[str, dict, str]] = []

        for b in all_blobs:
            name = b["name"]
            last_mod_iso = b.get("last_modified")
            if not self.is_blob_older_than_safety_window(last_mod_iso):
                logger.info("SKIP (safety window) %s last_modified=%s", name, last_mod_iso)
                continue

            try:
                blob_dt = parse_iso_to_utc(last_mod_iso)
            except Exception:
                logger.info("SKIP (bad last_modified) %s last_modified=%s", name, last_mod_iso)
                continue

            select = False
            if wm_dt is None:
                select = True
            else:
                if blob_dt > wm_dt:
                    select = True
                elif blob_dt == wm_dt and name > wm_last_blob:
                    select = True

            if select:
                fp = self.fingerprint_from_blob_props(b)
                candidates.append((name, fp, last_mod_iso))
                logger.info("SELECT %s last_modified=%s fp=%s", name, last_mod_iso, fp)
            else:
                logger.info("SKIP (watermark) %s last_modified=%s", name, last_mod_iso)

        # sort ascending by (last_modified, name)
        candidates.sort(key=lambda t: (parse_iso_to_utc(t[2]), t[0]))
        return candidates

    # metadata/content-type
    def determine_content_type(self, blob_name: str) -> str:
        ln = blob_name.lower()
        if ln.endswith(".json"):
            return "application/json"
        if ln.endswith(".insv"):
            return "application/octet-stream"
        return "application/octet-stream"

    def generate_metadata_obj(self, blob_name: str) -> dict:
        return {"content-type": self.determine_content_type(blob_name), "version": "001"}

    # main run (adapted to use new OpenAI container upload flow; azcopy path removed)
    def run_once(self) -> None:
        logger.info("Starting run (watermark mode).")
        to_send = self.blobs_to_send()
        if not to_send:
            logger.info("No blobs eligible. Exiting.")
            return

        # create OpenAI container
        logger.info("Creating OpenAI container via Containers API")
        try:
            create_resp = self.uploader.create_partner_container(name=f"ingest-{uuid.uuid4().hex}", purpose=self.purpose)
        except Exception as e:
            logger.error("Failed to create container: %s", e)
            raise

        container_id = create_resp.get("id") or create_resp.get("container_id") or create_resp.get("container")
        if not container_id:
            raise RuntimeError(f"Unexpected create response (no container id): {create_resp}")

        submission_manifest = {
            "container_id": container_id,
            "create_response": create_resp,
            "created_at": now_iso(),
            "files": []
        }
        processed: List[Tuple[str, str]] = []

        for blob_name, fp, last_mod_iso in to_send:
            logger.info("Processing %s", blob_name)
            metadata_blob_name = blob_name + ".metadata.json"
            tmp_blob_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(blob_name).name}")
            tmp_meta_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(metadata_blob_name).name}")

            try:
                # download the blob locally
                logger.info("Downloading blob %s", blob_name)
                self.source_client.download_blob_to_path(blob_name, tmp_blob_path)

                # # download or generate metadata
                # if self.source_client.metadata_blob_exists(metadata_blob_name):
                #     logger.info("Downloading metadata %s", metadata_blob_name)
                #     try:
                #         self.source_client.download_metadata_to_path(metadata_blob_name, tmp_meta_path)
                #     except Exception as e:
                #         logger.warning("Failed to download metadata %s: %s - generating instead", metadata_blob_name, e)
                #         with open(tmp_meta_path, "w", encoding="utf-8") as fm:
                #             json.dump(self.generate_metadata_obj(blob_name), fm)
                # else:
                #     logger.info("Generating metadata for %s", blob_name)
                #     with open(tmp_meta_path, "w", encoding="utf-8") as fm:
                #         json.dump(self.generate_metadata_obj(blob_name), fm)

                # upload blob file
                logger.info("Uploading blob %s -> container %s", blob_name, container_id)
                up_resp_blob = self.uploader.upload_file_to_container(container_id, blob_name, tmp_blob_path)
                # # upload metadata file
                # logger.info("Uploading metadata %s -> container %s", metadata_blob_name, container_id)
                # up_resp_meta = self.uploader.upload_file_to_container(container_id, metadata_blob_name, tmp_meta_path)

                # store file-level details in manifest
                submission_manifest["files"].append({
                    "blob_name": blob_name,
                    "fingerprint": fp,
                    # "metadata_blob": metadata_blob_name,
                    "upload_status": "success",
                    "uploaded_at": now_iso(),
                    "upload_response": {
                        "blob": up_resp_blob,
                        # "metadata": up_resp_meta
                    }
                })
                processed.append((last_mod_iso, blob_name))

            except Exception as e:
                logger.error("Error processing %s: %s", blob_name, e, exc_info=True)
                submission_manifest["files"].append({
                    "blob_name": blob_name,
                    "fingerprint": fp,
                    # "metadata_blob": metadata_blob_name,
                    "upload_status": f"failed: {str(e)}",
                    "uploaded_at": now_iso()
                })
            finally:
                for p in (tmp_blob_path, tmp_meta_path):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass

        # persist manifest: remote if configured else local
        manifest_blob_name = f"{now_utc().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex}.json"
        manifest_uploaded = False
        remote_blob_path = None
        if self.remote_manifests.enabled:
            try:
                remote_blob_path = self.remote_manifests.upload_manifest_dict(submission_manifest, blob_name=manifest_blob_name)
                manifest_uploaded = True
                logger.info("Uploaded submission manifest to remote container as %s", remote_blob_path)
            except Exception as e:
                logger.error("Failed to upload manifest to remote container: %s. Falling back to local.", e)

        if not manifest_uploaded:
            local_path = str(Path(self.manifests_dir) / manifest_blob_name)
            atomic_write_json(local_path, submission_manifest)
            logger.info("Wrote submission manifest locally to %s", local_path)

        # submit to OpenAI Responses (auto_submit)
        # submission_response = None
        # if self.auto_submit:
        #     attempt = 0
        #     while attempt < self.retry_attempts:
        #         attempt += 1
        #         try:
        #             logger.info("Posting submission to Responses API (attempt %d)", attempt)
        #             # choose model and prompt from config if present
        #             model = self.cfg.get("submission", "model", default=self.cfg.get("openai", "model", default="gpt-4o-mini"))
        #             prompt = self.cfg.get("submission", "prompt", default=f"Process files in container {container_id} and return a JSON summary.")
        #             extra_body = self.cfg.get("submission", "responses_extra_body", default=None)
        #             submission_response = self.uploader.post_submission(container_id, model=model, prompt=prompt, extra_body=extra_body)
        #             logger.info("Submission posted successfully")
        #             break
        #         except Exception as e:
        #             logger.error("Submission attempt %d failed: %s", attempt, e)
        #             if attempt < self.retry_attempts:
        #                 time.sleep(self.retry_delay_seconds)
        #             else:
        #                 logger.error("All submission attempts failed")

        # if submission_response:
        #     submission_manifest["submission_response"] = submission_response
        #     # update remote or local manifest with submission response
        #     if manifest_uploaded:
        #         try:
        #             # overwrite remote manifest with updated content
        #             self.remote_manifests.upload_manifest_dict(submission_manifest, blob_name=manifest_blob_name)
        #         except Exception:
        #             pass
        #     else:
        #         # overwrite local
        #         local_path = str(Path(self.manifests_dir) / manifest_blob_name)
        #         atomic_write_json(local_path, submission_manifest)

        # advance watermark using processed successes
        if processed:
            processed_dt_and_names: List[Tuple[datetime, str]] = []
            for ts_iso, name in processed:
                try:
                    dt = parse_iso_to_utc(ts_iso)
                except Exception:
                    dt = datetime.fromtimestamp(0, tz=timezone.utc)
                processed_dt_and_names.append((dt, name))

            max_dt = max(t for t, _ in processed_dt_and_names)
            names_at_max = [n for t, n in processed_dt_and_names if t == max_dt]
            watermark_last_blob = max(names_at_max) if names_at_max else ""
            watermark_iso = max_dt.isoformat()
            self.watermark_store.set_watermark(watermark_iso, watermark_last_blob)
            logger.info("Advanced watermark to %s last_blob=%s", watermark_iso, watermark_last_blob)
        else:
            logger.info("No successful processed blobs; watermark unchanged")

        logger.info("Run finished.")

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Push new blobs using watermark and store submission manifests remotely.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config)
    orchestrator = DataPushOrchestrator(cfg)
    orchestrator.run_once()

if __name__ == "__main__":
    main()
