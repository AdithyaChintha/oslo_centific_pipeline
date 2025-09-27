#!/usr/bin/env python3
"""
push_new_blobs_with_remote_watermark.py

Same pipeline as before but with watermark reading from remote manifests container first,
and writing watermark.json to remote manifests container (in addition to local file).

Run:
    python push_new_blobs_with_remote_watermark.py --config config.yaml
"""

import os
import sys
import json
import argparse
import uuid
import time
import base64
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import quote, urlparse, parse_qs, urlunparse

import requests
import yaml
from dateutil import parser as dtparser
import re

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import generate_blob_sas, generate_container_sas, BlobSasPermissions, ContainerSasPermissions
from azure.storage.filedatalake import DataLakeFileClient
from azure.core.exceptions import ResourceNotFoundError, AzureError
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse, quote

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

# -------------------------
# Logging
# -------------------------
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger("data_pusher")


from logging.handlers import RotatingFileHandler

def setup_logging_from_cfg(cfg: "ConfigLoader") -> None:
    # Read config
    level_name = (cfg.get("logging", "level", default="INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = cfg.get("logging", "file", default="./logs/data_pusher.log")
    max_mb = int(cfg.get("logging", "max_mb", default=10))
    backup_count = int(cfg.get("logging", "backup_count", default=5))

    # Ensure folder exists
    ensure_dir(str(Path(log_file).parent))

    # Common formatter
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Reset root logger to avoid duplicate handlers when rerun
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    fh = RotatingFileHandler(
        log_file, maxBytes=max_mb * 1024 * 1024, backupCount=backup_count, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Use a named logger in the rest of the code
    global logger
    logger = logging.getLogger("data_pusher")


# -------------------------
# Helpers
# -------------------------
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

def mask_sas(url: str) -> str:
    """Return a masked version of URL showing base and last 8 chars of token for debugging."""
    if not url:
        return "<empty>"
    if "?" not in url:
        return url
    base, q = url.split("?", 1)
    if len(q) <= 8:
        return base + "?***"
    return base + "?***" + q[-8:]



def load_renaming_map_from_ods(mapping_file: str, sheet_name, old_col: str, new_col: str) -> dict:
    """
    Load {old_blob_name -> new_blob_name} from an ODF (.ods) spreadsheet.
    Columns must be old_col and new_col.
    """
    from pathlib import Path
    import pandas as pd

    p = Path(mapping_file)
    if not p.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
    if p.stat().st_size == 0:
        raise ValueError(f"Mapping file is empty: {mapping_file}")

    df = pd.read_excel(mapping_file, sheet_name=sheet_name, engine="odf")
    if isinstance(df, dict):
        if sheet_name:
            if sheet_name not in df:
                raise ValueError(f"Worksheet '{sheet_name}' not found; available: {list(df.keys())}")
            df = df[sheet_name]
        else:
            df = df[next(iter(df))]

    cols = [str(c).strip() for c in df.columns]
    if old_col not in cols or new_col not in cols:
        raise ValueError(f"Mapping must have columns '{old_col}' and '{new_col}'. Found: {cols}")

    mp = {}
    for _, row in df.iterrows():
        k = str(row[old_col]).strip()
        v = str(row[new_col]).strip()
        if k:
            mp[k] = v
    return mp


# -------------------------
# Config loader
# -------------------------
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

# -------------------------
# RemoteManifestsUploader (extended)
# -------------------------
class RemoteManifestsUploader:
    """
    Uploads manifests to a configured Azure container (creates container if missing).
    Also supports downloading a blob as text for watermark retrieval.
    """
    def __init__(self, enabled: bool, connection_string: Optional[str], container_name: Optional[str], prefix: Optional[str] = None):
        self.enabled = bool(enabled)
        self.conn_str = connection_string
        self.container_name = container_name
        self.prefix = (prefix or "").lstrip("/")
        self._client: Optional[ContainerClient] = None
        if self.enabled:
            if not self.conn_str or not self.container_name:
                raise ValueError("remote_manifests enabled but connection string/container_name missing")
            svc = BlobServiceClient.from_connection_string(self.conn_str)
            self._client = svc.get_container_client(self.container_name)
            try:
                self._client.create_container()
            except Exception:
                pass

    def _apply_prefix(self, blob_name: str) -> str:
        if not self.prefix:
            return blob_name
        return f"{self.prefix.rstrip('/')}/{blob_name}"

    def upload_manifest_dict(self, manifest: dict, blob_name: Optional[str] = None) -> str:
        if not self.enabled or not self._client:
            raise RuntimeError("Remote manifests not enabled/configured")
        blob_name = blob_name or f"{now_utc().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex}.json"
        blob_name = self._apply_prefix(blob_name)
        blob_client = self._client.get_blob_client(blob_name)
        data = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        blob_client.upload_blob(data, overwrite=True)
        return blob_name

    def download_blob_as_text(self, blob_name: str) -> Optional[str]:
        """
        Download a blob and return its content as text (str). Returns None on errors/not found.
        """
        if not self.enabled or not self._client:
            return None
        blob_name = self._apply_prefix(blob_name)
        try:
            blob_client = self._client.get_blob_client(blob_name)
            stream = blob_client.download_blob()
            return stream.content_as_text(encoding="utf-8")
        except ResourceNotFoundError:
            logger.debug("Remote watermark blob not found: %s", blob_name)
            return None
        except Exception as e:
            logger.warning("Failed to download remote blob %s: %s", blob_name, e)
            return None

# -------------------------
# Watermark store (reads remote first)
# -------------------------
class WatermarkStore:
    """
    Reads watermark from remote manifests container first (if provided), falls back to local file.
    set_watermark writes both local and remote (if configured).
    """
    def __init__(self, local_path: str, remote_manifests: Optional[RemoteManifestsUploader] = None):
        self.local_path = Path(local_path)
        ensure_dir(str(self.local_path.parent))
        self.remote_manifests = remote_manifests

    def get_watermark(self) -> Tuple[Optional[str], Optional[str]]:
        # Try remote first
        if self.remote_manifests and self.remote_manifests.enabled:
            try:
                txt = self.remote_manifests.download_blob_as_text("watermark.json")
                if txt:
                    obj = json.loads(txt)
                    return obj.get("watermark_ts"), obj.get("watermark_last_blob")
            except Exception as e:
                logger.warning("Could not read watermark.json from remote manifests: %s", e)
        # Fallback local
        try:
            if self.local_path.exists():
                obj = read_json_if_exists(str(self.local_path))
                return obj.get("watermark_ts"), obj.get("watermark_last_blob")
        except Exception as e:
            logger.warning("Could not read local watermark.json: %s", e)
        return None, None

    def set_watermark(self, watermark_ts_iso: str, watermark_last_blob: Optional[str]) -> None:
        obj = {"watermark_ts": watermark_ts_iso, "watermark_last_blob": watermark_last_blob or ""}
        # write local
        atomic_write_json(str(self.local_path), obj)
        # write remote (best-effort)
        if self.remote_manifests and self.remote_manifests.enabled:
            try:
                # overwrite remote watermark.json
                self.remote_manifests.upload_manifest_dict(obj, blob_name="watermark.json")
                logger.info("Uploaded watermark.json to remote manifests container.")
            except Exception as e:
                logger.warning("Failed to upload watermark.json to remote manifests: %s", e)

# -------------------------
# Azure source client
# -------------------------
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

# -------------------------
# PartnerUploader (supports ADLS dir SAS and blob SAS)
# -------------------------
class PartnerUploader:
    def __init__(self, openai_api_key: str, containers_api: str, submissions_api: str, upload_retry: int = 3, upload_retry_delay: int = 5):
        self.openai_api_key = openai_api_key
        self.containers_api = containers_api
        self.submissions_api = submissions_api
        self.upload_retry = upload_retry
        self.upload_retry_delay = upload_retry_delay

    def create_partner_container(self, purpose: str = "ingestion", extra_body: Optional[dict] = None) -> dict:
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        body = {"purpose": purpose}
        if extra_body:
            body.update(extra_body)
        r = requests.post(self.containers_api, json=body, headers=headers)
        if not r.ok:
            logger.error("Partner create container failed: status=%s body=%s", r.status_code, r.text)
            r.raise_for_status()
        return r.json()

    def post_submission(self, container_id: str) -> dict:
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        body = {"container_id": container_id}
        r = requests.post(self.submissions_api, json=body, headers=headers)
        if not r.ok:
            logger.error("Partner post submission failed: status=%s body=%s", r.status_code, r.text)
            r.raise_for_status()
        return r.json()

    # ---------- helpers ----------
    def _parse_sas_info(self, url: str) -> dict:
        parsed = urlparse(url)
        q = parse_qs(parsed.query)
        return {"netloc": parsed.netloc, "path": parsed.path, "sp": q.get("sp", [None])[0], "se": q.get("se", [None])[0], "sr": q.get("sr", [None])[0]}

    def _is_dfs_dir_sas(self, url: str) -> bool:
        info = self._parse_sas_info(url)
        return info["netloc"].endswith(".dfs.core.windows.net") and info["sr"] == "d"

    def _build_file_url_for_dfs_dir_sas(self, dir_sas: str, blob_name: str) -> str:
        if "?" not in dir_sas:
            raise ValueError("dir_sas missing SAS query")
        base, sas = dir_sas.split("?", 1)
        base = base.rstrip("/")
        parts = blob_name.split("/")
        encoded_parts = [quote(p, safe="") for p in parts]
        appended = "/".join(encoded_parts)
        file_url = f"{base}/{appended}?{sas}"
        return file_url

    def _build_dest_url_for_blob_sas(self, container_sas: str, blob_name: str) -> str:
        if "?" not in container_sas:
            raise ValueError("container_sas missing SAS query")
        base, sas = container_sas.split("?", 1)
        base = base.rstrip("/")
        parsed = urlparse(base)
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) > 1:
            # base already has a blob path -> return as is
            return container_sas
        safe_blob = quote(blob_name, safe="/")
        return f"{base}/{safe_blob}?{sas}"

    # ---------- main upload ----------
    def upload_file_to_partner_using_sas(self, partner_container_sas: str, blob_name: str, local_file_path: str) -> None:
        info = self._parse_sas_info(partner_container_sas)
        logger.info("Partner SAS netloc=%s path=%s sp=%s se=%s sr=%s (masked)", info["netloc"], info["path"], info["sp"], info["se"], info["sr"])
        logger.info("Masked partner SAS: %s", mask_sas(partner_container_sas))

        last_exc = None
        attempt = 0

        if self._is_dfs_dir_sas(partner_container_sas):
            # ADLS Gen2 directory SAS -> use DataLakeFileClient
            file_url = self._build_file_url_for_dfs_dir_sas(partner_container_sas, blob_name)
            logger.info("Uploading to ADLS Gen2 directory via DataLakeFileClient. Masked dest: %s", mask_sas(file_url))
            while attempt < self.upload_retry:
                attempt += 1
                try:
                    # older SDKs may not have from_file_url; construct manually
                    try:
                        client = DataLakeFileClient.from_file_url(file_url)  # type: ignore[attr-defined]
                    except AttributeError:
                        parsed = urlparse(file_url)
                        path_parts = [p for p in parsed.path.split("/") if p]
                        if len(path_parts) < 2:
                            raise RuntimeError(f"Cannot parse filesystem/file path from ADLS URL: {file_url}")
                        filesystem = path_parts[0]
                        file_path = "/".join(path_parts[1:])
                        sas_token = parsed.query
                        account_url = f"{parsed.scheme}://{parsed.netloc}"
                        client = DataLakeFileClient(account_url, filesystem, file_path, credential=sas_token)
                    with open(local_file_path, "rb") as f:
                        data = f.read()
                    client.upload_data(data, overwrite=True)
                    return
                except Exception as e:
                    last_exc = e
                    txt = str(e)
                    if "AuthenticationFailed" in txt or "Signature" in txt or "403" in txt:
                        logger.error("AuthenticationFailed (ADLS) while uploading %s. Masked dest: %s", blob_name, mask_sas(file_url))
                        logger.error("Common causes: SAS missing 'w' permission, SAS expired, or URL was malformed.")
                        logger.debug("Full ADLS upload exception: %s", txt)
                    else:
                        logger.warning("Attempt %d failed for ADLS upload %s: %s", attempt, blob_name, txt)
                    if attempt < self.upload_retry:
                        time.sleep(self.upload_retry_delay)
            raise last_exc

        else:
            # Blob endpoint
            dest_url = self._build_dest_url_for_blob_sas(partner_container_sas, blob_name)
            logger.info("Uploading to Blob endpoint. Masked dest: %s", mask_sas(dest_url))
            while attempt < self.upload_retry:
                attempt += 1
                try:
                    dest_blob = BlobClient.from_blob_url(dest_url)
                    with open(local_file_path, "rb") as f:
                        dest_blob.upload_blob(f, overwrite=True)
                    return
                except Exception as e:
                    last_exc = e
                    txt = str(e)
                    if "AuthenticationFailed" in txt or "Signature" in txt or "403" in txt:
                        logger.error("AuthenticationFailed (Blob) while uploading %s. Masked dest: %s", blob_name, mask_sas(dest_url))
                        logger.error("Common causes: SAS missing 'w' permission, SAS expired, or URL was malformed.")
                        logger.debug("Full Blob upload exception: %s", txt)
                    else:
                        logger.warning("Attempt %d failed for Blob upload %s: %s", attempt, blob_name, txt)
                    if attempt < self.upload_retry:
                        time.sleep(self.upload_retry_delay)
            raise last_exc

    # ---------- small test helper ----------
    def test_upload_small_file(self, partner_container_sas: str) -> Tuple[bool, str]:
        tmp = Path("/tmp") / f"az_test_{uuid.uuid4().hex}.bin"
        try:
            tmp.write_bytes(b"ok")
            test_blob = f"__test__{uuid.uuid4().hex}.bin"
            try:
                self.upload_file_to_partner_using_sas(partner_container_sas, test_blob, str(tmp))
                return True, f"test upload succeeded -> {test_blob}"
            finally:
                try:
                    tmp.unlink()
                except Exception:
                    pass
        except Exception as e:
            return False, str(e)

# -------------------------
# Orchestrator
# -------------------------
class DataPushOrchestrator:
    def __init__(self, cfg: ConfigLoader):
        self.cfg = cfg

        # OpenAI API key
        self.openai_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_key:
            raise SystemExit("OpenAI API key not found in config or environment")

        # Azure source
        self.source_conn = cfg.get("azure_source", "connection_string", default="AZURE_SOURCE_CONN")
        # self.source_conn = os.getenv(self.source_conn_env)
        self.source_container = cfg.get("azure_source", "container_name")
        if not self.source_conn or not self.source_container:
            raise SystemExit("Azure source connection string or container name missing in config or env")

        # source SAS
        self.source_sas = cfg.get("azure_source", "container_sas") or os.getenv(cfg.get("azure_source", "container_sas_env", default=""))
        self.source_prefix = cfg.get("azure_source", "source_prefix", default=None)
        self.home_id = cfg.get("azure_source", "home_id")


        # partner endpoints
        self.containers_api = cfg.get("openai", "containers_api", default="https://api.openai.com/v1/training_data/containers")
        self.submissions_api = cfg.get("openai", "submissions_api", default="https://api.openai.com/v1/training_data/submissions")
        self.purpose = cfg.get("openai", "purpose", default="ingestion")

        # dedupe safety window
        self.safety_window_minutes = int(cfg.get("deduplication", "safety_window_minutes", default=2))

        # paths
        self.manifests_dir = cfg.get("local_paths", "manifests_dir", default="./local_manifests")
        self.watermark_file = cfg.get("local_paths", "watermark_file", default=str(Path(self.manifests_dir) / "watermark.json"))
        self.work_dir = cfg.get("local_paths", "work_dir", default="./work")
        ensure_dir(self.manifests_dir)
        ensure_dir(self.work_dir)

        # azcopy config
        self.use_azcopy = bool(cfg.get("azure_partner", "use_azcopy", default=False))
        self.azcopy_path = cfg.get("azure_partner", "azcopy_path", default="azcopy")

        # submission behavior
        self.auto_submit = bool(cfg.get("submission", "auto_submit", default=True))
        self.retry_attempts = int(cfg.get("submission", "retry_attempts", default=3))
        self.retry_delay_seconds = int(cfg.get("submission", "retry_delay_seconds", default=60))

        # remote manifests config - create this BEFORE WatermarkStore so store can read remote watermark
        rconf = cfg.get("remote_manifests", default={})
        self.remote_manifests_enabled = bool(rconf.get("enabled", False))
        if self.remote_manifests_enabled:
            rm_conn = rconf.get("connection_string")
            rm_container = rconf.get("container_name")
            rm_prefix = rconf.get("prefix", "")
            self.remote_manifests = RemoteManifestsUploader(True, rm_conn, rm_container, prefix=rm_prefix)
        else:
            self.remote_manifests = RemoteManifestsUploader(False, None, None)

        # helpers (watermark store uses remote_manifests)
        self.watermark_store = WatermarkStore(self.watermark_file, remote_manifests=self.remote_manifests)
        self.source_client = AzureSourceClient(self.source_conn, self.source_container)
        self.uploader = PartnerUploader(self.openai_key, self.containers_api, self.submissions_api)

        # SAS generation internals
        self._generated_container_sas: Optional[str] = None
        self._account_name, self._account_key = self._parse_account_from_connstr(self.source_conn)



        # ---- renaming config (from scratch) ----
        rnm = self.cfg.get("renaming", default={}) or {}
        self.renaming_enabled = bool(rnm.get("enabled", False))
        self.renaming_map: dict = {}
        self.renaming_patterns: list[tuple[re.Pattern, str]] = []  # (compiled_regex, replacement_template)
        self.rename_match_mode = (rnm.get("match_mode") or "auto").lower()

        if self.renaming_enabled:
            mapping_path = rnm.get("mapping_file")
            if not mapping_path:
                raise SystemExit("renaming.enabled is true but renaming.mapping_file is missing")

            # resolve relative to YAML file directory (if you have ConfigLoader.base_dir)
            try:
                base_dir = self.cfg.base_dir  # if you added earlier; otherwise use os.getcwd()
                mapping_path = str(Path(base_dir) / mapping_path) if not Path(mapping_path).is_absolute() else mapping_path
            except Exception:
                pass

            sheet_name = rnm.get("sheet_name")
            old_col = rnm.get("old_column", "old_blob_name")
            new_col = rnm.get("new_column", "new_blob_name")

            # ODF loader
            self.renaming_map = load_renaming_map_from_ods(mapping_path, sheet_name, old_col, new_col)
            logger.info("Loaded %d rename rules from %s", len(self.renaming_map), mapping_path)

            # Build regex patterns for placeholder keys (AUDIO_ID / VIDEO_ID), case-insensitive
            for k, v in self.renaming_map.items():
                if ("AUDIO_ID" in k) or ("VIDEO_ID" in k):
                    pat = re.escape(k)
                    # match audio<number> / video<number>, keep case-insensitive
                    pat = pat.replace("AUDIO_ID", r"(audio\d+)")
                    pat = pat.replace("VIDEO_ID", r"(video\d+)")
                    self.renaming_patterns.append((re.compile(r"^" + pat + r"$", re.IGNORECASE), v))


    # -------------------------
    # Account parsing + SAS generation
    # -------------------------
    def _parse_account_from_connstr(self, conn_str: str) -> Tuple[Optional[str], Optional[str]]:
        if not conn_str:
            return None, None
        parts = {}
        for part in conn_str.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                parts[k] = v
        return parts.get("AccountName"), parts.get("AccountKey")

    def generate_container_sas_url(self, expiry_minutes: int = 60) -> Optional[str]:
        if not self._account_name or not self._account_key:
            return None
        try:
            sas_token = generate_container_sas(
                account_name=self._account_name,
                container_name=self.source_container,
                account_key=self._account_key,
                permission=ContainerSasPermissions(read=True, list=True),
                expiry=datetime.utcnow() + timedelta(minutes=expiry_minutes)
            )
            url = f"https://{self._account_name}.blob.core.windows.net/{self.source_container}?{sas_token}"
            logger.debug("Generated container SAS (masked): %s", mask_sas(url))
            return url
        except Exception as e:
            logger.warning("Failed to generate container SAS: %s", e)
            return None

    def generate_blob_sas_url(self, blob_name: str, expiry_minutes: int = 60) -> Optional[str]:
        if not self._account_name or not self._account_key:
            return None
        try:
            sas_token = generate_blob_sas(
                account_name=self._account_name,
                container_name=self.source_container,
                blob_name=blob_name,
                account_key=self._account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(minutes=expiry_minutes)
            )
            url = f"https://{self._account_name}.blob.core.windows.net/{self.source_container}/{quote(blob_name, safe='')}" + f"?{sas_token}"
            logger.debug("Generated blob SAS (masked): %s", mask_sas(url))
            return url
        except Exception as e:
            logger.warning("Failed to generate blob SAS for %s: %s", blob_name, e)
            return None

    def ensure_container_sas(self) -> Optional[str]:
        if self.source_sas:
            return self.source_sas
        if self._generated_container_sas:
            return self._generated_container_sas
        sas = self.generate_container_sas_url(expiry_minutes=60)
        if sas:
            self._generated_container_sas = sas
        return sas

    # -------------------------
    # Watermark & selection
    # -------------------------
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
        
        def is_target(blob_path: str) -> bool:
            parts = blob_path.split("/")
            if len(parts) < 2:
                return False
            filename = parts[-1]
            parent_folders = parts[:-1]
            has_home = any(p.startswith(self.home_id) for p in parent_folders)
            is_media = filename.lower().endswith((".insv", ".wav"))
            return has_home and is_media
        
        all_blobs = [b for b in all_blobs if is_target(b["name"])]

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

        candidates.sort(key=lambda t: (parse_iso_to_utc(t[2]), t[0]))
        print(candidates)
        logger.info("Selected %d candidate blobs:\n%s", len(candidates), candidates)
        return candidates

    # -------------------------
    # metadata helpers
    # -------------------------
    def determine_content_type(self, blob_name: str) -> str:
        ln = blob_name.lower()
        if ln.endswith(".json"):
            return "application/json"
        if ln.endswith(".wav"):
            return "audio/wav"
        if ln.endswith(".insv"):
            return "application/octet-stream"
        return "application/octet-stream"

    def generate_metadata_obj(self, blob_name: str) -> dict:
        return {"content-type": self.determine_content_type(blob_name), "version": "001"}

    # -------------------------
    # azcopy wrapper
    # -------------------------
    def _is_dfs_dir_sas_url(self, url: str) -> bool:
        parsed = urlparse(url)
        q = parse_qs(parsed.query)
        return parsed.netloc.endswith(".dfs.core.windows.net") and q.get("sr", [""])[0] == "d"

    # def copy_blob_via_azcopy(self, source_url: str, partner_container_sas: str, blob_name: str) -> None:
    #     # src = source_url
    #     # if "?" in source_url and (f"/{self.source_container}/" in source_url or source_url.rstrip().endswith(self.source_container) or source_url.rstrip().endswith(self.source_container + "?") or source_url.rstrip().endswith(self.source_container + "/")):
    #     #     src = source_url.rstrip("/") + "/" + quote(blob_name, safe="")
    #     # dst = partner_container_sas.rstrip("/") + "/" + quote(blob_name, safe="")

    #     src = source_url.rstrip("/") + "/" + quote(blob_name, safe="/")
    #     dst = partner_container_sas
    #     if self._is_dfs_dir_sas_url(partner_container_sas):
    #         # ADLS Gen2 directory SAS -> append path with real slashes
    #         dst = partner_container_sas.rstrip("/") + "/" + quote(blob_name, safe="/")
    #     else:
    #         # Blob container SAS -> also append with slashes
    #         dst = partner_container_sas.rstrip("/") + "/" + quote(blob_name, safe="/")

    #     # cmd = [self.azcopy_path, "copy", src, dst, "--overwrite=false"]
    #     # logger.info("Running azcopy: %s", " ".join(cmd))
    #     # subprocess.check_call(cmd)

    #     cmd = [self.azcopy_path, "copy", src, dst, "--overwrite=false"]
    #     logger.info("Running azcopy:\nsrc: %s\ndst: %s", mask_sas(src), mask_sas(dst))
    #     subprocess.check_call(cmd)



    

    def _append_path_before_query(self,url: str, extra_path: str, keep_slashes: bool = True) -> str:
        """
        Append extra_path into the URL's *path* (before ?query).
        keep_slashes=True keeps '/' as real slashes (quote(..., safe='/')).
        """
        parsed = urlparse(url)
        safe_extra = quote(extra_path, safe="/" if keep_slashes else "")
        new_path = parsed.path.rstrip("/") + "/" + safe_extra.lstrip("/")
        return urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))

    def _is_dfs_dir_sas_url(self,url: str) -> bool:
        p = urlparse(url)
        from urllib.parse import parse_qs
        q = parse_qs(p.query)
        return p.netloc.endswith(".dfs.core.windows.net") and q.get("sr", [""])[0] == "d"

    def copy_blob_via_azcopy(self, source_url: str, partner_container_sas: str,
                            src_blob_name: str, dest_blob_name: str) -> None:
        # Build source: add *source* blob path into container SAS URL (before ?)
        src = self._append_path_before_query(source_url, src_blob_name, keep_slashes=True)

        # Build destination: add *dest* blob path into ADLS dir SAS or blob container SAS
        dst = self._append_path_before_query(partner_container_sas, dest_blob_name, keep_slashes=True)

        cmd = [self.azcopy_path, "copy", src, dst, "--overwrite=false"]
        logger.info("Running azcopy:\nsrc: %s\ndst: %s", mask_sas(src), mask_sas(dst))
        subprocess.check_call(cmd)


    # -------------------------
    # choose best source URL for azcopy
    # -------------------------
    def get_best_source_url_for_blob(self, blob_name: str) -> Optional[str]:
        if self.source_sas:
            return self.source_sas
        container_sas = self.ensure_container_sas()
        if container_sas:
            return container_sas
        blob_sas = self.generate_blob_sas_url(blob_name, expiry_minutes=60)
        return blob_sas

    # def resolve_upload_name(self, source_blob_name: str) -> str:
    #     """
    #     Decide the upload (destination) name:
    #     1) exact match on full path
    #     2) exact match on basename
    #     3) regex placeholders (AUDIO_ID/VIDEO_ID -> audio\d+/video\d+)
    #     If the target template contains the placeholder, we substitute it with the captured text.
    #     """
    #     if not self.renaming_enabled or not self.renaming_map:
    #         return source_blob_name

    #     full_key = source_blob_name
    #     base_key = Path(source_blob_name).name

    #     # Exact full-path match
    #     if full_key in self.renaming_map:
    #         return self.renaming_map[full_key]

    #     # Exact basename match
    #     if base_key in self.renaming_map:
    #         return self.renaming_map[base_key]

    #     # Regex placeholder match
    #     for pattern, template in self.renaming_patterns:
    #         m = pattern.match(full_key) or pattern.match(base_key)
    #         if m:
    #             replacement = template
    #             # If template includes placeholders, replace with captured token(s)
    #             # (We used a single capture group in the pattern)
    #             if "AUDIO_ID" in replacement and m.lastindex:
    #                 replacement = replacement.replace("AUDIO_ID", m.group(1))
    #             if "VIDEO_ID" in replacement and m.lastindex:
    #                 replacement = replacement.replace("VIDEO_ID", m.group(1))
    #             return replacement

    #     # No rule -> keep original
    #     return source_blob_name


    def resolve_upload_name(self, source_blob_name: str) -> str:
        if not self.renaming_enabled or not self.renaming_map:
            return source_blob_name

        full_key = source_blob_name
        base_key = Path(source_blob_name).name
        parent_dir = str(Path(source_blob_name).parent)  # e.g. "multi_chunk/raw_insv"

        # 1. Full-path exact match
        if full_key in self.renaming_map:
            return self.renaming_map[full_key]

        # 2. Basename exact match
        if base_key in self.renaming_map:
            return f"{parent_dir}/{self.renaming_map[base_key]}"

        # 3. Regex placeholder match
        for pattern, template in self.renaming_patterns:
            m = pattern.match(full_key) or pattern.match(base_key)
            if m:
                replacement = template
                if "AUDIO_ID" in replacement and m.lastindex:
                    replacement = replacement.replace("AUDIO_ID", m.group(1))
                if "VIDEO_ID" in replacement and m.lastindex:
                    replacement = replacement.replace("VIDEO_ID", m.group(1))
                # preserve the original folder structure
                return f"{parent_dir}/{replacement}"

        # no match → keep original
        return source_blob_name



    # -------------------------
    # run loop
    # -------------------------
    def run_once(self) -> None:
        logger.info("Starting data push run (with remote watermark).")
        to_send = self.blobs_to_send()
        if not to_send:
            print("No blobs eligible for sending. Exiting.")
            logger.info("No blobs eligible for sending. Exiting.")
            return

        # create partner container
        logger.info("Creating partner container via partner API")
        create_resp = self.uploader.create_partner_container(purpose=self.purpose)
        container_id = create_resp.get("id") or create_resp.get("container_id") or create_resp.get("container")
        partner_container_sas = (
            create_resp.get("container_url")
            or create_resp.get("url")
            or create_resp.get("container_sas")
            or create_resp.get("sas_url")
        )
        if not container_id or not partner_container_sas:
            raise RuntimeError(f"Unexpected partner create response: {create_resp}")
        logger.info("Partner container created. Masked SAS: %s", mask_sas(partner_container_sas))

        # optional quick test
        try:
            ok, msg = self.uploader.test_upload_small_file(partner_container_sas)
            logger.info("Partner SAS quick test: %s - %s", ok, msg)
            if not ok:
                logger.error("Partner SAS quick test failed; aborting run.")
                return
        except Exception as e:
            logger.warning("Partner SAS quick test raised: %s", e)

        submission_manifest = {
            "container_id": container_id,
            "create_response": create_resp,
            "created_at": now_iso(),
            "files": []
        }
        processed: List[Tuple[str, str]] = []

        # Pre-generate container SAS if possible
        if not self.source_sas:
            self.ensure_container_sas()

        for blob_name, fp, last_mod_iso in to_send:
            logger.info("Processing: %s", blob_name)
            # metadata_blob_name = blob_name + ".metadata.json"
            # tmp_blob_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(blob_name).name}")
            # tmp_meta_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(metadata_blob_name).name}")
            # upload_status = "failed: unknown"

            # try:
            #     if self.use_azcopy:
            #         source_for_azcopy = self.get_best_source_url_for_blob(blob_name)
            #         if source_for_azcopy:
            #             try:
            #                 logger.info("Using azcopy for blob: %s (source_sas masked: %s)", blob_name, mask_sas(source_for_azcopy))
            #                 self.copy_blob_via_azcopy(source_for_azcopy, partner_container_sas, blob_name)
            #                 try:
            #                     if self.source_client.metadata_blob_exists(metadata_blob_name):
            #                         self.copy_blob_via_azcopy(source_for_azcopy, partner_container_sas, metadata_blob_name)
            #                     else:
            #                         with open(tmp_meta_path, "w", encoding="utf-8") as fm:
            #                             json.dump(self.generate_metadata_obj(blob_name), fm)
            #                         self.uploader.upload_file_to_partner_using_sas(partner_container_sas, metadata_blob_name, tmp_meta_path)
            #                 except Exception as e_meta:
            #                     logger.warning("Metadata copy/upload warning for %s: %s", metadata_blob_name, e_meta)
            #                 upload_status = "success"
            #             except subprocess.CalledProcessError as e:
            #                 logger.error("azcopy command failed for %s: %s", blob_name, e)
            #                 raise
            #         else:
            #             logger.info("No usable source SAS for azcopy; falling back to SDK for %s", blob_name)
            #             raise RuntimeError("No source SAS for azcopy")
            #     else:
            #         raise RuntimeError("Azcopy disabled by config")

            # except Exception as az_err:
            #     logger.info("Falling back to SDK download/upload for %s due to: %s", blob_name, az_err)
            #     try:
            #         self.source_client.download_blob_to_path(blob_name, tmp_blob_path)

            #         if self.source_client.metadata_blob_exists(metadata_blob_name):
            #             try:
            #                 self.source_client.download_metadata_to_path(metadata_blob_name, tmp_meta_path)
            #             except Exception as e:
            #                 logger.warning("Failed to download metadata %s: %s - generating instead", metadata_blob_name, e)
            #                 with open(tmp_meta_path, "w", encoding="utf-8") as fm:
            #                     json.dump(self.generate_metadata_obj(blob_name), fm)
            #         else:
            #             with open(tmp_meta_path, "w", encoding="utf-8") as fm:
            #                 json.dump(self.generate_metadata_obj(blob_name), fm)

            #         try:
            #             self.uploader.upload_file_to_partner_using_sas(partner_container_sas, blob_name, tmp_blob_path)
            #             self.uploader.upload_file_to_partner_using_sas(partner_container_sas, metadata_blob_name, tmp_meta_path)
            #             upload_status = "success"
            #         except Exception as upl_ex:
            #             logger.error("Upload via partner SAS failed for %s: %s", blob_name, upl_ex)
            #             upload_status = f"failed: {str(upl_ex)}"
            #    except Exception as dl_ex:
            #        logger.error("Download failed for %s: %s", blob_name, dl_ex, exc_info=True)
            #        upload_status = f"failed: {str(dl_ex)}"


            # metadata_blob_name = blob_name + ".metadata.json"
            upload_blob_name = self.resolve_upload_name(blob_name)   # NEW
            metadata_blob_name = upload_blob_name + ".metadata.json"
            tmp_blob_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(blob_name).name}")
            tmp_meta_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(metadata_blob_name).name}")
            upload_status = "failed: unknown"

            try:
                if self.use_azcopy:
                    source_for_azcopy = self.get_best_source_url_for_blob(blob_name)
                    if source_for_azcopy:
                        try:
                            logger.info("Using azcopy for blob: %s (source_sas masked: %s)", blob_name, mask_sas(source_for_azcopy))
                            # 1) Copy the actual media blob via azcopy
                            # self.copy_blob_via_azcopy(source_for_azcopy, partner_container_sas, blob_name)

                            self.copy_blob_via_azcopy(source_for_azcopy, partner_container_sas,
                                                    src_blob_name=blob_name,
                                                    dest_blob_name=upload_blob_name)


                            # 2) ALWAYS generate fresh metadata locally and upload via partner SAS
                            with open(tmp_meta_path, "w", encoding="utf-8") as fm:
                                json.dump(self.generate_metadata_obj(blob_name), fm)
                            self.uploader.upload_file_to_partner_using_sas(
                                partner_container_sas, metadata_blob_name, tmp_meta_path
                            )

                            upload_status = "success"
                        except subprocess.CalledProcessError as e:
                            logger.error("azcopy command failed for %s: %s", blob_name, e)
                            raise
                    else:
                        logger.info("No usable source SAS for azcopy; falling back to SDK for %s", blob_name)
                        raise RuntimeError("No source SAS for azcopy")
                else:
                    raise RuntimeError("Azcopy disabled by config")

            except Exception as az_err:
                logger.info("Falling back to SDK download/upload for %s due to: %s", blob_name, az_err)
                # try:
                #     # 1) Download the media blob via SDK
                #     self.source_client.download_blob_to_path(blob_name, tmp_blob_path)

                #     # 2) ALWAYS generate fresh metadata locally
                #     with open(tmp_meta_path, "w", encoding="utf-8") as fm:
                #         json.dump(self.generate_metadata_obj(blob_name), fm)

                #     # 3) Upload both via partner SAS
                #     self.uploader.upload_file_to_partner_using_sas(partner_container_sas, blob_name, tmp_blob_path)
                #     self.uploader.upload_file_to_partner_using_sas(partner_container_sas, metadata_blob_name, tmp_meta_path)

                #     upload_status = "success"
                # except Exception as upl_ex:
                #     logger.error("SDK path failed for %s: %s", blob_name, upl_ex)
                #     upload_status = f"failed: {str(upl_ex)}"




            finally:
                for p in (tmp_blob_path, tmp_meta_path):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass

            submission_manifest["files"].append({
                "blob_name": blob_name,
                "upload_blob_name": upload_blob_name,
                "fingerprint": fp,
                "metadata_blob": metadata_blob_name,
                "upload_status": upload_status,
                "uploaded_at": now_iso()
            })
            if upload_status == "success":
                processed.append((last_mod_iso, blob_name))

        # persist submission manifest remotely or locally
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

        # post submission
        submission_response = None
        if self.auto_submit:
            attempt = 0
            while attempt < self.retry_attempts:
                attempt += 1
                try:
                    logger.info("Posting submission to partner (attempt %d)", attempt)
                    submission_response = self.uploader.post_submission(container_id)
                    logger.info("Submission posted successfully")
                    break
                except Exception as e:
                    logger.error("Submission attempt %d failed: %s", attempt, e)
                    if attempt < self.retry_attempts:
                        time.sleep(self.retry_delay_seconds)
                    else:
                        logger.error("All submission attempts failed")

        if submission_response:
            submission_manifest["submission_response"] = submission_response
            if manifest_uploaded:
                try:
                    self.remote_manifests.upload_manifest_dict(submission_manifest, blob_name=manifest_blob_name)
                except Exception:
                    pass
            else:
                atomic_write_json(str(Path(self.manifests_dir) / manifest_blob_name), submission_manifest)

        # advance watermark based on successes
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
            # write both local and remote inside WatermarkStore.set_watermark
            self.watermark_store.set_watermark(watermark_iso, watermark_last_blob)
            logger.info("Advanced watermark to %s last_blob=%s", watermark_iso, watermark_last_blob)
        else:
            logger.info("No blobs processed successfully; watermark unchanged")

        # summary
        total = len(to_send)
        success_files = [f["blob_name"] for f in submission_manifest["files"] if f["upload_status"] == "success"]
        failed_files = [f["blob_name"] for f in submission_manifest["files"] if f["upload_status"] != "success"]

        print("\n" + "=" * 72)
        print("SUMMARY")
        print(f"Total blobs considered: {total}")
        print(f"Successfully uploaded: {len(success_files)}")
        for b in success_files:
            print(f"  ✓ {b}")
        print(f"Failed uploads: {len(failed_files)}")
        for b in failed_files:
            entry = next((x for x in submission_manifest["files"] if x["blob_name"] == b), None)
            reason = entry["upload_status"] if entry else ""
            print(f"  ✖ {b} -> {reason}")
        wm_ts, wm_blob = self.watermark_store.get_watermark()
        print(f"Advanced watermark: {wm_ts} , last_blob: {wm_blob}")
        print("=" * 72 + "\n")

        logger.info("Run finished.")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Push new blobs (remote watermark) with SAS & azcopy support.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config)
    setup_logging_from_cfg(cfg)
    orchestrator = DataPushOrchestrator(cfg)
    orchestrator.run_once()

if __name__ == "__main__":
    main()
