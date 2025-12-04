#!/usr/bin/env python3
"""
tar_upload_pipeline.py

Simplified pipeline for uploading tar files to OpenAI Partner API.
No home-id filtering, no video duration, no renaming - just upload tar files.

Run:
    python tar_upload_pipeline.py --config config/tar_upload_config.yaml
"""

import os
import sys
import json
import argparse
import uuid
import time
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import quote, urlparse, parse_qs, urlunparse

import requests
import yaml
from dateutil import parser as dtparser

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import generate_blob_sas, generate_container_sas, BlobSasPermissions, ContainerSasPermissions
from azure.storage.filedatalake import DataLakeFileClient
from azure.core.exceptions import ResourceNotFoundError, AzureError
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_handling_to_client.utils import (
    now_utc, now_iso, atomic_write_json, ensure_dir,
    normalize_content_md5, parse_iso_to_utc, read_json_if_exists
)
from data_handling_to_client.state_manager import (
    StateStorageManager, EnhancedStateStore, ProcessingSession,
    VideoFingerprintManager, VideoProcessingResult
)

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

# -------------------------
# Logging
# -------------------------
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)
logging.getLogger("azure.storage.blob").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

from logging.handlers import RotatingFileHandler

def setup_logging_from_cfg(cfg: "ConfigLoader") -> None:
    level_name = (cfg.get("logging", "level", default="INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = cfg.get("logging", "file", default="./logs/tar_upload.log")
    max_mb = int(cfg.get("logging", "max_mb", default=10))
    backup_count = int(cfg.get("logging", "backup_count", default=5))

    ensure_dir(str(Path(log_file).parent))

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = RotatingFileHandler(
        log_file, maxBytes=max_mb * 1024 * 1024, backupCount=backup_count, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    global logger
    logger = logging.getLogger("tar_uploader")


# -------------------------
# Helpers
# -------------------------

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
# RemoteManifestsUploader
# -------------------------
class RemoteManifestsUploader:
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

    def json_metadata_exists(self, json_blob_name: str) -> bool:
        """Check if a JSON metadata file exists in the container"""
        try:
            self._container_client.get_blob_client(json_blob_name).get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except AzureError:
            return False

# -------------------------
# PartnerUploader
# -------------------------
class PartnerUploader:
    def __init__(self, openai_api_key: str, containers_api: str, submissions_api: str, upload_retry: int = 3, upload_retry_delay: int = 5,
                 testing_enabled: bool = False, test_container_id: str = "", test_container_url: str = ""):
        self.openai_api_key = openai_api_key
        self.containers_api = containers_api
        self.submissions_api = submissions_api
        self.upload_retry = upload_retry
        self.upload_retry_delay = upload_retry_delay
        self.testing_enabled = testing_enabled
        self.test_container_id = test_container_id
        self.test_container_url = test_container_url

    def create_partner_container(self, purpose: str = "ingestion", extra_body: Optional[dict] = None) -> dict:
        if self.testing_enabled:
            logger.info("TESTING MODE: Using test container instead of OpenAI API")
            if not self.test_container_url:
                raise ValueError("Testing enabled but test_container_url not configured")
            return {
                "id": self.test_container_id,
                "container_url": self.test_container_url
            }

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
            return container_sas
        safe_blob = quote(blob_name, safe="/")
        return f"{base}/{safe_blob}?{sas}"

    def upload_file_to_partner_using_sas(self, partner_container_sas: str, blob_name: str, local_file_path: str) -> None:
        info = self._parse_sas_info(partner_container_sas)
        logger.info("Partner SAS netloc=%s path=%s sp=%s se=%s sr=%s (masked)", info["netloc"], info["path"], info["sp"], info["se"], info["sr"])
        logger.info("Masked partner SAS: %s", mask_sas(partner_container_sas))

        last_exc = None
        attempt = 0

        if self._is_dfs_dir_sas(partner_container_sas):
            file_url = self._build_file_url_for_dfs_dir_sas(partner_container_sas, blob_name)
            logger.info("Uploading to ADLS Gen2 directory via DataLakeFileClient. Masked dest: %s", mask_sas(file_url))
            while attempt < self.upload_retry:
                attempt += 1
                try:
                    try:
                        client = DataLakeFileClient.from_file_url(file_url)
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
                    if attempt < self.upload_retry:
                        time.sleep(self.upload_retry_delay)
            raise last_exc
        else:
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
                    if attempt < self.upload_retry:
                        time.sleep(self.upload_retry_delay)
            raise last_exc

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
        self.source_container = cfg.get("azure_source", "container_name")
        if not self.source_conn or not self.source_container:
            raise SystemExit("Azure source connection string or container name missing in config or env")

        self.source_sas = cfg.get("azure_source", "container_sas") or os.getenv(cfg.get("azure_source", "container_sas_env", default=""))
        self.source_prefix = cfg.get("azure_source", "source_prefix", default=None)
        self.project_id = cfg.get("azure_source", "project_id", default="tar-upload")

        # partner endpoints
        self.containers_api = cfg.get("openai", "containers_api", default="https://api.openai.com/v1/training_data/containers")
        self.submissions_api = cfg.get("openai", "submissions_api", default="https://api.openai.com/v1/training_data/submissions")
        self.purpose = cfg.get("openai", "purpose", default="ingestion")

        # dedupe safety window
        self.safety_window_minutes = int(cfg.get("deduplication", "safety_window_minutes", default=2))

        # paths
        self.manifests_dir = cfg.get("local_paths", "manifests_dir", default="./local_manifests")
        self.work_dir = cfg.get("local_paths", "work_dir", default="./work")
        self.state_dir = cfg.get("state_tracking", "local_dir", default="./upload_state")
        ensure_dir(self.manifests_dir)
        ensure_dir(self.work_dir)
        ensure_dir(self.state_dir)

        # azcopy config
        self.use_azcopy = bool(cfg.get("azure_partner", "use_azcopy", default=False))
        self.azcopy_path = cfg.get("azure_partner", "azcopy_path", default="azcopy")

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

        state_config = cfg.get("state_storage", default={})
        if state_config.get("enabled", False):
            state_conn = state_config.get("connection_string")
            state_container = state_config.get("container_name")
            state_prefix = state_config.get("prefix", "")
            self.state_storage = StateStorageManager(state_conn, state_container, prefix=state_prefix)
        else:
            self.state_storage = None

        # State store with project_id instead of home_id
        self.state_store = EnhancedStateStore(self.project_id, self.state_dir, state_storage=self.state_storage)
        self._validate_state_configuration()

        # Testing configuration
        test_config = cfg.get("testing", default={})
        self.testing_enabled = bool(test_config.get("enabled", False))
        self.test_container_id = test_config.get("test_container_id", "test_container_123")
        self.test_container_url = test_config.get("test_container_url", "")

        self.source_client = AzureSourceClient(self.source_conn, self.source_container)
        self.uploader = PartnerUploader(self.openai_key, self.containers_api, self.submissions_api,
                                       testing_enabled=self.testing_enabled,
                                       test_container_id=self.test_container_id,
                                       test_container_url=self.test_container_url)

        # SAS generation internals
        self._generated_container_sas: Optional[str] = None
        self._account_name, self._account_key = self._parse_account_from_connstr(self.source_conn)

        # Tar-specific configuration
        self.require_metadata = bool(cfg.get("tar_upload", "require_metadata", default=False))
        self.require_metadata_json = bool(cfg.get("tar_upload", "require_metadata_json", default=True))
        logger.info("Tar upload pipeline initialized (project_id=%s, require_metadata=%s, require_metadata_json=%s)",
                   self.project_id, self.require_metadata, self.require_metadata_json)

    def _validate_state_configuration(self) -> None:
        """Validate state storage configuration"""
        if not hasattr(self, 'state_store'):
            raise SystemExit("State store not initialized. Check state_storage configuration.")

        if self.state_store.state_storage:
            try:
                logger.info("State storage configured and available")
            except Exception as e:
                logger.warning("State storage connectivity test failed: %s", e)
                logger.warning("Will fall back to local-only state storage")

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

    def is_blob_older_than_safety_window(self, last_modified_iso: Optional[str]) -> bool:
        if not last_modified_iso:
            return False
        try:
            lm = parse_iso_to_utc(last_modified_iso)
        except Exception:
            return False
        cutoff = now_utc() - timedelta(minutes=self.safety_window_minutes)
        return lm <= cutoff

    # -------------------------
    # JSON Metadata Validation
    # -------------------------
    def get_json_path_for_tar(self, tar_blob_name: str) -> str:
        """Get the expected JSON file path for a tar file.

        Example:
            one-data-platform/.../test-1gb.tar -> one-data-platform/.../test-1gb.json
        """
        if tar_blob_name.lower().endswith('.tar'):
            return tar_blob_name[:-4] + '.json'
        return tar_blob_name + '.json'

    def has_metadata_json(self, tar_blob_name: str) -> bool:
        """Check if a corresponding metadata JSON file exists for the given tar file.

        Args:
            tar_blob_name: Full blob path to the tar file (e.g., prefix/test-1gb.tar)

        Returns:
            bool: True if corresponding JSON file exists (e.g., prefix/test-1gb.json)
        """
        json_path = self.get_json_path_for_tar(tar_blob_name)
        try:
            exists = self.source_client.json_metadata_exists(json_path)
            if exists:
                logger.debug("Found metadata JSON: %s", json_path)
            else:
                logger.debug("Missing metadata JSON: %s", json_path)
            return exists
        except Exception as e:
            logger.warning("Error checking metadata JSON for %s: %s", tar_blob_name, e)
            return False

    def blobs_to_send(self) -> List[Tuple[str, dict, str]]:
        """Get list of tar files to upload"""
        all_blobs = self.source_client.list_blobs_with_props()
        if self.source_prefix:
            all_blobs = [b for b in all_blobs if b["name"].startswith(self.source_prefix)]

        def is_target(blob_path: str) -> bool:
            """Check if blob is a tar file"""
            parts = blob_path.split("/")
            if len(parts) < 2:
                return False
            filename = parts[-1]
            return filename.lower().endswith(".tar")

        target_blobs = [b for b in all_blobs if is_target(b["name"])]

        # Apply safety window
        safe_blobs = []
        for b in target_blobs:
            if self.is_blob_older_than_safety_window(b.get("last_modified")):
                safe_blobs.append(b)
            else:
                logger.info("SKIP (safety window) %s", b["name"])

        # Validate metadata JSON for each tar file
        if self.require_metadata_json:
            metadata_validated_blobs = []
            for b in safe_blobs:
                if self.has_metadata_json(b["name"]):
                    metadata_validated_blobs.append(b)
                    logger.info("METADATA JSON FOUND for %s", b["name"])
                else:
                    logger.warning("SKIP (no metadata JSON) %s - expected: %s",
                                 b["name"], self.get_json_path_for_tar(b["name"]))
        else:
            # No JSON validation required - accept all tar files
            metadata_validated_blobs = safe_blobs
            for b in metadata_validated_blobs:
                logger.info("ACCEPTED tar file (no JSON validation): %s", b["name"])

        # Convert to format expected by state store
        candidates = []
        for b in metadata_validated_blobs:
            fingerprint = VideoFingerprintManager.create_fingerprint(b)
            candidates.append((b["name"], fingerprint, b.get("last_modified")))

        # Use state store to filter out already processed files
        files_to_process = self.state_store.get_videos_to_process(candidates)

        logger.info("Selected %d tar files for processing (out of %d total)",
                len(files_to_process), len(target_blobs))
        return files_to_process

    def determine_content_type(self, blob_name: str) -> str:
        """Determine content type for blob"""
        ln = blob_name.lower()
        if ln.endswith(".json"):
            return "application/json"
        if ln.endswith(".tar"):
            return "application/x-tar"
        return "application/octet-stream"

    def generate_metadata_obj(self, blob_name: str) -> dict:
        return {"content-type": self.determine_content_type(blob_name), "version": "001"}

    def _is_dfs_dir_sas_url(self, url: str) -> bool:
        p = urlparse(url)
        from urllib.parse import parse_qs
        q = parse_qs(p.query)
        return p.netloc.endswith(".dfs.core.windows.net") and q.get("sr", [""])[0] == "d"

    def _append_path_before_query(self, url: str, extra_path: str, keep_slashes: bool = True) -> str:
        """Append extra_path into the URL's *path* (before ?query)."""
        parsed = urlparse(url)
        safe_extra = quote(extra_path, safe="/" if keep_slashes else "")
        new_path = parsed.path.rstrip("/") + "/" + safe_extra.lstrip("/")
        return urlunparse((parsed.scheme, parsed.netloc, new_path, parsed.params, parsed.query, parsed.fragment))

    def copy_blob_via_azcopy(self, source_url: str, partner_container_sas: str,
                            src_blob_name: str, dest_blob_name: str) -> None:
        # Build source: add *source* blob path into container SAS URL (before ?)
        src = self._append_path_before_query(source_url, src_blob_name, keep_slashes=True)

        # Build destination: add *dest* blob path into ADLS dir SAS or blob container SAS
        dst = self._append_path_before_query(partner_container_sas, dest_blob_name, keep_slashes=True)

        cmd = [self.azcopy_path, "copy", src, dst, "--overwrite=false"]
        logger.info("Running azcopy:\nsrc: %s\ndst: %s", mask_sas(src), mask_sas(dst))
        subprocess.check_call(cmd)

    def get_best_source_url_for_blob(self, blob_name: str) -> Optional[str]:
        if self.source_sas:
            return self.source_sas
        container_sas = self.ensure_container_sas()
        if container_sas:
            return container_sas
        blob_sas = self.generate_blob_sas_url(blob_name, expiry_minutes=60)
        return blob_sas

    def run_once(self) -> None:
        logger.info("Starting tar file upload run.")
        to_send = self.blobs_to_send()
        if not to_send:
            print("No tar files eligible for sending. Exiting.")
            logger.info("No tar files eligible for sending. Exiting.")
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

        # Initialize processing session
        session = ProcessingSession(self.project_id, container_id)

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

        # Pre-generate container SAS if possible
        if not self.source_sas:
            self.ensure_container_sas()

        for blob_name, fingerprint, last_modified in to_send:
            logger.info("Processing tar file: %s", blob_name)
            upload_blob_name = blob_name  # No renaming for tar files
            metadata_blob_name = upload_blob_name + ".metadata.json"
            tmp_blob_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(blob_name).name}")
            tmp_meta_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(metadata_blob_name).name}")

            video_result = VideoProcessingResult(
                blob_name=blob_name,
                upload_blob_name=upload_blob_name,
                fingerprint=fingerprint,
                file_size=fingerprint.get("size"),
                last_modified=last_modified,
                processed_at=now_iso(),
                upload_status="failed",
                container_id=container_id,
                metadata_uploaded=False,
                retry_count=0,
                error_details=None,
                upload_start_time=now_iso(),
                upload_end_time=None,
                upload_duration_seconds=None
            )

            try:
                if self.use_azcopy:
                    source_for_azcopy = self.get_best_source_url_for_blob(blob_name)
                    if source_for_azcopy:
                        try:
                            logger.info("Using azcopy for tar file: %s (source_sas masked: %s)",
                                      blob_name, mask_sas(source_for_azcopy))

                            # Copy the tar file via azcopy
                            self.copy_blob_via_azcopy(source_for_azcopy, partner_container_sas,
                                                    src_blob_name=blob_name,
                                                    dest_blob_name=upload_blob_name)

                            # Generate fresh metadata locally and upload via partner SAS
                            with open(tmp_meta_path, "w", encoding="utf-8") as fm:
                                json.dump(self.generate_metadata_obj(blob_name), fm)
                            self.uploader.upload_file_to_partner_using_sas(
                                partner_container_sas, metadata_blob_name, tmp_meta_path
                            )

                            # Record upload completion timing
                            video_result.upload_end_time = now_iso()
                            try:
                                start_dt = dtparser.parse(video_result.upload_start_time)
                                end_dt = dtparser.parse(video_result.upload_end_time)
                                video_result.upload_duration_seconds = (end_dt - start_dt).total_seconds()
                            except Exception as e:
                                logger.warning("Could not calculate upload duration for %s: %s", blob_name, e)

                            video_result.upload_status = "success"
                            video_result.metadata_uploaded = True
                        except subprocess.CalledProcessError as e:
                            logger.error("azcopy command failed for %s: %s", blob_name, e)
                            video_result.error_details = f"azcopy failed: {str(e)}"
                            raise
                    else:
                        logger.info("No usable source SAS for azcopy; falling back to SDK for %s", blob_name)
                        raise RuntimeError("No source SAS for azcopy")
                else:
                    raise RuntimeError("Azcopy disabled by config")

            except Exception as az_err:
                logger.info("Upload failed for %s: %s", blob_name, az_err)

            finally:
                # Record upload end time for failed uploads
                if video_result.upload_status != "success" and video_result.upload_end_time is None:
                    video_result.upload_end_time = now_iso()
                    try:
                        start_dt = dtparser.parse(video_result.upload_start_time)
                        end_dt = dtparser.parse(video_result.upload_end_time)
                        video_result.upload_duration_seconds = (end_dt - start_dt).total_seconds()
                    except Exception as e:
                        logger.warning("Could not calculate upload duration for failed upload %s: %s", blob_name, e)

                for p in (tmp_blob_path, tmp_meta_path):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass

            session.add_video_result(video_result.to_dict())
            submission_manifest["files"].append({
                "blob_name": blob_name,
                "upload_blob_name": upload_blob_name,
                "fingerprint": fingerprint,
                "metadata_blob": metadata_blob_name,
                "upload_status": video_result.upload_status,
                "uploaded_at": now_iso(),
                "file_type": "tar"
            })

        # persist submission manifest remotely or locally
        session.mark_completed()
        self.state_store.record_processing_session(session.get_session_info())

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

        # summary
        successful_files = [v for v in session.videos if v.get("upload_status") == "success"]
        failed_files = [v for v in session.videos if v.get("upload_status") != "success"]

        print(f"\nTar Upload Session Complete:")
        print(f"Session ID: {session.session_id}")
        print(f"Container ID: {container_id}")
        print(f"Total tar files: {len(session.videos)}")
        print(f"Successful: {len(successful_files)}")
        print(f"Failed: {len(failed_files)}")

        logger.info("Tar upload run finished.")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Upload tar files to OpenAI Partner API")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config)
    setup_logging_from_cfg(cfg)
    orchestrator = DataPushOrchestrator(cfg)
    orchestrator.run_once()

if __name__ == "__main__":
    main()
