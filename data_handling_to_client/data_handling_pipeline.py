#!/usr/bin/env python3
"""
push_new_blobs_final.py

Complete updated pipeline:
 - watermark + tie-break selection
 - container- and blob-level SAS generation (from connection string account key)
 - azcopy server-to-server copy when SAS available
 - robust partner uploads with URL-encoding, masking, retries, and AuthenticationFailed diagnostics
 - remote manifest upload (optional), local fallback
 - summary printed at end (success/failed lists)

Requirements:
    pip install azure-storage-blob requests pyyaml python-dateutil
    azcopy installed if using azcopy mode

Run:
    python push_new_blobs_final.py --config config.yaml
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
from urllib.parse import quote

import requests
import yaml
from dateutil import parser as dtparser
from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    generate_blob_sas,
    generate_container_sas,
    BlobSasPermissions,
    ContainerSasPermissions,
)
from azure.core.exceptions import ResourceNotFoundError, AzureError
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
# Watermark store
# -------------------------
class WatermarkStore:
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
# Partner uploader
# -------------------------
# class PartnerUploader:
#     def __init__(self, openai_api_key: str, containers_api: str, submissions_api: str, upload_retry: int = 3, upload_retry_delay: int = 5):
#         self.openai_api_key = openai_api_key
#         self.containers_api = containers_api
#         self.submissions_api = submissions_api
#         self.upload_retry = upload_retry
#         self.upload_retry_delay = upload_retry_delay

#     def create_partner_container(self, purpose: str = "ingestion", extra_body: Optional[dict] = None) -> dict:
#         headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
#         body = {"purpose": purpose}
#         if extra_body:
#             body.update(extra_body)
#         r = requests.post(self.containers_api, json=body, headers=headers)
#         if not r.ok:
#             logger.error("Partner create container failed: status=%s body=%s", r.status_code, r.text)
#             r.raise_for_status()
#         return r.json()

#     def post_submission(self, container_id: str) -> dict:
#         headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
#         body = {"container_id": container_id}
#         r = requests.post(self.submissions_api, json=body, headers=headers)
#         if not r.ok:
#             logger.error("Partner post submission failed: status=%s body=%s", r.status_code, r.text)
#             r.raise_for_status()
#         return r.json()

#     def upload_file_to_partner_using_sas(self, partner_container_sas: str, blob_name: str, local_file_path: str) -> None:
#         """
#         Upload local file to partner container using partner container SAS.
#         Uses retries and URL-encoding; handles AuthenticationFailed logging hints.
#         """
#         safe_blob = quote(blob_name, safe="/")
#         dest_url = partner_container_sas.rstrip("/") + "/" + safe_blob
#         logger.debug("Upload dest (masked): %s", mask_sas(dest_url))

#         attempt = 0
#         last_exc = None
#         while attempt < self.upload_retry:
#             attempt += 1
#             try:
#                 dest_blob = BlobClient.from_blob_url(dest_url)
#                 with open(local_file_path, "rb") as f:
#                     dest_blob.upload_blob(f, overwrite=True)
#                 return
#             except Exception as e:
#                 last_exc = e
#                 txt = str(e)
#                 if "AuthenticationFailed" in txt or "Server failed to authenticate the request" in txt or "403" in txt:
#                     logger.error("AuthenticationFailed while uploading %s. Masked dest: %s", blob_name, mask_sas(dest_url))
#                     logger.error("Common causes: expired/incorrect SAS token, missing 'w' permission on SAS, VM clock skew, or URL mangling.")
#                     logger.debug("Full upload exception: %s", txt)
#                     # do not keep retrying if auth failed — but allow a couple attempts (maybe transient)
#                 else:
#                     logger.warning("Upload attempt %d failed for %s: %s", attempt, blob_name, txt)
#                 if attempt < self.upload_retry:
#                     time.sleep(self.upload_retry_delay)
#         # if we exit loop with last_exc, raise it
#         raise last_exc



# -------------------------
# PartnerUploader (REPLACE this in your script)
# -------------------------
from urllib.parse import urlparse, parse_qs, urlunparse

# class PartnerUploader:
#     def __init__(self, openai_api_key: str, containers_api: str, submissions_api: str, upload_retry: int = 3, upload_retry_delay: int = 5):
#         self.openai_api_key = openai_api_key
#         self.containers_api = containers_api
#         self.submissions_api = submissions_api
#         self.upload_retry = upload_retry
#         self.upload_retry_delay = upload_retry_delay

#     def create_partner_container(self, purpose: str = "ingestion", extra_body: Optional[dict] = None) -> dict:
#         headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
#         body = {"purpose": purpose}
#         if extra_body:
#             body.update(extra_body)
#         r = requests.post(self.containers_api, json=body, headers=headers)
#         if not r.ok:
#             logger.error("Partner create container failed: status=%s body=%s", r.status_code, r.text)
#             r.raise_for_status()
#         return r.json()

#     def post_submission(self, container_id: str) -> dict:
#         headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
#         body = {"container_id": container_id}
#         r = requests.post(self.submissions_api, json=body, headers=headers)
#         if not r.ok:
#             logger.error("Partner post submission failed: status=%s body=%s", r.status_code, r.text)
#             r.raise_for_status()
#         return r.json()

#     # --------- helpers ----------
#     def _mask(self, url: str) -> str:
#         return mask_sas(url)

#     def _is_dfs_endpoint(self, url: str) -> bool:
#         parsed = urlparse(url)
#         return parsed.netloc.endswith(".dfs.core.windows.net")

#     def _convert_dfs_to_blob(self, url: str) -> str:
#         """
#         Convert a dfs.core.windows.net URL to blob.core.windows.net preserving path and query.
#         If url is already a blob endpoint, returns unchanged.
#         """
#         parsed = urlparse(url)
#         if not self._is_dfs_endpoint(url):
#             return url
#         blob_netloc = parsed.netloc.replace(".dfs.core.windows.net", ".blob.core.windows.net")
#         new_parsed = parsed._replace(netloc=blob_netloc)
#         return urlunparse(new_parsed)

#     def _inspect_sas(self, url: str) -> Tuple[Optional[str], Optional[str]]:
#         """
#         Return (sp, se) from SAS query if present (masked). sp = permissions, se = expiry.
#         """
#         if "?" not in url:
#             return None, None
#         parsed = urlparse(url)
#         q = parse_qs(parsed.query)
#         sp = q.get("sp", [None])[0]
#         se = q.get("se", [None])[0]
#         return sp, se

#     # --------- upload ----------
#     def upload_file_to_partner_using_sas(self, partner_container_sas: str, blob_name: str, local_file_path: str) -> None:
#         """
#         Upload local file to partner container using partner container SAS.
#         - Handles dfs->blob conversion (if partner returned a dfs endpoint).
#         - URL-encodes blob_name safely.
#         - Performs retries and logs helpful diagnostics when AuthenticationFailed occurs.
#         """
#         # encode blob name (preserve '/')
#         safe_blob = quote(blob_name, safe="/")
#         # build candidate dest (do not mutate partner_container_sas)
#         dest_candidate = partner_container_sas.rstrip("/") + "/" + safe_blob

#         # If partner SAS looks like DFS endpoint, convert to blob endpoint for BlobClient
#         if self._is_dfs_endpoint(partner_container_sas):
#             converted = self._convert_dfs_to_blob(partner_container_sas)
#             dest_url = converted.rstrip("/") + "/" + safe_blob
#             logger.debug("Converted DFS->Blob endpoint for upload.")
#         else:
#             dest_url = dest_candidate

#         # Log masked info and SAS permissions (masked)
#         sp, se = self._inspect_sas(partner_container_sas)
#         logger.info("Upload dest (masked): %s", self._mask(dest_url))
#         logger.info("Partner SAS permissions (sp)=%s expiry(se)=%s (masked)", sp if sp else "<none>", se if se else "<none>")

#         attempt = 0
#         last_exc = None
#         while attempt < self.upload_retry:
#             attempt += 1
#             try:
#                 dest_blob = BlobClient.from_blob_url(dest_url)
#                 with open(local_file_path, "rb") as f:
#                     dest_blob.upload_blob(f, overwrite=True)
#                 logger.debug("Upload succeeded for %s", blob_name)
#                 return
#             except Exception as e:
#                 last_exc = e
#                 txt = str(e)
#                 # If it's an auth issue, give actionable logs and don't blind-retry forever
#                 if "AuthenticationFailed" in txt or "Server failed to authenticate the request" in txt or "403" in txt:
#                     logger.error("AuthenticationFailed while uploading %s. Masked dest: %s", blob_name, self._mask(dest_url))
#                     logger.error("Common causes: expired/incorrect SAS token, missing 'w' permission on SAS, VM clock skew, or URL mangling.")
#                     logger.debug("Full upload exception: %s", txt)
#                     # don't do many retries for auth failure, but attempt a small number
#                 else:
#                     logger.warning("Upload attempt %d failed for %s: %s", attempt, blob_name, txt)
#                 if attempt < self.upload_retry:
#                     time.sleep(self.upload_retry_delay)

#         # if we exit loop, raise last exception
#         raise last_exc

#     # --------- small test helper (manual) ----------
#     def test_upload_small_file(self, partner_container_sas: str) -> Tuple[bool, str]:
#         """
#         Attempt to upload a tiny file to the partner SAS to test write permission quickly.
#         Returns (success_bool, message).
#         """
#         tmp = Path("/tmp") / f"az_test_{uuid.uuid4().hex}.bin"
#         try:
#             tmp.write_bytes(b"ok")
#             test_blob = f"__test__azcopy__{uuid.uuid4().hex}.bin"
#             try:
#                 self.upload_file_to_partner_using_sas(partner_container_sas, test_blob, str(tmp))
#                 return True, f"test upload succeeded -> {test_blob}"
#             finally:
#                 try:
#                     tmp.unlink()
#                 except Exception:
#                     pass
#         except Exception as e:
#             return False, str(e)
from urllib.parse import urlparse, parse_qs, quote, urlunparse
from azure.storage.filedatalake import DataLakeFileClient


class PartnerUploader:
    """
    Upload helper that supports:
      - ADLS Gen2 directory SAS (dfs.core.windows.net + sr=d) via DataLakeFileClient
      - Blob/container SAS (blob.core.windows.net) via BlobClient
    """

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
    def _mask(self, url: str) -> str:
        return mask_sas(url)

    def _parse_sas_info(self, url: str) -> dict:
        parsed = urlparse(url)
        q = parse_qs(parsed.query)
        return {"netloc": parsed.netloc, "path": parsed.path, "sp": q.get("sp", [None])[0], "se": q.get("se", [None])[0], "sr": q.get("sr", [None])[0]}

    def _is_dfs_dir_sas(self, url: str) -> bool:
        info = self._parse_sas_info(url)
        return (info["netloc"].endswith(".dfs.core.windows.net") or info["netloc"].endswith(".dfs.azure") ) and info["sr"] == "d"

    def _build_file_url_for_dfs_dir_sas(self, dir_sas: str, blob_name: str) -> str:
        """
        dir_sas: full SAS URL pointing to directory (i.e. https://account.dfs.core.windows.net/<filesystem>/<dir>?<sas>)
        blob_name: the relative path under that directory we want to upload (may include slashes)
        Returns: full file URL including SAS (suitable for DataLakeFileClient.from_file_url)
        """
        if "?" not in dir_sas:
            raise ValueError("dir_sas missing SAS query")
        base, sas = dir_sas.split("?", 1)
        base = base.rstrip("/")
        # Append encoded path
        # ADLS path separator is '/', so we encode each path segment
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
        # If base path already includes more than container (i.e., a specific blob path), assume blob-level SAS and return as-is
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) > 1:
            # base already references a blob; don't append
            return container_sas
        safe_blob = quote(blob_name, safe="/")
        return f"{base}/{safe_blob}?{sas}"

    # ---------- main upload ----------
    def upload_file_to_partner_using_sas(self, partner_container_sas: str, blob_name: str, local_file_path: str) -> None:
        """
        Upload local file to partner container using partner SAS.
        Automatically chooses DataLakeFileClient if partner SAS points to DFS directory (sr=d).
        Encodes path segments correctly; retries on transient errors; provides actionable logs on auth failure.
        """
        # Show masked info
        info = self._parse_sas_info(partner_container_sas)
        logger.info("Partner SAS netloc=%s path=%s sp=%s se=%s sr=%s (masked)", info["netloc"], info["path"], info["sp"], info["se"], info["sr"])
        logger.info("Masked partner SAS: %s", self._mask(partner_container_sas))

        last_exc = None
        attempt = 0

        if self._is_dfs_dir_sas(partner_container_sas):
            # ADLS Gen2 directory SAS -> use DataLakeFileClient
            file_url = self._build_file_url_for_dfs_dir_sas(partner_container_sas, blob_name)
            logger.info("Uploading to ADLS Gen2 directory via DataLakeFileClient. Masked dest: %s", self._mask(file_url))
            while attempt < self.upload_retry:
                attempt += 1
                try:
                    # client = DataLakeFileClient.from_file_url(file_url)
                    from urllib.parse import urlparse, unquote

                    # ... inside the ADLS upload branch ...
                    parsed = urlparse(file_url)
                    # parsed.path is like '/<filesystem>/<maybe/path/to/blob>'
                    path_parts = [p for p in parsed.path.split("/") if p]
                    if len(path_parts) < 2:
                        raise RuntimeError(f"Cannot parse filesystem/file path from ADLS URL: {file_url}")
                    filesystem = path_parts[0]
                    file_path = "/".join(path_parts[1:])  # ADLS path (may contain '/')
                    # SAS token is the query part
                    sas_token = parsed.query  # this is the query string without the leading '?'
                    # Build account (endpoint) URL for DataLakeFileClient constructor
                    account_url = f"{parsed.scheme}://{parsed.netloc}"

                    # Instantiate DataLakeFileClient using constructor compatible with older SDKs
                    client = DataLakeFileClient(account_url, filesystem, file_path, credential=sas_token)



                    # upload_data will handle chunking; overwrite True
                    with open(local_file_path, "rb") as f:
                        data = f.read()
                    client.upload_data(data, overwrite=True)
                    return
                except Exception as e:
                    last_exc = e
                    txt = str(e)
                    if "AuthenticationFailed" in txt or "Signature" in txt or "403" in txt:
                        logger.error("AuthenticationFailed (ADLS) while uploading %s. Masked dest: %s", blob_name, self._mask(file_url))
                        logger.error("Common causes: SAS missing 'w' permission, SAS expired, or URL was malformed.")
                        logger.debug("Full ADLS upload exception: %s", txt)
                    else:
                        logger.warning("Attempt %d failed for ADLS upload %s: %s", attempt, blob_name, txt)
                    if attempt < self.upload_retry:
                        time.sleep(self.upload_retry_delay)
            raise last_exc

        else:
            # Blob endpoint (use BlobClient)
            dest_url = self._build_dest_url_for_blob_sas(partner_container_sas, blob_name)
            logger.info("Uploading to Blob endpoint. Masked dest: %s", self._mask(dest_url))
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
                        logger.error("AuthenticationFailed (Blob) while uploading %s. Masked dest: %s", blob_name, self._mask(dest_url))
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
# Remote manifests uploader (optional)
# -------------------------
class RemoteManifestsUploader:
    def __init__(self, enabled: bool, connection_string: Optional[str], container_name: Optional[str], prefix: Optional[str] = None):
        self.enabled = bool(enabled)
        self.conn_str = connection_string
        self.container_name = container_name
        self.prefix = prefix or ""
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

    def upload_manifest_dict(self, manifest: dict, blob_name: Optional[str] = None) -> str:
        if not self.enabled or not self._client:
            raise RuntimeError("Remote manifests not enabled/configured")
        blob_name = blob_name or f"{self.prefix}{now_utc().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex}.json"
        blob_client = self._client.get_blob_client(blob_name)
        data = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        blob_client.upload_blob(data, overwrite=True)
        return blob_name

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

        # source SAS or its env var name (optional)
        self.source_sas = cfg.get("azure_source", "container_sas") or os.getenv(cfg.get("azure_source", "container_sas_env", default=""))
        self.source_prefix = cfg.get("azure_source", "source_prefix", default=None)

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
        self.uploader = PartnerUploader(self.openai_key, self.containers_api, self.submissions_api)

        # SAS generation internals
        self._generated_container_sas: Optional[str] = None
        self._account_name, self._account_key = self._parse_account_from_connstr(self.source_conn)

    # -------------------------
    # Account parsing and SAS generation
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
        return candidates

    # -------------------------
    # metadata helpers
    # -------------------------
    def determine_content_type(self, blob_name: str) -> str:
        ln = blob_name.lower()
        if ln.endswith(".json"):
            return "application/json"
        if ln.endswith(".insv"):
            return "application/octet-stream"
        return "application/octet-stream"

    def generate_metadata_obj(self, blob_name: str) -> dict:
        return {"content-type": self.determine_content_type(blob_name), "version": "001"}

    # -------------------------
    # azcopy wrapper
    # -------------------------
    def copy_blob_via_azcopy(self, source_url: str, partner_container_sas: str, blob_name: str) -> None:
        src = source_url
        # If source_url is a container SAS (no blob path), append properly encoded blob name
        if "?" in source_url and (f"/{self.source_container}/" in source_url or source_url.rstrip().endswith(self.source_container) or source_url.rstrip().endswith(self.source_container + "?") or source_url.rstrip().endswith(self.source_container + "/")):
            src = source_url.rstrip("/") + "/" + quote(blob_name, safe="")
        else:
            # If source_url seems to be a blob SAS already, assume it contains blob path
            pass
        dst = partner_container_sas.rstrip("/") + "/" + quote(blob_name, safe="")
        cmd = [self.azcopy_path, "copy", src, dst, "--overwrite=false"]
        logger.info("Running azcopy: %s", " ".join(cmd))
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

    # -------------------------
    # run loop
    # -------------------------
    def run_once(self) -> None:
        logger.info("Starting data push run (final).")
        to_send = self.blobs_to_send()
        if not to_send:
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

        submission_manifest = {
            "container_id": container_id,
            "create_response": create_resp,
            "created_at": now_iso(),
            "files": []
        }
        processed: List[Tuple[str, str]] = []

        # Pre-generate container SAS if possible to reuse
        if not self.source_sas:
            self.ensure_container_sas()

        for blob_name, fp, last_mod_iso in to_send:
            logger.info("Processing: %s", blob_name)
            metadata_blob_name = blob_name + ".metadata.json"
            tmp_blob_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(blob_name).name}")
            tmp_meta_path = str(Path(self.work_dir) / f"{uuid.uuid4().hex}_{Path(metadata_blob_name).name}")
            upload_status = "failed: unknown"

            try:
                if self.use_azcopy:
                    source_for_azcopy = self.get_best_source_url_for_blob(blob_name)
                    if source_for_azcopy:
                        try:
                            logger.info("Using azcopy for blob: %s (source_sas masked: %s)", blob_name, mask_sas(source_for_azcopy))
                            self.copy_blob_via_azcopy(source_for_azcopy, partner_container_sas, blob_name)
                            # metadata: copy if exists else generate and upload
                            try:
                                if self.source_client.metadata_blob_exists(metadata_blob_name):
                                    self.copy_blob_via_azcopy(source_for_azcopy, partner_container_sas, metadata_blob_name)
                                else:
                                    with open(tmp_meta_path, "w", encoding="utf-8") as fm:
                                        json.dump(self.generate_metadata_obj(blob_name), fm)
                                    # upload metadata via partner SAS using uploader helper (handles encoding & retries)
                                    self.uploader.upload_file_to_partner_using_sas(partner_container_sas, metadata_blob_name, tmp_meta_path)
                            except Exception as e_meta:
                                logger.warning("Metadata copy/upload warning for %s: %s", metadata_blob_name, e_meta)
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
                # Fallback to SDK download/upload path
                logger.info("Falling back to SDK download/upload for %s due to: %s", blob_name, az_err)
                try:
                    self.source_client.download_blob_to_path(blob_name, tmp_blob_path)

                    if self.source_client.metadata_blob_exists(metadata_blob_name):
                        try:
                            self.source_client.download_metadata_to_path(metadata_blob_name, tmp_meta_path)
                        except Exception as e:
                            logger.warning("Failed to download metadata %s: %s - generating instead", metadata_blob_name, e)
                            with open(tmp_meta_path, "w", encoding="utf-8") as fm:
                                json.dump(self.generate_metadata_obj(blob_name), fm)
                    else:
                        with open(tmp_meta_path, "w", encoding="utf-8") as fm:
                            json.dump(self.generate_metadata_obj(blob_name), fm)

                    # Upload blob and metadata via partner SAS (uploader handles encoding & retries)
                    try:
                        self.uploader.upload_file_to_partner_using_sas(partner_container_sas, blob_name, tmp_blob_path)
                        self.uploader.upload_file_to_partner_using_sas(partner_container_sas, metadata_blob_name, tmp_meta_path)
                        upload_status = "success"
                    except Exception as upl_ex:
                        logger.error("Upload via partner SAS failed for %s: %s", blob_name, upl_ex)
                        upload_status = f"failed: {str(upl_ex)}"

                except Exception as dl_ex:
                    logger.error("Download failed for %s: %s", blob_name, dl_ex, exc_info=True)
                    upload_status = f"failed: {str(dl_ex)}"

            finally:
                for p in (tmp_blob_path, tmp_meta_path):
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass

            submission_manifest["files"].append({
                "blob_name": blob_name,
                "fingerprint": fp,
                "metadata_blob": metadata_blob_name,
                "upload_status": upload_status,
                "uploaded_at": now_iso()
            })
            if upload_status == "success":
                processed.append((last_mod_iso, blob_name))

        # persist submission manifest (remote if configured, else local)
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

        # POST submission to partner (with retries)
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
            # update manifest storage with submission_response
            if manifest_uploaded:
                try:
                    self.remote_manifests.upload_manifest_dict(submission_manifest, blob_name=manifest_blob_name)
                except Exception:
                    pass
            else:
                atomic_write_json(str(Path(self.manifests_dir) / manifest_blob_name), submission_manifest)

        # Advance watermark using processed successes only
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
            logger.info("No blobs processed successfully; watermark unchanged")

        # -------------------------
        # Summary printed to terminal
        # -------------------------
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
            # print a trimmed explanation if available
            entry = next((x for x in submission_manifest["files"] if x["blob_name"] == b), None)
            reason = entry["upload_status"] if entry else ""
            print(f"  ✖ {b} -> {reason}")
        wm_ts, wm_blob = self.watermark_store.get_watermark()
        print(f"Advanced watermark: {wm_ts} , last_blob: {wm_blob}")
        print("=" * 72 + "\n")

        logger.info("Run finished.")

# -------------------------
# CLI entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Push new blobs (final) with SAS & azcopy support.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = ConfigLoader(args.config)
    orchestrator = DataPushOrchestrator(cfg)
    orchestrator.run_once()

if __name__ == "__main__":
    main()
