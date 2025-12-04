#!/usr/bin/env python3
"""
Copy all files from a OneDrive **shared link** (delegated login) to Azure Blob Storage.
- Uses Microsoft Graph delegated auth (interactive with device-code fallback)
- Uses Azure Blob **connection string**
- Recursively preserves folder structure
- Handles Graph pagination + basic throttling retries
"""

import base64
import sys
import os
from typing import Dict, Any, Iterator, Optional
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from msal import PublicClientApplication
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceExistsError

# ------------------------------------------------------------------------------------
# 🔐 SECRETS / SETTINGS — fill these in
# ------------------------------------------------------------------------------------
GRAPH_TENANT_ID = "9b415834-803a-4da0-afdc-fe6b1d52d649"   # Entra ID tenant GUID
GRAPH_CLIENT_ID = "cf36a429-766a-47eb-9b49-66b04a227ead"   # App registration (public client) Client ID
ONEDRIVE_SHARE_URL = "https://digitaltechedge-my.sharepoint.com/:f:/g/personal/v_shiva_githa_centific_com/EpZ1IU79WyFLnxSpKzhi3uQBVxq-d5u7-tgVOSOgtlpCrg?e=qhSeIz"

AZURE_STORAGE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=https;AccountName=oslotestvideo;AccountKey=zOevIegkZjld6ciTY+alA+YkzZ2gdAWVP7rkuhty5NAZ67AtiBB3fRTaa+eE3UbqhgwHZvWkOM0L+ASt4zCx6g==;EndpointSuffix=core.windows.net"
)
AZ_BLOB_CONTAINER = "instavideo"      # container must exist or will be created
AZ_BLOB_PREFIX = "oslo_stage_2_tar_files/onedrive_test"    # optional virtual folder in the container; "" for none

# Microsoft Graph delegated scopes required
GRAPH_SCOPES = ["Files.Read", "User.Read"] 
# ------------------------------------------------------------------------------------
# Auth & HTTP helpers
# ------------------------------------------------------------------------------------
GRAPH_AUTHORITY = f"https://login.microsoftonline.com/{GRAPH_TENANT_ID}"

class TransientHTTPError(Exception):
    pass

@retry(wait=wait_exponential(multiplier=1, min=1, max=30),
       stop=stop_after_attempt(6),
       retry=retry_if_exception_type(TransientHTTPError))
def graph_get(url: str, headers: Dict[str, str]) -> requests.Response:
    resp = requests.get(url, headers=headers)
    if resp.status_code in (429, 500, 502, 503):
        # Respect Retry-After if provided
        raise TransientHTTPError(f"Transient HTTP {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()
    return resp

GRAPH_SCOPES = ["Files.Read.All", "User.Read"]  # no offline_access

def acquire_graph_token() -> str:
    from msal import PublicClientApplication
    app = PublicClientApplication(
        client_id=GRAPH_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{GRAPH_TENANT_ID}",
    )

    # try silent first
    accts = app.get_accounts()
    if accts:
        res = app.acquire_token_silent(GRAPH_SCOPES, account=accts[0])
        if res and "access_token" in res:
            return res["access_token"]

    # device code only (no redirect URI needed)
    flow = app.initiate_device_flow(scopes=GRAPH_SCOPES)
    if "user_code" not in flow:
        raise RuntimeError(f"Failed to start device code flow: {flow}")
    print(f"To sign in, open {flow['verification_uri']} and enter code: {flow['user_code']}")
    res = app.acquire_token_by_device_flow(flow)
    if "access_token" not in res:
        raise RuntimeError(f"Device code auth failed: {res}")
    return res["access_token"]


def urlsafe_share_id(share_url: str) -> str:
    """Graph /shares API requires: 'u!' + URL-safe base64 (no padding) of the original share URL."""
    return "u!" + base64.urlsafe_b64encode(share_url.encode("utf-8")).decode("utf-8").rstrip("=")

# ------------------------------------------------------------------------------------
# Graph traversal (shared link)
# ------------------------------------------------------------------------------------
def list_children_recursive(base_share_id: str, headers: Dict[str, str], parent_relpath: str = "") -> Iterator[Dict[str, Any]]:
    """
    Yields items as:
      { name, id, is_folder, downloadUrl (files only), relpath }
    """
    # Build the correct listing URL for root vs subpath
    if parent_relpath:
        # Address by path inside the share
        encoded_path = requests.utils.quote(parent_relpath, safe="/")
        url = f"https://graph.microsoft.com/v1.0/shares/{base_share_id}/driveItem:/{encoded_path}:/children"
    else:
        url = f"https://graph.microsoft.com/v1.0/shares/{base_share_id}/driveItem/children"

    next_link: Optional[str] = url
    while next_link:
        resp = graph_get(next_link, headers).json()
        for it in resp.get("value", []):
            name = it["name"]
            is_folder = "folder" in it
            relpath = (parent_relpath + name) if parent_relpath else name
            if is_folder:
                # Recurse into subfolder
                yield from list_children_recursive(base_share_id, headers, parent_relpath=relpath + "/")
            else:
                yield {
                    "name": name,
                    "id": it["id"],
                    "is_folder": False,
                    "downloadUrl": it.get("@microsoft.graph.downloadUrl"),
                    "relpath": relpath
                }
        next_link = resp.get("@odata.nextLink")

# ------------------------------------------------------------------------------------
# Azure Blob helpers
# ------------------------------------------------------------------------------------
def get_blob_client(conn_str: str, container: str, blob_name: str) -> BlobClient:
    svc = BlobServiceClient.from_connection_string(conn_str)
    try:
        svc.create_container(container)
    except ResourceExistsError:
        pass
    return svc.get_blob_client(container=container, blob=blob_name)

def upload_stream_to_blob(download_url: str, blob: BlobClient) -> None:
    # Stream from Graph's pre-authenticated download URL directly to Blob
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        blob.upload_blob(r.raw, overwrite=True, max_concurrency=8)

# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
def main() -> None:
    # Acquire Graph token
    token = acquire_graph_token()
    headers = {"Authorization": f"Bearer {token}"}

    # Resolve the root of the share (works for folders or a single file share)
    share_id = urlsafe_share_id(ONEDRIVE_SHARE_URL)
    root_resp = graph_get(f"https://graph.microsoft.com/v1.0/shares/{share_id}/driveItem", headers).json()

    # If the link is a single file, upload it and exit
    if "file" in root_resp and root_resp.get("@microsoft.graph.downloadUrl"):
        relpath = root_resp["name"]
        blob_path = (AZ_BLOB_PREFIX + relpath).lstrip("/")
        blob = get_blob_client(AZURE_STORAGE_CONNECTION_STRING, AZ_BLOB_CONTAINER, blob_path)
        print(f"Uploading single file: {relpath} → {AZ_BLOB_CONTAINER}/{blob_path}")
        upload_stream_to_blob(root_resp["@microsoft.graph.downloadUrl"], blob)
        print("Done. Uploaded 1 file.")
        return

    # Otherwise, traverse folder recursively
    total_files = 0
    for item in list_children_recursive(share_id, headers, parent_relpath=""):
        if item["is_folder"]:
            continue
        if not item.get("downloadUrl"):
            print(f"Skip (no download URL): {item['relpath']}", file=sys.stderr)
            continue
        blob_path = (AZ_BLOB_PREFIX + item["relpath"]).lstrip("/")
        blob = get_blob_client(AZURE_STORAGE_CONNECTION_STRING, AZ_BLOB_CONTAINER, blob_path)
        print(f"Uploading: {item['relpath']} → {AZ_BLOB_CONTAINER}/{blob_path}")
        upload_stream_to_blob(item["downloadUrl"], blob)
        total_files += 1

    print(f"Done. Uploaded {total_files} files.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        sys.exit(130)
