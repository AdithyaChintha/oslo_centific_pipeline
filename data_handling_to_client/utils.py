#!/usr/bin/env python3
"""
Shared utility functions for the data handling pipeline.
This module contains utilities that are used by both the main pipeline and state manager.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import base64
from dateutil import parser as dtparser


def now_utc() -> datetime:
    """Get current UTC datetime"""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Get current UTC datetime as ISO string"""
    return now_utc().isoformat()


def atomic_write_json(path: str, obj: object) -> None:
    """Atomically write JSON to file using temporary file"""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json_if_exists(path: str) -> Dict:
    """Read JSON file if it exists, return empty dict otherwise"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def normalize_content_md5(md5) -> Optional[str]:
    """Normalize MD5 hash to hex string"""
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
    """Parse ISO datetime string to UTC datetime"""
    dt = dtparser.parse(iso_str)
    return dt.astimezone(timezone.utc)