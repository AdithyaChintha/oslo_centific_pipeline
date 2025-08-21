from typing import Callable, Dict, Iterator, List, Optional, Tuple
import os
import json
from copy import deepcopy
from numbers import Number
import numpy as np

# local timestamp helper
try:
    from .Tstamp import fmt_hhmmss_ms
except Exception:
    # fallback: simple hh:mm:ss.ms formatter
    def fmt_hhmmss_ms(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t - h * 3600 - m * 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"


def _to_py(x):
    """Convert numpy / numpy scalar to native Python types for json serialization."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Number) and (not isinstance(x, (int, float))):
        try:
            return float(x)
        except Exception:
            return x
    return x


def _format_event(row: dict, names_map: Optional[dict] = None) -> dict:
    """Return a normalized, JSON-serializable copy of an event row.

    Normalizations applied (idempotent):
      - round 't' to 3 decimals
      - ensure 'ts' exists (compute from 't' if missing)
      - ensure 'frame' is int if present
      - round 'conf' to 4 decimals
      - round bbox coords to 2 decimals and convert to list
      - map numeric 'cls' to name via names_map if provided
    """
    r = deepcopy(row)

    # convert numpy scalars/arrays to python types
    for k, v in list(r.items()):
        if isinstance(v, (np.generic, np.ndarray)):
            r[k] = _to_py(v)

    # timestamp
    if "t" in r:
        try:
            t = float(r["t"])
            r["t"] = round(t, 3)
        except Exception:
            r["t"] = r.get("t")
    else:
        r["t"] = float(r.get("ts", 0.0)) if isinstance(r.get("ts", None), (int, float)) else 0.0

    if "ts" not in r or not r["ts"]:
        try:
            r["ts"] = fmt_hhmmss_ms(float(r["t"]))
        except Exception:
            r["ts"] = fmt_hhmmss_ms(0.0)

    # frame
    if "frame" in r:
        try:
            r["frame"] = int(r["frame"])
        except Exception:
            pass

    # conf
    if "conf" in r:
        try:
            r["conf"] = round(float(r["conf"]), 4)
        except Exception:
            pass

    # bbox
    if "bbox" in r and r["bbox"] is not None:
        try:
            bbox = list(map(float, r["bbox"]))
            r["bbox"] = [round(x, 2) for x in bbox]
        except Exception:
            pass

    # class mapping
    if "cls" in r and names_map is not None:
        try:
            # if numeric, map; otherwise leave as-is
            if isinstance(r["cls"], (int, float)):
                cls_id = int(r["cls"])
                r["cls"] = names_map.get(cls_id, str(cls_id))
        except Exception:
            pass

    return r


def write_jsonl(path: str, rows: List[dict], meta: Optional[dict] = None, names_map: Optional[dict] = None) -> str:
    """Write a list of dicts to JSONL. If meta provided, write as first line under key '_meta'.

    This writer will normalize common event fields (t, ts, frame, conf, bbox, cls).
    If names_map is provided it will be used to map numeric class ids to names. You can
    also pass names_map inside meta under key 'names_map'.
    """
    # prefer names_map passed explicitly, else check meta
    if names_map is None and isinstance(meta, dict):
        names_map = meta.get("names_map")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if meta is not None:
            f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for r in rows:
            try:
                out = _format_event(r, names_map=names_map)
            except Exception:
                out = r
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    return path




