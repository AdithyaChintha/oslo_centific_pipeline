from typing import Callable, Dict, Iterator, List, Optional, Tuple
import os, json

def write_jsonl(path: str, rows: List[dict], meta: Optional[dict] = None) -> str:
    """Write a list of dicts to JSONL. If meta provided, write it as the first line under key '_meta'."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if meta is not None:
            f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path




