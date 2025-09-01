# src/nys_mfs_agent/logger.py
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Optional
from .config import CFG

def _str(x) -> str:
    try:
        return str(x)
    except Exception:
        return "unknown"

def _resolve_sessions_dir() -> Path:
    # Prefer CFG.sessions_dir if present
    sd = getattr(CFG, "sessions_dir", None)
    if sd:
        return Path(sd)
    # Else build from run_dir/doc_id
    run_dir = getattr(CFG, "run_dir", None)
    doc_id  = getattr(CFG, "doc_id", "doc")
    if run_dir:
        return Path(run_dir) / "sessions"
    # Last resort: ./data/runs/<doc_id>/sessions
    return Path("data") / "runs" / _str(doc_id) / "sessions"

def _resolve_model_str() -> str:
    # Try common config attributes in order; return path-ish string or "unknown"
    for attr in ("merged_model_dir", "base_model_dir", "model_dir"):
        p = getattr(CFG, attr, None)
        if p:
            try:
                p = Path(p)
                return str(p)
            except Exception:
                return _str(p)
    return "unknown"

class SessionLogger:
    """
    Lightweight JSONL session logger.
    - Writes to <sessions_dir>/logs/session_<ms>.jsonl
    - .event(kind, **payload) for steps; .close(meta=...) when done.
    - Exposes ._path for compatibility with your notebook prints.
    """
    def __init__(self, path: Optional[Path] = None):
        base_dir = _resolve_sessions_dir() / "logs"
        base_dir.mkdir(parents=True, exist_ok=True)
        if path is None:
            ts_ms = int(time.time() * 1000)
            path = base_dir / f"session_{ts_ms}.jsonl"
        self._path: str = str(path)
        self._open = True
        # Initial header
        self.event(
            "_init",
            doc_id=_str(getattr(CFG, "doc_id", "unknown")),
            model=_resolve_model_str(),
        )

    def event(self, kind: str, **payload: Any) -> None:
        if not self._open:
            return
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "kind": kind,
            **payload,
        }
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self, meta: Optional[dict] = None) -> None:
        if not self._open:
            return
        self.event("_close", **(meta or {}))
        self._open = False

__all__ = ["SessionLogger"]
