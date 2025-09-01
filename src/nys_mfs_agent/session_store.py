# src/nys_mfs_agent/session_store.py
from __future__ import annotations
import json, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import CFG

def _sid() -> str:
    return f"sess-{uuid.uuid4().hex[:8]}"

def sessions_dir() -> Path:
    CFG.sessions_dir.mkdir(parents=True, exist_ok=True)
    return CFG.sessions_dir

def list_sessions() -> List[str]:
    return sorted(p.stem for p in sessions_dir().glob("sess-*.json"))

def session_path(session_id: str) -> Path:
    return sessions_dir() / f"{session_id}.json"

def create_session(session_id: Optional[str] = None, intro_assistant: Optional[str] = None) -> str:
    sid = session_id or _sid()
    data = {
        "id": sid,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "messages": [],
        "meta": {"title": "New York MFS Session"},
    }
    if intro_assistant:
        data["messages"].append({"role": "assistant", "content": intro_assistant})
    session_path(sid).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return sid

def load_session(session_id: str) -> Dict[str, Any]:
    fp = session_path(session_id)
    if not fp.exists():
        return {"id": session_id, "messages": [], "meta": {}}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {"id": session_id, "messages": [], "meta": {}}

def save_session(session_id: str, data: Dict[str, Any]) -> None:
    fp = session_path(session_id)
    fp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def append_message(session_id: str, role: str, content: str) -> None:
    data = load_session(session_id)
    data.setdefault("messages", []).append({"role": role, "content": content})
    save_session(session_id, data)

def export_path(session_id: str) -> Optional[str]:
    fp = session_path(session_id)
    return str(fp) if fp.exists() else None
