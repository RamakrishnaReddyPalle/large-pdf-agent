# src/agent/memory.py
from __future__ import annotations

from pathlib import Path
import json, time
from typing import Dict, Any, List, Optional

from .config import CFG


# ---------- Simple per-session JSON transcript + rolling buffer ----------

class JsonSessionStore:
    def __init__(self, base_dir: Path | None = None):
        self.base = Path(base_dir or CFG.sessions_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _fp(self, session_id: str) -> Path:
        return self.base / f"{session_id}.json"

    def read(self, session_id: str) -> Dict[str, Any]:
        fp = self._fp(session_id)
        if not fp.exists():
            return {"session_id": session_id, "created": time.time(), "messages": []}
        return json.loads(fp.read_text(encoding="utf-8"))

    def append(self, session_id: str, role: str, content: str):
        data = self.read(session_id)
        data["messages"].append({"ts": time.time(), "role": role, "content": content})
        self._fp(session_id).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def last_messages(self, session_id: str, n: int = 6) -> List[Dict[str, Any]]:
        return self.read(session_id).get("messages", [])[-n:]


# ---------- Conversation summary memory (Ollama) ----------

# Prefer modern package; fall back to community (deprecated) if needed.
try:
    from langchain_ollama import ChatOllama  # pip install langchain-ollama
except Exception:
    try:
        from langchain_community.chat_models import ChatOllama  # type: ignore
    except Exception:
        ChatOllama = None  # type: ignore

try:
    from langchain.memory import ConversationSummaryBufferMemory
except Exception:
    ConversationSummaryBufferMemory = None  # type: ignore


def build_summary_memory(k_tokens: int = 800):
    """
    Optional ConversationSummaryBufferMemory using local Ollama model (CFG.ollama_summarizer).
    Returns None if packages are unavailable, to avoid breaking your agent.
    """
    if ConversationSummaryBufferMemory is None or ChatOllama is None:
        return None
    llm = ChatOllama(model=CFG.ollama_summarizer, temperature=0.2)
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=k_tokens,
        return_messages=True,
    )
