# src/nys_mfs_agent/chat_service.py
from __future__ import annotations
from typing import AsyncIterator, Optional, Dict, Any

from .multi_orchestrator import MultiAgent
from .logger import SessionLogger
from .sanitize import clean_text
from .session_store import (
    create_session, load_session, save_session, append_message,
    list_sessions, export_path
)
from .config import CFG

INTRO = (
    "Hi — I’m the **New York Workers’ Compensation Medical Fee Schedule Assistant**.\n"
    "Ask about sections, ground rules, billing/documentation, E/M, Physical Medicine, Radiology, Path/Lab, etc.\n"
    "Out-of-scope requests will be politely declined."
)

class ChatService:
    def __init__(self):
        self.agent = MultiAgent()

    def ensure_session(self, session_id: Optional[str]) -> str:
        if session_id and export_path(session_id):
            return session_id
        return create_session(intro_assistant=INTRO)

    def list_sessions(self):
        return list_sessions()

    async def stream_answer(self, session_id: str, user_text: str) -> AsyncIterator[str]:
        # write user message
        append_message(session_id, "user", user_text)

        # per-question log
        logger = SessionLogger()
        collected = []
        async for tok in self.agent.ask(user_text, logger=logger):
            collected.append(tok)
            yield tok
        final = clean_text("".join(collected).strip())

        # save assistant message
        append_message(session_id, "assistant", final)
