# src/agent/orchestrator.py
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Dict, Any, List

from .config import CFG
from .prompts import load_prompts, build_rag_prompt
from .llms import load_core_llm, stream_generate
from .retriever import HierBM25CEReranker
from .policy import guard_title17_scope, REFUSAL
from .memory import JsonSessionStore, build_summary_memory
from .logger import EventLogger


class Title17Agent:
    """
    Async, streaming agent for Title 17 Q&A with RAG.
    Keeps current behavior: guardrail -> retrieve -> build prompt -> stream core LLM.
    """

    def __init__(self):
        self.cfg = CFG
        self.prompts = load_prompts()
        self.core = load_core_llm()
        self.retr = HierBM25CEReranker()
        self.sessions = JsonSessionStore()
        self.mem = build_summary_memory(k_tokens=800)  # optional; may be None
        self.log = EventLogger()

    async def achat_stream(self, session_id: str, user_text: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Yields dict events:
          - {'type': 'token', 'text': '...'} for streaming tokens
          - {'type': 'final', 'text': '...', 'citations': [...]} at the end
          - {'type': 'error', 'text': '...'} on failure
        """
        # session + logs
        self.log.add_conv_if_missing(session_id)
        self.log.log_msg(session_id, "user", user_text)
        self.sessions.append(session_id, "user", user_text)

        # guardrails
        gd = guard_title17_scope(user_text)
        if not gd.allow:
            refusal = REFUSAL
            self.log.log_event("guard_refusal", {"session_id": session_id, "reason": gd.reason})
            self.log.log_msg(session_id, "assistant", refusal)
            self.sessions.append(session_id, "assistant", refusal)
            yield {"type": "final", "text": refusal, "guard_reason": gd.reason}
            return

        # (1) retrieve
        hits = self.retr.search(
            user_text,
            self.cfg.k_nodes, self.cfg.k_final_nodes,
            self.cfg.k_each_node, self.cfg.k_final_chunks
        )
        self.log.log_event("retrieve_done", {"session_id": session_id, "n_hits": len(hits)})

        # (2) memory: recent turns to prepend
        history = self.sessions.last_messages(session_id, n=6)
        history_blocks = []
        for m in history:
            if m["role"] == "user":
                history_blocks.append(f"User: {m['content']}")
            else:
                history_blocks.append(f"Assistant: {m['content']}")
        history_txt = "\n".join(history_blocks[-6:]) if history_blocks else ""

        # (2b) optional summary memory
        summary_txt = ""
        if self.mem is not None:
            try:
                mem_vars = self.mem.load_memory_variables({})
                val = mem_vars.get("history")
                if isinstance(val, str) and val.strip():
                    summary_txt = val.strip()
            except Exception:
                pass

        # (3) build RAG prompt
        prompt = build_rag_prompt(user_text, hits, self.prompts)

        # prepend chat summary / history (no backslashes inside f-string expressions)
        prefix_parts: List[str] = []
        if summary_txt:
            prefix_parts.append(f"[CHAT SUMMARY]\n{summary_txt}")
        if history_txt:
            prefix_parts.append(history_txt)
        if prefix_parts:
            prefix = "\n\n".join(prefix_parts) + "\n\n"
            prompt = prefix + prompt

        # (4) stream generation
        acc: List[str] = []

        async def _gen():
            # run blocking generator on a thread
            for tok in stream_generate(
                self.core,
                prompt,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature
            ):
                yield tok

        try:
            async for tok in _aiter_wrap(_gen()):
                acc.append(tok)
                yield {"type": "token", "text": tok}
        except Exception as e:
            err = f"[generation_error] {e}"
            self.log.log_event("gen_error", {"session_id": session_id, "error": str(e)})
            yield {"type": "error", "text": err}
            return

        final_text = "".join(acc).strip()
        self.log.log_msg(session_id, "assistant", final_text)
        self.sessions.append(session_id, "assistant", final_text)

        # (5) optional summary memory update
        if self.mem is not None:
            try:
                self.mem.chat_memory.add_user_message(user_text)
                self.mem.chat_memory.add_ai_message(final_text)
            except Exception:
                pass

        yield {"type": "final", "text": final_text, "citations": _mk_citations(hits)}


def _mk_citations(hits: List[Dict[str, Any]]):
    cits = []
    for h in hits:
        cits.append({
            "chunk_id": h["chunk_id"],
            "node_id": h["node_id"],
            "section": h.get("section"),
            "pages": h.get("pages") or [],
            "score": h.get("score"),
        })
    return cits


async def _aiter_wrap(sync_gen):
    """
    Expose a sync generator as async by pulling next() in a thread pool.
    """
    it = sync_gen
    if hasattr(it, "__aiter__"):
        async for x in it:
            yield x
    else:
        loop = asyncio.get_running_loop()

        def _next():
            try:
                return True, next(it)
            except StopIteration:
                return False, None

        while True:
            ok, val = await loop.run_in_executor(None, _next)
            if not ok:
                break
            yield val
