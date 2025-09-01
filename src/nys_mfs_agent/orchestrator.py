# src/nys_mfs_agent/orchestrator.py
from __future__ import annotations
import asyncio
from typing import List, Dict, AsyncIterator

# Optional helpers you already stubbed
from .rewrites import rewrite_query
from .sanitize import clean_text

from .config import CFG
from .prompts import render, load_prompt
from .llms_async import load_core_llm, render_chat, astream
from .policy import sanitize_answer, is_in_scope, REFUSAL
from .memory import SummaryBuffer
from .retriever import HybridRetriever, prepare_contexts
from .logger import SessionLogger

class Agent:
    def __init__(self):
        self.core = load_core_llm()
        self.retr = HybridRetriever(
            chunks_dir=CFG.chunks_dir,
            graph_dir=CFG.graph_dir,
            reranker_dir=CFG.reranker_dir,
        )
        self.mem = SummaryBuffer(max_chars=4000)

    def _system(self) -> str:
        return load_prompt("system")

    def _style(self) -> str:
        return load_prompt("style_rules")

    def _render_user(self, question: str, contexts: List[str]) -> str:
        return render(
            "answer_with_citations",
            question=question,
            contexts="\n\n".join(contexts),
        )

    async def _draft_stream(
        self,
        system_text: str,
        user_text: str,
        history: List[Dict[str, str]],
    ) -> AsyncIterator[str]:
        prompt = render_chat(self.core.tokenizer, system_text, user_text, history=history)
        # Slightly smaller cap for speed/discipline; greedy
        async for tok in astream(self.core, prompt, max_new_tokens=280, temperature=0.0):
            yield tok

    async def ask(self, question: str, logger: SessionLogger | None = None) -> AsyncIterator[str]:
        if logger is None:
            logger = SessionLogger()
        logger.event("question", text=question)

        # Rewrite for better retrieval (if applicable)
        q_for_retrieval = rewrite_query(question)
        if q_for_retrieval != question:
            logger.event("query_rewrite", original=question, rewritten=q_for_retrieval)

        # Retrieve: keep final chunk count modest for a small CPU model
        hits = self.retr.search(
            q_for_retrieval,
            k_nodes=50, k_final_nodes=8, k_each_node=14, k_final_chunks=6
        )
        logger.event(
            "retrieval_hits",
            n=len(hits),
            sample=[{k: h.get(k) for k in ("chunk_id", "node_id", "section", "pages", "score")} for h in hits[:5]],
        )

        # Use top 4–5 contexts to keep model focused and faster
        contexts = prepare_contexts(hits[:5], max_chars=1000)
        logger.event("contexts_ready", n=len(contexts))

        if not hits:
            if not is_in_scope(question):
                msg = REFUSAL
            else:
                msg = "I don’t have enough information in the provided document to answer that."
            logger.event("no_hits_reply", text=msg)
            yield msg
            logger.close({"status": "no_hits"})
            return

        system_text = (self._system() + "\n\n" + self._style()).strip()
        user_text = self._render_user(question, contexts)
        logger.event("prompt_built", system_len=len(system_text), user_len=len(user_text))

        # Stream draft tokens (UI shows this live)
        chunks: List[str] = []
        logger.event("stream_start")
        async for tok in self._draft_stream(system_text, user_text, history=self.mem.buffer[:]):
            chunks.append(tok)
            # Stream raw tokens for immediate responsiveness
            yield tok
        logger.event("stream_end", tokens=len(chunks))

        # --- Produce a final, cleaned answer and emit a 'final' event ---
        draft = clean_text("".join(chunks).strip())
        final = sanitize_answer(draft, contexts)
        final = clean_text(final)
        logger.event("finalized", draft_len=len(draft), final_len=len(final))

        # Emit structured final event so the UI can replace the last bubble
        yield {"type": "final", "text": final}

        # Memory (summary buffer)
        self.mem.add_turn("user", question)
        self.mem.add_turn("assistant", final)
        self.mem.maybe_summarize(lambda _: "")  # keep your current behavior

        logger.close({"status": "ok", "answer_preview": final[:200]})

    # one-shot convenience (collects stream)
    async def ask_text(self, question: str) -> str:
        out = []
        async for t in self.ask(question):
            out.append(t)
        return "".join(out)
