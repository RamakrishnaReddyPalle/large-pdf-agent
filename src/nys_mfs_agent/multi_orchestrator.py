# src/nys_mfs_agent/multi_orchestrator.py
from __future__ import annotations
from typing import AsyncIterator, List

from .orchestrator import Agent
from .splitter import split_queries
from .policy import is_in_scope, REFUSAL
from .combiner import combine_answers
from .logger import SessionLogger
from .sanitize import clean_text


class MultiAgent:
    def __init__(self):
        self.worker = Agent()

    async def ask(self, question: str, logger: SessionLogger | None = None) -> AsyncIterator[str]:
        """
        Streamed, multi-turn: splits the question when needed, streams each sub-answer,
        then streams a combined answer at the end. Each sub-answer and the final combined
        answer are sanitized to avoid JSON-like artifacts or code blocks.
        """
        # Ensure we have a logger
        if logger is None:
            logger = SessionLogger()
        logger.event("multi_question", text=question)

        # Domain guardrail
        if not is_in_scope(question):
            logger.event("out_of_scope", text=question)
            yield REFUSAL
            logger.close({"status": "refused"})
            return

        # Split into sub-queries
        parts = [p.strip() for p in split_queries(question) if p and p.strip()]
        logger.event("split", parts=parts)

        # Single question → just proxy to the worker (keeps streaming + logging)
        if len(parts) == 1:
            async for tok in self.worker.ask(parts[0], logger=logger):
                yield tok
            # Agent.ask() closes the logger in its success/failure branches.
            return

        # Multi-question → stream each separately; collect full texts
        subanswers: List[str] = []
        for i, pq in enumerate(parts, 1):
            header = f"\n### Part {i}/{len(parts)}: {pq}\n"
            logger.event("subquestion_start", idx=i, text=pq)
            yield header

            collected: List[str] = []
            async for tok in self.worker.ask(pq, logger=logger):
                collected.append(tok)
                yield tok
            ans = clean_text("".join(collected).strip())
            subanswers.append(f"Q{i}: {pq}\nA{i}: {ans}")
            logger.event("subquestion_end", idx=i, tokens=len(collected))

        # Combined answer (streamed as a single chunk at the end)
        logger.event("combine_start", n=len(subanswers))
        yield "\n\n### Combined answer\n"
        combined = await combine_answers(question, subanswers)
        combined = clean_text(combined)
        yield combined
        logger.event("combine_end", combined_len=len(combined))
        logger.close({"status": "ok", "n_parts": len(parts)})
