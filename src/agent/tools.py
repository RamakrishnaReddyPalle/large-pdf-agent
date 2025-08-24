# src/agent/tools.py
from __future__ import annotations

from typing import Optional, Type
import json
import requests
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool

from .retriever import HierBM25CEReranker
from .config import CFG


# -------- Retrieve Tool --------

class RetrieveInput(BaseModel):
    query: str = Field(..., description="User question about Title 17.")
    k_nodes: Optional[int] = Field(None, description="BM25 node candidates before CE rerank.")
    k_final_nodes: Optional[int] = Field(None, description="Final nodes kept after CE rerank.")
    k_each_node: Optional[int] = Field(None, description="Top chunks fetched per selected node.")
    k_final_chunks: Optional[int] = Field(None, description="Final chunks after CE rerank.")


class RetrieveTool(BaseTool):
    """
    LangChain Tool: retrieve_title17
    Returns JSON list of chunk dicts with fields:
    text, pages, section, node_id, chunk_id, score.
    """
    # Pydantic v2-compatible annotated attributes
    name: str = "retrieve_title17"
    description: str = (
        "Fetch relevant Title 17 chunks given a user question. "
        "Returns a JSON list of chunk dicts with fields: text, pages, section, node_id, chunk_id, score."
    )
    args_schema: Type[BaseModel] = RetrieveInput

    # Private state (kept out of Pydantic)
    _retr: HierBM25CEReranker | None = PrivateAttr(default=None)

    def _ensure(self) -> None:
        if self._retr is None:
            self._retr = HierBM25CEReranker()

    def _run(
        self,
        query: str,
        k_nodes: Optional[int] = None,
        k_final_nodes: Optional[int] = None,
        k_each_node: Optional[int] = None,
        k_final_chunks: Optional[int] = None,
    ) -> str:
        self._ensure()
        hits = self._retr.search(
            query,
            k_nodes or CFG.k_nodes,
            k_final_nodes or CFG.k_final_nodes,
            k_each_node or CFG.k_each_node,
            k_final_chunks or CFG.k_final_chunks,
        )
        return json.dumps(hits, ensure_ascii=False)

    async def _arun(
        self,
        query: str,
        k_nodes: Optional[int] = None,
        k_final_nodes: Optional[int] = None,
        k_each_node: Optional[int] = None,
        k_final_chunks: Optional[int] = None,
    ) -> str:
        # sync under the hood; keep API uniform
        return self._run(query, k_nodes, k_final_nodes, k_each_node, k_final_chunks)


# -------- Summarize Tool (Ollama HTTP) --------

class SummarizeInput(BaseModel):
    text: str = Field(..., description="Text to summarize.")
    max_bullets: int = Field(5, ge=3, le=8, description="Number of bullets (3–8).")


class SummarizeTool(BaseTool):
    """
    LangChain Tool: summarize_context
    Uses local Ollama (CFG.ollama_summarizer) via HTTP.
    """
    name: str = "summarize_context"
    description: str = (
        "Summarize a long context into 3–8 bullets, ≤25 words each. "
        "Uses local Ollama (default model from CFG.ollama_summarizer)."
    )
    args_schema: Type[BaseModel] = SummarizeInput

    base_url: str = "http://localhost:11434"
    model: str = CFG.ollama_summarizer

    def _run(self, text: str, max_bullets: int = 5) -> str:
        safe_text = (text or "").strip()
        if not safe_text:
            return "[SummarizeTool] EMPTY_INPUT"

        prompt = (
            "You are a precise legal summarizer.\n"
            f"Write exactly {max_bullets} bullets; each bullet ≤ 25 words.\n"
            "Preserve literal page refs like [pp. X–Y] if present. Do not invent citations.\n"
            "Only output bullets; no prefaces or apologies.\n\n"
            "### INPUT\n"
            f"{safe_text}\n\n"
            "### OUTPUT\n"
            "- "
        )
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=120,
            )
            r.raise_for_status()
            js = r.json()
            return (js.get("response", "") or "").strip()
        except Exception as e:
            return f"[SummarizeTool error] {e}"

    async def _arun(self, text: str, max_bullets: int = 5) -> str:
        return self._run(text, max_bullets)
