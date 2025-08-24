# src/agent/prompts.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .config import CFG

@dataclass
class PromptBundle:
    system: str
    style_rules: str
    answer_with_citations: str
    # NEW: keep compatibility with planner/polisher modules
    planner: str = ""
    polisher: str = ""

    # allow dict-like .get("planner") calls used by new modules
    def get(self, key: str, default=None):
        return getattr(self, key, default)

def _read(p: Path, default: str = "") -> str:
    try:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return default.strip()

def load_prompts() -> PromptBundle:
    pdir = CFG.prompts_dir
    return PromptBundle(
        system=_read(pdir / "system.txt", "You are Title 17 Assistant."),
        style_rules=_read(pdir / "style_rules.txt", "Be concise and cite pages."),
        answer_with_citations=_read(
            pdir / "answer_with_citations.txt",
            "Answer:\n<CONTENT>\n\nCitations:\n(none found in provided contexts)"
        ),
        # NEW: these files may or may not exist yet; default to empty strings
        planner=_read(pdir / "planner.txt", ""),
        polisher=_read(pdir / "polisher.txt", ""),
    )

def build_rag_prompt(question: str, contexts: list[dict], pb: PromptBundle) -> str:
    """
    contexts: list of dicts with keys: text, pages (list[int]), section (str), node_id, chunk_id
    """
    ctx_blocks = []
    for i, c in enumerate(contexts, 1):
        pages = c.get("pages") or []
        pages_str = ""
        if pages:
            if len(pages) == 1:
                pages_str = f"[p. {pages[0]}]"
            else:
                pages_str = f"[pp. {pages[0]}–{pages[-1]}]"
        title = c.get("section") or c.get("node_id") or c.get("chunk_id") or f"CTX-{i}"
        ctx_text = (c.get("text") or "").strip()
        ctx_blocks.append(f"[CTX {i}] {title} {pages_str}\n{ctx_text}")

    ctxs = "\n\n".join(ctx_blocks) if ctx_blocks else "(no contexts)"
    prompt = (
        f"System:\n{pb.system}\n\n"
        f"Style Rules:\n{pb.style_rules}\n\n"
        f"Task:\n{pb.answer_with_citations}\n\n"
        f"CONTEXTS:\n{ctxs}\n\n"
        f"User: {question}\nAssistant:"
    )
    return prompt
