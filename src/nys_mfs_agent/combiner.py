# src/nys_mfs_agent/combiner.py
from __future__ import annotations
from typing import List

from .prompts import render, load_prompt
from .llms_async import load_core_llm, astream, render_chat
from .policy import sanitize_answer

_COMBINER_PROMPT = """\
Merge the sub-answers below into ONE consistent answer to the user’s question.

RULES
- English only.
- Do NOT use JSON, code blocks, or tables.
- OUTPUT SHAPE:
  1) One-sentence direct answer first.
  2) Then 3–6 short hyphen bullets (≤ 30 words), with citations at the end of lines they support.
- Preserve any bracketed citations exactly as written (e.g., [pp. 77, 78] or [chunk …]).
- Do not invent page numbers or sections.
- Remove duplication and contradictions across sub-answers.

Question:
{question}

Sub-answers:
{subs}

Final answer:
"""

async def combine_answers(question: str, subanswers: List[str]) -> str:
    core = load_core_llm()

    # Prefer a dedicated combiner prompt if you have one in configs; else fall back to the strong inline one
    try:
        prompt_text = render("combiner",
                             question=question,
                             subanswers="\n\n---\n\n".join(subanswers))
    except Exception:
        prompt_text = _COMBINER_PROMPT.format(
            question=question,
            subs="\n\n---\n\n".join(subanswers)
        )

    system_text = "You merge the sub-answers into a clean, user-friendly final answer while preserving citations exactly."
    chat = render_chat(core.tokenizer, system_text, prompt_text, history=[])

    out_chunks: List[str] = []
    async for tok in astream(core, chat, max_new_tokens=360, temperature=0.0):
        out_chunks.append(tok)

    combined = "".join(out_chunks).strip()
    # final normalization pass (same as single-answer path)
    combined = sanitize_answer(combined, contexts=[])
    return combined
