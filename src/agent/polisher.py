from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
import json, requests
from .config import CFG
from .prompts import load_prompts

def _stream_ollama(prompt: str, model: str, base_url: str, timeout: int = 300) -> Iterable[str]:
    r = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True, "options": {"temperature": 0.2}},
        timeout=timeout,
        stream=True,
    )
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            js = json.loads(line)
            tok = js.get("response")
            if tok:
                yield tok
        except Exception:
            continue

class Polisher:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or str(CFG.ollama_polisher)
        self.base_url = str(base_url or CFG.ollama_base_url)
        self.prompts = load_prompts()

    def build_prompt(self, user_query: str, summary: str, subanswers: list[dict]) -> str:
        ptxt = self.prompts.get("polisher") or ""
        return (
            ptxt
            + "\n\nconversation_summary:\n"
            + (summary or "")
            + "\n\nuser_query:\n"
            + user_query.strip()
            + "\n\nsubanswers (JSON):\n"
            + json.dumps(subanswers, ensure_ascii=False, indent=2)
            + "\n\nProduce the final formatted answer now."
        )

    def stream_polish(self, user_query: str, summary: str, subanswers: list[dict]):
        prompt = self.build_prompt(user_query, summary, subanswers)
        yield from _stream_ollama(prompt, self.model, self.base_url)
