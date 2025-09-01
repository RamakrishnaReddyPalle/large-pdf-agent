
from .config import CFG
from pathlib import Path

def load_prompt(name: str) -> str:
    p = CFG.prompts_dir / f"{name}.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()

def render(name: str, **kw) -> str:
    s = load_prompt(name)
    for k, v in kw.items():
        s = s.replace("{"+k+"}", str(v))
    return s
