# src/nys_mfs_agent/splitter.py
from __future__ import annotations
import re
from typing import List

# Very light, deterministic splitter that works offline.

Q_SEP   = re.compile(r"[;\n]+")
HARD_Q  = re.compile(r"\?+")
AND_ALSO = re.compile(r"\b(?:and also|and,? also)\b", re.I)

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _ensure_qmark(s: str) -> str:
    s = _clean(s)
    return s if s.endswith("?") else (s + "?")

def split_queries(text: str, max_parts: int = 6) -> List[str]:
    """
    Split a compound user input into sub-questions.
    Priority:
      (1) Split on question marks.
      (2) Split on ';' or newlines.
      (3) Split on 'and also' if short but clearly two asks.
      (4) If long (>120 chars), split on 'and|also|plus' as a last resort.
    """
    t = _clean(text)
    if not t:
        return []

    # 1) Hard split on '?'
    parts = [p for p in HARD_Q.split(t) if _clean(p)]
    if len(parts) > 1:
        out = []
        for p in parts:
            out.append(_ensure_qmark(p))
        return out[:max_parts]

    # 2) Split on ; and newlines
    parts = [p for p in Q_SEP.split(t) if _clean(p)]
    if len(parts) > 1:
        return [_ensure_qmark(p) for p in parts[:max_parts]]

    # 3) Special-case "and also"
    if AND_ALSO.search(t):
        parts = [p for p in AND_ALSO.split(t) if _clean(p)]
        if len(parts) > 1:
            return [_ensure_qmark(p) for p in parts[:max_parts]]

    # 4) Heuristic split if very long input
    if len(t) > 120:
        maybe = re.split(r"\b(?:and|also|plus)\b", t, maxsplit=3, flags=re.I)
        maybe = [_clean(p) for p in maybe if _clean(p)]
        if len(maybe) > 1:
            return [_ensure_qmark(p) for p in maybe[:max_parts]]

    return [t if t.endswith("?") else t + "?"]
