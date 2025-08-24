from __future__ import annotations
import re
from dataclasses import dataclass

TITLE17_REGEX = re.compile(r"\b(17\s*U\.?S\.?C\.?|Title\s*17|copyright|§|\bsection\s+\d+)\b", re.I)

@dataclass
class GuardDecision:
    allow: bool
    reason: str

REFUSAL = (
    "I’m a Title 17 assistant. I can only answer questions that are clearly about U.S. Copyright "
    "law (Title 17). Please rephrase your question to reference the relevant sections or concepts."
)

def guard_title17_scope(user_text: str) -> GuardDecision:
    txt = (user_text or "").strip()
    if not txt:
        return GuardDecision(False, "empty input")
    # obvious jailbreaks / tool-use attempts—keep it light
    if any(k in txt.lower() for k in ["run code", "system prompt", "ignore previous", "browse the web"]):
        return GuardDecision(False, "unsafe/tooling request")
    # allow if seems related to Title 17
    if TITLE17_REGEX.search(txt):
        return GuardDecision(True, "matched title17 pattern")
    # weak signal: common copyright words
    if any(w in txt.lower() for w in ["copyright", "fair use", "dmca", "phonorecord", "compulsory license"]):
        return GuardDecision(True, "matched copyright lexicon")
    return GuardDecision(False, "out-of-scope")
