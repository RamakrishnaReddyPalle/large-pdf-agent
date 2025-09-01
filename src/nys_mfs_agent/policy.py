import re
from typing import List, Dict

from .config import CFG

# --- citation parsing ---
CIT_PAT   = re.compile(r"\[([^\[\]]+)\]")
PP_PAT    = re.compile(r"pp\.\s*([0-9,\-\–\s]+)", re.I)
CHUNK_PAT = re.compile(r"chunk\s+([A-Za-z0-9_\-\.]+)", re.I)

# --- light non-English (CJK etc.) matcher for spillover cleanup ---
CJK_PAT = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")

# --- JSON-ish keys we often see in legacy SFTs ---
JSONY_KEYS = re.compile(
    r"\b(?:short|medium|long|bullet\d?|bullets?|q|a|answer|citations?)\s*:",
    re.I
)

def _pages_from_pp_block(block: str) -> set[int]:
    m = PP_PAT.search(block)
    pages = set()
    if not m:
        return pages
    nums = m.group(1)
    for part in re.split(r"[,\s]+", nums.strip()):
        if not part:
            continue
        if "-" in part or "–" in part:
            a, b = re.split(r"[-–]", part)
            try:
                a = int(a); b = int(b)
                lo, hi = (a, b) if a <= b else (b, a)
                pages.update(range(lo, hi + 1))
            except:
                pass
        else:
            try:
                pages.add(int(part))
            except:
                pass
    return pages

def allowed_pages_from_contexts(contexts: List[str]) -> set[int]:
    allow = set()
    for c in contexts:
        for blk in CIT_PAT.findall(c):
            allow |= _pages_from_pp_block(blk)
    return allow

def strip_noise(txt: str) -> str:
    txt = re.sub(r"<\|/?(system|user|assistant)\|>", "", txt, flags=re.I)
    # strip weird control tokens and repeated whitespace
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return re.sub(r"\s{2,}", " ", txt).strip()

def _strip_jsonish(ans: str) -> str:
    # Remove leading/trailing braces and quotes if it looks like an object dump
    if re.search(r"^\s*\{.*\}\s*$", ans, flags=re.S):
        ans = re.sub(r"^\s*\{|\}\s*$", "", ans, flags=re.S)
    # Remove obvious JSON-like key scaffolding
    ans = JSONY_KEYS.sub("", ans)
    # Drop leftover quotes used as structure
    ans = ans.replace('\"', '').replace("”", "").replace("“", "")
    # Remove stray commas that were separators
    ans = re.sub(r"\s*,\s*(\n|$)", r"\1", ans)
    return ans.strip()

def _normalize_language(ans: str) -> str:
    # Remove non-English spillover blocks (very light touch)
    return CJK_PAT.sub("", ans)

def _to_lines(ans: str) -> List[str]:
    # split on newlines; also break on "• " and "- " midlines
    ans = ans.replace("•", "\n•").replace("- ", "\n- ")
    parts = [x.strip() for x in ans.split("\n")]
    return [p for p in parts if p]

def _normalize_bullets(lines: List[str]) -> List[str]:
    out = []
    for ln in lines:
        if ln.startswith(("•", "-", "–", "*")):
            content = ln.lstrip("•-*– ").strip()
            if content:
                out.append(f"- {content}")
        else:
            out.append(ln)
    return out

def _ensure_shape(ans: str) -> str:
    """
    Enforce:
      1) one-sentence direct answer (first non-bullet line),
      2) then 3–6 hyphen bullets.
    If there is no non-bullet line, convert the first bullet to the direct line.
    """
    lines = _to_lines(ans)
    lines = _normalize_bullets(lines)

    non_bullets = [l for l in lines if not l.startswith("- ")]
    bullets     = [l for l in lines if l.startswith("- ")]

    direct = ""
    if non_bullets:
        # take the first non-bullet as direct answer
        direct = non_bullets[0]
    elif bullets:
        # lift first bullet to direct answer (remove "- ")
        direct = bullets[0][2:].strip()
        bullets = bullets[1:]

    # limit bullets
    if len(bullets) > 6:
        bullets = bullets[:6]

    # If we somehow have no bullets but many lines, turn remaining into bullets
    if not bullets and len(lines) > 1:
        for l in lines[1:]:
            if l.strip():
                bullets.append(f"- {l.strip()}")
                if len(bullets) >= 3:
                    break

    # If still too few bullets (<3) but we have leftover sentences, promote them
    if len(bullets) < 3:
        extras = [l for l in lines if l not in [direct] + bullets and l.strip()]
        for e in extras:
            bullets.append(f"- {e.strip()}")
            if len(bullets) >= 3:
                break

    # final stitch
    final_lines = []
    if direct:
        final_lines.append(direct)
    final_lines.extend(bullets)
    return "\n".join(final_lines).strip()

def has_json_scaffold(txt: str) -> bool:
    if re.search(r'"\s*:\s*"', txt):
        return True
    if re.search(r"^\s*\{.*\}\s*$", txt, flags=re.S):
        return True
    if JSONY_KEYS.search(txt):
        return True
    return False

def sanitize_answer(ans: str, contexts: List[str]) -> str:
    # 1) strip system/user wrappers etc.
    ans = strip_noise(ans)
    # 2) remove non-English spillover
    ans = _normalize_language(ans)
    # 3) flatten JSON-like scaffolding if present
    if has_json_scaffold(ans):
        ans = _strip_jsonish(ans)
    # 4) enforce target shape
    ans = _ensure_shape(ans)

    # 5) citation sanity: remove citations that reference pages not included in contexts
    allow = allowed_pages_from_contexts(contexts)
    cites = CIT_PAT.findall(ans)
    used  = False
    bads  = []
    for blk in cites:
        if PP_PAT.search(blk):
            used = True
            pages = _pages_from_pp_block(blk)
            if allow and pages and not pages.issubset(allow):
                bads.append(blk)
        elif CHUNK_PAT.search(blk):
            used = True

    for b in bads:
        ans = ans.replace(f"[{b}]", "").strip()

    # 6) if no citation used anywhere, append the first context's tail (if any)
    if not used and contexts:
        tail = ""
        for line in reversed(contexts[0].splitlines()):
            s = line.strip()
            if s.startswith("[") and s.endswith("]"):
                tail = s
                break
        if tail:
            # Prefer to attach to the last bullet if present; else to the direct answer
            if "\n- " in ans:
                ans = ans.rsplit("\n- ", 1)
                if isinstance(ans, list):
                    head, last = ans
                    ans = head + "\n- " + last.rstrip(". ") + f" {tail}"
                else:
                    ans = ans.rstrip() + f" {tail}"
            else:
                ans = ans.rstrip() + f" {tail}"

    return ans

DOMAIN_HINTS = [
    "workers' compensation", "fee schedule", "cpt", "relative value", "conversion factor",
    "evaluation and management", "radiology", "pathology", "physical medicine", "surgery",
    "new york state", "nys", "wcb"
]

def is_in_scope(question: str) -> bool:
    q = (question or "").lower()
    return any(h in q for h in DOMAIN_HINTS)

REFUSAL = ("Sorry—I can only answer questions about the OFFICIAL NEW YORK STATE WORKERS’ "
           "COMPENSATION – MEDICAL FEE SCHEDULE. Please ask about rules, sections, coding, "
           "billing, or documentation from this fee schedule.")
