# src/ingest/md_to_chunks.py
from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ = True
except Exception:
    RAPIDFUZZ = False

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def load_pages_jsonl(pages_path: Optional[Path]) -> List[Dict[str, Any]]:
    if not pages_path or not pages_path.exists():
        return []
    pages = []
    with pages_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                pages.append(json.loads(line))
            except Exception:
                continue
    return pages

# ---------- text cleanup & quality ----------

_SPACED_LETTERS = re.compile(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b")
_DOT_LEADERS = re.compile(r"(?:\.\s){3,}\.?\s*")

def _fix_spaced_letters(s: str) -> str:
    # "c i r c u l a r  9 2" -> "circular 92"
    return _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), s)

def normalize_text(s: str) -> str:
    # Unicode normalize + common fixes
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u0131", "i").replace("\u0130", "I")  # dotless i / Turkish I
    s = _fix_spaced_letters(s)
    s = _DOT_LEADERS.sub(" … ", s)  # collapse dot leaders
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

def is_gibberish(s: str) -> bool:
    """
    Drop blocks that are mostly non-words or OCR noise.
    Simple heuristics: vowelful-word ratio & alphabetic ratio.
    """
    t = s.strip()
    if len(t) < 30:
        return False  # keep short headings/labels
    alpha = sum(c.isalpha() for c in t)
    if alpha == 0:
        return True
    alpha_ratio = alpha / max(1, len(t))
    words = re.findall(r"[A-Za-z]+", t)
    if not words:
        return True
    vowels = set("aeiouAEIOU")
    vowelful = sum(1 for w in words if any(ch in vowels for ch in w))
    vowel_ratio = vowelful / max(1, len(words))
    # noise if both “looks non-alpha” AND “most words have no vowels”
    return (alpha_ratio < 0.6) and (vowel_ratio < 0.5)

# ---------- parsing ----------

def chunk_long_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    # default to 1200 for legal text granularity
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]
    sents = _SENT_SPLIT.split(text)
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                out.append(cur)
            if overlap > 0 and out:
                tail = out[-1][-overlap:]
                cur = (tail + " " + s).strip()
            else:
                cur = s.strip()
    if cur:
        out.append(cur)
    return out

def parse_md(md: str) -> List[Dict[str, Any]]:
    lines = md.splitlines()
    blocks = []
    heading_stack: List[str] = []
    i, N = 0, len(lines)
    in_code = False
    current = {"type": None, "lines": []}

    def flush_current():
        nonlocal current
        if current["type"]:
            text = "\n".join(current["lines"]).strip()
            text = normalize_text(text)
            if text:
                blocks.append({"type": current["type"], "text": text, "heading_path": heading_stack.copy()})
        current = {"type": None, "lines": []}

    while i < N:
        ln = normalize_text(lines[i])

        # code fence
        if ln.startswith("```"):
            if not in_code:
                flush_current()
                in_code = True
                current = {"type": "code", "lines": []}
            else:
                in_code = False
                flush_current()
            i += 1
            continue

        if in_code:
            current["lines"].append(ln)
            i += 1
            continue

        # heading
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush_current()
            level = len(m.group(1))
            title = m.group(2).strip()
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            blocks.append({"type": "heading", "text": title, "heading_path": heading_stack.copy()})
            i += 1
            continue

        # simple table detection
        if "|" in ln and re.search(r"\|\s*-{3,}\s*\|", ln):
            flush_current()
            tbl_lines = [ln]
            i += 1
            while i < N and "|" in normalize_text(lines[i]):
                tbl_lines.append(normalize_text(lines[i]))
                i += 1
            blocks.append({"type": "table", "text": "\n".join(tbl_lines), "heading_path": heading_stack.copy()})
            continue

        # list item
        if re.match(r"^\s*([-*+]|\d+\.)\s+", ln):
            if current["type"] != "list":
                flush_current()
                current = {"type": "list", "lines": []}
            current["lines"].append(ln)
            i += 1
            continue

        # image
        if re.search(r"!\[.*?\]\(.*?\)", ln):
            flush_current()
            blocks.append({"type": "image", "text": ln.strip(), "heading_path": heading_stack.copy()})
            i += 1
            continue

        # blank line -> paragraph boundary
        if not ln.strip():
            flush_current()
            i += 1
            continue

        # paragraph
        if current["type"] not in (None, "para"):
            flush_current()
        current["type"] = "para"
        current["lines"].append(ln)
        i += 1

    flush_current()
    return blocks

# ---------- page alignment ----------

def align_to_pages(text: str, pages: List[Dict[str, Any]], min_score: int = 70) -> Dict[str, Optional[int]]:
    if not pages or not RAPIDFUZZ:
        return {"page_start": None, "page_end": None}
    probe = text[:400]
    best_score, best_page = -1, None
    for rec in pages:
        score = fuzz.partial_ratio(probe, (rec.get("text") or "")[:4000])
        if score > best_score:
            best_score, best_page = score, rec.get("page")
    if best_score < min_score:
        return {"page_start": None, "page_end": None}
    return {"page_start": int(best_page) if best_page else None, "page_end": int(best_page) if best_page else None}

# ---------- TOC detection ----------

def is_toc_block(block_type: str, heading_path: List[str], text: str) -> bool:
    hp = " / ".join(h.lower() for h in heading_path)
    if "contents" in hp:
        return True
    # also treat heavy TOC tables with many pipes or dot leaders as TOC
    if block_type in ("table", "list", "para"):
        if text.count("|") >= 3:
            return True
        if _DOT_LEADERS.search(text):
            return True
    return False

# ---------- public API ----------

def md_to_chunks(md_path: Path,
                 out_path: Path,
                 pages_jsonl: Optional[Path] = None,
                 max_chars: int = 1200,
                 overlap: int = 200,
                 drop_gibberish: bool = True,
                 drop_toc: bool = True,
                 min_align_score: int = 70) -> int:
    """
    Convert structured MD -> JSONL chunks with metadata.
    - Cleans text, drops gibberish, optionally drops TOC blocks.
    - Better page alignment with a minimum fuzzy score.
    """
    doc_id = md_path.stem
    pages = load_pages_jsonl(pages_jsonl) if pages_jsonl else []
    md = read_md(md_path)
    blocks = parse_md(md)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    with out_path.open("w", encoding="utf-8") as f:
        for b in blocks:
            btype = b["type"]
            btext = b["text"]
            hpath = b["heading_path"]

            # Skip TOC if requested
            if drop_toc and is_toc_block(btype, hpath, btext):
                continue

            # Skip gibberish paras/lists (keep headings/tables/images for context)
            if drop_gibberish and btype in ("para", "list") and is_gibberish(btext):
                continue

            # write heading as a tiny chunk
            if btype == "heading":
                meta = {"doc_id": doc_id, "block_type": "heading", "heading_path": hpath}
                meta.update(align_to_pages(btext, pages, min_align_score))
                n_chunks += 1
                f.write(json.dumps({"id": f"{doc_id}-h-{n_chunks}", "text": btext, "metadata": meta}, ensure_ascii=False) + "\n")
                continue

            # split large blocks
            for sub in chunk_long_text(btext, max_chars=max_chars, overlap=overlap):
                meta = {"doc_id": doc_id, "block_type": btype, "heading_path": hpath}
                meta.update(align_to_pages(sub, pages, min_align_score))
                n_chunks += 1
                f.write(json.dumps({"id": f"{doc_id}-{n_chunks}", "text": sub, "metadata": meta}, ensure_ascii=False) + "\n")
    return n_chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", default="data/chunks/chunks.jsonl")
    ap.add_argument("--pages_jsonl", default=None)
    ap.add_argument("--max_chars", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--keep_toc", action="store_true", help="Keep TOC-like blocks (tables/lists under 'Contents')")
    ap.add_argument("--no_drop_gibberish", action="store_true", help="Do not drop gibberish blocks")
    ap.add_argument("--min_align_score", type=int, default=70, help="Min fuzzy score to accept page alignment")
    a = ap.parse_args()
    n = md_to_chunks(
        Path(a.md),
        Path(a.out),
        Path(a.pages_jsonl) if a.pages_jsonl else None,
        a.max_chars,
        a.overlap,
        drop_gibberish=not a.no_drop_gibberish,
        drop_toc=not a.keep_toc,
        min_align_score=a.min_align_score,
    )
    print(f"[OK] Wrote {n} chunks → {a.out}")

if __name__ == "__main__":
    main()
