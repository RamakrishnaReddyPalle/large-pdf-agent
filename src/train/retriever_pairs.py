# src/train/retriever_pairs.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Set

import tqdm
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)

# Matches things like:
# [pp. 88–92], [pp. 189-201], [pp 10-11], [p. 67], [pages 12–14]
PP_RE = re.compile(
    r"\[(?:pp?\.?|pages?)\s*([0-9]{1,4})\s*(?:[-–—]\s*([0-9]{1,4}))?\s*\]",
    flags=re.IGNORECASE,
)

# Optional: extract "Heading: ..." lines (not required, but handy for meta)
HEADING_RE = re.compile(r"^Heading:\s*(.+)$", flags=re.IGNORECASE | re.MULTILINE)

def read_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def write_jsonl(rows: Iterable[Dict[str, Any]], p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

def load_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    """
    Expect *.jsonl with fields:
      - 'text' (or 'content')
      - optional 'page' (int) or 'pages' (list[int])
      - optional 'section'
    """
    rows: List[Dict[str, Any]] = []
    for fp in sorted(chunks_dir.glob("*.jsonl")):
        for idx, r in enumerate(read_jsonl(fp)):
            text = r.get("text") or r.get("content") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            page_set: Set[int] = set()
            page = r.get("page")
            if isinstance(page, int):
                page_set.add(page)
            pages = r.get("pages")
            if isinstance(pages, list):
                for p in pages:
                    if isinstance(p, int):
                        page_set.add(p)
            rows.append({
                "id": r.get("id") or f"{fp.name}:{idx}",
                "text": text,
                "section": r.get("section") or "",
                "pages": page_set,
            })
    return rows

def build_bm25(corpus_texts: List[str]) -> Tuple[BM25Okapi, List[List[str]]]:
    tokenized = [tokenize(t) for t in corpus_texts]
    return BM25Okapi(tokenized), tokenized

def parse_pages_from_text(s: str) -> List[int]:
    """
    Extract page numbers from patterns like:
    [pp. 88–92], [pp. 189-201], [p. 67], [pages 12–14]
    Returns deduplicated sorted list of ints.
    """
    pages: Set[int] = set()
    for m in PP_RE.finditer(s or ""):
        a = m.group(1)
        b = m.group(2)
        try:
            start = int(a)
            if b is not None:
                end = int(b)
                if end < start:
                    start, end = end, start
                for p in range(start, end + 1):
                    pages.add(p)
            else:
                pages.add(start)
        except Exception:
            continue
    return sorted(pages)

def overlap_pages(a: Iterable[int], b: Iterable[int]) -> bool:
    sa, sb = set(a), set(b)
    return len(sa & sb) > 0

def extract_from_row(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize any SFT row into:
      { 'query': str, 'answer': str, 'pages': List[int], 'section': str }

    Supports:
      A) Your chat format:
         - messages: list[ {role, content} ] (system + user)
         - response: assistant text
         Pages are parsed from the user content (e.g., "[pp. 88–92]").
      B) Chat format with assistant message in messages (fallback).
      C) Alpaca: instruction/input/output.
      D) QA: question/answer/pages/section.
    """
    query, answer, pages, section = "", "", [], ""

    # A) Your chat format (messages + response)
    msgs = ex.get("messages")
    resp = ex.get("response")
    if isinstance(msgs, list):
        # last user message becomes the query
        last_user = ""
        for m in msgs:
            role = (m.get("role") or "").lower()
            content = (m.get("content") or "").strip()
            if role == "user":
                last_user = content
        if last_user:
            query = last_user
            pages = parse_pages_from_text(last_user)
            # Optional section from "Heading: ..." inside query
            m = HEADING_RE.search(last_user)
            if m:
                section = (m.group(1) or "").strip()
        # answer from 'response' field if present
        if isinstance(resp, str) and resp.strip():
            answer = resp.strip()
        else:
            # B) fallback to last assistant in messages
            last_assistant = ""
            for m in msgs:
                role = (m.get("role") or "").lower()
                if role == "assistant":
                    last_assistant = (m.get("content") or "").strip()
            answer = last_assistant
        if query:
            return {"query": query, "answer": answer, "pages": pages, "section": section}

    # C) Alpaca-style
    if isinstance(ex.get("instruction"), str) or isinstance(ex.get("output"), str):
        instr = (ex.get("instruction") or "").strip()
        inp   = (ex.get("input") or "").strip()
        query = f"{instr}\n{inp}".strip() if inp else instr
        answer = (ex.get("output") or "").strip()
        pages = parse_pages_from_text(query)
        m = HEADING_RE.search(query)
        if m:
            section = (m.group(1) or "").strip()
        if query:
            return {"query": query, "answer": answer, "pages": pages, "section": section}

    # D) QA-style
    if isinstance(ex.get("question"), str) or isinstance(ex.get("answer"), str):
        query = (ex.get("question") or "").strip()
        answer = (ex.get("answer") or "").strip()
        if isinstance(ex.get("pages"), list):
            pages = [p for p in ex["pages"] if isinstance(p, int)]
        else:
            pages = parse_pages_from_text(query)
        section = (ex.get("section") or "").strip()
        if query:
            return {"query": query, "answer": answer, "pages": pages, "section": section}

    return {"query": "", "answer": "", "pages": [], "section": ""}

def mine_pairs(
    sft_path: Path,
    chunks: List[Dict[str, Any]],
    topk: int = 30,
    negatives_per_query: int = 4,
    prefer_page_overlap: bool = True,
    min_query_len: int = 12,
) -> List[Dict[str, Any]]:
    bm25, _ = build_bm25([c["text"] for c in chunks])

    stats = {"total": 0, "kept": 0, "skipped_short": 0,
             "picked_page": 0, "picked_substr": 0, "picked_top1": 0}

    out: List[Dict[str, Any]] = []

    for ex in tqdm.tqdm(list(read_jsonl(sft_path)), desc=f"Mining from {sft_path.name}"):
        stats["total"] += 1
        parsed = extract_from_row(ex)
        q = parsed["query"]
        ans = parsed["answer"]
        gold_pages = set(parsed["pages"])

        if not isinstance(q, str) or len(q.strip()) < min_query_len:
            stats["skipped_short"] += 1
            continue

        q_tokens = tokenize(q)
        scores = bm25.get_scores(q_tokens)
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:max(1, topk)]

        pos_idx = None

        # (1) prefer page overlap if we have gold pages and chunks have pages
        if prefer_page_overlap and gold_pages:
            for i in order:
                if chunks[i]["pages"] and overlap_pages(chunks[i]["pages"], gold_pages):
                    pos_idx = i
                    stats["picked_page"] += 1
                    break

        # (2) answer substring heuristic
        if pos_idx is None and isinstance(ans, str) and len(ans.strip()) >= 12:
            ans_low = ans.lower()
            for i in order:
                if ans_low in chunks[i]["text"].lower():
                    pos_idx = i
                    stats["picked_substr"] += 1
                    break

        # (3) fallback top-1
        if pos_idx is None and order:
            pos_idx = order[0]
            stats["picked_top1"] += 1

        pos_text = chunks[pos_idx]["text"]
        negs = []
        for i in order:
            if i == pos_idx:
                continue
            negs.append(chunks[i]["text"])
            if len(negs) >= negatives_per_query:
                break

        out.append({
            "query": q,
            "positive": pos_text,
            "negatives": negs,
            "meta": {
                "section": parsed["section"],
                "gold_pages": sorted(list(gold_pages)),
                "pos_chunk_id": chunks[pos_idx]["id"],
                "pos_reason": ("page" if stats["picked_page"] else
                               ("substr" if stats["picked_substr"] else "top1"))
            }
        })
        stats["kept"] += 1

    print(
        f"[stats] kept={stats['kept']}/{stats['total']} | skipped_short={stats['skipped_short']} | "
        f"page={stats['picked_page']} | substr={stats['picked_substr']} | top1={stats['picked_top1']}"
    )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", type=str, default="data/chunks")
    ap.add_argument("--train_jsonl", type=str, default="data/sft/train.jsonl")
    ap.add_argument("--dev_jsonl",   type=str, default="data/sft/dev.jsonl")
    ap.add_argument("--out_dir",     type=str, default="data/pairs")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--neg_per_q", type=int, default=4)
    ap.add_argument("--min_query_len", type=int, default=12)
    args = ap.parse_args()

    chunks = load_chunks(Path(args.chunks_dir))
    assert chunks, f"No chunks found under {args.chunks_dir}"
    print(f"[info] loaded {len(chunks)} chunks")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if Path(args.train_jsonl).exists():
        train_pairs = mine_pairs(Path(args.train_jsonl), chunks, args.topk, args.neg_per_q, True, args.min_query_len)
        write_jsonl(train_pairs, out_dir / "train.pairs.jsonl")
        print(f"[OK] wrote {len(train_pairs)} train pairs → {out_dir/'train.pairs.jsonl'}")

    if Path(args.dev_jsonl).exists():
        dev_pairs = mine_pairs(Path(args.dev_jsonl), chunks, args.topk, args.neg_per_q, True, args.min_query_len)
        write_jsonl(dev_pairs, out_dir / "dev.pairs.jsonl")
        print(f"[OK] wrote {len(dev_pairs)} dev pairs → {out_dir/'dev.pairs.jsonl'}")

if __name__ == "__main__":
    main()
