# build SFT dataset
# src/train/sft_build.py
from __future__ import annotations
import argparse, json, hashlib, math
from pathlib import Path
from typing import Dict, Any, List, Tuple

def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def _write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _hash_key(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)

def _mk_summary_examples(summ: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Produce 3 chat examples (short/medium/long)."""
    heading = summ["section"]
    pages = summ.get("pages", [None, None])
    pp = f"[pp. {pages[0]}–{pages[1]}]" if pages[0] and pages[1] else "[pp. N/A]"

    templates = [
        ("short",   "Give a brief 2–3 sentence summary."),
        ("medium",  "Summarize as 6–8 concise bullet points."),
        ("long",    "Summarize as 12–15 detailed bullet points."),
    ]
    out = []
    for key, instr in templates:
        answer = (summ.get(key) or "").strip()
        if not answer: continue
        msg = [
            {"role": "system", "content": "You are a helpful legal assistant. Be precise and cite pages if known."},
            {"role": "user", "content": f"Summarize the section:\nHeading: {heading}\n{instr} End with {pp}."},
        ]
        out.append({"messages": msg, "response": answer})
    return out

def _mk_qa_examples(qa: Dict[str, Any]) -> Dict[str, Any]:
    """One chat example per Q/A pair."""
    heading = qa["section"]
    pages = qa.get("pages", [None, None])
    pp = f"[pp. {pages[0]}–{pages[1]}]" if pages[0] and pages[1] else "[pp. N/A]"
    q = qa["question"].strip()
    a = qa["answer"].strip()
    msg = [
        {"role": "system", "content": "You answer strictly based on the document content and cite pages."},
        {"role": "user", "content": f"({qa['type']}) {q}\nPrefer concise, correct answers. End with {pp}."},
    ]
    return {"messages": msg, "response": a}

def build_sft(summaries_path: Path, qa_path: Path,
              out_dir: Path,
              train_frac: float = 0.85,
              dev_frac: float = 0.075,
              test_frac: float = 0.075) -> Tuple[Path, Path, Path]:
    assert abs((train_frac + dev_frac + test_frac) - 1.0) < 1e-6
    summaries = _read_jsonl(summaries_path) if summaries_path.exists() else []
    qa = _read_jsonl(qa_path) if qa_path.exists() else []

    examples: List[Dict[str, Any]] = []
    for s in summaries:
        examples.extend(_mk_summary_examples(s))
    for r in qa:
        examples.append(_mk_qa_examples(r))

    # deterministic split by hash of user+response (or heading)
    keyed = []
    for ex in examples:
        user = next((m["content"] for m in ex["messages"] if m["role"]=="user"), "")
        key = _hash_key(user + "||" + ex["response"])
        keyed.append((key, ex))
    keyed.sort(key=lambda x: x[0])

    n = len(keyed)
    n_train = int(math.floor(n * train_frac))
    n_dev = int(math.floor(n * dev_frac))
    n_test = n - n_train - n_dev

    train = [x[1] for x in keyed[:n_train]]
    dev = [x[1] for x in keyed[n_train:n_train+n_dev]]
    test = [x[1] for x in keyed[n_train+n_dev:]]

    out_dir.mkdir(parents=True, exist_ok=True)
    p_train = out_dir / "train.jsonl"
    p_dev = out_dir / "dev.jsonl"
    p_test = out_dir / "test.jsonl"
    _write_jsonl(p_train, train)
    _write_jsonl(p_dev, dev)
    _write_jsonl(p_test, test)

    print(f"[OK] SFT examples: total={n} | train={len(train)} dev={len(dev)} test={len(test)}")
    return p_train, p_dev, p_test

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", default="data/sft/summaries.jsonl")
    ap.add_argument("--qa", default="data/sft/qa.jsonl")
    ap.add_argument("--out_dir", default="data/sft")
    ap.add_argument("--train_frac", type=float, default=0.85)
    ap.add_argument("--dev_frac", type=float, default=0.075)
    ap.add_argument("--test_frac", type=float, default=0.075)
    a = ap.parse_args()
    build_sft(Path(a.summaries), Path(a.qa), Path(a.out_dir), a.train_frac, a.dev_frac, a.test_frac)

if __name__ == "__main__":
    main()
