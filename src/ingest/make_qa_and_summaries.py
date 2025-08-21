# src/ingest/make_qa_and_summaries.py
from __future__ import annotations
import argparse, json, re, time, random, hashlib, os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


# ---------- Ollama helpers ----------
import requests
import os, time

def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

def _list_ollama_models() -> list[str]:
    try:
        r = requests.get(f"{_ollama_host()}/api/tags", timeout=10)
        r.raise_for_status()
        models = r.json().get("models", []) or []
        return [m.get("name","") for m in models if m.get("name")]
    except requests.RequestException as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {_ollama_host()} — start it with `ollama serve`."
        ) from e

def _resolve_ollama_model_tag(requested: str) -> str:
    """
    Return an installed tag to use. Priority:
      1) exact tag match
      2) prefix match (e.g., 'llama3.1:8b-instruct' -> 'llama3.1:8b-instruct-q8_0')
    """
    installed = _list_ollama_models()
    if requested in installed:
        return requested

    # try to find a tag that starts with the requested string
    candidates = [name for name in installed if name.startswith(requested)]
    if candidates:
        # prefer the longest (often includes quant suffix like -q8_0)
        candidates.sort(key=len, reverse=True)
        return candidates[0]

    # no match at all
    msg = (
        f"Ollama model '{requested}' not found.\n"
        f"- Installed: {installed or '[]'}\n"
        f"- Fix: either `ollama pull {requested}` OR pass an installed tag such as one above."
    )
    raise RuntimeError(msg)

def ollama_generate(model: str, prompt: str, system: str | None = None,
                    temperature: float = 0.2, top_p: float = 0.9, num_predict: int = 512,
                    json_mode: bool = False, retries: int = 2) -> str:
    """Call /api/generate (non-stream) to keep things simple."""
    resolved_model = _resolve_ollama_model_tag(model)
    payload = {
        "model": resolved_model,
        "prompt": prompt if system is None else f"<<SYS>>\n{system}\n<</SYS>>\n{prompt}",
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    }
    if json_mode:
        payload["format"] = "json"
    last_err = None
    for _ in range(retries + 1):
        try:
            r = requests.post(f"{_ollama_host()}/api/generate", json=payload, timeout=180)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.2)
    raise RuntimeError(f"Ollama generate failed after retries: {last_err}")


# ---------- IO helpers ----------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- Section builder from chunks ----------

def load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(chunks_path)
    # keep only useful types; headings are used for grouping, but we won’t feed them as content
    return rows

def _hp_to_key(hp: List[str]) -> str:
    return " > ".join([h.strip() for h in hp if h and h.strip()])

def build_sections(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Group chunks by their full heading_path string.
    Returns mapping: section_key -> {
        "heading_path": str,
        "pages": (min_page, max_page),
        "texts": [str, ...],    # content-only (no headings)
        "count": int
    }
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in chunks:
        meta = r.get("metadata", {})
        hp = meta.get("heading_path") or []
        if isinstance(hp, str):
            hp_str = hp
        else:
            hp_str = _hp_to_key(hp)
        if not hp_str:
            # put into a default bucket
            hp_str = "ROOT"

        key = hp_str
        b = buckets.setdefault(key, {"heading_path": hp_str, "pages": [10**9, -1], "texts": [], "count": 0})
        # track pages
        p = meta.get("page_start")
        if isinstance(p, int):
            b["pages"][0] = min(b["pages"][0], p)
            b["pages"][1] = max(b["pages"][1], p)

        # collect content
        block_type = meta.get("block_type")
        txt = r.get("text", "").strip()
        if block_type != "heading" and txt:
            b["texts"].append(txt)
        b["count"] += 1

    # clean pages
    for k, b in list(buckets.items()):
        lo, hi = b["pages"]
        if lo == 10**9 or hi == -1:
            buckets[k]["pages"] = (None, None)
        else:
            buckets[k]["pages"] = (lo, hi)
    return buckets

def _truncate(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[:max_chars]

# ---------- Prompts ----------

SUMMARY_SYSTEM = (
    "You are a legal summarizer. Be precise, faithful, concise, and avoid speculation. "
    "If the supplied text is noisy or non-substantive (e.g., TOC), say 'No substantive content'."
)

SUMMARY_PROMPT = """Summarize the following section from U.S. copyright law.

Heading Path: {heading}
Pages: {pages}

Return **strict JSON** with keys exactly:
  "short"  – 2-3 sentences, plain text.
  "medium" – 6-8 bullet points (use '-' bullets).
  "long"   – 12-15 bullet points (use '-' bullets).

Text:
\"\"\"{text}\"\"\""""

QA_SYSTEM = (
    "You generate high-quality question/answer pairs from legal text. "
    "Answers must be fully grounded in the text; cite page(s) at the end as [p. N] or [pp. X–Y]. "
    "Avoid hallucinations; if unclear, write 'insufficient context'."
)

QA_PROMPT = """From the following excerpt, produce {n} diverse Q/A pairs as a **JSON array**.
Each item must have keys:
  "type": one of ["factual","why","application"]
  "question": a clear, answerable question
  "answer": concise answer grounded in the text and citing page(s) as [p. N] or [pp. X–Y]

Prefer coverage of definitions, obligations, exceptions, factors, and edge cases.

Heading Path: {heading}
Pages: {pages}

Text:
\"\"\"{text}\"\"\""""

# ---------- JSON extraction ----------

def _json_extract(s: str) -> Any:
    """Robust JSON parse: try direct, then extract the outermost JSON block."""
    s = s.strip()
    # direct
    try:
        return json.loads(s)
    except Exception:
        pass
    # fenced code?
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", s, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # fallback: find first { or [ and last } or ]
    start = min([i for i in [s.find("{"), s.find("[")] if i != -1] or [-1])
    end = max(s.rfind("}"), s.rfind("]"))
    if start != -1 and end != -1 and end > start:
        frag = s[start:end+1]
        try:
            return json.loads(frag)
        except Exception:
            pass
    raise ValueError("Could not parse JSON from model output")

# ---------- Generators ----------

def gen_summaries_for_section(model: str, heading: str, pages: Tuple[Optional[int], Optional[int]],
                              text: str, max_chars: int = 3500) -> Dict[str, str]:
    pages_str = f"pp. {pages[0]}–{pages[1]}" if pages[0] and pages[1] else "pp. N/A"
    prompt = SUMMARY_PROMPT.format(heading=heading, pages=pages_str, text=_truncate(text, max_chars))
    out = ollama_generate(model, prompt, system=SUMMARY_SYSTEM, temperature=0.2, top_p=0.9,
                          num_predict=600, json_mode=False)
    try:
        obj = _json_extract(out)
        # basic shape enforcement
        return {
            "short": str(obj.get("short","")).strip(),
            "medium": str(obj.get("medium","")).strip(),
            "long": str(obj.get("long","")).strip(),
        }
    except Exception:
        # degrade gracefully
        return {"short": out.strip(), "medium": "", "long": ""}

def gen_qa_for_section(model: str, heading: str, pages: Tuple[Optional[int], Optional[int]],
                       text: str, n: int = 5, max_chars: int = 2400) -> List[Dict[str, str]]:
    pages_str = f"pp. {pages[0]}–{pages[1]}" if pages[0] and pages[1] else "pp. N/A"
    prompt = QA_PROMPT.format(heading=heading, pages=pages_str, text=_truncate(text, max_chars), n=n)
    out = ollama_generate(model, prompt, system=QA_SYSTEM, temperature=0.3, top_p=0.9,
                          num_predict=700, json_mode=False)
    try:
        arr = _json_extract(out)
        if not isinstance(arr, list): raise ValueError("Expected list")
        cleaned = []
        for item in arr:
            t = str(item.get("type","factual")).strip().lower()
            if t not in ("factual","why","application"): t = "factual"
            q = str(item.get("question","")).strip()
            a = str(item.get("answer","")).strip()
            if q and a:
                cleaned.append({"type": t, "question": q, "answer": a})
        return cleaned
    except Exception:
        # fallback: no usable JSON
        return []

# ---------- Orchestration ----------

def make_data(chunks_path: Path,
              out_dir: Path,
              model: str = "llama3.1:8b-instruct",
              seed: int = 7,
              max_sections: int = 40,
              min_tokens_per_section: int = 400,
              qa_per_section: int = 5) -> Tuple[Path, Path]:
    """
    Build:
      - data/sft/summaries.jsonl
      - data/sft/qa.jsonl
    """
    random.seed(seed)
    chunks = load_chunks(chunks_path)
    sections = build_sections(chunks)

    # score sections by content length; pick top max_sections
    scored: List[Tuple[str,int]] = []
    for k, v in sections.items():
        L = sum(len(x) for x in v["texts"])
        scored.append((k, L))
    scored.sort(key=lambda x: x[1], reverse=True)
    picked = [k for k, L in scored if L >= min_tokens_per_section][:max_sections]

    summaries_rows: List[Dict[str, Any]] = []
    qa_rows: List[Dict[str, Any]] = []

    for i, key in enumerate(picked, 1):
        sec = sections[key]
        heading = sec["heading_path"]
        pages = sec["pages"]
        text = "\n\n".join(sec["texts"])

        print(f"[{i}/{len(picked)}] Section: {heading[:80]} | chars={len(text)} | pages={pages}")

        # Summaries
        summ = gen_summaries_for_section(model, heading, pages, text)
        summaries_rows.append({
            "section": heading,
            "pages": pages,
            "short": summ.get("short",""),
            "medium": summ.get("medium",""),
            "long": summ.get("long",""),
        })

        # Q/A
        qas = gen_qa_for_section(model, heading, pages, text, n=qa_per_section)
        for qa in qas:
            qa_rows.append({
                "section": heading,
                "pages": pages,
                "type": qa["type"],
                "question": qa["question"],
                "answer": qa["answer"],
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    summaries_path = out_dir / "summaries.jsonl"
    qa_path = out_dir / "qa.jsonl"
    _write_jsonl(summaries_path, summaries_rows)
    _write_jsonl(qa_path, qa_rows)
    print(f"[OK] Wrote {len(summaries_rows)} summaries → {summaries_path}")
    print(f"[OK] Wrote {len(qa_rows)} Q/A → {qa_path}")
    return summaries_path, qa_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--out_dir", default="data/sft")
    ap.add_argument("--model", default="llama3.1:8b-instruct")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_sections", type=int, default=40)
    ap.add_argument("--min_tokens_per_section", type=int, default=400)
    ap.add_argument("--qa_per_section", type=int, default=5)
    a = ap.parse_args()
    make_data(Path(a.chunks), Path(a.out_dir), a.model, a.seed, a.max_sections,
              a.min_tokens_per_section, a.qa_per_section)

if __name__ == "__main__":
    main()
