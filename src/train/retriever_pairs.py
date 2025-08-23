# pos/neg mining
from __future__ import annotations
import json, random, argparse
from pathlib import Path
from typing import List, Dict, Any
import chromadb

def _read_jsonl(p: Path) -> List[Dict[str,Any]]:
    return [json.loads(l) for l in open(p, "r", encoding="utf-8") if l.strip()]

def mine_pairs(index_dir: Path,
               chunks_path: Path,
               qa_path: Path,
               out_path: Path,
               n_per_q: int = 4,
               k_candidates: int = 12,
               seed: int = 7) -> Path:
    random.seed(seed)

    # Load QA questions as queries
    qa = _read_jsonl(qa_path)
    queries = []
    for r in qa:
        q = (r.get("question") or "").strip()
        if q:
            queries.append({"q": q, "section": r.get("section","")})

    # Open Chroma
    client = chromadb.PersistentClient(path=str(index_dir))
    # If you used collection_name==doc_id, grab the first collection
    cols = client.list_collections()
    if not cols:
        raise RuntimeError("No Chroma collections found.")
    coll = client.get_collection(cols[0].name)

    # Build pairs
    pairs = []
    for item in queries:
        res = coll.query(query_texts=[item["q"]], n_results=k_candidates,
                         include=["documents","metadatas","distances"])
        docs   = res["documents"][0]
        metas  = res["metadatas"][0]

        # Heuristic: top-1 as positive if it's not a heading/TOC
        pos_idx = None
        for i,(d,m) in enumerate(zip(docs, metas)):
            if (m.get("block_type") != "heading") and "|" not in d:
                pos_idx = i; break
        if pos_idx is None:  # fall back to 0
            pos_idx = 0

        pos_doc = docs[pos_idx]
        pos_meta = metas[pos_idx]

        # Negatives: next few non-identical docs
        negs = []
        for i,(d,m) in enumerate(zip(docs, metas)):
            if i == pos_idx: continue
            if m.get("id") == pos_meta.get("id"): continue
            if m.get("block_type") == "heading": continue
            negs.append((d,m))
            if len(negs) >= n_per_q: break

        # Save one positive + N negatives
        if negs:
            pairs.append({
                "query": item["q"],
                "positive": {"text": pos_doc, "meta": pos_meta},
                "negatives": [{"text": d, "meta": m} for d,m in negs],
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(pairs)} pairs → {out_path}")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    ap.add_argument("--qa", default="data/sft/qa.v3b.jsonl")
    ap.add_argument("--out", default="data/pairs/pairs.jsonl")
    ap.add_argument("--n_per_q", type=int, default=4)
    ap.add_argument("--k_candidates", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    mine_pairs(Path(a.index), Path(a.chunks), Path(a.qa), Path(a.out),
               n_per_q=a.n_per_q, k_candidates=a.k_candidates, seed=a.seed)

if __name__ == "__main__":
    main()
