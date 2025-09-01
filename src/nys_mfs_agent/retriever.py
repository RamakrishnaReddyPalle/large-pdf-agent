# src/nys_mfs_agent/retriever.py
from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# -------------------------
# helpers
# -------------------------

def _tok(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (s or "").lower())

def _resolve_graph_dir(gdir: Path) -> Path:
    gdir = Path(gdir)
    if (gdir / "node_texts.jsonl").exists() and (gdir / "hierarchy.json").exists():
        return gdir
    if (gdir / "graph" / "node_texts.jsonl").exists():
        return gdir / "graph"
    raise FileNotFoundError(f"node_texts.jsonl not found under {gdir}")

def _query_expand(q: str) -> str:
    """
    Very small domain synonym expansion to stabilize BM25.
    This is intentionally conservative (no external deps).
    """
    ql = (q or "").lower()
    add: list[str] = []
    if "e/m" in ql or "evaluation and management" in ql or "evaluation & management" in ql:
        add += ["E/M", "Evaluation and Management"]
    if "rvu" in ql or "relative value" in ql:
        add += ["RVU", "relative value units"]
    if "radiology" in ql:
        add += ["imaging", "diagnostic studies"]
    if "physical medicine" in ql:
        add += ["therapy", "modalities", "PT"]
    if "pathology" in ql or "laboratory" in ql:
        add += ["lab", "pathology and laboratory"]
    if "surgery" in ql:
        add += ["surgical", "operative"]
    if "introduction and general guidelines" in ql or "general guidelines" in ql:
        add += ["ground rules", "introduction"]
    if add:
        return q + " " + " ".join(set(add))
    return q

def _extract_pages_from_meta(meta: dict) -> list[int]:
    p = meta.get("pages") or meta.get("page")
    if isinstance(p, list):
        out = []
        for v in p:
            try:
                out.append(int(v))
            except:
                pass
        return out
    if isinstance(p, (int, float)):
        v = int(p)
        return [v, v]
    for k in ("page_start", "start_page", "pageStart"):
        if k in meta:
            try:
                a = int(meta[k])
                b = int(meta.get("page_end", meta.get("end_page", meta.get("pageEnd", a))))
                return [a, b]
            except:
                pass
    for k in ("page_idx", "page_no", "pageNumber"):
        if k in meta:
            try:
                v = int(meta[k])
                return [v, v]
            except:
                pass
    return []

def _normalize_section(obj: dict) -> str:
    meta = obj.get("metadata") or {}
    hp = meta.get("heading_path")
    if isinstance(hp, list):
        hp = " > ".join([str(x).strip() for x in hp if str(x).strip()])
    elif isinstance(hp, str):
        hp = hp.strip()
    else:
        hp = ""
    return hp or (obj.get("section") or "")

def _norm_pages(pages: list[int]) -> list[int]:
    try:
        return sorted(set(int(x) for x in pages))
    except Exception:
        return pages or []

def _hash_text(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()

# -------------------------
# loaders
# -------------------------

def load_graph(graph_dir: Path):
    graph_dir = _resolve_graph_dir(graph_dir)
    node_records = [json.loads(x) for x in open(graph_dir / "node_texts.jsonl", "r", encoding="utf-8")]
    hier = json.loads((graph_dir / "hierarchy.json").read_text(encoding="utf-8"))
    for n in hier.get("nodes", []):
        if "id" not in n and "node_id" in n:
            n["id"] = n["node_id"]
    hier_by_id = {n["id"]: n for n in hier["nodes"]}

    node2chunks: dict[str, list[str]] = {}
    nm_path = graph_dir / "node_members.jsonl"
    if nm_path.exists():
        for line in open(nm_path, "r", encoding="utf-8"):
            row = json.loads(line)
            nid = row.get("node_id")
            cids = row.get("chunk_ids") or []
            if nid:
                node2chunks[nid] = cids
    else:
        for n in hier["nodes"]:
            node2chunks[n["id"]] = n.get("chunk_ids") or []
    return node_records, hier_by_id, node2chunks

def load_chunks(chunks_dir: Path) -> dict[str, dict]:
    id2chunk: dict[str, dict] = {}
    for fp in sorted(Path(chunks_dir).glob("*.jsonl")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                cid = obj.get("id") or obj.get("chunk_id")
                txt = obj.get("text") or obj.get("content") or ""
                if not cid or not isinstance(txt, str) or not txt.strip():
                    continue
                meta = obj.get("metadata") or {}
                id2chunk[cid] = {
                    "id": cid,
                    "text": txt,
                    "section": _normalize_section(obj),
                    "pages": _norm_pages(_extract_pages_from_meta(meta)),
                }
    return id2chunk

# -------------------------
# context renderer
# -------------------------

def prepare_contexts(hits: list[dict], max_chars: int = 1100) -> list[str]:
    """
    Turn ranked chunk hits into compact context strings and attach a proper tail citation.
    De-dups near-identical chunks (by text hash) to avoid flooding.
    """
    out = []
    seen_hash = set()
    for h in hits:
        body = (h.get("text") or "").strip()
        if not body:
            continue
        hsh = _hash_text(body[:1000])
        if hsh in seen_hash:
            continue
        seen_hash.add(hsh)

        sec = (h.get("section") or "").strip()
        pages = _norm_pages(h.get("pages") or [])
        cid = h.get("chunk_id") or "unknown"
        head = sec if sec else "(untitled)"

        if pages:
            if len(pages) == 1:
                tail = f"[pp. {pages[0]}]"
            else:
                tail = f"[pp. {', '.join(map(str, pages))}]"
        else:
            tail = f"[chunk {cid}]"

        out.append(f"{head}\n{body[:max_chars]}\n{tail}")
    return out

# -------------------------
# retrievers
# -------------------------

class GraphRetriever:
    """BM25 over node texts -> CE node rerank -> BM25 within node -> CE chunk rerank."""
    def __init__(self, chunks_dir: str | Path, graph_dir: str | Path, reranker_dir: str | Path):
        self.chunks_dir = Path(chunks_dir)
        self.graph_dir  = Path(graph_dir)
        self.reranker   = CrossEncoder(str(reranker_dir), device="cpu")

        self.node_records, self.hier_by_id, self.node2chunks = load_graph(self.graph_dir)
        self.node_texts  = [r["text"] for r in self.node_records]
        self.node_ids    = [r["node_id"] for r in self.node_records]
        self.node_names  = [r["name"]    for r in self.node_records]

        self.bm25_nodes  = BM25Okapi([_tok(t) for t in self.node_texts])
        self.id2chunk    = load_chunks(self.chunks_dir)

    def _best_nodes(self, query: str, k_nodes: int, k_final_nodes: int) -> list[int]:
        qx = _query_expand(query)
        scores = self.bm25_nodes.get_scores(_tok(qx))
        idxs   = sorted(range(len(self.node_texts)), key=lambda i: scores[i], reverse=True)[:k_nodes]
        if not idxs:
            return []
        ce = self.reranker.predict([[query, self.node_texts[i]] for i in idxs])
        reranked = sorted(zip(idxs, ce), key=lambda x: x[1], reverse=True)[:k_final_nodes]
        return [i for i, _ in reranked]

    def _best_chunks_from_nodes(self, query: str, node_idxs: list[int],
                                k_each_node: int, k_final_chunks: int) -> list[dict]:
        qx = _query_expand(query)
        cands: list[dict] = []
        for ni in node_idxs:
            node_id = self.node_ids[ni]
            node_name = self.node_names[ni] if ni < len(self.node_names) else node_id
            member_ids = [cid for cid in self.node2chunks.get(node_id, []) if cid in self.id2chunk]
            if not member_ids:
                continue
            sub_texts = [f"{node_name}\n{self.id2chunk[cid]['text']}" for cid in member_ids]
            bm25_local = BM25Okapi([_tok(t) for t in sub_texts])
            local_scores = bm25_local.get_scores(_tok(qx))
            order = sorted(range(len(member_ids)), key=lambda i: local_scores[i], reverse=True)[:k_each_node]
            for j in order:
                cid = member_ids[j]
                ch  = self.id2chunk[cid]
                cands.append({
                    "chunk_id": cid,
                    "node_id": node_id,
                    "node_name": node_name,
                    "text": ch["text"],
                    "pages": ch["pages"],
                    "section": ch["section"],
                })
        if not cands:
            return []
        ce = self.reranker.predict([[query, c["text"]] for c in cands])
        for c, s in zip(cands, ce):
            c["score"] = float(s)
        cands.sort(key=lambda x: x["score"], reverse=True)
        return cands[:k_final_chunks]

    def search(self, query: str, k_nodes=60, k_final_nodes=10, k_each_node=16, k_final_chunks=10) -> list[dict]:
        node_idxs = self._best_nodes(query, k_nodes=k_nodes, k_final_nodes=k_final_nodes)
        if not node_idxs:
            return []
        return self._best_chunks_from_nodes(query, node_idxs, k_each_node, k_final_chunks)

class FlatRetriever:
    """BM25 over all chunks -> CE rerank."""
    def __init__(self, chunks_dir: str | Path, reranker_dir: str | Path):
        self.reranker = CrossEncoder(str(reranker_dir), device="cpu")
        self.id2chunk = load_chunks(Path(chunks_dir))
        # stable order for ids
        self.ids      = sorted(self.id2chunk.keys())
        self.corpus   = [self.id2chunk[i]["text"] for i in self.ids]
        self.bm25     = BM25Okapi([_tok(t) for t in self.corpus])

    def search(self, query: str, k_bm25=80, k_final=10) -> list[dict]:
        qx = _query_expand(query)
        scores = self.bm25.get_scores(_tok(qx))
        idxs   = sorted(range(len(self.corpus)), key=lambda i: scores[i], reverse=True)[:k_bm25]
        cands  = [{"id": self.ids[i], **self.id2chunk[self.ids[i]]} for i in idxs]
        ce     = self.reranker.predict([[query, c["text"]] for c in cands])
        ranked = sorted(zip(cands, ce), key=lambda x: x[1], reverse=True)[:k_final]
        out = []
        for ch, sc in ranked:
            out.append({
                "chunk_id": ch["id"],
                "node_id": None, "node_name": None,
                "text": ch["text"], "pages": ch["pages"], "section": ch["section"],
                "score": float(sc),
            })
        return out

class HybridRetriever:
    """
    Union strategy by default:
      - get hier candidates
      - get flat candidates
      - union by chunk_id
      - global CE re-rank across the union
    Fallback to hier-only behavior if you pass mode='hier_only'.
    """
    def __init__(self, chunks_dir: str | Path, graph_dir: str | Path, reranker_dir: str | Path):
        self.hier = GraphRetriever(chunks_dir, graph_dir, reranker_dir)
        self.flat = FlatRetriever(chunks_dir, reranker_dir)
        self.final_ce = CrossEncoder(str(reranker_dir), device="cpu")

    def _union(self, a: list[dict], b: list[dict]) -> list[dict]:
        by_id: dict[str, dict] = {}
        for x in a + b:
            cid = x.get("chunk_id")
            if cid and cid not in by_id:
                by_id[cid] = x
        return list(by_id.values())

    def search(self, query: str, mode: str = "union", **kw) -> list[dict]:
        k_final = int(kw.get("k_final_chunks", 10))

        # get hier
        h_hits = self.hier.search(
            query,
            k_nodes=kw.get("k_nodes", 60),
            k_final_nodes=kw.get("k_final_nodes", 10),
            k_each_node=kw.get("k_each_node", 16),
            k_final_chunks=min(k_final, kw.get("k_final_chunks", 10)),
        )

        if mode == "hier_only":
            return h_hits

        # get flat
        f_hits = self.flat.search(query, k_bm25=100, k_final=max(10, k_final))

        # union + final CE rerank
        union = self._union(h_hits, f_hits)
        if not union:
            return []

        ce = self.final_ce.predict([[query, c["text"]] for c in union])
        for c, s in zip(union, ce):
            c["score"] = float(s)
        union.sort(key=lambda x: x["score"], reverse=True)
        return union[:k_final]
