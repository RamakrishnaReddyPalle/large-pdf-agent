# hybrid + rerank
# src/serve/retriever.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def _tok(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (s or "").lower())


def load_graph(graph_dir: Path) -> tuple[list[dict], dict]:
    """
    Returns:
      node_records: list of {node_id, name, level, text}
      hier_by_id  : {node_id -> node_dict(with chunk_ids)}
    """
    node_records = [json.loads(x) for x in open(graph_dir / "node_texts.jsonl", "r", encoding="utf-8")]
    hier = json.loads((graph_dir / "hierarchy.json").read_text(encoding="utf-8"))
    hier_by_id = {n["id"]: n for n in hier["nodes"]}
    return node_records, hier_by_id


def load_chunks(chunks_dir: Path) -> dict[str, dict]:
    """
    Loads every *.jsonl chunk into a dict keyed by chunk_id.
    Chunk schema is flexible; we try common fields: id/chunk_id, text/content, pages/page/metadata.pages, section.
    """
    id2chunk: dict[str, dict] = {}
    for fp in sorted(chunks_dir.glob("*.jsonl")):
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
                pages = (
                    obj.get("pages")
                    or obj.get("page")
                    or (obj.get("metadata") or {}).get("pages")
                    or []
                )
                if isinstance(pages, int):
                    pages = [pages]
                id2chunk[cid] = {
                    "id": cid,
                    "text": txt,
                    "section": obj.get("section") or "",
                    "pages": pages if isinstance(pages, list) else [],
                }
    return id2chunk


def prepare_contexts(hits: list[dict], max_chars: int = 1200) -> list[str]:
    """
    Turn ranked chunk hits into compact context strings.
    We prepend section, pages (if any), and chunk id so the model can preserve citations.
    """
    out = []
    for h in hits:
        sec = (h.get("section") or "").strip()
        pages = h.get("pages") or []
        cid = h.get("chunk_id") or "unknown"
        head = sec if sec else "(untitled)"
        tail_cite = ""
        if pages:
            tail_cite = f"[pp. {', '.join(map(str, pages))}]"
        else:
            # fallback cite via chunk id; lets the model keep a stable reference
            tail_cite = f"[chunk {cid}]"
        body = (h.get("text") or "").strip()
        body = body[:max_chars]
        out.append(f"{head}\n{body}\n{tail_cite}")
    return out


class GraphRetriever:
    """
    BM25 over hierarchy nodes -> CE rerank nodes -> BM25 within-node (with slight heading boost) -> CE final rerank.
    """
    def __init__(
        self,
        chunks_dir: str | Path,
        graph_dir: str | Path,
        reranker_dir: str | Path,
    ):
        self.chunks_dir = Path(chunks_dir)
        self.graph_dir  = Path(graph_dir)
        self.reranker   = CrossEncoder(str(reranker_dir), device="cpu")

        self.node_records, self.hier_by_id = load_graph(self.graph_dir)
        self.node_texts  = [r["text"] for r in self.node_records]
        self.node_ids    = [r["node_id"] for r in self.node_records]
        self.node_names  = [r["name"]    for r in self.node_records]

        # BM25 over nodes
        self.bm25_nodes = BM25Okapi([_tok(t) for t in self.node_texts])

        # All chunks (global pool), subset per node when needed
        self.id2chunk = load_chunks(self.chunks_dir)

    def _best_nodes(self, query: str, k_nodes: int, k_final_nodes: int) -> list[int]:
        # Lexical candidate nodes
        scores = self.bm25_nodes.get_scores(_tok(query))
        idxs   = sorted(range(len(self.node_texts)), key=lambda i: scores[i], reverse=True)[:k_nodes]
        if not idxs:
            return []
        # Cross-encode rerank
        pairs  = [[query, self.node_texts[i]] for i in idxs]
        ce     = self.reranker.predict(pairs)
        reranked = sorted(zip(idxs, ce), key=lambda x: x[1], reverse=True)[:k_final_nodes]
        return [i for i, _ in reranked]

    def _best_chunks_from_nodes(
        self, query: str, node_idxs: list[int], k_each_node: int, k_final_chunks: int
    ) -> list[dict]:
        cand_entries: list[dict] = []

        for ni in node_idxs:
            node_id = self.node_ids[ni]
            node    = self.hier_by_id.get(node_id) or {}
            cids    = node.get("chunk_ids") or []
            sub_ids = [cid for cid in cids if cid in self.id2chunk]
            if not sub_ids:
                continue

            # Slight heading boost: include node name with chunk text for local BM25
            node_name = self.node_names[ni] if ni < len(self.node_names) else ""
            sub_texts = [f"{node_name}\n{self.id2chunk[cid]['text']}" for cid in sub_ids]

            bm25_local   = BM25Okapi([_tok(t) for t in sub_texts])
            local_scores = bm25_local.get_scores(_tok(query))
            order = sorted(range(len(sub_ids)), key=lambda i: local_scores[i], reverse=True)[:k_each_node]
            for j in order:
                cid = sub_ids[j]
                ch  = self.id2chunk[cid]
                cand_entries.append({
                    "chunk_id": cid,
                    "node_id": node_id,
                    "node_name": node_name or node_id,
                    "text": ch["text"],
                    "pages": ch.get("pages") or [],
                    "section": ch.get("section") or "",
                })

        if not cand_entries:
            return []

        # CE rerank final candidates
        ce = self.reranker.predict([[query, c["text"]] for c in cand_entries])
        for c, s in zip(cand_entries, ce):
            c["score"] = float(s)
        cand_entries = sorted(cand_entries, key=lambda x: x["score"], reverse=True)[:k_final_chunks]
        return cand_entries

    def search(
        self,
        query: str,
        k_nodes: int = 50,          # slightly higher recall
        k_final_nodes: int = 8,
        k_each_node: int = 14,
        k_final_chunks: int = 6,
    ) -> list[dict]:
        node_idxs = self._best_nodes(query, k_nodes=k_nodes, k_final_nodes=k_final_nodes)
        if not node_idxs:
            return []
        return self._best_chunks_from_nodes(
            query, node_idxs=node_idxs, k_each_node=k_each_node, k_final_chunks=k_final_chunks
        )


# Back-compat alias (some callers import this name)
HierBM25CEReranker = GraphRetriever

__all__ = [
    "GraphRetriever",
    "HierBM25CEReranker",
    "load_graph",
    "load_chunks",
    "prepare_contexts",
]
