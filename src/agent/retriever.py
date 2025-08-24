from __future__ import annotations
from pathlib import Path
import json, re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import CFG

_rx = re.compile(r"[A-Za-z0-9_]+")

def _tok(s: str): return _rx.findall((s or "").lower())

class HierBM25CEReranker:
    """
    Loads hierarchy & chunks; BM25 on nodes -> CE rerank nodes ->
    BM25 within node -> CE rerank final chunks; returns top chunks w/ meta.
    """
    def __init__(self,
                 graph_dir: Path | None = None,
                 chunks_dir: Path | None = None,
                 reranker_dir: Path | None = None):
        self.graph_dir = Path(graph_dir or CFG.graph_dir)
        self.chunks_dir = Path(chunks_dir or CFG.chunks_dir)
        self.reranker_dir = Path(reranker_dir or CFG.reranker_dir)

        # hierarchy
        self.hier = json.loads((self.graph_dir / "hierarchy.json").read_text(encoding="utf-8"))
        self.nodes = self.hier["nodes"]
        node_records = [json.loads(l) for l in open(self.graph_dir / "node_texts.jsonl", "r", encoding="utf-8")]
        self.node_text_by_id = {r["node_id"]: r["text"] for r in node_records}
        self.node_name_by_id = {r["node_id"]: r["name"] for r in node_records}
        self.node_ids = list(self.node_text_by_id.keys())
        self.node_texts = [self.node_text_by_id[n] for n in self.node_ids]

        # node->chunk_ids
        self.node2chunks = {nd["id"]: (nd.get("chunk_ids") or []) for nd in self.nodes}

        # chunk corpus
        self.chunk_text, self.chunk_pages, self.chunk_sec = {}, {}, {}
        for fp in sorted(self.chunks_dir.glob("*.jsonl")):
            for line in open(fp, "r", encoding="utf-8"):
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                cid = obj.get("id") or obj.get("chunk_id")
                if not cid: continue
                txt = obj.get("text") or obj.get("content") or ""
                if not txt.strip(): continue
                self.chunk_text[cid]  = txt
                self.chunk_pages[cid] = obj.get("pages") or []
                self.chunk_sec[cid]   = obj.get("section") or ""

        # lexical node index
        self.bm25_nodes = BM25Okapi([_tok(t) for t in self.node_texts])

        # CE reranker
        self.ce = CrossEncoder(str(self.reranker_dir), device="cpu")

    def search(self, query: str,
               k_nodes: int | None = None, k_final_nodes: int | None = None,
               k_each_node: int | None = None, k_final_chunks: int | None = None):
        k_nodes = k_nodes or CFG.k_nodes
        k_final_nodes = k_final_nodes or CFG.k_final_nodes
        k_each_node = k_each_node or CFG.k_each_node
        k_final_chunks = k_final_chunks or CFG.k_final_chunks

        # (1) BM25 on nodes
        scores = self.bm25_nodes.get_scores(_tok(query))
        idxs = sorted(range(len(self.node_ids)), key=lambda i: scores[i], reverse=True)[:k_nodes]
        cand_nodes = [(self.node_ids[i], self.node_texts[i]) for i in idxs]

        # (2) CE rerank nodes
        ce_scores = self.ce.predict([[query, t] for _, t in cand_nodes])
        order = sorted(range(len(cand_nodes)), key=lambda j: ce_scores[j], reverse=True)[:k_final_nodes]
        top_node_ids = [cand_nodes[j][0] for j in order]

        # (3) BM25 within each node
        final_cands = []
        for nid in top_node_ids:
            cids = [c for c in self.node2chunks.get(nid, []) if c in self.chunk_text]
            if not cids: continue
            bm25_local = BM25Okapi([_tok(self.chunk_text[c]) for c in cids])
            sc = bm25_local.get_scores(_tok(query))
            loc = sorted(range(len(cids)), key=lambda i: sc[i], reverse=True)[:k_each_node]
            for i in loc:
                cid = cids[i]
                final_cands.append({
                    "chunk_id": cid,
                    "node_id": nid,
                    "node_name": self.node_name_by_id.get(nid, nid),
                    "text": self.chunk_text[cid],
                    "pages": self.chunk_pages.get(cid) or [],
                    "section": self.chunk_sec.get(cid) or "",
                })

        if not final_cands: return []

        # (4) CE rerank final chunks
        ce2 = self.ce.predict([[query, c["text"]] for c in final_cands])
        for c, s in zip(final_cands, ce2): c["score"] = float(s)
        final = sorted(final_cands, key=lambda x: x["score"], reverse=True)[:k_final_chunks]
        return final
