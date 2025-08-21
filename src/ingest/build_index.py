# src/ingest/build_index.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

def _batched(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

def _maybe_bge_passage_prefix(embed_model: str, texts: list[str], use_prompt: bool = True) -> list[str]:
    """
    BGE v1.5 models recommend adding 'passage: ' for corpus and 'query: ' for queries.
    We add 'passage: ' here at index time when using bge-* v1.5 models.
    """
    name = embed_model.lower()
    if use_prompt and ("bge-" in name and "v1.5" in name):
        return [f"passage: {t}" for t in texts]
    return texts

def _flatten_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma requires metadata values to be scalars and non-None.
    - Drop keys with None values.
    - Convert lists (e.g., heading_path) to strings.
    - JSON-stringify any unexpected types as a fallback.
    """
    flat: Dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue  # <-- IMPORTANT: drop None entirely
        if isinstance(v, list):
            flat[k] = " > ".join(str(x) for x in v)
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            try:
                flat[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                flat[k] = str(v)
    return flat

def _embed_input_with_headings(doc_text: str, meta: Dict[str, Any], max_levels: int = 3) -> str:
    """
    Prefix the last N heading levels to the text for embedding only.
    Stored document remains the original text.
    """
    hp = meta.get("heading_path")
    # If md_to_chunks wrote heading_path as list:
    if isinstance(hp, list) and hp:
        prefix = " > ".join(hp[-max_levels:])
        return f"{prefix}\n{doc_text}"
    # If already stored as string for some reason:
    if isinstance(hp, str) and hp.strip():
        parts = [p.strip() for p in hp.split(">") if p.strip()]
        if parts:
            prefix = " > ".join(parts[-max_levels:])
            return f"{prefix}\n{doc_text}"
    return doc_text

def build_index(chunks_path: Path,
                persist: Path = Path("data/index"),
                collection: Optional[str] = None,
                embed_model: str = "BAAI/bge-base-en-v1.5",
                batch_size: int = 64,
                bge_use_prompt: bool = True,
                reset: bool = False) -> str:
    """
    Build a persistent Chroma index from JSONL chunks using a SentenceTransformers model.
    - Flattens list-valued metadata (e.g., heading_path) to strings and drops None values.
    - Embeds a heading-prefixed variant of the text (improves retrieval).
    - Stores original text in documents.
    """
    records: List[Dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if not records:
        raise RuntimeError("No chunks found. Did you run md_to_chunks?")

    first_doc_id = records[0]["metadata"]["doc_id"]
    collection_name = collection or first_doc_id

    client = PersistentClient(path=str(persist))
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[INFO] Dropped existing collection '{collection_name}'")
        except Exception:
            pass
    coll = client.get_or_create_collection(collection_name)

    model = SentenceTransformer(embed_model, device="cpu")

    total = len(records)
    added = 0
    for batch in _batched(records, batch_size):
        # Embedding input: (headings + text), then add BGE 'passage:' prompt
        embed_texts = [_embed_input_with_headings(r["text"], r["metadata"]) for r in batch]
        embed_texts = _maybe_bge_passage_prefix(embed_model, embed_texts, use_prompt=bge_use_prompt)
        embs = model.encode(embed_texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        ids = [r["id"] for r in batch]
        docs = [r["text"] for r in batch]  # store original text
        metas = [_flatten_meta(r["metadata"]) for r in batch]  # <-- drop None, flatten lists

        coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        added += len(batch)
        print(f"Indexed {added}/{total}")

    print(f"[OK] Chroma collection '{collection_name}' built at {persist}")
    print(f"[INFO] Embed model: {embed_model} | BGE passage prompt: {bge_use_prompt}")
    return collection_name

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/chunks/chunks.jsonl")
    p.add_argument("--persist", default="data/index")
    p.add_argument("--collection", default=None)
    p.add_argument("--embed_model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--no_bge_prompt", action="store_true", help="Disable 'passage:' prefix for BGE v1.5")
    p.add_argument("--reset", action="store_true", help="Drop existing collection before indexing")
    a = p.parse_args()
    name = build_index(
        Path(a.chunks),
        Path(a.persist),
        a.collection,
        a.embed_model,
        a.batch_size,
        bge_use_prompt=not a.no_bge_prompt,
        reset=a.reset,
    )
    print(f"[OK] Chroma collection '{name}' at {a.persist}")

if __name__ == "__main__":
    main()
