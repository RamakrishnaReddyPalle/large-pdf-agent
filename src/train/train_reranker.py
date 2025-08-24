# src/train/train_reranker.py
from __future__ import annotations
import argparse, json, os, random, math
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, CrossEncoder
from tqdm import tqdm


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def build_ce_examples(pairs_path: Path) -> List[InputExample]:
    """
    Consume pairs like:
      { "query": str, "positive": str, "negatives": [str, ...], ... }

    Create CE binary examples:
      [query, positive] -> label 1.0
      [query, negative] -> label 0.0
    """
    items = read_jsonl(pairs_path)
    examples: List[InputExample] = []
    for r in items:
        q = (r.get("query") or "").strip()
        pos = (r.get("positive") or "").strip()
        negs = r.get("negatives") or []
        if not q or not pos:
            continue
        examples.append(InputExample(texts=[q, pos], label=1.0))
        for n in negs:
            if isinstance(n, str) and n.strip():
                examples.append(InputExample(texts=[q, n.strip()], label=0.0))
    return examples


def prep_dev_rerank(dev_path: Path) -> List[Tuple[str, List[str], int]]:
    """
    For dev evaluation, return list of (query, [cands], pos_index)
    where cands = [positive] + negatives and pos_index == 0.
    """
    triples = []
    if not dev_path.exists():
        return triples
    for r in read_jsonl(dev_path):
        q = (r.get("query") or "").strip()
        pos = (r.get("positive") or "").strip()
        negs = [n for n in (r.get("negatives") or []) if isinstance(n, str) and n.strip()]
        if q and pos:
            cands = [pos] + negs
            triples.append((q, cands, 0))
    return triples


@torch.no_grad()
def evaluate_ce(model: CrossEncoder, dev_triples: List[Tuple[str, List[str], int]], batch_size: int = 32) -> Dict[str, float]:
    """
    Compute simple ranking metrics: MRR@10, Recall@10, Top1 accuracy.
    """
    if not dev_triples:
        return {}

    mrr10 = 0.0
    recall10 = 0.0
    top1 = 0.0
    total = 0

    for q, cands, pos_idx in tqdm(dev_triples, desc="[dev] rerank", leave=False):
        pairs = [[q, c] for c in cands]

        # Batch predict to keep RAM low
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            s = model.predict(batch)
            # predict returns numpy array
            scores.extend(s.tolist())

        # higher score = more relevant
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)

        # rank of the known positive (at index 0)
        rank = order.index(pos_idx) + 1  # 1-based
        if rank <= 10:
            mrr10 += 1.0 / rank
            recall10 += 1.0
        if rank == 1:
            top1 += 1.0
        total += 1

    return {
        "mrr@10": mrr10 / total,
        "recall@10": recall10 / total,
        "top1": top1 / total,
        "n": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pairs", type=str, required=True)
    ap.add_argument("--dev_pairs",   type=str, default="")
    ap.add_argument("--base_ce",     type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--local_base_dir", type=str, default="", help="Path to local CE folder (preferred on Windows/offline)")
    ap.add_argument("--out_dir",     type=str, required=True)

    ap.add_argument("--epochs",      type=int, default=2)
    ap.add_argument("--batch_size",  type=int, default=8)
    ap.add_argument("--lr",          type=float, default=2e-5)
    ap.add_argument("--max_len",     type=int, default=384)
    ap.add_argument("--seed",        type=int, default=7)
    ap.add_argument("--warmup_ratio",type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_pairs = Path(args.train_pairs)
    dev_pairs   = Path(args.dev_pairs) if args.dev_pairs else None
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load base model (CPU)
    base_name = args.local_base_dir if args.local_base_dir and Path(args.local_base_dir).exists() else args.base_ce
    print(f"[info] base CE: {base_name}")
    model = CrossEncoder(
        base_name,
        num_labels=1,
        max_length=args.max_len,
        device="cpu",  # CPU-only
    )

    # Build datasets
    train_examples = build_ce_examples(train_pairs)
    assert train_examples, f"No training examples parsed from {train_pairs}"
    print(f"[info] train examples: {len(train_examples)}")

    train_loader = DataLoader(
        train_examples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows-safe
        pin_memory=False,
    )

    # Warmup steps
    total_steps = math.ceil(len(train_loader) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"[info] total_steps={total_steps} | warmup_steps={warmup_steps}")

    # Optional dev evaluation triples
    dev_triples: List[Tuple[str, List[str], int]] = []
    if dev_pairs and dev_pairs.exists():
        dev_triples = prep_dev_rerank(dev_pairs)
        print(f"[info] dev queries for rerank eval: {len(dev_triples)}")

    # Train
    model.fit(
        train_dataloader=train_loader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=str(out_dir),             # will save tokenizer + model
        optimizer_params={"lr": args.lr},
        evaluation_steps=0,                   # custom eval after epochs
        show_progress_bar=True,
        use_amp=False,                        # CPU
    )

    # Save again explicitly (some versions only save at end of epoch)
    model.save(str(out_dir))

    # Epoch-end eval (after training)
    if dev_triples:
        metrics = evaluate_ce(model, dev_triples, batch_size=max(8, args.batch_size))
        print("[dev metrics]", metrics)

    print(f"[OK] saved reranker → {out_dir}")


if __name__ == "__main__":
    main()
