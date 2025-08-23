from __future__ import annotations
import json, argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from sentence_transformers import CrossEncoder, InputExample, losses
from torch.utils.data import DataLoader
import torch

def _read_jsonl(p: Path) -> List[Dict[str,Any]]:
    return [json.loads(l) for l in open(p, "r", encoding="utf-8") if l.strip()]

def load_pair_examples(pairs_path: Path, max_negs: int = 3) -> List[InputExample]:
    pairs = _read_jsonl(pairs_path)
    exs: List[InputExample] = []
    for r in pairs:
        q = r["query"]
        pos = r["positive"]["text"]
        exs.append(InputExample(texts=[q, pos], label=1.0))
        for neg in r["negatives"][:max_negs]:
            exs.append(InputExample(texts=[q, neg["text"]], label=0.0))
    return exs

def train_reranker(pairs_path: Path,
                   out_dir: Path,
                   model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                   batch_size: int = 8,
                   epochs: int = 2,
                   lr: float = 1e-5,
                   seed: int = 7) -> Path:
    torch.manual_seed(seed)

    exs = load_pair_examples(pairs_path)
    train_loader = DataLoader(exs, shuffle=True, batch_size=batch_size)

    model = CrossEncoder(model_name, device="cpu")  # CPU-friendly
    loss_fn = losses.SoftmaxLoss(model)

    model.fit(
        train_dataloader=train_loader,
        loss_fct=loss_fn,
        epochs=epochs,
        warmup_steps=0,
        optimizer_params={"lr": lr},
        output_path=str(out_dir),
        show_progress_bar=True,
    )
    print(f"[OK] saved reranker → {out_dir}")
    return out_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs/pairs.jsonl")
    ap.add_argument("--out_dir", default="outputs/reranker")
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    train_reranker(Path(a.pairs), Path(a.out_dir), a.model, a.batch_size, a.epochs, a.lr)

if __name__ == "__main__":
    main()
