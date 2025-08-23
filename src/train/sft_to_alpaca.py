from __future__ import annotations
import json, argparse
from pathlib import Path

def to_alpaca(src: Path, dst: Path):
    out = []
    for line in open(src, "r", encoding="utf-8"):
        ex = json.loads(line)
        msgs = ex.get("messages", [])
        resp = ex.get("response", "").strip()
        if not msgs or not resp:
            continue
        # take the last user message as the instruction (simple & robust for our SFT)
        user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user" and m.get("content")), "")
        if not user: 
            continue
        out.append({
            "instruction": user.strip(),
            "input": "",
            "output": resp.strip()
        })
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(out)} alpaca examples → {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/sft/train.jsonl")
    ap.add_argument("--dst", default="data/sft/alpaca.train.jsonl")
    a = ap.parse_args()
    to_alpaca(Path(a.src), Path(a.dst))
