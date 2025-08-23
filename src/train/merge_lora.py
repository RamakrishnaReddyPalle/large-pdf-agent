# src/train/merge_lora.py
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="HF repo id or LOCAL DIR of base")
    ap.add_argument("--lora_dir",   required=True, help="Directory containing LoRA adapter (the folder saved by cpu_lora_hf.py)")
    ap.add_argument("--out_dir",    required=True, help="Directory to save merged full model")
    ap.add_argument("--local_files_only", action="store_true", default=False)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device_map={"": "cpu"},
    )

    peft_model = PeftModel.from_pretrained(
        base,
        args.lora_dir,
        local_files_only=True,
    )
    merged = peft_model.merge_and_unload()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))
    tok.save_pretrained(str(out))

    print("[OK] merged model saved to:", out)


if __name__ == "__main__":
    main()