# src/train/cpu_lora_hf.py
from __future__ import annotations
import os, json, argparse, random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def format_alpaca_example(instr: str, inp: str, out: str, eos: str) -> str:
    instr = (instr or "").strip()
    inp   = (inp or "").strip()
    out   = (out or "").strip()
    if inp:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{out}{eos}"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instr}\n\n"
            f"### Response:\n{out}{eos}"
        )


def make_train_dataset(tokenizer, jsonl_path: Path, max_len: int):
    ds = load_dataset("json", data_files=str(jsonl_path))["train"]
    eos = tokenizer.eos_token or "</s>"

    def _tok(batch):
        instrs = batch["instruction"]
        inputs = batch.get("input") or [""] * len(instrs)
        outs   = batch["output"]
        texts  = [
            format_alpaca_example(i, x, o, eos)
            for i, x, o in zip(instrs, inputs, outs)
        ]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        enc["labels"] = [ids[:] for ids in enc["input_ids"]]
        return enc

    ds = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="HF repo id OR LOCAL DIRECTORY")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--num_threads", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--local_files_only", action="store_true", default=False)
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.num_threads))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Tokenizer (CPU / offline-friendly)
    tok = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Dataset
    train_ds = make_train_dataset(tok, Path(args.train_jsonl), args.max_seq_len)

    # Base model (CPU)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        device_map={"": "cpu"},
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # LoRA config (Qwen/LLaMA-style)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tok, mlm=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=10_000_000,   # effectively "don't mid-save"
        save_total_limit=1,
        bf16=False,
        fp16=False,
        dataloader_num_workers=0,  # Windows-friendly
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        tokenizer=tok,
        data_collator=data_collator,
    )

    trainer.train()

    # Save LoRA adapter
    (out_dir / "adapter").mkdir(exist_ok=True, parents=True)
    model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))

    with open(out_dir / "base_model.txt", "w", encoding="utf-8") as f:
        f.write(str(args.model_id))

    print("[OK] LoRA adapter saved to:", out_dir / "adapter")


if __name__ == "__main__":
    main()
