# src/agent/llms.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Iterator, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread

from .config import CFG

# Hard block GPU/auto-mapping; keep it simple and CPU-only.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ACCELERATE_DISABLE_RICH", "1")

torch.set_num_threads(max(1, os.cpu_count() or 1))

@dataclass
class CoreLLM:
    tokenizer: AutoTokenizer
    model: torch.nn.Module

# Simple singleton cache to avoid reloading every time in notebooks
_CORE_CACHE: Optional[CoreLLM] = None

def load_core_llm(force_reload: bool = False) -> CoreLLM:
    """
    Load base Qwen + attach LoRA adapter on CPU, robust to reruns.
    Set `force_reload=True` if you changed the adapter on disk and want a fresh load.
    """
    global _CORE_CACHE
    if _CORE_CACHE is not None and not force_reload:
        return _CORE_CACHE

    tok = AutoTokenizer.from_pretrained(
        str(CFG.base_model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Always pin to CPU and avoid auto sharding/offloading.
    base = AutoModelForCausalLM.from_pretrained(
        str(CFG.base_model_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": "cpu"},        # <- explicit CPU
        low_cpu_mem_usage=False,       # <- avoid Accelerate deciding to shard
    )
    # Inference settings
    base.config.use_cache = True

    # Create an offload folder just in case Accelerate is triggered internally by PEFT
    offload_dir = CFG.logs_dir / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)

    model = PeftModel.from_pretrained(
        base,
        str(CFG.adapter_dir),
        local_files_only=True,
        device_map={"": "cpu"},                 # <- explicit CPU again for adapter
        offload_folder=str(offload_dir),        # <- satisfies Accelerate if it tries anything
    )
    model.to("cpu").eval()

    _CORE_CACHE = CoreLLM(tokenizer=tok, model=model)
    return _CORE_CACHE

def stream_generate(core: CoreLLM, prompt: str, max_new_tokens: int = 256,
                    temperature: float = 0.0) -> Iterator[str]:
    """
    CPU streaming generator.
    Greedy if temperature<=0; otherwise sampling.
    """
    do_sample = bool(temperature and temperature > 0.0)

    inputs = core.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    streamer = TextIteratorStreamer(core.tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=float(max(0.01, temperature)) if do_sample else None,
        pad_token_id=core.tokenizer.eos_token_id,
        eos_token_id=core.tokenizer.eos_token_id,
        streamer=streamer,
    )
    # remove None keys to avoid "invalid generation flag" warnings
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    thread = Thread(target=core.model.generate, kwargs=gen_kwargs)
    thread.start()
    for token in streamer:
        yield token
    thread.join()
