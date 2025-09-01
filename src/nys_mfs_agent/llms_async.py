# src/nys_mfs_agent/llms_async.py
from __future__ import annotations
import os, asyncio
from dataclasses import dataclass
from typing import Iterator, Optional, AsyncIterator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import logging as hf_logging
from threading import Thread

from .config import CFG

# Silence noisy warnings completely
hf_logging.set_verbosity_error()

# CPU only & offline
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ACCELERATE_DISABLE_RICH", "1")

torch.set_num_threads(max(1, os.cpu_count() or 1))

@dataclass
class CoreLLM:
    tokenizer: AutoTokenizer
    model: torch.nn.Module

_CORE: Optional[CoreLLM] = None

def load_core_llm(force_reload: bool = False) -> CoreLLM:
    global _CORE
    if _CORE is not None and not force_reload:
        return _CORE

    tok = AutoTokenizer.from_pretrained(
        str(CFG.core_model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(CFG.core_model_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
        trust_remote_code=True,
        device_map={"": "cpu"},
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = True
    model.eval()
    _CORE = CoreLLM(tokenizer=tok, model=model)
    return _CORE

def render_chat(tok, system_text: str, user_text: str, history=None) -> str:
    msgs = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    if history:
        for m in history:
            if m.get("role") in ("user","assistant") and m.get("content"):
                msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": user_text})
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # light fallback
        return f"<|im_start|>system\n{system_text}\n<|im_end|>\n<|im_start|>user\n{user_text}\n<|im_end|>\n<|im_start|>assistant\n"

def stream(core: CoreLLM, prompt: str, max_new_tokens=256, temperature=0.0) -> Iterator[str]:
    do_sample = bool(temperature and temperature > 0.0)
    inputs = core.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    streamer = TextIteratorStreamer(core.tokenizer, skip_prompt=True, skip_special_tokens=True)

    kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        pad_token_id=core.tokenizer.eos_token_id,
        eos_token_id=core.tokenizer.eos_token_id,
        streamer=streamer,
    )
    # Include sampling args only when sampling to avoid "invalid generation flag" banners
    if do_sample:
        kwargs["temperature"] = float(max(0.01, temperature))
        kwargs["top_p"] = 0.9
        # NOTE: Do NOT pass top_k; it's often the culprit for warnings

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    th = Thread(target=core.model.generate, kwargs=kwargs, daemon=True)
    th.start()
    for tok in streamer:
        yield tok
    th.join()

async def astream(core: CoreLLM, prompt: str, max_new_tokens=256, temperature=0.0) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()

    def _producer():
        for tok in stream(core, prompt, max_new_tokens=max_new_tokens, temperature=temperature):
            asyncio.run_coroutine_threadsafe(queue.put(tok), loop)
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    Thread(target=_producer, daemon=True).start()

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
