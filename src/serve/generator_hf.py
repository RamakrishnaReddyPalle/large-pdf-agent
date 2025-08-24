# src/serve/generator_hf.py
from __future__ import annotations
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from typing import Iterable

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class HFGeneratorCPU:
    def __init__(self, model_dir: str, trust_remote_code: bool = True, max_new_tokens: int = 256):
        self.model_dir = model_dir
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32, local_files_only=True,
            trust_remote_code=trust_remote_code, device_map={"": "cpu"}
        )
        self.model.eval()

    def generate_stream(self, prompt: str) -> Iterable[str]:
        inputs = self.tok(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tok, skip_special_tokens=True, skip_prompt=True)
        gen_kwargs = dict(
            **inputs, streamer=streamer, max_new_tokens=self.max_new_tokens,
            do_sample=False, pad_token_id=self.tok.eos_token_id, eos_token_id=self.tok.eos_token_id
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for token in streamer:
            yield token

    def generate(self, prompt: str) -> str:
        return "".join(self.generate_stream(prompt))
