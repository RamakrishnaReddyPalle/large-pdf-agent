# src/train/cpu_lora_infer.py
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def chat(model_id, adapter_dir, prompt, max_new_tokens=256, num_threads=8):
    torch.set_num_threads(num_threads)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    ids = tok(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapter_dir", default="outputs/lora_hf/title17")
    ap.add_argument("--num_threads", type=int, default=8)
    a = ap.parse_args()

    test_q = "Summarize the fair use factors in §107. Keep it concise and cite [pp. 40] at the end."
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{test_q}

### Response:
"""
    chat(a.model_id, a.adapter_dir, prompt, num_threads=a.num_threads)
