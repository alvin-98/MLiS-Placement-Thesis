import torch, time, threading
from accelerate import infer_auto_device_map
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TextIteratorStreamer
)

import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"   # pick the 3 GPUs you want
print("Visible devices ->", torch.cuda.device_count())

# -- 1. low‑precision & kernel flags -------------------------------------------------
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.set_float32_matmul_precision("high")



model_name = "Qwen/Qwen3-235B-A22B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -- 2. balanced sharding to avoid GPU‑0 overload -----------------------------------
device_map = "balanced"                # Spread router & experts :contentReference[oaicite:10]{index=10}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # attn_implementation="flash_attention_2",  # fused FA2 :contentReference[oaicite:11]{index=11}
    # quantization_config removed – run full-precision bfloat16
)

# -- 3. compile once, before first forward ------------------------------------------
# model = torch.compile(model, mode="reduce-overhead")   # :contentReference[oaicite:12]{index=12}

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

print("\nHF device map ⬇️")
print(model.hf_device_map)          # <- nothing here should say "cpu" or "disk"

# print("bnb CUDA installed:", bnb.cuda.is_available())
print("Any Linear4bit layers on CPU?",
      any(p.device.type == "cpu" for p in model.parameters()))

# -- 4. prompt & static‑shape input --------------------------------------------------
prompt = "Give me a short introduction to large language model."
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False   
)


inputs = tokenizer(
    text, return_tensors="pt",
    # pad_to_multiple_of=16,
).to(model.device)

# -- 5. non‑blocking streamer + direct logits ---------------------------------------

gen_kwargs = dict(
    **inputs,
    max_new_tokens=256,                 # small window for fair timing
    use_cache=True,
    return_dict_in_generate=True,
    output_scores=True,
)
print("\n--- generating ---")
torch.cuda.synchronize()
t0 = time.time()
# model.forward = torch.compile(model.forward, mode="max-autotune")
print(model.device)
print("HF device map:", getattr(model, "hf_device_map", None))

out = model.generate(**gen_kwargs)   # single blocking call

torch.cuda.synchronize()
dt = time.time() - t0

# ------------------------------------------------------------------
# post‑processing
# ------------------------------------------------------------------
generated_ids   = out.sequences
new_tokens      = generated_ids.shape[1] - inputs.input_ids.shape[1]
tok_per_second  = new_tokens / dt

print(f"\n{new_tokens} tokens in {dt:.2f}s → {tok_per_second:.2f} tok/s")

# full text (optional)
print("\n---- decoded text ----")
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))

# logits tensor:  shape = [steps, batch, vocab]
logits = torch.stack(out.scores)    