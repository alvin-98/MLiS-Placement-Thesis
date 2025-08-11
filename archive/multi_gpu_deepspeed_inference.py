#!/usr/bin/env python3
"""
Multi-GPU single-node inference script for Qwen-3-235B using DeepSpeed.

Key speedups implemented:
  • DeepSpeed Tensor-Parallel engine with kernel injection.
  • 4-bit NF4 weight quantisation + 8-bit KV-cache quant.
  • Flash-Attention-2 kernels where available.
  • TF32 matrix-multiply enabled.
  • KV-cache used for token-by-token streaming.

Launch example (on 4 GPUs):
    deepspeed --num_gpus 4 multi_gpu_deepspeed_inference.py --prompt "Hello!" --max_new 256
"""
import argparse
import os
import glob
from typing import List

import torch
import deepspeed
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    TextStreamer,
    
)

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def enable_flash_attention(config):
    """Switch the model config to flash-attention kernels if supported."""
    try:
        config.attn_implementation = "flash_attention_2"  # HF ≥ 4.38
    except AttributeError:
        pass  # Ignore if not supported


def enable_tf32():
    """Turn on TF32 for faster matmul on Ampere+ GPUs."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class MyStreamer(TextStreamer):
    """Token streamer that also stores generated ids for post-processing."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.token_ids: List[int] = []

    def put(self, value):
        super().put(value)  # prints token
        self.token_ids.extend(value.squeeze(0).tolist())

    def end(self):
        super().end()


# -----------------------------------------------------------------------------
# Main generation routine
# -----------------------------------------------------------------------------

def generate(args):
    enable_tf32()

    # Determine this process's rank and target model
    local_rank = int(args.local_rank if args.local_rank is not None else os.environ.get("LOCAL_RANK", 0))
    model_name = args.checkpoint

    # ---------------------------
    # Optional fresh download
    # ---------------------------
    if args.fresh and local_rank == 0:
        print("[Rank 0] Forcing fresh download from Hugging Face…")
        snapshot_download(repo_id=model_name)
    # Block until download done (when running multi-GPU)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    tp_size = args.tp or torch.cuda.device_count()

    # Resolve model weights location
    if os.path.isdir(model_name):
        # User supplied a local snapshot directory – skip any download
        weights_dir = model_name
    else:
        # Hugging Face repo ID – download only missing shards (cached otherwise)
        weights_dir = snapshot_download(
            repo_id=model_name,
            force_download=args.fresh,
            resume_download=not args.fresh,
        )

    # Build DeepSpeed checkpoint description expected by SDLoaderFactory
    ckpt_list = sorted(glob.glob(os.path.join(weights_dir, "model-*.safetensors")))
    checkpoint_dict = {
        "type": "Megatron",        # Let DeepSpeed treat this as Megatron-style checkpoint
        "version": 0,
        "checkpoints": ckpt_list,
        "parallelization": "tp",
        "tp_size": tp_size,
    }
    

    # 1) Tokenizer + empty weights skeleton (saves GPU RAM during load)
    print(f"[Rank {local_rank}] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"[Rank {local_rank}] Building empty model skeleton…")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    enable_flash_attention(config)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    model.eval()

    # 2) DeepSpeed Inference initialisation ­­– 4-bit NF4 + kernel injection
    ds_cfg = {
        "tensor_parallel": {"tp_size": tp_size},
        "dtype": "fp16",
        "replace_with_kernel_inject": True,
        "quant": {
            "enabled": True,
            "weight": {"enabled": True},  # 4-bit NF4 weight quant
            "qkv": {"enabled": True},    # 8-bit KV-cache quant
        },
    }
    print(f"[Rank {local_rank}] Initialising DeepSpeed engine…")
    engine = deepspeed.init_inference(
        model=model,
        config=ds_cfg,
        checkpoint=checkpoint_dict,
    )
    model = engine  # Alias for clarity

    # 3) Prepare prompt & inputs
    if args.chat:
        messages = [{"role": "user", "content": args.prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    else:
        prompt_text = args.prompt

    inputs = tokenizer([prompt_text], return_tensors="pt").to(local_rank)
    print(f"[Rank {local_rank}] Prefill phase…")

    # 4) One-token prefill to build KV-cache
    prefill = model.generate(
        **inputs,
        max_new_tokens=1,
        use_cache=True,
        return_dict_in_generate=True,
    )
    pkv = prefill.past_key_values
    first_id = prefill.sequences[:, -1:].to(local_rank)
    print("First token:", tokenizer.decode(first_id[0]))

    # 5) Stream remainder
    streamer = MyStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Expand attention mask for new tokens
    attention_mask = torch.cat(
        [inputs.attention_mask, torch.ones((1, 1), dtype=torch.long, device=local_rank)],
        dim=-1,
    )
    past_len = pkv[0][0].shape[2]
    cache_pos = torch.arange(past_len, past_len + 1, device=local_rank)

    print("\n--- Streaming ---")
    _ = model.generate(
        input_ids=first_id,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new,
        past_key_values=pkv,
        cache_position=cache_pos,
        use_cache=True,
        streamer=streamer,
    )
    print("\n--- Completed ---")

    # 6) Decode & display
    out_ids = [first_id.item()] + streamer.token_ids
    try:
        think_token_id = tokenizer.convert_tokens_to_ids("</think>")
        split = len(out_ids) - out_ids[::-1].index(think_token_id)
    except ValueError:
        split = 0

    thinking = tokenizer.decode(out_ids[:split], skip_special_tokens=True).strip()
    answer = tokenizer.decode(out_ids[split:], skip_special_tokens=True).strip()

    if thinking:
        print("\n[Thinking]\n", thinking)
    print("\n[Answer]\n", answer)


# -----------------------------------------------------------------------------
# Entry point / CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-3-235B DeepSpeed inference")
    parser.add_argument("--prompt", type=str, default="Hello.", help="User prompt text")
    parser.add_argument("--checkpoint", type=str, default="Qwen/Qwen3-235B-A22B", help="HF model path or local checkpoint")
    parser.add_argument("--max_new", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--tp", type=int, default=None, help="Tensor-parallel size (defaults to #GPUs)")
    parser.add_argument("--chat", action="store_true", help="Treat prompt as chat message and apply template")
    parser.add_argument("--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fresh", action="store_true", help="Force fresh download of model files from HF hub before loading")

    args = parser.parse_args()
    generate(args)
