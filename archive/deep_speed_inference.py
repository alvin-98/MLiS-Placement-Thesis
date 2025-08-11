#!/usr/bin/env python3
"""
Script to run Qwen3-235B 4-bit inference with DeepSpeed on 3 A100 GPUs.

Requirements:
    pip install deepspeed accelerate transformers
"""
import os
import torch
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig, # Import AutoConfig
    TextStreamer,
)
from accelerate import init_empty_weights # Import init_empty_weights

# Custom streamer to store token IDs while printing to stdout
class MyStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.token_ids = []

    def put(self, value):
        # Print token and store its ID
        super().put(value)
        if value.dim() > 1:
            self.token_ids.extend(value[0].tolist())
        else:
            self.token_ids.extend(value.tolist())

    def end(self):
        super().end()


def main():
    model_name = "Qwen/Qwen3-235B-A22B"

    # 1) Load tokenizer & create an empty model skeleton
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Creating empty model skeleton...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()

    # 2) Initialize DeepSpeed inference with on-the-fly 4-bit quantization
    print("Initializing DeepSpeed Inference engine with 4-bit NF4 quant...")
    # Use a dictionary for configuration for better compatibility
    ds_config = {
        "tensor_parallel": {"tp_size": 3},      # Shard across 3 A100 GPUs
        "dtype": "fp16",                       # Compute in FP16
        "replace_with_kernel_inject": True,     # Use fused kernels
        "quant": {                              # Quantization section per DeepSpeed schema
            "enabled": True,                    # Enable quantization
            "weight": {"enabled": True},      # 4-bit NF4 weight quant (default)
            "qkv": {"enabled": True}          # 8-bit KV-cache quant
        }
    }

    ds_engine = deepspeed.init_inference(
        model=model,           # Pass the empty skeleton
        config=ds_config,      # Use the unified config dictionary
        checkpoint=model_name, # Tell DeepSpeed where to find the weights
    )
    model = ds_engine
    local_rank = ds_engine.local_rank

    # 3) Prepare prompt and tokenization
    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    print("Prepared prompt text:", text)
    inputs = tokenizer([text], return_tensors="pt").to(local_rank)

    # 4) Prefill to build KV-cache
    prefill = model.generate(
        **inputs,
        max_new_tokens=1,
        use_cache=True,
        return_dict_in_generate=True,
    )
    pkv = prefill.past_key_values
    first_id = prefill.sequences[:, -1:].to(local_rank)
    print("First token:", tokenizer.decode(first_id[0]))

    # 5) Stream remaining tokens
    streamer = MyStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    attention_mask = torch.cat([
        inputs.attention_mask,
        torch.ones((1, 1), dtype=torch.long, device=local_rank)
    ], dim=-1)
    past_len = pkv[0][0].shape[2]
    cache_pos = torch.arange(past_len, past_len + 1, device=local_rank)

    print("\n--- Streaming generation ---")
    _ = model.generate(
        input_ids=first_id,
        attention_mask=attention_mask,
        max_new_tokens=256,
        past_key_values=pkv,
        cache_position=cache_pos,
        use_cache=True,
        streamer=streamer,
    )
    print("\n--- Done ---")

    # 6) Decode output
    output_ids = [first_id.item()] + streamer.token_ids
    print(f"Total tokens generated: {len(output_ids)}")

    # Split thinking vs response
    try:
        think_end = len(output_ids) - output_ids[::-1].index(tokenizer.convert_tokens_to_ids("</think>"))
    except ValueError:
        think_end = 0

    thinking = tokenizer.decode(output_ids[:think_end], skip_special_tokens=True).strip()
    content  = tokenizer.decode(output_ids[think_end:], skip_special_tokens=True).strip()

    if thinking:
        print("\n[Thinking]\n", thinking)
    print("\n[Response]\n", content)

if __name__ == "__main__":
    main()
