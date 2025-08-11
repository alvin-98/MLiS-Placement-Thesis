import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TextStreamer
from accelerate import init_empty_weights

# A custom streamer to store token IDs while also printing to console
# This is useful for post-processing the output
class MyStreamer(TextStreamer):
    def __init__(self, tokenizer: "AutoTokenizer", **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.token_ids = []

    def put(self, value):
        """
        Receives tokens, prints them to stdout, and stores their IDs.
        """
        super().put(value)
        if len(value.shape) > 1:
            self.token_ids.extend(value[0].tolist())
        else:
            self.token_ids.extend(value.tolist())

    def end(self):
        """
        Cleans up the output stream.
        """
        super().end()

# --- Configuration ---
model_name = "Qwen/Qwen3-235B-A22B"
# Get the local rank from the environment variable set by the DeepSpeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

# --- Model Loading (Memory-Efficient Method) ---
print(f"[Rank {local_rank}] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"[Rank {local_rank}] Creating empty model skeleton...")
# 1. Load the model config only
config = AutoConfig.from_pretrained(model_name)

# 2. Create an empty model on the 'meta' device. This uses almost no RAM.
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
model.eval() # Set to inference mode

# --- DeepSpeed Initialization ---
print(f"[Rank {local_rank}] Initializing DeepSpeed-Inference engine...")
# 3. Let DeepSpeed instantiate the model and load the checkpoint directly to GPUs.
ds_config = {
    "tensor_parallel": {"tp_size": world_size},
    "dtype": "fp16",
    "replace_with_kernel_inject": True,
    "quantization": {
        "enabled": True,
        "bits": 4,
    },
    "kv_cache": {
        "enabled": True,
        "dtype": "int8"
    }
}

# Initialize the DeepSpeed Inference engine, passing the model name as the checkpoint
# DeepSpeed will handle loading the weights into the empty model skeleton.
model = deepspeed.init_inference(
    model=model,
    config=ds_config,
    checkpoint=model_name # Key change: Tell DeepSpeed where to find the weights
)
print(f"[Rank {local_rank}] DeepSpeed engine initialized successfully.")

# --- Prompt Preparation ---
# Only rank 0 needs to prepare and broadcast the prompt
if local_rank == 0:
    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    print("Prepared prompt text:", text)
    model_inputs = tokenizer([text], return_tensors="pt")
else:
    # Other ranks get dummy inputs that will be overwritten by broadcast
    model_inputs = tokenizer([""], return_tensors="pt")

# Move inputs to the correct device for the current rank
for key in model_inputs:
    model_inputs[key] = model_inputs[key].to(f'cuda:{local_rank}')

# Broadcast the tokenized inputs from rank 0 to all other ranks
if world_size > 1:
    torch.distributed.broadcast(model_inputs.input_ids, src=0)
    torch.distributed.broadcast(model_inputs.attention_mask, src=0)

# --- Text Generation ---
# The streamer only needs to be created on the rank that will print the output.
streamer = MyStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if local_rank == 0 else None

print(f"\n[Rank {local_rank}] --- Generating tokens ---")
output = model.generate(
    input_ids=model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=512, # Set a reasonable max length
    streamer=streamer,
)
print(f"\n[Rank {local_rank}] --- End of generation ---")

# --- Output Processing ---
# Only rank 0, which has the full output, should process and print the result.
if local_rank == 0:
    print(f"Generated {len(output[0])} ids in total.")
    
    full_output_ids = output[0].tolist()

    try:
        # rindex finding 151668 (</think>)
        index = len(full_output_ids) - full_output_ids[::-1].index(151668)
    except ValueError:
        index = 0 # If the token isn't found, assume no thinking content
    print("Split index:", index)

    thinking_content = tokenizer.decode(full_output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(full_output_ids[index:], skip_special_tokens=True).strip("\n")

    print("\n--- Parsed Output ---")
    print("Thinking content:", thinking_content)
    print("Content:", content)
