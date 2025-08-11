import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import os

model_name = "Qwen/Qwen3-235B-A22B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",

    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
# setup DeepSpeed inference across nodes
if not torch.distributed.is_initialized():
    # compute global rank and world size for multi-node multi-gpu
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("SLURM_NODEID", 0))
    gpus_per_node = torch.cuda.device_count()
    rank = node_rank * gpus_per_node + local_rank
    world_size = int(os.environ.get("SLURM_NTASKS", 1)) * gpus_per_node
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
engine = deepspeed.init_inference(
    model,
    mp_size=torch.distributed.get_world_size(),  # partition model across all ranks
    dtype=torch.float16,
    replace_with_kernel_inject=True,
    zero={
        "stage": 3,
        "offload_param": {"device": "cpu", "pin_memory": True}
    }
)
model = engine.module
# verify distributed setup
if torch.distributed.is_initialized():
    print(f"Distributed world size: {torch.distributed.get_world_size()}, rank: {torch.distributed.get_rank()}")

print(model.device)


# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
print("Prepared prompt text:", text)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print("Model inputs keys:", model_inputs.keys())
print("Input_ids shape:", model_inputs.input_ids.shape)

# conduct text completion (timing)
torch.cuda.synchronize()
start_time = time.time()
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128
)
torch.cuda.synchronize()
end_time = time.time()
print(f"Inference time: {end_time - start_time:.3f} seconds")
print("Generated ids shape:", generated_ids.shape)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
print("Output_ids length:", len(output_ids)) 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0
print("Split index:", index)

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
