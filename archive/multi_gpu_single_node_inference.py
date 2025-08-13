from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
import os
import time

# Enable TF32 tensor-core acceleration
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Disable tokenizer parallelism warnings as recommended when using torch.compile
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Custom streamer to store token IDs while printing
class MyStreamer(TextStreamer):
    def __init__(self, tokenizer: "AutoTokenizer", **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.token_ids = []

    def put(self, value):
        """
        Receives tokens, prints them to stdout, and stores their IDs.
        """
        super().put(value)  # This prints the token to the console
        # The 'value' tensor is of shape (batch_size, num_new_tokens).
        # We handle the batch dimension to ensure we're adding a list of ints.
        if len(value.shape) > 1:
            self.token_ids.extend(value[0].tolist())
        else:
            self.token_ids.extend(value.tolist())

    def end(self):
        """
        Cleans up the output stream.
        """
        super().end()

model_name = "Qwen/Qwen3-235B-A22B"
quantized_model_dir = "qwen-235b-4bit"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Check if the quantized model is already saved
if os.path.exists(quantized_model_dir) and os.path.isdir(quantized_model_dir):
    print(f"Loading saved 4-bit model from {quantized_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        quantized_model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )
else:
    print(f"Model not found locally. Downloading, quantizing, and saving to {quantized_model_dir}...")
    # load the tokenizer and the model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        force_download=True,
        resume_download=True,
        trust_remote_code=True,
        quantization_config=quantization_config,
        # attn_implementation="flash_attention_2",
    )
    
    # Save the quantized model and tokenizer
    print(f"Saving quantized model to {quantized_model_dir}...")
    model.save_pretrained(quantized_model_dir)
    tokenizer.save_pretrained(quantized_model_dir)
    print("Model saved.")
# Compile the model for faster inference
# try:
    # Compile the model's forward pass; the first call will trigger compilation.
    model.forward = torch.compile(model.forward, mode="max-autotune", fullgraph=True)
    print("model.forward compiled with torch.compile.")
except Exception as e:
    print("WARNING: Failed to compile model with torch.compile:", e)

print(model.device)
print("HF device map:", getattr(model, "hf_device_map", None))

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
print("Prepared prompt text:", text)
# Padding to a multiple of 16 avoids shape-related recompilations when using static kv-cache
model_inputs = tokenizer([text], return_tensors="pt", pad_to_multiple_of=16).to(model.device)
print("Model inputs keys:", model_inputs.keys())
print("Input_ids shape:", model_inputs.input_ids.shape)

# conduct text completion with KV caching
# Step 1: generate first token to build KV cache
with torch.inference_mode():
    prefill_outputs = model.generate(
        **model_inputs,
        max_new_tokens=1,
        use_cache=True,
        return_dict_in_generate=True,
    )
past_key_values = prefill_outputs.past_key_values
first_token_id = prefill_outputs.sequences[:, -1:].to(model.device)
print("First token:", tokenizer.decode(first_token_id[0]))
print("Prefill complete, KV cache built.")

# Step 2: generate remaining tokens and stream them
streamer = MyStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# The attention mask must be updated to cover the cached tokens plus the new token.
attention_mask = torch.cat([
    model_inputs.attention_mask, 
    torch.ones((1, 1), dtype=torch.long, device=model.device)
], dim=-1)

# Compute cache_position manually to avoid empty tensor in HF `generate`
past_seq_len = past_key_values[0][0].shape[2]
cache_position = torch.arange(past_seq_len, past_seq_len + 1, dtype=torch.long, device=model.device)

print("\n--- Streaming new tokens ---")
start_time = time.time()
with torch.inference_mode():
    _ = model.generate(
        input_ids=first_token_id,
        attention_mask=attention_mask,
        max_new_tokens=32767,
        past_key_values=past_key_values,
        cache_position=cache_position,
        use_cache=True,
        streamer=streamer,
    )
end_time = time.time()
stream_time = end_time - start_time
token_count = len(streamer.token_ids)
print(f"\n--- End of streaming ---\nStreamed {token_count} tokens in {stream_time:.2f} seconds ({token_count/stream_time:.2f} tokens/s)")

# Combine the first token with the rest of the generated tokens
output_ids = [first_token_id.item()] + streamer.token_ids
print(f"Generated {len(output_ids)} ids in total.") 

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