import os
# Keep downloads inside the project and use the robust HTTP downloader
os.environ.setdefault("HF_HOME", os.path.abspath("hf_home"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.abspath("hf_home/transformers"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # use Rust accel
os.environ.setdefault("HF_HUB_ENABLE_XET", "0")          # disable xet path

from huggingface_hub import snapshot_download

MODELS = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "models/deepseek-70b",
    "meta-llama/Llama-3.1-70B-Instruct":         "models/llama3-70b",
}

for repo_id, local_dir in MODELS.items():
    print(f"\nDownloading {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,  # friendlier to shared FS
        allow_patterns=[
            "*.safetensors",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
        ],
    )
print("All done.")

