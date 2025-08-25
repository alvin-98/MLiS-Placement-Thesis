uv venv
uv pip install "sglang[all]>=0.5.0rc2" %as per documentation


module purge
module load gcc-uoneasy/12.3.0
module load Clang/16.0.6-GCCcore-12.3.0
module load CUDA/12.4.0
export CC=clang
export CXX=clang++
export TRITON_BUILD_WITH_CLANG_LLD=1
rm -rf ~/.triton ~/.cache/triton

python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --context-length 262144 --tp 2

A40s-

python -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tp 4 \
  --context-length 262144 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend triton

and parallelize to make concurrent requests and super fast tokens / second

import os
import shutil
import nest_asyncio
import asyncio
import json
from pydantic import BaseModel, Field
import sglang as sgl
nest_asyncio.apply()
# Set CUDA environment from module show output
os.environ["CUDA_HOME"] = "/gpfs01/software/easybuild-ada-uon/software/CUDA/12.4.0"
os.environ["PATH"] = f"/gpfs01/software/easybuild-ada-uon/software/CUDA/12.4.0/bin:{os.environ.get('PATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/gpfs01/software/easybuild-ada-uon/software/CUDA/12.4.0/lib:/gpfs01/software/easybuild-ada-uon/software/CUDA/12.4.0/extras/CUPTI/lib64:/gpfs01/software/easybuild-ada-uon/software/CUDA/12.4.0/nvvm/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ["OUTLINES_CACHE_DIR"] = f"/tmp/sglang_cache_{os.getpid()}"  # Unique cache dir for Jupyter
os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "True"  # For mixed GPU memory

# Clear caches
shutil.rmtree(os.path.expanduser("~/.cache/flashinfer"), ignore_errors=True)
shutil.rmtree(os.path.expanduser("~/.triton"), ignore_errors=True)
shutil.rmtree(os.path.expanduser("~/.cache/triton"), ignore_errors=True)

# Verify environment
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("PATH:", os.environ.get("PATH"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CC:", os.environ.get("CC"))
print("CXX:", os.environ.get("CXX"))
print("TRITON_BUILD_WITH_CLANG_LLD:", os.environ.get("TRITON_BUILD_WITH_CLANG_LLD"))
print("nvcc version:")
os.system("nvcc --version")
print("clang version:")
os.system("clang --version")

import sglang as sgl

llm = sgl.Engine(
    model_path="Qwen/Qwen3-30B-A3B-Instruct-2507", grammar_backend="llguidance", tp_size=2, 
)

sampling_params = {
    "temperature": 0.1,
    "top_p": 0.95,
    "json_schema": json.dumps(CapitalInfo.model_json_schema()),
    "max_new_tokens": 1024
}


It is important to set max_new_tokens in sampling params. As otherwise, you get validation errors.