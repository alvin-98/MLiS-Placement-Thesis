from vllm import LLM, SamplingParams
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
llm = LLM(
        model="openai/gpt-oss-20b",
        tensor_parallel_size=4,
    )
