from vllm import LLM, SamplingParams
import os

def main():
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    llm = LLM(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        tensor_parallel_size=4
    )

    prompts = [
        "What is the capital of France?",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    # This is the entry point, and this code will only run once.
    main()