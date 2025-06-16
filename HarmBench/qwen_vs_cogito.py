import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import pandas as pd
import re

print("Torch has GPU access: ", torch.cuda.is_available())
def generate_completions(model_name, dataset, device, cogito_df, qwen_df, do_sample, temperature, max_samples=None, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts = []
    metadata = []

    for ex in dataset:
        prompt = f"USER: {ex['prompt']}\nASSISTANT:"
        prompts.append(prompt)
        metadata.append((ex['category'], ex['prompt']))

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name} Generation Progress"):
        batch_prompts = prompts[i:i+batch_size]
        batch_metadata = metadata[i:i+batch_size]

        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)

        for _ in range(5):  
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.batch_decode(outputs[:, inputs['input_ids'].size(1):], skip_special_tokens=True)

            for j, gen in enumerate(decoded):
                prompt_number = i + j + 1
                category, orig_prompt = batch_metadata[j]
                if model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B':
                    qwen_df.loc[len(qwen_df)] = [model_name,temperature, prompt_number, category, orig_prompt, gen]
                elif model_name == 'deepcogito/cogito-v1-preview-llama-8B':
                    cogito_df.loc[len(cogito_df)] = [model_name,temperature, prompt_number, category, orig_prompt, gen]

    


def main():
    cogito_df = pd.DataFrame(columns=['model_name','temperature','prompt_number','category','prompt','LLM_responce'])
    qwen_df = pd.DataFrame(columns=['model_name','temperature','prompt_number','category','prompt','LLM_responce'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cogito', default='deepcogito/cogito-v1-preview-llama-8B')
    parser.add_argument('--qwen', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--do_sample', default=True)
    parser.add_argument('--temperature', type=float, default=1.1)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    print(f"Loading HarmBench...")
    dataset = load_dataset("walledai/HarmBench",'standard', split="train")
    print(f"Loaded {len(dataset)} examples")
    
    for model_name, label in [(args.cogito, 'Cogito'), (args.qwen, 'Qwen')]:
        print(f"\nGenerating with {label} model: {model_name}")
        generate_completions(model_name, dataset, args.device, cogito_df, qwen_df, args.do_sample, args.temperature, args.max_samples, args.batch_size)
        
    
    cogito_df.to_csv(f'HarmBench/Cogito_HarmBench_llm_judge_larger_context_samples_{args.do_sample}_temperature_{args.temperature}.csv', index=True)
    qwen_df.to_csv(f'HarmBench/Qwen_HarmBench_llm_judge_larger_context_samples_{args.do_sample}_temperature_{args.temperature}.csv', index=True)

if __name__ == '__main__':
    main()
