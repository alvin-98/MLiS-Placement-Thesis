import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import pandas as pd
import re

print("Torch has GPU access: ", torch.cuda.is_available())
def generate_completions(model_name, dataset, device, cogito_df, qwen_df, do_sample, temperature, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    
    
    
    prompt_number = 0
    for ex in tqdm(dataset, desc=f"Evaluating {model_name}", total=len(dataset)):
        dataset_instruction = ex['prompt']
        category = ex['category']
        prompt_number += 1
        prompt = f"USER: {dataset_instruction}\nASSISTANT:"
        
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        for i in range(5):
            with torch.no_grad():
                if model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': 
                    outputs = model.generate(**inputs, max_new_tokens=250, do_sample=do_sample,temperature=temperature, pad_token_id=tokenizer.eos_token_id)
                elif model_name == 'deepcogito/cogito-v1-preview-llama-8B':
                    outputs = model.generate(**inputs, max_new_tokens=250, do_sample=do_sample,temperature=temperature, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(outputs[0][inputs['input_ids'].size(-1):], skip_special_tokens=True)
                                   
            if model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B':
                qwen_df.loc[len(qwen_df)] = [prompt_number, category,dataset_instruction, gen]
            elif model_name == 'deepcogito/cogito-v1-preview-llama-8B':
                cogito_df.loc[len(cogito_df)] = [prompt_number, category,dataset_instruction, gen]
        
    


def main():
    cogito_df = pd.DataFrame(columns=['prompt_number','category','prompt','LLM_responce'])
    qwen_df = pd.DataFrame(columns=['prompt_number','category','prompt','LLM_responce',])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cogito', default='deepcogito/cogito-v1-preview-llama-8B')
    parser.add_argument('--qwen', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--do_sample', default=True)
    parser.add_argument('--temperature', type=float, default=1.1)
    args = parser.parse_args()
    
    print(f"Loading HarmBench...")
    dataset = load_dataset("walledai/HarmBench",'standard', split="train")
    print(f"Loaded {len(dataset)} examples")
    
    for model_name, label in [(args.cogito, 'Cogito'), (args.qwen, 'Qwen')]:
        print(f"\nEvaluating {label} model: {model_name}")
        generate_completions(model_name, dataset, args.device, cogito_df, qwen_df, args.do_sample, args.temperature, args.max_samples)
        
    
    cogito_df.to_csv(f'HarmBench/Cogito_HarmBench_llm_judge_larger_context_samples_{args.do_sample}_temperature_{args.temperature}.csv', index=True)
    qwen_df.to_csv(f'HarmBench/Qwen_HarmBench_llm_judge_larger_context_samples_{args.do_sample}_temperature_{args.temperature}.csv', index=True)

if __name__ == '__main__':
    main()
