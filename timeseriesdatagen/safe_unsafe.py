import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import pandas as pd
import re
from itertools import islice

print("Torch has GPU access: ", torch.cuda.is_available())
def generate_completions(model_name, dataset,dataset_name, device, df, do_sample, temperature, max_samples=None, batch_size=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if max_samples:
        if dataset_name == 'unsafe':
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        elif dataset_name == 'safe':
            dataset = dataset[:max_samples]

    prompts = []
    metadata = []
    if dataset_name == 'unsafe':
        for ex in dataset:
            prompt = f"USER: {ex['prompt']}\nASSISTANT:"
            prompts.append(prompt)
            metadata.append((ex['category'], ex['prompt']))
    elif dataset_name == 'safe':
        for ex in dataset:
            prompt = f"USER: {ex}\nASSISTANT:"
            prompts.append(prompt)
            category = 'safe prompt'
            metadata.append((category, ex))

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name} Generation Progress"):
        batch_prompts = prompts[i:i+batch_size]
        batch_metadata = metadata[i:i+batch_size]

        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)

        for _ in range(5):  
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded = tokenizer.batch_decode(outputs[:, inputs['input_ids'].size(1):], skip_special_tokens=True)

            for j, gen in enumerate(decoded):
                prompt_number = i + j + 1
                category, orig_prompt = batch_metadata[j]
                df.loc[len(df)] = [model_name,dataset_name,temperature,prompt_number, category, orig_prompt, gen]
                
    return df

    


def main():
    unsafe_df = pd.DataFrame(columns=['model_name','dataset','temperature','prompt_number','category','prompt','LLM_responce'])
    safe_df = pd.DataFrame(columns=['model_name','dataset','temperature','prompt_number','category','prompt','LLM_responce'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--do_sample', default=True)
    parser.add_argument('--temperature', type=float, default=1.2)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    print(f"Loading HarmBench...")
    unsafe_dataset = load_dataset("walledai/HarmBench",'standard', split="train")
    print(f"Loaded {len(unsafe_dataset)} examples")
    print('loading safe dataset...')
    streaming_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", streaming=True)
    train_stream = streaming_dataset["train_sft"]
    n = 200
    first_n = list(islice(train_stream, n))
    safe_dataset = []
    for example in first_n:
        for message in example["messages"]:
            if message["role"] == "user":
                safe_dataset.append(message["content"])
                break 
    print(f"Loaded {len(safe_dataset)} user prompts from the safe dataset")
    model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    for dataset in [unsafe_dataset, safe_dataset]:
        if dataset == unsafe_dataset:
            print(f"Generating completions for unsafe dataset with {model_name}...")
            dataset_name = 'unsafe'
            unsafe_df = generate_completions(model_name, dataset,dataset_name, args.device, unsafe_df, args.do_sample, args.temperature, args.max_samples, args.batch_size)
        else:
            print(f"Generating completions for safe dataset with {model_name}...")
            dataset_name = 'safe'
            safe_df = generate_completions(model_name, dataset,dataset_name, args.device, safe_df, args.do_sample, args.temperature, args.max_samples, args.batch_size)
        
    
    safe_df.to_csv('timeseriesdatagen/safe_data.csv', index=True)
    unsafe_df.to_csv('timeseriesdatagen/unsafe_data.csv', index=True)

if __name__ == '__main__':
    main()
