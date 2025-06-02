

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import pandas as pd
import re


def evaluate_model(model_name, dataset, device, cogito_df, qwen_df, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    correct = 0
    total = 0
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # system instruction for multiple-choice exam - experiment with different instructions for reasoning models
    if model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B':
        system_instruction = """You are a knowledgeable assistant taking a multiple-choice exam. Read the multiple‑choice question
        and begin your chain of thought in the format 'the answer is (A, B, C or D) because'. DO NOT INCLUDE BRACKETS AROUND THE LETTER"""
    elif model_name == 'deepcogito/cogito-v1-preview-llama-8B':
        system_instruction = """You are a knowledgeable assistant taking a multiple-choice exam. Read the multiple‑choice question
        and answer with a single capital letter (A, B, C or D) only. DO NOT INCLUDE BRACKETS AROUND THE LETTER"""
    
    for ex in tqdm(dataset, desc=f"Evaluating {model_name}", total=len(dataset)):
        question = ex['question']
        choices = ex.get('choices', [])
        label = ex['answer']
        
        choices_str = ' '.join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        prompt = f"{system_instruction}\n Question: {question}\nChoices: {choices_str}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            if model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': #reasoning model breaks due to being unable to think
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            elif model_name == 'deepcogito/cogito-v1-preview-llama-8B':
                outputs = model.generate(**inputs, max_new_tokens=3, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(outputs[0][inputs['input_ids'].size(-1):], skip_special_tokens=True)
                      
        tokens = gen.strip().split()
        if not tokens: #fix for if the model generates nothing
            total += 1
            continue
        
        if model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B':  
            if 'Answer:' in gen: #if enough tokens are generated to include explicit answer - will explore more when on full HPC
                match = re.search(r"Answer:\s*(.*)", gen, re.IGNORECASE)
                answer_part = match.group(1).strip() if match else ""
                match = re.search(r"\b[\(\[]?([A-D])[\)\]]?\b", answer_part)
                pred = match.group(1) if match else None
            else:
                match = re.search(r"\b[\(\[]?([A-D])[\)\]]?\b", gen)
                pred = match.group(1) if match else None

            
        elif model_name == 'deepcogito/cogito-v1-preview-llama-8B':  
            pred = gen.strip().split()[0].strip("()")
        
        # normalize true label to letter (index -> letter)
        true = chr(65 + int(label))
        
        total += 1
        if pred == true:
            correct += 1
        

        
        if model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B':
            qwen_df.loc[len(qwen_df)] = [prompt,gen, pred, true]
        elif model_name == 'deepcogito/cogito-v1-preview-llama-8B':
            cogito_df.loc[len(cogito_df)] = [prompt,gen, pred, true]
    
    return correct, total


def main():
    cogito_df = pd.DataFrame(columns=['prompt','raw prediction', 'submitted prediction', 'true'])
    qwen_df = pd.DataFrame(columns=['prompt', 'raw prediction','submitted prediction', 'true'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cogito', default='deepcogito/cogito-v1-preview-llama-8B')
    parser.add_argument('--qwen', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Loading MMLU validation split...")
    dataset = load_dataset("cais/mmlu", 'all', split="validation")
    print(f"Loaded {len(dataset)} examples")
    
    for model_name, label in [(args.cogito, 'Cogito'), (args.qwen, 'Qwen')]:
        print(f"\nEvaluating {label} model: {model_name}")
        correct, total = evaluate_model(model_name, dataset, args.device, cogito_df, qwen_df, args.max_samples)
        print(f"{label} accuracy: {correct}/{total} = {correct/total:.4f}")
    
    cogito_df.to_csv('Cogito_MMLU.csv', index=True)
    qwen_df.to_csv('Qwen_MMLU.csv', index=True)

if __name__ == '__main__':
    main()
