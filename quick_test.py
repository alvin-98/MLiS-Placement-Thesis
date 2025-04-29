'File to quickly test changes to the code - runs 10 samples on weak model and saves predictions to csv files'

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import pandas as pd


def evaluate_model(model_name, dataset, device, weak_model_df, strong_model_df, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    correct = 0
    total = 0
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # system instruction for multiple-choice exam
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
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(outputs[0][inputs['input_ids'].size(-1):], skip_special_tokens=True)
                      
        tokens = gen.strip().split()
        if not tokens: #fix for if the model generates nothing
            total += 1
            continue
            
        pred = gen.strip().split()[0].strip("()")
        
        # normalize true label to letter (index -> letter)
        true = chr(65 + int(label))
        
        
        if pred == true:
            correct += 1
        total += 1
        
        if model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B':
            strong_model_df.loc[len(strong_model_df)] = [prompt, pred, true]
        elif model_name == 'distilbert/distilgpt2':
            weak_model_df.loc[len(weak_model_df)] = [prompt, pred, true]
    
    return correct, total


def main():
    weak_model_df = pd.DataFrame(columns=['prompt', 'prediction', 'true'])
    strong_model_df = pd.DataFrame(columns=['prompt', 'prediction', 'true'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak_model', default='distilbert/distilgpt2')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Loading MMLU validation split...")
    dataset = load_dataset("cais/mmlu", 'all', split="validation")
    print(f"Loaded {len(dataset)} examples")
    
    for model_name, label in [(args.weak_model, 'Weak')]:
        print(f"\nEvaluating {label} model: {model_name}")
        correct, total = evaluate_model(model_name, dataset, args.device, weak_model_df, strong_model_df, 10)
        print(f"{label} accuracy: {correct}/{total} = {correct/total:.4f}")
    
    weak_model_df.to_csv('weak_model_predictions.csv', index=True)
    strong_model_df.to_csv('strong_model_predictions.csv', index=True)

if __name__ == '__main__':
    main()
