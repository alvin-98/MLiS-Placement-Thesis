import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import pandas as pd
import os


def get_responses(model_name, dataset, device, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    correct = 0
    total = 0
    prompts = []
    completions = []
    responses = []
    reference_responces = []
    
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
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id, num_beams=1)
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

        prompts.append(prompt)
        completions.append(gen)
        responses.append(pred)
        reference_responces.append(true)
    
    return correct, total, prompts, completions, responses, reference_responces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='distilgpt2', help='Model to evaluate')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    parser.add_argument('--num-rows', type=int, default=100, help='Number of rows to evaluate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--concat-results', type=bool, default=True, help='Concatenate results to CSV file if it exists')
    args = parser.parse_args()

    _print = print if args.verbose else lambda *args, **kwargs: None

    DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(DATA_ROOT, exist_ok=True)

    if args.concat_results:
        results_path = os.path.join(DATA_ROOT, 'results.csv')
    else:
        results_path = os.path.join(DATA_ROOT, f'{args.model.replace("/", "_")}_results.csv')

    responses_path = os.path.join(DATA_ROOT, f'{args.model.replace("/", "_")}_responses.csv')
    if os.path.exists(responses_path):
        _print(f"Warning: {responses_path} already exists. Responses will be appended to the file.")

    model_name = args.model

    if args.seed is not None:
        torch.manual_seed(args.seed)
        _print(f"Setting random seed to {args.seed}")

    if args.num_samples is None or (args.num_samples == 1):
        num_samples = 1
    else:
        raise NotImplementedError("num_samples > 1 is not implemented yet")

    
    _print(f"Loading MMLU validation split...")
    dataset = load_dataset("cais/mmlu", 'all', split="validation")
    _print(f"Loaded {len(dataset)} examples")
    
    _print(f"\nEvaluating: {model_name}")
    correct, total, prompts, completions, responses, reference_responces = get_responses(model_name, dataset, args.device, args.num_rows)

    _print(f"{model_name} accuracy: {correct}/{total} = {correct/total:.4f}")

    df = pd.DataFrame({
        'prompt': prompts,
        'completion': completions,
        'response': responses,
        'true': reference_responces
    })

    res = {"model": model_name, "seed": args.seed, "num_rows":args.num_rows, "num_samples":args.num_samples, "accuracy": correct/total}
    res = pd.DataFrame([res])

    if args.concat_results and os.path.exists(results_path):
        res = pd.concat([pd.read_csv(results_path), res], ignore_index=True)
        res.to_csv(results_path, index=False)
    else:
        res.to_csv(results_path, index=False)

    if os.path.exists(responses_path):
        df = pd.concat([pd.read_csv(responses_path), df], ignore_index=True)
        df.to_csv(responses_path, index=False)
    else:
        df.to_csv(responses_path, index=False)

    _print(f"Results saved to {results_path}")
    _print(f"Responses saved to {responses_path}")
    
if __name__ == '__main__':
    main()
