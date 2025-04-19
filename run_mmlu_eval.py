import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm


def evaluate_model(model_name, dataset, device, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # system instruction for multiple-choice exam
    system_instruction = """You are a knowledgeable assistant taking a multiple-choice exam. Read the multiple‑choice question and answer with a single capital letter (A, B, C or D) only."""

    for ex in tqdm(dataset, desc=f"Evaluating {model_name}", total=len(dataset)):
        question = ex['question']
        choices = ex.get('choices', [])
        label = ex['answer']

        choices_str = ' '.join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        prompt = f"{system_instruction}\n Question: {question}\nChoices: {choices_str}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs['input_ids'].size(-1):], skip_special_tokens=True)
        
        tokens = gen.strip().split()
        if not tokens: #fix for if the model generates nothing
            total += 1  
            continue    
        pred = gen.strip().split()[0]

        # normalize true label to letter (index -> letter)
        true = chr(65 + int(label))

        if pred == true:
            correct += 1
        total += 1

    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak_model', default='distilbert/distilgpt2')
    parser.add_argument('--strong_model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to evaluate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Loading MMLU validation split...")
    dataset = load_dataset("cais/mmlu", 'all', split="validation")
    print(f"Loaded {len(dataset)} examples")

    for model_name, label in [(args.weak_model, 'Weak'), (args.strong_model, 'Strong')]:
        print(f"\nEvaluating {label} model: {model_name}")
        correct, total = evaluate_model(model_name, dataset, args.device, args.max_samples)
        print(f"{label} accuracy: {correct}/{total} = {correct/total:.4f}")


if __name__ == '__main__':
    main()
