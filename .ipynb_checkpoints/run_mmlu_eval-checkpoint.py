import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def evaluate_model(model_name, dataset, device, max_samples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    correct = 0
    total = 0

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    for ex in dataset:
        question = ex['question']
        # options may be under 'choices' or 'options'
        choices = ex.get('choices', ex.get('options', []))
        label = ex['answer']

        choices_str = ' '.join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
        prompt = f"Question: {question}\nChoices: {choices_str}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        gen = tokenizer.decode(outputs[0][inputs['input_ids'].size(-1):], skip_special_tokens=True)
        pred = gen.strip().split()[0]

        # normalize true label to letter
        if len(label) == 1 and label in [chr(65+i) for i in range(len(choices))]:
            true = label
        else:
            true = chr(65 + choices.index(label))

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
    dataset = load_dataset("cais/mmlu", split="validation")
    print(f"Loaded {len(dataset)} examples")

    for model_name, label in [(args.weak_model, 'Weak'), (args.strong_model, 'Strong')]:
        print(f"\nEvaluating {label} model: {model_name}")
        correct, total = evaluate_model(model_name, dataset, args.device, args.max_samples)
        print(f"{label} accuracy: {correct}/{total} = {correct/total:.4f}")


if __name__ == '__main__':
    main()
