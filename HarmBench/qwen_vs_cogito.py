import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import pandas as pd
import re
import os

def generate_completions(model_name, dataset, device, df, do_sample, temperature, max_samples=None, batch_size=4, num_generations=5):
    os.makedirs(f"HarmBench/data/{model_name}", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts = []
    metadata = []

    for ex in dataset:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex['prompt']}],
            tokenize=False,
            add_generation_prompt=True)
        prompts.append(prompt)
        metadata.append((ex['category'], ex['prompt']))

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name} Generation Progress"):
        batch_prompts = prompts[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True, truncation=True).to(device)

        for sample_num in range(num_generations):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=do_sample,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id
                )

            decoded = tokenizer.batch_decode(outputs[:, inputs['input_ids'].size(1):], skip_special_tokens=True)

            for j, gen in enumerate(decoded):
                prompt_number = i + j + 1
                category, orig_prompt = batch_metadata[j]
                row = [model_name, temperature, prompt_number, category, orig_prompt, gen, sample_num]
                df.loc[len(df)] = row

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama', default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('--deepseek', default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--no_sample', dest='do_sample', action='store_false')
    parser.set_defaults(do_sample=True)
    parser.add_argument('--temperature', type=float, default=1.1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_generations', type=int, default=5)
    args = parser.parse_args()

    print("Torch has GPU access:", torch.cuda.is_available())
    print(f"Loading HarmBench...")
    dataset = load_dataset("walledai/HarmBench", 'standard', split="train")
    print(f"Loaded {len(dataset)} examples")

    for model_name, label in [(args.llama, 'Llama'), (args.deepseek, 'DeepSeek')]:
        print(f"\nGenerating with {label} model: {model_name}")

        df = pd.DataFrame(columns=['model_name', 'temperature', 'prompt_number', 'category', 'prompt', 'LLM_response', 'sample_num'])

        generate_completions(model_name, dataset, args.device, df, args.do_sample, args.temperature,
                             args.max_samples, args.batch_size, args.num_generations)

        df.to_csv(
            f'HarmBench/data/{model_name}/HarmBench_llm_judge_samples_{args.do_sample}_temperature_{args.temperature}.csv',
            index=False
        )

if __name__ == '__main__':
    main()
