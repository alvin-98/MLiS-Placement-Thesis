import argparse
import os
import re
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
import pandas as pd
from tqdm.auto import tqdm


def load_4bit(model_name_or_path: str, local_only: bool = False):
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_only,
        use_fast=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 4-bit config
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        local_files_only=local_only,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    return tok, model

@torch.inference_mode()
def generate_completions(
    model_name: str,
    dataset,
    df: pd.DataFrame,
    do_sample: bool,
    temperature: float,
    max_samples=None,
    batch_size: int = 1,          
    num_generations: int = 5,
    local_only: bool = False,
):
    os.makedirs(f"HarmBench/data/{model_name}", exist_ok=True)

    tokenizer, model = load_4bit(model_name, local_only=local_only)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    prompts, metadata = [], []
    for ex in dataset:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        metadata.append((ex["category"], ex["prompt"]))


    first_device = next(model.parameters()).device

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"{model_name} Generation Progress"):
        batch_prompts = prompts[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(first_device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]

        for sample_num in range(num_generations):
            gen_ids = model.generate(
                **enc,
                max_new_tokens=512,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.9 if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.05,
            )
            new_tokens = gen_ids[:, input_len:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for j, gen in enumerate(decoded):
                prompt_number = i + j + 1
                category, orig_prompt = batch_metadata[j]
                row = [model_name, temperature, prompt_number, category, orig_prompt, gen, sample_num]
                df.loc[len(df)] = row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama', default='Meta-Llama-3.1-70B-Instruct')
    parser.add_argument('--deepseek', default='Deepseek-70B')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--no_sample', dest='do_sample', action='store_false')
    parser.set_defaults(do_sample=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1)          
    parser.add_argument('--num_generations', type=int, default=5)      
    parser.add_argument('--local_only', action='store_true')           
    args = parser.parse_args()

    print("Torch has GPU access:", torch.cuda.is_available())
    print("Loading HarmBench...")
    dataset = load_dataset("walledai/HarmBench", "standard", split="train")
    print(f"Loaded {len(dataset)} examples")

    for model_name, label in [(args.llama, 'Llama-3.1-70B-4bit'), (args.deepseek, 'DeepSeek-70B-4bit')]:
        print(f"\nGenerating with {label}: {model_name}")
        df = pd.DataFrame(columns=['model_name', 'temperature', 'prompt_number', 'category', 'prompt', 'LLM_response', 'sample_num'])

        generate_completions(
            model_name=model_name,
            dataset=dataset,
            df=df,
            do_sample=args.do_sample,
            temperature=args.temperature,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            num_generations=args.num_generations,
            local_only=args.local_only,
        )

        out_path = f'HarmBench/data/{model_name}/HarmBench_llm_judge_samples_{args.do_sample}_temperature_{args.temperature}.csv'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print("Saved:", out_path)

if __name__ == '__main__':
    main()
