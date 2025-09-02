import json
import re
import random
import argparse
import glob
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from prompts import one_shot_example_1, one_shot_example_2, one_shot_example_3


class LLMClient:
    """
    Simple client for interacting with HF models, with safe defaults for large models.
    - quantization: "4bit" (best for 1x80GB), "8bit", or None (use offload or multi-GPU)
    - model_name_or_path: repo id OR local folder (recommended: local folder you downloaded)
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct",
        quantization: str = "4bit",           # "4bit" | "8bit" | None
        local_only: bool = True,              # use local cache/files only
        device_map: str = "auto",             # let HF place weights across devices
        offload_folder: str | None = None,    # e.g., "/gpfs01/scratch/$USER/offload-70b" (for None/8bit)
    ):
        # Tokenizer (safe offline)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_only,
            use_fast=False,                   # Llama tokenizers are often non-fast
        )

        model_kwargs = dict(
            local_files_only=local_only,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

        # Choose quantization / offload strategy
        if quantization == "4bit":
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_cfg

        elif quantization == "8bit":
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_cfg

        else:
            # Full-precision/bf16 path (does NOT fit in 1x80GB without offload)
            model_kwargs["torch_dtype"] = torch.bfloat16
            if offload_folder:
                model_kwargs["offload_folder"] = offload_folder  # CPU/GPU offload
            # Otherwise expect multiple GPUs for sharding

        # Important: do NOT .to(device); let device_map/quant handle placement
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        # Pipeline can use the already-placed model (no device arg needed)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_at: list[str] | None = None,   # e.g., ["</table>"] to clamp HTML tables
    ) -> str:
        # 1) System message
        system_prompt = (
            "You are a professional assistant trained to generate policy documentation and HTML tables for cloud pricing documents. "
            "You always follow instructions precisely and do not include any explanation, commentary, or additional text beyond what is requested. "
            "When asked to generate a table, output only valid HTML starting with a <table> tag. "
            "When asked to write a paragraph, respond in a concise, informative, and professional tone. "
            "Do not refer to the user, and do not include disclaimers or formatting notes. "
            "Maintain consistency with real-world cloud pricing policies and use appropriate domain terminology. "
            "The goal is to create documentation as close to human-written as possible."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # 2) Build a chat prompt using the model's template (string form)
        chat_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 3) Tokenize once; we’ll use model.generate() directly for precise slicing
        #    (more reliable than string-slicing pipeline output)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            # Llama tokenizers often have no pad_token; align to eos to avoid warnings
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # Move to the same device(s) as the model without collapsing placement.
        # If the model is sharded (device_map="auto"), .generate handles it; just send inputs to the first device.
        first_device = next(self.model.parameters()).device
        for k in inputs:
            inputs[k] = inputs[k].to(first_device)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            repetition_penalty=1.05,
        )

        # 4) Slice out ONLY the newly generated tokens
        new_tokens = gen_ids[0, inputs["input_ids"].shape[1]:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 5) Optional hard stop trimming (useful for HTML tables)
        if stop_at:
            lower = output_text.lower()
            cut = None
            for s in stop_at:
                i = lower.find(s.lower())
                if i != -1:
                    j = i + len(s)
                    cut = j if cut is None else min(cut, j)
            if cut is not None:
                output_text = output_text[:cut].strip()

        return output_text


def extract_table(html_str: str) -> str:
    m = re.search(r"<table.*?>.*?</table>", html_str, re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else "This charge is temporarily suspended due to technical issues."


def generate_document_table(llm_client: LLMClient, provider_name: str, currency:str, charge_description: str, max_retries: int = 3) -> str:
    if random.random() < 0.5: # add more variety to tables with two different options
        one_shot_example = one_shot_example_1
    else:
        one_shot_example = one_shot_example_2
    prompt = f"""Write a professionally human readable HTML table for a cloud pricing policy from {provider_name}.
    The currency of this table is {currency} - be sure to include this
    Return only the HTML table, nothing else.
    All information should be in a single table.
    examples of the type of charge expected are as follows:
    {one_shot_example}
    {one_shot_example_3}
    INPUT: {charge_description}
    OUTPUT:"""
    max_tokens = 1000
    for attempt in range(max_retries):
        html = llm_client.generate_text(prompt, max_tokens=max_tokens)
        if "<table" in html.lower() and "</table>" in html.lower():
            return html
        if attempt == 1:
            max_tokens = 2000
    return ""


def generate_charge_description(llm_client: LLMClient, provider_name: str,currency: str, charge_description: str) -> str:
    prompt = f"""Write a professional description for a cloud pricing charge category from {provider_name}.
Be concise, informative, and suitable for a pricing policy document.
Charge description: {charge_description}. 
Be sure to mention the curreny : {currency}
Return 1–2 paragraphs explaining what the charge covers and its purpose.
The description should be written in enough detail such that a user could calculate their exact charge from the description
Do not include disclaimers or additional text."""
    return llm_client.generate_text(prompt, max_tokens=500)


def decide_charge_format(llm_client: LLMClient, provider_name: str,currency: str, entry: Dict[str, Any], table_chance: float = 0.7) -> str:
    desc = entry["description"]
    # variables_used is now a dict like {var_name: {unit, description, dtype}}
    used_meta = entry.get("variables_used", {}) or {}
    used_count = len(used_meta) if isinstance(used_meta, dict) else len(used_meta)
    want_table = used_count > 1 or random.random() < table_chance
    if want_table:
        html = generate_document_table(llm_client, provider_name,currency, desc, max_retries=2)
        return extract_table(html) if html else generate_charge_description(llm_client, provider_name, currency, desc)
    return generate_charge_description(llm_client, provider_name, currency, desc)


def generate_provider_name() -> str: #can be updated to multiple i suppose
    return "Microsoft Azure"


def make_section_title(llm_client: LLMClient, provider_name: str, category: str) -> str:
    prompt = f"""Create a professional section title for a cloud pricing policy from {provider_name}.
The title should be concise and clearly indicate the charge category(s).
Category: {category}.
No more than 20 words."""
    return llm_client.generate_text(prompt, max_tokens=20)


def generate_document_introduction(llm_client: LLMClient, provider_name: str, currency: str) -> str:
    prompt = f"""Write a professional introduction for a cloud pricing policy from {provider_name}.
This document outlines the fee structure and pricing policies for cloud services in {currency}.
Keep it formal and welcoming, 2–3 sentences. Include the provider name and note this is an official policy document."""
    return llm_client.generate_text(prompt, max_tokens=150)


def generate_document_conclusion(llm_client: LLMClient, provider_name: str, currency: str) -> str:
    prompt = f"""Write a professional conclusion for a cloud pricing policy from {provider_name}.
The policies cover various cloud services charged in {currency}.
Include a brief note on how to contact support for questions. Keep it formal and brief, 2–3 sentences."""
    return llm_client.generate_text(prompt, max_tokens=150)


def extract_first_paragraph(text: str) -> str:
    m = re.search(r'(.*?)(?:\n\s*\n|$)', text.strip(), re.DOTALL)
    return m.group(1).strip() if m else ""


def choose_currency() -> str:
    return random.choice(["USD", "EUR", "GBP"])


def load_generated_charges(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, dict) and "description" in v and "code" in v}


def find_latest_output_json(pattern: str = "output_structure_*.json") -> str:
    base_dir = os.path.join("LLM_generated_data", "synthetic_dataset")
    search_pattern = os.path.join(base_dir, pattern)
    files = glob.glob(search_pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {search_pattern} found.")
    try:
        files_sorted = sorted(files, key=lambda f: datetime.strptime(
            os.path.basename(f)[17:-5], "%Y-%m-%d_%H-%M-%S"), reverse=True)
    except Exception:
        files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    return files_sorted[0]


def generate_document_from_json(client: LLMClient, output_json_path: str) -> Tuple[str, List[str], List[str], Dict[str, Any]]:
    data = load_generated_charges(output_json_path)
    currency = choose_currency()
    provider_name = extract_first_paragraph(generate_provider_name())
    introduction = generate_document_introduction(client, provider_name, currency)
    conclusion = generate_document_conclusion(client, provider_name, currency)

    charge_names = list(data.keys())
    titles = [make_section_title(client, provider_name, charge.replace("_", " ").title()) for charge in charge_names]
    rendered_sections, answers, names = [], [], []

    for charge, title in zip(charge_names, titles):
        entry = data[charge]
        section_html_or_text = decide_charge_format(client, provider_name, currency, entry)
        rendered_sections.append(f"## {title}\n\n{section_html_or_text}")
        answers.append(entry.get("code", ""))
        names.append(charge)

    doc = f"# {provider_name} Cloud Pricing Policy Document\n\n{introduction}\n\n"
    doc += "\n\n".join(rendered_sections)
    doc += f"\n\n{conclusion}\n\n"
    doc += f"**Policy Effective Date:** {datetime.now().strftime('%B %d, %Y')}\n\n"
    return doc, answers, names, data  



def create_charge_answer_pair(answers: List[str], variables: List[str]) -> Tuple[str, str]:
    if not answers or not variables:
        return "", ""
    idx = random.randint(0, len(answers) - 1)
    return answers[idx], variables[idx]


def save_document(content: str, filename: str = None) -> str:
    base_dir = os.path.join("LLM_generated_data", "synthetic_dataset")
    os.makedirs(base_dir, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(base_dir, f"charging_policy_{timestamp}.md")
    else:
        if not os.path.isabs(filename):
            filename = os.path.join(base_dir, filename)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return filename


def save_datapoint_to_jsonl(datapoint: Any, filename: str = None):
    base_dir = os.path.join("LLM_generated_data", "synthetic_dataset")
    os.makedirs(base_dir, exist_ok=True)
    if filename is None:
        filename = os.path.join(base_dir, "datapoints_70B_4bit.jsonl")
    with open(filename, "a", encoding="utf-8") as f:
        json.dump(datapoint, f)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="Path to a specific output_structure JSON file", default=None)
    args = parser.parse_args()

    if args.json:
        json_path = os.path.join("LLM_generated_data", "synthetic_dataset", args.json)
    else:
        json_path = find_latest_output_json()

    client = LLMClient(model_name='Meta-Llama-3.1-70B-Instruct')
    # 3) Capture `source_dict` from the generator
    document_content, answers, names, source_dict = generate_document_from_json(client, json_path)

    # (You can keep target/charge_type logic if you still need it elsewhere)
    target, charge_type = create_charge_answer_pair(answers, names)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 4) Build the exact structure requested: [document, dict]
    datapoint_list = [document_content, source_dict]

    # 5) Save the list to JSONL (one list per line)
    save_datapoint_to_jsonl(datapoint_list)

    saved_filename = save_document(document_content, filename=f"document_{run_id}.md")
    print(f"Using JSON file: {json_path}")
    print(f"Document saved to {saved_filename}")
