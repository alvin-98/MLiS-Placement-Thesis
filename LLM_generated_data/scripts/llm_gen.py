import json
import re
import random
import argparse
import glob
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from prompts import one_shot_example_1, one_shot_example_2, one_shot_example_3


class LLMClient:
    """Simple client for interacting with Hugging Face Transformers models."""
    def __init__(self, model_name: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct', device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1
        )

    def generate_text(self, prompt: str, max_tokens: int = 200) -> str:
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
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.generator(
            chat_prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        return generated[len(chat_prompt):].strip()


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


def generate_document_from_json(client: LLMClient, output_json_path: str) -> Tuple[str, List[str], List[str]]:
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
        section_html_or_text = decide_charge_format(client, provider_name,currency, entry)
        rendered_sections.append(f"## {title}\n\n{section_html_or_text}")
        answers.append(entry.get("code", ""))
        names.append(charge)

    doc = f"# {provider_name} Cloud Pricing Policy Document\n\n{introduction}\n\n"
    doc += "\n\n".join(rendered_sections)
    doc += f"\n\n{conclusion}\n\n"
    doc += f"**Policy Effective Date:** {datetime.now().strftime('%B %d, %Y')}\n\n"
    return doc, answers, names


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


def save_datapoint_to_jsonl(datapoint: Dict[str, Any], filename: str = None):
    base_dir = os.path.join("LLM_generated_data", "synthetic_dataset")
    os.makedirs(base_dir, exist_ok=True)
    if filename is None:
        filename = os.path.join(base_dir, "datapoints.jsonl")
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

    client = LLMClient(model_name='meta-llama/Meta-Llama-3.1-8B-Instruct')
    document_content, answers, names = generate_document_from_json(client, json_path)

    target, charge_type = create_charge_answer_pair(answers, names)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    datapoint = {
        "charge_type": charge_type,
        "document": document_content,
        "target": target,
        "document_id": run_id,
    }
    save_datapoint_to_jsonl(datapoint)

    saved_filename = save_document(document_content, filename=f"document_{run_id}.md")
    print(f"Using JSON file: {json_path}")
    print(f"Document saved to {saved_filename}")
