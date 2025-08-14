import json
import os
from datetime import datetime
from typing import Dict, Any
import glob

########################################################
# Import/Export Utilities
########################################################

def load_jsonl(filename):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_documents_as_markdown(dataset, output_dir="synthetic_dataset/markdown_outputs"):

    os.makedirs(output_dir, exist_ok=True)

    for i, entry in enumerate(dataset):
        doc_text = entry.get("document")
        doc_id = entry.get("document_id", f"doc_{i+1}")
        if doc_text:
            filename = os.path.join(output_dir, f"document_{doc_id}.md")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(doc_text)
            print(f"Saved: {filename}")
        else:
            print(f"Skipping entry {i} (no document field)")

# # Usage
# dataset = load_jsonl("synthetic_dataset/datapoints.jsonl")
# save_documents_as_markdown(dataset)

def save_document(content: str, filename: str = None) -> str:
    base_dir = os.path.join("LLM_generated_data", "synthetic_dataset")
    os.makedirs(base_dir, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(base_dir, f"charging_policy_{timestamp}.md")
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

def load_generated_charges(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if isinstance(v, dict) and "description" in v and "code" in v}