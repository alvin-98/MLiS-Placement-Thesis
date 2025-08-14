from datetime import datetime
import json
from .function_creation import create_charge_func, generate_output_structure
import os
from .structs.airports import CHARGES, VARIABLES


if __name__ == "__main__":
    output_structure = generate_output_structure(CHARGES, VARIABLES)

    # Generate an identifier like 2025-08-08_11-42-05
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_structure["run_id"] = run_id

    # Ensure target folder exists
    save_dir = "LLM_generated_data/synthetic_dataset"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"output_structure_{run_id}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_structure, f, indent=2)

    print(f"Saved output structure to {save_path}")


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

    saved_filename = save_document(document_content, filename=f"synthetic_dataset/document_{run_id}.md")
    print(f"Using JSON file: {json_path}")
    print(f"Document saved to {saved_filename}")