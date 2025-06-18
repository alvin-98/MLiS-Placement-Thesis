import json
import argparse
import sys
from transformers import pipeline

def read_markdown_file(filepath):
    """Reads the content of a markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

def get_charge_rules_from_llm(text):
    """
    Uses a Hugging Face LLM to extract charge rules from text into a JSON format.
    """
    prompt = f"""
    You are an expert aviation cost analyst. Your task is to extract all aircraft charging rules from the provided document text.
    Convert all extracted rules into a single, valid JSON object.

    The JSON object must have a single root key: "charges", which contains a list of charge objects.
    
    Each charge object must have:
    - "charge_name": A descriptive name for the charge.
    - "charge_type": A standardized type (e.g., "landing", "passenger", "parking").
    - "rules": A list of rule objects.

    Each rule object defines a rate and the conditions under which it applies. A rule object can have one of two calculation methods:

    1. For simple, non-banded rates, the rule object must have:
      - "calculation_method": "standard"
      - "rate": The numeric value of the charge.
      - "unit": The unit for the rate (e.g., "per_departing_passenger").
      - "conditions": A list of condition objects.

    2. For tiered/banded rates (like charges that change based on weight bands), the rule object must have:
      - "calculation_method": "tiered"
      - "unit": The base unit for the tiers (e.g., "per_tonne_mtow").
      - "tiers": A list of tier objects, ordered from the lowest band to the highest.
          - Each tier object must have a "rate" and a band definition (e.g., {{"up_to": 136}} or {{"above": 136}}).
      - "conditions": A list of general conditions that apply to the entire tiered rule (e.g., season).

    Each condition object must have:
    - "field": The parameter the condition applies to (e.g., "season", "stand_type").
    - "operator": The logical operator (e.g., "equals").
    - "value": The value for the condition.

    Document Text to Analyze:
    ---
    {text}
    ---

    JSON Output:
    """

    print("INFO: Sending prompt to LLM...")

    try:
        # Switched to a smaller model to fit within memory constraints.
        extractor = pipeline('text-generation', model='google/gemma-2-2b-it', model_kwargs={"load_in_8bit": True}) # Use device=0 for GPU
        response = extractor(prompt, max_new_tokens=35000, pad_token_id=extractor.tokenizer.eos_token_id)
        # Clean the response to get only the JSON part
        json_string = response[0]['generated_text'].split("JSON Output:")[1].strip()
        # Find the start and end of the JSON object
        start = json_string.find('{')
        end = json_string.rfind('}') + 1
        json_string = json_string[start:end]
    except Exception as e:
        print(f"Error during LLM inference: {e}", file=sys.stderr)
        return None

    try:
        print("INFO: Parsing JSON response...")
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error: LLM did not return valid JSON. Parser error: {e}", file=sys.stderr)
        print(f"--- Raw LLM Output ---\n{json_string}\n----------------------", file=sys.stderr)
        return None

def write_json_output(data, filepath):
    """Writes the JSON data to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"INFO: Successfully wrote structured data to '{filepath}'")
    except Exception as e:
        print(f"Error writing to output file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to orchestrate the script."""
    parser = argparse.ArgumentParser(
        description="Extract structured charge data from a markdown file using an LLM."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input markdown file."
    )
    parser.add_argument(
        "-o", "--output_file",
        default="output.json",
        help="Path to the output JSON file (default: output.json)."
    )
    args = parser.parse_args()

    # 1. Read input file
    markdown_text = read_markdown_file(args.input_file)

    # 2. Extract rules using the LLM
    extracted_rules = get_charge_rules_from_llm(markdown_text)

    # 3. Write output to JSON file
    if extracted_rules:
        write_json_output(extracted_rules, args.output_file)
    else:
        print("ERROR: Failed to extract rules. Exiting.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
