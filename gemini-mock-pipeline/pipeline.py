import argparse
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
import google.genai as genai 

from schemas import Charges


load_dotenv()
client = genai.Client()

def generate_structured_output(markdown_text: str, model_name: str = "gemini-1.5-flash") -> tuple[Charges | None, dict]:
    """Generate a Charges object using Gemini structured output and return performance metrics."""
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
    {markdown_text}
    ---
    """

    start_time = time.time()
    response_stream = client.models.generate_content(
        model=model_name,
        contents=prompt,
        generation_config={
            "response_mime_type": "application/json",
        },
        stream=True
    )

    ttft = 0
    first_chunk_received_time = 0
    last_chunk_received_time = 0
    token_latencies = []
    full_response_text = ""
    usage_metadata = None

    for chunk in response_stream:
        if not first_chunk_received_time:
            first_chunk_received_time = time.time()
            ttft = first_chunk_received_time - start_time
        
        if last_chunk_received_time:
            token_latencies.append(time.time() - last_chunk_received_time)
        
        last_chunk_received_time = time.time()
        full_response_text += chunk.text
        
        if chunk.usage_metadata:
            usage_metadata = chunk.usage_metadata
            
    total_generation_time = last_chunk_received_time - start_time if last_chunk_received_time else 0
    inter_token_latency = sum(token_latencies) / len(token_latencies) if token_latencies else 0

    metrics = {
        "time_to_first_token_s": ttft,
        "inter_token_latency_ms": inter_token_latency * 1000,
        "total_generation_time_s": total_generation_time,
        "input_tokens": usage_metadata.prompt_token_count if usage_metadata else "N/A",
        "output_tokens": usage_metadata.candidates_token_count if usage_metadata else "N/A",
    }

    try:
        # The model might wrap the JSON in markdown fences
        if full_response_text.strip().startswith("```json"):
            full_response_text = full_response_text.strip()[7:-3]
        parsed_data = Charges.model_validate_json(full_response_text)
        return parsed_data, metrics
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"Error: Failed to parse model output into Charges schema.\n{e}", file=sys.stderr)
        print(f"Raw model output:\n{full_response_text}", file=sys.stderr)
        return None, metrics


def run_pipeline(input_path: Path, output_path: Path | None) -> None:
    markdown_text = input_path.read_text(encoding="utf-8")
    structured, metrics = generate_structured_output(markdown_text)

    if structured:
        # Pretty-print to stdout
        json_output = json.dumps(structured.model_dump(), ensure_ascii=False, indent=2)

        if output_path:
            output_path.write_text(json_output, encoding="utf-8")
            print(f"Structured output written to {output_path}")
        else:
            print(json_output)

    print("\n--- Performance Metrics ---", file=sys.stderr)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}", file=sys.stderr)
        else:
            print(f"{key}: {value}", file=sys.stderr)
    print("-------------------------", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert markdown to structured JSON via Gemini API.")
    parser.add_argument("markdown", type=Path, help="Path to the input markdown file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output file path for the JSON result (defaults to stdout)",
    )

    args = parser.parse_args()

    if not args.markdown.exists():
        parser.error(f"File not found: {args.markdown}")

    run_pipeline(args.markdown, args.output)
