import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel


try:
    from outlines import Generator, from_transformers
    from outlines.types import JsonSchema
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    missing_pkg = str(e).split(" ")[-1].strip("'")
    print(f"Error: Required package '{missing_pkg}' is not installed. Please run 'pip install outlines transformers torch pydantic'.", file=sys.stderr)
    sys.exit(1)

class Condition(BaseModel):
    field: str
    operator: str
    value: str

class Tier(BaseModel):
    rate: float
    up_to: Optional[int] = None
    above: Optional[int] = None

class Rule(BaseModel):
    calculation_method: str
    rate: Optional[float] = None
    unit: str
    tiers: Optional[List[Tier]] = None
    conditions: Optional[List[Condition]] = None

class Charge(BaseModel):
    charge_name: str
    charge_type: str
    rules: List[Rule]

class Charges(BaseModel):
    charges: List[Charge]

def generate_structured_output_hf(
    markdown_text: str, model_name: str = "Qwen/Qwen3-30B-A3B"
) -> Tuple[Optional[Charges], Optional[dict]]:
    """Generate a Charges object using a Hugging Face model with Outlines, and compute performance metrics."""
    prompt = f"""
    You are an expert aviation cost analyst. Your task is to extract all aircraft charging rules from the provided document text.
    Convert all extracted rules into a single, valid JSON object that strictly follows the provided schema.

    Document Text to Analyze:
    ---
    {markdown_text}
    ---
    """

    try:
        print(f"INFO: Loading model '{model_name}'... (this may take a while on first run)")
        model = from_transformers(
            AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
            ),
            AutoTokenizer.from_pretrained(model_name, trust_remote_code=True),
        )

        print("INFO: Sending prompt to LLM via Outlines...")
        tokenizer = model.tokenizer
        input_token_count = len(tokenizer.encode(prompt))

        # Set generation parameters on the underlying model's config
        model.model.generation_config.max_new_tokens = 4096

        # Convert Pydantic model to JSON Schema and create a generator
        schema_str = json.dumps(Charges.model_json_schema())
        generator = Generator(model, JsonSchema(schema_str))

        # Use the generator to get a stream of tokens
        stream_generator = generator(prompt)

        start_time = time.perf_counter()
        first_token_time = None
        token_times = []
        output_tokens = []

        for token in stream_generator:
            current_time = time.perf_counter()
            if first_token_time is None:
                first_token_time = current_time
            token_times.append(current_time)
            output_tokens.append(token)

        if not output_tokens:
            raise ValueError("Model produced no output.")

        # --- Performance Metrics Calculation ---
        ttft = first_token_time - start_time
        total_generation_time = token_times[-1] - start_time
        raw_json = "".join(output_tokens)
        output_token_count = len(tokenizer.encode(raw_json))
        
        # Inter-token latency
        if len(token_times) > 1:
            inter_token_latencies = [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))]
            avg_itl = sum(inter_token_latencies) / len(inter_token_latencies)
        else:
            avg_itl = 0  # Not applicable for a single token

        metrics = {
            "time_to_first_token_s": ttft,
            "avg_inter_token_latency_ms": avg_itl * 1000,
            "total_generation_time_s": total_generation_time,
            "input_tokens": input_token_count,
            "output_tokens": output_token_count,
            "tokens_per_second": output_token_count / total_generation_time if total_generation_time > 0 else 0,
        }

        # --- Final Output Processing ---
        structured_output = Charges.model_validate_json(raw_json)
        return structured_output, metrics

    except Exception as e:
        print(f"Error during LLM inference: {e}", file=sys.stderr)
        return None, None

def run_pipeline(input_path: Path, output_path: Path) -> None:
    """Read input, generate structured data, and write to output file.""" 
    print(f"INFO: Reading input file: {input_path}")
    try:
        markdown_text = input_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
        sys.exit(1)

    structured_data, metrics = generate_structured_output_hf(markdown_text)

    if structured_data and metrics:
        print(f"INFO: Writing structured output to {output_path}")
        json_output = json.dumps(structured_data.model_dump(), ensure_ascii=False, indent=2)
        output_path.write_text(json_output, encoding="utf-8")
        
        # Print performance metrics
        print("\n--- Performance Metrics ---")
        print(f"Input Tokens:              {metrics['input_tokens']}")
        print(f"Output Tokens:             {metrics['output_tokens']}")
        print(f"Time to First Token:       {metrics['time_to_first_token_s']:.3f} s")
        print(f"Avg. Inter-Token Latency:  {metrics['avg_inter_token_latency_ms']:.2f} ms")
        print(f"Tokens per Second:         {metrics['tokens_per_second']:.2f}")
        print(f"Total Generation Time:     {metrics['total_generation_time_s']:.3f} s")
        print("---------------------------")
        print("INFO: Pipeline completed successfully.")
    else:
        print("ERROR: Failed to generate structured data. Exiting.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert markdown to structured JSON via a Hugging Face model.")
    parser.add_argument("markdown_file", type=Path, help="Path to the input markdown file")
    args = parser.parse_args()

    output_file = Path("output.json")

    if not args.markdown_file.exists():
        parser.error(f"File not found: {args.markdown_file}")

    run_pipeline(args.markdown_file, output_file)
