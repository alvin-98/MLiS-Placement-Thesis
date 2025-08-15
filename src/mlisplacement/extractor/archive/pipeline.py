import json
import argparse
import sys
from typing import List, Literal, Union, Optional

# Third-party imports
try:
    import outlines  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pydantic import BaseModel
except ImportError as e:
    missing_pkg = str(e).split(" ")[-1].strip("'")
    print(f"Error: Required package '{missing_pkg}' is not installed. Please 'pip install outlines transformers pydantic'.", file=sys.stderr)
    sys.exit(1)

###############################################################################
# 1. Pydantic schema matching the required JSON structure                     #
###############################################################################

class Condition(BaseModel):
    field: str
    operator: str
    value: str | int | float


class Tier(BaseModel):
    rate: float
    # Exactly one of the following keys should be present for a given tier.
    up_to: Optional[float] = None  # e.g. {"up_to": 136}
    above: Optional[float] = None  # e.g. {"above": 136}


class RuleStandard(BaseModel):
    calculation_method: Literal["standard"] = "standard"
    rate: float
    unit: str
    conditions: List[Condition]


class RuleTiered(BaseModel):
    calculation_method: Literal["tiered"] = "tiered"
    unit: str
    tiers: List[Tier]
    conditions: List[Condition]


Rule = Union[RuleStandard, RuleTiered]


class Charge(BaseModel):
    charge_name: str
    charge_type: str
    rules: List[Rule]


class ChargesOutput(BaseModel):
    charges: List[Charge]

###############################################################################
# 2. Utility functions                                                        #
###############################################################################

def read_markdown_file(filepath: str) -> str:
    """Read and return the contents of a markdown file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at '{filepath}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error reading file: {exc}", file=sys.stderr)
        sys.exit(1)


def write_json_output(data: ChargesOutput, filepath: str) -> None:
    """Serialize `data` to JSON and write it to `filepath`."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data.model_dump(mode="json"), f, indent=2)
        print(f"INFO: Successfully wrote structured data to '{filepath}'.")
    except Exception as exc:
        print(f"Error writing to output file: {exc}", file=sys.stderr)
        sys.exit(1)

###############################################################################
# 3. LLM extraction using Outlines                                            #
###############################################################################

def get_charge_rules_with_outlines(text: str) -> ChargesOutput | None:
    """Extract charge rules from the supplied text using Outlines for structured output."""

    prompt = (
        "You are an expert aviation cost analyst. "
        "Extract all aircraft charging rules from the following document text. "
        "Return ONLY a JSON object that matches the provided schema.\n\n"
        "Document Text:\n---\n" + text + "\n---\n"
    )

    # NOTE: Adjust the model below if you prefer another LLM.
    MODEL_NAME = "Qwen/Qwen3-30B-A3B"

    try:
        print("INFO: Loading model… (this may take a while on first run)")
        model = outlines.from_transformers(
            AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                trust_remote_code=True,
                # load_in_8bit=True,  # keeps memory usage reasonable
            ),
            AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True),
        )

        print("INFO: Sending prompt to LLM via Outlines…")
        raw_json = model(
            prompt,
            ChargesOutput,  # The Pydantic schema that Outlines must follow
            max_new_tokens=4096,
        )

        # Convert the returned JSON string into a validated ChargesOutput instance
        structured = ChargesOutput.model_validate_json(raw_json)
        return structured

    except Exception as exc:
        print(f"Error during LLM inference: {exc}", file=sys.stderr)
        return None

###############################################################################
# 4. CLI entry-point                                                          #
###############################################################################

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured charge data from a markdown file using an LLM with Outlines for guaranteed JSON output.",
    )
    parser.add_argument("input_file", help="Path to the input markdown file.")
    parser.add_argument(
        "-o",
        "--output_file",
        default="output.json",
        help="Path to the output JSON file (default: output.json).",
    )
    args = parser.parse_args()

    markdown_text = read_markdown_file(args.input_file)

    extracted = get_charge_rules_with_outlines(markdown_text)
    if extracted is None:
        print("ERROR: Failed to extract rules.", file=sys.stderr)
        sys.exit(1)

    write_json_output(extracted, args.output_file)


if __name__ == "__main__":
    main()
