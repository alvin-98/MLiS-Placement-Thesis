import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Field, RootModel
from typing import Optional, Union, Literal
from enum import Enum
from guidance import models, system, user, assistant, json as gen_json

# 1. Load Model and Tokenizer
MODEL_ID = "Qwen/Qwen3-30B-A3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    low_cpu_mem_usage=True,
    load_in_4bit=True  # Prevents OutOfResources errors
)

# 2. Create Guidance model
model = models.Transformers(hf_model, tokenizer)

# 3. Define the expression tree structure
class Op(str, Enum):
    ADD = "ADD"
    MULTIPLY = "MULTIPLY"

class ValueNode(BaseModel):
    """A leaf node with a numeric value"""
    type: Literal["VALUE"] = "VALUE"
    value: float
    description: str = Field(description="Explanation of what this value represents")

# Define OpNode without its recursive fields first
class OpNode(BaseModel):
    """An operation node with two children"""
    type: Literal["OPERATION"] = "OPERATION"
    operator: Op
    # The 'left' and 'right' fields will be added dynamically

# Now, define the Union for the recursive fields
AnyNode = Union[OpNode, ValueNode]

# Dynamically update the OpNode to add the recursive fields
# This is the standard way to handle recursive models and avoid schema errors
OpNode.model_fields.update({
    'left': (AnyNode, Field(..., discriminator='type')),
    'right': (AnyNode, Field(..., discriminator='type')),
})

# The root of the expression tree must be an operation.
class Node(RootModel):
    """The root of the expression tree, which must be an OpNode."""
    root: OpNode
    model_config = dict(extra="forbid")





# 4. Define conditions
transfer_passenger_count = 30
condition = [
    {"charge_name": "transfer passenger charge", "transfer_passenger_count": transfer_passenger_count, "period": "summer airline scheduling season", "rate": "?"},
    # {"charge_name": "transfer passenger charge", "transfer_passenger_count": transfer_passenger_count, "period": "winter airline scheduling season", "rate": "?"},
    # {"charge_name": "runway landing and takeoff charge", "period": "summer airline scheduling season", "atm": "landing", "per tonne MTOW": 1, "rate": "?"},
    # {"charge_name": "runway landing and takeoff charge", "period": "summer airline scheduling season", "atm": "takeoff", "per tonne MTOW": 1, "rate": "?"},
    # {"charge_name": "runway landing and takeoff charge", "period": "winter airline scheduling season", "atm": "landing", "per tonne MTOW": 1, "rate": "?"},
    # {"charge_name": "runway landing and takeoff charge", "period": "winter airline scheduling season", "atm": "takeoff", "per tonne MTOW": 1, "rate": "?"}
]

# 5. Read the markdown file
with open("output/2025-airport-charges-terms-and-conditions/tinychargesmarkdown.md", "r") as f:
    markdown_content = f.read()

# 6. Define the guidance program using the new chat API
@guidance
def create_expression_tree(markdown_content, cond, pydantic_class):
    
    with system():
        lm = """You are a world-class algorithm for building expression trees from text. Your goal is to construct a JSON object that represents the calculation logic for a 'rate' based on a document and a set of conditions.
        
        You MUST follow the Node schema exactly. The root of the tree must be an 'OpNode'. 'ValueNode's can only be leaf nodes.
        1. A 'ValueNode' has 'type': 'VALUE', a 'value' field containing a number, and a 'description'.
        2. An 'OpNode' has 'type': 'OPERATION', an 'operator' ('ADD' or 'MULTIPLY'), and 'left'/'right' fields which can be other 'OpNode's or 'ValueNode's.

        Here is an example of a valid expression tree:
        {\"type\": \"OPERATION\", \"operator\": \"MULTIPLY\", \"left\": {\"type\": \"VALUE\", \"value\": 3.9, \"description\": \"Transfer passenger charge per passenger\"}, \"right\": {\"type\": \"VALUE\", \"value\": 30.0, \"description\": \"Number of transfer passengers\"}}"""

    with user():
        lm += f"""Here is the document:
---
{markdown_content}
---

Given the following condition:
{cond}

Construct the expression tree for the rate based on the document and condition."""

    with assistant():
        lm += gen_json(
            name="expression_tree", 
            schema=pydantic_class,
            max_tokens=200
        )
    
    return lm

# Set guidance's default LLM to our model
guidance.llm = model

# 7. Loop over conditions and generate the expression tree
all_trees = []
for cond in condition:
    print(f"--- Condition: {cond} ---")
    try:
        # Execute the guidance program
        print("Running inference...")
        result_lm = create_expression_tree(markdown_content=markdown_content, cond=cond, pydantic_class=Node)
        print("Inference completed")
        
        # Extract the generated tree
        expression_tree = result_lm["expression_tree"]
        print("Generated expression tree:")
        # Use .json() for Pydantic v1
        print(expression_tree.json(indent=2))
        
        # Store the dictionary representation in the condition
        cond["rate"] = expression_tree.dict()
        all_trees.append(expression_tree)

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Full error: {repr(e)}")
        cond["rate"] = None

# 8. Print the final conditions with expression trees
print("\n--- Final Result ---")
print(condition)
