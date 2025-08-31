import torch
torch.cuda.empty_cache()

import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Field, RootModel
from typing import Optional, Union, Literal, ForwardRef, List, Any
from enum import Enum
from guidance import models, system, user, assistant, json as gen_json, gen
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel, Field, RootModel
from typing import Optional, Union
from enum import Enum
from guidance import models, system, user, assistant, json as gen_json
import guidance
from utils import timing_decorator
from guidance.chat import ChatTemplate
import json
import traceback
from pydantic import BaseModel, Field, RootModel
from typing import Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum
import sympy
from sympy import Symbol, Piecewise, sympify, Add, Mul, Pow

import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"

# MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",       
    low_cpu_mem_usage=True,          
)


tok        = AutoTokenizer.from_pretrained(MODEL_ID)


class QwenChatTemplate(ChatTemplate):
    template_str = tok.chat_template
    def get_role_start(self, role_name):
        # adjust tokens if you use a derivative model with different tags
        if role_name == "system":
            return "<|im_start|>system\n"
        elif role_name == "user":
            return "<|im_start|>user\n"
        elif role_name == "assistant":
            return "<|im_start|>assistant\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        # same token for every role in Qwen‑3
        return "<|im_end|>\n"



model = guidance.models.Transformers(hf_model, tok, chat_template=QwenChatTemplate)


class DomainVariable(BaseModel):
    """Defines a single variable the LLM can use in the computation graph."""
    name: str = Field(..., description="The unique identifier for the variable.")
    description: str = Field(..., description="A detailed explanation of what this variable represents.")
    # Optional: You could add type hints, units, etc. for more advanced validation
    unit: Optional[str] = Field(..., description="The unit of the variable")
    data_type : type = Field(..., description="The data type of the variable")

def create_dynamic_variable_enum(charge_category: str, charge_category_variables: dict) -> type(Enum):
    """
    Creates a new Enum class containing only the variables relevant
    to the specified charge category.
    """
    variable_names = charge_category_variables.get(charge_category)
    if not variable_names:
        raise ValueError(f"Unknown charge category: {charge_category}")
    
    # The dictionary for the Enum must have {MEMBER_NAME: value}
    # We'll use uppercase for the member name for convention.
    enum_dict = {name.upper(): name for name in variable_names}
    
    # Create the Enum class dynamically
    return Enum("Var", enum_dict)


# Separate enums for clarity and type safety
class MathOperator(str, Enum):
    ADD = "ADD"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    CEIL = "CEIL"
    
class Comparator(str, Enum):
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    EQUAL_TO = "EQUAL_TO"

class Units(str, Enum):
    HOURS = "HOURS"
    MINUTES = "MINUTES"
    EUROS = "EUROS"
    PERCENT = "PERCENT"
    UNITLESS = "UNITLESS"
    
# --- Node Definitions ---

class ValueNode(BaseModel):
    type: Literal["VALUE"] = "VALUE"
    value: Union[float, int, str]
    description: str
    unit: Units

class CategoricalValueNode(BaseModel):
    type: Literal["CATEGORICAL_VALUE_NODE"] = "CATEGORICAL_VALUE_NODE"
    value: str
    description: str
    unit: Units

class VariableNode(BaseModel):
    type: Literal["VARIABLE"] = "VARIABLE"
    name: str 
    description: str
    unit: Units

class BinaryOpNode(BaseModel):
    """Node for mathematical operations that produce a number."""
    type: Literal["BINARY_OPERATION"] = "BINARY_OPERATION"
    operator: MathOperator
    left: 'AnyNode'
    right: 'AnyNode'

class ComparisonNode(BaseModel):
    """Node for comparison operations that produce a boolean."""
    type: Literal["COMPARISON"] = "COMPARISON"
    operator: Comparator
    left: 'AnyNode'
    right: 'AnyNode'

class ConditionalNode(BaseModel):
    """Node for if-then-else logic."""
    type: Literal["CONDITIONAL"] = "CONDITIONAL"
    condition: ComparisonNode # Condition must be a comparison
    if_true: 'AnyNode'
    if_false: 'AnyNode'

# --- Recursive Setup ---

AnyNode = Union[
    ValueNode, 
    CategoricalValueNode,
    VariableNode, 
    BinaryOpNode, 
    ConditionalNode
]

# Use model_rebuild() to safely resolve all forward references
BinaryOpNode.model_rebuild()
ConditionalNode.model_rebuild()
ComparisonNode.model_rebuild()

class Node(RootModel):
    root: BinaryOpNode



class ParameterStatus(str, Enum):
    """An enumeration for clear, explicit parameter statuses."""
    KNOWN = "KNOWN"
    SYMBOLIC = "SYMBOLIC"

class ParameterDetail(BaseModel):
    """A structured model to describe each parameter identified from the query."""
    name: str = Field(..., description="The name of the parameter.")
    status: ParameterStatus = Field(..., description="Whether the parameter's value is known from the query or is a symbolic variable.")
    value: Optional[Any] = Field(None, description="The actual value of the parameter, if its status is 'KNOWN'. Must be null if status is 'SYMBOLIC'.")


class ReasoningSchemaStep1(BaseModel):
    query_parameters: List[ParameterDetail] = Field(
        ...,
        description="A structured list of all parameters identified from the query and their status."
    )
class ReasoningSchemaStep2(BaseModel):
    """
    A simplified schema for Step 2 that captures all constants and rules as a simple list of descriptive strings.
    """
    identified_constants_and_rules: List[str] = Field(
        ...,
        description="A comprehensive list of all facts, constants, and conditional rules extracted from the document that are necessary for the final calculation. Each string in the list should be a self-contained, clear statement. For example: 'The rate for a Narrow Satellite stand is €32.90 per 15 minutes' or 'A 100% surcharge is applied if parking duration is between 48 and 72 hours'."
    )
class ReasoningSchemaStep3(BaseModel):
    synthesis_plan: str = Field(
        ...,
        description="A concise, step-by-step plan describing how the variables and constants are combined into the final computation graph."
    )
class ReasoningSchemaStep4(BaseModel):
    rethink: str = Field(
        ...,
        description="Final check to ensure the plan correctly uses variables and constants and handles all logic from the document."
    )


@guidance
def create_graph_with_cot(llm, allowed_variables_prompt, document, query, output_schema):
    
    with system():
        llm += f"""You are an expert system that converts textual calculation rules into structured JSON expression trees.
        You MUST think step-by-step and reason before generating the final JSON.
        
        **Reasoning Guidelines:**
        1.  **Analyze Query Parameters (Coverage Required):** Enumerate EVERY variable listed in the 'Allowed Variables' section below. For each variable, output an object with fields: 'name', 'status' ('KNOWN' if the query provides a concrete value; 'SYMBOLIC' otherwise), and 'value' (present only when status is 'KNOWN', null when 'SYMBOLIC').
            - If a variable is categorical (string-valued), you must still include it. Do NOT assume a default category when it is not provided; mark it 'SYMBOLIC'.
            - Do NOT introduce variables that are not in the Allowed Variables list.
            Example: `[ {{"name": "aircraft_stand_type", "status": "KNOWN", "value": "Wide Remote"}}, {{"name": "parking_duration_hours", "status": "SYMBOLIC", "value": null}} ]`
        **Allowed Variables for this Task:**
        ---
        {allowed_variables_prompt}
        ---
        2. **Identify All Relevant Information**: Review the document and extract every fact, constant, and conditional rule needed for the calculation. Each piece of information should be written as a clear, self-contained sentence and collected into a list of strings.
        3.  **Synthesize Plan:** Briefly describe how you will combine these pieces into a final expression tree.
        4. **Rethink and Finalize Approach**: Before processing with generation, rethink your progress so far and make adjustments if necessary, then finalize and proceed to generate the expression tree.

        **Crucial Rule 1:** If a parameter from the 'Allowed Variables' list is given a specific value in the query, you MUST treat it as a fixed value to find constants. You MUST NOT include it as a `VARIABLE` node in the final JSON.
        **Crucial Rule 2**: If a calculation path or value depends on the value of a symbolic variable, you MUST capture the rules for all possible values and represent this logic using CONDITIONAL nodes in the final expression tree. You MUST NOT assume a default value for the variable to simplify the logic.
        **Crucial Rule 3 (Categorical Variables)**: Variables whose domain is categorical (e.g., string-valued like 'flight_type', 'stand_type') must be handled explicitly.
            - If the query supplies a category, treat it as fixed (per Rule 1).
            - If the query does not supply a category, mark the variable as 'SYMBOLIC' in Step 1 and ensure the final expression covers all category-dependent branches stated in the document using `CONDITIONAL` nodes. Do NOT assume a default category. Always use `CATEGORICAL_VALUE_NODE` for category constants.
            - Ensure every variable in the Allowed Variables list appears in Step 1 with a status of KNOWN or SYMBOLIC.
            - Encoding guidance: when checking a categorical branch, use a `COMPARISON` node with operator `EQUAL_TO`, `left` as a `VARIABLE` node for the categorical variable, and `right` as a `CATEGORICAL_VALUE_NODE` whose `value` is the category string (e.g., "International").

        **Crucial Rule 4 (Numeric vs Categorical Comparisons)**:
            - Numeric variables must be compared against numeric `VALUE` nodes whose `value` is a number.
            - Categorical variables must be compared against `CATEGORICAL_VALUE_NODE` nodes whose `value` is the category string.
            - Do NOT compare a categorical variable to a numeric `VALUE` node, and do NOT compare a numeric variable to a `CATEGORICAL_VALUE_NODE`.
        
        After writing your reasoning, you WILL generate the JSON object.


        """

    with user():
        llm += f"""
        **Document:**
        ---
        {document}
        ---

        **Query:**
        
        Based on the document, construct the computation graph for the following request:
        
        "{query}"
        
        """

    with assistant():
        llm += "I will now follow the reasoning guidelines step-by-step before generating the final JSON.\n"
        llm += "Step1. Analyze Query Parameters:\n"
        llm += gen_json(
            name="thought1", 
            schema=ReasoningSchemaStep1, 
            max_tokens=600)

        llm += "Step2. Identify All Relevant Information:\n"
        llm += gen_json(
            name="thought2", 
            schema=ReasoningSchemaStep2, 
            max_tokens=600)

        llm += "Step3. Synthesize Plan:\n"
        llm += gen_json(
            name="thought3", 
            schema=ReasoningSchemaStep3, 
            max_tokens=600)

        llm += "Step4. Rethink and Finalize Approach:\n"
        llm += gen_json(
            name="thought4", 
            schema=ReasoningSchemaStep4, 
            max_tokens=600)

        # After thinking, it generates the JSON.
        llm += "\n\nFinal JSON object:\n"
        llm += gen_json(
            name="result_graph", 
            schema=output_schema,
            max_tokens=2000 
        )
        
    return llm


class ComputationGraphBuilder:
    """
    Orchestrates the creation of a computation graph by preparing dynamic
    constraints and prompting the LLM.
    """
    
    def __init__(self, model):
        """
        Initializes the builder with a guidance model.
        """
        self.model = model
        # Set the default LLM for all guidance programs
        # guidance.llm = self.model
    @timing_decorator
    def build(self, document_content: str, query: str, charge_category: str, charge_category_variables: dict, all_variables: dict) -> dict:
        """
        Generates a computation graph for a given query and document.

        Args:
            document_content: The text containing the rules.
            query: A natural language question about what to calculate.
            charge_category: The specific charge context used to filter variables.

        Returns:
            A dictionary representing the computation graph or an error.
        """
        print(f"--- Building graph for charge category: '{charge_category}' ---")
        
        # 1. Dynamically create the filtered Enum for this specific task
        try:
            Var = create_dynamic_variable_enum(charge_category, charge_category_variables)
        except ValueError as e:
            print(f"Error: {e}")
            return {"error": str(e)}

        # 3. Create a formatted prompt string of allowed variables for the LLM
        allowed_variables = [el.value for el in list(Var)]
        def _var_type(v):
            try:
                return "CATEGORICAL" if v.data_type is str else "NUMERIC"
            except Exception:
                return "UNKNOWN"
        allowed_variables_prompt = "\n".join(
            [
                f"- **{v.name}** | type: {_var_type(v)} | unit: {v.unit} — {v.description}"
                for name, v in all_variables.items() if name in allowed_variables
            ]
        )

        try:
            # 4. Execute the guidance program with all dynamic components
            result_lm = self.model + create_graph_with_cot(
                allowed_variables_prompt=allowed_variables_prompt,
                document=document_content,
                query=query,
                output_schema=Node
            )
            
            
            # print("\nSuccessfully generated graph:")
            # # Use model_dump_json for Pydantic v2
            # print(pydantic_graph.model_dump_json(indent=2)) 
            return result_lm
            
        except Exception as e:
            print(f"\nAn error occurred while building the graph for '{query}': {e}")
            return {"error": str(e)}


def compose_expression(node: dict):
    """
    Recursively parses a JSON graph into a SymPy expression,
    preventing automatic simplification.

    Args:
        node: A dictionary representing a node in the computation graph.

    Returns:
        A non-evaluated sympy expression representing the computation.
    """

    node_type = node.get('type')

    if node_type == "VALUE":
        val = node['value']
        # VALUE should represent numeric constants; allow numeric sympification
        return sympify(val)

    elif node_type in ("CATEGORICAL_VALUE_NODE", "CATEGORICAL_VALUE"):
        # Map raw category strings to valid SymPy Symbols for safe equality checks
        val = node['value']
        sanitized = re.sub(r"\W+", "_", str(val)).strip("_")
        if not sanitized:
            sanitized = "_VAL"
        return Symbol(sanitized)

    elif node_type == "VARIABLE":
        return Symbol(node['name'])

    elif node_type == "BINARY_OPERATION":
        left = compose_expression(node['left'])
        right = compose_expression(node['right'])
        operator = node['operator']
        
        # Use class constructors with evaluate=False to prevent simplification
        if operator == "ADD":
            return Add(left, right, evaluate=False)
        elif operator == "MULTIPLY":
            return Mul(left, right, evaluate=False)
        elif operator == "DIVIDE":
            # Division (a/b) is represented as a * (b**-1)
            power = Pow(right, -1, evaluate=False)
            return Mul(left, power, evaluate=False)
        else:
            raise ValueError(f"Unsupported binary operator: {operator}")

    elif node_type == "COMPARISON":
        left = compose_expression(node['left'])
        right = compose_expression(node['right'])
        operator = node['operator']

        if operator == "GREATER_THAN":
            return left > right
        elif operator == "LESS_THAN":
            return left < right
        elif operator == "EQUAL_TO":
            return sympy.Eq(left, right, evaluate=False)
        elif operator == "NOT_EQUAL_TO":
            return sympy.Ne(left, right, evaluate=False)
        elif operator == "LESS_THAN_OR_EQUAL_TO":
            return sympy.Le(left, right, evaluate=False)
        elif operator == "GREATER_THAN_OR_EQUAL_TO":
            return sympy.Ge(left, right, evaluate=False)
        else:
            raise ValueError(f"Unsupported comparison operator: {operator}")

    elif node_type == "CONDITIONAL":
        condition = compose_expression(node['condition'])
        if_true_expr = compose_expression(node['if_true'])
        if_false_expr = compose_expression(node['if_false'])
        
        return Piecewise((if_true_expr, condition), (if_false_expr, True))

    else:
        raise ValueError(f"Unknown node type: {node_type}")



def create_computation_graph(model, query, charge_category, markdown_content, charge_category_variables, all_variables):
    graph_builder = ComputationGraphBuilder(model=model)

    start_time = time.perf_counter()
    llm_structured_response = graph_builder.build(
        document_content=markdown_content,
        query=query,
        charge_category=charge_category,
        charge_category_variables=charge_category_variables,
        all_variables=all_variables
    )
    end_time = time.perf_counter()
    build_time = end_time - start_time

    return llm_structured_response, build_time

#/gpfs01/home/ppxac9/MLiS-Placement-Thesis/LLM_generated_data/synthetic_dataset/document_20250808_123124.md
#/gpfs01/home/ppxac9/MLiS-Placement-Thesis/LLM_generated_data/synthetic_dataset/output_structure_2025-08-08_11-47-52.json

with open("/gpfs01/home/ppxac9/MLiS-Placement-Thesis/LLM_generated_data/synthetic_dataset/document_20250808_123124.md", "r") as f:
    markdown_content = f.read()

with open("/gpfs01/home/ppxac9/MLiS-Placement-Thesis/LLM_generated_data/synthetic_dataset/output_structure_2025-08-08_11-47-52.json", "r") as f:
    output_structure = json.load(f)

# print(markdown_content)
# print(output_structure)
output_structure.popitem()


charge_category_variables = {k: v['variables_used'] for k, v in output_structure.items()}

all_variables = {
    'aircraft_weight': DomainVariable(
        name='aircraft_weight',
        description='Weight of the aircraft in pounds.',
        unit='pounds',
        data_type=float
    ),
    'flight_type': DomainVariable(
        name='flight_type',
        description='Type of flight (e.g., Domestic, International, Charter).',
        unit='none',
        data_type=str
    ),
    'baggage_weight': DomainVariable(
        name='baggage_weight',
        description='Weight of baggage in kilograms.',
        unit='kilograms',
        data_type=float
    ),
    'fuel_consumption': DomainVariable(
        name='fuel_consumption',
        description='Fuel consumption in gallons.',
        unit='gallons',
        data_type=float
    ),
    'aircraft_type': DomainVariable(
        name='aircraft_type',
        description='Type of aircraft (e.g., Military, Passenger, Cargo, Private).',
        unit='none',
        data_type=str
    )
}

query = "Calculate the total security fee."
charge_category = "security_fee"

result = create_computation_graph(model, query, charge_category, markdown_content, charge_category_variables, all_variables)
