from enum import Enum
import guidance
from guidance import system, user, assistant
from guidance import json as gen_json
from mlisplacement.utils import timing_decorator
from mlisplacement.structs.reasoning import (
    ReasoningSchemaStep1,
    ReasoningSchemaStep2,
    ReasoningSchemaStep3,
    ReasoningSchemaStep4
)
from mlisplacement.structs.base import Node
import time

@guidance
def create_graph_with_cot(model: guidance.models.Transformers, allowed_variables_prompt: list[str], document: str, query: str, output_schema) -> guidance.models.Transformers:
    """Creates a structured guidance prompt for reasoning about a document.
    
    Args:
        model (guidance.models.Transformers): The language model to use for reasoning.
        allowed_variables_prompt (list[str]): A list of allowed variables for the task.
        document (str): The document to analyze.
        query (str): The user's query.
        output_schema (dict): A pydantic schema defining the output structure.

    Returns:
        guidance.models.Transformers: The updated language model with reasoning steps.

    Example: #TODO

    """

    with system():
        model += f"""You are an expert system that converts textual calculation rules into structured JSON expression trees.
        You MUST think step-by-step and reason before generating the final JSON.
        
        **Reasoning Guidelines:**
        1.  **Analyze Query Parameters:** Identify all relevant parameters from the user's query. For each parameter, create a structured object specifying its 'name', its 'status' ('KNOWN' if the value is given, or 'SYMBOLIC' if it's a variable), and its 'value' (or null if symbolic). For example: `[ {{"name": "aircraft_stand_type", "status": "KNOWN", "value": "Wide Remote"}}, {{"name": "parking_duration_hours", "status": "SYMBOLIC", "value": null}} ]`
        **Allowed Variables for this Task:**
        ---
        {allowed_variables_prompt}
        ---
        2. **Identify All Relevant Information**: Review the document and extract every fact, constant, and conditional rule needed for the calculation. Each piece of information should be written as a clear, self-contained sentence and collected into a list of strings.
        3.  **Synthesize Plan:** Briefly describe how you will combine these pieces into a final expression tree.
        4. **Rethink and Finalize Approach**: Before processing with generation, rethink your progress so far and make adjustments if necessary, then finalize and proceed to generate the expression tree.

        **Crucial Rule 1:** If a parameter from the 'Allowed Variables' list is given a specific value in the query, you MUST treat it as a fixed value to find constants. You MUST NOT include it as a `VARIABLE` node in the final JSON.
        **Crucial Rule 2**: If a calculation path or value depends on the value of a symbolic variable, you MUST capture the rules for all possible values and represent this logic using CONDITIONAL nodes in the final expression tree. You MUST NOT assume a default value for the variable to simplify the logic.
        
        After writing your reasoning, you WILL generate the JSON object.


        """

    with user():
        model += f"""
        **Document:**
        ---
        {document}
        ---

        **Query:**
        
        Based on the document, construct the computation graph for the following request:
        
        "{query}"
        
        """

    with assistant():
        model += "I will now follow the reasoning guidelines step-by-step before generating the final JSON.\n"
        model += "Step1. Analyze Query Parameters:\n"
        model += gen_json(
            name="thought1", 
            schema=ReasoningSchemaStep1, 
            max_tokens=600)

        model += "Step2. Identify All Relevant Information:\n"
        model += gen_json(
            name="thought2", 
            schema=ReasoningSchemaStep2, 
            max_tokens=600)

        model += "Step3. Synthesize Plan:\n"
        model += gen_json(
            name="thought3", 
            schema=ReasoningSchemaStep3, 
            max_tokens=600)

        model += "Step4. Rethink and Finalize Approach:\n"
        model += gen_json(
            name="thought4", 
            schema=ReasoningSchemaStep4, 
            max_tokens=600)

        # After thinking, it generates the JSON.
        model += "\n\nFinal JSON object:\n"
        model += gen_json(
            name="result_graph", 
            schema=output_schema,
            max_tokens=2000 
        )

    return model

def create_variables_prompt(charge_name, all_charges: dict, all_variables: dict) -> str:
    """
    Creates a formatted prompt string of allowed variables for the LLM.

    Args:
        charge_name (str): The name of the charge category.
        all_charges (dict): A dictionary of all charge categories and their variables.
        all_variables (dict): A dictionary of all variable names and their descriptions.

    Returns:
        str: A formatted string listing all allowed variables.
    """
    # check charge_name is valid
    if charge_name not in all_charges:
        raise ValueError(f"Unknown charge category: {charge_name}. Available categories: {list(all_charges.keys())}")

    allowed_variables = all_charges[charge_name]["variables"]
    if not allowed_variables:
        raise ValueError(f"Unknown charge category: {charge_name}")
    return "\n".join(
        [f"- **{v['name']}**: {v['description']}" for name, v in all_variables.items() if name in allowed_variables]
    )
class ComputationGraphBuilder:
    """
    Orchestrates the creation of a computation graph by preparing dynamic
    constraints and prompting the LLM.
    """

    def __init__(self, model, output_schema = Node):
        """
        Initializes the builder with a guidance model.
        """
        self.model = model
        self.output_schema = output_schema

    @timing_decorator
    def build(self, document_content: str, query: str, charge_name: str, all_charges: dict, all_variables: dict) -> dict:
        """
        Generates a computation graph for a given query and document.

        Args:
            document_content: The text containing the rules.
            query: A natural language question about what to calculate.
            charge_name: The specific charge context used to filter variables.


        Returns:
            A dictionary representing the computation graph or an error.
        """
        print(f"--- Building graph for charge: '{charge_name}' ---")

        allowed_variables_prompt = create_variables_prompt(
            charge_name, 
            all_charges, 
            all_variables
        )

        try:
            # Execute the guidance program with all dynamic components
            result_lm = self.model + create_graph_with_cot(
                allowed_variables_prompt=allowed_variables_prompt,
                document=document_content,
                query=query,
                output_schema=self.output_schema
            )
            
            return result_lm
            
        except Exception as e:
            print(f"\nAn error occurred while building the graph for '{query}': {e}")
            return {"error": str(e)}
        
def generate(model, query, charge_name, markdown_content, all_charges, all_variables):
    """Generate text using structured reasoning"""
    graph_builder = ComputationGraphBuilder(model=model)

    start_time = time.perf_counter()
    llm_structured_response = graph_builder.build(
        document_content=markdown_content,
        query=query,
        charge_name=charge_name,
        all_charges=all_charges,
        all_variables=all_variables
    )
    end_time = time.perf_counter()
    build_time = end_time - start_time

    return llm_structured_response, build_time

