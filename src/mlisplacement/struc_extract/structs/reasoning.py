from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum
import guidance
from guidance import system, user, assistant
from guidance import json as gen_json
from mlisplacement.utils import timing_decorator

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

#TODO: abstract the global CHARGE_CATEGORY_VARIABLES
def create_dynamic_variable_enum(charge_category: str) -> Enum:
    """
    Creates a new Enum class containing only the variables relevant
    to the specified charge category.

    CHARGE_CATEGORY_VARIABLES provides a mapping from charge categories to the variable names they can use.
    """
    variable_names = CHARGE_CATEGORY_VARIABLES.get(charge_category)
    if not variable_names:
        raise ValueError(f"Unknown charge category: {charge_category}")
    
    # The dictionary for the Enum must have {MEMBER_NAME: value}
    # We'll use uppercase for the member name for convention.
    enum_dict = {name.upper(): name for name in variable_names}
    
    # Create the Enum class dynamically
    return Enum("Var", enum_dict)


#TODO: abstract the global ALL_VARIABLES
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

    @timing_decorator
    def build(self, document_content: str, query: str, charge_category: str) -> dict:
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
            Var = create_dynamic_variable_enum(charge_category)
        except ValueError as e:
            print(f"Error: {e}")
            return {"error": str(e)}

        # 3. Create a formatted prompt string of allowed variables for the LLM
        allowed_variables = [el.value for el in list(Var)]
        allowed_variables_prompt = "\n".join(
            [f"- **{v.name}**: {v.description}" for name, v in ALL_VARIABLES.items() if name in allowed_variables]
        )

        try:
            # 4. Execute the guidance program with all dynamic components
            result_lm = self.model + create_graph_with_cot(
                allowed_variables_prompt=allowed_variables_prompt,
                document=document_content,
                query=query,
                output_schema=Node
            )
            
            return result_lm
            
        except Exception as e:
            print(f"\nAn error occurred while building the graph for '{query}': {e}")
            return {"error": str(e)}
