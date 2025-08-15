"""Defines structures used for structured reasoning with llms.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum

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
