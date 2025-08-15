"""Contains the base structures used in the packages, implmented in pydantic.
This includes domain variables, mathematical operators, and node definitions.
"""
from pydantic import BaseModel, Field, RootModel
from typing import Optional, Literal, Union, List
from enum import Enum

# VARIABLES = {
#     "aircraft_weight": {
#         "dtype": float,
#         "values": (5000.0, 400000.0),
#         "units": "kg",
#         "alternative_names": ["aircraft mass", "plane weight"]
#     },
#     "passenger_count": {
#         "dtype": int,
#         "values": (10, 500),
#         "units": "passengers",
#         "alternative_names": ["passenger number", "traveler count"]
#     },
#     "fuel_consumption": {
#         "dtype": float,
#         "values": (500.0, 5000.0),
#         "units": "liters",
#         "alternative_names": ["fuel usage", "fuel burn"]
#     },
#     "baggage_weight": {
#         "dtype": float,
#         "values": (5.0, 50.0),
#         "units": "kg",
#         "alternative_names": ["luggage weight", "baggage mass"]
#     },
#     "aircraft_type": {
#         "dtype": str,
#         "values": ["passenger", "cargo", "private", "military"],
#         "units": "aircraft type",
#         "alternative_names": ["plane type", "aircraft category"]
#     },
#     "flight_type": {
#         "dtype": str,
#         "values": ["domestic", "international", "charter"],
#         "units": "flight type",
#         "alternative_names": ["flight category", "flight class"]
#     },
# }

class DomainVariable(BaseModel):
    """Defines a single variable the LLM can use in the computation graph."""
    name: str = Field(..., description="The unique identifier for the variable.")
    description: str = Field(..., description="A detailed explanation of what this variable represents.")
    # Optional: You could add type hints, units, etc. for more advanced validation
    units: Optional[str] = Field(..., description="The unit of the variable")
    dtype : type = Field(..., description="The data type of the variable")
    alternative_names: Optional[List[str]] = Field(
        None,
        description="A list of alternative names or terms used for this variable"
    )
    values: Optional[tuple | list[str]] = Field(
        None,
        description=""""A tuple or list of valid values for this variable. 
        If the variable is categorical, this should be a list of strings. 
        If it is numerical, this should be a tuple indicating the range (min, max)."""
    )

class ChargeModel(BaseModel):
    """Defines a charge category with its description, variables, and synonyms."""
    name: str = Field(..., description="The unique identifier for the charge category.")
    description: str = Field(..., description="A detailed explanation of the charge.")
    variables: list[str] = Field(..., description="List of variable names that are relevant to this charge.")
    synonyms: list[str] = Field(..., description="Alternative names or terms used for this charge.")

# Separate enums for clarity and type safety
class MathOperator(str, Enum):
    ADD = "ADD"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"

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
    value: float
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
    VariableNode, 
    BinaryOpNode, 
    ConditionalNode
]

#TODO: place this...
# Use model_rebuild() to safely resolve all forward references
BinaryOpNode.model_rebuild()
ConditionalNode.model_rebuild()
ComparisonNode.model_rebuild()

class Node(RootModel):
    root: BinaryOpNode