from pydantic import BaseModel, Field, RootModel
from typing import Optional, Literal, Union
from enum import Enum

class DomainVariable(BaseModel):
    """Defines a single variable the LLM can use in the computation graph."""
    name: str = Field(..., description="The unique identifier for the variable.")
    description: str = Field(..., description="A detailed explanation of what this variable represents.")
    # Optional: You could add type hints, units, etc. for more advanced validation
    unit: Optional[str] = Field(..., description="The unit of the variable")
    data_type : type = Field(..., description="The data type of the variable")

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