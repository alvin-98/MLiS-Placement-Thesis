from pydantic import BaseModel
from typing import List, Optional

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
    tiers: Optional[List['Tier']] = None
    conditions: Optional[List[Condition]] = None

class Charge(BaseModel):
    charge_name: str
    charge_type: str
    rules: List[Rule]

class Charges(BaseModel):
    charges: List[Charge]

# Resolve forward references in Pydantic models
Rule.model_rebuild()
