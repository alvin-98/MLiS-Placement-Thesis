from .base import DomainVariable, ChargeModel
from typing import Dict, List
from dataclasses import dataclass, field

CHARGES = {"cpu_compute": {
    "description": "The charge associated with a CPU compute instance.",
    "variables": ["cpu_usage", "processing_time", "cpu_count", "cpu_type"],
    "synonyms": ["CPU time", "processing hours"]
}}

VARIABLES = {
    "cpu_usage": {
        "description": "The CPU usage during processing.",
        "dtype": float,
        "values": (0, 100.0),
        "units": "percentage",
        "alternative_names": ["CPU load", "CPU utilization"]
    },
    "processing_time": {
        "description": "The time taken for processing.",
        "dtype": float,
        "values": (0, 3600.0),  # in seconds
        "units": "seconds",
        "alternative_names": ["processing duration", "execution time"]
    },
    "cpu_count": {
        "description": "The number of CPUs used for processing.",
        "dtype": int,
        "values": (0, 64),  # assuming a maximum of 64 CPUs
        "units": "CPUs",
        "alternative_names": ["CPU cores", "processor count"]
    },
    "cpu_type": {
        "description": "The type of CPU used for processing.",
        "dtype": str,
        "values": ["Intel", "AMD", "ARM"],
        "units": "CPU type",
        "alternative_names": ["processor type", "CPU model"]
    }
}

_VARIABLES: List[DomainVariable] = [
    DomainVariable(name="cpu_usage", description=VARIABLES["cpu_usage"]["description"], dtype=VARIABLES["cpu_usage"]["dtype"], values=VARIABLES["cpu_usage"]["values"], units=VARIABLES["cpu_usage"]["units"], alternative_names=VARIABLES["cpu_usage"]["alternative_names"]),
    DomainVariable(name="processing_time", description=VARIABLES["processing_time"]["description"], dtype=VARIABLES["processing_time"]["dtype"], values=VARIABLES["processing_time"]["values"], units=VARIABLES["processing_time"]["units"], alternative_names=VARIABLES["processing_time"]["alternative_names"]),
    DomainVariable(name="cpu_count", description=VARIABLES["cpu_count"]["description"], dtype=VARIABLES["cpu_count"]["dtype"], values=VARIABLES["cpu_count"]["values"], units=VARIABLES["cpu_count"]["units"], alternative_names=VARIABLES["cpu_count"]["alternative_names"]),
    DomainVariable(name="cpu_type", description=VARIABLES["cpu_type"]["description"], dtype=VARIABLES["cpu_type"]["dtype"], values=VARIABLES["cpu_type"]["values"], units=VARIABLES["cpu_type"]["units"], alternative_names=VARIABLES["cpu_type"]["alternative_names"])
]

_CHARGES: List[ChargeModel] = [
    ChargeModel(
        name="cpu_compute",
        description=CHARGES["cpu_compute"]["description"],
        variables=CHARGES["cpu_compute"]["variables"],
        synonyms=CHARGES["cpu_compute"]["synonyms"]
    )
]   

@dataclass
class AzureVariables:
    """Container for Azure-related variables."""
    variables: List[DomainVariable] = field(default_factory=lambda: list(_VARIABLES))

    def to_dict(self) -> Dict[str, Dict]:
        """Converts the list of variables to a dictionary."""
        return {var.name: var.model_dump() for var in self.variables}
    
@dataclass
class AzureCharges:
    """Container for Azure-related charges."""
    charges: List[ChargeModel] = field(default_factory=lambda: list(_CHARGES))

    def to_dict(self) -> Dict[str, Dict]:
        """Converts the list of charges to a dictionary."""
        return {charge.name: charge.model_dump() for charge in self.charges}