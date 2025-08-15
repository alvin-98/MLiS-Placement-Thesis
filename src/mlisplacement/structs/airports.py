"""Defines the domain variables and charge categories for airport-related data.
It includes variables for aircraft, passengers, and various airport charges.

Terminology:
A **variable** is a named entity that can hold a value, such as the weight of an aircraft or the number of passengers.
A *charge**
"""
from .base import DomainVariable, ChargeModel
from typing import Dict, List
from dataclasses import dataclass, field

_VARIABLES: List[DomainVariable] = [
    DomainVariable(
        name="aircraft_weight",
        dtype=float,
        values=(5000.0, 400000.0),
        units="kg",
        alternative_names=["aircraft mass", "plane weight"],
        description="The weight of the aircraft in kilograms."
    ),
    DomainVariable(
        name="passenger_count",
        dtype=int,
        values=(10, 500),
        units="passengers",
        alternative_names=["passenger number", "traveler count"],
        description="The number of passengers on the aircraft."
    ),
    DomainVariable(
        name="fuel_consumption",
        dtype=float,
        values=(500.0, 5000.0),
        units="liters",
        alternative_names=["fuel usage", "fuel burn"],
        description="The amount of fuel consumed by the aircraft in liters."
    ),
    DomainVariable(
        name="baggage_weight",
        dtype=float,
        values=(5.0, 50.0),
        units="kg",
        alternative_names=["luggage weight", "baggage mass"],
        description="The weight of the baggage carried by passengers in kilograms."
    ),
    DomainVariable(
        name="aircraft_type",
        dtype=str,
        values=["passenger", "cargo", "private", "military"],
        units="aircraft type",
        alternative_names=["plane type", "aircraft category"],
        description="The type of the aircraft."
    ),
    DomainVariable(
        name="flight_type",
        dtype=str,
        values=["domestic", "international", "charter"],
        units="flight type",
        alternative_names=["flight category", "flight class"],
        description="The type of the flight."
    ),
]

_CHARGES = [
    ChargeModel(
        name="security_fee",
        description="A fee charged for security measures at the airport.",
        variables=["aircraft_weight", "passenger_count", "baggage_weight"],
        synonyms=["security charge", "safety fee"],
    ),
    ChargeModel(
        name="landing_fee",
        description="A fee charged for landing the aircraft at the airport.",
        variables=["aircraft_weight", "passenger_count", "fuel_consumption", "baggage_weight", "aircraft_type", "flight_type"],
        synonyms=["landing charge", "arrival fee"],
    ),
    ChargeModel(
        name="fuel_tax",
        description="A tax applied to the fuel consumed by the aircraft.",
        variables=["fuel_consumption", "aircraft_weight", "flight_type"],
        synonyms=["fuel charge", "fuel levy"],
    ),
    ChargeModel(
        name="baggage_fee",
        description="A fee charged for the baggage carried by passengers.",
        variables=["baggage_weight", "aircraft_type"],
        synonyms=["luggage charge", "baggage levy"],
    ),
    ChargeModel(
        name="passenger_service_fee",
        description="A fee charged for services provided to passengers at the airport.",
        variables=["passenger_count", "aircraft_type", "flight_type"],
        synonyms=["passenger charge", "service fee"],
    ),
    ChargeModel(
        name="airport_facility_fee",
        description="A fee charged for the use of airport facilities.",
        variables=["aircraft_type", "flight_type"],
        synonyms=["facility charge", "airport usage fee"],
    ),
]

@dataclass
class AirportVariables:
    """Container for airport-related variables."""
    variables: List[DomainVariable] = field(default_factory=lambda: list(_VARIABLES))

    def to_dict(self) -> Dict[str, Dict]:
        """Converts the list of variables to a dictionary."""
        return {var.name: var.model_dump() for var in self.variables}

@dataclass
class AirportCharges:
    """Container for airport-related charges."""
    charges: List[ChargeModel] = field(default_factory=lambda: list(_CHARGES))

    def to_dict(self) -> Dict[str, Dict]:
        """Converts the list of charges to a dictionary."""
        return {charge.name: charge.model_dump() for charge in self.charges}