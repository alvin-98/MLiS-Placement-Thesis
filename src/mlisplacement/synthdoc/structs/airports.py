CHARGES = {
    "security_fee": {
        "description": "A fee charged for security measures at the airport.",
        "variables": ["aircraft_weight", "passenger_count", "baggage_weight"],
        "synonyms": ["security charge", "safety fee"],
    },
    "landing_fee": {
        "description": "A fee charged for landing the aircraft at the airport.",
        "variables": ["aircraft_weight", "passenger_count", "fuel_consumption", "baggage_weight", "aircraft_type", "flight_type"],
        "synonyms": ["landing charge", "arrival fee"],
    },
    "fuel_tax": {
        "description": "A tax applied to the fuel consumed by the aircraft.",
        "variables": ["fuel_consumption", "aircraft_weight", "flight_type"],
        "synonyms": ["fuel charge", "fuel levy"],
    },
    "baggage_fee": {
        "description": "A fee charged for the baggage carried by passengers.",
        "variables": ["baggage_weight", "aircraft_type"],
        "synonyms": ["luggage charge", "baggage levy"],
    },
    "passenger_service_fee": {
        "description": "A fee charged for services provided to passengers at the airport.",
        "variables": ["passenger_count", "aircraft_type", "flight_type"],
        "synonyms": ["passenger charge", "service fee"],
    },
    "airport_facility_fee": {
        "description": "A fee charged for the use of airport facilities.",
        "variables": ["aircraft_type", "flight_type"],
        "synonyms": ["facility charge", "airport usage fee"],
    }
}

VARIABLES = {
    "aircraft_weight": {
        "dtype": float,
        "values": (5000.0, 400000.0),
        "units": "kg",
        "alternative_names": ["aircraft mass", "plane weight"]
    },
    "passenger_count": {
        "dtype": int,
        "values": (10, 500),
        "units": "passengers",
        "alternative_names": ["passenger number", "traveler count"]
    },
    "fuel_consumption": {
        "dtype": float,
        "values": (500.0, 5000.0),
        "units": "liters",
        "alternative_names": ["fuel usage", "fuel burn"]
    },
    "baggage_weight": {
        "dtype": float,
        "values": (5.0, 50.0),
        "units": "kg",
        "alternative_names": ["luggage weight", "baggage mass"]
    },
    "aircraft_type": {
        "dtype": str,
        "values": ["passenger", "cargo", "private", "military"],
        "units": "aircraft type",
        "alternative_names": ["plane type", "aircraft category"]
    },
    "flight_type": {
        "dtype": str,
        "values": ["domestic", "international", "charter"],
        "units": "flight type",
        "alternative_names": ["flight category", "flight class"]
    },
}