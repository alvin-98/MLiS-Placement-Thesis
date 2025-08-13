from .base import DomainVariable

ALL_VARIABLES = {
    
    'transfer_passenger_count': DomainVariable(name='transfer_passenger_count', description='Total number of transferring passengers.', unit=None, data_type=float),
    'airline_scheduling_season': DomainVariable(name='airline_scheduling_season', description='Whether summer/winter airline scheduling season.', unit=None, data_type=float),
    'takeoff_aircraft_mtow_tonnes': DomainVariable(name='takeoff_aircraft_mtow_tonnes', description='The Maximum Take-Off Weight in tonnes.', unit='tonne', data_type=float),
    'landing_aircraft_mtow_tonnes': DomainVariable(name='landing_aircraft_mtow_tonnes', description='The Maximum Landing Weight in tonnes.', unit='tonne', data_type=float),
    
    'parking_duration_hours': DomainVariable(
        name='parking_duration_hours',
        description='Total duration of parking in hours. Surcharges apply at 48 and 72 hours.',
        unit='hours',
        data_type=float
    ),
    'aircraft_stand_type': DomainVariable(
        name='aircraft_stand_type',
        description='The type of aircraft stand used for parking. E.g., "Wide Contact", "Narrow Remote", "LAP", "Long Term Remote".',
        unit=None,
        data_type=str
    ),
    'parking_location': DomainVariable(
        name='parking_location',
        description='The location of the parking stand, either "EAP" (East Aerodrome Parking) or "WAP" (West Aerodrome Parking).',
        unit=None,
        data_type=str
    ),
    'is_overnight_parking': DomainVariable(
        name='is_overnight_parking',
        description='True if the parking occurs during the free overnight period (2300-0600hrs).',
        unit=None,
        data_type=bool
    )
}
# 2. Map charge types to the variable names they are allowed to use
CHARGE_CATEGORY_VARIABLES = {
    # --- Existing Categories ---
    "transfer_passenger_charge": [
        'transfer_passenger_count', 
        'airline_scheduling_season'
    ],
    "runway_landing_charge": [
        'landing_aircraft_mtow_tonnes', 
        'airline_scheduling_season'
    ],
    "runway_takeoff_charge": [
        'takeoff_aircraft_mtow_tonnes', 
        'airline_scheduling_season'
    ],

    # --- New Categories for Parking Charges ---
    "east_aerodrome_parking_charge": [
        'parking_duration_hours',
        'aircraft_stand_type',
        'is_overnight_parking'
        # 'parking_location' is implicitly 'EAP' for this category
    ],
    "west_aerodrome_parking_charge": [
        'parking_duration_hours',
        'aircraft_stand_type',
        'is_overnight_parking'
        # 'parking_location' is implicitly 'WAP' for this category
    ]
}