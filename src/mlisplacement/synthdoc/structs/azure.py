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