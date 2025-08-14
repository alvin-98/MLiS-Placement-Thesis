import random
from typing import List, Tuple, Dict, Any
import itertools

def create_conditional_func_categorical(variable_name: str, possible_values: list[str]) -> tuple[str, str]:
    """Creates the string representation of a function that computes charges based on categorical variable conditions.
    
    Args:
        variable_name (str): The name of the variable to create conditions for.
        possible_values (list[str]): The possible values for the categorical variable.

    Returns:
        tuple[str, str]: A tuple containing the function code as a string and a description of

    Example usage:
        >>> func, desc = create_categorical_conditions("binary", ["0", "1"])
        >>> # func will contain the function code, desc will contain the description
        >>> # Example output:
        >>> print(func)
        def compute_charge(binary):
            if binary == '1': return 0
            elif binary == '0': return 1
            else: raise ValueError(f'Unknown value for binary: {binary}')
        >>> print(desc)
            if binary is '1', charge is 0; if binary is '0', charge is 1

    """
    shuffled_values = random.sample(possible_values, len(possible_values))
    body_lines = []
    description_clauses = []
    for idx, val in enumerate(shuffled_values):
        line = f"{'if' if idx == 0 else 'elif'} {variable_name} == '{val}': return {idx}"
        description = f"if {variable_name} is '{val}', charge is {idx}"
        body_lines.append(f"    {line}")
        description_clauses.append(description)
    body_lines.append(f"    else: raise ValueError(f'Unknown value for {variable_name}: {{{variable_name}}}')")
    function_code = f"def compute_charge({variable_name}):\n" + "\n".join(body_lines)
    full_description = "; ".join(description_clauses)
    return function_code, full_description

def create_conditional_func_numeric(
    variable_name: str,
    value_range: tuple[float, float],
    value_type: type = float
) -> Tuple[str, str]:
    """Creates the string representation of a function that computes charges based on numeric variable conditions.

    Args:
        variable_name (str): The name of the variable to create conditions for.
        value_range (tuple[float, float]): The range of values for the numeric variable.
        value_type (type): The type of the variable (float or int).

    Returns:
        Tuple[str, str]: A tuple containing the function code as a string and a description of the conditions.

    Example usage:
        >>> func, desc = create_numeric_conditions("weight", (100.0, 2000.0), float)
        >>> # func will contain the function code, desc will contain the description
        >>> # Example output:
        >>> print(func)
        def compute_charge(weight):
            if 1861.7 <= weight < 2000.0: return 0 * weight
            elif 624.4 <= weight < 1861.7: return 1 * weight
            elif 100.0 <= weight < 624.4: return 2 * weight
            else: raise ValueError(f'weight is out of expected range')
        >>> print(desc)
        if weight in [1861.7, 2000.0), charge is 0 x weight; if weight in [624.4, 1861.7), charge is 1 x weight; if weight in [100.0, 624.4), charge is 2 x weight
        
    """
    min_val, max_val = value_range
    num_splits = random.randint(1, 3)
    if value_type == float:
        thresholds = sorted(round(random.uniform(min_val, max_val), 1) for _ in range(num_splits))
        fmt = lambda x: f"{round(x, 1):.1f}"
    elif value_type == int:
        thresholds = sorted(random.randint(int(min_val), int(max_val)) for _ in range(num_splits))
        fmt = lambda x: str(int(x))
    else:
        raise ValueError("value_type must be type float or int")
    split_bounds = [min_val] + thresholds + [max_val]
    if random.choice([True, False]):
        split_bounds = split_bounds[::-1]
    body_lines = []
    description_clauses = []
    for idx in range(len(split_bounds) - 1):
        lower = min(split_bounds[idx], split_bounds[idx+1])
        upper = max(split_bounds[idx], split_bounds[idx+1])
        lower_str = fmt(lower)
        upper_str = fmt(upper)
        line = f"{'if' if idx == 0 else 'elif'} {lower_str} <= {variable_name} < {upper_str}: return {idx} * {variable_name}"
        body_lines.append(f"    {line}")
        description_clauses.append(f"if {variable_name} in [{lower_str}, {upper_str}), charge is {idx} x {variable_name}")
    body_lines.append(f"    else: raise ValueError(f'{variable_name} is out of expected range')")
    function_code = f"def compute_charge({variable_name}):\n" + "\n".join(body_lines)
    full_description = "; ".join(description_clauses)
    return function_code, full_description

def create_conditional_func_categorical_numeric(
    charge_name: str,
    cat_var: str,
    cont_var: str,
    cat_vals: List[str],
    cont_vals: Tuple[float, float],
    cat_dtype: type,
    cont_dtype: type
) -> Tuple[str, str]:
    """Create a conditional function for a charge based on a combination of categorical and continuous variables.
    
    Args:
        charge_name (str): The name of the charge being computed.
        cat_var (str): The name of the categorical variable.
        cont_var (str): The name of the continuous variable.
        cat_vals (List[str]): The possible values for the categorical variable.
        cont_vals (Tuple[float, float]): The range of values for the continuous variable.
        cat_dtype (type): The data type of the categorical variable (should be str).
        cont_dtype (type): The data type of the continuous variable (should be float or int).

    Returns:
        Tuple[str, str]: A tuple containing the function-body code as a string and a description of the conditions.

    Example usage:
        >>> func, description = create_conditional_categorical_numeric(
        "my_charge", "binary_cat_var", "float_var", ["0", "1"], (100.0, 2000.0), str, float)
        >>> # func will contain the function code, description will contain the description
        >>> # Example output:
        >>> print(func)
        def compute_charge(binary_cat_var, float_var):
            if binary_cat_var == '0' and 100.0 <= float_var < 353.6: return 0 * float_var * 1.0
            elif binary_cat_var == '0' and 353.6 <= float_var < 1105.9: return 1 * float_var * 1.0
            elif binary_cat_var == '0' and 1105.9 <= float_var < 1793.5: return 2 * float_var * 1.0
            elif binary_cat_var == '0' and 1793.5 <= float_var < 2000.0: return 3 * float_var * 1.0
            elif binary_cat_var == '1' and 100.0 <= float_var < 353.6: return 0 * float_var * 1.1
            elif binary_cat_var == '1' and 353.6 <= float_var < 1105.9: return 1 * float_var * 1.1
            elif binary_cat_var == '1' and 1105.9 <= float_var < 1793.5: return 2 * float_var * 1.1
            elif binary_cat_var == '1' and 1793.5 <= float_var < 2000.0: return 3 * float_var * 1.1
            else: raise ValueError(f'binary_cat_var or float_var is out of expected range')
        >>> print(description)
            my_charge conditions by binary_cat_var and float_var:

            for 0 the following charges apply:
            - if float_var in [100.0, 922.0), charge is 0.0 x float_var
            - if float_var in [922.0, 1078.7), charge is 1.0 x float_var
            - if float_var in [1078.7, 2000.0), charge is 2.0 x float_var
            for 1 the following charges apply:
            - if float_var in [100.0, 922.0), charge is 0.0 x float_var
            - if float_var in [922.0, 1078.7), charge is 1.1 x float_var
            - if float_var in [1078.7, 2000.0), charge is 2.2 x float_var
    """
    code_lines: List[str] = []
    description_clauses = f"{charge_name} conditions by {cat_var} and {cont_var}:\n"

    # cont_dtype = VARIABLES[cont_var]["dtype"]
    min_val, max_val = cont_vals
    num_splits = random.randint(1, 3)

    if cont_dtype == float:
        thresholds = sorted(round(random.uniform(min_val, max_val), 1) for _ in range(num_splits))
        fmt = lambda x: f"{round(x, 1):.1f}"
    elif cont_dtype == int:
        thresholds = sorted(random.randint(int(min_val), int(max_val)) for _ in range(num_splits))
        fmt = lambda x: f"{int(x)}"
    else:
        raise ValueError(f"Unsupported continuous dtype: {cont_dtype}")
    
    split_bounds = [min_val] + thresholds + [max_val]
    if random.choice([True, False]):
        split_bounds = split_bounds[::-1]
    for outer_idx, val in enumerate(cat_vals):
        intermediate_code_lines = []
        intermediate_description = f"\nfor {val} the following charges apply:"
        multiplier = 1 + outer_idx if cont_dtype == int else 1 + (outer_idx / 10)
        for idx in range(len(split_bounds) - 1):
            lower = min(split_bounds[idx], split_bounds[idx+1])
            upper = max(split_bounds[idx], split_bounds[idx+1])
            lower_str = fmt(lower)
            upper_str = fmt(upper)
            charge_str = round(idx * multiplier, 2) if cont_dtype == float else int(idx * multiplier)
            condition_line = (
                f"{'if' if idx == 0 and outer_idx == 0 else 'elif'} "
                f"{cat_var} == '{val}' and {lower_str} <= {cont_var} < {upper_str}: "
                f"return {idx} * {cont_var} * {multiplier}"
            )
            description_line = (
                f"  - if {cont_var} in [{lower_str}, {upper_str}), charge is {charge_str} x {cont_var}"
            )
            intermediate_code_lines.append(condition_line)
            intermediate_description += f"\n{description_line}"
        code_lines.extend(intermediate_code_lines)
        description_clauses += intermediate_description
    code_lines.append(f"else: raise ValueError(f'{cat_var} or {cont_var} is out of expected range')")
    body = "\n".join(code_lines)

    # wrap into a full function for consistency with single-variable path
    code = f"def compute_charge({cat_var}, {cont_var}):\n    " + body.replace("\n", "\n    ")

    return code, description_clauses

def create_charge_func(charge_name: str, all_charges: dict, all_variables: dict) -> tuple[str, str, list[str]]:
    """Randomly chooses from the variables associated with a chosen charge and creates a conditional function based on the variable types.
    Preference is given to generating a function that combines a categorical variable with a numeric variable.
    If no such pair exists, a single variable is randomly selected and processed.

    Args:
        charge_name (str): The name of the charge to process.
        all_charges (dict): A dictionary containing all charges and their associated variables.
        all_variables (dict): A dictionary containing all variables and their properties.

    Returns:
        tuple[str, str, list[str]]: A tuple containing the description of the charge, the function code as a string, and a list of variable names used in the function.

    Raises:
        ValueError: If the charge name is not found in all_charges or if no variables are available for the charge.
        KeyError: If a variable name is not found in all_variables.
        TypeError: If the variable's dtype is not supported.

    Example usage:
        #Consider the following example of a charge for a CPU compute instance

        all_charges = {"cpu_compute": {
        "description": "The charge associated with a CPU compute instance.",
        "variables": ["cpu_usage", "processing_time", "cpu_count", "cpu_type"],
        "synonyms": ["CPU time", "processing hours"]
        }}

        all_variables = {
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

        >>> desc, code, var_names = create_charge_func("cpu_compute", all_charges, all_variables)
        >>> # desc will contain the description of the charge, code will contain the function code, var_names will contain the variable names used

        >>> print(desc)
            cpu_compute conditions by cpu_type and cpu_usage:

            for Intel the following charges apply:
            - if cpu_usage in [0.0, 34.3), charge is 0.0 x cpu_usage
            - if cpu_usage in [34.3, 100.0), charge is 1.0 x cpu_usage
            for AMD the following charges apply:
            - if cpu_usage in [0.0, 34.3), charge is 0.0 x cpu_usage
            - if cpu_usage in [34.3, 100.0), charge is 1.1 x cpu_usage
            for ARM the following charges apply:
            - if cpu_usage in [0.0, 34.3), charge is 0.0 x cpu_usage
            - if cpu_usage in [34.3, 100.0), charge is 1.2 x cpu_usage

        >>> print(code)
            def compute_charge(cpu_type, cpu_usage):
                if cpu_type == 'Intel' and 0.0 <= cpu_usage < 34.3: return 0 * cpu_usage * 1.0
                elif cpu_type == 'Intel' and 34.3 <= cpu_usage < 100.0: return 1 * cpu_usage * 1.0
                elif cpu_type == 'AMD' and 0.0 <= cpu_usage < 34.3: return 0 * cpu_usage * 1.1
                elif cpu_type == 'AMD' and 34.3 <= cpu_usage < 100.0: return 1 * cpu_usage * 1.1
                elif cpu_type == 'ARM' and 0.0 <= cpu_usage < 34.3: return 0 * cpu_usage * 1.2
                elif cpu_type == 'ARM' and 34.3 <= cpu_usage < 100.0: return 1 * cpu_usage * 1.2
                else: raise ValueError(f'cpu_type or cpu_usage is out of expected range')
        >>> print(var_names)
            ['cpu_type', 'cpu_usage']

    """

    if charge_name not in all_charges:
        raise ValueError(f"Charge '{charge_name}' not found in all_charges.")

    variables = all_charges[charge_name]["variables"]

    if not variables:
        raise ValueError(f"No variables available for charge '{charge_name}'.")
    
    var_pairs = list(itertools.combinations(variables, 2))
    valid_pairs = []

    for v1, v2 in var_pairs:
        dtype1 = all_variables[v1]["dtype"]
        dtype2 = all_variables[v2]["dtype"]
        if (dtype1 == str and dtype2 in (int, float)) or (dtype2 == str and dtype1 in (int, float)):
            valid_pairs.append((v1, v2))

    if valid_pairs: #randomly select a pair of variables to process
        cat_var_name, cont_var_name = random.choice(valid_pairs)
        if all_variables[cat_var_name]["dtype"] != str:
            cat_var_name, cont_var_name = cont_var_name, cat_var_name

        cat_vals = all_variables[cat_var_name]["values"]
        cont_vals = all_variables[cont_var_name]["values"]
        cat_dtype = all_variables[cat_var_name]["dtype"]
        cont_dtype = all_variables[cont_var_name]["dtype"]

        if cat_dtype != str or cont_dtype not in (float, int):
            raise ValueError(f"Unsupported variable types: {cat_var_name} ({cat_dtype}), {cont_var_name} ({cont_dtype})")
        code, desc = create_conditional_func_categorical_numeric(charge_name, cat_var_name, cont_var_name, cat_vals, cont_vals, cat_dtype, cont_dtype)
        return desc, code, [cat_var_name, cont_var_name]

    else: # randomly select a single variable to process
        variable_name = random.choice(variables)
        var = all_variables.get(variable_name)
        if var is None:
            raise KeyError(f"Variable with name '{variable_name}' not found.")
        dtype = var["dtype"]
        values = var["values"]
        units = var["units"]
        if dtype == str:
            code, desc = create_conditional_func_categorical(variable_name, values)
            # omit units sentence for categoricals to avoid awkward phrasing
        elif dtype in (int, float):
            code, desc = create_conditional_func_numeric(variable_name, values, dtype)
            if units:
                desc += f" The variable '{variable_name}' is measured in {units}."
        else:
            raise TypeError(f"Unsupported dtype for variable with name '{variable_name}': {dtype}")

    return desc, code, [variable_name]

def generate_output_structure(all_charges: dict, all_variables: dict) -> Dict[str, Any]:
    """Generates functions for all charges and modifies their descriptions based on variable names.
    
    Args:
        all_charges (dict): A dictionary containing all charges and their associated variables.
        all_variables (dict): A dictionary containing all variables and their properties.

    Returns:
        Dict[str, Any]: A dictionary containing the output structure with charge names as keys and their
                        descriptions, code, variables used, synonyms, and charge descriptions as values.

    """
    output_structure: Dict[str, Any] = {}
    for charge_name, charge_info in all_charges.items():
        desc, code, used_var_names = create_charge_func(charge_name, all_charges, all_variables)

        for var_name in used_var_names: #iterate over used variables and modify the description if alternative names are available
            alt_names = all_variables[var_name]["alternative_names"]
            if alt_names:  # only modify if there are alternative names available
                alt_name = random.choice(alt_names)
                desc = desc.replace(var_name, alt_name)

        output_structure[charge_name] = {
            "description": desc,
            "code": code,
            "variables_used": used_var_names,
            "synonyms": charge_info.get("synonyms", []),
            "charge_description": charge_info["description"],
        }
    return output_structure