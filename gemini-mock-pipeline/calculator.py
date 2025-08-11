import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from schemas import Charges, Condition, Rule


class ChargeCalculator:
    """Calculates aviation charges based on a set of rules and flight inputs."""

    def __init__(self, charges_data: Charges):
        self.charges_data = charges_data

    def _clean_value(self, value: Any) -> Any:
        """Sanitizes a value for comparison, converting to float if possible."""
        if isinstance(value, str):
            value = value.replace('%', '').strip()
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    def _check_condition(self, condition: Condition, inputs: Dict[str, Any]) -> bool:
        """Checks if a single condition is met by the provided inputs."""
        field = condition.field
        operator = condition.operator
        rule_value = condition.value

        if field not in inputs:
            return False

        input_value = inputs[field]

        # Handle special time_of_day between operator (e.g., 2300-0600)
        if field == "time_of_day" and operator == "between" and rule_value == "2300_0600":
            try:
                time = int(self._clean_value(input_value))
                return time >= 2300 or time < 600
            except (ValueError, TypeError):
                return False

        cleaned_input = self._clean_value(input_value)
        
        if operator == "equals":
            return str(input_value).lower() == str(rule_value).lower()
        if operator == "less_than":
            return isinstance(cleaned_input, (int, float)) and cleaned_input < self._clean_value(rule_value)
        if operator == "greater_than_or_equals":
            return isinstance(cleaned_input, (int, float)) and cleaned_input >= self._clean_value(rule_value)
        if operator == "in":
            return str(input_value) in str(rule_value).split(',')
        if operator == "between":
            try:
                low, high = map(self._clean_value, rule_value.split('_and_'))
                return isinstance(cleaned_input, (int, float)) and low <= cleaned_input < high
            except ValueError:
                return False

        return False

    def _check_all_conditions(self, rule: Rule, inputs: Dict[str, Any]) -> bool:
        """Checks if all conditions for a given rule are met."""
        if not rule.conditions:
            return True
        return all(self._check_condition(c, inputs) for c in rule.conditions)

    def _calculate_standard_cost(self, rule: Rule, inputs: Dict[str, Any]) -> float:
        """Calculates cost for a simple, non-tiered rule."""
        rate = rule.rate if rule.rate is not None else 0.0
        unit = rule.unit
        cleaned_inputs = {k: self._clean_value(v) for k, v in inputs.items()}

        if unit == "per_departing_passenger":
            return rate * cleaned_inputs.get("departing_passengers", 0)
        if unit == "per_transfer_passenger":
            return rate * cleaned_inputs.get("transfer_passengers", 0)
        if unit == "per_passenger":
            passengers = cleaned_inputs.get("departing_passengers", 0) + cleaned_inputs.get("transfer_passengers", 0)
            return rate * passengers
        if unit == "per_15_minutes":
            duration_hours = cleaned_inputs.get("parking_duration_hours", 0)
            return rate * (duration_hours * 60 / 15)
        if unit == "per_day":
            duration_hours = cleaned_inputs.get("parking_duration_hours", 0)
            return rate * math.ceil(duration_hours / 24) # Charge per full or partial day
        if unit == "per_movement":
            return rate
        if unit == "per_tonne":
            return rate * cleaned_inputs.get("mtow", 0)
        if unit == "per_kg_nox":
            return rate * cleaned_inputs.get("nox_kg", 0)
        
        return 0.0

    def _calculate_tiered_cost(self, rule: Rule, inputs: Dict[str, Any]) -> float:
        """Calculates cost for a tiered/banded rule."""
        if rule.unit != "per_tonne_mtow":
            return 0.0

        mtow = self._clean_value(inputs.get("mtow", 0))
        if not isinstance(mtow, (int, float)) or mtow <= 0:
            return 0.0

        total_cost = 0.0
        sorted_tiers = sorted([t for t in rule.tiers if t.up_to is not None], key=lambda t: t.up_to)
        final_tier = next((t for t in rule.tiers if t.above is not None), None)
        if final_tier:
            sorted_tiers.append(final_tier)

        last_band_limit = 0
        for tier in sorted_tiers:
            if mtow <= last_band_limit:
                break
            
            weight_in_band = 0
            if tier.up_to is not None:
                current_band_limit = tier.up_to
                weight_in_band = min(mtow, current_band_limit) - last_band_limit
                last_band_limit = current_band_limit
            elif tier.above is not None:
                weight_in_band = mtow - last_band_limit
            
            if weight_in_band > 0:
                total_cost += weight_in_band * tier.rate
        
        return total_cost

    def calculate_total_cost(self, inputs: Dict[str, Any]) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculates the total cost by evaluating all charges and rules."""
        total_cost = 0.0
        summary = []

        for charge in self.charges_data.charges:
            for rule in charge.rules:
                if self._check_all_conditions(rule, inputs):
                    cost = 0.0
                    if rule.calculation_method == "standard":
                        cost = self._calculate_standard_cost(rule, inputs)
                    elif rule.calculation_method == "tiered":
                        cost = self._calculate_tiered_cost(rule, inputs)

                    if cost > 0:
                        total_cost += cost
                        summary.append({
                            "charge_name": charge.charge_name,
                            "cost": cost,
                            "rule_description": f"Method: {rule.calculation_method}, Unit: {rule.unit}",
                            "conditions": [f'{c.field} {c.operator} {c.value}' for c in rule.conditions] if rule.conditions else ["Unconditional"],
                        })
        return total_cost, summary

def get_input_specifications(charges_data: Charges) -> Dict[str, Dict[str, Any]]:
    """Scans rules to build a specification for each required input field."""
    specs: Dict[str, Dict[str, Any]] = {}

    def add_spec(field: str, type: str):
        if field not in specs:
            specs[field] = {"options": set(), "ranges": set(), "type": type}

    numeric_fields = ["mtow", "nox_kg", "parking_duration_hours", "departing_passengers", "transfer_passengers", "passenger_seating_capacity", "passenger_age_years"]
    unit_to_input_map = {
        "per_departing_passenger": "departing_passengers",
        "per_transfer_passenger": "transfer_passengers",
        "per_passenger": ["departing_passengers", "transfer_passengers"],
        "per_15_minutes": "parking_duration_hours",
        "per_day": "parking_duration_hours",
        "per_tonne_mtow": "mtow",
        "per_tonne": "mtow",
        "per_kg_nox": "nox_kg",
    }

    for charge in charges_data.charges:
        for rule in charge.rules:
            # Determine required fields from units
            if rule.unit in unit_to_input_map:
                fields = unit_to_input_map[rule.unit]
                field_list = fields if isinstance(fields, list) else [fields]
                for f in field_list:
                    add_spec(f, "numeric")
            
            # Determine required fields and guidance from conditions
            if rule.conditions:
                for c in rule.conditions:
                    is_numeric = c.field in numeric_fields
                    add_spec(c.field, "numeric" if is_numeric else "string")

                    if c.operator == "equals":
                        specs[c.field]["options"].add(f"'{c.value}'")
                    elif c.operator == "in":
                        options = [f"'{v.strip()}'" for v in c.value.split(',')]
                        specs[c.field]["options"].update(options)
                    elif is_numeric:
                        specs[c.field]["ranges"].add(f"{c.operator} {c.value}")
    return specs

def main():
    """Main function to run the command-line interface."""
    parser = argparse.ArgumentParser(description="Calculate aviation charges from a JSON rules file.")
    parser.add_argument("json_file", type=Path, help="Path to the input JSON file (e.g., output.json)")
    args = parser.parse_args()

    if not args.json_file.exists():
        print(f"Error: File not found at {args.json_file}")
        return

    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    try:
        charges_data = Charges(**data)
    except Exception as e:
        print(f"Error parsing JSON file: {e}")
        return

    input_specs = get_input_specifications(charges_data)

    print("\nPlease provide the following flight details:")
    inputs = {}
    for field, spec in sorted(input_specs.items()):
        prompt = f"- {field}"
        guidance = []
        if spec['type'] == 'numeric':
            guidance.append("numeric")
        if spec['options']:
            guidance.append(f"e.g., {', '.join(sorted(list(spec['options'])))}")
        if spec['ranges']:
            guidance.append(f"conditions like: {', '.join(sorted(list(spec['ranges'])))}")
        
        if guidance:
            prompt += f" ({'; '.join(guidance)})"
        
        value = input(f"{prompt}: ")
        inputs[field] = value

    calculator = ChargeCalculator(charges_data)
    total_cost, summary = calculator.calculate_total_cost(inputs)

    print("\n--- Calculation Results ---")
    if not summary:
        print("No charges were applicable based on the provided inputs.")
    else:
        for item in summary:
            print(f"\nCharge: {item['charge_name']}")
            print(f"  - Cost: £{item['cost']:.2f}")
            print(f"  - Rule: {item['rule_description']}")
            print(f"  - Conditions Met:")
            for cond in item['conditions']:
                print(f"    - {cond}")

    print("\n-------------------------")
    print(f"TOTAL COST: £{total_cost:.2f}")
    print("-------------------------")

if __name__ == "__main__":
    main()
