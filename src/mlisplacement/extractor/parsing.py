"""Parse a JSON graph into a SymPy expression."""

import sympy
from sympy import Symbol, Piecewise, sympify, Add, Mul, Pow

def compose_expression(node: dict):
    """
    Recursively parses a JSON graph into a SymPy expression,
    preventing automatic simplification.

    Args:
        node: A dictionary representing a node in the computation graph.

    Returns:
        A non-evaluated sympy expression representing the computation.
    """

    node_type = node.get('type')

    if node_type == "VALUE":
        return sympify(node['value'])

    elif node_type == "VARIABLE":
        return Symbol(node['name'])

    elif node_type == "BINARY_OPERATION":
        left = compose_expression(node['left'])
        right = compose_expression(node['right'])
        operator = node['operator']
        
        # Use class constructors with evaluate=False to prevent simplification
        if operator == "ADD":
            return Add(left, right, evaluate=False)
        elif operator == "MULTIPLY":
            return Mul(left, right, evaluate=False)
        elif operator == "DIVIDE":
            # Division (a/b) is represented as a * (b**-1)
            power = Pow(right, -1, evaluate=False)
            return Mul(left, power, evaluate=False)
        else:
            raise ValueError(f"Unsupported binary operator: {operator}")

    elif node_type == "COMPARISON":
        left = compose_expression(node['left'])
        right = compose_expression(node['right'])
        operator = node['operator']

        if operator == "GREATER_THAN":
            return left > right
        if operator == "LESS_THAN":
            return left < right
        else:
            raise ValueError(f"Unsupported comparison operator: {operator}")

    elif node_type == "CONDITIONAL":
        condition = compose_expression(node['condition'])
        if_true_expr = compose_expression(node['if_true'])
        if_false_expr = compose_expression(node['if_false'])
        
        return Piecewise((if_true_expr, condition), (if_false_expr, True))

    else:
        raise ValueError(f"Unknown node type: {node_type}")