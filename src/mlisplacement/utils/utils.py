import time
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from enum import Enum

def read_md(file_path):
    """
    Reads a Markdown file and returns its content as a string.
    
    Args:
        file_path (str): The path to the Markdown file.
        
    Returns:
        str: The content of the Markdown file.
    """
    with open(file_path, 'r') as file:
        return file.read()
    
def load_model(model_id, low_cpu_mem_usage=False):
    """
    Loads a pre-trained model and tokenizer from Hugging Face.
    
    Args:
        model_id (str): The identifier of the model to load.
        
    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=low_cpu_mem_usage)
    model.eval()
    return model, tokenizer

def timing_decorator(func):
    """
    A decorator that prints the execution time of the function it wraps.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start the clock
        result = func(*args, **kwargs)    # Call the original function
        end_time = time.perf_counter()    # Stop the clock
        
        execution_time = end_time - start_time
        print(f"Execution time for '{func.__name__}': {execution_time:.4f} seconds")
        
        return result
    return wrapper

#TODO: abstract the global CHARGE_CATEGORY_VARIABLES
def create_dynamic_variable_enum(charge_category: str, charge_category_variables: dict) -> Enum:
    """
    Creates a new Enum class containing only the variables relevant
    to the specified charge category.

    charge_category_variables provides a mapping from charge categories to the variable names they can use.
    """
    variable_names = charge_category_variables.get(charge_category)
    if not variable_names:
        raise ValueError(f"Unknown charge category: {charge_category}")
    
    # The dictionary for the Enum must have {MEMBER_NAME: value}
    # We'll use uppercase for the member name for convention.
    enum_dict = {name.upper(): name for name in variable_names}
    
    # Create the Enum class dynamically
    return Enum("Var", enum_dict)