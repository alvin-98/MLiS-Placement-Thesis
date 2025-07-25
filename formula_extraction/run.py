# %%
#imports
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import inspect
model_name = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# %%
def extract_python_function(text):
    """
    Extracts a Python function from a given string.
    """
    match = re.search(r'```python\s+(def [\s\S]+?)```', text)
    return match.group(1) if match else None
# %% some extra ideas for post processing the output source function
import ast #can also use this for augmentation of the formulas

def extract_function_arguments(function_str):
    """
    Extracts the arguments of a Python function from its string representation.
    """
    tree = ast.parse(function_str)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = []
            for arg in node.args.args:
                arg_name = arg.arg
                arg_type = ast.unparse(arg.annotation) if arg.annotation else None
                args.append((arg_name, arg_type))
            return args
    return None
# %%
data=pd.read_csv('formula_extraction/sampledata.csv')
# %%
# function to complete - needs manually updating for now
def calculate_total_fee(qc_value: float, tonnage: float, time_min: float, time_period: str = 'day', park_type: str = 'Wide Remote') -> float:
  """This function calculates the total fee based on the QC value, tonnage, and time period of noise as well as the time spent parked and the park type. 
  
  Args:
      qc_value (float): The QC value.
      tonnage (float): The weight in tonnes.
      time_min (float): Time spent parked
      time_period (str): The time period, either 'day' or 'night'.
      park_type (str): Type of parking undertaken

  Returns:
      float: The total fee calculated based on the provided inputs.
  """

# convert to a string representation for the LLM
function_to_complete = inspect.getsource(calculate_total_fee)
# %%
sys_prompt = '''You are a helpful Python programming assistant. You are given multiple HTML tables combined into a single string. Each table represents a separate pricing scheme for an airport charge (e.g., noise, parking). Your task is to mentally interpret all tables and define a single Python function that computes the total cost by combining the relevant charges.

Each table may contain <th colspan>, <br> tags, currency symbols, and style attributes. You should **not parse** the HTML at runtime. Instead, read the tables and hardcode the extracted rates into the function as dictionaries.'''

few_shot = '''Below is an HTML table containing fee rates based on QC Set values. Write a Python function that:

- Matches the exact QC Set value (no interpolation),
- Takes as input: QC Set, tonnage, and time period ('day' or 'night'),
- Returns the total fee using the appropriate per-tonne rate.

---

Example:

HTML Table:
<table>
  <thead>
    <tr><th>QC Set</th><th>Day Rate</th><th>Night Rate</th></tr>
  </thead>
  <tbody>
    <tr><td>0.0</td><td>€5.00</td><td>€4.00</td></tr>
    <tr><td>1.0</td><td>€10.00</td><td>€8.00</td></tr>
    <tr><td>2.0</td><td>€15.00</td><td>€12.00</td></tr>
  </tbody>
</table>

Python Function:
```python
def calculate_total_fee(qc_value: float, tonnage: float, time_period: str = 'day') -> float:
    fees = {
        0.0: {'day': 5.00, 'night': 4.00},
        1.0: {'day': 10.00, 'night': 8.00},
        2.0: {'day': 15.00, 'night': 12.00},
    }

    if qc_value not in fees:
        raise ValueError(f"QC value {qc_value} not found.")
    if time_period not in ['day', 'night']:
        raise ValueError("time_period must be 'day' or 'night'")

    return fees[qc_value][time_period] * tonnage'''

user_prompt = f'''
Below is an HTML string containing multiple pricing tables for different types of airport charges.

Complete the following function:

{function_to_complete}

The function should:
- Extract all fee rates by interpreting the tables mentally and hardcoding them into the function,
- Take all relevant inputs (e.g., QC, tonnage, time, parking duration/type),
- Compute individual fees for each applicable charge (e.g., noise, parking),
- Return the **total combined** fee,
- Do not perform any HTML parsing at runtime.

Only output the function. Do not include any comments or explanations.
'''


def make_prompt(html, sys_prompt=sys_prompt, few_shot=few_shot, user_prompt=user_prompt):
    full_prompt = f'''
    <|system|>
    {sys_prompt}
    {few_shot}
    <|user|>
    {user_prompt}
    {html}
    <|assistant|>'''
    return full_prompt

# %%
completions = []
llm_funcs=[]
func_vars = []
for i in range(data.shape[0]):
    print(f"Processing row: {i+1}/{data.shape[0]}")
    html=data['html'][i] 
    full_prompt = make_prompt(html)
    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1000)
    completion = tokenizer.batch_decode(outputs[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
    _func_str = extract_python_function(completion[0])
    _func_vars = extract_function_arguments(_func_str)
    llm_funcs.append(_func_str) #some post-processing to extract the function code
    func_vars.append(_func_vars) #some post-processing to extract the function arguments
    completions.append(completion[0])
    
# %%
print('loop finished')
data = data.copy()
#truncate to length of completions for testing
data = data.iloc[:len(completions)].reset_index(drop=True)

data['function']=llm_funcs
data['completion']=completions
data['function_vars'] = str(func_vars)
data.to_csv('formula_extraction/llmdata.csv', index=False)
# %%