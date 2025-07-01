# %%
#imports
import pandas as pd
from bs4 import BeautifulSoup
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
model_name = "codellama/CodeLlama-7b-Python-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
def extract_function_code(text):
    'because its difficult to put an exact token limit'
    match = re.search(r'def [\s\S]+?(?=<\|[^|]+?\|>|$)', text)
    return match.group(0) if match else None

data=pd.read_csv('Syntheticdata/sampledata.csv')

sys_prompt = '''sys_prompt = 
You are a helpful Python programming assistant. You are given an HTML document containing one or more pricing tables. Your job is to write clean, readable Python code that defines a function to compute a cost or fee based on user input.

Important instructions:
- Do not parse or manipulate the HTML in the function. Just read and interpret it yourself.
- Extract and hardcode any relevant rates or rules from the table(s).
- Use clear variable names and simple logic.
- You may use dictionaries, conditionals, or any clean structure to represent the logic — choose the most appropriate form.
- If input validation is needed (e.g. invalid category or time), raise a `ValueError` with a helpful message.
- The function should be self-contained and return the final cost/fee as a float.
'''


few_shot = '''
Example 1:

Task:
You are given an HTML table with fee rates depending on aircraft noise QC values. Your function should:
- Match the QC Set value exactly (no interpolation),
- Take `qc_value`, `tonnage`, and `time_period` ('day' or 'night') as arguments,
- Return the total cost using the corresponding per-tonne rate.

HTML:
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
    rates = {
        0.0: {'day': 5.00, 'night': 4.00},
        1.0: {'day': 10.00, 'night': 8.00},
        2.0: {'day': 15.00, 'night': 12.00},
    }

    if qc_value not in rates:
        raise ValueError(f"QC value {qc_value} not found.")
    if time_period not in ['day', 'night']:
        raise ValueError("time_period must be 'day' or 'night'")

    return rates[qc_value][time_period] * tonnage

Example 2:
    
Task:
Given an HTML table with time-based parking charges, write a function that:
- Takes `hours_parked` as input,
- Applies the correct flat fee or hourly rate based on duration band.

HTML:
<table>
  <thead><tr><th>Duration</th><th>Rate</th></tr></thead>
  <tbody>
    <tr><td>0–1 hours</td><td>€5</td></tr>
    <tr><td>1–3 hours</td><td>€10</td></tr>
    <tr><td>Over 3 hours</td><td>€15 + €2 per additional hour</td></tr>
  </tbody>
</table>

Python Function:
```python
def calculate_total_fee(hours_parked: float) -> float:
    if hours_parked <= 1:
        return 5
    elif hours_parked <= 3:
        return 10
    else:
        extra_hours = hours_parked - 3
        return 15 + 2 * extra_hours
'''

user_prompt = '''
Below is an HTML table describing a cost or pricing scheme.

Write a Python function `calculate_total_fee` that:
- Computes and returns the total cost or fee based on appropriate inputs,
- Uses logic derived from the table(s),
- Encodes all relevant constants/rates directly in the code.

Only output the function. Do not parse the HTML at runtime. Avoid unnecessary complexity.
'''

llm_func=[]
for i in range(data.shape[0]):
    html=data['html'][i] 
    full_prompt = f'''
    <|system|>
    {sys_prompt}
    {few_shot}
    <|user|>
    {user_prompt}
    {html}
    <|assistant|>'''
    inputs = tokenizer(full_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=500)
    formula = tokenizer.batch_decode(outputs[:, inputs['input_ids'].size(1):], skip_special_tokens=True)
    llm_func.append(extract_function_code(formula[0]))
data['function']=llm_func
data.to_csv('Syntheticdata/llmdata_newsample.csv')
 



