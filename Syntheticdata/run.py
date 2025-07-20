#%% #imports
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

sys_prompt = '''You are a helpful Python programming assistant. You are given an HTML document that contains a pricing table. Your job is to write clean, readable Python code that defines a function to compute a total fee based on inputs like 'QC', weight in tonnes, and whether it's 'day' or 'night'.

The HTML may contain <th colspan>, <br> tags, currency symbols, and style attributes. You should mentally interpret the table and output a pure Python function that hardcodes the rates in a dictionary. Do not parse the HTML in code. Just read it and write a clean function.'''

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

user_prompt = '''
Below is an HTML table containing noise charge data.

Write a function `calculate_total_fee(qc_value: float, tonnage: float, time_period: str)` that:
- Matches the QC Set value exactly (do not interpolate),
- Multiplies the corresponding per-tonne fee by the tonnage,
- Uses the correct rate for 'day' or 'night',
- Returns the total fee.

Only output the function. Do not parse the HTML at runtime.
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
data.to_csv('Syntheticdata/llmdata.csv')
 
