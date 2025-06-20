# %%
#imports
import pandas as pd
from bs4 import BeautifulSoup
import random

# %%
html = """
<html>
<table><thead><tr><th colspan="3"><strong>Noise Charges</strong></th></tr><tr><th>QC</th><th>Set fee per Tonne 2025<br>Day</th><th>Set fee per Tonne 2025<br>Night</th></tr></thead><tbody><tr><td>0</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.125</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.25</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.5</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€2.00</span></td></tr><tr><td>1</td><td><span style="color: green;">€1.00</span></td><td><span style="color: green;">€4.00</span></td></tr><tr><td>2</td><td><span style="color: green;">€2.00</span></td><td><span style="color: green;">€8.00</span></td></tr><tr><td>4</td><td><span style="color: green;">€4.00</span></td><td><span style="color: green;">€12.00</span></td></tr><tr><td>8</td><td><span style="color: green;">€6.00</span></td><td><span style="color: green;">€16.00</span></td></tr><tr><td>16</td><td><span style="color: green;">€8.00</span></td><td><span style="color: green;">€20.00</span></td></tr></tbody></table>
</html>
"""


soup = BeautifulSoup(html, "html.parser")

# Find all value cells (inside <span>)
for span in soup.find_all("span"):
    # Replace the € value with a random number
    new_value = round(random.uniform(0, 25), 2)
    span.string = f"€{new_value:.2f}"


updated_html = str(soup)

print(updated_html)

dfs = pd.read_html(updated_html)
df = dfs[0]

print(df)
def extract_formula(df, qc, weight, day_night):
    row = df[df[('Noise Charges', 'QC')] == qc]
    if row.empty:
        return None
    
    if day_night.lower() == 'day':
        fee = row[('Noise Charges', 'Set fee per Tonne 2025 Day')].values[0]
    elif day_night.lower() == 'night':
        fee = row[('Noise Charges', 'Set fee per Tonne 2025 Night')].values[0]
    else:
        raise ValueError("day_night must be 'day' or 'night'")
    
    total_fee = float(fee.replace('€', '').replace(',', '.')) * weight
    return total_fee

    
print(extract_formula(df, 1, 10, 'day'))  
print(extract_formula(df, 2, 5, 'night'))  


# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "codellama/CodeLlama-7b-Python-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

sys_prompt = '''You are a helpful Python programming assistant. You are given an HTML document that contains a pricing table. Your job is to write clean, readable Python code that defines a function to compute a total fee based on inputs like 'QC', weight in tonnes, and whether it's 'day' or 'night'.

The HTML may contain <th colspan> or <br> tags and style attributes. You should only provide the formula. Read the HTML and use your HTML reading abilities to understand the structure and values of the HTML and use them to make a function'''
#add few shot learning? will need to produce examples
few_shot = f'''Example: For the following HTML table **{updated_html}** you would be expected to provide the following output **def extract_formula(df, qc, weight, day_night):
    row = df[df[('Noise Charges', 'QC')] == qc]
    if row.empty:
        return None
    
    if day_night.lower() == 'day':
        fee = row[('Noise Charges', 'Set fee per Tonne 2025 Day')].values[0]
    elif day_night.lower() == 'night':
        fee = row[('Noise Charges', 'Set fee per Tonne 2025 Night')].values[0]
    else:
        raise ValueError("day_night must be 'day' or 'night'")
    
    total_fee = float(fee.replace('€', '').replace(',', '.')) * weight
    return total_fee**'''
user_prompt = '''
# Below is an HTML table containing noise charge data.
# This table is presented to you as a string for easy reading
# Your task is to write a function `extract_formula(html_text, qc, weight, day_night)` that:
# - Extracts the relevant fee per tonne for a given `qc` (float) and `day_night` ("day" or "night")
# - Multiplies the fee by the given `weight` in tonnes
# - Returns the total fee as a float
# Output only the function definition. Do not include explanatory comments or examples.
'''
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
print(formula)
 



