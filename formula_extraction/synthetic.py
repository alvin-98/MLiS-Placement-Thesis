# %%
#imports
import pandas as pd
from bs4 import BeautifulSoup
import random
import inspect
import numpy as np
import math
from input_data import input_values #saved locally

# %% Load the document to augment
noise_html = """
<html>
<table><thead><tr><th colspan="3"><strong>Noise Charges</strong></th></tr><tr><th>QC</th><th>Set fee per Tonne 2025<br>Day</th><th>Set fee per Tonne 2025<br>Night</th></tr></thead><tbody><tr><td>0</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.125</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.25</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.5</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€2.00</span></td></tr><tr><td>1</td><td><span style="color: green;">€1.00</span></td><td><span style="color: green;">€4.00</span></td></tr><tr><td>2</td><td><span style="color: green;">€2.00</span></td><td><span style="color: green;">€8.00</span></td></tr><tr><td>4</td><td><span style="color: green;">€4.00</span></td><td><span style="color: green;">€12.00</span></td></tr><tr><td>8</td><td><span style="color: green;">€6.00</span></td><td><span style="color: green;">€16.00</span></td></tr><tr><td>16</td><td><span style="color: green;">€8.00</span></td><td><span style="color: green;">€20.00</span></td></tr></tbody></table>
</html>
"""
parking_html = """
<html>
<table><thead><tr><th>Charging Basis (€)</th><th>Detail</th><th>Per 15 minutes or part thereof except for Long Term Remote which is per day or part thereof</th></tr></thead><tbody><tr><td rowspan="2">Standard Charge per Aircraft/Stand type</td><td>Wide Remote</td><td>9.60</td></tr><tr><td>Narrow Remote</td><td>7.70</td></tr><tr><td></td><td>Long Term Remote*</td><td>180.00</td></tr><tr><td></td><td>Light Aircraft Parking (LAP)</td><td>2.65</td></tr><tr><th colspan="3"><strong>Aircraft parking for extended periods in WAP attract the following surcharges:</strong></th></tr><tr><th>Aircraft Parking Duration</th><th></th><th>Parking Surcharge</th></tr><tr><th>Charging Basis</th><th></th><th>Per 15 minutes or part thereof</th></tr><tr><td>48 hours up to 72 hours (including night-time)</td><td></td><td>Standard rate</td></tr><tr><td>72 hours and over (including night-time)</td><td></td><td>Standard rate +200%</td></tr></tbody></table>
</html>
"""
# %% Load the formula

def compute_noise_charge(html, qc, weight, day_night):
    '''This function calculates the total noise charge based on the QC value, weight in tonnes, and whether it's day or night.
     df[df.iloc[:,0] == qc] is used to find the row corresponding to the QC value.
     row.iloc[:,1].values[0] is used to select the fee for daytime.
     row.iloc[:,2].values[0] is used to select the fee for nighttime.
     '''
    df = pd.read_html(html)[0]
    row = df[df.iloc[:,0] == qc]
    if row.empty:
        return None
    
    if day_night.lower() == 'day':
        fee = row.iloc[:,1].values[0]
    elif day_night.lower() == 'night':
        fee = row.iloc[:,2].values[0]
    else:
        raise ValueError("day_night must be 'day' or 'night'")
    
    total_fee = float(fee.replace('€', '').replace(',', '.')) * weight
    return total_fee

# let's imagine we are given a string representation of the function
compute_noise_charge = inspect.getsource(compute_noise_charge)
# %% a function to calculate charnges for our second html
def compute_park_charge_west(html, park_type, time_min):
    """
    Calculate the parking charge for an aircraft In west parking location
    (WAP) based on stand type, and parking duration (in minutes).
    Applies standard charges and WAP surcharges for stays over 72 hours.

    Parameters:
    - park_type: str, one of the defined stand types
    - time_min: float or int, total parking time in minutes
    - html: a version of the parking html

    Returns:
    - Total parking charge as float
    """
    
    df = pd.read_html(html)[0]
    if park_type == 'Wide Remote':
        i = 0
    elif park_type == 'Narrow Remote':
        i = 1
    elif park_type == 'Long Term Remote*':
        i = 2
    elif park_type == 'Light Aircraft Parking (LAP)':
        i = 3
    else:
        raise ValueError(f"Unknown park_type: {park_type}")

    rate = float(df.iloc[:, 2].values[i])

    if park_type == 'Long Term Remote*':
        days = math.ceil(time_min / (24 * 60))
        charge = rate * days
    else:
        intervals_15 = math.ceil(time_min / 15)
        charge = rate * intervals_15

        if time_min >= (72 * 60):
            normal_time = 72 * 60
            surcharge_time = time_min - normal_time
            normal_intervals = math.ceil(normal_time / 15)
            surcharge_intervals = math.ceil(surcharge_time / 15)
            charge = (rate * normal_intervals) + (rate * surcharge_intervals * 3)

    return charge

compute_park_charge_west = inspect.getsource(compute_park_charge_west)
# %% Some functions to augment the document

def add_random_values(document, formula):
    """Augmentation function that replaces the € values in the HTML with random numbers.
    Since the formula uses the document as input, it does not need to be changed.
    """
    soup = BeautifulSoup(document, "html.parser")
    for span in soup.find_all("span"):
        # Replace the € value with a random number
        new_value = round(random.uniform(0, 25), 2)
        span.string = f"€{new_value:.2f}"
    return str(soup), formula

def replace_text(document, formula, old_word, new_word):
    """Change a word (or phrase) in the document into a new one. requires inputting the
    old and new words into the function as strings
    """
    soup = BeautifulSoup(document, "html.parser")
    for strong in soup.find_all("strong"):
        if strong.string == old_word:
            strong.string = new_word
    new_doc = str(soup)
    new_formula = formula.replace(old_word, new_word) #redundant as formula now iloc based however keeping in as doesnt throw error maybe useful later
    return new_doc, new_formula

def augment_document(document,formula,n_augmentations,old_word=None,new_word=None):
    '''augments numerical values of a document, by providing str old word and new word can also
    modify words. In future can optionally have old word and new word take lists as input'''
    synthetic_content=[]
    synthetic_formulas=[]
    for i in range(n_augmentations):
        _content,_formula = add_random_values(document,formula)
        if old_word is not None and new_word is not None:
            _content,_formula = replace_text(_content,_formula,old_word,new_word)
        synthetic_content.append(_content)
        synthetic_formulas.append(_formula)
    return synthetic_content,synthetic_formulas
        
synthetic_main=[]
repeats_per_doc=2
synthetic_main.append(augment_document(noise_html,compute_noise_charge,repeats_per_doc,"Noise Charges","Noise Fees"))
synthetic_main.append(augment_document(parking_html,compute_park_charge_west,repeats_per_doc))


for i in range(len(synthetic_main)):
    content,formulas = synthetic_main[i]
    synthetic_main[i] = (content, formulas, input_values[i])
# %% run the eval code
charges = []

for content_list, formula_list, input_dicts in synthetic_main:
    for html, code in zip(content_list, formula_list):
        # Prepare isolated namespace for exec
        namespace = {}
        namespace['html'] = html
        exec(code, {'pd': pd, 'math': math, 'np': np}, namespace)
        # Grab the function (assumes only one function is defined)
        fn = [f for f in namespace.values() if callable(f)][0]

        for inputs in input_dicts:
            # Filter out any keys with None values
            filtered_vals = {k: v for k, v in inputs.items() if v is not None}
            print(fn)
            print(filtered_vals)
            result = fn(html, **filtered_vals)
            print(f"Result for inputs {filtered_vals}: {result}")
            charges.append(result)
# %%   collecting results together
from collections import defaultdict

#grouping htmls together
grouped_htmls = defaultdict(list)

for content_list, formula_list, input_dicts in synthetic_main:
    for idx, html in enumerate(content_list):
        grouped_htmls[idx].append(html)

joined_htmls = [ "".join(grouped_htmls[i]) for i in sorted(grouped_htmls) ]

#grouping charges 
n_htmls=2
n=repeats_per_doc*n_htmls
m=len(input_values[0])
intermediate=[]
sorted_charges=[]
for i in range(n):
    intermediate.append([charges[i],charges[i+n]])
for i in range(m):
    sorted_charges.append([intermediate[i],intermediate[i+m]])
       

                      

# %% Put into a DataFrame and export

i=0
dataframe_dataset= pd.DataFrame(columns=['html', 'checks'])
for html in joined_htmls:
    dataframe_dataset.loc[i] = [html, sorted_charges[i]]
    i+=1

print(dataframe_dataset)
print('\n full def \n sample element')
dataframe_dataset.to_csv('./sampledata.csv', index=False)


# %% Reload to check
dataframe_dataset = pd.read_csv('./sampledata.csv')

# %%
dataframe_dataset.head()
# %%
