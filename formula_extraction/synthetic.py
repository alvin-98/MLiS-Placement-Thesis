# %%
#imports
import pandas as pd
from bs4 import BeautifulSoup
import random
import inspect
import numpy as np
# %% Load the document to augment
document_content = """
<html>
<table><thead><tr><th colspan="3"><strong>Noise Charges</strong></th></tr><tr><th>QC</th><th>Set fee per Tonne 2025<br>Day</th><th>Set fee per Tonne 2025<br>Night</th></tr></thead><tbody><tr><td>0</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.125</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.25</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.5</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€2.00</span></td></tr><tr><td>1</td><td><span style="color: green;">€1.00</span></td><td><span style="color: green;">€4.00</span></td></tr><tr><td>2</td><td><span style="color: green;">€2.00</span></td><td><span style="color: green;">€8.00</span></td></tr><tr><td>4</td><td><span style="color: green;">€4.00</span></td><td><span style="color: green;">€12.00</span></td></tr><tr><td>8</td><td><span style="color: green;">€6.00</span></td><td><span style="color: green;">€16.00</span></td></tr><tr><td>16</td><td><span style="color: green;">€8.00</span></td><td><span style="color: green;">€20.00</span></td></tr></tbody></table>
</html>
"""
# %% Load the formula

def compute_noise_charge(document, qc, weight, day_night):
    df = pd.read_html(document)[0]
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

# let's imagine we are given a string representation of the function
compute_noise_charge = inspect.getsource(compute_noise_charge)
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

def replace_text(document, formula):
    """Change the word 'Noise Charges' to 'Noise Fees' in the HTML content and the formula.
    Since the formula uses the document as input, it needs to be changed.
    """
    soup = BeautifulSoup(document, "html.parser")
    for strong in soup.find_all("strong"):
        if strong.string == "Noise Charges":
            strong.string = "Noise Fees"
    new_doc = str(soup)
    new_formula = formula.replace("Noise Charges", "Noise Fees")
    return new_doc, new_formula

# %% Generate synthetic data

num_repeats = 2 # number of times to repeat the synthetic data generation

source_content = [document_content]
source_formulas = [compute_noise_charge]

synthetic_content = []
synthetic_formulas = []

for i in range(num_repeats):

    _content, _formula = add_random_values(document_content, compute_noise_charge)
    synthetic_content += [_content]
    synthetic_formulas += [_formula]  # no change to the formula here

    _content, _formula = replace_text(document_content, compute_noise_charge)
    synthetic_content += [_content]
    synthetic_formulas += [_formula]  # here need to transform the formula
# %% Compute some example charges
# these should all be correctby construction but need some way to verify
test_values = [(1, 10, 'day'), (2, 5, 'night')]
charges = []
for html, formula in zip(synthetic_content, synthetic_formulas):
    _checks = []
    for qc, weight, day_night in test_values:
        exec(formula)  # Ensure the function is defined in the current scope
        charge = compute_noise_charge(html, qc, weight, day_night)
        _checks.append(np.around(charge, 4).item())  # Round. Need to check this doesn't raise issues later...
    charges.append(_checks)

# %% Put into a DataFrame and export

i=0
dataframe_dataset= pd.DataFrame(columns=['html', 'formula', 'check1','check2'])
for html in synthetic_content:
    dataframe_dataset.loc[i] = [html, synthetic_formulas[i], charges[i][0], charges[i][1]]
    i+=1

print(dataframe_dataset)
print('\n full def \n sample element')
dataframe_dataset.to_csv('./sampledata.csv', index=False)


# %% Reload to check
dataframe_dataset = pd.read_csv('./sampledata.csv')

# %%
dataframe_dataset.head()
# %%
