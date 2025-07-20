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
# In this alternative formulation, the formula does not take the document itself as an argument.
# I've made up these numbers, but they will need to be consistent with the document...

def compute_noise_charge(qc_value: float, tonnage: float, time_period: str = 'day') -> float:
    """This function calculates the total noise charge based on the QC value, tonnage, and time period.

    Args:
        qc_value (float): The QC value.
        tonnage (float): The weight in tonnes.
        time_period (str): The time period, either 'day' or 'night'.

    Returns:
        float: The total fee calculated based on the provided inputs.
    """

    fees = {
        0: {'day': 16.20, 'night': 17.72},
        0.125: {'day': 21.54, 'night': 5.57},
        0.25: {'day': 10.97, 'night': 8.09},
        0.5: {'day': 11.51, 'night': 11.12},
        1: {'day': 22.46, 'night': 1.35},
        2: {'day': 0.50, 'night': 0.14},
        4: {'day': 0.17, 'night': 19.43},
        8: {'day': 24.89, 'night': 6.11},
        16: {'day': 11.89, 'night': 4.83},
    }

    if qc_value not in fees:
        raise ValueError(f"QC value {qc_value} not found.")
    if time_period not in ['day', 'night']:
        raise ValueError("time_period must be 'day' or 'night'")

    return fees[qc_value][time_period] * tonnage

# let's imagine we are given a string representation of the function
compute_noise_charge = inspect.getsource(compute_noise_charge)
# %% Augmentation Functions
# For this case, we should use the formulation where an augmentation function takes both the document and the formula as arguments.

def add_random_values(document, formula):
    """Augmentation function that replaces the € values in the HTML with random numbers.
    In this case, the formula is not actually changed. Constrast this with the case in sytnthetic.py where the formula is changed...

    **As I'm not sure how to update the formula correctly, I'm leaving it unchanged as an example here.**

    """
    return document, formula

def replace_text(document, formula):
    """Change the word 'Noise Charges' to 'Noise Fees' in the HTML content.
    In this case, the formula is not actually changed. Constrast this with the case in sytnthetic.py where the formula is changed...
    """
    soup = BeautifulSoup(document, "html.parser")
    for strong in soup.find_all("strong"):
        if strong.string == "Noise Charges":
            strong.string = "Noise Fees"
    return str(soup), formula
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
        charge = compute_noise_charge(qc, weight, day_night)
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
dataframe_dataset.to_csv('./sampledata_alt.csv', index=False)


# %% Reload to check
dataframe_dataset = pd.read_csv('./sampledata_alt.csv')

# %%
dataframe_dataset.head()
# %%
