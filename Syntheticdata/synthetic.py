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
dataset=[]
# Find all value cells (inside <span>)
dataset_size=5
for i in range(dataset_size):
    for span in soup.find_all("span"):
        # Replace the € value with a random number
        new_value = round(random.uniform(0, 25), 2)
        span.string = f"€{new_value:.2f}"
    updated_html = str(soup)
    dataset.append(updated_html)


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
i=0
dataframe_dataset= pd.DataFrame(columns=['html','check1','check2'])
for html in dataset:    
    df = pd.read_html(html)
    check1=extract_formula(df[0], 1, 10, 'day')
    check2=extract_formula(df[0], 2, 5, 'night') 
    #obvious issue with this method = doesnt cover edge cases unless specifically designed to
    dataframe_dataset.loc[i] = [html,check1,check2]
    i+=1

print(dataframe_dataset)
print('\n full def \n sample element')
dataframe_dataset.to_csv('Syntheticdata/sampledata.csv')

