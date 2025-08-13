# %%
#imports
import pandas as pd
from bs4 import BeautifulSoup
import random
import re

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
#%%
html2 = '''<html>
<table><thead><tr><th>Charging Basis (€)</th><th>Detail</th><th>Per 15 minutes or part thereof except for Long Term Remote which is per day or part thereof</th></tr></thead><tbody><tr><td rowspan="2">Standard Charge per Aircraft/Stand type</td><td>Wide Remote</td><td>9.60</td></tr><tr><td>Narrow Remote</td><td>7.70</td></tr><tr><td></td><td>Long Term Remote*</td><td>180.00</td></tr><tr><td></td><td>Light Aircraft Parking (LAP)</td><td>2.65</td></tr><tr><th colspan="3"><strong>Aircraft parking for extended periods in WAP attract the following surcharges:</strong></th></tr><tr><th>Aircraft Parking Duration</th><th></th><th>Parking Surcharge</th></tr><tr><th>Charging Basis</th><th></th><th>Per 15 minutes or part thereof</th></tr><tr><td>48 hours up to 72 hours (including night-time)</td><td></td><td>Standard rate</td></tr><tr><td>72 hours and over (including night-time)</td><td></td><td>Standard rate +200%</td></tr></tbody></table>
</html>'''
soup2 = BeautifulSoup(html2, "html.parser")
dataset2=[]
for i in range(dataset_size):
    for td in soup2.find_all("td"):
        text = td.get_text(strip=True).replace('€', '').replace(',', '.')
        try:
            # If the text is a number (e.g., "180.00"), replace it
            float(text)
            new_value = round(random.uniform(0, 25), 2)
            td.string = f"€{new_value:.2f}"
        except ValueError:
        # If it's not a number (e.g., "Long Term Remote*"), don't touch it
            continue
    updated_html = str(soup2)
    dataset2.append(updated_html)


#%%
# %%
def compute_parking_charge(parking_datatable, stand_type, total_duration_minutes):

    def normalize(s):
        return str(s).strip().lower().replace('*', '').replace('(', '').replace(')', '')

    target = normalize(stand_type)
    df = parking_datatable.copy()
    df['Detail'] = df['Detail'].astype(str).apply(normalize)

    match_row = df[df['Detail'] == target]
    if match_row.empty:
        raise ValueError(f"No matching stand type found: {stand_type}")

    rate_col_candidates = [
        col for col in df.columns if "per 15" in col.lower() or "per day" in col.lower()
    ]
    if not rate_col_candidates:
        raise ValueError("No valid rate column found.")

    rate_col = rate_col_candidates[0]
    rate_str = str(match_row.iloc[0][rate_col])

    try:
        rate = float(rate_str.replace("€", "").replace(",", "."))
    except ValueError:
        raise ValueError(f"Could not parse rate value: {rate_str}")

    if 'long term remote' in target:
        return rate * ((total_duration_minutes + 1439) // 1440)  # per day
    else:
        return rate * ((total_duration_minutes + 14) // 15)  # per 15 mins

#%%
    
for html2 in dataset2:    
    i+=1
    df2 = pd.read_html(html2)
    check1=compute_parking_charge(df2[0],'long term remote*',100)
    check2=compute_parking_charge(df2[0],'Light Aircraft Parking (LAP)',1000)
    dataframe_dataset.loc[i] = [html2,check1,check2]

print(dataframe_dataset.loc[6])
print(dataframe_dataset.loc[7])
dataframe_dataset.to_csv('Syntheticdata/sampledata.csv')


# %%
