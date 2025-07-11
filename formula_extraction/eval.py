# %%
import pandas as pd
import numpy as np
from input_data import input_values
import ast
df=pd.read_csv('llmdata.csv')
for i in range(df.shape[0]):
    print(df)
test_values = []
#should be checking that input values consistent with expected
for inputs in zip(*input_values):
    merged = {}
    for d in inputs:
        merged.update(d)
    test_tuple = (
        merged.get('qc'),
        merged.get('weight'),
        merged.get('time_min'),
        merged.get('day_night'),
        merged.get('park_type')
    )
    test_values.append(test_tuple)
    
# %%   
outputs_formula=[] 
for i in range(df.shape[0]):
    data = ast.literal_eval(df['checks'][i])
    for idx,d in enumerate(data):
        d=sum(d)
        data[idx]=d
    outputs_formula.append(data)

# %%
evals = []
outputs_llm_func = []
for i in range(df.shape[0]):
    func = df['function'][i]  # the function string from the LLM
    intermediate_llm_outputs=[]
    outputs_llm_func = []
    for qc, weight, time_min, day_night, park_type in test_values:
        exec(func)  # execute the function string to define the function in the current scope
        # Extract the function name from the string (assuming it's named `calculate_total_fee`)
        function_name = func.split('(')[0].split()[-1]
        output_llm = globals()[function_name](qc, weight, time_min, day_night, park_type )
        intermediate_llm_outputs.append(output_llm)
    outputs_llm_func.append(intermediate_llm_outputs)
    evals.append({
        'doc_id': i,
        'func': func,
        'outputs_formula': outputs_formula[i],
        'outputs_llm_func': outputs_llm_func,
        'outputs_close' : np.allclose(outputs_formula, outputs_llm_func, rtol=1e-04, atol=1e-04),
    })

df_evals = pd.DataFrame(evals)
print(df_evals.head())
# %%
df_evals.to_csv('evals.csv', index=False)
# %%
