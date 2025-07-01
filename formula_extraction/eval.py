# %%
import pandas as pd
import numpy as np
test_values = [(1, 10, 'day'), (2, 5, 'night'), (8, 20, 'day'), (0, 15, 'night'), (0.25, 25, 'day'), (4, 30, 'night')]
df=pd.read_csv('llmdata.csv')

evals = []

for i in range(df.shape[0]):
    doc = df['html'][i]  # the HTML document
    formula = df['formula'][i]  # the true formula
    func = df['function'][i]  # the function string from the LLM
    outputs_formula = []
    outputs_llm_func = []
    for qc, weight, day_night in test_values:
        exec(formula)  # Ensure the function is defined in the current scope
        formula_name = formula.split('(')[0].split()[-1]  # Extract the function name from the string
        exec(func)  # execute the function string to define the function in the current scope
        # Extract the function name from the string (assuming it's named `calculate_total_fee`)
        function_name = func.split('(')[0].split()[-1]

        # Call the function dynamically using `globals()`
        output_formula = globals()[formula_name](doc, qc, weight, day_night)
        outputs_formula.append(output_formula)
        output_llm = globals()[function_name](qc, weight, day_night)
        outputs_llm_func.append(output_llm)

    evals.append({
        'doc_id': i,
        'formula': formula,
        'func': func,
        'outputs_formula': outputs_formula,
        'outputs_llm_func': outputs_llm_func,
        'outputs_close' : np.allclose(outputs_formula, outputs_llm_func, rtol=1e-04, atol=1e-04),
    })

df_evals = pd.DataFrame(evals)
# %%
df_evals.to_csv('evals.csv', index=False)
# %%
