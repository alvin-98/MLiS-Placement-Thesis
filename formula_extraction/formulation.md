### Source Data

The source data we are considering (i.e. the data available to us) are documents and formulas for each document. Let $i= 1, ..., N$ indicate the document in the source data. Each document has some content denoted as $x_i$. This is typically in pdf form, but let us assume it is in text form here, i.e a long string, $x \in \mathcal{S}$, where $\mathcal{S}$ is the set of all strings.

For example, we might have two documents, $i=1$ the "Heathrow Airport Charges 2025" and $i=2$ the "Gatwick Airport Charges 2025". The content, $x_i$, for each document contains information about various charges applicable to that airport.

A formula, $f$, takes in an $x$ and a set of values for some variables, $v$, and outputs the corresponding charge amount $y$,

$$y = f(x, v)$$

In general, this forumula will depend on:
1. The document being considered, i.e. the airport and year, and
2. The charge being computed

Overall, there will be a set of charges that can be considered. Label these as $c = 1, 2, ..., C$ where $C$ is the total number of different charges. (We could have this depend on the document as well, but for simplicity we will take a fixed set of charges.)

We can then label the set of formula as,

$$f_{i,c}(x_i, v)$$

where $i$ denotes the document and $c$ denotes the charge being computed. We can also collect all the forumulas into a single set for notational convenience,
$$
\mathbf{f}_{i}(x_{i}, \mathbf{v}) = \{f_{i,c}(x_i, v_{c})\}_{c=1}^{C}
$$
where $\mathbf{v}$ is the values of all variables, and $v_c$ are the values for the variable corresponding to charge $c$.


For example, if we have a document, "Heathrow Airport Charges 2025", then this will have information concerning different charges for that airport, e.g. Noise Charge, Landing Charge, etc.

The formula for the Noise Charge will be different from the formula for the Landing Charge.

Additionally, the formula for the Noise Charge for this document will in general differ from the formula for the Noise Charge for another document, e.g. "Gatwick Airport Charges 2025".

Our source data can thus be represented as,
$$
\mathcal{D} = \{(x_i, \mathbf{f}_{i})\}_{i=1}^{N} $$

### Document Classes and Augmentation

While the source data provides a different forumula for each document, we can expect the same formula to compute the correct charge for different documents. 

For a given charge $c$, we will define a class of documents, $\chi_{c}(i)$, with representative document $i$, such that all documents in this class have the same formula for the given charge.

As a simple example, two documents that have the exact same structure apart from their numerical values, will likely have the same formula for a given charge. Thus, they would belong to the same class.

We can use this idea to create synthetic data.

Let $\mathcal{A}: x \to x'$ be a function that takes in a document $x$ and outputs a new document $x'$. We will refer to this as augmentation. An augmentation function can be as simple as replacing numerical values with new random values.

Let us then denote $\mathcal{A}_{\chi_{c}}$ as the augmentation function for the class of documents $\chi_{c}$, i.e. the augmentation function that takes in a document from this class and outputs a new document from the same class. 

For example, let $i=1$ be the "Heathrow Airport Charges 2025" document and $c=1$ be the Noise Charge. An example augmentation function $\mathcal{A}_{\chi_{c}}$ could be to replace the numerical values in the document with new random values, while keeping the structure of the document intact. This will create new document content, $x'_1$, that belongs to the same class $\chi_{c}$, i.e. it will have the same formula for the Noise Charge. In fact, in this case we can expect all forumulas for all charges to remain valid.

As such, we now have a new data point,
$$
(x'_1, \mathbf{f}_{1}) = \left(\mathcal{A}_{\chi_{c}}(x_1), \mathbf{f}_{1})\right)
$$

Let us now generalise this idea. Rather than just changing the document content and leaving the formula unchanged, we can also change the formula to reflect the changes made to the document.

To this end, let us define the augmentation functions as a function of both the document content and the formula. That is,
$$
\mathcal{A}_{\chi_{c}}(x_i, f_{i,c}) = (x'_i, f'_{i,c})
$$
where $f'_{i,c}$ is the transformed formula for the augmented document $x'_i$. We then require that,
$$
f'_{i,c}(x_i', v_c) = f_{i,c}(x_i, v_c)
$$

In the simple example above, the augmentation function would not change the formula, i.e.,
$$
\mathcal{A}_{\chi_{c}}(x_i, f_{i,c}) = (x'_i, f_{i,c}) ~.
$$

Generally, a given augementation will only work for a particular class of documents, e.g. for documents with different structures, or for different charges it may not be satsify the requirement that the requirement above.

### Example

Consider the case of a source dataset with a single document, $N=1$ and a single charge, $C=1$.

Let us assume that this document consists of a single table in HTML format. Thus, $x$ is a string such as,

```html
<html>
<table><thead><tr><th colspan="3"><strong>Noise Charges</strong></th></tr><tr><th>QC</th><th>Set fee per Tonne 2025<br>Day</th><th>Set fee per Tonne 2025<br>Night</th></tr></thead><tbody><tr><td>0</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.125</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.25</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€0.00</span></td></tr><tr><td>0.5</td><td><span style="color: green;">€0.00</span></td><td><span style="color: green;">€2.00</span></td></tr><tr><td>1</td><td><span style="color: green;">€1.00</span></td><td><span style="color: green;">€4.00</span></td></tr><tr><td>2</td><td><span style="color: green;">€2.00</span></td><td><span style="color: green;">€8.00</span></td></tr><tr><td>4</td><td><span style="color: green;">€4.00</span></td><td><span style="color: green;">€12.00</span></td></tr><tr><td>8</td><td><span style="color: green;">€6.00</span></td><td><span style="color: green;">€16.00</span></td></tr><tr><td>16</td><td><span style="color: green;">€8.00</span></td><td><span style="color: green;">€20.00</span></td></tr></tbody></table>
</html>
```

We are also given the formula,
```python
def compute_noise_charge(document, qc, weight, day_night):
    """Compute the noise charge based on the document, QC, weight, and day/night.
    Args:
        document (str): The HTML content of the document.
        qc (float): The QC value.
        weight (float): The weight in tonnes.
        day_night (str): 'day' or 'night'.
    Returns:
        float: The computed noise charge.
    """
    df = pd.read_html(html)
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
```
This formula correctly computes the noise charge given the document and the values for the variables, $v = \{\text{qc}, \text{weight}, \text{day\_night}\}$.

Define the augmentation function,
```python
def augment_document(document):
    """Augment the document by replacing the numerical values in the HTML with random values.
    Args:
        document (str): The HTML content of the document.
    Returns:
        str: The augmented HTML content with random values.
    """
    import random
    from bs4 import BeautifulSoup

    # Parse the HTML content
    soup = BeautifulSoup(document, "html.parser")
    
    # Find all value cells (inside <span>)
    for span in soup.find_all("span"):
        # Replace the € value with a random number
        new_value = round(random.uniform(0, 25), 2)
        span.string = f"€{new_value:.2f}"
    
    updated_html = str(soup)
    return updated_html
```

This augmentation function maps the document to a new document for which the formula, `compute_noise_charge`, will still be valid.

As such, we can apply the augmentation function to the document a number of times to get a new synthetic dataset.

As a second example of an augmentation function, we could define a function that replaces the names of the charges in the document, e.g. changing "Noise Charges" to "Sound Charges". This would require us to also transform the forumula to reflect this change.

### An Equivalent Formulation

While we have above defined our formulas as $f_{i,c}(x_i, v_c)$, we can also define $g_{i,c}(v_c)$ as the formula that computes the same charge, but does not depend on the document content. In this case, we can write,
$$
g_{i,c}(v_c) = f_{i,c}(x_i, v_c)$$

In our example above, we then might have,
```python
def compute_noise_charge(qc_value, weight, day_night) -> float:
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
```
This function computes the same charge as the original formula, but does not use the document as an input. 

**Note**: It is not clear to me yet which is more useful. The former function makes sythetic data generation very simple, while the latter function is more like the expected output. Intuitively, the latter function seems better if possible.

In this case we can define the augmentation function as,
$$
\mathcal{A}_{\chi_{c}}(x_i, g_{i,c}) = (x'_i, g'_{i,c})
$$
where $g'_{i,c}$ is the transformed formula for the augmented document $x'_i$. We then require that,
$$
g'_{i,c}(v_c) = g_{i,c}(v_c)
$$

We then just have to figure out how to transform the formula. For example, in the case of the random number augmentation of the document, we will need to replace the numerical values in the formula with the same random values.