# MLiS Placement Thesis

## About

Mono-repo consisting of packages developed as part of the MLiS Placement and associated Thesis work.

- `synthdoc`: for generation of synthetic documents
- `red_team`: for automated LLM red-teaming
- `infer_opt`: for inference optimisation
- `struc_extract`: for structured extraction of information from documents
- [...]

## Setup

Example with conda. First create and activate the environment:
```bash
conda create -n mlisp python==3.11
```
then,
```bash
conda activate mlisp
```
or, on the HPC,
```bash
source activate mlisp
```
Then install the package. From the top-level directory (containing setup.py):
```bash
pip install -e .
```

To install addition dependencies for a given package, include the options, e.g.,
```bash
pip install -e .[struc_extract]
```

## Example Usage

Different packages can be accessed as modules,

```python
import mlisplacement as mlp
mlp.red_team.get_prompt_style("default")
```

## Synthdoc

See example usage in `examples\synthdoc`.

This package provides functionality to synthesize documents. Including:
- **Function creation**: Uses hard-coded methods to generate python functions from given combinations of charges and variables.
- **LLM Assisted Document Generation**: Uses LLMs to synthesise documents with a given structure.

### Function Creation

- Functions are provided in `synthdoc.function_creation` module.
- These can be combined with example charge-variable structures from `synthdoc.structs.airports`, `synthdoc.structs.azure` and so on.
- A cli tool is also available via `generate_functions`