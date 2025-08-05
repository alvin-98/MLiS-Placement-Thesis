## Setup

These scripts rely on the `lmblocks` package on the `adversarial_decoding` branch.

These can be obtained by cloning the relevant repository:
```bash
git clone https://github.com/UoN-MLiS/lmblocks
```

To install the packages onto a new environment, first create the environment. Note that the `python` version `3.11` is supported (other versions may work but have not been tested):
```bash
conda create --name lmb python==3.11
conda activate lmb
```

Then install the packages:
```bash
cd <path/to/lmblocks>
pip install -e .
```

On Ada, to create the conda environment, run:
```
module load anaconda-uoneasy
conda create --name lmb python==3.11
source activate lmb
```
Note the use of `source activate` rather than `conda activate`.

Credentials are managed by utils in `lmblocks`. As stated in the docstrings, by default assumes a file in a hidden folder `~/mlflow/credentials` with the structure:

```
[mlflow]
MLFLOW_TRACKING_USERNAME=XXXX
MLFLOW_TRACKING_PASSWORD=XXXXX
MLFLOW_TRACKING_URI=XXXXXX
HF_TOKEN=XXXXX
```