#!/bin/bash

#SBATCH --partition=shortq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=env_setup_%j.out

module purge
module load anaconda-uoneasy

source /gpfs01/software/easybuild-ada-uon/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh

conda deactivate
conda remove -y --name nlp --all

cat <<EOF > nlp_env.yaml
name: nlp
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy
  - scipy
  - pip:
      - torch
      - transformers
      - datasets
EOF

conda env create -f nlp_env.yaml
