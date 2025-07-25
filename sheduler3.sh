#!/bin/bash

#SBATCH --partition=shortq
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=outs/ampereq_%j.out

module purge
module load anaconda-uoneasy

source /gpfs01/software/easybuild-ada-uon/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nlp

python Syntheticdata/synthetic.py
