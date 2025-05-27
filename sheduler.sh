#!/bin/bash

#SBATCH --partition=shortq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=myscript_%j.out

module purge
module load python-uoneasy
module load anaconda-uoneasy 
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1


source /gpfs01/software/easybuild-ada-uon/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nlp
conda config --add channels conda-forge
conda install -y transformers datasets accelerate || \

python quick_test.py






