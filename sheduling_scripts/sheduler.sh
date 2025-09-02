#!/bin/bash

#SBATCH --partition=ampereq
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --time=10:00:00
#SBATCH --output=outs/ampereq_%j.out

module purge
module load anaconda-uoneasy
source /gpfs01/software/easybuild-ada-uon/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nlp

python PAIR/analysis_chatTemplates.py




