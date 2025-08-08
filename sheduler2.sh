#!/bin/bash

#SBATCH --partition=ampereq
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=outs/ampereq_%j.out

module purge
module load anaconda-uoneasy

source /gpfs01/software/easybuild-ada-uon/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nlp

cd /gpfs01/home/ppytr13/MLiS-Placement-Thesis-2

# Ensure logs dir exists
mkdir -p logs hf_home

# Optional: make HF token available if needed
# export HUGGINGFACE_TOKEN=hf_xxx

python download_model.py
