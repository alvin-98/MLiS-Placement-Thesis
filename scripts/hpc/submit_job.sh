#!/bin/bash
#SBATCH --partition=ampereq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:7
#SBATCH --time=10:00:00
#SBATCH --job-name=test_script
#SBATCH --output=test_script_%j.out
#SBATCH --error=test_script_%j.err

module purge
module load cuda-12.2.2
module load anaconda-uoneasy/2023.09-0
source /gpfs01/home/ppxac9/.zshrc
conda activate placement_dev

srun python multi_gpu_single_node_inference.py
