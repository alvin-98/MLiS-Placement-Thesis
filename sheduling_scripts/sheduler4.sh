#!/bin/bash
#SBATCH --job-name=hb-loop
#SBATCH --partition=ampereq
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=400G
#SBATCH --time=100:00:00
#SBATCH --output=outs/ampereq_%j.out
#SBATCH --open-mode=append

set -euo pipefail

# how many times to repeat the triplet
N_ITER=100

module purge
module load anaconda-uoneasy
source /gpfs01/software/easybuild-ada-uon/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nlp

echo "Starting pipeline on $(date)"
for i in $(seq 1 "$N_ITER"); do
  echo "=== Iteration $i ==="

  srun --kill-on-bad-exit=1 --label python LLM_generated_data/scripts/input_gen.py

  srun --kill-on-bad-exit=1 --label python LLM_generated_data/scripts/llm_gen_8B.py
 
  srun --kill-on-bad-exit=1 --label python LLM_generated_data/scripts/llm_gen_70B.py

  echo "=== Iteration $i complete ==="
done
echo "Done on $(date)"
