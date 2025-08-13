#!/bin/bash

#SBATCH --partition=ampereq
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=100:00:00
#SBATCH --output=outs/ampereq_%j.out

module purge
module load anaconda-uoneasy

source /gpfs01/software/easybuild-ada-uon/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate nlp


python HarmBench/qwen_vs_cogito.py --temperature 1.0   --batch_size 4
python HarmBench/llm_as_judge_probs.py --temperature 1.0   --batch_size 4
python HarmBench/qwen_vs_cogito.py --temperature 1.2   --batch_size 4
python HarmBench/llm_as_judge_probs.py --temperature 1.2   --batch_size 4
python HarmBench/qwen_vs_cogito.py --temperature 1.4   --batch_size 4
python HarmBench/llm_as_judge_probs.py --temperature 1.4   --batch_size 4
python HarmBench/qwen_vs_cogito.py --temperature 1.6   --batch_size 4
python HarmBench/llm_as_judge_probs.py --temperature 1.6   --batch_size 4




