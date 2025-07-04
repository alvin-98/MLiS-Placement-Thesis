#!/bin/bash
echo "Running interactive session for 12 hours"
echo "Use control + a to deattach from screen and screen -r cpuinter to reconnect"
module load cuda-12.2.2
module load anaconda-uoneasy/2023.09-0
source /gpfs01/home/ppxac9/.zshrc
conda activate MonkeyOCR
screen -S cpuinter srun --partition=compchemq --qos=compchem --nodes=1 --ntasks-per-node=1 --cpus-per-task=30 --ntasks-per-socket=1 --gres=gpu:4 --mem=100G --time=12:00:00 --pty bash