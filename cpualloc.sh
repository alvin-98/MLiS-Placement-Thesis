#!/bin/bash
echo "Running interactive session for 12 hours"
echo "Use control + a to deattach from screen and screen -r cpuinter to reconnect"
sleep 1
screen -S cpuinter srun --partition=ampereq --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:7 --mem=300G --time=12:00:00 --pty bash