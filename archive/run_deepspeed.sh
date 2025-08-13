#!/bin/bash

# This script launches the DeepSpeed inference script.
# It will automatically detect and use all available GPUs on the node.
# The launcher sets environment variables like WORLD_SIZE and LOCAL_RANK,
# which are used by the Python script to manage multi-GPU communication.

deepspeed deep_speed_multi_gpu_single_node_inference.py
