#!/usr/bin/env bash
# Load toolchains
module purge
module load gcc-uoneasy/12.3.0
module load Clang/16.0.6-GCCcore-12.3.0
module load CUDA/12.4.0

# Compiler/env for Triton
export CC=clang
export CXX=clang++
export TRITON_BUILD_WITH_CLANG_LLD=1
rm -rf ~/.triton ~/.cache/triton

module list
echo "Environment ready."