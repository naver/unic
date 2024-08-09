#!/bin/bash

args="$*"

echo "Args passed to bash ${0##*/}:"
echo "=> ${args}"

# initialize the conda environment
source ./scripts/setup_env.sh

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=${N_GPUS} main_unic.py  \
    --seed=${RANDOM} \
    ${args}