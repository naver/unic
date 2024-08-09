#!/bin/bash

args="$*"
echo "Args passed to bash ${0##*/}:"
echo "=> ${args}"

source ./scripts/setup_env.sh
pretrained="/path/to/unic/checkpoint.pth"

python eval_dense/eval_semseg.py --pretrained=${pretrained}