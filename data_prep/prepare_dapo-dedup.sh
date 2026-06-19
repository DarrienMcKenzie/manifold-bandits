#!/bin/bash
set -uxo pipefail

export VERL_HOME=${VERL_HOME:-"."}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-17k.parquet"}
export OVERWRITE=${OVERWRITE:-0}

mkdir -p "${VERL_HOME}/data"

if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed/resolve/main/all/train-00000-of-00001.parquet?download=true"
  python data_prep/dapo_fix.py
fi