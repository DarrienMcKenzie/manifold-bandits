#!/bin/bash
bash data_prep/prepare_dapo-dedup.sh
python data_prep/eval_process.py --tasks math --filename math_eval.parquet
python data_prep/eval_prep.py --tasks math 