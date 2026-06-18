#!/bin/bash
python data_prep/eval_process.py --tasks alphamed --filename alphamed-19k.parquet
python data_prep/eval_prep.py --tasks alphamed
python data_prep/eval_process.py --tasks med --filename med_eval.parquet
python data_prep/eval_prep.py --tasks med