#!/usr/bin/env bash
set -e
SEED=${1:-42}
export PYTHONHASHSEED=${SEED}
python -u main.py \
  -d ICEWS18 \
  --description optimized_cenet_seed${SEED}_ \
  --max-epochs 30 \
  --oracle-epochs 20 \
  --valid-epochs 5 \
  --save_dir SAVE/cenet/ICEWS18/seed${SEED} \
  2>&1 | tee logs/cenet_icews18_seed${SEED}.log
