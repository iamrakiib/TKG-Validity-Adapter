#!/usr/bin/env bash
set -e
SEED=${1:-42}
python -u main.py \
  --dataset-dir extrapolation \
  -d icews18 \
  --module GRRGCN \
  --score-function complex \
  --n-gpu 0 \
  --max-nb-epochs 100 \
  --patience 10 \
  --train-seq-len 15 \
  --test-seq-len 30 \
  --batch-size 8 \
  --filtered \
  --rec-only-last-layer \
  --use-time-embedding \
  --post-ensemble \
  --seed ${SEED}
