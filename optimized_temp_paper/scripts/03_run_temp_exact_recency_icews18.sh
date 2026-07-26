#!/usr/bin/env bash
set -e
python -m hva.run_exact_recency \
  --dataset ICEWS18 \
  --data-root data \
  --dump score_dumps/temp/ICEWS18/test_scores.npz \
  --split test \
  --alpha 0.1 \
  --lambda-decay 0.1 \
  --stale-threshold 10 \
  --out-dir results/temp/ICEWS18/exact_recency
