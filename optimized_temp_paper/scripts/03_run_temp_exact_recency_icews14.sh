#!/usr/bin/env bash
set -e
python -m hva.run_exact_recency \
  --dataset ICEWS14 \
  --data-root data \
  --dump score_dumps/temp/ICEWS14/test_scores.npz \
  --split test \
  --alpha 0.1 \
  --lambda-decay 0.1 \
  --stale-threshold 10 \
  --out-dir results/temp/ICEWS14/exact_recency
