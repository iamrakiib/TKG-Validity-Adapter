#!/usr/bin/env bash
set -e
SEED=${1:-42}
python -m hva.run_exact_recency --dataset ICEWS18 --data-root data \
  --dump score_dumps/tirgn/ICEWS18/test_scores_seed${SEED}.npz --split test \
  --alpha 0.1 --lambda-decay 0.1 --stale-threshold 10 \
  --out-dir results/tirgn/ICEWS18/exact_recency_seed${SEED}
