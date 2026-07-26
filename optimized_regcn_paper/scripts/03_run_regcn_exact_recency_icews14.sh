#!/usr/bin/env bash
set -e
SEED=${1:-42}
python -m hva.run_exact_recency --dataset ICEWS14 --data-root data \
  --dump score_dumps/regcn/ICEWS14/test_scores_seed${SEED}.npz --split test \
  --alpha 0.1 --lambda-decay 0.1 --stale-threshold 10 \
  --out-dir results/regcn/ICEWS14/exact_recency_seed${SEED}
