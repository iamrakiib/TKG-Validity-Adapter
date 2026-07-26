#!/usr/bin/env bash
set -e
SEED=${1:-42}
python -m hva.run_hva_from_scores --dataset ICEWS14 --data-root data \
  --valid-dump score_dumps/tirgn/ICEWS14/valid_scores_seed${SEED}.npz \
  --test-dump score_dumps/tirgn/ICEWS14/test_scores_seed${SEED}.npz \
  --out-dir results/tirgn/ICEWS14/hva_exact_seed${SEED} \
  --mode exact_only --topk 100 --eval-topk 100 --epochs 12 --seed ${SEED} --save-adjusted-scores
