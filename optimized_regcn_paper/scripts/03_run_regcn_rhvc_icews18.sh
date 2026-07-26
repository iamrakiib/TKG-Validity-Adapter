#!/usr/bin/env bash
set -e
SEED=${1:-42}
python -m hva.run_rhvc_from_scores --dataset ICEWS18 --data-root data \
  --valid-dump score_dumps/regcn/ICEWS18/valid_scores_seed${SEED}.npz \
  --test-dump score_dumps/regcn/ICEWS18/test_scores_seed${SEED}.npz \
  --out-dir results/regcn/ICEWS18/rhvc_seed${SEED} \
  --topk 100 --save-adjusted-scores --seed ${SEED}
