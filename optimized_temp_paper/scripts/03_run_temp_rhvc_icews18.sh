#!/usr/bin/env bash
set -e
SEED=${1:-42}
python -m hva.run_rhvc_from_scores \
  --dataset ICEWS18 \
  --data-root data \
  --valid-dump score_dumps/temp/ICEWS18/valid_scores.npz \
  --test-dump score_dumps/temp/ICEWS18/test_scores.npz \
  --out-dir results/temp/ICEWS18/rhvc_seed${SEED} \
  --topk 100 \
  --seed ${SEED} \
  --save-adjusted-scores
