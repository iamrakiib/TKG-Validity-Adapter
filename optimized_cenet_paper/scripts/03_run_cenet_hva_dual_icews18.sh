#!/usr/bin/env bash
set -e
SEED=${1:-42}
python -m hva.run_hva_from_scores \
  --dataset ICEWS18 \
  --data-root data \
  --valid-dump score_dumps/cenet/ICEWS18/valid_scores.npz \
  --test-dump score_dumps/cenet/ICEWS18/test_scores.npz \
  --out-dir results/cenet/ICEWS18/hva_dual_seed${SEED} \
  --mode dual_branch \
  --topk 100 --eval-topk 100 \
  --epochs 12 --seed ${SEED} \
  --save-adjusted-scores
