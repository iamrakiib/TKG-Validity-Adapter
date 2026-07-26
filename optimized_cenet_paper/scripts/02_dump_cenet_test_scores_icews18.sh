#!/usr/bin/env bash
set -e
CKPT_DIR=${1:?"Usage: bash scripts/02_dump_cenet_test_scores_icews18.sh <CENET experiment folder>"}
python -m hva.dump_cenet_scores \
  --dataset ICEWS18 \
  --split test \
  --data-root data \
  --model-dir "${CKPT_DIR}" \
  --out score_dumps/cenet/ICEWS18/test_scores.npz \
  --score-kind oracle
