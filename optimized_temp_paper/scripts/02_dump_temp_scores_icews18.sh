#!/usr/bin/env bash
set -e
CKPT_DIR=${1:?"Usage: bash scripts/02_dump_temp_scores_icews18.sh <TeMP checkpoint experiment folder>"}
rm -rf score_dumps/temp/ICEWS18/test_chunks
mkdir -p score_dumps/temp/ICEWS18/test_chunks
python -u test.py \
  --checkpoint-path "${CKPT_DIR}" \
  --n-gpu 0 \
  --dump-score-dir score_dumps/temp/ICEWS18/test_chunks \
  --dump-score-object-only
python -m hva.merge_temp_score_chunks \
  --chunk-dir score_dumps/temp/ICEWS18/test_chunks \
  --out score_dumps/temp/ICEWS18/test_scores.npz
