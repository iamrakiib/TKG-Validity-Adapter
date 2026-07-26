#!/usr/bin/env bash
set -e
CKPT_DIR=${1:?"Usage: bash scripts/02_dump_temp_valid_scores_icews18.sh <TeMP checkpoint experiment folder>"}
rm -rf score_dumps/temp/ICEWS18/valid_chunks
mkdir -p score_dumps/temp/ICEWS18/valid_chunks
python -u test.py \
  --checkpoint-path "${CKPT_DIR}" \
  --n-gpu 0 \
  --hva-dump-split valid \
  --dump-score-dir score_dumps/temp/ICEWS18/valid_chunks \
  --dump-score-object-only
python -m hva.merge_temp_score_chunks \
  --chunk-dir score_dumps/temp/ICEWS18/valid_chunks \
  --out score_dumps/temp/ICEWS18/valid_scores.npz
