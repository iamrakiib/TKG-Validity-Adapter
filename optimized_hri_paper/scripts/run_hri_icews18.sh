#!/usr/bin/env bash
set -euo pipefail

python src/tune_and_run_hri.py \
  --dataset ICEWS18 \
  --data-root data \
  --out-dir results/ICEWS18/optimized_hri \
  --tie-policy stable_id \
  --export-topk \
  --topk 100
