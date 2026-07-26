#!/usr/bin/env bash
set -euo pipefail

# HRI-inspired recurrence baseline for the HVA-TKG paper.
# Validation is used only for hyperparameter selection; test is used once.

python src/tune_and_run_hri.py \
  --dataset ICEWS14 \
  --data-root data \
  --out-dir results/ICEWS14/optimized_hri \
  --tie-policy stable_id \
  --export-topk \
  --topk 100
