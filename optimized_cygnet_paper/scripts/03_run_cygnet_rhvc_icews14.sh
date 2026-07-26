#!/usr/bin/env bash
set -euo pipefail
SEED=${1:-42}
TOPK=${TOPK:-100}
RUN_NAME="seed${SEED}"
BASE="runs/cygnet/ICEWS14/native_baseline/${RUN_NAME}"
OUT="results/ICEWS14/cygnet/rhvc/${RUN_NAME}"

mkdir -p "${OUT}/object" "${OUT}/subject" "${OUT}/combined"

python -m hva.run_rhvc_from_scores --dataset ICEWS14 --data-root data   --valid-dump "${BASE}/object/dumps/valid_scores.npz"   --test-dump "${BASE}/object/dumps/test_scores.npz"   --out-dir "${OUT}/object" --topk ${TOPK} --eval-topk ${TOPK}   --seed ${SEED}  --save-adjusted-scores

python -m hva.run_rhvc_from_scores --dataset ICEWS14 --data-root data   --valid-dump "${BASE}/subject/dumps/valid_scores.npz"   --test-dump "${BASE}/subject/dumps/test_scores.npz"   --out-dir "${OUT}/subject" --topk ${TOPK} --eval-topk ${TOPK}   --seed ${SEED}  --save-adjusted-scores

# Average object and subject branch results for CyGNet reporting.
OBJ_JSON="${OUT}/object/rhvc_results.json"
SUB_JSON="${OUT}/subject/rhvc_results.json"
python -m hva.combine_branch_results --object-result "${OBJ_JSON}" --subject-result "${SUB_JSON}" --out "${OUT}/combined/combined_results.json"
