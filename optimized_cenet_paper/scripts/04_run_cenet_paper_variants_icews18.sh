#!/usr/bin/env bash
set -e
SEED=${1:-42}
bash scripts/03_run_cenet_exact_recency_icews18.sh
bash scripts/03_run_cenet_rhvc_icews18.sh ${SEED}
bash scripts/03_run_cenet_hva_dual_icews18.sh ${SEED}
bash scripts/03_run_cenet_hva_exact_icews18.sh ${SEED}
