#!/usr/bin/env bash
set -e
SEED=${1:-42}
bash scripts/03_run_regcn_exact_recency_icews18.sh ${SEED}
bash scripts/03_run_regcn_rhvc_icews18.sh ${SEED}
bash scripts/03_run_regcn_hva_dual_icews18.sh ${SEED}
bash scripts/03_run_regcn_hva_exact_icews18.sh ${SEED}
