#!/usr/bin/env bash
set -e
SEED=${1:-42}
bash scripts/03_run_temp_exact_recency_icews14.sh
bash scripts/03_run_temp_rhvc_icews14.sh ${SEED}
bash scripts/03_run_temp_hva_dual_icews14.sh ${SEED}
bash scripts/03_run_temp_hva_exact_icews14.sh ${SEED}
