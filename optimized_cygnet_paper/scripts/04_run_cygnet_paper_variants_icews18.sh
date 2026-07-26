#!/usr/bin/env bash
set -euo pipefail
SEED=${1:-42}
bash scripts/02_dump_cygnet_valid_test_scores_icews18.sh ${SEED}
bash scripts/03_run_cygnet_exact_recency_icews18.sh ${SEED}
bash scripts/03_run_cygnet_rhvc_icews18.sh ${SEED}
bash scripts/03_run_cygnet_hva_dual_icews18.sh ${SEED}
bash scripts/03_run_cygnet_hva_exact_icews18.sh ${SEED}
