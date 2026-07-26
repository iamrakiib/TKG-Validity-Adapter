#!/usr/bin/env bash
set -e
SEED=${1:-42}
bash scripts/04_run_regcn_paper_variants_icews14.sh ${SEED}
bash scripts/04_run_regcn_paper_variants_icews18.sh ${SEED}
