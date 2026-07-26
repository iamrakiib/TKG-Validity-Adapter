#!/usr/bin/env bash
set -euo pipefail
SEED=${1:-42}
bash scripts/04_run_cygnet_paper_variants_icews14.sh ${SEED}
bash scripts/04_run_cygnet_paper_variants_icews18.sh ${SEED}
