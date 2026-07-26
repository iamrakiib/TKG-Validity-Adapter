#!/usr/bin/env bash
set -euo pipefail
for SEED in 42 123 2026 7 3407; do
  bash scripts/01_train_cygnet_icews14.sh ${SEED}
  bash scripts/04_run_cygnet_paper_variants_icews14.sh ${SEED}
  bash scripts/01_train_cygnet_icews18.sh ${SEED}
  bash scripts/04_run_cygnet_paper_variants_icews18.sh ${SEED}
done
