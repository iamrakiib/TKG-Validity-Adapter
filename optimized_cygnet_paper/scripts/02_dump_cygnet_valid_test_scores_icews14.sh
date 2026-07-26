#!/usr/bin/env bash
set -euo pipefail
SEED=${1:-42}
bash scripts/02_dump_cygnet_valid_scores_icews14.sh ${SEED}
bash scripts/02_dump_cygnet_test_scores_icews14.sh ${SEED}
