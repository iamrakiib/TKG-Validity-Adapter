#!/usr/bin/env bash
set -e
for S in 42 123 2026 7 3407; do
  bash scripts/03_run_temp_hva_icews14.sh $S
  bash scripts/03_run_temp_hva_icews18.sh $S
done
