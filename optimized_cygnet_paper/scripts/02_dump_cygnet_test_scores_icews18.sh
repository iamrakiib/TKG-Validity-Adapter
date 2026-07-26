#!/usr/bin/env bash
set -euo pipefail
SEED=${1:-42}
GPU=${GPU:-0}
RUN_NAME="seed${SEED}"

python test.py --dataset ICEWS18 --entity object --time-stamp 24 --alpha 0.8   --hidden-dim 200 --gpu ${GPU} --batch-size 1024 --seed ${SEED}   --row-name native_baseline --run-name ${RUN_NAME} --eval-split test   --dump-full-scores --dump-test

python test.py --dataset ICEWS18 --entity subject --time-stamp 24 --alpha 0.8   --hidden-dim 200 --gpu ${GPU} --batch-size 1024 --seed ${SEED}   --row-name native_baseline --run-name ${RUN_NAME} --eval-split test   --dump-full-scores --dump-test
