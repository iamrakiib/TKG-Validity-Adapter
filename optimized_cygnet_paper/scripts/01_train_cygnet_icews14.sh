#!/usr/bin/env bash
set -euo pipefail
SEED=${1:-42}
GPU=${GPU:-0}
RUN_NAME="seed${SEED}"

bash scripts/00_prepare_cygnet_history_icews14.sh

python train.py --dataset ICEWS14 --entity object --time-stamp 24 --alpha 0.8   --lr 0.001 --n-epochs 30 --hidden-dim 200 --gpu ${GPU} --batch-size 1024   --counts 4 --valid-epoch 5 --seed ${SEED} --row-name native_baseline   --run-name ${RUN_NAME} --method native_baseline --save-best --save-latest

python train.py --dataset ICEWS14 --entity subject --time-stamp 24 --alpha 0.8   --lr 0.001 --n-epochs 30 --hidden-dim 200 --gpu ${GPU} --batch-size 1024   --counts 4 --valid-epoch 5 --seed ${SEED} --row-name native_baseline   --run-name ${RUN_NAME} --method native_baseline --save-best --save-latest
