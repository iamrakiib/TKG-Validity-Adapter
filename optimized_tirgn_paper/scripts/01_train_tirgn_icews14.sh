#!/usr/bin/env bash
set -e
SEED=${1:-42}
cd src
python main.py -d ICEWS14 \
  --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 \
  --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop \
  --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5 \
  --entity-prediction --relation-prediction --add-static-graph --angle 14 \
  --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint \
  --ckpt-dir ../checkpoints/tirgn/ICEWS14/seed${SEED} \
  --train-log-path ../logs/tirgn_icews14_seed${SEED}_train.json
