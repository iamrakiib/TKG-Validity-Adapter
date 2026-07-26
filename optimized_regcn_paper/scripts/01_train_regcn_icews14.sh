#!/usr/bin/env bash
set -e
SEED=${1:-42}
cd src
python main.py -d ICEWS14 \
  --train-history-len 3 --test-history-len 3 --dilate-len 1 \
  --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop \
  --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
  --entity-prediction --relation-prediction --add-static-graph --angle 10 \
  --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint \
  --ckpt-dir ../checkpoints/regcn/ICEWS14/seed${SEED} \
  --train-log-path ../logs/regcn_icews14_seed${SEED}_train.json
