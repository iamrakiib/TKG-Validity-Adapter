#!/usr/bin/env bash
set -e
SEED=${1:-42}
CKPT=${2:-../checkpoints/regcn/ICEWS14/seed${SEED}/best.pt}
cd src
python main.py -d ICEWS14 --test --eval-mode dump_test --dump-full-scores \
  --full-score-path ../score_dumps/regcn/ICEWS14/test_scores_seed${SEED}.npz \
  --resume-ckpt ${CKPT} \
  --train-history-len 3 --test-history-len 3 --dilate-len 1 \
  --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop \
  --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5 \
  --entity-prediction --relation-prediction --add-static-graph --angle 10 \
  --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint
