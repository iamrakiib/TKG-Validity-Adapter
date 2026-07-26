#!/usr/bin/env bash
set -e
SEED=${1:-42}
CKPT=${2:-../checkpoints/tirgn/ICEWS14/seed${SEED}/best.pt}
cd src
python main.py -d ICEWS14 --test --eval-mode dump_valid --dump-full-scores \
  --full-score-path ../score_dumps/tirgn/ICEWS14/valid_scores_seed${SEED}.npz \
  --resume-ckpt ${CKPT} \
  --history-rate 0.3 --train-history-len 9 --test-history-len 9 --dilate-len 1 \
  --lr 0.001 --n-layers 2 --evaluate-every 1 --n-hidden 200 --self-loop \
  --decoder timeconvtranse --encoder convgcn --layer-norm --weight 0.5 \
  --entity-prediction --relation-prediction --add-static-graph --angle 14 \
  --discount 1 --task-weight 0.7 --gpu 0 --save checkpoint
