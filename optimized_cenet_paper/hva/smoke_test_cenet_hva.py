from __future__ import annotations
import torch
from hva.history_utils import score_only_topk


def main():
    scores = torch.tensor([[0.10, 0.20, 0.30, -9.0]])
    gold = 3
    topk = score_only_topk(scores, 2)
    assert gold not in topk[0].tolist(), 'Gold was inserted into top-K. This violates the paper protocol.'
    print('OK: CENET-HVA top-K selection is score-only and leakage-free.')


if __name__ == '__main__':
    main()
