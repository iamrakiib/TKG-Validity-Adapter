from __future__ import annotations
import numpy as np
import torch
from hva.history_utils import score_only_topk


def main():
    scores = torch.tensor([[0.1, 0.2, 0.3, -5.0]])
    gold = 3
    topk = score_only_topk(scores, 2)
    assert gold not in topk[0].tolist(), "Gold was incorrectly inserted into top-k"
    print("OK: TeMP-HVA top-k selection is score-only and leakage-free.")

if __name__ == "__main__":
    main()
