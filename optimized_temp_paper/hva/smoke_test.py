from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from hva.history_utils import score_only_topk


def test_score_only_topk_does_not_insert_gold():
    scores = torch.tensor([[0.9, 0.8, 0.7, -10.0]])
    topk = score_only_topk(scores, 2)
    assert topk.tolist() == [[0, 1]], topk.tolist()
    assert 3 not in topk.tolist()[0], "gold-like low-score entity was inserted into top-k"


def main():
    test_score_only_topk_does_not_insert_gold()
    print("Smoke test passed: score-only top-k does not insert gold candidates.")


if __name__ == "__main__":
    main()
