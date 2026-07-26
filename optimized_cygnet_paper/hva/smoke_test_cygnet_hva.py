from __future__ import annotations
import torch
from hva.history_utils import score_only_topk


def main():
    scores = torch.tensor([[0.9, 0.8, 0.1, 0.2], [0.1, 0.2, 0.3, 0.4]])
    gold = torch.tensor([2, 0])  # deliberately outside top-2 for both rows
    topk = score_only_topk(scores, topk=2)
    assert topk.tolist() == [[0, 1], [3, 2]], topk.tolist()
    assert not bool((topk == gold.view(-1, 1)).all()), "target labels must not define top-k candidate membership"
    print("CyGNet HVA smoke test passed: top-K is selected from backbone scores only; target labels are not used for candidate membership.")


if __name__ == "__main__":
    main()
