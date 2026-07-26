from __future__ import annotations
import torch
from hva.history_utils import score_only_topk

def test_topk_does_not_insert_gold():
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.3]])
    topk = score_only_topk(scores, 2)
    # The gold id could be 0, but it is not in top-k and must not be inserted.
    assert topk.tolist() == [[1, 3]], topk.tolist()

if __name__ == "__main__":
    test_topk_does_not_insert_gold()
    print("RE-GCN-HVA smoke test passed: top-k uses backbone scores only.")
