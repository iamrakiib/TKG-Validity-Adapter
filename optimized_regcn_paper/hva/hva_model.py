from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .history_utils import (
    freq_before,
    last_time_before,
    score_only_topk,
    scatter_topk_back,
)


class HistoryValidityAdapter(nn.Module):
    """Candidate-level History Validity Adapter.

    exact_only features per selected candidate:
      [exact_seen, recency_score, log_frequency, stale_flag]

    dual_branch additionally appends the same three histories for (s,*,c)
    and (*,r,c). The final paper setting should usually use mode='exact_only'.
    """

    def __init__(
        self,
        num_relations: int,
        mode: str = "exact_only",
        rel_emb_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        gamma: float = 0.1,
        stale_threshold: int = 10,
    ) -> None:
        super().__init__()
        if mode not in {"exact_only", "dual_branch"}:
            raise ValueError("mode must be exact_only or dual_branch")
        self.mode = mode
        self.stale_threshold = int(stale_threshold)
        self.rel_emb = nn.Embedding(int(max(1, num_relations)), int(rel_emb_dim))
        feat_dim = 4 if mode == "exact_only" else 12
        in_dim = feat_dim + int(rel_emb_dim) + 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        self.gamma_raw = nn.Parameter(torch.tensor(float(gamma)))

    def forward(self, base_scores_topk: torch.Tensor, rel_ids: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Small rank/score context prevents the adapter from treating all top-k
        # candidates as equally reliable.
        centered = base_scores_topk - base_scores_topk.mean(dim=1, keepdim=True)
        gap_from_top1 = base_scores_topk.max(dim=1, keepdim=True).values - base_scores_topk
        k = base_scores_topk.size(1)
        rank_context = torch.arange(k, device=base_scores_topk.device, dtype=torch.float32).view(1, k)
        rank_context = rank_context.expand_as(base_scores_topk) / max(k - 1, 1)

        rel_ctx = self.rel_emb(rel_ids.clamp_min(0) % self.rel_emb.num_embeddings)
        rel_ctx = rel_ctx.unsqueeze(1).expand(-1, k, -1)

        x = torch.cat([
            features,
            rel_ctx,
            centered.unsqueeze(-1),
            gap_from_top1.unsqueeze(-1),
            rank_context.unsqueeze(-1),
        ], dim=-1)
        delta = self.mlp(x).squeeze(-1)
        gamma = torch.tanh(self.gamma_raw)
        adjusted = base_scores_topk + gamma * delta
        return adjusted, gamma * delta


def _feature_triplet(times, t: int, stale_threshold: int):
    lt = last_time_before(times, int(t))
    if lt is None:
        return 0.0, 0.0, 0.0, 0.0
    gap = max(0, int(t) - int(lt))
    freq = freq_before(times, int(t))
    seen = 1.0
    recency = 1.0 / (1.0 + float(gap))
    log_freq = float(np.log1p(freq))
    stale = 1.0 if gap > int(stale_threshold) else 0.0
    return seen, recency, log_freq, stale


def build_hva_features(query_triples, candidate_ids, histories: Dict[str, object], device, mode: str = "exact_only", stale_threshold: int = 10) -> torch.Tensor:
    cand_np = candidate_ids.detach().cpu().numpy() if torch.is_tensor(candidate_ids) else np.asarray(candidate_ids)
    query_np = query_triples.detach().cpu().numpy() if torch.is_tensor(query_triples) else np.asarray(query_triples)
    sr_hist = histories["sr"]
    so_hist = histories["so"]
    ro_hist = histories["ro"]

    bsz, k = cand_np.shape
    feat_dim = 4 if mode == "exact_only" else 12
    feats = np.zeros((bsz, k, feat_dim), dtype=np.float32)

    for i in range(bsz):
        s, r, _gold, t = map(int, query_np[i][:4])
        exact_map = sr_hist.get((s, r), {})
        so_map = so_hist.get(s, {})
        ro_map = ro_hist.get(r, {})
        for j in range(k):
            c = int(cand_np[i, j])
            feats[i, j, 0:4] = _feature_triplet(exact_map.get(c, []), t, stale_threshold)
            if mode == "dual_branch":
                feats[i, j, 4:8] = _feature_triplet(so_map.get(c, []), t, stale_threshold)
                feats[i, j, 8:12] = _feature_triplet(ro_map.get(c, []), t, stale_threshold)
    return torch.tensor(feats, dtype=torch.float32, device=device)


def apply_hva_batch(model: HistoryValidityAdapter, base_scores: torch.Tensor, query_triples, histories: Dict[str, object], topk: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_ids = score_only_topk(base_scores, topk)
    base_topk = torch.gather(base_scores, 1, candidate_ids)
    q_np = np.asarray(query_triples).copy()
    # Defence-in-depth: feature construction must not depend on the gold column.
    # Current feature code ignores the third column, but masking prevents future
    # accidental target use.
    q_np[:, 2] = -1
    rel_ids = torch.tensor(np.asarray(query_triples)[:, 1], dtype=torch.long, device=device)
    features = build_hva_features(q_np, candidate_ids, histories, device, mode=model.mode, stale_threshold=model.stale_threshold)
    adjusted_topk, delta = model(base_topk, rel_ids, features)
    adjusted_full = scatter_topk_back(base_scores, candidate_ids, adjusted_topk)
    return adjusted_full, adjusted_topk, candidate_ids, delta
