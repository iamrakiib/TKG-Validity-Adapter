from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .history_utils import build_filter_map_from_triples, novelty_bucket_from_history, stale_exact_bucket


def _rank_filtered(scores_row: np.ndarray, s: int, r: int, o: int, t: int, filter_map) -> int:
    row = scores_row.copy()
    filt = filter_map.get((int(s), int(r), int(t)), set())
    for cand in filt:
        if int(cand) != int(o) and 0 <= int(cand) < row.shape[0]:
            row[int(cand)] = -np.inf
    gold_score = row[int(o)]
    return int(np.sum(row > gold_score) + 1)


def ranking_metrics_from_ranks(ranks: Sequence[int]) -> Dict[str, float]:
    if len(ranks) == 0:
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0}
    r = np.asarray(ranks, dtype=np.float64)
    return {
        "MRR": float(np.mean(1.0 / r)),
        "Hits@1": float(np.mean(r <= 1)),
        "Hits@3": float(np.mean(r <= 3)),
        "Hits@10": float(np.mean(r <= 10)),
        "count": int(len(r)),
    }


def evaluate_filtered(scores: np.ndarray, queries: np.ndarray, filter_triples) -> Dict[str, float]:
    fmap = build_filter_map_from_triples(filter_triples)
    ranks = []
    for row, q in zip(scores, queries):
        s, r, o, t = map(int, q[:4])
        ranks.append(_rank_filtered(row, s, r, o, t, fmap))
    return ranking_metrics_from_ranks(ranks)


def evaluate_diagnostics(scores: np.ndarray, queries: np.ndarray, filter_triples, histories, stale_threshold: int = 10) -> Dict[str, object]:
    fmap = build_filter_map_from_triples(filter_triples)
    bucket_ranks = {"repeat": [], "near_repeat": [], "novel": []}
    stale_top1 = 0
    total = 0
    sr = histories["sr"]
    so = histories["so"]
    ro = histories["ro"]
    for row, q in zip(scores, queries):
        s, r, o, t = map(int, q[:4])
        rank = _rank_filtered(row, s, r, o, t, fmap)
        bucket = novelty_bucket_from_history(s, r, o, t, sr, so, ro)
        bucket_ranks[bucket].append(rank)

        filtered = row.copy()
        for cand in fmap.get((s, r, t), set()):
            if int(cand) != o and 0 <= int(cand) < filtered.shape[0]:
                filtered[int(cand)] = -np.inf
        top1 = int(np.argmax(filtered))
        if top1 != o and stale_exact_bucket(s, r, top1, t, sr, stale_threshold) == "stale":
            stale_top1 += 1
        total += 1

    return {
        "repeat": ranking_metrics_from_ranks(bucket_ranks["repeat"]),
        "near_repeat": ranking_metrics_from_ranks(bucket_ranks["near_repeat"]),
        "novel": ranking_metrics_from_ranks(bucket_ranks["novel"]),
        "StaleTop1": float(stale_top1 / total) if total else 0.0,
        "count": int(total),
    }
