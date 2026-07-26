from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Set
import json
import numpy as np

@dataclass
class RankingMetrics:
    mrr: float
    hits1: float
    hits3: float
    hits10: float
    mean_rank: float
    num_queries: int

    def to_dict(self):
        return asdict(self)


def filtered_rank_from_sparse_scores(
    scores: Mapping[int, float],
    gold: int,
    num_entities: int,
    filter_objects: Set[int] | None = None,
    tie_policy: str = "stable_id",
) -> float:
    """Compute filtered rank from a sparse score dictionary.

    Unmentioned entities have score 0.0. Gold is never inserted into top-k;
    if it receives no recurrence score, its score is naturally 0.0.

    tie_policy:
      stable_id: deterministic entity-id tie break, useful for reproducible code.
      average: average rank among tied candidates.
      optimistic: count only strictly greater scores.
    """
    filter_objects = set(filter_objects or set())
    filter_objects.discard(int(gold))
    gold = int(gold)
    gold_score = float(scores.get(gold, 0.0))

    greater = 0
    equal = 0
    lower_id_equal = 0

    # Count explicit scored candidates.
    seen = set()
    for ent, score in scores.items():
        ent = int(ent)
        if ent in filter_objects:
            continue
        seen.add(ent)
        score = float(score)
        if score > gold_score:
            greater += 1
        elif score == gold_score:
            equal += 1
            if ent < gold:
                lower_id_equal += 1

    # Add implicit zero-score candidates not present in scores.
    implicit_zero_count = num_entities - len(filter_objects) - len(seen)
    if gold_score == 0.0:
        equal += implicit_zero_count
        # For stable-id tie-break, count implicit zero entities with id < gold.
        # Some may be filtered or explicitly scored; remove those.
        filtered_lower = sum(1 for ent in filter_objects if ent < gold)
        explicit_lower_seen = sum(1 for ent in seen if ent < gold)
        lower_id_equal += max(0, gold - filtered_lower - explicit_lower_seen)

    if tie_policy == "optimistic":
        return float(1 + greater)
    if tie_policy == "average":
        return float(1 + greater + max(0, equal - 1) / 2.0)
    if tie_policy == "stable_id":
        return float(1 + greater + lower_id_equal)
    raise ValueError(f"Unknown tie policy: {tie_policy}")


def metrics_from_ranks(ranks: List[float]) -> RankingMetrics:
    arr = np.asarray(ranks, dtype=np.float64)
    if arr.size == 0:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    return RankingMetrics(
        mrr=float(np.mean(1.0 / arr)),
        hits1=float(np.mean(arr <= 1)),
        hits3=float(np.mean(arr <= 3)),
        hits10=float(np.mean(arr <= 10)),
        mean_rank=float(np.mean(arr)),
        num_queries=int(arr.size),
    )


def save_metrics(path, metrics: RankingMetrics, extra: dict | None = None) -> None:
    payload = metrics.to_dict()
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
