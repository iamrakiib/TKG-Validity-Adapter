from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple
import math
import numpy as np

@dataclass
class HRIParams:
    w_occurrence: float = 1.0
    w_frequency: float = 0.25
    w_recency: float = 1.0
    decay: float = 0.03
    stale_threshold: int = -1
    stale_penalty: float = 0.0


class HRIHistory:
    """Causal recurrence memory for the HRI-inspired baseline.

    For each exact pair (subject, relation), the memory stores candidate objects,
    their frequency, and their most recent timestamp. This matches the paper's
    recurrence baseline role: no neural backbone, no HVA correction, no gold
    insertion, and only facts with t' < t are used for prediction.
    """

    def __init__(self) -> None:
        self.count = defaultdict(lambda: defaultdict(int))       # (s,r) -> c -> count
        self.last_t = defaultdict(dict)                          # (s,r) -> c -> last timestamp
        self.subject_any = defaultdict(set)                      # s -> historical objects
        self.relation_any = defaultdict(set)                     # r -> historical objects

    def update_many(self, quads: np.ndarray) -> None:
        for s, r, o, t in quads.tolist():
            self.update(int(s), int(r), int(o), int(t))

    def update(self, s: int, r: int, o: int, t: int) -> None:
        key = (int(s), int(r))
        o = int(o)
        t = int(t)
        self.count[key][o] += 1
        old = self.last_t[key].get(o)
        if old is None or t > old:
            self.last_t[key][o] = t
        self.subject_any[int(s)].add(o)
        self.relation_any[int(r)].add(o)

    def score_candidates(self, s: int, r: int, query_t: int, params: HRIParams) -> Dict[int, float]:
        key = (int(s), int(r))
        query_t = int(query_t)
        scores: Dict[int, float] = {}
        if key not in self.count:
            return scores
        for c, freq in self.count[key].items():
            last = self.last_t[key].get(c, query_t)
            delta = max(0, query_t - int(last))
            score = 0.0
            score += params.w_occurrence
            score += params.w_frequency * math.log1p(float(freq))
            score += params.w_recency * math.exp(-params.decay * float(delta))
            if params.stale_threshold is not None and params.stale_threshold >= 0 and delta > params.stale_threshold:
                score += params.stale_penalty
            scores[int(c)] = float(score)
        return scores

    def diagnostic_label(self, s: int, r: int, o: int) -> str:
        key = (int(s), int(r))
        if int(o) in self.count.get(key, {}):
            return "repeat"
        if int(o) in self.subject_any.get(int(s), set()) or int(o) in self.relation_any.get(int(r), set()):
            return "near_repeat"
        return "novel"
