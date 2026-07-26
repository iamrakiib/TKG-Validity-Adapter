from __future__ import annotations

import bisect
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

Triple = Tuple[int, int, int, int]
History = Dict[Tuple[int, int], Dict[int, List[int]]]
PairHistory = Dict[int, Dict[int, List[int]]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_stat(data_dir: str) -> Tuple[int, int]:
    path = os.path.join(data_dir, "stat.txt")
    with open(path, "r", encoding="utf-8") as f:
        parts = f.readline().strip().split()
    if len(parts) < 2:
        raise ValueError(f"stat.txt must contain at least num_entities and num_relations: {path}")
    return int(parts[0]), int(parts[1])


def read_triples(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            triples.append(tuple(map(int, parts[:4])))
    return triples


def read_split_arrays(data_dir: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        triples = read_triples(os.path.join(data_dir, f"{split}.txt"))
        out[split] = np.asarray(triples, dtype=np.int64)
    return out


def triples_array_to_list(array_like: np.ndarray) -> List[Triple]:
    arr = np.asarray(array_like)
    if arr.size == 0:
        return []
    return [tuple(map(int, row[:4])) for row in arr]


def augment_with_inverse(triples: Sequence[Triple], num_rels: int) -> List[Triple]:
    aug: List[Triple] = []
    for s, r, o, t in triples:
        aug.append((int(s), int(r), int(o), int(t)))
        aug.append((int(o), int(r) + int(num_rels), int(s), int(t)))
    return aug


def canonicalize_queries(triples: np.ndarray, entity: str = "object", num_rels: Optional[int] = None) -> np.ndarray:
    """Return object-prediction query rows [s, r, gold_object, t].

    If entity='subject', raw rows [s,r,o,t] are converted to inverse-object
    queries [o, r + num_rels, s, t]. This keeps HVA features and metrics in a
    single object-ranking format.
    """
    arr = np.asarray(triples, dtype=np.int64)
    if entity in ("object", "tail", "o"):
        return arr.copy()
    if entity in ("subject", "head", "s"):
        if num_rels is None:
            raise ValueError("num_rels is required to canonicalize subject prediction")
        out = arr.copy()
        out[:, 0] = arr[:, 2]
        out[:, 1] = arr[:, 1] + int(num_rels)
        out[:, 2] = arr[:, 0]
        out[:, 3] = arr[:, 3]
        return out
    raise ValueError(f"Unknown entity branch: {entity}")


def build_sr_history(triples: Sequence[Triple]) -> History:
    hist: History = {}
    for s, r, o, t in triples:
        hist.setdefault((int(s), int(r)), {}).setdefault(int(o), []).append(int(t))
    for cmap in hist.values():
        for times in cmap.values():
            times.sort()
    return hist


def build_so_history(triples: Sequence[Triple]) -> PairHistory:
    hist: PairHistory = {}
    for s, _r, o, t in triples:
        hist.setdefault(int(s), {}).setdefault(int(o), []).append(int(t))
    for cmap in hist.values():
        for times in cmap.values():
            times.sort()
    return hist


def build_ro_history(triples: Sequence[Triple]) -> PairHistory:
    hist: PairHistory = {}
    for _s, r, o, t in triples:
        hist.setdefault(int(r), {}).setdefault(int(o), []).append(int(t))
    for cmap in hist.values():
        for times in cmap.values():
            times.sort()
    return hist


def build_histories(triples: Sequence[Triple]) -> Dict[str, object]:
    return {
        "sr": build_sr_history(triples),
        "so": build_so_history(triples),
        "ro": build_ro_history(triples),
    }


def last_time_before(times: Sequence[int], t: int) -> Optional[int]:
    idx = bisect.bisect_left(times, int(t)) - 1
    if idx < 0:
        return None
    return int(times[idx])


def freq_before(times: Sequence[int], t: int) -> int:
    return int(bisect.bisect_left(times, int(t)))


def novelty_bucket_from_history(s: int, r: int, o: int, t: int, sr_hist: History, so_hist: PairHistory, ro_hist: PairHistory) -> str:
    if last_time_before(sr_hist.get((int(s), int(r)), {}).get(int(o), []), int(t)) is not None:
        return "repeat"
    near_so = last_time_before(so_hist.get(int(s), {}).get(int(o), []), int(t)) is not None
    near_ro = last_time_before(ro_hist.get(int(r), {}).get(int(o), []), int(t)) is not None
    if near_so or near_ro:
        return "near_repeat"
    return "novel"


def stale_exact_bucket(s: int, r: int, o: int, t: int, sr_hist: History, stale_threshold: int = 10) -> str:
    lt = last_time_before(sr_hist.get((int(s), int(r)), {}).get(int(o), []), int(t))
    if lt is None:
        return "novel"
    gap = int(t) - lt
    if gap > stale_threshold:
        return "stale"
    if gap <= 1:
        return "recent"
    return "mid"


def build_filter_map_from_triples(triples: Sequence[Triple]) -> Dict[Tuple[int, int, int], set]:
    """Map (s,r,t) -> set of true object entities for filtered object ranking."""
    fmap: Dict[Tuple[int, int, int], set] = {}
    for s, r, o, t in triples:
        fmap.setdefault((int(s), int(r), int(t)), set()).add(int(o))
    return fmap


def score_only_topk(base_scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Leakage-safe top-k: candidate membership depends only on model scores.

    Never pass or insert the gold/target entity here. This is the core fix for
    HVA/RHVC candidate selection.
    """
    if base_scores.ndim != 2:
        raise ValueError(f"base_scores must be [batch, num_entities], got {tuple(base_scores.shape)}")
    if int(topk) <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    k = min(int(topk), int(base_scores.size(1)))
    return torch.topk(base_scores.detach(), k=k, dim=1).indices


def scatter_topk_back(full_scores: torch.Tensor, candidate_ids: torch.Tensor, adjusted_topk_scores: torch.Tensor) -> torch.Tensor:
    out = full_scores.clone()
    out.scatter_(1, candidate_ids, adjusted_topk_scores)
    return out


def load_score_dump(path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    obj = np.load(path)
    if "scores" not in obj or "triples" not in obj:
        raise KeyError(f"Score dump must contain 'scores' and 'triples': {path}")
    scores = obj["scores"].astype(np.float32)
    triples = obj["triples"].astype(np.int64)
    entity = "object"
    if "entity" in obj:
        entity_arr = obj["entity"]
        if len(entity_arr) > 0:
            entity = str(entity_arr[0])
    return scores, triples, entity


def split_by_time_fraction(scores: np.ndarray, triples: np.ndarray, dev_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/dev split used for adapter tuning on validation dumps."""
    if not (0.0 <= dev_frac < 1.0):
        raise ValueError("dev_frac must be in [0,1)")
    n = len(triples)
    if n == 0 or dev_frac == 0.0:
        return scores, triples, np.empty((0,) + scores.shape[1:], dtype=scores.dtype), np.empty((0, 4), dtype=np.int64)
    order = np.argsort(triples[:, 3], kind="stable")
    cut = max(1, int(round(n * (1.0 - dev_frac))))
    cut = min(cut, n)
    train_idx = order[:cut]
    dev_idx = order[cut:]
    return scores[train_idx], triples[train_idx], scores[dev_idx], triples[dev_idx]
