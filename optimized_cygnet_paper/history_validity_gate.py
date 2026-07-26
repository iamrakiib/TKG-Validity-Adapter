import bisect
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Triple = Tuple[int, int, int, int]
ArrayLike = Union[np.ndarray, torch.Tensor, List[List[int]], List[Tuple[int, int, int, int]]]


def triples_array_to_list(array_like: ArrayLike) -> List[Triple]:
    arr = np.asarray(array_like)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected shape [N, >=4], got {arr.shape}")
    return [tuple(map(int, row[:4])) for row in arr]


def read_triples(path: str) -> List[Triple]:
    triples: List[Triple] = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            s, r, o, t = map(int, parts[:4])
            triples.append((s, r, o, t))
    return triples


def augment_with_inverse(triples: List[Triple], num_rels: int) -> List[Triple]:
    aug: List[Triple] = []
    for s, r, o, t in triples:
        aug.append((s, r, o, t))
        aug.append((o, r + num_rels, s, t))
    return aug


def canonicalize_cygnet_queries(
    query_triples: ArrayLike,
    entity: str,
    num_rels: int,
) -> np.ndarray:
    """
    Convert CyGNet branch queries into a single object-prediction-style canonical form.

    object branch:
        (s, r, o, t) -> (s, r, o, t)

    subject branch:
        CyGNet predicts subject from the original quadruple, but for shared
        history-validity logic we canonicalize it as inverse-form:
        (o, r + num_rels, s, t)

    This lets the same exact/near history logic work for both branches.
    """
    arr = np.asarray(query_triples)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected shape [N, >=4], got {arr.shape}")

    out = arr[:, :4].astype(np.int64, copy=True)

    if entity == "object":
        return out

    if entity == "subject":
        s = out[:, 0].copy()
        r = out[:, 1].copy()
        o = out[:, 2].copy()
        t = out[:, 3].copy()

        out[:, 0] = o
        out[:, 1] = r + int(num_rels)
        out[:, 2] = s
        out[:, 3] = t
        return out

    raise ValueError(f"Unknown entity branch: {entity}")


def build_sr_history(triples: List[Triple]) -> Dict[Tuple[int, int], Dict[int, List[int]]]:
    sr_hist: Dict[Tuple[int, int], Dict[int, List[int]]] = {}
    for s, r, o, t in triples:
        sr_hist.setdefault((s, r), {}).setdefault(o, []).append(t)

    for sr_key in sr_hist:
        for obj in sr_hist[sr_key]:
            sr_hist[sr_key][obj].sort()

    return sr_hist


def build_so_history(triples: List[Triple]) -> Dict[int, Dict[int, List[int]]]:
    so_hist: Dict[int, Dict[int, List[int]]] = {}
    for s, r, o, t in triples:
        so_hist.setdefault(s, {}).setdefault(o, []).append(t)

    for s in so_hist:
        for obj in so_hist[s]:
            so_hist[s][obj].sort()

    return so_hist


def build_ro_history(triples: List[Triple]) -> Dict[int, Dict[int, List[int]]]:
    ro_hist: Dict[int, Dict[int, List[int]]] = {}
    for s, r, o, t in triples:
        ro_hist.setdefault(r, {}).setdefault(o, []).append(t)

    for r in ro_hist:
        for obj in ro_hist[r]:
            ro_hist[r][obj].sort()

    return ro_hist


def build_train_and_train_valid_histories(
    train_triples: List[Triple],
    valid_triples: List[Triple],
    num_rels: int,
):
    train_aug = augment_with_inverse(train_triples, num_rels)
    train_valid_aug = augment_with_inverse(train_triples + valid_triples, num_rels)

    train_hist = {
        "sr": build_sr_history(train_aug),
        "so": build_so_history(train_aug),
        "ro": build_ro_history(train_aug),
    }
    train_valid_hist = {
        "sr": build_sr_history(train_valid_aug),
        "so": build_so_history(train_valid_aug),
        "ro": build_ro_history(train_valid_aug),
    }
    return train_hist, train_valid_hist


def last_time_before(times: List[int], t: int):
    idx = bisect.bisect_left(times, t) - 1
    if idx < 0:
        return None
    return times[idx]


def freq_before(times: List[int], t: int) -> int:
    return bisect.bisect_left(times, t)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_device_float(np_array: np.ndarray, device: torch.device):
    return torch.from_numpy(np_array).to(device=device, dtype=torch.float32)


def build_topk_candidate_ids(
    base_scores: torch.Tensor,
    gold_ids: torch.Tensor = None,
    topk_cands: int = 100,
) -> torch.Tensor:
    """Evaluation-safe top-K candidate selection.

    Candidate membership must depend only on CyGNet backbone scores. The gold
    entity is intentionally ignored and is never inserted into the top-K set.
    This keeps candidate membership independent of validation or test targets.
    """
    if base_scores.ndim != 2:
        raise ValueError(f"base_scores must be [batch, num_entities], got {tuple(base_scores.shape)}")
    k = min(int(topk_cands), int(base_scores.size(1)))
    return torch.topk(base_scores.detach(), k=k, dim=1).indices


def scatter_topk_back(
    full_scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    adjusted_topk_scores: torch.Tensor,
) -> torch.Tensor:
    out = full_scores.clone()
    out.scatter_(1, candidate_ids, adjusted_topk_scores)
    return out


def build_topk_history_features_dual(
    query_triples,
    candidate_ids,
    sr_hist,
    so_hist,
    ro_hist,
    device,
    mode: str = "dual_branch",
):
    """
    Build exact/near history features for top-K candidates.

    query_triples must already be canonicalized into object-style form:
        (subject_like, relation_like, gold_candidate, time)

    mode:
        - exact_only
        - dual_branch
    """
    if mode not in {"exact_only", "dual_branch"}:
        raise ValueError(f"Unsupported HVA mode: {mode}")

    cand_np = _to_numpy(candidate_ids)
    query_np = _to_numpy(query_triples)

    batch_size, k = cand_np.shape

    seen_sr = np.zeros((batch_size, k), dtype=np.float32)
    dt_sr = np.zeros((batch_size, k), dtype=np.float32)
    freq_sr = np.zeros((batch_size, k), dtype=np.float32)

    seen_so = np.zeros((batch_size, k), dtype=np.float32)
    dt_so = np.zeros((batch_size, k), dtype=np.float32)
    freq_so = np.zeros((batch_size, k), dtype=np.float32)

    seen_ro = np.zeros((batch_size, k), dtype=np.float32)
    dt_ro = np.zeros((batch_size, k), dtype=np.float32)
    freq_ro = np.zeros((batch_size, k), dtype=np.float32)

    exact_only = (mode == "exact_only")

    for i in range(batch_size):
        s, r, _, t = map(int, query_np[i][:4])

        cand_map_sr = sr_hist.get((s, r), {})
        cand_map_so = {} if exact_only else so_hist.get(s, {})
        cand_map_ro = {} if exact_only else ro_hist.get(r, {})

        for j, cand_o in enumerate(cand_np[i]):
            cand_o = int(cand_o)

            times_sr = cand_map_sr.get(cand_o, [])
            if times_sr:
                lt = last_time_before(times_sr, t)
                if lt is not None:
                    seen_sr[i, j] = 1.0
                    dt_sr[i, j] = float(t - lt)
                    freq_sr[i, j] = float(freq_before(times_sr, t))

            if not exact_only:
                times_so = cand_map_so.get(cand_o, [])
                if times_so:
                    lt = last_time_before(times_so, t)
                    if lt is not None:
                        seen_so[i, j] = 1.0
                        dt_so[i, j] = float(t - lt)
                        freq_so[i, j] = float(freq_before(times_so, t))

                times_ro = cand_map_ro.get(cand_o, [])
                if times_ro:
                    lt = last_time_before(times_ro, t)
                    if lt is not None:
                        seen_ro[i, j] = 1.0
                        dt_ro[i, j] = float(t - lt)
                        freq_ro[i, j] = float(freq_before(times_ro, t))

    return (
        _to_device_float(seen_sr, device),
        _to_device_float(dt_sr, device),
        _to_device_float(freq_sr, device),
        _to_device_float(seen_so, device),
        _to_device_float(dt_so, device),
        _to_device_float(freq_so, device),
        _to_device_float(seen_ro, device),
        _to_device_float(dt_ro, device),
        _to_device_float(freq_ro, device),
    )


def novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist):
    times_sr = sr_hist.get((s, r), {}).get(o, [])
    lt_sr = last_time_before(times_sr, t)
    if lt_sr is not None:
        return "repeat"

    times_so = so_hist.get(s, {}).get(o, [])
    lt_so = last_time_before(times_so, t)

    times_ro = ro_hist.get(r, {}).get(o, [])
    lt_ro = last_time_before(times_ro, t)

    if lt_so is not None or lt_ro is not None:
        return "near_repeat"

    return "novel"


def stale_exact_bucket(s, r, o, t, sr_hist):
    times = sr_hist.get((s, r), {}).get(o, [])
    lt = last_time_before(times, t)
    if lt is None:
        return "novel"

    gap = t - lt
    if gap <= 1:
        return "recent"
    if gap <= 10:
        return "mid"
    return "stale"


class HistoryValidityAdapter(nn.Module):
    """
    End-to-end candidate-level history-validity adapter for CyGNet.

    This module operates on top-K candidate score slices in score/logit space.

    Inputs:
        - base_scores: [B, K]
        - rel_ids: [B]
        - exact branch: seen_sr, dt_sr, freq_sr
        - near branches: seen_so, dt_so, freq_so, seen_ro, dt_ro, freq_ro

    Modes:
        - exact_only
        - dual_branch
    """

    def __init__(
        self,
        num_relations: int,
        mode: str = "dual_branch",
        gamma_exact: float = 0.005,
        gamma_near: float = 0.08,
        stale_init: float = 0.2,
    ):
        super().__init__()
        if mode not in {"exact_only", "dual_branch"}:
            raise ValueError("mode must be one of {'exact_only', 'dual_branch'}")

        self.mode = mode
        self.register_buffer("gamma_exact", torch.tensor(float(gamma_exact), dtype=torch.float32))
        self.register_buffer("gamma_near", torch.tensor(float(gamma_near), dtype=torch.float32))

        self.rel_lambda_sr = nn.Embedding(num_relations, 1)
        self.rel_w_rec_sr = nn.Embedding(num_relations, 1)
        self.rel_w_freq_sr = nn.Embedding(num_relations, 1)
        self.rel_w_stale_sr = nn.Embedding(num_relations, 1)
        self.rel_bias_sr = nn.Embedding(num_relations, 1)

        self.rel_lambda_so = nn.Embedding(num_relations, 1)
        self.rel_w_rec_so = nn.Embedding(num_relations, 1)
        self.rel_w_freq_so = nn.Embedding(num_relations, 1)
        self.rel_bias_so = nn.Embedding(num_relations, 1)

        self.rel_lambda_ro = nn.Embedding(num_relations, 1)
        self.rel_w_rec_ro = nn.Embedding(num_relations, 1)
        self.rel_w_freq_ro = nn.Embedding(num_relations, 1)
        self.rel_bias_ro = nn.Embedding(num_relations, 1)

        for emb in [self.rel_lambda_sr, self.rel_lambda_so, self.rel_lambda_ro]:
            nn.init.constant_(emb.weight, 0.05)

        for emb in [self.rel_w_rec_sr, self.rel_w_rec_so, self.rel_w_rec_ro]:
            nn.init.constant_(emb.weight, 1.0)

        for emb in [self.rel_w_freq_sr, self.rel_w_freq_so, self.rel_w_freq_ro]:
            nn.init.constant_(emb.weight, 0.25)

        nn.init.constant_(self.rel_w_stale_sr.weight, float(stale_init))

        for emb in [self.rel_bias_sr, self.rel_bias_so, self.rel_bias_ro]:
            nn.init.constant_(emb.weight, 0.0)

    def _normalize_freq(self, freq, seen):
        freq_feat = torch.log1p(torch.clamp(freq, min=0.0))
        denom = freq_feat.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        return (freq_feat / denom) * seen

    def _branch_exact(self, rel_ids, seen, dt, freq):
        lam = F.softplus(self.rel_lambda_sr(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = self.rel_w_rec_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = self.rel_w_freq_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wstale = self.rel_w_stale_sr(rel_ids).squeeze(-1).unsqueeze(1)
        b = self.rel_bias_sr(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        stale = (1.0 - rec) * seen
        freq_feat = self._normalize_freq(freq, seen)

        score = wrec * rec + wfreq * freq_feat - wstale * stale + b
        return torch.tanh(score) * seen

    def _branch_near(self, rel_ids, seen, dt, freq, emb_lambda, emb_wrec, emb_wfreq, emb_bias):
        lam = F.softplus(emb_lambda(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = emb_wrec(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = emb_wfreq(rel_ids).squeeze(-1).unsqueeze(1)
        b = emb_bias(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        freq_feat = self._normalize_freq(freq, seen)

        score = wrec * rec + wfreq * freq_feat + b
        return torch.tanh(score) * seen

    def forward(
        self,
        base_scores,
        rel_ids,
        seen_sr, dt_sr, freq_sr,
        seen_so, dt_so, freq_so,
        seen_ro, dt_ro, freq_ro,
    ):
        g_sr = self._branch_exact(rel_ids, seen_sr, dt_sr, freq_sr)

        if self.mode == "exact_only":
            hist_bias = self.gamma_exact * g_sr
        else:
            g_so = self._branch_near(
                rel_ids,
                seen_so, dt_so, freq_so,
                self.rel_lambda_so, self.rel_w_rec_so, self.rel_w_freq_so, self.rel_bias_so,
            )
            g_ro = self._branch_near(
                rel_ids,
                seen_ro, dt_ro, freq_ro,
                self.rel_lambda_ro, self.rel_w_rec_ro, self.rel_w_freq_ro, self.rel_bias_ro,
            )
            hist_bias = self.gamma_exact * g_sr + self.gamma_near * 0.5 * (g_so + g_ro)

        adjusted_scores = base_scores + hist_bias
        return adjusted_scores, hist_bias
