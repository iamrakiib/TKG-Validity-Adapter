import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from evolution import calc_filtered_mrr, calc_filtered_test_mrr
from history_validity_gate import (
    augment_with_inverse,
    build_ro_history,
    build_so_history,
    build_sr_history,
    build_topk_candidate_ids,
    build_topk_history_features_dual,
    canonicalize_cygnet_queries,
    novelty_bucket_from_history,
    read_triples,
    scatter_topk_back,
    stale_exact_bucket,
    triples_array_to_list,
)

Triple = Tuple[int, int, int, int]


def inverse_softplus(x: float) -> float:
    x = float(x)
    if x <= 0:
        raise ValueError("inverse_softplus expects a positive value")
    return math.log(math.expm1(x))


def safe_div(x: float, y: float) -> float:
    return 0.0 if y == 0 else x / y


def load_dump(path: str):
    obj = np.load(path)
    scores = obj["scores"].astype(np.float32)
    triples = obj["triples"].astype(np.int64)
    entity = "object"
    if "entity" in obj:
        entity_arr = obj["entity"]
        if len(entity_arr) > 0:
            entity = str(entity_arr[0])
    return scores, triples, entity


def build_filter_map_from_arrays(arrays: Sequence[np.ndarray], num_r: int) -> Dict[Tuple[int, int], set]:
    triples: List[Triple] = []
    for arr in arrays:
        if arr is None or len(arr) == 0:
            continue
        triples.extend(triples_array_to_list(arr))
    aug = augment_with_inverse(triples, num_r)
    out: Dict[Tuple[int, int], set] = {}
    for s, r, o, _ in aug:
        out.setdefault((s, r), set()).add(o)
    return out


def split_dump_by_time_fraction(
    scores_np: np.ndarray,
    triples_np: np.ndarray,
    dev_frac: float = 0.2,
):
    if dev_frac <= 0.0:
        return {
            "train_scores": scores_np,
            "train_triples": triples_np,
            "dev_scores": None,
            "dev_triples": None,
            "train_times": sorted({int(t) for t in triples_np[:, 3].tolist()}),
            "dev_times": [],
        }

    ordered_times = []
    seen = set()
    for t in triples_np[:, 3].tolist():
        t = int(t)
        if t not in seen:
            ordered_times.append(t)
            seen.add(t)

    if len(ordered_times) < 2:
        return {
            "train_scores": scores_np,
            "train_triples": triples_np,
            "dev_scores": None,
            "dev_triples": None,
            "train_times": ordered_times,
            "dev_times": [],
        }

    num_dev = max(1, int(round(len(ordered_times) * dev_frac)))
    num_dev = min(num_dev, len(ordered_times) - 1)

    train_times = ordered_times[:-num_dev]
    dev_times = ordered_times[-num_dev:]

    train_mask = np.isin(triples_np[:, 3], np.asarray(train_times, dtype=np.int64))
    dev_mask = np.isin(triples_np[:, 3], np.asarray(dev_times, dtype=np.int64))

    return {
        "train_scores": scores_np[train_mask],
        "train_triples": triples_np[train_mask],
        "dev_scores": scores_np[dev_mask] if np.any(dev_mask) else None,
        "dev_triples": triples_np[dev_mask] if np.any(dev_mask) else None,
        "train_times": train_times,
        "dev_times": dev_times,
    }


def filtered_rank_from_scores(
    scores_row: np.ndarray,
    query_s: int,
    query_r: int,
    gold_o: int,
    filter_map: Dict[Tuple[int, int], set],
) -> int:
    gold_score = float(scores_row[gold_o])
    better = int(np.sum(scores_row > gold_score))

    disallowed = filter_map.get((query_s, query_r), set())
    if disallowed:
        disallowed = [idx for idx in disallowed if idx != gold_o]
        if disallowed:
            better -= int(np.sum(scores_row[np.asarray(disallowed, dtype=np.int64)] > gold_score))

    return better + 1


def finalize_bucket_stats(bucket_stats):
    out = {}
    for k, v in bucket_stats.items():
        c = max(v["count"], 1)
        out[k] = {
            "count": int(v["count"]),
            "MRR": safe_div(v["MRR"], c),
            "Hits@1": safe_div(v["Hits@1"], c),
            "Hits@3": safe_div(v["Hits@3"], c),
            "Hits@10": safe_div(v["Hits@10"], c),
        }
    return out


def aggregate_native_filtered_metrics(
    scores_np: np.ndarray,
    triples_np: np.ndarray,
    entity: str,
    num_e: int,
    train_data: np.ndarray,
    valid_data: np.ndarray,
    test_data: np.ndarray,
    split: str,
    batch_size: int = 128,
):
    total_mrr = 0.0
    total_hits1 = 0.0
    total_hits3 = 0.0
    total_hits10 = 0.0
    total_count = 0

    train_tensor = torch.LongTensor(train_data)
    valid_tensor = torch.LongTensor(valid_data)
    test_tensor = torch.LongTensor(test_data)

    n = len(triples_np)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_scores = torch.tensor(scores_np[start:end], dtype=torch.float32)
        batch_triples = torch.LongTensor(triples_np[start:end])

        if split == "valid":
            mrr, h1, h3, h10 = calc_filtered_mrr(
                num_e,
                batch_scores,
                train_tensor,
                valid_tensor,
                batch_triples,
                entity=entity,
                hits=[1, 3, 10],
            )
        elif split == "test":
            mrr, h1, h3, h10 = calc_filtered_test_mrr(
                num_e,
                batch_scores,
                train_tensor,
                valid_tensor,
                test_tensor,
                batch_triples,
                entity=entity,
                hits=[1, 3, 10],
            )
        else:
            raise ValueError(f"Unsupported split: {split}")

        batch_n = end - start
        total_mrr += mrr * batch_n
        total_hits1 += h1 * batch_n
        total_hits3 += h3 * batch_n
        total_hits10 += h10 * batch_n
        total_count += batch_n

    return {
        "count": int(total_count),
        "MRR": safe_div(total_mrr, total_count),
        "Hits@1": safe_div(total_hits1, total_count),
        "Hits@3": safe_div(total_hits3, total_count),
        "Hits@10": safe_div(total_hits10, total_count),
    }


def evaluate_bucket_metrics_filtered(
    scores_np,
    triples_np,
    entity,
    num_r,
    filter_map,
    sr_hist,
    so_hist,
    ro_hist,
):
    canonical_queries = canonicalize_cygnet_queries(triples_np, entity=entity, num_rels=num_r)

    bucket_stats = {
        "repeat": {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0},
        "near_repeat": {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0},
        "novel": {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0},
    }

    for i in range(len(triples_np)):
        s, r, o, t = map(int, canonical_queries[i][:4])
        rank = filtered_rank_from_scores(scores_np[i], s, r, o, filter_map)

        mrr = 1.0 / rank
        h1 = 1.0 if rank <= 1 else 0.0
        h3 = 1.0 if rank <= 3 else 0.0
        h10 = 1.0 if rank <= 10 else 0.0

        bucket = novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist)
        bucket_stats[bucket]["MRR"] += mrr
        bucket_stats[bucket]["Hits@1"] += h1
        bucket_stats[bucket]["Hits@3"] += h3
        bucket_stats[bucket]["Hits@10"] += h10
        bucket_stats[bucket]["count"] += 1

    return finalize_bucket_stats(bucket_stats)


def stale_top1_interference_from_scores(
    scores_np: np.ndarray,
    triples_np: np.ndarray,
    entity: str,
    num_r: int,
    sr_hist,
    so_hist,
    ro_hist,
):
    canonical_queries = canonicalize_cygnet_queries(triples_np, entity=entity, num_rels=num_r)
    top1 = np.argmax(scores_np, axis=1)

    total = 0
    stale_count = 0

    for i in range(len(triples_np)):
        s, r, o, t = map(int, canonical_queries[i][:4])
        true_bucket = novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist)

        if true_bucket in {"near_repeat", "novel"}:
            total += 1
            pred_o = int(top1[i])
            pred_bucket = stale_exact_bucket(s, r, pred_o, t, sr_hist)
            if pred_bucket == "stale":
                stale_count += 1

    return {
        "count": int(total),
        "stale_top1_count": int(stale_count),
        "stale_top1_rate": safe_div(stale_count, total),
    }


class RelationHistoryValidityCalibrator(nn.Module):
    """
    Stronger post-hoc relation-aware history-validity calibrator.
    Prototype / comparison only.
    Modes:
        - exact_only
        - dual_branch
    """

    def __init__(
        self,
        num_relations: int,
        mode: str = "dual_branch",
        rel_emb_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.10,
        init_gamma_exact: float = 0.02,
        init_gamma_near: float = 0.10,
        stale_init: float = 0.40,
        init_base_scale: float = 1.0,
        max_bias: float = 2.5,
        use_score_mlp: bool = True,
        use_uncertainty_gate: bool = True,
    ):
        super().__init__()
        assert mode in {"exact_only", "dual_branch"}

        self.mode = mode
        self.max_bias = float(max_bias)
        self.use_score_mlp = bool(use_score_mlp)
        self.use_uncertainty_gate = bool(use_uncertainty_gate)

        self.rel_lambda_sr = nn.Embedding(num_relations, 1)
        self.rel_w_rec_sr = nn.Embedding(num_relations, 1)
        self.rel_w_freq_sr = nn.Embedding(num_relations, 1)
        self.rel_w_recent_sr = nn.Embedding(num_relations, 1)
        self.rel_w_mid_sr = nn.Embedding(num_relations, 1)
        self.rel_w_stale_sr = nn.Embedding(num_relations, 1)
        self.rel_bias_sr = nn.Embedding(num_relations, 1)

        self.rel_lambda_so = nn.Embedding(num_relations, 1)
        self.rel_w_rec_so = nn.Embedding(num_relations, 1)
        self.rel_w_freq_so = nn.Embedding(num_relations, 1)
        self.rel_w_presence_so = nn.Embedding(num_relations, 1)
        self.rel_bias_so = nn.Embedding(num_relations, 1)

        self.rel_lambda_ro = nn.Embedding(num_relations, 1)
        self.rel_w_rec_ro = nn.Embedding(num_relations, 1)
        self.rel_w_freq_ro = nn.Embedding(num_relations, 1)
        self.rel_w_presence_ro = nn.Embedding(num_relations, 1)
        self.rel_bias_ro = nn.Embedding(num_relations, 1)

        self.rel_context = nn.Embedding(num_relations, rel_emb_dim)

        self.gamma_exact_raw = nn.Parameter(torch.tensor(inverse_softplus(init_gamma_exact), dtype=torch.float32))
        self.gamma_near_raw = nn.Parameter(torch.tensor(inverse_softplus(init_gamma_near), dtype=torch.float32))
        self.base_scale_raw = nn.Parameter(torch.tensor(inverse_softplus(init_base_scale), dtype=torch.float32))

        feature_dim = 17
        gate_dim = 5

        if self.use_score_mlp:
            self.score_mlp = nn.Sequential(
                nn.Linear(feature_dim + rel_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.score_mlp = None

        if self.use_uncertainty_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_dim + rel_emb_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.gate_mlp = None

        for emb in [self.rel_lambda_sr, self.rel_lambda_so, self.rel_lambda_ro]:
            nn.init.constant_(emb.weight, 0.05)

        for emb in [self.rel_w_rec_sr, self.rel_w_rec_so, self.rel_w_rec_ro]:
            nn.init.constant_(emb.weight, 1.0)

        for emb in [self.rel_w_freq_sr, self.rel_w_freq_so, self.rel_w_freq_ro]:
            nn.init.constant_(emb.weight, 0.25)

        nn.init.constant_(self.rel_w_recent_sr.weight, 0.45)
        nn.init.constant_(self.rel_w_mid_sr.weight, 0.12)
        nn.init.constant_(self.rel_w_stale_sr.weight, float(stale_init))

        nn.init.constant_(self.rel_w_presence_so.weight, 0.18)
        nn.init.constant_(self.rel_w_presence_ro.weight, 0.18)

        for emb in [self.rel_bias_sr, self.rel_bias_so, self.rel_bias_ro]:
            nn.init.constant_(emb.weight, 0.0)

        nn.init.normal_(self.rel_context.weight, mean=0.0, std=0.02)

    def _normalize_freq(self, freq, seen):
        freq_feat = torch.log1p(torch.clamp(freq, min=0.0))
        denom = freq_feat.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        return (freq_feat / denom) * seen

    def _score_context(self, base_scores):
        mean = base_scores.mean(dim=1, keepdim=True)
        std = base_scores.std(dim=1, keepdim=True).clamp_min(1e-6)
        centered = (base_scores - mean) / std

        top1 = base_scores.max(dim=1, keepdim=True).values
        gap_from_top1 = top1 - base_scores

        order = torch.argsort(base_scores, dim=1, descending=True)
        rank = torch.argsort(order, dim=1).float()
        denom = max(base_scores.size(1) - 1, 1)
        rank_norm = rank / denom
        return centered, gap_from_top1, rank_norm

    def _bucket_flags(self, seen, dt):
        recent = ((seen > 0) & (dt <= 1)).float()
        mid = ((seen > 0) & (dt > 1) & (dt <= 10)).float()
        stale = ((seen > 0) & (dt > 10)).float()
        return recent, mid, stale

    def _branch_exact(self, rel_ids, seen, dt, freq):
        lam = F.softplus(self.rel_lambda_sr(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = self.rel_w_rec_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = self.rel_w_freq_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wrecent = self.rel_w_recent_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wmid = self.rel_w_mid_sr(rel_ids).squeeze(-1).unsqueeze(1)
        wstale = self.rel_w_stale_sr(rel_ids).squeeze(-1).unsqueeze(1)
        b = self.rel_bias_sr(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        freq_feat = self._normalize_freq(freq, seen)
        recent, mid, stale = self._bucket_flags(seen, dt)

        score = (
            wrec * rec
            + wfreq * freq_feat
            + wrecent * recent
            + 0.5 * wmid * mid
            - wstale * stale
            + b
        )
        return torch.tanh(score) * seen, rec, freq_feat, recent, mid, stale

    def _branch_near(self, rel_ids, seen, dt, freq, emb_lambda, emb_wrec, emb_wfreq, emb_wpresence, emb_bias):
        lam = F.softplus(emb_lambda(rel_ids)).squeeze(-1).unsqueeze(1)
        wrec = emb_wrec(rel_ids).squeeze(-1).unsqueeze(1)
        wfreq = emb_wfreq(rel_ids).squeeze(-1).unsqueeze(1)
        wpres = emb_wpresence(rel_ids).squeeze(-1).unsqueeze(1)
        b = emb_bias(rel_ids).squeeze(-1).unsqueeze(1)

        dt_feat = torch.log1p(torch.clamp(dt, min=0.0))
        rec = torch.exp(-lam * dt_feat) * seen
        freq_feat = self._normalize_freq(freq, seen)

        score = wrec * rec + wfreq * freq_feat + wpres * seen + b
        return torch.tanh(score) * seen, rec, freq_feat

    def forward(
        self,
        base_scores,
        rel_ids,
        seen_sr, dt_sr, freq_sr,
        seen_so, dt_so, freq_so,
        seen_ro, dt_ro, freq_ro,
    ):
        g_sr, rec_sr, freq_sr_norm, recent_sr, mid_sr, stale_sr = self._branch_exact(
            rel_ids, seen_sr, dt_sr, freq_sr
        )

        if self.mode == "exact_only":
            g_so = torch.zeros_like(g_sr)
            g_ro = torch.zeros_like(g_sr)
            rec_so = torch.zeros_like(g_sr)
            rec_ro = torch.zeros_like(g_sr)
            freq_so_norm = torch.zeros_like(g_sr)
            freq_ro_norm = torch.zeros_like(g_sr)
            near_presence = torch.zeros_like(g_sr)
            branch_raw = F.softplus(self.gamma_exact_raw) * g_sr
        else:
            g_so, rec_so, freq_so_norm = self._branch_near(
                rel_ids,
                seen_so, dt_so, freq_so,
                self.rel_lambda_so, self.rel_w_rec_so, self.rel_w_freq_so, self.rel_w_presence_so, self.rel_bias_so
            )
            g_ro, rec_ro, freq_ro_norm = self._branch_near(
                rel_ids,
                seen_ro, dt_ro, freq_ro,
                self.rel_lambda_ro, self.rel_w_rec_ro, self.rel_w_freq_ro, self.rel_w_presence_ro, self.rel_bias_ro
            )
            near_presence = torch.clamp(seen_so + seen_ro, min=0.0, max=2.0) / 2.0
            branch_raw = (
                F.softplus(self.gamma_exact_raw) * g_sr
                + F.softplus(self.gamma_near_raw) * 0.5 * (g_so + g_ro)
            )

        centered_score, gap_from_top1, rank_norm = self._score_context(base_scores)
        hist_presence = torch.clamp(seen_sr + 0.5 * (seen_so + seen_ro), min=0.0, max=1.0)

        rel_ctx = self.rel_context(rel_ids).unsqueeze(1).expand(-1, base_scores.size(1), -1)

        feature_stack = torch.stack(
            [
                base_scores,
                centered_score,
                gap_from_top1,
                rank_norm,
                seen_sr,
                rec_sr,
                freq_sr_norm,
                recent_sr,
                mid_sr,
                stale_sr,
                near_presence,
                seen_so,
                rec_so,
                freq_so_norm,
                seen_ro,
                rec_ro,
                freq_ro_norm,
            ],
            dim=-1,
        )

        if self.score_mlp is not None:
            score_delta = self.score_mlp(torch.cat([feature_stack, rel_ctx], dim=-1)).squeeze(-1)
        else:
            score_delta = torch.zeros_like(base_scores)

        combined_raw = branch_raw + score_delta

        if self.gate_mlp is not None:
            gate_features = torch.stack(
                [
                    centered_score,
                    gap_from_top1,
                    rank_norm,
                    hist_presence,
                    stale_sr,
                ],
                dim=-1,
            )
            gate = torch.sigmoid(self.gate_mlp(torch.cat([gate_features, rel_ctx], dim=-1)).squeeze(-1))
        else:
            gate = torch.ones_like(base_scores)

        hist_bias = self.max_bias * torch.tanh(gate * combined_raw)
        base_scale = F.softplus(self.base_scale_raw) + 1e-4
        logits = base_scale * base_scores + hist_bias

        return logits, hist_bias


@torch.no_grad()
def apply_calibrator_to_scores(
    model: RelationHistoryValidityCalibrator,
    scores_np: np.ndarray,
    triples_np: np.ndarray,
    entity: str,
    num_r: int,
    sr_hist,
    so_hist,
    ro_hist,
    device,
    batch_size: int = 128,
    topk_cands: int = 256,
):
    model.eval()
    out_batches = []
    canonical_queries_np = canonicalize_cygnet_queries(triples_np, entity=entity, num_rels=num_r)

    n = len(triples_np)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        batch_scores = torch.tensor(scores_np[start:end], dtype=torch.float32, device=device)
        batch_queries_np = canonical_queries_np[start:end]
        batch_queries = torch.tensor(batch_queries_np, dtype=torch.long, device=device)

        rel_ids = batch_queries[:, 1]
        gold_ids = batch_queries[:, 2]

        candidate_ids = build_topk_candidate_ids(batch_scores, gold_ids, topk_cands)
        base_scores_topk = torch.gather(batch_scores, 1, candidate_ids)

        (
            seen_sr,
            dt_sr,
            freq_sr,
            seen_so,
            dt_so,
            freq_so,
            seen_ro,
            dt_ro,
            freq_ro,
        ) = build_topk_history_features_dual(
            query_triples=batch_queries_np,
            candidate_ids=candidate_ids,
            sr_hist=sr_hist,
            so_hist=so_hist,
            ro_hist=ro_hist,
            device=device,
            mode=model.mode,
        )

        adjusted_topk_scores, _ = model(
            base_scores_topk,
            rel_ids,
            seen_sr, dt_sr, freq_sr,
            seen_so, dt_so, freq_so,
            seen_ro, dt_ro, freq_ro,
        )
        adjusted_scores = scatter_topk_back(batch_scores, candidate_ids, adjusted_topk_scores)
        out_batches.append(adjusted_scores.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(out_batches, axis=0)


@torch.no_grad()
def evaluate_model_filtered(
    model: RelationHistoryValidityCalibrator,
    scores_np: np.ndarray,
    triples_np: np.ndarray,
    entity: str,
    num_r: int,
    num_e: int,
    train_data: np.ndarray,
    valid_data: np.ndarray,
    test_data: np.ndarray,
    split: str,
    filter_map,
    sr_hist,
    so_hist,
    ro_hist,
    device,
    batch_size: int = 128,
    topk_cands: int = 256,
):
    adjusted_scores_np = apply_calibrator_to_scores(
        model=model,
        scores_np=scores_np,
        triples_np=triples_np,
        entity=entity,
        num_r=num_r,
        sr_hist=sr_hist,
        so_hist=so_hist,
        ro_hist=ro_hist,
        device=device,
        batch_size=batch_size,
        topk_cands=topk_cands,
    )

    overall = aggregate_native_filtered_metrics(
        scores_np=adjusted_scores_np,
        triples_np=triples_np,
        entity=entity,
        num_e=num_e,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        split=split,
        batch_size=batch_size,
    )
    bucket = evaluate_bucket_metrics_filtered(
        scores_np=adjusted_scores_np,
        triples_np=triples_np,
        entity=entity,
        num_r=num_r,
        filter_map=filter_map,
        sr_hist=sr_hist,
        so_hist=so_hist,
        ro_hist=ro_hist,
    )
    interference = stale_top1_interference_from_scores(
        scores_np=adjusted_scores_np,
        triples_np=triples_np,
        entity=entity,
        num_r=num_r,
        sr_hist=sr_hist,
        so_hist=so_hist,
        ro_hist=ro_hist,
    )
    return adjusted_scores_np, overall, bucket, interference


def compute_delta(base_obj, cal_obj):
    out = {}
    for k in base_obj.keys():
        if isinstance(base_obj[k], (int, float)) and isinstance(cal_obj.get(k, None), (int, float)):
            out[k] = cal_obj[k] - base_obj[k]
    return out


def average_branch_results(obj_result: Dict, sub_result: Dict) -> Dict:
    combined = {
        "overall_filtered": {
            "count": obj_result["overall_filtered"]["count"] + sub_result["overall_filtered"]["count"],
            "MRR": 0.5 * (obj_result["overall_filtered"]["MRR"] + sub_result["overall_filtered"]["MRR"]),
            "Hits@1": 0.5 * (obj_result["overall_filtered"]["Hits@1"] + sub_result["overall_filtered"]["Hits@1"]),
            "Hits@3": 0.5 * (obj_result["overall_filtered"]["Hits@3"] + sub_result["overall_filtered"]["Hits@3"]),
            "Hits@10": 0.5 * (obj_result["overall_filtered"]["Hits@10"] + sub_result["overall_filtered"]["Hits@10"]),
        },
        "bucket_metrics_filtered": {},
        "stale_top1_interference": {
            "count": obj_result["stale_top1_interference"]["count"] + sub_result["stale_top1_interference"]["count"],
            "stale_top1_count": obj_result["stale_top1_interference"]["stale_top1_count"] + sub_result["stale_top1_interference"]["stale_top1_count"],
            "stale_top1_rate": 0.5 * (
                obj_result["stale_top1_interference"]["stale_top1_rate"]
                + sub_result["stale_top1_interference"]["stale_top1_rate"]
            ),
        },
    }

    all_buckets = set(obj_result["bucket_metrics_filtered"].keys()) | set(sub_result["bucket_metrics_filtered"].keys())
    for bucket in all_buckets:
        obj_bucket = obj_result["bucket_metrics_filtered"].get(bucket, {"count": 0, "MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0})
        sub_bucket = sub_result["bucket_metrics_filtered"].get(bucket, {"count": 0, "MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0})
        combined["bucket_metrics_filtered"][bucket] = {
            "count": int(obj_bucket["count"]) + int(sub_bucket["count"]),
            "MRR": 0.5 * (obj_bucket["MRR"] + sub_bucket["MRR"]),
            "Hits@1": 0.5 * (obj_bucket["Hits@1"] + sub_bucket["Hits@1"]),
            "Hits@3": 0.5 * (obj_bucket["Hits@3"] + sub_bucket["Hits@3"]),
            "Hits@10": 0.5 * (obj_bucket["Hits@10"] + sub_bucket["Hits@10"]),
        }

    return combined


def train_calibrator(
    model: RelationHistoryValidityCalibrator,
    train_scores_np: np.ndarray,
    train_triples_np: np.ndarray,
    entity: str,
    num_r: int,
    sr_hist,
    so_hist,
    ro_hist,
    device,
    dev_scores_np: np.ndarray = None,
    dev_triples_np: np.ndarray = None,
    dev_num_e: int = None,
    dev_train_data: np.ndarray = None,
    dev_valid_data: np.ndarray = None,
    dev_test_data: np.ndarray = None,
    dev_split: str = "valid",
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    topk_cands: int = 256,
    eval_topk_cands: int = 256,
    patience: int = 3,
    min_epochs: int = 4,
    pairwise_weight: float = 0.25,
    margin: float = 0.20,
    bias_reg: float = 1e-4,
    label_smoothing: float = 0.0,
    grad_norm: float = 1.0,
):
    model.train()

    dataset = TensorDataset(
        torch.tensor(train_scores_np, dtype=torch.float32),
        torch.tensor(train_triples_np, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_epoch = -1
    best_dev_mrr = -1.0
    wait = 0
    history = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_ce = 0.0
        total_pair = 0.0
        total_bias = 0.0
        total_count = 0

        for batch_scores_cpu, batch_triples_cpu in loader:
            full_scores = batch_scores_cpu.to(device)
            batch_triples_np = batch_triples_cpu.cpu().numpy()

            canonical_queries_np = canonicalize_cygnet_queries(
                batch_triples_np,
                entity=entity,
                num_rels=num_r,
            )
            canonical_queries = torch.tensor(canonical_queries_np, dtype=torch.long, device=device)

            rel_ids = canonical_queries[:, 1]
            gold_ids = canonical_queries[:, 2]

            candidate_ids = build_topk_candidate_ids(full_scores, gold_ids, topk_cands)
            base_scores_topk = torch.gather(full_scores, 1, candidate_ids)

            (
                seen_sr,
                dt_sr,
                freq_sr,
                seen_so,
                dt_so,
                freq_so,
                seen_ro,
                dt_ro,
                freq_ro,
            ) = build_topk_history_features_dual(
                query_triples=canonical_queries_np,
                candidate_ids=candidate_ids,
                sr_hist=sr_hist,
                so_hist=so_hist,
                ro_hist=ro_hist,
                device=device,
                mode=model.mode,
            )

            adjusted_topk_scores, hist_bias = model(
                base_scores_topk,
                rel_ids,
                seen_sr, dt_sr, freq_sr,
                seen_so, dt_so, freq_so,
                seen_ro, dt_ro, freq_ro,
            )

            local_gold = (candidate_ids == gold_ids.unsqueeze(1)).long().argmax(dim=1)

            ce_loss = F.cross_entropy(
                adjusted_topk_scores,
                local_gold,
                label_smoothing=label_smoothing,
            )

            gold_score = adjusted_topk_scores.gather(1, local_gold.unsqueeze(1)).squeeze(1)
            neg_mask = candidate_ids == gold_ids.unsqueeze(1)
            hardest_neg = adjusted_topk_scores.masked_fill(neg_mask, float("-inf")).max(dim=1).values
            pairwise_loss = F.relu(margin - (gold_score - hardest_neg)).mean()

            bias_penalty = hist_bias.pow(2).mean()
            loss = ce_loss + pairwise_weight * pairwise_loss + bias_reg * bias_penalty

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            opt.step()

            batch_n = full_scores.size(0)
            total_loss += float(loss.item()) * batch_n
            total_ce += float(ce_loss.item()) * batch_n
            total_pair += float(pairwise_loss.item()) * batch_n
            total_bias += float(bias_penalty.item()) * batch_n
            total_count += batch_n

        train_loss = safe_div(total_loss, total_count)
        train_ce = safe_div(total_ce, total_count)
        train_pair = safe_div(total_pair, total_count)
        train_bias = safe_div(total_bias, total_count)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_ce": train_ce,
            "train_pairwise": train_pair,
            "train_bias_penalty": train_bias,
            "dev_mrr": None,
        }

        if (
            dev_scores_np is not None
            and dev_triples_np is not None
            and dev_num_e is not None
            and dev_train_data is not None
            and dev_valid_data is not None
            and dev_test_data is not None
        ):
            _, dev_overall, _, _ = evaluate_model_filtered(
                model=model,
                scores_np=dev_scores_np,
                triples_np=dev_triples_np,
                entity=entity,
                num_r=num_r,
                num_e=dev_num_e,
                train_data=dev_train_data,
                valid_data=dev_valid_data,
                test_data=dev_test_data,
                split=dev_split,
                filter_map=build_filter_map_from_arrays(
                    [dev_train_data, dev_valid_data] if dev_split == "valid" else [dev_train_data, dev_valid_data, dev_test_data],
                    num_r,
                ),
                sr_hist=sr_hist,
                so_hist=so_hist,
                ro_hist=ro_hist,
                device=device,
                batch_size=batch_size,
                topk_cands=eval_topk_cands,
            )
            dev_mrr = float(dev_overall["MRR"])
            epoch_info["dev_mrr"] = dev_mrr

            if dev_mrr > best_dev_mrr:
                best_dev_mrr = dev_mrr
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if (epoch + 1) >= min_epochs and wait >= patience:
                history.append(epoch_info)
                break

        history.append(epoch_info)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_epoch": best_epoch,
        "best_dev_mrr": best_dev_mrr,
        "history": history,
    }