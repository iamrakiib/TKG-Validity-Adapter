from __future__ import annotations

"""RHVC diagnostic prototype for TeMP score dumps.

This file implements the paper role of RHVC:
  - post-hoc diagnostic calibration prototype,
  - not the final HVA adapter,
  - candidate set selected only from backbone scores,
  - exact historical evidence only from t' < t,
  - no gold candidate insertion into top-K.

The correction is intentionally transparent:
  base_score + gamma * (occurrence + frequency + recency - stale penalty)
computed only on the backbone top-K candidates.
"""

import argparse
import itertools
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from hva.history_utils import (
    augment_with_inverse,
    build_histories,
    canonicalize_queries,
    ensure_dir,
    freq_before,
    last_time_before,
    load_score_dump,
    read_split_arrays,
    read_stat,
    save_json,
    score_only_topk,
    set_seed,
)
from hva.metrics import evaluate_diagnostics, evaluate_filtered


@dataclass(frozen=True)
class RHVCParams:
    gamma: float = 0.10
    w_occurrence: float = 1.00
    w_frequency: float = 0.15
    w_recency: float = 0.30
    w_stale: float = 0.30
    lambda_decay: float = 0.10
    stale_threshold: int = 10
    topk: int = 100


def _candidate_correction(times, query_t: int, p: RHVCParams) -> float:
    lt = last_time_before(times, int(query_t))
    if lt is None:
        return 0.0
    gap = max(0, int(query_t) - int(lt))
    freq = freq_before(times, int(query_t))
    correction = 0.0
    correction += float(p.w_occurrence)
    correction += float(p.w_frequency) * float(np.log1p(freq))
    correction += float(p.w_recency) * float(np.exp(-float(p.lambda_decay) * float(gap)))
    if gap > int(p.stale_threshold):
        correction -= float(p.w_stale)
    return float(correction)


def apply_rhvc(scores: np.ndarray, queries: np.ndarray, histories: Dict[str, object], p: RHVCParams) -> np.ndarray:
    """Apply RHVC only to top-K candidates selected from base scores.

    Gold labels are never used in candidate selection. The gold column in
    queries is ignored for feature construction by design.
    """
    out = scores.copy().astype(np.float32)
    sr = histories["sr"]
    scores_t = torch.tensor(scores, dtype=torch.float32)
    topk_ids = score_only_topk(scores_t, p.topk).cpu().numpy()

    for i, q in enumerate(queries):
        s, r, _gold, t = map(int, q[:4])
        exact_map = sr.get((s, r), {})
        for cand in topk_ids[i]:
            cand = int(cand)
            corr = _candidate_correction(exact_map.get(cand, []), t, p)
            if corr != 0.0:
                out[i, cand] += float(p.gamma) * corr
    return out


def _parse_float_grid(text: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _parse_int_grid(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def build_data_context(data_root: str, dataset: str, valid_dump: str, test_dump: str):
    data_dir = os.path.join(data_root, dataset)
    num_e, num_rels = read_stat(data_dir)
    arrays = read_split_arrays(data_dir)
    valid_scores_raw, valid_triples_raw, valid_entity = load_score_dump(valid_dump)
    test_scores_raw, test_triples_raw, test_entity = load_score_dump(test_dump)
    if valid_entity != test_entity:
        raise ValueError(f"valid entity={valid_entity}, test entity={test_entity}; use matching score branches")

    valid_queries = canonicalize_queries(valid_triples_raw, valid_entity, num_rels)
    test_queries = canonicalize_queries(test_triples_raw, test_entity, num_rels)

    train_aug = augment_with_inverse([tuple(x) for x in arrays["train"]], num_rels)
    train_valid_aug = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"]], axis=0)], num_rels)
    valid_histories = build_histories(train_aug)
    test_histories = build_histories(train_valid_aug)

    filter_valid = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"]], axis=0)], num_rels)
    filter_test = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"], arrays["test"]], axis=0)], num_rels)
    return {
        "num_entities": num_e,
        "num_relations": num_rels,
        "valid_scores": valid_scores_raw,
        "test_scores": test_scores_raw,
        "valid_queries": valid_queries,
        "test_queries": test_queries,
        "valid_histories": valid_histories,
        "test_histories": test_histories,
        "filter_valid": filter_valid,
        "filter_test": filter_test,
        "entity_branch": valid_entity,
    }


def main():
    p = argparse.ArgumentParser(description="RHVC post-hoc diagnostic prototype from TeMP score dumps")
    p.add_argument("--dataset", required=True, choices=["ICEWS14", "ICEWS18"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--valid-dump", required=True)
    p.add_argument("--test-dump", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--gamma-grid", default="0.05,0.10,0.20")
    p.add_argument("--frequency-grid", default="0.05,0.15,0.30")
    p.add_argument("--recency-grid", default="0.10,0.30,0.50")
    p.add_argument("--stale-grid", default="0.10,0.30,0.50")
    p.add_argument("--lambda-grid", default="0.05,0.10,0.20")
    p.add_argument("--stale-threshold-grid", default="5,10,20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-adjusted-scores", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ctx = build_data_context(args.data_root, args.dataset, args.valid_dump, args.test_dump)

    gamma_grid = _parse_float_grid(args.gamma_grid)
    freq_grid = _parse_float_grid(args.frequency_grid)
    rec_grid = _parse_float_grid(args.recency_grid)
    stale_grid = _parse_float_grid(args.stale_grid)
    lam_grid = _parse_float_grid(args.lambda_grid)
    stale_thr_grid = _parse_int_grid(args.stale_threshold_grid)

    trials = []
    best = None
    best_mrr = -1.0
    for gamma, wf, wr, ws, lam, thr in itertools.product(gamma_grid, freq_grid, rec_grid, stale_grid, lam_grid, stale_thr_grid):
        params = RHVCParams(
            gamma=gamma,
            w_frequency=wf,
            w_recency=wr,
            w_stale=ws,
            lambda_decay=lam,
            stale_threshold=thr,
            topk=args.topk,
        )
        valid_adjusted = apply_rhvc(ctx["valid_scores"], ctx["valid_queries"], ctx["valid_histories"], params)
        valid_overall = evaluate_filtered(valid_adjusted, ctx["valid_queries"], ctx["filter_valid"])
        record = {"params": asdict(params), "valid_MRR": valid_overall["MRR"], "valid_Hits@1": valid_overall["Hits@1"]}
        trials.append(record)
        if valid_overall["MRR"] > best_mrr:
            best_mrr = valid_overall["MRR"]
            best = params

    assert best is not None
    valid_adjusted = apply_rhvc(ctx["valid_scores"], ctx["valid_queries"], ctx["valid_histories"], best)
    test_adjusted = apply_rhvc(ctx["test_scores"], ctx["test_queries"], ctx["test_histories"], best)

    result = {
        "dataset": args.dataset,
        "method": "RHVC_post_hoc_diagnostic_prototype",
        "entity_branch": ctx["entity_branch"],
        "selected_params": asdict(best),
        "valid_overall": evaluate_filtered(valid_adjusted, ctx["valid_queries"], ctx["filter_valid"]),
        "valid_diagnostics": evaluate_diagnostics(valid_adjusted, ctx["valid_queries"], ctx["filter_valid"], ctx["valid_histories"], best.stale_threshold),
        "test_overall": evaluate_filtered(test_adjusted, ctx["test_queries"], ctx["filter_test"]),
        "test_diagnostics": evaluate_diagnostics(test_adjusted, ctx["test_queries"], ctx["filter_test"], ctx["test_histories"], best.stale_threshold),
        "validation_trials": trials,
        "leakage_control": {
            "role": "RHVC is a post-hoc diagnostic prototype, not the final HVA adapter.",
            "topk_candidate_selection": "torch.topk(base_scores.detach()) only; gold target is never inserted",
            "feature_construction": "exact historical evidence from t' < t only",
            "test_history": "test-time features use train+valid history only",
        },
    }
    save_json(result, os.path.join(args.out_dir, "rhvc_results.json"))
    if args.save_adjusted_scores:
        np.savez_compressed(os.path.join(args.out_dir, "valid_rhvc_scores.npz"), scores=valid_adjusted, triples=ctx["valid_queries"], entity=np.asarray(["object"]))
        np.savez_compressed(os.path.join(args.out_dir, "test_rhvc_scores.npz"), scores=test_adjusted, triples=ctx["test_queries"], entity=np.asarray(["object"]))
    print("==== RHVC TEST RESULT ====")
    print(result["test_overall"])
    print(result["test_diagnostics"])
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
