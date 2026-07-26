from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from data_utils import load_dataset, group_by_time, build_time_filter_map
from hri_recurrence import HRIHistory, HRIParams
from metrics import filtered_rank_from_sparse_scores, metrics_from_ranks, save_metrics


def topk_from_sparse(scores: Dict[int, float], num_entities: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return top-k candidate ids and scores from recurrence scores only.

    This is leakage-safe: the gold entity is not inserted. Unscored entities have
    score zero and are used only to fill remaining positions deterministically.
    """
    k = min(int(k), int(num_entities))
    items = sorted(((int(e), float(v)) for e, v in scores.items()), key=lambda x: (-x[1], x[0]))
    ids = [e for e, _ in items[:k]]
    vals = [v for _, v in items[:k]]
    if len(ids) < k:
        used = set(ids)
        for e in range(num_entities):
            if e not in used:
                ids.append(e)
                vals.append(0.0)
                if len(ids) == k:
                    break
    return np.asarray(ids, dtype=np.int64), np.asarray(vals, dtype=np.float64)


def evaluate_split(ds, split: str, params: HRIParams, out_dir: Path, tie_policy: str, export_topk: bool, topk: int, max_queries: int | None = None):
    if split == "valid":
        init_history = ds.train
        eval_quads = ds.valid
    elif split == "test":
        init_history = np.concatenate([ds.train, ds.valid], axis=0)
        eval_quads = ds.test
    else:
        raise ValueError("split must be valid or test")

    all_quads = np.concatenate([ds.train, ds.valid, ds.test], axis=0)
    filter_map = build_time_filter_map(all_quads)

    history = HRIHistory()
    history.update_many(init_history)

    ranks: List[float] = []
    rows: List[dict] = []
    diagnostic_hits = {"repeat": [], "near_repeat": [], "novel": []}
    stale_top1_flags: List[int] = []

    topk_ids_all = []
    topk_scores_all = []
    query_quads_all = []

    processed = 0
    for t, facts_t in group_by_time(eval_quads):
        # Evaluate all facts at this timestamp before adding same-timestamp facts.
        for s, r, o, qt in facts_t.tolist():
            if max_queries is not None and processed >= max_queries:
                break
            s, r, o, qt = int(s), int(r), int(o), int(qt)
            scores = history.score_candidates(s, r, qt, params)
            rank = filtered_rank_from_sparse_scores(
                scores=scores,
                gold=o,
                num_entities=ds.num_entities,
                filter_objects=filter_map.get((s, r, qt), set()),
                tie_policy=tie_policy,
            )
            ranks.append(rank)
            label = history.diagnostic_label(s, r, o)
            diagnostic_hits[label].append(1 if rank <= 1 else 0)

            # StaleTop1 diagnostic for HRI: top-ranked recurrence candidate is stale.
            top1_id, top1_score = topk_from_sparse(scores, ds.num_entities, 1)
            top1 = int(top1_id[0])
            stale_flag = 0
            if top1 in history.last_t.get((s, r), {}):
                last_t = history.last_t[(s, r)][top1]
                if params.stale_threshold >= 0 and (qt - last_t) > params.stale_threshold:
                    stale_flag = 1
            stale_top1_flags.append(stale_flag)

            rows.append({"s": s, "r": r, "o": o, "t": qt, "rank": rank, "label": label, "top1": top1, "top1_score": float(top1_score[0]), "stale_top1": stale_flag})

            if export_topk:
                ids, vals = topk_from_sparse(scores, ds.num_entities, topk)
                topk_ids_all.append(ids)
                topk_scores_all.append(vals)
                query_quads_all.append([s, r, o, qt])
            processed += 1
        if max_queries is not None and processed >= max_queries:
            break
        history.update_many(facts_t)

    metrics = metrics_from_ranks(ranks)
    extra = {
        "dataset": ds.dataset,
        "split": split,
        "tie_policy": tie_policy,
        "params": params.__dict__,
        "diagnostic_repeat_h1": float(np.mean(diagnostic_hits["repeat"])) if diagnostic_hits["repeat"] else None,
        "diagnostic_near_repeat_h1": float(np.mean(diagnostic_hits["near_repeat"])) if diagnostic_hits["near_repeat"] else None,
        "diagnostic_novel_h1": float(np.mean(diagnostic_hits["novel"])) if diagnostic_hits["novel"] else None,
        "stale_top1": float(np.mean(stale_top1_flags)) if stale_top1_flags else None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(out_dir / f"{split}_metrics.json", metrics, extra)
    with (out_dir / f"{split}_query_ranks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["s", "r", "o", "t", "rank", "label", "top1", "top1_score", "stale_top1"])
        writer.writeheader()
        writer.writerows(rows)

    if export_topk:
        np.savez_compressed(
            out_dir / f"{split}_hri_topk_scores.npz",
            query_quads=np.asarray(query_quads_all, dtype=np.int64),
            topk_ids=np.asarray(topk_ids_all, dtype=np.int64),
            topk_scores=np.asarray(topk_scores_all, dtype=np.float64),
        )
    return metrics, extra


def parse_args():
    p = argparse.ArgumentParser(description="Optimized HRI-inspired recurrence baseline for ICEWS14/ICEWS18")
    p.add_argument("--dataset", choices=["ICEWS14", "ICEWS18"], required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--w-occurrence", type=float, default=1.0)
    p.add_argument("--w-frequency", type=float, default=0.25)
    p.add_argument("--w-recency", type=float, default=1.0)
    p.add_argument("--decay", type=float, default=0.03)
    p.add_argument("--stale-threshold", type=int, default=30)
    p.add_argument("--stale-penalty", type=float, default=0.0)
    p.add_argument("--tie-policy", choices=["stable_id", "average", "optimistic"], default="stable_id")
    p.add_argument("--export-topk", action="store_true")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--max-queries", type=int, default=None, help="Debug only: evaluate first N queries")
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_dataset(args.dataset, args.data_root)
    params = HRIParams(
        w_occurrence=args.w_occurrence,
        w_frequency=args.w_frequency,
        w_recency=args.w_recency,
        decay=args.decay,
        stale_threshold=args.stale_threshold,
        stale_penalty=args.stale_penalty,
    )
    out_dir = Path(args.out_dir or f"results/{args.dataset}/optimized_hri")
    metrics, extra = evaluate_split(ds, args.split, params, out_dir, args.tie_policy, args.export_topk, args.topk, args.max_queries)
    print(json.dumps({**metrics.to_dict(), **extra}, indent=2))


if __name__ == "__main__":
    main()
