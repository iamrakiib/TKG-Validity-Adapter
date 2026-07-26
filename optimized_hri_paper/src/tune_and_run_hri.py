from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

from data_utils import load_dataset
from hri_recurrence import HRIParams
from run_hri import evaluate_split


def parse_args():
    p = argparse.ArgumentParser(description="Tune HRI-inspired recurrence baseline on validation, then run test.")
    p.add_argument("--dataset", choices=["ICEWS14", "ICEWS18"], required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tie-policy", choices=["stable_id", "average", "optimistic"], default="stable_id")
    p.add_argument("--export-topk", action="store_true")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--max-queries", type=int, default=None, help="Debug only")
    return p.parse_args()


def main():
    args = parse_args()
    ds = load_dataset(args.dataset, args.data_root)
    root = Path(args.out_dir or f"results/{args.dataset}/optimized_hri_tuned")
    root.mkdir(parents=True, exist_ok=True)

    # Small, reviewer-readable grid. Extend this if the original experiment used a larger search.
    w_frequency_grid = [0.0, 0.10, 0.25, 0.50]
    w_recency_grid = [0.50, 1.00, 1.50]
    decay_grid = [0.01, 0.03, 0.05, 0.10]
    stale_grid = [15, 30, 60]

    rows = []
    best = None
    best_payload = None
    for wf, wr, decay, stale in product(w_frequency_grid, w_recency_grid, decay_grid, stale_grid):
        params = HRIParams(w_occurrence=1.0, w_frequency=wf, w_recency=wr, decay=decay, stale_threshold=stale, stale_penalty=0.0)
        metrics, extra = evaluate_split(ds, "valid", params, root / "valid_grid" / f"wf{wf}_wr{wr}_d{decay}_st{stale}", args.tie_policy, False, args.topk, args.max_queries)
        row = {"w_frequency": wf, "w_recency": wr, "decay": decay, "stale_threshold": stale, **metrics.to_dict()}
        rows.append(row)
        if best is None or metrics.mrr > best.mrr:
            best = metrics
            best_payload = params

    with (root / "validation_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (root / "best_valid_config.json").open("w", encoding="utf-8") as f:
        json.dump({"params": best_payload.__dict__, "valid_metrics": best.to_dict(), "tie_policy": args.tie_policy}, f, indent=2, sort_keys=True)

    test_metrics, test_extra = evaluate_split(ds, "test", best_payload, root / "test_best", args.tie_policy, args.export_topk, args.topk, args.max_queries)
    with (root / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"test_metrics": test_metrics.to_dict(), "extra": test_extra, "selected_params": best_payload.__dict__}, f, indent=2, sort_keys=True)
    print(json.dumps({"selected_params": best_payload.__dict__, "test_metrics": test_metrics.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
