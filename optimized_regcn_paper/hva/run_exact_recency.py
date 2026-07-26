from __future__ import annotations

import argparse
import os

import numpy as np

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
)
from hva.metrics import evaluate_diagnostics, evaluate_filtered


def apply_exact_recency(scores, queries, histories, alpha=0.1, lam=0.1):
    out = scores.copy().astype(np.float32)
    sr = histories["sr"]
    for i, q in enumerate(queries):
        s, r, _o, t = map(int, q[:4])
        cand_map = sr.get((s, r), {})
        for cand, times in cand_map.items():
            if cand < 0 or cand >= out.shape[1]:
                continue
            lt = last_time_before(times, t)
            if lt is None:
                continue
            gap = max(0, t - lt)
            out[i, int(cand)] += float(alpha) * np.exp(-float(lam) * float(gap))
    return out


def main():
    p = argparse.ArgumentParser(description="Exact Recency Heuristic over backbone score dump")
    p.add_argument("--dataset", required=True, choices=["ICEWS14", "ICEWS18"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--dump", required=True)
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--lambda-decay", type=float, default=0.1)
    p.add_argument("--stale-threshold", type=int, default=10)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    ensure_dir(args.out_dir)
    num_e, num_rels = read_stat(os.path.join(args.data_root, args.dataset))
    arrays = read_split_arrays(os.path.join(args.data_root, args.dataset))
    scores, triples, entity = load_score_dump(args.dump)
    queries = canonicalize_queries(triples, entity, num_rels)

    if args.split == "valid":
        hist_triples = augment_with_inverse([tuple(x) for x in arrays["train"]], num_rels)
        filter_triples = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"]], axis=0)], num_rels)
    else:
        hist_triples = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"]], axis=0)], num_rels)
        filter_triples = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"], arrays["test"]], axis=0)], num_rels)
    histories = build_histories(hist_triples)
    adjusted = apply_exact_recency(scores, queries, histories, args.alpha, args.lambda_decay)
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "alpha": args.alpha,
        "lambda_decay": args.lambda_decay,
        "overall": evaluate_filtered(adjusted, queries, filter_triples),
        "diagnostics": evaluate_diagnostics(adjusted, queries, filter_triples, histories, args.stale_threshold),
    }
    save_json(result, os.path.join(args.out_dir, "exact_recency_results.json"))
    np.savez_compressed(os.path.join(args.out_dir, "exact_recency_scores.npz"), scores=adjusted, triples=queries, entity=np.asarray(["object"]))
    print(result)


if __name__ == "__main__":
    main()
