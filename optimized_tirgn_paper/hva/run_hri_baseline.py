from __future__ import annotations

import argparse
import os

import numpy as np

from hva.history_utils import augment_with_inverse, build_histories, read_split_arrays, read_stat, save_json, ensure_dir, last_time_before, freq_before
from hva.metrics import evaluate_diagnostics, evaluate_filtered


def build_hri_scores(queries, num_entities, histories, alpha_freq=1.0, alpha_recency=1.0, lambda_decay=0.1):
    scores = np.zeros((len(queries), int(num_entities)), dtype=np.float32)
    sr = histories["sr"]
    for i, q in enumerate(queries):
        s, r, _o, t = map(int, q[:4])
        for cand, times in sr.get((s, r), {}).items():
            lt = last_time_before(times, t)
            if lt is None:
                continue
            gap = max(0, t - lt)
            freq = freq_before(times, t)
            scores[i, int(cand)] = alpha_freq * np.log1p(freq) + alpha_recency * np.exp(-lambda_decay * gap)
    return scores


def main():
    p = argparse.ArgumentParser(description="HRI-inspired recurrence-only baseline")
    p.add_argument("--dataset", required=True, choices=["ICEWS14", "ICEWS18"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--split", choices=["valid", "test"], default="test")
    p.add_argument("--alpha-freq", type=float, default=1.0)
    p.add_argument("--alpha-recency", type=float, default=1.0)
    p.add_argument("--lambda-decay", type=float, default=0.1)
    p.add_argument("--stale-threshold", type=int, default=10)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    ensure_dir(args.out_dir)
    data_dir = os.path.join(args.data_root, args.dataset)
    num_e, num_rels = read_stat(data_dir)
    arrays = read_split_arrays(data_dir)
    if args.split == "valid":
        queries = arrays["valid"]
        hist_triples = augment_with_inverse([tuple(x) for x in arrays["train"]], num_rels)
        filter_triples = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"]], axis=0)], num_rels)
    else:
        queries = arrays["test"]
        hist_triples = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"]], axis=0)], num_rels)
        filter_triples = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"], arrays["test"]], axis=0)], num_rels)
    histories = build_histories(hist_triples)
    scores = build_hri_scores(queries, num_e, histories, args.alpha_freq, args.alpha_recency, args.lambda_decay)
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "overall": evaluate_filtered(scores, queries, filter_triples),
        "diagnostics": evaluate_diagnostics(scores, queries, filter_triples, histories, args.stale_threshold),
    }
    save_json(result, os.path.join(args.out_dir, "hri_results.json"))
    np.savez_compressed(os.path.join(args.out_dir, "hri_scores.npz"), scores=scores, triples=queries, entity=np.asarray(["object"]))
    print(result)


if __name__ == "__main__":
    main()
