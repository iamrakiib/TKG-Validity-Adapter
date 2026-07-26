from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from hva.history_utils import (
    augment_with_inverse,
    build_histories,
    canonicalize_queries,
    ensure_dir,
    load_score_dump,
    read_split_arrays,
    read_stat,
    save_json,
    set_seed,
    split_by_time_fraction,
)
from hva.hva_model import HistoryValidityAdapter, apply_hva_batch
from hva.metrics import evaluate_diagnostics, evaluate_filtered


def evaluate_model(model, scores_np, queries_np, histories, filter_triples, args, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(queries_np), args.eval_batch_size):
            end = min(start + args.eval_batch_size, len(queries_np))
            batch_scores = torch.tensor(scores_np[start:end], dtype=torch.float32, device=device)
            adjusted_full, _, _, _ = apply_hva_batch(
                model, batch_scores, queries_np[start:end], histories, args.eval_topk, device
            )
            outs.append(adjusted_full.detach().cpu().numpy().astype(np.float32))
    adjusted = np.concatenate(outs, axis=0) if outs else np.empty_like(scores_np, dtype=np.float32)
    overall = evaluate_filtered(adjusted, queries_np, filter_triples)
    diag = evaluate_diagnostics(adjusted, queries_np, filter_triples, histories, args.stale_threshold)
    return adjusted, overall, diag


def train_adapter(model, train_scores, train_queries, dev_scores, dev_queries, histories, filter_triples_dev, args, device):
    dataset = TensorDataset(torch.tensor(train_scores, dtype=torch.float32), torch.tensor(train_queries, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_mrr = -1.0
    bad = 0
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = ce_total = pair_total = reg_total = count = 0.0
        for batch_scores_cpu, batch_queries_cpu in loader:
            full_scores = batch_scores_cpu.to(device)
            batch_queries = batch_queries_cpu.cpu().numpy()
            gold_ids = batch_queries_cpu[:, 2].to(device=device, dtype=torch.long)

            adjusted_full, adjusted_topk, candidate_ids, delta = apply_hva_batch(
                model, full_scores, batch_queries, histories, args.topk, device
            )
            ce_loss = F.cross_entropy(adjusted_full, gold_ids, label_smoothing=args.label_smoothing)

            gold_score = adjusted_full.gather(1, gold_ids.unsqueeze(1)).squeeze(1)
            neg_scores = adjusted_full.clone()
            neg_scores.scatter_(1, gold_ids.unsqueeze(1), float("-inf"))
            hardest_neg = neg_scores.max(dim=1).values
            pairwise = F.relu(args.margin - (gold_score - hardest_neg)).mean()
            reg = delta.pow(2).mean()
            loss = ce_loss + args.pairwise_weight * pairwise + args.bias_reg * reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            opt.step()

            bsz = float(full_scores.size(0))
            total += float(loss.item()) * bsz
            ce_total += float(ce_loss.item()) * bsz
            pair_total += float(pairwise.item()) * bsz
            reg_total += float(reg.item()) * bsz
            count += bsz

        epoch_info = {
            "epoch": epoch,
            "train_loss": total / max(count, 1.0),
            "train_ce": ce_total / max(count, 1.0),
            "train_pairwise": pair_total / max(count, 1.0),
            "train_delta_reg": reg_total / max(count, 1.0),
        }
        if len(dev_queries) > 0:
            _, dev_overall, _ = evaluate_model(model, dev_scores, dev_queries, histories, filter_triples_dev, args, device)
            epoch_info["dev_MRR"] = dev_overall["MRR"]
            if dev_overall["MRR"] > best_mrr:
                best_mrr = dev_overall["MRR"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
        log.append(epoch_info)
        print(epoch_info)
        if len(dev_queries) > 0 and epoch >= args.min_epochs and bad >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return log


def main():
    p = argparse.ArgumentParser(description="Leakage-safe HVA exact-only runner from backbone score dumps")
    p.add_argument("--dataset", required=True, choices=["ICEWS14", "ICEWS18"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--valid-dump", required=True, help="npz with keys scores,trips/triples for validation")
    p.add_argument("--test-dump", required=True, help="npz with keys scores,triples for test")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["exact_only", "dual_branch"], default="exact_only")
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--eval-topk", type=int, default=100)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--min-epochs", type=int, default=4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--rel-emb-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--gamma", type=float, default=0.10)
    p.add_argument("--stale-threshold", type=int, default=10)
    p.add_argument("--dev-frac", type=float, default=0.20)
    p.add_argument("--pairwise-weight", type=float, default=0.25)
    p.add_argument("--margin", type=float, default=0.20)
    p.add_argument("--bias-reg", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-adjusted-scores", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(args.data_root, args.dataset)
    num_e, num_rels = read_stat(data_dir)
    arrays = read_split_arrays(data_dir)

    valid_scores_raw, valid_triples_raw, valid_entity = load_score_dump(args.valid_dump)
    test_scores_raw, test_triples_raw, test_entity = load_score_dump(args.test_dump)
    if valid_entity != test_entity:
        raise ValueError(f"valid dump entity={valid_entity}, test dump entity={test_entity}; use matching branches")

    valid_queries = canonicalize_queries(valid_triples_raw, valid_entity, num_rels)
    test_queries = canonicalize_queries(test_triples_raw, test_entity, num_rels)

    # Histories must be causal. Valid uses train only; test uses train+valid.
    train_aug = augment_with_inverse([tuple(x) for x in arrays["train"]], num_rels)
    train_valid_aug = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"]], axis=0)], num_rels)
    valid_histories = build_histories(train_aug)
    test_histories = build_histories(train_valid_aug)

    filter_valid = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"]], axis=0)], num_rels)
    filter_test = augment_with_inverse([tuple(x) for x in np.concatenate([arrays["train"], arrays["valid"], arrays["test"]], axis=0)], num_rels)

    train_scores, train_queries, dev_scores, dev_queries = split_by_time_fraction(valid_scores_raw, valid_queries, args.dev_frac)
    model = HistoryValidityAdapter(
        num_relations=max(num_rels * 2, int(np.max(valid_queries[:, 1]) + 1), int(np.max(test_queries[:, 1]) + 1)),
        mode=args.mode,
        rel_emb_dim=args.rel_emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gamma=args.gamma,
        stale_threshold=args.stale_threshold,
    ).to(device)

    train_log = train_adapter(model, train_scores, train_queries, dev_scores, dev_queries, valid_histories, filter_valid, args, device)

    valid_adjusted, valid_overall, valid_diag = evaluate_model(model, valid_scores_raw, valid_queries, valid_histories, filter_valid, args, device)
    test_adjusted, test_overall, test_diag = evaluate_model(model, test_scores_raw, test_queries, test_histories, filter_test, args, device)

    result = {
        "dataset": args.dataset,
        "entity_branch": valid_entity,
        "mode": args.mode,
        "topk": args.topk,
        "eval_topk": args.eval_topk,
        "seed": args.seed,
        "valid_overall": valid_overall,
        "valid_diagnostics": valid_diag,
        "test_overall": test_overall,
        "test_diagnostics": test_diag,
        "train_log": train_log,
        "leakage_control": {
            "topk_candidate_selection": "torch.topk(base_scores.detach()) only; gold target is never inserted",
            "feature_construction": "gold/object column is masked before feature construction",
            "test_history": "train+valid only; no test facts before evaluating each test query except filtered evaluation labels",
        },
    }
    save_json(result, os.path.join(args.out_dir, "hva_results.json"))
    torch.save(model.state_dict(), os.path.join(args.out_dir, "hva_adapter.pt"))
    if args.save_adjusted_scores:
        np.savez_compressed(os.path.join(args.out_dir, "valid_adjusted_scores.npz"), scores=valid_adjusted, triples=valid_queries, entity=np.asarray(["object"]))
        np.savez_compressed(os.path.join(args.out_dir, "test_adjusted_scores.npz"), scores=test_adjusted, triples=test_queries, entity=np.asarray(["object"]))

    print("==== HVA TEST RESULT ====")
    print(result["test_overall"])
    print(result["test_diagnostics"])
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
