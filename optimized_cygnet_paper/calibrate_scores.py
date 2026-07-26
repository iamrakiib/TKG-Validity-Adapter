import argparse
import json
import os
import random

import numpy as np
import torch

from history_validity_calibration import (
    RelationHistoryValidityCalibrator,
    average_branch_results,
    build_filter_map_from_arrays,
    build_ro_history,
    build_so_history,
    build_sr_history,
    compute_delta,
    evaluate_bucket_metrics_filtered,
    evaluate_model_filtered,
    aggregate_native_filtered_metrics,
    load_dump,
    read_triples,
    split_dump_by_time_fraction,
    stale_top1_interference_from_scores,
    train_calibrator,
)
from history_validity_gate import augment_with_inverse


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_stat(data_dir: str):
    with open(os.path.join(data_dir, "stat.txt"), "r") as f:
        line = f.readline().strip().split()
        return int(line[0]), int(line[1])


def load_original_split_arrays(data_dir: str):
    def _read(path: str):
        rows = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                rows.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
        return np.asarray(rows, dtype=np.int64)

    return {
        "train": _read(os.path.join(data_dir, "train.txt")),
        "valid": _read(os.path.join(data_dir, "valid.txt")),
        "test": _read(os.path.join(data_dir, "test.txt")),
    }


def build_histories(train_triples, valid_triples, num_rels):
    train_aug = augment_with_inverse(train_triples, num_rels)
    train_valid_aug = augment_with_inverse(train_triples + valid_triples, num_rels)
    return (
        {
            "sr": build_sr_history(train_aug),
            "so": build_so_history(train_aug),
            "ro": build_ro_history(train_aug),
        },
        {
            "sr": build_sr_history(train_valid_aug),
            "so": build_so_history(train_valid_aug),
            "ro": build_ro_history(train_valid_aug),
        },
    )


def resolve_dump_path(args, branch: str, split: str) -> str:
    explicit = ""
    if branch == "object" and split == "valid":
        explicit = args.valid_dump_object
    elif branch == "object" and split == "test":
        explicit = args.test_dump_object
    elif branch == "subject" and split == "valid":
        explicit = args.valid_dump_subject
    elif branch == "subject" and split == "test":
        explicit = args.test_dump_subject

    if explicit:
        return explicit

    return os.path.join(
        args.run_root,
        args.dataset,
        args.source_row,
        args.source_run,
        branch,
        "dumps",
        f"{split}_scores.npz",
    )


def branch_out_dir(args, branch: str) -> str:
    return os.path.join(
        args.run_root,
        args.dataset,
        args.target_row,
        args.target_run,
        branch,
    )


def combined_out_dir(args) -> str:
    return os.path.join(
        args.run_root,
        args.dataset,
        args.target_row,
        args.target_run,
        "combined",
    )


def print_priority_result_block(result: dict):
    bucket_metrics = result.get("bucket_metrics_filtered", {})
    near_h1 = bucket_metrics.get("near_repeat", {}).get("Hits@1", 0.0)
    novel_h1 = bucket_metrics.get("novel", {}).get("Hits@1", 0.0)
    repeat_h1 = bucket_metrics.get("repeat", {}).get("Hits@1", 0.0)

    print("================ PRIORITY RESULT ================")
    print(f"row: {result.get('row_name', '')}")
    print(f"split: {result.get('split', '')}")
    print(f"branch: {result.get('branch', '')}")
    print(f"filtered/test MRR:          {result['overall_filtered']['MRR']:.4f}")
    print(f"near-repeat Hits@1:         {near_h1:.4f}")
    print(f"novel Hits@1:               {novel_h1:.4f}")
    print(f"stale-top1 interference:    {result['stale_top1_interference']['stale_top1_rate']:.4f}")
    print(f"repeat Hits@1:              {repeat_h1:.4f}")
    print(f"checkpoint path: {result.get('checkpoint_path', '')}")
    print(f"dump path:       {result.get('dump_path', '')}")
    print("===============================================")


def main():
    parser = argparse.ArgumentParser(description="Post-hoc RHVC runner for CyGNet dumps")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--run-root", type=str, default="./runs/cygnet")

    parser.add_argument("--source-row", type=str, required=True)
    parser.add_argument("--source-run", type=str, required=True)
    parser.add_argument("--target-row", type=str, default="rhvc_prototype")
    parser.add_argument("--target-run", type=str, required=True)

    parser.add_argument("--valid-dump-object", type=str, default="")
    parser.add_argument("--test-dump-object", type=str, default="")
    parser.add_argument("--valid-dump-subject", type=str, default="")
    parser.add_argument("--test-dump-subject", type=str, default="")

    parser.add_argument("--mode", type=str, default="dual_branch", choices=["exact_only", "dual_branch"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--topk-cands", type=int, default=256)
    parser.add_argument("--eval-topk-cands", type=int, default=256)

    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=4)

    parser.add_argument("--pairwise-weight", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--bias-reg", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-norm", type=float, default=1.0)

    parser.add_argument("--rel-emb-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--init-gamma-exact", type=float, default=0.02)
    parser.add_argument("--init-gamma-near", type=float, default=0.10)
    parser.add_argument("--stale-init", type=float, default=0.40)
    parser.add_argument("--init-base-scale", type=float, default=1.0)
    parser.add_argument("--max-bias", type=float, default=2.5)

    parser.add_argument("--disable-score-mlp", action="store_true", default=False)
    parser.add_argument("--disable-uncertainty-gate", action="store_true", default=False)

    parser.add_argument("--save-adjusted-scores", action="store_true", default=False)
    parser.add_argument("--include-valid-history-in-test", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(args.data_root, args.dataset)
    num_e, num_rels = read_stat(data_dir)
    arrays = load_original_split_arrays(data_dir)

    train_triples = read_triples(os.path.join(data_dir, "train.txt"))
    valid_triples = read_triples(os.path.join(data_dir, "valid.txt"))
    train_hist, train_valid_hist = build_histories(train_triples, valid_triples, num_rels)

    valid_filter_map = build_filter_map_from_arrays([arrays["train"], arrays["valid"]], num_rels)
    test_filter_map = build_filter_map_from_arrays([arrays["train"], arrays["valid"], arrays["test"]], num_rels)

    test_histories = train_valid_hist if args.include_valid_history_in_test else train_hist
    branch_results = {}

    for branch in ["object", "subject"]:
        valid_dump_path = resolve_dump_path(args, branch=branch, split="valid")
        test_dump_path = resolve_dump_path(args, branch=branch, split="test")

        valid_scores, valid_queries, valid_entity = load_dump(valid_dump_path)
        test_scores, test_queries, test_entity = load_dump(test_dump_path)

        if valid_entity != branch or test_entity != branch:
            print(f"Warning: dump entity tags do not match requested branch={branch}. Using branch={branch} anyway.")

        valid_split = split_dump_by_time_fraction(valid_scores, valid_queries, dev_frac=args.dev_frac)

        model = RelationHistoryValidityCalibrator(
            num_relations=2 * num_rels,
            mode=args.mode,
            rel_emb_dim=args.rel_emb_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            init_gamma_exact=args.init_gamma_exact,
            init_gamma_near=args.init_gamma_near,
            stale_init=args.stale_init,
            init_base_scale=args.init_base_scale,
            max_bias=args.max_bias,
            use_score_mlp=not args.disable_score_mlp,
            use_uncertainty_gate=not args.disable_uncertainty_gate,
        ).to(device)

        model, train_summary = train_calibrator(
            model=model,
            train_scores_np=valid_split["train_scores"],
            train_triples_np=valid_split["train_triples"],
            entity=branch,
            num_r=num_rels,
            sr_hist=train_hist["sr"],
            so_hist=train_hist["so"],
            ro_hist=train_hist["ro"],
            device=device,
            dev_scores_np=valid_split["dev_scores"],
            dev_triples_np=valid_split["dev_triples"],
            dev_num_e=num_e,
            dev_train_data=arrays["train"],
            dev_valid_data=arrays["valid"],
            dev_test_data=arrays["test"],
            dev_split="valid",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            topk_cands=args.topk_cands,
            eval_topk_cands=args.eval_topk_cands,
            patience=args.patience,
            min_epochs=args.min_epochs,
            pairwise_weight=args.pairwise_weight,
            margin=args.margin,
            bias_reg=args.bias_reg,
            label_smoothing=args.label_smoothing,
            grad_norm=args.grad_norm,
        )

        out_dir = branch_out_dir(args, branch)
        ckpt_dir = ensure_dir(os.path.join(out_dir, "checkpoints"))
        reports_dir = ensure_dir(os.path.join(out_dir, "reports"))
        dumps_dir = ensure_dir(os.path.join(out_dir, "dumps"))

        ckpt_path = os.path.join(ckpt_dir, f"rhvc_{args.mode}.pt")
        torch.save(model.state_dict(), ckpt_path)

        valid_base_overall = aggregate_native_filtered_metrics(
            scores_np=valid_scores,
            triples_np=valid_queries,
            entity=branch,
            num_e=num_e,
            train_data=arrays["train"],
            valid_data=arrays["valid"],
            test_data=arrays["test"],
            split="valid",
            batch_size=args.eval_batch_size,
        )
        valid_base_bucket = evaluate_bucket_metrics_filtered(
            scores_np=valid_scores,
            triples_np=valid_queries,
            entity=branch,
            num_r=num_rels,
            filter_map=valid_filter_map,
            sr_hist=train_hist["sr"],
            so_hist=train_hist["so"],
            ro_hist=train_hist["ro"],
        )
        valid_base_interference = stale_top1_interference_from_scores(
            scores_np=valid_scores,
            triples_np=valid_queries,
            entity=branch,
            num_r=num_rels,
            sr_hist=train_hist["sr"],
            so_hist=train_hist["so"],
            ro_hist=train_hist["ro"],
        )

        test_base_overall = aggregate_native_filtered_metrics(
            scores_np=test_scores,
            triples_np=test_queries,
            entity=branch,
            num_e=num_e,
            train_data=arrays["train"],
            valid_data=arrays["valid"],
            test_data=arrays["test"],
            split="test",
            batch_size=args.eval_batch_size,
        )
        test_base_bucket = evaluate_bucket_metrics_filtered(
            scores_np=test_scores,
            triples_np=test_queries,
            entity=branch,
            num_r=num_rels,
            filter_map=test_filter_map,
            sr_hist=test_histories["sr"],
            so_hist=test_histories["so"],
            ro_hist=test_histories["ro"],
        )
        test_base_interference = stale_top1_interference_from_scores(
            scores_np=test_scores,
            triples_np=test_queries,
            entity=branch,
            num_r=num_rels,
            sr_hist=test_histories["sr"],
            so_hist=test_histories["so"],
            ro_hist=test_histories["ro"],
        )

        valid_adjusted_scores, valid_overall, valid_bucket, valid_interference = evaluate_model_filtered(
            model=model,
            scores_np=valid_scores,
            triples_np=valid_queries,
            entity=branch,
            num_r=num_rels,
            num_e=num_e,
            train_data=arrays["train"],
            valid_data=arrays["valid"],
            test_data=arrays["test"],
            split="valid",
            filter_map=valid_filter_map,
            sr_hist=train_hist["sr"],
            so_hist=train_hist["so"],
            ro_hist=train_hist["ro"],
            device=device,
            batch_size=args.eval_batch_size,
            topk_cands=args.eval_topk_cands,
        )

        test_adjusted_scores, test_overall, test_bucket, test_interference = evaluate_model_filtered(
            model=model,
            scores_np=test_scores,
            triples_np=test_queries,
            entity=branch,
            num_r=num_rels,
            num_e=num_e,
            train_data=arrays["train"],
            valid_data=arrays["valid"],
            test_data=arrays["test"],
            split="test",
            filter_map=test_filter_map,
            sr_hist=test_histories["sr"],
            so_hist=test_histories["so"],
            ro_hist=test_histories["ro"],
            device=device,
            batch_size=args.eval_batch_size,
            topk_cands=args.eval_topk_cands,
        )

        valid_adjusted_path = ""
        test_adjusted_path = ""
        if args.save_adjusted_scores:
            valid_adjusted_path = os.path.join(dumps_dir, "valid_scores_adjusted.npz")
            test_adjusted_path = os.path.join(dumps_dir, "test_scores_adjusted.npz")
            np.savez_compressed(valid_adjusted_path, scores=valid_adjusted_scores, triples=valid_queries, entity=np.array([branch]))
            np.savez_compressed(test_adjusted_path, scores=test_adjusted_scores, triples=test_queries, entity=np.array([branch]))

        result = {
            "row_name": args.target_row,
            "method": "rhvc_prototype",
            "branch": branch,
            "mode": args.mode,
            "checkpoint_path": ckpt_path,
            "history_scope": "train_plus_valid" if args.include_valid_history_in_test else "train_only",
            "overall_metric_source": "native_batchwise_filtered",
            "bucket_metric_source": "custom_filtered_rank",
            "valid_dump_path": valid_dump_path,
            "test_dump_path": test_dump_path,
            "valid_adjusted_dump_path": valid_adjusted_path,
            "test_adjusted_dump_path": test_adjusted_path,
            "training_summary": train_summary,
            "valid_base_overall_filtered": valid_base_overall,
            "valid_base_bucket_metrics_filtered": valid_base_bucket,
            "valid_base_stale_top1_interference": valid_base_interference,
            "valid_overall_filtered": valid_overall,
            "valid_bucket_metrics_filtered": valid_bucket,
            "valid_stale_top1_interference": valid_interference,
            "valid_improvement_over_base": compute_delta(valid_base_overall, valid_overall),
            "test_base_overall_filtered": test_base_overall,
            "test_base_bucket_metrics_filtered": test_base_bucket,
            "test_base_stale_top1_interference": test_base_interference,
            "overall_filtered": test_overall,
            "bucket_metrics_filtered": test_bucket,
            "stale_top1_interference": test_interference,
            "test_improvement_over_base": compute_delta(test_base_overall, test_overall),
            "split": "test",
            "dump_path": test_adjusted_path if args.save_adjusted_scores else test_dump_path,
        }

        result_path = os.path.join(reports_dir, f"results_{args.mode}.json")
        save_json(result, result_path)
        result["result_path"] = result_path

        print_priority_result_block(result)
        branch_results[branch] = result

    combined = average_branch_results(branch_results["object"], branch_results["subject"])
    combined["row_name"] = args.target_row
    combined["method"] = "rhvc_prototype"
    combined["branch"] = "combined"
    combined["split"] = "test"
    combined["checkpoint_path"] = ""
    combined["dump_path"] = ""
    combined["history_scope"] = "train_plus_valid" if args.include_valid_history_in_test else "train_only"
    combined["overall_metric_source"] = "native_batchwise_filtered"
    combined["bucket_metric_source"] = "custom_filtered_rank"
    combined["object_result_path"] = branch_results["object"]["result_path"]
    combined["subject_result_path"] = branch_results["subject"]["result_path"]

    combined_dir = combined_out_dir(args)
    combined_reports = ensure_dir(os.path.join(combined_dir, "reports"))
    combined_path = os.path.join(combined_reports, f"results_{args.mode}.json")
    save_json(combined, combined_path)

    print_priority_result_block(combined)


if __name__ == "__main__":
    main()