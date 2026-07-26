#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import random
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from config import (
    args,
    ensure_dir,
    get_branch_checkpoint_path,
    get_branch_dump_path,
    get_branch_result_path,
    get_combined_result_path,
    resolve_use_history_gate,
)
from evolution import calc_filtered_mrr, calc_filtered_test_mrr, calc_raw_mrr
from history_validity_gate import (
    augment_with_inverse,
    build_train_and_train_valid_histories,
    canonicalize_cygnet_queries,
    novelty_bucket_from_history,
    read_triples,
    stale_exact_bucket,
)
from link_prediction import link_prediction
from utils import get_total_number, load_quadruples

warnings.filterwarnings(action="ignore")
torch.set_num_threads(2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_dataset():
    data_dir = os.path.join(args.data_root, args.dataset)
    train_data, train_times = load_quadruples(data_dir, "train.txt")
    valid_data, valid_times = load_quadruples(data_dir, "valid.txt")
    test_data, test_times = load_quadruples(data_dir, "test.txt")
    num_e, num_r = get_total_number(data_dir, "stat.txt")
    all_times = np.concatenate([train_times, valid_times, test_times])
    num_times = int(max(all_times) / args.time_stamp) + 1
    return {
        "data_dir": data_dir,
        "train_data": train_data,
        "train_times": train_times,
        "valid_data": valid_data,
        "valid_times": valid_times,
        "test_data": test_data,
        "test_times": test_times,
        "num_e": num_e,
        "num_r": num_r,
        "num_times": num_times,
    }


def history_npz_path(dataset: str, data_root: str, entity: str, timestamp: int) -> str:
    subdir = "copy_seq" if entity == "object" else "copy_seq_sub"
    return os.path.join(data_root, dataset, subdir, f"train_h_r_copy_seq_{int(timestamp)}.npz")


def load_train_history_matrix(
    dataset: str,
    data_root: str,
    entity: str,
    train_times: np.ndarray,
    num_e: int,
    num_r: int,
) -> sp.csr_matrix:
    matrix = sp.csr_matrix((num_e * num_r, num_e), dtype=np.float32)
    for ts in train_times:
        matrix = matrix + sp.load_npz(history_npz_path(dataset, data_root, entity, int(ts))).astype(np.float32).tocsr()
    return matrix


def build_sparse_history_from_triples(
    triples_np: np.ndarray,
    num_e: int,
    num_r: int,
    entity: str,
) -> sp.csr_matrix:
    if triples_np is None or len(triples_np) == 0:
        return sp.csr_matrix((num_e * num_r, num_e), dtype=np.float32)

    if entity == "object":
        row = triples_np[:, 0] * num_r + triples_np[:, 1]
        col = triples_np[:, 2]
    elif entity == "subject":
        row = triples_np[:, 2] * num_r + triples_np[:, 1]
        col = triples_np[:, 0]
    else:
        raise ValueError(f"Unsupported branch: {entity}")

    data = np.ones(len(row), dtype=np.float32)
    return sp.csr_matrix((data, (row, col)), shape=(num_e * num_r, num_e), dtype=np.float32)


def build_eval_history_matrix(
    train_history: sp.csr_matrix,
    valid_data: np.ndarray,
    split: str,
    num_e: int,
    num_r: int,
    entity: str,
) -> sp.csr_matrix:
    if split == "test" and args.include_valid_history_in_test:
        return train_history + build_sparse_history_from_triples(valid_data, num_e, num_r, entity)
    return train_history


def build_batch_copy_inputs(
    history_matrix: sp.csr_matrix,
    batch_data: np.ndarray,
    num_r: int,
    entity: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if entity == "object":
        labels = torch.LongTensor(batch_data[:, 2])
        seq_idx = batch_data[:, 0] * num_r + batch_data[:, 1]
    elif entity == "subject":
        labels = torch.LongTensor(batch_data[:, 0])
        seq_idx = batch_data[:, 2] * num_r + batch_data[:, 1]
    else:
        raise ValueError(f"Unsupported branch: {entity}")

    tail_seq = np.asarray(history_matrix[seq_idx].todense(), dtype=np.float32)
    one_hot_tail_seq = torch.from_numpy(tail_seq)
    one_hot_tail_seq = one_hot_tail_seq.masked_fill(one_hot_tail_seq != 0, 1.0)

    return labels.to(device), one_hot_tail_seq.to(device)


def build_filter_map_from_arrays(arrays: Sequence[np.ndarray], num_r: int) -> Dict[Tuple[int, int], set]:
    triples = []
    for arr in arrays:
        if arr is None or len(arr) == 0:
            continue
        triples.extend([tuple(map(int, row[:4])) for row in np.asarray(arr)])
    aug = augment_with_inverse(triples, num_r)

    out: Dict[Tuple[int, int], set] = {}
    for s, r, o, _ in aug:
        out.setdefault((s, r), set()).add(o)
    return out


def safe_div(x: float, y: float) -> float:
    return 0.0 if y == 0 else x / y


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


def finalize_bucket_stats(bucket_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for bucket, values in bucket_stats.items():
        count = max(int(values["count"]), 1)
        out[bucket] = {
            "count": int(values["count"]),
            "MRR": safe_div(values["MRR"], count),
            "Hits@1": safe_div(values["Hits@1"], count),
            "Hits@3": safe_div(values["Hits@3"], count),
            "Hits@10": safe_div(values["Hits@10"], count),
        }
    return out


def init_bucket_stats() -> Dict[str, Dict[str, float]]:
    return {
        "repeat": {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0},
        "near_repeat": {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0},
        "novel": {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0, "count": 0},
    }


def update_bucket_and_stale_from_batch(
    scores_np: np.ndarray,
    triples_np: np.ndarray,
    entity: str,
    num_r: int,
    filter_map: Dict[Tuple[int, int], set],
    sr_hist,
    so_hist,
    ro_hist,
    bucket_stats: Dict[str, Dict[str, float]],
    stale_state: Dict[str, int],
) -> None:
    canonical_queries = canonicalize_cygnet_queries(triples_np, entity=entity, num_rels=num_r)
    top1 = np.argmax(scores_np, axis=1)

    for i in range(len(triples_np)):
        s, r, o, t = map(int, canonical_queries[i][:4])
        rank = filtered_rank_from_scores(scores_np[i], s, r, o, filter_map)

        mrr = 1.0 / rank
        hits1 = 1.0 if rank <= 1 else 0.0
        hits3 = 1.0 if rank <= 3 else 0.0
        hits10 = 1.0 if rank <= 10 else 0.0

        bucket = novelty_bucket_from_history(s, r, o, t, sr_hist, so_hist, ro_hist)
        bucket_stats[bucket]["MRR"] += mrr
        bucket_stats[bucket]["Hits@1"] += hits1
        bucket_stats[bucket]["Hits@3"] += hits3
        bucket_stats[bucket]["Hits@10"] += hits10
        bucket_stats[bucket]["count"] += 1

        if bucket in {"near_repeat", "novel"}:
            stale_state["count"] += 1
            pred_o = int(top1[i])
            pred_bucket = stale_exact_bucket(s, r, pred_o, t, sr_hist)
            if pred_bucket == "stale":
                stale_state["stale_top1_count"] += 1


def average_branch_results(obj_result: Dict, sub_result: Dict, split: str) -> Dict:
    combined = {
        "row_name": obj_result["row_name"],
        "method": obj_result["method"],
        "split": split,
        "branch": "combined",
        "checkpoint_path": "",
        "dump_path": "",
        "history_scope": obj_result.get("history_scope", ""),
        "overall_metric_source": obj_result.get("overall_metric_source", ""),
        "bucket_metric_source": obj_result.get("bucket_metric_source", ""),
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
        "object_result_path": obj_result.get("result_path", ""),
        "subject_result_path": sub_result.get("result_path", ""),
    }

    all_buckets = set(obj_result["bucket_metrics_filtered"].keys()) | set(sub_result["bucket_metrics_filtered"].keys())
    for bucket in all_buckets:
        obj_bucket = obj_result["bucket_metrics_filtered"].get(
            bucket, {"count": 0, "MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        )
        sub_bucket = sub_result["bucket_metrics_filtered"].get(
            bucket, {"count": 0, "MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        )
        combined["bucket_metrics_filtered"][bucket] = {
            "count": int(obj_bucket["count"]) + int(sub_bucket["count"]),
            "MRR": 0.5 * (obj_bucket["MRR"] + sub_bucket["MRR"]),
            "Hits@1": 0.5 * (obj_bucket["Hits@1"] + sub_bucket["Hits@1"]),
            "Hits@3": 0.5 * (obj_bucket["Hits@3"] + sub_bucket["Hits@3"]),
            "Hits@10": 0.5 * (obj_bucket["Hits@10"] + sub_bucket["Hits@10"]),
        }

    return combined


def print_priority_result_block(result: Dict) -> None:
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


def build_model(num_e: int, num_r: int, num_times: int, use_cuda: bool) -> link_prediction:
    return link_prediction(
        i_dim=num_e,
        h_dim=args.hidden_dim,
        num_rels=num_r,
        num_times=num_times,
        time_stamp=args.time_stamp,
        alpha=args.alpha,
        use_cuda=use_cuda,
        use_history_gate=resolve_use_history_gate(args),
        hva_topk=args.hva_topk,
        hva_mode=args.hva_mode,
        hva_gamma_exact=args.hva_gamma_exact,
        hva_gamma_near=args.hva_gamma_near,
        hva_stale_init=args.hva_stale_init,
    )


def evaluate_branch(
    model: link_prediction,
    checkpoint_path: str,
    split: str,
    entity: str,
    eval_data: np.ndarray,
    history_matrix: sp.csr_matrix,
    num_e: int,
    num_r: int,
    train_data: np.ndarray,
    valid_data: np.ndarray,
    test_data: np.ndarray,
    device: torch.device,
    history_context,
    filter_map,
    sr_hist,
    so_hist,
    ro_hist,
) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dump_requested = (
        args.dump_full_scores
        or (split == "valid" and args.dump_valid)
        or (split == "test" and args.dump_test)
    )

    dump_path = get_branch_dump_path(args, split=split, branch=entity)
    result_path = get_branch_result_path(args, split=split, branch=entity)

    all_scores = [] if dump_requested else None
    all_triples = [] if dump_requested else None

    native_mrr = 0.0
    native_hits1 = 0.0
    native_hits3 = 0.0
    native_hits10 = 0.0
    native_count = 0

    bucket_stats = init_bucket_stats()
    stale_state = {"count": 0, "stale_top1_count": 0}

    train_tensor = torch.LongTensor(train_data)
    valid_tensor = torch.LongTensor(valid_data)
    test_tensor = torch.LongTensor(test_data)

    n_batch = (eval_data.shape[0] + args.batch_size - 1) // args.batch_size
    if args.smoke and args.max_eval_batches > 0:
        n_batch = min(n_batch, args.max_eval_batches)

    print(
        f"start eval | row={args.row_name} | split={split} | branch={entity} | "
        f"batches={n_batch} | batch_size={args.batch_size} | dump={dump_requested}",
        flush=True,
    )

    for batch_idx in range(n_batch):
        batch_start = batch_idx * args.batch_size
        batch_end = min(eval_data.shape[0], (batch_idx + 1) * args.batch_size)
        batch_data = eval_data[batch_start:batch_end]

        _, one_hot_tail_seq = build_batch_copy_inputs(
            history_matrix=history_matrix,
            batch_data=batch_data,
            num_r=num_r,
            entity=entity,
            device=device,
        )

        with torch.no_grad():
            scores = model(
                batch_data,
                one_hot_tail_seq,
                entity=entity,
                history_context=history_context,
            )

        if args.raw:
            batch_mrr, batch_hits1, batch_hits3, batch_hits10 = calc_raw_mrr(
                scores,
                torch.LongTensor(batch_data[:, 2] if entity == "object" else batch_data[:, 0]).to(device),
                hits=[1, 3, 10],
            )
        else:
            if split == "valid":
                batch_mrr, batch_hits1, batch_hits3, batch_hits10 = calc_filtered_mrr(
                    num_e,
                    scores,
                    train_tensor,
                    valid_tensor,
                    torch.LongTensor(batch_data),
                    entity=entity,
                    hits=[1, 3, 10],
                )
            else:
                batch_mrr, batch_hits1, batch_hits3, batch_hits10 = calc_filtered_test_mrr(
                    num_e,
                    scores,
                    train_tensor,
                    valid_tensor,
                    test_tensor,
                    torch.LongTensor(batch_data),
                    entity=entity,
                    hits=[1, 3, 10],
                )

        batch_n = len(batch_data)
        native_mrr += batch_mrr * batch_n
        native_hits1 += batch_hits1 * batch_n
        native_hits3 += batch_hits3 * batch_n
        native_hits10 += batch_hits10 * batch_n
        native_count += batch_n

        scores_np = scores.detach().cpu().numpy().astype(np.float32)
        triples_np = batch_data.astype(np.int64)

        update_bucket_and_stale_from_batch(
            scores_np=scores_np,
            triples_np=triples_np,
            entity=entity,
            num_r=num_r,
            filter_map=filter_map,
            sr_hist=sr_hist,
            so_hist=so_hist,
            ro_hist=ro_hist,
            bucket_stats=bucket_stats,
            stale_state=stale_state,
        )

        if dump_requested:
            all_scores.append(scores_np)
            all_triples.append(triples_np)

        if batch_idx == 0 or (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batch:
            print(
                f"eval progress | row={args.row_name} | split={split} | branch={entity} | "
                f"batch={batch_idx + 1}/{n_batch}",
                flush=True,
            )

    print(
        f"finished eval loop | row={args.row_name} | split={split} | branch={entity} | "
        f"count={native_count}",
        flush=True,
    )

    if dump_requested:
        all_scores_np = np.concatenate(all_scores, axis=0)
        all_triples_np = np.concatenate(all_triples, axis=0)

        print(
            f"saving dump | row={args.row_name} | split={split} | branch={entity} | path={dump_path}",
            flush=True,
        )

        ensure_dir(os.path.dirname(dump_path))
        np.savez_compressed(
            dump_path,
            scores=all_scores_np,
            triples=all_triples_np,
            entity=np.array([entity]),
        )

    bucket_metrics_filtered = finalize_bucket_stats(bucket_stats)
    stale_top1 = {
        "count": int(stale_state["count"]),
        "stale_top1_count": int(stale_state["stale_top1_count"]),
        "stale_top1_rate": safe_div(stale_state["stale_top1_count"], stale_state["count"]),
    }

    result = {
        "row_name": args.row_name,
        "method": args.method,
        "split": split,
        "branch": entity,
        "checkpoint_path": checkpoint_path,
        "dump_path": dump_path if dump_requested else "",
        "history_scope": "train_plus_valid" if (split == "test" and args.include_valid_history_in_test) else "train_only",
        "overall_metric_source": "native_batchwise_filtered" if not args.raw else "native_batchwise_raw",
        "bucket_metric_source": "custom_filtered_rank",
        "overall_filtered": {
            "count": int(native_count),
            "MRR": safe_div(native_mrr, native_count),
            "Hits@1": safe_div(native_hits1, native_count),
            "Hits@3": safe_div(native_hits3, native_count),
            "Hits@10": safe_div(native_hits10, native_count),
        },
        "bucket_metrics_filtered": bucket_metrics_filtered,
        "stale_top1_interference": stale_top1,
        "best_epoch": int(checkpoint.get("epoch", -1)) + 1 if "epoch" in checkpoint else -1,
        "result_path": result_path,
    }

    ensure_dir(os.path.dirname(result_path))
    save_json(result, result_path)

    if args.print_priority_block:
        print_priority_result_block(result)

    return result


def main():
    set_seed(args.seed)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    data = load_dataset()
    train_data = data["train_data"]
    valid_data = data["valid_data"]
    test_data = data["test_data"]
    train_times = data["train_times"]
    num_e = data["num_e"]
    num_r = data["num_r"]
    num_times = data["num_times"]

    train_triples = read_triples(os.path.join(data["data_dir"], "train.txt"))
    valid_triples = read_triples(os.path.join(data["data_dir"], "valid.txt"))
    train_hist, train_valid_hist = build_train_and_train_valid_histories(train_triples, valid_triples, num_r)

    valid_filter_map = build_filter_map_from_arrays([train_data, valid_data], num_r)
    test_filter_map = build_filter_map_from_arrays([train_data, valid_data, test_data], num_r)

    if args.entity == "combined":
        branches: List[str] = ["object", "subject"]
    else:
        branches = [args.entity]

    split_list = ["valid", "test"] if args.eval_split == "both" else [args.eval_split]
    branch_results = {split: {} for split in split_list}

    for branch in branches:
        checkpoint_path = ""
        if branch == "object" and args.checkpoint_obj:
            checkpoint_path = args.checkpoint_obj
        elif branch == "subject" and args.checkpoint_sub:
            checkpoint_path = args.checkpoint_sub
        elif args.checkpoint_path:
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_path = get_branch_checkpoint_path(args, branch=branch, which="best")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found for branch={branch}: {checkpoint_path}")

        train_history_matrix = load_train_history_matrix(
            dataset=args.dataset,
            data_root=args.data_root,
            entity=branch,
            train_times=train_times,
            num_e=num_e,
            num_r=num_r,
        )

        model = build_model(
            num_e=num_e,
            num_r=num_r,
            num_times=num_times,
            use_cuda=use_cuda,
        ).to(device)

        for split in split_list:
            if split == "valid":
                eval_data = valid_data
                history_matrix = train_history_matrix
                history_context = train_hist if resolve_use_history_gate(args) else None
                filter_map = valid_filter_map
                bucket_histories = train_hist
            else:
                eval_data = test_data
                history_matrix = build_eval_history_matrix(
                    train_history=train_history_matrix,
                    valid_data=valid_data,
                    split="test",
                    num_e=num_e,
                    num_r=num_r,
                    entity=branch,
                )
                test_histories = train_valid_hist if args.include_valid_history_in_test else train_hist
                history_context = test_histories if resolve_use_history_gate(args) else None
                filter_map = test_filter_map
                bucket_histories = test_histories

            result = evaluate_branch(
                model=model,
                checkpoint_path=checkpoint_path,
                split=split,
                entity=branch,
                eval_data=eval_data,
                history_matrix=history_matrix,
                num_e=num_e,
                num_r=num_r,
                train_data=train_data,
                valid_data=valid_data,
                test_data=test_data,
                device=device,
                history_context=history_context,
                filter_map=filter_map,
                sr_hist=bucket_histories["sr"],
                so_hist=bucket_histories["so"],
                ro_hist=bucket_histories["ro"],
            )
            branch_results[split][branch] = result

    if set(branches) == {"object", "subject"}:
        for split in split_list:
            combined = average_branch_results(
                obj_result=branch_results[split]["object"],
                sub_result=branch_results[split]["subject"],
                split=split,
            )
            combined_result_path = get_combined_result_path(args, split=split)
            ensure_dir(os.path.dirname(combined_result_path))
            save_json(combined, combined_result_path)
            if args.print_priority_block:
                print_priority_result_block(combined)

    print("evaluation done", flush=True)


if __name__ == "__main__":
    main()
