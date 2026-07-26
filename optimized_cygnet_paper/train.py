#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import random
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import optim

from config import (
    args,
    ensure_dir,
    get_branch_checkpoint_path,
    get_branch_ckpt_dir,
    get_branch_result_path,
    get_branch_run_dir,
    resolve_use_history_gate,
)
from evolution import calc_filtered_mrr, calc_raw_mrr
from history_validity_gate import read_triples, build_train_and_train_valid_histories
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


def load_history_npz(dataset: str, data_root: str, entity: str, timestamp: int) -> sp.csr_matrix:
    return sp.load_npz(history_npz_path(dataset, data_root, entity, timestamp)).astype(np.float32).tocsr()


def build_full_train_history_matrix(
    dataset: str,
    data_root: str,
    entity: str,
    train_times: np.ndarray,
    num_e: int,
    num_r: int,
) -> sp.csr_matrix:
    matrix = sp.csr_matrix((num_e * num_r, num_e), dtype=np.float32)
    for ts in train_times:
        matrix = matrix + load_history_npz(dataset, data_root, entity, int(ts))
    return matrix


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
        raise ValueError(f"Unsupported entity branch: {entity}")

    tail_seq = np.asarray(history_matrix[seq_idx].todense(), dtype=np.float32)
    one_hot_tail_seq = torch.from_numpy(tail_seq)
    one_hot_tail_seq = one_hot_tail_seq.masked_fill(one_hot_tail_seq != 0, 1.0)

    return labels.to(device), one_hot_tail_seq.to(device)


@torch.no_grad()
def evaluate_valid(
    model: link_prediction,
    train_data: np.ndarray,
    valid_data: np.ndarray,
    full_train_history: sp.csr_matrix,
    num_e: int,
    num_r: int,
    device: torch.device,
    history_context,
) -> Dict[str, float]:
    model.eval()

    total_mrr = 0.0
    total_hits1 = 0.0
    total_hits3 = 0.0
    total_hits10 = 0.0
    total_examples = 0

    n_batch = (valid_data.shape[0] + args.batch_size - 1) // args.batch_size
    if args.smoke and args.max_eval_batches > 0:
        n_batch = min(n_batch, args.max_eval_batches)

    for batch_idx in range(n_batch):
        batch_start = batch_idx * args.batch_size
        batch_end = min(valid_data.shape[0], (batch_idx + 1) * args.batch_size)
        valid_batch_data = valid_data[batch_start:batch_end]

        labels, one_hot_tail_seq = build_batch_copy_inputs(
            history_matrix=full_train_history,
            batch_data=valid_batch_data,
            num_r=num_r,
            entity=args.entity,
            device=device,
        )

        valid_score = model(
            valid_batch_data,
            one_hot_tail_seq,
            entity=args.entity,
            history_context=history_context,
        )

        if args.raw:
            mrr, hits1, hits3, hits10 = calc_raw_mrr(valid_score, labels, hits=[1, 3, 10])
        else:
            mrr, hits1, hits3, hits10 = calc_filtered_mrr(
                num_e,
                valid_score,
                torch.LongTensor(train_data),
                torch.LongTensor(valid_data),
                torch.LongTensor(valid_batch_data),
                entity=args.entity,
                hits=[1, 3, 10],
            )

        batch_n = len(valid_batch_data)
        total_mrr += mrr * batch_n
        total_hits1 += hits1 * batch_n
        total_hits3 += hits3 * batch_n
        total_hits10 += hits10 * batch_n
        total_examples += batch_n

    total_examples = max(total_examples, 1)
    return {
        "MRR": total_mrr / total_examples,
        "Hits@1": total_hits1 / total_examples,
        "Hits@3": total_hits3 / total_examples,
        "Hits@10": total_hits10 / total_examples,
        "count": total_examples,
    }


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


def main():
    if args.entity not in {"object", "subject"}:
        raise ValueError("train.py only supports --entity object or --entity subject")

    set_seed(args.seed)

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    data = load_dataset()
    train_data = data["train_data"]
    train_times = data["train_times"]
    valid_data = data["valid_data"]
    num_e = data["num_e"]
    num_r = data["num_r"]
    num_times = data["num_times"]

    run_dir = ensure_dir(get_branch_run_dir(args, branch=args.entity))
    ckpt_dir = ensure_dir(get_branch_ckpt_dir(args, branch=args.entity))
    reports_dir = ensure_dir(os.path.join(run_dir, "reports"))
    dump_dir = ensure_dir(os.path.join(run_dir, "dumps"))

    train_log_path = os.path.join(run_dir, "training_log.json")
    train_summary_path = os.path.join(reports_dir, "train_summary.json")

    model = build_model(num_e=num_e, num_r=num_r, num_times=num_times, use_cuda=use_cuda).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    use_history_gate = resolve_use_history_gate(args)
    train_history_context = None
    if use_history_gate:
        train_triples = read_triples(os.path.join(data["data_dir"], "train.txt"))
        valid_triples = read_triples(os.path.join(data["data_dir"], "valid.txt"))
        train_hist, _ = build_train_and_train_valid_histories(train_triples, valid_triples, num_r)
        train_history_context = train_hist

    full_train_history = build_full_train_history_matrix(
        dataset=args.dataset,
        data_root=args.data_root,
        entity=args.entity,
        train_times=train_times,
        num_e=num_e,
        num_r=num_r,
    )

    best_mrr = 0.0
    best_epoch = -1
    non_improve_rounds = 0
    start_epoch = 0

    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        best_mrr = float(checkpoint.get("best_mrr", 0.0))
        best_epoch = int(checkpoint.get("epoch", -1))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    train_log = {
        "config": vars(args),
        "epochs": [],
        "best_mrr": best_mrr,
        "best_epoch": best_epoch,
        "run_dir": run_dir,
        "checkpoints": {
            "latest": get_branch_checkpoint_path(args, branch=args.entity, which="latest"),
            "best": get_branch_checkpoint_path(args, branch=args.entity, which="best"),
        },
    }

    history_time_indices = list(range(1, len(train_times)))
    if args.smoke and args.max_train_times > 0:
        history_time_indices = history_time_indices[:args.max_train_times]

    print(f"start train | entity={args.entity} | use_history_gate={use_history_gate} | run_dir={run_dir}")

    for epoch in range(start_epoch, args.n_epochs):
        model.train()

        cumulative_history = sp.csr_matrix((num_e * num_r, num_e), dtype=np.float32)

        epoch_losses = []
        epoch_forward_times = []
        epoch_backward_times = []

        for history_pos in history_time_indices:
            prev_time = int(train_times[history_pos - 1])
            current_time = int(train_times[history_pos])

            cumulative_history = cumulative_history + load_history_npz(
                dataset=args.dataset,
                data_root=args.data_root,
                entity=args.entity,
                timestamp=prev_time,
            )

            train_sample_data = train_data[train_data[:, 3] == current_time]
            if len(train_sample_data) == 0:
                continue

            batch_order = np.arange(len(train_sample_data))
            np.random.shuffle(batch_order)
            train_sample_data = train_sample_data[batch_order]

            n_batch = (train_sample_data.shape[0] + args.batch_size - 1) // args.batch_size

            for batch_idx in range(n_batch):
                batch_start = batch_idx * args.batch_size
                batch_end = min(train_sample_data.shape[0], (batch_idx + 1) * args.batch_size)
                train_batch_data = train_sample_data[batch_start:batch_end]

                labels, one_hot_tail_seq = build_batch_copy_inputs(
                    history_matrix=cumulative_history,
                    batch_data=train_batch_data,
                    num_r=num_r,
                    entity=args.entity,
                    device=device,
                )

                optimizer.zero_grad(set_to_none=True)

                t0 = time.time()
                score = model(
                    train_batch_data,
                    one_hot_tail_seq,
                    entity=args.entity,
                    history_context=train_history_context,
                )
                t1 = time.time()

                if use_history_gate:
                    loss_main = F.cross_entropy(score, labels)
                else:
                    loss_main = F.nll_loss(score, labels)

                loss = loss_main + model.regularization_loss(reg_param=args.regularization)
                loss.backward()
                optimizer.step()
                t2 = time.time()

                epoch_losses.append(float(loss.item()))
                epoch_forward_times.append(t1 - t0)
                epoch_backward_times.append(t2 - t1)

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "forward_time_mean": float(np.mean(epoch_forward_times)) if epoch_forward_times else None,
            "backward_time_mean": float(np.mean(epoch_backward_times)) if epoch_backward_times else None,
            "val_mrr_filter": None,
            "val_hits1_filter": None,
            "val_hits3_filter": None,
            "val_hits10_filter": None,
            "is_best": False,
        }

        if args.save_latest:
            latest_path = get_branch_checkpoint_path(args, branch=args.entity, which="latest")
            ensure_dir(os.path.dirname(latest_path))
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_mrr": best_mrr,
                    "config": vars(args),
                },
                latest_path,
            )

        if (epoch + 1) >= args.valid_epoch:
            valid_metrics = evaluate_valid(
                model=model,
                train_data=train_data,
                valid_data=valid_data,
                full_train_history=full_train_history,
                num_e=num_e,
                num_r=num_r,
                device=device,
                history_context=train_history_context,
            )

            current_mrr = float(valid_metrics["MRR"])
            epoch_row["val_mrr_filter"] = current_mrr
            epoch_row["val_hits1_filter"] = float(valid_metrics["Hits@1"])
            epoch_row["val_hits3_filter"] = float(valid_metrics["Hits@3"])
            epoch_row["val_hits10_filter"] = float(valid_metrics["Hits@10"])

            print(
                f"valid | epoch={epoch + 1:04d} | MRR={current_mrr:.6f} | "
                f"H@1={valid_metrics['Hits@1']:.6f} | H@3={valid_metrics['Hits@3']:.6f} | "
                f"H@10={valid_metrics['Hits@10']:.6f}"
            )

            if current_mrr > best_mrr:
                best_mrr = current_mrr
                best_epoch = epoch + 1
                non_improve_rounds = 0
                epoch_row["is_best"] = True

                if args.save_best:
                    best_path = get_branch_checkpoint_path(args, branch=args.entity, which="best")
                    ensure_dir(os.path.dirname(best_path))
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_mrr": best_mrr,
                            "config": vars(args),
                        },
                        best_path,
                    )
            else:
                non_improve_rounds += 1

        train_log["best_mrr"] = float(best_mrr)
        train_log["best_epoch"] = int(best_epoch)
        train_log["epochs"].append(epoch_row)
        save_json(train_log, train_log_path)

        print(
            "epoch {:04d} | loss {:.6f} | best_mrr {:.6f} | forward {:.6f}s | backward {:.6f}s".format(
                epoch + 1,
                float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
                best_mrr,
                float(np.mean(epoch_forward_times)) if epoch_forward_times else 0.0,
                float(np.mean(epoch_backward_times)) if epoch_backward_times else 0.0,
            )
        )

        if (epoch + 1) >= args.valid_epoch and non_improve_rounds >= args.counts:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    best_path = get_branch_checkpoint_path(args, branch=args.entity, which="best")
    latest_path = get_branch_checkpoint_path(args, branch=args.entity, which="latest")
    if args.save_best and not os.path.exists(best_path):
        ensure_dir(os.path.dirname(best_path))
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": max(best_epoch - 1, 0),
                "best_mrr": best_mrr,
                "config": vars(args),
            },
            best_path,
        )

    train_summary = {
        "row_name": args.row_name,
        "method": args.method,
        "entity": args.entity,
        "use_history_gate": use_history_gate,
        "hva_mode": args.hva_mode,
        "best_epoch": int(best_epoch),
        "best_valid_mrr": float(best_mrr),
        "checkpoint_latest": latest_path if args.save_latest else "",
        "checkpoint_best": best_path if args.save_best else "",
        "training_log_path": train_log_path,
        "dump_dir": dump_dir,
    }
    save_json(train_summary, train_summary_path)

    print("training done")
    print(json.dumps(train_summary, indent=2))


if __name__ == "__main__":
    main()
