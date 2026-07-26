#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from typing import Optional


ROW_CHOICES = [
    "native_baseline",
    "rhvc_prototype",
    "hva_exact_only",
    "hva_dual_branch",
]

METHOD_CHOICES = [
    "native_baseline",
    "hva",
]

ENTITY_CHOICES = [
    "object",
    "subject",
    "combined",
]

HVA_MODE_CHOICES = [
    "off",
    "exact_only",
    "dual_branch",
]

EVAL_SPLIT_CHOICES = [
    "valid",
    "test",
    "both",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized CyGNet thesis runner")

    # Native CyGNet defaults
    parser.add_argument("--dataset", type=str, default="YAGO")
    parser.add_argument("--time-stamp", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, use -1 for CPU")
    parser.add_argument("--regularization", type=float, default=0.01, help="L2 regularization weight")
    parser.add_argument("--valid-epoch", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--raw", action="store_true", default=False)
    parser.add_argument("--counts", type=int, default=4, help="Early-stop patience in evaluation rounds")
    parser.add_argument("--entity", type=str, default="subject", choices=ENTITY_CHOICES)

    # Data / run structure
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--run-root", type=str, default="./runs/cygnet")
    parser.add_argument("--row-name", type=str, default="native_baseline", choices=ROW_CHOICES)
    parser.add_argument("--run-name", type=str, default="run")
    parser.add_argument("--method", type=str, default="native_baseline", choices=METHOD_CHOICES)
    parser.add_argument("--seed", type=int, default=7)

    # Train / eval orchestration
    parser.add_argument("--eval-split", type=str, default="test", choices=EVAL_SPLIT_CHOICES)
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--checkpoint-obj", type=str, default="")
    parser.add_argument("--checkpoint-sub", type=str, default="")
    parser.add_argument("--resume-ckpt", type=str, default="")
    parser.add_argument("--save-latest", action="store_true", default=False)
    parser.add_argument("--save-best", action="store_true", default=False)

    # Dump / report control
    parser.add_argument("--dump-valid", action="store_true", default=False)
    parser.add_argument("--dump-test", action="store_true", default=False)
    parser.add_argument("--dump-full-scores", action="store_true", default=False)
    parser.add_argument("--dump-topk", type=int, default=256)

    # Smoke-mode control
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--max-train-times", type=int, default=0, help="0 means all")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 means all")

    # History semantics
    # Keep native baseline clean by default. Turning this on is explicit.
    parser.add_argument("--include-valid-history-in-test", dest="include_valid_history_in_test", action="store_true")
    parser.add_argument("--no-include-valid-history-in-test", dest="include_valid_history_in_test", action="store_false")
    parser.set_defaults(include_valid_history_in_test=False)

    # Result printing
    parser.add_argument("--print-priority-block", dest="print_priority_block", action="store_true")
    parser.add_argument("--no-print-priority-block", dest="print_priority_block", action="store_false")
    parser.set_defaults(print_priority_block=True)

    # HVA controls
    parser.add_argument("--use-history-gate", action="store_true", default=False)
    parser.add_argument("--hva-mode", type=str, default="off", choices=HVA_MODE_CHOICES)
    parser.add_argument("--hva-topk", type=int, default=256)
    parser.add_argument("--hva-gamma-exact", type=float, default=0.005)
    parser.add_argument("--hva-gamma-near", type=float, default=0.08)
    parser.add_argument("--hva-stale-init", type=float, default=0.2)

    return parser


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_use_history_gate(parsed_args) -> bool:
    return bool(
        parsed_args.use_history_gate
        or parsed_args.method == "hva"
        or parsed_args.hva_mode in {"exact_only", "dual_branch"}
    )


def get_row_root(
    run_root: str,
    dataset: str,
    row_name: str,
    run_name: str,
) -> str:
    return os.path.join(run_root, dataset, row_name, run_name)


def get_branch_run_dir(
    parsed_args,
    branch: Optional[str] = None,
    row_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    branch = branch or parsed_args.entity
    row_name = row_name or parsed_args.row_name
    run_name = run_name or parsed_args.run_name
    return os.path.join(get_row_root(parsed_args.run_root, parsed_args.dataset, row_name, run_name), branch)


def get_combined_run_dir(
    parsed_args,
    row_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> str:
    row_name = row_name or parsed_args.row_name
    run_name = run_name or parsed_args.run_name
    return os.path.join(get_row_root(parsed_args.run_root, parsed_args.dataset, row_name, run_name), "combined")


def get_branch_ckpt_dir(parsed_args, branch: Optional[str] = None) -> str:
    return os.path.join(get_branch_run_dir(parsed_args, branch=branch), "checkpoints")


def get_branch_dump_dir(parsed_args, branch: Optional[str] = None) -> str:
    return os.path.join(get_branch_run_dir(parsed_args, branch=branch), "dumps")


def get_branch_report_dir(parsed_args, branch: Optional[str] = None) -> str:
    return os.path.join(get_branch_run_dir(parsed_args, branch=branch), "reports")


def get_combined_report_dir(parsed_args) -> str:
    return os.path.join(get_combined_run_dir(parsed_args), "reports")


def get_branch_checkpoint_path(parsed_args, branch: Optional[str] = None, which: str = "best") -> str:
    return os.path.join(get_branch_ckpt_dir(parsed_args, branch=branch), f"{which}.pt")


def get_branch_dump_path(parsed_args, split: str, branch: Optional[str] = None) -> str:
    return os.path.join(get_branch_dump_dir(parsed_args, branch=branch), f"{split}_scores.npz")


def get_branch_result_path(parsed_args, split: str, branch: Optional[str] = None) -> str:
    return os.path.join(get_branch_report_dir(parsed_args, branch=branch), f"{split}_results.json")


def get_combined_result_path(parsed_args, split: str) -> str:
    return os.path.join(get_combined_report_dir(parsed_args), f"{split}_results.json")


args = build_parser().parse_args()


if __name__ == "__main__":
    print(args)
