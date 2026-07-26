# TKG-Validity-Adapter

Reproducibility code for candidate-level history validity correction in temporal knowledge graph forecasting.

## Overview

This repository contains anonymized reproducibility code, scripts, and documentation for experiments on candidate-level history validity correction in temporal knowledge graph forecasting.

The main idea is that historical occurrence does not always imply current validity. A candidate entity may appear frequently in previous temporal snapshots, but that candidate may become stale for the current query timestamp. This repository provides backbone-specific reproducibility folders for evaluating this idea across multiple temporal knowledge graph forecasting models.

## Main Paper Logic

For neural temporal knowledge graph forecasting backbones, the experimental protocol follows the same ranking-stage correction pipeline:

```text
Backbone training
→ Backbone candidate score generation
→ Top-K candidate selection from backbone scores only
→ History-validity feature construction using only t' < t
→ Ranking-stage correction
→ Filtered evaluation and diagnostic analysis
```

The following paper variants are provided for neural backbone folders:

1. Original backbone baseline
2. Exact Recency Heuristic
3. RHVC diagnostic prototype
4. HVA dual-branch
5. HVA exact-only

The HVA exact-only variant is the final proposed method. RHVC is included as a post-hoc diagnostic prototype, and HVA dual-branch is included as an ablation/extension.

## Repository Structure

```text
TKG-Validity-Adapter/
│
├── optimized_hri/
│   └── HRI-inspired recurrence-only baseline
│
├── optimized_tirgn/
│   └── TiRGN backbone with paper variants
│
├── optimized_regcn/
│   └── RE-GCN backbone with paper variants
│
├── optimized_cygnet/
│   └── CyGNet backbone with paper variants
│
├── optimized_temp/
│   └── TeMP backbone with paper variants
│
├── optimized_cenet/
│   └── CENET backbone with paper variants
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Backbone Folders

Each neural backbone folder contains backbone-specific training and evaluation code, together with paper-aligned ranking-stage variants.

| Folder | Role |
|---|---|
| `optimized_tirgn/` | TiRGN backbone with Exact Recency, RHVC, HVA dual-branch, and HVA exact-only |
| `optimized_regcn/` | RE-GCN backbone with Exact Recency, RHVC, HVA dual-branch, and HVA exact-only |
| `optimized_cygnet/` | CyGNet backbone with Exact Recency, RHVC, HVA dual-branch, and HVA exact-only |
| `optimized_temp/` | TeMP backbone with Exact Recency, RHVC, HVA dual-branch, and HVA exact-only |
| `optimized_cenet/` | CENET backbone with Exact Recency, RHVC, HVA dual-branch, and HVA exact-only |
| `optimized_hri/` | HRI-inspired recurrence-only baseline |

## Important Protocol Notes

All neural backbone variants follow the same leakage-controlled protocol:

```text
1. Top-K candidates are selected only from backbone scores.
2. Target entities are not inserted into the candidate set.
3. Historical features are computed only from timestamps earlier than the query timestamp.
4. Validation data is used for hyperparameter selection.
5. Test data is reserved for final evaluation.
```

The HRI-inspired baseline is kept separate because it is not a neural backbone. It is used to evaluate the strength of recurrence-only forecasting and is not modified with HVA, RHVC, or Exact Recency variants.

## Datasets

The experiments use the publicly available ICEWS14 and ICEWS18 temporal knowledge graph benchmark datasets.

No new dataset was created for this study.

Expected dataset structure:

```text
data/
├── ICEWS14/
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│   └── stat.txt
│
└── ICEWS18/
    ├── train.txt
    ├── valid.txt
    ├── test.txt
    └── stat.txt
```

Some backbone folders may require their own dataset formatting or preprocessing. Please check the README file inside each optimized backbone folder before running the scripts.

## Paper Variants

### Original Backbone

The original temporal knowledge graph forecasting model is trained and evaluated using its own backbone-specific implementation.

### Exact Recency Heuristic

Exact Recency applies a fixed recency-based boost to the backbone score when a candidate has exact historical evidence.

### RHVC Diagnostic Prototype

RHVC is a post-hoc diagnostic prototype that adjusts backbone rankings using recurrence, recency, frequency, and stale-history evidence.

### HVA Dual-Branch

HVA dual-branch uses exact-history features together with broader near-history signals.

### HVA Exact-Only

HVA exact-only is the final proposed method. It uses exact historical occurrence, recency, frequency, and stale tendency to learn a candidate-level score correction.

## Evaluation Metrics

The repository supports standard filtered ranking metrics:

```text
MRR
Hits@1
Hits@3
Hits@10
```

It also supports diagnostic metrics used in the paper:

```text
Repeat H@1
Near-repeat H@1
Novel H@1
StaleTop1
```

Higher values are better for MRR, Hits@K, Repeat H@1, Near-repeat H@1, and Novel H@1. Lower values are better for StaleTop1.

## Running Experiments

Each optimized backbone folder contains its own `scripts/` directory.

A typical workflow is:

```bash
cd optimized_temp

bash scripts/01_train_temp_icews14.sh 42
bash scripts/02_dump_temp_valid_scores_icews14.sh <checkpoint_folder>
bash scripts/02_dump_temp_scores_icews14.sh <checkpoint_folder>
bash scripts/03_run_temp_hva_exact_icews14.sh 42
```

The exact command names may differ by backbone. Please read the README inside each optimized folder.

## Generated Files

Large generated files are not included in this repository because of GitHub size constraints.

Excluded generated files include:

```text
checkpoints/
logs/
results/
score_dumps/
SAVE/
experiments/
*.pt
*.pth
*.ckpt
*.npz
*.pkl
```

Empty output folders may contain `.gitkeep` files. These files are placeholders only and will be populated after running the scripts.

## Reproducibility Note

This repository is organized for reproducibility and review. Full GPU reruns may be required to regenerate the exact checkpoints, score dumps, logs, and final result files.

The provided scripts are intended to reproduce the experimental pipeline:

```text
train backbone
dump scores
run paper variants
evaluate results
```

## Double-Blind Review Note

This repository has been prepared for double-blind review. Identifying author information, institutional information, personal paths, and private acknowledgements have been removed.

## Citation

Citation information will be added after the review process.

## License

This repository contains adapted code from multiple temporal knowledge graph forecasting baselines. Please check the license files and documentation inside each optimized backbone folder before redistribution.
