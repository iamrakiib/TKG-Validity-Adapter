# Optimized HRI-Inspired Recurrence Baseline for HVA-TKG

This folder contains the optimized HRI-inspired recurrence baseline used as a **comparison baseline** for the HVA-TKG paper.

HRI is **not a neural backbone** and HVA is **not applied to HRI** in the paper. The role of this folder is to reproduce the recurrence-only comparison setting: how strong is history repetition by itself?

## Paper-consistent logic

For each query \((s,r,?,t)\), this baseline:

1. uses only facts with timestamp \(t' < t\);
2. checks exact historical recurrence \((s,r,c,t')\);
3. scores candidates using occurrence, frequency, and recency;
4. performs filtered ranking evaluation;
5. never inserts the gold answer into a candidate set;
6. exports optional top-\(K\) recurrence scores only for reproducibility/debugging.

This is consistent with the paper's baseline role:

> HRI-inspired recurrence baseline evaluates recurrence alone without a neural backbone.

## Folder structure

```text
optimized_hri/
├── data/
│   ├── ICEWS14/
│   └── ICEWS18/
├── src/
│   ├── data_utils.py
│   ├── hri_recurrence.py
│   ├── metrics.py
│   ├── run_hri.py
│   └── tune_and_run_hri.py
├── scripts/
│   ├── run_hri_icews14.sh
│   ├── run_hri_icews18.sh
│   ├── run_hri_both.sh
│   └── colab_setup_and_run.sh
├── configs/
├── results/
├── logs/
└── docs/
```

## Run in Colab / Linux terminal

```bash
pip install -r requirements.txt
bash scripts/run_hri_icews14.sh
bash scripts/run_hri_icews18.sh
```

Or run both:

```bash
bash scripts/run_hri_both.sh
```

## Outputs

For each dataset, the script writes:

```text
results/ICEWS14/optimized_hri/
├── validation_grid.csv
├── best_valid_config.json
├── test_summary.json
└── test_best/
    ├── test_metrics.json
    ├── test_query_ranks.csv
    └── test_hri_topk_scores.npz
```

The `.npz` file contains:

```text
query_quads: [s, r, o, t]
topk_ids: candidate ids selected from HRI scores only
topk_scores: recurrence scores
```

Gold candidate insertion is disabled by design.

## Important note for reviewers

This folder reproduces the recurrence-only baseline. It is intentionally separate from TiRGN, RE-GCN, CyGNet, TeMP, and CENET because HRI is not a trainable temporal graph backbone. HVA exact-only should be attached to neural backbone score dumps, not to this recurrence-only baseline.

## Upstream source

The original uploaded baseline package was `recurrency_baseline_tkg-master.zip`, associated with:

> History Repeats Itself: A Baseline for Temporal Knowledge Graph Forecasting.

The original upstream README and license are preserved in `docs/UPSTREAM_HRI_README.md` and `UPSTREAM_HRI_LICENSE.txt`.
