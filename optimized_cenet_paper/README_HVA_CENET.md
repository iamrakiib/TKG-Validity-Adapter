# Optimized CENET with HVA/RHVC Paper Variants

This folder keeps **CENET** as the temporal knowledge graph forecasting backbone and adds the paper-aligned ranking-stage variants used in the manuscript:

- CENET baseline score export
- Exact Recency Heuristic
- RHVC post-hoc diagnostic prototype
- HVA dual-branch
- HVA exact-only final method

The implementation follows the paper protocol:

1. Train/evaluate CENET as the backbone.
2. Export object-candidate score matrices for validation and test.
3. Select top-K candidates from CENET scores only.
4. Build history-validity features using only facts with `t' < t`.
5. Apply Exact Recency, RHVC, HVA dual-branch, or HVA exact-only.
6. Report filtered ranking metrics and diagnostic metrics including StaleTop1.

## Leakage-safe rule

The gold entity is never inserted into the top-K candidate set. Top-K is obtained from backbone scores only.

## Reviewer workflow

```bash
bash scripts/00_colab_install_cenet.sh
bash scripts/00_prepare_cenet_history_icews18.sh
bash scripts/01_train_cenet_icews18.sh 42
```

After training, identify the produced CENET experiment folder under `SAVE/cenet/ICEWS18/seed42/`, then dump scores:

```bash
bash scripts/02_dump_cenet_valid_scores_icews18.sh SAVE/cenet/ICEWS18/seed42/<experiment-folder>
bash scripts/02_dump_cenet_test_scores_icews18.sh  SAVE/cenet/ICEWS18/seed42/<experiment-folder>
```

Run the paper variants:

```bash
bash scripts/04_run_cenet_paper_variants_icews18.sh 42
```

For ICEWS14, place the official `valid.txt` split in `data/ICEWS14/` before CENET history preprocessing if it is missing in the upstream CENET copy.
