# Optimized RE-GCN with HVA paper variants

This folder keeps RE-GCN as the temporal knowledge graph forecasting backbone and adds the paper-aligned ranking-stage variants used in the HVA study.

## Paper variants included

- **RE-GCN baseline**: original RE-GCN backbone score.
- **RE-GCN + Exact Recency**: fixed exact-recency heuristic applied to RE-GCN scores.
- **RE-GCN + RHVC**: post-hoc diagnostic Relation History Validity Calibration prototype.
- **RE-GCN + HVA dual-branch**: learned HVA adapter using exact and near-history branches.
- **RE-GCN + HVA exact-only**: final proposed method using exact-history validity features.

## Protocol

The protocol is consistent with the paper:

1. Train RE-GCN normally.
2. Dump validation and test candidate scores from RE-GCN.
3. Apply Exact Recency, RHVC, HVA dual-branch, and HVA exact-only as ranking-stage variants.
4. Select top-K candidates only from RE-GCN backbone scores.
5. Do not insert the gold entity into top-K.
6. Build history features using only facts with timestamp t' < t.
7. Report filtered ranking metrics and diagnostic metrics.

HVA is not treated as a new RE-GCN encoder. It is a post-backbone ranking-stage adapter, which matches the paper formulation.

## Colab/CLI run order

```bash
bash scripts/00_colab_install_regcn.sh
bash scripts/00_prepare_regcn_static_icews14.sh
bash scripts/01_train_regcn_icews14.sh 42
bash scripts/02_dump_regcn_valid_scores_icews14.sh 42
bash scripts/02_dump_regcn_test_scores_icews14.sh 42
bash scripts/04_run_regcn_paper_variants_icews14.sh 42
```

Use the corresponding ICEWS18 scripts for ICEWS18.

## Reviewer note

This package is organized for reproducibility. Full GPU training should be rerun in the same environment before claiming exact reproduction of reported table values.
