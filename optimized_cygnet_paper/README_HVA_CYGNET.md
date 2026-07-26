# Optimized CyGNet with paper-aligned HVA/RHVC variants

This folder keeps CyGNet as the temporal knowledge graph forecasting backbone and adds the reviewer-safe paper variants used in the manuscript:

- CyGNet baseline
- CyGNet + Exact Recency Heuristic
- CyGNet + RHVC post-hoc diagnostic prototype
- CyGNet + HVA dual-branch
- CyGNet + HVA exact-only final method

## Protocol

The protocol is the same as the other optimized backbone folders:

1. Train CyGNet object and subject branches.
2. Dump validation and test candidate score matrices.
3. Apply Exact Recency, RHVC, HVA dual-branch, and HVA exact-only from the dumped CyGNet scores.
4. Average object and subject branch results for CyGNet reporting.

All variants obey the paper's evaluation-control rules:

- top-K candidates are selected only from CyGNet backbone scores;
- the gold entity is never inserted into top-K;
- history features use only timestamps earlier than the query timestamp;
- validation uses train history only;
- test uses train+valid history only.

## Quick Colab/CLI run

```bash
cd optimized_cygnet
bash scripts/00_colab_install_cygnet.sh
bash scripts/01_train_cygnet_icews14.sh 42
bash scripts/04_run_cygnet_paper_variants_icews14.sh 42
```

For ICEWS18:

```bash
bash scripts/01_train_cygnet_icews18.sh 42
bash scripts/04_run_cygnet_paper_variants_icews18.sh 42
```

## Reviewer note

HVA is attached as a ranking-stage adapter after CyGNet scoring. This is intentional and matches the manuscript: HVA is not a replacement for the CyGNet encoder/copy-generation mechanism; it recalibrates top-K candidate scores using candidate-level historical validity features.
