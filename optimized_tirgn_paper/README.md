# Optimized TiRGN reproducibility folder

This repository folder contains the optimized TiRGN backbone code and paper-aligned ranking-stage variants for the HVA temporal knowledge graph forecasting study.

## Main workflow

```bash
bash scripts/00_colab_install_tirgn.sh
bash scripts/00_prepare_tirgn_history_icews14.sh
bash scripts/01_train_tirgn_icews14.sh 42
bash scripts/02_dump_tirgn_valid_scores_icews14.sh 42
bash scripts/02_dump_tirgn_test_scores_icews14.sh 42
bash scripts/04_run_tirgn_paper_variants_icews14.sh 42
```

For ICEWS18, replace `icews14` with `icews18` in the script names.

## Paper variants

The folder supports:

- TiRGN baseline
- TiRGN + Exact Recency
- TiRGN + RHVC
- TiRGN + HVA dual-branch
- TiRGN + HVA exact-only

All variants follow the same leakage-free protocol: top-K candidates are selected only from TiRGN scores, target labels are not used for candidate membership, and history features use only timestamps earlier than the query.

See `README_HVA_TIRGN.md` and `docs/TIRGN_HVA_PROTOCOL.md` for details.
