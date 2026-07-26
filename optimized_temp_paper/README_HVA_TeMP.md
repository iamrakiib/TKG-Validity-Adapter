# Optimized TeMP + paper-aligned HVA/RHVC reproduction folder

This folder is based on the supplied `TeMP-master.zip` code and is organized for the paper's reviewer-facing reproduction workflow.

## Paper roles implemented here

| Paper method | File / script | Role |
|---|---|---|
| TeMP baseline | `main.py`, `test.py`, `scripts/01_train_temp_*.sh` | Neural temporal forecasting backbone |
| Exact Recency Heuristic | `hva/run_exact_recency.py`, `scripts/03_run_temp_exact_recency_*.sh` | Fixed recency comparison over TeMP scores |
| RHVC | `hva/run_rhvc_from_scores.py`, `scripts/03_run_temp_rhvc_*.sh` | Post-hoc diagnostic prototype |
| HVA dual-branch | `hva/run_hva_from_scores.py --mode dual_branch`, `scripts/03_run_temp_hva_dual_*.sh` | Ablation/extension with broader near-history features |
| HVA exact-only | `hva/run_hva_from_scores.py --mode exact_only`, `scripts/03_run_temp_hva_exact_*.sh` | Final proposed method |

## Reviewer-safe pipeline

1. Train TeMP backbone.
2. Dump validation and test full candidate score matrices from TeMP.
3. Run paper variants from the same score dumps:
   - Exact Recency Heuristic,
   - RHVC diagnostic prototype,
   - HVA dual-branch,
   - HVA exact-only.

This matches the manuscript logic: HVA is not a replacement TeMP encoder. It is a leakage-safe ranking-stage adapter applied after backbone scoring.

## Leakage controls

- Top-K candidates are selected only from backbone scores with `torch.topk(base_scores.detach())`.
- The gold answer is never inserted into the top-K set.
- HVA masks the gold column before feature construction.
- History features use only `t' < t`.
- Test-time histories use train+validation facts only.

## Example Colab/CLI workflow

```bash
cd optimized_temp_paper
bash scripts/00_colab_install_legacy_temp.sh

# Train TeMP backbone
bash scripts/01_train_temp_icews14.sh 42

# After choosing the TeMP experiment/checkpoint folder, dump scores
bash scripts/02_dump_temp_valid_scores_icews14.sh experiments/<checkpoint-folder>
bash scripts/02_dump_temp_scores_icews14.sh experiments/<checkpoint-folder>

# Run all paper variants for TeMP
bash scripts/04_run_temp_paper_variants_icews14.sh 42
```

For ICEWS18, replace `icews14` with `icews18`.

## Important note

This folder is code- and protocol-aligned with the paper. Final table numbers should be verified by running the complete training and score-dump pipeline in the same GPU/Colab environment used for the experiments.
