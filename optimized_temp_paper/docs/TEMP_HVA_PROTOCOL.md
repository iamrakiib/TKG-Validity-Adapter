# TeMP + HVA exact-only protocol

This folder keeps TeMP as the backbone and adds the paper-aligned HVA exact-only ranking-stage correction.

## Role of this folder

- TeMP remains the temporal message passing backbone.
- The patched TeMP evaluation can export candidate score dumps.
- HVA exact-only is trained/applied from those score dumps.
- Top-K candidate selection is score-only and never inserts the gold entity.
- Historical validity features use only facts before the query timestamp.

## Order of execution

1. Train the TeMP backbone with `scripts/01_train_temp_icews14.sh` or `scripts/01_train_temp_icews18.sh`.
2. Export candidate score dumps using `scripts/02_dump_temp_scores_*.sh`.
3. Prepare both validation and test score dumps. If your checkpoint workflow only exports test scores, rerun evaluation on the validation checkpoint/split or provide validation dumps from the training stage.
4. Run HVA exact-only using `scripts/03_run_temp_hva_*.sh`.

## Important reviewer note

This code is organized to match the paper logic: backbone score -> top-K score-only candidate selection -> exact historical validity features -> HVA correction -> filtered evaluation. The repository should not claim that HVA is a new TeMP encoder. HVA is a ranking-stage adapter attached after TeMP scoring.
