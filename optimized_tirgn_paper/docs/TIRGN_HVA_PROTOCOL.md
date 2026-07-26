# TiRGN HVA protocol

TiRGN is used as a backbone model. Paper variants are applied after TiRGN candidate score generation.

## Variants

- Baseline: TiRGN candidate scores.
- Exact Recency: fixed recency boost over exact (s, r, c) history.
- RHVC: post-hoc diagnostic prototype.
- HVA dual-branch: learned correction using exact history plus near-history branches (s, *, c) and (*, r, c).
- HVA exact-only: final method using exact (s, r, c) historical validity features.

## Leakage control

- Top-K candidate membership is selected only from TiRGN scores.
- The gold answer is not inserted into the top-K set.
- History features are built only from timestamps earlier than the query timestamp.
- Validation is used for tuning; test is reserved for final reporting.
