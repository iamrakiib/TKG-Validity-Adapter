# Optimized TiRGN with HVA paper variants

This folder keeps TiRGN as the temporal knowledge graph forecasting backbone and adds the paper-aligned ranking-stage variants used in the HVA study.

## Paper variants included

- **TiRGN baseline**: original TiRGN backbone score.
- **TiRGN + Exact Recency**: fixed exact-recency heuristic applied to TiRGN scores.
- **TiRGN + RHVC**: post-hoc diagnostic Relation History Validity Calibration prototype.
- **TiRGN + HVA dual-branch**: learned HVA adapter using exact and near-history branches.
- **TiRGN + HVA exact-only**: final proposed method using exact-history validity features.

## Protocol

The protocol is consistent with the paper:

1. Train TiRGN normally.
2. Dump validation and test candidate scores from TiRGN.
3. Apply Exact Recency, RHVC, HVA dual-branch, and HVA exact-only as ranking-stage variants.
4. Select top-K candidates only from TiRGN backbone scores.
5. Do not insert the gold entity into top-K.
6. Build history features using only facts with timestamp t' < t.
7. Report filtered ranking metrics and diagnostic metrics.

HVA is not treated as a new TiRGN encoder. It is a post-backbone ranking-stage adapter, which matches the paper formulation.
