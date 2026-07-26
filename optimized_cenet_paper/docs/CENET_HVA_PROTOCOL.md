# CENET + HVA/RHVC Protocol

CENET is treated as a neural backbone. HVA is not inserted inside CENET's temporal encoder. Instead, it is attached after CENET produces candidate scores, matching the manuscript's ranking-stage formulation.

## Variants

- **CENET baseline:** original CENET object scores.
- **Exact Recency:** fixed recency reward added to CENET scores.
- **RHVC:** post-hoc diagnostic prototype using exact recurrence, frequency, recency, and stale penalty.
- **HVA dual-branch:** learned adapter using exact, subject-candidate, and relation-candidate historical signals.
- **HVA exact-only:** final method using exact historical occurrence, recency, frequency, and stale tendency.

## Safety checks

- Top-K is selected from CENET score matrices only.
- Gold target is not inserted into top-K.
- History features use only previous timestamps (`t' < t`).
- Validation score dumps are used for hyperparameter selection/training.
- Test score dumps are reserved for final evaluation.
