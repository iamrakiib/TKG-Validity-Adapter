# HRI Reproduction Guide

## Why HRI is different from other baselines

TiRGN, RE-GCN, CyGNet, TeMP, and CENET are neural forecasting backbones. HRI is different: it is a recurrence-only comparison baseline. It does not learn entity embeddings or temporal graph encoders.

Therefore, the correct GitHub organization is:

```text
optimized_tirgn/      -> neural backbone + HVA adapter path
optimized_regcn/      -> neural backbone + HVA adapter path
optimized_cygnet/     -> neural backbone + leakage-safe HVA adapter path
optimized_temp/       -> neural backbone + HVA adapter path from score dump
optimized_cenet/      -> neural backbone + HVA adapter path from score dump
optimized_hri/        -> recurrence-only baseline, no HVA injection
hva_common/           -> shared HVA exact-only adapter used by neural backbones
```

## What the script does

`src/tune_and_run_hri.py` first evaluates a small validation grid, selects the best recurrence parameters by validation MRR, and then evaluates once on test.

The temporal protocol is causal:

- validation starts with train history;
- test starts with train + validation history;
- facts at the current timestamp are evaluated before being added to history;
- only facts with \(t' < t\) influence a query at \(t\).

## Why no HVA injection into HRI

In the paper, HVA is designed as a ranking-stage adapter for backbone scores. HRI is already a recurrence-only baseline. Applying HVA to HRI would change its role and make the comparison unclear.

The consistent logic is not "HVA on HRI". The consistent logic is:

1. all methods use leakage-free temporal history;
2. no method inserts the gold candidate into top-K;
3. HRI remains recurrence-only;
4. neural backbones can be followed by HVA exact-only correction.
