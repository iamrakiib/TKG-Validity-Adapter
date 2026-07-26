# RE-GCN HVA/RHVC protocol

RE-GCN remains the backbone. The paper variants are applied after RE-GCN score generation.

The shared protocol is:

```text
RE-GCN scores
  -> top-K candidates selected from RE-GCN scores only
  -> historical features computed with t' < t
  -> Exact Recency / RHVC / HVA dual-branch / HVA exact-only
  -> filtered ranking and diagnostic evaluation
```

## Variant roles

- Exact Recency is a fixed recency-based score boost.
- RHVC is a post-hoc diagnostic prototype.
- HVA dual-branch is a learned adapter using exact and near-history branches.
- HVA exact-only is the final proposed method using exact-history features.

The gold entity is not used for candidate membership during top-K selection.
