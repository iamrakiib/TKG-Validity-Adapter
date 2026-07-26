# CyGNet HVA/RHVC protocol

CyGNet is used as the copy-generation temporal knowledge graph forecasting backbone. Because CyGNet has separate object and subject prediction branches, this package trains and evaluates both branches and then combines the branch-level results for reporting.

Paper variants included in this folder:

- CyGNet baseline
- CyGNet + Exact Recency Heuristic
- CyGNet + RHVC post-hoc diagnostic prototype
- CyGNet + HVA dual-branch
- CyGNet + HVA exact-only final method

Candidate selection follows the manuscript protocol: top-K candidates are selected only from CyGNet backbone scores, target labels are not used for candidate membership, and historical features are computed only from timestamps earlier than the query timestamp.
