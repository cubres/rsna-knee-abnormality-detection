# V14 completed: official public score 0.949

Raw server row, observed 2026-10-08 10:54 UTC: submission **56948803** (V14, session 356365376, submitted 10:20:42 UTC) has raw status **COMPLETE** and **public score 0.949**. Exact row: [evidence/row_56948803_complete.json](evidence/row_56948803_complete.json).

This is the first completed official row from our own fault-tolerant implementation of the public anatomical-mirror recipe, and it raises our verified best from 0.943 (row 56859861) to 0.949. It equals the upstream notebook's displayed public score, which is consistent with the recipe being reproduced faithfully: same selected checkpoint, same geometry and views, same probability average and rank transform, and the visible-cohort CSV identical byte for byte (see [README.md](README.md)). It is a public-split score only; the private split is unknown until the competition ends.

What the result establishes and what it does not:

- The strict reader was the cause of V13's hidden failure in the sense that matters: the same model, geometry and arithmetic completed the hidden rerun once the reference's fault handling was adopted. The hidden log remains unavailable, so the specific malformed study is unknown.
- The hidden rerun finished within about 30 minutes of acceptance, so the reader's throughput comfortably fits the competition limit.
- No new modelling is claimed. The predeclared next candidate is in [frontier-audits/2026-10-08-public-0950](../frontier-audits/2026-10-08-public-0950/).
