# Knee V64: a runtime hold without a restart checkpoint

This is a dated follow-up to the [optimizer-call witness](https://github.com/cubres/rsna-knee-abnormality-detection/tree/52023b65c4d4d8820c2a63137d79338a626d8448/optimizer-call-witness-2026-10-11). The accepted training worker V64, native session **357083876**, was refreshed read-only on October 11, 2026. Its kernel status is **ERROR**, while its retained training receipt explicitly declares **HOLD_PROJECTED_OVER_SOFT**. The source deliberately exits with code 2 for this hold.

| Receipt observation | Value |
|---|---:|
| Consumed batch groups | 50 / 434 |
| Elapsed time | 849.02 seconds |
| Projected session time | 8640.4 seconds |
| Soft budget | 8400 seconds |
| Recorded process-tree memory metric, first to last | 2021.8 to 6494.8 MiB |
| Recorded host-memory percentage, first to last | 6.80 to 21.71% |
| Exported resumable checkpoints | 0 |

This version does **not** establish an OOM or an AMP overflow. Its projection hold returns during group 50 before the first worker-recycle cleanup. Thus the receipt cannot show whether recycling would reset memory. The exact kind of process-tree memory metric is not declared; these samples must not be renamed as unique physical RSS or used to extrapolate an OOM time.

The complete output listing contains only the training receipt. The prepared continuation looks for a checkpoint and sidecar, but can fall back to its epoch-1 initializer when none exists. Attaching V64 outputs is therefore insufficient to admit it as a continuation. Any restart must be explicitly registered as a restart; its 50 uncheckpointed groups cannot be recovered by inventing state.

The next proposal is a bounded, contiguous checkpoint study. A nominal partition of 434 groups is 100, 100, 100, 100 and 34. It must project the registered chunk horizon, checkpoint before a planned stop, require the exact predecessor for every continuation, preserve optimizer/scaler/RNG/data order and the absolute schedule, and defer OOF evaluation until the epoch finishes. Merely setting a shorter stop can still trigger the old full-epoch projection at group 50. The first recycle boundary must be measured before expanding. No such successor has run in this audit, and numerical equivalence across process restarts remains unproven.

Separately, the best completed account submission in the **16:52:19 UTC October 11** inventory remains **0.949**. The exact latest row is **57043015**, COMPLETE, bench worker V10/native **357057303**, CSV SHA256 **381253ef942b8f7da0488b13f9e6aabed6a9b660ae40e68b17533a9eba14b37b**. It credits the existing anatomical mirror and goodpjw2008 ConvNeXt 2.5D reader, with the per-finding lineage credited to sujanmajhisuzan. This result matches the established 0.949 score; it proves no new gain from the training recovery or CPU witness.

[EVIDENCE.json](EVIDENCE.json) pins the privately retained source, exact receipt, fresh status and official-row snapshot. Only aggregate observations and hashes are published; no medical data, labels, weights, notebook source or third-party implementation is redistributed. All notebooks remain intact. The run coordinator owns subsequent launches and submissions.
