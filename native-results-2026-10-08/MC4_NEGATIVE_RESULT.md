# MC4 V59: successful execution, retain the unchanged reference

Both fixed four-epoch fine-tunes completed two networks each, with **3,488 actual AdamW updates per network**. All three prediction arms covered the same ordered 862 studies and twelve findings, and their prediction files were sealed before the held target numerics were opened. Shared training took **4,803.706 seconds**; total scientific execution took **5,694.458 seconds**.

| Prediction arm | Macro report-proxy ROC AUC |
|---|---:|
| Unchanged two-network initialization | 0.879416982 |
| Plain BCE fine-tune | 0.876943525 |
| Study-coherent mild affine BCE fine-tune | 0.877472933 |

| Paired comparison | Macro AUC difference | Pointwise descriptive 95% interval |
|---|---:|---:|
| Mild affine minus plain | +0.000529408 | [−0.000869929, +0.001928930] |
| Plain minus unchanged | −0.002473457 | [−0.004198660, −0.000867229] |
| Mild affine minus unchanged | −0.001944049 | [−0.003233649, −0.000624841] |

The registered mild-versus-plain interval includes zero, and both trained ensembles ranked below the unchanged initialization. No final fine-tune is promoted. Preserve their checkpoints: a rejected candidate can still explain where a training recipe loses ranking quality.

## What this experiment teaches

Checking only the two fine-tunes would have hidden their shared degradation. The unchanged initialization is a useful third prediction arm because it measures whether further fitting improved the starting point at all. Matching initial states, training length and prediction order also makes the difference interpretable within this experiment.

This result does not establish why the models degraded. Learning rate, additional training, target construction, initialization history and augmentation remain candidate explanations. A useful next test is inference with the preserved epoch-one and epoch-two checkpoints, registered before inspecting their results. That can locate when ranking changes without repeating a full fit. Then compare one material adaptation at a time, such as shorter or lower-rate fitting, against the frozen reference on an independent patient-group evaluation when provenance permits.

## Limits and reproducibility

The panel was previously exposed. Report overlap is unknown, patient independence is unproven, and report-derived proxies are not independent structured MRI annotations. This is neither OOF nor clinical validation. A single training seed/device assignment was used.

The paired whole-study bootstrap requested and obtained 2,000 valid resamples, with zero rejected draws. Intervals are conditional on surviving class support, pointwise and descriptive; they are not simultaneous or confirmatory and do not quantify training-seed variability. The plot shows aggregate comparisons only.

The public [scalar evidence](evidence/aggregate_results.json) contains exact arm means, deltas, intervals and twelve per-finding deltas. The standard-library summarizer verifies the arithmetic and equal-weight macro aggregation. Reproducing the original AUC or bootstrap distribution would require the private study-level data; this packet does not claim to reproduce them from scalar summaries. No competition submission or official score resulted from this experiment.
