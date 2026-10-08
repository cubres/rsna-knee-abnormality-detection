# Efficiency track: the measured fast-mirror path, and why we are not publishing it (2026-10-08)

On 2026-10-08 we ran a private bench version of the mirror arm that tries to be faster without changing its output. The public notebook stays as it is: no fast-mirror version is published, and no score is claimed.

## What was measured

- The gate accepted one approximate flag (fused two-view TTA) alongside three exact flags (shared gather, pooled header and pixel reads). The bench's mirror output is byte-identical to the public mirror output (CSV sha256 `faa20e95…`). The largest per-cell probability deviation measured was 6.1e-5, against a predeclared tolerance of 1e-4.
- Chunk size 32 and channels-last failed the gate, at about 1.5e-4 each.
- Steady-state study time (studies 1 and 2 on the three visible studies, one T4) is 4.3% lower with the accepted path. The GPU forward is 3.0% slower, and only the preprocessing is faster. Preprocessing runs under the GPU in the pipeline, so the saving does not reach the total.
- The parity gate itself costs 95 seconds on the commit, on the first GPU only.

## Projection for about 1,300 studies on two T4 GPUs

- Reference path (no gate): 1,287 s if preprocessing overlaps the GPU, 2,013 s if it does not.
- Accepted path (gated): 1,411 s and 2,015 s respectively. In the overlapped case it is about 124 s slower.
- Efficiency at AUC 0.949 (lower is better): the reference scores −2.023 and the accepted path −2.019. A 0.001 AUC step is worth about 70 seconds of runtime, so the accepted path is about 1.8 AUC steps worse.

## Decision

- We are not publishing a fast-mirror version. Its speed-up does not improve the efficiency score.
- The lever that does move the score is the view count, not kernel speed. Dropping the mirror view removes about half the GPU time. It is worth it only if its private AUC cost is below about 0.0075. The public ladder suggests the mirror step is worth about 0.001. That claim is not yet measured on the private set, so the next step is to measure the plain-only reader on the private bench.

Analysis and evidence are kept in the takeover workspace, which is not part of this repository.
