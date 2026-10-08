# Efficiency Prize track, and the mirror-arm speed bench, 2026-10-08

This note records four things from 2026-10-08 (UTC): the completed score of the public version 35, the restored mirror-only head in version 36, the Efficiency Prize formula and the public board facts we checked, and the plan for a faster mirror arm. No new score is claimed.

## Public notebook: version 35 scored 0.948, so the head stays on version 34

- Version 35 (mirror plus the ConvNeXt blend at 0.70 / 0.30) completed with a public score of 0.948, row 56957785. The private bench twin of the same blend (row 56955785) also scored 0.948.
- The predeclared rule in [PUBLIC_V35_MIRROR_CONVNEXT_BLEND.md](../official-results-2026-10-08/PUBLIC_V35_MIRROR_CONVNEXT_BLEND.md) keeps version 35 as the head only if its completed row beats version 34's 0.949. A 0.948 is a loss, so the rule is applied and the blend is not adopted.
- Version 36 restores the mirror-only head: candidate sha256 `6180bbc1…`, 140 cells, 775 KB on the wire. Its commit run reproduced the version 34 submission byte for byte (sha256 `faa20e95…`, the CSV that scored 0.949 as row 56949940). Version 36 was not submitted, because all five Knee submission slots for the day were used and the card keeps 0.949.

## Efficiency Prize: the formula and what the public board shows

- Formula, as shown on the competition overview page on 2026-10-08: Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400, minimised on the private data. With Benchmark about 0.5 and maxAUC about 0.96, one AUC step of 0.001 is worth about 70 seconds of hidden runtime.
- Eligibility: one of the team's selected final submissions that beats `sample_submission`. Prizes are 7,000 / 6,000 / 5,000 USD for ranks 1 to 3.
- Daily public board: `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`. The snapshot we took on 2026-10-08 (notebook version 69) has no runtime column, so it ranks entries but cannot show how the formula trades runtime against AUC. In that snapshot, our team is at rank 3666 with the 0.943 row submitted on 2026-10-05. The first 0.949 entry is at rank 15, and rank 1 is 0.959.

## Mirror arm: what is measured and what is planned

- Measured: the version 36 commit run takes 39.6 seconds of native inference for the three visible studies on two T4 GPUs (receipt field `elapsed_s`). Hidden-set runtime for about 1,300 studies is not measured. Our estimate is about 25 minutes, and the estimate is labelled as such.
- Plan: a private bench candidate, the same r384 anatomical-mirror recipe with the same weights, plus speed flags. Each flag is off in the reference path. The candidate requests six flags: a shared gather for the two test-time views, pooled header and pixel reads (exact), and a fused backbone batch, a backbone chunk size and channels-last (approximate).
- Parity gate: the first three studies run through the reference path and through each candidate. A flag set is accepted only if every per-cell probability deviates from the reference by at most 1e-4. Otherwise the reader falls back to the reference path for that flag set. A study that fails on the fast path is retried on the reference path.
- Timing: a per-study, per-stage receipt with study indices only (no study identifiers), so the commit run measures the speed-up on the three visible studies.
- Status: the candidate is built offline. Its reader was tested on CPU with random weights at reduced geometry, which checks the logic but not the qualified weights or fp16 numerics. A read-only safe push dry run passed. It has not been submitted, and the GPU parity and timing on the visible studies are still pending a commit run.
