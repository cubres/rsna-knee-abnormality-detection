# Mirror reader V12: native runtime check

The exact accepted V12 canary completed native inference on **three visible studies**. For those studies, the constructed volume/mask bytes matched the reference and all twelve label rankings matched its visible reference ordering. Whole native elapsed time was **53.630134489 seconds**.

The reader source SHA was `4c0f5e4ef362941927e79ca3ab7e98a2c17a46d5a9230928b9057ce3e4a7b387`, identical to the earlier published [original reader](../public-mirror-reader-2026-10-08/src/public_mirror_reader.py). The worker and checkpoint have separate hashes in the scalar evidence. A successful scientific-worker return and an ownership-bound teardown were observed; this note does not describe the supervisor as having a normal zero exit.

The observed runtime included torch 2.11.0+cu128, timm 1.0.29 and NumPy 2.1.3. Two Tesla T4 devices were visible. Visibility does not establish that both were used, and timing three studies does not estimate full-test throughput.

## What can be reused

The [source-only release](../public-mirror-reader-2026-10-08/README.md) explains anatomical mirroring before normalization, source/schema checks and model usage. Its implementation is original and explicitly attributes nartaa's inference recipe and learned checkpoint. This canary adds a measured runtime and preprocessing/ranking check to that earlier release.

Three studies provide little power to check ranking differences. The result does not establish probability-byte equivalence, full population parity, hidden-test parity or a reproduced leaderboard score. No competition submission or own new official score resulted from V12. The public notebook scorecard 0.949 belongs to the upstream nartaa V2 notebook.

A full inference release must complete exact coverage and schema validation, and any resulting competition score must be recorded from its exact completed official row. The full source/metadata and raw canary predictions are kept outside this public aggregate packet.
