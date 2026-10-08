# Official outcome of V13, root cause, and the V14 fault-tolerant successor

Recorded 2026-10-08 10:21 UTC from raw server rows. No score is claimed in this note.

## 1. What the V13 submission actually returned

Submission **56923965** (V13, submitted 2026-10-07 23:22 UTC) finished with raw status **COMPLETE** and this error description:

> Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset

No public score was returned. Our verified official best therefore remains **0.943** (row 56859861). The upstream 0.949 stays credited to [nartaa's notebook](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397). Exact row: [evidence/row_56923965_final.json](evidence/row_56923965_final.json).

## 2. Root cause (from source; the hidden log is not retrievable)

The V13 reader (`public-mirror-reader-2026-10-08/src/public_mirror_reader.py`) was deliberately strict, and the strictness was fatal on a cohort it had never seen:

| V13 behaviour | Upstream reference behaviour |
|---|---|
| `pydicom.dcmread(stop_before_pixels=True)` with no `try`; non-finite geometry raises | header exception keeps the file as a neutral `(0.0, path, 0.5)` record |
| pixel decode errors, non-2D or non-finite arrays raise | pixel exception returns `None`; that slice is skipped |
| a study with no usable series raises ("No usable acquisition") | zero volume and zero mask, still predicted |
| any per-study model error is collected and then fatal (`require(not errors)`) | that study keeps 0.5 probabilities |
| exactly two T4 GPUs required | `torch.cuda.device_count()` used as found |
| a study with no `test_series.csv` rows raises | zero volume, still predicted |

Each row was reproduced offline on synthetic DICOM: the V13 reader raised on 7 of 9 faulty cases and its end-to-end run died; the reference semantics complete. With about 1,300 hidden studies, one malformed or unusual study is enough. This is the most probable cause, not a proven one; memory, timeout and environment differences are not excluded by evidence.

## 3. V14: same recipe, reference fault semantics

[src/public_mirror_reader.py](src/public_mirror_reader.py) (SHA-256 `e9ec648e1b2f6121f36bbe6ae806a5714571db83804c719ea02d5c4bf7fe47d7`) differs from V13 only in fault handling and device count; the happy-path arithmetic is unchanged. The full diff is [src/public_mirror_reader_v13_to_v14.diff](src/public_mirror_reader_v13_to_v14.diff). Launcher changes: the first-three-study parity comparator is recorded instead of blocking, the `>=3 studies` gate became `>=1`, and the owned watchdog deadline is aligned to the 32,350 s rerun budget (commit budget 1,750 s unchanged).

Offline proof ([src/test_v14_offline.py](src/test_v14_offline.py), results in [evidence/v14_offline_test_results.json](evidence/v14_offline_test_results.json)): happy-path submissions of V13 and V14 are byte-identical; V14 volumes equal the reference builders on 12 synthetic cases (truncated headers and pixels, garbage files, multi-frame, NaN, missing series directory, missing series rows, all-garbage series, two-plane studies); V14 never raises; an injected model failure becomes a 0.5 row; one-GPU and two-GPU outputs are identical.

## 4. V14 visible run on Kaggle (2026-10-08 10:18 UTC)

| Check | Result |
|---|---|
| `submission.csv` SHA-256 equals V13's `faa20e95…` | yes |
| header / pixel / study-build / model fallbacks / consumer errors | 0 / 0 / 0 / 0 / 0 |
| first-three volume and mask parity | PASS |
| devices | two Tesla T4, Torch 2.11.0+cu128, timm 1.0.29 |
| reader `run()` elapsed, 3 studies | 20.3 s |

Receipts: [evidence/v14_visible_native_receipt.json](evidence/v14_visible_native_receipt.json), [evidence/v14_visible_coordinator_verify.json](evidence/v14_visible_coordinator_verify.json).

## 5. Official submission of V14

Submission **56948803** was accepted at 2026-10-08 10:20:42 UTC from [V14](https://www.kaggle.com/code/prvsiyan/bee-s-knees-rsna-knees-final-push?scriptVersionId=356365376) (notebook SHA-256 `73958c9f612721d0df368d7709a5ee82379ed812e68302d54ee42cf733bb87a5`). Its status field was absent at acceptance, which means UNKNOWN. A completed row and a non-empty score will be recorded in a separately dated note; until then no result is claimed. Receipt: [evidence/row_56948803_accepted.json](evidence/row_56948803_accepted.json).

## Credits and licences

The reader is an original implementation of the public Raptor CoAtNet/MIL anatomical-mirror recipe published by nartaa (Danial Zakaria) under Apache-2.0; the selected SWA checkpoint comes from the publisher's [weights dataset V1](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007/versions/1) under its own "Other" terms (research and educational use, including this competition). This repository distributes no weights, MRI data, identifiers or prediction values. The complete attribution and data acknowledgements in [public-mirror-reader-2026-10-08/NOTICE.md](../public-mirror-reader-2026-10-08/NOTICE.md) and [LICENSE_SCOPE.md](../public-mirror-reader-2026-10-08/LICENSE_SCOPE.md) apply unchanged. Source files here are Apache-2.0 as marked in their headers; this documentation is MIT like its siblings.
