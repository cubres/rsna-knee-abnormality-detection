# Reader canary cap: the hidden-rerun failure, the fix, and today's rows (2026-10-09)

This note records measurements and decisions. The count-argument defect is also analysed in `independent-review-2026-10-09/`. That review proposes a command-only repair (omit the study count from inference commands). The fix recorded here changes the reader instead. Both avoid the refusal. Where the two differ, this note says so.

## 1. Root cause

- The reader's argument validation requires `1 <= max_studies <= 16`, and it applies in inference mode as well as canary mode. Inference ignores the value and processes every study.
- Each bench chain passed its study count as `--max-studies`. On the visible commit the count is three, so the check passes. On a hidden rerun the count is about 1,300, so the reader refuses at startup, before any GPU work.
- Reproduced: the reader body used by V22 and V23 refuses 1,300 studies with "Canary study count must be 1..16". The same body with the cap limited to canary mode runs 20 unseen studies to completion on CPU.
- Effect: the arm writes no file, the root step finds none, and the notebook ends. Kaggle reports "completed, submission file not found". V22 (row 57016738) and V23 (row 57016750) were COMPLETE by 17:57 UTC, about three minutes after submission, with no submission file and no score. The rerun log text has not been fetched, so the cause rests on the code, the reproduction and the timing.
- The bench V19 chain passed the same count to its mirror arm. Its row can therefore only measure BK-A on its own at best. The MaVIT probe's mirror component was affected in the same way.
- The public mirror notebook was not affected. Its launcher passes a canary value of 3, which the reader accepts. Inference ignores the value and processes every study, which is how the 0.949 rerun ran.

## 2. Fix (built and tested offline)

- Reader: the study cap applies only in canary mode. The change is one line. Comments and docstrings were pruned under an AST-identity check, and the reader's pin was recorded.
- Visible gates run only when the test set has exactly three studies whose identifiers hash to the three visible identifiers. The notebook stores hashes, not identifiers. A mismatch falls back to the reference path. It does not stop the run.
- Candidate first, then reference. On a candidate failure the reference path runs, provided the remaining budget covers 3.5 seconds per study plus five minutes. For a plain-view arm the reference is the reader's reference plain half. For a four-view arm it is the reader's two-view run.
- The root is written last, exclusively, and always when the reference path succeeded. The receipt field `root_source` names the path: `candidate`, `reference_fallback`, `blend`, `bka_only`, `mirror_only_fallback_v36` or `none`.
- BK-A chain: a mirror failure gives a BK-A-only root (`bka_only`). A BK-A gate failure gives the mirror-only root, which is byte-identical to the public version 36 output. The gate needs 27,000 s remaining on a rerun of about 1,300 studies, and 1,333 s on a three-study commit.
- Decision: the degraded-study fraction (0.5%) is recorded in the receipt and no longer blocks the root. The earlier chains failed above that fraction.
- Tests: reader cap tests (1,300 studies in inference pass the cap; the earlier body refuses 1,300; canary mode refuses 20). A CPU run of 20 unseen studies with speed flags and the parity gate completes with one row per study. Chain harness scenarios cover commit, rerun on 1,300 unseen studies (candidate, fallback, BK-A-only and mirror-only roots), internal mismatch, candidate crash, visible mismatch and total failure. Replays with no GPU end fatal at the root step, as designed.

## 3. Today's rows (as reported to the campaign)

| Row | Submission | Status | Note |
|---|---|---|---|
| 57016738 | bench V22, plain-view | COMPLETE, no submission file | cap refusal (section 1) |
| 57016750 | bench V23, four-view | COMPLETE, no submission file | cap refusal (section 1) |
| 57016729 | bench V19, BK-A and mirror | pending | mirror arm refused at the cap; at best BK-A alone |
| 57016763 | MaVIT probe, version 3 | COMPLETE | standalone 0.935, below the 0.943 gate; dropped as a lever, consistent with weak readers adding nothing to the blend |
| 57017206 | bench V25, four-view, rerun-safe | pending | see the open item in section 5 |

- V24 (plain-view, rerun-safe) and the corrected BK-A and mirror chain on the second private bench worker are queued for tomorrow.

## 4. Fine-tune session readiness

The fine-tune is built as a new version of the private trainer. The candidate is 576,656 bytes, and its 17 CPU tests pass on Python 3.12, torch 2.8.0, timm 1.0.19 and pydicom 3.0.1. Dry runs for the control, gemlow and lexblend arms, and for the canary, pass on random initialisation. Nothing has run on GPU yet. Sessions run one after another on one pair of T4 GPUs. A canary of 600 seconds comes first: 8 fit and 8 out-of-fold studies, one epoch, two steps. It measures preparation time per study and GPU time per step, and it checks that the training DICOM folder exists. Control, gemlow and lexblend follow at about 9,600 quota-seconds each, then three public inference rows at about 900 quota-seconds. The hidden reruns are not charged. The central total is about 29,300 quota-seconds (range 22,000 to 41,000), which is 41% of a 72,000 allowance. Caps: a soft budget of 25,200 s checked per micro-batch, a hard budget of 30,000 s enforced by a daemon that exits the process, and Kaggle's 32,400 s limit. A checkpoint is written after every epoch. The arms are capped at three additional submissions within the five-per-day limit. Gates: G0 (validity: receipt status, step count, build failures at most 0.5% of fit studies, checkpoint digest recorded); G1 (the control's public row must fall in [0.948, 0.950]); G2 (an arm is adopted only if its public score rises by at least 0.002 over control and its out-of-fold fold-0 AUC does not fall). Before a live run: the host ruling on the OAI-derived checkpoint is still open, the training DICOM folder is unverified (the trainer fails closed if it is missing), and the shipped trainer cannot start from the SWA checkpoint, so the arms start from the base checkpoint. Pushing this version also makes the notebook's default run HOLD, which stops the affine trial in that notebook. That needs an explicit decision.

## 5. Open items

- Bench V25's four-view arm runs the V23 driver, which switches the shared reader's input resolution (320 to 384) while consumer threads may be predicting. The independent review in `independent-review-2026-10-09/` reproduces a concurrency case in that path. The reruns made in this note cover the cap, the fallbacks and the root logic only. They do not qualify the concurrent path. V23's visible runs made no post-parity predictions, so they never exercised it. Row 57017206 should be read with this in mind until a serial comparison and a runtime qualification have passed.
- Bench V19 and the MaVIT probe: their mirror components were refused at the cap, so their roots, when they finish, describe BK-A alone or the MaVIT standalone table.
- The public version 37 candidate passes the canary value 3 to its mirror arm. It has not been pushed.
