# One experiment after faithful 0.957 reproduction

Add anatomical mirror averaging to **Raptor224 only**, with zero training. Complete and validate the faithful baseline first; this document is a source-only proposal, not a launch, measured improvement or promise of 0.958.

## Direct evidence and attribution

- [Helios V1/native 358332966](https://www.kaggle.com/code/heliosli/this-is-new-version-0-9576-in-one-click?scriptVersionId=358332966) has a previously observed visible **0.957** badge, not independently established 0.9576 precision. Exact source SHA256: `16a9d953972d9d82717f279f75e57e798d3902f93d1071c30f3caa67326e3601`. Keep historical [R6 dataset V1](https://www.kaggle.com/datasets/heliosli/rsna-knee-r6-frozen-fe055406/versions/1): latest V2 changes the model family. R6 already has flip TTA and Raptor384 already has anatomical mirror TTA; Raptor224 is single view.
- [varshithakolupuri V1/native 358309699](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up?scriptVersionId=358309699), exact source SHA256 `d271973c53e6e327aede374d54740f7cfbbe1ced3dbe44231a9da552b9489787`, provides the closest direct public contrast. Root's 2026-10-15 15:25 UTC native observation bound Apache-2.0, public/best **0.956**, T4x2 and 484.7-second successful public example run. Rank 357→289 is the author's report; accepted-row causality was not verified. This is evidence for a candidate, not a verified final winning solution or proof of improvement over R6 0.957. Fresh web text retrieval failed, so those facts retain their explicit earlier observation time.
- The 224 mirror helpers are identical to the scored 384 helpers. Preserve credits to varshithakolupuri, nartaa, dreaddevelopment and Helios. Use original public mounts with their existing terms and required OAI acknowledgement; this proposal distributes no checkpoint or OAI/competition data.

## Frozen contrast

```text
same study / centres / mask
          ├─ plain uint8 triplets ────────────┐
          └─ anatomical mirror before LUT ──┤
                                            ↓
                         same LUT → crop .8 → 224 model → sigmoid
                                            ↓
                                 mean of two probability vectors
                                            ↓
              unchanged ranks and blend: .5 R6 + .325 Raptor384 + .175 Raptor224
```

Slots are `[26,22,18,12,18]`; the centre slice determines the axis. Slots 0–1 reverse triplet channels; slots 2–4 reverse width. No output-label permutation, logit averaging, new crop, new weights or other arm change. Preserve the baseline's raw prediction dumps and required-arm failure handling; the source donor's optional-arm behavior is not part of this transplant. Keep Raptor224 checkpoint SHA256 `3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6`.

## What can actually be measured

One paired three-study T4x2 canary, **900 seconds wall time**, after baseline parity. Synthetic geometry checks precede model work. Require identical row/label order, finite probabilities, no extra fallbacks, the frozen plain-224 control and unchanged R6/384 predictions within the existing `1e-7` gate; verify the paired mean directly. A rank CSV may remain unchanged on three studies. Measure the extra preparation **and** forward time with CUDA synchronization, process RSS, and per-device allocated/reserved peaks; control-pass overhead is logged separately. GPU allocator peaks alone are not total device memory. Preserve the current 28 GiB process-RSS cap and ownership/closure guards. No CPU-worker change belongs in this contrast.

No new training, download or tuning budget: **0**. No blend/threshold search and no Gold58 model selection. Gold58 was reused for Raptor checkpoint and label choices; R6 held-fold predictions mixed with those Raptor weights are not full-ensemble OOF. Three public example studies cannot establish AUC gain. The useful immediate endpoint is correct implementation and measured incremental cost, followed by at most **one** root-authorized native score comparison. Public-LB movement would remain a descriptive result, not an independent causal or clinical claim.

For leakage-safe labeled validation, require a patient-disjoint set absent from every arm's representation training, gradient training, label revision, checkpoint selection and blend tuning; no such set is currently qualified. A clean study-UID audit is necessary but insufficient. If future retraining is chosen, partition patients before all task training and selection, use common outer folds across **all** branches and inner-only epoch selection, then produce routed outer predictions with lineage checks. Starting from already exposed task checkpoints would not repair leakage. That is a separate budgeted project and is excluded here; do not label today's existing-fold test blind. See the original [UID exposure tool](https://github.com/cubres/rsna-knee-abnormality-detection/tree/main/tools/uid-exposure-audit/2026-10-15) and [Raptor publication account](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792).

Only after the paired canary passes, project full runtime using the **completed full baseline** stage timings plus measured mirror cost. Public three-study runtime is not a full-run bound. One future full run is capped at **29,880 seconds (8.3 hours)**, with 30 minutes of validation reserve inside the nine-hour platform window and sufficient remaining coordinator allocation. Hold if either timing evidence or budget is absent. Root alone schedules, preserves notebook identity/artwork, uses unchanged safe_push and performs any once-only native-menu submission; accepted/failed/ambiguous attempts remain consumed.

## Useful artifact now

`synthetic_geometry.py` supplies ten deterministic invented boundary fixtures, an independent pixel oracle, six refusal cases, a double-reflection check and a before-LUT counterexample. It is original MIT code with a runnable stdlib demo; generated JSON is MIT as well. It contains no real identifiers, labels, predictions, images, reports, weights or private paths. This can be reviewed and later added to the existing public repository without a new Kaggle dataset identity. The artifact validates geometry contracts only; no accuracy result has been produced.
