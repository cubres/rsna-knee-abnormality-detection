# Knee status, 2026-10-08 (night): measurements and decisions not yet recorded

This note consolidates what has been measured or decided since the last dated folder and is not yet in this repository. Times are UTC. Scores are official Kaggle scores where stated; anything marked planned has not happened yet. No score is claimed for a candidate that has not been submitted.

## Four-view test-time augmentation on the private bench (bench V23)

- Design: the V36 mirror arm at 320 px (plain and mirror) plus the same checkpoint at 384 px full field (plain and mirror). The four probability vectors are averaged with equal weights. The equal weights were predeclared before the run.
- Run: COMPLETE after 454 s on two T4 GPUs. Chain status PASS_V23_FOUR_VIEW.
- Timing on the three visible studies: 384 px plain view 1.646 s per view per study; 384 px mirror view 1.671 s per view per study.
- Two-view self-check: PASS. The four-view output CSV equals the two-view reference CSV (sha256 prefix faa20e95). With three studies the export ranks are 0, 0.5 or 1, so this equality says little about accuracy.
- Planned: submission at 00:00Z 2026-10-09 as slot 3.

## MaVIT third-lineage probe (private bench worker)

- Version 2 failed closed: "MaVIT not mounted" (a mount-path miss). The root was written as the constant 0.5 table (470 bytes), so no score can come from it.
- Version 3 clean: the MaVIT arm reported ok, not degraded, no failure. The run was COMPLETE after 454 s. The standalone table and a staged 0.80 mirror / 0.20 MaVIT blend were both produced.
- Licence decision: the dataset that carries the weights is MIT, but the upstream MALA encoder is unlicensed. The MaVIT lineage is a private probe only and is not for any public notebook.
- Planned: the standalone row at 00:00Z 2026-10-09 as slot 4. Gate: a standalone score of at least 0.943 to keep the lineage.

## Public candidates (built offline, dry run clean, nothing pushed)

- Public V37, the BK-A port (Rad branch off) plus the predeclared 0.55 / 0.45 rank blend with the mirror arm. The version 36 mirror-only root is the fallback when the BK-A time gate does not pass, and the chain receipt records which path wrote the root.
  - Wire 1,026,483 bytes against the 1,036,069 limit (headroom 9,586). An offload variant with the dormant student glue moved out is 992,804 bytes.
  - Checks: static checks, a chain harness with a BK-A stub over six scenarios, a cell replay (ends in the designed fail-closed state without a GPU), reader CPU checks, and a safe_push dry run with is_private False (DRY_RUN_OK, 10 dataset sources, no kernel sources).
- Public V38, the four-view port on the public lineage (320 px plain and mirror as in version 36, plus 384 px full-field plain and mirror with equal weights).
  - Wire 812,098 bytes, 145 cells. Safe_push dry run DRY_RUN_OK with is_private False.
- Push rule (decision): a public candidate is pushed only if its bench row scores at least 0.950, judged on Saturday 2026-10-10.

## Fine-tune programme (design only, not run)

- Three arms on the same checkpoint family, each for three epochs: control, gemlow (label variant gemlow) and lexblend (label variant lexblend, with a lexicon override where the blend and the lexicon disagree by at least 0.5).
- Cost: about 29,300 quota-seconds central (range 22,000 to 41,000), which is 41% of a 72,000 quota-second allowance. Work starts 2026-10-10.
- Gate G1 (control validity): the control's public row must fall in [0.948, 0.950]. Outside that band the procedure itself moved the model, and the arms are read against that.
- Gate G2 (adoption, per arm against control): adopt only if the public score rises by at least +0.002 and the fold-0 out-of-fold macro AUC does not fall below the base checkpoint. Rows are rounded to 0.001, so +0.002 is two ticks. G2 is a threshold, not a significance test.
- Engineering findings:
  - The training script cannot resume from the SWA checkpoint as shipped. Its loader copies only backbone tensors from a dictionary under student or model keys, with strict=False.
  - The training script has no flag to disable the OAI-derived masked supervision. Every arm inherits it.
- Open risk: the checkpoint is OAI-derived and its release says OAI access remains governed by the NDA terms. The host ruling is open, so eligibility is not settled.

## Strategy sweep verdict

- No inference-time lever beyond one tick (0.001) survived the sweep. The measured paths are the mirror arm, the BK-A family, and the four-view variants above.
- Probability of a final at or above 0.951 by 2026-10-22: about 6% (range 4% to 10%). Component estimates from the sweep: V23 four-view at or above 0.950 about 0.40, at or above 0.951 about 0.04; V19 public at or above 0.951 about 0.08. Stacking is unmeasured. The planning assumption is 0.949 or 0.950 as the realistic outcome.
- Plan: two finals. One is the efficiency entry, which must be one of the two. The other is an accuracy hedge. The efficiency entry follows the adopt rule in efficiency-track-2026-10-08-v22. From 2026-10-18 only pre-verified submissions are used, and the account owner selects the two finals in the Kaggle interface.

## Not yet recorded here

- Bench V19 row (scores 2026-10-09). The decision on the public chain follows it.
- Planned submissions at 00:00Z 2026-10-09: four-view slot 3, MaVIT standalone slot 4.
- Host ruling on OAI eligibility.
