# V16 predeclaration (knee): one candidate, fixed before any build or run

Author: Claude frontier analyst. Date: 2026-10-08 (UTC). Status: PREDECLARED. Nothing has been built, pushed, run or submitted. Changing any formula, weight, gate or rule below after a V16 result exists requires a new predeclaration (V17), not an edit.
Evidence labels: [V] verified from source, receipts or a quoted public document; [I] inference; [P] author claim.
Companion: COMPOSITION.md (public notebook evidence) and NOTES.md (log).

## 0. The candidate

Two arms, both on hidden test (about 1,300 studies, 2 x T4, 9 h = 32,400 s).

- **A = BK-A with the RadImageNet branch off.** This is our 0.943 stack (T = DINO-20 + A5, R = Raptor 4 views at 94 windows, C = four CoAt readers, fused as in the brief section 2.2) with the Rad branch removed. It is the V6 configuration ("T without the RadImageNet branch"). Reason in section 1.
- **M = V14 mirror arm.** Our fault-tolerant reimplementation of nartaa's 384 accuracy reader: checkpoint raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt from nartaa/rsna-knee-publication-swa-weights-20261007 (sha256 7e5315dad125...), native 320 centre crop, K94, probabilities averaged equally over the plain and anatomical-mirror view, sigmoid per finding.

**Formula (exact).** For each finding j and study i, with ranks computed over the test studies using pandas `rank(method="average", pct=True)`, per column:

    r_A = rank(A)            (per finding)
    r_M = rank(M)            (per finding)
    S   = 0.55 * r_A + 0.45 * r_M       (one global weight, all 12 findings)
    F   = rank(S)            (per finding, same rank convention)

Output: `StudyInstanceUID` plus the 12 findings in sample_submission column order, row order of test.csv. Required checks: no NaN or Inf; every value in [0, 1]; each column has more than one distinct value; StudyInstanceUID equals the test IDs.

**Weight.** w_M = 0.45, global. No per-label weights. No tie jitter. The weight is fixed here and may not be tuned on the public board (section 6).

## 1. Why this candidate

Evidence [V]:
- The public card for nartaa's accuracy notebook is 0.949 with a single 384 checkpoint and mirror TTA (author ladder: 0.948 to 0.949 from the mirror on identical weights). M is that reader.
- Our BK-A stack scores 0.942 to 0.943 officially (orchestrator brief; receipts show PENDING at submit time, so I did not verify the scores).
- Public blends at one global weight: goodpjw's 2.5D ConvNeXt (0.929 alone) at 0.30 into the 0.943 stack gives 0.944. pjmathematician's private fleet at 0.45 gives 0.946 on the 0.943 family (no mirror in that notebook; its fleet assets return 403 for us). kozykappa uses 0.22 for a ConvNeXt, gated on 58 gold studies.
- Our own ConvNeXt student fleet at 15% (V8) gave no tick (0.943 with or without it); the student alone scored 0.918.

Inference [I]:
- The weight should scale with the added arm's strength. goodpjw's reader is 0.929 alone, so 0.30 fits a weak arm. M is about 0.02 AUC stronger than that reader and about 0.006 stronger than the stack, so 0.30 would under-use it. 0.45 is the only public precedent with a strong, separate arm.
- I did not choose 0.50. Equal weight would give a 0.943 stack the same say as a 0.949 reader whose hidden-run score is not known yet (see G1). Keeping the stack as the majority arm is the conservative choice.
- Expected gain over the better arm: unknown, somewhere between 0 and about +0.003. The two readers are both CoAt-Raptor lineage, so their errors may be correlated and the gain small.

Excluded, with reasons:
- Per-label weights (as in the matterhorn family): tuned on 58 gold studies or on the public board. A 12-weight search is not supported by the data (a public author's own note says so, for 870 studies). Excluded.
- Student fleet (V8 component): no measured gain, and its runtime is not in our receipts. Excluded.
- Rad branch: the RadImageNet encoder is CC BY-NC-SA 4.0 (marwanmath/resnet-50-radimagenet-marwan), and a prize-eligibility question under rule 6.c is open (forum thread 745283, no host answer found). Cost of dropping it: about +0.001 by the brief's estimate [I]. Saves about 0.35 h hidden (cost model, section 4). Dropping it keeps this candidate usable if the host rules against NC-SA weights.
- r224 arm: optional extension only, not in V16 (section 5).

## 2. Gates before the build

All must hold. Any failure means no V16 build, and the coordinator is told why.

- **G1. Mirror arm proven on hidden.** Official row of V14 (56948803) has been read and is at least 0.947 public. Reason: V13 failed on the hidden run; V14 is its fix, and its row is the first hidden evidence for M. Status: pending according to the orchestrator brief (not re-checked by me).
- **G2. Stack baseline.** Official row of V6 (56795572, Rad off) has been read and is at least 0.940. If it is not, the Rad-off base is too weak and the Rad question must be reopened first.
- **G3. Mirror parity on the 3 visible studies.** A real GPU run of V14 on the 3 visible studies reproduces nartaa's reference probabilities within 1e-3 absolute (predeclared tolerance). Reason: V14's offline proof uses synthetic volumes and no real weights (knee/NOTES.md). This is a parity check, not a score.
- **G4. Eligibility decision by the coordinator, in writing.** M is built on nartaa's checkpoint, whose training used OAI masked supervision. The host ruled on 2026-09-21 that OAI use does not break the rules if it is free and generally accessible without institutional review. OAI's own terms are disclaimed by the host, and a follow-up on OAI access was posted 2026-10-08 with no answer. Rule 2.8 (weights and code for winners) is also unanswered (threads 743420, 744056). This gate is a decision, not a technical check. Without it, V16 can be run as a research row but cannot be selected as final.
- **G5. Quota.** At least 7.5 h of GPU quota remains in the current weekly window. The quota is 21 h per week (75,600 s). The brief says it resets Saturdays 00:00 UTC, so the next resets are 10-10 and 10-17. Two receipts give conflicting usage (61,486 s on 10-04, 1,759 s on 10-05), so read the live value before scheduling.

## 3. Build and run rules

- Build from a clean BK-A (Rad off) notebook plus the V14 reader modules. Do not carry V13's 39 preserved cells. The V14 candidate is 828 KB, close to our 850 KB builder margin; BK-A alone is about 159 KB.
- Order inside the run:
  1. BK-A stack (A). Compute r_A.
  2. Write submission.csv = r_A as the fallback, before M starts.
  3. Time gate for M (section 4). If it fails, stop here: the run is V16_DEGRADED_TO_A.
  4. M (V14 mirror arm). Compute r_M.
  5. Blend per section 0. Write submission.csv. Write the receipt.
- Mirror time gate: M starts only if elapsed time is at most 27,900 s (7.75 h). That equals 32,400 minus the 40-minute safety reserve, minus 5 minutes for finalising, minus the 1,800 s budget for M. M's own watchdog is min(2 x 1,800 s, remaining).
- M failure policy: study-level fallbacks (V14 0.5 rows) are allowed up to 0.5% of test studies. Above that, or on any arm exception, publish A alone and mark V16_DEGRADED_TO_A.
- Diagnostics logged, not used for decisions: per-finding Spearman correlation between r_A and r_M on the hidden test. A mean above 0.95 means the arms are near-duplicates; that is recorded, and the decision rule in section 6 still applies unchanged.

## 4. Runtime budget (hidden test, 1,300 studies)

Model costs from the BK-A notebook COST table (fixed + per study x 1,300), which is also the brief's efficiency-table basis [I for the constants]:

| Block | Model cost | Seconds |
|---|---|---|
| inventory | 30 + 0.5 x 1300 | 680 |
| T: DINO-20 | 120 + 1.6 x 1300 | 2,200 |
| T: A5 | 90 + 0.7 x 1300 | 1,000 |
| R: Raptor (4 views, 94 windows) | 120 + 3.4 x 1300 | 4,540 |
| C: ResGated, D4, Global96, Repair-v1 | 3,370 + 3,210 + 3,370 + 3,370 | 13,320 |
| fusion, audit, receipt | fixed | 300 |
| **A subtotal (Rad off)** | | **22,040 = 6.12 h** |
| Rad (dropped in V16) | 90 + 0.9 x 1300 | (1,260; would give 6.47 h for A with Rad) |
| M: V14 mirror arm | gate value | 1,800 (expected 1,300 to 1,650) |
| blend and export | fixed | about 120 |
| **V16 nominal** | | **23,960 = 6.66 h** |

Checks against other evidence:
- Brief section 3.2: the efficiency bound gives 6.13 h median for the 0.943 stack, and 6.2 to 6.5 h typical. Our model with Rad is 6.47 h, consistent.
- M: nartaa's accuracy entry took 26.1 min submit-to-score, including queue and orchestration [P, author]. So 1,800 s is a safe gate, not a measurement.
- Slow draw: brief q90 7.0 h for A, plus M gives 7.5 h. Still under 9 h with about 1.5 h spare.
- Worst published: goodpjw reports 6.5 to 8.0 h submit-to-score for the 0.943 stack, including overhead [P]. With M, that is up to 8.5 h. In that case the 7.75 h gate would skip M, and the run stays at A (the fallback), which is the intended behaviour.

Drops. Nominally nothing needs to be dropped to fit 9 h. Dropped on other grounds:
- Rad branch: licence (section 1).
- Student fleet: no gain measured (section 1).
- No CoAt reader is dropped by default. Time pressure is handled by BK-A's own block gates (its COST constants) and by the mirror gate, both logged in the receipt.

Quota. One V16 official run is about 6.7 h nominal (7.5 h slow) of the 21 h weekly allowance. That allows at most one V16 run per week unless the live quota shows more.

## 5. Cheaper alternatives (predeclared, used only if V16 cannot run)

- **V16-lite (first choice if quota or the gate blocks V16).** S = 0.55 x rank(T' + R) + 0.45 x rank(M), where T' = DINO-20 + A5 with the Rad branch off, and R = Raptor. No CoAt readers. Model cost: 680 + 2,200 + 1,000 + 4,540 + 300 + 1,800 + 120 = 10,640 s = 2.96 h. Expected public score: not measured. [I] It should sit between the 0.941 DINOsaur family and the 0.949 reader; I expect no more than 0.949. Treat as a separate candidate with its own official row.
- **V14 alone (w = 1.0).** About 0.4 h. This is the cheapest arm-level candidate and implements nartaa's public 0.949 recipe. Use only if V14's official row is at least 0.948 and no blend has been run.
- **Optional r224 arm inside M (not the primary).** Replace M by 0.75 x rank(M384) + 0.25 x rank(M224), the e3 builder's weights. Adds about 0.15 h. Use only if the e3 V15 official row is at least 0.001 above V14. This is an alternative to V16, not an addition to it.

## 6. Decision rule for the official row

Let S_M = official row of V14 (56948803). Let S_A = official row of V6 (56795572). Let S16 = official row of V16 once scored. All are public scorecard values to three decimals.

1. **Select V16** if S16 is at least max(S_M, S_A) + 0.002 and S16 is at least 0.949.
2. **Tie, select V16** if S16 is within 0.001 of max(S_M, S_A) and S16 is at least 0.948. This is a judgement: the blend is the more diversified bet, and a 0.001 difference is noise.
3. **Otherwise do not select V16.** Select the better of V14 and V6 by official row. Record V16 as a negative result.
4. **One fallback run, decided before its outcome is known.** Only if S16 is in [0.945, 0.948) and at least 7.5 h of quota remain: one run with w_M = 0.30 (goodpjw's value), identical otherwise. No other weight is tried on the public board.

Notes on the rule:
- A repeat of the same notebook is not new evidence: the hidden test is fixed and the run is deterministic in intent.
- The public board is a subset of the test (about 30% is an assumption in the brief, not an API fact) and gives a rounded score. Differences of 0.001 are within noise unless repeated.
- Gold58 (58 studies, reused for selection in several cards) is a diagnostic only. It is not a decision input for this rule.

## 7. Blockers and open items

1. **V14 official row pending (G1).** The mirror arm has no hidden-run evidence yet. V13's hidden run failed.
2. **Eligibility of nartaa's checkpoint (G4).** Host ruling of 2026-09-21 (OAI use allowed if generally accessible without institutional review; the host disclaims OAI's own terms). OAI access status re-asked 2026-10-08, unanswered. Rule 2.8 unanswered. The coordinator must decide before any selection. This also applies, in part, to BK-A, whose components are third-party weights (Mattia Angeli's CC0 readers, dreaddevelopment's CC0 Raptor, pilkwang's CC0 DINO).
3. **Rad branch (rule 6.c).** The RadImageNet encoder is CC BY-NC-SA 4.0 and a host answer is pending. V16 is built Rad-off for this reason.
4. **A5 weights.** Fine-tuned from DINOv3-small. Meta's DINOv3 licence governs the upstream weights and has not been read (brief section 5.1 flag). Open.
5. **Quota.** 21 h per week, conflicting snapshots (section 2, G5).
6. **Cost constants.** The COST table is our own estimate. The first hidden-scale BK-A run is the calibration point; the receipt must record each block's real time.
7. **Receipts.** The V7 and V8 run receipts are not under bka_v*_output/ (only V3 to V6 are there). V8 submission receipts exist (refs 56818632 and 56859861).
8. **Build size.** Build clean (section 3).
9. **M parity.** No real-weights GPU run of V14 yet (G3).

## 8. Evidence versus inference, in one place

- Verified [V]: the arms, the formula, the cost constants, nartaa's ladder and runtime note, goodpjw's 0.929 / 0.944 / 30% statements, pjmathematician's 0.45 weight and 403 fleet assets, the nartaa weights licence text, the forum rulings quoted above, the quota and reset facts in the brief.
- Inference [I]: w_M = 0.45 as the best choice among 0.30, 0.45 and 0.50; expected gain of 0 to +0.003; the 0.3 to 0.45 h runtime of M; V16-lite score; the correlation between arms.
- Not verified by me: official scores of V3 to V8 and V14 (only the brief and pending receipts); the displayed public scores (rounded, one version each).
