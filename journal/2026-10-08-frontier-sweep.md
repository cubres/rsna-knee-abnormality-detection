# Knee frontier sweep, 2026-10-08

A read-only sweep ran from 14:41 to 15:30 UTC on 8 October. Nothing on the competition was pushed, submitted, created, deleted or modified. A separate verification pass re-checked the sweep's claims. Its results are in section 5. Labels follow the README.

## 1. Bottom line

1. [V] The public notebook frontier was 0.950, one tick (0.001) above our 0.949. Nine notebooks displayed 0.950. They share two public artifacts: a 0.949 CoAtNet checkpoint from another author, and a 2.5D ConvNeXt reader from a second author, rank-blended in at a low weight.
2. [V] Our ConvNeXt blend at a global weight of 0.30 scored 0.948. The 0.950 forks give the ConvNeXt reader a mean weight of 0.16 to 0.22. The second author measured 0.948 at 0.30 on their own notebook [P]. The whole spread fits inside one leaderboard tick.
3. [V] A blend of our planned BK-A and mirror stack had already been run publicly by another author. Their 0.943 stack at 40% with the 0.949 reader scored 0.949, the same as the reader alone [P, author's table]. Our bench version of that idea would cost about 7 hours of hidden runtime [I].
4. [V] The real frontier is private and depends on labels. The leaderboard top was 0.964. 151 teams were above 0.950 and 285 sat at exactly 0.950. Forum reports of single private models reach 0.950 to 0.955 at 224 to 384 px [P]. Their authors credit report-label rework and image-based pseudo-labels, not architecture [P].
5. [I] The cheap levers are worth at most one tick (+0.001). A larger gain needs our own training on better labels, which fits the remaining quota only if committed early.

## 2. Where we stood on 8 October

| Item | Value | Basis |
|---|---|---|
| Best official public score | 0.949, row 56948803 (version 14 of the private submission notebook; mirror-only reader) | [V] ledger |
| Mirror 0.70 plus ConvNeXt 0.30 | 0.948, validation row 56955785; public version 35 scored 0.948 later the same day | [V] ledger |
| Mirror 0.75 plus 224-pixel reader 0.25 | 0.949, validation row 56952077 (tie with mirror-only) | [V] ledger |
| Earlier BK-A stack | 0.943, row 56859861 | [V] ledger |
| Team rank at 15:00 UTC | 443 of 3,000 or more teams, at 0.949 | [V] leaderboard pages |
| Public notebook frontier | 0.950, nine notebooks | [V] signed-out scorecard |
| Leaderboard | top 0.964; 10th 0.961; 50th 0.956; 100th 0.953; 151 teams above 0.950; 285 at 0.950; 97 at 0.949 | [V] leaderboard |
| Efficiency Prize board | top 13 scored 0.952 to 0.962; a 224-pixel reader of another author ranked 20th at 0.945; we were not in the top 100 | [V] rendered board, signed out |

Gaps: +0.001 to the public frontier, +0.002 to the next 22 teams, +0.007 to the top 50, +0.015 to first place. Displayed scores are rounded and cover part of the test set, so one tick is not evidence of anything.

## 3. Public notebooks

Recipes are paraphrased from the notebook sources. Licences were rendered for some notebooks only.

| Notebook (version) | Displayed | Recipe | Note |
|---|---|---|---|
| evgendvorkin/rsna-versia-5 (v19) | 0.950 | Unchanged 384 mirror cell; ConvNeXt reader from another author; per-label rank blend copied from another notebook with credit; falls back to the base reader if a step fails | Apache 2.0 [V]; 304 votes |
| matterhorn3838 forks (v7, v5) and sujanmajhisuzan/rsna-knee-apex-grandmaster-stack (v1) | 0.950 | Same fusion as the row above | [V] |
| haideptry/rsna-knee-v2-vs-apex-comparative-study (v5) | 0.950 | Three tiers; tier 3 uses fixed higher weights (mean 0.21) or a grid on the 58 gold studies | [V] |
| kozykappa/rsna-knee-gold-gated-triple-reader | 0.950 | One global ConvNeXt weight of 0.22, chosen on the gold set according to its own card | [V] |
| hitarthjain0/rsna-knee-apex-grandmaster-stack-v5 (v2) | 0.950 | Per-label weights (mean 0.22); ConvNeXt at two resolutions and two slice offsets, averaged. The forum claim of 0.954 or more is not on the card | [V] card; [P] claim |
| ranjeet258/rsna-knee-apex-b-under-150 (v1) | 0.950 | 224-pixel crop at 0.3 inside the CoAt side, then per-label ConvNeXt weights | [V]; 2 votes |
| nartaa/rsna-knee-0949-anatomical-mirror (v2) | 0.949 | Single 384 SWA checkpoint, native 320 crop, anatomical mirror. The base of the frontier | Apache 2.0 [V] |
| goodpjw2008 stack blend (v4) | 0.949 | 0.943 stack at 40% plus the base reader; measured null | [V] card; [P] probes |
| bliverrigpdoge raptor student blend | 0.944 | 0.943 stack with a student reader rank-blended 0.55 / 0.45; the student scored 0.942 alone [P] | [P] for the student |
| goodpjw2008 stack-2-5d-convnext (v5) | 0.944 | Stack 70% plus ConvNeXt 30% | Apache 2.0 [V] |
| Ours: the-bee-s-knees-final-rsna-push (v34, v35) | 0.949 (card at the sweep) | v34 mirror-only; v35 mirror plus ConvNeXt 0.30, row 0.948 scored after the sweep | [V] |

The ConvNeXt weight against the displayed score across the forks [V code, V scores; the pattern is our reading]:

| Mean ConvNeXt weight | Notebooks | Displayed score |
|---|---|---|
| 0.16 | matterhorn, sujanmajhisuzan (v1), the Versia repack, ranjeet258 | 0.950 |
| about 0.21 | haideptry tier 3 | 0.950 |
| 0.22 | kozykappa (global), hitarthjain0 v5 | 0.950 |
| 0.20 | sujanmajhisuzan v5 | 0.949 |
| 0.30 | ours, goodpjw2008 | 0.948 |

The pattern suggests a small optimum between 0.15 and 0.22. It sits within the noise of one tick [I].

## 4. Measured probes and community evidence

- [P] Probes by goodpjw2008 on top of the 0.949 reader, from their table: the reader alone 0.949; plus the 0.943 stack at 40%, 0.949; plus their own Raptor fine-tune (0.933) and ConvNeXt (0.929) at 30%, 0.948; per-finding weights on the stack, no change; a public meniscus specialist, no change.
- [P] A blend-weight sweep on a 0.934 model with a 0.936 public model gave a plateau of 0.935 to 0.937 for weights from 0.15 to 0.90. Changing the operator (rank, probability, logit, geometric, power) did nothing, and neither did per-label or per-study weights.
- [P] The base author's write-up lists a ladder: 0.939, 0.940 (96 slices), 0.942 (a label set), 0.945 (masked OAI supervision, stated as +0.005), 0.948 (384 full field), 0.948 (native 320 crop), 0.949 (anatomical mirror, +0.001). Negative steps are listed too. Verification found the OAI step was not compute-matched (section 5).

## 5. What was verified and what was refuted

Corrections to the sweep, from the verification pass:

1. [V] The base checkpoint was trained with masked supervision on 2,399 OAI knees. Its release grants no OAI rights. The sweep recorded the licence correctly but not the training use.
2. [V] The second author's 0.944 figure is for the ConvNeXt blended at 30% into a community stack, not into the base reader. The card says so.
3. [V] "Single fold 0.949 in about 5 minutes" refers to scoring or submit time, not training time, according to the thread.
4. [V] The "independent 2.5D reader" at 0.944 is a reproduction of another author's notebook. The source says that author trained the weights.
5. [V] The "Efficiency LB" is the Efficiency Prize track; its formula could not be read signed out.
6. [V] The slice-order bug in one forum thread (+0.028 on one fold) does not apply to our readers, which sort slices by through-plane position.
7. [V] The 0.20 weight in lever L1 was not set in advance. It was read off the public board and a gold-set gate.
8. [V] A full label run could not start before the 10 October quota reset. Free GPU time was 25,821 s at 14:59 UTC, and the next weekly refresh was 10 October at 00:00 UTC.

Lever verdicts:

| Lever | Sweep claim | Verdict | Main reason |
|---|---|---|---|
| L1 global ConvNeXt weight 0.20 | 0 to +0.001 | Refuted as a lever [V] | Not set in advance; our 0.30 blend measured 0.948; a 0.20 row would sit inside one rounded tick |
| L2 CC0 Raptor student as second reader | 0 to +0.001 | Refuted [V] | The closest measured analogue (another author's fine-tune at 30%) gave 0.948 against 0.949 [P]; same family, so errors are likely correlated [I] |
| L3 224-pixel crop at 0.3 plus L1 | 0 to +0.001 | Refuted [V] | Our 0.25 weight scored 0.949 (a tie); the predeclared rule (at least +0.002) was not met |
| L4 anatomical-mirror TTA on ConvNeXt | About 0 | Refuted [I] | Diluted at a weight near 0.2; unmeasured |
| L5 own second reader on better labels | 0 to +0.003 blended | Refuted for the deadline; research only [V] | The gauges are in-sample: the label columns were chosen on the gold set; retrains of the base recipe landed at 0.944 to 0.946; timing does not fit |
| L6 BK-A plus mirror (bench) | About 0 | Refuted as a lever [V] | The other author's stack-plus-reader probe was null (0.949); predeclared gates were not yet met |
| L7 OAI external data | +0.003 to +0.008 | Refuted [V] | NDA-controlled data; the host had not ruled on it; not doable before 22 October |
| L8 hidden-run robustness | Insurance, no score delta | Kept [V] | The V13 hidden rerun failed (row 56923965); V14 completed (row 56948803); keep the fallback root written before optional arms |

Two candidate levers were not refuted and not adopted:

- M1, an Efficiency Prize entry with a fast single-view reader. Its formula and eligibility were unverified. The verification proposed a first experiment: confirm the formula and eligibility in writing, then time the arm on the three visible studies [I].
- M2, a third-family CNN at 224 to 288 pixels trained on CC0 soft labels. Expected gain 0 to +0.001 [I]. The verification found no third missing lever that passes the evidence bar.

## 6. Label-quality findings

Host statements, paraphrased from forum posts [P]:

- Labels were assigned from the MRI images, not from the reports. Where the image and the report disagree, the image label is authoritative (topic 733826).
- LLM APIs may be used to read the reports for labels (topic 733965).
- A non-commercial licence alone does not exclude a dataset. Click-through access is generally acceptable; institutional approvals are not (topic 733965).
- The KneeCoT approach is banned (topic 734109).
- OAI data is allowed only if it is generally accessible without institutional review and free (topic 741819). The host does not comment on OAI's own terms.

Open questions with no host answer at the sweep: OAI yes or no and OAI-trained public checkpoints (743416); rule 2.5.a.4 against 2.8.a.1 (743420); prize eligibility when using public weights trained by another team (744056); RadImageNet licence under rule 6.c (745283); whether every hidden study has all three planes (744519).

Measured or substantive community findings [P unless marked]:

- Changing labels moved one model from 0.931 to 0.942. A gold-set gain to 0.945 scored 0.910 on the board (topic 745214).
- LLM labels scored 0.878 against the gold set versus 0.814 for regex labels, and about a quarter of cells were "not addressed" (topic 733932).
- One gold label contradicted its report, so the labels follow the image (topic 745861).
- Report labels top out near 0.90 without image refinement (topic 745820).
- An image model trained on pseudo-labels beat an LLM read of the reports on two findings (topic 745949).
- Private single models reached 0.950 to 0.955 at 224 to 384 px, many with pseudo-labels (topic 735304).
- DICOM metadata alone gives about 0.65 with random folds and 0.60 with scanner-grouped folds, so there is no shortcut (topic 733517).
- Seven training studies contain both knees, which is a crop risk (topic 735639).
- Two notebooks failed one to two hours into the hidden run with no traceback, and one hit a dataset mount error (topics 745984, 745826, 746542).

## 7. Caveats

- Displayed scores are rounded to three decimals and cover part of the test set. One-tick differences are noise.
- Forum numbers are the authors' claims. Private teams publish few ablations.
- Licences were opened for only some forks. Kaggle's default for public notebooks is Apache 2.0 [I].
- The leaderboard API stopped at 3,000 rows; the team count is higher. Read-only calls were paced after two HTTP 429 responses.
