# Knee engineering lessons

Each lesson gives the situation, the rule adopted, the evidence (row ids and UTC dates from the campaign ledger) and how the rule was checked. The repository folders named below hold the public write-ups.

## 1. A strict reader fails a hidden rerun on one odd study

- Situation: the reader raised on any header, pixel, geometry or model error. Its rerun over about 1,300 hidden studies ended with an unhandled error. Row 56923965 (7 October, 23:22 UTC) has no score.
- Rule: a reader that runs on the hidden set follows the reference fault semantics. A bad file becomes a neutral record, a failed study keeps neutral probabilities, and the root is always written. Strict checks belong in the visible run only.
- Evidence: row 56948803 (reference semantics) completed with public score 0.949 on 8 October at 10:54 UTC. The root cause is probable, not proven (folder official-results-2026-10-08).
- Check: on synthetic DICOM, the strict reader raised on 7 of 9 faulty cases. The new reader matched the reference builders on 12 synthetic cases.

## 2. A canary-only limit must not reach inference

- Situation: the reader required a study count from 1 to 16, and the check also ran in inference. The bench chains passed about 1,300 studies in the rerun, so the reader refused at start-up and wrote no file. Rows 57016738 and 57016750 were rejected at the file check within about a minute (9 October, 17:54 to 17:57 UTC). Row 57016729 could only measure BK-A.
- Rule: a canary limit applies in canary mode only. Chains are tested with a rerun-sized study count.
- Evidence: rows 57016738, 57016750 and 57016729; the MaVIT probe's mirror component had the same cap (row 57016763). Folder knee-rerun-cap-2026-10-09.
- Check: a 1,300-study inference test passed after the fix, and a CPU run of 20 unseen studies gave one row per study.

## 3. A gate that only records is not a gate

- Situation: the four-view chain recorded degraded studies but no longer blocked the root. Row 57017206 (9 October, 18:21 UTC) scored 0.505 publicly.
- Rule: a chain that can write a root blocks when degraded studies exceed max(0.5% of the test studies, 1). Above the limit, every study goes to the reference path, or no root is written. The receipt is written before the root, and the root last.
- Evidence: the rule was written at 18:50 UTC and the one-study floor added at 19:30 UTC. Rows 57024089, 57024205 and 57024289 (10 October) each gave a valid row under it (folder knee-fail-closed-2026-10-09).
- Check: degraded-above-limit and both-above-limit scenarios. At 1,300 studies the limit is 6.5: six degraded studies pass, seven do not. Up to six degraded studies can still give a root; the effect is not measured.

## 4. Concurrent device threads must not share mutable global settings

- Situation: the four-view driver set a process-wide input size of 384 while another device thread was still predicting at 320. That thread raised and fell back to 0.5 per study (post-mortem, 9 October 18:58 UTC).
- Rule: each device has its own model instance and fixed input size. No global is written inside a concurrent section.
- Evidence: row 57017206 (0.505). The race-free replicate, row 57024205 (10 October, 0.949, zero degraded studies), scored level with mirror-only. The ledger calls the race's role suspected, not proven.
- Check: a two-thread shared-state test matched serial results bitwise in 12 of 12 cases.

## 5. A module-level name can shadow a helper

- Situation: a BK-A cell bound a module-level name to a float that a gate helper also used. The gate raised a TypeError after both arms finished, leaving no receipt (version 7, 9 October, 19:20 UTC).
- Rule: gate code uses its own helpers and snapshots the names defined before any model cell runs. Gate tests include a colliding binding.
- Evidence: version 9 completed in 452 s after the fix; its root was the blend (row 57024089, 0.947).
- Check: replaying version 7's fetched artefacts reproduced the TypeError; without the binding the replay completed.

## 6. Canary results come from receipts, and mounts differ by layout

- Situation: canary version 60 ended ERROR after 635 s on 10 October; its receipt read HOLD_HARD_WATCHDOG after a 600 s watchdog fired. A readiness review found that a passing canary would exit with code 2 (10:25 UTC). Version 61 failed in 45 s: the competition data mounted under a competitions/ path, while the code expected the dataset path (11:02 UTC).
- Rule: the receipt decides the outcome, not the kernel status. A passing canary exits 0. Path resolution tries both layouts and records the one used.
- Evidence: folder finetune-readiness-2026-10-10. Version 62, with both fixes, completed in 136 s (11:24 UTC); its receipt was not read at the snapshot.
- Check: the exit-code build passed 23 of 23 tests; the layout build passed 24 of 24, including a two-layout test.

## 7. Predeclared adoption rules apply to near-ties

- Situation: several rows landed within one rounded tick of the anchor, or below it.
- Rule: the threshold is written before scoring. A tie is not a gain, and misses are recorded as negative results.
- Evidence: row 56952077 (0.949, tie) not adopted under the +0.002 rule (8 October, 12:30 UTC); row 56955785 (0.948) not adopted; row 57024089 (0.947) rejected the blend; row 57024205 (0.949, tie) not pushed (10 October, 10:17 UTC).
- Check: each decision was logged against the predeclared rule. Scores round to 0.001, so the rule does not establish significance.

## 8. Plain-only efficiency only with a measured break-even

- Situation: a single-view reader costs about half the GPU time of the two-view arm. The trade pays only on the efficiency board, and the efficiency entry must be one of the two selected finals.
- Rule: a plain-only row is the efficiency final only if its public score is at least 0.9415 and its private AUC cost is inside the break-even. The ledger computes break-even 0.0088 (overlap) and 0.0107 (serial) and adopts 0.0075 as the conservative line.
- Evidence: row 57024289 (10 October, 0.948) clears the public bar. The private check was pending at the snapshot.
- Check: the arm's plain output equals the reference plain half bitwise on CPU.

## 9. Check which rows the efficiency board scores

- Situation: on 9 October the board placed the team at rank 3,678, on a 0.943 row. The ledger reads the board as scoring the selected submissions (18:01 UTC).
- Rule: check the selected finals before reading a board position.
- Evidence: the 10 October snapshot put the 0.949 row at rank 176 (10:23 UTC). Not confirmed in the competition interface.

## 10. Notebook identity: never delete, rename or replace

- Situation: each version must keep all prior cells. The bench notebook sat at the 1 MiB wire limit (1,033,378 bytes against 1,036,069) and could not take the MaVIT arm (8 October, 19:56 UTC).
- Rule: every push runs one path: fresh pull, identity check, not running or queued, push, post-push cell comparison, receipt. Notebooks, versions, slugs and source directories are never deleted, renamed or moved. A full notebook is kept, a new private worker is created, and the public notebook is untouched.
- Evidence: the second private worker (8 October, 19:56 UTC); a successor version (17:23 UTC) that kept earlier cells raw and deactivated the BK-A cells; the 8 October 10:18 push logged identical cells.
- Check: each push logged its identity check and post-push comparison in its receipt.

## 11. Session and time limits

- Situation: each run has a commit cap and a rerun budget. The commit cap was set for quota budgeting. The rerun budget is 32,350 s, inside the 9-hour hidden limit (32,400 s); a commit's budget is 1,750 s. At most two GPU sessions run per account.
- Rule: each run has an explicit cap and a watchdog aligned to its budget. Reruns are pushed without a session timeout, because the policy line of 8 October 10:50 UTC says the commit cap's effect on reruns was unverified.
- Evidence: row 56948803 was pushed with a 1,800 s commit cap; its rerun completed in about 30 minutes including queue. A canary was ended by its 600 s watchdog (10 October, 10:29 UTC). Launches waited for a free session (10:40 UTC).
- Check: the watchdog deadline was aligned to the 32,350 s budget (folder official-results-2026-10-08). Whether the 1,800 s commit cap bounds a rerun has not been tested separately.
