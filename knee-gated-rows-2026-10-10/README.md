# Gated bench rows and the 10-10 decisions (2026-10-10)

This note records measurements and decisions. It contains no paths, worker slugs or credentials. Scores are official public scores.

## 1. Rows

| Row | Submission | Public score | Route in the receipt | Note |
|---|---|---|---|---|
| 57024089 | Bench V9, gated BK-A plus mirror (blend 0.55 / 0.45) | 0.947 | blend | Below the 0.949 mirror-only score |
| 57024205 | Bench V8, gated four-view | 0.949 | four_view, degraded 0 | Ties the 0.949 mirror-only score |
| 57024289 | Bench V6, gated plain-only | 0.948 | candidate | About half the GPU time of the two-view arm |
| 57016729 | Bench V19 (2026-10-09), BK-A only, Rad off | 0.938 | BK-A only | Predicted; the full stack with the Rad member was 0.943 in an earlier measurement |

The three gated versions (V9, V8, V6) each produced a valid public row, with a route recorded in its receipt. This is the first clean outcome of the standing fail-closed rule (section 3).

## 2. Decisions

- **Blend rejected.** The BK-A plus mirror blend scored 0.947, below the 0.949 mirror-only score. The public BK-A port is not pushed.
- **Four-view: tie, no public push.** The gated four-view scored 0.949, equal to the mirror-only score. The public four-view candidate is not pushed.
- **Plain-only is the efficiency final candidate.** At 0.948 it costs 0.001 AUC against the 0.949 row for about half the GPU time. The ledger records a break-even of 0.0075 AUC for this trade, and that the row clears the efficiency-entry bar of 0.9415.
- **Efficiency board, measured 10-10.** The 0.949 row of 8 October sits at rank 176. Rank 15 is at 0.963. The first 0.948 entry is at rank 32. The projected landing for the plain-only row is near rank 30; that is a projection, not a measurement.
- **Accuracy lever: the fine-tune programme.** Its canary is running today, per the coordinator. The readiness packet `finetune-readiness-2026-10-10/` records a source-proven exit-code mismatch in the prepared canary entrypoint (a passing canary exits with code 2), and open items for the control, gemlow and lexblend arms. Those items are to be closed before the canary's result is read as a run result.

## 3. Standing fail-closed rule: first clean outcome

The rule: degraded studies above max(0.5% of the test studies, 1 study) go to the reference path for all studies, or to no root. The receipt is written before the root, and the root is written last. Its origin and the V7 name-collision finding are recorded in `knee-fail-closed-2026-10-09/`.

All three gated versions were submitted under this rule and scored (rows above). Each receipt names its route. The rule's outcome for the three is therefore a valid row each, not a refusal.

## 4. Not in this note

- The public notebook narrative is being rewritten by a separate workflow. This note does not describe it.
