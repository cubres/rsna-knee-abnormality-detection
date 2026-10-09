# Fail-closed gate, the V25 post-mortem, and the 10-10 submission order (2026-10-09)

This note records measurements and decisions. It contains no paths, worker slugs or credentials.

## 1. V25: the 0.505 row and the post-mortem

- Row 57017206 (four-view, rerun-safe) scored 0.505 on the public split.
- Source-level post-mortem, from the four-view driver: the four-view section sets the process-global input resolution to 384 while the other device's 320-pixel consumer may still be predicting. A concurrency case in that path is reproduced in `independent-review-2026-10-09/`.
- The chain's fail-soft rule recorded the degraded fraction but did not block the root. The degraded gate had been made record-only in the V24 and V25 chains, so a degraded table could be exported.
- Not established: that the causal role of the resolution race in the 0.505 is proven, or that the submitted CSV was a literal constant 0.5 table. The regression note `hidden-rerun-regression-2026-10-09/` lists what the official score and the source audit do and do not establish.

## 2. Standing fail-closed rule

Applies to every chain written from 2026-10-09 onward.

- A study is degraded when its 320-pair row is the reader's constant 0.5 row, or when its volume build failed. Build failures are added on top, so the count is an upper bound.
- Limit: max(0.5% of the test studies, 1 study). At 1,300 studies it is 6.5, so six degraded studies pass and seven do not. At the three visible studies the limit is one study, but the four-view launcher's visible-set gate runs first and refuses any degraded visible study (a degraded study changes the two-view CSV). So at three studies one or two degraded studies give no root in that launcher. The floor therefore applies on the hidden set (see section 5, open decision).
- A count above the limit sends every study to the reference path (the version 36 reader, with no 384-pixel views). If the reference path is also above the limit, no root is written.
- The receipt is written before the root. The root is written last, exclusively, and only after the limit check.
- A member fill is counted once per study, whatever the number of members filled.

## 3. The V7 name-collision finding

- The bench V7 chain (BK-A with mirror blend) ended in error after 407 s on the visible commit. Both arms had produced their outputs; no receipt and no root were written.
- Cause: a BK-A cell at module level binds the name `_need` to a float. The chain's gate helper has the same name and is called after that binding. The gate raised `TypeError: 'float' object is not callable` before writing its receipt.
- Reproduced offline on the fetched artefacts: the same cells raise the same error when the binding is present, and complete with the blend root when it is absent. The member-fill audit was not the cause; the fetched branch member arrays contain no exact 0.5 members.
- Fix (bench V9): the gate restores a snapshot of the chain names taken before any BK-A cell runs, and uses its own helpers. Tests include the binding.

## 4. Bench versions committed today, submitted 10-10 in this order

1. **Blend: bench V9.** Root `blend`. Visible CSV c1478aac…, equal to the earlier blend. Static 21/21, chain 12/12 (including two replays of the fetched V7 artefacts, both asserting the blend root), dry run `DRY_RUN_OK`. The row completed on the private worker in 452 s.
2. **Four-view: bench V8.** Root `four_view`, degraded 0, visible CSV faa20e95…. Static 21/21, shared-state 12/12, chain 10/10, dry run `DRY_RUN_OK`. Completed in 90 s.
3. **Plain-only: bench V6.** Root `candidate`. Static 19/19, chain 7/7, dry run `DRY_RUN_OK`. Completed in 91 s.

Decision (coordinator): V6 and V8 stay as committed. They use the limit 0.5% × N without the one-study floor. The floor only changes the outcome at three studies, where the visible commits already produced roots. At 1,300 studies the limit is 6.5 studies either way, so the hidden reruns follow the standing rule. Only V9 carries the floor in the bench lineage.

## 5. Public candidates: same gate, held

- **BK-A plus mirror port (public notebook, embedded).** sha 22d78490…, 1,029,288 B on the wire (headroom 6,781). Offload variant sha 1053a9e1…, 995,609 B on the wire. Static 25/25 (offload 26/26); chain 10/10 on both variants; replay fatal at the root step as designed; dry run `DRY_RUN_OK`, not private. Held until bench V9's row scores at least 0.950. Its mirror arm is the only version 36 reader in the notebook, so a degraded mirror table is refused and the root falls to the BK-A-only route or to no root.
- **Four-view (public notebook, gated).** sha c71e3f84…, 804,734 B, 149 cells; wire 840,379 B (840,410 with timeout), headroom 195,659. Static 33/33; shared-state 12/12; chain 11/11 (the 1,300-study scenarios keep limit 6.5); dry run DRY_RUN_OK, not private, explicit identity triple, source sha matches. Held until bench V8's row scores at least 0.950. The reference reserve is 3.0 s per study plus 300 s, an estimate not yet measured. Above 483 studies a non-rerun commit cannot fit the reserve and fails closed; the visible commit is unaffected. The 384-pixel GPU path has not run on hardware.
- Open decision for the coordinator: whether the visible-set gate should defer to the degraded rule on the three visible studies, so that one degraded visible study stays on the four-view root. The gate is as committed; the brief required the existing checks to stay.

## 6. MaVIT (closed)

Standalone MaVIT row 57016763 scored 0.935 on the public split, below the 0.943 gate. It produced no blend row and is dropped as a lever. Closed.
