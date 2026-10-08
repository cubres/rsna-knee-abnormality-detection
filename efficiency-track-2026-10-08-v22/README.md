# Efficiency track: the plain-view arm measured on GPU (2026-10-08, late)

A private bench run of the single-view reader (the mirror view switched off, exact preprocessing flags kept) ran on two T4 GPUs on 2026-10-08. This note records what it measured, what the visible CSV can and cannot tell us, and the rule for choosing an efficiency entry. No score is claimed.

## What was measured

- On the three visible studies, the single-view output matched the reader's own reference single-view output exactly (maximum deviation 0.0 on each study, with the real weights on the GPU).
- The visible CSV is byte-identical to the mirror arm's CSV. The export ranks each column across the submitted studies, so with three studies every column holds only 0, 0.5 or 1. The identity shows the two views order those three studies the same way; it says nothing about AUC on the full test set.
- Steady-state study time is 1.86 s per study for the single-view arm. The two-view reference costs about 3.1 s per study by the same accounting (V21 measurements). The GPU work per study halves, from 1.93 s to 0.95 s, because the mirror view is gone.

## Projection for about 1,300 studies on two T4 GPUs

- Overlapped preprocessing: about 670 s for the single-view arm, against about 1,290 s for the mirror arm.
- Serial preprocessing: about 1,260 s, against about 2,010 s.
- Efficiency at AUC 0.949 minus x (lower is better): the single-view arm stays ahead of the mirror arm in both regimes while x is at most about 0.0088 (overlapped) or 0.0107 (serial). The adopt line below sits inside both.

## Adopt rule for the efficiency entry

- The single-view row becomes the efficiency entry only if its official public score is at least 0.9415, which is the mirror arm's public 0.949 minus 0.0075.
- At the deadline its private score is checked the same way: it must be no more than 0.0075 below the mirror row's private score.
- The efficiency entry must be one of the two selected final submissions. The accuracy entry stays on the mirror (BK-A) row.

The single-view arm's hidden score arrives after its Knee slot runs. This note will be updated with the public score once it is known.
