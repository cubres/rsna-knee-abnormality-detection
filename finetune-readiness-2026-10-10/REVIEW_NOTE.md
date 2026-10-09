# Independent source review and invented controls

The corrected helper received an independent source-review PASS. The reviewer
checked nonempty findings, rows and fit/held sets; explicit exposure inventories;
held-study and held-patient initialization exposure; mask-aware binary class
coverage; overlap holds; strict integer counters; and aggregate-only output.
This review did not run the helper, calculate AUC or qualify a native model.

Exact reviewed and executed helper SHA256:
`b1ec2aea802e439fbec04793bd71b9af716e6093821c1e66b0497b6d9c69e504`.

One author-owned normal-Python invocation passed all twelve invented controls.
It ran at nice 5 under a 90 CPU-second resource ceiling, using approximately
0.0085 CPU seconds. No optimized-Python run or independent test reproduction
is claimed. The code uses explicit exceptions rather than assertions for its
control checks.

Public synthetic proof SHA256:
`2602b9dba92c6286c1211a389075464f0b006a8107061d8e06d27d89ad4ab7a4`.

The private invocation receipt retains interpreter, optimization, command,
source and result bindings. The public proof contains invented case names,
aggregate flags and no private joining values. These controls validate ledger
mechanics only; they do not authenticate upstream inventories or establish
patient independence, confidence intervals, clinical validity or score gain.

The source review also found the prepared canary's successful receipt maps to
process exit 2. No canary execution was observed in this review. No prepared
candidate, scheduler, notebook, identity artifact or guard was changed.
