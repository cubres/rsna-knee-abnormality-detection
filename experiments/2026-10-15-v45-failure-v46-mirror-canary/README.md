# V46 passes the public mirror canary; hidden scoring is pending

**Status snapshot: 15 October 2026, 20:19 UTC.** [Bee V45](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push?scriptVersionId=358775631) matched the three public example studies exactly, but its accepted competition submission, row `57241363`, ended with **“Notebook Threw Exception” and no score**. Kaggle exposed no traceback or failing stage. The API returned `COMPLETE` alongside an error and empty scores: that status alone did not mean success.

This updates the [earlier public-example report](https://github.com/cubres/rsna-knee-abnormality-detection/tree/1647bc2198d7ed80b0a37e3aad819b30b6fa6c9b/experiments/2026-10-15-v45-public-example-parity), which correctly recorded the submission as pending at publication. Its measured public parity still holds. It did not establish hidden-test accuracy, memory requirements or runtime. The upstream reference’s visible 0.957 is not an official score reproduced by V45.

[V46](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push?scriptVersionId=358794056) **passed its paired three-study public canary**. Its first full competition attempt was submitted once through that public version’s native menu and accepted as row `57243233`. At 20:19 UTC, the API reported `PENDING`, with no error and no score. [EVIDENCE.json](EVIDENCE.json) records the version and observation times. Hidden completion and accuracy remain unverified.

## What V46 changes

The scientific experiment adds anatomical left/right reflection to **Raptor224**, before the intensity lookup, and averages its plain and reflected probabilities. R6 and Raptor384 inference logic, all checkpoints and the final rank-blend weights remain unchanged: `0.500 / 0.325 / 0.175`.

```text
same study -> plain 224 probabilities -----\
                                           mean -> unchanged rank blend
           -> reflected 224 probabilities -/
```

The completed output passed source-bound validation and independent review. Its receipts report:

| Public-canary check | Observed result |
| --- | --- |
| Output coverage | 3 studies, 36 finite predictions |
| Reported failures across all arms | 0 |
| Maximum plain/control difference | 0 |
| Maximum probability-mean residual | 6.5 × 10⁻⁸ |
| Measured prepublication elapsed time | 466.639 seconds |
| Native successful-run duration | 490 seconds, T4 × 2 |
| Peak sampled owned-process RSS | 22,616,350,720 bytes (21.063 GiB) |

The raw paired matrices stayed private; these numerical checks use the source-bound aggregate receipts, not an independent recomputation from exported matrices. Unchanged R6/Raptor384 inference logic does not establish their numerical parity. Sampled RSS is not a continuous physical-memory peak.

The public budget was 900 seconds with a 28 GiB sampled aggregate process-RSS cap. The accepted full attempt retains 29,880 seconds and the same cap. Three public examples are not cross-validation or a hidden-runtime forecast; passing them does not demonstrate an accuracy improvement.

Two independently reviewed runtime corrections accompany this experiment:

- **Test identities define the output.** The test CSV determines row count, unique IDs and order. The sample submission supplies the required thirteen-column header. An invented four-row output showed why a three-row or reordered sample template must not define hidden-test subjects. Exact output order, schema and finite `[0,1]` checks remain.
- **Bound preprocessing work before it reaches the GPU queue.** A bounded consumer queue does not bound the separate `multiprocessing` result iterator. V46 feeds the same pool in batches of `max(8, 2*nproc)` studies, retaining the workers, two-study chunks and preprocessing math. Tests cover 1,300 invented studies, exceptions and cancellation. This bounds outstanding result batches; it does not establish physical RSS or faster inference.

These are demonstrated source-level risks and reviewed corrections. **Neither has been established as V45’s failure cause.** Owner, deadline, cleanup and memory checks remain in place.

## Reproduce the public checks

The [existing reproduction guide](https://github.com/cubres/rsna-knee-abnormality-detection/blob/1647bc2198d7ed80b0a37e3aad819b30b6fa6c9b/tools/public0957-reproduction/2026-10-15/README.md) binds the exact upstream version, three public V1 datasets, dependency order and use conditions. Retain those historical versions; a dataset’s latest version is not interchangeable with its pinned V1 bytes.

From this repository’s root, run the existing standard-library fixture demo:

```bash
python3 -B tools/public0957-reproduction/2026-10-15/validate_canary_parity.py --self-test
```

For independently retained faithful baseline public-example files:

```bash
python3 -B tools/public0957-reproduction/2026-10-15/validate_canary_parity.py BASELINE.csv REFERENCE.csv
```

The filenames are placeholders. Obtain the reference from [Heliosli V1, native version 358332966](https://www.kaggle.com/code/heliosli/this-is-new-version-0-9576-in-one-click?scriptVersionId=358332966) and verify its provenance first. This checker validates schema/order, finite bounds and numerical parity; it cannot qualify hidden performance. V45’s recorded comparison was stronger: its entire public CSV was byte-identical. V46’s averaged output may legitimately differ from that reference; its paired checks assess the unchanged plain path and the new probability mean.

Credit Heliosli, nartaa, dreaddevelopment, varshithakolupuri and rostigor for the upstream sources and assets. The [full credits and OAI/NDA acknowledgement](https://github.com/cubres/rsna-knee-abnormality-detection/blob/1647bc2198d7ed80b0a37e3aad819b30b6fa6c9b/tools/public0957-reproduction/2026-10-15/README.md#credits-and-use-conditions) remain applicable. This report adds links and aggregate evidence; it rehosts no weights, patient data, reference CSV or labels.
