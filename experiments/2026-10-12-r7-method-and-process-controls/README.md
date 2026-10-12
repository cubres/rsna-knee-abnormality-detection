# Bee's Knees: R7 method notes and an owned-process helper

This is a small source-only publication package. Its process helper passed two local synthetic controls; **no R7 native run, held-out gain or new Bee official score is established**. The Kaggle launch workflow remains stopped after a notebook-preservation hook rejection. This package does not restart it or change its candidate, driver, admission rules or preserved notebook.

Heliosli's [V1 / run 357456820](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click?scriptVersionId=357456820) displayed a foreign public notebook score of **0.956**; the title says approximately 0.957. The frozen download contains a saved three-public-example execution. These examples establish neither hidden-set reproduction nor clean held-out improvement.

The reusable computational difference is the publisher's four-fold R7 ConvNeXt-small reader with exact Effusion/Baker state and fluid stackers. Its v4 caches, crop352, stride1, three slices and physical spacing3.5mm feed per-finding ranks. The publisher blends those ranks with the exact source Raptor reader at fixed **0.5 Raptor / 0.5 R7**. Raptor retains native384, a 140mm crop, central320 input, slots26/22/18/12/18, K94, embedded `study_lut` reading and anatomical mirroring. This method note neither substitutes an older owned Raptor implementation nor tunes coefficients against the foreign score or test outputs.

The [editable pipeline diagram](pipeline.mmd) describes that recipe. [PROVENANCE.json](PROVENANCE.json) records the frozen public source, observed licence terms and the scope of our evidence. No notebook, foreign source bundle, model weights, auxiliary helper, reports, labels, predictions or offline wheels are distributed here.

## Process helper and qualification

[owned_process.py](owned_process.py) is a separate MIT utility derived from our reviewed Enveda ownership pattern. It starts a new POSIX session, retains the leader unreaped, checks the exact leader PID/PGID in process snapshots, closes live descendants on normal exit as well as timeout, and waits for the leader only after two snapshots show group closure. TERM/KILL escalation, process-table probes and final wait share one absolute `time.monotonic()` deadline; cleanup time is reserved inside that budget. A timeout remains a HOLD even when cleanup succeeds.

Optional output goes directly to an exclusive regular file, which is closed before return. There is no captured pipe or reader thread. The helper does **not** cap file output volume. It requires default SIGCHLD handling and exclusive ownership of child reaping. Its closure claim covers processes that remain in the new group; it does not contain descendants that deliberately create another session. Evidence errors return HOLD. If closure cannot be proven, the helper retains a strong reference to the unreaped leader while its owner process remains alive. Ordinary operating-system calls cannot provide a hard real-time guarantee under a stalled host.

[test_owned_process.py](test_owned_process.py) exercised two actual local synthetic process trees on macOS26.5.1/arm64 with Python3.9.6:

| Control | Observed result | Caller elapsed | Global budget |
| --- | --- | ---: | ---: |
| Leader exits0, TERM-resistant child remains | TERM→KILL; empty live group; leader reaped; execution PASS | 0.660s | 3.0s |
| TERM-resistant leader and child time out | TERM→KILL; empty live group; leader reaped; execution HOLD | 1.526s | 3.0s |

Both controls independently observed neither fixture PID after return. Logs closed, no reader threads existed, and neither global deadline was exceeded. The frozen [control receipt](PROCESS_CONTROL_RECEIPT.json) pins the exact helper and test source. Earlier preserved control attempts exposed macOS `getpgid(zombie)` returning ESRCH; the corrected helper uses the unreaped leader's process-table row instead.

A separate reviewer checked the exact helper, fixture, test and receipt hashes plus38 source/evidence consistency predicates, with no blocking findings. That reviewer did not rerun the controls. The post-return observation is a fresh snapshot using the same process-table parser, rather than a second containment implementation.

These controls qualify this exact helper's tested local process behavior only. Linux process-table behavior, adversarial session escape, checkpoint compatibility, GPU execution, the original R7 adapter and Kaggle runtime/resources remain unqualified. The earlier adapter's591 static checks are not runtime qualification. Its preserved `r7_safety.py` has a known process-cleanup defect and is deliberately absent from this package. The new helper has not been integrated into any stopped launch candidate.

## Attribution and terms

- **Heliosli**: Apache-2.0 notebook source and seven embedded modules; R7 four-fold reader, stackers and fixed rank blend. [Original R7 mount](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss), V1, bundle `8c7b666e27dc0e8f`.
- **nartaa / Danial Zakaria**: [selected Raptor checkpoint](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007), publisher SHA256 `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`; research/educational checkpoint terms are separate from source-code terms.
- **dreaddevelopment** and **timm**: Raptor inference/model lineage and backbones. **stevenleehans**, **pilkwang** and **riadmohamed42**: upstream weak-label work credited by the publisher.
- **Arjun Desai / SKM-TEA**: anatomy-checkpoint lineage. The original auxiliary mount retains component-specific terms; the seven-module Apache grant does not relicense imported helpers.

The observed R7 dataset licence is **Other (specified in description)**. The author publicly invites reproduction using the original Kaggle mount; a complete weight-rehosting or relicensing grant remains **unverified**. Model hashes are publisher declarations; we did not download or byte-attest the binaries. Publisher training exclusions are declarations, not our independent audit. Gold58 influenced upstream selection and cannot provide a clean test of this proposal.

The publisher credits OAI/NDA research tools and dataset identifier10.15154/0hcg-f676. Retain the publisher's complete acknowledgement in any permitted manuscript or notebook adaptation: [NDA manuscript preparation](https://nda.nih.gov/nda/manuscript-preparation), [shared study](https://nda.nih.gov/study.html?id=3407), [dataset DOI](https://doi.org/10.15154/0hcg-f676).

Our method notes and diagram are offered under Apache-2.0 ([LICENSE](LICENSE)); `owned_process.py` and its synthetic test are MIT ([LICENSE-MIT](LICENSE-MIT)). These grants cover the included authored files only.
