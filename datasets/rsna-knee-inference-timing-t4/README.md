## What

How much does one more test-time view cost a knee MRI reader on two Tesla T4 GPUs, and where does the time go? This dataset holds per-stage timing receipts from 10 benchmark runs of the reader on three visible studies, the summary tables derived from them, and a calculator that projects a 1,300-study hidden run and its efficiency score.

- **One extra view.** Going from one view to two adds about 0.99 s of GPU time per study. Over 1,300 studies on two GPUs that is about 645 s of hidden runtime, worth about 0.009 AUC at the approximate efficiency constants (see Efficiency arithmetic).
- **Four views** (two 320-pixel views plus a 384-pixel full-field pair) take about 4.8 s of GPU time per study, roughly 2.5 times the two-view cost.
- **Speed flags.** Accepted speed flags cut CPU prep by 8% to 17% per study and added 1% to 9% GPU time. For two-view and four-view builds, whole-study time stayed within about 4% of the reference.
- **Where the time goes.** CPU prep (DICOM header and pixel reads, volume build) is about 1.0 to 1.1 s per study in the parity reference rows (1.01 to 1.12 s). It is higher in the main-phase rows, 1.39 to 1.44 s per study (see Caveats). Prep is the binding term for one view and is hidden under GPU time for two and four views.

### Key numbers

Steady state, per study, on two Tesla T4s. The hidden projection is for 1,300 studies with prep and GPU overlapped.

| Configuration | GPU s per study | CPU prep s per study | Projected 1,300-study run (s) |
|---|---|---|---|
| One view, reference (bench V22) | 0.94 | 1.08 | 717 |
| One view, speed flags (bench V22) | 0.95 | 0.91 | 633 |
| Two views, reference (bench V21) | 1.93 | 1.12 | 1,279 |
| Two views, reference (bench-2 V3) | 1.95 | 1.01 | 1,278 |
| Two views, reference, main phase (bench-2 V9) | 1.88 | 1.44 | 1,252 |
| Four views, reference (bench V25) * | 4.84 | 1.10 | 3,153 |
| Four views, speed flags (bench V25) * | 4.92 | 0.94 | 3,204 |

Each row is a separately measured build. Compare GPU and prep within a row. Across rows, differences under about 7% fall within the two-view reference GPU range given in Caveats.

\* Four-view build. Official row 57017206 has a public score of 0.505, and the snapshot records no cause (see Caveats). These timings are visible-study measurements only.

### Speed flags and the parity gate

Before a speed configuration is used, the reader runs it against the reference path on the three visible studies and compares the output probabilities. The gate tolerance is 0.0001 absolute. Only configurations that pass are eligible, and the table shows the one each build accepted.

| Build | Accepted configuration | GPU change | CPU prep change | Study total change |
|---|---|---|---|---|
| bench V21, two views | fused TTA only | +3.0% | -16.7% | -4.2% |
| bench V22, one view | all requested flags | +0.8% | -15.4% | n/a (see Timing scope) |
| bench V23, four views | all requested flags | +4.5% | -14.0% | +1.5% |
| bench V24, one view | all requested flags | +1.1% | -11.6% | n/a (see Timing scope) |
| bench V25, four views | all requested flags | +1.6% | -15.1% | -1.5% |
| bench-2 V2, two views | fused TTA only | +8.6% | -11.3% | +1.5% |
| bench-2 V3, two views | fused TTA only | +4.3% | -8.0% | +0.1% |

GPU and prep changes are against the same build's reference row. The study-total change compares whole-study wall time. It is marked n/a for plain-view builds (see Timing scope). bench V22, V23 and V24 passed the gate but were not scored: the reader rejected two at hidden-run startup and one was pulled before submission. Their timings are visible-study checks only.

### Builds in this dataset

Each build is one version of the same reader notebook. The build labels are our own measurement-run labels, numbered by the team: "bench V21 to V25" is one series and "bench-2 V2 to V9" is another. Names are kept so rows can be matched across files. The arm column says which views a build ran.

| Build | Views | Per-stage timing file | Public row |
|---|---|---|---|
| bench V21 | two views (320 px) | yes | none |
| bench V22 | one view (320 px) | yes | none |
| bench V23 | four views (320 + 384 px) | yes | none |
| bench V24 | one view (320 px) | yes | none |
| bench V25 | four views (320 + 384 px) | yes | 57017206 |
| bench-2 V2 | two views (320 px) | yes | none |
| bench-2 V3 | two views (320 px) | yes | none |
| bench-2 V4 | two views (320 px) | yes | none |
| bench-2 V5 | four views (320 + 384 px) | no | none |
| bench-2 V6 | one view (320 px) | no | 57024289 |
| bench-2 V7 | two views (320 px) | yes | none |
| bench-2 V8 | four views (320 + 384 px) | no | 57024205 |
| bench-2 V9 | two views (320 px) | yes | 57024089 |

- **One view** (`plain_320`): the plain view only, on the 320-pixel centre crop.
- **Two views** (`two_view_320`): the plain view and a mirrored view (the same windows with the channel order reflected) on the 320-pixel crop, averaged with equal weight.
- **Four views** (`four_view_320_384`): the two 320-pixel views plus a plain and a mirrored view on the 384-pixel full field, averaged with equal weight.

## Why

A test-time view is a runtime decision as well as an accuracy one. Each view adds forward passes on the GPU, and the efficiency formula in this card counts runtime against AUC. Anyone planning a build needs the measured per-stage cost to see whether an extra view pays for itself. This dataset gives those costs for each build on the same hardware, the parity checks that gate each speed flag, and the arithmetic that turns them into a projected hidden run. It is most useful for planning and for checking that arithmetic. It does not replace a run on the hidden studies, whose preprocessing cost was not measured (see Caveats).

## Files

| File | What it is |
|---|---|
| `README.md` | The complete dataset card. The Kaggle description field carries only its opening sections (What, Why and Files). |
| `LICENSE` | Licence for this dataset (CC BY-NC-SA 4.0), with the link to the full legal code and the scope of the code-file exception. |
| `LICENSE-APACHE-2.0` | Full Apache License 2.0 text, covering build_tables.py and efficiency_calculator.py. |
| `NOTICE.md` | Per-file licence table and third-party notices. |
| `MANIFEST.json` | SHA-256 and byte size of every other file in this folder. |
| `timings.csv` | Long format: one row per stage, per study, per configuration, per build. 630 rows. |
| `summary.csv` | One row per build, phase and configuration: steady-state seconds, projections and parity results. 26 rows. |
| `native_runs.csv` | One row per build's native run (end-to-end on three visible studies): elapsed time, parity, library versions and failure counters. 13 rows. |
| `efficiency_calculator.py` | Standard-library Python: hidden-run projection, efficiency score and break-even AUC. Command line and import. |
| `build_tables.py` | Rebuilds timings.csv, summary.csv and native_runs.csv from the receipt files, indexed by receipts__index.csv. Standard library only. Apache-2.0. |
| `schema.json` | Machine-readable column dictionary, stage glossary and licence fields. |
| `receipts__index.csv` | Maps each receipt file to its kind (timing or native), build version and arm. |
| `receipts__timing__BUILD.json` | Per-stage timing receipts, one per build with a timing file (10 files). Study and session identifiers removed. |
| `receipts__native__BUILD.json` | Native inference receipts, one per build (13 files). Session identifiers removed. |

build_tables.py rebuilds the three CSV files from the receipt files (`receipts__timing__*.json` and `receipts__native__*.json`), using `receipts__index.csv` to find them. Running it on the shipped receipts reproduces the tables exactly. Receipt files sit at the top level of this folder, with the prefix `receipts__` in place of the sub-folder path.

## Columns

### Stage glossary

| Stage | Category | What it covers |
|---|---|---|
| `select_s` | CPU prep | Choosing the slices for each series slot from the series metadata. |
| `header_s` | CPU prep | Reading DICOM headers. |
| `pixel_s` | CPU prep | Reading pixel data for the selected slices. |
| `volume_s` | CPU prep | Per-slot percentile normalisation, centre crop and resize into a 384-pixel volume. |
| `windows_s` | GPU | Building the three-slice windows for the 320-pixel views, including the host-to-device copy. |
| `forward_plain_s` | GPU | Forward pass over the plain windows at 320 px. |
| `forward_mirror_s` | GPU | Forward pass over the mirrored windows at 320 px. |
| `forward_fused_s` | GPU | Plain and mirrored windows in one forward call (fused TTA flag). |
| `full_windows_s` | GPU | Building the 384-pixel full-field triplets, shared by the two full-field passes. |
| `full_forward_plain_s` | GPU | Plain full-field pass at 384 px, including gathering its windows from the shared triplets. |
| `full_forward_mirror_s` | GPU | Mirrored full-field pass at 384 px, including its window gather. |
| `study_total_s` | total | Study wall time as recorded by the reader. Scope is in study_total_scope. |

### timings.csv

| Column | Unit | Meaning |
|---|---|---|
| `run` | - | Build and arm key, for example bench_v22_plain_320 (version plus arm). |
| `version` | - | Build name: bench_v21 to bench_v25, or bench2_v2 to bench2_v9. |
| `arm` | - | Views the build ran: two_view_320, plain_320 or four_view_320_384. |
| `phase` | - | parity_reference (reference path on device 0), parity_candidate (one speed configuration on device 0) or main (three studies across both GPUs). |
| `config` | - | reference = no speed flags. full = every flag the build requested (the flags column is authoritative). single:FLAG = one flag switched on. |
| `study_index` | index | Position of the study among the three visible studies: 0, 1 or 2. Not a study identifier. |
| `stage` | - | Stage name. See the stage glossary. |
| `seconds` | s | Wall-clock seconds for this stage on this study. |
| `flags` | - | Speed flags in effect, as key=value pairs separated by semicolons: channels_last, chunk (windows per forward call), fused_tta, header_parallel, pixel_parallel, shared_triplets. |

### summary.csv

| Column | Unit | Meaning |
|---|---|---|
| `run` | - | Build and arm key, as in timings.csv. |
| `version` | - | Build name, as in timings.csv. |
| `arm` | - | Views the build ran, as in timings.csv. |
| `phase` | - | parity_reference, parity_candidate or main, as in timings.csv. |
| `config` | - | Configuration name, as in timings.csv. |
| `flags` | - | Speed flags in effect, as in timings.csv. |
| `studies` | count | Studies recorded for this run and configuration (three in every case). |
| `steady_indices` | index | Study indices averaged for the steady-state columns: 1-2 for parity rows, 2 for main-phase rows. |
| `study_total_by_index_s` | s | study_total_s for each study index, as index:seconds pairs separated by a vertical bar. |
| `cold_study0_total_s` | s | study_total_s for study index 0 (the cold-start study in parity runs). |
| `steady_prep_s` | s per study | Mean CPU prep per steady study: select_s + header_s + pixel_s + volume_s. |
| `steady_gpu_s` | s per study | Mean GPU time per steady study: windows_s + full_windows_s + forward_plain_s + forward_mirror_s + forward_fused_s + full_forward_plain_s + full_forward_mirror_s. |
| `study_total_scope` | - | build+predict (parity rows: prep and GPU timed in one window) or predict_only (main rows: GPU consumer only, prep overlapped). |
| `steady_residual_s` | s per study | steady_study_total_s minus steady_prep_s minus steady_gpu_s. Blank for predict_only rows. About 0.01 s in every parity row except the two plain-view reference rows, about 1.9 s (see Timing scope). |
| `steady_study_total_s` | s per study | Mean study_total_s over the steady studies. |
| `startup_s` | s | Startup before the first study (imports, model load, CUDA start), from the receipt. |
| `projected_hidden_overlap_s` | s | Projected 1,300-study run, overlapped: startup + 1300 x max(prep / 2, gpu / 2). |
| `projected_hidden_serial_s` | s | Projected 1,300-study run, serial: startup + 1300 x (prep + gpu) / 2. |
| `projected_hidden_overlap_prep3x_s` | s | As projected_hidden_overlap_s with prep three times larger. A sensitivity check for hidden studies that decode more slowly. |
| `parity_passed` | boolean | parity_candidate rows only: whether this configuration passed the parity gate on the visible studies. Blank for other rows. |
| `parity_max_abs_dev` | probability | parity_candidate rows only: largest absolute deviation from the reference probabilities on the visible studies. Gate tolerance is 0.0001. Blank for other rows. |
| `accepted` | boolean | parity_candidate rows only: whether this is the configuration the reader accepted for the build. Blank for other rows. |
| `gpu_vs_run_reference` | ratio | steady_gpu_s divided by the same build's reference row. 1.0 for reference rows. |

### native_runs.csv

| Column | Unit | Meaning |
|---|---|---|
| `run` | - | Build and arm key, as in timings.csv. |
| `version` | - | Build name, as in timings.csv. |
| `arm` | - | Views the build ran, as in timings.csv. |
| `has_timing_receipt` | boolean | true if a per-stage timing file for this build is in timings.csv. |
| `mode` | - | Reader mode (inference). |
| `gpu_count` | count | GPUs used (two in every run). |
| `device_name` | - | GPU model reported by the reader (Tesla T4). |
| `studies` | count | Studies processed (three visible studies). |
| `selected_slices` | count | Slices selected across the studies. |
| `missing_slots` | count | Series slots with no usable series across the studies, as counted by the reader. |
| `elapsed_s` | s | End-to-end wall time of the native run, including startup. Not comparable to per-study totals. |
| `startup_s` | s | Startup seconds, where the receipt records them. Blank where it does not. |
| `fast_decision` | - | Gate outcome: accepted_full, accepted_subset (with the accepted flag), or reference_only when no speed flags were requested. |
| `fast_config_accepted_name` | - | Name of the accepted configuration (full, subset or reference). |
| `prep_workers` | count | CPU prep workers (two in every run). |
| `header_failures` | count | DICOM header reads that failed. Zero in every receipt. |
| `pixel_failures` | count | Pixel reads that failed. Zero in every receipt. |
| `study_build_failures` | count | Studies whose volume could not be built. Zero in every receipt. |
| `model_fallback_studies` | count | Studies that fell back to the 0.5 default. Zero in every receipt. |
| `fast_fallback_studies` | count | Studies that fell back from the speed configuration to the reference. Zero in every receipt. |
| `consumer_error_count` | count | Errors raised by GPU consumer threads. Zero in every receipt. |
| `parity_studies` | count | Visible studies used by the parity gate. |
| `parity_configs_passed` | - | Configurations that passed, as 'passed of tested'. |
| `parity_max_abs_dev_passing` | probability | Largest deviation from the reference over passing configurations. Tolerance is 0.0001. |
| `parity_max_abs_dev_all_tested` | probability | Largest deviation over every tested configuration, passing or not. |
| `torch_version` | - | PyTorch version recorded by the reader. |
| `timm_version` | - | timm version recorded by the reader. |
| `visible_prediction_sha256` | hex | SHA-256 that the reader records for its prediction file on the three visible studies. Identical in all 13 native receipts. The snapshot gives the same value as the submission.csv sha256 of three public rows with different scores, so it does not link a receipt to a public row (see Caveats). |
| `official_row` | Kaggle row id | Public Kaggle row produced by the same notebook version, where the team recorded one. Read as text. |
| `official_public_auc` | AUC | Public score of that row, as recorded in the Kaggle submissions snapshot of 2026-10-10. |
| `official_note` | - | How the row relates to this build. |
| `note` | - | Context for this build. |

### receipts__index.csv

| Column | Unit | Meaning |
|---|---|---|
| `file` | - | Name of the receipt file in this folder, for example receipts__timing__bench_v21_two_view_320.json. |
| `kind` | - | timing (per-stage timing receipt) or native (native inference receipt). |
| `version` | - | Build name, as in timings.csv: bench_v21 to bench_v25, or bench2_v2 to bench2_v9. |
| `arm` | - | Views the build ran, as in timings.csv: two_view_320, plain_320 or four_view_320_384. |

## How to load

Reproduce the steady-state reference rows from summary.csv:

```python
import pandas as pd

summary = pd.read_csv("summary.csv")
ref = summary[(summary["config"] == "reference") & (summary["phase"].isin(["parity_reference", "main"]))]
print(ref[["run", "steady_gpu_s", "steady_prep_s", "projected_hidden_overlap_s"]].to_string(index=False))
```

Per-stage seconds for one build, one row per stage and one column per study:

```python
import pandas as pd

timings = pd.read_csv("timings.csv")
one_view = timings[(timings["run"] == "bench_v22_plain_320") & (timings["phase"] == "parity_candidate")]
print(one_view.pivot_table(index="stage", columns="study_index", values="seconds").round(3).to_string())
```

Runtime and efficiency arithmetic, from the folder that holds `efficiency_calculator.py`. The AUC is a placeholder, not a measured score. The two runtimes come from different builds, so the break-even line compares runtimes only.

```python
from efficiency_calculator import breakeven_auc_cost, efficiency_score, hidden_runtime

one_view = hidden_runtime(startup_s=17.102, gpu_s=0.9482, prep_s=0.9102)  # bench V22, speed flags
two_view = hidden_runtime(startup_s=22.331, gpu_s=1.9327, prep_s=1.1177)  # bench V21, reference
print(round(one_view), round(two_view))                     # 633 1279
print(round(breakeven_auc_cost(two_view - one_view), 4))    # 0.0092
print(round(efficiency_score(0.950, one_view), 4))          # -2.0457 (placeholder AUC 0.950)
```

To read the official row ids as text, use `pd.read_csv("native_runs.csv", dtype={"official_row": "string"})`. To map receipt files to builds, use `pd.read_csv("receipts__index.csv")`.

### Efficiency arithmetic (assumption-based)

Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400, where lower is better. The formula and the constants Benchmark ≈ 0.5 and maxAUC ≈ 0.96 are assumptions of this card, not values taken from a verified source. Both constants are approximations. The competition rules page could not be read when this card was prepared, so neither the constants, the RuntimeSeconds definition nor the hidden-set size of 1,300 studies is verified. Treat the figures below as a worked example. Use the private-leaderboard AUC for any real score.

- One AUC step of 0.001 is worth about 70 s of runtime.
- A saving of ΔT seconds pays for an AUC loss of up to about 1.42 × 10⁻⁵ × ΔT.
- One extra view costs about 645 s over 1,300 studies on two GPUs. That is the GPU difference between the two reference rows, in the overlapped projection. It is worth about 0.009 AUC.
- Four views against two (bench V25 against bench V21, runtimes only) add about 1,875 s. That needs about 0.027 AUC to break even.
- The projection is startup plus 1,300 studies times the larger of prep/2 and GPU/2. It excludes platform overhead such as queueing and commit time.

## Provenance

- **Source.** The timings were measured on the competition's three visible sample studies, indexed 0 to 2, on Kaggle GPU sessions with two Tesla T4 devices. Study and session identifiers were removed from the receipts. No competition images, labels, predictions or identifiers are included.
- **Official rows.** Official Kaggle rows are referenced by row id only. Their public scores are in the table under Public scores referenced.

### How the timings were produced

- **Hardware and software.** Kaggle GPU sessions with two Tesla T4 devices, PyTorch 2.11.0 with CUDA 12.8, timm 1.0.29, fp16 autocast.
- **Per study.** Five series slots supply 96 slices (26, 22, 18, 12 and 18 slices per slot), and the model reads 94 windows of three adjacent slices, both as configured in the public mirror reader (nartaa's recipe notebook rsna-knee-0949-anatomical-mirror, Apache 2.0). The slices are resampled to a 384-pixel volume.
- **Clock.** Stage times are wall-clock seconds from `time.monotonic()`, with GPU synchronisation at each GPU stage boundary. The reader writes them to JSON receipts. The receipts shipped here are those files with study and session identifiers removed.
- **Visible studies.** Three studies per run, indexed 0, 1 and 2. In parity runs, index 0 is the cold-start study.
- **Slice count.** Across the three studies, 264 slices were selected and two series slots were missing (`selected_slices` and `missing_slots` in native_runs.csv). The reader counts a missing slot's slices as zero.

#### Steady state

- Parity runs (`parity_reference`, `parity_candidate`) execute on one device. Index 0 is cold, so steady-state columns average indices 1 and 2.
- Main-phase runs spread studies across both GPUs. The first study on each GPU is cold, so steady-state columns use index 2 only. Receipts do not record device identifiers, so this assignment is inferred from the reader's scheduling.

#### Timing scope and the plain-view residual

- In parity runs, study_total_s covers CPU prep and GPU work in one window. The residual (study total minus prep minus GPU) is about 0.01 s in every parity row except the two plain-view reference rows.
- In the two plain-view builds (bench V22 and bench V24), the reference-configuration check also runs the full two-view reference predictor inside the timed window, without timers. This adds about 1.9 s to each reference study_total_s. Stage columns are unaffected, and the check runs only in the parity phase of these builds. For those two builds, compare stage columns, not study totals.
- In the plain-view builds the predictor is replaced by a plain-only function that always builds shared triplets, so the shared_triplets flag does not change their GPU stages.
- In the main phase, study_total_s times only the GPU consumer. Prep runs in producer threads that overlap the GPU, so the residual is blank.
- In four-view builds, the 384-pixel pass is timed in three stages: full_windows_s (shared triplets), full_forward_plain_s and full_forward_mirror_s (each including its own window gather). These do not overlap, so steady_gpu_s sums them directly.

### Public scores referenced

These are Kaggle public scores as recorded in a Kaggle submissions snapshot taken 2026-10-10 at 11:47 UTC. Each row id and score below appears in that snapshot. They were not re-read live for this card.

| Official row | Build | Public score | What it means here |
|---|---|---|---|
| 57024289 | bench-2 V6, one view | 0.948 | Same notebook version as its native receipt. No per-stage timing file exists for this build. |
| 57024205 | bench-2 V8, four views | 0.949 | Same notebook version as its native receipt. No per-stage timing file exists for this build. |
| 57024089 | bench-2 V9 | 0.947 | Same notebook version label as the bench-2 V9 native receipt. The snapshot describes this row as a per-finding rank blend (0.55/0.45); no timing in this dataset covers that blend. |
| 57017206 | bench V25, four views | 0.505 | Same notebook version as the bench V25 native receipt. The snapshot records no cause for this score, and this card does not establish one. |

The snapshot descriptions follow, paraphrased. Internal build labels are replaced with generic wording, and the notebook-reference and file-hash fields are left out. The paraphrases keep the snapshot's meaning and its own terms for the method.

- **57024289:** Plain-view-only anatomical reader (nartaa r384 recipe, mirror view off). Fail-closed chain: above 0.5% degraded output it falls back to the reference path, or gives no root. The row measures the plain reader's AUC for the Efficiency Prize break-even.
- **57024205:** Four-view anatomical-mirror TTA on the nartaa r384 checkpoint (320 crop plain+mirror, plus 384 full-field plain+mirror, equal weights predeclared), race-free driver, fail-closed chain (above 0.5% degraded output it falls back to the reference path, or gives no root). A clean replicate of an earlier four-view row.
- **57024089:** Fault-tolerant anatomical-mirror reader (nartaa recipe), with a per-finding rank blend of 0.55/0.45 predeclared. Fail-closed chain (above 0.5% degraded output it falls back to the reference path for all studies, or gives no root). It replaces an earlier row whose mirror arm could not run in the rerun.
- **57017206:** Four-view anatomical-mirror TTA on the nartaa r384 checkpoint (320 crop plain+mirror as in the reference path, plus 384 full-field plain+mirror, equal weights predeclared), rerun-safe (reader canary cap only in canary mode; visible gate only on the 3 visible studies; two-view fallback). It replaces an earlier four-view row that produced no file.

### Caveats

- **Three studies per run.** Steady-state values average one or two studies. Reference two-view GPU time ranges from 1.85 s to 1.98 s across six rows, a spread of about 7%, so this card does not treat build-to-build differences under about 7% as meaningful.
- **Hidden-study prep is unknown.** Prep was measured on three visible studies. If hidden DICOM files decode more slowly, the 3× prep column is the relevant projection. For one view with speed flags it moves the projection from 633 s to 1,792 s.
- **Main-phase prep was higher.** Main-phase rows measured 1.39 to 1.44 s of prep per study, against 1.01 to 1.12 s in the parity reference rows. The cause is not established. CPU contention between the two GPU consumers is one candidate.
- **Startup is included, platform time is not.** Projections add the receipt's startup seconds only. They exclude queueing, notebook commit overhead and any other platform time. The official RuntimeSeconds definition was not verified.
- **Four-view builds need care.** Official row 57017206 (four views) has a public score of 0.505, far below the other official rows in this card. The snapshot records no cause, and this card does not establish one. The team's suspected cause, not tested here, is that four-view builds switch a shared input-resolution setting and that two GPU consumers ran at once in the main phase. The four-view timings measure the code on visible studies. They do not show that the arm is safe.
- **Plain-view study totals.** See Timing scope. Study-total comparisons for bench V22 and bench V24 are not meaningful.
- **Visible prediction hash.** visible_prediction_sha256 is the SHA-256 that the reader records for its prediction file on the three visible studies. All 13 native receipts carry the same value. The snapshot descriptions of rows 57024289, 57024205 and 57017206 give that same value as their submission.csv sha256, although their public scores differ. The field therefore has no established meaning, and it does not link a receipt to a public row.
- **Flag names differ by build.** The name full means different flag sets in different builds. The flags column is authoritative.
- **Receipt detail varies.** Bench-2 V5, V6 and V8 have native receipts only. Per-stage files exist for bench V21 to V25 and bench-2 V2, V3, V4, V7 and V9. Each native receipt also records the reader's SHA-256 values for its checkpoint and its source (checkpoint_sha256 and source_sha256). Neither file is shipped, and build_tables.py does not read them.
- **Not included.** No images, labels, predictions, model weights, study identifiers, session identifiers or file paths.

## Credits

- **Reader structure.** Follows the public Kaggle notebook nartaa/rsna-knee-0949-anatomical-mirror (https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) by nartaa (Danial Zakaria), which is listed as Apache 2.0 on its Kaggle page (checked 2026-10-10). The reader source is not shipped.
- **Model lineage.** Credited in the reader source to dreaddevelopment (Raptor CoAtNet and MIL lineage). Nothing shipped here uses it, and no licence is stated for it.
- **Weights.** From the public dataset nartaa/rsna-knee-publication-swa-weights-20261007. Its licence is "Other", as shown on the Kaggle page on 2026-10-10 (not independently re-verified). No weights are redistributed here.
- **Efficiency arithmetic.** The efficiency formula and its two constants (Benchmark about 0.5, maxAUC about 0.96) are assumptions of this card, not values verified against the competition rules. The constants are approximations, and the RuntimeSeconds definition was not verified against the rules.

### Dependencies

build_tables.py and efficiency_calculator.py import only the Python standard library. The table lists the third-party components that the measurements and the How to load snippets depend on. Licence names are as published by each project; they were not re-read for this card.

| Dependency | Version recorded | Used for | Licence |
|---|---|---|---|
| PyTorch | 2.11.0 (CUDA 12.8 build) | Reader inference on the two T4s | BSD-3-Clause |
| timm | 1.0.29 | Recorded by the reader | Apache-2.0 |
| NumPy | 2.1.3 | Recorded in every native receipt | BSD-3-Clause |
| pandas | not recorded | How to load snippets only; not shipped | BSD-3-Clause |
| CUDA runtime | 12.8, bundled with the PyTorch build | GPU execution | NVIDIA CUDA licence terms; not redistributed |

### How to cite

prvsiyan (2026). RSNA Knee Inference Timing T4 [Dataset]. Kaggle. https://www.kaggle.com/datasets/prvsiyan/rsna-knee-inference-timing-t4 (CC-BY-NC-SA-4.0)

## Licence

- **Dataset: CC-BY-NC-SA-4.0.** The Kaggle licence field for this dataset is CC-BY-NC-SA-4.0, Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International. It covers the data files (timings.csv, summary.csv, native_runs.csv, the receipts__ files and receipts__index.csv), schema.json, MANIFEST.json, dataset-metadata.json, NOTICE.md, LICENSE and this card. Full legal code: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
- **Code: Apache-2.0.** build_tables.py and efficiency_calculator.py are licensed under the Apache License 2.0. The full text is in LICENSE-APACHE-2.0, and NOTICE.md gives the per-file terms.
- **Public score facts: CC-BY-4.0 as well.** The official row ids, public scores and snapshot date in the Public scores table are compilations of public Kaggle listing facts. They are also offered under CC BY 4.0 (attribution), as NOTICE.md explains. The rest of the data files are CC-BY-NC-SA-4.0 only.
- **Basis.** The timings are measurements of the competition's reader on the competition's visible sample studies. They are treated as covered by the competition's data terms, and the non-commercial and share-alike conditions are applied as the conservative choice. The competition rules page was not read in full when this card was prepared. Confirm the rules before any commercial use.
- **Third-party material.** Not redistributed: the reader source, model weights and runtime libraries (see Credits). No competition images, labels, predictions or identifiers are included.

## Changelog

- **v1 (2026-10-10).** First release. 10 timing receipts and 13 native receipts, 630 timing rows, 26 summary rows and 13 native rows. Study and session identifiers removed. Includes efficiency_calculator.py, build_tables.py, the sanitized receipts and the receipt index. Licence CC-BY-NC-SA-4.0, with the two code files under Apache-2.0.
