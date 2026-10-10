# RSNA Knee Fail-Closed Reader + Gate (Python code)

Python code for running a medical-image model on a hidden test set without ever writing a silently degraded submission. The pack holds an anatomical-mirror reader in four versions, a four-view driver, a fail-closed gate with a reference fallback, an owned-process watchdog, and offline tests. It contains **no model weights, no competition data, no labels and no predictions**.

## File names in this upload

Kaggle does not keep sub-folders in a dataset upload, so every file sits at the top level. A file's original folder becomes a prefix separated by a double underscore: `src/owned_watchdog.py` is uploaded as `src__owned_watchdog.py`, and `docs/HOWTO_safe_hidden_reruns.md` as `docs__HOWTO_safe_hidden_reruns.md`. The text of this card and of the documents uses the original paths (`src/`, `gate/`, `tests/`, `docs/`). Flattening does not change file contents, so the sha256 values quoted below still match.

Only the smoke test runs directly from the upload folder (`tests__smoke_standalone.py`). Every other command that uses the original paths needs the folder layout restored first. See "Restore the folder layout".

## The four reader variants

The four reader variants are related as follows. `src/public_mirror_reader_v36_canary_cap.py` differs from the pinned `src/public_mirror_reader.py` by one changed line. `src/public_mirror_reader_v20.py` is a separate reader: against the pinned reader it is 60 lines removed and 411 added (standard diff). `src/public_mirror_reader_v24_canary_cap.py` differs from V20 by one changed line.

## What you can use it for

- **A fail-closed export gate.** Count degraded outputs from the saved probabilities before any file is exported. Refuse above a limit, fall back to a simpler route, and refuse again if that also degrades. The gate is in `gate/launcher_v38g.py`, and its rules are in `docs/FAIL_CLOSED_RULES.md`.
- **An owned-process watchdog.** Run a child in its own process group, wait for it, and get a receipt that says whether teardown finished. The helper is in `src/owned_watchdog.py` and is stdlib-only.
- **Canary-only input checks.** Apply a study-count check only in canary mode, so a full rerun is not refused at startup. Two reader variants do this.
- **A checklist and tests for hidden reruns.** `docs/HOWTO_safe_hidden_reruns.md` lists the failure modes and the checks. `tests/smoke_standalone.py` runs on a plain CPU with no notebook.
- **A four-view driver that does not switch shared state.** Two device consumers can run at once without changing each other's input resolution. `tests/test_shared_state_v38_gated.py` checks this against a serial run, when a notebook is supplied.

## Terms

- **Hidden rerun.** A notebook is re-run by Kaggle on a test cohort that nobody can inspect. A failure there is silent unless the code makes it loud.
- **Canary.** A short run on a few studies (the canary cap, for example 3) that checks a notebook before the full run.
- **Degraded study.** A study whose output is a constant 0.5 row, or whose volume could not be built. Defined in `docs/FAIL_CLOSED_RULES.md`, section 1.
- **Root.** The submission file that the gate accepts as final output. Only one root is written, and only after its receipt.
- **Receipt.** A JSON record written by each run with its route, counts, mode and file hashes.
- **Owned process group.** A child process and everything it starts, which the watchdog can signal and wait for as one unit.
- **Four-view.** The reader's 320-pixel plain and mirrored views, plus 384-pixel plain and mirrored views (sizes as configured in the shipped reader and four-view driver), averaged with equal weights over the views a study has.
- **Fail-closed.** The run refuses to write a root rather than write a degraded one.
- **Reference route.** A second run of the simpler two-view reader on every study, used only when the four-view output is degraded above the limit.
- **Chain.** A notebook pipeline that runs the reader or driver, the gate and the export, from the reader to the root file.
- **Version labels.** These are our own measurement-run labels for reader variants, driver versions, notebook cell ids and receipts. V1, V14, V19, V20, V21, V22, V23, V24, V25, V34, V36 and V38 (and their lower-case forms in cell ids) appear in file names, receipt strings, cell ids and some comments. They are kept so that the pinned hashes, cell ids and receipt names still match. V1 names the attached checkpoint dataset version that the readers check against a pinned sha256. V14 names the fault-policy version. V19 is the earlier reader whose reference behaviour V20 keeps. V25 is our own label for row 57017206, the row that the post-mortem analyses, and V36 labels the two-view canary-cap reader variant used for the 320 pair of the four-view rows. Some code comments, and one receipt schema string (`bench-v20-timing-1`), use the word "bench" for our measurement runs, such as "bench V23". "V2" in "nartaa public V2" is the upstream notebook's own version number, not ours.

## Why it is useful

A hidden rerun fails silently, so three failure modes cost real submissions:

1. **Refused at startup.** A study-count check that applied in every mode ended the run before a submission file was written. Official rows 57016738 and 57016750 have no score, and the submission snapshot records Kaggle's error that no expected submission file was produced. The cause is inferred from the code and a local reproduction, not from the rerun log, which was not fetched.
2. **Degraded in the middle.** A per-study failure falls back to a neutral 0.5 probability, and the rank export can give tied neutral rows distinct ranks (check R1 of the smoke test shows the tie). Row 57017206 scored 0.505 on the public split. Whether its output was degraded in this way is not established (see "What is not established").
3. **A race.** An earlier four-view driver set the reader's global input resolution to 384 for its 384 pass, while another consumer could be predicting at 320. The race was reproduced synthetically by the publisher; that reproduction is not part of this pack. The fix is one model per device and no writes to the global input resolution.

The code in this pack is what came out of those failures:

- The gate counts degraded studies from the output itself and refuses to export above a limit. Above that limit it recomputes every study on a simpler path, and refuses again if that also degrades.
- The four-view driver does not change the reader's input resolution. Inside `main()` it installs its own `predict_pair` on the reader module once, before the reader starts, and keeps lock-guarded counters.
- The study-count check applies only in canary mode in the two canary-cap readers.
- Every run writes its receipt before the root file. The root is created exclusively and then re-hashed against its receipt.

Start with `docs/HOWTO_safe_hidden_reruns.md` for the checklist, and `docs/FAIL_CLOSED_RULES.md` for the rules and every receipt field.

## Start here

Run the smoke test from the upload folder. It needs no notebook, checkpoint or GPU:

```bash
python3 -I tests__smoke_standalone.py
```

The gate limit is `max(0.5% x N, 1)`, where N is the number of test studies; the 0.5% and the formula are as written in the shipped gate launcher (`DEGRADE_FRACTION`). As a worked example at N = 1,300 the limit is 6.5, so six degraded studies pass and seven fail:

```python
def degrade_limit(n_studies, fraction=0.005):
    return max(fraction * n_studies, 1.0)

assert degrade_limit(3) == 1.0
assert abs(degrade_limit(1300) - 6.5) < 1e-9
```

The watchdog, as a runnable snippet. It needs the restored layout (see below), so `PACK` is the folder that contains `src/`:

```python
import importlib.util, os, sys, time
PACK = "/kaggle/working/pack"   # or "." from the restored pack root
spec = importlib.util.spec_from_file_location("owned_watchdog", os.path.join(PACK, "src", "owned_watchdog.py"))
wd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wd
spec.loader.exec_module(wd)
receipt = wd.run_owned([sys.executable, "-c", "print('child ok')"], timeout=60,
                       absolute_deadline=time.monotonic() + 60, cleanup_on_success=True, cwd=PACK, env=None)
print(receipt["status"] == "PASS_OWNED_COMMAND" and receipt["command_returncode"] == 0)
```

## Restore the folder layout

Use this to copy the flattened files into the original folder layout (`src/`, `tests/`, `gate/`, `docs/`) under `/kaggle/working/pack`. The command reads the flattened names and nothing else. Change `SRC` if you run it outside Kaggle:

```bash
SRC=/kaggle/input/rsna-knee-reader-code-pack
DST=/kaggle/working/pack
mkdir -p "$DST"
for f in "$SRC"/*; do
  name=$(basename "$f")
  case "$name" in
    *__*) dir=${name%%__*}; mkdir -p "$DST/$dir"; cp "$f" "$DST/$dir/${name#*__}" ;;
    *)    cp "$f" "$DST/" ;;
  esac
done
cd "$DST"
```

After this, the commands in the rest of this card work from `$DST`. Run the smoke test from the upload folder, before restoring, because it reads the flattened names.

## What is inside

| Uploaded file | Original path | Role |
|---|---|---|
| `src__public_mirror_reader.py` | `src/public_mirror_reader.py` | The pinned reader (sha256 `e9ec648e…`). Unmodified. Its study-count check applies in every mode. |
| `src__public_mirror_reader_v36_canary_cap.py` | `src/public_mirror_reader_v36_canary_cap.py` | The same file with one line changed: the study-count check applies only in canary mode. sha256 `96af4b84…`. |
| `src__public_mirror_reader_v20.py` | `src/public_mirror_reader_v20.py` | V20 reader with parity-gated speed flags and timing receipts. Study-count check in every mode. |
| `src__public_mirror_reader_v24_canary_cap.py` | `src/public_mirror_reader_v24_canary_cap.py` | The V20 reader with the same one-line canary-only change. |
| `src__four_view_arm.py` | `src/four_view_arm.py` | Four-view driver (sha256 `a76e9d7f…`). Calls the reader's own `predict_pair` for the 320 pair, adds 384 plain and mirrored views with one model per device, and fails soft per view. |
| `src__owned_watchdog.py` | `src/owned_watchdog.py` | Owned process-group runner. Its header declares SPDX Apache-2.0 and names no upstream project. Unmodified (sha256 `54636411…`). See Credits. |
| `gate__launcher_v38g.py` | `gate/launcher_v38g.py` | Fail-closed gate: degraded-fraction limit, reference route, receipts, root-last export. A notebook-cell body with placeholders (see below). |
| `gate__validation_v38g.py` | `gate/validation_v38g.py` | Final aggregate validation cell for the gate. |
| `tests__smoke_standalone.py` | `tests/smoke_standalone.py` | Standalone CPU smoke test. Needs no notebook, checkpoint or GPU. Standard library, plus numpy for one optional check. Reads the flattened names. |
| `tests__replay_active_cells.py` | `tests/replay_active_cells.py` | Replays a notebook's active cells in a fake `/kaggle` tree, with fatal or non-fatal expectations. Needs a notebook. |
| `tests__test_chain_v38_gated.py` | `tests/test_chain_v38_gated.py` | Scenario harness for the gate: visible commit, budget refusal, 1,300-study reruns with injected failures, fail-closed branches. Needs a notebook. |
| `tests__test_shared_state_v38_gated.py` | `tests/test_shared_state_v38_gated.py` | Shared-state test: two concurrent device consumers, bitwise equal to serial, no writes to the reader's input resolution, locks held. Needs a notebook and a random checkpoint. |
| `tests__test_reader_v20_cpu.py` | `tests/test_reader_v20_cpu.py` | CPU equivalence tests for the V20 reader against an earlier reader, on random weights and reduced geometry. Needs a notebook that holds the earlier reader, and scipy. |
| `tests__test_fourview_cpu.py` | `tests/test_fourview_cpu.py` | CPU tests for the four-view driver: bitwise combination, byte-identical rank exports, view differences, per-view timing. Needs a notebook. |
| `docs__HOWTO_safe_hidden_reruns.md` | `docs/HOWTO_safe_hidden_reruns.md` | Checklist for hidden reruns: canary caps, mount layouts, resolution-switch races, degraded outputs, budgets. |
| `docs__FAIL_CLOSED_RULES.md` | `docs/FAIL_CLOSED_RULES.md` | The gate rules, the root sources, every receipt field, and a comparison with earlier gate versions. |
| `docs__V25_POSTMORTEM.md` | `docs/V25_POSTMORTEM.md` | Analysis of row 57017206 (public score 0.505): the failure mechanisms considered, the evidence for each, the rule that followed, and open items. |
| `README.md`, `NOTICE.md`, `LICENSE` | (same) | This card, lineage and changes, and the Apache-2.0 text. |
| `MANIFEST.json` | (same) | sha256 and byte size of every other file in this folder. |
| `dataset-metadata.json` | (same) | Kaggle dataset metadata (title, description, licence, resources). Not listed in `resources`, because it is the metadata itself. |

## Columns and fields

The pack has no CSV or other table files, so it has no data columns. Two structured records are documented here:

- **`MANIFEST.json`**, one record per file, with three fields: `path` (the uploaded file name), `bytes` (integer byte count) and `sha256` (64 hexadecimal characters). Its top-level keys are `dataset` (the Kaggle dataset id), `generated` (the date the manifest was written), `scope` (the files it covers), `file_count` (the number of records) and `total_bytes` (the sum of the listed byte counts).
- **The results table** in "Results recorded for these chains", with five columns: `Kaggle submission id` (the row id that Kaggle returned for the submission), `What was tested` (paraphrased in part from the row's own description in the submission snapshot), `Notebook version id` (the `scriptVersionId` of the notebook version that made the submission), `Public score` (the public-leaderboard score, three decimals, or no score when the submission produced no file) and `Notes` (what the row shows and what this card says about it).

## How the pieces connect

```
 320 pair: plain + mirror (reader's own predict_pair)
   -> four-view arm adds 384 plain + 384 mirror (one model per device, fail-soft per view)
   -> count degraded studies (320 pair is a constant 0.5 row, or volume build failed)
   -> degraded <= max(0.005 * N, 1) ?
        yes -> root_source = four_view
        no  -> reference route: reader's own two-view run on every study (no 384 views)
                 -> degraded <= limit ?  yes -> root_source = reference_v36
                                         no  -> no root; receipt; raise
 receipt written first; root written last (exclusive create), then re-hashed against the receipt
```

## Use it on Kaggle

1. **Attach this dataset.** It mounts at `/kaggle/input/rsna-knee-reader-code-pack/`. The pack contains no checkpoint and no competition data. Attach the competition and the upstream checkpoint separately, under their own terms.
2. **Restore the folder layout** with the snippet above, into a writable folder such as `/kaggle/working/pack/`. Python then writes its bytecode caches there rather than into the read-only input.
3. **Choose the reader for a full run.** The pinned reader and the V20 reader require `--max-studies` in 1..16 in every mode, including inference, even though inference ignores the value. For a full test-set run either pass `--max-studies 16` (or less) to any of the four readers, or use one of the two canary-cap variants, which apply the check only in canary mode. The four-view driver imports the pinned reader, so the same rule applies to it. The gated launcher passes its canary value for this reason.
4. **The gate is notebook-bound.** `gate__launcher_v38g.py` (`gate/launcher_v38g.py`) is the body of one notebook cell. It needs the placeholders filled and constants defined in earlier cells (`CONFIG`, `RUN_TAG`, `FULL_DEADLINE`, `SOURCE_DIR`, `READER_PATH`, `WATCHDOG_PATH`, the sha256 pins and `require`). It will not run on its own. Its `REFERENCE_320` placeholder holds no values in this pack.

## Quick start

**1. Smoke test (no notebook, plain CPU).** From the upload folder:

```bash
python3 -I tests__smoke_standalone.py
```

It checks the manifest, the pinned hashes, the one-line variant claims, the study-count refusal and its absence in the canary-cap readers, the watchdog, the gate arithmetic and the budget projection. The tie-hazard check needs numpy, which `-I` hides when numpy is only in a user site-packages folder; drop `-I` to run it. Exit status 0 means every check passed.

**2. Owned watchdog.** Runs a child in its own process group and returns a receipt. It needs the restored layout, and `PACK` is the folder that contains `src/` (on Kaggle, `/kaggle/working/pack`).

```python
import importlib.util, os, sys, time
PACK = "."  # the restored pack root: the folder that contains src/
spec = importlib.util.spec_from_file_location("owned_watchdog", os.path.join(PACK, "src", "owned_watchdog.py"))
wd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wd
spec.loader.exec_module(wd)
receipt = wd.run_owned([sys.executable, "-c", "print('child ok')"],
                       timeout=60, absolute_deadline=time.monotonic() + 60,
                       cleanup_on_success=True, cwd=PACK, env=None)
ok = receipt["status"] == "PASS_OWNED_COMMAND" and receipt["command_returncode"] == 0
print("watchdog ok:", ok)
```

**3. The gate limit.** The arithmetic, which the gate uses as written:

```python
def degrade_limit(n_studies, fraction=0.005):
    return max(fraction * n_studies, 1.0)

assert degrade_limit(3) == 1.0
assert abs(degrade_limit(1300) - 6.5) < 1e-9   # 6 degraded studies pass, 7 do not
```

After a gated run, the launcher writes `public_v38_receipt.json` in its stage folder. Read `root_source`, `degraded_studies` and `degrade_limit` from it.

**4. Reader canary run.** Needs the restored layout, the upstream checkpoint attached and a CUDA device, otherwise the reader exits early:

```bash
python3 src/public_mirror_reader_v24_canary_cap.py \
  --input-root /kaggle/input \
  --competition-root /kaggle/input/competitions/rsna-knee-abnormality-detection \
  --output-dir /kaggle/working/reader_out \
  --mode canary --max-studies 3 --prep-workers 2
```

## Runtime requirements

- **Python.** The smoke test runs on Python 3.9 or newer. The notebook-bound tests were written for Python 3.11 or newer, and the usage line of `test_reader_v20_cpu.py` names Python 3.13.
- **Pinned by the launcher.** The launcher checks the installed `torch` and `timm` versions against values in its `CONFIG`, which is defined in a notebook cell that is not in this pack. The usage lines of the tests name torch 2.11 and timm 1.0.29.
- **Other imports.** numpy, pandas, opencv-python (`cv2`), pydicom, timm and torch. pydicom is not pinned. The reader imports `pydicom.pixels` and falls back to `pydicom.pixel_data_handlers.util`. The CPU test `test_reader_v20_cpu.py` also needs scipy, which its usage line names. Credits lists the licence of each.
- **What was run here.** The smoke test was run from the upload folder on Python 3.9.6, in the flattened layout. With `-I`, every check except R1 passed; R1 is skipped because numpy is not visible to an isolated interpreter on that machine. Without `-I`, every check passed, including R1. The notebook-bound suites were not run.

## Results recorded for these chains

Each row is a Kaggle submission. The score is the official public-leaderboard value for that row, as recorded in the Kaggle submission snapshot taken on 2026-10-10 at 11:47 UTC. Rows with no score had no submission file. The "What was tested" column paraphrases in part the row's own description from the same snapshot; internal build labels are paraphrased out. The "Notebook version id" column is the `scriptVersionId` in that snapshot's notebook link. The "Notes" column says what the row shows and what this card does and does not claim about it.

| Kaggle submission id | What was tested (paraphrased in part from the row's description) | Notebook version id | Public score | Notes |
|---|---|---|---|---|
| 56949940 | fault-tolerant anatomical-mirror reader (nartaa recipe, Apache code, checkpoint weights under their own terms) with the corrected owned launcher | 356373560 | 0.949 | Reference row for the four-view comparison below. |
| 57024205 | four-view anatomical-mirror TTA on the nartaa r384 checkpoint (320 crop plain+mirror + 384 full-field plain+mirror, equal weights predeclared), race-free driver, fail-closed chain | 356846883 | 0.949 | Same public score as row 56949940. Its description calls it a clean replicate of row 57017206. |
| 57024289 | plain-view-only anatomical reader (nartaa r384 recipe, mirror view off), fail-closed chain | 356842162 | 0.948 | Its description says it measures the plain reader's AUC for the Efficiency Prize break-even. |
| 57024089 | fault-tolerant anatomical mirror (nartaa recipe), per-finding rank blend 0.55/0.45 predeclared | 356849298 | 0.947 | Lower public score than row 56949940. |
| 57017206 | four-view anatomical-mirror TTA on the nartaa r384 checkpoint (320 crop plain+mirror from the two-view canary-cap reader variant plus 384 full-field plain+mirror, equal weights predeclared), rerun-safe (reader canary cap only in canary mode; visible gate only on the 3 visible studies; two-view fallback) | 356834227 | 0.505 | Far below the 0.949 rows. The cause is not established; see "What is not established". |
| 57016738 | plain-view-only anatomical reader (nartaa r384 recipe with the mirror view off; exact prep flags; on-GPU plain-half equality 0.0 on the visible studies) | 356486430 | no score (no submission file) | Kaggle's error text reads, in part, "did not output the expected submission file." The cause is inferred from the code; the rerun log was not fetched. |
| 57016750 | four-view anatomical-mirror TTA on the nartaa r384 checkpoint (320 centre-crop plain+mirror from the two-view canary-cap reader variant, plus 384 full-field plain+mirror; equal-weight sigmoid average, per-column rank export; two-view self-check equals the 0.949 CSV on the visible studies) | 356520735 | no score (no submission file) | Same error text as row 57016738. The cause is inferred from the code; the rerun log was not fetched. |

## What is not established

- **The 0.505 row.** The publisher's analysis of row 57017206 considers two causes: a process-wide input-resolution switch shared by two consumer threads, and a per-study failure at scale, such as an out-of-memory error. It favours the first, with moderate to high confidence, but the causal role of the resolution race is not proven. The same analysis records that neither the official score nor the source audit establishes that the submitted table was a literal constant 0.5 table.
- **Hidden-scale runs.** No hidden-scale timing or receipt for the driver in this pack is included. Its budget constants are labelled below.
- **Cause of the refusals.** The rerun logs for rows 57016738 and 57016750 were not fetched. The cause rests on the code and a local reproduction. The snapshot's error text is the only official record.
- **Budget constants are estimates.** The 3.0 s per study for the reference route (`REFERENCE_PER_STUDY_S` in the shipped gate launcher) is an estimate, labelled as one in the receipt. The constant `PER_STUDY_SERIAL_S` (6.7 s) in `src/four_view_arm.py` is, according to its own comment, a serial timing on the visible studies, taken with an earlier driver. Measure both on your hardware before relying on them.
- **Wrong-but-finite outputs.** The gate counts only constant 0.5 rows and build failures. A 320 model that accepts 384 input and returns finite but wrong values is not counted.
- **Tests.** The notebook-bound suites were not run in the staging environment. Only the smoke test was run, from the upload folder.
- **Performance.** No speed claim is made for the four-view driver beyond its budget projection.

## Licence

- **Kaggle licence field: `apache-2.0`.** The pack's primary content is code, tests and documents.
- **Code, tests, documents, this README and NOTICE.md: Apache-2.0.** See `LICENSE`. That file is the canonical Apache License 2.0 text, with the template appendix placeholder and no copyright holder named in it. The pack's copyright is stated in NOTICE.md (Copyright 2026 prvsiyan).
- **Results table (this README): CC-BY-4.0.** The table compiles the publisher's own Kaggle submission ids, notebook version ids and the public scores shown for them. That compilation is offered under CC-BY-4.0, as NOTICE.md states. The code is offered under Apache-2.0 and the table under CC-BY-4.0. Apache-2.0 and CC-BY-4.0 content may sit in one pack.
- **Upstream notebook.** The reader and the four-view driver derive from nartaa's RSNA Knee 0949 Anatomical Mirror notebook. Its [Kaggle page](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) states Apache 2.0 (checked 2026-10-10). The five derived files, with attribution, are listed in `NOTICE.md`.
- **Not included.** The learned checkpoint, which has its own publisher terms; competition data; OAI data; label tables; and any prediction table.
- **Per-file detail and the OAI / NDA wording** are in `NOTICE.md`.

## Credits

- **nartaa (Danial Zakaria)**: the plain and anatomical-mirror inference recipe and the selected learned checkpoint. Notebook: [RSNA Knee 0949 Anatomical Mirror, V2](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397).
- **dreaddevelopment**: the Raptor CoAtNet / MIL architecture and geometry lineage ([raptor-knee-widedense](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)). No baseline weights or upstream model code are included.
- **timm (Ross Wightman and contributors)**: the CoAtNet implementation the reader instantiates. timm is Apache-2.0 and is not bundled ([huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)).
- **Checkpoint publisher and weak-label contributors.** The checkpoint is attached from [nartaa/rsna-knee-publication-swa-weights-20261007](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) under that publisher's own terms. Weak-label contributions are credited, by that publisher, to [stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels), [pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) and [riadmohamed42](https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification). None of their label tables are included here.
- **RSNA and the competition data contributors.** Data use remains subject to the [RSNA Knee competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules). The rules page was not readable when checked on 2026-10-10, so the rules text has not been checked.
- **Owned-process watchdog (`src/owned_watchdog.py`).** The publisher's own helper for this pack. Its file header declares SPDX `Apache-2.0` and names no upstream project or author. It is not among the upstream-derived files listed in NOTICE.md.
- **Publisher:** prvsiyan.

### Dependency credits

Each third-party package imported by a shipped file, the package named in a usage line, and each model component the measurements depend on:

| Component | Where it is used | Licence |
|---|---|---|
| PyTorch (`torch`) | reader, four-view driver, tests | BSD-3-Clause |
| NumPy (`numpy`) | gate launcher, reader, four-view driver, tests | BSD-3-Clause |
| pandas | gate launcher, reader, tests | BSD-3-Clause |
| SciPy (`scipy`) | named in the usage line of `tests__test_reader_v20_cpu.py`; not imported by a shipped file | BSD-3-Clause |
| OpenCV (`cv2`, opencv-python) | reader, shared-state test | Apache-2.0 |
| pydicom | reader, tests | MIT |
| timm | reader, four-view driver, tests (builds the backbone the reader names) | Apache-2.0 |
| nartaa checkpoint (not included) | attached separately at run time | the publisher's own terms; the reader docstring describes separate research and educational terms. Not re-verified in this revision |
| dreaddevelopment Raptor CoAtNet / MIL lineage (no code or weights included) | the architecture and geometry the reader names | the terms of the dataset's Kaggle page. Not restated or re-verified in this revision |

The licences in the first seven rows are those of the projects, as listed here. They were not re-read from each project's repository in this revision.

### OAI / NDA acknowledgement (verbatim from the publisher's credits)

The following paragraphs are reproduced verbatim from the upstream publisher's credits and dataset description. They acknowledge the upstream research lineage. Distributing this code does not assert that this pack obtained or redistributed controlled-access OAI records.

Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): 10.15154/0hcg-f676.

[Shared NDA Study 3407](https://nda.nih.gov/study.html?id=3407) · [DOI: 10.15154/0hcg-f676](https://doi.org/10.15154/0hcg-f676). Official acknowledgement source: [NDA manuscript preparation](https://nda.nih.gov/nda/manuscript-preparation).

This pack contains no OAI or competition MRI, no reports, no label tables, no participant identifiers and no checkpoint payloads.

## How to cite

prvsiyan (2026). *RSNA Knee Fail-Closed Reader + Gate (Python code)* [Dataset]. Kaggle. Upstream recipe: nartaa, *RSNA Knee 0949 Anatomical Mirror* (kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror).

## Changelog

- **1.0 (2026-10-10).** First release: readers in four versions, four-view driver, fail-closed gate template, owned-process watchdog, offline tests, standalone smoke test and documents. The upstream notebook's licence was confirmed as Apache-2.0 before publication (see `NOTICE.md`).
