# Hidden-Rerun Safety Kit: Checklist + Tested Code

## What

A small, tested kit for one failure: a code-competition notebook that passes its commit run and then fails, or quietly degrades, on the hidden rerun that produces the score. It contains:

- a ten-point checklist (`CHECKLIST.md`), each check with the code that implements it and its evidence;
- eight Python modules and offline tests (no Kaggle calls, no credentials needed);
- an example notebook for the offline replay;
- platform facts observed on Kaggle in October 2026, with IDs and statuses, in prose and in CSV form, plus two more CSV tables (the checklist and the ledger excerpts);
- a post-mortem of a constant 0.5 table that the campaign ledger records as written and submitted on a hidden rerun.

The kit replays your active notebook cells in a fresh directory tree, gates the output before the root is written, and guards the launch and the submission.

## Why

The commit run is what you see. The hidden rerun is what is scored. The two can differ in input layout, time budget, data size and environment variables. In one campaign, the campaign ledger records that a rerun wrote a constant 0.5 table and links that table to the submission with public score 0.505 (official row 57017206). The row's public description does not mention the table. The degraded count was recorded, but nothing stopped the root from being written.

## Files

Folders are flattened in this download. The kit file `code/root_gate.py` ships as `code__root_gate.py`, and `data/checklist.csv` ships as `data__checklist.csv`. The checklist, the platform facts and the code refer to the folder form. The restore step under "How to load" rebuilds the folders.

| Shipped file | Folder path in the kit | Contents |
|---|---|---|
| `CHECKLIST.md` | `CHECKLIST.md` | The ten hidden-rerun checks, each with the code that implements it and its evidence IDs. Read first. |
| `README.md` | `README.md` | This dataset card. |
| `LICENSE` | `LICENSE` | Apache License 2.0, full text. It covers the Apache-2.0 files listed in `NOTICE.md`. |
| `NOTICE.md` | `NOTICE.md` | Licence by file. Apache-2.0 for the code, tests, documentation and configuration. CC-BY-4.0 for the three compilation files (`data__platform_facts.csv`, `data__ledger_excerpts.csv`, `docs__PLATFORM_FACTS.md`). |
| `MANIFEST.json` | `MANIFEST.json` | sha256 and byte size of every content file, under the shipped names. |
| `code__launch_preflight.py` | `code/launch_preflight.py` | Pre-launch rules: raw quota parsing that refuses on a missing or unparseable field, the two-session limit, and a free-GPU check. Pure functions, tested offline. |
| `code__launch_worker.py` | `code/launch_worker.py` | Reference driver for one GPU notebook version. Its quota helper is untested on kaggle 1.7.4.5. It needs two helpers that are not included. |
| `code__mount_resolver.py` | `code/mount_resolver.py` | Finds an input folder under the competition, legacy competition, dataset and owner-prefixed dataset layouts, and names the roots tried when none matches. |
| `code__owned_process.py` | `code/owned_process.py` | POSIX owned process-group watchdog: runs one command in its own session with a deadline, signals only that group, and returns or raises with a receipt. |
| `code__raw_submissions.py` | `code/raw_submissions.py` | Read-only dump of the raw ListSubmissions JSON for the competitions named on the command line. Needs Kaggle credentials. |
| `code__replay_active_cells.py` | `code/replay_active_cells.py` | Offline replay of a notebook's active code cells in a fresh tree, with GPU, data and pip stubbed. Flags writes into directories that no active cell creates. Needs no credentials. |
| `code__root_gate.py` | `code/root_gate.py` | Degraded-study gate (0.5 percent, fail closed), a constant-row counter, CSV validation that returns False rather than raising, and staged-then-root publication with `os.replace`. |
| `code__submission_guard.py` | `code/submission_guard.py` | Submission guard: sha256 check, daily count by UTC date, exact-token refusal of a duplicate pending version, a dry-run mode, and a receipt for every outcome. |
| `code__tests__test_kit.py` | `code/tests/test_kit.py` | 53 offline unit and regression tests for the modules in `code/`. The integrity tests also check `MANIFEST.json` and the metadata against the files. |
| `data__checklist.csv` | `data/checklist.csv` | The ten checks as a table. Columns are listed under "Columns". |
| `data__platform_facts.csv` | `data/platform_facts.csv` | One row per platform fact, with its topic, observed value, UTC time, ledger source and status. Columns are listed under "Columns". Licence CC-BY-4.0 (see `NOTICE.md`). |
| `data__ledger_excerpts.csv` | `data/ledger_excerpts.csv` | Sanitised paraphrases of the campaign-ledger rows that the facts cite. The full ledger is not included. Columns are listed under "Columns". Licence CC-BY-4.0 (see `NOTICE.md`). |
| `docs__PLATFORM_FACTS.md` | `docs/PLATFORM_FACTS.md` | The platform facts in prose, grouped by topic, with IDs, statuses and the mount-layout reconciliation. Licence CC-BY-4.0 (see `NOTICE.md`). |
| `docs__degraded_output_postmortem.md` | `docs/degraded_output_postmortem.md` | Post-mortem of a constant 0.5 table that the campaign ledger records as written and submitted on a hidden rerun: the code path, the two candidate causes, and the standing fail-closed rule. |
| `examples__example_notebook.ipynb` | `examples/example_notebook.ipynb` | A four-cell example for the replay (one markdown cell, three code cells): a `!pip` line and an `os.makedirs` in one cell, a `%%writefile` into that folder, and a `%matplotlib` line. |

`dataset-metadata.json` is the Kaggle metadata for this dataset and is not part of the kit's content. `MANIFEST.json` does not list it or itself.

## Columns

Each CSV file has a header row. The names below are the exact header names.

**`data__checklist.csv`** (5 columns)

| Column | Meaning |
|---|---|
| `check_id` | Check number, 1 to 10, matching `CHECKLIST.md`. |
| `check` | The check, in one sentence. |
| `code_file` | The file in the kit that implements the check (folder path), or a note that it is a one-line rule. |
| `function` | The function or entry point in `code_file`. |
| `evidence_ref` | Evidence IDs (`PF-nn`) from `data__platform_facts.csv`, or a design-rule note. |

**`data__platform_facts.csv`** (7 columns)

| Column | Meaning |
|---|---|
| `fact_id` | Fact identifier, `PF-nn`. |
| `topic` | One of `push_size`, `gpu_sessions`, `rerun`, `daily_limit`, `mount_layout`, `canary_cap`, `degraded_output`, `root_publication`, `replay`, `rerun_flag`. |
| `fact` | The fact, in one or two sentences. |
| `observed_value` | The observed value, the estimate, or the rule. |
| `observed_utc` | Date and time of the observation, UTC, as written in the ledger or to the minute. |
| `source_ref` | Ledger timestamp(s) that match `ledger_utc` in `data__ledger_excerpts.csv`, separated by semicolons, or `none` with the reason given in `fact`. |
| `status` | `observed`, `estimate`, `rule` or `unverified`. |

**`data__ledger_excerpts.csv`** (6 columns)

| Column | Meaning |
|---|---|
| `excerpt_id` | Excerpt identifier, `LX-nn`. |
| `ledger_utc` | Timestamp of the ledger row, as written in the ledger. |
| `ledger_area` | Short label for the subject of the row. |
| `excerpt` | Sanitised paraphrase of the ledger row. |
| `supports` | The platform fact IDs (`PF-nn`) that this row supports, separated by semicolons. |
| `removed_from_original` | What was removed from the original row before it was summarised (for example worker or session identifiers, or internal labels). |

## How to load

1. Attach the dataset to a Kaggle notebook, or download it. An attached dataset is mounted read-only at `/kaggle/input/<slug>/`, so copy the folder to a writable place first.
2. Restore the folder layout in the copy. The loop moves each `__`-joined file name back into its sub-folder:

```bash
mkdir -p /kaggle/working/kit && cp -r /kaggle/input/<slug>/. /kaggle/working/kit/
cd /kaggle/working/kit
for f in *__*; do p="${f//__//}"; mkdir -p "$(dirname "$p")"; mv "$f" "$p"; done
```

3. Run the offline tests from the restored `code` folder. They need Python 3.9 or later and no credentials:

```bash
cd code && python3 -I tests/test_kit.py
```

4. Replay a notebook before any push. The positive case must pass. Remove the cell that creates the directory (the negative control), and the same replay must fail:

```bash
python3 -I code/replay_active_cells.py examples/example_notebook.ipynb --expect pass --competition-slug my-competition-slug
```

Find an attached competition or dataset folder, whichever layout the platform used:

```python
import sys; sys.path.insert(0, "code")
from mount_resolver import resolve_competition, resolve_dataset

comp_root = resolve_competition("my-competition-slug", marker="test.csv")   # raises, naming the roots tried
weights_root = resolve_dataset("my-dataset-slug", marker="model.bin", owner="my-owner")
```

Gate a root and publish it only after validation:

```python
from root_gate import degraded_count, check_degraded_fraction, csv_shape_ok, publish_root

degraded = degraded_count(prob_rows, receipt_count=receipt_fallbacks)  # larger of the two counts
check_degraded_fraction(degraded, total_studies)                       # raises above 0.5 %
publish_root("staging/submission.csv", "working/submission.csv",
             validate=lambda p: csv_shape_ok(p, expected_rows=total_studies, expected_columns=13,
                                             expected_ids=study_ids))  # ID column must match exactly
```

Check a submission before the single submit call:

```python
from submission_guard import decide, sha256_file, utc_date

verdict, detail = decide(rows, expected_sha=EXPECTED_SHA, actual_sha=sha256_file("artifact.py"),
                         tag="owner/notebook v7", daily_limit=DAILY_LIMIT, today=utc_date())
# verdict is "OK", "REFUSED_sha_mismatch", "REFUSED_daily_limit" or "REFUSED_pending_same_version"
```

`DAILY_LIMIT` is competition-specific. Read it from your own competition's rules or account page.

## Platform facts (short form)

The full list, with IDs and ledger sources, is in `docs/PLATFORM_FACTS.md` and `data__platform_facts.csv`.

- **Push size.** A push of 1,036,069 bytes on the wire was accepted, and one of 1,181,475 bytes was rejected (HTTP 400). The exact limit was not measured, so treat 1 MiB as an estimate (PF-02 to PF-04).
- **GPU sessions.** A third concurrent batch GPU session is rejected with "Maximum batch GPU session count of 2 reached" (PF-07, PF-08).
- **Mounts.** Competition data is mounted at `/kaggle/input/competitions/<slug>/`. An attached dataset is mounted at `/kaggle/input/<slug>/` (PF-12, PF-13). The owner-prefixed dataset path is an unverified candidate (PF-14).
- **Canary cap.** A canary-only cap that also applied in inference refused about 1,300 studies at startup. No root was written (PF-16).
- **Degraded output.** Official row 57017206 has public score 0.505. The campaign ledger records that a four-view arm of that submission degraded to a constant 0.5 table in the hidden rerun, and that the degraded count did not block the root (PF-17). The row's public description does not mention the degradation.
- **Rerun limit.** The campaign ledger records a hidden-rerun time limit of 32,400 s, not charged to quota, for submission row 56948803. The row's public description does not state a limit. The campaign's commit runs used a 1800 s cap (PF-09). The limit is as the ledger records it.
- **Unverified.** The daily reset was assumed to be 00:00 UTC (PF-11). The rerun flag `KAGGLE_IS_COMPETITION_RERUN` was not verified on Kaggle (PF-23).

## Provenance

- The platform facts are observations from one campaign of private notebooks on Kaggle in October 2026. Each fact carries its status (observed, estimate, rule or unverified). Each `source_ref` points to a campaign-ledger row, which is paraphrased in `data__ledger_excerpts.csv`. The full ledger is not included.
- Official row IDs were checked against a public submissions-list snapshot taken on 2026-10-10 at 11:47 UTC. In that snapshot, row 57017206 has public score 0.505 and row 56948803 has public score 0.949. The public descriptions of these rows do not mention a rerun time limit or a constant 0.5 table. Those two statements come from the campaign ledger only (excerpts LX-01 and LX-14).
- `docs/degraded_output_postmortem.md` is a design-level write-up, with run labels replaced by roles. The hidden-rerun log was not available. The two candidate causes are weighed in the document. The favoured one is a process-global switch shared by two consumer threads, which is not proven.
- `code/owned_process.py` is the campaign's owned process-group watchdog. The shipped copy is the canonical one.
- Generalised from campaign tools, with competition and notebook slugs, local paths, pipeline-specific identifiers and credentials removed: `code/replay_active_cells.py`, `code/launch_worker.py`, `code/submission_guard.py`, `code/raw_submissions.py` and `code/launch_preflight.py`. Written for this kit from the campaign's rules: `code/mount_resolver.py` and `code/root_gate.py`.
- No competition data, model checkpoints, third-party notebook code or private notebook identifiers are included.

## Credits

- Publisher and copyright holder: prvsiyan (Kaggle).
- Apache License 2.0 text, from the Apache Software Foundation (`LICENSE`).
- Observations were made on the Kaggle platform.
- Third-party packages imported by the shipped code. Each licence is as shown in the installed package metadata on 2026-10-10 (not independently re-verified against the package index). Every other import is from the Python standard library or from the kit's own modules, except `safe_push`, an external helper that `launch_worker` imports and that is not shipped. `launch_worker` also runs `fetch_version_output.py` as a separate process through `FETCH_HELPER`; that file is not shipped either, as stated under "Limits".

| Package | Used by | Licence | Version checked |
|---|---|---|---|
| `kaggle` (official Kaggle client) | `code__raw_submissions.py`, `code__submission_guard.py`, `code__launch_worker.py` (inside functions) | Apache-2.0 | 2.2.3 |
| `kagglesdk` | `code__submission_guard.py` (inside a function) | Apache-2.0 | 0.1.34 |
| `requests` | `code__submission_guard.py`, `code__launch_worker.py` (inside functions) | Apache-2.0 | 2.34.2 |
| `torch` (PyTorch) | Not imported by shipped code. Named in `docs__degraded_output_postmortem.md` among the exception sources of the reader arm behind PF-17. | BSD-3-Clause | 2.8.0 (local venv) |
| Reader checkpoint (a model, not a package) | Not shipped and not loaded by this kit. The four-view arm behind PF-17; see the bullet below the table. | Not stated in the public snapshot; not checked | n/a |

The Kaggle client and `kagglesdk` are needed only by the live-API helpers under "Limits". The offline tests and the replay do not import them.

- Model checkpoint. The four-view arm in the degraded-output observation (PF-17) ran a reader checkpoint from a public third-party notebook recipe. The checkpoint is not shipped, and this kit does not load it. Its licence is not stated in the public snapshot and was not checked for this kit.

## Licence

- Dataset licence field: `apache-2.0` (Apache License, Version 2.0). The full text is in `LICENSE`.
- Per-file terms are in `NOTICE.md`:
  - Apache-2.0: the code and its tests, `CHECKLIST.md`, `data__checklist.csv`, `docs__degraded_output_postmortem.md`, this README, the example notebook, `MANIFEST.json` and `NOTICE.md`.
  - CC-BY-4.0 (Creative Commons Attribution 4.0): `data__platform_facts.csv`, `data__ledger_excerpts.csv` and `docs__PLATFORM_FACTS.md`. These compile the publisher's own submission rows and public scores. The licence text is at https://creativecommons.org/licenses/by/4.0/legalcode.
- Both licences apply to content in this dataset, each to the files stated for it. Both require attribution.
- The CC-BY-4.0 files are paraphrases of the publisher's own campaign records. They contain no competition data.
- Copyright: prvsiyan. The licence field is also named in `dataset-metadata.json` and `NOTICE.md`.

## Limits

- The live-API helpers (`list_submission_rows`, `session_status`, `raw_submissions`, the submit step) need the `kaggle` package and credentials. They were not run against Kaggle for this kit. The offline tests cover the decision logic and the failure paths that use fakes.
- `quota_from_raw` in `launch_worker` calls `api.quota_view()`, which does not exist in kaggle 1.7.4.5 (checked against that release). The driver refuses to launch when the quota cannot be read, so this fails closed. The statement is version-specific: kaggle 2.2.3, installed locally on 2026-10-10, does define `quota_view`. The helper was not run against either version.
- `launch_worker` is a reference driver. It needs `safe_push.py` and `fetch_version_output.py`, which are not included.
- The owner-prefixed dataset layout, `/kaggle/input/datasets/<owner>/<slug>/`, is an unverified candidate (PF-14).
- The daily reset time was assumed to be 00:00 UTC and was not verified. Keep a retry, or a wait, in case the reset is a rolling 24-hour window.
- The rerun flag `KAGGLE_IS_COMPETITION_RERUN` is the name the campaign's design notes used. It is not in the campaign ledger or code. Its value on Kaggle was not verified (PF-23). Confirm it in your own runs.
- The replay and the watchdog use POSIX-only calls (`SIGALRM`, `os.killpg`, `os.getsid`). They were tested on macOS only, not on Linux, which is what Kaggle runs.
- The replay does not execute shell commands other than `mkdir`, and does not run `%%bash`-style cells. See `CHECKLIST.md`, check 7.
- The 1 MiB wire limit is an estimate from two measurements.

## How to cite

prvsiyan (2026). *Hidden-Rerun Safety Kit: Checklist + Tested Code*, version 2. Kaggle dataset, licence field apache-2.0 (data files CC-BY-4.0, see `NOTICE.md`).

## Changelog

- **2, revision** (October 2026): licence by file recorded in `NOTICE.md`, naming the three CC-BY-4.0 compilation files. Dependency table with licences added to Credits. Row attribution corrected: the public descriptions of rows 57017206 and 56948803 do not state a constant table or a rerun limit, so those statements are now attributed to the campaign ledger. The watchdog provenance names the shipped copy as the canonical one. An internal label is removed from the launch driver, and a budget figure is removed from the watchdog header. The kaggle `quota_view` statement is version-specific.
- **2** (October 2026): restructured the card around What, Why, Files, Columns, How to load, Provenance, Credits, Licence and Changelog. Named the exact CSV column headers. Flattened the sub-folders into double-underscore file names, with a restore step; the offline integrity test accepts the flattened names. Named the shipped watchdog copy as the canonical one. Licence set to apache-2.0 across the metadata, the licence file and this card. Reconciled the mount layouts across the checklist, platform facts and resolver. The replay translates IPython line and cell magics and treats a non-compiling cell as fatal. `csv_shape_ok` returns False instead of raising, takes an exact expected-ID list, and `publish_root` converts validator errors to `StagedOutputInvalid`. The submission guard matches version tokens exactly and writes a receipt for failed submit calls and failed re-reads. The pre-flight rules fail closed on a missing or unparseable quota. The launch driver refuses without `KNOWN_SLUGS` and counts unreadable sessions as active. The campaign-only budget helper is removed from the watchdog. Added the structured CSVs, the ledger excerpts, an example notebook and regression tests.
- **1** (October 2026): first release. Checklist, platform facts, post-mortem, eight code modules and offline tests.
