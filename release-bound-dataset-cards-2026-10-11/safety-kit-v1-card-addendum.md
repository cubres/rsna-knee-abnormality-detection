## Verify version 1 before using the code

For dataset **`prvsiyan/kaggle-hidden-rerun-safety-kit`**, numeric dataset ID **12489897**, version **1**, the `MANIFEST.json` downloaded from that exact version on **October 11, 2026, 17:50 UTC** is **3,431 bytes** with SHA-256:

```text
0cb4bd97842c329dcdbabc89e29bb6226cbbb95058016bf8987021d33f90b88e
```

The complete version-1 API listing contains 20 files: this manifest plus its 19 recorded content files. Their recorded sizes match the listing. The manifest records **158,955 content bytes**. This check downloaded only the manifest; content payload digests have not been independently fetched in this live check. Verify those files on the mount before running them.

Use the original MIT [read-only verifier](https://github.com/cubres/gemma-4-developer-agent/blob/764878bd6d214ef8aedea18a3027793bb9b65273/dataset-flatpack-verifier-2026-10-11/verify_flatpack.py) and its [guide](https://github.com/cubres/gemma-4-developer-agent/blob/764878bd6d214ef8aedea18a3027793bb9b65273/dataset-flatpack-verifier-2026-10-11/README.md). Save the verifier alongside your local experiment scripts, leaving the attached dataset intact. Its source SHA-256 at that commit is `560eed0c11f1f76a12f8f447d80bbb7130e796aa51fb8854160bd919aabd1b81`.

From the directory holding that helper:

```bash
python3 -I -B verify_flatpack.py /kaggle/input/kaggle-hidden-rerun-safety-kit \
  --manifest-sha256 0cb4bd97842c329dcdbabc89e29bb6226cbbb95058016bf8987021d33f90b88e
```

Use the actual attached root if your mount has an owner-prefixed layout. The digest is bound to **version 1**; a different release must use its own verified pin. `PASS_PINNED_BYTES` means the manifest and recorded files match the supplied pin, not that the code is numerically correct, compatible, independently evaluated or scored. `HOLD` exits 2 and names the inconsistency. The helper uses explicit checks, including under Python `-O`.

### Inspect paths without moving files

The folder-form examples in this kit require a layout that the flattened mount does not supply. The old folder-restoration loop moves files and also reaches the included example notebook. Keep the flat mount and all notebooks intact. The helper can instead inspect a complete explicit layout map and print the resulting plan without creating, renaming, moving, overwriting or executing a file.

Save the following JSON as `kaggle-hidden-rerun-safety-kit-layout-v1.json` beside the helper. This is a reviewed path plan, not a restoration command:

```json
{
  "manifest_sha256": "0cb4bd97842c329dcdbabc89e29bb6226cbbb95058016bf8987021d33f90b88e",
  "paths": {
    "CHECKLIST.md": "CHECKLIST.md",
    "LICENSE": "LICENSE",
    "NOTICE.md": "NOTICE.md",
    "README.md": "README.md",
    "code__launch_preflight.py": "code/launch_preflight.py",
    "code__launch_worker.py": "code/launch_worker.py",
    "code__mount_resolver.py": "code/mount_resolver.py",
    "code__owned_process.py": "code/owned_process.py",
    "code__raw_submissions.py": "code/raw_submissions.py",
    "code__replay_active_cells.py": "code/replay_active_cells.py",
    "code__root_gate.py": "code/root_gate.py",
    "code__submission_guard.py": "code/submission_guard.py",
    "code__tests__test_kit.py": "code/tests/test_kit.py",
    "data__checklist.csv": "data/checklist.csv",
    "data__ledger_excerpts.csv": "data/ledger_excerpts.csv",
    "data__platform_facts.csv": "data/platform_facts.csv",
    "docs__PLATFORM_FACTS.md": "docs/PLATFORM_FACTS.md",
    "docs__degraded_output_postmortem.md": "docs/degraded_output_postmortem.md",
    "examples__example_notebook.ipynb": "examples/example_notebook.ipynb"
  },
  "schema": "flatpack-layout-1"
}
```

Inspect that plan with the same release pin:

```bash
python3 -I -B verify_flatpack.py /kaggle/input/kaggle-hidden-rerun-safety-kit \
  --manifest-sha256 0cb4bd97842c329dcdbabc89e29bb6226cbbb95058016bf8987021d33f90b88e \
  --layout kaggle-hidden-rerun-safety-kit-layout-v1.json
```

`PASS_EXPLICIT_MAP` proves only that the complete plan has valid relative paths and no file/parent or case collisions. It does not materialize folders or prove the author's intended semantics. The map is bound to the manifest; its JSON byte digest is `da7b535576dd7b6fa3488b195c7d3f13109d459587928766753c553cc9213c02`.

The verifier works on a stable read-only mount and does not promise an atomic snapshot of a changing or hostile filesystem. Folder-form kit tests remain a separate execution step requiring a publisher-reviewed, non-destructive working layout. This addendum does not claim those tests ran here.

The helper's MIT terms cover the helper alone. All kit file terms, credits and notebook identities remain as recorded by this dataset. No Gemma competition package or agent ZIP is needed to use this general utility.
