"""Exercise the OOF contract with entirely synthetic studies and artifacts."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from oof_contract import Artifact, Bank, Fold, format_diagnostics, sha256_file, validate_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mismatch", action="store_true", help="Introduce one synthetic attached-label mismatch")
    args = parser.parse_args()
    # Each invocation has a fresh, retained directory. No user inputs are edited.
    root = Path(tempfile.mkdtemp(prefix="synthetic_knee_oof_contract_"))
    official = {"synthetic-study-a": {"ACL": 0, "MCL": 1}, "synthetic-study-b": {"ACL": 1, "MCL": 0}}
    probabilities = np.array([[.2, .7], [.9, .1]])
    files = {
        "synthetic-input.json": json.dumps(official, sort_keys=True) + "\n",
        "synthetic-state.json": json.dumps({"constant_probabilities": probabilities.tolist(), "trained_model": False}) + "\n",
        "synthetic-exposure.txt": "Fresh synthetic fixture; no real validation cohort or training.\n",
    }
    for name, content in files.items():
        with (root / name).open("x", encoding="utf-8") as f:
            f.write(content)
    source = Path(__file__)
    artifacts = [
        Artifact("generator_source", source, "source", sha256_file(source)),
        Artifact("synthetic_state", root / "synthetic-state.json", "state", sha256_file(root / "synthetic-state.json")),
        Artifact("synthetic_input", root / "synthetic-input.json", "input", sha256_file(root / "synthetic-input.json")),
        Artifact("synthetic_exposure", root / "synthetic-exposure.txt", "exposure", sha256_file(root / "synthetic-exposure.txt")),
    ]
    attached_labels = np.array([[0, 1], [1, 0]])
    if args.mismatch:
        attached_labels[0, 0] = 1
    report = validate_contract(
        official=official,
        label_names=("ACL", "MCL"),
        expected_held_ids=("synthetic-study-a", "synthetic-study-b"),
        banks=[Bank("synthetic_bank", ("synthetic-study-a", "synthetic-study-b"), ("ACL", "MCL"),
                    probabilities, attached_labels, ("0", "1"))],
        folds=[Fold("0", ("synthetic-study-b",), ("synthetic-study-a",)),
               Fold("1", ("synthetic-study-a",), ("synthetic-study-b",))],
        artifacts=artifacts,
        components=[{"name": "synthetic_generator", "source_artifacts": ["generator_source"],
                     "state_artifacts": ["synthetic_state"], "input_artifacts": ["synthetic_input"]}],
        bank_components={"synthetic_bank": ["synthetic_generator"]},
        group_by_id={"synthetic-study-a": "synthetic-group-a", "synthetic-study-b": "synthetic-group-b"},
        group_kind="synthetic", runtime_input_closure_attested=True,
        exposure={"status": "attested_unexposed", "artifact_names": ["synthetic_exposure"], "scope": "synthetic fixture only"},
        current_ensemble_equivalent=False,
    )
    with (root / "evidence_manifest.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(format_diagnostics(report), end="")
    print(f"Synthetic evidence retained at: {root}")
    return 2 if report["status"] == "HOLD" else 0


if __name__ == "__main__":
    raise SystemExit(main())
