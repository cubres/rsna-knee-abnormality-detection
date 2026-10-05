"""Named-label OOF data and declared-provenance checks; no model fitting.

Inputs remain untouched. A passing result concerns the declared contract only:
it does not establish model quality, patient independence, or an official score.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Bank:
    name: str
    ids: tuple[str, ...]
    label_names: tuple[str, ...]
    probabilities: np.ndarray
    supplied_labels: np.ndarray | None = None
    fold_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Fold:
    name: str
    train_ids: tuple[str, ...]
    held_ids: tuple[str, ...]


@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
    kind: str
    expected_sha256: str | None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest_ids(ids: Sequence[str]) -> str:
    # Length framing avoids ambiguous concatenations and preserves multiplicity.
    h = hashlib.sha256()
    for uid in sorted(ids):
        value = uid.encode("utf-8")
        h.update(len(value).to_bytes(8, "big"))
        h.update(value)
    return h.hexdigest()


def validate_contract(
    official: Mapping[str, Mapping[str, float | None]],
    label_names: Sequence[str],
    expected_held_ids: Sequence[str],
    banks: Sequence[Bank],
    folds: Sequence[Fold],
    artifacts: Sequence[Artifact],
    components: Sequence[Mapping],
    *,
    bank_components: Mapping[str, Sequence[str]] | None = None,
    group_by_id: Mapping[str, str] | None = None,
    group_kind: str = "unspecified",
    runtime_input_closure_attested: bool = False,
    exposure: Mapping | None = None,
    current_ensemble_equivalent: bool = False,
) -> dict:
    """Check joins by names/IDs, folds, file pins and exposure independently.

    ``components`` explicitly declares required source/state artifact names.
    A caller's closure attestation is reported as an attestation, not inferred
    from a checkpoint hash. Exposure must be known; absence is not untouchedness.
    """
    issues: list[dict] = []
    rows: list[dict] = []
    evidence: list[dict] = []

    def issue(code: str, message: str, *, severity: str = "error", **detail) -> None:
        issues.append({"code": code, "severity": severity, "message": message, **detail})

    labels = tuple(label_names)
    expected = tuple(str(x) for x in expected_held_ids)
    if not labels or len(labels) != len(set(labels)):
        issue("LABEL_SCHEMA", "Official target names must be nonempty and unique.")
    if len(expected) != len(set(expected)):
        issue("DUPLICATE_EXPECTED_ID", "Expected held cohort contains duplicate study IDs.")
    if not expected:
        issue("EMPTY_COHORT", "Expected held cohort is empty.")
    for uid in expected:
        if uid not in official:
            issue("UNKNOWN_EXPECTED_ID", "Expected study is absent from official labels.", id_sha256=_digest_ids([uid]))
    assignment: dict[str, str] = {}
    fold_evidence: list[dict] = []
    if not folds:
        issue("MISSING_FOLDS", "No train/held split declarations were supplied.")
    if len({f.name for f in folds}) != len(folds):
        issue("DUPLICATE_FOLD_NAME", "Fold names must be unique.")
    for fold in folds:
        train, held = set(fold.train_ids), set(fold.held_ids)
        if len(train) != len(fold.train_ids) or len(held) != len(fold.held_ids):
            issue("DUPLICATE_SPLIT_ID", "A split contains duplicate study IDs.", fold=fold.name)
        overlap = train & held
        if overlap:
            issue("TRAIN_HELD_OVERLAP", "Training and held studies overlap.", fold=fold.name, count=len(overlap))
        if not train or not held:
            issue("EMPTY_SPLIT", "Both training and held splits must be nonempty.", fold=fold.name)
        for uid in train | held:
            if uid not in official:
                issue("UNKNOWN_SPLIT_ID", "Split study is absent from official data.", fold=fold.name, id_sha256=_digest_ids([uid]))
        for uid in held:
            if uid in assignment:
                issue("MULTIPLE_HELD_FOLDS", "A held study belongs to more than one fold.", id_sha256=_digest_ids([uid]))
            assignment[uid] = fold.name
        group_overlap = None
        if group_by_id is not None:
            missing = [uid for uid in train | held if uid not in group_by_id or not group_by_id[uid]]
            if missing:
                issue("MISSING_GROUP", "Group identity is incomplete for the declared split.", fold=fold.name, count=len(missing))
            train_groups = {group_by_id[uid] for uid in train if uid in group_by_id}
            held_groups = {group_by_id[uid] for uid in held if uid in group_by_id}
            group_overlap = len(train_groups & held_groups)
            if group_overlap:
                issue("GROUP_OVERLAP", "A declared group occurs in training and held splits.", fold=fold.name, group_kind=group_kind, count=group_overlap)
        fold_evidence.append({"fold": fold.name, "train_count": len(train), "held_count": len(held),
                              "train_ids_sha256": _digest_ids(fold.train_ids), "held_ids_sha256": _digest_ids(fold.held_ids),
                              "study_overlap_count": len(overlap), "group_overlap_count": group_overlap})
    expected_set = set(expected)
    if set(assignment) != expected_set:
        issue("HELD_COVERAGE", "Fold-held union differs from the declared cohort.", missing=len(expected_set-set(assignment)), extra=len(set(assignment)-expected_set))
    if not banks:
        issue("MISSING_BANK", "No predictions were supplied.")
    if len({b.name for b in banks}) != len(banks):
        issue("DUPLICATE_BANK_NAME", "Prediction bank names must be unique.")
    for bank in banks:
        counts = Counter(bank.ids)
        duplicates = sum(n-1 for n in counts.values() if n > 1)
        if duplicates:
            issue("DUPLICATE_PREDICTION_ID", "Prediction rows contain duplicate studies.", bank=bank.name, count=duplicates)
        observed_ids = set(bank.ids)
        if observed_ids != expected_set:
            issue("PREDICTION_COVERAGE", "Prediction IDs differ from the declared cohort.", bank=bank.name, missing=len(expected_set-observed_ids), extra=len(observed_ids-expected_set))
        if len(bank.label_names) != len(set(bank.label_names)) or set(bank.label_names) != set(labels):
            issue("BANK_LABEL_SCHEMA", "Prediction targets do not match the named official targets.", bank=bank.name)
            continue
        p = np.asarray(bank.probabilities)
        shape = (len(bank.ids), len(bank.label_names))
        if p.shape != shape or p.dtype.kind not in "iuf":
            issue("PROBABILITY_SHAPE_TYPE", "Probability matrix has the wrong shape or nonnumeric dtype.", bank=bank.name, shape=list(p.shape))
            continue
        if not np.isfinite(p).all():
            issue("NONFINITE_PROBABILITY", "Probabilities contain NaN or infinity.", bank=bank.name, count=int((~np.isfinite(p)).sum()))
        out = np.isfinite(p) & ((p < 0) | (p > 1))
        if out.any():
            issue("PROBABILITY_RANGE", "Probabilities lie outside [0, 1].", bank=bank.name, count=int(out.sum()))
        if bank.fold_ids is not None:
            if len(bank.fold_ids) != len(bank.ids):
                issue("FOLD_VECTOR_SHAPE", "Prediction fold vector has the wrong length.", bank=bank.name)
            else:
                bad = sum(str(f) != assignment.get(uid) for uid, f in zip(bank.ids, bank.fold_ids))
                if bad:
                    issue("PREDICTION_FOLD_MISMATCH", "A prediction is assigned to a different declared held fold.", bank=bank.name, count=bad)
        else:
            issue("MISSING_PREDICTION_FOLD", "Prediction rows lack their generating held-fold identity.", bank=bank.name)
        supplied = None if bank.supplied_labels is None else np.asarray(bank.supplied_labels)
        if supplied is None:
            issue("MISSING_SUPPLIED_LABELS", "No labels are attached to this bank for consistency comparison.", severity="warning", bank=bank.name)
        elif supplied.shape != shape or supplied.dtype.kind not in "iufb":
            issue("SUPPLIED_LABEL_SHAPE_TYPE", "Attached label matrix has the wrong shape or dtype.", bank=bank.name)
            supplied = None
        per_label = {name: 0 for name in labels}
        examples: list[dict] = []
        official_missing = 0
        invalid_supplied = 0
        compared_valid_cells = 0
        for row_index, uid in enumerate(bank.ids):
            if uid not in official:
                issue("UNKNOWN_PREDICTION_ID", "Prediction study is absent from official data.", bank=bank.name, id_sha256=_digest_ids([uid]))
                continue
            for col, label in enumerate(bank.label_names):
                gold = official[uid].get(label)
                if gold is None or not math.isfinite(float(gold)) or float(gold) not in (0.0, 1.0):
                    official_missing += 1
                    continue
                if supplied is not None:
                    value = float(supplied[row_index, col])
                    if not math.isfinite(value) or value not in (0.0, 1.0):
                        invalid_supplied += 1
                    else:
                        compared_valid_cells += 1
                        if value != float(gold):
                            per_label[label] += 1
                            if len(examples) < 8:
                                examples.append({"id_sha256": _digest_ids([uid]), "label": label, "supplied": value, "official": float(gold)})
        if official_missing:
            issue("MISSING_OFFICIAL_TARGET", "Official labels are missing or nonbinary for evaluated cells.", bank=bank.name, count=official_missing)
        if invalid_supplied:
            issue("NONBINARY_SUPPLIED_LABEL", "Attached labels contain nonbinary or nonfinite cells.", bank=bank.name, count=invalid_supplied)
        mismatches = sum(per_label.values())
        if mismatches:
            issue("LABEL_MISMATCH", "Attached labels disagree with the official ID/name join; inputs are retained unchanged.", bank=bank.name, count=mismatches)
        rows.append({"bank": bank.name, "rows": len(bank.ids), "targets": len(bank.label_names), "ids_sha256": _digest_ids(bank.ids),
                     "label_mismatch_count": mismatches, "label_mismatches_by_target": per_label, "mismatch_examples": examples,
                     "compared_valid_label_cells": compared_valid_cells,
                     "label_comparison_complete": compared_valid_cells == len(bank.ids) * len(bank.label_names) and bool(bank.ids),
                     "source_labels_used_as_truth": False})
    artifact_names = [a.name for a in artifacts]
    if len(artifact_names) != len(set(artifact_names)):
        issue("DUPLICATE_ARTIFACT_NAME", "Artifact names must be unique.")
    verified_names: set[str] = set()
    for artifact in artifacts:
        entry = {"name": artifact.name, "path": str(artifact.path.resolve()), "kind": artifact.kind, "expected_sha256": artifact.expected_sha256}
        if not artifact.path.is_file():
            issue("MISSING_ARTIFACT", "A declared source/state/input artifact is unavailable.", artifact=artifact.name)
            entry["status"] = "MISSING"
        else:
            observed = sha256_file(artifact.path)
            entry.update({"observed_sha256": observed, "bytes": artifact.path.stat().st_size})
            if artifact.expected_sha256 is None:
                issue("UNPINNED_ARTIFACT", "No independently retained expected hash was supplied.", artifact=artifact.name)
                entry["status"] = "OBSERVED_UNPINNED"
            elif len(artifact.expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in artifact.expected_sha256):
                issue("INVALID_HASH_PIN", "Expected SHA-256 must be lowercase hexadecimal.", artifact=artifact.name)
                entry["status"] = "INVALID_PIN"
            elif observed != artifact.expected_sha256:
                issue("ARTIFACT_HASH_MISMATCH", "Artifact bytes differ from the retained expected hash.", artifact=artifact.name)
                entry["status"] = "HASH_MISMATCH"
            else:
                verified_names.add(artifact.name)
                entry["status"] = "PIN_MATCH"
        evidence.append(entry)
    if not components:
        issue("MISSING_COMPONENT_PROVENANCE", "No model component provenance is declared.")
    seen_components: set[str] = set()
    for component in components:
        name = str(component.get("name", ""))
        if not name or name in seen_components:
            issue("COMPONENT_NAME", "Model component names must be nonempty and unique.")
        seen_components.add(name)
        for kind in ("source", "state", "input"):
            required = component.get(kind + "_artifacts", [])
            if not required or not isinstance(required, list) or any(not isinstance(v, str) for v in required):
                issue("INCOMPLETE_COMPONENT_PROVENANCE", "Component lacks an explicit artifact closure.", component=name, kind=kind)
                continue
            by_name = {a.name: a for a in artifacts}
            missing = [v for v in required if v not in verified_names or by_name[v].kind != kind]
            if missing:
                issue("INCOMPLETE_COMPONENT_PROVENANCE", "Required component artifacts are missing, unpinned, mismatched or of the wrong kind.", component=name, kind=kind, artifacts=missing)
    for bank in banks:
        origins = (bank_components or {}).get(bank.name, ())
        if not origins or isinstance(origins, str) or len(origins) != len(set(origins)):
            issue("BANK_COMPONENT_BINDING", "Every prediction bank needs an explicit, unique generating-component declaration.", bank=bank.name)
        elif any(name not in seen_components for name in origins):
            issue("BANK_COMPONENT_BINDING", "Prediction bank refers to an undeclared generating component.", bank=bank.name)
    if not runtime_input_closure_attested:
        issue("RUNTIME_CLOSURE_UNATTESTED", "Input/source/state hashes do not by themselves attest the generating runtime's complete dependency closure.")
    exposure_state = (exposure or {}).get("status", "unknown")
    if exposure_state not in ("exposed", "attested_unexposed"):
        issue("VALIDATION_EXPOSURE_UNKNOWN", "Absence of an exposure record cannot establish an untouched holdout.")
    elif exposure_state == "exposed":
        issue("VALIDATION_EXPOSED", "This cohort was already used for selection or tuning; it is unsuitable as a fresh holdout.")
    exposure_artifacts = (exposure or {}).get("artifact_names", [])
    by_name = {a.name: a for a in artifacts}
    if exposure_state != "unknown" and (not exposure_artifacts or any(x not in verified_names or by_name[x].kind != "exposure" for x in exposure_artifacts)):
        issue("EXPOSURE_EVIDENCE_INCOMPLETE", "Exposure declaration lacks matching evidence artifacts.")
    if not current_ensemble_equivalent:
        issue("CURRENT_ENSEMBLE_NOT_ATTESTED", "Historical component predictions do not establish current ensemble OOF equivalence.", severity="warning")
    errors = sum(i["severity"] == "error" for i in issues)
    return {"schema_version": 1, "status": "HOLD" if errors else "PASS_DATA_AND_DECLARED_PROVENANCE_ONLY",
            "error_count": errors, "warning_count": sum(i["severity"] == "warning" for i in issues), "issues": issues,
            "cohort": {"count": len(expected), "ids_sha256": _digest_ids(expected), "label_names": list(labels)},
            "banks": rows, "folds": fold_evidence, "artifacts": evidence, "components": list(components),
            "bank_components": {k: list(v) for k, v in (bank_components or {}).items()},
            "group_kind": group_kind, "patient_independence_proven": False,
            "runtime_input_closure_attested": runtime_input_closure_attested, "validation_exposure": dict(exposure or {"status": "unknown"}),
            "current_ensemble_equivalent": current_ensemble_equivalent, "quality_promotion_authorized": False, "official_score": None}


def format_diagnostics(report: Mapping) -> str:
    lines = [f"OOF contract: {report['status']} ({report['error_count']} errors, {report['warning_count']} warnings)"]
    for bank in report["banks"]:
        suffix = "" if bank["label_comparison_complete"] else "; label comparison incomplete"
        lines.append(f"  {bank['bank']}: {bank['rows']} studies x {bank['targets']} targets; {bank['label_mismatch_count']} observed label mismatches{suffix}")
    for item in report["issues"]:
        context = ", ".join(f"{k}={v}" for k, v in item.items() if k not in ("code", "severity", "message"))
        lines.append(f"  {item['severity'].upper()} {item['code']}: {item['message']}" + (f" [{context}]" if context else ""))
    lines.append("No fitting, input corrections, current-ensemble equivalence, or official score is implied.")
    return "\n".join(lines) + "\n"
