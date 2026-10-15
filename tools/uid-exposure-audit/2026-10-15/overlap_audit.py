#!/usr/bin/env python3
"""Original, offline, aggregate-only UID exposure audit. MIT licensed."""

import csv
import io
import json
import os
import stat
import sys
from pathlib import Path

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_CSV_BYTES = 16 * 1024 * 1024
MAX_ROWS = 250000
MAX_UID_BYTES = 256
MAX_ARMS = 64
TOP_KEYS = {"schema_version", "uid_column", "validation_csv", "arms_complete",
            "arms", "ensemble_training", "ensemble_selection"}


def _result():
    return {"schema_version": 1, "uid_exposure_gate": "BLOCKED",
            "full_ensemble_oof_claim": "REFUSED",
            "independent_evaluation_claim": "NOT_ESTABLISHED",
            "claim_scope": "SUPPLIED_UID_MANIFESTS_ONLY",
            "issues": [], "validation": None, "arms": [], "ensemble": {}}


def _bounded_bytes(path, limit):
    # Opening a FIFO normally could hang before any byte limit can apply.
    # Check the opened descriptor so a path replacement cannot defeat the check.
    if not hasattr(os, "O_NONBLOCK"):
        raise ValueError("nonblocking regular-file open unavailable")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("regular file and size cap required")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            value = stream.read(limit + 1)
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(value) > limit:
        raise ValueError("size cap")
    return value


def _csv_uids(path, uid_column):
    summary = {"rows": 0, "unique_uids": 0, "missing_uid_rows": 0,
               "duplicate_uid_rows": 0, "invalid_uid_rows": 0,
               "malformed_rows": 0, "issues": []}
    uids = set()
    try:
        content = _bounded_bytes(path, MAX_CSV_BYTES).decode("utf-8-sig")
        reader = csv.reader(io.StringIO(content, newline=""), strict=True)
        if next(reader, None) != [uid_column]:
            summary["issues"].append("EXPECTED_SINGLE_UID_HEADER")
            return summary, uids
        for row in reader:
            summary["rows"] += 1
            if summary["rows"] > MAX_ROWS:
                summary["issues"].append("ROW_LIMIT_EXCEEDED")
                break
            if len(row) != 1:
                summary["malformed_rows"] += 1
                if not row:
                    summary["missing_uid_rows"] += 1
                continue
            uid = row[0]
            if not uid or not uid.strip():
                summary["missing_uid_rows"] += 1
            elif (len(uid.encode("utf-8")) > MAX_UID_BYTES or
                  any(character.isspace() or ord(character) < 32 or
                      ord(character) == 127 for character in uid)):
                summary["invalid_uid_rows"] += 1
            elif uid in uids:
                summary["duplicate_uid_rows"] += 1
            else:
                uids.add(uid)
    except (OSError, ValueError, UnicodeError, csv.Error):
        summary["issues"].append("CSV_UNREADABLE_OR_LIMIT_EXCEEDED")
    summary["unique_uids"] = len(uids)
    for count, code in [("missing_uid_rows", "MISSING_UID"),
                        ("duplicate_uid_rows", "DUPLICATE_UID"),
                        ("invalid_uid_rows", "INVALID_UID"),
                        ("malformed_rows", "MALFORMED_ROW")]:
        if summary[count]:
            summary["issues"].append(code)
    return summary, uids


def _pool(spec, base, uid_column, validation):
    result = {"declared_complete": False, "overlap_uids": None,
              "csv": None, "issues": []}
    if not isinstance(spec, dict) or set(spec) - {"complete", "csv"}:
        result["issues"].append("INVALID_POOL_SPEC")
        return result
    result["declared_complete"] = spec.get("complete") is True
    if not result["declared_complete"]:
        result["issues"].append("POOL_COMPLETENESS_UNKNOWN")
    filename = spec.get("csv")
    if not isinstance(filename, str) or not filename:
        result["issues"].append("POOL_MANIFEST_MISSING")
        return result
    summary, uids = _csv_uids(base / filename, uid_column)
    result["csv"] = summary
    if summary["issues"]:
        result["issues"].append("POOL_UID_INPUT_INVALID")
    else:
        result["overlap_uids"] = len(validation.intersection(uids))
        if result["overlap_uids"]:
            result["issues"].append("VALIDATION_EXPOSURE")
    return result


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def audit_manifest(manifest_path):
    """Read only specified local files; return fixed keys and aggregate counts."""
    report = _result()
    try:
        manifest_path = Path(manifest_path)
        config = json.loads(_bounded_bytes(manifest_path, MAX_MANIFEST_BYTES),
                            object_pairs_hook=_strict_object)
        if (not isinstance(config, dict) or set(config) - TOP_KEYS or
                type(config.get("schema_version")) is not int or
                config["schema_version"] != 1):
            raise ValueError("manifest schema")
        column = config.get("uid_column", "StudyInstanceUID")
        filename = config.get("validation_csv")
        if (not isinstance(column, str) or not column or len(column) > 128 or
                not isinstance(filename, str) or not filename):
            raise ValueError("validation schema")
        base = manifest_path.parent
        validation_summary, validation = _csv_uids(base / filename, column)
        report["validation"] = validation_summary
        if validation_summary["issues"] or not validation:
            report["issues"].append("VALIDATION_UID_INPUT_INVALID_OR_EMPTY")
        if config.get("arms_complete") is not True:
            report["issues"].append("CONTRIBUTING_ARM_COVERAGE_UNKNOWN")
        arms = config.get("arms")
        if not isinstance(arms, list) or not 1 <= len(arms) <= MAX_ARMS:
            raise ValueError("arm count")
        for index, arm in enumerate(arms, 1):
            if not isinstance(arm, dict) or set(arm) != {"training", "selection"}:
                raise ValueError("arm schema")
            entry = {"arm_index": index}
            for stage in ("training", "selection"):
                entry[stage] = _pool(arm[stage], base, column, validation)
                if entry[stage]["issues"]:
                    report["issues"].append("ARM_" + stage.upper() + "_UNQUALIFIED")
            report["arms"].append(entry)
        for stage in ("training", "selection"):
            pool = _pool(config.get("ensemble_" + stage), base, column, validation)
            report["ensemble"][stage] = pool
            if pool["issues"]:
                report["issues"].append("ENSEMBLE_" + stage.upper() + "_UNQUALIFIED")
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        report["issues"].append("MANIFEST_INVALID_OR_UNREADABLE")
    report["issues"] = sorted(set(report["issues"]))
    if not report["issues"]:
        report["uid_exposure_gate"] = "PASS_DECLARED_MANIFESTS"
        report["full_ensemble_oof_claim"] = "NOT_ESTABLISHED_BY_UID_AUDIT"
    return report


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments == ["--help"]:
        print("Usage: python overlap_audit.py LOCAL_MANIFEST.json")
        print("Writes aggregate JSON to stdout. Exit 0: UID gate passes; 2: refused.")
        return 0
    if len(arguments) != 1:
        report = _result()
        report["issues"].append("EXPECTED_ONE_LOCAL_MANIFEST_ARGUMENT")
    else:
        report = audit_manifest(arguments[0])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["uid_exposure_gate"] == "PASS_DECLARED_MANIFESTS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
