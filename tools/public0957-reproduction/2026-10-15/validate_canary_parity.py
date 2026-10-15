#!/usr/bin/env python3
"""Compare a three-study Knee canary with an exact public-version CSV.

Usage: python3 validate_canary_parity.py CANARY.csv REFERENCE.csv
       python3 validate_canary_parity.py --self-test

Standard library only. Reads two local files; never runs a model, contacts a
service, writes outputs, or prints study identifiers or individual predictions.
Exit status: 0 = parity passed; 1 = failed; 2 = invalid invocation/read error.
Passing public-example parity does not establish hidden-test score or runtime.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path


HEADER = [
    "StudyInstanceUID", "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
EXPECTED_ROWS = 3
EXPECTED_COLUMNS = 13
TOLERANCE = Decimal("0.0000001")


def _decode(raw, label, errors):
    """Return parsed records and decimal values; diagnostics contain no values."""
    try:
        records = list(csv.reader(io.StringIO(raw.decode("utf-8-sig")), strict=True))
    except (UnicodeDecodeError, csv.Error):
        errors.append(label + "_invalid_csv")
        return None
    if not records:
        errors.append(label + "_empty_csv")
        return None
    header, rows = records[0], records[1:]
    if header != HEADER:
        errors.append(label + "_wrong_header")
    if len(rows) != EXPECTED_ROWS:
        errors.append(label + "_wrong_row_count")
    if any(len(row) != EXPECTED_COLUMNS for row in rows):
        errors.append(label + "_wrong_column_count")
        return {"header": header, "rows": rows, "ids": None, "values": None}
    ids = [row[0] for row in rows]
    if any(not uid.strip() for uid in ids) or len(set(ids)) != len(ids):
        errors.append(label + "_invalid_identifiers")
    values = []
    try:
        for row in rows:
            for item in row[1:]:
                value = Decimal(item)
                # Decimal preserves the exact inclusive 1e-7 tolerance boundary.
                if not value.is_finite() or not math.isfinite(float(value)):
                    errors.append(label + "_nonfinite_prediction")
                    return {"header": header, "rows": rows, "ids": ids, "values": None}
                if not Decimal(0) <= value <= Decimal(1):
                    errors.append(label + "_prediction_out_of_range")
                    return {"header": header, "rows": rows, "ids": ids, "values": None}
                values.append(value)
    except (InvalidOperation, ValueError, OverflowError):
        errors.append(label + "_nonnumeric_prediction")
        values = None
    return {"header": header, "rows": rows, "ids": ids, "values": values}


def validate_bytes(canary_bytes, reference_bytes):
    """Validate inputs and return a JSON-safe aggregate-only report."""
    errors = []
    canary = _decode(canary_bytes, "canary", errors)
    reference = _decode(reference_bytes, "reference", errors)
    report = {
        "status": "FAIL",
        "byte_equal": canary_bytes == reference_bytes,
        "canary_sha256": hashlib.sha256(canary_bytes).hexdigest(),
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "canary_rows": None if canary is None else len(canary["rows"]),
        "reference_rows": None if reference is None else len(reference["rows"]),
        "canary_columns": None if canary is None else len(canary["header"]),
        "reference_columns": None if reference is None else len(reference["header"]),
        "prediction_count": None,
        "max_abs_diff": None,
        "tolerance": float(TOLERANCE),
        "errors": errors,
    }
    if canary is not None and reference is not None:
        if canary["header"] != reference["header"]:
            errors.append("headers_differ")
        if canary["ids"] is not None and reference["ids"] is not None:
            if canary["ids"] != reference["ids"]:
                errors.append("ordered_identifiers_differ")
    if not errors:
        # Schema validation above guarantees exactly 3 * 12 predictions per CSV.
        # Keep even unusually precise CSV decimals exact at the tolerance boundary.
        values = canary["values"] + reference["values"]
        precision = max(28, max(len(v.as_tuple().digits) + max(0, -v.as_tuple().exponent)
                                for v in values) + 2)
        with localcontext() as context:
            context.prec = precision
            differences = [abs(a - b) for a, b in zip(canary["values"], reference["values"])]
        maximum = max(differences)
        report["prediction_count"] = len(differences)
        report["max_abs_diff"] = float(maximum)
        if maximum > TOLERANCE:
            errors.append("prediction_difference_exceeds_tolerance")
    report["status"] = "FAIL" if errors else "PASS"
    return report


def _fixture(rows=None, header=None):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(HEADER if header is None else header)
    writer.writerows(rows if rows is not None else [
        ["invented-example-" + str(i)] + ["0.25"] * 12 for i in range(3)
    ])
    return stream.getvalue().encode("utf-8")


def self_test():
    """Only invented in-memory fixtures. No real identifiers or files printed."""
    base = _fixture()
    cases = []

    def check(name, candidate, expected, error=None, reference=base):
        result = validate_bytes(candidate, reference)
        passed = result["status"] == expected
        if error is not None:
            passed = passed and error in result["errors"]
        cases.append({"case": name, "passed": passed})

    def changed(value):
        rows = list(csv.reader(io.StringIO(base.decode())))[1:]
        rows[1][5] = value
        return _fixture(rows)

    check("identical", base, "PASS")
    check("within_numeric_tolerance", changed("0.25000005"), "PASS")
    check("inclusive_tolerance_boundary", changed("0.2500001"), "PASS")
    check("different_serialization_same_numbers", base.replace(b"0.25", b"0.250"), "PASS")
    rows = list(csv.reader(io.StringIO(base.decode())))[1:]
    check("shuffled_identifiers", _fixture(rows[::-1]), "FAIL", "ordered_identifiers_differ")
    check("wrong_header", _fixture(header=["wrong"] + HEADER[1:]), "FAIL", "canary_wrong_header")
    check("wrong_reference_header", base, "FAIL", "reference_wrong_header",
          reference=_fixture(header=["wrong"] + HEADER[1:]))
    check("nan_prediction", changed("NaN"), "FAIL", "canary_nonfinite_prediction")
    check("infinite_prediction", changed("Infinity"), "FAIL", "canary_nonfinite_prediction")
    check("difference_above_tolerance", changed("0.2500001001"), "FAIL",
          "prediction_difference_exceeds_tolerance")
    check("high_precision_difference_above_tolerance", changed("0.25000010000000000000000000000001"),
          "FAIL", "prediction_difference_exceeds_tolerance")
    check("prediction_out_of_range", changed("1.01"), "FAIL", "canary_prediction_out_of_range")
    check("wrong_row_count", _fixture(rows[:2]), "FAIL", "canary_wrong_row_count")
    check("wrong_column_count", _fixture([row[:-1] for row in rows]), "FAIL", "canary_wrong_column_count")
    duplicate = [row[:] for row in rows]
    duplicate[2][0] = duplicate[1][0]
    check("duplicate_identifier", _fixture(duplicate), "FAIL", "canary_invalid_identifiers")
    check("nonnumeric_prediction", changed("invalid"), "FAIL", "canary_nonnumeric_prediction")
    check("empty_file", b"", "FAIL", "canary_empty_csv")
    check("invalid_utf8", b"\xff", "FAIL", "canary_invalid_csv")
    check("invalid_reference_nonfinite", base, "FAIL", "reference_nonfinite_prediction",
          reference=changed("NaN"))
    result = {
        "status": "PASS" if all(case["passed"] for case in cases) else "FAIL",
        "tests": len(cases), "passed": sum(case["passed"] for case in cases),
        "cases": cases,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canary", nargs="?")
    parser.add_argument("reference", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        if args.canary or args.reference:
            parser.error("--self-test accepts no file arguments")
        result = self_test()
    else:
        if not args.canary or not args.reference:
            parser.error("supply both canary and reference files")
        try:
            canary_bytes = Path(args.canary).read_bytes()
            reference_bytes = Path(args.reference).read_bytes()
        except OSError:
            print(json.dumps({"status": "FAIL", "errors": ["input_read_error"]}))
            return 2
        result = validate_bytes(canary_bytes, reference_bytes)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
