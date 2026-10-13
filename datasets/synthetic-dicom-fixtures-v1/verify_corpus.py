#!/usr/bin/env python3
"""Read-only oracle and manifest checks, without importing the generator. MIT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from fixture_reader import FixtureRefusal, read_fixture

if not __debug__:
    raise RuntimeError("Verification requires assertions: run Python without -O")


def analytic_truth(row):
    """Independent rational truth from the documented phantom family."""
    shape = (int(row["rows"]), int(row["columns"]))
    name = row["oracle"]
    if name in ("ramp_up", "ramp_down"):
        n = shape[0] * shape[1]
        # Explicit scalar arithmetic, rather than the reader's extrema algorithm.
        values = [j / (n - 1) if name == "ramp_up" else (n - 1 - j) / (n - 1)
                  for j in range(n)]
    elif name in ("constant_zero", "constant_one"):
        values = [int(name == "constant_one")] * (shape[0] * shape[1])
    elif name.startswith("rescale_window_"):
        rational = {"rescale_window_half": (1, 1, 8),
                    "rescale_window_double": (4, -13, 8),
                    "rescale_window_negative": (-2, 13, 8),
                    "rescale_window_negative_quarter": (-1, 20, 16)}
        step, offset, divisor = rational[name]
        values = [min(1, max(0, (step * j + offset) / divisor))
                  for j in range(shape[0] * shape[1])]
    elif name.startswith("window5"):
        values = [0, 1/8, 3/8, 5/8, 7/8, 1, 1]
    elif name.startswith("window1"):
        values = [0, 0, 1, 1, 1]
    elif name == "signed_endpoints":
        values = [v / 65535 for v in [0, 16384, 32767, 32768, 32769, 49152, 65535]]
    elif name == "unsigned_endpoints":
        values = [v / 65535 for v in [0, 1, 32767, 32768, 40000, 65534, 65535]]
    else:
        raise AssertionError("Unknown analytic family: " + name)
    if name.endswith("_inverse"):
        values = [1 - v for v in values]
    return np.array(values, dtype=np.float32).reshape(shape)


def safe_relative(root, value):
    path = Path(value)
    assert not path.is_absolute() and ".." not in path.parts, "Nonrelative manifest path"
    return root / path


def verify(root):
    root = Path(root)
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 48 and len({row["case_id"] for row in rows}) == 48
    assert len(list((root / "dicom").glob("*.dcm"))) == 48
    assert len(list((root / "golden").glob("*.npy"))) == 43
    accepted, refused = 0, 0
    outputs = {}
    for row in rows:
        path = safe_relative(root, row["dicom_path"])
        blob = path.read_bytes()
        assert len(blob) == int(row["dicom_bytes"]), row["case_id"]
        assert hashlib.sha256(blob).hexdigest() == row["dicom_sha256"], row["case_id"]
        if row["expected_error"]:
            assert row["expected_contract"] == "refuse" and not row["golden_path"]
            try:
                read_fixture(path)
            except FixtureRefusal as exc:
                assert str(exc) == row["expected_error"], (row["case_id"], str(exc))
            else:
                raise AssertionError("Expected refusal: " + row["case_id"])
            refused += 1
            continue
        golden_path = safe_relative(root, row["golden_path"])
        assert hashlib.sha256(golden_path.read_bytes()).hexdigest() == row["golden_sha256"]
        golden = np.load(golden_path, allow_pickle=False)
        truth = analytic_truth(row)
        assert golden.dtype == np.dtype("float32") and golden.shape == truth.shape
        np.testing.assert_array_equal(golden, truth, err_msg=row["case_id"] + " golden")
        actual, metadata = read_fixture(path)
        np.testing.assert_allclose(actual, truth, rtol=0, atol=1e-7, err_msg=row["case_id"])
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        assert float(actual.min()) >= 0 and float(actual.max()) <= 1
        np.testing.assert_allclose(metadata["orientation"],
            [float(x) for x in row["orientation"].split(";")], rtol=0, atol=1e-12)
        outputs[row["case_id"]] = actual
        accepted += 1
    # Pair relations catch accidental sign handling and photometric inversion.
    for r, c in [(1, 7), (3, 11), (8, 5), (16, 16)]:
        prefix = f"ramp_{r}x{c}_"
        np.testing.assert_array_equal(outputs[prefix + "i16_m2"], outputs[prefix + "u16_m2"])
        np.testing.assert_allclose(outputs[prefix + "i16_m1"] + outputs[prefix + "i16_m2"],
                                   1, rtol=0, atol=1e-7)
    assert (accepted, refused) == (43, 5)
    print(json.dumps(dict(status="PASS", accepted=accepted, expected_refusals=refused,
                         dicom_count=len(rows), analytic_goldens=accepted), sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="Directory containing manifest.csv")
    verify(parser.parse_args().corpus)
