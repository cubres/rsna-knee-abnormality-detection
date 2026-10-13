#!/usr/bin/env python3
"""Inspect one accepted fixture and one intentional refusal, read-only. MIT."""
import argparse
from pathlib import Path
from fixture_reader import FixtureRefusal, read_fixture

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("corpus")
root = Path(parser.parse_args().corpus)
image, geometry = read_fixture(root / "dicom/signed_endpoints.dcm")
print("Signed endpoint intensities:", image.tolist())
print("Geometry (no reorientation):", geometry)
try:
    read_fixture(root / "dicom/multiframe_two.dcm")
except FixtureRefusal as exc:
    print("Expected single-frame reader refusal:", str(exc))
