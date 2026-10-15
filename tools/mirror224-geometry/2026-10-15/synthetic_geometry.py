#!/usr/bin/env python3
"""Original MIT-licensed invented geometry fixtures; no MRI/model dependency."""
import argparse
import copy
import json

SLOT_SIZES = (26, 22, 18, 12, 18)
# Production centres are clipped to [1, 94]. Cover every slot boundary.
CASES = ((1, 0), (25, 0), (26, 1), (47, 1), (48, 2),
         (65, 2), (66, 3), (77, 3), (78, 4), (94, 4))


def reflect_triplet(triplet, centre, depth=96):
    """Reflect a uint8-like [3,H,W] toy triplet using its centre's slot."""
    if type(depth) is not int or depth != sum(SLOT_SIZES):
        raise ValueError("DEPTH_MUST_BE_96")
    if type(centre) is not int or not 0 <= centre < depth:
        raise ValueError("CENTRE_OUT_OF_RANGE")
    if not isinstance(triplet, list) or len(triplet) != 3:
        raise ValueError("TRIPLET_SHAPE_REQUIRED")
    height = len(triplet[0]) if isinstance(triplet[0], list) else 0
    width = len(triplet[0][0]) if height and isinstance(triplet[0][0], list) else 0
    if not height or not width:
        raise ValueError("TRIPLET_SHAPE_REQUIRED")
    for plane in triplet:
        if not isinstance(plane, list) or len(plane) != height:
            raise ValueError("TRIPLET_SHAPE_REQUIRED")
        for row in plane:
            if not isinstance(row, list) or len(row) != width:
                raise ValueError("TRIPLET_SHAPE_REQUIRED")
            if any(type(v) is not int or not 0 <= v <= 255 for v in row):
                raise ValueError("UINT8_REQUIRED")
    boundary = 0
    for slot, size in enumerate(SLOT_SIZES):
        boundary += size
        if centre < boundary:
            break
    if slot < 2:
        return copy.deepcopy(triplet[::-1])
    return [[row[::-1] for row in plane] for plane in triplet]


def toy_triplet():
    return [[[8 * channel + 4 * y + x for x in range(4)]
             for y in range(2)] for channel in range(3)]


def fixture_document():
    fixtures = []
    for centre, slot in CASES:
        original = toy_triplet()
        # Independent explicit coordinate oracle; no reflect_triplet call.
        expected = [[[8 * (2 - c if slot < 2 else c) + 4 * y
                      + (x if slot < 2 else 3 - x) for x in range(4)]
                     for y in range(2)] for c in range(3)]
        fixtures.append({"case_id": "invented-centre-%02d" % centre,
                         "centre": centre, "slot": slot,
                         "axis": "slice_channel" if slot < 2 else "width",
                         "input_uint8": original, "expected_uint8": expected})
    return {"schema_version": 1, "license": "MIT", "synthetic_only": True,
            "volume_depth": 96, "slot_sizes": list(SLOT_SIZES),
            "normalization_stage": "after_reflection",
            "claims": "Geometry examples only; no MRI, model parity or accuracy claim.",
            "fixtures": fixtures}


def self_test():
    for case in fixture_document()["fixtures"]:
        before = copy.deepcopy(case["input_uint8"])
        actual = reflect_triplet(before, case["centre"])
        assert actual == case["expected_uint8"], case["case_id"]
        assert reflect_triplet(actual, case["centre"]) == before
        assert before == case["input_uint8"], "input mutated"
    failures = ((toy_triplet(), -1, 96, "CENTRE_OUT_OF_RANGE"),
                (toy_triplet(), 96, 96, "CENTRE_OUT_OF_RANGE"),
                (toy_triplet(), 1, 95, "DEPTH_MUST_BE_96"),
                (toy_triplet()[:2], 1, 96, "TRIPLET_SHAPE_REQUIRED"),
                ([[[256]], [[0]], [[0]]], 1, 96, "UINT8_REQUIRED"),
                ([[[True]], [[0]], [[0]]], 1, 96, "UINT8_REQUIRED"))
    for triplet, centre, depth, expected_code in failures:
        try:
            reflect_triplet(triplet, centre, depth)
        except ValueError as exc:
            assert str(exc) == expected_code
        else:
            raise AssertionError("invalid fixture accepted")
    # Channel-dependent toy LUTs show why reflecting AFTER normalizing is wrong.
    def toy_normalize(triplet):
        return [[[value + 50 * c for value in row] for row in plane]
                for c, plane in enumerate(triplet)]
    right = toy_normalize(reflect_triplet(toy_triplet(), 1))
    wrong = toy_normalize(toy_triplet())[::-1]
    assert right != wrong
    return {"status": "PASS_SYNTHETIC_GEOMETRY_ONLY", "boundary_cases": 10,
            "refusal_cases": 6, "checks": ["explicit_coordinate_oracle",
            "double_reflection", "input_preserved", "before_LUT_order"],
            "model_or_patient_data_used": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    print(json.dumps(self_test() if args.self_test else fixture_document(),
                     sort_keys=True, indent=2))
