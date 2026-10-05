"""Standalone synthetic checks; never loads model, patient data or targets."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import platform

import numpy as np
from peer_selection_dense import (
    dense_peer_mask, dense_peer_mask_torch, random_stratified_mask, selected_support,
)


def sorted_list_reference(loss, positive, retain):
    """Independent scalar/list implementation of the frozen original rule."""
    answer = np.zeros(positive.shape, dtype=bool)
    for finding in range(positive.shape[1]):
        for class_value in (False, True):
            rows = [i for i in range(positive.shape[0]) if bool(positive[i, finding]) == class_value]
            if not rows:
                continue
            ordered = sorted(rows, key=lambda i: (loss[i, finding], i))
            number = max(1, math.floor(float(retain) * len(rows)))
            for i in ordered[:number]:
                answer[i, finding] = True
    return answer


def class_count_reference(mask, positive):
    answer = []
    for finding in range(positive.shape[1]):
        answer.append([sum(bool(mask[i, finding]) and bool(positive[i, finding]) == cls
                           for i in range(positive.shape[0])) for cls in (False, True)])
    return np.asarray(answer, dtype=np.int64)


def check_expected_exception(fn, exception):
    try:
        fn()
    except exception:
        return
    raise AssertionError("Expected contract exception: " + exception.__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="A new JSON report file; existing files are refused")
    args = parser.parse_args()
    retain_values = (0.0, 0.2, 0.5, 0.8, 1.0)
    exhaustive = {}
    count = 0
    for batch in (3, 4):
        seen = 0
        for classes in product((False, True), repeat=batch):
            positive = np.asarray(classes, dtype=bool).reshape(batch, 1)
            for values in product((-1.0, 0.0, 1.0), repeat=batch):
                loss = np.asarray(values).reshape(batch, 1)
                for retain in retain_values:
                    actual = dense_peer_mask(loss, positive, retain)
                    expected = sorted_list_reference(loss, positive, retain)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(selected_support(actual, positive),
                                                  class_count_reference(expected, positive))
                    seen += 1
                    count += 1
                mask = dense_peer_mask(loss, positive, .8)
                total = int(mask.sum())
                expected_total = 2 if batch == 3 or sum(classes) == 2 else 3
                assert total == expected_total
        exhaustive[str(batch)] = seen

    generator = np.random.Generator(np.random.PCG64(20261005))
    randomized = 1000
    multicolumn_cases = 0
    parameter_combinations = set()
    random_masks_different_from_low_loss = 0
    for trial in range(randomized):
        batch = (1, 2, 3, 4, 7, 16)[trial % 6]
        findings = (1, 3, 12)[(trial // 6) % 3]
        positive = generator.integers(0, 2, size=(batch, findings)).astype(bool)
        # Many deliberate ties, including negative loss values.
        loss = generator.integers(-3, 4, size=(batch, findings)).astype(np.float64)
        retain = (0.0, 0.05, 0.33, 0.5, 0.8, 1.0)[(trial // 18) % 6]
        multicolumn_cases += int(findings > 1)
        parameter_combinations.add((batch, findings, retain))
        old_loss, old_classes = loss.copy(), positive.copy()
        peer = dense_peer_mask(loss, positive, retain)
        np.testing.assert_array_equal(peer, sorted_list_reference(loss, positive, retain))
        control = random_stratified_mask(positive, retain, seed=trial)
        np.testing.assert_array_equal(control, random_stratified_mask(positive, retain, seed=trial))
        priorities = np.random.Generator(np.random.PCG64(trial)).random(positive.shape)
        np.testing.assert_array_equal(control, sorted_list_reference(priorities, positive, retain))
        np.testing.assert_array_equal(selected_support(peer, positive), selected_support(control, positive))
        np.testing.assert_array_equal(loss, old_loss)
        np.testing.assert_array_equal(positive, old_classes)
        random_masks_different_from_low_loss += int(not np.array_equal(peer, control))

    # Constant priorities exercise stable tie handling in the random ranking rule.
    tied_priority_cases = 0
    for batch in (3, 4):
        for classes in product((False, True), repeat=batch):
            positive = np.asarray(classes, dtype=bool).reshape(batch, 1)
            loss = np.zeros((batch, 1))
            np.testing.assert_array_equal(dense_peer_mask(loss, positive, .8),
                                          sorted_list_reference(loss, positive, .8))
            tied_priority_cases += 1

    good_loss = np.zeros((4, 2))
    good_classes = np.zeros((4, 2), dtype=bool)
    contract_cases = [
        (lambda: dense_peer_mask(good_loss, good_classes, True), TypeError),
        (lambda: dense_peer_mask(good_loss, good_classes, "0.8"), TypeError),
        (lambda: dense_peer_mask(good_loss, good_classes, np.nan), ValueError),
        (lambda: dense_peer_mask(good_loss, good_classes, np.inf), ValueError),
        (lambda: dense_peer_mask(good_loss, good_classes, -.1), ValueError),
        (lambda: dense_peer_mask(good_loss, good_classes, 1.1), ValueError),
        (lambda: dense_peer_mask(good_loss[:, 0], good_classes, .8), ValueError),
        (lambda: dense_peer_mask(np.zeros((0, 2)), np.zeros((0, 2), dtype=bool), .8), ValueError),
        (lambda: dense_peer_mask(good_loss, good_classes.astype(float), .8), TypeError),
        (lambda: dense_peer_mask(np.full((4, 2), np.nan), good_classes, .8), ValueError),
        (lambda: dense_peer_mask(np.full((4, 2), np.inf), good_classes, .8), ValueError),
        (lambda: dense_peer_mask(np.full((4, 2), 1j), good_classes, .8), TypeError),
        (lambda: dense_peer_mask(good_classes, good_classes, .8), TypeError),
        (lambda: random_stratified_mask(good_classes, .8, seed=-1), ValueError),
        (lambda: random_stratified_mask(good_classes, .8, seed=2**64), ValueError),
        (lambda: random_stratified_mask(good_classes, .8, seed=True), TypeError),
        (lambda: random_stratified_mask(good_classes, .8, seed=1.5), TypeError),
        (lambda: selected_support(np.zeros((4, 2)), good_classes), ValueError),
    ]
    for fn, exception in contract_cases:
        check_expected_exception(fn, exception)

    # The isolated random control must leave global legacy NumPy RNG untouched.
    global_before = np.random.get_state()
    random_stratified_mask(good_classes, .8, seed=42)
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]

    fixture = json.loads((Path(__file__).parent / "v54_support_counts.json").read_text())
    full = np.asarray(fixture["full_support_per_network"], dtype=np.int64)
    selected = np.asarray(fixture["peer_epoch2_support_per_network"], dtype=np.int64)
    assert full.shape == selected.shape == (12, 2)
    assert (full.sum(axis=1) == 3487).all()
    assert ((0 < selected) & (selected <= full)).all()
    balanced = 3 * 871 + 2 - selected.sum(axis=1)
    assert ((0 <= balanced) & (balanced <= 871)).all()
    # From exhaustive rule: balanced B4 keeps2, other B4 keeps3, final B3 keeps2.
    np.testing.assert_array_equal(2 * balanced + 3 * (871 - balanced) + 2,
                                  selected.sum(axis=1))
    assert int(full.sum()) == 41844 and int(selected.sum()) == 29309
    assert int(selected[:, 0].sum()) == 21381 and int(selected[:, 1].sum()) == 7928

    torch_result = {"status": "not_installed", "synthetic_cpu_cases": 0}
    if importlib.util.find_spec("torch") is not None:
        import torch
        torch_cases = 0
        for batch in (3, 4):
            for classes in product((False, True), repeat=batch):
                positive = np.asarray(classes, dtype=bool).reshape(batch, 1)
                for values in product((-1.0, 0.0, 1.0), repeat=batch):
                    loss = np.asarray(values).reshape(batch, 1)
                    actual = dense_peer_mask_torch(torch.from_numpy(loss), torch.from_numpy(positive), .8)
                    np.testing.assert_array_equal(actual.numpy(), sorted_list_reference(loss, positive, .8))
                    torch_cases += 1
        for trial in range(100):
            loss = generator.integers(-2, 3, size=(4, 12)).astype(np.float64)
            positive = generator.integers(0, 2, size=(4, 12)).astype(bool)
            retain = retain_values[trial % len(retain_values)]
            actual = dense_peer_mask_torch(torch.from_numpy(loss), torch.from_numpy(positive), retain)
            np.testing.assert_array_equal(actual.numpy(), sorted_list_reference(loss, positive, retain))
            torch_cases += 1
        torch_rejected_dtypes = []
        for name in ("uint16", "uint32", "uint64", "float8_e4m3fn", "float8_e5m2", "bool", "complex64"):
            if hasattr(torch, name):
                loss = torch.zeros((4, 2), dtype=getattr(torch, name))
                positive = torch.zeros((4, 2), dtype=torch.bool)
                check_expected_exception(lambda: dense_peer_mask_torch(loss, positive, .8), TypeError)
                torch_rejected_dtypes.append(name)
        torch_supported_dtype_cases = 0
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                      torch.float16, torch.bfloat16, torch.float32, torch.float64):
            loss = torch.tensor([[0], [1], [1], [2]], dtype=dtype)
            positive = torch.tensor([[False], [True], [True], [False]], dtype=torch.bool)
            actual = dense_peer_mask_torch(loss, positive, .8)
            expected = sorted_list_reference(np.asarray([[0], [1], [1], [2]]), positive.numpy(), .8)
            np.testing.assert_array_equal(actual.numpy(), expected)
            torch_supported_dtype_cases += 1
        torch_result = {"status": "synthetic_cpu_pass", "version": torch.__version__,
                        "synthetic_cpu_cases": torch_cases,
                        "supported_dtype_cases": torch_supported_dtype_cases,
                        "unsupported_dtypes_explicitly_rejected": torch_rejected_dtypes,
                        "accelerator_tests": 0}

    report = {
        "status": "PASS_STANDALONE_SYNTHETIC_METHOD_CHECKS",
        "exhaustive_cases_by_batch": exhaustive, "exhaustive_total_cases": count,
        "exhaustive_retain_values": retain_values,
        "randomized_cases": randomized,
        "randomized_multicolumn_cases": multicolumn_cases,
        "randomized_parameter_combinations": len(parameter_combinations),
        "tied_priority_cases": tied_priority_cases,
        "random_masks_different_from_low_loss": random_masks_different_from_low_loss,
        "contract_rejection_cases": len(contract_cases),
        "global_numpy_rng_preserved": True,
        "v54_aggregate_counts_checked": True,
        "v54_balanced_b4_finding_batches_inferred": balanced.tolist(),
        "v54_peer_epoch2_retained": int(selected.sum()),
        "v54_full_exposure": int(full.sum()),
        "v54_realized_retention_fraction": float(selected.sum()/full.sum()),
        "v54_counts_scope": "Per-network finding/class aggregates; balanced batch counts are inferred, not independently logged.",
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch_result,
        "source_files_sha256": {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                for name in ("peer_selection_dense.py", "verify_method.py", "v54_support_counts.json")},
        "model_calls": 0, "target_or_weight_reads": False,
        "gpu_tpu_runs": 0, "speed_measurement": None, "quality_metric": None, "official_score": None,
        "current_v55_modified": False, "remote_mutations": False,
    }
    if args.report:
        with args.report.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
