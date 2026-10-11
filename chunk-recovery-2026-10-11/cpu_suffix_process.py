"""Fresh-process suffix training witness using the retained CPU reference.

This trains groups101..868 from the exact first100-group commit, with actual
synthetic DataLoader generations and RNG-driven training. It is CPU evidence;
it does not execute MRI readers, CUDA/DataParallel workers or evaluation.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import resource
import time

import torch

from chunk_recovery import (CheckpointStore, ChunkPlan, CommitRef, OptimizerCallObserver,
                            capture_state, file_hash, require, restore_state)
from cpu_training_experiment import POLICY, SEED, build, contract, full_order, same_tree, train_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started, cpu = time.monotonic(), time.process_time()
    reference = Path(args.reference)
    original = json.loads((reference / "CPU_RECEIPT.json").read_text())
    require(original["status"] == "PASS_CPU_TRAINING_AND_PROCESS_CRASH_WITNESS"
            and all(original["bitwise_cpu_comparisons"].values()), "uninterrupted reference unqualified")
    declared = contract()
    require(original["contract"] == declared, "reference contract drift")
    store = CheckpointStore(reference / "chunked")
    first, last = CommitRef(**original["commits"][0]), CommitRef(**original["commits"][-1])
    # Actual separate-process initialization consumes unrelated RNG state.
    objects = build(44001)
    torch.rand(17); random.random()
    payload, ledger = store.load_exact(first, ChunkPlan(1, 100, 200), POLICY, declared)
    restore_state(payload["state"], *objects)
    observer = OptimizerCallObserver(objects[1], ledger)
    order_log, generations = [], []
    for epoch in (1, 2):
        endpoints = ((100, 200), (200, 300), (300, 400), (400, 434)) if epoch == 1 else ((0, 100), (100, 200), (200, 300), (300, 400), (400, 434))
        for start, stop in endpoints:
            require(time.monotonic() - started < 45, "fresh suffix wall cap")
            train_chunk(objects, ledger, observer, ChunkPlan(epoch, start, stop), declared, order_log, generations)
    actual = capture_state(*objects)
    expected, expected_ledger = store.verify_commit(last, ChunkPlan(2, 400, 434), POLICY, declared)
    comparisons = {key: same_tree(actual[key], expected["state"][key]) for key in actual}
    comparisons["health_ledger"] = asdict(ledger) == asdict(expected_ledger)
    expected_order = [[1, index] for index in full_order(1)[800:3472]] + [[2, index] for index in full_order(2)[:3472]]
    expected_generations = []
    for epoch in (1, 2):
        for start in range(100 if epoch == 1 else 0, 434, 50):
            expected_generations.append([epoch, start, SEED + epoch * 100003 + start, (min(start + 50, 434) - start) * 8])
    comparisons["actual_suffix_order"] = order_log == expected_order
    comparisons["absolute_loader_generations"] = generations == expected_generations
    require(all(comparisons.values()), "fresh-process suffix equivalence failed")
    receipt = {"status": "PASS_FRESH_PROCESS_CPU_SUFFIX_TRAINING", "device": "cpu", "torch": torch.__version__,
               "dtype": "float32", "threads": 1, "workers": 0, "DataParallel": False,
               "starting_global_group": 100, "trained_suffix_groups": 768, "final_global_group": 868,
               "suffix_studies_observed": len(order_log), "suffix_loader_generations": len(generations),
               "full_schedule_horizon": 1302, "comparisons_to_cpu_uninterrupted_reference": comparisons,
               "reference_receipt_sha256": file_hash(reference / "CPU_RECEIPT.json"),
               "first_commit": asdict(first), "final_reference_commit": asdict(last),
               "cpu_s": time.process_time() - cpu, "wall_s": time.monotonic() - started,
               "cpu_cap_s": 60, "wall_cap_s": 45, "native_knee_execution": False,
               "evaluation_execution": False, "CUDA_equivalence_proven": False,
               "source_sha256": file_hash(Path(__file__)),
               "training_source_sha256": file_hash(Path(__file__).with_name("cpu_training_experiment.py")),
               "library_sha256": file_hash(Path(__file__).with_name("chunk_recovery.py")), "official_score": None}
    with open(args.output, "x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("first_commit", "final_reference_commit")}, indent=2))


if __name__ == "__main__":
    main()
