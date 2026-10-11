"""Targeted negative controls using the measured synthetic CPU100 commit.

No MRI data, remote APIs, deletions, overwritten files or training repetition.
Outputs must be fresh and every partial artifact remains preserved.
"""
import argparse
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import resource
import time
import unittest

import torch

from chunk_recovery import (CheckpointStore, ChunkPlan, CommitRef, HealthLedger, HealthPolicy, Hold,
                            OptimizerCallObserver, capture_state, digest, file_hash, require_initialization)
from cpu_training_experiment import POLICY, build, contract


FIXTURE = None
OUTPUT = None


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        receipt = json.loads((FIXTURE / "CPU_RECEIPT.json").read_text())
        cls.ref = CommitRef(**receipt["commits"][0])
        cls.source_store = CheckpointStore(FIXTURE / "chunked")
        cls.plan = ChunkPlan(1, 0, 100)
        cls.next_plan = ChunkPlan(1, 100, 200)
        cls.declared = contract()
        cls.payload, _ = cls.source_store.load_exact(cls.ref, cls.next_plan, POLICY, cls.declared)

    def setUp(self):
        self.store = CheckpointStore(OUTPUT / self.id().split(".")[-1])
        self.ledger = HealthLedger(**copy.deepcopy(self.payload["health"]["ledger"]))

    def save(self, ledger=None, policy=POLICY):
        return self.store.save("negative", self.payload["state"], self.plan, ledger or self.ledger,
                               policy, self.declared, self.ref.predecessor_source_sha256, self.ref.predecessor_origin)

    def test_missing_health_allowance_is_hold(self):
        with self.assertRaises(Hold):
            HealthPolicy(0, None, "absent-native-allowance")

    def test_missing_exact_predecessor_is_hold(self):
        with self.assertRaises(Hold):
            self.source_store.load_exact(None, self.next_plan, POLICY, self.declared)

    def test_fresh_mode_cannot_bypass_continuation(self):
        with self.assertRaises(Hold):
            require_initialization(self.next_plan, "START_FROM_INITIAL_STATE", None)

    def test_wrong_absolute_predecessor_is_hold(self):
        with self.assertRaises(Hold):
            self.source_store.load_exact(self.ref, ChunkPlan(1, 200, 300), POLICY, self.declared)

    def test_wrong_source_binding_is_hold(self):
        with self.assertRaises(Hold):
            self.source_store.load_exact(replace(self.ref, predecessor_source_sha256="a" * 64), self.next_plan, POLICY, self.declared)

    def test_changed_training_contract_is_hold(self):
        changed = copy.deepcopy(self.declared); changed["invariants_sha256"] = "a" * 64
        with self.assertRaises(Hold):
            self.source_store.load_exact(self.ref, self.next_plan, POLICY, changed)

    def test_saved_plan_geometry_must_match_declared_contract(self):
        payload = copy.deepcopy(self.payload)
        payload["plan"]["recycle"] = 25
        pt, side = self.store.root / "chunk-drift.pt", self.store.root / "chunk-drift.json"
        with open(pt, "xb") as stream:
            torch.save(payload, stream)
        marker = json.loads((self.source_store.root / self.ref.sidecar).read_text())
        marker.update(plan=payload["plan"], checkpoint=pt.name, checkpoint_sha256=file_hash(pt), checkpoint_bytes=pt.stat().st_size)
        with open(side, "x") as stream:
            json.dump(marker, stream)
        ref = CommitRef(pt.name, file_hash(pt), side.name, file_hash(side), self.ref.predecessor_source_sha256, self.ref.predecessor_origin)
        with self.assertRaises(Hold):
            self.store.load_exact(ref, self.next_plan, POLICY, self.declared)

    def test_full_fit_corpus_and_434_geometry_required(self):
        with self.assertRaises(Hold):
            ChunkPlan(1, 0, 100, groups=433)
        with self.assertRaises(Hold):
            list(self.plan.slices(list(range(3472)), digest(list(range(3472)))))

    def test_same_count_reordered_consumed_prefix_is_hold(self):
        self.ledger.order_hash = digest(["same-count-reordered-microbatch"])
        with self.assertRaises(Hold):
            self.save()
        self.assertEqual(list(self.store.root.iterdir()), [])

    def test_unused_seven_position_tail_drift_is_hold(self):
        order = list(range(3479)); changed = order.copy()
        changed[-1], changed[-2] = changed[-2], changed[-1]
        with self.assertRaises(Hold):
            list(self.plan.slices(changed, digest(order)))

    def test_partial_accumulation_is_hold(self):
        self.ledger.pending_micro = 1
        with self.assertRaises(Hold):
            self.save()

    def test_epoch_loss_diagnostics_cannot_be_truncated(self):
        self.ledger.epoch_loss_count = 4
        with self.assertRaises(Hold):
            self.save()

    def test_scheduler_call_clock_cannot_be_invented(self):
        self.ledger.scheduler_calls = 99
        with self.assertRaises(Hold):
            self.save()

    def test_build_and_missing_allowances_are_explicit(self):
        self.ledger.build_failures = 1
        with self.assertRaises(Hold):
            self.save()
        self.ledger.build_failures = 0; self.ledger.missing_slots = 1001
        with self.assertRaises(Hold):
            self.save()

    def test_nonzero_declared_missing_slots_can_pass(self):
        self.assertGreater(self.ledger.missing_slots, 0)
        ref = self.save()
        payload, _ = self.store.verify_commit(ref, self.plan, POLICY, self.declared)
        self.assertEqual(payload["health"]["ledger"]["missing_slots"], self.ledger.missing_slots)

    def test_existing_pair_never_overwritten_even_on_retry(self):
        ref = self.save()
        before = {p.name: file_hash(p) for p in self.store.root.iterdir()}
        with self.assertRaises(Hold):
            self.save()
        self.assertEqual(before, {p.name: file_hash(p) for p in self.store.root.iterdir()})
        self.store.verify_commit(ref, self.plan, POLICY, self.declared)

    def test_orphan_is_preserved_and_not_reused(self):
        orphan = self.store.root / "chunk-negative.pt"
        with open(orphan, "xb") as stream:
            stream.write(b"partial-checkpoint")
        with self.assertRaises(Hold):
            self.save()
        self.assertEqual(orphan.read_bytes(), b"partial-checkpoint")
        self.assertFalse((self.store.root / "chunk-negative.json").exists())

    def test_wrong_checkpoint_hash_is_rejected_before_deserialization(self):
        with self.assertRaises(Hold):
            self.source_store.load_exact(replace(self.ref, checkpoint_sha256="0" * 64), self.next_plan, POLICY, self.declared)

    def test_malformed_sidecar_is_rejected_and_retained(self):
        pt, side = self.store.root / "chunk-malformed.pt", self.store.root / "chunk-malformed.json"
        with open(pt, "xb") as stream:
            stream.write((self.source_store.root / self.ref.checkpoint).read_bytes())
        with open(side, "xb") as stream:
            stream.write(b"{")
        ref = CommitRef(pt.name, file_hash(pt), side.name, file_hash(side), self.ref.predecessor_source_sha256, self.ref.predecessor_origin)
        with self.assertRaises(json.JSONDecodeError):
            self.store.load_exact(ref, self.next_plan, POLICY, self.declared)
        self.assertEqual(side.read_bytes(), b"{")
        self.assertTrue(pt.is_file())

    def test_nonfinite_loss_permanently_taints_ledger(self):
        with self.assertRaises(Hold):
            self.ledger.microbatch(float("nan"), [1, 2], 1)
        self.assertTrue(self.ledger.tainted)
        with self.assertRaises(Hold):
            self.save()

    def test_live_gradients_cannot_be_captured(self):
        objects = build(31)
        for p in objects[0].parameters():
            p.grad = torch.ones_like(p)
        with self.assertRaises(Hold):
            capture_state(*objects)

    def observer(self):
        ledger = HealthLedger()
        p = torch.nn.Parameter(torch.tensor([1.0])); p.grad = torch.ones_like(p)
        opt = torch.optim.SGD([p], lr=0.1)
        return ledger, opt, OptimizerCallObserver(opt, ledger)

    def test_skipped_call_permanently_taints_observer_and_ledger(self):
        ledger, opt, observer = self.observer()
        with self.assertRaises(Hold):
            observer.step(lambda: None)
        with self.assertRaises(Hold):
            observer.step(opt.step)
        self.assertTrue(observer.tainted and ledger.tainted)

    def test_callback_exception_permanently_taints(self):
        ledger, opt, observer = self.observer()
        def failure():
            raise ValueError("invented callback failure")
        with self.assertRaises(ValueError):
            observer.step(failure)
        with self.assertRaises(Hold):
            observer.step(opt.step)
        self.assertTrue(ledger.tainted)

    def test_external_optimizer_call_is_permanently_ineligible(self):
        ledger, opt, observer = self.observer()
        opt.step()
        with self.assertRaises(Hold):
            observer.step(opt.step)
        self.assertTrue(ledger.tainted)

    def test_partial_projection_excludes_434_tail_and_evaluation(self):
        self.assertEqual(self.plan.total_schedule, 1302)
        self.assertAlmostEqual(self.plan.project(849, 15.878, 50, 240), 1882.9)
        with self.assertRaises(Hold):
            self.plan.project(849, 15.878, 50, 240, evaluation_tail=1679.1)


def main():
    global FIXTURE, OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    FIXTURE, OUTPUT = Path(args.fixture), Path(args.output)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    torch.set_num_threads(1)
    cpu, wall = time.process_time(), time.monotonic()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    receipt = {"status": "PASS_TARGETED_CONTROLS" if result.wasSuccessful() else "FAIL_TARGETED_CONTROLS",
               "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "cpu_s": time.process_time() - cpu, "wall_s": time.monotonic() - wall,
               "source_sha256": file_hash(Path(__file__)),
               "library_sha256": file_hash(Path(__file__).with_name("chunk_recovery.py")),
               "fixture_receipt_sha256": file_hash(FIXTURE / "CPU_RECEIPT.json"),
               "optimized": __import__("sys").flags.optimize, "native_execution": False}
    with open(OUTPUT / "CONTROLS_LOG.txt", "x") as f:
        f.write(stream.getvalue())
    with open(OUTPUT / "CONTROLS_RECEIPT.json", "x") as f:
        json.dump(receipt, f, indent=2)
    print(stream.getvalue())
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
