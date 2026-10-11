"""Invented controls; no competition records or tensor checkpoints."""
import copy
import unittest

from optimizer_call_witness import OptimizerCallWitness, ProgressHold, continuation_report


class ToyOptimizer:
    def __init__(self):
        self.hook = None
        self.calls = 0

    def register_step_post_hook(self, hook):
        self.hook = hook
        return object()

    def step(self):
        self.calls += 1
        self.hook(self, (), {})


def envelope(witness=None):
    return {
        "native_status": "COMPLETE", "receipt_status": "PASS_EPOCH",
        "candidate_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
        "contract_sha256": "c" * 64, "planned_batch_groups": 2,
        "consumed_batch_groups": 2, "checkpoint_contract_matches": True,
        "finite_loss_and_gradient_checks": True, "training_build_failures": 0,
        "witness": witness or {
            "schema": "optimizer-call-witness/1", "attempts": 2,
            "completed_optimizer_calls": 2, "skipped_optimizer_calls": 0,
            "scheduler_calls": 2, "tainted": False, "active": False,
        },
    }


class WitnessTests(unittest.TestCase):
    def setUp(self):
        self.opt = ToyOptimizer()
        self.w = OptimizerCallWitness(self.opt)
        self.scheduler_calls = 0

    def schedule(self):
        self.scheduler_calls += 1

    def test_success_counts_call_and_schedule(self):
        self.assertEqual(self.w.attempt(self.opt.step, self.schedule), "COMPLETED_OPTIMIZER_CALL")
        self.assertEqual(self.w.snapshot()["completed_optimizer_calls"], 1)
        self.assertEqual(self.scheduler_calls, 1)

    def test_skip_leaves_scheduler_unchanged(self):
        self.assertEqual(self.w.attempt(lambda: None, self.schedule), "SKIPPED_OPTIMIZER_CALL")
        self.assertEqual(self.scheduler_calls, 0)
        self.assertEqual(self.w.snapshot()["skipped_optimizer_calls"], 1)

    def test_success_skip_success(self):
        self.w.attempt(self.opt.step, self.schedule)
        self.w.attempt(lambda: None, self.schedule)
        self.w.attempt(self.opt.step, self.schedule)
        self.assertEqual(self.scheduler_calls, 2)
        self.assertEqual(self.w.snapshot()["attempts"], 3)

    def test_optimizer_exception_taints_permanently(self):
        def fail():
            raise RuntimeError("invented")
        with self.assertRaises(RuntimeError):
            self.w.attempt(fail, self.schedule)
        with self.assertRaises(ProgressHold):
            self.w.attempt(self.opt.step, self.schedule)
        self.assertEqual(self.scheduler_calls, 0)

    def test_exception_after_step_does_not_schedule(self):
        def fail_after():
            self.opt.step()
            raise RuntimeError("invented")
        with self.assertRaises(RuntimeError):
            self.w.attempt(fail_after, self.schedule)
        self.assertEqual(self.scheduler_calls, 0)
        self.assertTrue(self.w.snapshot()["tainted"])

    def test_scheduler_exception_taints(self):
        def fail():
            raise RuntimeError("invented")
        with self.assertRaises(RuntimeError):
            self.w.attempt(self.opt.step, fail)
        self.assertEqual(self.w.snapshot()["completed_optimizer_calls"], 1)
        self.assertEqual(self.w.snapshot()["scheduler_calls"], 0)

    def test_outside_call_is_hold(self):
        with self.assertRaises(ProgressHold):
            self.opt.step()
        self.assertTrue(self.w.snapshot()["tainted"])

    def test_double_optimizer_call_is_hold(self):
        def twice():
            self.opt.step()
            self.opt.step()
        with self.assertRaises(ProgressHold):
            self.w.attempt(twice, self.schedule)
        self.assertEqual(self.scheduler_calls, 0)

    def test_nested_caught_exception_still_holds(self):
        def nested():
            try:
                self.w.attempt(self.opt.step, self.schedule)
            except ProgressHold:
                pass
        with self.assertRaises(ProgressHold):
            self.w.attempt(nested, self.schedule)

    def test_scheduler_cannot_step_optimizer(self):
        with self.assertRaises(ProgressHold):
            self.w.attempt(self.opt.step, self.opt.step)

    def test_invalid_callback_does_not_consume_attempt(self):
        with self.assertRaises(TypeError):
            self.w.attempt(None, self.schedule)
        self.assertEqual(self.w.snapshot()["attempts"], 0)

    def test_adapter_missing(self):
        with self.assertRaises(TypeError):
            OptimizerCallWitness(object())

    def test_snapshot_makes_no_parameter_or_finite_claim(self):
        s = self.w.snapshot()
        self.assertFalse(s["parameter_change_proven"])
        self.assertFalse(s["gradient_finiteness_proven"])


class EnvelopeTests(unittest.TestCase):
    def test_declared_contract_pass(self):
        r = continuation_report(envelope())
        self.assertEqual(r["status"], "DECLARED_CONTINUATION_CONTRACT_PASS")
        self.assertFalse(r["authenticity_proven"])
        self.assertFalse(r["independent_validation_proven"])
        self.assertIsNone(r["official_score"])

    def test_running_is_hold(self):
        e = envelope(); e["native_status"] = "RUNNING"
        self.assertIn("NATIVE_NOT_COMPLETE", continuation_report(e)["reasons"])

    def test_incomplete_epoch_is_hold(self):
        e = envelope(); e["consumed_batch_groups"] = 1
        self.assertIn("BATCH_GROUP_COVERAGE", continuation_report(e)["reasons"])

    def test_old_receipt_without_witness_is_hold(self):
        e = envelope(); e.pop("witness")
        self.assertEqual(continuation_report(e)["status"], "HOLD")

    def test_skip_policy_and_scheduler_counts(self):
        e = envelope(); e["witness"]["completed_optimizer_calls"] = 1
        e["witness"]["skipped_optimizer_calls"] = 1
        e["witness"]["scheduler_calls"] = 1
        self.assertIn("SKIP_POLICY", continuation_report(e)["reasons"])
        self.assertEqual(continuation_report(e, max_skips=1)["status"], "DECLARED_CONTINUATION_CONTRACT_PASS")
        e["witness"]["scheduler_calls"] = 2
        self.assertIn("SCHEDULER_CALL_ACCOUNTING", continuation_report(e, 1)["reasons"])

    def test_boolean_count_is_not_integer(self):
        e = envelope(); e["witness"]["attempts"] = True
        self.assertIn("MALFORMED_OR_INCOMPLETE_ENVELOPE", continuation_report(e)["reasons"])

    def test_nonfinite_and_oversized_counts_hold(self):
        for value in (float("nan"), float("inf"), -1, 10**10, "2"):
            e = envelope(); e["planned_batch_groups"] = value
            self.assertEqual(continuation_report(e)["status"], "HOLD")

    def test_tainted_or_active_hold(self):
        for field in ("tainted", "active"):
            e = envelope(); e["witness"][field] = True
            self.assertIn("AMBIGUOUS_WITNESS", continuation_report(e)["reasons"])

    def test_explicit_build_failure_and_checks_hold(self):
        e = envelope(); e["training_build_failures"] = 1
        e["finite_loss_and_gradient_checks"] = False
        e["checkpoint_contract_matches"] = False
        r = continuation_report(e)
        self.assertIn("TRAINING_BUILD_FAILURES", r["reasons"])
        self.assertIn("FINITE_CHECKS_UNOBSERVED", r["reasons"])
        self.assertIn("CHECKPOINT_CONTRACT_UNBOUND", r["reasons"])

    def test_invalid_hashes_hold(self):
        e = envelope(); e["candidate_sha256"] = "A" * 64
        self.assertEqual(continuation_report(e)["status"], "HOLD")

    def test_missing_and_nondictionary_hold(self):
        for e in ({}, None, [], ""):
            self.assertEqual(continuation_report(e)["status"], "HOLD")

    def test_does_not_mutate_envelope(self):
        e = envelope(); before = copy.deepcopy(e)
        continuation_report(e)
        self.assertEqual(e, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
