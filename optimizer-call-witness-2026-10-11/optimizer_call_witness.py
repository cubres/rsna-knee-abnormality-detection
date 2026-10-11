"""Original, in-memory instrumentation for optimizer calls and scheduler progress.

No tensor loading, file writes, credentials, network, or Kaggle operations. A
completed optimizer call is not a proof that parameters changed or were finite.
The supported adapter is PyTorch's Optimizer.register_step_post_hook API.
"""
from collections.abc import Mapping
import re


class ProgressHold(RuntimeError):
    """The training loop must stop and retain its current evidence."""


def _count(value, name):
    if type(value) is not int or not 0 <= value <= 10**9:
        raise ValueError(name + " must be a bounded nonnegative integer")
    return value


class OptimizerCallWitness:
    """Track one optimizer, one attempt callback, and one scheduler callback.

    Attach before any optimizer call. For each already-accumulated batch group,
    run attempt(lambda: scaler.step(optimizer), scheduler.step), then perform
    scaler.update and zero_grad in the owning loop. A skipped scaler attempt
    does not advance the scheduler. Exceptions and calls outside an attempt
    taint this instance permanently. Keep the hook attached for its lifetime.

    Callbacks must be synchronous and this instance is single-threaded. This
    helper does not inspect gradients, change parameters, retry work, resample
    batches, or supply a policy for accepting an overflow.
    """

    def __init__(self, optimizer):
        register = getattr(optimizer, "register_step_post_hook", None)
        if not callable(register):
            raise TypeError("optimizer lacks register_step_post_hook")
        self._optimizer = optimizer
        self._active = False
        self._tainted = False
        self._hook_calls = 0
        self._attempts = 0
        self._completed = 0
        self._skipped = 0
        self._scheduler_calls = 0
        self._handle = register(self._after_step)

    def _after_step(self, optimizer, args, kwargs):
        if optimizer is not self._optimizer or not self._active:
            self._tainted = True
            raise ProgressHold("optimizer call was outside its active attempt")
        self._hook_calls += 1

    def attempt(self, optimizer_attempt, scheduler_step):
        if self._tainted:
            raise ProgressHold("a prior ambiguous attempt requires investigation")
        if self._active:
            self._tainted = True
            raise ProgressHold("nested attempts are unsupported")
        if not callable(optimizer_attempt) or not callable(scheduler_step):
            raise TypeError("both callbacks must be callable")
        before = self._hook_calls
        self._active = True
        self._attempts += 1
        try:
            optimizer_attempt()
            if self._tainted:
                raise ProgressHold("an invalid nested or external call was caught by the callback")
            calls = self._hook_calls - before
            if calls not in (0, 1):
                raise ProgressHold("one attempt completed multiple optimizer calls")
            if calls == 0:
                self._skipped += 1
                return "SKIPPED_OPTIMIZER_CALL"
            self._completed += 1
            # Preserve the exception if this callback fails. No automatic retry.
            scheduler_step()
            self._scheduler_calls += 1
            if self._tainted:
                raise ProgressHold("scheduler callback caught an invalid nested call")
            if self._hook_calls - before != 1:
                raise ProgressHold("scheduler callback called the optimizer")
            return "COMPLETED_OPTIMIZER_CALL"
        except BaseException:
            self._tainted = True
            raise
        finally:
            self._active = False

    def snapshot(self):
        return {
            "schema": "optimizer-call-witness/1",
            "attempts": self._attempts,
            "completed_optimizer_calls": self._completed,
            "skipped_optimizer_calls": self._skipped,
            "scheduler_calls": self._scheduler_calls,
            "tainted": self._tainted,
            "active": self._active,
            "parameter_change_proven": False,
            "gradient_finiteness_proven": False,
        }


def continuation_report(envelope, max_skips=0):
    """Check a caller-supplied normalized terminal/epoch envelope.

    No authentication or checkpoint deserialization is performed. This report
    cannot substitute for fresh server identity and native artifact inspection.
    It answers only whether the declared envelope meets this narrow contract.
    Invalid, missing, running, interrupted, or tainted evidence returns HOLD.
    """
    reasons = []
    try:
        _count(max_skips, "max_skips")
        if not isinstance(envelope, Mapping):
            raise ValueError("envelope must be a mapping")
        if envelope.get("native_status") != "COMPLETE":
            reasons.append("NATIVE_NOT_COMPLETE")
        if envelope.get("receipt_status") != "PASS_EPOCH":
            reasons.append("EPOCH_NOT_PASSED")
        for field in ("candidate_sha256", "checkpoint_sha256", "contract_sha256"):
            value = envelope.get(field)
            if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
                reasons.append("MISSING_OR_INVALID_" + field.upper())
        planned = _count(envelope.get("planned_batch_groups"), "planned_batch_groups")
        consumed = _count(envelope.get("consumed_batch_groups"), "consumed_batch_groups")
        if planned == 0 or consumed != planned:
            reasons.append("BATCH_GROUP_COVERAGE")
        witness = envelope.get("witness")
        if not isinstance(witness, Mapping) or witness.get("schema") != "optimizer-call-witness/1":
            raise ValueError("witness is absent or has an unknown schema")
        attempts = _count(witness.get("attempts"), "attempts")
        completed = _count(witness.get("completed_optimizer_calls"), "completed_optimizer_calls")
        skipped = _count(witness.get("skipped_optimizer_calls"), "skipped_optimizer_calls")
        scheduled = _count(witness.get("scheduler_calls"), "scheduler_calls")
        if witness.get("tainted") is not False or witness.get("active") is not False:
            reasons.append("AMBIGUOUS_WITNESS")
        if attempts != consumed or completed + skipped != attempts:
            reasons.append("CALL_ACCOUNTING")
        if scheduled != completed:
            reasons.append("SCHEDULER_CALL_ACCOUNTING")
        if skipped > max_skips:
            reasons.append("SKIP_POLICY")
        if envelope.get("finite_loss_and_gradient_checks") is not True:
            reasons.append("FINITE_CHECKS_UNOBSERVED")
        if envelope.get("training_build_failures") != 0 or type(envelope.get("training_build_failures")) is not int:
            reasons.append("TRAINING_BUILD_FAILURES")
        if envelope.get("checkpoint_contract_matches") is not True:
            reasons.append("CHECKPOINT_CONTRACT_UNBOUND")
    except (TypeError, ValueError, KeyError):
        reasons.append("MALFORMED_OR_INCOMPLETE_ENVELOPE")
    return {
        "status": "HOLD" if reasons else "DECLARED_CONTINUATION_CONTRACT_PASS",
        "reasons": sorted(set(reasons)),
        "authenticity_proven": False,
        "checkpoint_numerics_proven": False,
        "independent_validation_proven": False,
        "official_score": None,
    }
