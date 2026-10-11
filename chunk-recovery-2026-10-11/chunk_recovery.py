"""MIT original checkpoint/horizon helpers. No discovery, cleanup or remote API.

Hash bindings prove integrity against caller-supplied pins, not authenticity of
a Kaggle receipt. Missing data-health allowances fail closed. All writes are
exclusive new .pt/.json pairs; existing files are never replaced or removed.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re

import torch


class Hold(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Hold(message)


def integer(value, name, minimum=0):
    require(type(value) is int and minimum <= value <= 10**7, name)
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid SHA256 pin")
    return value


@dataclass(frozen=True)
class HealthPolicy:
    max_build_failures: int
    max_missing_slots: int
    registration: str
    basis: str = "cumulative-consumed-studies"

    def __post_init__(self):
        integer(self.max_build_failures, "explicit build-failure allowance required")
        integer(self.max_missing_slots, "explicit missing-slot allowance required")
        require(type(self.registration) is str and bool(self.registration), "health-policy registration required")
        require(self.basis == "cumulative-consumed-studies", "unknown health-policy denominator")

    @property
    def hash(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ChunkPlan:
    epoch: int
    start: int
    stop: int
    groups: int = 434
    fit_studies: int = 3479
    micro: int = 2
    accum: int = 4
    checkpoint_every: int = 100
    recycle: int = 50
    train_epochs: int = 2
    schedule_epochs: int = 3

    def __post_init__(self):
        for key, value in asdict(self).items():
            integer(value, key, 0 if key == "start" else 1)
        require(self.epoch <= self.train_epochs <= self.schedule_epochs, "invalid training/schedule horizon")
        require(self.groups == (self.fit_studies // self.micro) // self.accum, "group geometry differs from full fit corpus")
        require(self.checkpoint_every % self.recycle == 0, "checkpoint/recycle grid mismatch")
        require(0 <= self.start < self.stop <= self.groups, "invalid session interval")
        require(self.start % self.checkpoint_every == 0, "unregistered start point")
        require(self.stop == min(self.start + self.checkpoint_every, self.groups), "unregistered stop point")

    @property
    def global_start(self):
        return (self.epoch - 1) * self.groups + self.start

    @property
    def global_stop(self):
        return (self.epoch - 1) * self.groups + self.stop

    @property
    def total_schedule(self):
        return self.groups * self.schedule_epochs

    @property
    def evaluation_due(self):
        return self.stop == self.groups

    @property
    def predecessor_point(self):
        require(self.global_start > 0, "fresh start has no predecessor")
        return {"epoch": self.epoch if self.start else self.epoch - 1,
                "step": self.start if self.start else self.groups, "global_group": self.global_start}

    @property
    def point(self):
        return {"epoch": self.epoch, "step": self.stop, "global_group": self.global_stop}

    def slices(self, order, expected_full_order_hash):
        require(type(order) is list and len(order) == self.fit_studies, "full declared fit permutation required")
        for index in order:
            integer(index, "absolute fit index")
        require(set(order) == set(range(self.fit_studies)), "order is not the full fit permutation")
        require(digest(order) == sha(expected_full_order_hash), "full permutation including unused tail drift")
        for start in range(self.start, self.stop, self.recycle):
            end = min(start + self.recycle, self.stop)
            yield start, order[start * self.micro * self.accum:end * self.micro * self.accum]

    def project(self, elapsed, mean_group, consumed, persistence_tail, evaluation_tail=0):
        integer(consumed, "consumed group")
        require(self.start <= consumed <= self.stop, "projection cursor outside session")
        for value in (elapsed, mean_group, persistence_tail, evaluation_tail):
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid timing bound")
        require(mean_group > 0, "missing measured group speed")
        require(self.evaluation_due or evaluation_tail == 0, "partial chunk cannot reserve evaluation")
        return elapsed + (self.stop - consumed) * mean_group + persistence_tail + evaluation_tail


@dataclass
class HealthLedger:
    consumed_groups: int = 0
    optimizer_calls: int = 0
    scheduler_calls: int = 0
    microbatches: int = 0
    studies: int = 0
    build_failures: int = 0
    missing_slots: int = 0
    finite_losses: int = 0
    finite_gradient_groups: int = 0
    pending_micro: int = 0
    loss_sum: float = 0.0
    epoch_loss_sum: float = 0.0
    epoch_loss_count: int = 0
    tainted: bool = False
    order_hash: str = "0" * 64

    def begin_epoch(self):
        require(self.pending_micro == 0 and not self.tainted, "invalid epoch handoff")
        self.epoch_loss_sum, self.epoch_loss_count = 0.0, 0

    def microbatch(self, loss, study_indices, epoch, build_failures=0, missing_slots=0):
        if type(loss) not in (int, float) or not math.isfinite(loss):
            self.tainted = True
            raise Hold("nonfinite loss")
        integer(epoch, "epoch", 1)
        for key, value in (("build_failures", build_failures), ("missing_slots", missing_slots)):
            integer(value, key)
        require(type(study_indices) is list and len(study_indices) > 0, "missing consumed order")
        for index in study_indices:
            integer(index, "study index")
        self.order_hash = digest([self.order_hash, epoch, study_indices])
        self.microbatches += 1
        self.pending_micro += 1
        self.studies += len(study_indices)
        self.build_failures += build_failures
        self.missing_slots += missing_slots
        self.finite_losses += 1
        self.loss_sum += loss
        self.epoch_loss_sum += loss
        self.epoch_loss_count += 1

    def group(self, accum, optimizer_called, scheduler_called, gradients_finite):
        if self.pending_micro != accum or optimizer_called is not True or scheduler_called is not True or gradients_finite is not True:
            self.tainted = True
            raise Hold("incomplete accumulation or unqualified optimizer/scheduler/gradient observation")
        self.consumed_groups += 1
        self.optimizer_calls += 1
        self.scheduler_calls += 1
        self.finite_gradient_groups += 1
        self.pending_micro = 0

    def qualify(self, plan, policy, expected_order_hash, at_start=False):
        require(isinstance(policy, HealthPolicy), "explicit data-health policy missing")
        d = asdict(self)
        for key, value in d.items():
            if key not in ("tainted", "order_hash", "loss_sum", "epoch_loss_sum"):
                integer(value, "health " + key)
        require(self.tainted is False and self.pending_micro == 0, "tainted or partial group")
        point = plan.global_start if at_start else plan.global_stop
        require(self.consumed_groups == self.optimizer_calls == self.scheduler_calls == self.finite_gradient_groups == point,
                "absolute consumed/call/gradient clock mismatch")
        require(self.microbatches == self.finite_losses == point * plan.accum, "incomplete finite-loss evidence")
        epoch_step = (plan.start or plan.groups) if at_start and point else (0 if at_start else plan.stop)
        require(self.epoch_loss_count == epoch_step * plan.accum, "partial epoch diagnostic count mismatch")
        require(self.studies == point * plan.accum * plan.micro, "consumed study count mismatch")
        require(self.build_failures <= min(policy.max_build_failures, self.studies), "build-failure policy failed")
        require(self.missing_slots <= policy.max_missing_slots, "missing-slot policy failed")
        require(math.isfinite(self.loss_sum) and math.isfinite(self.epoch_loss_sum), "nonfinite cumulative diagnostic")
        require(sha(self.order_hash) == sha(expected_order_hash), "consumed order prefix differs from registered full order")
        return {"status": "PASS_DECLARED_HEALTH", "policy_sha256": policy.hash,
                "policy": asdict(policy), "ledger": d, "parameter_change_proven": False}


class OptimizerCallObserver:
    """Observes post-step calls; never equates them with parameter changes."""
    def __init__(self, optimizer, ledger):
        require(isinstance(ledger, HealthLedger), "observer must bind the cumulative health ledger")
        self.ledger = ledger
        self.calls = 0
        self.active = False
        self.tainted = False
        self.hook = optimizer.register_step_post_hook(self._called)

    def _taint(self):
        self.tainted = self.ledger.tainted = True

    def _called(self, optimizer, args, kwargs):
        self.calls += 1
        if not self.active:
            self._taint()

    def step(self, callback):
        if self.tainted or self.ledger.tainted or self.active:
            self._taint()
            raise Hold("tainted or reentrant optimizer observation")
        before = self.calls
        self.active = True
        try:
            callback()
            if self.calls - before != 1 or self.tainted or self.ledger.tainted:
                self._taint()
                raise Hold("optimizer call skipped/unobserved/multiple")
        except BaseException:
            self._taint()
            raise
        finally:
            self.active = False
        return True


def gradients_are_finite(model):
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    return bool(gradients) and all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    require(type(value) in (str, int, float, bool, type(None)), "unsupported state container")
    return value


def capture_state(model, optimizer, scheduler, scaler, cuda=False):
    require(all(p.grad is None for p in model.parameters()), "checkpoint has uncleared gradients")
    return {"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
            "scheduler": cpu_tree(scheduler.state_dict()),
            "scaler": cpu_tree(scaler.state_dict()), "model_training": model.training,
            "rng": {"torch_cpu": torch.get_rng_state(), "python": random.getstate(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if cuda else None}}


def restore_state(state, model, optimizer, scheduler, scaler, cuda=False):
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    model.train(state["model_training"])
    torch.set_rng_state(state["rng"]["torch_cpu"])
    random.setstate(state["rng"]["python"])
    if cuda:
        require(state["rng"]["torch_cuda"] is not None and len(state["rng"]["torch_cuda"]) == torch.cuda.device_count(),
                "CUDA RNG device topology mismatch")
        torch.cuda.set_rng_state_all(state["rng"]["torch_cuda"])
    else:
        require(state["rng"]["torch_cuda"] is None, "refusing to ignore CUDA RNG state")


@dataclass(frozen=True)
class CommitRef:
    checkpoint: str
    checkpoint_sha256: str
    sidecar: str
    sidecar_sha256: str
    predecessor_source_sha256: str
    predecessor_origin: str


class CheckpointStore:
    """Append-only pair store; a fsynced sidecar is the commit marker.

    Caller owns authenticity of origin/source pins and the output-directory
    registration. Orphans and failed writes remain preserved, never repaired,
    selected by glob, replaced, renamed, or deleted.
    """
    def __init__(self, root, max_checkpoint_bytes=256 * 2**20):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        require(type(max_checkpoint_bytes) is int and 0 < max_checkpoint_bytes <= 2**31, "checkpoint size cap")
        self.cap = max_checkpoint_bytes

    def path(self, name, extension):
        require(type(name) is str and re.fullmatch(r"chunk-[A-Za-z0-9_-]+" + re.escape(extension), name) is not None,
                "checkpoint name is not a registered single basename")
        p = self.root / name
        require(not p.is_symlink(), "symlink artifact refused")
        return p

    def sync_dir(self):
        fd = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def save(self, stem, state, plan, ledger, policy, contract, source_sha256, origin, fault=None):
        require(type(stem) is str and re.fullmatch(r"[A-Za-z0-9_-]+", stem) is not None, "invalid output stem")
        sha(source_sha256)
        require(type(origin) is str and bool(origin), "exact declared source origin required")
        validate_contract(contract, plan, policy)
        health = ledger.qualify(plan, policy, contract["consumed_prefix_sha256"][str(plan.global_stop)])
        require(state["scheduler"]["last_epoch"] == plan.global_stop, "scheduler state does not match absolute clock")
        contract_hash = digest(contract)
        pt = self.path("chunk-" + stem + ".pt", ".pt")
        side = self.path("chunk-" + stem + ".json", ".json")
        require(not pt.exists() and not side.exists(), "existing pair/orphan must remain preserved")
        payload = {"schema": "chunk-recovery/1", "point": plan.point, "plan": asdict(plan),
                   "contract": contract, "contract_sha256": contract_hash, "health": health,
                   "source_sha256": source_sha256, "origin": origin, "state": state,
                   "rng_handoff": "train-boundary-before-evaluation"}
        if fault:
            fault("before_checkpoint")
        with open(pt, "xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        self.sync_dir()
        require(pt.stat().st_size <= self.cap, "checkpoint exceeds registered cap")
        if fault:
            fault("after_checkpoint_fsync")
        marker = {key: payload[key] for key in ("schema", "point", "plan", "contract_sha256", "health", "source_sha256", "origin", "rng_handoff")}
        marker.update(checkpoint=pt.name, checkpoint_sha256=file_hash(pt), checkpoint_bytes=pt.stat().st_size)
        with open(side, "xb") as stream:
            stream.write(canonical(marker))
            stream.flush()
            os.fsync(stream.fileno())
        self.sync_dir()
        ref = CommitRef(pt.name, marker["checkpoint_sha256"], side.name, file_hash(side), source_sha256, origin)
        if fault:
            fault("after_sidecar_fsync")
        self.verify_commit(ref, plan, policy, contract)
        return ref

    def load_exact(self, ref, next_plan, policy, contract):
        return self._load(ref, next_plan, policy, contract, next_plan.predecessor_point, True)

    def verify_commit(self, ref, saved_plan, policy, contract):
        return self._load(ref, saved_plan, policy, contract, saved_plan.point, False)

    def _load(self, ref, plan, policy, contract, expected_point, at_start):
        require(isinstance(ref, CommitRef), "missing exact predecessor; no fresh-start fallback")
        require(isinstance(policy, HealthPolicy), "explicit health policy required")
        validate_contract(contract, plan, policy)
        for value in (ref.checkpoint_sha256, ref.sidecar_sha256, ref.predecessor_source_sha256):
            sha(value)
        pt, side = self.path(ref.checkpoint, ".pt"), self.path(ref.sidecar, ".json")
        require(pt.is_file() and side.is_file(), "partial/missing checkpoint pair")
        require(0 < pt.stat().st_size <= self.cap and 0 < side.stat().st_size <= 1 << 20, "artifact size bound")
        require(file_hash(pt) == ref.checkpoint_sha256 and file_hash(side) == ref.sidecar_sha256, "exact predecessor hash mismatch")
        marker = json.loads(side.read_bytes())
        require(marker["checkpoint"] == ref.checkpoint and marker["checkpoint_sha256"] == ref.checkpoint_sha256,
                "sidecar checkpoint binding mismatch")
        require(marker["checkpoint_bytes"] == pt.stat().st_size, "sidecar byte count mismatch")
        require(marker["schema"] == "chunk-recovery/1", "unsupported sidecar schema")
        for key in ("epoch", "step", "global_group"):
            integer(marker["point"][key], "predecessor " + key, 1)
        saved_plan = ChunkPlan(**marker["plan"])
        validate_contract(contract, saved_plan, policy)
        require(marker["point"] == saved_plan.point == expected_point, "wrong/ambiguous predecessor cursor")
        require(marker["source_sha256"] == ref.predecessor_source_sha256 and marker["origin"] == ref.predecessor_origin,
                "predecessor source/origin mismatch")
        require(marker["contract_sha256"] == digest(contract) and marker["health"]["policy_sha256"] == policy.hash,
                "training/data-health contract drift")
        payload = torch.load(pt, map_location="cpu", weights_only=True)
        require(payload["schema"] == "chunk-recovery/1" and payload["rng_handoff"] == "train-boundary-before-evaluation",
                "unsupported checkpoint/handoff schema")
        for key in ("point", "plan", "contract_sha256", "health", "source_sha256", "origin", "rng_handoff"):
            require(payload[key] == marker[key], "checkpoint/sidecar semantic mismatch: " + key)
        require(payload["contract"] == contract, "checkpoint training contract mismatch")
        ledger = HealthLedger(**payload["health"]["ledger"])
        ledger.qualify(plan, policy, contract["consumed_prefix_sha256"][str(expected_point["global_group"])], at_start=at_start)
        require(payload["state"]["scheduler"]["last_epoch"] == expected_point["global_group"], "resumed scheduler clock mismatch")
        return payload, ledger


def validate_contract(contract, plan, policy):
    require(isinstance(contract, dict) and contract.get("schema") == "chunk-contract/1", "explicit training contract required")
    geometry = {key: value for key, value in asdict(plan).items() if key not in ("epoch", "start", "stop")}
    require(contract.get("geometry") == geometry, "training geometry/schedule horizon drift")
    require(contract.get("health_policy_sha256") == policy.hash, "health policy contract drift")
    require(contract.get("rng_handoff") == "train-boundary-before-evaluation", "RNG handoff contract drift")
    for key in ("invariants_sha256", "initialization_sha256", "source_data_contract_sha256",
                "augmentation_contract_sha256", "worker_contract_sha256"):
        sha(contract.get(key))
    orders = contract.get("epoch_order_sha256")
    require(isinstance(orders, dict) and set(orders) == {str(epoch) for epoch in range(1, plan.train_epochs + 1)},
            "full ordered epoch bindings required")
    for value in orders.values():
        sha(value)
    prefixes = contract.get("consumed_prefix_sha256")
    expected_points = {"0"}
    for epoch in range(1, plan.train_epochs + 1):
        expected_points.update(str((epoch - 1) * plan.groups + stop)
                               for stop in list(range(plan.checkpoint_every, plan.groups, plan.checkpoint_every)) + [plan.groups])
    require(isinstance(prefixes, dict) and set(prefixes) == expected_points, "registered consumed order prefix pins required")
    for value in prefixes.values():
        sha(value)
    digest(contract)


def require_initialization(plan, mode, ref):
    if mode == "START_FROM_INITIAL_STATE":
        require(plan.epoch == 1 and plan.start == 0 and ref is None, "fresh mode cannot bypass continuation")
    else:
        require(mode == "REQUIRE_PREDECESSOR" and isinstance(ref, CommitRef), "explicit exact predecessor required")
