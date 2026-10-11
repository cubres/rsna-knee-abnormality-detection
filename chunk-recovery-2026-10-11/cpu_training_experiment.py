"""Real bounded CPU training, exact continuation and process-crash experiment.

Synthetic data only; two434-group epochs, four2-study microbatches per group,
absolute50-group DataLoader generations, CPU GradScaler, dropout and random
input noise. No CUDA, DataParallel, multiprocessing workers or OOF execution.
Artifacts are create-only and retained. Run only in a fresh output directory.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch

from chunk_recovery import (CheckpointStore, ChunkPlan, CommitRef, HealthLedger, HealthPolicy, Hold,
                            OptimizerCallObserver, capture_state, digest, file_hash,
                            gradients_are_finite, require, require_initialization, restore_state)


SEED = 701
SOURCE_SHA = file_hash(Path(__file__))
POLICY = HealthPolicy(0, 1000, "synthetic-fixture-only-explicit-allowance")


def full_order(epoch):
    return torch.randperm(3479, generator=torch.Generator().manual_seed(SEED + epoch)).tolist()


def contract():
    prototype = ChunkPlan(1, 0, 100)
    geometry = {k: v for k, v in asdict(prototype).items() if k not in ("epoch", "start", "stop")}
    h, prefix, epoch_hashes = "0" * 64, {"0": "0" * 64}, {}
    for epoch in (1, 2):
        order = full_order(epoch)
        epoch_hashes[str(epoch)] = digest(order)
        for group in range(434):
            for micro in range(4):
                start = group * 8 + micro * 2
                h = digest([h, epoch, order[start:start + 2]])
            step = group + 1
            if step % 100 == 0 or step == 434:
                prefix[str((epoch - 1) * 434 + step)] = h
    invariants = {"seed": SEED, "model": "8-16-dropout0.25-2", "dtype": "float32", "device": "cpu",
                  "AdamW": {"lr": 0.001, "weight_decay": 0.02}, "cosine_floor": 0.1,
                  "scheduler_total": 1302, "grad_clip": 3.0,
                  "cpu_scaler": {"init_scale": 8.0, "growth_interval": 80},
                  "data": "3479 synthetic studies; stateless per-epoch/index torch augmentation",
                  "augmentation": "seed701+epoch*1000003+absolute_index",
                  "loader": "workers0; generator701+epoch*100003+absolute_chunk_start; recycle50",
                  "evaluation_coverage": "none; pre-evaluation handoff scoped explicitly",
                  "DataParallel": False, "CUDA_topology": []}
    return {"schema": "chunk-contract/1", "geometry": geometry, "health_policy_sha256": POLICY.hash,
            "rng_handoff": "train-boundary-before-evaluation", "invariants_sha256": digest(invariants),
            "initialization_sha256": digest({"seed": SEED, "model": invariants["model"]}),
            "source_data_contract_sha256": digest(invariants["data"]),
            "augmentation_contract_sha256": digest(invariants["augmentation"]),
            "worker_contract_sha256": digest(invariants["loader"]),
            "epoch_order_sha256": epoch_hashes, "consumed_prefix_sha256": prefix, "declarations": invariants}


class Studies(torch.utils.data.Dataset):
    def __init__(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 3479

    def __getitem__(self, index):
        gen = torch.Generator().manual_seed(SEED + self.epoch * 1000003 + index)
        x = torch.sin(torch.arange(8, dtype=torch.float32) * 0.2 + index * 0.01)
        x = x + torch.randn(8, generator=gen) * 0.01
        y = torch.stack((x[0] * 0.4 + x[1] * 0.2, x[2] * 0.3 - x[3] * 0.1))
        return x, y, index, int(index % 17 == 0)


def build(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Dropout(0.25), torch.nn.Linear(16, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, 1302) / 1302)))
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=80)
    return model, optimizer, scheduler, scaler


def train_chunk(objects, ledger, observer, plan, declared, order_log, generation_log):
    model, optimizer, scheduler, scaler = objects
    if plan.start == 0:
        ledger.begin_epoch()
    order = full_order(plan.epoch)
    require(digest(order) == declared["epoch_order_sha256"][str(plan.epoch)], "full epoch order drift")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for absolute_start, studies in plan.slices(order, declared["epoch_order_sha256"][str(plan.epoch)]):
        loader_seed = SEED + plan.epoch * 100003 + absolute_start
        loader = torch.utils.data.DataLoader(Studies(plan.epoch), batch_size=2, sampler=studies,
                                            drop_last=True, num_workers=0,
                                            generator=torch.Generator().manual_seed(loader_seed))
        generation_log.append([plan.epoch, absolute_start, loader_seed, len(studies)])
        for x, y, indices, missing in loader:
            if random.random() < 0.5:
                x = x + torch.randn_like(x) * 0.001
            loss = torch.nn.functional.mse_loss(model(x), y)
            ids = indices.tolist()
            ledger.microbatch(float(loss.detach()), ids, plan.epoch, missing_slots=int(missing.sum()))
            order_log.extend([plan.epoch, index] for index in ids)
            scaler.scale(loss / 4).backward()
            if ledger.pending_micro == 4:
                scaler.unscale_(optimizer)
                finite = gradients_are_finite(model)
                require(finite, "nonfinite synthetic gradient")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                require(bool(torch.isfinite(norm)) and gradients_are_finite(model), "clip sequence nonfinite")
                observed = observer.step(lambda: scaler.step(optimizer))
                scaler.update()
                before = scheduler.last_epoch
                try:
                    scheduler.step()
                except BaseException:
                    ledger.tainted = True
                    raise
                ledger.group(4, observed, scheduler.last_epoch == before + 1, finite)
                optimizer.zero_grad(set_to_none=True)
        del loader
    ledger.qualify(plan, POLICY, declared["consumed_prefix_sha256"][str(plan.global_stop)])


def same_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def run_training(folder, resumed):
    declared = contract()
    ledger, refs, order_log, generation_log = HealthLedger(), [], [], []
    objects = build(SEED)
    observer = OptimizerCallObserver(objects[1], ledger)
    store = CheckpointStore(folder)
    for epoch in (1, 2):
        for start, stop in ((0, 100), (100, 200), (200, 300), (300, 400), (400, 434)):
            plan = ChunkPlan(epoch, start, stop)
            if resumed:
                ref = refs[-1] if refs else None
                require_initialization(plan, "REQUIRE_PREDECESSOR" if ref else "START_FROM_INITIAL_STATE", ref)
                if ref:
                    # Deliberately perturb model initialization and both global
                    # RNGs. A missing restore cannot accidentally pass equality.
                    objects = build(9000 + plan.global_start)
                    torch.rand(13); random.random()
                    payload, ledger = store.load_exact(ref, plan, POLICY, declared)
                    restore_state(payload["state"], *objects)
                    observer = OptimizerCallObserver(objects[1], ledger)
            train_chunk(objects, ledger, observer, plan, declared, order_log, generation_log)
            if resumed:
                ref = store.save("e%d-g%d" % (epoch, stop), capture_state(*objects), plan, ledger,
                                 POLICY, declared, SOURCE_SHA, "cpu-toy-source-v1")
                refs.append(ref)
    final = capture_state(*objects)
    if resumed:
        final_payload, _ = store.verify_commit(refs[-1], ChunkPlan(2, 400, 434), POLICY, declared)
        require(same_tree(final_payload["state"], final), "final persisted state changed")
    return final, asdict(ledger), order_log, generation_log, refs


def fault_worker(spec_path, stage):
    spec = json.loads(Path(spec_path).read_text())
    ref = CommitRef(**spec["ref"])
    payload, ledger = CheckpointStore(spec["source_store"]).load_exact(ref, ChunkPlan(1, 100, 200), POLICY, contract())
    target = CheckpointStore(Path(spec["fault_root"]) / stage)
    def crash(where):
        if where == stage:
            os._exit(83)
    target.save(stage, payload["state"], ChunkPlan(1, 0, 100), ledger, POLICY, contract(),
                SOURCE_SHA, "cpu-toy-source-v1", fault=crash)
    raise RuntimeError("crash stage not reached")


def process_crashes(folder, first_ref):
    spec = {"ref": asdict(first_ref), "source_store": str(folder / "chunked"), "fault_root": str(folder / "crashes")}
    spec_path = folder / "FAULT_INPUT.json"
    with open(spec_path, "x") as stream:
        json.dump(spec, stream, indent=2)
    rows = []
    for stage in ("before_checkpoint", "after_checkpoint_fsync", "after_sidecar_fsync"):
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--fault-worker", str(spec_path), "--stage", stage],
                                   capture_output=True, text=True, timeout=30)
        require(completed.returncode == 83, "unexpected crash-worker outcome: " + completed.stderr[-300:])
        store = CheckpointStore(folder / "crashes" / stage)
        pt, side = "chunk-" + stage + ".pt", "chunk-" + stage + ".json"
        files = sorted(p.name for p in store.root.iterdir())
        if stage == "after_sidecar_fsync":
            ref = CommitRef(pt, file_hash(store.root / pt), side, file_hash(store.root / side), SOURCE_SHA, "cpu-toy-source-v1")
            payload, _ = store.load_exact(ref, ChunkPlan(1, 100, 200), POLICY, contract())
            origin, _ = CheckpointStore(folder / "chunked").load_exact(first_ref, ChunkPlan(1, 100, 200), POLICY, contract())
            require(same_tree(payload["state"], origin["state"]), "post-commit crash lost state")
            outcome = "EXACT_COMMITTED_PAIR_RECOVERABLE"
        else:
            ref = CommitRef(pt, "0" * 64, side, "0" * 64, SOURCE_SHA, "cpu-toy-source-v1")
            try:
                store.load_exact(ref, ChunkPlan(1, 100, 200), POLICY, contract())
            except Hold:
                outcome = "HOLD_PARTIAL_OR_MISSING_PAIR"
            else:
                raise RuntimeError("partial pair admitted")
        rows.append({"stage": stage, "exit_code": 83, "retained_files": files, "outcome": outcome})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--fault-worker")
    parser.add_argument("--stage")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.fault_worker:
        fault_worker(args.fault_worker, args.stage)
    require(args.output is not None, "fresh output directory required")
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.monotonic(), time.process_time()
    a = run_training(folder / "uninterrupted", False)
    b = run_training(folder / "chunked", True)
    comparisons = {key: same_tree(a[0][key], b[0][key]) for key in a[0]}
    comparisons.update(health_ledger=a[1] == b[1], consumed_order=a[2] == b[2], loader_generations=a[3] == b[3])
    require(all(comparisons.values()), "CPU state-continuation equivalence failed")
    crashes = process_crashes(folder, b[4][0])
    receipt = {"status": "PASS_CPU_TRAINING_AND_PROCESS_CRASH_WITNESS", "torch": torch.__version__,
               "device": "cpu", "dtype": "float32", "threads": 1, "DataParallel": False, "workers": 0,
               "epochs": 2, "groups_per_epoch": 434, "schedule_total_groups": 1302,
               "completed_groups": b[1]["consumed_groups"], "microbatches": b[1]["microbatches"],
               "bitwise_cpu_comparisons": comparisons, "final_health": b[1],
               "loader_generation_count_per_path": len(b[3]), "cpu_s_main_process": time.process_time() - cpu,
               "wall_s_including_crash_workers": time.monotonic() - wall, "process_crash_controls": crashes,
               "commits": [asdict(ref) for ref in b[4]], "contract": contract(),
               "evaluation_execution": False, "native_knee_execution": False, "CUDA_equivalence_proven": False,
               "Knee_health_allowance_supplied": False, "official_score": None,
               "source_sha256": SOURCE_SHA, "library_sha256": file_hash(Path(__file__).with_name("chunk_recovery.py"))}
    with open(folder / "CPU_RECEIPT.json", "x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("commits", "contract", "final_health")}, indent=2))


if __name__ == "__main__":
    main()
