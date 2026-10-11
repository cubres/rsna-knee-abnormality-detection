"""Small, real PyTorch CPU GradScaler/AdamW probe; no model/checkpoint files.

An intentional infinity induces one skipped optimizer call. This verifies the
public hook on this CPU build, not CUDA/DataParallel behavior or Knee outcomes.
"""
import json
import math

import torch

from optimizer_call_witness import OptimizerCallWitness


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(701)
    p = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
    opt = torch.optim.AdamW([p], lr=0.1, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=100)
    witness = OptimizerCallWitness(opt)
    rows = []
    for loss_factor in (1.0, float("inf"), 1.0):
        before = p.detach().clone()
        old_lr = opt.param_groups[0]["lr"]
        opt.zero_grad(set_to_none=True)
        loss = p.square().sum() * loss_factor
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        outcome = witness.attempt(lambda: scaler.step(opt), schedule.step)
        scaler.update()
        changed = bool(torch.any(p.detach() != before))
        rows.append({"infinite_gradient_fixture": not math.isfinite(loss_factor),
                     "outcome": outcome, "parameter_changed": changed,
                     "lr_before": old_lr, "lr_after": opt.param_groups[0]["lr"]})
    require(rows[0]["parameter_changed"] and rows[2]["parameter_changed"], "finite calls did not change the toy parameter")
    require(not rows[1]["parameter_changed"], "overflow changed the toy parameter")
    require(rows[1]["lr_before"] == rows[1]["lr_after"], "overflow advanced the scheduler")
    s = witness.snapshot()
    require(s["attempts"] == 3 and s["completed_optimizer_calls"] == 2 and s["skipped_optimizer_calls"] == 1,
            "unexpected hook accounting")
    require(s["scheduler_calls"] == 2 and not s["tainted"], "unexpected scheduler accounting")
    # Reproduce the source's unconditional-scheduler pattern on another toy.
    p_naive = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float32))
    opt_naive = torch.optim.AdamW([p_naive], lr=0.1, weight_decay=0.0)
    schedule_naive = torch.optim.lr_scheduler.StepLR(opt_naive, step_size=1, gamma=0.5)
    scaler_naive = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=100)
    naive_calls = [0]
    def naive_post_hook(optimizer, args, kwargs):
        naive_calls[0] += 1
    opt_naive.register_step_post_hook(naive_post_hook)
    naive_rows = []
    for loss_factor in (1.0, float("inf"), 1.0):
        before = p_naive.detach().clone()
        old_lr = opt_naive.param_groups[0]["lr"]
        opt_naive.zero_grad(set_to_none=True)
        scaler_naive.scale(p_naive.square().sum() * loss_factor).backward()
        scaler_naive.unscale_(opt_naive)
        scaler_naive.step(opt_naive)
        scaler_naive.update()
        schedule_naive.step()
        naive_rows.append({"infinite_gradient_fixture": not math.isfinite(loss_factor),
                           "parameter_changed": bool(torch.any(p_naive.detach() != before)),
                           "lr_before": old_lr, "lr_after": opt_naive.param_groups[0]["lr"]})
    require(naive_calls[0] == 2 and schedule_naive.last_epoch == 3, "naive mismatch was not reproduced")
    require(not naive_rows[1]["parameter_changed"] and naive_rows[1]["lr_after"] == 0.025,
            "naive overflow fixture did not advance its scheduler")
    print(json.dumps({"status": "PASS_CPU_API_PROBE", "torch_version": torch.__version__,
                      "device": "cpu", "rows": rows, "witness": s,
                      "unconditional_scheduler_fixture": {"rows": naive_rows,
                          "completed_optimizer_calls": naive_calls[0],
                          "scheduler_calls": schedule_naive.last_epoch},
                      "native_knee_execution": False, "cuda_compatibility_proven": False,
                      "official_score": None}, indent=2))


if __name__ == "__main__":
    main()
