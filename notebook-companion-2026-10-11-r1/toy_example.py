# SPDX-License-Identifier: MIT
"""Original invented example; no network, model, candidate edit or submission."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

EXPECTED_TOOL_SHA256 = '8a890ea67cd8382d31d7b2a98d78ecf45b6604cc9b1eb284ba769737b1a5eef3'

def load_tool(path):
    # Obtain this original helper from the separately recorded pinned commit.
    # A digest checks consistency; it does not authenticate an untrusted pin.
    with Path(path).open("rb") as f:
        raw = f.read(1_048_577)
    if len(raw) > 1_048_576 or hashlib.sha256(raw).hexdigest() != EXPECTED_TOOL_SHA256:
        raise ValueError("PINNED_TOOL_MISMATCH")
    module = types.ModuleType("companion_pinned_tool")
    module.__file__ = str(Path(path))
    sys.modules[module.__name__] = module
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    return module

def run(tool):
    class ToyOptimizer:
        # Simulates the synchronous public hook protocol. No tensor or weight.
        def __init__(self):
            self.hooks = []
        def register_step_post_hook(self, hook):
            self.hooks.append(hook)
            return object()
        def step(self):
            for hook in self.hooks:
                hook(self, (), {})

    optimizer = ToyOptimizer()
    scheduled = []
    witness = tool.OptimizerCallWitness(optimizer)
    outcomes = [
        witness.attempt(optimizer.step, lambda: scheduled.append("tick")),
        witness.attempt(lambda: None, lambda: scheduled.append("tick")),
        witness.attempt(optimizer.step, lambda: scheduled.append("tick")),
    ]
    evidence = witness.snapshot()
    if (evidence["attempts"], evidence["completed_optimizer_calls"],
        evidence["skipped_optimizer_calls"], evidence["scheduler_calls"]) != (3, 2, 1, 2):
        raise ValueError("INVENTED_EXPECTATION_MISMATCH")
    if len(scheduled) != 2 or evidence["parameter_change_proven"] is not False:
        raise ValueError("INVENTED_EXPECTATION_MISMATCH")
    return {"status": "INVENTED_HOOK_PROTOCOL_ONLY", "outcomes": outcomes,
            "witness": evidence, "tensor_or_parameter_created": False,
            "native_pytorch_or_cuda_test": False, "official_score": None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", required=True, help="Path to the pinned original helper")
    args = parser.parse_args()
    try:
        result = run(load_tool(args.tool))
    except (OSError, ValueError) as exc:
        # No input path, content, credentials or traceback is printed on HOLD.
        print(json.dumps({"status": "HOLD", "reason": "SOURCE_OR_EXAMPLE_CHECK_FAILED"}))
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
