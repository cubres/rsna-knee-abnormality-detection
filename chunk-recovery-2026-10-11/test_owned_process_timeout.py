"""Benign actual parent+child timeout control; owns only its newly spawned tree."""
import argparse
import json
from pathlib import Path
import sys
import time

import psutil

from chunk_recovery import file_hash, require
from run_cpu_witness import run_owned_tree


def active(pid):
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    code = ("import resource,subprocess,sys,time;resource.setrlimit(resource.RLIMIT_CPU,(2,2));"
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            "print(child.pid,flush=True);time.sleep(60)")
    started = time.monotonic()
    result = run_owned_tree([sys.executable, "-B", "-c", code], 1)
    child_pid = int(result["stdout"].strip())
    deadline = time.monotonic() + 3
    while active(child_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    require(result["status"] == "HOLD_WALL_TIMEOUT" and not active(result["owned_group"]) and not active(child_pid),
            "owned parent/child remains active")
    receipt = {"status": "PASS_BENIGN_OWNED_TREE_TIMEOUT", "parent_and_child_inactive": True,
               "wall_cap_s": 1, "termination_grace_s": 2, "cpu_cap_s_worker": 2,
               "actual_wall_s": time.monotonic() - started, "worker_exit_code": result["exit_code"],
               "only_new_owned_local_cpu_tree_targeted": True, "native_execution": False,
               "runner_sha256": file_hash(Path(__file__).with_name("run_cpu_witness.py")),
               "control_source_sha256": file_hash(Path(__file__))}
    with open(args.output, "x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
