"""Run the original synthetic CPU evidence pipeline with process bounds.

POSIX only. Each selected worker is nice+5, CPU-capped90s and wall-capped60s;
the training worker's crash subprocesses have30s wall caps. Fresh output only.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run_owned_tree(argv, wall_cap):
    """Own only the new local process group created by this call."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=wall_cap)
        return {"status": "PASS" if proc.returncode == 0 else "FAIL", "exit_code": proc.returncode,
                "stdout": stdout, "stderr": stderr, "owned_group": proc.pid}
    except subprocess.TimeoutExpired:
        # The process group is this exact newly created CPU worker tree. No
        # external PID, notebook, Kaggle session or preexisting job is accepted.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate(timeout=2)
        return {"status": "HOLD_WALL_TIMEOUT", "exit_code": proc.returncode,
                "stdout": stdout, "stderr": stderr, "owned_group": proc.pid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    folder = Path(__file__).resolve().parent
    commands = [
        ("training", "cpu_training_experiment.py", ["--output", str(root / "training")], False),
        ("fresh_suffix", "cpu_suffix_process.py", ["--reference", str(root / "training"), "--output", str(root / "CPU_SUFFIX_RECEIPT.json")], False),
        ("controls_normal", "test_chunk_recovery.py", ["--fixture", str(root / "training"), "--output", str(root / "controls_normal")], False),
        ("controls_optimized", "test_chunk_recovery.py", ["--fixture", str(root / "training"), "--output", str(root / "controls_optimized")], True),
    ]
    bootstrap = ("import os,resource,runpy,sys;os.nice(5);resource.setrlimit(resource.RLIMIT_CPU,(90,90));"
                 "script=sys.argv[1];sys.path.insert(0,os.path.dirname(script));sys.argv=sys.argv[1:];runpy.run_path(script,run_name='__main__')")
    rows = []
    for name, script, extra, optimized in commands:
        argv = [sys.executable, "-B"] + (["-O"] if optimized else []) + ["-c", bootstrap, str(folder / script)] + extra
        started = time.monotonic()
        result = run_owned_tree(argv, 60)
        status, code, stdout, stderr = (result[key] for key in ("status", "exit_code", "stdout", "stderr"))
        with open(root / (name + "_stdout.txt"), "x") as stream:
            stream.write(stdout)
        with open(root / (name + "_stderr.txt"), "x") as stream:
            stream.write(stderr)
        rows.append({"worker": name, "status": status, "exit_code": code,
                     "wall_s": time.monotonic() - started, "nice_increment": 5,
                     "cpu_cap_s": 90, "wall_cap_s": 60})
        if status != "PASS":
            break
    receipt = {"status": "PASS_BOUNDED_CPU_PIPELINE" if len(rows) == 4 and all(row["status"] == "PASS" for row in rows)
               else "HOLD_BOUNDED_CPU_PIPELINE", "workers": rows, "native_execution": False,
               "all_failed_or_partial_artifacts_preserved": True}
    with open(root / "BOUNDED_PIPELINE_RECEIPT.json", "x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if receipt["status"] == "PASS_BOUNDED_CPU_PIPELINE" else 1)


if __name__ == "__main__":
    main()
