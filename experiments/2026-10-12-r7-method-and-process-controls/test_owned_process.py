# SPDX-License-Identifier: MIT
"""Two local synthetic controls; no Kaggle, model, network, or GPU operation."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import owned_process


FIXTURE = r'''
import json, os, resource, signal, subprocess, sys, time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
mode, output = sys.argv[1:]
ready = output + '.child-ready'
child_source = r"""
import os, resource, signal, sys, time
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1], 'x') as f:
    f.write(str(os.getpid()))
time.sleep(60)
"""
child = subprocess.Popen([sys.executable, '-c', child_source, ready])
until = time.monotonic() + 2
while not Path(ready).exists():
    if time.monotonic() >= until:
        raise RuntimeError('synthetic child readiness expired')
    time.sleep(.01)
with open(output, 'x') as f:
    json.dump({'leader_pid': os.getpid(), 'child_pid': child.pid,
               'leader_pgid': os.getpgrp(), 'mode': mode}, f)
print('synthetic leader and TERM-resistant child ready', flush=True)
if mode == 'normal_exit':
    sys.exit(0)
time.sleep(60)
'''


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(mode, directory, fixture):
    pid_path = directory / (mode + '.pids.json')
    log_path = directory / (mode + '.log')
    started = time.monotonic()
    deadline = started + 3.0
    result = owned_process.run_owned(
        [sys.executable, str(fixture), mode, str(pid_path)],
        hard_deadline=deadline, log_path=log_path,
        cleanup_reserve=2.0, term_grace=.4)
    ended = time.monotonic()
    with (directory / (mode + '.process.json')).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    require(pid_path.exists(), mode + ': fixture did not establish two processes')
    fixture_pids = json.loads(pid_path.read_text())
    require(fixture_pids['leader_pid'] == result['pid'], mode + ': leader differs')
    require(fixture_pids['leader_pgid'] == result['pgid'], mode + ': owned group differs')
    child_pid = fixture_pids['child_pid']
    require(child_pid in result['live_group_members_observed'], mode + ': live descendant never observed')
    require('SIGKILL' in result['signals'], mode + ': TERM-resistant control was not closed by KILL')
    require(result['cleanup_proven'] and result['leader_reaped'], mode + ': closure/reaping unproven')
    require(result['live_group_members_remaining'] == [], mode + ': helper reports live members')
    require(result['log_closed'] and not result['reader_thread_used'], mode + ': logging incomplete')
    require(not result['hard_deadline_exceeded'] and ended < deadline, mode + ': global budget exceeded')
    require(result['error'] is None, mode + ': process evidence failed')
    # A fresh independent process-table observation after return establishes that
    # neither fixture process remains live; the helper leader must be reaped.
    table = owned_process.proc_table(time.monotonic() + .8)
    after = [{'pid': pid, 'state': state, 'pgid': pgid}
             for pid, state, pgid in table
             if pid in (result['pid'], child_pid)]
    require(not any(row['pid'] == result['pid'] for row in after), mode + ': leader not reaped')
    require(not any(row['state'] not in ('Z', 'X') for row in after), mode + ': fixture remains live')
    if mode == 'normal_exit':
        require(result['returncode'] == 0 and not result['timed_out'], 'normal control lost exit evidence')
        require(result['status'] == 'PASS_PROCESS_GROUP_CLOSED', 'normal control did not pass')
    else:
        require(result['timed_out'] and result['returncode'] == -9, 'timeout control lost timeout evidence')
        require(result['status'] == 'HOLD_EXECUTION_TIMEOUT_GROUP_CLOSED', 'timeout control falsely accepted execution')
    return {'mode': mode, 'process_receipt': result, 'fixture_pids': fixture_pids,
            'independent_after_return_observation': after,
            'caller_elapsed_seconds': ended - started, 'budget_seconds': 3.0,
            'log_sha256': digest(log_path), 'control_passed': True}


def main():
    base = Path(__file__).resolve().parent
    stamp = str(time.time_ns())
    output = base / ('synthetic_controls_' + stamp)
    output.mkdir()
    fixture = output / 'synthetic_fixture.py'
    with fixture.open('x') as stream:
        stream.write(FIXTURE)
    cases = [run_case(mode, output, fixture) for mode in ('normal_exit', 'timeout')]
    require(not owned_process._UNREAPED, 'retained workload leader after successful controls')
    require(not owned_process._UNREAPED_PROBES, 'retained diagnostic process after controls')
    receipt = {'status': 'PASS_TWO_SYNTHETIC_PROCESS_CONTROLS',
               'qualification_scope': 'Local POSIX process closure only; no R7 science, checkpoint, GPU or Kaggle qualification',
               'python': sys.version, 'platform': platform.platform(),
               'source_sha256': {'owned_process.py': digest(base / 'owned_process.py'),
                                 'test_owned_process.py': digest(Path(__file__)),
                                 'synthetic_fixture.py': digest(fixture)},
               'cases': cases}
    receipt_path = output / 'PROCESS_CONTROL_RECEIPT.json'
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'receipt_path': str(receipt_path),
                      'receipt_sha256': digest(receipt_path),
                      'elapsed_seconds': [case['caller_elapsed_seconds'] for case in cases]}))


if __name__ == '__main__':
    main()
