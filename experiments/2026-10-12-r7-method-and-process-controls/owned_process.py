# SPDX-License-Identifier: MIT
"""Close only a newly owned POSIX process group within one monotonic budget.

Derived from the MIT ownership pattern in the independently reviewed Enveda
source at codex_enveda_compact_smoke_r1_20261012T194000Z/owned_process.py.
The unreaped session leader pins the group until live-member closure is proven.
This contract covers processes remaining in that group, not processes that
deliberately create another session. Failed process evidence yields HOLD.
"""
import os
import math
from pathlib import Path
import signal
import subprocess
import time

_UNREAPED = []
_UNREAPED_PROBES = []


class EvidenceError(RuntimeError):
    pass


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise EvidenceError('MONOTONIC_BUDGET_EXPIRED')
    return value


def pause(deadline, interval=0.03):
    time.sleep(min(interval, remaining(deadline)))


def _ps_table(deadline, cleanup_deadline=None):
    # Diagnostic child ownership is distinct from the workload group. All
    # communicate/wait operations have a finite remaining-budget timeout.
    probe = subprocess.Popen(['/bin/ps', '-A', '-o', 'pid=,pgid=,stat='],
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
    _UNREAPED_PROBES.append(probe)
    try:
        out, err = probe.communicate(timeout=min(0.5, remaining(deadline)))
    except BaseException as exc:
        try:
            probe.kill()  # only the exact diagnostic child created above
        except ProcessLookupError:
            pass  # exact child has exited; bounded communicate still reaps it
        try:
            probe.communicate(timeout=remaining(cleanup_deadline if cleanup_deadline is not None else deadline))
        except (subprocess.TimeoutExpired, EvidenceError):
            raise EvidenceError('PROCESS_TABLE_PROBE_CLEANUP_HOLD')
        else:
            _UNREAPED_PROBES.remove(probe)
        if time.monotonic() >= deadline and isinstance(exc, (subprocess.TimeoutExpired, EvidenceError)):
            raise EvidenceError('EXECUTION_OBSERVATION_DEADLINE') from exc
        raise EvidenceError('PROCESS_TABLE_PROBE_FAILED') from exc
    _UNREAPED_PROBES.remove(probe)
    if probe.returncode != 0 or err.strip() or not out.strip():
        raise EvidenceError('PROCESS_TABLE_PROBE_INVALID')
    table = []
    for line in out.decode('ascii', errors='strict').splitlines():
        fields = line.split()
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit() or not fields[2]:
            raise EvidenceError('PROCESS_TABLE_ROW_INVALID')
        table.append((int(fields[0]), fields[2][0], int(fields[1])))
    return table


def proc_table(deadline, *, cleanup_deadline=None):
    remaining(deadline)
    if not Path('/proc').is_dir():
        return _ps_table(deadline, cleanup_deadline)
    table = []
    with os.scandir('/proc') as entries:
        for entry in entries:
            remaining(deadline)
            if not entry.name.isdigit():
                continue
            try:
                text = (Path(entry.path) / 'stat').read_text()
            except FileNotFoundError:
                continue  # unrelated process disappeared during the snapshot
            except OSError as exc:
                raise EvidenceError('PROCESS_TABLE_READ_FAILED') from exc
            fields = text[text.rfind(')') + 2:].split()
            if len(fields) < 3 or not fields[2].isdigit() or not fields[0]:
                raise EvidenceError('PROCESS_TABLE_STAT_INVALID')
            table.append((int(entry.name), fields[0], int(fields[2])))
    return table


def snapshot(pid, pgid, deadline, *, cleanup_deadline=None):
    table = proc_table(deadline, cleanup_deadline=cleanup_deadline)
    leaders = [(state, group) for p, state, group in table if p == pid]
    if len(leaders) != 1 or leaders[0][1] != pgid:
        raise EvidenceError('UNREAPED_LEADER_NOT_PROVEN')
    return leaders[0][0] in ('Z', 'X'), [p for p, state, group in table if group == pgid and state not in ('Z', 'X')]


def run_owned(command, *, hard_deadline, env=None, cwd=None, log_path=None,
              cleanup_reserve=2.0, term_grace=0.4):
    """Return separate execution/closure evidence; never reap before closure.

    hard_deadline is an absolute time.monotonic deadline. The caller starts it
    before staging. Cleanup is reserved inside it, including final reaping.
    Optional logs go directly to an exclusive file; no pipe-reader thread exists.
    """
    started = time.monotonic()
    for name, value in (('hard_deadline', hard_deadline), ('cleanup_reserve', cleanup_reserve), ('term_grace', term_grace)):
        try:
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except (OverflowError, TypeError):
            valid = False
        if not valid:
            raise ValueError(name + ' must be a finite number')
    if cleanup_reserve <= 0:
        raise ValueError('cleanup_reserve must be positive')
    if not 0 <= term_grace < cleanup_reserve:
        raise ValueError('term_grace must be nonnegative and less than cleanup_reserve')
    if remaining(hard_deadline) <= cleanup_reserve + 0.05:
        raise EvidenceError('NO_EXECUTION_AND_CLEANUP_BUDGET')
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise EvidenceError('EXCLUSIVE_CHILD_REAP_OWNERSHIP_REQUIRED')
    log = Path(log_path).open('xb', buffering=0) if log_path is not None else None
    try:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT if log else None,
                                start_new_session=True)
    except BaseException:
        if log:
            log.close()
        raise
    _UNREAPED.append(proc)
    pgid = proc.pid
    compute_deadline = hard_deadline - cleanup_reserve
    timed_out, leader_done, closure, code, error = False, False, False, None, None
    live, sent, observed = None, [], set()
    try:
        if pgid == os.getpgrp():
            raise EvidenceError('NEW_OWNED_SESSION_NOT_PROVEN')
        while True:
            if time.monotonic() >= compute_deadline:
                timed_out = True
                break
            leader_done, live = snapshot(proc.pid, pgid, compute_deadline,
                                         cleanup_deadline=hard_deadline)
            observed.update(live)
            if leader_done:
                break
            if time.monotonic() >= compute_deadline:
                timed_out = True
                break
            delay = compute_deadline - time.monotonic()
            if delay > 0:
                time.sleep(min(0.03, delay))
    except BaseException as exc:
        timed_out = timed_out or time.monotonic() >= compute_deadline
        if not (isinstance(exc, EvidenceError) and str(exc) in
                ('MONOTONIC_BUDGET_EXPIRED', 'EXECUTION_OBSERVATION_DEADLINE')
                and timed_out and not _UNREAPED_PROBES):
            error = type(exc).__name__ + ': ' + str(exc)
    finally:
        try:
            # Always close the owned group, even after a successful leader exit.
            # No workload poll/wait/communicate has reaped its identity above.
            for sig, grace in ((signal.SIGTERM, term_grace), (signal.SIGKILL, 0.4)):
                remaining(hard_deadline)
                # macOS getpgid(zombie) may report ESRCH even while ps retains
                # its PID/PGID. The unreaped leader row pins the identity here.
                leader_done, live = snapshot(proc.pid, pgid, hard_deadline)
                observed.update(live)
                if leader_done and not live:
                    leader_done, live = snapshot(proc.pid, pgid, hard_deadline)
                    closure = leader_done and not live
                    if closure:
                        break
                if pgid == os.getpgrp():
                    raise EvidenceError('OWNED_GROUP_IDENTITY_LOST')
                os.killpg(pgid, sig)
                sent.append(sig.name)
                end = min(hard_deadline - 0.25, time.monotonic() + grace)
                while True:
                    leader_done, live = snapshot(proc.pid, pgid, hard_deadline)
                    observed.update(live)
                    if leader_done and not live:
                        # A second current snapshot guards a transient table view.
                        leader_done, live = snapshot(proc.pid, pgid, hard_deadline)
                        closure = leader_done and not live
                        if closure:
                            break
                    if time.monotonic() >= end:
                        break
                    delay = end - time.monotonic()
                    if delay > 0:
                        time.sleep(min(0.03, delay))
                if closure:
                    break
            while not closure and time.monotonic() < hard_deadline - 0.25:
                leader_done, live = snapshot(proc.pid, pgid, hard_deadline)
                observed.update(live)
                if leader_done and not live:
                    leader_done, live = snapshot(proc.pid, pgid, hard_deadline)
                    closure = leader_done and not live
                if not closure:
                    delay = hard_deadline - 0.25 - time.monotonic()
                    if delay > 0:
                        time.sleep(min(0.03, delay))
            if closure:
                # First workload reaping operation, only after group closure.
                code = proc.wait(timeout=remaining(hard_deadline))
                _UNREAPED.remove(proc)
        except BaseException as exc:
            error = (error + '; ' if error else '') + type(exc).__name__ + ': ' + str(exc)
            closure = False
        if log:
            log.close()  # direct regular-file output: no reader or pipe to join
    elapsed = time.monotonic() - started
    over = time.monotonic() >= hard_deadline
    status = ('HOLD_CLEANUP_NOT_PROVEN' if not closure or over else
              'HOLD_PROCESS_EVIDENCE_GROUP_CLOSED' if error else
              'HOLD_EXECUTION_TIMEOUT_GROUP_CLOSED' if timed_out else
              'HOLD_EXIT_CODE_GROUP_CLOSED' if code else 'PASS_PROCESS_GROUP_CLOSED')
    return {'status': status, 'pid': proc.pid, 'pgid': pgid, 'returncode': code,
            'leader_exited_observed': leader_done, 'timed_out': timed_out,
            'signals': sent, 'live_group_members_remaining': live,
            'live_group_members_observed': sorted(observed),
            'cleanup_proven': closure, 'leader_reaped': proc not in _UNREAPED,
            'log_closed': log is None or log.closed, 'reader_thread_used': False,
            'hard_deadline_exceeded': over, 'elapsed_seconds': elapsed,
            'error': error, 'scope': 'Processes remaining in this newly owned group only'}
