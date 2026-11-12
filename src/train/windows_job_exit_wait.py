"""Bounded exit confirmation for already identified Windows process handles.

Only stdlib and the frozen Windows API bindings; no ML imports or new processes.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes as wt
import math
import time

from src.train import windows_job_process as win


def exit_timing(process):
    """Read exit FILETIME only after the retained process handle is signaled."""
    if process.alive():
        raise RuntimeError('Exit time is undefined for a live process')
    created, exited, kernel, user = (wt.FILETIME() for _ in range(4))
    win._check(win._GetTimes(process.handle, ctypes.byref(created), ctypes.byref(exited),
                            ctypes.byref(kernel), ctypes.byref(user)), 'GetProcessTimes')
    creation = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
    exit_time = (int(exited.dwHighDateTime) << 32) | int(exited.dwLowDateTime)
    if creation != process.identity()['creation_time_100ns'] or exit_time < creation:
        raise ValueError('Invalid process timing or changed identity')
    return {'creation_time_100ns': creation, 'exit_time_100ns': exit_time,
            'lifetime_seconds': (exit_time - creation) / 10_000_000}


def terminate_and_wait(job, processes, record, *, exit_code=124, timeout=2.0):
    """One shared deadline includes termination, per-role waits and Job-empty check.

    Caller retains ownership of handles and supplies a durable record callback.
    An exception stops acceptance; cleanup must not retry this acceptance helper.
    """
    if not math.isfinite(timeout) or not 0 < timeout <= 2.0:
        raise ValueError('Exit confirmation budget must be in (0, 2] seconds')
    processes = tuple(processes)
    pids = [process.pid for process in processes]
    active = job.active_pids()
    record('exit_wait_scope', retained_pids=pids, active_pids=active)
    if not pids or len(set(pids)) != len(pids) or set(pids) != set(active):
        raise ValueError('Retained handles must cover the complete current Job')
    started = time.perf_counter()
    deadline = started + timeout
    record('terminate_requested', requested_monotonic=started,
           deadline_monotonic=deadline, timeout_seconds=timeout, exit_code=exit_code)
    ok = win._TerminateJob(job.job, int(exit_code))
    error = ctypes.get_last_error() if not ok else 0
    record('terminate_returned', api_success=bool(ok), winerror=error)
    if not ok:
        raise ctypes.WinError(error, 'TerminateJobObject')
    rows = []
    for process in processes:
        record('role_wait_started', target_pid=process.pid, deadline_monotonic=deadline)
        wait_started = time.perf_counter()
        remaining = max(0.0, deadline - wait_started)
        try:
            code = process.wait(remaining)
            observed = time.perf_counter()
            row = {'pid': process.pid, 'result': 'signaled', 'exit_code': code,
                   'wait_started_monotonic': wait_started, 'observed_exit_monotonic': observed,
                   'remaining_seconds_at_wait': remaining,
                   'wait_seconds': observed - wait_started,
                   'seconds_since_termination_request': observed - started,
                   'deadline_monotonic': deadline, 'alive': process.alive(),
                   **exit_timing(process)}
        except Exception as failure:
            record('role_wait_failed', target_pid=process.pid, remaining_seconds_at_wait=remaining,
                   wait_started_monotonic=wait_started, deadline_monotonic=deadline,
                   failed_monotonic=time.perf_counter(), error=f'{type(failure).__name__}: {failure}')
            raise
        record('role_wait_finished', target_pid=process.pid, result=row)
        rows.append(row)
        if row['alive'] or observed > deadline:
            raise TimeoutError('Role did not confirm exit within the shared deadline')
        if code != exit_code or not 0 <= row['lifetime_seconds'] <= 10.0:
            raise ValueError('Unexpected exit code or lifetime exceeds 10 seconds')
    active_after = job.active_pids()
    checked = time.perf_counter()
    record('job_empty_checked', active_pids=active_after, checked_monotonic=checked,
           deadline_monotonic=deadline, seconds_since_termination_request=checked - started)
    if active_after or checked > deadline:
        raise TimeoutError('Job-empty confirmation exceeded the shared deadline')
    return {'requested_monotonic': started, 'deadline_monotonic': deadline,
            'confirmed_monotonic': checked, 'confirmation_seconds': checked - started,
            'roles': rows, 'job_active_pids_after': active_after}


def close_without_new_wait(job):
    """Close kill-on-close Job and launcher handles, without a fresh wait budget."""
    errors = []
    for name in ('job', 'handle'):
        handle = getattr(job, name, None)
        if handle:
            if not win._Close(handle):
                errors.append(f'CloseHandle({name}): {ctypes.get_last_error()}')
            else:
                setattr(job, name, None)
    job._closed = not (job.job or job.handle)
    if errors:
        raise OSError('; '.join(errors))
