"""Native-Python supervision with explicit CPU, GPU-telemetry and SFT budgets.

Frozen v1 and failed runs stay unchanged. No ML package is imported here.
"""
from __future__ import annotations
import os
import math
from pathlib import Path
import queue
import secrets
import threading
import time

from src.train.cache_release_watchdog import BudgetWatch, append_durable, read_optional, write_atomic
from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack, verify_identity
from src.train.windows_job_process_v2 import JobProcess
from src.train.windows_job_exit_wait_v3 import terminate_and_wait, exit_timing, close_without_new_wait
from src.train import windows_job_process as win
from src.train.stage_file_io_v2 import read_stage


def confirm_natural_exit(job, handles, record, deadline, *, max_lifetime_seconds=10.0):
    """No termination request: confirm all retained roles under one deadline."""
    rows = []
    for handle in handles:
        record('natural_wait_started', target_pid=handle.pid, deadline_monotonic=deadline)
        began = time.perf_counter()
        try:
            code = handle.wait(max(0.0, deadline - began))
            observed = time.perf_counter()
            row = {'pid': handle.pid, 'result': 'signaled', 'exit_code': code, 'alive': handle.alive(),
                   'observed_exit_monotonic': observed, 'wait_seconds': observed - began,
                   'deadline_monotonic': deadline, **exit_timing(handle)}
        except Exception as error:
            record('natural_wait_failed', target_pid=handle.pid, error=str(error))
            raise
        record('natural_wait_finished', result=row)
        rows.append(row)
        if observed > deadline or row['alive'] or not 0 <= row['lifetime_seconds'] <= max_lifetime_seconds:
            raise TimeoutError('Natural exit confirmation exceeded its deadline or lifetime cap')
    active = job.active_pids()
    observed = time.perf_counter()
    record('natural_job_empty_checked', active_pids=active, deadline_monotonic=deadline,
           checked_monotonic=observed)
    if active or observed > deadline:
        raise TimeoutError('Natural exit Job-empty check failed')
    return rows


def supervise(command, cwd, env, run_dir, limits, *, sample_provider=None, suite_deadline=None):
    """One worker launch, no retries; test budgets are explicit caller configuration."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ('supervision_claim.json', 'console.log', 'result.json')):
        raise FileExistsError('This run was already claimed; no replay')
    started = time.perf_counter()
    options = dict(limits)
    budget_profile = options.pop('budget_profile', 'cpu')
    profiles = {'cpu': (8.0, 9.5), 'gpu_telemetry': (25.0, 27.0),
                'bounded_sft': (304.0, 306.0)}
    if budget_profile not in profiles:
        raise ValueError('Unknown explicit supervisor budget profile')
    max_self_seconds, max_case_seconds = profiles[budget_profile]
    handshake_seconds = options.pop('handshake_seconds', 2.5)
    cleanup_seconds = options.pop('cleanup_seconds', 2.0)
    expected_count = options.pop('expected_process_count', 1)
    child_self_seconds = options.pop('child_self_seconds', 8.0)
    case_seconds = options.pop('case_seconds', 9.5)
    budget_values = (handshake_seconds, cleanup_seconds, child_self_seconds, case_seconds)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in budget_values):
        raise ValueError('Supervisor budgets must be finite numbers')
    if (cleanup_seconds != 2 or not 0 < handshake_seconds < child_self_seconds
            or not 0 < child_self_seconds <= max_self_seconds
            or not child_self_seconds < case_seconds <= max_case_seconds):
        raise ValueError('Budget exceeds explicit profile or leaves no independent cleanup margin')
    if not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError('Expected process count must be positive')
    case_deadline = min(started + case_seconds, suite_deadline or float('inf'))
    if case_deadline - started < case_seconds:
        raise TimeoutError('Insufficient suite budget for a new case')
    watch = BudgetWatch(started, options)
    if watch.limits['total_hard_seconds'] + cleanup_seconds >= child_self_seconds:
        raise ValueError('Hard stop and confirmation must finish before fixture self-expiry')
    nonce = secrets.token_hex(24)
    write_atomic(directory / 'supervision_claim.json', {'nonce': nonce, 'started_monotonic': started,
        'case_deadline_monotonic': case_deadline, 'child_self_deadline': started + child_self_seconds,
        'budget_profile': budget_profile, 'max_lifetime_seconds': case_seconds,
        'expected_process_count': expected_count, 'automatic_retries': 0})
    child_env = dict(env, PID_RUN_DIR=str(directory.resolve()), PID_RUN_NONCE=nonce,
        PID_SUPERVISOR_PID=str(os.getpid()), PID_RUN_STARTED=str(started),
        PID_CHILD_DEADLINE=str(started + child_self_seconds))
    job = thread = None
    retained = {}
    retained_identities = {}
    all_seen = set()
    errors, cleanup_errors, events, rows, role_exits = [], [], [], [], []
    samples = queue.Queue()
    sampling_stop = threading.Event()
    worker_pid = bound_pid = exit_code = None
    verified = killed = closed = False
    membership_sent = False
    ancestry = []
    request_written = None
    valid_count = 0
    outcome, stop_reason, identity_error = 'failed', None, None
    termination = None

    def sampler_loop(pid):
        sampler = None
        try:
            if sample_provider is None:
                from src.train.windows_gpu_memory import WindowsGPUMemorySampler
                sampler = WindowsGPUMemorySampler(pid, watch.limits['sample_interval_seconds'])
            with (directory / 'pid_memory.jsonl').open('x', encoding='utf-8') as output:
                while not sampling_stop.is_set():
                    stage = read_stage(directory) or {}
                    tick = time.perf_counter()
                    if sampler:
                        sampler.stage = stage.get('stage', 'before_cuda')
                        row = sampler.sample()
                    else:
                        row = dict(sample_provider(pid, stage))
                    if row.get('pid', pid) != pid:
                        raise ValueError('Sample provider changed the bound PID')
                    row.update(pid=pid, sampled_monotonic=tick, stage=stage.get('stage', 'before_cuda'),
                        sample_source='Windows PDH' if sampler else 'injected_cpu_fixture')
                    append_durable(output, row)
                    samples.put(row)
                    if budget_profile in ('gpu_telemetry', 'bounded_sft') and row.get('status') not in ('ok', 'not_registered'):
                        raise RuntimeError(f"GPU telemetry provider failure: {row.get('status')}")
                    sampling_stop.wait(max(0, watch.limits['sample_interval_seconds'] - (time.perf_counter() - tick)))
        except Exception as error:
            errors.append(f'Sampling: {type(error).__name__}: {error}')
        finally:
            if sampler is not None and sampler._query is not None:
                sampler._query.close()

    def drain():
        nonlocal valid_count
        fresh = []
        while True:
            try:
                row = samples.get_nowait()
                rows.append(row)
                fresh.append(row)
                valid_count += row.get('status') == 'ok'
            except queue.Empty:
                return fresh

    def stop_sampler():
        sampling_stop.set()
        if thread:
            thread.join(timeout=max(0, min(0.5, case_deadline - time.perf_counter())))
            if thread.is_alive():
                raise TimeoutError('Sampler did not stop within the remaining case budget')
        drain()

    with (directory / 'supervisor_events.jsonl').open('x', encoding='utf-8') as log:
        def event(name, **details):
            row = {'event': name, 'monotonic': time.perf_counter(), **details}
            events.append(row)
            append_durable(log, row)

        def retain_members():
            active = job.active_pids()
            all_seen.update(active)
            for pid in active:
                if pid not in retained:
                    handle = job.open_process(pid)
                    retained[pid] = handle
                    retained_identities[pid] = handle.identity()
                    event('process_handle_retained', identity=handle.identity())
            if len(retained) > expected_count:
                raise ValueError('Unexpected process topology; stop instead of guessing ownership')
            return active

        def force_stop():
            nonlocal killed, termination, role_exits
            killed = True
            stop_sampler()
            active = retain_members()
            event('force_stop_scope', active_pids=active, retained_pids=list(retained))
            # Fixed launcher/worker roles are retained before business activity begins.
            termination = terminate_and_wait(job, list(retained.values()), event,
                exit_code=124, timeout=min(2.0, case_deadline - time.perf_counter()),
                max_lifetime_seconds=case_seconds)
            role_exits = termination['roles']

        try:
            job = JobProcess(command, cwd, child_env, directory / 'console.log')
            event('launcher_in_job', launcher_pid=job.pid, start_order=job.start_order)
            while True:
                now = time.perf_counter()
                if now >= case_deadline - cleanup_seconds:
                    raise TimeoutError('Case budget reserved for cleanup')
                active = retain_members()
                if not active:
                    break
                if len(retained) == expected_count and not membership_sent:
                    write_atomic(directory / 'membership_ack.json', {'nonce': nonce, 'retained_pids': sorted(retained)})
                    event('membership_acknowledged', retained_pids=sorted(retained))
                    membership_sent = True
                if membership_sent and not verified:
                    reported = read_optional(directory / 'worker_identity.json')
                    if reported is not None:
                        event('worker_identity_received', reported=reported)
                        try:
                            handle, actual, ancestry = verify_identity(job, reported, nonce)
                        except ValueError as error:
                            identity_error = str(error)
                            outcome, stop_reason = 'identity_rejected', identity_error
                            event('worker_identity_rejected', reason=identity_error)
                            force_stop()
                            break
                        try:
                            if actual['pid'] not in retained or actual != retained_identities[actual['pid']]:
                                raise ValueError('Verified worker differs from retained process identity')
                        finally:
                            handle.close()
                        worker_pid = bound_pid = actual['pid']
                        verified = True
                        event('worker_identity_verified', worker_pid=worker_pid, ancestry=ancestry)
                        thread = threading.Thread(target=sampler_loop, args=(bound_pid,), daemon=True)
                        thread.start()
                        write_atomic(directory / 'identity_ack.json', {'nonce': nonce, 'worker_pid': worker_pid,
                            'launcher_pid': job.pid, 'creation_time_100ns': actual['creation_time_100ns']})
                if not verified and now - started >= handshake_seconds:
                    raise TimeoutError('Worker identity handshake deadline')
                stage = read_stage(directory) or {}
                fresh = drain()
                if valid_count >= 3 and not (directory / 'telemetry_ready.json').exists():
                    write_atomic(directory / 'telemetry_ready.json', {'nonce': nonce, 'worker_pid': worker_pid,
                        'valid_sample_count': valid_count, 'monotonic': now})
                if stage.get('telemetry_complete') and not closed:
                    if not verified:
                        raise ValueError('Completion before verified identity')
                    stop_sampler()
                    if errors or valid_count < 3:
                        raise RuntimeError('Completion without sufficient valid samples')
                    closed = True
                    write_atomic(directory / 'telemetry_closed.json', {'nonce': nonce, 'worker_pid': worker_pid,
                        'sample_count': len(rows), 'valid_sample_count': valid_count})
                    event('telemetry_closed', sample_count=len(rows), valid_sample_count=valid_count)
                if errors:
                    raise RuntimeError(errors[0])
                policy_stage = dict(stage)
                if closed:
                    policy_stage['telemetry_required'] = False
                decision = watch.evaluate(now, policy_stage, fresh, read_optional(directory / 'stop_request.json'))
                if decision['request'] and decision['request'] != request_written:
                    write_atomic(directory / 'stop_request.json', decision['request'])
                    request_written = dict(decision['request'])
                    event('stop_requested', **request_written)
                if decision['hard_reason']:
                    outcome, stop_reason = 'hard_timeout', decision['hard_reason']
                    event('hard_stop_triggered', reason=stop_reason)
                    force_stop()
                    break
                time.sleep(0.015)
            if not role_exits:
                stop_sampler()
                role_exits = confirm_natural_exit(job, list(retained.values()), event,
                    min(time.perf_counter() + 2.0, case_deadline), max_lifetime_seconds=case_seconds)
            exit_code = next(row['exit_code'] for row in role_exits if row['pid'] == job.pid)
            if outcome == 'failed':
                if verified and closed and not watch.request and exit_code == 0:
                    outcome, stop_reason = 'completed', 'normal_completion'
                elif verified and closed and watch.request and exit_code == 20:
                    outcome, stop_reason = 'soft_stopped', watch.request['reason']
                else:
                    raise RuntimeError('Process exit did not match a completed or cooperative stop protocol')
        except Exception as error:
            errors.append(f'{type(error).__name__}: {error}')
            outcome = 'failed'
            stop_reason = stop_reason or errors[-1]
            event('supervisor_error', error=errors[-1])
        finally:
            try:
                stop_sampler()
            except Exception as error:
                cleanup_errors.append(str(error))
            if job:
                try:
                    active = retain_members()
                    if active:
                        event('emergency_cleanup', active_pids=active, acceptance_retry=False)
                        win._check(win._TerminateJob(job.job, 125), 'Cleanup TerminateJobObject')
                except Exception as error:
                    cleanup_errors.append(str(error))
                # No old close()->5s retry; independent retained handles survive Job close.
                try:
                    active_after = job.active_pids()
                    close_without_new_wait(job)
                except Exception as error:
                    cleanup_errors.append(str(error))
                    active_after = None
            else:
                active_after = []
            cleanup_rows = []
            for handle in retained.values():
                try:
                    code = handle.wait(max(0.0, case_deadline - time.perf_counter()))
                    row = {'pid': handle.pid, 'exit_code': code, 'alive': handle.alive(),
                           'observed_exit_monotonic': time.perf_counter(), **exit_timing(handle)}
                    cleanup_rows.append(row)
                    event('retained_handle_cleanup', result=row)
                    if not 0 <= row['lifetime_seconds'] <= case_seconds or row['alive']:
                        cleanup_errors.append(f'PID {handle.pid} violated lifetime/exit limit')
                except Exception as error:
                    cleanup_errors.append(str(error))
                    event('cleanup_wait_failed', target_pid=handle.pid, error=str(error))
                finally:
                    handle.close()
            if active_after or len(cleanup_rows) != expected_count:
                cleanup_errors.append('Job not confirmed empty or expected process count not observed')
            if time.perf_counter() > case_deadline:
                cleanup_errors.append('Case cleanup deadline exceeded')
            event('cleanup_complete', job_active_pids=active_after, errors=cleanup_errors)
    if errors or cleanup_errors:
        outcome = 'failed'
    result = {'status': {'completed': 'pass', 'soft_stopped': 'stopped', 'hard_timeout': 'stopped',
                         'identity_rejected': 'rejected', 'failed': 'failed'}[outcome],
        'outcome': outcome, 'stop_reason': stop_reason, 'identity_error': identity_error,
        'launcher_pid': job.pid if job else None, 'worker_pid': worker_pid, 'bound_pid': bound_pid,
        'identity_verified': verified, 'ancestry': ancestry, 'exit_code': exit_code, 'killed': killed,
        'retained_identities': list(retained_identities.values()), 'role_exits': role_exits,
        'cleanup_exits': cleanup_rows, 'job_active_pids_after_cleanup': active_after,
        'observed_job_pids': sorted(all_seen), 'sample_count': len(rows), 'valid_sample_count': valid_count,
        'observed_sample_pids': sorted({row['pid'] for row in rows}), 'errors': errors,
        'cleanup_errors': cleanup_errors, 'termination': termination, 'stop_request': watch.request,
        'wall_seconds': time.perf_counter() - started, 'automatic_retries': 0, 'worker_launches': 1 if job else 0,
        'sample_source': 'Windows PDH' if sample_provider is None else 'injected_cpu_fixture',
        'telemetry_closed': closed, 'budget_profile': budget_profile,
        'max_lifetime_seconds': case_seconds, 'limits': dict(limits), 'events': events}
    write_atomic(directory / 'result.json', result)
    return result
