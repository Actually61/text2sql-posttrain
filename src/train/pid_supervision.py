"""Identity-verified, job-contained supervisor; imports no ML/CUDA packages."""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import secrets
import threading
import time

from src.train.cache_release_watchdog import BudgetWatch, append_durable, read_optional, write_atomic
from src.train.windows_job_process import JobProcess, ProcessHandle


def publish_worker_identity(run_dir, nonce):
    """Call before importing Torch. Both PID and creation time identify this process."""
    directory = Path(run_dir)
    with ProcessHandle(os.getpid()) as handle:
        identity = handle.identity()
    identity.update(nonce=nonce, published_monotonic=time.perf_counter())
    write_atomic(directory / 'worker_identity.json', identity)
    return identity


def wait_for_identity_ack(run_dir, nonce, timeout=5.0):
    deadline = time.perf_counter() + timeout
    directory = Path(run_dir)
    while time.perf_counter() < deadline:
        if (directory / 'stop_request.json').exists():
            raise RuntimeError('Supervisor stopped before identity acknowledgement')
        ack = read_optional(directory / 'identity_ack.json')
        if ack:
            if ack.get('nonce') != nonce or ack.get('worker_pid') != os.getpid():
                raise ValueError('Wrong identity acknowledgement')
            return ack
        time.sleep(0.02)
    raise TimeoutError('Identity acknowledgement deadline')


def verify_identity(job, reported, nonce):
    if reported.get('nonce') != nonce:
        raise ValueError('Run nonce mismatch')
    pid = reported.get('pid')
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError('Invalid worker PID')
    handle = job.open_process(pid)  # Retain OS handle, not just a recyclable PID.
    try:
        actual = handle.identity()
        for key in ('pid', 'parent_pid', 'creation_time_100ns', 'executable'):
            if reported.get(key) != actual[key]:
                raise ValueError(f'Worker identity mismatch: {key}')
        if not handle.alive():
            raise ValueError('Worker already exited')
        # Confirm the current ancestry reaches our suspended-at-start launcher.
        chain, seen, current = [actual], {pid}, actual
        while current['pid'] != job.pid:
            parent = current['parent_pid']
            if parent in seen or len(chain) > 16:
                raise ValueError('Invalid launcher ancestry')
            seen.add(parent)
            with job.open_process(parent) as parent_handle:
                parent_identity = parent_handle.identity()
            if parent_identity['creation_time_100ns'] > current['creation_time_100ns']:
                raise ValueError('Parent PID reused after worker creation')
            chain.append(parent_identity)
            current = parent_identity
        return handle, actual, chain
    except Exception:
        handle.close()
        raise


def supervise(command, cwd, env, run_dir, limits, *, sample_provider=None):
    """One single-use run; sample_provider is only for CPU test fixtures."""
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ('supervision_claim.json', 'console.log', 'result.json')):
        raise FileExistsError('This run was already claimed; no replay')
    nonce = secrets.token_hex(24)
    started = time.perf_counter()
    write_atomic(directory / 'supervision_claim.json', {'nonce': nonce, 'started_monotonic': started})
    limits = dict(limits)
    handshake_seconds = limits.pop('handshake_seconds', 5.0)
    cleanup_seconds = limits.pop('cleanup_seconds', 3.0)
    watch = BudgetWatch(started, limits)
    child_env = dict(env, PID_RUN_DIR=str(directory.resolve()), PID_RUN_NONCE=nonce,
                     PID_SUPERVISOR_PID=str(os.getpid()), PID_RUN_STARTED=str(started))
    job = handle = thread = None
    errors, events = [], []
    sample_queue = queue.Queue()
    sampling_stop = threading.Event()
    worker_pid = bound_pid = None
    identity_verified = killed = closed = False
    ancestry, all_seen_pids = [], set()
    exit_code = None
    request_written = None
    rows, valid_count = [], 0

    def sampler_loop(pid):
        sampler = None
        try:
            if sample_provider is None:
                from src.train.windows_gpu_memory import WindowsGPUMemorySampler
                sampler = WindowsGPUMemorySampler(pid, watch.limits['sample_interval_seconds'])
            with (directory / 'pid_memory.jsonl').open('x', encoding='utf-8') as output:
                while not sampling_stop.is_set():
                    stage = read_optional(directory / 'stage.json') or {}
                    tick = time.perf_counter()
                    if sampler is not None:
                        sampler.stage = stage.get('stage', 'before_cuda')
                        row = sampler.sample()
                    else:
                        row = dict(sample_provider(pid, stage))
                    if row.get('pid', pid) != pid:
                        raise ValueError('Sample provider changed the bound PID')
                    row.update(pid=pid, sampled_monotonic=tick, stage=stage.get('stage', 'before_cuda'),
                               sample_source='Windows PDH' if sampler else 'injected_cpu_fixture')
                    append_durable(output, row)
                    sample_queue.put(row)
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
                row = sample_queue.get_nowait()
                rows.append(row)
                fresh.append(row)
                valid_count += row.get('status') == 'ok'
            except queue.Empty:
                break
        return fresh

    with (directory / 'supervisor_events.jsonl').open('x', encoding='utf-8') as log:
        def event(name, **details):
            record = {'event': name, 'monotonic': time.perf_counter(), **details}
            events.append(record)
            append_durable(log, record)

        try:
            job = JobProcess(command, cwd, child_env, directory / 'console.log')
            event('launcher_in_job', launcher_pid=job.pid)
            all_seen_pids.add(job.pid)
            while True:
                now = time.perf_counter()
                active = job.active_pids()
                all_seen_pids.update(active)
                if not active:
                    break
                if not identity_verified:
                    reported = read_optional(directory / 'worker_identity.json')
                    if reported is not None:
                        handle, actual, ancestry = verify_identity(job, reported, nonce)
                        worker_pid = bound_pid = actual['pid']
                        identity_verified = True
                        event('worker_identity_verified', worker_pid=worker_pid, ancestry=ancestry)
                        thread = threading.Thread(target=sampler_loop, args=(bound_pid,), daemon=True)
                        thread.start()
                        write_atomic(directory / 'identity_ack.json', {'nonce': nonce, 'worker_pid': worker_pid,
                            'launcher_pid': job.pid, 'creation_time_100ns': actual['creation_time_100ns']})
                    elif now - started >= handshake_seconds:
                        watch.trigger(now, 'Worker identity handshake deadline', 'resource')
                stage = read_optional(directory / 'stage.json') or {}
                fresh = drain()
                if valid_count >= 3 and not (directory / 'telemetry_ready.json').exists():
                    write_atomic(directory / 'telemetry_ready.json', {'nonce': nonce, 'worker_pid': worker_pid,
                        'valid_sample_count': valid_count, 'monotonic': now})
                if stage.get('telemetry_complete') and not closed:
                    if not identity_verified:
                        raise ValueError('Completion before verified identity')
                    sampling_stop.set()
                    thread.join(timeout=min(1.0, cleanup_seconds))
                    if thread.is_alive():
                        raise RuntimeError('Sampler failed to close')
                    fresh += drain()
                    if errors or valid_count < 3:
                        raise RuntimeError('Telemetry completion without sufficient valid samples')
                    closed = True
                    write_atomic(directory / 'telemetry_closed.json', {'nonce': nonce, 'worker_pid': worker_pid,
                        'sample_count': len(rows), 'valid_sample_count': valid_count})
                    event('telemetry_closed', sample_count=len(rows), valid_sample_count=valid_count)
                if errors:
                    watch.trigger(now, errors[0], 'resource')
                policy_stage = dict(stage)
                if closed:
                    policy_stage['telemetry_required'] = False
                decision = watch.evaluate(now, policy_stage, fresh, read_optional(directory / 'stop_request.json'))
                if decision['request'] and decision['request'] != request_written:
                    write_atomic(directory / 'stop_request.json', decision['request'])
                    request_written = dict(decision['request'])
                    event('stop_requested', **request_written)
                if decision['hard_reason']:
                    killed = True
                    event('terminate_job_tree', reason=decision['hard_reason'], active_pids=active)
                    job.terminate_tree()
                    break
                time.sleep(0.015)
            exit_code = job.wait(timeout=cleanup_seconds)
        except Exception as error:
            errors.append(f'{type(error).__name__}: {error}')
            event('supervisor_error', error=errors[-1])
            if job is not None:
                killed = True
                event('terminate_job_tree_after_error', active_pids=job.active_pids())
                job.terminate_tree()
                exit_code = job.wait(timeout=cleanup_seconds)
        finally:
            sampling_stop.set()
            if thread:
                thread.join(timeout=min(1.0, cleanup_seconds))
                if thread.is_alive():
                    errors.append('Sampler thread still alive after cleanup deadline')
            drain()
            active_after = job.active_pids() if job is not None else []
            worker_alive_after = handle.alive() if handle is not None else None
            if active_after or worker_alive_after:
                errors.append('Processes remained before final job cleanup')
            if handle:
                handle.close()
            if job:
                job.close()
            event('cleanup_complete', job_active_pids=active_after, worker_alive=worker_alive_after)
    valid = [r for r in rows if r.get('status') == 'ok']
    result = {'status': 'pass' if identity_verified and closed and exit_code == 0 and not killed and not watch.request
              and not errors and len(valid) >= 3 and not active_after and worker_alive_after is False else 'failed',
        'launcher_pid': job.pid if job else None, 'worker_pid': worker_pid, 'bound_pid': bound_pid,
        'identity_verified': identity_verified, 'ancestry': ancestry, 'exit_code': exit_code, 'killed': killed,
        'job_active_pids_after_cleanup': active_after, 'worker_alive_after_cleanup': worker_alive_after,
        'observed_job_pids': sorted(all_seen_pids), 'sample_count': len(rows), 'valid_sample_count': len(valid),
        'observed_sample_pids': sorted({r['pid'] for r in rows}), 'errors': errors,
        'pid_dedicated_peak_mib': max((r['dedicated_bytes'] / 2**20 for r in valid), default=None),
        'pid_shared_peak_mib': max((r['shared_bytes'] / 2**20 for r in valid), default=None),
        'stop_request': watch.request, 'wall_seconds': time.perf_counter() - started,
        'automatic_retries': 0, 'worker_launches': 1 if job else 0, 'events': events,
        'sample_source': 'Windows PDH' if sample_provider is None else 'injected_cpu_fixture',
        'limits': dict(watch.limits, handshake_seconds=handshake_seconds, cleanup_seconds=cleanup_seconds)}
    write_atomic(directory / 'result.json', result)
    return result
