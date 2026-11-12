"""Single CPU round: one writer, two readers, a bounded named stage mutex."""
from __future__ import annotations
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from src.train.windows_json_ipc import read_json, write_json_atomic, open_shared_binary
from src.train.stage_file_io import stage_evidence
from src.train.stage_file_io_v2 import read_stage, write_stage
from src.train.windows_stage_mutex import locked_stage
from src.train.windows_job_process_v2 import JobProcess, ProcessHandle, VENV_PYTHON, NATIVE_PYTHON, normalize
from src.train.windows_job_exit_wait_v3 import exit_timing, close_without_new_wait
from src.train import windows_job_process as win

RUN = ROOT / 'outputs/checks/stage_mutex_v1'
REPORT = ROOT / 'reports/stage_mutex_cpu_checks.json'
SOURCES = ['src/train/windows_stage_mutex.py', 'src/train/stage_file_io_v2.py',
    'src/train/test_stage_mutex.py', 'src/train/pid_supervision_v6.py',
    'src/train/sft_cache_resume_worker_v3.py']
FORBIDDEN = ('torch', 'transformers', 'peft', 'bitsandbytes')
FINAL_SEQUENCE = 34


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    """Closed-result files only; live protocol messages use publish instead."""
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def publish(path, value):
    # Each protocol path has one publisher and is published exactly once.
    if Path(path).exists():
        raise FileExistsError(f'Protocol message already published: {path}')
    write_json_atomic(path, value)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot(sequence, nonce, done=False):
    payload = f'阶段快照-{sequence}-' + '完整输入' * (1 + sequence % 37)
    return {'sequence': sequence, 'nonce': nonce, 'done': done, 'payload': payload,
            'sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest()}


def check(value, nonce):
    assert value['nonce'] == nonce
    assert value == snapshot(value['sequence'], nonce, value['done'])
    assert 0 <= value['sequence'] <= FINAL_SEQUENCE
    return value['sequence']


def child(role):
    directory = Path(os.environ['IPC_DIRECTORY'])
    nonce = os.environ['IPC_NONCE']
    deadline = float(os.environ['IPC_CHILD_DEADLINE'])
    def expire():
        time.sleep(max(0.0, deadline - time.perf_counter()))
        os._exit(92)
    threading.Thread(target=expire, daemon=True).start()
    def guard():
        if time.perf_counter() >= deadline:
            raise TimeoutError('Child self deadline')
    def wait_value(path):
        while True:
            guard()
            value = read_json(path, missing_ok=True)
            if value is not None:
                assert value['nonce'] == nonce
                return value
            time.sleep(0.002)  # Wait for publication, never retry a failed I/O operation.
    started = time.perf_counter()
    evidence = stage_evidence(directory)
    try:
        with ProcessHandle(os.getpid()) as handle:
            identity = handle.identity()
        publish(directory / f'{role}_identity.json', {**identity, 'role': role,
            'sys_executable': sys.executable, 'nonce': nonce, 'started_monotonic': started})
        wait_value(directory / 'start.json')
        if role == 'writer':
            write_stage(directory, snapshot(0, nonce))
            publish(directory / 'initial_published.json', {'nonce': nonce})
            wait_value(directory / 'reader1_held.json')
            intent_at = time.perf_counter()
            publish(directory / 'writer_intent.json', {'nonce': nonce, 'monotonic': intent_at})
            write_stage(directory, snapshot(1, nonce))
            replaced = time.perf_counter()
            publish(directory / 'barrier_complete.json', {'nonce': nonce, 'monotonic': replaced})
            for reader in ('reader1', 'reader2'):
                wait_value(directory / f'{reader}_observed1.json')
            for sequence in range(2, FINAL_SEQUENCE):
                guard()
                write_stage(directory, snapshot(sequence, nonce))
                time.sleep(0.004)
            write_stage(directory, snapshot(FINAL_SEQUENCE, nonce, True))
            for reader in ('reader1', 'reader2'):
                wait_value(directory / f'{reader}_done.json')
            details = {'published_versions': FINAL_SEQUENCE + 1,
                'barrier_write_intent_monotonic': intent_at, 'barrier_replace_monotonic': replaced}
        else:
            wait_value(directory / 'initial_published.json')
            details = {}
            if role == 'reader1':
                with locked_stage(directory, evidence=evidence):
                    acquired = time.perf_counter()
                    with open_shared_binary(directory / 'stage.json', evidence=evidence) as held:
                        old = json.loads(held.read().decode('utf-8'))
                        assert check(old, nonce) == 0
                        publish(directory / 'reader1_held.json', {'nonce': nonce, 'monotonic': acquired})
                        wait_value(directory / 'writer_intent.json')
                        # Hold briefly to force a real, bounded writer wait.
                        time.sleep(0.12)
                        held.seek(0)
                        assert json.loads(held.read().decode('utf-8')) == old
                    file_closed = time.perf_counter()  # Still owns mutex; HANDLE already closed.
                unlocked = time.perf_counter()
                details.update(barrier_lock_acquired_monotonic=acquired,
                    barrier_file_closed_monotonic=file_closed, barrier_mutex_released_monotonic=unlocked)
            wait_value(directory / 'barrier_complete.json')
            assert check(read_stage(directory), nonce) == 1
            publish(directory / f'{role}_observed1.json', {'nonce': nonce})
            seen, reads, last = {1}, 1, 1
            while True:
                guard()
                value = read_stage(directory)
                current = check(value, nonce)
                assert current >= last, 'Sequence regressed'
                seen.add(current); reads += 1; last = current
                if value['done']:
                    assert current == FINAL_SEQUENCE
                    break
                time.sleep(0.001)
            assert len(seen) >= 5, 'Insufficient observed concurrent versions'
            details.update(read_count=reads, distinct_versions=len(seen), final_sequence=last,
                json_and_checksum_valid=True, sequence_never_regressed=True)
            publish(directory / f'{role}_done.json', {'nonce': nonce})
        assert not any(name in sys.modules for name in FORBIDDEN)
        save(directory / f'{role}_result.json', {'status': 'pass', 'role': role, 'pid': os.getpid(),
            'seconds': time.perf_counter() - started, 'ml_imported': False, 'automatic_retries': 0, **details})
        return 0
    except Exception as error:
        publish(directory / f'{role}_failure.json', {'status': 'failed', 'role': role, 'pid': os.getpid(),
            'errno': getattr(error, 'errno', None), 'winerror': getattr(error, 'winerror', None),
            'error': repr(error), 'traceback': traceback.format_exc(), 'monotonic': time.perf_counter()})
        return 2


def protected_files():
    old = ROOT / 'outputs/checks/stage_json_ipc_v1'
    mapping = dict(read(old / 'preservation_before.json'))
    mapping.update(read(ROOT / 'reports/stage_json_ipc_cpu_checks.json')['source_sha256'])
    mapping.update({str(old / name): sha for name, sha in read(old / 'artifact_sha256.json').items()})
    for path in [old / 'artifact_sha256.json', ROOT / 'reports/stage_json_ipc_cpu_checks.json',
                 ROOT / 'reports/stage_json_ipc_readonly_audit.json', ROOT / 'reports/stage_json_ipc_walkthrough.md']:
        mapping[str(path)] = digest(path)
    for name, sha in mapping.items():
        assert digest(ROOT / name) == sha, f'Historical file changed: {name}'
    return mapping


def audit_mutex_logs(directory, roles):
    """Prove serialization using acquired→releasing, not post-release timestamps."""
    all_rows, intervals = [], []
    for path in directory.glob('stage_ipc_*.jsonl'):
        content = path.read_text(encoding='utf-8')
        assert not content or content.endswith('\n'), f'Incomplete JSONL: {path}'
        rows = [json.loads(line) for line in content.splitlines()]
        all_rows.extend(rows)
        current = None
        for row in rows:
            assert row['automatic_retries'] == 0
            assert row['status'] not in ('failed', 'timeout', 'abandoned'), row
            if row['operation'] == 'mutex' and row['phase'] == 'acquired':
                assert current is None, 'Unexpected recursive mutex acquisition'
                current = {'start': row['monotonic'], 'pid': row['pid'],
                           'mutex_name': row['mutex_name'], 'rows': [row]}
            elif row['operation'] == 'mutex' and row['phase'] == 'releasing':
                assert current is not None
                current['end'] = row['monotonic']
                current['rows'].append(row)
                intervals.append(current)
                current = None
            elif current is not None:
                current['rows'].append(row)
        assert current is None, 'Mutex critical section left open'
    assert all_rows and intervals
    intervals.sort(key=lambda value: value['start'])
    assert len({value['mutex_name'] for value in intervals}) == 1
    for previous, following in zip(intervals, intervals[1:]):
        assert previous['end'] <= following['start'], 'Critical sections overlap'
    closes = replacements = 0
    for interval in intervals:
        operations = interval['rows']
        close_rows = [row for row in operations if row['operation'] == 'close_read' and row['status'] == 'ok']
        replace_rows = [row for row in operations if row['operation'] == 'write_json_atomic'
                        and row['phase'] == 'os.replace' and row['status'] == 'ok']
        assert bool(close_rows) != bool(replace_rows), 'Critical section must read-close OR replace'
        closes += len(close_rows); replacements += len(replace_rows)
        assert all(interval['start'] <= row['monotonic'] <= interval['end'] for row in close_rows + replace_rows)
    assert replacements == FINAL_SEQUENCE + 1
    writer_pid = roles['writer']['pid']
    intent = roles['writer']['barrier_write_intent_monotonic']
    replaced = roles['writer']['barrier_replace_monotonic']
    barrier = [value for value in intervals if value['pid'] == writer_pid and intent <= value['start'] <= replaced]
    assert len(barrier) == 1
    first_reader = roles['reader1']
    assert first_reader['barrier_file_closed_monotonic'] <= barrier[0]['start']
    waits = [row for row in all_rows if row['operation'] == 'mutex' and row['phase'] == 'wait'
             and row['pid'] == writer_pid and intent <= row['monotonic'] <= barrier[0]['start']]
    assert len(waits) == 1 and waits[0]['monotonic'] < first_reader['barrier_file_closed_monotonic']
    waited = barrier[0]['start'] - waits[0]['monotonic']
    assert 0 < waited < 1.0
    return {'status': 'pass', 'critical_sections': len(intervals), 'json_reads_closed_inside_lock': closes,
        'atomic_replacements_inside_lock': replacements, 'overlapping_critical_sections': 0,
        'writer_wait_started_before_reader_close': True, 'barrier_writer_wait_seconds': waited,
        'replacement_started_after_reader_close': True, 'mutex_name': intervals[0]['mutex_name'],
        'log_event_count': len(all_rows), 'runtime_abandoned_or_timeout_case_executed': False}


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('Only one CPU acceptance round; no retry')
    RUN.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + 30.0
    jobs, handles = {}, {}
    report = {'status': 'running', 'attempts': 1, 'automatic_retries': 0,
        'budget_seconds': 30, 'per_child_limit_seconds': 10, 'mutex_wait_limit_seconds': 1,
        'gpu_initialized': False, 'model_loaded': False, 'training_updates': 0,
        'final_examples_read': False, 'events': []}
    save(RUN / 'claim.json', report)
    protected = None
    hard_done = threading.Event()
    def hard_deadline():
        if not hard_done.wait(max(0, deadline - time.perf_counter())):
            save(RUN / 'hard_timeout.json', {'status': 'failed', 'seconds': time.perf_counter() - started})
            os._exit(93)  # Noninherited Job handles close and contain children.
    threading.Thread(target=hard_deadline, daemon=True).start()
    try:
        protected = protected_files()
        save(RUN / 'preservation_before.json', protected)
        report['source_sha256'] = {name: digest(ROOT / name) for name in SOURCES}
        snapshot_dir = RUN / 'source_snapshot'
        snapshot_dir.mkdir()
        for name in SOURCES:
            ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
            shutil.copyfile(ROOT / name, snapshot_dir / Path(name).name)
        old_super = (ROOT / 'src/train/pid_supervision_v5.py').read_text(encoding='utf-8')
        new_super = (ROOT / 'src/train/pid_supervision_v6.py').read_text(encoding='utf-8')
        assert new_super == old_super.replace('from src.train.stage_file_io import read_stage',
                                              'from src.train.stage_file_io_v2 import read_stage')
        old_worker = (ROOT / 'src/train/sft_cache_resume_worker_v2.py').read_text(encoding='utf-8')
        new_worker = (ROOT / 'src/train/sft_cache_resume_worker_v3.py').read_text(encoding='utf-8')
        assert new_worker == old_worker.replace('from src.train.stage_file_io import write_stage',
                                                'from src.train.stage_file_io_v2 import write_stage')
        assert deadline - time.perf_counter() > 12, 'Insufficient total budget to launch children'
        nonce = secrets.token_hex(24)
        directory = RUN / 'concurrent'
        directory.mkdir()
        concurrent_start = time.perf_counter()
        run_stop = concurrent_start + 7.0
        child_deadline = concurrent_start + 9.0
        for role in ('writer', 'reader1', 'reader2'):
            job = JobProcess([str(VENV_PYTHON), '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--role', role],
                ROOT, dict(os.environ, IPC_DIRECTORY=str(directory), IPC_NONCE=nonce,
                           IPC_CHILD_DEADLINE=str(child_deadline)), directory / f'{role}_console.log')
            jobs[role] = job
            handles[role] = job.open_process(job.pid)
            report['events'].append({'event': 'child_started', 'role': role,
                'identity': handles[role].identity(), 'monotonic': time.perf_counter()})
        while not all((directory / f'{role}_identity.json').exists() for role in jobs):
            if time.perf_counter() >= run_stop or any(job.poll() is not None for job in jobs.values()):
                raise RuntimeError('Child startup failed or deadline exceeded')
            time.sleep(0.002)
        for role, handle in handles.items():
            item = read(directory / f'{role}_identity.json')
            assert item['pid'] == handle.pid and item['nonce'] == nonce
            assert normalize(item['executable']) == normalize(NATIVE_PYTHON)
            assert normalize(item['sys_executable']) == normalize(VENV_PYTHON)
        publish(directory / 'start.json', {'nonce': nonce})
        while any(job.poll() is None for job in jobs.values()):
            if any((directory / f'{role}_failure.json').exists() or job.poll() not in (None, 0)
                   for role, job in jobs.items()):
                raise RuntimeError('Concurrent child failed; stop all, no retry')
            if time.perf_counter() >= run_stop:
                raise TimeoutError('Concurrent phase reached 7-second stop budget')
            time.sleep(0.005)
        report['roles'] = {role: read(directory / f'{role}_result.json') for role in jobs}
        report['mutex_audit'] = audit_mutex_logs(directory, report['roles'])
        report['concurrent_seconds'] = time.perf_counter() - concurrent_start
        report['status'] = 'pass'
    except Exception as error:
        report.update(status='failed', error=repr(error), errno=getattr(error, 'errno', None),
                      winerror=getattr(error, 'winerror', None), traceback=traceback.format_exc())
    finally:
        cleanup_deadline = min(deadline, time.perf_counter() + 2.0)
        cleanup_errors, exits = [], []
        # Terminate all survivors first; every retained handle shares ONE wait deadline.
        for role, job in jobs.items():
            try:
                if job.poll() is None:
                    ok = win._TerminateJob(job.job, 125)
                    report['events'].append({'event': 'terminate', 'role': role, 'success': bool(ok),
                        'winerror': 0 if ok else ctypes.get_last_error(), 'monotonic': time.perf_counter()})
                    if not ok:
                        raise ctypes.WinError(ctypes.get_last_error())
            except Exception as error:
                cleanup_errors.append({'role': role, 'phase': 'terminate', 'error': repr(error)})
        for role, handle in handles.items():
            try:
                code = handle.wait(max(0, cleanup_deadline - time.perf_counter()))
                row = {'role': role, 'pid': handle.pid, 'exit_code': code, 'alive': handle.alive(),
                       **exit_timing(handle), 'job_active_pids': jobs[role].active_pids()}
                exits.append(row)
                assert not row['alive'] and not row['job_active_pids'] and row['lifetime_seconds'] <= 10
                assert time.perf_counter() <= cleanup_deadline
                if report['status'] == 'pass':
                    assert code == 0
            except Exception as error:
                cleanup_errors.append({'role': role, 'error': repr(error), 'traceback': traceback.format_exc()})
        for role, job in jobs.items():
            try:
                close_without_new_wait(job)
                if role in handles:
                    handles[role].close()
            except Exception as error:
                cleanup_errors.append({'role': role, 'error': repr(error)})
        report['exits'], report['cleanup_errors'] = exits, cleanup_errors
        if cleanup_errors:
            report['status'] = 'failed'
        try:
            if protected:
                for name, expected in protected.items():
                    assert digest(ROOT / name) == expected, f'Old file changed: {name}'
                report['protected_unique_paths_verified'] = len({str((ROOT / name).resolve()).lower() for name in protected})
            assert not any(name in sys.modules for name in FORBIDDEN)
            assert time.perf_counter() < deadline
        except Exception as error:
            report.update(status='failed', final_audit_error=repr(error), final_audit_traceback=traceback.format_exc())
        report['total_seconds'] = time.perf_counter() - started
        save(RUN / 'report.json', report)
        save(REPORT, report)
        save(RUN / 'artifact_sha256.json', {p.relative_to(RUN).as_posix(): digest(p)
            for p in RUN.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'})
        hard_done.set()
    print(json.dumps({key: report[key] for key in ('status', 'total_seconds', 'cleanup_errors')}, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(child(sys.argv[2]) if sys.argv[1:2] == ['--role'] else main())
