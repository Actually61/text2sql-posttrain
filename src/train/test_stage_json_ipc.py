"""One CPU-only writer/two-reader round; no CUDA/ML imports or retries."""
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
from src.train.stage_file_io import read_stage, write_stage, stage_evidence
from src.train.windows_job_process_v2 import JobProcess, ProcessHandle, VENV_PYTHON, NATIVE_PYTHON, normalize
from src.train.windows_job_exit_wait_v3 import exit_timing, close_without_new_wait
from src.train import windows_job_process as win

RUN = ROOT / 'outputs/checks/stage_json_ipc_v1'
REPORT = ROOT / 'reports/stage_json_ipc_cpu_checks.json'
SOURCES = ['src/train/windows_json_ipc.py', 'src/train/stage_file_io.py',
    'src/train/pid_supervision_v5.py', 'src/train/sft_cache_resume_worker_v2.py',
    'src/train/test_stage_json_ipc.py']
FORBIDDEN = ('torch', 'transformers', 'peft', 'bitsandbytes')
FINAL_SEQUENCE = 102


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


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
            time.sleep(0.002)  # Wait for protocol state, never retry failed I/O.
    started = time.perf_counter()
    evidence = stage_evidence(directory)
    try:
        with ProcessHandle(os.getpid()) as handle:
            identity = handle.identity()
        save(directory / f'{role}_identity.json', {**identity, 'role': role,
            'sys_executable': sys.executable, 'nonce': nonce, 'started_monotonic': started})
        wait_value(directory / 'start.json')
        if role == 'writer':
            write_stage(directory, snapshot(0, nonce))
            for reader in ('reader1', 'reader2'):
                wait_value(directory / f'{reader}_held.json')
            write_stage(directory, snapshot(1, nonce))
            replaced = time.perf_counter()
            save(directory / 'replace_complete.json', {'nonce': nonce, 'monotonic': replaced})
            for reader in ('reader1', 'reader2'):
                wait_value(directory / f'{reader}_released.json')
            for sequence in range(2, FINAL_SEQUENCE):
                guard()
                write_stage(directory, snapshot(sequence, nonce))
                time.sleep(0.004)
            write_stage(directory, snapshot(FINAL_SEQUENCE, nonce, True))
            for reader in ('reader1', 'reader2'):
                wait_value(directory / f'{reader}_done.json')
            details = {'published_versions': FINAL_SEQUENCE + 1, 'barrier_replace_monotonic': replaced}
        else:
            while read_stage(directory) is None:
                guard(); time.sleep(0.002)
            with open_shared_binary(directory / 'stage.json', evidence=evidence) as held:
                old = json.loads(held.read().decode('utf-8'))
                assert check(old, nonce) == 0
                acquired = time.perf_counter()
                save(directory / f'{role}_held.json', {'nonce': nonce, 'monotonic': acquired})
                while True:
                    guard()
                    fresh = read_stage(directory)
                    if check(fresh, nonce) == 1:
                        break
                    time.sleep(0.002)
                observed_new = time.perf_counter()
                wait_value(directory / 'replace_complete.json')
                held.seek(0)
                old_again = json.loads(held.read().decode('utf-8'))
                assert old_again == old
            released = time.perf_counter()
            save(directory / f'{role}_released.json', {'nonce': nonce, 'monotonic': released})
            seen, reads, last = {0, 1}, 2, 1
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
            details = {'read_count': reads, 'distinct_versions': len(seen), 'final_sequence': last,
                'held_sequence_after_replace': old_again['sequence'], 'new_open_sequence_during_hold': fresh['sequence'],
                'held_from_monotonic': acquired, 'new_seen_monotonic': observed_new, 'released_monotonic': released}
            save(directory / f'{role}_done.json', {'nonce': nonce, 'monotonic': time.perf_counter()})
        assert not any(name in sys.modules for name in FORBIDDEN)
        save(directory / f'{role}_result.json', {'status': 'pass', 'role': role, 'pid': os.getpid(),
            'seconds': time.perf_counter() - started, 'ml_imported': False, 'automatic_retries': 0, **details})
        return 0
    except Exception as error:
        save(directory / f'{role}_failure.json', {'status': 'failed', 'role': role, 'pid': os.getpid(),
            'errno': getattr(error, 'errno', None), 'winerror': getattr(error, 'winerror', None),
            'error': repr(error), 'traceback': traceback.format_exc(), 'monotonic': time.perf_counter()})
        return 2


def protected_files():
    old = ROOT / 'outputs/runs/sft_cache_resume_v1'
    mapping = dict(read(ROOT / 'reports/sft_cache_resume_preflight.json')['protected_sha256'])
    mapping.update(read(ROOT / 'reports/sft_cache_resume_cpu_checks.json')['source_sha256'])
    mapping.update({str(old / name): sha for name, sha in read(old / 'artifact_sha256.json').items()})
    for path in [old / 'artifact_sha256.json', *ROOT.glob('reports/sft_cache_resume*.json'),
                 ROOT / 'reports/sft_cache_resume_walkthrough.md']:
        mapping[str(path)] = digest(path)
    for name, sha in mapping.items():
        assert digest(ROOT / name) == sha, f'Historical file changed: {name}'
    return mapping


def negative_controls():
    """Expected errors are assertions about propagation, not failed runs/retries."""
    evidence = []
    bad = RUN / 'invalid.json'
    bad.write_bytes(b'{broken json')
    try:
        read_json(bad, evidence=evidence.append)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError('Malformed JSON was silently accepted')
    # A separate, deliberately exclusive handle creates a real WinError 32.
    from src.train import windows_json_ipc as ipc
    locked = RUN / 'exclusive.json'
    locked.write_text('{}', encoding='utf-8')
    handle = ipc._CreateFile(str(locked), ipc._GENERIC_READ, 0, None, ipc._OPEN_EXISTING, ipc._NORMAL, None)
    if handle == ipc._INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        try:
            read_json(locked, evidence=evidence.append)
        except OSError as error:
            assert error.winerror == 32, error
        else:
            raise AssertionError('A real sharing violation was hidden')
    finally:
        if not ipc._CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())
    failed = [row for row in evidence if row['operation'] == 'read_json' and row['status'] == 'failed']
    save(RUN / 'negative_control_evidence.json', evidence)
    assert len(failed) == 2 and all(row['traceback'] and row['automatic_retries'] == 0 for row in failed)
    assert any(row['winerror'] == 32 and row['errno'] == 13 for row in failed)
    return {'bad_json_propagated': True, 'real_sharing_violation_propagated': True,
            'expected_failures': failed, 'historical_error_reproduced': False}


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('Only one CPU acceptance round; no retry')
    RUN.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + 30.0
    jobs, handles = {}, {}
    report = {'status': 'running', 'attempts': 1, 'automatic_retries': 0,
        'budget_seconds': 30, 'per_child_limit_seconds': 10, 'gpu_initialized': False,
        'model_loaded': False, 'training_updates': 0, 'final_examples_read': False, 'events': []}
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
        old_super = (ROOT / 'src/train/pid_supervision_v4.py').read_text(encoding='utf-8')
        new_super = (ROOT / 'src/train/pid_supervision_v5.py').read_text(encoding='utf-8')
        assert new_super == old_super.replace('from src.train import windows_job_process as win',
            'from src.train import windows_job_process as win\nfrom src.train.stage_file_io import read_stage').replace(
            "read_optional(directory / 'stage.json')", 'read_stage(directory)')
        old_worker = (ROOT / 'src/train/sft_cache_resume_worker.py').read_text(encoding='utf-8')
        new_worker = (ROOT / 'src/train/sft_cache_resume_worker_v2.py').read_text(encoding='utf-8')
        assert new_worker == old_worker.replace('from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack',
            'from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack\nfrom src.train.stage_file_io import write_stage').replace(
            "write(directory / 'stage.json',", 'write_stage(directory,')
        report['negative_controls'] = negative_controls()
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
        write_json_atomic(directory / 'start.json', {'nonce': nonce})
        while any(job.poll() is None for job in jobs.values()):
            if any((directory / f'{role}_failure.json').exists() or job.poll() not in (None, 0)
                   for role, job in jobs.items()):
                raise RuntimeError('Concurrent child failed; stop all, no retry')
            if time.perf_counter() >= run_stop:
                raise TimeoutError('Concurrent phase reached 7-second stop budget')
            time.sleep(0.005)
        report['roles'] = {role: read(directory / f'{role}_result.json') for role in jobs}
        replaced = report['roles']['writer']['barrier_replace_monotonic']
        for role in ('reader1', 'reader2'):
            result = report['roles'][role]
            assert result['held_from_monotonic'] < replaced < result['released_monotonic']
            assert result['held_sequence_after_replace'] == 0 and result['new_open_sequence_during_hold'] == 1
            assert result['final_sequence'] == FINAL_SEQUENCE
        report['concurrent_seconds'] = time.perf_counter() - concurrent_start
        report['status'] = 'pass'
    except Exception as error:
        report.update(status='failed', error=repr(error), errno=getattr(error, 'errno', None),
                      winerror=getattr(error, 'winerror', None), traceback=traceback.format_exc())
    finally:
        cleanup_deadline = min(deadline, time.perf_counter() + 2.0)
        # Terminate every surviving Job first, then wait all retained handles on ONE deadline.
        cleanup_errors, exits = [], []
        for role, job in jobs.items():
            if job.poll() is None:
                ok = win._TerminateJob(job.job, 125)
                report['events'].append({'event': 'terminate', 'role': role, 'success': bool(ok),
                    'winerror': 0 if ok else ctypes.get_last_error(), 'monotonic': time.perf_counter()})
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
            for path in (RUN / 'concurrent').glob('*.jsonl'):
                content = path.read_text(encoding='utf-8')
                assert not content or content.endswith('\n'), f'Incomplete IPC log: {path}'
                rows = [json.loads(line) for line in content.splitlines()]
                # Explicit pre-publication missing is permitted; other I/O errors are not.
                assert not any(row['status'] == 'failed' and row['winerror'] != 2 for row in rows)
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
