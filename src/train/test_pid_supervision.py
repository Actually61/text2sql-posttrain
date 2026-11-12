"""CPU integration checks for real Windows launcher/worker identity and Job cleanup."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from src.train.pid_supervision import supervise

SOURCES = ['src/train/pid_supervision.py', 'src/train/windows_job_process.py',
           'src/train/pid_supervision_fixture.py', 'src/train/test_pid_supervision.py',
           'src/train/windows_gpu_memory.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def process_alive(pid):
    """Read-only post-run check. No termination by numeric PID in these tests."""
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100000, False, int(pid))  # SYNCHRONIZE only.
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: exited PID no longer exists.
            return False
        raise ctypes.WinError(error)
    try:
        status = kernel.WaitForSingleObject(handle, 0)
        if status == 0:
            return False
        if status == 258:
            return True
        raise RuntimeError(f'Unexpected process wait status: {status}')
    finally:
        kernel.CloseHandle(handle)


def complete_jsonl(path):
    data = Path(path).read_text(encoding='utf-8')
    if data and not data.endswith('\n'):
        raise AssertionError(f'Incomplete trailing log line: {path}')
    return [json.loads(line) for line in data.splitlines()]


def cpu_provider(directory, called):
    def sample(pid, stage):
        actual = read(directory / 'fixture_worker.json')['pid']
        if pid != actual:
            raise AssertionError(f'Sampler bound to PID {pid}, real worker is {actual}')
        called.append(pid)
        return {'pid': pid, 'status': 'ok', 'dedicated_bytes': 0, 'shared_bytes': 0,
                'fixture': 'cpu_injected_counter_not_real_gpu'}
    return sample


def run_case(attempt, mode):
    directory = attempt / mode
    called = []
    limits = {'total_soft_seconds': 3.0 if mode != 'normal' else 8.0,
              'total_hard_seconds': 4.0 if mode != 'normal' else 10.0,
              'handshake_seconds': 2.5, 'cleanup_seconds': 3.0,
              'resource_grace_seconds': 0.3, 'sample_interval_seconds': 0.05,
              'telemetry_stale_seconds': 1.0}
    command = [sys.executable, '-X', 'utf8', str(ROOT / 'src/train/pid_supervision_fixture.py'), '--launcher', mode]
    result = supervise(command, ROOT, dict(os.environ), directory, limits,
                       sample_provider=cpu_provider(directory, called))
    worker_pid = read(directory / 'fixture_worker.json')['pid']
    fixture_launcher_pid = read(directory / 'fixture_launcher.json')['pid']
    assert worker_pid != result['launcher_pid'], 'Fixture must cover distinct launcher and worker PID'
    assert worker_pid != fixture_launcher_pid
    assert result['job_active_pids_after_cleanup'] == []
    checked_pids = sorted({worker_pid, fixture_launcher_pid, result['launcher_pid'],
                          read(directory / 'fixture_launcher_child.json')['popen_child_pid']})
    assert all(not process_alive(pid) for pid in checked_pids), 'Actual worker or launcher remains alive'
    samples = complete_jsonl(directory / 'pid_memory.jsonl')
    events = complete_jsonl(directory / 'supervisor_events.jsonl')
    assert events, 'No durable supervisor lifecycle evidence'
    assert (directory / 'result.json').exists()
    if mode in ('normal', 'ignore_stop', 'cooperative'):
        assert result['identity_verified'] is True
        assert result['worker_pid'] == result['bound_pid'] == worker_pid
        assert len(samples) >= 3 and result['valid_sample_count'] >= 3
        assert called and set(called) == {worker_pid}
        assert all(row['pid'] == worker_pid for row in samples)
        if mode == 'normal':
            assert result['status'] == 'pass' and result['exit_code'] == 0 and not result['killed']
            assert read(directory / 'telemetry_closed.json')['valid_sample_count'] >= 3
            assert read(directory / 'worker_result.json')['torch_imported'] is False
        elif mode == 'ignore_stop':
            assert result['status'] == 'failed' and result['killed']
            assert not (directory / 'worker_result.json').exists(), 'Ignored stop must not complete normally'
        else:
            assert result['status'] == 'failed' and not result['killed']
            assert read(directory / 'worker_result.json')['status'] == 'stopped'
    else:
        assert result['status'] == 'failed' and result['identity_verified'] is False
        assert result['bound_pid'] is None and not called and not samples
    try:
        supervise(command, ROOT, dict(os.environ), directory, limits,
                  sample_provider=cpu_provider(directory, called))
    except FileExistsError:
        replay_rejected = True
    else:
        raise AssertionError('Reused run directory was accepted')
    return {'mode': mode, 'status': 'pass', 'supervisor_result': result,
            'actual_fixture_worker_pid': worker_pid, 'fixture_launcher_pid': fixture_launcher_pid,
            'all_checked_pids_exited': checked_pids, 'sampler_called_pids': sorted(set(called)),
            'durable_jsonl_samples': len(samples), 'durable_jsonl_events': len(events),
            'replay_rejected': replay_rejected, 'gpu_sample_values_are_simulated': True}


def main():
    destination = ROOT / 'reports/pid_supervision_cpu_checks.json'
    if destination.exists():
        raise FileExistsError('Published CPU report already exists; no overwrite or automatic rerun')
    attempt = ROOT / 'outputs/checks/pid_supervision_cpu' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    attempt.mkdir(parents=True, exist_ok=False)
    for name in SOURCES:
        shutil.copyfile(ROOT / name, attempt / Path(name).name)
    report = {'status': 'running', 'cases': [], 'attempt_directory': str(attempt),
              'torch_imported': False, 'cuda_initialized': False, 'model_loaded': False,
              'training_updates': 0, 'final_examples_read': False,
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}}
    code = 0
    try:
        for mode in ['normal', 'ignore_stop', 'cooperative', 'invalid_nonce', 'forged_pid']:
            report['cases'].append(run_case(attempt, mode))
        assert 'torch' not in sys.modules
        report.update(status='pass', check_count=len(report['cases']))
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        (attempt / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        code = 1
    finally:
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        (attempt / 'report.json').write_text(text, encoding='utf-8')
        destination.write_text(text, encoding='utf-8')
    print(json.dumps({'status': report['status'], 'completed_cases': len(report['cases']),
                      'attempt_directory': str(attempt), 'error': report.get('error')}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
