"""Single four-case CPU integration round; no retries, ML imports, or CUDA."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from src.train.pid_supervision_v2 import supervise
from src.train.windows_job_process_v2 import VENV_PYTHON, NATIVE_PYTHON, normalize

RUN = ROOT / 'outputs/checks/pid_supervision_v2_cpu_v1'
REPORT = ROOT / 'reports/pid_supervision_v2_cpu_checks.json'
MODES = ['normal', 'cooperative', 'ignore_stop', 'invalid_nonce']
SOURCES = ['src/train/pid_supervision_v2.py', 'src/train/pid_supervision_fixture_v2.py',
           'src/train/test_pid_supervision_v2.py', 'src/train/windows_job_process_v2.py',
           'src/train/windows_job_exit_wait.py', 'src/train/windows_job_process.py',
           'src/train/pid_supervision.py', 'src/train/pid_supervision_fixture.py',
           'src/train/test_pid_supervision.py', 'src/train/cache_release_watchdog.py']
LIMITS = {'total_soft_seconds': 3.0, 'total_hard_seconds': 4.0,
          'handshake_seconds': 2.5, 'cleanup_seconds': 2.0,
          'expected_process_count': 2, 'child_self_seconds': 8.0, 'case_seconds': 9.5,
          'sample_interval_seconds': 0.05, 'resource_grace_seconds': 0.3,
          'telemetry_stale_seconds': 1.0}
EXPECTED = {
    'normal': ('completed', 'pass', 'normal_completion', 0, False),
    'cooperative': ('soft_stopped', 'stopped', 'Total soft time limit', 20, False),
    'ignore_stop': ('hard_timeout', 'stopped', 'Total hard time limit', 124, True),
    'invalid_nonce': ('identity_rejected', 'rejected', 'Run nonce mismatch', 124, True),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_preservation():
    protected = read(ROOT / 'reports/cache_release_pilot_preflight.json')['protected_sha256']
    protected.update(read(ROOT / 'reports/pid_supervision_repair_result.json')['source_sha256'])
    protected.update(read(ROOT / 'reports/windows_job_v2_cpu_checks.json')['source_sha256'])
    protected.update(read(ROOT / 'reports/windows_job_v3_preservation_manifest.json'))
    protected.update(read(ROOT / 'reports/windows_job_v3_cpu_checks.json')['source_sha256'])
    protected.update(read(ROOT / 'reports/pid_supervision_v2_preservation_manifest.json'))
    for name, expected in protected.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen source/report/ledger changed: {name}')
    counts = {}
    for old_run in ('outputs/runs/cache_release_pilot_v1', 'outputs/checks/windows_job_v2_cpu_v1',
                    'outputs/checks/windows_job_v3_cpu_v1'):
        manifest = read(ROOT / old_run / 'artifact_sha256.json')
        for name, expected in manifest.items():
            if digest(ROOT / old_run / name) != expected:
                raise ValueError(f'Old run artifact changed: {old_run}/{name}')
        counts[old_run] = len(manifest)
    first_failure = read(ROOT / 'reports/windows_job_v2_before_run.json')['old_cpu_failure_sha256']
    for name, expected in first_failure.items():
        if digest(name) != expected:
            raise ValueError(f'First CPU failure artifact changed: {name}')
    return {'protected_files': len(protected), 'old_run_artifact_counts': counts,
            'first_cpu_failure_files': len(first_failure)}


def complete_jsonl(path, *, missing_ok=False):
    path = Path(path)
    if missing_ok and not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    if text and not text.endswith('\n'):
        raise AssertionError(f'Incomplete trailing JSONL record: {path}')
    return [json.loads(line) for line in text.splitlines()]


def cpu_provider(directory, called):
    def sample(pid, stage):
        actual = read(directory / 'fixture_worker.json')['pid']
        if pid != actual:
            raise AssertionError(f'Sampling wrong process: {pid}, expected worker {actual}')
        called.append(pid)
        return {'pid': pid, 'status': 'ok', 'dedicated_bytes': 0, 'shared_bytes': 0,
                'fixture': 'cpu_injected_counter_not_real_gpu'}
    return sample


def run_case(mode, suite_deadline):
    directory = RUN / mode
    called = []
    command = [str(VENV_PYTHON), '-X', 'utf8', str(ROOT / 'src/train/pid_supervision_fixture_v2.py'),
               '--launcher', mode]
    result = supervise(command, ROOT, dict(os.environ), directory, dict(LIMITS),
                       sample_provider=cpu_provider(directory, called), suite_deadline=suite_deadline)
    # Preserve exactly what the supervisor returned before any acceptance assertion.
    write(directory / 'returned_result.json', result)
    launcher, worker = read(directory / 'fixture_launcher.json'), read(directory / 'fixture_worker.json')
    spawn = read(directory / 'fixture_launcher_child.json')
    expected_outcome, expected_status, reason, code, killed = EXPECTED[mode]
    assert result['outcome'] == expected_outcome, result
    assert result['status'] == expected_status, result
    assert result['stop_reason'] == reason, result
    assert result['exit_code'] == code and result['killed'] is killed, result
    assert not result['errors'] and not result['cleanup_errors'], result
    assert result['automatic_retries'] == 0 and result['worker_launches'] == 1
    assert result['launcher_pid'] == launcher['pid'] != worker['pid']
    assert worker['parent_pid'] == launcher['pid'] and spawn['popen_child_pid'] == worker['pid']
    assert result['job_active_pids_after_cleanup'] == []
    roles = result['role_exits']
    assert len(roles) == 2 and {row['pid'] for row in roles} == {launcher['pid'], worker['pid']}
    for row in roles:
        assert row['alive'] is False and row['exit_code'] == code, row
        assert 0 <= row['lifetime_seconds'] <= 10.0, row
        calculated = (row['exit_time_100ns'] - row['creation_time_100ns']) / 10_000_000
        assert abs(calculated - row['lifetime_seconds']) < 1e-9
    for identity in (launcher, worker):
        assert normalize(identity['executable']) == normalize(NATIVE_PYTHON)
        assert normalize(identity['sys_executable']) == normalize(VENV_PYTHON)
        assert normalize(identity['sys_prefix']) == normalize(ROOT / '.venv')
        assert not identity['forbidden_imports']
        events = complete_jsonl(directory / f"fixture_events_{identity['pid']}.jsonl")
        assert events, 'Missing durable role events'
    samples = complete_jsonl(directory / 'pid_memory.jsonl', missing_ok=mode == 'invalid_nonce')
    supervisor_events = complete_jsonl(directory / 'supervisor_events.jsonl')
    assert supervisor_events and (directory / 'result.json').is_file()
    for path in directory.glob('*.jsonl'):
        complete_jsonl(path)
    if mode == 'invalid_nonce':
        assert result['identity_verified'] is False and result['bound_pid'] is None
        assert result['worker_pid'] is None and not called and not samples
        assert result['sample_count'] == result['valid_sample_count'] == 0
        assert result['observed_sample_pids'] == []
        assert not (directory / 'identity_ack.json').exists()
        assert not (directory / 'telemetry_ready.json').exists()
    else:
        assert result['identity_verified'] is True
        assert result['worker_pid'] == result['bound_pid'] == worker['pid']
        assert result['valid_sample_count'] >= 3 and len(samples) >= 3
        assert called and set(called) == {worker['pid']}
        assert result['observed_sample_pids'] == [worker['pid']]
        assert all(row['pid'] == worker['pid'] and row['sample_source'] == 'injected_cpu_fixture' for row in samples)
        ack = read(directory / 'identity_ack.json')
        assert ack['worker_pid'] == worker['pid']
    if mode in ('normal', 'cooperative'):
        stopped = read(directory / 'worker_result.json')
        assert stopped['status'] == ('pass' if mode == 'normal' else 'stopped')
        assert stopped['torch_imported'] is False
        assert read(directory / 'telemetry_closed.json')['valid_sample_count'] >= 3
    else:
        assert not (directory / 'worker_result.json').exists()
        assert result['termination']['confirmation_seconds'] <= 2.0
        assert result['termination']['job_active_pids_after'] == []
    if time.perf_counter() > suite_deadline:
        raise TimeoutError('Overall 60-second budget expired')
    acceptance = {'mode': mode, 'case_status': 'pass', 'expected_outcome': expected_outcome,
                  'worker_pid': worker['pid'], 'launcher_pid': launcher['pid'],
                  'stop_reason': result['stop_reason'], 'exit_code': result['exit_code'],
                  'role_exits': roles, 'supervisor_result': result,
                  'injected_sample_pids': sorted(set(called)), 'sample_values_are_simulated': True}
    write(directory / 'case_acceptance.json', acceptance)
    return acceptance


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('This single-use integration round was already claimed; no replay')
    suite_started = time.perf_counter()
    suite_deadline = suite_started + 60.0
    RUN.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'acceptance_rounds': 1, 'automatic_retries': 0,
              'case_order': MODES, 'current_case': None, 'cases': [],
              'suite_started_monotonic': suite_started, 'suite_deadline_monotonic': suite_deadline,
              'total_budget_seconds': 60.0, 'case_limits': LIMITS,
              'cuda_initialized': False, 'model_loaded': False, 'training_updates': 0,
              'final_examples_read': False, 'sample_values_are_simulated': True}
    write(RUN / 'claim.json', report)
    exit_status = 1
    try:
        snapshot = RUN / 'source_snapshot'
        snapshot.mkdir()
        for name in SOURCES:
            shutil.copyfile(ROOT / name, snapshot / Path(name).name)
        report['source_sha256'] = {name: digest(ROOT / name) for name in SOURCES}
        report['preservation_before'] = check_preservation()
        write(RUN / 'before_cases.json', report)
        for mode in MODES:
            report['current_case'] = mode
            remaining = suite_deadline - time.perf_counter()
            if remaining < 12.0:
                raise TimeoutError('Cannot start another case without 10s child and 2s wrap-up reserve')
            write(RUN / f'{mode}_started.json', {'mode': mode, 'monotonic': time.perf_counter(),
                  'suite_remaining_seconds': remaining, 'case_launch_number': len(report['cases']) + 1})
            report['cases'].append(run_case(mode, suite_deadline))
            write(RUN / 'progress.json', report)
        assert len(report['cases']) == 4
        assert not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'bitsandbytes'))
        report.update(status='pass', current_case=None, completed_cases=4)
        exit_status = 0
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                      completed_cases=len(report['cases']))
    finally:
        try:
            report['preservation_after'] = check_preservation()
        except Exception as error:
            report.update(status='failed', preservation_error=f'{type(error).__name__}: {error}')
            exit_status = 1
        report['wall_seconds_including_preservation_checks'] = time.perf_counter() - suite_started
        if time.perf_counter() > suite_deadline:
            report.update(status='failed', budget_error='Overall 60-second budget exceeded')
            exit_status = 1
        report['not_started_cases'] = [mode for mode in MODES if not (RUN / f'{mode}_started.json').exists()]
        write(RUN / 'report.json', report)
        write(REPORT, report)
        write(RUN / 'artifact_sha256.json', {p.relative_to(RUN).as_posix(): digest(p)
              for p in RUN.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'})
    print(json.dumps({'status': report['status'], 'completed_cases': len(report['cases']),
                      'current_case': report['current_case'], 'error': report.get('error'),
                      'wall_seconds': report['wall_seconds_including_preservation_checks'],
                      'cuda_initialized': False}, ensure_ascii=False))
    return exit_status


if __name__ == '__main__':
    raise SystemExit(main())
