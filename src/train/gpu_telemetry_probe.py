"""Single model-free GPU telemetry attempt; CPU preflight never imports Torch."""
from __future__ import annotations
import ast
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Controller remains CPU-only; child gets explicit device 0.
from src.train.pid_supervision_v3 import supervise
from src.train.windows_job_process_v2 import VENV_PYTHON, NATIVE_PYTHON, normalize, interpreter_evidence
from src.train.windows_gpu_memory import telemetry_preflight
from src.train.test_pid_supervision_v2 import check_preservation as previous_preservation

RUN = ROOT / 'outputs/checks/gpu_telemetry_probe_v1'
REPORT = ROOT / 'reports/gpu_telemetry_probe_result.json'
STAGES = ['cuda_context_only', 'tensor_live', 'tensor_released']
LIMITS = {'budget_profile': 'gpu_telemetry', 'total_soft_seconds': 20.0, 'total_hard_seconds': 22.0,
          'handshake_seconds': 3.0, 'cleanup_seconds': 2.0, 'expected_process_count': 2,
          'child_self_seconds': 25.0, 'case_seconds': 27.0, 'sample_interval_seconds': 0.1,
          'resource_grace_seconds': 0.5, 'telemetry_stale_seconds': 2.0}
SOURCES = ['src/train/gpu_telemetry_probe.py', 'src/train/gpu_telemetry_probe_worker.py',
           'src/train/pid_supervision_v3.py', 'src/train/windows_job_exit_wait_v2.py',
           'src/train/pid_supervision_v2.py', 'src/train/windows_job_exit_wait.py',
           'src/train/windows_job_process_v2.py', 'src/train/windows_job_process.py',
           'src/train/pid_supervision.py', 'src/train/cache_release_watchdog.py',
           'src/train/windows_gpu_memory.py', 'src/train/test_pid_supervision_v2.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def preservation():
    previous = previous_preservation()
    frozen = read(ROOT / 'reports/pid_supervision_v2_cpu_checks.json')['source_sha256']
    frozen.update(read(ROOT / 'reports/gpu_telemetry_preservation_manifest.json'))
    for name, expected in frozen.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen integration source/report changed: {name}')
    prior = ROOT / 'outputs/checks/pid_supervision_v2_cpu_v1'
    manifest = read(prior / 'artifact_sha256.json')
    for name, expected in manifest.items():
        if digest(prior / name) != expected:
            raise ValueError(f'Old CPU integration artifact changed: {name}')
    return {'prior_protection': previous, 'additional_source_report_entries': len(frozen),
            'cpu_integration_artifacts': len(manifest)}


def jsonl(path):
    value = Path(path).read_text(encoding='utf-8')
    if value and not value.endswith('\n'):
        raise ValueError(f'Incomplete JSONL: {path}')
    return [json.loads(line) for line in value.splitlines()]


def cpu_preflight():
    parsed = {}
    for name in SOURCES:
        source = (ROOT / name).read_text(encoding='utf-8-sig')
        ast.parse(source, filename=name)  # Syntax only; does not execute worker/CUDA code.
        parsed[name] = hashlib.sha256(source.encode('utf-8')).hexdigest()
    tree = ast.parse((ROOT / 'src/train/gpu_telemetry_probe_worker.py').read_text(encoding='utf-8'))
    allocations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                   and node.func.value.id == 'torch' and node.func.attr == 'empty']
    assert len(allocations) == 1, 'Only one explicit GPU test tensor allocation is allowed'
    files = interpreter_evidence()
    expected = read(ROOT / 'reports/sft_resume_pilot_preflight.json')['packages']
    versions = {name: metadata.version(name) for name in expected}
    counters = telemetry_preflight()
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    selected = [line for line in inventory.splitlines() if line.split(',')[0].strip() == '0']
    inventory_ok = len(selected) == 1 and int(selected[0].split(',')[-1].strip()) >= 1024
    record = {'status': 'evidence_recorded', 'interpreter_files': files, 'package_versions': versions,
              'syntax_checked_files': list(parsed), 'explicit_test_tensor_allocation_sites': len(allocations),
              'counter_provider': counters, 'nvidia_smi': inventory, 'minimum_free_mib': 1024,
              'controller_torch_imported': 'torch' in sys.modules, 'cuda_context_created': False}
    write(RUN / 'cpu_preflight.json', record)
    assert all(row['matches'] for row in files) and versions == expected
    assert counters['status'] == 'available' and inventory_ok
    assert not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'bitsandbytes'))
    return record


def validate_probe(supervision, gpu_started):
    directory = RUN / 'supervised'
    worker = read(directory / 'worker_result.json')
    write(RUN / 'worker_result_copy.json', worker)
    gate = read(directory / 'gpu_gate.json')
    identity = read(directory / 'fixture_worker.json')
    launcher = read(directory / 'fixture_launcher.json')
    rows = jsonl(directory / 'pid_memory.jsonl')
    events = jsonl(directory / 'supervisor_events.jsonl')
    for path in directory.glob('*.jsonl'):
        jsonl(path)
    assert supervision['status'] == 'pass' and supervision['outcome'] == 'completed', supervision
    assert supervision['exit_code'] == 0 and not supervision['killed']
    assert not supervision['errors'] and not supervision['cleanup_errors']
    assert supervision['worker_launches'] == 1 and supervision['automatic_retries'] == 0
    assert supervision['identity_verified'] and supervision['bound_pid'] == worker['pid'] == identity['pid']
    assert supervision['launcher_pid'] == launcher['pid'] != worker['pid']
    assert identity['parent_pid'] == launcher['pid']
    for item in (identity, launcher):
        assert normalize(item['executable']) == normalize(NATIVE_PYTHON)
        assert normalize(item['sys_executable']) == normalize(VENV_PYTHON)
        assert not item['forbidden_imports']
    assert worker['status'] == 'pass' and worker['cuda_initialized']
    assert gate['worker_pid'] == worker['pid'] and not gate['torch_imported']
    assert not gate['cuda_initialization_requested'] and not worker['cuda_initialized_before_explicit_init']
    verified_at = next(row['monotonic'] for row in events if row['event'] == 'worker_identity_verified')
    assert verified_at <= worker['identity_ack_received_monotonic'] <= worker['torch_import_started_monotonic']
    assert worker['torch_import_finished_monotonic'] <= worker['cuda_init_started_monotonic'] < worker['cuda_init_finished_monotonic']
    assert worker['test_tensor_bytes'] == worker['test_tensor_requested_bytes'] == 1024 * 1024
    assert worker['allocation_attempts'] == worker['successful_test_allocations'] == 1
    assert worker['test_tensor_released'] and not worker['forbidden_imports']
    assert not worker['model_loaded'] and worker['training_updates'] == 0 and not worker['final_examples_read']
    assert all(row['pid'] == worker['pid'] and row['sample_source'] == 'Windows PDH' for row in rows)
    assert all(row['status'] in ('ok', 'not_registered') for row in rows)
    assert all(row['dedicated_bytes'] is None and row['shared_bytes'] is None
               for row in rows if row['status'] == 'not_registered')
    valid = [row for row in rows if row['status'] == 'ok']
    assert valid and all(row['dedicated_bytes'] >= 0 and row['shared_bytes'] >= 0 for row in valid)
    assert [phase['stage'] for phase in worker['phases']] == STAGES
    phases = []
    for phase in worker['phases']:
        observed = [row for row in rows if row['stage'] == phase['stage']
                    and phase['started_monotonic'] <= row['sampled_monotonic'] <= phase['finished_monotonic']]
        matched = [row for row in observed if row['status'] == 'ok']
        assert len(matched) >= 3 and phase['valid_sample_count'] >= 3
        for timestamp in phase['sample_timestamps']:
            assert any(row['sampled_monotonic'] == timestamp for row in matched)
        phases.append({'stage': phase['stage'], 'valid_samples': len(matched),
            'dedicated_peak_mib': max(row['dedicated_bytes'] for row in matched) / 2**20,
            'shared_peak_mib': max(row['shared_bytes'] for row in matched) / 2**20,
            'torch_memory': phase['torch_memory'], 'test_tensor_live_bytes': phase['test_tensor_live_bytes']})
    baseline, live, released = worker['phases']
    assert live['torch_memory']['allocated_bytes'] - baseline['torch_memory']['allocated_bytes'] == 1024 * 1024
    assert released['torch_memory']['allocated_bytes'] == baseline['torch_memory']['allocated_bytes']
    assert len(supervision['role_exits']) == 2 and supervision['job_active_pids_after_cleanup'] == []
    assert all(not row['alive'] and row['exit_code'] == 0 and 0 <= row['lifetime_seconds'] <= 27
               for row in supervision['role_exits'])
    elapsed = time.perf_counter() - gpu_started
    assert elapsed <= 30 and supervision['wall_seconds'] <= 27
    return {'status': 'pass', 'worker_pid': worker['pid'], 'launcher_pid': launcher['pid'],
        'gpu_window_to_validation_seconds': elapsed, 'supervisor_seconds': supervision['wall_seconds'],
        'torch_import_seconds': worker['torch_import_finished_monotonic'] - worker['torch_import_started_monotonic'],
        'cuda_initialization_seconds': worker['cuda_init_finished_monotonic'] - worker['cuda_init_started_monotonic'],
        'test_tensor_bytes': worker['test_tensor_bytes'], 'gpu': worker['gpu'], 'phases': phases,
        'sample_count': len(rows), 'valid_sample_count': len(valid),
        'not_registered_count': sum(row['status'] == 'not_registered' for row in rows),
        'unavailable_count': sum(row['status'] == 'unavailable' for row in rows),
        'ambiguous_or_invalid_count': sum(row['status'] == 'ambiguous_or_invalid_counter' for row in rows),
        'dedicated_peak_mib': max(row['dedicated_bytes'] for row in valid) / 2**20,
        'shared_peak_mib': max(row['shared_bytes'] for row in valid) / 2**20,
        'role_exits': supervision['role_exits'], 'sample_source': 'Windows PDH; real counters, not CPU simulation',
        'limitations': 'Sampled WDDM per-process accounting; allocator and CUDA context are additional to the 1MiB test tensor; no model or training evidence.'}


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('This one-shot GPU check already has a claim; do not rerun')
    whole_started = time.perf_counter()
    RUN.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'gpu_attempts': 0, 'automatic_retries': 0, 'gpu_window_budget_seconds': 30,
              'limits': LIMITS, 'model_loaded': False, 'training_updates': 0, 'final_examples_read': False,
              'controller_cuda_initialized': False, 'worker_cuda_initialized': False,
              'budget_clock': '30 seconds from process-tree launch request through validation; CPU preservation preflight is separate'}
    write(RUN / 'claim.json', report)
    code, gpu_started = 1, None
    try:
        snapshot = RUN / 'source_snapshot'
        snapshot.mkdir()
        for name in SOURCES:
            shutil.copyfile(ROOT / name, snapshot / Path(name).name)
        report['source_sha256'] = {name: digest(ROOT / name) for name in SOURCES}
        report['preservation_before'] = preservation()
        report['cpu_preflight'] = cpu_preflight()
        write(RUN / 'before_gpu.json', report)
        report['gpu_attempts'] = 1
        report['worker_cuda_initialized'] = None  # Unknown until durable worker evidence is available.
        gpu_started = time.perf_counter()
        write(RUN / 'gpu_launch_claim.json', {'attempt': 1, 'started_monotonic': gpu_started,
              'deadline_monotonic': gpu_started + 30.0, 'limits': LIMITS})
        command = [str(VENV_PYTHON), '-X', 'utf8', str(ROOT / 'src/train/gpu_telemetry_probe_worker.py'), '--launcher']
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
        result = supervise(command, ROOT, child_env, RUN / 'supervised', dict(LIMITS),
                           sample_provider=None, suite_deadline=gpu_started + 30.0)
        report['supervisor'] = result
        write(RUN / 'supervisor_returned.json', result)
        report['measurement'] = validate_probe(result, gpu_started)
        report.update(status='pass', worker_cuda_initialized=True)
        assert 'torch' not in sys.modules
        code = 0
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        for evidence in ('worker_result.json', 'probe_progress.json'):
            path = RUN / 'supervised' / evidence
            if path.exists():
                saved = read(path)
                report['partial_worker_evidence'] = saved
                report['worker_cuda_initialized'] = saved.get('cuda_initialized')
                break
    finally:
        if gpu_started is not None:
            report['gpu_window_seconds_before_cpu_preservation'] = time.perf_counter() - gpu_started
            if report['gpu_window_seconds_before_cpu_preservation'] > 30:
                report['status'], report['budget_error'], code = 'failed', 'GPU selfcheck exceeded 30 seconds', 1
        try:
            report['preservation_after'] = preservation()
        except Exception as error:
            report.update(status='failed', preservation_error=str(error)); code = 1
        report['controller_total_seconds_including_cpu_checks'] = time.perf_counter() - whole_started
        write(RUN / 'report.json', report)
        write(REPORT, report)
        write(RUN / 'artifact_sha256.json', {p.relative_to(RUN).as_posix(): digest(p)
              for p in RUN.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'})
    print(json.dumps({'status': report['status'], 'gpu_attempts': report['gpu_attempts'],
        'gpu_window_seconds': report.get('gpu_window_seconds_before_cpu_preservation'),
        'error': report.get('error'), 'worker_cuda_initialized': report['worker_cuda_initialized']}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
