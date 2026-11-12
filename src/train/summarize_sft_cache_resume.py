"""Read existing evidence and verify hashes; never import ML or launch workers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    run = ROOT / 'outputs/runs/sft_cache_resume_v1'
    report = read(ROOT / 'reports/sft_cache_resume_result.json')
    cpu = read(ROOT / 'reports/sft_cache_resume_cpu_checks.json')
    preflight = read(ROOT / 'reports/sft_cache_resume_preflight.json')
    manifest = read(run / 'artifact_sha256.json')
    hashes = dict(preflight['protected_sha256'])
    hashes.update(cpu['source_sha256'])
    hashes.update({str(run / name): value for name, value in manifest.items()})
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, f'Changed evidence: {name}'
    directory = run / 'segment_1'
    logs = {}
    for path in directory.glob('*.jsonl'):
        content = path.read_text(encoding='utf-8')
        assert not content or content.endswith('\n'), f'Incomplete log: {path}'
        logs[path.name] = [json.loads(line) for line in content.splitlines()]
    events = logs['worker_events.jsonl']
    samples = logs['pid_memory.jsonl']
    supervised = report['segments'][0]
    end = next(row for row in supervised['events'] if row['event'] == 'supervisor_error')
    cuda_stage = next(row for row in events if row.get('stage') == 'cuda_initializing')
    assert report['status'] == 'failed' and report['worker_launches'] == 1 and report['automatic_retries'] == 0
    assert report['ledger']['attempted_updates'] == report['ledger']['completed_updates'] == 0
    assert report['ledger']['failed'] and not (run / 'segment_2').exists() and not (run / 'checkpoints').exists()
    assert supervised['identity_verified'] and supervised['bound_pid'] == supervised['launcher_pid']
    assert supervised['job_active_pids_after_cleanup'] == [] and not supervised['cleanup_errors']
    assert len(supervised['cleanup_exits']) == 1 and not supervised['cleanup_exits'][0]['alive']
    assert all(row['pid'] == supervised['bound_pid'] for row in samples)
    assert all(row['status'] == 'not_registered' and row['dedicated_bytes'] is None
               and row['shared_bytes'] is None for row in samples)
    old_ledgers = {name: read(ROOT / name) for name in (
        'outputs/runs/sft_resume_pilot_v1/progress.json', 'outputs/runs/cache_release_pilot_v1/progress.json')}
    result = {
        'status': 'audit_pass_run_failed', 'training_status': 'failed',
        'cpu_status': cpu['status'], 'cpu_seconds': cpu['seconds'],
        'cpu_resume_exact': cpu['checks']['four_update_resume']['exact_comparison'],
        'cpu_parameter_max_difference': cpu['checks']['four_update_resume']['maximum_parameter_difference'],
        'cpu_single_native_worker_seconds': cpu['checks']['single_native_worker']['supervisor']['wall_seconds'],
        'prepared_examples': 32, 'input_tokens': 47654, 'target_tokens': 1350,
        'worker_launches': 1, 'automatic_retries': 0, 'completed_gpu_updates': 0,
        'loss': None, 'gradients': None, 'cache_release_calls': 0, 'gpu_restore_validated': False,
        'gpu_window_seconds': report['gpu_window_seconds'], 'worker': supervised['cleanup_exits'][0],
        'identity_verified': True, 'observed_sample_pids': supervised['observed_sample_pids'],
        'sample_count': len(samples), 'valid_samples': 0, 'dedicated_peak_mib': None, 'shared_peak_mib': None,
        'allocated_peak_mib': None, 'reserved_peak_mib': None,
        'cuda_initialization_completed': None,
        'cuda_evidence': 'Last persisted stage announces cuda_initializing before torch.cuda.init; no completion evidence. Initial worker_started false is stale.',
        'model_loaded': False,
        'model_evidence': 'No telemetry_ready or cuda_registering stage; fresh_model is gated after those steps.',
        'failure': end['error'], 'error_monotonic': end['monotonic'],
        'cuda_stage_written_monotonic': cuda_stage['monotonic'],
        'error_minus_stage_seconds': end['monotonic'] - cuda_stage['monotonic'],
        'confirmed': 'Supervisor stage.json read raised PermissionError; identity matched; emergency cleanup exit125 and Job empty.',
        'hypothesis': 'Concurrent Windows file replacement/read sharing conflict; exact OS error code and holder were not captured, so cause is not proved.',
        'old_ledger_completed_updates': {name: value['completed_updates'] for name, value in old_ledgers.items()},
        'protected_hash_entries_verified': len(hashes),
        'unique_protected_paths_verified': len({str((ROOT / name).resolve()).lower() for name in hashes}),
        'current_run_artifact_count': len(manifest), 'complete_jsonl_logs': len(logs),
        'final_examples_read': False, 'validation_examples_read': False,
        'next_scope': 'CPU-only bounded IPC fix and concurrent-read acceptance, no GPU or training replay.'}
    destination = ROOT / 'reports/sft_cache_resume_readonly_audit.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('status', 'completed_gpu_updates', 'gpu_window_seconds',
        'sample_count', 'valid_samples', 'unique_protected_paths_verified', 'current_run_artifact_count')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
