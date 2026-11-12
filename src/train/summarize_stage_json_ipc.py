"""Read-only audit after the single failed CPU round; no processes or CUDA."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/checks/stage_json_ipc_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    report = read(ROOT / 'reports/stage_json_ipc_cpu_checks.json')
    failure = read(RUN / 'concurrent/writer_failure.json')
    manifest = read(RUN / 'artifact_sha256.json')
    preserved = read(RUN / 'preservation_before.json')
    for name, sha in preserved.items():
        assert digest(ROOT / name) == sha
    for name, sha in manifest.items():
        assert digest(RUN / name) == sha
    for name, sha in report['source_sha256'].items():
        assert digest(ROOT / name) == sha
    logs = {}
    for path in (RUN / 'concurrent').glob('*.jsonl'):
        text = path.read_text(encoding='utf-8')
        assert not text or text.endswith('\n')
        logs[path.name] = [json.loads(line) for line in text.splitlines()]
    errors = [row for rows in logs.values() for row in rows if row['status'] == 'failed' and row['winerror'] != 2]
    assert report['status'] == 'failed' and report['attempts'] == 1
    assert failure['errno'] == 13 and failure['winerror'] == 5 and 'os.replace' in failure['traceback']
    assert len(errors) == 1 and errors[0]['operation'] == 'write_json_atomic' and errors[0]['phase'] == 'os.replace'
    assert all(not row['alive'] and row['exit_code'] == 125 and not row['job_active_pids']
               and row['lifetime_seconds'] <= 10 for row in report['exits'])
    assert all(row['automatic_retries'] == 0 for rows in logs.values() for row in rows)
    result = {
        'status': 'evidence_verified_acceptance_failed', 'fixed_and_validated': False,
        'cpu_rounds': 1, 'automatic_retries': 0, 'seconds': report['total_seconds'],
        'actual_failure': failure, 'ipc_failure_evidence': errors,
        'child_exits': report['exits'], 'cleanup_errors': report['cleanup_errors'],
        'source_sha256': report['source_sha256'], 'artifact_count_verified': len(manifest),
        'old_unique_paths_verified': len({str((ROOT / name).resolve()).lower() for name in preserved}),
        'complete_jsonl_files': len(logs), 'ipc_log_events': sum(len(rows) for rows in logs.values()),
        'last_published_sequence': read(RUN / 'concurrent/stage.json')['sequence'],
        'reader_held_markers': {role: read(RUN / f'concurrent/{role}_held.json') for role in ('reader1', 'reader2')},
        'gpu_capacity_caused_this_cpu_failure': False, 'cuda_initialized': False, 'model_loaded': False,
        'gpu_training_updates': 0, 'old_sft_cache_ledger': read(ROOT / 'outputs/runs/sft_cache_resume_v1/progress.json'),
        'final_examples_read': False,
        'confirmed': 'Initial stage0 publication succeeded. Both readers held share7 handles. os.replace to publish stage1 returned WinError5/errno13; no second attempt.',
        'not_confirmed': 'Historical read error did not capture winerror. Current write denial does not identify the exact OS/filesystem/filter cause or prove the same historical error.',
        'next_minimum_change_unverified': 'Serialize stage read-open/read/close and atomic replacement with one named Windows mutex; bounded acquisition, fail on abandoned mutex or timeout; validate in a new single CPU round.',
        'background_app_cleanup': {'closed_application': 'NVIDIA Overlay', 'closed_pids': [9580, 30576, 32100, 32120, 32220],
            'free_mib_before': 6638, 'free_mib_immediately_after': 6825, 'difference_mib': 187,
            'limitation': 'A point-in-time reading; background GPU usage changes. This does not fix the file I/O failure.'}}
    with (ROOT / 'reports/stage_json_ipc_readonly_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('status', 'seconds', 'artifact_count_verified',
        'old_unique_paths_verified', 'complete_jsonl_files', 'ipc_log_events')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
