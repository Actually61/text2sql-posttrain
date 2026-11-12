"""One stdlib-only preflight for the mutex-integrated 32-example SFT pilot.

Reuse hash-verified CPU math evidence. Only one native CPU fixture is launched;
neither controller nor fixture imports Torch or opens an evaluation dataset.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
RUN = ROOT / 'outputs/checks/sft_mutex_resume_cpu_v1'
REPORT = ROOT / 'reports/sft_mutex_resume_cpu_checks.json'
CONFIG = ROOT / 'configs/sft_mutex_resume.json'
FORBIDDEN = ('torch', 'transformers', 'peft', 'bitsandbytes')
SOURCES = ['configs/sft_mutex_resume.json', 'src/train/sft_mutex_resume.py',
    'src/train/sft_mutex_resume_worker.py', 'src/train/test_sft_mutex_resume.py',
    'src/train/pid_supervision_v6.py', 'src/train/windows_stage_mutex.py',
    'src/train/stage_file_io_v2.py', 'src/train/stage_file_io.py',
    'src/train/windows_json_ipc.py', 'src/train/windows_job_process_v2.py',
    'src/train/windows_job_exit_wait_v3.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    from src.train.windows_json_ipc import write_json_atomic
    write_json_atomic(path, value)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(mapping):
    for name, expected in mapping.items():
        assert digest(ROOT / name) == expected, f'Frozen evidence changed: {name}'


def fixture():
    """Actual native worker, simulated counter provider, real stage-file mutex."""
    from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack
    from src.train.stage_file_io_v2 import write_stage
    from src.train.windows_json_ipc import read_json
    directory = Path(os.environ['PID_RUN_DIR'])
    nonce = os.environ['PID_RUN_NONCE']
    deadline = float(os.environ['PID_CHILD_DEADLINE'])
    def expire():
        time.sleep(max(0, deadline - time.perf_counter()))
        os._exit(92)
    threading.Thread(target=expire, daemon=True).start()
    def wait_file(name):
        while time.perf_counter() < deadline:
            if (directory / 'stop_request.json').exists():
                raise RuntimeError('CPU fixture received stop request')
            value = read_json(directory / name, missing_ok=True)
            if value is not None:
                assert value['nonce'] == nonce
                return value
            time.sleep(0.01)  # Wait for a protocol publication, not retry failed I/O.
        raise TimeoutError(f'CPU fixture acknowledgement deadline: {name}')
    try:
        write_stage(directory, {'stage': 'cpu_before_identity', 'telemetry_required': False,
                                 'micro_started_monotonic': None})
        membership = wait_file('membership_ack.json')
        assert membership['retained_pids'] == [os.getpid()]
        identity = publish_worker_identity(directory, nonce)
        write(directory / 'fixture_worker.json', {**identity, 'sys_executable': sys.executable,
            'sys_prefix': sys.prefix, 'forbidden_imports': [name for name in FORBIDDEN if name in sys.modules]})
        ack = wait_for_identity_ack(directory, nonce, timeout=min(2.5, deadline - time.perf_counter()))
        assert ack['launcher_pid'] == ack['worker_pid'] == os.getpid()
        write_stage(directory, {'stage': 'cpu_registered', 'telemetry_required': True,
                                 'micro_started_monotonic': None, 'fixture_sequence': 0})
        ready = wait_file('telemetry_ready.json')
        assert ready['worker_pid'] == os.getpid() and ready['valid_sample_count'] >= 3
        for sequence in range(1, 4):
            if time.perf_counter() >= deadline or (directory / 'stop_request.json').exists():
                raise RuntimeError('CPU fixture cannot advance after stop/deadline')
            write_stage(directory, {'stage': f'cpu_sequence_{sequence}', 'telemetry_required': True,
                'micro_started_monotonic': None, 'fixture_sequence': sequence})
            time.sleep(0.06)
        write_stage(directory, {'stage': 'cpu_complete', 'telemetry_required': True,
            'micro_started_monotonic': None, 'fixture_sequence': 4, 'telemetry_complete': True})
        closed = wait_file('telemetry_closed.json')
        assert closed['worker_pid'] == os.getpid() and closed['valid_sample_count'] >= 3
        assert not any(name in sys.modules for name in FORBIDDEN)
        write(directory / 'worker_result.json', {'status': 'pass', 'pid': os.getpid(),
            'torch_imported': False, 'ml_imported': False, 'telemetry_closed': True,
            'mutex_stage_publications': 6, 'automatic_retries': 0})
        return 0
    except Exception as error:
        write(directory / 'fixture_failure.json', {'status': 'failed', 'pid': os.getpid(),
            'error': repr(error), 'errno': getattr(error, 'errno', None),
            'winerror': getattr(error, 'winerror', None), 'traceback': traceback.format_exc()})
        return 2


def reused_prerequisites():
    old_cpu_path = ROOT / 'reports/sft_cache_resume_cpu_checks.json'
    mutex_path = ROOT / 'reports/stage_mutex_cpu_checks.json'
    old_cpu, mutex = read(old_cpu_path), read(mutex_path)
    assert old_cpu['status'] == mutex['status'] == 'pass'
    numerical = old_cpu['checks']['four_update_resume']
    assert numerical['status'] == 'pass' and numerical['exact_comparison']
    assert numerical['maximum_parameter_difference'] == 0
    assert numerical['fifth_update_rejected'] and numerical['ledger_rewind_rejected']
    assert mutex['mutex_audit']['overlapping_critical_sections'] == 0
    assert mutex['mutex_audit']['replacement_started_after_reader_close']
    verify(old_cpu['source_sha256'])
    verify(mutex['source_sha256'])
    verify(old_cpu['checks']['original_32_full_examples']['data_sha256'])
    return {'status': 'pass', 'math_equivalence_reused_not_rerun': True,
        'math_exact_comparison': True, 'math_maximum_parameter_difference': 0,
        'mutex_acceptance_reused_not_rerun': True,
        'reports_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
                           for path in (old_cpu_path, mutex_path)},
        'source_hash_entries_verified': len(old_cpu['source_sha256']) + len(mutex['source_sha256'])}


def original_examples():
    directory = ROOT / 'data/processed/spider_sft_resume_pilot_v1'
    selection, rows = read(directory / 'selection.json'), read(directory / 'features.json')
    by_id = {row['source_index']: row for row in rows}
    ordered = [index for group in selection['groups'] for index in group['source_indices']]
    assert len(rows) == len(by_id) == len(set(ordered)) == 32
    assert ordered == selection['selected_source_indices'] == [row['source_index'] for row in rows]
    groups = []
    for group_index, group in enumerate(selection['groups']):
        assert group['group_index'] == group_index and group['update'] == group_index + 1
        assert len(group['source_indices']) == len(group['sequence_lengths']) == len(group['target_counts']) == 8
        lengths = targets = 0
        for index, length, count in zip(group['source_indices'], group['sequence_lengths'], group['target_counts']):
            row = by_id[index]
            ids, mask, labels = (row[key] for key in ('input_ids', 'attention_mask', 'labels'))
            assert row['project_split'] == 'train'
            assert length == row['sequence_length'] == len(ids) == len(mask) == len(labels)
            assert 2 <= length <= 4096 and all(type(value) is int for value in ids + mask + labels)
            assert all(value >= 0 for value in ids) and all(value == 1 for value in mask)
            assert labels[0] == -100 and all(label == -100 or label == token for label, token in zip(labels, ids))
            assert sum(label != -100 for label in labels[1:]) == count == row['supervised_token_count']
            lengths += length; targets += count
        assert lengths == group['group_input_tokens'] and targets == group['group_target_tokens']
        groups.append({'group_index': group_index, 'input_tokens': lengths, 'target_tokens': targets})
    assert len(groups) == 4 and selection['bucket_counts'] == {'512': 8, '1024': 8, '2048': 8, '4096': 8}
    assert sum(g['input_tokens'] for g in groups) == selection['input_tokens'] == 47654
    assert sum(g['target_tokens'] for g in groups) == selection['target_tokens'] == 1350
    assert max(row['sequence_length'] for row in rows) == 3761
    assert max(row['supervised_token_count'] for row in rows) == 202
    return {'status': 'pass', 'source_indices': ordered, 'groups': groups,
        'input_tokens': 47654, 'target_tokens': 1350, 'selected_examples': 32,
        'unchanged_full_inputs': True, 'truncation_or_deletion': False,
        'data_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
                       for path in (directory / 'selection.json', directory / 'features.json')}}


def syntax_and_config():
    config = read(CONFIG)
    assert (config['selected_examples'], config['batch_size'], config['accumulation_steps'],
            config['maximum_updates'], config['global_update_budget'], config['segment_ranges']) == (32, 1, 8, 4, 4, [[0, 2], [2, 4]])
    assert config['run_dir'] == 'outputs/runs/sft_mutex_resume_v1'
    assert config['prepared_dir'] == 'data/processed/spider_sft_resume_pilot_v1'
    expected = {'total_gpu_window_seconds': 620, 'cpu_check_budget_seconds': 30,
                'minimum_free_mib': 6656, 'allocated_limit_mib': 6144, 'reserved_limit_mib': 7168}
    assert all(config[key] == value for key, value in expected.items())
    limits = config['limits']
    expected_limits = {'budget_profile': 'bounded_sft', 'expected_process_count': 1,
        'total_soft_seconds': 240, 'total_hard_seconds': 300, 'child_self_seconds': 304,
        'case_seconds': 306, 'cleanup_seconds': 2, 'handshake_seconds': 3,
        'micro_soft_seconds': 60, 'micro_hard_seconds': 90,
        'shared_limit_mib': 512, 'shared_duration_seconds': 1}
    assert all(limits[key] == value for key, value in expected_limits.items())
    for name in SOURCES:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
    old_worker = (ROOT / 'src/train/sft_cache_resume_worker_v3.py').read_text(encoding='utf-8')
    new_worker = (ROOT / 'src/train/sft_mutex_resume_worker.py').read_text(encoding='utf-8')
    assert new_worker == old_worker.replace('configs/sft_cache_resume.json', 'configs/sft_mutex_resume.json')
    tree = ast.parse((ROOT / 'src/train/pid_supervision_v6.py').read_text(encoding='utf-8'))
    profiles = [ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'profiles' for target in node.targets)]
    assert len(profiles) == 1 and profiles[0]['bounded_sft'] == (304.0, 306.0)
    return {'status': 'pass', 'config': config, 'worker_training_logic_unchanged': True,
            'model_loaded': False, 'all_sources_parsed_without_execution': True}


def single_native_worker(deadline):
    from src.train.pid_supervision_v6 import supervise
    from src.train.windows_job_process_v2 import VENV_PYTHON, NATIVE_PYTHON, normalize
    directory = RUN / 'single_native_worker'
    sampled = []
    def provider(pid, stage):
        assert read(directory / 'worker_identity.json')['pid'] == pid
        sampled.append(pid)
        return {'pid': pid, 'status': 'ok', 'dedicated_bytes': 0, 'shared_bytes': 0,
                'fixture_sequence': stage.get('fixture_sequence'), 'fixture': 'cpu_injected_not_gpu'}
    limits = {'budget_profile': 'cpu', 'total_soft_seconds': 3.0, 'total_hard_seconds': 4.0,
        'handshake_seconds': 2.5, 'cleanup_seconds': 2.0, 'expected_process_count': 1,
        'child_self_seconds': 8.0, 'case_seconds': 9.5, 'sample_interval_seconds': 0.05,
        'resource_grace_seconds': 0.3, 'telemetry_stale_seconds': 1.0}
    result = supervise([str(VENV_PYTHON), '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--fixture'],
        ROOT, dict(os.environ), directory, limits, sample_provider=provider, suite_deadline=deadline)
    write(directory / 'returned_result.json', result)
    assert result['status'] == 'pass' and result['outcome'] == 'completed', result
    identity, worker = read(directory / 'fixture_worker.json'), read(directory / 'worker_result.json')
    pid = identity['pid']
    assert result['identity_verified'] and result['bound_pid'] == result['worker_pid'] == result['launcher_pid'] == pid
    assert result['exit_code'] == 0 and not result['killed'] and result['telemetry_closed']
    assert not result['errors'] and not result['cleanup_errors'] and result['job_active_pids_after_cleanup'] == []
    assert len(result['role_exits']) == 1 and not result['role_exits'][0]['alive']
    assert 0 <= result['role_exits'][0]['lifetime_seconds'] <= 10
    assert sampled and set(sampled) == {pid} and result['observed_sample_pids'] == [pid]
    assert normalize(identity['executable']) == normalize(NATIVE_PYTHON)
    assert normalize(identity['sys_executable']) == normalize(VENV_PYTHON)
    assert normalize(identity['sys_prefix']) == normalize(ROOT / '.venv')
    assert not identity['forbidden_imports'] and worker['status'] == 'pass' and not worker['ml_imported']
    assert worker['mutex_stage_publications'] == 6
    rows = []
    for path in directory.glob('*.jsonl'):
        text = path.read_text(encoding='utf-8')
        assert not text or text.endswith('\n'), f'Incomplete JSONL: {path}'
        records = [json.loads(line) for line in text.splitlines()]
        if path.name.startswith('stage_ipc_'):
            assert not any(row['status'] == 'failed' and row.get('winerror') != 2 for row in records)
        if path.name == 'pid_memory.jsonl':
            rows = records
    assert any(row.get('fixture_sequence') in (1, 2, 3) for row in rows)
    return {'status': 'pass', 'supervisor': result, 'same_native_launcher_worker_pid': pid,
        'sample_values_are_simulated': True, 'mutex_stage_publications': 6,
        'cuda_initialized': False, 'ml_imported': False}


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('Single-use CPU preflight already claimed; no retry')
    RUN.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + 30.0
    report = {'status': 'running', 'attempts': 1, 'automatic_retries': 0, 'cpu_budget_seconds': 30,
        'model_loaded': False, 'cuda_initialized': False, 'gpu_training_updates': 0,
        'validation_examples_read': False, 'final_examples_read': False, 'checks': {}}
    write(RUN / 'claim.json', report)
    done = threading.Event()
    def expire():
        if not done.wait(max(0, deadline - time.perf_counter())):
            write(RUN / 'cpu_budget_exceeded.json', {'status': 'failed', 'seconds': time.perf_counter() - started})
            os._exit(93)
    threading.Thread(target=expire, daemon=True).start()
    code = 1
    try:
        report['source_sha256'] = {name: digest(ROOT / name) for name in SOURCES}
        snapshot_dir = RUN / 'source_snapshot'
        snapshot_dir.mkdir()
        for name in SOURCES:
            shutil.copyfile(ROOT / name, snapshot_dir / Path(name).name)
        report['checks']['syntax_and_budget'] = syntax_and_config()
        report['checks']['reused_prerequisites'] = reused_prerequisites()
        report['checks']['original_32_full_examples'] = original_examples()
        report['checks']['single_native_worker'] = single_native_worker(deadline)
        verify(report['source_sha256'])
        verify(report['checks']['original_32_full_examples']['data_sha256'])
        verify(report['checks']['reused_prerequisites']['reports_sha256'])
        assert not any(name in sys.modules for name in FORBIDDEN)
        assert time.perf_counter() < deadline
        report['status'] = 'pass'
        code = 0
    except Exception as error:
        report.update(status='failed', error=repr(error), errno=getattr(error, 'errno', None),
                      winerror=getattr(error, 'winerror', None), traceback=traceback.format_exc())
    finally:
        report['seconds'] = time.perf_counter() - started
        report['ml_imported'] = any(name in sys.modules for name in FORBIDDEN)
        write(RUN / 'report.json', report)
        write(REPORT, report)
        write(RUN / 'artifact_sha256.json', {path.relative_to(RUN).as_posix(): digest(path)
            for path in RUN.rglob('*') if path.is_file() and path.name != 'artifact_sha256.json'})
        done.set()
    print(json.dumps({key: report[key] for key in ('status', 'seconds', 'ml_imported')}, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(fixture() if sys.argv[1:] == ['--fixture'] else main())
