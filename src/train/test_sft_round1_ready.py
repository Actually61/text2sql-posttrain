"""One stdlib-only formal-readiness check; no process, model, CUDA or training."""
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
RUN = ROOT / 'outputs/checks/sft_round1_ready_cpu_v1'
REPORT = ROOT / 'reports/sft_round1_ready_cpu_checks.json'
CONFIG = ROOT / 'configs/sft_round1_ready.json'
FORBIDDEN = ('torch', 'transformers', 'peft', 'bitsandbytes')
SOURCES = ['configs/sft_round1_ready.json', 'src/train/sft_round1_ready.py',
    'src/train/sft_round1_ready_worker.py', 'src/train/test_sft_round1_ready.py',
    'src/train/pid_supervision_formal.py', 'src/train/windows_job_exit_wait_formal.py',
    'src/train/windows_stage_mutex.py', 'src/train/stage_file_io_v2.py',
    'src/train/windows_json_ipc.py', 'src/train/cache_release_core.py',
    'src/train/sft_checkpoint.py', 'src/train/sft_update_budget.py',
    'src/train/target_loss.py', 'src/train/sft_train.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(mapping):
    for name, expected in mapping.items():
        assert digest(ROOT / name) == expected, f'Frozen evidence changed: {name}'


def worker_generalization(pilot):
    """The full approved literal change list, without executing either worker."""
    changes = [
        ('independent 32-example cache/resume pilot', '6364-example formal SFT round'),
        ('import os', 'import json\nimport os'),
        ('configs/sft_mutex_resume.json', 'configs/sft_round1_ready.json'),
        ("    root = ROOT / config['run_dir']", "    if os.environ.get('SFT_FORMAL_TRAINING_CONFIRMED') != '1':\n        raise RuntimeError('Formal worker requires explicit controller confirmation')\n    root = ROOT / config['run_dir']"),
        ('segment not in (1, 2)', 'segment not in (1, 2, 3, 4)'),
        ("config['segment_ranges'] != [[0, 2], [2, 4]]", "config['segment_ranges'] != [[0, 200], [200, 400], [400, 600], [600, 796]]"),
        ('Only two fresh processes with two updates each are allowed', 'Only the frozen four-segment 796-update round is allowed'),
        ("ledger['total_budget'] != 4", "ledger['total_budget'] != 796"),
        ("read(ROOT / config['prepared_dir'] / 'selection.json')['groups']", "read(ROOT / config['schedule_path'])['groups']"),
        ("        original = read(ROOT / config['prepared_dir'] / 'features.json')", "        with (ROOT / config['train_path']).open(encoding='utf-8') as stream:\n            original = [json.loads(line) for line in stream]"),
        ("        if len(groups) != 4 or any(len(group['source_indices']) != 8 for group in groups) or len(set(ids)) != 32:", "        if (len(groups) != 796 or any(len(group['source_indices']) != 8 for group in groups[:-1])\n                or len(groups[-1]['source_indices']) != 4 or len(set(ids)) != 6364 or len(original) != 6364):"),
        ('Schedule must contain exactly the original 32 unique examples', 'Schedule must contain the 6364 unique full examples in 795 full groups plus one four-example tail'),
        ('total_budget=4,', 'total_budget=796,'),
        ("root / 'checkpoints/update_000002'", "root / 'checkpoints' / f'update_{start:06d}'"),
        ('len(cache_records) != 8', 'len(cache_records) != len(rows)'),
        ('choices=(1, 2)', 'choices=(1, 2, 3, 4)'),
    ]
    expected = pilot
    for old, new in changes:
        assert expected.count(old) == 1, f'Ambiguous expected worker change: {old}'
        expected = expected.replace(old, new)
    return expected, len(changes)


def source_and_config():
    config = read(CONFIG)
    assert config['gpu_training_enabled'] is False and config['evaluation_enabled'] is False
    assert config['formal_config'] == 'configs/sft_round1.json'
    assert config['train_path'] == 'data/processed/spider_sft_tokens_v1/train_tokens.jsonl'
    assert config['schedule_path'] == 'data/processed/spider_sft_round1_plan_v1/epoch_0.json'
    assert config['segment_ranges'] == [[0, 200], [200, 400], [400, 600], [600, 796]]
    expected = {'selected_examples': 6364, 'expected_examples': 6364, 'expected_databases': 126,
        'batch_size': 1, 'accumulation_steps': 8, 'maximum_updates': 796, 'global_update_budget': 796,
        'expected_tail_group_size': 4, 'minimum_free_mib': 6656, 'minimum_disk_gib': 8,
        'allocated_limit_mib': 6144, 'reserved_limit_mib': 7168,
        'total_gpu_window_seconds': 15660, 'cpu_check_budget_seconds': 60, 'mutex_wait_seconds': 1}
    assert all(config[key] == value for key, value in expected.items())
    assert config['save_after_updates'] == [200, 400, 600, 796]
    limits = config['limits']
    expected_limits = {'budget_profile': 'formal_sft', 'total_soft_seconds': 3600,
        'total_hard_seconds': 3900, 'child_self_seconds': 3904, 'case_seconds': 3906,
        'cleanup_seconds': 2, 'handshake_seconds': 3, 'expected_process_count': 1,
        'micro_soft_seconds': 60, 'micro_hard_seconds': 90,
        'shared_limit_mib': 512, 'shared_duration_seconds': 1}
    assert all(limits[key] == value for key, value in expected_limits.items())
    formal = read(ROOT / config['formal_config'])
    original_formal = read(ROOT / 'reports/sft_round1_preflight.json')
    assert formal == original_formal['config'] and original_formal['status'] == 'pass'
    assert formal['training']['epochs'] == 1 and not formal['data']['truncate'] and not formal['data']['drop_last']
    assert formal['training']['scheduler']['total_updates'] == 796
    for name in SOURCES:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
    pilot = (ROOT / 'src/train/sft_mutex_resume_worker.py').read_text(encoding='utf-8')
    actual = (ROOT / 'src/train/sft_round1_ready_worker.py').read_text(encoding='utf-8')
    expected_worker, change_count = worker_generalization(pilot)
    assert actual == expected_worker, 'Formal worker includes an unapproved change beyond scale/config/paths'
    def core_loop(source):
        loops = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'group_index']
        assert len(loops) == 1
        return ast.dump(loops[0], include_attributes=False)
    normalized_pilot_loop = core_loop(pilot.replace('len(cache_records) != 8', 'len(cache_records) != len(rows)'))
    assert core_loop(actual) == normalized_pilot_loop
    supervisor = ast.parse((ROOT / 'src/train/pid_supervision_formal.py').read_text(encoding='utf-8'))
    profiles = [ast.literal_eval(node.value) for node in ast.walk(supervisor)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'profiles' for target in node.targets)]
    assert len(profiles) == 1 and profiles[0]['formal_sft'] == (3904.0, 3906.0)
    old_exit = (ROOT / 'src/train/windows_job_exit_wait_v3.py').read_text(encoding='utf-8')
    new_exit = (ROOT / 'src/train/windows_job_exit_wait_formal.py').read_text(encoding='utf-8')
    assert new_exit == old_exit.replace('306.0', '3906.0').replace('(0, 306]', '(0, 3906]')
    controller = (ROOT / 'src/train/sft_round1_ready.py').read_text(encoding='utf-8')
    assert '--confirm-formal-training' in controller and 'SFT_FORMAL_TRAINING_CONFIRMED' in controller
    assert not (ROOT / config['run_dir']).exists(), 'CPU readiness must not claim the formal GPU run'
    return {'status': 'pass', 'config': config, 'formal_config_sha256': digest(ROOT / config['formal_config']),
        'worker_allowed_literal_changes': change_count, 'worker_exact_allowed_transform': True,
        'core_loop_ast_equal_except_tail_cache_count': True,
        'normalized_core_loop_ast_sha256': hashlib.sha256(normalized_pilot_loop.encode()).hexdigest(),
        'formal_run_directory_exists': False, 'gpu_training_enabled': False,
        'formal_profile': profiles[0]['formal_sft']}


def full_training_data(config, deadline):
    train_path, schedule_path = ROOT / config['train_path'], ROOT / config['schedule_path']
    prior = read(ROOT / 'reports/sft_round1_preflight.json')
    verify({config['train_path']: prior['protected_sha256'][config['train_path']],
            config['schedule_path']: prior['protected_sha256'][config['schedule_path']]})
    schedule = read(schedule_path)
    assert schedule['version'] == 'sft_round1_v1' and schedule['epoch_index'] == 0 and schedule['seed'] == 42
    assert schedule['train_sha256'] == digest(train_path)
    by_id, databases = {}, set()
    input_total = target_total = 0
    with train_path.open(encoding='utf-8') as stream:
        for line in stream:
            if time.perf_counter() >= deadline:
                raise TimeoutError('CPU readiness deadline while validating full training rows')
            row = json.loads(line)
            index = row['source_index']
            assert index not in by_id and row['project_split'] == 'train'
            ids, mask, labels = (row[key] for key in ('input_ids', 'attention_mask', 'labels'))
            length = row['sequence_length']
            assert 2 <= length <= 4096 and length == len(ids) == len(mask) == len(labels)
            assert all(type(value) is int and value >= 0 for value in ids)
            assert all(type(value) is int and value == 1 for value in mask)
            assert all(type(label) is int and (label == -100 or label == token) for label, token in zip(labels, ids))
            assert labels[0] == -100
            supervised = [label for label in labels[1:] if label != -100]
            assert supervised and supervised[-1] == 151645
            assert len(supervised) == row['supervised_token_count']
            by_id[index] = {'sequence_length': length, 'target_count': len(supervised)}
            databases.add(row['db_id'])
            input_total += length; target_total += len(supervised)
    assert len(by_id) == 6364 and len(databases) == 126
    assert input_total == 3932044 and target_total == 206073
    groups = schedule['groups']
    assert len(groups) == 796
    ordered, all_targets = [], 0
    for zero_index, group in enumerate(groups):
        assert group['update'] == zero_index + 1 and group['epoch_index'] == 0
        ids = group['source_indices']
        expected_size = 4 if zero_index == 795 else 8
        assert len(ids) == len(group['sequence_lengths']) == len(group['target_counts']) == expected_size
        assert [by_id[index]['sequence_length'] for index in ids] == group['sequence_lengths']
        assert [by_id[index]['target_count'] for index in ids] == group['target_counts']
        assert sum(group['target_counts']) == group['group_target_tokens']
        update = zero_index + 1
        expected_lr = 1e-4 * update / 24 if update <= 24 else 1e-4 * (796 - update + 1) / (796 - 24 + 1)
        assert group['learning_rate'] == expected_lr
        ordered.extend(ids); all_targets += group['group_target_tokens']
    assert len(ordered) == len(set(ordered)) == 6364 and set(ordered) == set(by_id)
    assert all_targets == 206073
    hashes = {config['train_path']: digest(train_path), config['schedule_path']: digest(schedule_path)}
    return {'status': 'pass', 'examples': 6364, 'databases': 126, 'groups': groups,
        'source_indices': ordered, 'group_count': 796, 'full_groups': 795, 'tail_group_examples': 4,
        'tail_group_target_tokens': groups[-1]['group_target_tokens'],
        'input_tokens': input_total, 'target_tokens': target_total,
        'padding_present': False, 'arrays_truncated_or_modified': False,
        'source_order_is_frozen_schedule': True, 'data_sha256': hashes}


def reused_evidence():
    numerical_path = ROOT / 'reports/token_accumulation_cpu_checks.json'
    numerical = read(numerical_path)
    assert numerical['status'] == 'pass'
    verify(numerical['source_sha256'])
    tail = [case for case in numerical['case_results'] if case['name'] == 'tail_four_lora_nonzero_b']
    assert len(tail) == 1 and tail[0]['accumulation']['microbatch_count'] == 4
    assert tail[0]['optimizer_state_comparison'] == 'pass' and tail[0]['input_unchanged']
    assert tail[0]['group_has_no_updates_or_gradient_clears']
    pilot_path = ROOT / 'reports/sft_mutex_resume_result.json'
    pilot = read(pilot_path)
    assert pilot['status'] == 'pass' and pilot['worker_launches'] == 2 and pilot['automatic_retries'] == 0
    assert pilot['ledger'] == {'attempted_updates': 4, 'completed_updates': 4,
        'total_budget': 4, 'partial_update_possible': False, 'failed': False}
    verify(pilot['source_sha256'])
    directory = ROOT / pilot['config']['run_dir']
    artifacts = read(directory / 'artifact_sha256.json')
    verify({str(directory / name): expected for name, expected in artifacts.items()})
    restored = read(directory / 'segment_2/restore_verification.json')['verification']
    assert all(restored[key] for key in ('adapter_exact', 'optimizer_exact', 'rng_exact', 'parameter_order_exact', 'cuda_rng_restored'))
    assert all(segment['status'] == 'pass' and segment['exit_code'] == 0 and not segment['job_active_pids_after_cleanup']
               for segment in pilot['segments'])
    previous_preflight = read(ROOT / 'reports/sft_mutex_resume_preflight.json')
    verify(previous_preflight['protected_sha256'])
    return {'status': 'pass', 'cpu_tail_math_reused_without_rerun': True,
        'tail_case': tail[0]['name'], 'gpu_two_plus_two_reused_without_rerun': True,
        'gpu_resume_exact': True, 'gpu_pilot_artifacts_verified': len(artifacts),
        'prior_protected_entries_verified': len(previous_preflight['protected_sha256']),
        'reports_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
                          for path in (numerical_path, pilot_path)}}


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('Formal readiness CPU check is single-use; do not rerun')
    RUN.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + 60.0
    done = threading.Event()
    report = {'status': 'running', 'attempts': 1, 'automatic_retries': 0, 'cpu_budget_seconds': 60,
        'cuda_initialized': False, 'model_loaded': False, 'training_updates': 0, 'child_processes_started': 0,
        'final_examples_read': False, 'validation_examples_read': False, 'checks': {}}
    save(RUN / 'claim.json', report)
    def expire():
        if not done.wait(max(0, deadline - time.perf_counter())):
            save(RUN / 'hard_timeout.json', {'status': 'failed', 'seconds': time.perf_counter() - started})
            os._exit(93)
    threading.Thread(target=expire, daemon=True).start()
    code = 1
    try:
        report['source_sha256'] = {name: digest(ROOT / name) for name in SOURCES}
        snapshot = RUN / 'source_snapshot'
        snapshot.mkdir()
        for name in SOURCES:
            shutil.copyfile(ROOT / name, snapshot / Path(name).name)
        report['checks']['source_and_config'] = source_and_config()
        config = report['checks']['source_and_config']['config']
        report['checks']['full_training_data'] = full_training_data(config, deadline)
        report['data_sha256'] = report['checks']['full_training_data']['data_sha256']
        report['checks']['reused_evidence'] = reused_evidence()
        verify(report['source_sha256']); verify(report['data_sha256'])
        assert not (ROOT / config['run_dir']).exists()
        assert not any(name in sys.modules for name in FORBIDDEN)
        assert time.perf_counter() < deadline
        report['status'] = 'pass'
        code = 0
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    finally:
        report['seconds'] = time.perf_counter() - started
        report['ml_imported'] = any(name in sys.modules for name in FORBIDDEN)
        save(RUN / 'report.json', report)
        save(REPORT, report)
        save(RUN / 'artifact_sha256.json', {path.relative_to(RUN).as_posix(): digest(path)
            for path in RUN.rglob('*') if path.is_file() and path.name != 'artifact_sha256.json'})
        done.set()
    print(json.dumps({key: report[key] for key in ('status', 'seconds', 'ml_imported')}, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
