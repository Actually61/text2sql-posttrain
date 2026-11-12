"""Frozen weighted formal SFT: 6364 full rows, 796 updates, four saved segments."""
from __future__ import annotations
import argparse
import ast
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train.cache_release_pilot import read, digest, write, verify, gate_resources
from src.train.sft_round1_resource_v3 import stable_resource_gate, lines
from src.train.pid_supervision_formal_v2 import supervise
from src.train.windows_job_process_v2 import VENV_PYTHON, interpreter_evidence
from src.train.windows_gpu_memory import telemetry_preflight
from src.train.sft_round2_binding_v1 import build_binding, verify_binding, verify_group_manifest

CONFIG = ROOT / 'configs/sft_round2_formal_v1.json'
FREEZE = ROOT / 'reports/sft_round2_formal_v1_freeze.json'
SOURCES = [
    'configs/sft_round2_formal_v1.json', 'src/train/sft_round2_formal_v1.py',
    'src/train/sft_round2_formal_v1_worker.py', 'src/train/test_round2_formal_cpu_v1.py',
    'src/train/sft_round2_weight_core_v1.py', 'src/train/sft_round2_binding_v1.py',
    'src/train/pid_supervision_formal_v2.py', 'src/train/windows_job_exit_wait_formal.py',
    'src/train/windows_job_process_v2.py', 'src/train/windows_job_process.py',
    'src/train/cache_release_pilot.py', 'src/train/cache_release_watchdog.py',
    'src/train/sft_checkpoint.py', 'src/train/sft_checkpoint_cpu_v1.py',
    'src/train/sft_update_budget.py', 'src/train/sft_train.py', 'src/train/sft_round1_plan.py',
    'src/train/target_loss.py', 'src/train/token_accumulation.py', 'src/train/qlora_stability.py',
    'src/train/qlora_smoke.py', 'src/train/qlora_step.py', 'src/data/sft_features.py',
    'src/train/windows_stage_mutex.py', 'src/train/stage_file_io_v2.py',
    'src/train/stage_file_io.py', 'src/train/windows_json_ipc.py', 'src/train/windows_gpu_memory.py',
    'src/train/sft_round1_resource_v3.py', 'src/train/long_probe_core.py',
    'src/train/freeze_round2_formal_v1.py']


def objective_files(config, proposal):
    return {
        'full_train_tokens': proposal['train_path'], 'full_weights': proposal['weights'],
        'original_schedule': proposal['frozen_schedule'], 'weighted_groups': proposal['weighted_groups'],
        'proposal_config': config['formal_config'], 'run_config': str(CONFIG.relative_to(ROOT)),
        'weighted_core_source': 'src/train/sft_round2_weight_core_v1.py',
        'binding_helper_source': 'src/train/sft_round2_binding_v1.py',
        'count_helper_source': 'src/train/token_accumulation.py',
        'target_loss_source': 'src/train/target_loss.py',
        'prompt_config': 'configs/prompt.yaml', 'model_config': 'configs/model.yaml',
        'token_manifest': 'data/processed/spider_sft_tokens_v1/manifest.json',
        'data_manifest': proposal['manifest'], 'package_lock': 'requirements.lock.txt',
    }


def preserved_history():
    old = read(ROOT / 'reports/sft_round1_resource_v3_preflight.json')
    protected = dict(old['protected_sha256'])
    for run in ('sft_round1_resource_v3', 'sft_mutex_resume_v1', 'sft_round2_pilot_v2'):
        location = ROOT / 'outputs/runs' / run
        protected.update({str(location / name): sha for name, sha in read(location / 'artifact_sha256.json').items()})
        protected[str(location / 'artifact_sha256.json')] = digest(location / 'artifact_sha256.json')
    for name in ('reports/sft_round1_resource_v3_result.json', 'reports/sft_round1_eval_v2_eval.json',
                 'reports/sft_round1_eval_v2_analysis.json', 'reports/sft_round2_weight_cpu_v1_checks.json',
                 'configs/sft_round2_weight_proposal_v1.json'):
        protected[name] = digest(ROOT / name)
    frozen = read(FREEZE)
    protected.update(frozen['previous_attempt_sha256'])
    protected[str(FREEZE.relative_to(ROOT))] = digest(FREEZE)
    verify(protected)
    return protected


def prepare():
    config = read(CONFIG)
    destination = ROOT / config['preflight_report']
    if destination.exists():
        raise FileExistsError('Preparation already claimed; no retry')
    started = time.perf_counter()
    result = {'status': 'running', 'cuda_initialized': False, 'final_examples_read': False}
    write(destination, result, exclusive=True)
    try:
        assert (config['selected_examples'], config['batch_size'], config['accumulation_steps'],
                config['maximum_updates'], config['segment_ranges']) == (6364, 1, 8, 796, [[0, 200], [200, 400], [400, 600], [600, 796]])
        assert not config['gpu_training_enabled'] and not config['evaluation_enabled']
        assert config['global_update_budget'] == 796 and config['save_after_updates'] == [200, 400, 600, 796]
        assert not (ROOT / config['run_dir']).exists(), 'CPU preparation must not claim the GPU run'
        cpu = read(ROOT / config['cpu_report'])
        assert cpu['status'] == 'pass' and cpu['cuda_initialized'] is False
        frozen = read(FREEZE)
        assert cpu['freeze_manifest_sha256'] == digest(FREEZE)
        assert cpu['source_sha256'] == frozen['source_sha256']
        assert cpu['frozen_sources_verified_before_and_after'] is True
        verify(frozen['previous_attempt_sha256'])
        verify(cpu['source_sha256'])
        previous_cpu = read(ROOT / 'reports/sft_round2_weight_cpu_v1_checks.json')
        assert previous_cpu['status'] == 'pass'
        verify(previous_cpu['source_sha256'])
        pilot = read(ROOT / config['successful_pilot_report'])
        assert pilot['status'] == 'pass' and pilot['ledger']['completed_updates'] == 4
        verify(pilot['source_sha256'])
        selection = read(ROOT / config['schedule_path'])
        rows, weights = lines(ROOT / config['train_path']), lines(ROOT / config['weights_path'])
        old_schedule = read(ROOT / config['original_schedule_path'])
        old_groups = old_schedule['groups']
        assert old_schedule['train_sha256'] == digest(ROOT / config['train_path'])
        assert selection['source_schedule_sha256'] == digest(ROOT / config['original_schedule_path'])
        group_check = verify_group_manifest(rows, weights, old_groups, selection['groups'])
        groups = [{**original, **weighted} for original, weighted in zip(old_groups, selection['groups'])]
        assert group_check['databases'] == 126 and group_check['last_group_examples'] == 4
        assert all(len(group['source_indices']) == 8 for group in groups[:-1])
        assert max(row['sequence_length'] for row in rows) == 3761
        assert sum(row['loss_weight'] == 2 for row in weights) == 2528
        proposal = read(ROOT / config['formal_config'])
        for actual, expected in ((config['train_path'], proposal['train_path']),
                                 (config['weights_path'], proposal['weights']),
                                 (config['schedule_path'], proposal['weighted_groups']),
                                 (config['original_schedule_path'], proposal['frozen_schedule'])):
            assert Path(actual) == Path(expected), 'Runtime path differs from frozen proposal'
        resource_reference = read(ROOT / 'configs/sft_round1_resource_v3.json')
        for key in ('seed', 'selected_examples', 'batch_size', 'accumulation_steps', 'maximum_updates',
                    'global_update_budget', 'segment_ranges', 'save_after_updates', 'minimum_free_mib',
                    'minimum_disk_gib', 'allocated_limit_mib', 'reserved_limit_mib', 'total_gpu_window_seconds',
                    'limits', 'mutex_wait_seconds', 'start_headroom_mib', 'start_resource_samples',
                    'start_resource_interval_seconds', 'start_resource_budget_seconds'):
            assert config[key] == resource_reference[key], f'Formal runtime budget changed: {key}'
        original_recipe = read(ROOT / 'configs/sft_round1.json')
        original_training = dict(original_recipe['training'])
        weighted_training = dict(proposal['training'])
        original_training.pop('normalization'); weighted_training.pop('normalization')
        assert original_training == weighted_training and proposal['model'] == original_recipe['model']
        protected = preserved_history()
        for name in SOURCES:
            if name.endswith('.py'):
                ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
            protected[name] = digest(ROOT / name)
        files = objective_files(config, proposal)
        binding = build_binding(ROOT, files)
        for value in binding['files'].values():
            protected[value['path']] = value['sha256']
        protected[config['cpu_report']] = digest(ROOT / config['cpu_report'])
        old_preflight = read(ROOT / 'reports/sft_round1_resource_v3_preflight.json')
        packages = {name: version(name) for name in old_preflight['packages']}
        assert packages == old_preflight['packages']
        interpreters, counters = interpreter_evidence(), telemetry_preflight()
        assert all(row['matches'] for row in interpreters) and counters['status'] == 'available'
        assert not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'bitsandbytes'))
        verify(protected)
        result.update(status='pass', config=config, formal_config=proposal, selection=selection,
            groups=groups, packages=packages, group_check=group_check,
            protected_sha256=protected, source_sha256={name: digest(ROOT / name) for name in SOURCES},
            schedule_sha256=digest(ROOT / config['schedule_path']), run_config_sha256=digest(CONFIG),
            objective_binding=binding, objective_binding_files=files,
            interpreter_evidence=interpreters, telemetry_preflight=counters,
            cpu_report_sha256=digest(ROOT / config['cpu_report']),
            cpu_integration_passed=True, selected_examples=6364, input_tokens=sum(row['sequence_length'] for row in rows),
            target_tokens=sum(row['supervised_token_count'] for row in rows), gpu_training_started=False)
    except Exception as error:
        result.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result['cpu_prepare_seconds'] = time.perf_counter() - started
        write(destination, result)
    print({'status': result['status'], 'selected_examples': 6364, 'seconds': result['cpu_prepare_seconds']}, flush=True)


def checkpoint_check(path, update, preflight):
    manifest = read(path / 'complete.json')
    assert manifest['format'] == 'sft_complete_peft_v1'
    assert set(manifest['files']) == {'adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'rng.pt', 'metadata.json'}
    for name, expected in manifest['files'].items():
        assert digest(path / name) == expected, f'Checkpoint file changed: {name}'
    metadata = read(path / 'metadata.json')
    assert metadata['completed_updates'] == metadata['next_group_index'] == update
    assert metadata['schedule_sha256'] == preflight['schedule_sha256']
    assert metadata['run_config_sha256'] == preflight['run_config_sha256']
    assert metadata['protected_source_sha256'] == preflight['protected_sha256']
    current = build_binding(ROOT, preflight['objective_binding_files'])
    verify_binding(preflight['objective_binding'], current)
    verify_binding(metadata['objective_binding'], current)
    return {'metadata': metadata, 'fingerprints': manifest['fingerprints']}


def run(confirm_formal_training=False):
    if confirm_formal_training is not True:
        raise PermissionError('Explicit --run --confirm-formal-training required')
    config = read(CONFIG)
    preflight = read(ROOT / config['preflight_report'])
    root = ROOT / config['run_dir']
    if root.exists() or (ROOT / config['result_report']).exists():
        raise FileExistsError('Formal run already claimed; no replay/retry')
    assert preflight['status'] == 'pass' and preflight['config'] == config
    verify(preflight['protected_sha256'])
    verify_binding(preflight['objective_binding'], build_binding(ROOT, preflight['objective_binding_files']))
    readiness = stable_resource_gate(config)
    if readiness['status'] != 'ready':
        print({'status': 'not_ready', 'error': readiness.get('error'), 'worker_launches': 0}, flush=True)
        return 2
    root.mkdir(parents=True, exist_ok=False)
    ledger_path = root / 'progress.json'
    write(ledger_path, {'attempted_updates': 0, 'completed_updates': 0, 'total_budget': 796,
                       'partial_update_possible': False, 'failed': False}, exclusive=True)
    result = {'status': 'running', 'worker_launches': 0, 'automatic_retries': 0, 'segments': [],
              'worker_results': [], 'config': config, 'final_examples_read': False,
              'validation_examples_read': False, 'evaluation_performed': False}
    write(root / 'claim.json', result, exclusive=True)
    started, window_started = time.perf_counter(), None
    try:
        for name in SOURCES:
            target = root / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        result['source_sha256'] = preflight['source_sha256']
        write(root / 'preflight.json', preflight, exclusive=True)
        result['initial_resources'] = gate_resources(config)
        window_started = time.perf_counter()
        deadline = window_started + config['total_gpu_window_seconds']
        write(root / 'gpu_window_claim.json', {'started_monotonic': window_started, 'deadline_monotonic': deadline})
        for segment, (start, end) in enumerate(config['segment_ranges'], 1):
            ledger = read(ledger_path)
            assert ledger['attempted_updates'] == ledger['completed_updates'] == start and not ledger['failed']
            assert deadline - time.perf_counter() >= config['limits']['case_seconds']
            write(root / f'segment_{segment}_resources.json', gate_resources(config))
            command = [str(VENV_PYTHON), '-B', '-X', 'utf8', str(ROOT / 'src/train/sft_round2_formal_v1_worker.py'), '--segment', str(segment)]
            child_env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', SFT_CACHE_CLAIM=str(root.resolve()),
                             SFT_ROUND2_FORMAL_CONFIRMED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
            result['worker_launches'] += 1
            write(root / 'launch_progress.json', result)
            supervised = supervise(command, ROOT, child_env, root / f'segment_{segment}',
                                   dict(config['limits']), suite_deadline=deadline)
            result['segments'].append(supervised)
            write(root / f'segment_{segment}_returned.json', supervised)
            assert supervised['status'] == 'pass' and supervised['outcome'] == 'completed', supervised
            assert supervised['exit_code'] == 0 and supervised['identity_verified'] and not supervised['killed']
            assert not supervised['errors'] and not supervised['cleanup_errors']
            assert supervised['job_active_pids_after_cleanup'] == []
            assert len(supervised['role_exits']) == 1 and not supervised['role_exits'][0]['alive']
            worker = read(root / f'segment_{segment}/worker_result.json')
            assert worker['status'] == 'pass' and worker['pid'] == supervised['bound_pid']
            assert worker['start'] == start and worker['end'] == end
            expected_microbatches = sum(len(group['source_indices']) for group in preflight['groups'][start:end])
            assert worker['completed_microbatches'] == expected_microbatches and worker['global_completed_updates'] == end
            assert worker['base_unchanged'] and worker['cuda_rng_saved']
            assert [row['update'] for row in worker['history']] == list(range(start + 1, end + 1))
            for row, group in zip(worker['history'], preflight['groups'][start:end]):
                assert row['source_indices'] == group['source_indices']
                assert row['total_target_tokens'] == group['group_target_tokens']
                assert row['weighted_target_denominator'] == group['weighted_target_denominator']
                assert row['weights'] == group['loss_weights']
                assert row['micro_mean_loss_coefficients'] == group['micro_mean_loss_coefficients']
                assert row['gradients_finite_nonzero'] and row['changed_parameter_elements'] > 0
                assert row['cache_releases'] == len(group['source_indices'])
            releases = lines(root / f'segment_{segment}/cache_releases.jsonl')
            assert len(releases) == expected_microbatches and all(row['gradient_and_adam_identity_version_preserved'] for row in releases)
            ledger = read(ledger_path)
            assert ledger['attempted_updates'] == ledger['completed_updates'] == end and not ledger['failed']
            checkpoint_check(root / 'checkpoints' / f'update_{end:06d}', end, preflight)
            if start:
                restored = read(root / f'segment_{segment}/restore_verification.json')
                previous = checkpoint_check(root / 'checkpoints' / f'update_{start:06d}', start, preflight)
                assert restored['metadata'] == previous['metadata']
                assert restored['verification']['fingerprints'] == previous['fingerprints']
                assert all(restored['verification'][key] for key in ('adapter_exact', 'optimizer_exact', 'rng_exact', 'parameter_order_exact', 'cuda_rng_restored'))
            result['worker_results'].append(worker)
        assert read(ledger_path)['completed_updates'] == 796
        result['status'] = 'pass'
    except Exception as error:
        result.update(status='failed', error=str(error), traceback=traceback.format_exc())
        ledger = read(ledger_path)
        ledger.update(failed=True, error=str(error))
        write(ledger_path, ledger)
    finally:
        result['gpu_window_seconds'] = None if window_started is None else time.perf_counter() - window_started
        if result['gpu_window_seconds'] is not None and result['gpu_window_seconds'] > config['total_gpu_window_seconds']:
            result.update(status='failed', budget_error='Total GPU window exceeded')
        try:
            verify(preflight['protected_sha256'])
            verify_binding(preflight['objective_binding'], build_binding(ROOT, preflight['objective_binding_files']))
            result['protected_files_verified_after'] = len(preflight['protected_sha256'])
        except Exception as error:
            result.update(status='failed', preservation_error=str(error))
        if result['status'] != 'pass':
            ledger = read(ledger_path); ledger['failed'] = True; write(ledger_path, ledger)
        result['ledger'] = read(ledger_path)
        result['total_controller_seconds'] = time.perf_counter() - started
        result['controller_torch_imported'] = 'torch' in sys.modules
        write(root / 'report.json', result)
        write(ROOT / config['result_report'], result, exclusive=True)
        write(root / 'artifact_sha256.json', {p.relative_to(root).as_posix(): digest(p)
              for p in root.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'}, exclusive=True)
    print({'status': result['status'], 'ledger': result['ledger'], 'gpu_window_seconds': result['gpu_window_seconds']}, flush=True)
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--confirm-formal-training', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        raise SystemExit(run(confirm_formal_training=args.confirm_formal_training))
