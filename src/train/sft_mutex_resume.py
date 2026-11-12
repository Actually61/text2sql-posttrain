"""One bounded 2+2 SFT run; CPU controller never imports ML packages."""
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
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from src.train.cache_release_pilot import read, digest, write, verify, gate_resources
from src.train.pid_supervision_v6 import supervise
from src.train.windows_job_process_v2 import VENV_PYTHON, interpreter_evidence
from src.train.windows_gpu_memory import telemetry_preflight

CONFIG = ROOT / 'configs/sft_mutex_resume.json'
SOURCES = ['configs/sft_mutex_resume.json', 'src/train/sft_mutex_resume.py',
    'src/train/sft_mutex_resume_worker.py', 'src/train/test_sft_mutex_resume.py',
    'src/train/pid_supervision_v6.py', 'src/train/windows_job_exit_wait_v3.py',
    'src/train/cache_release_core.py', 'src/train/sft_checkpoint.py',
    'src/train/sft_update_budget.py', 'src/train/sft_train.py',
    'src/train/target_loss.py', 'src/train/qlora_stability.py',
    'src/train/windows_stage_mutex.py', 'src/train/stage_file_io_v2.py',
    'src/train/stage_file_io.py', 'src/train/windows_json_ipc.py']


def preservation_map():
    """Expand existing manifests only; never open evaluation datasets."""
    protected = {}
    def merge(mapping):
        for name, expected in mapping.items():
            path = Path(name)
            absolute = path if path.is_absolute() else ROOT / path
            key = str(absolute.resolve())
            if key in protected and protected[key] != expected:
                raise ValueError(f'Conflicting historical hashes: {name}')
            protected[key] = expected
    previous = ROOT / 'outputs/checks/stage_mutex_v1'
    # This frozen map already contains the earlier failed sources, runs and ledgers.
    merge(read(previous / 'preservation_before.json'))
    mutex_report = read(ROOT / 'reports/stage_mutex_cpu_checks.json')
    assert mutex_report['status'] == 'pass'
    merge(mutex_report['source_sha256'])
    merge({str(previous / name): sha for name, sha in read(previous / 'artifact_sha256.json').items()})
    for path in [previous / 'artifact_sha256.json', ROOT / 'reports/stage_mutex_cpu_checks.json',
                 ROOT / 'reports/stage_mutex_readonly_audit.json', ROOT / 'reports/stage_mutex_walkthrough.md']:
        merge({str(path): digest(path)})
    verify(protected)
    return protected


def prepare():
    config = read(CONFIG)
    destination = ROOT / config['preflight_report']
    if destination.exists():
        raise FileExistsError('CPU preparation already claimed; no rerun')
    started = time.perf_counter()
    report = {'status': 'running', 'final_examples_read': False, 'cuda_initialized': False}
    write(destination, report, exclusive=True)
    try:
        assert (config['selected_examples'], config['batch_size'], config['accumulation_steps'],
                config['segment_ranges'], config['maximum_updates']) == (32, 1, 8, [[0, 2], [2, 4]], 4)
        prior = read(ROOT / 'reports/sft_resume_pilot_preflight.json')
        cpu = read(ROOT / config['cpu_report'])
        assert cpu['status'] == 'pass', cpu
        verify(cpu['source_sha256'])
        assert cpu['checks']['original_32_full_examples']['status'] == 'pass'
        assert cpu['checks']['single_native_worker']['status'] == 'pass'
        reused = {}
        for name in config['reused_cpu_reports']:
            earlier = read(ROOT / name)
            assert earlier['status'] == 'pass', name
            verify(earlier['source_sha256'])
            reused[name] = {'sha256': digest(ROOT / name), 'source_hashes_verified': len(earlier['source_sha256'])}
        math_evidence = read(ROOT / 'reports/sft_cache_resume_cpu_checks.json')['checks']['four_update_resume']
        assert math_evidence['exact_comparison'] and math_evidence['maximum_parameter_difference'] == 0
        protected = preservation_map()
        for name in SOURCES:
            if name.endswith('.py'):
                ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
            protected[name] = digest(ROOT / name)
        protected[config['cpu_report']] = digest(ROOT / config['cpu_report'])
        selection = read(ROOT / config['prepared_dir'] / 'selection.json')
        assert selection == prior['selection'], 'Original 32-example schedule changed'
        assert cpu['checks']['original_32_full_examples']['source_indices'] == selection['selected_source_indices']
        verify(cpu['checks']['original_32_full_examples']['data_sha256'])
        assert config['mutex_wait_seconds'] == 1 and config['total_gpu_window_seconds'] == 620
        packages = {name: version(name) for name in prior['packages']}
        assert packages == prior['packages']
        for rel, count in [('outputs/runs/sft_resume_pilot_v1/progress.json', 3),
                           ('outputs/runs/cache_release_pilot_v1/progress.json', 0),
                           ('outputs/runs/sft_cache_resume_v1/progress.json', 0)]:
            old = read(ROOT / rel)
            assert old['completed_updates'] == count and old['failed']
        interpreters = interpreter_evidence()
        counters = telemetry_preflight()
        assert all(row['matches'] for row in interpreters) and counters['status'] == 'available'
        assert not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'bitsandbytes'))
        report.update(status='pass', config=config, formal_config=prior['formal_config'],
            groups=selection['groups'], selection=selection, packages=packages,
            protected_sha256=protected, schedule_sha256=digest(ROOT / config['prepared_dir'] / 'selection.json'),
            run_config_sha256=digest(CONFIG), interpreter_evidence=interpreters, telemetry_preflight=counters,
            selected_examples=32, input_tokens=selection['input_tokens'], target_tokens=selection['target_tokens'],
            source_sha256={name: digest(ROOT / name) for name in SOURCES},
            reused_cpu_evidence=reused, cpu_math_rerun=False, mutex_wait_limit_seconds=1,
            budget={'gpu_window_seconds': config['total_gpu_window_seconds'], 'limits': config['limits'],
                'minimum_free_mib': config['minimum_free_mib'], 'allocated_limit_mib': config['allocated_limit_mib'],
                'reserved_limit_mib': config['reserved_limit_mib'], 'mutex_wait_seconds': config['mutex_wait_seconds']})
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['cpu_prepare_seconds'] = time.perf_counter() - started
        write(destination, report)
    print({key: report[key] for key in ('status', 'selected_examples', 'input_tokens', 'target_tokens', 'cpu_prepare_seconds')}, flush=True)


def lines(path):
    import json
    if not path.exists():
        return []
    contents = path.read_text(encoding='utf-8')
    if contents and not contents.endswith('\n'):
        raise ValueError(f'Incomplete JSONL: {path}')
    return [json.loads(line) for line in contents.splitlines()]


def checkpoint_check(path, update, preflight):
    saved = read(path / 'complete.json')
    assert saved['format'] == 'sft_complete_peft_v1'
    assert set(saved['files']) == {'adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'rng.pt', 'metadata.json'}
    for name, expected in saved['files'].items():
        assert digest(path / name) == expected, f'Checkpoint changed: {name}'
    metadata = read(path / 'metadata.json')
    assert metadata['completed_updates'] == metadata['next_group_index'] == update
    assert metadata['schedule_sha256'] == preflight['schedule_sha256']
    assert metadata['run_config_sha256'] == preflight['run_config_sha256']
    assert metadata['protected_source_sha256'] == preflight['protected_sha256']
    return {'update': update, 'fingerprints': saved['fingerprints'], 'metadata': metadata}


def run():
    config = read(CONFIG)
    preflight = read(ROOT / config['preflight_report'])
    root = ROOT / config['run_dir']
    if root.exists() or (ROOT / config['result_report']).exists():
        raise FileExistsError('One-shot run already claimed; no retry')
    root.mkdir(parents=True, exist_ok=False)
    ledger_path = root / 'progress.json'
    write(ledger_path, {'attempted_updates': 0, 'completed_updates': 0, 'total_budget': 4,
                        'partial_update_possible': False, 'failed': False}, exclusive=True)
    report = {'status': 'running', 'worker_launches': 0, 'automatic_retries': 0,
              'final_examples_read': False, 'validation_examples_read': False,
              'controller_torch_imported': False, 'segments': [], 'worker_results': [], 'config': config}
    started = time.perf_counter()
    window_started = None
    write(root / 'claim.json', report, exclusive=True)
    try:
        assert preflight['status'] == 'pass' and preflight['config'] == config
        verify(preflight['protected_sha256'])
        for name in SOURCES:
            destination = root / 'source_snapshot' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        report['source_sha256'] = preflight['source_sha256']
        write(root / 'preflight.json', preflight, exclusive=True)
        report['initial_resources'] = gate_resources(config)
        window_started = time.perf_counter()
        deadline = window_started + config['total_gpu_window_seconds']
        write(root / 'gpu_window_claim.json', {'started_monotonic': window_started, 'deadline_monotonic': deadline})
        for segment, (start, end) in enumerate(config['segment_ranges'], 1):
            ledger = read(ledger_path)
            assert ledger['attempted_updates'] == ledger['completed_updates'] == start and not ledger['failed']
            assert deadline - time.perf_counter() >= config['limits']['case_seconds'], 'Insufficient remaining window'
            resource = gate_resources(config)
            write(root / f'segment_{segment}_resources.json', resource)
            command = [str(VENV_PYTHON), '-X', 'utf8', str(ROOT / 'src/train/sft_mutex_resume_worker.py'),
                       '--segment', str(segment)]
            child_env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', SFT_CACHE_CLAIM=str(root.resolve()),
                             HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
            report['worker_launches'] += 1
            write(root / 'launch_progress.json', report)
            result = supervise(command, ROOT, child_env, root / f'segment_{segment}',
                               dict(config['limits']), sample_provider=None, suite_deadline=deadline)
            report['segments'].append(result)
            write(root / f'segment_{segment}_returned.json', result)
            assert result['status'] == 'pass' and result['outcome'] == 'completed', result
            assert result['exit_code'] == 0 and result['identity_verified'] and not result['killed']
            assert not result['errors'] and not result['cleanup_errors']
            assert result['job_active_pids_after_cleanup'] == []
            assert len(result['role_exits']) == 1 and not result['role_exits'][0]['alive']
            worker = read(root / f'segment_{segment}' / 'worker_result.json')
            assert worker['status'] == 'pass', worker
            assert worker['pid'] == result['bound_pid'] and worker['start'] == start and worker['end'] == end
            assert worker['completed_microbatches'] == 16 and worker['global_completed_updates'] == end
            assert worker['base_unchanged'] and worker['cuda_rng_saved']
            assert [row['update'] for row in worker['history']] == list(range(start + 1, end + 1))
            for row, group in zip(worker['history'], preflight['groups'][start:end]):
                assert row['source_indices'] == group['source_indices']
                assert row['total_target_tokens'] == group['group_target_tokens']
                assert row['gradients_finite_nonzero'] and row['changed_parameter_elements'] > 0
                assert row['cache_releases'] == 8
            releases = lines(root / f'segment_{segment}' / 'cache_releases.jsonl')
            assert len(releases) == 16 and all(row['gradient_and_adam_identity_version_preserved'] for row in releases)
            ledger = read(ledger_path)
            assert ledger['attempted_updates'] == ledger['completed_updates'] == end and not ledger['failed']
            saved = checkpoint_check(root / 'checkpoints' / f'update_{end:06d}', end, preflight)
            if segment == 2:
                restored = read(root / 'segment_2/restore_verification.json')
                previous = checkpoint_check(root / 'checkpoints/update_000002', 2, preflight)
                assert restored['metadata'] == previous['metadata']
                assert restored['verification']['fingerprints'] == previous['fingerprints']
                assert all(restored['verification'][key] for key in (
                    'adapter_exact', 'optimizer_exact', 'rng_exact', 'parameter_order_exact', 'cuda_rng_restored'))
            report['worker_results'].append(worker)
            for path in (root / f'segment_{segment}').glob('*.jsonl'):
                lines(path)
        assert read(ledger_path)['completed_updates'] == 4
        report['status'] = 'pass'
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        ledger = read(ledger_path)
        ledger.update(failed=True, error=str(error))
        write(ledger_path, ledger)
    finally:
        report['gpu_window_seconds'] = None if window_started is None else time.perf_counter() - window_started
        if report['gpu_window_seconds'] is not None and report['gpu_window_seconds'] > config['total_gpu_window_seconds']:
            report.update(status='failed', budget_error='620 second total window exceeded')
        try:
            verify(preflight['protected_sha256'])
            report['protected_files_verified_after'] = len(preflight['protected_sha256'])
        except Exception as error:
            report.update(status='failed', preservation_error=str(error))
        if report['status'] != 'pass':
            failed_ledger = read(ledger_path)
            failed_ledger.update(failed=True)
            write(ledger_path, failed_ledger)
        report['ledger'] = read(ledger_path)
        report['total_controller_seconds'] = time.perf_counter() - started
        report['controller_torch_imported'] = 'torch' in sys.modules
        write(root / 'report.json', report)
        write(ROOT / config['result_report'], report, exclusive=True)
        write(root / 'artifact_sha256.json', {p.relative_to(root).as_posix(): digest(p)
              for p in root.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'}, exclusive=True)
    print({'status': report['status'], 'ledger': report['ledger'], 'gpu_window_seconds': report['gpu_window_seconds']}, flush=True)
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        raise SystemExit(run())
