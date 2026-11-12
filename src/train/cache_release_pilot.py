"""Independent 16-example cache-release trial. CPU parent never imports Torch."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
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
CONFIG = ROOT / 'configs/cache_release_pilot.json'
NEW_SOURCES = ['configs/cache_release_pilot.json', 'src/train/cache_release_pilot.py',
    'src/train/cache_release_core.py', 'src/train/test_cache_release_core.py',
    'src/train/cache_release_watchdog.py', 'src/train/test_cache_release_watchdog.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def append(path, row):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify(mapping):
    for name, expected in mapping.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen file changed: {name}')


def resources():
    raw = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,driver_version',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    fields = [part.strip() for part in raw.split(',')]
    return {'name': fields[0], 'total_mib': float(fields[1]), 'used_mib': float(fields[2]),
            'free_mib': float(fields[3]), 'driver': fields[4], 'disk_free_gib': shutil.disk_usage(ROOT).free / 2**30}


def gate_resources(config):
    result = resources()
    if result['free_mib'] < config['minimum_free_mib'] or result['disk_free_gib'] < config['minimum_disk_gib']:
        raise RuntimeError(f'Resource gate failed before model load: {result}')
    return result


def prepare():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from src.data.sft_features import collate_sft_batch
    from src.train.windows_gpu_memory import telemetry_preflight
    config = read(CONFIG)
    if (config['selected_examples'], config['batch_size'], config['accumulation_steps'], config['maximum_updates']) != (16, 1, 8, 2):
        raise ValueError('Only the authorized 16-example/two-update trial is supported')
    prior = read(ROOT / config['previous_preflight'])
    protected = dict(prior['protected_sha256'])
    if any('final' in name.lower() for name in protected):
        raise ValueError('Do not access final-set files')
    verify(protected)
    old_run = ROOT / config['old_run']
    for path in old_run.rglob('*'):
        if path.is_file():
            protected[path.relative_to(ROOT).as_posix()] = digest(path)
    for name, expected in read(old_run / 'artifact_sha256.json').items():
        if digest(old_run / name) != expected:
            raise ValueError(f'Old failed run changed: {name}')
    old_ledger = read(old_run / 'progress.json')
    if old_ledger['completed_updates'] != 3 or not old_ledger['failed']:
        raise ValueError('Old failed ledger must stay at three completed updates')
    for name, pinned in prior['packages'].items():
        if version(name) != pinned:
            raise ValueError(f'Package changed: {name}')
    for name in config['cpu_reports']:
        report = read(ROOT / name)
        if report['status'] != 'pass':
            raise ValueError(f'CPU prerequisite failed: {name}')
        for source, expected in report.get('source_sha256', {}).items():
            if digest(ROOT / source) != expected:
                raise ValueError(f'CPU test source changed: {source}')
        protected[name] = digest(ROOT / name)
    selection = read(ROOT / 'data/processed/spider_sft_resume_pilot_v1/selection.json')
    original = read(ROOT / 'data/processed/spider_sft_resume_pilot_v1/features.json')
    groups = selection['groups'][:2]
    ids = [index for group in groups for index in group['source_indices']]
    if len(ids) != len(set(ids)) or len(ids) != 16 or any(len(g['source_indices']) != 8 for g in groups):
        raise ValueError('Only the exact original first sixteen examples are allowed')
    by_id = {row['source_index']: row for row in original}
    features = [by_id[index] for index in ids]
    for group in groups:
        for index, length, targets in zip(group['source_indices'], group['sequence_lengths'], group['target_counts']):
            row = by_id[index]
            batch = collate_sft_batch([row], 151643, 4096)
            if list(batch['input_ids'].shape) != [1, length] or int((batch['labels'][:, 1:] != -100).sum()) != targets:
                raise ValueError('Full input/assistant target mismatch')
    prepared = ROOT / config['prepared_dir']
    for name, value in [('features.json', features), ('schedule.json', groups)]:
        write(prepared / name, value, exclusive=True)
        protected[(prepared / name).relative_to(ROOT).as_posix()] = digest(prepared / name)
    for name in NEW_SOURCES + [config['previous_preflight']]:
        protected[name] = digest(ROOT / name)
    telemetry = telemetry_preflight()
    if telemetry['status'] != 'available' or torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation/PDH prerequisite failed')
    report = {'status': 'pass', 'config': config, 'formal_config': prior['formal_config'], 'groups': groups,
        'packages': prior['packages'], 'protected_sha256': protected, 'old_ledger': old_ledger,
        'schedule_sha256': digest(prepared / 'schedule.json'), 'run_config_sha256': digest(CONFIG),
        'source_indices': ids, 'selected_examples': 16, 'input_tokens': sum(r['sequence_length'] for r in features),
        'target_tokens': sum(r['supervised_token_count'] for r in features),
        'cuda_initialized': False, 'model_loaded': False, 'final_examples_read': False,
        'telemetry_preflight': telemetry}
    write(ROOT / config['preflight_report'], report, exclusive=True)
    print(json.dumps({key: report[key] for key in ['status', 'selected_examples', 'input_tokens', 'target_tokens', 'cuda_initialized']}))


def checked():
    config = read(CONFIG)
    preflight = read(ROOT / config['preflight_report'])
    if config != preflight['config'] or preflight['status'] != 'pass':
        raise ValueError('Preflight/config changed')
    verify(preflight['protected_sha256'])
    for name, pinned in preflight['packages'].items():
        if version(name) != pinned:
            raise ValueError(f'Package changed: {name}')
    return config, preflight


def worker():
    # Publish stage before heavy imports so the parent can time the entire worker.
    config = read(CONFIG)
    directory = ROOT / config['run_dir']
    if os.environ.get('CACHE_PILOT_CLAIM') != str(directory.resolve()):
        raise RuntimeError('Worker requires single-use parent claim')
    write(directory / 'worker_started.json', {'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat()}, exclusive=True)
    write(directory / 'stage.json', {'stage': 'imports', 'micro_started_monotonic': None, 'telemetry_required': False})
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import random
    import numpy as np
    import torch
    from src.data.sft_features import collate_sft_batch
    from src.train.qlora_stability import fresh_model
    from src.train.qlora_smoke import frozen_parameter_hash
    from src.train.qlora_step import adapter_parameters, check_gradients
    from src.train.sft_update_budget import JournaledAdamW
    from src.train.sft_checkpoint import save_checkpoint
    from src.train.sft_train import set_next_lr, adam_steps
    from src.train.long_probe_core import optimizer_state_inventory
    from src.train.target_loss import target_position_loss
    from src.train.cache_release_core import backward_cache_group

    def guard():
        if (directory / 'stop_request.json').exists():
            raise RuntimeError(f"Stop requested: {read(directory / 'stop_request.json')}")
        if time.perf_counter() - float(os.environ['CACHE_PILOT_STARTED']) >= config['limits']['total_soft_seconds']:
            write(directory / 'stop_request.json', {'reason': 'Total soft deadline at worker boundary',
                'kind': 'time', 'monotonic': time.perf_counter()})
            raise RuntimeError('Total soft time limit')

    def stop_resource(reason):
        if not (directory / 'stop_request.json').exists():
            write(directory / 'stop_request.json', {'reason': reason, 'kind': 'resource', 'monotonic': time.perf_counter()})
        raise RuntimeError(reason)

    def memory():
        stats = torch.cuda.memory_stats()
        backend = torch.cuda.get_allocator_backend()
        native = backend == 'native'
        return {'allocated_mib': torch.cuda.memory_allocated() / 2**20,
            'reserved_mib': torch.cuda.memory_reserved() / 2**20,
            'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            'backend': backend, 'inactive_split_bytes': stats.get('inactive_split_bytes.all.current') if native else None,
            'num_alloc_retries': stats.get('num_alloc_retries') if native else None,
            'num_ooms': stats.get('num_ooms') if native else None,
            'native_stats_applicable': native}

    def sample(stage, **extra):
        guard()
        torch.cuda.synchronize()
        row = {'stage': stage, 'monotonic': time.perf_counter(), **memory(), **extra}
        append(directory / 'memory_events.jsonl', row)
        if row['peak_allocated_mib'] > config['allocated_limit_mib'] or row['peak_reserved_mib'] > config['reserved_limit_mib']:
            stop_resource(f'Memory threshold exceeded at {stage}: {row}')
        guard()
        return row

    def stage(name, micro_start=None):
        guard()
        write(directory / 'stage.json', {'stage': name, 'micro_started_monotonic': micro_start,
                                         'telemetry_required': torch.cuda.is_initialized()})

    history, micro_records = [], []
    optimizer = model = None
    started = time.perf_counter()
    try:
        config, preflight = checked()
        write(directory / 'worker_resources.json', gate_resources(config))
        guard()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 is required')
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed']); torch.backends.cudnn.benchmark = False
        # Establish the CUDA context and wait briefly for Windows to register this PID.
        # During this handshake no model is loaded; missing counters are not reported as zero.
        torch.cuda.init()
        register_deadline = time.perf_counter() + 2.0
        while True:
            guard()
            sample_path = directory / 'pid_memory.jsonl'
            persisted = []
            if sample_path.exists():
                for line in sample_path.read_text(encoding='utf-8').splitlines():
                    try:
                        persisted.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # The parent may currently be appending its last line.
            if any(row.get('status') == 'ok' and row.get('pid') == os.getpid()
                   and time.perf_counter() - row.get('sampled_monotonic', 0) < 2.0 for row in persisted):
                break
            if time.perf_counter() >= register_deadline:
                stop_resource('PID memory counters did not register within two seconds')
            time.sleep(0.05)
        stage('load_and_prepare')
        formal = preflight['formal_config']
        model = fresh_model(read(ROOT / formal['model']['inherit_loading_and_lora_from']))
        named = adapter_parameters(model)
        base_before = frozen_parameter_hash(model)
        optimizer = JournaledAdamW(list(named.values()), ledger_path=directory / 'progress.json',
            start=0, segment_end=2, total_budget=2,
            **{k: v for k, v in formal['training']['optimizer'].items() if k != 'name'})
        if optimizer.state:
            raise ValueError('Fresh optimizer must be empty')
        sample('load_end')
        write(directory / 'initial_state.json', {'base_hash': base_before,
            'trainable_parameters': sum(p.numel() for p in named.values()),
            'fresh_optimizer': True, 'previous_adapter_loaded': False,
            'allocator_environment': {k: v for k, v in os.environ.items() if k.startswith('PYTORCH_') and 'ALLOC' in k}})
        by_id = {r['source_index']: r for r in read(ROOT / config['prepared_dir'] / 'features.json')}
        model.train()
        for group_index, group in enumerate(preflight['groups']):
            guard()
            update = group_index + 1
            stage(f'update_{update}_setup')
            rows = [by_id[index] for index in group['source_indices']]
            batches = [collate_sft_batch([row], 151643, 4096) for row in rows]
            optimizer.zero_grad(set_to_none=True)
            before = {n: p.detach().cpu().clone() for n, p in named.items()}
            lr = set_next_lr(optimizer, group_index, formal['training'])
            state_before = optimizer_state_inventory(optimizer)
            if state_before['cuda_bytes'] != (0 if update == 1 else 73859072):
                raise ValueError('Adam state reset or changed size')
            micro_started = None
            cache_records = []

            def observe(event, index, metadata):
                nonlocal micro_started
                row = rows[index]
                if event == 'micro_start':
                    guard()
                    micro_started = time.perf_counter()
                    stage(f'update_{update}_micro_{index + 1}', micro_started)
                    torch.cuda.reset_peak_memory_stats()
                if event == 'forward_end':
                    if metadata['input_shape'] != [1, row['sequence_length']] or metadata['valid_target_count'] != row['supervised_token_count']:
                        raise ValueError('Full input/assistant supervision changed')
                record = sample(event, update=update, microbatch=index + 1, source_index=row['source_index'], **metadata)
                if event == 'micro_end':
                    record = {**record, 'seconds': time.perf_counter() - micro_started,
                              'sequence_length': row['sequence_length'], 'valid_targets': row['supervised_token_count']}
                    micro_records.append(record)
                    append(directory / 'micro_intervals.jsonl', record)
                    if record['seconds'] >= config['limits']['micro_soft_seconds']:
                        write(directory / 'stop_request.json', {'reason': 'Microbatch soft deadline at boundary',
                            'kind': 'time', 'monotonic': time.perf_counter()})
                        raise RuntimeError('Microbatch soft time limit; no next microbatch or update')
                    stage(f'update_{update}_between_micro')

            def retained_signature():
                gradients = [(name, p.grad.data_ptr(), p.grad._version, tuple(p.grad.shape), str(p.grad.dtype))
                             for name, p in named.items() if p.grad is not None]
                states = [(id(p), key, v.data_ptr(), v._version, tuple(v.shape), str(v.dtype), str(v.device))
                          for p, state_values in optimizer.state.items() for key, v in state_values.items() if isinstance(v, torch.Tensor)]
                return gradients, states

            def release(index):
                guard()
                signature = retained_signature()
                if len(signature[0]) != len(named):
                    raise ValueError('Missing accumulated gradients before cache release')
                before_cache = sample('cache_before', update=update, microbatch=index + 1)
                tick = time.perf_counter()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - tick
                after_cache = sample('cache_after', update=update, microbatch=index + 1)
                if retained_signature() != signature:
                    raise ValueError('Cache release changed live gradient/Adam identity or version')
                result = {'update': update, 'microbatch': index + 1, 'seconds': seconds,
                    'before': before_cache, 'after': after_cache, 'gradient_and_adam_identity_version_preserved': True}
                cache_records.append(result)
                append(directory / 'cache_releases.jsonl', result)

            def loss_fn(current_model, batch):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return target_position_loss(current_model, batch)

            group_result = backward_cache_group(model, batches, device='cuda:0', loss_fn=loss_fn,
                                                 observe=observe, release=release)
            guard()
            if group_result['total_target_tokens'] != group['group_target_tokens'] or len(cache_records) != 8:
                raise ValueError('Incomplete group/cache release coverage')
            stage(f'update_{update}_gradient_and_clip')
            gradients = check_gradients(model)
            write(directory / f'update_{update}_gradients.json', gradients)
            torch.nn.utils.clip_grad_norm_(list(named.values()), 1.0, error_if_nonfinite=True)
            post_clip = check_gradients(model)['global_l2_norm']
            if post_clip > 1.00001:
                raise ValueError('Gradient clipping failed')
            sample('gradient_clip_end', update=update)
            stage(f'update_{update}_optimizer')
            guard()
            optimizer.step()
            sample('optimizer_end', update=update)
            state_after = optimizer_state_inventory(optimizer)
            if state_after['cuda_bytes'] != 73859072 or adam_steps(optimizer) != [float(update)]:
                raise ValueError('Adam moments/step mismatch')
            changes = []
            for name, parameter in named.items():
                value = parameter.detach().cpu()
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite adapter')
                changes.append({'name': name, 'changed_elements': int(torch.count_nonzero(value - before[name]))})
            if not any(c['changed_elements'] for c in changes):
                raise ValueError('No adapter parameter changed')
            write(directory / f'update_{update}_adapter_changes.json', changes)
            next_lr = set_next_lr(optimizer, update, formal['training'])
            optimizer.zero_grad(set_to_none=True)
            record = {'update': update, 'source_indices': group['source_indices'], **group_result,
                'lr_used': lr, 'next_update_lr': next_lr, 'grad_norm_before_clip': gradients['global_l2_norm'],
                'grad_norm_after_clip': post_clip, 'gradients_finite': True,
                'optimizer_cuda_mib_before': state_before['cuda_bytes'] / 2**20,
                'optimizer_cuda_mib_after': state_after['cuda_bytes'] / 2**20,
                'changed_parameter_elements': sum(c['changed_elements'] for c in changes)}
            history.append(record)
            write(directory / 'history.json', history)
            del before, changes, batches
            print(json.dumps(record), flush=True)
        guard()
        stage('save_complete_checkpoint_2')
        metadata = {'completed_updates': 2, 'next_group_index': 2,
            'next_update_lr': history[-1]['next_update_lr'], 'schedule_sha256': preflight['schedule_sha256'],
            'run_config_sha256': preflight['run_config_sha256'], 'base_hash': base_before,
            'protected_source_sha256': preflight['protected_sha256']}
        saved = save_checkpoint(directory / 'checkpoints/update_000002', model, optimizer, metadata)
        write(directory / 'checkpoint_save.json', saved)
        sample('save_end')
        stage('final_audit')
        base_after = frozen_parameter_hash(model)
        if base_before != base_after:
            raise ValueError('Frozen base changed')
        write(directory / 'base_hashes.json', {'before': base_before, 'after': base_after})
        sample('final_audit_end')
        guard()
        result = {'status': 'pass', 'history': history, 'completed_microbatches': len(micro_records),
            'base_unchanged': True, 'cuda_rng_saved': saved['cuda_rng_saved'],
            'model_scope_seconds': time.perf_counter() - started}
        code = 0
    except Exception as error:
        # No retry, no step on partial group, and no checkpoint of partial gradients.
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        ledger = read(directory / 'progress.json')
        ledger.update(failed=True, error=f'{type(error).__name__}: {error}')
        write(directory / 'progress.json', ledger)
        (directory / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        result = {'status': 'failed', 'error': str(error), 'history': history, 'ledger': ledger,
                  'completed_microbatches': len(micro_records), 'model_scope_seconds': time.perf_counter() - started}
        code = 2
    write(directory / 'worker_result.json', result)
    return code


def launch():
    config, preflight = checked()
    initial_resources = gate_resources(config)
    directory = ROOT / config['run_dir']
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / 'claim.json', {'utc': datetime.now(timezone.utc).isoformat(), 'maximum_updates': 2}, exclusive=True)
    write(directory / 'progress.json', {'attempted_updates': 0, 'completed_updates': 0,
        'total_budget': 2, 'partial_update_possible': False, 'failed': False}, exclusive=True)
    write(directory / 'resources.json', initial_resources, exclusive=True)
    write(directory / 'preflight.json', preflight, exclusive=True)
    for name in NEW_SOURCES + config['cpu_reports']:
        path = directory / 'snapshot' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    from src.train.cache_release_watchdog import run_supervised
    monitored = run_supervised([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker'],
        ROOT, dict(os.environ, CACHE_PILOT_CLAIM=str(directory.resolve()), PYTHONUTF8='1',
                   CACHE_PILOT_STARTED=str(time.perf_counter())), directory, config['limits'])
    worker_result = read(directory / 'worker_result.json') if (directory / 'worker_result.json').exists() else None
    ledger = read(directory / 'progress.json')
    success = bool(worker_result and worker_result['status'] == 'pass' and monitored['status'] == 'pass'
                   and monitored['valid_sample_count'] > 0 and not monitored['telemetry_errors'] and monitored['exit_code'] == 0
                   and not (directory / 'stop_request.json').exists() and ledger['completed_updates'] == ledger['attempted_updates'] == 2
                   and not ledger['failed'] and not ledger['partial_update_possible']
                   and worker_result['completed_microbatches'] == 16 and worker_result['cuda_rng_saved'])
    if not success:
        ledger.update(failed=True, controller_error='Trial failed or interrupted; no retry')
        write(directory / 'progress.json', ledger)
    verify(preflight['protected_sha256'])
    result = {'status': 'pass' if success else 'failed', 'ledger': ledger, 'watchdog': monitored,
        'worker': worker_result, 'old_files_unchanged': True, 'old_ledger': read(ROOT / config['old_run'] / 'progress.json'),
        'new_process': True, 'restored_old_run': False, 'full_training': False,
        'evaluation_performed': False, 'final_examples_read': False}
    write(directory / 'result.json', result)
    write(ROOT / config['result_report'], result, exclusive=True)
    write(directory / 'artifact_sha256.json', {p.relative_to(directory).as_posix(): digest(p)
        for p in directory.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'}, exclusive=True)
    print(json.dumps({'status': result['status'], 'ledger': ledger, 'watchdog': monitored}, ensure_ascii=False))
    return 0 if success else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare', action='store_true')
    choice.add_argument('--run', action='store_true')
    choice.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        raise SystemExit(worker() if args.worker else launch())
