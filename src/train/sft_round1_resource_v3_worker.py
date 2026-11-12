"""One identity-gated segment of the 6364-example formal SFT round.

This module imports no ML library until the supervisor has retained and verified
this native worker. The controller owns the one-shot claim and global ledger.
"""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train.cache_release_pilot import read, write, append, verify, gate_resources
from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack
from src.train.stage_file_io_v2 import write_stage

CONFIG = ROOT / 'configs/sft_round1_resource_v3.json'


def run(segment):
    config = read(CONFIG)
    if os.environ.get('SFT_FORMAL_TRAINING_CONFIRMED') != '1':
        raise RuntimeError('Formal worker requires explicit controller confirmation')
    root = ROOT / config['run_dir']
    directory = root / f'segment_{segment}'
    if segment not in (1, 2, 3, 4) or os.environ.get('SFT_CACHE_CLAIM') != str(root.resolve()):
        raise RuntimeError('Worker requires the new, single-use controller claim')
    if Path(os.environ['PID_RUN_DIR']).resolve() != directory.resolve():
        raise ValueError('Supervisor and training segment directories differ')
    deadline = float(os.environ['PID_CHILD_DEADLINE'])
    started = float(os.environ['PID_RUN_STARTED'])
    nonce = os.environ['PID_RUN_NONCE']
    ledger_path = root / 'progress.json'

    def self_expire():
        time.sleep(max(0.0, deadline - time.perf_counter()))
        os._exit(92)

    threading.Thread(target=self_expire, daemon=True).start()

    def guard():
        now = time.perf_counter()
        if now >= deadline:
            raise TimeoutError('Independent worker deadline')
        if (directory / 'stop_request.json').exists():
            raise RuntimeError(f"Supervisor stop: {read(directory / 'stop_request.json')}")
        if now - started >= config['limits']['total_soft_seconds']:
            write(directory / 'stop_request.json', {'kind': 'time', 'reason': 'Worker soft deadline', 'monotonic': now})
            raise TimeoutError('Worker soft deadline; no further update')

    def event(name, **values):
        append(directory / 'worker_events.jsonl', {'event': name, 'pid': os.getpid(),
            'monotonic': time.perf_counter(), **values})

    registered = False

    def stage(name, micro_start=None, *, complete=False):
        guard()
        write_stage(directory, {'stage': name, 'micro_started_monotonic': micro_start,
            'telemetry_required': registered, 'telemetry_complete': complete})
        event('stage', stage=name, micro_started_monotonic=micro_start)

    def wait_ack(filename, *, membership=False, timeout=None):
        end = min(deadline, time.perf_counter() + timeout) if timeout else deadline
        while time.perf_counter() < end:
            guard()
            path = directory / filename
            if path.exists():
                ack = read(path)
                correct_pid = (set(ack.get('retained_pids', [])) == {os.getpid()} if membership
                               else ack.get('worker_pid') == os.getpid())
                if ack.get('nonce') != nonce or not correct_pid:
                    raise ValueError(f'Wrong worker acknowledgement: {filename}')
                return ack
            time.sleep(0.01)
        raise TimeoutError(f'Acknowledgement deadline: {filename}')

    optimizer = model = None
    history, micro_records = [], []
    result = {'status': 'running', 'pid': os.getpid(), 'segment': segment,
        'automatic_retries': 0, 'cuda_initialized': False, 'model_loaded': False,
        'evaluation_performed': False, 'final_examples_read': False}
    write(directory / 'worker_started.json', result, exclusive=True)
    try:
        stage('before_identity_ack')
        wait_ack('membership_ack.json', membership=True, timeout=config['limits']['handshake_seconds'])
        identity = publish_worker_identity(directory, nonce)
        ack = wait_for_identity_ack(directory, nonce, timeout=config['limits']['handshake_seconds'])
        event('identity_acknowledged', acknowledgement=ack)
        result['identity_ack_received_monotonic'] = time.perf_counter()
        if any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'bitsandbytes')):
            raise RuntimeError('ML module imported before verified worker identity')
        write(directory / 'gpu_gate.json', {'identity': identity, 'acknowledgement': ack,
            'torch_imported': False, 'cuda_initialization_requested': False,
            'identity_ack_received_monotonic': result['identity_ack_received_monotonic']})
        if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
            raise RuntimeError('CUDA hidden in worker environment; do not override silently')

        stage('cpu_integrity_checks')
        preflight = read(ROOT / config['preflight_report'])
        if preflight['status'] != 'pass' or config != preflight['config']:
            raise ValueError('CPU preflight/config changed')
        verify(preflight['protected_sha256'])
        for package, expected in preflight['packages'].items():
            if version(package) != expected:
                raise ValueError(f'Frozen package changed: {package}')
        start, end = config['segment_ranges'][segment - 1]
        if config['segment_ranges'] != [[0, 200], [200, 400], [400, 600], [600, 796]]:
            raise ValueError('Only the frozen four-segment 796-update round is allowed')
        ledger = read(ledger_path)
        if (ledger['attempted_updates'] != start or ledger['completed_updates'] != start
                or ledger['total_budget'] != 796 or ledger.get('failed') or ledger.get('partial_update_possible')):
            raise ValueError('Global ledger forbids replay or continuation after failure')
        groups = preflight['groups']
        if groups != read(ROOT / config['schedule_path'])['groups']:
            raise ValueError('Original full-example schedule changed')
        with (ROOT / config['train_path']).open(encoding='utf-8') as stream:
            original = [json.loads(line) for line in stream]
        by_id = {row['source_index']: row for row in original}
        ids = [index for group in groups for index in group['source_indices']]
        if (len(groups) != 796 or any(len(group['source_indices']) != 8 for group in groups[:-1])
                or len(groups[-1]['source_indices']) != 4 or len(set(ids)) != 6364 or len(original) != 6364):
            raise ValueError('Schedule must contain the 6364 unique full examples in 795 full groups plus one four-example tail')
        if set(by_id) != set(ids):
            raise ValueError('Prepared full examples and frozen schedule differ')
        write(directory / 'worker_resources.json', gate_resources(config))
        guard()
        os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
        stage('torch_import')
        result['torch_import_started_monotonic'] = time.perf_counter()
        import torch
        result['torch_import_finished_monotonic'] = time.perf_counter()
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA initialized before explicit initialization point')
        stage('cuda_initializing')
        result['cuda_init_started_monotonic'] = time.perf_counter()
        torch.cuda.init()
        torch.cuda.synchronize()
        result['cuda_initialized'] = torch.cuda.is_initialized()
        result['cuda_init_finished_monotonic'] = time.perf_counter()
        stage('cuda_registering')
        ready = wait_ack('telemetry_ready.json', timeout=5.0)
        if ready.get('valid_sample_count', 0) < 3:
            raise ValueError('Too few valid real-PID telemetry samples')
        registered = True
        event('telemetry_ready', acknowledgement=ready)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA BF16 required')

        stage('training_imports')
        import random
        import numpy as np
        from src.data.sft_features import collate_sft_batch
        from src.train.qlora_stability import fresh_model
        from src.train.qlora_smoke import frozen_parameter_hash
        from src.train.qlora_step import adapter_parameters, check_gradients
        from src.train.sft_update_budget import JournaledAdamW
        from src.train.sft_checkpoint import save_checkpoint, load_checkpoint
        from src.train.sft_train import set_next_lr, adam_steps
        from src.train.sft_round1_plan import learning_rate_for_update
        from src.train.long_probe_core import optimizer_state_inventory
        from src.train.target_loss import target_position_loss
        from src.train.cache_release_core import backward_cache_group

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
                'num_ooms': stats.get('num_ooms') if native else None}

        def sample(name, **extra):
            guard()
            torch.cuda.synchronize()
            row = {'stage': name, 'monotonic': time.perf_counter(), **memory(), **extra}
            append(directory / 'memory_events.jsonl', row)
            if (row['peak_allocated_mib'] > config['allocated_limit_mib']
                    or row['peak_reserved_mib'] > config['reserved_limit_mib']):
                write(directory / 'stop_request.json', {'kind': 'resource',
                    'reason': f'PyTorch memory threshold at {name}', 'monotonic': time.perf_counter(), 'memory': row})
                raise RuntimeError('PyTorch allocated/reserved limit exceeded')
            guard()
            return row

        formal = preflight['formal_config']
        training = formal['training']

        def metadata(completed, base_hash):
            schedule = training['scheduler']
            return {'completed_updates': completed, 'next_group_index': completed,
                'next_update_lr': learning_rate_for_update(completed + 1, schedule['total_updates'],
                    schedule['warmup_updates'], training['optimizer']['lr']),
                'schedule_sha256': preflight['schedule_sha256'], 'run_config_sha256': preflight['run_config_sha256'],
                'base_hash': base_hash, 'protected_source_sha256': preflight['protected_sha256']}

        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.benchmark = False
        stage('load_and_prepare')
        model = fresh_model(read(ROOT / formal['model']['inherit_loading_and_lora_from']))
        result['model_loaded'] = True
        named = adapter_parameters(model)
        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in named.values())
        base_before = frozen_parameter_hash(model)
        optimizer = JournaledAdamW(list(named.values()), ledger_path=ledger_path,
            start=start, segment_end=end, total_budget=796,
            **{key: value for key, value in training['optimizer'].items() if key != 'name'})
        if optimizer.state:
            raise ValueError('New optimizer state is not empty')
        sample('load_end')
        if start:
            stage('restore_complete_checkpoint')
            restored = load_checkpoint(root / 'checkpoints' / f'update_{start:06d}', model, optimizer, metadata(start, base_before))
            if (adam_steps(optimizer) != [float(start)]
                    or optimizer_state_inventory(optimizer)['cuda_bytes'] != 2 * parameter_bytes
                    or not restored['verification']['cuda_rng_restored']):
                raise ValueError('Restored CUDA Adam/RNG state mismatch')
            write(directory / 'restore_verification.json', restored)
            sample('restore_end')
        write(directory / 'initial_state.json', {'base_hash': base_before,
            'completed_updates': start, 'adam_steps': adam_steps(optimizer),
            'optimizer': optimizer_state_inventory(optimizer),
            'trainable_parameters': sum(parameter.numel() for parameter in named.values()),
            'fresh_original_base': True, 'restored_this_run_checkpoint': bool(start),
            'restored_old_run': False})
        model.train()

        for group_index in range(start, end):
            update = group_index + 1
            group = groups[group_index]
            stage(f'update_{update}_setup')
            rows = [by_id[index] for index in group['source_indices']]
            batches = [collate_sft_batch([row], 151643, max_length=4096) for row in rows]
            optimizer.zero_grad(set_to_none=True)
            before = {name: parameter.detach().cpu().clone() for name, parameter in named.items()}
            lr = set_next_lr(optimizer, group_index, training)
            state_before = optimizer_state_inventory(optimizer)
            if (state_before['cuda_bytes'] != (0 if update == 1 else 2 * parameter_bytes)
                    or adam_steps(optimizer) != ([] if update == 1 else [float(group_index)])):
                raise ValueError('Persistent Adam state/cursor mismatch')
            micro_started = None
            cache_records = []

            def observe(event_name, index, payload):
                nonlocal micro_started
                row = rows[index]
                if event_name == 'micro_start':
                    micro_started = time.perf_counter()
                    stage(f'update_{update}_micro_{index + 1}_L{row["sequence_length"]}', micro_started)
                    torch.cuda.reset_peak_memory_stats()
                if event_name == 'forward_end':
                    if (payload['input_shape'] != [1, row['sequence_length']]
                            or payload['valid_target_count'] != row['supervised_token_count']):
                        raise ValueError('Full input or assistant-only supervision changed')
                record = sample(event_name, update=update, microbatch=index + 1,
                    source_index=row['source_index'], **payload)
                if event_name == 'micro_end':
                    record.update(seconds=time.perf_counter() - micro_started,
                        sequence_length=row['sequence_length'], valid_targets=row['supervised_token_count'])
                    micro_records.append(record)
                    append(directory / 'micro_intervals.jsonl', record)
                    if record['seconds'] >= config['limits']['micro_soft_seconds']:
                        write(directory / 'stop_request.json', {'kind': 'time',
                            'reason': 'Microbatch soft limit at boundary', 'monotonic': time.perf_counter()})
                        raise TimeoutError('Microbatch soft limit; no next microbatch or update')
                    stage(f'update_{update}_between_micro')

            def retained_signature():
                gradients = [(name, parameter.grad.data_ptr(), parameter.grad._version,
                    tuple(parameter.grad.shape), str(parameter.grad.dtype))
                    for name, parameter in named.items() if parameter.grad is not None]
                states = [(id(parameter), key, value.data_ptr(), value._version, tuple(value.shape),
                    str(value.dtype), str(value.device)) for parameter, state_values in optimizer.state.items()
                    for key, value in state_values.items() if isinstance(value, torch.Tensor)]
                return gradients, states

            def release(index):
                signature = retained_signature()
                if len(signature[0]) != len(named):
                    raise ValueError('Missing accumulated gradients before cache release')
                before_cache = sample('cache_before', update=update, microbatch=index + 1)
                tick = time.perf_counter()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after_cache = sample('cache_after', update=update, microbatch=index + 1)
                if retained_signature() != signature:
                    raise ValueError('Cache release changed live gradient or Adam identity/version')
                cache_row = {'update': update, 'microbatch': index + 1,
                    'seconds': time.perf_counter() - tick, 'before': before_cache, 'after': after_cache,
                    'gradient_and_adam_identity_version_preserved': True}
                cache_records.append(cache_row)
                append(directory / 'cache_releases.jsonl', cache_row)

            def loss_fn(current_model, batch):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return target_position_loss(current_model, batch)

            group_result = backward_cache_group(model, batches, device='cuda:0', loss_fn=loss_fn,
                observe=observe, release=release)
            guard()
            if group_result['total_target_tokens'] != group['group_target_tokens'] or len(cache_records) != len(rows):
                raise ValueError('Incomplete group or wrong effective-token denominator')
            stage(f'update_{update}_gradient_and_clip')
            gradients = check_gradients(model)
            write(directory / f'update_{update}_gradients.json', gradients)
            torch.nn.utils.clip_grad_norm_(list(named.values()), training['max_gradient_norm'], error_if_nonfinite=True)
            post_clip = check_gradients(model)['global_l2_norm']
            if post_clip > training['max_gradient_norm'] + 1e-5:
                raise ValueError('Gradient clipping failed')
            sample('gradient_clip_end', update=update)
            stage(f'update_{update}_optimizer_step')
            guard()
            optimizer.step()
            sample('optimizer_end', update=update)
            state_after = optimizer_state_inventory(optimizer)
            if (state_after['cuda_bytes'] != 2 * parameter_bytes
                    or state_after['parameter_states'] != len(named) or adam_steps(optimizer) != [float(update)]):
                raise ValueError('Adam moments, state count or step mismatch')
            changes = []
            for name, parameter in named.items():
                value = parameter.detach().cpu()
                if not torch.isfinite(value).all().item():
                    raise ValueError('Nonfinite adapter')
                changes.append({'name': name, 'changed_elements': int(torch.count_nonzero(value - before[name]))})
            if not any(row['changed_elements'] for row in changes):
                raise ValueError('No adapter parameter changed')
            write(directory / f'update_{update}_adapter_changes.json', changes)
            next_lr = set_next_lr(optimizer, update, training)
            optimizer.zero_grad(set_to_none=True)
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise ValueError('Post-update gradients not cleared')
            record = {'update': update, 'source_indices': group['source_indices'], **group_result,
                'lr_used': lr, 'next_update_lr': next_lr, 'grad_norm_before_clip': gradients['global_l2_norm'],
                'grad_norm_after_clip': post_clip, 'gradients_finite_nonzero': True,
                'optimizer_cuda_mib_before': state_before['cuda_bytes'] / 2**20,
                'optimizer_cuda_mib_after': state_after['cuda_bytes'] / 2**20,
                'cache_releases': len(cache_records),
                'changed_parameter_elements': sum(row['changed_elements'] for row in changes)}
            history.append(record)
            write(directory / 'history.json', history)
            del before, changes, batches
            print(f'Completed update {update}; token-weighted loss={group_result["group_loss"]:.6f}', flush=True)

        stage(f'save_complete_checkpoint_{end}')
        saved = save_checkpoint(root / 'checkpoints' / f'update_{end:06d}', model, optimizer, metadata(end, base_before))
        write(directory / 'checkpoint_save.json', saved)
        if not saved['cuda_rng_saved']:
            raise ValueError('CUDA RNG not saved')
        sample('save_end')
        stage('final_base_audit')
        base_after = frozen_parameter_hash(model)
        write(directory / 'base_hashes.json', {'before': base_before, 'after': base_after})
        if base_before != base_after:
            raise ValueError('Frozen base changed')
        sample('final_audit_end')
        stage('segment_complete', complete=True)
        closed = wait_ack('telemetry_closed.json', timeout=2.0)
        result.update(status='pass', start=start, end=end, history=history,
            global_completed_updates=optimizer.completed_updates, completed_microbatches=len(micro_records),
            base_unchanged=True, cuda_rng_saved=True, telemetry_closed=closed,
            segment_seconds=time.perf_counter() - started)
        write(directory / 'worker_result.json', result)
        return 0
    except Exception as error:
        # No retry and no optimizer step on an incomplete accumulation group.
        ledger = read(ledger_path)
        ledger.update(failed=True, error=f'{type(error).__name__}: {error}', failed_segment=segment)
        write(ledger_path, ledger)
        result.update(status='failed', error=f'{type(error).__name__}: {error}',
            history=history, completed_microbatches=len(micro_records), ledger=ledger,
            failed_stage=read(directory / 'stage.json') if (directory / 'stage.json').exists() else None,
            segment_seconds=time.perf_counter() - started)
        (directory / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write(directory / 'worker_result.json', result)
        event('failed', error=result['error'])
        return 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment', type=int, choices=(1, 2, 3, 4), required=True)
    raise SystemExit(run(parser.parse_args().segment))
