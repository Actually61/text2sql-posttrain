"""One bounded CPU integration of weighted cache accumulation and 2+2 state restore.

Only synthetic tiny-model tensors are optimized. Selected real training rows are
read for immutable-content checks. Formal PEFT/GPU/cross-process restore is not
performed by this acceptance; the separate GPU pilot must establish that evidence.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import gc
import importlib.abc
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'outputs/checks/sft_round2_pilot_cpu_v2'
REPORT = ROOT / 'reports/sft_round2_pilot_cpu_v2_checks.json'
PREPARED = ROOT / 'data/processed/spider_sft_round2_pilot_v1'
CONFIG = ROOT / 'configs/sft_round2_pilot_v2.json'
FREEZE = ROOT / 'reports/sft_round2_pilot_v2_freeze.json'
SOURCES = ['src/train/test_round2_pilot_cpu_v2.py', 'src/train/sft_round2_weight_core_v1.py',
    'src/train/sft_round2_pilot_v2.py', 'src/train/sft_round2_pilot_v2_worker.py',
    'src/train/sft_round2_binding_v1.py', 'src/train/test_round2_weight_cpu_v1.py',
    'src/train/weighted_token_accumulation_v1.py',
    'src/train/sft_checkpoint_cpu_v1.py', 'src/train/sft_update_budget.py',
    'src/train/target_loss.py', 'src/train/token_accumulation.py',
    'src/data/sft_features.py', 'configs/sft_round2_pilot_v2.json']
BUDGET_SECONDS = 120


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, mode='w'):
    with Path(path).open(mode, encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check_selected_rows():
    from src.train.sft_round2_binding_v1 import file_digest, build_binding
    from src.data.sft_features import collate_sft_batch
    selection_path = PREPARED / 'selection.json'
    selected_path = PREPARED / 'train_tokens.jsonl'
    selection = read(selection_path)
    rows = [json.loads(line) for line in selected_path.read_text(encoding='utf-8').splitlines()]
    by_id = {row['source_index']: row for row in rows}
    ordered = [i for group in selection['groups'] for i in group['source_indices']]
    assert len(rows) == len(by_id) == len(set(ordered)) == 32
    assert ordered == selection['selected_source_indices'] == [row['source_index'] for row in rows]
    assert file_digest(selected_path) == selection['selected_sha256']
    selected_weights_path = PREPARED / 'weights.jsonl'
    assert file_digest(selected_weights_path) == selection['weights_sha256']
    selected_weights = [json.loads(line) for line in selected_weights_path.read_text(encoding='utf-8').splitlines()]
    for path, sha in selection['source_sha256'].items():
        assert file_digest(ROOT / path) == sha, f'Selection source changed: {path}'
    full_path = ROOT / 'data/processed/spider_sft_tokens_v1/train_tokens.jsonl'
    full_count = matches = global_length = global_targets = 0
    with full_path.open(encoding='utf-8') as stream:
        for line in stream:
            original = json.loads(line)
            full_count += 1
            global_length = max(global_length, original['sequence_length'])
            global_targets = max(global_targets, original['supervised_token_count'])
            if original['source_index'] in by_id:
                assert original == by_id[original['source_index']], 'Selected row differs from frozen original'
                matches += 1
    assert full_count == 6364 and matches == 32
    assert max(row['sequence_length'] for row in rows) == global_length == selection['max_sequence_length'] == 3761
    assert max(row['supervised_token_count'] for row in rows) == global_targets == selection['max_answer_tokens']
    global_weights = {row['source_index']: row['loss_weight'] for row in
        (json.loads(line) for line in (ROOT / 'data/processed/spider_round2_weight_plan_v1/train_loss_weights.jsonl').read_text(encoding='utf-8').splitlines())}
    assert len(selected_weights) == 32 and {row['source_index'] for row in selected_weights} == set(by_id)
    assert all(item['loss_weight'] == global_weights[item['source_index']] for item in selected_weights)
    buckets = {str(bound): 0 for bound in (512, 1024, 2048, 4096)}
    weight_counts = {'1': 0, '2': 0}
    for ordinal, group in enumerate(selection['groups']):
        assert group['group_index'] == ordinal and group['update'] == ordinal + 1
        assert len(group['source_indices']) == len(group['target_counts']) == len(group['loss_weights']) == 8
        counts, lengths = [], []
        for i in group['source_indices']:
            row = by_id[i]
            assert row['project_split'] == 'train'
            batch = collate_sft_batch([row], 151643, max_length=4096)
            for field in ('input_ids', 'attention_mask', 'labels'):
                assert batch[field][0].tolist() == row[field], 'CPU collation altered selected content'
            count = int((batch['labels'][:, 1:] != -100).sum())
            assert count == row['supervised_token_count']
            counts.append(count)
            lengths.append(row['sequence_length'])
            bound = next(bound for bound in (512, 1024, 2048, 4096) if row['sequence_length'] <= bound)
            buckets[str(bound)] += 1
            weight_counts[str(global_weights[i])] += 1
        weights = [global_weights[i] for i in group['source_indices']]
        denominator = sum(w * n for w, n in zip(weights, counts))
        assert group['sequence_lengths'] == lengths and group['target_counts'] == counts
        assert group['group_target_tokens'] == sum(counts) and group['group_input_tokens'] == sum(lengths)
        assert group['loss_weights'] == weights and group['weighted_target_denominator'] == denominator
        assert group['micro_mean_loss_coefficients'] == [w * n / denominator for w, n in zip(weights, counts)]
    assert len(selection['groups']) == 4 and all(value == 8 for value in buckets.values())
    assert buckets == selection['bucket_counts'] and weight_counts == selection['weight_counts']
    assert all(weight_counts[str(w)] > 0 for w in (1, 2))
    worker_path = ROOT / 'src/train/sft_round2_pilot_v2_worker.py'
    worker_tree = ast.parse(worker_path.read_text(encoding='utf-8'))
    definitions = [node for node in worker_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'verify_selected_group_weights']
    assert len(definitions) == 1
    namespace = {'math': math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(worker_path) + '::pure_weight_guard', 'exec'), namespace)
    worker_guard = namespace['verify_selected_group_weights']
    assert worker_guard(rows, selected_weights, selection['groups'])['status'] == 'pass'
    bad_groups = copy.deepcopy(selection['groups'])
    bad_groups[0]['weighted_target_denominator'] += 1
    try:
        worker_guard(rows, selected_weights, bad_groups)
    except ValueError:
        pass
    else:
        raise AssertionError('Actual worker guard accepted a bad denominator')
    files = {'train_tokens': str(full_path.relative_to(ROOT)),
        'weights': 'data/processed/spider_round2_weight_plan_v1/train_loss_weights.jsonl',
        'original_schedule': 'data/processed/spider_sft_round1_plan_v1/epoch_0.json',
        'weighted_groups': 'data/processed/spider_round2_weight_plan_v1/weighted_groups.json',
        'pilot_selection': str(selection_path.relative_to(ROOT)),
        'selected_tokens': str(selected_path.relative_to(ROOT)),
        'selected_weights': str(selected_weights_path.relative_to(ROOT)),
        'pilot_config': str(CONFIG.relative_to(ROOT)),
        'weighted_cache_source': 'src/train/sft_round2_weight_core_v1.py'}
    binding = build_binding(ROOT, files)
    write(RUN / 'objective_binding.json', binding, 'x')
    return {'status': 'pass', 'source_indices': ordered, 'bucket_counts': buckets,
        'weight_counts': weight_counts, 'max_sequence_length': global_length, 'max_answer_tokens': global_targets,
        'selected_rows_equal_original': True, 'full_training_rows_scanned': full_count,
        'data_sha256': {str(path.relative_to(ROOT)): file_digest(path) for path in (selection_path, selected_path, selected_weights_path)},
        'actual_worker_pure_guard_positive_and_negative': True,
        'actual_training_rows_optimized_on_cpu': 0}, selection, binding, files


def check_worker_wiring():
    for path in SOURCES:
        if path.endswith('.py'):
            ast.parse((ROOT / path).read_text(encoding='utf-8'), filename=path)
    source = (ROOT / 'src/train/sft_round2_pilot_v2_worker.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'backward_weighted_cache_group']
    assert len(calls) == 1, 'Worker must call the CPU-tested weighted cache core exactly once in its loop'
    assert {'rows', 'group', 'device', 'observe', 'release'} <= {item.arg for item in calls[0].keywords}
    assert 'objective_binding' in source and 'verify_binding' in source
    assert 'sft_round2_weight_core_v1' in source
    return {'status': 'pass', 'sources_parse': True, 'weighted_core_called_with_frozen_rows_and_group': True,
            'worker_imported': False, 'scope': 'AST wiring check, not execution of GPU worker'}


def four_update_check(selection, binding, binding_files, deadline):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from src.train import test_round2_weight_cpu_v1 as toy
    from src.train.sft_round2_weight_core_v1 import backward_weighted_cache_group
    from src.train.weighted_token_accumulation_v1 import backward_weighted_token_group
    from src.train.sft_checkpoint_cpu_v1 import state_fingerprint
    from src.train.sft_round2_binding_v1 import build_binding, verify_binding, file_digest
    from src.train.sft_update_budget import JournaledAdamW, read_ledger, write_atomic
    toy.torch, toy.F = torch, F
    torch.set_num_threads(2)

    class CPUJournaledAdamW(JournaledAdamW):
        def _accelerator_graph_capture_health_check(self):
            assert all(not group.get('capturable') and not group.get('fused') for group in self.param_groups)
            assert all(p.device.type == 'cpu' for group in self.param_groups for p in group['params'])
            assert all(value.device.type == 'cpu' for state in self.state.values()
                       for value in state.values() if isinstance(value, torch.Tensor))

    def rng():
        return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch_cpu': torch.get_rng_state().clone()}

    def seed(value):
        random.seed(value); np.random.seed(value); torch.random.default_generator.manual_seed(value)

    groups = []
    for g, real in enumerate(selection['groups']):
        counts = [1 + (g * 7 + 2 * i) % 9 for i in range(8)]
        batch = toy.make_batch(counts)
        micros = toy.split_microbatches(batch)
        weights = real['loss_weights']
        denominator = sum(w * n for w, n in zip(weights, counts))
        group = {'source_indices': real['source_indices'], 'target_counts': counts, 'loss_weights': weights,
                 'group_target_tokens': sum(counts), 'weighted_target_denominator': denominator,
                 'micro_mean_loss_coefficients': [w * n / denominator for w, n in zip(weights, counts)]}
        rows = [{'source_index': i, 'supervised_token_count': n} for i, n in zip(real['source_indices'], counts)]
        groups.append((micros, rows, group))
    schedule_hash = state_fingerprint(groups)
    # The new hook-enabled core must also equal the already independently checked
    # weighted helper; 4-vs-2+2 alone could let an identical formula bug pass twice.
    reference_model = toy.make_model(795, low_rank=True)
    core_model = copy.deepcopy(reference_model)
    batches, metadata_rows, metadata_group = next(item for item in groups if set(item[2]['loss_weights']) == {1, 2})
    reference_loss = backward_weighted_token_group(reference_model, batches,
        metadata_group['loss_weights'], device='cpu')
    core_loss = backward_weighted_cache_group(core_model, batches, rows=metadata_rows,
        group=metadata_group, device='cpu')
    assert core_loss['group_loss'] == reference_loss['group_loss']
    assert state_fingerprint({name: p.grad for name, p in toy.parameters(core_model).items()}) == state_fingerprint(
        {name: p.grad for name, p in toy.parameters(reference_model).items()})

    def initialize(name, start=0, segment_end=4, ledger=None):
        directory = RUN / name
        directory.mkdir()
        if ledger is None:
            ledger = directory / 'progress.json'
            write_atomic(ledger, {'attempted_updates': 0, 'completed_updates': 0, 'total_budget': 4,
                                  'partial_update_possible': False, 'failed': False})
        model = toy.make_model(801, low_rank=True)
        opt = CPUJournaledAdamW(list(toy.parameters(model).values()), ledger_path=ledger,
            start=start, segment_end=segment_end, total_budget=4, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.0, foreach=False, capturable=False, fused=False)
        seed(802)
        return model, opt, ledger

    def update(model, opt, g):
        if time.perf_counter() >= deadline:
            raise TimeoutError('CPU integration deadline exceeded')
        batches, rows, group = groups[g]
        opt.zero_grad(set_to_none=True)
        for param_group in opt.param_groups:
            param_group['lr'] = 1e-4 * (g + 1) / 24
        draws = [random.random(), float(np.random.rand()), *torch.rand(3, device='cpu').tolist()]
        named = toy.parameters(model)
        parameter_before = state_fingerprint(model.state_dict())
        adam_before = state_fingerprint(opt.state_dict())
        events, releases = [], []

        def observe(stage, index, payload):
            if time.perf_counter() >= deadline:
                raise TimeoutError('CPU integration deadline exceeded during a group')
            events.append([index, stage])
            assert state_fingerprint(model.state_dict()) == parameter_before
            assert state_fingerprint(opt.state_dict()) == adam_before
            if stage == 'forward_end':
                assert payload['input_shape'] == list(batches[index]['input_ids'].shape)
                assert payload['sample_weight'] == group['loss_weights'][index]
                assert payload['weight'] == group['micro_mean_loss_coefficients'][index]

        def release(index):
            assert events[-1] == [index, 'backward_end']
            gradients = {name: p.grad for name, p in named.items()}
            assert all(value is not None for value in gradients.values())
            before_hash = state_fingerprint(gradients)
            identities = {name: id(p.grad) for name, p in named.items()}
            gc.collect()  # CPU simulation, no CUDA cache call.
            assert identities == {name: id(p.grad) for name, p in named.items()}
            assert before_hash == state_fingerprint({name: p.grad for name, p in named.items()})
            assert adam_before == state_fingerprint(opt.state_dict())
            releases.append(index); events.append([index, 'release'])

        result = backward_weighted_cache_group(model, batches, rows=rows, group=group,
                                               device='cpu', observe=observe, release=release)
        assert events == [[i, stage] for i in range(8) for stage in
                          ('micro_start', 'forward_end', 'backward_end', 'release', 'micro_end')]
        assert releases == list(range(8))
        grads = {name: p.grad.detach().clone() for name, p in named.items()}
        assert all(torch.isfinite(value).all() for value in grads.values())
        assert all(torch.count_nonzero(value) for value in grads.values())
        norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1.0, error_if_nonfinite=True)
        opt.step()
        assert opt.completed_updates == g + 1
        assert {float(state['step']) for state in opt.state.values()} == {float(g + 1)}
        assert state_fingerprint(model.state_dict()) != parameter_before
        for param_group in opt.param_groups:
            param_group['lr'] = 1e-4 * (g + 2) / 24
        opt.zero_grad(set_to_none=True)
        return {'update': g + 1, 'group_loss': result['group_loss'], 'weighted_denominator': result['weighted_target_denominator'],
            'weights': result['loss_weights'], 'gradient_sha256': state_fingerprint(grads), 'gradient_norm': float(norm),
            'model_sha256': state_fingerprint(model.state_dict()), 'optimizer_sha256': state_fingerprint(opt.state_dict()),
            'rng_sha256': state_fingerprint(rng()), 'rng_draws': draws, 'next_lr': opt.param_groups[0]['lr'],
            'next_group_index': g + 1, 'cache_callbacks': len(releases), 'cache_gradient_adam_preserved': True}

    continuous, full_opt, full_ledger = initialize('continuous')
    expected = [update(continuous, full_opt, i) for i in range(4)]
    split, split_opt, split_ledger = initialize('split', segment_end=2)
    actual = [update(split, split_opt, i) for i in range(2)]
    assert actual == expected[:2]
    checkpoint = RUN / 'split/checkpoint_2'
    checkpoint.mkdir()
    payload = {'model': split.state_dict(), 'optimizer': split_opt.state_dict(), 'rng': rng()}
    torch.save(payload, checkpoint / 'state.pt')
    metadata = {'format': 'cpu_round2_integration_fixture_NOT_PEFT', 'completed_updates': 2,
                'next_group_index': 2, 'next_update_lr': split_opt.param_groups[0]['lr'],
                'objective_binding': binding, 'state_sha256': file_digest(checkpoint / 'state.pt')}
    write(checkpoint / 'metadata.json', metadata, 'x')
    restored, restored_opt, _ = initialize('restored', start=2, segment_end=4, ledger=split_ledger)
    seed(999)
    current_binding = build_binding(ROOT, binding_files)
    verify_binding(read(checkpoint / 'metadata.json')['objective_binding'], current_binding)
    assert file_digest(checkpoint / 'state.pt') == metadata['state_sha256']
    loaded = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
    assert state_fingerprint(loaded) == state_fingerprint(payload)
    restored.load_state_dict(loaded['model'], strict=True)
    restored_opt.load_state_dict(loaded['optimizer'])
    random.setstate(loaded['rng']['python']); np.random.set_state(loaded['rng']['numpy'])
    torch.set_rng_state(loaded['rng']['torch_cpu'])
    assert state_fingerprint(restored.state_dict()) == state_fingerprint(split.state_dict())
    assert state_fingerprint(restored_opt.state_dict()) == state_fingerprint(split_opt.state_dict())
    assert state_fingerprint(rng()) == state_fingerprint(loaded['rng'])
    actual.extend(update(restored, restored_opt, i) for i in range(2, 4))
    assert actual == expected, '4 versus 2+2 differs in gradients/parameters/Adam/LR/RNG/cursor'
    assert read_ledger(full_ledger) == read_ledger(split_ledger)
    before = read_ledger(split_ledger)
    try:
        restored_opt.step()
    except RuntimeError as error:
        assert 'budget exhausted' in str(error)
    else:
        raise AssertionError('Fifth update accepted')
    assert read_ledger(split_ledger) == before
    try:
        CPUJournaledAdamW(list(toy.parameters(restored).values()), ledger_path=split_ledger, start=2,
                         segment_end=4, total_budget=4, lr=1e-4)
    except RuntimeError as error:
        assert 'Ledger mismatch' in str(error)
    else:
        raise AssertionError('Ledger rewind accepted')
    assert state_fingerprint(groups) == schedule_hash
    baseline_state = state_fingerprint([restored.state_dict(), restored_opt.state_dict(), rng()])
    mismatch = copy.deepcopy(binding)
    mismatch['files']['weights']['sha256'] = '0' * 64
    try:
        verify_binding(mismatch, current_binding)
    except ValueError:
        assert baseline_state == state_fingerprint([restored.state_dict(), restored_opt.state_dict(), rng()])
    else:
        raise AssertionError('Wrong weight binding accepted')
    # Callback failure must propagate before any optimizer step or ledger mutation.
    interrupted, interrupted_opt, interrupted_ledger = initialize('callback_stop', segment_end=2)
    batches, rows, group = groups[0]
    def stop_release(index):
        raise RuntimeError('injected_release_stop')
    try:
        backward_weighted_cache_group(interrupted, batches, rows=rows, group=group, device='cpu', release=stop_release)
    except RuntimeError as error:
        assert str(error) == 'injected_release_stop'
        assert read_ledger(interrupted_ledger)['attempted_updates'] == 0
        assert interrupted_opt.completed_updates == 0
    else:
        raise AssertionError('Callback failure was swallowed')
    write(RUN / 'continuous_updates.json', expected, 'x')
    write(RUN / 'resumed_updates.json', actual, 'x')
    return {'status': 'pass', 'exact_comparison': True, 'maximum_parameter_difference': 0,
            'four_update_records': actual, 'cache_callbacks_per_path': 32,
            'cpu_optimizer_updates_total': 8, 'real_gpu_optimizer_updates': 0,
            'new_cache_core_matches_prevalidated_weighted_helper_exactly': True,
            'checkpoint_restored_model_optimizer_rng_exact': True,
            'objective_binding_recomputed_before_payload_load': True,
            'wrong_binding_rejected_without_mutation': True, 'callback_failure_stops_without_update': True,
            'fifth_update_rejected': True, 'ledger_rewind_rejected': True, 'final_ledger': before,
            'limitation': 'Synthetic CPU fresh-object disk resume; CUDA cache and real cross-process PEFT resume remain for GPU pilot'}


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('CPU integration already claimed; no automatic retry or overwrite')
    RUN.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + BUDGET_SECONDS
    record = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'budget_seconds': BUDGET_SECONDS, 'attempts': 1, 'automatic_retries': 0, 'checks': {},
        'real_model_loaded': False, 'gpu_training_updates': 0, 'final_examples_read': False,
        'validation_examples_read': False, 'attempted_cuda_initializations': 0,
        'attempted_forbidden_imports': [], 'attempted_non_cpu_operations': 0}
    write(REPORT, record, 'x'); write(RUN / 'claim.json', {'pid': os.getpid(), **record}, 'x')
    done = threading.Event()
    def expire():
        if not done.wait(max(0.0, deadline - time.perf_counter())):
            record.update(status='failed', error='CPU integration exceeded 120-second deadline')
            try:
                write(RUN / 'timeout.json', record, 'x'); write(REPORT, record)
            finally:
                os._exit(124)
    threading.Thread(target=expire, daemon=True).start()
    try:
        from src.train.sft_round2_binding_v1 import file_digest
        frozen = read(FREEZE)
        record['freeze_manifest_sha256'] = file_digest(FREEZE)
        assert set(SOURCES).issubset(frozen['source_sha256'])
        def verify_frozen():
            assert file_digest(FREEZE) == record['freeze_manifest_sha256'], 'Freeze manifest changed'
            for mapping in ('source_sha256', 'previous_attempt_sha256'):
                for name, expected in frozen[mapping].items():
                    assert file_digest(ROOT / name) == expected, f'Frozen file changed: {name}'
        verify_frozen()
        record['source_sha256'] = dict(frozen['source_sha256'])
        for source in record['source_sha256']:
            destination = RUN / 'source_snapshot' / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, destination)
        record['checks']['worker_wiring'] = check_worker_wiring()
        class BlockImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'peft', 'bitsandbytes', 'transformers'}:
                    record['attempted_forbidden_imports'].append(fullname)
                    raise ImportError(f'CPU integration forbids {fullname}')
        sys.meta_path.insert(0, BlockImports())
        import torch
        assert not torch.cuda.is_initialized()
        from torch.utils._python_dispatch import TorchDispatchMode
        def block_init(*args, **kwargs):
            record['attempted_cuda_initializations'] += 1
            raise RuntimeError('CUDA initialization forbidden in CPU integration')
        def check_cpu(value):
            if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
                record['attempted_non_cpu_operations'] += 1
                raise RuntimeError('Non-CPU tensor forbidden')
            if isinstance(value, (tuple, list)):
                for item in value: check_cpu(item)
            if isinstance(value, dict):
                for item in value.values(): check_cpu(item)
        class CPUOnly(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                check_cpu(args); check_cpu(kwargs)
                if kwargs.get('device') is not None and torch.device(kwargs['device']).type != 'cpu':
                    record['attempted_non_cpu_operations'] += 1
                    raise RuntimeError('Non-CPU allocation forbidden')
                result = func(*args, **kwargs)
                check_cpu(result)
                return result
        with patch.object(torch.cuda, '_lazy_init', block_init), patch.object(torch._C, '_cuda_init', block_init), CPUOnly():
            selected, selection, binding, files = check_selected_rows()
            record['checks']['selected_32_full_examples'] = selected
            print('PASS selected 32 complete training rows, all length buckets and weights', flush=True)
            record['checks']['four_update_resume'] = four_update_check(selection, binding, files, deadline)
        assert not torch.cuda.is_initialized()
        assert not record['attempted_forbidden_imports'] and record['attempted_cuda_initializations'] == record['attempted_non_cpu_operations'] == 0
        verify_frozen()
        record.update(status='pass', cuda_initialized=False, cpu_optimizer_updates=8,
                      frozen_sources_verified_before_and_after=True)
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        done.set()
        record['seconds'] = time.perf_counter() - started
        write(REPORT, record); write(RUN / 'result.json', record, 'x')
    print(json.dumps({key: record.get(key) for key in ('status', 'seconds', 'cuda_initialized', 'error')}, ensure_ascii=False), flush=True)
    return 0 if record['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
