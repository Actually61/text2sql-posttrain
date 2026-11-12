"""One bounded CPU-only weighted-loss acceptance; never import PEFT/Transformers.

Synthetic tiny causal model only. Frozen real token data is inspected, not trained.
New outputs only; an exclusive claim prevents silently rerunning this acceptance.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib.abc
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'outputs/checks/sft_round2_weight_cpu_v1'
REPORT = ROOT / 'reports/sft_round2_weight_cpu_v1_checks.json'
CONFIG = ROOT / 'configs/sft_round2_weight_proposal_v1.json'
BUDGET_SECONDS = 120
ATOL, RTOL = 1e-6, 1e-5  # Fixed before execution; do not widen after failures.
VOCAB, EOS, PAD = 67, 2, 0
SOURCES = ['src/train/weighted_token_accumulation_v1.py', 'src/train/sft_round2_binding_v1.py',
           'src/train/round2_binding_fixtures_v1.py', 'src/train/test_round2_weight_cpu_v1.py',
           'src/train/target_loss.py', 'src/train/token_accumulation.py',
           'src/train/sft_checkpoint_cpu_v1.py']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value, mode='w'):
    with Path(path).open(mode, encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def actual_data_preflight():
    from src.train.sft_round2_binding_v1 import build_binding, file_digest, verify_group_manifest
    config = read(CONFIG)
    if config['gpu_training_enabled'] or config['evaluation_enabled']:
        raise ValueError('Proposal must not enable real training/evaluation')
    manifest = read(ROOT / config['manifest'])
    inventory = read(ROOT / 'data/processed/spider_round2_train_inventory_v1/inventory.json')
    original = read(ROOT / config['frozen_schedule'])
    weighted = read(ROOT / config['weighted_groups'])
    weights = [json.loads(line) for line in (ROOT / config['weights']).read_text(encoding='utf-8').splitlines()]
    checked = {}
    for mapping in (manifest['source_sha256'], manifest['output_sha256'], inventory['source_sha256']):
        for path, expected in mapping.items():
            actual = file_digest(ROOT / path)
            if actual != expected:
                raise ValueError(f'Frozen artifact changed: {path}')
            checked[path] = actual
    token_meta, target_sum, max_length = [], 0, 0
    # Inspect all stored labels on CPU. Do not tokenize, truncate or use them in the toy model.
    with (ROOT / config['train_path']).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            ids, labels, mask = (row[key] for key in ('input_ids', 'labels', 'attention_mask'))
            length, start, eos = row['sequence_length'], row['prompt_length'], row['eos_position']
            if not (row['project_split'] == 'train' and len(ids) == len(labels) == len(mask) == length
                    and 1 <= start <= eos < length and all(value == 1 for value in mask)
                    and labels[:start] == [-100] * start and labels[eos + 1:] == [-100] * (length - eos - 1)
                    and labels[start:eos + 1] == ids[start:eos + 1] and ids[eos] == 151645):
                raise ValueError(f'Unexpected assistant-only/EOS layout: {row["source_index"]}')
            count = sum(value != -100 for value in labels[1:])
            if count != row['supervised_token_count']:
                raise ValueError('Actual shifted target count differs from stored metadata')
            target_sum += count
            max_length = max(max_length, length)
            token_meta.append({key: row[key] for key in ('source_index', 'db_id', 'project_split', 'supervised_token_count')})
    result = verify_group_manifest(token_meta, weights, original['groups'], weighted['groups'])
    if target_sum != 206073 or max_length != 3761 or result['databases'] != 126:
        raise ValueError('Frozen training totals differ')
    files = {'train_tokens': config['train_path'], 'original_schedule': config['frozen_schedule'],
             'weights': config['weights'], 'weighted_groups': config['weighted_groups'],
             'proposal_config': str(CONFIG.relative_to(ROOT)), 'data_manifest': config['manifest'],
             'token_manifest': 'data/processed/spider_sft_tokens_v1/manifest.json',
             'weighted_helper_source': SOURCES[0], 'binding_helper_source': SOURCES[1],
             'target_loss_source': 'src/train/target_loss.py',
             'prompt_config': 'configs/prompt.yaml', 'model_config': 'configs/model.yaml',
             'package_lock': 'requirements.lock.txt'}
    binding = build_binding(ROOT, files)
    write(RUN / 'objective_binding.json', binding, 'x')
    result.update(status='pass', actual_answer_tokens=target_sum, actual_max_length=max_length,
                  all_actual_labels_and_eos_checked=True, checked_sha256=checked,
                  tail_group=weighted['groups'][-1], no_training_examples_fed_to_toy_model=True)
    return result, binding


def make_batch(counts, *, hole=False):
    rows = []
    for index, count in enumerate(counts):
        prompt = [3 + (index * 7 + j) % (VOCAB - 3) for j in range(2 + index % 5)]
        answer = [3 + (index * 11 + j + 5) % (VOCAB - 3) for j in range(count - 1)] + [EOS]
        ids = prompt + answer + [4]  # A real template tail, ignored by labels.
        labels = [-100] * len(prompt) + answer + [-100]
        if hole and index == 1:
            labels[len(prompt) + 1] = -100
        rows.append((ids, labels))
    longest = max(len(ids) for ids, _ in rows)
    return {
        'input_ids': torch.tensor([ids + [PAD] * (longest - len(ids)) for ids, _ in rows], dtype=torch.long, device='cpu'),
        'labels': torch.tensor([labels + [-100] * (longest - len(labels)) for _, labels in rows], dtype=torch.long, device='cpu'),
        'attention_mask': torch.tensor([[1] * len(ids) + [0] * (longest - len(ids)) for ids, _ in rows], dtype=torch.long, device='cpu'),
    }


def split_microbatches(batch):
    return [{key: value[i:i + 1, :int(batch['attention_mask'][i].sum())].clone()
             for key, value in batch.items()} for i in range(batch['input_ids'].shape[0])]


def make_model(seed, low_rank=False):
    # Seed only the CPU generator; torch.manual_seed also queues accelerator seeds.
    torch.random.default_generator.manual_seed(seed)

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vocab_size=VOCAB)
            self.embedding = torch.nn.Embedding(VOCAB, 16, device='cpu')
            self.context = torch.nn.Linear(16, 16, bias=False, device='cpu')
            self.head = torch.nn.Linear(16, VOCAB, bias=False, device='cpu')
            if low_rank:
                for parameter in self.parameters():
                    parameter.requires_grad_(False)
                self.lora_A = torch.nn.Linear(16, 4, bias=False, device='cpu')
                self.lora_B = torch.nn.Linear(4, VOCAB, bias=False, device='cpu')
                # Nonzero B exercises A and B gradients; this is NOT real PEFT initialization.
            self.forward_calls = 0

        def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=None):
            self.forward_calls += 1
            if use_cache:
                raise ValueError('No KV cache in this CPU fixture')
            x = self.embedding(input_ids)
            valid = attention_mask.unsqueeze(-1).to(x.dtype)
            prefix = (x * valid).cumsum(dim=1) / valid.cumsum(dim=1).clamp_min(1)
            hidden = torch.tanh(self.context(prefix) + x)
            if logits_to_keep is not None:
                hidden = hidden.index_select(1, logits_to_keep)
            logits = self.head(hidden)
            if low_rank:
                logits = logits + 2 * self.lora_B(self.lora_A(hidden))
            return SimpleNamespace(logits=logits)

    return TinyCausalLM().float().train()


def direct_group_loss(model, batch, weights):
    # Independent reference: compute ALL logits and unreduced next-token cross entropy.
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits[:, :-1]
    targets = batch['labels'][:, 1:]
    per_token = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1),
                                ignore_index=-100, reduction='none').reshape_as(targets)
    row_weights = torch.tensor(weights, dtype=per_token.dtype, device='cpu').unsqueeze(1)
    return (per_token * row_weights).sum() / ((targets != -100) * row_weights).sum()


def make_optimizer(model):
    class CPUAdamW(torch.optim.AdamW):
        def _accelerator_graph_capture_health_check(self):
            # Current torch's default health check probes CUDA even for CPU tensors.
            # Replace ONLY this probe with a stronger CPU-only check; AdamW math is unchanged.
            if any(group.get('capturable') or group.get('fused') for group in self.param_groups):
                raise ValueError('CPU fixture forbids capturable/fused optimizer')
            tensors = [p for group in self.param_groups for p in group['params']]
            tensors += [v for state in self.state.values() for v in state.values() if isinstance(v, torch.Tensor)]
            if any(t.device.type != 'cpu' for t in tensors):
                raise ValueError('CPU AdamW received a non-CPU tensor')
    return CPUAdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4,
                    betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False,
                    capturable=False, fused=False)


def parameters(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def compare(left, right):
    if set(left) != set(right):
        raise AssertionError('Tensor names differ')
    maximum = 0.0
    for name in left:
        if left[name] is None or right[name] is None or not torch.isfinite(right[name]).all():
            raise AssertionError(f'Missing/nonfinite tensor: {name}')
        torch.testing.assert_close(right[name], left[name], atol=ATOL, rtol=RTOL)
        maximum = max(maximum, float((right[name] - left[name]).abs().max()))
    return maximum


def compare_case(case):
    from src.train.token_accumulation import backward_token_normalized_group
    batch = make_batch(case['counts'], hole=case.get('hole', False))
    prototype = make_model(case['seed'], case.get('low_rank', False))
    prototype_opt = make_optimizer(prototype)
    priming = int(case.get('existing_adam', False))
    if priming:
        direct_group_loss(prototype, batch, case['weights']).backward()
        torch.nn.utils.clip_grad_norm_(list(parameters(prototype).values()), 1.0, error_if_nonfinite=True)
        prototype_opt.step()
        prototype_opt.zero_grad(set_to_none=True)
    models = {name: copy.deepcopy(prototype) for name in ('direct', 'accumulated')}
    if len(set(case['weights'])) == 1:
        models['original_unweighted'] = copy.deepcopy(prototype)
    optimizers = {name: make_optimizer(model) for name, model in models.items()}
    for optimizer in optimizers.values():
        optimizer.load_state_dict(copy.deepcopy(prototype_opt.state_dict()))
        optimizer.zero_grad(set_to_none=True)
    before = {name: p.detach().clone() for name, p in parameters(prototype).items()}
    batch_before = {key: value.clone() for key, value in batch.items()}
    loss = direct_group_loss(models['direct'], batch, case['weights'])
    loss.backward()
    losses = {'direct': float(loss.detach())}
    with patch.object(optimizers['accumulated'], 'step', side_effect=AssertionError('No update within group')), \
         patch.object(optimizers['accumulated'], 'zero_grad', side_effect=AssertionError('No gradient clearing within group')):
        group = accumulate(models['accumulated'], split_microbatches(batch), case['weights'], device='cpu')
    losses['accumulated'] = group['group_loss']
    if 'original_unweighted' in models:
        old = backward_token_normalized_group(models['original_unweighted'], split_microbatches(batch), device='cpu')
        losses['original_unweighted'] = old['group_loss']
    max_grad, max_clipped, max_update, max_state = 0.0, 0.0, 0.0, 0.0
    for name, model in models.items():
        torch.testing.assert_close(torch.tensor(losses[name]), loss.detach(), atol=ATOL, rtol=RTOL)
        max_grad = max(max_grad, compare({k: p.grad for k, p in parameters(models['direct']).items()},
                                         {k: p.grad for k, p in parameters(model).items()}))
        for key, p in parameters(model).items():
            torch.testing.assert_close(p.detach(), before[key], atol=0, rtol=0)
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise AssertionError('Frozen parameters received gradients')
        if case.get('low_rank') and any(torch.count_nonzero(p.grad) == 0 for p in parameters(model).values()):
            raise AssertionError('Low-rank comparison did not exercise both matrices')
    norms = {}
    for name, model in models.items():
        norms[name] = float(torch.nn.utils.clip_grad_norm_(list(parameters(model).values()), 1.0, error_if_nonfinite=True))
    for model in models.values():
        max_clipped = max(max_clipped, compare({k: p.grad for k, p in parameters(models['direct']).items()},
                                               {k: p.grad for k, p in parameters(model).items()}))
    for optimizer in optimizers.values():
        optimizer.step()
    for name, model in models.items():
        max_update = max(max_update, compare({k: p.detach() for k, p in parameters(models['direct']).items()},
                                             {k: p.detach() for k, p in parameters(model).items()}))
        for p_ref, p_other in zip(parameters(models['direct']).values(), parameters(model).values()):
            max_state = max(max_state, compare(optimizers['direct'].state[p_ref], optimizers[name].state[p_other]))
        if not any(torch.count_nonzero(p.detach() - before[k]) for k, p in parameters(model).items()):
            raise AssertionError('No parameter changed')
        if {float(state['step']) for state in optimizers[name].state.values()} != {float(priming + 1)}:
            raise AssertionError('Optimizer update count differs')
    for key in batch:
        torch.testing.assert_close(batch[key], batch_before[key], atol=0, rtol=0)
    result = {'name': case['name'], 'status': 'pass', 'losses': losses,
              'loss_max_absolute_difference': max(abs(value - losses['direct']) for value in losses.values()),
              'gradient_max_absolute_difference': max_grad, 'clipped_gradient_max_absolute_difference': max_clipped,
              'parameter_after_update_max_absolute_difference': max_update,
              'adam_state_max_absolute_difference': max_state, 'gradient_norm_before_clip': norms,
              'group': group, 'low_rank_fixture_not_peft': case.get('low_rank', False),
              'comparison_updates_per_path': 1, 'priming_updates': priming,
              'cpu_optimizer_step_calls': priming + len(models), 'inputs_unchanged': True,
              'trainable_parameter_elements': sum(p.numel() for p in parameters(prototype).values())}
    return result, models['accumulated'], optimizers['accumulated']


def negative_math_checks():
    from src.train.target_loss import target_position_loss
    batch = make_batch([1, 9, 3, 5])
    weights = [2, 1, 2, 1]
    good = make_model(601)
    expected = direct_group_loss(good, batch, weights)
    expected.backward()
    expected_grads = {name: p.grad.clone() for name, p in parameters(good).items()}
    results = []
    for name in ('extra_divide_by_8', 'unweighted_denominator', 'equal_example_average'):
        bad = make_model(601)
        micros = split_microbatches(batch)
        counts = [int((micro['labels'][:, 1:] != -100).sum()) for micro in micros]
        losses = []
        for micro, w, n in zip(micros, weights, counts):
            loss, _ = target_position_loss(bad, micro)
            if name == 'extra_divide_by_8':
                factor = w * n / sum(x * y for x, y in zip(weights, counts)) / 8
            elif name == 'unweighted_denominator':
                factor = w * n / sum(counts)
            else:
                factor = w / sum(weights)
            (loss * factor).backward()
            losses.append(float(loss.detach()) * factor)
        grad_error = max(float((p.grad - expected_grads[k]).abs().max()) for k, p in parameters(bad).items())
        loss_error = abs(sum(losses) - float(expected.detach()))
        if grad_error <= 1e-5 or loss_error <= 1e-5:
            raise AssertionError(f'Negative control not detected: {name}')
        results.append({'name': name, 'detected': True, 'loss_error': loss_error, 'gradient_error': grad_error})
    invalids = []
    micros = split_microbatches(batch)
    ignored = {key: value.clone() for key, value in micros[-1].items()}
    ignored['labels'].fill_(-100)
    invalid_cases = [('empty', [], []), ('missing_weight', micros, [1]),
                     ('bool_weight', micros, [True, 1, 1, 1]), ('zero_weight', micros, [0, 1, 1, 1]),
                     ('nan_weight', micros, [float('nan'), 1, 1, 1]),
                     ('infinite_weight', micros, [float('inf'), 1, 1, 1]),
                     ('negative_weight', micros, [-1, 1, 1, 1]),
                     ('ignored_later_micro', micros[:-1] + [ignored], [1] * 4)]
    for name, batches, ws in invalid_cases:
        model = make_model(603)
        try:
            accumulate(model, batches, ws, device='cpu')
        except ValueError as error:
            if model.forward_calls or any(p.grad is not None for p in model.parameters()):
                raise AssertionError('Invalid input was rejected after computation began')
            invalids.append({'name': name, 'rejected_before_forward': True, 'error': str(error)})
        else:
            raise AssertionError(f'Invalid group accepted: {name}')
    return {'incorrect_formulas': results, 'invalid_inputs': invalids}


def checkpoint_checks(model, optimizer, binding):
    from src.train.sft_round2_binding_v1 import build_binding, file_digest, verify_binding
    from src.train.sft_checkpoint_cpu_v1 import state_fingerprint
    optimizer.zero_grad(set_to_none=True)
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'torch_cpu_rng': torch.get_rng_state(), 'next_group_index': 1, 'completed_updates': 1}
    destination = RUN / 'tiny_cpu_checkpoint'
    destination.mkdir()
    torch.save(payload, destination / 'state.pt')
    metadata = {'format': 'cpu_tiny_weight_probe_v1_NOT_PEFT', 'objective_binding': binding,
                'state_sha256': file_digest(destination / 'state.pt'), 'completed_updates': 1,
                'next_group_index': 1, 'next_update_lr': 1e-4}
    write(destination / 'metadata.json', metadata, 'x')
    # Recompute actual project bytes; do not derive the expected binding from checkpoint data.
    config = read(CONFIG)
    current_files = {'train_tokens': config['train_path'], 'original_schedule': config['frozen_schedule'],
                     'weights': config['weights'], 'weighted_groups': config['weighted_groups'],
                     'proposal_config': str(CONFIG.relative_to(ROOT)), 'data_manifest': config['manifest'],
                     'token_manifest': 'data/processed/spider_sft_tokens_v1/manifest.json',
                     'weighted_helper_source': SOURCES[0], 'binding_helper_source': SOURCES[1],
                     'target_loss_source': 'src/train/target_loss.py', 'prompt_config': 'configs/prompt.yaml',
                     'model_config': 'configs/model.yaml', 'package_lock': 'requirements.lock.txt'}
    current = build_binding(ROOT, current_files)
    restored = make_model(999, low_rank=True)
    restored_opt = make_optimizer(restored)
    state_load_calls = 0

    def restore(saved_metadata):
        nonlocal state_load_calls
        verify_binding(saved_metadata.get('objective_binding'), current)
        if file_digest(destination / 'state.pt') != saved_metadata['state_sha256']:
            raise ValueError('CPU fixture payload hash mismatch')
        state_load_calls += 1
        state = torch.load(destination / 'state.pt', map_location='cpu', weights_only=True)
        restored.load_state_dict(state['model'], strict=True)
        restored_opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_cpu_rng'])
        return state

    saved = read(destination / 'metadata.json')
    state = restore(saved)
    if state_fingerprint(state) != state_fingerprint(payload):
        raise AssertionError('CPU checkpoint payload round-trip changed')
    if state_fingerprint(restored.state_dict()) != state_fingerprint(model.state_dict()):
        raise AssertionError('CPU model state not restored exactly')
    if state_fingerprint(restored_opt.state_dict()) != state_fingerprint(optimizer.state_dict()):
        raise AssertionError('CPU optimizer state not restored exactly')
    before = state_fingerprint([restored.state_dict(), restored_opt.state_dict(), torch.get_rng_state()])
    rejected = []
    for name in ('missing_binding', 'wrong_weights_hash', 'wrong_data_hash', 'wrong_schedule_hash'):
        bad = copy.deepcopy(saved)
        if name == 'missing_binding':
            del bad['objective_binding']
        else:
            role = {'wrong_weights_hash': 'weights', 'wrong_data_hash': 'train_tokens',
                    'wrong_schedule_hash': 'original_schedule'}[name]
            bad['objective_binding']['files'][role]['sha256'] = '0' * 64
        try:
            restore(bad)
        except ValueError:
            if state_load_calls != 1 or before != state_fingerprint([restored.state_dict(), restored_opt.state_dict(), torch.get_rng_state()]):
                raise AssertionError('Mismatch caused payload load or state mutation')
            rejected.append(name)
        else:
            raise AssertionError('Mismatched binding accepted')
    return {'status': 'pass', 'scope': 'CPU toy disk payload and metadata only; formal PEFT/GPU resume NOT tested',
            'actual_file_binding_recomputed': True, 'model_optimizer_rng_roundtrip_exact': True,
            'payload_deserializations': state_load_calls, 'rejected_before_deserialization_and_mutation': rejected,
            'checkpoint_bytes': (destination / 'state.pt').stat().st_size}


def main():
    global torch, F, accumulate
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('CPU preflight already claimed; no automatic retry/overwrite')
    RUN.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'budget_seconds': BUDGET_SECONDS, 'cpu_threads': 2, 'atol': ATOL, 'rtol': RTOL,
              'device': 'cpu', 'dtype': 'float32', 'formal_training_updates': 0,
              'evaluation_performed': False, 'large_model_loaded': False, 'final_read': False,
              'attempted_forbidden_imports': [], 'attempted_cuda_initializations': 0,
              'attempted_non_cpu_operations': 0, 'cases': []}
    write(REPORT, record, 'x')
    write(RUN / 'claim.json', {'pid': os.getpid(), 'started_utc': record['started_utc'],
                              'budget_seconds': BUDGET_SECONDS, 'automatic_retries': 0}, 'x')
    lock = threading.Lock()
    finished = False

    def timeout():
        nonlocal finished
        with lock:
            if not finished:
                finished = True
                record.update(status='failed', error='CPU preflight exceeded 120 seconds', seconds=time.perf_counter() - started)
                try:
                    write(REPORT, record)
                    write(RUN / 'result.json', record, 'x')
                finally:
                    os._exit(124)

    timer = threading.Timer(BUDGET_SECONDS, timeout)
    timer.daemon = True
    timer.start()
    try:
        from src.train.sft_round2_binding_v1 import file_digest
        for source in SOURCES:
            target = RUN / 'source_snapshot' / source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / source, target)
        record['source_sha256'] = {name: file_digest(ROOT / name) for name in SOURCES}
        record['data'], binding = actual_data_preflight()
        from src.train.round2_binding_fixtures_v1 import run_binding_checks
        record['binding_fixture_checks'] = run_binding_checks(RUN / 'binding_fixtures')
        print('PASS frozen data, weighted schedule and binding mutation checks', flush=True)

        class BlockImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'peft', 'bitsandbytes', 'transformers'}:
                    record['attempted_forbidden_imports'].append(fullname)
                    raise ImportError(f'CPU test forbids {fullname}')

        blocker = BlockImports()
        sys.meta_path.insert(0, blocker)
        import torch
        import torch.nn.functional as F
        if torch.cuda.is_initialized():
            raise AssertionError('Plain torch import initialized CUDA')
        torch.set_num_threads(2)
        from torch.utils._python_dispatch import TorchDispatchMode

        def blocked_init(*args, **kwargs):
            record['attempted_cuda_initializations'] += 1
            raise RuntimeError('CUDA initialization forbidden')

        def check_cpu(value):
            if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
                record['attempted_non_cpu_operations'] += 1
                raise RuntimeError('Non-CPU tensor forbidden')
            if isinstance(value, (tuple, list)):
                for item in value:
                    check_cpu(item)
            if isinstance(value, dict):
                for item in value.values():
                    check_cpu(item)

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

        with patch.object(torch.cuda, '_lazy_init', blocked_init), patch.object(torch._C, '_cuda_init', blocked_init), CPUOnly():
            from src.train.weighted_token_accumulation_v1 import backward_weighted_token_group as accumulate
            tail = record['data']['tail_group']
            counts = [1, 3, 8, 2, 6, 11, 4, 7]
            cases = [
                {'name': 'all_one_full_eight', 'counts': counts, 'weights': [1] * 8, 'seed': 401},
                {'name': 'all_two_equals_original', 'counts': counts, 'weights': [2] * 8, 'seed': 401},
                {'name': 'mixed_eight_dense', 'counts': counts, 'weights': [1, 2, 1, 2, 2, 1, 1, 2], 'seed': 403},
                {'name': 'mixed_eight_low_rank', 'counts': counts, 'weights': [2, 1, 2, 1, 1, 2, 2, 1], 'seed': 405, 'low_rank': True},
                {'name': 'tail_four_actual_counts_and_weights', 'counts': tail['target_counts'], 'weights': tail['loss_weights'], 'seed': 407, 'low_rank': True},
                {'name': 'mixed_eos_only_hole_and_padding', 'counts': [1, 5, 2, 9], 'weights': [2, 1, 1, 2], 'seed': 409, 'hole': True},
                {'name': 'mixed_existing_adam_moments', 'counts': counts, 'weights': [1, 2, 1, 2, 1, 2, 1, 2], 'seed': 411, 'low_rank': True, 'existing_adam': True},
            ]
            last_model = last_optimizer = None
            for case in cases:
                result, model, optimizer = compare_case(case)
                record['cases'].append(result)
                print(f"PASS {case['name']}: loss={result['loss_max_absolute_difference']:.3g}, grad={result['gradient_max_absolute_difference']:.3g}, update={result['parameter_after_update_max_absolute_difference']:.3g}", flush=True)
                if case['name'] == 'tail_four_actual_counts_and_weights':
                    last_model, last_optimizer = model, optimizer
            record['negative_math_checks'] = negative_math_checks()
            record['checkpoint_checks'] = checkpoint_checks(last_model, last_optimizer, binding)
        if (torch.cuda.is_initialized() or record['attempted_forbidden_imports']
                or record['attempted_cuda_initializations'] or record['attempted_non_cpu_operations']):
            raise AssertionError('CPU isolation boundary was attempted')
        record.update(status='pass', cuda_initialized=False, torch_version=version('torch'),
                      cpu_test_optimizer_step_calls=sum(case['cpu_optimizer_step_calls'] for case in record['cases']),
                      toy_model='random tiny causal prefix-mean LM; optional nonzero low-rank head, NOT Qwen/PEFT/4bit',
                      adamw_cpu_guard='Only accelerator capture health probe replaced by CPU tensor/capturable checks; optimizer math unchanged',
                      gpu_or_formal_resume_validated=False, automatic_retries=0)
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        with lock:
            finished = True
            timer.cancel()
            record['seconds'] = time.perf_counter() - started
            write(REPORT, record)
            write(RUN / 'result.json', record, 'x')
    print(json.dumps({key: record.get(key) for key in ('status', 'seconds', 'cuda_initialized', 'cpu_test_optimizer_step_calls', 'error')}, ensure_ascii=False), flush=True)
    return 0 if record['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
