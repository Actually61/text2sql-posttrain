"""One CPU-only integration acceptance: one native worker and exact 2+2 resume.

The --fixture branch runs before any ML import. This file never initializes CUDA,
loads the real Qwen checkpoint, or reads a validation/final dataset.
"""
from __future__ import annotations
import ast
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
RUN = ROOT / 'outputs/checks/sft_cache_resume_cpu_v1'
REPORT = ROOT / 'reports/sft_cache_resume_cpu_checks.json'
SOURCES = ['src/train/test_sft_cache_resume.py', 'src/train/sft_cache_resume_worker.py',
    'src/train/pid_supervision_v4.py', 'src/train/windows_job_exit_wait_v3.py',
    'src/train/cache_release_core.py', 'src/train/sft_checkpoint.py',
    'src/train/sft_update_budget.py', 'src/train/sft_train.py',
    'src/train/token_accumulation.py', 'src/train/target_loss.py',
    'src/train/test_target_loss.py', 'src/train/test_token_accumulation.py',
    'configs/sft_cache_resume.json']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fixture():
    """A single real native process, with injected counters and no ML imports."""
    from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack
    directory = Path(os.environ['PID_RUN_DIR'])
    nonce = os.environ['PID_RUN_NONCE']
    deadline = min(time.perf_counter() + 8.0, float(os.environ['PID_CHILD_DEADLINE']))
    def expire():
        time.sleep(max(0.0, deadline - time.perf_counter()))
        os._exit(92)
    threading.Thread(target=expire, daemon=True).start()
    write(directory / 'stage.json', {'stage': 'cpu_identity', 'telemetry_required': False})
    identity = publish_worker_identity(directory, nonce)
    write(directory / 'fixture_worker.json', {**identity, 'sys_executable': sys.executable,
        'sys_prefix': sys.prefix, 'forbidden_imports': [name for name in
            ('torch', 'transformers', 'peft', 'bitsandbytes') if name in sys.modules]})
    def wait_file(name):
        while time.perf_counter() < deadline:
            if (directory / 'stop_request.json').exists():
                raise RuntimeError('CPU fixture received a stop request')
            if (directory / name).exists():
                value = read(directory / name)
                if value.get('nonce') != nonce:
                    raise ValueError('CPU fixture acknowledgement nonce mismatch')
                return value
            time.sleep(0.01)
        raise TimeoutError(f'CPU fixture expired while waiting for {name}')
    membership = wait_file('membership_ack.json')
    if membership['retained_pids'] != [os.getpid()]:
        raise ValueError('CPU fixture expected exactly its own process')
    ack = wait_for_identity_ack(directory, nonce, timeout=min(2.5, deadline - time.perf_counter()))
    if ack['launcher_pid'] != os.getpid():
        raise ValueError('Native launcher is not the actual single worker')
    write(directory / 'stage.json', {'stage': 'cpu_sampling', 'telemetry_required': True})
    ready = wait_file('telemetry_ready.json')
    if ready['worker_pid'] != os.getpid() or ready['valid_sample_count'] < 3:
        raise ValueError('CPU fixture counters not bound to its own PID')
    write(directory / 'stage.json', {'stage': 'cpu_complete', 'telemetry_required': True,
                                     'telemetry_complete': True})
    closed = wait_file('telemetry_closed.json')
    if closed['worker_pid'] != os.getpid() or closed['valid_sample_count'] < 3:
        raise ValueError('CPU fixture sampler close mismatch')
    forbidden = [name for name in ('torch', 'transformers', 'peft', 'bitsandbytes') if name in sys.modules]
    if forbidden:
        raise AssertionError(f'CPU identity fixture imported ML modules: {forbidden}')
    write(directory / 'worker_result.json', {'status': 'pass', 'pid': os.getpid(),
        'torch_imported': False, 'telemetry_closed': True})
    return 0


def check_native_single_worker(deadline):
    from src.train.pid_supervision_v4 import supervise
    from src.train.windows_job_process_v2 import VENV_PYTHON, NATIVE_PYTHON, normalize
    directory = RUN / 'single_native_worker'
    called = []
    def provider(pid, stage):
        # This file must exist before the supervisor can bind the worker; the
        # auxiliary fixture record is deliberately not a synchronization token.
        actual = read(directory / 'worker_identity.json')['pid']
        if actual != pid:
            raise AssertionError('Injected counter was bound to a different PID')
        called.append(pid)
        return {'pid': pid, 'status': 'ok', 'dedicated_bytes': 0, 'shared_bytes': 0,
                'fixture': 'cpu_injected_not_gpu_measurement'}
    limits = {'budget_profile': 'cpu', 'total_soft_seconds': 3.0, 'total_hard_seconds': 4.0,
        'handshake_seconds': 2.5, 'cleanup_seconds': 2.0, 'expected_process_count': 1,
        'child_self_seconds': 8.0, 'case_seconds': 9.5, 'sample_interval_seconds': 0.05,
        'resource_grace_seconds': 0.3, 'telemetry_stale_seconds': 1.0}
    result = supervise([str(VENV_PYTHON), '-X', 'utf8', str(Path(__file__).resolve()), '--fixture'],
        ROOT, dict(os.environ), directory, limits, sample_provider=provider, suite_deadline=deadline)
    write(directory / 'returned_result.json', result)  # Evidence before assertions.
    identity = read(directory / 'fixture_worker.json')
    pid = identity['pid']
    assert result['status'] == 'pass' and result['outcome'] == 'completed', result
    assert result['identity_verified'] and result['worker_pid'] == result['launcher_pid'] == result['bound_pid'] == pid
    assert result['exit_code'] == 0 and not result['killed']
    assert result['telemetry_closed'] and result['valid_sample_count'] >= 3
    assert result['job_active_pids_after_cleanup'] == [] and not result['errors'] and not result['cleanup_errors']
    assert len(result['role_exits']) == 1 and result['role_exits'][0]['alive'] is False
    assert 0 <= result['role_exits'][0]['lifetime_seconds'] <= 10
    assert called and set(called) == {pid} and result['observed_sample_pids'] == [pid]
    assert normalize(identity['executable']) == normalize(NATIVE_PYTHON)
    assert normalize(identity['sys_executable']) == normalize(VENV_PYTHON)
    assert normalize(identity['sys_prefix']) == normalize(ROOT / '.venv')
    assert not identity['forbidden_imports'] and not read(directory / 'worker_result.json')['torch_imported']
    for path in directory.glob('*.jsonl'):
        text = path.read_text(encoding='utf-8')
        assert not text or text.endswith('\n'), f'Incomplete JSONL: {path}'
        for line in text.splitlines():
            json.loads(line)
    return {'status': 'pass', 'same_native_launcher_worker_pid': pid, 'supervisor': result,
            'sample_values_are_simulated': True, 'cuda_initialized': False}


def check_original_schedule():
    from src.data.sft_features import collate_sft_batch
    selection_path = ROOT / 'data/processed/spider_sft_resume_pilot_v1/selection.json'
    features_path = ROOT / 'data/processed/spider_sft_resume_pilot_v1/features.json'
    selection, rows = read(selection_path), read(features_path)
    by_id = {row['source_index']: row for row in rows}
    ordered = [index for group in selection['groups'] for index in group['source_indices']]
    assert len(rows) == len(by_id) == len(set(ordered)) == 32
    assert ordered == selection['selected_source_indices'] == [row['source_index'] for row in rows]
    groups = []
    for group_index, group in enumerate(selection['groups']):
        assert group['group_index'] == group_index and group['update'] == group_index + 1
        assert len(group['source_indices']) == len(group['sequence_lengths']) == len(group['target_counts']) == 8
        targets = total_length = 0
        for index, expected_length, expected_targets in zip(group['source_indices'], group['sequence_lengths'], group['target_counts']):
            row = by_id[index]
            batch = collate_sft_batch([row], 151643, 4096)
            assert list(batch['input_ids'].shape) == [1, expected_length]
            assert batch['input_ids'][0].tolist() == row['input_ids']
            assert batch['attention_mask'][0].tolist() == row['attention_mask']
            assert batch['labels'][0].tolist() == row['labels']
            assert int((batch['labels'][:, 1:] != -100).sum()) == expected_targets == row['supervised_token_count']
            assert bool(batch['attention_mask'].all())
            total_length += expected_length
            targets += expected_targets
        assert targets == group['group_target_tokens'] and total_length == group['group_input_tokens']
        groups.append({'group_index': group_index, 'input_tokens': total_length, 'target_tokens': targets})
    assert len(groups) == 4 and selection['bucket_counts'] == {'512': 8, '1024': 8, '2048': 8, '4096': 8}
    assert max(row['sequence_length'] for row in rows) == 3761
    assert max(row['supervised_token_count'] for row in rows) == 202
    return {'status': 'pass', 'source_indices': ordered, 'groups': groups, 'input_tokens': 47654,
        'target_tokens': 1350, 'unchanged_full_inputs': True, 'truncation_or_deletion': False,
        'data_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in (selection_path, features_path)}}


def check_profile_and_sources():
    for name in SOURCES:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
    tree = ast.parse((ROOT / 'src/train/pid_supervision_v4.py').read_text(encoding='utf-8-sig'))
    profile_values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'profiles' for target in node.targets):
            profile_values.append(ast.literal_eval(node.value))
    assert len(profile_values) == 1 and profile_values[0]['bounded_sft'] == (304.0, 306.0)
    config = read(ROOT / 'configs/sft_cache_resume.json')
    limits = config['limits']
    assert limits['budget_profile'] == 'bounded_sft' and limits['expected_process_count'] == 1
    assert limits['cleanup_seconds'] == 2 and limits['total_soft_seconds'] == 240 and limits['total_hard_seconds'] == 300
    assert limits['child_self_seconds'] == 304 and limits['case_seconds'] == 306
    assert limits['micro_soft_seconds'] == 60 and limits['micro_hard_seconds'] == 90
    assert limits['total_hard_seconds'] + limits['cleanup_seconds'] < limits['child_self_seconds'] < limits['case_seconds']
    return {'status': 'pass', 'profile': 'bounded_sft', 'limits': limits,
            'new_sources_compile_via_ast': True, 'models_loaded_during_static_checks': False}


def check_four_update_equivalence(source_hashes, deadline):
    import numpy as np
    import torch
    from src.train.test_target_loss import make_batch, make_model
    from src.train.test_token_accumulation import microbatches, named_trainable, OPTIMIZER_KWARGS
    from src.train.cache_release_core import backward_cache_group
    from src.train.sft_checkpoint import capture_rng, load_checkpoint, save_checkpoint, state_fingerprint
    from src.train.sft_update_budget import JournaledAdamW, read_ledger, write_atomic
    from src.train.sft_round1_plan import learning_rate_for_update
    torch.set_num_threads(2)
    groups = [microbatches(make_batch([(2 + (g * 3 + i) % 11, 1 + (g + 2 * i) % 8, bool(i % 2))
                                      for i in range(8)])) for g in range(4)]
    schedule_hash = state_fingerprint(groups)
    def initialize(name):
        directory = RUN / name
        directory.mkdir()
        ledger = directory / 'progress.json'
        write_atomic(ledger, {'attempted_updates': 0, 'completed_updates': 0, 'total_budget': 4,
                              'partial_update_possible': False, 'failed': False})
        model = make_model(801, tied=True, lora=True)
        opt = JournaledAdamW(list(named_trainable(model).values()), ledger_path=ledger, start=0,
                            segment_end=4 if name == 'continuous' else 2, total_budget=4, **OPTIMIZER_KWARGS)
        random.seed(802); np.random.seed(802); torch.manual_seed(802)
        return model, opt, ledger
    def update(model, opt, group_index):
        if time.perf_counter() >= deadline:
            raise TimeoutError('CPU acceptance budget exhausted')
        named = named_trainable(model)
        before = {name: p.detach().clone() for name, p in named.items()}
        opt.zero_grad(set_to_none=True)
        lr = learning_rate_for_update(group_index + 1)
        for group in opt.param_groups:
            group['lr'] = lr
        draws = [random.random(), float(np.random.rand()), *torch.rand(3).tolist()]
        events = []
        releases = []
        adam_before = state_fingerprint(opt.state_dict())
        params_before = state_fingerprint(before)
        def observe(stage, index, payload):
            if time.perf_counter() >= deadline:
                raise TimeoutError('CPU acceptance budget exhausted inside accumulation')
            events.append([index, stage])
            if stage == 'forward_end':
                assert payload['input_shape'] == list(groups[group_index][index]['input_ids'].shape)
            assert state_fingerprint(opt.state_dict()) == adam_before
            assert state_fingerprint({name: p.detach() for name, p in named.items()}) == params_before
        def release(index):
            grads = {name: p.grad for name, p in named.items()}
            assert all(value is not None for value in grads.values())
            identities = {name: id(value) for name, value in grads.items()}
            gradient_hash = state_fingerprint(grads)
            gc.collect()  # CPU callback simulation, never torch.cuda.empty_cache().
            assert identities == {name: id(p.grad) for name, p in named.items()}
            assert gradient_hash == state_fingerprint({name: p.grad for name, p in named.items()})
            assert adam_before == state_fingerprint(opt.state_dict())
            events.append([index, 'release'])
            releases.append(index)
        result = backward_cache_group(model, groups[group_index], device='cpu', observe=observe, release=release)
        expected_events = [[index, stage] for index in range(8) for stage in
                           ('micro_start', 'forward_end', 'backward_end', 'release', 'micro_end')]
        assert events == expected_events and releases == list(range(8))
        grads = {name: p.grad.detach().clone() for name, p in named.items()}
        assert all(torch.isfinite(value).all() for value in grads.values())
        assert any(torch.count_nonzero(value) for value in grads.values())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1.0, error_if_nonfinite=True)
        opt.step()
        assert opt.completed_updates == group_index + 1
        assert sorted({float(state['step']) for state in opt.state.values()}) == [float(group_index + 1)]
        assert any(torch.count_nonzero(p.detach() - before[name]) for name, p in named.items())
        next_lr = learning_rate_for_update(group_index + 2)
        for group in opt.param_groups:
            group['lr'] = next_lr
        opt.zero_grad(set_to_none=True)
        return {'update': group_index + 1, 'group_loss': result['group_loss'],
            'target_counts': result['micro_target_counts'], 'group_targets': result['total_target_tokens'],
            'gradient_sha256': state_fingerprint(grads), 'gradient_norm': float(norm),
            'adapter_sha256': state_fingerprint({name: p.detach() for name, p in named.items()}),
            'optimizer_sha256': state_fingerprint(opt.state_dict()), 'rng_sha256': state_fingerprint(capture_rng()),
            'rng_draws': draws, 'lr_used': lr, 'next_lr': next_lr, 'next_group_index': group_index + 1,
            'cache_callbacks': len(releases), 'release_preserves_gradient_identities_values_and_adam': True}
    continuous, continuous_opt, continuous_ledger = initialize('continuous')
    expected = [update(continuous, continuous_opt, index) for index in range(4)]
    split, split_opt, split_ledger = initialize('split_resume')
    actual = [update(split, split_opt, index) for index in range(2)]
    assert actual == expected[:2]
    metadata = {'completed_updates': 2, 'next_group_index': 2, 'next_update_lr': learning_rate_for_update(3),
        'schedule_sha256': schedule_hash, 'run_config_sha256': 'b' * 64, 'base_hash': 'cpu_seed_801',
        'protected_source_sha256': source_hashes}
    checkpoint = RUN / 'split_resume/checkpoint_2'
    saved = save_checkpoint(checkpoint, split, split_opt, metadata)
    assert saved['readback_equal'] and not saved['cuda_rng_saved']
    restored = make_model(801, tied=True, lora=True)
    restored_opt = JournaledAdamW(list(named_trainable(restored).values()), ledger_path=split_ledger,
        start=2, segment_end=4, total_budget=4, **OPTIMIZER_KWARGS)
    random.seed(999); np.random.seed(999); torch.manual_seed(999)
    loaded = load_checkpoint(checkpoint, restored, restored_opt, metadata)
    assert loaded['verification']['adapter_exact'] and loaded['verification']['optimizer_exact']
    assert loaded['verification']['rng_exact'] and not loaded['verification']['cuda_rng_restored']
    assert loaded['metadata']['next_group_index'] == 2
    actual.extend(update(restored, restored_opt, index) for index in range(2, 4))
    assert actual == expected, 'Continuous versus 2+2 differs in loss, gradients, parameters, Adam, LR, RNG or cursor'
    assert read_ledger(continuous_ledger) == read_ledger(split_ledger)
    before_rejected = read_ledger(split_ledger)
    try:
        restored_opt.step()
    except RuntimeError as error:
        assert 'budget exhausted' in str(error)
    else:
        raise AssertionError('A fifth update was accepted')
    assert read_ledger(split_ledger) == before_rejected
    try:
        JournaledAdamW(list(named_trainable(restored).values()), ledger_path=split_ledger,
            start=2, segment_end=4, total_budget=4, **OPTIMIZER_KWARGS)
    except RuntimeError as error:
        assert 'Ledger mismatch' in str(error)
    else:
        raise AssertionError('Ledger rewind was accepted')
    assert state_fingerprint(groups) == schedule_hash
    assert not torch.cuda.is_initialized()
    write(RUN / 'continuous_updates.json', expected)
    write(RUN / 'resumed_updates.json', actual)
    return {'status': 'pass', 'exact_comparison': True, 'maximum_parameter_difference': 0,
        'four_update_records': actual, 'cache_callbacks_per_path': 32, 'checkpoint_restore': loaded,
        'fifth_update_rejected': True, 'ledger_rewind_rejected': True, 'final_ledger': read_ledger(split_ledger),
        'cpu_optimizer_updates_total': 8, 'real_gpu_optimizer_updates': 0,
        'limitation': 'CPU fresh-object resume and simulated cache release; GPU and cross-process results require the separately authorized pilot'}


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('Single-use CPU acceptance already claimed; no retry or overwrite')
    RUN.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started + 120.0
    report = {'status': 'running', 'attempts': 1, 'automatic_retries': 0, 'cpu_budget_seconds': 120,
        'real_model_loaded': False, 'gpu_training_updates': 0, 'final_examples_read': False,
        'validation_examples_read': False, 'checks': {}}
    write(RUN / 'claim.json', report)
    done = threading.Event()
    def hard_expire():
        if not done.wait(max(0.0, deadline - time.perf_counter())):
            write(RUN / 'cpu_budget_exceeded.json', {'status': 'failed', 'seconds': time.perf_counter() - started,
                'reason': 'CPU suite hard 120-second deadline', 'automatic_retries': 0})
            os._exit(93)
    threading.Thread(target=hard_expire, daemon=True).start()
    code = 1
    try:
        report['source_sha256'] = {name: digest(ROOT / name) for name in SOURCES}
        snapshot = RUN / 'source_snapshot'
        snapshot.mkdir()
        for name in SOURCES:
            shutil.copyfile(ROOT / name, snapshot / Path(name).name)
        report['checks']['syntax_and_budget'] = check_profile_and_sources()
        report['checks']['single_native_worker'] = check_native_single_worker(deadline)
        report['checks']['original_32_full_examples'] = check_original_schedule()
        report['checks']['four_update_resume'] = check_four_update_equivalence(report['source_sha256'], deadline)
        import torch
        assert not torch.cuda.is_initialized()
        assert time.perf_counter() < deadline
        report.update(status='pass', cuda_initialized=False)
        code = 0
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        report['seconds'] = time.perf_counter() - started
        report['cuda_initialized'] = bool('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized())
        write(RUN / 'report.json', report)
        with REPORT.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        done.set()
    print(json.dumps({key: report[key] for key in ('status', 'seconds', 'cuda_initialized')}, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(fixture() if sys.argv[1:] == ['--fixture'] else main())
