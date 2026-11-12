"""One frozen, stdlib-only readiness acceptance for the 6364-example weighted round.

Reuses verified CPU gradient mathematics and GPU pilot evidence. Does not import
ML libraries, initialize CUDA, optimize a model, or read evaluation examples.
"""
from __future__ import annotations
import ast
import copy
from datetime import datetime, timezone
import importlib.abc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'outputs/checks/sft_round2_formal_cpu_v1'
REPORT = ROOT / 'reports/sft_round2_formal_cpu_v1_checks.json'
CONFIG = ROOT / 'configs/sft_round2_formal_v1.json'
FREEZE = ROOT / 'reports/sft_round2_formal_v1_freeze.json'
WORKER = ROOT / 'src/train/sft_round2_formal_v1_worker.py'
SOURCES = ['src/train/test_round2_formal_cpu_v1.py', 'configs/sft_round2_formal_v1.json',
    'src/train/sft_round2_formal_v1.py', 'src/train/sft_round2_formal_v1_worker.py',
    'src/train/sft_round2_weight_core_v1.py', 'src/train/sft_round2_binding_v1.py',
    'src/train/sft_round2_pilot_v2_worker.py', 'src/train/test_round2_pilot_cpu_v2.py',
    'src/train/test_round2_weight_cpu_v1.py', 'src/train/weighted_token_accumulation_v1.py',
    'src/train/sft_checkpoint.py', 'src/train/sft_checkpoint_cpu_v1.py',
    'src/train/sft_update_budget.py', 'src/train/sft_train.py',
    'src/train/target_loss.py', 'src/train/token_accumulation.py']
FORBIDDEN = {'torch', 'transformers', 'peft', 'bitsandbytes', 'numpy'}
BUDGET_SECONDS = 120


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, mode='w'):
    with Path(path).open(mode, encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify(mapping):
    from src.train.sft_round2_binding_v1 import file_digest
    for name, expected in mapping.items():
        assert file_digest(ROOT / name) == expected, f'Frozen evidence changed: {name}'


def check_config():
    config = read(CONFIG)
    proposal = read(ROOT / config['formal_config'])
    original = read(ROOT / 'configs/sft_round1.json')
    round1_run = read(ROOT / 'configs/sft_round1_resource_v3.json')
    expected = {'selected_examples': 6364, 'batch_size': 1, 'accumulation_steps': 8,
        'maximum_updates': 796, 'global_update_budget': 796,
        'segment_ranges': [[0, 200], [200, 400], [400, 600], [600, 796]],
        'save_after_updates': [200, 400, 600, 796],
        'minimum_free_mib': 6656, 'start_headroom_mib': 128,
        'start_resource_samples': 6, 'start_resource_interval_seconds': 2,
        'start_resource_budget_seconds': 20, 'minimum_disk_gib': 8,
        'allocated_limit_mib': 6144, 'reserved_limit_mib': 7168,
        'total_gpu_window_seconds': 15660, 'mutex_wait_seconds': 1}
    for key, value in expected.items():
        assert config[key] == value, f'Formal setting changed: {key}'
    assert config['limits'] == round1_run['limits'] == proposal['future_resource_budget']['limits']
    assert config['seed'] == proposal['seed'] == original['seed'] == 42
    assert config['gpu_training_enabled'] is False and config['evaluation_enabled'] is False
    assert config['train_path'].replace('\\', '/') == proposal['train_path'].replace('\\', '/')
    for key, proposal_key in [('weights_path', 'weights'), ('schedule_path', 'weighted_groups'),
                              ('original_schedule_path', 'frozen_schedule')]:
        assert config[key].replace('\\', '/') == proposal[proposal_key].replace('\\', '/')
    assert proposal['model'] == original['model']
    earlier, weighted = dict(original['training']), dict(proposal['training'])
    earlier.pop('normalization'); weighted.pop('normalization')
    assert earlier == weighted, 'Only normalization/weights may differ from round one'
    assert proposal['single_active_change'] == {'focus_a_loss_weight': 2, 'other_loss_weight': 1,
        'round1_focus_a_loss_weight': 1, 'focus_b_intervention': False}
    assert proposal['training']['epochs'] == 1
    assert proposal['training']['expected_final_group_examples'] == 4
    assert not (ROOT / config['run_dir']).exists(), 'CPU check must precede new run claim'
    return {'status': 'pass', 'config': config, 'same_round1_model_and_training_except_weights': True}, config, proposal


def check_reused_evidence():
    from src.train.sft_round2_binding_v1 import file_digest
    paths = ('reports/sft_round2_weight_cpu_v1_checks.json',
             'reports/sft_round2_pilot_cpu_v2_checks.json',
             'reports/sft_round2_pilot_v2_result.json',
             'reports/sft_round1_ready_cpu_checks.json')
    evidence = {}
    for name in paths:
        report = read(ROOT / name)
        assert report['status'] == 'pass', name
        verify(report['source_sha256'])
        evidence[name] = {'sha256': file_digest(ROOT / name),
                          'source_hashes_verified': len(report['source_sha256'])}
    math_report, pilot_cpu, pilot_gpu, full_cpu = [read(ROOT / name) for name in paths]
    assert math_report['cuda_initialized'] is False and len(math_report['cases']) == 7
    assert all(case['status'] == 'pass' for case in math_report['cases'])
    assert 'tail_four_actual_counts_and_weights' in {case['name'] for case in math_report['cases']}
    assert pilot_cpu['checks']['four_update_resume']['exact_comparison'] is True
    assert pilot_cpu['frozen_sources_verified_before_and_after'] is True
    assert pilot_cpu['checks']['four_update_resume']['new_cache_core_matches_prevalidated_weighted_helper_exactly'] is True
    assert pilot_gpu['ledger']['completed_updates'] == 4 and not pilot_gpu['ledger']['failed']
    assert pilot_gpu['worker_launches'] == 2 and pilot_gpu['automatic_retries'] == 0
    assert all(worker['base_unchanged'] for worker in pilot_gpu['worker_results'])
    verify(full_cpu['data_sha256'])
    return {'status': 'pass', 'reports': evidence, 'math_rerun': False,
            'gpu_pilot_rerun': False, 'cpu_or_gpu_optimizer_updates': 0}, full_cpu


def check_full_data(config, proposal, earlier_cpu):
    from src.train.sft_round2_binding_v1 import file_digest, verify_group_manifest
    train_path, weight_path = ROOT / config['train_path'], ROOT / config['weights_path']
    original_path, weighted_path = ROOT / config['original_schedule_path'], ROOT / config['schedule_path']
    token_manifest = read(ROOT / 'data/processed/spider_sft_tokens_v1/manifest.json')
    weight_manifest = read(ROOT / proposal['manifest'])
    verify(token_manifest['source_sha256'])
    verify(weight_manifest['source_sha256']); verify(weight_manifest['output_sha256'])
    assert token_manifest['stored_arrays_unpadded'] and not token_manifest['labels_shifted_in_preprocessing']
    assert token_manifest['truncated_examples'] == token_manifest['dropped_examples'] == 0
    assert file_digest(train_path) == token_manifest['output_sha256']['train_tokens.jsonl']
    original, weighted = read(original_path), read(weighted_path)
    assert original['train_sha256'] == file_digest(train_path)
    assert weighted['source_schedule_sha256'] == file_digest(original_path)
    rows = [json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines()]
    weights = [json.loads(line) for line in weight_path.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len({row['source_index'] for row in rows}) == 6364
    sequence_total = target_total = maximum_length = maximum_answer = 0
    buckets = {str(bound): 0 for bound in (512, 1024, 2048, 4096)}
    for row in rows:
        assert row['project_split'] == 'train'
        ids, mask, labels = row['input_ids'], row['attention_mask'], row['labels']
        length, prompt, eos = row['sequence_length'], row['prompt_length'], row['eos_position']
        assert all(type(values) is list and len(values) == length for values in (ids, mask, labels))
        assert all(type(value) is int and value >= 0 for value in ids)
        assert all(type(value) is int and value == 1 for value in mask), 'Stored full rows must have no padding'
        assert all(type(value) is int for value in labels)
        assert 0 < prompt <= eos < length <= 4096
        assert ids[eos] == token_manifest['eos_token_id'] == 151645
        assert labels[:prompt] == [-100] * prompt
        assert labels[prompt:eos + 1] == ids[prompt:eos + 1]
        assert labels[eos + 1:] == [-100] * (length - eos - 1)
        count = sum(value != -100 for value in labels[1:])
        assert count == row['supervised_token_count'] == eos - prompt + 1
        assert row['trailing_template_token_count'] == length - eos - 1
        sequence_total += length; target_total += count
        maximum_length = max(maximum_length, length); maximum_answer = max(maximum_answer, count)
        buckets[str(next(bound for bound in (512, 1024, 2048, 4096) if length <= bound))] += 1
    verified = verify_group_manifest(rows, weights, original['groups'], weighted['groups'])
    groups = [{**old, **new} for old, new in zip(original['groups'], weighted['groups'])]
    assert len(groups) == 796 and all(len(group['source_indices']) == 8 for group in groups[:-1])
    assert len(groups[-1]['source_indices']) == 4
    old_full = earlier_cpu['checks']['full_training_data']
    assert original['groups'] == old_full['groups']
    assert sequence_total == old_full['input_tokens'] == 3932044
    assert target_total == old_full['target_tokens'] == 206073
    assert maximum_length == 3761 and maximum_answer == 202 and verified['databases'] == 126
    assert verified['weighted_target_tokens_total'] == 345003
    weight_counts = {str(weight): sum(row['loss_weight'] == weight for row in weights) for weight in (1, 2)}
    assert weight_counts == {'1': 3836, '2': 2528}
    assert verified['group_composition_counts'] == {'mixed': 777, 'all_focus': 1, 'all_other': 18}
    by_id = {row['source_index']: row for row in rows}
    for index, group in enumerate(groups):
        assert group['update'] == index + 1
        assert group['sequence_lengths'] == [by_id[i]['sequence_length'] for i in group['source_indices']]
        assert group['group_target_tokens'] == sum(group['target_counts'])
    worker_tree = ast.parse(WORKER.read_text(encoding='utf-8'))
    definitions = [node for node in worker_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'verify_full_group_weights']
    assert len(definitions) == 1
    namespace = {'math': math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(WORKER) + '::pure_weight_guard', 'exec'), namespace)
    guard = namespace['verify_full_group_weights']
    assert guard(rows, weights, groups)['status'] == 'pass'
    bad_groups = copy.deepcopy(groups)
    bad_groups[-1]['weighted_target_denominator'] += 1
    try:
        guard(rows, weights, bad_groups)
    except ValueError:
        pass
    else:
        raise AssertionError('Actual formal worker guard accepted an invalid tail denominator')
    data_files = (train_path, weight_path, original_path, weighted_path,
                  ROOT / proposal['manifest'], ROOT / 'data/processed/spider_sft_tokens_v1/manifest.json')
    hashes = {path.relative_to(ROOT).as_posix(): file_digest(path) for path in data_files}
    return {'status': 'pass', **verified, 'groups': groups,
        'source_indices': [i for group in groups for i in group['source_indices']],
        'input_tokens': sequence_total, 'target_tokens': target_total,
        'max_sequence_length': maximum_length, 'max_answer_tokens': maximum_answer,
        'weight_counts': weight_counts, 'bucket_counts': buckets, 'full_arrays_inspected': True,
        'assistant_only_eos_and_unpadded_rows_verified': True, 'labels_shifted_in_preprocessing': False,
        'truncated_or_dropped_examples': 0, 'original_group_order_preserved': True,
        'actual_worker_guard_positive_and_bad_tail_denominator': True,
        'actual_training_rows_optimized_on_cpu': 0}, hashes


def check_worker_wiring():
    for name in SOURCES:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
    source = WORKER.read_text(encoding='utf-8')
    tree = ast.parse(source)
    previous = (ROOT / 'src/train/sft_round2_pilot_v2_worker.py').read_text(encoding='utf-8')
    def core_loop(text):
        loops = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == 'group_index']
        assert len(loops) == 1
        return ast.dump(loops[0], include_attributes=False)
    normalized = previous.replace('len(cache_records) != 8', 'len(cache_records) != len(rows)')
    assert core_loop(source) == core_loop(normalized), 'Training loop differs beyond tail-group cache count'
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'backward_weighted_cache_group']
    assert len(calls) == 1 and {'rows', 'group', 'device', 'observe', 'release'} <= {kw.arg for kw in calls[0].keywords}
    optimizers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == 'JournaledAdamW']
    assert len(optimizers) == 1
    budget = next(kw.value for kw in optimizers[0].keywords if kw.arg == 'total_budget')
    assert ast.literal_eval(budget) == 796
    assert 'SFT_ROUND2_FORMAL_CONFIRMED' in source
    assert 'configs/sft_round2_formal_v1.json' in source
    assert 'SFT_ROUND2_PILOT_CONFIRMED' not in source
    assert 'verify_binding' in source and 'build_binding' in source and 'objective_binding' in source
    return {'status': 'pass', 'source_ast_parsed': True,
        'training_loop_matches_validated_pilot_except_tail_cache_count': True,
        'weighted_core_and_restore_binding_wired': True, 'global_optimizer_budget': 796,
        'worker_imported': False, 'scope': 'Structural check; full formal GPU execution remains untested'}


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('CPU acceptance already claimed; no retry or overwrite')
    RUN.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
        'budget_seconds': BUDGET_SECONDS, 'attempts': 1, 'automatic_retries': 0,
        'checks': {}, 'cuda_initialized': False, 'model_loaded': False,
        'cpu_optimizer_updates': 0, 'gpu_training_updates': 0, 'child_processes_started': 0,
        'validation_examples_read': False, 'final_examples_read': False,
        'attempted_forbidden_imports': []}
    write(REPORT, record, 'x'); write(RUN / 'claim.json', {'pid': os.getpid(), **record}, 'x')
    done = threading.Event()
    def expire():
        if not done.wait(max(0.0, started + BUDGET_SECONDS - time.perf_counter())):
            record.update(status='failed', error='CPU formal readiness exceeded 120-second deadline')
            try:
                write(RUN / 'timeout.json', record, 'x'); write(REPORT, record)
            finally:
                os._exit(124)
    threading.Thread(target=expire, daemon=True).start()
    try:
        assert not FORBIDDEN.intersection(sys.modules)
        class BlockImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in FORBIDDEN:
                    record['attempted_forbidden_imports'].append(fullname)
                    raise ImportError(f'Stdlib CPU readiness forbids {fullname}')
        sys.meta_path.insert(0, BlockImports())
        from src.train.sft_round2_binding_v1 import file_digest
        frozen = read(FREEZE)
        record['freeze_manifest_sha256'] = file_digest(FREEZE)
        assert set(SOURCES).issubset(frozen['source_sha256'])
        def verify_frozen():
            assert file_digest(FREEZE) == record['freeze_manifest_sha256'], 'Freeze manifest changed'
            verify(frozen['source_sha256']); verify(frozen['previous_attempt_sha256'])
        verify_frozen()
        record['source_sha256'] = dict(frozen['source_sha256'])
        for name in record['source_sha256']:
            destination = RUN / 'source_snapshot' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
            assert file_digest(destination) == record['source_sha256'][name]
        record['checks']['config'], config, proposal = check_config()
        record['checks']['reused_evidence'], earlier_cpu = check_reused_evidence()
        record['checks']['full_training_data'], record['data_sha256'] = check_full_data(config, proposal, earlier_cpu)
        record['checks']['worker_wiring'] = check_worker_wiring()
        verify_frozen()
        assert not FORBIDDEN.intersection(sys.modules) and not record['attempted_forbidden_imports']
        record.update(status='pass', frozen_sources_verified_before_and_after=True,
                      math_reused_not_rerun=True, formal_training_started=False)
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
