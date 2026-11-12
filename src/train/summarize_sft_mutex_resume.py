"""Read completed 32-example pilot evidence; no model imports, training or edits."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/runs/sft_mutex_resume_v1'
REPORT = ROOT / 'reports/sft_mutex_resume_analysis.json'
BOUNDS = (512, 1024, 2048, 4096)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jsonl(path):
    content = Path(path).read_text(encoding='utf-8')
    assert not content or content.endswith('\n'), f'Incomplete JSONL: {path}'
    return [json.loads(line) for line in content.splitlines()]


def stats(values):
    assert values and all(math.isfinite(value) for value in values)
    return {'count': len(values), 'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'max': max(values), 'sum': sum(values)}


def main():
    if REPORT.exists():
        raise FileExistsError('Completed evidence analysis already exists; do not overwrite')
    report = read(ROOT / 'reports/sft_mutex_resume_result.json')
    preflight = read(ROOT / 'reports/sft_mutex_resume_preflight.json')
    ledger = read(RUN / 'progress.json')
    assert report['status'] == 'pass' and report['worker_launches'] == 2 and report['automatic_retries'] == 0
    assert ledger['attempted_updates'] == ledger['completed_updates'] == ledger['total_budget'] == 4
    assert not ledger['failed'] and not ledger['partial_update_possible']
    manifest = read(RUN / 'artifact_sha256.json')
    for name, expected in manifest.items():
        assert digest(RUN / name) == expected, f'Run artifact changed: {name}'
    for name, expected in preflight['protected_sha256'].items():
        assert digest(ROOT / name) == expected, f'Protected file changed: {name}'
    for name, expected in report['source_sha256'].items():
        assert digest(ROOT / name) == expected, f'Pilot source changed: {name}'

    updates, micros, caches, memory, process_samples, segment_rows, group_times = [], [], [], [], [], [], []
    for number, supervision in enumerate(report['segments'], 1):
        directory = RUN / f'segment_{number}'
        worker = read(directory / 'worker_result.json')
        assert worker['status'] == 'pass' and worker['base_unchanged'] and worker['cuda_rng_saved']
        assert worker['pid'] == supervision['bound_pid'] and supervision['identity_verified']
        assert supervision['exit_code'] == 0 and not supervision['killed']
        assert not supervision['errors'] and not supervision['cleanup_errors']
        assert not supervision['job_active_pids_after_cleanup']
        assert all(not row['alive'] and row['exit_code'] == 0 for row in supervision['role_exits'])
        history = read(directory / 'history.json')
        batch_rows = jsonl(directory / 'micro_intervals.jsonl')
        releases = jsonl(directory / 'cache_releases.jsonl')
        memory_rows = jsonl(directory / 'memory_events.jsonl')
        pdh = jsonl(directory / 'pid_memory.jsonl')
        stages = [row for row in jsonl(directory / 'worker_events.jsonl') if row['event'] == 'stage']
        assert len(history) == 2 and len(batch_rows) == len(releases) == 16
        assert all(row['pid'] == worker['pid'] for row in pdh)
        assert all(row['status'] in ('ok', 'not_registered') for row in pdh)
        assert all(row['dedicated_bytes'] is None and row['shared_bytes'] is None
                   for row in pdh if row['status'] == 'not_registered')
        base = read(directory / 'base_hashes.json')
        assert base['before'] == base['after']
        for item in history:
            number_update = item['update']
            start_event = next(row for row in stages if row['stage'] == f'update_{number_update}_setup')
            end_event = next(row for row in stages if row['monotonic'] > start_event['monotonic'] and
                (re.fullmatch(r'update_\d+_setup', row['stage']) or row['stage'].startswith('save_complete_checkpoint_')))
            group_seconds = end_event['monotonic'] - start_event['monotonic']
            micro_seconds = sum(row['seconds'] for row in batch_rows if row['update'] == number_update)
            assert group_seconds >= micro_seconds
            group_times.append({'update': number_update, 'group_seconds': group_seconds,
                'micro_seconds': micro_seconds, 'non_micro_seconds': group_seconds - micro_seconds})
        segment_group_seconds = sum(row['group_seconds'] for row in group_times if row['update'] in [h['update'] for h in history])
        valid = [row for row in pdh if row['status'] == 'ok']
        assert valid
        segment_rows.append({'segment': number, 'pid': worker['pid'], 'wall_seconds': supervision['wall_seconds'],
            'worker_seconds': worker['segment_seconds'], 'group_seconds': segment_group_seconds,
            'non_group_seconds': supervision['wall_seconds'] - segment_group_seconds,
            'sample_count': len(pdh), 'valid_samples': len(valid),
            'dedicated_sampled_peak_mib': max(row['dedicated_bytes'] for row in valid) / 2**20,
            'shared_sampled_peak_mib': max(row['shared_bytes'] for row in valid) / 2**20,
            'role_exits': supervision['role_exits']})
        updates.extend(history); micros.extend(batch_rows); caches.extend(releases)
        memory.extend(memory_rows); process_samples.extend(pdh)
    assert [row['update'] for row in updates] == [1, 2, 3, 4]
    expected_ids = [item for group in preflight['groups'] for item in group['source_indices']]
    assert [row['source_index'] for row in micros] == expected_ids and len(set(expected_ids)) == 32
    assert sum(row['sequence_length'] for row in micros) == 47654
    assert sum(row['valid_targets'] for row in micros) == 1350
    assert max(row['sequence_length'] for row in micros) == 3761
    assert max(row['valid_targets'] for row in micros) == 202
    assert all(row['gradients_finite_nonzero'] and row['changed_parameter_elements'] > 0
               and math.isfinite(row['group_loss']) and row['grad_norm_after_clip'] <= 1.00001 for row in updates)
    assert all(row['gradient_and_adam_identity_version_preserved'] for row in caches)
    allocated_changes = [row['after']['allocated_mib'] - row['before']['allocated_mib'] for row in caches]
    reserved_drops = [row['before']['reserved_mib'] - row['after']['reserved_mib'] for row in caches]
    assert all(delta == 0 for delta in allocated_changes)
    assert all(drop >= 0 for drop in reserved_drops)

    saved2 = read(RUN / 'checkpoints/update_000002/complete.json')
    metadata2 = read(RUN / 'checkpoints/update_000002/metadata.json')
    restored = read(RUN / 'segment_2/restore_verification.json')
    assert restored['metadata'] == metadata2 and restored['verification']['fingerprints'] == saved2['fingerprints']
    assert all(restored['verification'][key] for key in ('adapter_exact', 'optimizer_exact', 'rng_exact',
        'parameter_order_exact', 'cuda_rng_restored'))
    for step in (2, 4):
        saved = read(RUN / f'checkpoints/update_{step:06d}/complete.json')
        for name, expected in saved['files'].items():
            assert digest(RUN / f'checkpoints/update_{step:06d}' / name) == expected

    selection = read(ROOT / 'data/processed/spider_sft_resume_pilot_v1/selection.json')
    buckets = []
    for upper in BOUNDS:
        rows = [row for row in micros if next(bound for bound in BOUNDS if row['sequence_length'] <= bound) == upper]
        count = selection['candidate_bucket_counts'][str(upper)]
        assert len(rows) == 8
        buckets.append({'upper_bound': upper, 'training_examples': count,
            'pilot_micro_seconds': stats([row['seconds'] for row in rows]),
            'pilot_lengths': stats([row['sequence_length'] for row in rows]),
            'pilot_target_counts': stats([row['valid_targets'] for row in rows])})
    assert sum(row['training_examples'] for row in buckets) == 6364
    micro_estimate = sum(row['training_examples'] * row['pilot_micro_seconds']['mean'] for row in buckets)
    group_non_micro = stats([row['non_micro_seconds'] for row in group_times])
    non_micro_estimate = 796 * group_non_micro['mean']
    fixed_segment_estimate = 4 * statistics.mean(row['non_group_seconds'] for row in segment_rows)
    estimated_seconds = micro_estimate + non_micro_estimate + fixed_segment_estimate
    # These are explicit planning allowances, not a confidence interval or a guaranteed finish time.
    conservative_hours = [estimated_seconds * 1.25 / 3600, estimated_seconds * 2 / 3600]
    all_paths = [path for path in RUN.rglob('*') if path.is_file()]
    stage_log_bytes = sum(path.stat().st_size for path in all_paths if path.name.startswith('stage_ipc_') and path.suffix == '.jsonl')
    logs_bytes = sum(path.stat().st_size for path in all_paths if path.suffix in ('.jsonl', '.log'))
    checkpoint_bytes = sum(path.stat().st_size for path in (RUN / 'checkpoints').rglob('*') if path.is_file())
    predicted_stage_log_gib = stage_log_bytes / report['gpu_window_seconds'] * estimated_seconds / 2**30
    valid = [row for row in process_samples if row['status'] == 'ok']
    result = {
        'status': 'pass', 'analysis_scope': 'Read-only completed pilot; no model imports or training',
        'worker_launches': 2, 'automatic_retries': 0, 'completed_updates': 4,
        'gpu_window_seconds': report['gpu_window_seconds'], 'controller_seconds': report['total_controller_seconds'],
        'ledger': ledger, 'segments': segment_rows,
        'updates': [{key: row[key] for key in ('update', 'group_loss', 'total_target_tokens', 'lr_used',
            'next_update_lr', 'grad_norm_before_clip', 'grad_norm_after_clip', 'gradients_finite_nonzero',
            'changed_parameter_elements', 'optimizer_cuda_mib_before', 'optimizer_cuda_mib_after')} for row in updates],
        'loss_interpretation': 'Four different groups and changing model states; this is not a before/after quality comparison or proof of EX improvement.',
        'selected_examples': 32, 'input_tokens': 47654, 'valid_target_tokens': 1350,
        'maximum_input_tokens': 3761, 'maximum_answer_tokens': 202,
        'cache': {'calls': len(caches), 'reserved_drop_mib': stats(reserved_drops),
            'positive_drop_count': sum(value > 0 for value in reserved_drops),
            'allocated_change_mib': stats(allocated_changes), 'all_gradient_adam_identity_versions_preserved': True,
            'cache_release_seconds': stats([row['seconds'] for row in caches]),
            'reserved_after_mib': stats([row['after']['reserved_mib'] for row in caches]),
            'meaning': 'Unused allocator cache released; live tensors remain. GPU identity/version preservation is not a new bitwise CPU-equivalence experiment.'},
        'memory': {'torch_peak_allocated_mib': max(row['peak_allocated_mib'] for row in memory),
            'torch_peak_reserved_mib': max(row['peak_reserved_mib'] for row in memory),
            'pdh_sampled_dedicated_peak_mib': max(row['dedicated_bytes'] for row in valid) / 2**20,
            'pdh_sampled_shared_peak_mib': max(row['shared_bytes'] for row in valid) / 2**20,
            'sample_count': len(process_samples), 'valid_samples': len(valid),
            'not_registered_samples': len(process_samples) - len(valid),
            'allocator_retry_max': max(row['num_alloc_retries'] for row in memory if row['num_alloc_retries'] is not None),
            'allocator_oom_max': max(row['num_ooms'] for row in memory if row['num_ooms'] is not None),
            'limit': 'PDH maxima are sampled per-process WDDM accounting, not instantaneous whole-GPU peaks. Missing registration samples are null, not zero.'},
        'resume': {'checkpoint_update': 2, 'next_group_index': 2, 'next_update_lr': metadata2['next_update_lr'],
            'adapter_exact': True, 'optimizer_exact': True, 'rng_exact': True, 'parameter_order_exact': True,
            'cuda_rng_restored': True, 'restored_fingerprints_equal_saved': True,
            'scope': 'Exact saved-state restoration across new processes, not a second continuous GPU trajectory experiment.'},
        'base_unchanged_in_both_segments': True, 'group_timing': group_times, 'buckets': buckets,
        'formal_estimate': {'training_examples': 6364, 'updates': 796, 'planned_checkpoint_segments': 4,
            'weighted_micro_seconds': micro_estimate, 'per_update_non_micro_seconds': group_non_micro,
            'update_overhead_seconds': non_micro_estimate, 'four_segment_fixed_seconds': fixed_segment_estimate,
            'measured_rate_point_hours': estimated_seconds / 3600,
            'planning_allowance_multipliers': [1.25, 2.0], 'conservative_planning_hours': conservative_hours,
            'formula': 'Sum(natural bucket count * pilot mean micro time) + 796*mean measured non-micro group time + 4*mean segment non-group time; planning range is point*1.25 through point*2.',
            'limitations': ['Only eight selected examples per bucket; stress selection includes extremes and is not a random sample.',
                'Mixed-length pilot groups differ from formally bucketed groups; target length, thermal behavior, background apps and caching may change throughput.',
                'Current per-access durable logging and full gradient/adapter audits are included in measured rates; changing them invalidates this direct estimate.',
                'Planning range is an explicit contingency allowance, not a statistical confidence interval or proof of multi-hour stability.',
                'No 625-example evaluation time is included; no evaluation or final-set access was performed.']},
        'disk': {'pilot_stage_ipc_bytes': stage_log_bytes, 'pilot_all_jsonl_console_bytes': logs_bytes,
            'pilot_two_checkpoint_bytes': checkpoint_bytes,
            'formal_stage_ipc_gib_at_measured_rate_point': predicted_stage_log_gib,
            'formal_stage_ipc_gib_at_conservative_upper': predicted_stage_log_gib * 2,
            'four_checkpoints_gib_estimate': 2 * checkpoint_bytes / 2**30,
            'limitation': 'Stage-log projection uses observed bytes/second; other audits, checkpoints and source snapshots also need disk reserve.'},
        'integrity': {'run_artifacts_verified': len(manifest),
            'protected_hash_entries_verified': len(preflight['protected_sha256']),
            'protected_unique_paths_verified': len({str((ROOT / name).resolve()).lower() for name in preflight['protected_sha256']}),
            'source_files_verified': len(report['source_sha256'])},
        'validation_examples_read': False, 'final_examples_read': False,
        'formal_training_started': False}
    with REPORT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('status', 'completed_updates', 'gpu_window_seconds', 'memory', 'formal_estimate', 'disk')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
