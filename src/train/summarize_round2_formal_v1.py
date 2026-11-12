"""Read a completed successful weighted SFT run; never import training libraries.

Large JSONL telemetry files and checkpoint payloads are streamed. The only write
is an exclusive new analysis report after all consistency checks have passed.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/runs/sft_round2_formal_v1'
DESTINATION = ROOT / 'reports/sft_round2_formal_v1_analysis.json'
RESTORE_KEYS = ('adapter_exact', 'optimizer_exact', 'rng_exact',
                'parameter_order_exact', 'cuda_rng_restored')
CHECKPOINT_FILES = {'adapter_model.safetensors', 'adapter_config.json',
                    'optimizer.pt', 'rng.pt', 'metadata.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def lines(path):
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f'Unexpected blank JSONL record: {path}:{number}')
            yield json.loads(line)


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def finite(value, label, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f'Invalid {label}')
    require(value > 0 if positive else value >= 0, f'Negative/zero {label}')
    return float(value)


def main():
    require(not DESTINATION.exists(), 'Analysis already exists; no overwrite')
    report = read(RUN / 'report.json')
    require(report['status'] == 'pass', 'Only a completed successful formal run may be summarized')
    require((RUN / 'artifact_sha256.json').is_file(), 'Controller artifact finalization is incomplete')
    config = read(ROOT / 'configs/sft_round2_formal_v1.json')
    require(report['config'] == config, 'Formal run configuration changed')
    require(read(ROOT / config['result_report']) == report, 'Controller report copies disagree')
    preflight = read(RUN / 'preflight.json')
    require(preflight['status'] == 'pass' and preflight['config'] == config, 'Preflight mismatch')
    ledger = read(RUN / 'progress.json')
    require(ledger == report['ledger'], 'Ledger/report mismatch')
    require(ledger == {'attempted_updates': 796, 'completed_updates': 796,
                       'total_budget': 796, 'partial_update_possible': False, 'failed': False},
            'Formal ledger is not exactly 796 completed updates')
    require(report['automatic_retries'] == 0 and report['worker_launches'] == 4,
            'Unexpected retry or worker count')
    for field in ('validation_examples_read', 'final_examples_read', 'evaluation_performed'):
        require(report[field] is False, f'Unexpected scope: {field}')
    require(len(report['segments']) == len(report['worker_results']) == 4, 'Expected four segments')

    original = read(ROOT / config['original_schedule_path'])['groups']
    weighted = read(ROOT / config['schedule_path'])['groups']
    require(len(original) == len(weighted) == 796, 'Schedule must contain 796 updates')
    require(preflight['groups'] == [{**a, **b} for a, b in zip(original, weighted)],
            'Merged schedule changed')
    original_ids = [index for group in original for index in group['source_indices']]
    require(len(original_ids) == len(set(original_ids)) == 6364, 'Original schedule coverage differs')
    require(all(len(group['source_indices']) == 8 for group in original[:-1])
            and len(original[-1]['source_indices']) == 4, 'Expected 795 full groups and one tail of four')
    weight_rows = list(lines(ROOT / config['weights_path']))
    by_weight = {row['source_index']: row for row in weight_rows}
    require(len(weight_rows) == len(by_weight) == 6364 and set(by_weight) == set(original_ids),
            'Weight coverage differs')
    require(all(row['project_split'] == 'train' and type(row['loss_weight']) is int
                and row['loss_weight'] in (1, 2) for row in weight_rows), 'Invalid train-only weights')
    weight_counts = Counter(str(row['loss_weight']) for row in weight_rows)
    require(weight_counts == {'1': 3836, '2': 2528}, 'Frozen weight counts changed')

    checkpoint_records, checkpoint_metadata, checkpoint_fingerprints = [], {}, {}
    for update in (200, 400, 600, 796):
        folder = RUN / 'checkpoints' / f'update_{update:06d}'
        manifest = read(folder / 'complete.json')
        require(manifest['format'] == 'sft_complete_peft_v1', 'Unknown checkpoint format')
        require(set(manifest['files']) == CHECKPOINT_FILES, 'Checkpoint file coverage differs')
        for name, expected in manifest['files'].items():
            require(digest(folder / name) == expected, f'Checkpoint hash mismatch: {update}/{name}')
        metadata = read(folder / 'metadata.json')
        require(metadata['completed_updates'] == metadata['next_group_index'] == update,
                f'Checkpoint cursor mismatch: {update}')
        require(metadata['schedule_sha256'] == preflight['schedule_sha256']
                and metadata['run_config_sha256'] == preflight['run_config_sha256'],
                f'Checkpoint schedule/config mismatch: {update}')
        require(metadata['objective_binding'] == preflight['objective_binding']
                and metadata['protected_source_sha256'] == preflight['protected_sha256'],
                f'Checkpoint objective/source binding mismatch: {update}')
        checkpoint_metadata[update] = metadata
        checkpoint_fingerprints[update] = manifest['fingerprints']
        checkpoint_records.append({'update': update, 'path': folder.relative_to(ROOT).as_posix(),
                                   'file_hashes_verified': len(manifest['files']),
                                   'complete_manifest_sha256': digest(folder / 'complete.json'),
                                   'adapter_sha256': manifest['files']['adapter_model.safetensors']})

    history, segments, restore_checks, base_hashes = [], [], [], []
    memory_count = telemetry_count = valid_telemetry_count = cache_count = 0
    peak_allocated = peak_reserved = peak_dedicated = peak_shared = cache_delta = 0.0
    telemetry_status_counts = Counter()
    allocator_backends = Counter()
    process_ooms, process_alloc_retries, cache_per_update = [], [], Counter()
    for index, (start, end) in enumerate(config['segment_ranges'], 1):
        folder = RUN / f'segment_{index}'
        worker = read(folder / 'worker_result.json')
        supervised = report['segments'][index - 1]
        require(worker == report['worker_results'][index - 1], f'Segment {index} worker copies disagree')
        require(worker['status'] == supervised['status'] == 'pass'
                and supervised['outcome'] == 'completed', f'Segment {index} failed')
        require(worker['pid'] == supervised['bound_pid'] == supervised['worker_pid']
                and supervised['identity_verified'], f'Segment {index} PID mismatch')
        require(supervised['exit_code'] == 0 and not supervised['killed']
                and supervised['job_active_pids_after_cleanup'] == []
                and not supervised['errors'] and not supervised['cleanup_errors'],
                f'Segment {index} did not exit cleanly')
        require(len(supervised['role_exits']) == 1 and not supervised['role_exits'][0]['alive'],
                f'Segment {index} retained live process')
        expected_micros = sum(len(group['source_indices']) for group in original[start:end])
        require(worker['start'] == start and worker['end'] == end
                and worker['global_completed_updates'] == end
                and worker['completed_microbatches'] == worker['expected_microbatches'] == expected_micros,
                f'Segment {index} counts differ')
        require(worker['base_unchanged'] and worker['cuda_rng_saved'], f'Segment {index} base/RNG audit failed')
        initial = read(folder / 'initial_state.json')
        require(initial['fresh_original_base'] and not initial['restored_old_run']
                and initial['completed_updates'] == start
                and initial['restored_this_run_checkpoint'] == bool(start),
                f'Segment {index} initialization lineage differs')
        audit = read(folder / 'base_hashes.json')
        require(audit['before'] == audit['after'] == initial['base_hash'], f'Segment {index} base hash differs')
        base_hashes.append(audit['before'])
        require([row['update'] for row in worker['history']] == list(range(start + 1, end + 1)),
                f'Segment {index} update sequence differs')
        history.extend(worker['history'])
        if start:
            restored = read(folder / 'restore_verification.json')
            require(restored['metadata'] == checkpoint_metadata[start], f'Segment {index} restored metadata differs')
            verification = restored['verification']
            require(verification['fingerprints'] == checkpoint_fingerprints[start],
                    f'Segment {index} restored fingerprints differ')
            require(all(verification[key] is True for key in RESTORE_KEYS), f'Segment {index} restore check failed')
            restore_checks.append({'segment': index, 'from_update': start,
                                   **{key: verification[key] for key in RESTORE_KEYS}})

        segment_memory_count = 0
        segment_ooms = segment_retries = None
        for row in lines(folder / 'memory_events.jsonl'):
            segment_memory_count += 1
            allocator_backends[row['backend']] += 1
            peak_allocated = max(peak_allocated, finite(row['peak_allocated_mib'], 'allocated peak'))
            peak_reserved = max(peak_reserved, finite(row['peak_reserved_mib'], 'reserved peak'))
            if row['num_ooms'] is not None:
                segment_ooms = max(0 if segment_ooms is None else segment_ooms, finite(row['num_ooms'], 'OOM counter'))
            if row['num_alloc_retries'] is not None:
                segment_retries = max(0 if segment_retries is None else segment_retries, finite(row['num_alloc_retries'], 'allocation retry counter'))
        require(segment_memory_count > 0, f'Segment {index} has no allocator measurements')
        memory_count += segment_memory_count
        process_ooms.append(None if segment_ooms is None else int(segment_ooms))
        process_alloc_retries.append(None if segment_retries is None else int(segment_retries))
        segment_valid = 0
        for row in lines(folder / 'pid_memory.jsonl'):
            telemetry_count += 1
            telemetry_status_counts[row['status']] += 1
            require(row['pid'] == worker['pid'], f'Segment {index} telemetry PID differs')
            if row['status'] == 'ok':
                valid_telemetry_count += 1; segment_valid += 1
                peak_dedicated = max(peak_dedicated, finite(row['dedicated_bytes'], 'dedicated memory') / 2**20)
                peak_shared = max(peak_shared, finite(row['shared_bytes'], 'shared memory') / 2**20)
        require(segment_valid > 0, f'Segment {index} has no valid process memory measurements')
        expected_release_order = ((update + 1, micro + 1)
                                  for update in range(start, end)
                                  for micro in range(len(original[update]['source_indices'])))
        segment_releases = 0
        for row in lines(folder / 'cache_releases.jsonl'):
            require((row['update'], row['microbatch']) == next(expected_release_order, None),
                    f'Segment {index} cache release order differs')
            require(row['gradient_and_adam_identity_version_preserved'] is True,
                    f'Segment {index} cache release changed live state')
            cache_delta = max(cache_delta, abs(row['after']['allocated_mib'] - row['before']['allocated_mib']))
            segment_releases += 1; cache_count += 1; cache_per_update[row['update']] += 1
        require(segment_releases == expected_micros and next(expected_release_order, None) is None,
                f'Segment {index} cache release count differs')
        segments.append({'segment': index, 'start': start, 'end': end, 'worker_pid': worker['pid'],
                         'wall_seconds': supervised['wall_seconds'], 'exit_code': supervised['exit_code'],
                         'job_empty': True, 'completed_microbatches': expected_micros,
                         'valid_memory_samples': segment_valid, 'base_unchanged': True})

    require(all(value == base_hashes[0] for value in base_hashes), 'Segments used different frozen base hashes')
    require(len(history) == 796 and [row['update'] for row in history] == list(range(1, 797)),
            'Full update sequence differs')
    require([i for row in history for i in row['source_indices']] == original_ids,
            'Training sample order differs from the original epoch')
    require(cache_count == 6364 and len(restore_checks) == 3, 'Missing cache releases or restore checks')
    training = preflight['formal_config']['training']
    peak_lr = training['optimizer']['lr']
    warmup = training['scheduler']['warmup_updates']
    total_steps = training['scheduler']['total_updates']

    def expected_lr(update):
        if update > total_steps:
            return 0.0
        return peak_lr * update / warmup if update <= warmup else peak_lr * (total_steps - update + 1) / (total_steps - warmup + 1)

    for row, raw_group, weights in zip(history, original, weighted):
        ids = raw_group['source_indices']
        counts = raw_group['target_counts']
        expected_weights = [by_weight[i]['loss_weight'] for i in ids]
        denominator = sum(w * n for w, n in zip(expected_weights, counts))
        coefficients = [w * n / denominator for w, n in zip(expected_weights, counts)]
        require(row['source_indices'] == weights['source_indices'] == ids, 'Weighted source order differs')
        require(row['weights'] == row['loss_weights'] == weights['loss_weights'] == expected_weights,
                'Actual sample weights differ')
        require(row['micro_target_counts'] == weights['target_counts'] == counts
                and row['total_target_tokens'] == raw_group['group_target_tokens'] == sum(counts),
                'Actual answer-token counts differ')
        require(row['weighted_target_denominator'] == weights['weighted_target_denominator'] == denominator,
                'Weighted group denominator differs')
        require(row['micro_mean_loss_coefficients'] == weights['micro_mean_loss_coefficients'] == coefficients,
                'Weighted microbatch coefficients differ')
        require(row['microbatch_count'] == row['cache_releases'] == cache_per_update[row['update']] == len(ids),
                'Microbatch/cache count differs')
        require(row['gradients_finite_nonzero'] and row['changed_parameter_elements'] > 0,
                'Gradient or parameter-change evidence failed')
        finite(row['group_loss'], 'training loss')
        finite(row['grad_norm_before_clip'], 'gradient norm before clipping', positive=True)
        finite(row['grad_norm_after_clip'], 'gradient norm after clipping', positive=True)
        require(row['grad_norm_after_clip'] <= training['max_gradient_norm'] + 1e-5,
                'Gradient clipping threshold exceeded')
        require(math.isclose(row['lr_used'], expected_lr(row['update']), rel_tol=0, abs_tol=1e-15)
                and math.isclose(row['next_update_lr'], expected_lr(row['update'] + 1), rel_tol=0, abs_tol=1e-15),
                'Learning-rate position differs')
    require(history[-1]['next_update_lr'] == 0.0, 'Learning rate did not finish the frozen schedule')
    losses = [row['group_loss'] for row in history]
    before_norms = [row['grad_norm_before_clip'] for row in history]
    after_norms = [row['grad_norm_after_clip'] for row in history]
    summary = {
        'status': 'pass', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'run_dir': config['run_dir'], 'ledger': ledger, 'automatic_retries': 0,
        'full_training_examples': 6364, 'optimizer_updates': 796, 'last_group_examples': 4,
        'same_original_epoch_order': True, 'all_weighted_denominators_verified': True,
        'weight_counts': dict(sorted(weight_counts.items())),
        'gpu_window_seconds': report['gpu_window_seconds'],
        'total_controller_seconds': report['total_controller_seconds'], 'segments': segments,
        'peak_allocated_mib': peak_allocated, 'peak_reserved_mib': peak_reserved,
        'peak_process_dedicated_mib': peak_dedicated, 'peak_process_shared_mib': peak_shared,
        'allocator_measurement_count': memory_count, 'memory_sample_count': telemetry_count,
        'valid_memory_sample_count': valid_telemetry_count, 'memory_sample_status_counts': dict(telemetry_status_counts),
        'allocator_backends': dict(allocator_backends),
        'num_ooms_per_segment': process_ooms,
        'num_ooms_total': None if None in process_ooms else sum(process_ooms),
        'num_alloc_retries_per_segment': process_alloc_retries,
        'num_alloc_retries_total': None if None in process_alloc_retries else sum(process_alloc_retries),
        'cache_release_count': cache_count, 'cache_gradient_adam_preserved': True,
        'cache_maximum_allocated_delta_mib': cache_delta,
        'base_unchanged_all_segments': True, 'base_hash': base_hashes[0],
        'restore_checks': restore_checks, 'checkpoints': checkpoint_records,
        'loss_statistics': {'first': losses[0], 'last': losses[-1], 'minimum': min(losses), 'maximum': max(losses),
                            'last_50_update_arithmetic_mean': math.fsum(losses[-50:]) / 50,
                            'interpretation': 'Descriptive training trace over different weighted groups; not EX, quality, or comparable to round1 unweighted loss.'},
        'gradient_norm_before_clip_range': [min(before_norms), max(before_norms)],
        'gradient_norm_after_clip_range': [min(after_norms), max(after_norms)],
        'last_update_lr_used': history[-1]['lr_used'], 'next_update_lr_after_completion': history[-1]['next_update_lr'],
        'protected_files_verified_after': report['protected_files_verified_after'],
        'validation_examples_read': False, 'final_examples_read': False, 'evaluation_performed': False,
        'evidence': {'report_sha256': digest(RUN / 'report.json'), 'ledger_sha256': digest(RUN / 'progress.json'),
                     'summarizer_sha256': digest(Path(__file__)),
                     'artifact_manifest_sha256': digest(RUN / 'artifact_sha256.json')},
        'limitation': 'Successful full training and checkpoint integrity; quality gains remain unmeasured until a separately authorized evaluation.'}
    with DESTINATION.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
