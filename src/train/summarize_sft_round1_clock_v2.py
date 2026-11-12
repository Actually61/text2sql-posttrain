"""Read-only, stdlib-only analysis AFTER the formal controller and workers exit.

Large JSONL logs are streamed. Missing failure-side artifacts remain explicitly
incomplete; no training, checkpoint loading, retries or ledger mutation occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/runs/sft_round1_clock_v2'
REPORT = ROOT / 'reports/sft_round1_clock_v2_analysis.json'
PRIOR_RUN = ROOT / 'outputs/runs/sft_round1_ready_v1'
PLANNED = ((0, 200), (200, 400), (400, 600), (600, 796))


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Stat:
    """Constant-memory numeric summary; unknown values never become zeros."""
    def __init__(self):
        self.count, self.total, self.low, self.high = 0, 0.0, None, None

    def add(self, value):
        if number(value):
            self.count += 1
            self.total += value
            self.low = value if self.low is None else min(self.low, value)
            self.high = value if self.high is None else max(self.high, value)

    def result(self):
        return {'count': self.count, 'min': self.low, 'max': self.high,
                'mean': self.total / self.count if self.count else None,
                'sum': self.total if self.count else None}


class Evidence:
    def __init__(self):
        self.missing, self.errors, self.hash_cache = [], [], {}

    def read(self, path, required=False):
        path = Path(path)
        if not path.exists():
            self.missing.append(str(path))
            if required:
                self.errors.append({'path': str(path), 'error': 'Required artifact missing'})
            return None
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'))
        except Exception as error:
            self.errors.append({'path': str(path), 'error': repr(error)})
            return None

    def rows(self, path):
        """Yield one JSONL record at a time, report an incomplete final record."""
        path = Path(path)
        if not path.exists():
            self.missing.append(str(path))
            return
        try:
            with path.open(encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.endswith('\n'):
                        self.errors.append({'path': str(path), 'line': line_number,
                                            'error': 'Incomplete JSONL final line; excluded'})
                        break
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError('Expected an object record')
                        yield row
                    except Exception as error:
                        self.errors.append({'path': str(path), 'line': line_number, 'error': repr(error)})
                        break
        except Exception as error:
            self.errors.append({'path': str(path), 'error': repr(error)})

    def digest(self, path):
        path = Path(path).resolve()
        key = str(path).casefold()
        if key not in self.hash_cache:
            with path.open('rb') as stream:
                self.hash_cache[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return self.hash_cache[key]

    def hashes(self, mapping, base):
        failures, verified = [], 0
        if not isinstance(mapping, dict):
            return {'status': 'not_completed', 'verified': 0, 'failures': [], 'reason': 'No hash manifest'}
        for name, expected in mapping.items():
            path = Path(name) if Path(name).is_absolute() else base / name
            try:
                actual = self.digest(path)
                if actual != expected:
                    failures.append({'path': str(path), 'expected': expected, 'actual': actual})
                else:
                    verified += 1
            except Exception as error:
                failures.append({'path': str(path), 'error': repr(error)})
        return {'status': 'pass' if not failures else 'failed', 'verified': verified,
                'expected_entries': len(mapping), 'failures': failures}


def checkpoint(evidence, update, preflight):
    path = RUN / 'checkpoints' / f'update_{update:06d}'
    complete = evidence.read(path / 'complete.json')
    metadata = evidence.read(path / 'metadata.json')
    if not complete or not metadata:
        return {'update': update, 'status': 'not_completed', 'path': str(path)}
    hashes = evidence.hashes(complete.get('files'), path)
    checks = {
        'format': complete.get('format') == 'sft_complete_peft_v1',
        'files': set(complete.get('files', {})) == {
            'adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'rng.pt', 'metadata.json'},
        'update_and_cursor': metadata.get('completed_updates') == metadata.get('next_group_index') == update,
        'schedule_hash': metadata.get('schedule_sha256') == preflight.get('schedule_sha256'),
        'run_config_hash': metadata.get('run_config_sha256') == preflight.get('run_config_sha256'),
        'protected_source_hashes': metadata.get('protected_source_sha256') == preflight.get('protected_sha256'),
    }
    return {'update': update, 'path': str(path),
            'status': 'pass' if hashes['status'] == 'pass' and all(checks.values()) else 'failed',
            'file_hashes': hashes, 'checks': checks, 'metadata': metadata,
            'fingerprints': complete.get('fingerprints')}


def segment(evidence, index, limits, groups, controller):
    start, end = PLANNED[index - 1]
    directory = RUN / f'segment_{index}'
    if not directory.exists():
        return {'segment': index, 'status': 'not_started', 'range': [start, end]}, [], [], []
    supervision = evidence.read(RUN / f'segment_{index}_returned.json')
    if supervision is None:
        supervision = evidence.read(directory / 'result.json')
    if supervision is None and len(controller.get('segments', [])) >= index:
        supervision = controller['segments'][index - 1]
    supervision = supervision or {}
    worker = evidence.read(directory / 'worker_result.json') or {}
    history = evidence.read(directory / 'history.json')
    if history is None:
        history = worker.get('history', [])
    if not isinstance(history, list):
        evidence.errors.append({'path': str(directory / 'history.json'), 'error': 'History must be a list'})
        history = []
    memory_keys = ('peak_allocated_mib', 'peak_reserved_mib', 'num_alloc_retries', 'num_ooms')
    memory = {key: Stat() for key in memory_keys}
    for row in evidence.rows(directory / 'memory_events.jsonl'):
        for key in memory_keys:
            memory[key].add(row.get(key))
    status_counts = Counter()
    dedicated, shared = Stat(), Stat()
    sampled_pids = set()
    missing_registration_nonnull = 0
    for row in evidence.rows(directory / 'pid_memory.jsonl'):
        status_counts[str(row.get('status'))] += 1
        sampled_pids.add(row.get('pid'))
        if row.get('status') == 'ok':
            dedicated.add(row.get('dedicated_bytes'))
            shared.add(row.get('shared_bytes'))
        elif row.get('status') == 'not_registered':
            missing_registration_nonnull += int(row.get('dedicated_bytes') is not None or row.get('shared_bytes') is not None)
    reserved_drop, allocated_delta, cache_seconds = Stat(), Stat(), Stat()
    calls = preserved = 0
    for row in evidence.rows(directory / 'cache_releases.jsonl'):
        calls += 1
        preserved += int(row.get('gradient_and_adam_identity_version_preserved') is True)
        before, after = row.get('before', {}), row.get('after', {})
        if number(before.get('reserved_mib')) and number(after.get('reserved_mib')):
            reserved_drop.add(before['reserved_mib'] - after['reserved_mib'])
        if number(before.get('allocated_mib')) and number(after.get('allocated_mib')):
            allocated_delta.add(after['allocated_mib'] - before['allocated_mib'])
        cache_seconds.add(row.get('seconds'))
    micro_ids, micro_updates = [], []
    micro_time, input_tokens, targets = Stat(), Stat(), Stat()
    shape_mismatches = []
    expected_rows = {}
    for group in groups[start:end]:
        for source_id, length, count in zip(group['source_indices'], group['sequence_lengths'], group['target_counts']):
            expected_rows[(group['update'], source_id)] = (length, count)
    for row in evidence.rows(directory / 'micro_intervals.jsonl'):
        micro_ids.append(row.get('source_index'))
        micro_updates.append(row.get('update'))
        micro_time.add(row.get('seconds'))
        input_tokens.add(row.get('sequence_length'))
        targets.add(row.get('valid_targets'))
        observed = (row.get('sequence_length'), row.get('valid_targets'))
        if expected_rows.get((row.get('update'), row.get('source_index'))) != observed:
            shape_mismatches.append({'update': row.get('update'), 'source_index': row.get('source_index'), 'observed': observed})
    base = evidence.read(directory / 'base_hashes.json')
    initial = evidence.read(directory / 'initial_state.json')
    stop = evidence.read(directory / 'stop_request.json') if (directory / 'stop_request.json').exists() else None
    observed_pid = worker.get('pid', supervision.get('bound_pid'))
    row = {'segment': index, 'range': [start, end], 'status': supervision.get('status', 'not_completed'),
           'worker_status': worker.get('status', 'not_completed'), 'outcome': supervision.get('outcome'),
           'wall_seconds': supervision.get('wall_seconds'), 'worker_seconds': worker.get('segment_seconds'),
           'pid': observed_pid, 'launcher_pid': supervision.get('launcher_pid'),
           'identity_verified': supervision.get('identity_verified'),
           'worker_matches_binding': worker.get('pid') == supervision.get('bound_pid') if worker and supervision else None,
           'sampled_pids': sorted(sampled_pids, key=str),
           'samples_match_binding': sampled_pids == {observed_pid} if sampled_pids and observed_pid is not None else None,
           'exit_code': supervision.get('exit_code'), 'killed': supervision.get('killed'),
           'stop_reason': supervision.get('stop_reason'), 'stop_request': stop,
           'errors': supervision.get('errors'), 'cleanup_errors': supervision.get('cleanup_errors'),
           'worker_error': worker.get('error'), 'failed_stage': worker.get('failed_stage'),
           'role_exits': supervision.get('role_exits'), 'cleanup_exits': supervision.get('cleanup_exits'),
           'job_active_pids_after_cleanup': supervision.get('job_active_pids_after_cleanup'),
           'initial_state': initial, 'base_unchanged': (base.get('before') == base.get('after')) if base else None,
           'cuda_rng_saved': worker.get('cuda_rng_saved'), 'history_updates': len(history),
           'completed_microbatches': len(micro_ids), 'micro_seconds': micro_time.result(),
           'input_tokens': input_tokens.result(), 'valid_target_tokens': targets.result(),
           'full_length_mismatches': shape_mismatches,
           'memory': {key: stat.result() for key, stat in memory.items()},
           'pdh': {'sample_count': sum(status_counts.values()), 'status_counts': dict(status_counts),
                   'valid_samples': status_counts['ok'],
                   'dedicated_sampled_peak_mib': dedicated.high / 2**20 if dedicated.count else None,
                   'shared_sampled_peak_mib': shared.high / 2**20 if shared.count else None,
                   'not_registered_nonnull_values': missing_registration_nonnull},
           'cache': {'calls': calls, 'identity_preserved_calls': preserved,
                     'all_identity_preserved': preserved == calls if calls else None,
                     'reserved_drop_mib': reserved_drop.result(), 'allocated_change_mib': allocated_delta.result(),
                     'seconds': cache_seconds.result()},
           'budget': limits}
    return row, history, micro_ids, micro_updates


def formal_accounting(evidence, controller, ledger, preflight):
    """Keep the failed prior 142 separate from this independently trained adapter."""
    prior_ledger = evidence.read(PRIOR_RUN / 'progress.json', required=True) or {}
    prior_result = evidence.read(ROOT / 'reports/sft_round1_ready_result.json', required=True) or {}
    config = preflight.get('config', {})
    def count(value):
        return type(value) is int and value >= 0
    attempted, completed = ledger.get('attempted_updates'), ledger.get('completed_updates')
    cumulative_attempted = 142 + attempted if count(attempted) else None
    cumulative_completed = 142 + completed if count(completed) else None
    checks = {
        'prior_failed_ledger_142_preserved': (
            prior_ledger.get('attempted_updates') == prior_ledger.get('completed_updates') == 142
            and prior_ledger.get('total_budget') == 796 and prior_ledger.get('failed') is True
            and prior_ledger.get('partial_update_possible') is False),
        'prior_result_matches_ledger': prior_result.get('status') == 'failed' and prior_result.get('ledger') == prior_ledger,
        'prior_run_identity_matches': (
            controller.get('prior_failed_formal_run') == preflight.get('prior_failed_formal_run')
            == config.get('previous_failed_formal_run') == 'outputs/runs/sft_round1_ready_v1'),
        'prior_counts_match': (
            controller.get('prior_formal_attempted_updates') == controller.get('prior_formal_completed_updates')
            == preflight.get('prior_formal_attempted_updates') == preflight.get('prior_formal_completed_updates')
            == config.get('previous_formal_attempted_updates') == config.get('previous_formal_completed_updates') == 142),
        'new_run_is_independent': (
            controller.get('new_independent_run_from_original_base') is True
            and preflight.get('new_run_starts_from_original_base') is True
            and preflight.get('new_run_initial_update') == 0
            and config.get('run_dir') == 'outputs/runs/sft_round1_clock_v2'),
        'new_ledger_budget_796': ledger.get('total_budget') == 796,
        'cumulative_attempted_matches': cumulative_attempted is not None
            and controller.get('cumulative_formal_attempted_updates') == cumulative_attempted,
        'cumulative_completed_matches': cumulative_completed is not None
            and controller.get('cumulative_formal_completed_updates') == cumulative_completed,
    }
    return {'status': 'pass' if all(checks.values()) else 'failed', 'checks': checks,
        'prior_ledger': prior_ledger,
        'prior_ledger_sha256': evidence.digest(PRIOR_RUN / 'progress.json') if prior_ledger else None,
        'new_attempted_updates': attempted, 'new_completed_updates': completed,
        'computed_cumulative_formal_attempted_updates': cumulative_attempted,
        'computed_cumulative_formal_completed_updates': cumulative_completed,
        'reported_cumulative_formal_attempted_updates': controller.get('cumulative_formal_attempted_updates'),
        'reported_cumulative_formal_completed_updates': controller.get('cumulative_formal_completed_updates'),
        'note': 'Prior 142 remain failed historical work; they are never added to this adapter or replayed by this analysis.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-process-exit', action='store_true', required=True,
                        help='Operator confirms controller and all workers have exited; never run concurrently')
    parser.parse_args()
    if REPORT.exists():
        raise FileExistsError('Read-only analysis report already exists; do not overwrite or rerun')
    started = time.perf_counter()
    evidence = Evidence()
    result = {'status': 'incomplete', 'analysis_scope': 'Read-only after process exit; no model or data-set imports',
              'training_rerun': False, 'validation_examples_read': False, 'final_examples_read': False}
    try:
        controller = evidence.read(ROOT / 'reports/sft_round1_clock_v2_result.json')
        if controller is None:
            controller = evidence.read(RUN / 'report.json') or {}
        if controller.get('status') == 'running':
            raise RuntimeError('Controller record is still running; require terminal process-exit confirmation')
        preflight = evidence.read(RUN / 'preflight.json')
        if preflight is None:
            preflight = evidence.read(ROOT / 'reports/sft_round1_clock_v2_preflight.json', required=True) or {}
        ledger = evidence.read(RUN / 'progress.json', required=True) or {}
        groups = preflight.get('groups', [])
        config = preflight.get('config', {})
        accounting = formal_accounting(evidence, controller, ledger, preflight)
        manifest = evidence.read(RUN / 'artifact_sha256.json')
        integrity = {
            'run_artifacts': evidence.hashes(manifest, RUN),
            'frozen_sources': evidence.hashes(preflight.get('source_sha256'), ROOT),
            'old_and_frozen_protection': evidence.hashes(preflight.get('protected_sha256'), ROOT),
        }
        checkpoints = [checkpoint(evidence, end, preflight) for _, end in PLANNED]
        checkpoint_map = {item['update']: item for item in checkpoints}
        segments, updates, micro_ids, micro_updates = [], [], [], []
        for index in range(1, 5):
            row, history, ids, update_ids = segment(evidence, index, config.get('limits'), groups, controller)
            start, end = PLANNED[index - 1]
            if start and row['status'] != 'not_started':
                restored = evidence.read(RUN / f'segment_{index}/restore_verification.json')
                previous = checkpoint_map[start]
                if restored is None:
                    row['resume'] = {'status': 'not_completed', 'checkpoint_update': start}
                else:
                    verification = restored.get('verification', {})
                    exact = {key: verification.get(key) for key in ('adapter_exact', 'optimizer_exact', 'rng_exact',
                                                                  'parameter_order_exact', 'cuda_rng_restored')}
                    metadata_equal = restored.get('metadata') == previous.get('metadata')
                    fingerprints_equal = verification.get('fingerprints') == previous.get('fingerprints')
                    passed = previous['status'] == 'pass' and all(value is True for value in exact.values()) and metadata_equal and fingerprints_equal
                    row['resume'] = {'status': 'pass' if passed else 'failed', 'checkpoint_update': start,
                                     **exact, 'metadata_equal': metadata_equal, 'fingerprints_equal': fingerprints_equal}
            elif start:
                row['resume'] = {'status': 'not_started', 'checkpoint_update': start}
            else:
                row['resume'] = {'status': 'not_applicable', 'reason': 'Fresh original base and LoRA'}
            segments.append(row); updates.extend(history); micro_ids.extend(ids); micro_updates.extend(update_ids)
        expected_ids = [source_id for group in groups for source_id in group['source_indices']]
        committed_ids = [source_id for row in updates for source_id in row.get('source_indices', [])]
        update_numbers = [row.get('update') for row in updates]
        schedule_mismatches = []
        for row in updates:
            update = row.get('update')
            if not isinstance(update, int) or not 1 <= update <= len(groups):
                schedule_mismatches.append({'update': update, 'error': 'Unexpected update index'})
                continue
            group = groups[update - 1]
            for row_key, group_key in (('source_indices', 'source_indices'), ('total_target_tokens', 'group_target_tokens'), ('lr_used', 'learning_rate')):
                if row.get(row_key) != group.get(group_key):
                    schedule_mismatches.append({'update': update, 'field': row_key})
        blocks = []
        gradient_failures = []
        for start, end in PLANNED:
            rows = [row for row in updates if isinstance(row.get('update'), int) and start < row['update'] <= end]
            before, after, losses = Stat(), Stat(), Stat()
            weighted_sum, total_tokens = 0.0, 0
            for row in rows:
                loss, count = row.get('group_loss'), row.get('total_target_tokens')
                before.add(row.get('grad_norm_before_clip')); after.add(row.get('grad_norm_after_clip')); losses.add(loss)
                if number(loss) and number(count) and count > 0:
                    weighted_sum += loss * count; total_tokens += count
                if (not number(loss) or row.get('gradients_finite_nonzero') is not True
                        or not number(row.get('changed_parameter_elements')) or row['changed_parameter_elements'] <= 0
                        or not number(row.get('grad_norm_after_clip')) or row['grad_norm_after_clip'] > 1.00001):
                    gradient_failures.append(row.get('update'))
            blocks.append({'updates': [start + 1, end], 'recorded_updates': len(rows),
                           'status': 'complete' if [row.get('update') for row in rows] == list(range(start + 1, end + 1)) else 'not_completed',
                           'valid_target_tokens': total_tokens,
                           'token_weighted_loss': weighted_sum / total_tokens if total_tokens else None,
                           'group_loss': losses.result(), 'gradient_norm_before_clip': before.result(), 'gradient_norm_after_clip': after.result()})
        active_segments = [row for row in segments if row['status'] != 'not_started']
        def maxima(field, subfield):
            values = [row[field].get(subfield) for row in active_segments]
            values = [value for value in values if number(value)]
            return max(values) if values else None
        def torch_max(key):
            values = [row['memory'][key]['max'] for row in active_segments]
            values = [value for value in values if number(value)]
            return max(values) if values else None
        data_complete = (len(expected_ids) == 6364 and len(set(expected_ids)) == 6364
                         and committed_ids == expected_ids and micro_ids == expected_ids
                         and update_numbers == list(range(1, 797)) and not schedule_mismatches
                         and not any(row['full_length_mismatches'] for row in active_segments))
        ledger_complete = ledger.get('attempted_updates') == ledger.get('completed_updates') == ledger.get('total_budget') == 796 and ledger.get('failed') is False and ledger.get('partial_update_possible') is False
        checkpoint_ok = all(row['status'] == 'pass' for row in checkpoints)
        process_ok = len(active_segments) == 4 and all(row['status'] == row.get('worker_status') == 'pass'
            and row.get('identity_verified') is True and row.get('worker_matches_binding') is True
            and row.get('exit_code') == 0 and row.get('job_active_pids_after_cleanup') == []
            and not row.get('errors') and not row.get('cleanup_errors')
            and row.get('base_unchanged') is True for row in active_segments)
        restores_ok = all(row.get('resume', {}).get('status') == 'pass' for row in segments[1:])
        cache_ok = sum(row['cache']['calls'] for row in active_segments) == 6364 and all(row['cache']['all_identity_preserved'] is True for row in active_segments)
        integrity_ok = all(row['status'] == 'pass' for row in integrity.values())
        acceptance = {'controller_pass': controller.get('status') == 'pass', 'ledger_complete': ledger_complete,
                      'prior_and_cumulative_accounting_verified': accounting['status'] == 'pass',
                      'full_data_complete': data_complete, 'checkpoints_verified': checkpoint_ok,
                      'processes_and_base_verified': process_ok, 'restores_verified': restores_ok,
                      'cache_complete_and_preserved': cache_ok, 'finite_gradients_and_losses': not gradient_failures and len(updates) == 796,
                      'frozen_and_run_hashes_verified': integrity_ok, 'no_analysis_errors': not evidence.errors}
        verified_checkpoints = [item for item in checkpoints if item['status'] == 'pass']
        last = max(verified_checkpoints, key=lambda item: item['update']) if verified_checkpoints else None
        result.update(status='pass' if all(acceptance.values()) else 'failed' if controller.get('status') in ('pass', 'failed') or ledger.get('failed') else 'incomplete',
            controller_status=controller.get('status', 'not_completed'), controller_error=controller.get('error'),
            controller_traceback=controller.get('traceback'), ledger=ledger,
            formal_accounting=accounting,
            prior_formal_attempted_updates=142, prior_formal_completed_updates=142,
            cumulative_formal_attempted_updates=accounting['computed_cumulative_formal_attempted_updates'],
            cumulative_formal_completed_updates=accounting['computed_cumulative_formal_completed_updates'],
            attempted_updates=ledger.get('attempted_updates'), completed_updates=ledger.get('completed_updates'),
            history_recorded_updates=len(updates), worker_launches=controller.get('worker_launches'),
            automatic_retries=controller.get('automatic_retries'), gpu_window_seconds=controller.get('gpu_window_seconds'),
            controller_seconds=controller.get('total_controller_seconds'), segments=segments, loss_blocks=blocks,
            loss_interpretation='Each block contains different training examples and changing model states; these training losses are not before/after evaluation and do not demonstrate EX improvement.',
            gradient_failure_updates=gradient_failures, schedule_mismatches=schedule_mismatches,
            data={'scheduled_examples': len(expected_ids), 'committed_history_examples': len(committed_ids),
                  'committed_unique_examples': len(set(committed_ids)), 'completed_microbatches': len(micro_ids),
                  'completed_micro_unique_examples': len(set(micro_ids)),
                  'micro_order_matches_schedule_prefix': micro_ids == expected_ids[:len(micro_ids)],
                  'committed_order_matches_schedule_prefix': committed_ids == expected_ids[:len(committed_ids)],
                  'full_6364_complete': data_complete,
                  'note': 'A completed microbatch can belong to an unfinished accumulation group; only committed history is counted as a completed update.'},
            memory={'torch_peak_allocated_mib': torch_max('peak_allocated_mib'),
                    'torch_peak_reserved_mib': torch_max('peak_reserved_mib'),
                    'allocator_retry_max_per_process': torch_max('num_alloc_retries'),
                    'allocator_oom_max_per_process': torch_max('num_ooms'),
                    'pdh_sampled_dedicated_peak_mib': maxima('pdh', 'dedicated_sampled_peak_mib'),
                    'pdh_sampled_shared_peak_mib': maxima('pdh', 'shared_sampled_peak_mib'),
                    'sample_count': sum(row['pdh']['sample_count'] for row in active_segments),
                    'valid_samples': sum(row['pdh']['valid_samples'] for row in active_segments),
                    'limit': 'PDH maxima are sampled per-process WDDM values, not instantaneous global GPU peaks. Unknown samples are not zero.'},
            cache={'calls': sum(row['cache']['calls'] for row in active_segments),
                   'identity_preserved_calls': sum(row['cache']['identity_preserved_calls'] for row in active_segments)},
            checkpoints=checkpoints, last_durable_checkpoint=None if last is None else {'update': last['update'], 'path': last['path']},
            integrity=integrity, acceptance=acceptance,
            updates=[{key: row.get(key) for key in ('update', 'group_loss', 'total_target_tokens', 'lr_used', 'next_update_lr',
                     'grad_norm_before_clip', 'grad_norm_after_clip', 'gradients_finite_nonzero', 'changed_parameter_elements')} for row in updates])
    except Exception as error:
        result.update(status='analysis_failed', analysis_error=repr(error), traceback=traceback.format_exc())
    result.update(missing_artifacts=sorted(set(evidence.missing)), analysis_errors=evidence.errors,
                  analysis_seconds=time.perf_counter() - started)
    with REPORT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({key: result.get(key) for key in ('status', 'controller_status', 'attempted_updates',
          'completed_updates', 'cumulative_formal_attempted_updates', 'cumulative_formal_completed_updates',
          'worker_launches', 'gpu_window_seconds', 'memory', 'last_durable_checkpoint', 'analysis_error')}, ensure_ascii=False))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
