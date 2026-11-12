"""Read the completed pilot evidence without importing any model library."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/runs/sft_round2_pilot_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def main():
    report = read(RUN / 'report.json')
    assert report['status'] == 'pass', 'Only summarize a completed successful pilot'
    cpu = read(ROOT / 'reports/sft_round2_pilot_cpu_v2_checks.json')
    restored = read(RUN / 'segment_2/restore_verification.json')
    memory, telemetry, releases = [], [], []
    for index in (1, 2):
        folder = RUN / f'segment_{index}'
        memory.extend(lines(folder / 'memory_events.jsonl'))
        telemetry.extend(lines(folder / 'pid_memory.jsonl'))
        releases.extend(lines(folder / 'cache_releases.jsonl'))
    history = [row for worker in report['worker_results'] for row in worker['history']]
    fields = ('update', 'group_loss', 'grad_norm_before_clip', 'grad_norm_after_clip',
              'weighted_target_denominator', 'total_target_tokens', 'lr_used',
              'optimizer_cuda_mib_after', 'changed_parameter_elements')
    valid = [row for row in telemetry if row['status'] == 'ok']
    selection = read(ROOT / 'data/processed/spider_sft_round2_pilot_v1/selection.json')
    assert [i for row in history for i in row['source_indices']] == selection['selected_source_indices']
    summary = {
        'status': 'pass', 'cpu_seconds': cpu['seconds'],
        'cpu_cuda_initialized': cpu['cuda_initialized'],
        'cpu_continuous_vs_resumed_exact': cpu['checks']['four_update_resume']['exact_comparison'],
        'frozen_sources_count': len(cpu['source_sha256']),
        'freeze_verified_before_and_after': cpu['frozen_sources_verified_before_and_after'],
        'gpu_window_seconds': report['gpu_window_seconds'],
        'total_controller_seconds': report['total_controller_seconds'],
        'ledger': report['ledger'], 'automatic_retries': report['automatic_retries'],
        'updates': [{field: row[field] for field in fields} for row in history],
        'same_32_full_samples_and_order': True,
        'bucket_counts': selection['bucket_counts'], 'weight_counts': selection['weight_counts'],
        'maximum_sequence_length': selection['max_sequence_length'],
        'maximum_answer_tokens': selection['max_answer_tokens'],
        'peak_allocated_mib': max(row['peak_allocated_mib'] for row in memory),
        'peak_reserved_mib': max(row['peak_reserved_mib'] for row in memory),
        'peak_process_dedicated_mib': max(row['dedicated_bytes'] for row in valid) / 2**20,
        'peak_process_shared_mib': max(row['shared_bytes'] for row in valid) / 2**20,
        'memory_sample_count': len(telemetry), 'valid_memory_sample_count': len(valid),
        'maximum_num_ooms': max(row['num_ooms'] for row in memory if row['num_ooms'] is not None),
        'maximum_num_alloc_retries': max(row['num_alloc_retries'] for row in memory if row['num_alloc_retries'] is not None),
        'cache_release_count': len(releases),
        'cache_gradient_adam_preserved': all(row['gradient_and_adam_identity_version_preserved'] for row in releases),
        'cache_maximum_allocated_delta_mib': max(abs(row['after']['allocated_mib'] - row['before']['allocated_mib']) for row in releases),
        'restore_checks': {key: restored['verification'][key] for key in
                           ('adapter_exact', 'optimizer_exact', 'rng_exact', 'parameter_order_exact', 'cuda_rng_restored')},
        'segments': [{key: segment[key] for key in ('worker_pid', 'bound_pid', 'identity_verified',
                     'exit_code', 'job_active_pids_after_cleanup', 'wall_seconds')} for segment in report['segments']],
        'protected_files_verified_after': report['protected_files_verified_after'],
        'base_unchanged_both_segments': all(worker['base_unchanged'] for worker in report['worker_results']),
        'validation_examples_read': report['validation_examples_read'],
        'final_examples_read': report['final_examples_read'], 'evaluation_performed': report['evaluation_performed'],
        'limitations': 'Four-step engineering acceptance only; no quality-gain conclusion or GPU uninterrupted-vs-resumed reference run.',
    }
    path = ROOT / 'reports/sft_round2_pilot_v2_analysis.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
