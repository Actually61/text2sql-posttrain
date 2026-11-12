"""Offline evidence audit only: no GPU command, model imports or SQL execution."""
from __future__ import annotations
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/checks/final_telemetry_audit_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def write_new(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    root = ROOT / 'outputs/final_holdout_v1'
    arm = root / 'base'
    frozen = read(ROOT / 'reports/final_holdout_v1_freeze.json')
    preflight = read(ROOT / 'reports/final_holdout_v1_preflight.json')
    failure = read(root / 'failure_analysis.json')
    protected = dict(frozen['source_sha256'])
    protected.update(preflight['output_sha256'])
    protected.update(failure['source_sha256'])
    # Compare old expected bytes before establishing this audit's own snapshot.
    for name, expected in protected.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Existing frozen evidence changed: {name}')
    for path in root.rglob('*'):
        if path.is_file():
            protected[rel(path)] = digest(path)
    for name in ('data/processed/spider_final_holdout_v1/manifest.json',
                 'reports/final_holdout_v1_base_start_resources.json',
                 'docs/项目总结与简历面试手册.md', 'README.md'):
        protected[name] = digest(ROOT / name)
    write_new(OUT / 'preservation.json', {'scope':'Preserve failed run, 616 predictions, frozen sources and model/data/protocol bytes',
        'source_sha256':protected, 'training_or_scoring':False})

    samples = read(arm / 'gpu_samples.json')
    times = [s['elapsed_seconds'] for s in samples['samples']]
    intervals = [b-a for a,b in zip(times,times[1:])]
    if not all(math.isfinite(x) and x >= 0 for x in intervals):
        raise ValueError('Invalid/non-monotonic sample timestamps')
    ordered = sorted(intervals)
    def percentile(p):
        return ordered[math.ceil(p*len(ordered))-1]
    largest = max(range(len(intervals)), key=intervals.__getitem__)
    before = samples['samples'][largest]
    after = samples['samples'][largest+1]
    predictions_bytes = (arm / 'predictions.jsonl').read_bytes()
    if not predictions_bytes.endswith(b'\n'):
        raise ValueError('Partial final prediction record')
    rows = [json.loads(line) for line in predictions_bytes.decode('utf-8').splitlines()]
    if len(rows) != 616 or [row['source_index'] for row in rows] != list(range(616)):
        raise ValueError('Unexpected prediction membership/order')
    if not all(row['ordinal']==i+1 and row['attempt_number']==1 and row['generation_ok']
               and row['generation_status']=='generated' and row['ended_with_eos'] for i,row in enumerate(rows)):
        raise ValueError('Unexpected saved generation state')
    controller = read(arm / 'controller_result.json')
    status = read(arm / 'status.json')
    if status['status'] != 'failed' or controller['exit_code'] != 1 or controller['job_active_pids_after_cleanup']:
        raise ValueError('Failure/cleanup state differs')
    if any((root / p).exists() for p in ('sft','scores_base','scores_sft','comparison.json')):
        raise ValueError('Later experiment was unexpectedly started')
    exit_record = controller['exits'][0]
    epoch = datetime(1601,1,1,tzinfo=timezone.utc)
    def file_time(value):
        return epoch + timedelta(microseconds=value//10)
    created = file_time(exit_record['creation_time_100ns'])
    exited = file_time(exit_record['exit_time_100ns'])

    # Record the installed Python implementation behind the 5-second message.
    import subprocess
    sub_path = Path(subprocess.__file__)
    source = sub_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    run_node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    lines = source.splitlines()
    hits = [{'line':i+1,'code':lines[i]} for i in range(run_node.lineno-1,run_node.end_lineno)
            if any(term in lines[i] for term in ('with Popen','communicate(','process.kill()', 'except TimeoutExpired','process.wait()'))]
    (OUT / 'installed_subprocess_run.txt').write_text('\n'.join(f'{i+1}: {lines[i]}'
        for i in range(run_node.lineno-1,run_node.end_lineno))+'\n', encoding='utf-8')
    result = {
        'status':'audit_complete_not_a_runtime_fix', 'saved_predictions':len(rows),
        'prediction_sha256':digest(arm/'predictions.jsonl'), 'prediction_ids_contiguous_unique':True,
        'all_attempts_one_and_eos':True, 'unattempted_final_base_examples':1034-len(rows),
        'valid_sample_count':len(times), 'error_count':len(samples['errors']),
        'errors':samples['errors'], 'gap_seconds':{'median':statistics.median(intervals),
            'p95':percentile(.95),'p99':percentile(.99),'max':max(intervals),
            'above_1s':sum(x>1 for x in intervals),'other_max':max(x for i,x in enumerate(intervals) if i!=largest)},
        'largest_gap':{'before':before,'after':after,'seconds':intervals[largest],
            'later_successful_samples':samples['samples'][largest+2:]},
        'generation_seconds_sum':sum(row['generation_seconds'] for row in rows),
        'maximum_query_seconds':max(row['generation_seconds'] for row in rows),
        'last_query_seconds':rows[-1]['generation_seconds'],
        'controller_seconds':controller['controller_seconds'],
        'allocated_peak_mib':max(row['peak_allocated_mib'] for row in rows),
        'reserved_peak_mib':max(row['peak_reserved_mib'] for row in rows),
        'whole_gpu_sampled_peak_mib':max(row['used_mib'] for row in samples['samples']),
        'worker_created_utc':created.isoformat(), 'worker_exited_utc':exited.isoformat(),
        'worker_created_shanghai':created.astimezone(timezone(timedelta(hours=8))).isoformat(),
        'worker_exited_shanghai':exited.astimezone(timezone(timedelta(hours=8))).isoformat(),
        'worker_exit_code':1,'job_empty':True,
        'python_runtime':{'version':sys.version,'executable':sys.executable,
            'subprocess_path':str(sub_path),'subprocess_sha256':digest(sub_path),'run_timeout_code':hits},
        'timing_limits':'Sample error has no timestamp/PID or call boundaries; prediction durations have no shared absolute anchor. Do not equate the 17.955s gap with command duration or a specific query.',
        'confirmed_trigger':'Sampler TimeoutExpired was retained in errors; inference stopped new-question dispatch at its next error check.',
        'root_cause':'Undetermined: no evidence separates OS scheduling, creation/cleanup/pipe delay, or driver/NVML response.',
        'oom_exception_observed':False, 'final_allocator_oom_counter':'not available, not verified zero',
        'models_loaded_this_audit':False,'cuda_initialized_this_audit':False,
        'sql_executed_this_audit':False,'live_gpu_commands_run_this_audit':False,
        'original_predictions_or_sources_modified':False,'remediation_implemented':False,
        'model_run_retried':False,'final_scores_available':False,
        'protected_files_verified':len(protected),
        'evidence_sha256':{rel(p):digest(p) for p in OUT.iterdir() if p.is_file()},
    }
    if any(name in sys.modules for name in ('torch','transformers','peft','bitsandbytes')):
        raise RuntimeError('Audit imported a model/CUDA library')
    for name, expected in protected.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Evidence mutated during audit: {name}')
    write_new(OUT/'analysis.json',result)
    print(json.dumps({k:result[k] for k in ('status','saved_predictions','valid_sample_count','error_count',
        'gap_seconds','protected_files_verified','remediation_implemented')},ensure_ascii=False))


if __name__ == '__main__':
    main()
