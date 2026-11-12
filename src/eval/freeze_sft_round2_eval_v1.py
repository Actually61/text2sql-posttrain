"""Freeze a new single-pass round2 evaluation after static protocol review."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'reports/sft_round2_eval_v1_freeze.json'
SNAPSHOT = ROOT / 'outputs/checks/sft_round2_eval_v1_freeze'
SOURCES = [
    'configs/sft_round2_eval_v1.json', 'src/infer/sft_round2_eval_v1.py',
    'src/eval/evaluate_sft_round2_v1.py', 'src/eval/summarize_sft_round2_eval_v1.py',
    'src/eval/freeze_sft_round2_eval_v1.py', 'src/infer/baseline_full.py',
    'src/infer/baseline_pilot.py', 'src/eval/evaluate_spider.py', 'src/eval/spider_pair_worker.py',
    'src/eval/summarize_baseline_full.py', 'src/eval/summarize_sft_round1_eval_v2.py',
    'src/train/sft_checkpoint_cpu_v1.py', 'src/train/windows_job_process_v2.py',
    'src/train/windows_job_process.py', 'src/train/windows_job_exit_wait_formal.py',
    'src/train/pid_supervision_formal_v2.py', 'src/train/sft_round1_resource_v3.py',
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def function(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    return ast.dump(next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                         and node.name == name), include_attributes=False)


def main():
    if DEST.exists() or SNAPSHOT.exists():
        raise FileExistsError('Evaluation freeze already claimed; no overwrite')
    old_config = read(ROOT / 'configs/sft_round1_eval_v2.json')
    config = read(ROOT / 'configs/sft_round2_eval_v1.json')
    changed = {'version', 'adapter_path', 'training_report', 'run_dir', 'preflight_report',
               'generation_report', 'resource_readiness_report', 'freeze_manifest'}
    assert {k: v for k, v in config.items() if k not in changed} == {
        k: v for k, v in old_config.items() if k not in changed}
    assert config['adapter_path'] == 'outputs/runs/sft_round2_formal_v1/checkpoints/update_000796'
    for name in ('worker', 'run', 'preflight'):
        assert function('src/infer/sft_round2_eval_v1.py', name) == function('src/infer/sft_round1_eval_v2.py', name), name
    for name in ('dispatch', 'infrastructure_problem', 'check_generation'):
        assert function('src/eval/evaluate_sft_round2_v1.py', name) == function('src/eval/evaluate_sft_round1_v2.py', name), name
    for name in SOURCES:
        if name.endswith('.py'):
            tree = ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
            compile(tree, name, 'exec')
    preserved = {}
    def retain(path, expected=None):
        name = path.resolve().relative_to(ROOT.resolve()).as_posix()
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Prior evidence changed: {name}')
        if name in preserved and preserved[name] != actual:
            raise ValueError(f'Conflicting prior evidence: {name}')
        preserved[name] = actual
    first = read(ROOT / 'reports/sft_round1_eval_v2_generation.json')
    baseline = read(ROOT / 'reports/baseline_full_generation.json')
    for report in (first, baseline):
        for name, expected in report['source_sha256'].items():
            retain(ROOT / name, expected)
        retain(ROOT / report['prediction_path'], report['prediction_sha256'])
        for path in (ROOT / report['run_dir']).rglob('*'):
            if path.is_file():
                retain(path)
    for name in ('reports/sft_round1_eval_v2_eval.json', 'reports/baseline_full_eval.json'):
        scoring = read(ROOT / name)
        for path in (ROOT / scoring['run_dir']).rglob('*'):
            if path.is_file():
                retain(path)
        retain(ROOT / name)
    training_root = ROOT / 'outputs/runs/sft_round2_formal_v1'
    training_manifest = read(training_root / 'artifact_sha256.json')
    for name, expected in training_manifest.items():
        if name.startswith('checkpoints/') or name in ('report.json', 'progress.json', 'preflight.json'):
            retain(training_root / name, expected)
    retain(training_root / 'artifact_sha256.json')
    for name in ('reports/sft_round1_eval_v2_generation.json', 'reports/sft_round1_eval_v2_analysis.json',
                 'reports/baseline_full_generation.json', 'reports/sft_round2_formal_v1_result.json',
                 'reports/sft_round2_formal_v1_analysis.json', 'configs/sft_round2_weight_proposal_v1.json'):
        retain(ROOT / name)
    record = {'status': 'frozen', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': {name: digest(ROOT / name) for name in SOURCES},
              'preserved_sha256': preserved,
              'static_checks': ['same_generation_worker_controller_and_cuda_guard_ast',
                  'same_sql_dispatch_and_error_classification_ast', 'same_resource_budgets',
                  'sources_compile_without_import', 'prior_predictions_and_checkpoints_preserved'],
              'policy': 'No protected source edits after freeze. One CPU preflight, one 625-question generation and one EX scoring; no automatic retries.',
              'cuda_initialized': False, 'model_loaded': False, 'sql_executed': False,
              'final_examples_read': False}
    SNAPSHOT.mkdir(parents=True, exist_ok=False)
    for name, expected in record['source_sha256'].items():
        target = SNAPSHOT / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        assert digest(target) == expected
    with DEST.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'frozen', 'source_count': len(SOURCES),
                      'preserved_count': len(preserved), 'freeze_sha256': digest(DEST)}))


if __name__ == '__main__':
    main()
