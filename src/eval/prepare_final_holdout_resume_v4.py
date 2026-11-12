"""Freeze an SFT-only attempt; bind the existing complete v3 base in place.

This is saved-record validation only: no CUDA/model/SQL, no base copying and no
new base generation.  v3 source, preflight, failed scoring and resource-gate
evidence remain immutable.  The only inference changes are paths, two base-dir
bindings and explicit rejection of the base arm.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.final_holdout_common_v1 import (
    read_json, write, sha256, relative, verify_prepared_binding,
)

CONFIG = ROOT / 'configs/final_holdout_resume_v4.json'
V3_CONFIG = ROOT / 'configs/final_holdout_resume_v3.json'
V3_FREEZE = ROOT / 'reports/final_holdout_resume_v3_freeze.json'
V3_PREFLIGHT = ROOT / 'reports/final_holdout_resume_v3_preflight.json'
V3_GATE = ROOT / 'reports/final_holdout_resume_v3_sft_start_resources.json'
V3_GENERATOR = ROOT / 'src/infer/final_holdout_resume_v3.py'
V4_GENERATOR = ROOT / 'src/infer/final_holdout_resume_v4.py'
BASE = ROOT / 'outputs/final_holdout_v3/base'
FAILED_SCORE = ROOT / 'outputs/final_holdout_v3/scores_base'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_clone(cfg, parent_cfg):
    expected_cfg = dict(parent_cfg, version='final_holdout_resume_v4',
        run_root='outputs/final_holdout_v4',
        frozen_report='reports/final_holdout_resume_v4_freeze.json',
        preflight_report='reports/final_holdout_resume_v4_preflight.json',
        arms_in_order=['sft'], base_generation_dir=relative(BASE))
    require(cfg == expected_cfg, 'v4 changed unapproved model/generation/resource configuration')
    body = V3_GENERATOR.read_text(encoding='utf-8').replace('final_holdout_resume_v3', 'final_holdout_resume_v4')
    patches = (
        ("base_prompts = ROOT / config['run_root'] / 'base/prompts.jsonl'",
         "base_prompts = ROOT / config['base_generation_dir'] / 'prompts.jsonl'"),
        ("base_dir = ROOT / config['run_root'] / 'base'", "base_dir = ROOT / config['base_generation_dir']"),
        ("parser.add_argument('--arm', choices=['base', 'sft'], required=True)",
         "parser.add_argument('--arm', choices=['sft'], required=True)"),
        ('def worker(arm):\n', "def worker(arm):\n    if arm != 'sft':\n        raise ValueError('v4 is SFT-only; completed base predictions must never be regenerated')\n"),
        ('def run(arm):\n', "def run(arm):\n    if arm != 'sft':\n        raise ValueError('v4 is SFT-only; completed base predictions must never be regenerated')\n"),
    )
    for old, new in patches:
        require(body.count(old) == 1, f'Source patch is not unique: {old}')
        body = body.replace(old, new, 1)
    actual = V4_GENERATOR.read_text(encoding='utf-8')
    require(actual == body, 'v4 generator changed beyond approved path/binding/base-rejection patches')
    ast.parse(actual, filename=relative(V4_GENERATOR))
    ast.parse(Path(__file__).read_text(encoding='utf-8'), filename=relative(Path(__file__)))
    return {'parent_generator_sha256': sha256(V3_GENERATOR), 'generator_sha256': sha256(V4_GENERATOR),
            'parent_config_sha256': sha256(V3_CONFIG), 'config_sha256': sha256(CONFIG),
            'approved_path_changes': ['version', 'run_root', 'frozen_report', 'preflight_report'],
            'base_binding': relative(BASE), 'allowed_arm': 'sft', 'new_base_generation_forbidden': True,
            'generator_text_matches_exact_approved_patches': True, 'ast_parse_passed': True}


def prepare():
    started = time.perf_counter(); cfg = read_json(CONFIG)
    destination = ROOT / cfg['preflight_report']; freeze_path = ROOT / cfg['frozen_report']
    require(not destination.exists() and not freeze_path.exists() and not (ROOT/cfg['run_root']).exists(),
            'v4 attempt already claimed')
    write(destination, {'status': 'checking', 'training': False, 'automatic_retries': 0}, True)
    result = {'status': 'failed'}
    try:
        v3_cfg, v3_frozen, v3_pref = read_json(V3_CONFIG), read_json(V3_FREEZE), read_json(V3_PREFLIGHT)
        require(v3_pref['status'] == 'pass' and v3_pref['config'] == v3_cfg == v3_frozen['config'],
                'v3 preflight/config was not successful and unchanged')
        require(v3_pref['frozen_report_sha256'] == sha256(V3_FREEZE), 'v3 freeze binding differs')
        protected = dict(v3_frozen['source_sha256'])
        for name, expected in protected.items():
            require(sha256(ROOT/name) == expected, f'Frozen historical source changed: {name}')
        gate = read_json(V3_GATE)
        require(gate['status'] == 'not_ready' and gate['run_directory_created'] is False
                and gate['cuda_initialized'] is False and gate['model_loaded'] is False
                and gate['training_updates'] == 0, 'Expected v3 SFT zero-worker resource gate failure')
        require(not (ROOT/v3_cfg['run_root']/'sft').exists(), 'v3 SFT arm already exists; cannot claim no prior attempt')
        clone_evidence = verify_clone(cfg, v3_cfg)
        parent = verify_prepared_binding()
        for package, expected in parent['packages'].items():
            require(version(package) == expected, f'Package changed: {package}')
        require(version('peft') == parent['peft_version'], 'PEFT changed')
        for mode in ('cpu', 'live'):
            path = ROOT/f'outputs/checks/final_telemetry_v2_{mode}_v1/result.json'
            report = read_json(path)
            require(report['status'] == 'pass' and report['sources_unchanged'], f'{mode} infrastructure checks did not pass')
            for name, expected in report['source_sha256'].items():
                require(sha256(ROOT/name) == expected == protected[name], f'Tested helper changed: {name}')
        from src.infer.final_holdout_resume_v4 import preserved_prefix, PARENT_PROMPTS
        prefix, prior_rows = preserved_prefix()
        base, controller = read_json(BASE/'summary.json'), read_json(BASE/'controller_result.json')
        require(base['status'] == controller['status'] == 'pass' and controller['exit_code'] == 0
                and controller['job_active_pids_after_cleanup'] == [], 'Existing base did not finish and clean up')
        require(all(base[key] == 1034 for key in ('prediction_count', 'generated_count', 'attempted_count')),
                'Base population is incomplete')
        require(base['reused_prefix_count'] == 616 and base['new_generated_count'] == 418
                and base['generation_failure_or_skip_count'] == 0 and base['automatic_retries'] == 0,
                'Base continuation lineage differs')
        raw = (BASE/'predictions.jsonl').read_bytes()
        require(sha256(BASE/'predictions.jsonl') == base['prediction_sha256'] and raw.startswith(prefix),
                'Base prediction hash or original 616 bytes changed')
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
        require([row['source_index'] for row in rows] == list(range(1034)), 'Base prediction IDs differ')
        require(all(row['attempt_number'] == 1 and row['generation_ok'] is True for row in rows),
                'Base records contain repeated or failed predictions')
        require(sha256(BASE/'prompts.jsonl') == sha256(PARENT_PROMPTS), 'Base prompts differ from original freeze')
        require(base['model_config'] == parent['model_config'] and base['model_weight_sha256'] == parent['weight_sha256'],
                'Base model differs from preflight')
        require(base['generation_config'] == parent['generation_config'], 'Base generation protocol differs')
        failed_score = read_json(FAILED_SCORE/'summary.json')
        require(failed_score['status'] == 'failed' and failed_score['complete_final_scored'] is False,
                'Expected preserved incomplete strict-decoding score attempt')
        from src.train.windows_job_process_v2 import interpreter_evidence
        interpreter = interpreter_evidence()
        require(all(row['matches'] for row in interpreter), 'Interpreter changed')
        for directory in (BASE, FAILED_SCORE):
            for path in directory.rglob('*'):
                if path.is_file():
                    protected[relative(path)] = sha256(path)
        for path in (CONFIG, Path(__file__), V4_GENERATOR, V3_CONFIG, V3_FREEZE, V3_PREFLIGHT, V3_GATE):
            protected[relative(path)] = sha256(path)
        run = ROOT/cfg['run_root']; run.mkdir(parents=True, exist_ok=False)
        for path in (CONFIG, Path(__file__), V4_GENERATOR):
            target = run/'frozen_snapshot'/relative(path)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        frozen = {'status': 'frozen_sft_only_new_attempt_after_zero_worker_gate',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg, 'source_sha256': protected,
            'v3_freeze_sha256': sha256(V3_FREEZE), 'approved_clone_evidence': clone_evidence,
            'base_generation_dir': relative(BASE), 'base_prediction_sha256': base['prediction_sha256'],
            'base_reused_prefix_examples': len(prior_rows), 'base_examples_complete': 1034,
            'base_copy_performed': False, 'base_generation_authorized': False,
            'sft_expected_examples': 1034, 'v3_sft_worker_launches': 0,
            'model_prompt_generation_unchanged': True, 'scoring_not_performed': True,
            'failed_scoring_preserved': True, 'training': False, 'final_result_used_for_tuning': False}
        write(freeze_path, frozen, True)
        result = {**parent, 'status': 'pass', 'config': cfg, 'frozen_report_sha256': sha256(freeze_path),
            'parent_preflight_sha256': sha256(ROOT/cfg['parent_preflight']),
            'telemetry_executable_sha256': v3_pref['telemetry_executable_sha256'],
            'telemetry_executable_path': v3_pref['telemetry_executable_path'],
            'approved_clone_evidence': clone_evidence, 'protected_source_count': len(protected),
            'base_generation_dir': relative(BASE), 'base_prediction_sha256': base['prediction_sha256'],
            'reused_prefix_count': 616, 'new_base_examples': 0, 'new_sft_examples': 1034,
            'parent_cpu_checks_reused_by_unchanged_hash': True, 'interpreter_evidence': interpreter,
            'cuda_initialized': False, 'model_loaded': False, 'training_performed': False,
            'final_used_for_tuning': False, 'failed_scoring_preserved': True}
        require(not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'bitsandbytes')),
                'CPU preflight imported model libraries')
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        result['sft_only_cpu_preflight_seconds'] = time.perf_counter()-started
        write(destination, result)
    print({key: result[key] for key in ('status', 'base_generation_dir', 'new_base_examples',
          'new_sft_examples', 'protected_source_count', 'sft_only_cpu_preflight_seconds')}, flush=True)


if __name__ == '__main__':
    prepare()
