"""Freeze the selected recipe before unsealing official dev; CPU-only preparation."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/final_holdout_v1.json'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src/data'))
sys.path.insert(0, str(ROOT / 'src/infer'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def verify_hashes(mapping):
    for name, expected in mapping.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Frozen file changed: {name}')


def verify_frozen():
    frozen = read_json(ROOT / read_json(CONFIG)['frozen_report'])
    if frozen['status'] != 'frozen_before_final_examples':
        raise ValueError('Recipe not frozen')
    verify_hashes(frozen['source_sha256'])
    return frozen


def verify_prepared_binding():
    verify_frozen()
    cfg = read_json(CONFIG)
    checked = read_json(ROOT / cfg['preflight_report'])
    if checked['status'] != 'pass' or checked['config'] != cfg:
        raise ValueError('Successful matching preflight required')
    frozen_hash = sha256(ROOT / cfg['frozen_report'])
    path = ROOT / cfg['input_dir'] / 'manifest.json'
    manifest = read_json(path)
    if (checked['frozen_report_sha256'] != frozen_hash
            or checked['input_manifest_sha256'] != sha256(path)
            or manifest['recipe_freeze_sha256'] != frozen_hash
            or manifest['output_sha256'] != checked['output_sha256']):
        raise ValueError('Prepared inputs/references no longer bind to CPU preflight')
    verify_hashes(checked['output_sha256'])
    return checked


def freeze():
    cfg = read_json(CONFIG)
    destination = ROOT / cfg['frozen_report']
    if destination.exists() or (ROOT / cfg['input_dir']).exists() or (ROOT / cfg['run_root']).exists():
        raise FileExistsError('Final experiment already claimed')
    prior = read_json(ROOT / 'reports/sft_round1_eval_v2_preflight.json')
    verify_hashes(prior['source_sha256'])
    preservation = read_json(ROOT / 'reports/sft_showcase_v1/preservation.json')['source_sha256']
    verify_hashes(preservation)
    source_hashes = {**preservation, **prior['source_sha256']}
    source_hashes.update(read_json(ROOT / 'data/processed/spider_inputs_v1/manifest.json')['input_sha256'])
    # The previously sealed manifest/database bytes are verified without parsing dev examples.
    final = read_json(ROOT / cfg['final_manifest'])
    split = read_json(ROOT / 'data/processed/spider_v1/split_manifest.json')
    selected = [final['source_file'], 'data/raw/spider/tables.json']
    selected += [f'data/raw/spider/database/{db}/{db}.sqlite' for db in final['db_ids']]
    expected = {name: split['source_sha256'][name] for name in selected}
    verify_hashes(expected); source_hashes.update(expected)
    if final['sha256'] != expected[final['source_file']] or final['example_count'] != 1034 or final['database_count'] != 20:
        raise ValueError('Sealed final identity differs')
    source_hashes['models/qwen2.5-coder-1.5b-instruct/model.safetensors'] = prior['weight_sha256']
    extra = [relative(CONFIG), relative(Path(__file__)), cfg['final_manifest'], cfg['reference_generation_report'],
             'reports/sft_round1_eval_v2_preflight.json', 'reports/spider_evaluator_source.json',
             'src/infer/final_holdout_v1.py', 'src/eval/evaluate_final_holdout_v1.py',
             'src/train/pid_supervision_formal_v2.py', 'src/train/windows_job_process.py',
             'src/train/windows_gpu_memory.py', 'src/train/cache_release_pilot.py',
             'src/eval/summarize_sft_round1_eval_v1.py']
    provenance = read_json(ROOT / 'reports/spider_evaluator_source.json')
    for name, meta in provenance['source_files'].items():
        key = str(Path(provenance['local_dir']) / name).replace('\\', '/')
        if sha256(ROOT / key) != meta['sha256']:
            raise ValueError('Comparator source drift')
        source_hashes[key] = meta['sha256']
    for name in extra:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
        source_hashes[name] = sha256(ROOT / name)
    verify_hashes(source_hashes)
    run = ROOT / cfg['run_root']; run.mkdir(parents=True, exist_ok=False)
    for name in extra:
        dest = run / 'frozen_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dest)
    write(run / 'handbook_before_final.md.json', {'path': 'docs/项目总结与简历面试手册.md',
        'sha256': sha256(ROOT / 'docs/项目总结与简历面试手册.md')}, True)
    shutil.copyfile(ROOT / 'docs/项目总结与简历面试手册.md', run / 'handbook_before_final.md')
    report = {'status': 'frozen_before_final_examples', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': cfg, 'source_sha256': source_hashes,
        'model_selected': 'round1 update796 based exclusively on internal validation',
        'adapter_sha256': prior['adapter_sha256'], 'adapter_fingerprint': prior['adapter_fingerprint'],
        'final_examples_parsed': False, 'training_performed': False,
        'automatic_retries': 0, 'final_used_for_tuning': False}
    write(destination, report, True)
    print(json.dumps({'status': report['status'], 'protected_files': len(source_hashes)}), flush=True)


def _write_jsonl(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def load_prepared():
    """Inference reads only answer-free inputs; references are hashed as opaque bytes."""
    import yaml
    from transformers import AutoTokenizer
    cfg = read_json(CONFIG)
    directory = ROOT / cfg['input_dir']
    manifest = read_json(directory / 'manifest.json')
    verify_hashes(manifest['output_sha256'])
    model = yaml.safe_load((ROOT / 'configs/model.yaml').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / model['local_dir'], local_files_only=True, trust_remote_code=False)
    rows = [json.loads(line) for line in (directory / 'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    if [r['source_index'] for r in rows] != list(range(cfg['expected_examples'])):
        raise ValueError('Final input identity/order changed')
    prepared = []
    for row in rows:
        if set(row) != {'source_index', 'source_file', 'db_id', 'project_split', 'messages'} or row['project_split'] != 'final':
            raise ValueError('Reference leaked into inference fields')
        if [m['role'] for m in row['messages']] != ['system', 'user']:
            raise ValueError('Unexpected inference roles')
        encoded = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True,
                                               return_tensors='pt', return_dict=True)
        if encoded['input_ids'].shape[1] > cfg['max_input_tokens']:
            raise ValueError('Input exceeds frozen complete-input guard; never truncate')
        prepared.append((row, encoded))
    return tokenizer, prepared


def prepare():
    cfg = read_json(CONFIG); target = ROOT / cfg['preflight_report']
    write(target, {'status': 'checking', 'automatic_retries': 0}, True)
    started = time.perf_counter()
    report = {'status': 'failed', 'automatic_retries': 0}
    try:
        frozen = verify_frozen()
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU preflight starts with CUDA initialized')
        previous = torch.cuda._lazy_init
        attempts = []
        def reject(*args, **kwargs):
            attempts.append(traceback.format_stack())
            raise RuntimeError('CUDA initialization forbidden in CPU preparation')
        torch.cuda._lazy_init = reject
        try:
            import yaml
            from transformers import GenerationConfig
            from transformers.generation.utils import GenerationMixin
            from types import SimpleNamespace
            from baseline_full import make_generation_config, verify_effective_config
            from schema_serialization import serialize_schema
            from prepare_model_inputs import build_messages
            from src.train.sft_checkpoint_cpu_v1 import inspect_final_adapter_cpu
            from src.train.windows_job_process_v2 import interpreter_evidence, JobProcess, VENV_PYTHON
            from src.train.windows_job_exit_wait_formal import terminate_and_wait, close_without_new_wait
            from src.train.pid_supervision_formal_v2 import confirm_natural_exit
            earlier = read_json(ROOT / 'reports/sft_round1_eval_v2_preflight.json')
            model = yaml.safe_load((ROOT / 'configs/model.yaml').read_text(encoding='utf-8'))
            baseline = read_json(ROOT / cfg['baseline_config'])
            generation = read_json(ROOT / cfg['reference_generation_report'])
            if baseline != generation['requested_config'] or model != generation['model_config']:
                raise ValueError('Original model or generation protocol drift')
            packages = {name: version(name) for name in earlier['packages']}
            if packages != earlier['packages'] or version('peft') != earlier['peft_version']:
                raise ValueError('Dependency versions drifted')
            provenance = read_json(ROOT / 'reports/spider_evaluator_source.json')
            if any(version(name) != v for name, v in provenance['dependency_versions'].items()):
                raise ValueError('Evaluator dependency drift')
            interpreter = interpreter_evidence()
            if not all(r['matches'] for r in interpreter):
                raise ValueError('Native Python launch files differ')
            # One bounded CPU child confirms that the packaged interpreter can launch
            # its native Job worker. Direct PowerShell launch is a different route.
            launch_log = ROOT / cfg['run_root'] / 'cpu_native_launch.log'
            job = None; handle = None
            try:
                job = JobProcess([str(VENV_PYTHON), '-B', '-c',
                    'import os,sys,json;print(json.dumps(dict(pid=os.getpid(),executable=sys.executable,prefix=sys.prefix)),flush=True)'],
                    ROOT, dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'), launch_log)
                handle = job.open_process(job.pid)
                exit_code = job.wait(8)
                confirm_natural_exit(job, [handle], lambda *a, **kw: None, time.perf_counter()+2, max_lifetime_seconds=10)
                if exit_code != 0 or handle.alive() or job.active_pids():
                    raise RuntimeError('Bounded CPU native launch/exit did not pass')
                identity = read_json(launch_log)
                if identity['pid'] != job.pid or Path(identity['prefix']).resolve() != ROOT / '.venv':
                    raise ValueError('CPU native child identity differs')
            finally:
                if job is not None and job.active_pids():
                    terminate_and_wait(job, [handle] if handle else [], lambda *a, **kw: None, timeout=2, max_lifetime_seconds=10)
                if handle is not None:
                    handle.close()
                if job is not None:
                    close_without_new_wait(job)
            adapter = inspect_final_adapter_cpu(ROOT / cfg['adapter_path'])
            if adapter['adapter_fingerprint'] != frozen['adapter_fingerprint']:
                raise ValueError('Selected adapter drift')
            trained = read_json(ROOT / cfg['training_report'])
            if trained['status'] != 'pass' or trained['ledger']['completed_updates'] != 796 or trained['ledger']['failed']:
                raise ValueError('Round1 training incomplete')
            final = read_json(ROOT / cfg['final_manifest'])
            split = read_json(ROOT / 'data/processed/spider_v1/split_manifest.json')
            allowed = set(final['db_ids'])
            if allowed & (set(split['splits']['train']['db_ids']) | set(split['splits']['validation']['db_ids'])):
                raise ValueError('Final database overlap')
            directory = ROOT / cfg['input_dir']; directory.mkdir(parents=True, exist_ok=False)
            # First authorized parsing of held-out examples, strictly after recipe/source freeze.
            examples = read_json(ROOT / final['source_file'])
            schemas = {r['db_id']: r for r in read_json(ROOT / 'data/raw/spider/tables.json') if r['db_id'] in allowed}
            if len(examples) != 1034 or {r['db_id'] for r in examples} != allowed or set(schemas) != allowed:
                raise ValueError('Final data membership differs')
            prompt = yaml.safe_load((ROOT / 'configs/prompt.yaml').read_text(encoding='utf-8'))
            schema_text = {db: serialize_schema(schemas[db]) for db in sorted(allowed)}
            inputs, references = [], []
            for i, row in enumerate(examples):
                common = {'source_index': i, 'source_file': final['source_file'], 'db_id': row['db_id'], 'project_split': 'final'}
                inputs.append({**common, 'messages': build_messages(schema_text[row['db_id']], row['question'], prompt)})
                references.append({**common, 'question': row['question'], 'gold_sql': row['query'], 'query': row['query'], 'sql': row['sql']})
            _write_jsonl(directory / 'inputs.jsonl', inputs)
            _write_jsonl(directory / 'references.jsonl', references)
            output_hashes = {relative(directory / name): sha256(directory / name) for name in ('inputs.jsonl', 'references.jsonl')}
            write(directory / 'manifest.json', {'example_count':1034, 'database_count':20, 'source_file':final['source_file'],
                'source_sha256':final['sha256'], 'recipe_freeze_sha256':sha256(ROOT / cfg['frozen_report']),
                'output_sha256':output_hashes, 'training_performed':False, 'final_used_for_tuning':False}, True)
            tokenizer, prepared = load_prepared()
            tokens = [int(encoded['input_ids'].shape[1]) for _, encoded in prepared]
            context = read_json(ROOT / model['local_dir'] / 'config.json')['max_position_embeddings']
            if max(tokens) + baseline['generation']['max_new_tokens'] > context:
                raise ValueError('Complete prompt plus output exceeds model context')
            defaults = GenerationConfig.from_pretrained(ROOT / model['local_dir'], local_files_only=True)
            request = make_generation_config(baseline['generation'], defaults, tokenizer.pad_token_id)
            effective, _ = GenerationMixin._prepare_generation_config(SimpleNamespace(generation_config=defaults), generation_config=request)
            verify_effective_config(effective, generation)
            if attempts or torch.cuda.is_initialized() or any(m in sys.modules for m in ('peft','bitsandbytes')):
                raise RuntimeError('CPU isolation violated')
            verify_frozen()
            report.update(status='pass', config=cfg, model_config=model, source_indices=list(range(1034)),
                examples=1034, databases=20, input_tokens_min=min(tokens), input_tokens_max=max(tokens),
                input_tokens_total=sum(tokens), inputs_exceeding_historical_1024_guard=sum(n>1024 for n in tokens),
                weight_path=earlier['weight_path'], weight_sha256=earlier['weight_sha256'], packages=packages,
                peft_version=version('peft'), adapter_fingerprint=adapter['adapter_fingerprint'],
                adapter_parameter_count=adapter['adapter_parameter_count'], generation_config=effective.to_dict(),
                input_manifest_sha256=sha256(directory/'manifest.json'), output_sha256=output_hashes,
                frozen_report_sha256=sha256(ROOT/cfg['frozen_report']), interpreter_evidence=interpreter,
                cpu_native_launch={'status':'pass','identity':identity,'exit_code':exit_code},
                cuda_initialized=False, cuda_initialization_attempts=0, peft_imported=False,bitsandbytes_imported=False,
                training_performed=False, final_used_for_tuning=False, examples_removed=0, truncated_examples=0,
                final_examples_prepared_after_recipe_freeze=True, reference_answers_in_model_inputs=False,
                gold_SQL_executed=False, inference_models_loaded=False)
        finally:
            torch.cuda._lazy_init = previous
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        report['cpu_preflight_seconds'] = time.perf_counter()-started
        write(target, report)
    print(json.dumps({k:report[k] for k in ('status','examples','input_tokens_max','inputs_exceeding_historical_1024_guard','cuda_initialized','cpu_preflight_seconds')}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true'); group.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    freeze() if args.freeze else prepare()
