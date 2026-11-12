"""One-shot evaluation generation, reusing the frozen baseline inference protocol."""
from __future__ import annotations
import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import baseline_full as baseline
from baseline_pilot import GPUSampler, extract_sql, read_json, relative, sha256

CONFIG = ROOT / 'configs/sft_round1_eval_v1.json'


def write(path, obj, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def preflight():
    """Tokenize inputs and inspect checkpoint bytes on CPU; never load a model."""
    started = time.perf_counter()
    config = read_json(CONFIG)
    frozen, pilot, model_config, tokenizer, prepared, checked = baseline.preflight()
    prior = read_json(ROOT / config['baseline_generation_report'])
    if prior['requested_config'] != frozen or prior['generation_config'] != checked['generation_config']:
        raise ValueError('Frozen full-baseline protocol differs')
    if not prior['complete_validation_generated'] or prior['prediction_count'] != 625:
        raise ValueError('Baseline is not complete')
    baseline.verify_hashes(prior['source_sha256'], 'Frozen baseline sources')
    baseline.verify_hashes({prior['prediction_path']: prior['prediction_sha256']}, 'Baseline predictions')
    prompts = [{'source_index': row['source_index'], 'db_id': row['db_id'],
        'input_token_count': int(encoded['input_ids'].shape[1]),
        'rendered_prompt': tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)}
        for row, encoded in prepared]
    prior_prompts = [json.loads(line) for line in (ROOT / prior['run_dir'] / 'prompts.jsonl').read_text(encoding='utf-8').splitlines()]
    if prompts != prior_prompts:
        raise ValueError('Prompt bytes/order/token lengths differ from baseline')
    from src.train.sft_checkpoint import inspect_checkpoint, state_fingerprint
    from safetensors.torch import load_file
    inspected = inspect_checkpoint(ROOT / config['adapter_path'])
    meta, complete = inspected['metadata'], inspected['manifest']
    if meta['completed_updates'] != 796 or meta['next_group_index'] != 796:
        raise ValueError('Only final 796 checkpoint is authorized')
    payload = load_file(str(ROOT / config['adapter_path'] / 'adapter_model.safetensors'), device='cpu')
    if state_fingerprint(payload) != complete['fingerprints']['adapter']:
        raise ValueError('Adapter tensor fingerprint mismatch')
    adapter_config = read_json(ROOT / config['adapter_path'] / 'adapter_config.json')
    if adapter_config['revision'] != model_config['revision'] or adapter_config['peft_type'] != 'LORA':
        raise ValueError('Adapter revision/type mismatch')
    trained = read_json(ROOT / config['training_report'])
    if trained['status'] != 'pass' or trained['ledger']['completed_updates'] != 796 or trained['ledger']['failed']:
        raise ValueError('Training did not finish successfully')
    protected = dict(checked['input_sha256'])
    for directory in (ROOT / prior['run_dir'], ROOT / config['adapter_path']):
        for path in directory.rglob('*'):
            if path.is_file():
                protected[relative(path)] = sha256(path)
    for name in (relative(CONFIG), relative(Path(__file__)), config['baseline_generation_report'],
                 config['training_report'], 'reports/baseline_full_eval.json', 'configs/evaluation.json',
                 'src/eval/evaluate_spider.py', 'src/eval/spider_pair_worker.py',
                 'src/eval/evaluate_sft_round1_v1.py', 'src/eval/summarize_sft_round1_eval_v1.py',
                 'src/eval/summarize_baseline_full.py',
                 'src/train/sft_checkpoint.py', 'src/train/sft_round1_resource_v3.py',
                 'src/train/windows_job_process_v2.py', 'src/train/windows_job_exit_wait_formal.py'):
        protected[name] = sha256(ROOT / name)
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    report = {**checked, 'status': 'pass', 'eval_config': config, 'source_sha256': protected,
        'adapter_path': config['adapter_path'], 'adapter_sha256': complete['files']['adapter_model.safetensors'],
        'adapter_fingerprint': complete['fingerprints']['adapter'], 'adapter_parameter_count': sum(t.numel() for t in payload.values()),
        'adapter_tensor_count': len(payload), 'peft_version': version('peft'),
        'baseline_prediction_sha256': prior['prediction_sha256'], 'prompts_identical_to_baseline': True,
        'cuda_initialized': False, 'training_performed': False,
        'cpu_preflight_seconds': time.perf_counter() - started}
    return config, frozen, prior, model_config, tokenizer, prepared, prompts, report


def worker():
    task_started = time.perf_counter()
    config = read_json(CONFIG)
    directory = ROOT / config['run_dir']
    if os.environ.get('SFT_EVAL_CLAIM') != str(directory.resolve()):
        raise RuntimeError('Worker requires its one-shot controller claim')
    predictions, model, sampler, cleanup_errors = [], None, None, []
    try:
        config, frozen, prior, model_config, tokenizer, prepared, prompts, checked = preflight()
        initial = read_json(ROOT / config['preflight_report'])
        if initial['source_sha256'] != checked['source_sha256']:
            raise ValueError('Sources changed after CPU preflight')
        baseline.verify_hashes(initial['source_sha256'], 'Frozen source before model loading')
        write(directory / 'preflight.json', checked, True)
        write(directory / 'worker_identity.json', {'pid': os.getpid(), 'executable': sys.executable, 'prefix': sys.prefix}, True)
        with (directory / 'prompts.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for prompt in prompts:
                stream.write(json.dumps(prompt, ensure_ascii=False) + '\n')
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel, get_peft_model_state_dict
        from src.train.sft_checkpoint import state_fingerprint
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA and BF16 required')
        random.seed(frozen['seed']); np.random.seed(frozen['seed']); torch.manual_seed(frozen['seed'])
        torch.cuda.manual_seed_all(frozen['seed']); torch.backends.cudnn.benchmark = False
        sampler = GPUSampler(frozen['gpu_sampling_interval_seconds']); sampler.start()
        if sampler.errors:
            raise RuntimeError('Initial GPU sampling failed')
        free_before, total = torch.cuda.mem_get_info()
        if free_before / 2**20 < config['minimum_free_mib']:
            raise RuntimeError('Free memory fell below 6656 MiB before model load')
        torch.cuda.reset_peak_memory_stats(); sampler.stage = 'load'
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(ROOT / model_config['local_dir'], local_files_only=True,
            trust_remote_code=False, device_map={'': 'cuda:0'}, dtype=torch.bfloat16,
            attn_implementation=frozen['attention_implementation'],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        model = PeftModel.from_pretrained(model, ROOT / config['adapter_path'], is_trainable=False,
                                         local_files_only=True).eval()
        model.requires_grad_(False)
        if state_fingerprint(get_peft_model_state_dict(model, adapter_name='default', save_embedding_layers=False)) != checked['adapter_fingerprint']:
            raise ValueError('Loaded adapter does not exactly match final checkpoint')
        if any(p.requires_grad for p in model.parameters()) or model.training or model.active_adapters != ['default']:
            raise ValueError('Inference-only adapter state invalid')
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        devices = sorted({str(p.device) for p in model.parameters()})
        if load_seconds > config['load_budget_seconds'] or devices != ['cuda:0'] or not model.is_loaded_in_4bit:
            raise RuntimeError('Model load budget/device/quantization check failed')
        load_allocated, load_reserved = torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20
        if load_allocated > config['allocated_limit_mib'] or load_reserved > config['reserved_limit_mib']:
            raise RuntimeError('Load memory limit exceeded')
        settings = frozen['generation']
        generation_config = baseline.make_generation_config(settings, model.generation_config, tokenizer.pad_token_id)
        effective, _ = model.base_model.model._prepare_generation_config(generation_config=generation_config)
        baseline.verify_effective_config(effective, prior)
        write(directory / 'generation_config.json', effective.to_dict(), True)
        write(directory / 'adapter_loaded.json', {'exact_fingerprint': checked['adapter_fingerprint'],
            'all_parameters_frozen': True, 'training_mode': False, 'active_adapters': model.active_adapters,
            'adapter_dtype': sorted({str(p.dtype) for n,p in model.named_parameters() if 'lora_' in n}),
            'base_loading_matches_baseline': True, 'optimizer_created': False}, True)
        print(f'Loaded final adapter exactly; all parameters frozen. Load={load_seconds:.3f}s. Generating 625 once.', flush=True)
        eos = effective.eos_token_id if isinstance(effective.eos_token_id, list) else [effective.eos_token_id]
        loop_started = time.perf_counter(); sampler.stage = 'generation'
        with (directory / 'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for ordinal, (source, encoded) in enumerate(prepared, 1):
                if time.perf_counter() - loop_started >= frozen['generation_loop_budget_seconds']:
                    raise TimeoutError('Generation loop exhausted its unchanged 3600-second budget')
                if sampler.errors:
                    raise RuntimeError('GPU telemetry failed; do not continue')
                inputs = output = None
                begun = time.perf_counter()
                error = None
                try:
                    inputs = {k: v.to('cuda:0') for k,v in encoded.items()}
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    begun = time.perf_counter()
                    with torch.inference_mode():
                        output = model.generate(**inputs, generation_config=deepcopy(generation_config))
                    torch.cuda.synchronize()
                    seconds = time.perf_counter() - begun
                    new_ids = output[0, encoded['input_ids'].shape[1]:].tolist()
                    text = tokenizer.decode(new_ids, skip_special_tokens=True)
                    ended = bool(new_ids and new_ids[-1] in eos)
                    limited = len(new_ids) >= settings['max_new_tokens'] and not ended
                    row = {'source_index': source['source_index'], 'db_id': source['db_id'], 'generation_ok': True,
                        'generation_status': 'generated', 'predicted_sql': extract_sql(text), 'generated_text': text,
                        'raw_output_with_special_tokens': tokenizer.decode(new_ids, skip_special_tokens=False),
                        'new_token_ids': new_ids, 'generated_token_count': len(new_ids), 'generation_seconds': seconds,
                        'ended_with_eos': ended, 'reached_token_limit': limited,
                        'stop_reason': 'eos' if ended else 'token_limit' if limited else 'time_or_other_stopping_criterion',
                        'effective_max_time_seconds': settings['max_time_seconds'],
                        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                        'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
                    if not ended and not limited:
                        error = TimeoutError('Generation ended by time or unexpected criterion; retain result and stop')
                    if row['peak_allocated_mib'] > config['allocated_limit_mib'] or row['peak_reserved_mib'] > config['reserved_limit_mib']:
                        error = RuntimeError('Generation memory threshold exceeded')
                except Exception as exc:
                    error = exc
                    row = baseline.record_failure(source, f'{type(exc).__name__}: {exc}', seconds=time.perf_counter()-begun)
                finally:
                    del output, inputs
                row.update(ordinal=ordinal, input_token_count=int(encoded['input_ids'].shape[1]), attempt_number=1)
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()
                if ordinal == 1 or ordinal % 25 == 0 or ordinal == 625:
                    print(f'[{ordinal}/625] id={row["source_index"]} tokens={row["generated_token_count"]} loop={time.perf_counter()-loop_started:.2f}s', flush=True)
                if error is not None:
                    raise error
        loop_seconds = time.perf_counter() - loop_started
        if [r['source_index'] for r in predictions] != checked['source_indices'] or len(predictions) != 625:
            raise ValueError('Incomplete or repeated predictions')
        if state_fingerprint(get_peft_model_state_dict(model, adapter_name='default', save_embedding_layers=False)) != checked['adapter_fingerprint']:
            raise ValueError('Adapter changed during inference')
        stats = torch.cuda.memory_stats()
        allocator = {k: int(stats.get(k,0)) for k in ('num_ooms','num_alloc_retries')}
        del model; model = None; torch.cuda.empty_cache()
        sampler.stage = 'after_release'; sampler.close()
        if sampler.errors:
            raise RuntimeError('GPU sampling errors at completion')
        baseline.verify_hashes(checked['source_sha256'], 'Frozen inputs/checkpoint/baseline after inference')
        if sha256(ROOT / checked['weight_path']) != checked['weight_sha256']:
            raise ValueError('Base weight changed')
        summary = {'run_id': config['version'], 'scope': config['scope'], 'status': 'pass',
            'prediction_count': 625, 'generated_count': 625, 'attempted_count': 625,
            'generation_failure_or_skip_count': 0, 'automatic_retries': 0,
            'prediction_selection_performed': False, 'pilot_predictions_reused': False,
            'ended_with_eos_count': sum(r['ended_with_eos'] for r in predictions),
            'token_limit_count': sum(r['reached_token_limit'] for r in predictions),
            'generated_tokens_total': sum(r['generated_token_count'] for r in predictions),
            'generation_seconds_total': sum(r['generation_seconds'] for r in predictions),
            'generation_seconds_median': statistics.median(r['generation_seconds'] for r in predictions),
            'generation_loop_wall_seconds': loop_seconds, 'load_seconds': load_seconds,
            'end_to_end_seconds': time.perf_counter()-task_started, 'first_example_seconds': predictions[0]['generation_seconds'],
            'warmup_generations': 0, 'load_peak_allocated_mib': load_allocated, 'load_peak_reserved_mib': load_reserved,
            'generation_peak_allocated_mib': max(r['peak_allocated_mib'] for r in predictions),
            'generation_peak_reserved_mib': max(r['peak_reserved_mib'] for r in predictions),
            'nvidia_smi_sampled_peak_used_mib': max(r['used_mib'] for r in sampler.samples),
            'nvidia_smi_sample_count': len(sampler.samples), 'nvidia_smi_sampling_error_count': len(sampler.errors),
            'nvidia_smi_scope': 'Whole GPU sampled memory including other applications, not instantaneous process peak',
            'pytorch_memory_scope': 'Process allocator peaks; allocated and reserved are nested, not additive',
            'free_before_load_mib': free_before/2**20, 'total_gpu_memory_mib': total/2**20,
            'gpu_name': torch.cuda.get_device_name(0), 'actual_parameter_devices': devices, **allocator,
            'model_config': model_config, 'model_weight_sha256': checked['weight_sha256'],
            'project_fine_tuning_applied': True, 'is_loaded_in_4bit': True, 'quantization': frozen['quantization'],
            'generation_config': effective.to_dict(), 'requested_config': frozen, 'eval_config': config,
            'generation_protocol_matches_baseline': True, 'packages': checked['packages'], 'peft_version': checked['peft_version'],
            'source_sha256': checked['source_sha256'], 'source_files_unchanged': True,
            'adapter_path': config['adapter_path'], 'adapter_sha256': checked['adapter_sha256'],
            'adapter_fingerprint': checked['adapter_fingerprint'], 'adapter_exact_before_and_after': True,
            'weight_sha256_unchanged_after_run': True, 'reference_answers_loaded_for_inference': False,
            'validation_reference_files_read': False, 'final_examples_read': False, 'databases_opened': False,
            'training_performed': False, 'complete_validation_records': True, 'complete_validation_generated': True,
            'run_dir': config['run_dir'], 'prediction_path': relative(directory/'predictions.jsonl'),
            'prediction_sha256': sha256(directory/'predictions.jsonl'),
            'validation_input_sha256': checked['input_sha256'][frozen['validation_inputs']]}
        write(directory/'summary.json', summary, True)
        write(ROOT/config['generation_report'], summary, True)
        write(directory/'status.json', {'status':'pass','completed':625,'attempted':625,'automatic_retries':0})
        print(json.dumps({k:summary[k] for k in ('status','prediction_count','generation_loop_wall_seconds','generation_peak_allocated_mib','generation_peak_reserved_mib')}),flush=True)
    except BaseException as exc:
        write(directory/'status.json', {'status':'failed','completed_records':len(predictions),
            'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(),'automatic_retries':0})
        raise
    finally:
        if model is not None:
            del model
        if sampler is not None:
            if sampler.thread and sampler.thread.is_alive():
                sampler.stage = 'after_failure'; sampler.close()
            write(directory/'gpu_samples.json', {'samples':sampler.samples,'errors':sampler.errors,'scope':'whole GPU'})


def run():
    """Native Windows process containment; one launch and a hard 65-minute bound."""
    config = read_json(CONFIG); directory = ROOT/config['run_dir']
    if directory.exists() or (ROOT/config['generation_report']).exists():
        raise FileExistsError('Run already claimed; no replay')
    checked = read_json(ROOT/config['preflight_report'])
    if checked['status'] != 'pass' or checked['eval_config'] != config:
        raise ValueError('Successful matching CPU preflight required')
    baseline.verify_hashes(checked['source_sha256'], 'Sources before startup gate')
    from src.train.sft_round1_resource_v3 import stable_resource_gate
    from src.train.windows_job_process_v2 import JobProcess, VENV_PYTHON
    from src.train.windows_job_exit_wait_formal import terminate_and_wait, close_without_new_wait
    from src.train.pid_supervision_formal_v2 import confirm_natural_exit
    readiness = stable_resource_gate(config)
    if readiness['status'] != 'ready':
        raise RuntimeError('Startup resources not ready; no worker launched')
    directory.mkdir(parents=True, exist_ok=False)
    write(directory/'claim.json', {'status':'claimed','automatic_retries':0,'worker_launches':1,
        'hard_seconds':config['worker_hard_seconds'],'training':False,'final_examples_read':False},True)
    write(directory/'start_resources.json', readiness, True)
    for name in (relative(CONFIG),relative(Path(__file__)),'src/infer/baseline_full.py','src/infer/baseline_pilot.py'):
        dest=directory/'snapshot'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,dest)
    started=time.perf_counter(); job=None; handles=[]; result={'status':'failed','events':[]}
    def record(event, **values):
        result['events'].append({'event':event,'monotonic':time.perf_counter(),**values})
    try:
        job=JobProcess([str(VENV_PYTHON),'-B','-X','utf8',str(Path(__file__).resolve()),'--worker'],ROOT,
            dict(os.environ,CUDA_VISIBLE_DEVICES='0',SFT_EVAL_CLAIM=str(directory.resolve()),
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),directory/'console.log')
        handles.append(job.open_process(job.pid))
        result.update(worker_pid=job.pid,identity=handles[0].identity(),job_settings=job.settings_evidence())
        code=job.wait(max(0,started+config['worker_hard_seconds']-time.perf_counter()))
        result.update(exit_code=code,exits=confirm_natural_exit(job,handles,record,time.perf_counter()+2,
            max_lifetime_seconds=3906),job_active_pids_after_cleanup=job.active_pids())
        if code != 0:
            raise RuntimeError(f'Worker exit={code}; no retry or evaluation')
        worker_id=read_json(directory/'worker_identity.json')
        if worker_id['pid'] != job.pid:
            raise ValueError('Native worker identity mismatch')
        result['status']='pass'
    except BaseException as exc:
        result.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        if job is not None and job.active_pids():
            retained={h.pid:h for h in handles}
            for pid in job.active_pids():
                if pid not in retained:
                    h=job.open_process(pid); handles.append(h); retained[pid]=h
            active=job.active_pids()
            result['termination']=terminate_and_wait(job,[retained[pid] for pid in active],record,
                timeout=2,max_lifetime_seconds=3906)
        raise
    finally:
        for handle in handles: handle.close()
        if job is not None: close_without_new_wait(job)
        result['controller_seconds']=time.perf_counter()-started
        write(directory/'controller_result.json',result,True)
    print(json.dumps({k:result[k] for k in ('status','exit_code','controller_seconds','job_active_pids_after_cleanup')}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true'); group.add_argument('--run',action='store_true'); group.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.check:
        config=read_json(CONFIG)
        report_path=ROOT/config['preflight_report']
        write(report_path,{'status':'checking','automatic_retries':0},True)
        try:
            config,*_,checked=preflight()
        except BaseException as error:
            write(report_path,{'status':'failed','error':f'{type(error).__name__}: {error}',
                'traceback':traceback.format_exc(),'automatic_retries':0})
            raise
        write(report_path,checked)
        print(json.dumps({k:checked[k] for k in ('status','examples','input_tokens_max','adapter_parameter_count','cuda_initialized','cpu_preflight_seconds')}),flush=True)
    elif args.run: run()
    else: worker()
