"""One-shot sealed Spider dev generation for base or frozen round-one adapter.

Only prepared question/schema messages enter inference.  Each arm owns an
exclusive directory and native Windows Job; no resume, selection, or retries.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src/infer'))
import baseline_full as baseline
from baseline_pilot import GPUSampler, extract_sql
from src.eval.final_holdout_common_v1 import (
    CONFIG, read_json, write, sha256, relative, verify_frozen, load_prepared, verify_prepared_binding,
)


def publish_once(path, payload):
    """Publish one completed handshake file; readers never see a partial JSON."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_suffix(path.suffix + '.pending')
    write(pending, payload, True)
    pending.replace(path)


def verify_preflight(config):
    checked = verify_prepared_binding()
    if checked['status'] != 'pass':
        raise ValueError('Successful CPU preflight required')
    if checked['config'] != config:
        raise ValueError('Preflight config differs from the frozen config')
    if len(checked['source_indices']) != config['expected_examples']:
        raise ValueError('Preflight count differs from fixed denominator')
    return checked


def worker(arm):
    started = time.perf_counter()
    config = read_json(CONFIG)
    directory = ROOT / config['run_root'] / arm
    if os.environ.get('FINAL_HOLDOUT_CLAIM') != str(directory.resolve()):
        raise RuntimeError('Worker requires its exclusive controller claim')
    predictions, model, sampler = [], None, None
    try:
        nonce = os.environ['FINAL_HOLDOUT_NONCE']
        publish_once(directory / 'worker_identity.json', {
            'pid': os.getpid(), 'executable': sys.executable, 'prefix': sys.prefix,
            'arm': arm, 'nonce': nonce, 'cuda_initialized': False,
        })
        identity_deadline = started + 20.0
        acknowledgment = directory / 'identity_accepted.json'
        while not acknowledgment.exists():
            if time.perf_counter() >= identity_deadline:
                raise TimeoutError('Controller did not bind the real worker before CUDA')
            time.sleep(0.02)
        bound = read_json(acknowledgment)
        if bound['pid'] != os.getpid() or bound['nonce'] != nonce or bound['arm'] != arm:
            raise ValueError('Controller identity acknowledgement mismatch')

        checked = verify_preflight(config)
        frozen = read_json(ROOT / config['baseline_config'])
        prior = read_json(ROOT / config['reference_generation_report'])
        model_config = checked['model_config']
        # Loading/tokenizing prepared inputs must remain CPU-only.
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA initialized before prepared-input verification')
        old_lazy_init = torch.cuda._lazy_init
        def reject_cuda(*args, **kwargs):
            raise RuntimeError('Prepared-input CPU path attempted CUDA initialization')
        torch.cuda._lazy_init = reject_cuda
        try:
            tokenizer, prepared = load_prepared()
        finally:
            torch.cuda._lazy_init = old_lazy_init
        if torch.cuda.is_initialized():
            raise RuntimeError('Prepared-input path initialized CUDA')
        if [row['source_index'] for row, _ in prepared] != checked['source_indices']:
            raise ValueError('Prepared-input order/count changed')
        with (directory / 'prompts.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for source, encoded in prepared:
                prompt = {'source_index': source['source_index'], 'db_id': source['db_id'],
                    'input_token_count': int(encoded['input_ids'].shape[1]),
                    'rendered_prompt': tokenizer.apply_chat_template(source['messages'],
                        tokenize=False, add_generation_prompt=True)}
                stream.write(json.dumps(prompt, ensure_ascii=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        if arm == 'sft':
            base_prompts = ROOT / config['run_root'] / 'base/prompts.jsonl'
            if sha256(base_prompts) != sha256(directory / 'prompts.jsonl'):
                raise ValueError('Base/SFT prompt bytes differ')
        write(directory / 'preflight.json', checked, True)

        import numpy as np
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        if arm == 'sft':
            from peft import PeftModel, get_peft_model_state_dict
            from src.train.sft_checkpoint_cpu_v1 import state_fingerprint
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA and BF16 required')
        random.seed(frozen['seed']); np.random.seed(frozen['seed'])
        torch.manual_seed(frozen['seed']); torch.cuda.manual_seed_all(frozen['seed'])
        torch.backends.cudnn.benchmark = False
        sampler = GPUSampler(frozen['gpu_sampling_interval_seconds']); sampler.start()
        if sampler.errors:
            raise RuntimeError('Initial GPU telemetry failed')
        free_before, total = torch.cuda.mem_get_info()
        if free_before / 2**20 < config['minimum_free_mib']:
            raise RuntimeError('Free GPU memory fell below the frozen startup threshold')
        torch.cuda.reset_peak_memory_stats(); sampler.stage = 'load'
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(ROOT / model_config['local_dir'],
            local_files_only=True, trust_remote_code=False, device_map={'': 'cuda:0'},
            dtype=torch.bfloat16, attn_implementation=frozen['attention_implementation'],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        if arm == 'sft':
            model = PeftModel.from_pretrained(model, ROOT / config['adapter_path'],
                is_trainable=False, local_files_only=True).eval()
            if state_fingerprint(get_peft_model_state_dict(model, adapter_name='default',
                    save_embedding_layers=False)) != checked['adapter_fingerprint']:
                raise ValueError('Loaded adapter differs from the frozen 796-step checkpoint')
            if model.active_adapters != ['default']:
                raise ValueError('Unexpected active adapter')
        model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
            raise ValueError('Inference parameters/mode must be frozen')
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        devices = sorted({str(parameter.device) for parameter in model.parameters()})
        if (load_seconds > config['load_budget_seconds'] or devices != ['cuda:0']
                or not model.is_loaded_in_4bit):
            raise RuntimeError('Model load budget/device/quantization check failed')
        load_allocated = torch.cuda.max_memory_allocated() / 2**20
        load_reserved = torch.cuda.max_memory_reserved() / 2**20
        if load_allocated > config['allocated_limit_mib'] or load_reserved > config['reserved_limit_mib']:
            raise RuntimeError('Load memory limit exceeded')

        settings = frozen['generation']
        generation_config = baseline.make_generation_config(settings, model.generation_config,
                                                            tokenizer.pad_token_id)
        effective_model = model.base_model.model if arm == 'sft' else model
        effective, _ = effective_model._prepare_generation_config(generation_config=generation_config)
        baseline.verify_effective_config(effective, prior)
        write(directory / 'generation_config.json', effective.to_dict(), True)
        write(directory / 'model_loaded.json', {
            'arm': arm, 'all_parameters_frozen': True, 'optimizer_created': False,
            'model_revision': model_config['revision'], 'training_mode': False,
            'adapter_fingerprint': checked['adapter_fingerprint'] if arm == 'sft' else None,
            'identity_accepted_before_cuda': True,
        }, True)
        expected = config['expected_examples']
        print(f'{arm}: loaded once in {load_seconds:.3f}s; generating {expected} complete inputs once.', flush=True)
        eos = effective.eos_token_id if isinstance(effective.eos_token_id, list) else [effective.eos_token_id]
        loop_started = time.perf_counter(); sampler.stage = 'generation'
        with (directory / 'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for ordinal, (source, encoded) in enumerate(prepared, 1):
                if time.perf_counter() - loop_started >= config['generation_loop_budget_seconds']:
                    raise TimeoutError('Frozen final-evaluation generation loop budget exhausted')
                if sampler.errors:
                    raise RuntimeError('GPU telemetry failed; stop without retry')
                inputs = output = None
                begun, error = time.perf_counter(), None
                try:
                    inputs = {key: value.to('cuda:0') for key, value in encoded.items()}
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
                    row = {'source_index': source['source_index'], 'db_id': source['db_id'],
                        'generation_ok': True, 'generation_status': 'generated',
                        'predicted_sql': extract_sql(text), 'generated_text': text,
                        'raw_output_with_special_tokens': tokenizer.decode(new_ids, skip_special_tokens=False),
                        'new_token_ids': new_ids, 'generated_token_count': len(new_ids),
                        'generation_seconds': seconds, 'ended_with_eos': ended,
                        'reached_token_limit': limited,
                        'stop_reason': 'eos' if ended else 'token_limit' if limited else 'time_or_other_stopping_criterion',
                        'effective_max_time_seconds': settings['max_time_seconds'],
                        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
                    if not ended and not limited:
                        error = TimeoutError('Unexpected/time stopping criterion; retain result and stop')
                    if row['peak_allocated_mib'] > config['allocated_limit_mib'] or row['peak_reserved_mib'] > config['reserved_limit_mib']:
                        error = RuntimeError('Generation memory threshold exceeded')
                except Exception as exc:
                    error = exc
                    row = baseline.record_failure(source, f'{type(exc).__name__}: {exc}',
                                                  seconds=time.perf_counter() - begun)
                finally:
                    del output, inputs
                row.update(ordinal=ordinal, input_token_count=int(encoded['input_ids'].shape[1]), attempt_number=1)
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush(); os.fsync(stream.fileno())
                if ordinal == 1 or ordinal % 25 == 0 or ordinal == expected:
                    print(f'[{ordinal}/{expected}] id={row["source_index"]} tokens={row["generated_token_count"]} loop={time.perf_counter()-loop_started:.2f}s', flush=True)
                if error is not None:
                    raise error
        loop_seconds = time.perf_counter() - loop_started
        if [row['source_index'] for row in predictions] != checked['source_indices'] or len(predictions) != expected:
            raise ValueError('Incomplete or repeated final predictions')
        if arm == 'sft' and state_fingerprint(get_peft_model_state_dict(model, adapter_name='default',
                save_embedding_layers=False)) != checked['adapter_fingerprint']:
            raise ValueError('Adapter changed during inference')
        allocator = {key: int(torch.cuda.memory_stats().get(key, 0))
                     for key in ('num_ooms', 'num_alloc_retries')}
        gpu_name = torch.cuda.get_device_name(0)
        del model; model = None; torch.cuda.empty_cache()
        sampler.stage = 'after_release'; sampler.close()
        if sampler.errors:
            raise RuntimeError('GPU telemetry errors at completion')
        verify_frozen()
        if sha256(ROOT / checked['weight_path']) != checked['weight_sha256']:
            raise ValueError('Original base weight changed')
        summary = {
            'status': 'pass', 'arm': arm, 'prediction_count': expected, 'generated_count': expected,
            'attempted_count': expected, 'automatic_retries': 0, 'warmup_generations': 0,
            'generation_failure_or_skip_count': 0, 'prediction_selection_performed': False,
            'ended_with_eos_count': sum(row['ended_with_eos'] for row in predictions),
            'token_limit_count': sum(row['reached_token_limit'] for row in predictions),
            'generated_tokens_total': sum(row['generated_token_count'] for row in predictions),
            'generation_seconds_total': sum(row['generation_seconds'] for row in predictions),
            'generation_seconds_median': statistics.median(row['generation_seconds'] for row in predictions),
            'generation_loop_wall_seconds': loop_seconds, 'load_seconds': load_seconds,
            'end_to_end_seconds': time.perf_counter() - started,
            'load_peak_allocated_mib': load_allocated, 'load_peak_reserved_mib': load_reserved,
            'generation_peak_allocated_mib': max(row['peak_allocated_mib'] for row in predictions),
            'generation_peak_reserved_mib': max(row['peak_reserved_mib'] for row in predictions),
            'nvidia_smi_sampled_peak_used_mib': max(row['used_mib'] for row in sampler.samples),
            'nvidia_smi_sample_count': len(sampler.samples), 'nvidia_smi_sampling_error_count': 0,
            'nvidia_smi_scope': 'Whole GPU sampled memory including other applications; not a process instantaneous peak',
            'pytorch_memory_scope': 'Process allocator peaks; allocated and reserved are nested, not additive',
            'free_before_load_mib': free_before / 2**20, 'total_gpu_memory_mib': total / 2**20,
            'gpu_name': gpu_name, 'actual_parameter_devices': devices, **allocator,
            'model_config': model_config, 'model_weight_sha256': checked['weight_sha256'],
            'adapter_path': config['adapter_path'] if arm == 'sft' else None,
            'adapter_fingerprint': checked['adapter_fingerprint'] if arm == 'sft' else None,
            'project_fine_tuning_applied': arm == 'sft', 'is_loaded_in_4bit': True,
            'quantization': frozen['quantization'], 'generation_config': effective.to_dict(),
            'generation_protocol_matches_internal_validation': True, 'packages': checked['packages'],
            'reference_answers_loaded_for_inference': False, 'databases_opened': False,
            'final_examples_read': True, 'training_performed': False, 'all_inputs_complete': True,
            'input_tokens_max': checked['input_tokens_max'], 'source_indices': checked['source_indices'],
            'run_dir': relative(directory), 'prediction_path': relative(directory / 'predictions.jsonl'),
            'prediction_sha256': sha256(directory / 'predictions.jsonl'),
            'prompts_sha256': sha256(directory / 'prompts.jsonl'),
        }
        write(directory / 'summary.json', summary, True)
        write(directory / 'status.json', {'status': 'pass', 'completed': expected,
            'attempted': expected, 'automatic_retries': 0})
        print(json.dumps({key: summary[key] for key in ('status', 'arm', 'prediction_count',
            'generation_loop_wall_seconds', 'generation_peak_allocated_mib', 'generation_peak_reserved_mib')}), flush=True)
    except BaseException as exc:
        write(directory / 'status.json', {'status': 'failed', 'completed_records': len(predictions),
            'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc(), 'automatic_retries': 0})
        raise
    finally:
        if model is not None:
            del model
        if sampler is not None:
            if sampler.thread and sampler.thread.is_alive():
                sampler.stage = 'after_failure'; sampler.close()
            write(directory / 'gpu_samples.json', {
                'samples': sampler.samples, 'errors': sampler.errors, 'scope': 'whole GPU'}, True)


def run(arm):
    config = read_json(CONFIG)
    directory = ROOT / config['run_root'] / arm
    if directory.exists():
        raise FileExistsError('Final evaluation arm already claimed; no replay')
    verify_preflight(config)
    if arm == 'sft':
        base_dir = ROOT / config['run_root'] / 'base'
        if (read_json(base_dir / 'summary.json')['status'] != 'pass'
                or read_json(base_dir / 'controller_result.json')['status'] != 'pass'):
            raise RuntimeError('Base arm did not finish successfully; stop the entire comparison')
    from src.train.sft_round1_resource_v3 import stable_resource_gate
    from src.train.windows_job_process_v2 import JobProcess, VENV_PYTHON, NATIVE_PYTHON, normalize
    from src.train.windows_job_exit_wait_formal import terminate_and_wait, close_without_new_wait
    from src.train.pid_supervision_formal_v2 import confirm_natural_exit
    readiness_config = dict(config, resource_readiness_report=f'reports/final_holdout_v1_{arm}_start_resources.json')
    readiness = stable_resource_gate(readiness_config)
    if readiness['status'] != 'ready':
        raise RuntimeError('Startup resources not ready; no worker launched and no automatic retry')
    directory.mkdir(parents=True, exist_ok=False)
    nonce = secrets.token_hex(24)
    write(directory / 'claim.json', {'status': 'claimed', 'arm': arm, 'automatic_retries': 0,
        'worker_launches': 1, 'hard_seconds': config['worker_hard_seconds'],
        'training': False, 'nonce': nonce}, True)
    write(directory / 'start_resources.json', readiness, True)
    for name in (relative(CONFIG), relative(Path(__file__)), 'src/eval/final_holdout_common_v1.py'):
        destination = directory / 'snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    started, job, handles = time.perf_counter(), None, []
    result = {'status': 'failed', 'arm': arm, 'events': [], 'automatic_retries': 0}
    def record(event, **values):
        result['events'].append({'event': event, 'monotonic': time.perf_counter(), **values})
    try:
        job = JobProcess([str(VENV_PYTHON), '-B', '-X', 'utf8', str(Path(__file__).resolve()),
            '--arm', arm, '--worker'], ROOT, dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                FINAL_HOLDOUT_CLAIM=str(directory.resolve()), FINAL_HOLDOUT_NONCE=nonce,
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1'),
            directory / 'console.log')
        handles.append(job.open_process(job.pid))
        result.update(worker_pid=job.pid, identity=handles[0].identity(), job_settings=job.settings_evidence())
        deadline = min(started + config['worker_hard_seconds'], time.perf_counter() + 20.0)
        identity_path = directory / 'worker_identity.json'
        while not identity_path.exists():
            if not handles[0].alive():
                raise RuntimeError('Worker exited before its identity handshake')
            if time.perf_counter() >= deadline:
                raise TimeoutError('Worker identity handshake exceeded 20 seconds')
            time.sleep(0.02)
        identity = read_json(identity_path)
        if (identity['pid'] != job.pid or identity['nonce'] != nonce or identity['arm'] != arm
                or normalize(identity['executable']) != normalize(VENV_PYTHON)
                or normalize(identity['prefix']) != normalize(ROOT / '.venv')
                or normalize(result['identity']['executable']) != normalize(NATIVE_PYTHON)
                or not job.contains(job.pid) or not handles[0].alive()):
            raise ValueError('Real worker PID/interpreter/Job/nonce binding mismatch')
        record('real_worker_bound_before_cuda', pid=job.pid, worker_identity=identity)
        publish_once(directory / 'identity_accepted.json', {'pid': job.pid, 'nonce': nonce, 'arm': arm})
        code = job.wait(max(0, started + config['worker_hard_seconds'] - time.perf_counter()))
        result.update(exit_code=code, exits=confirm_natural_exit(job, handles, record, time.perf_counter() + 2,
            max_lifetime_seconds=config['worker_hard_seconds'] + 6), job_active_pids_after_cleanup=job.active_pids())
        if code != 0:
            raise RuntimeError(f'Worker exit={code}; no retry or evaluation')
        if read_json(directory / 'summary.json')['status'] != 'pass':
            raise RuntimeError('Worker summary did not pass')
        result['status'] = 'pass'
    except BaseException as exc:
        result.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        if job is not None and job.active_pids():
            retained = {handle.pid: handle for handle in handles}
            for pid in job.active_pids():
                if pid not in retained:
                    handle = job.open_process(pid); handles.append(handle); retained[pid] = handle
            result['termination'] = terminate_and_wait(job,
                [retained[pid] for pid in job.active_pids()], record,
                timeout=2, max_lifetime_seconds=config['worker_hard_seconds'] + 6)
        raise
    finally:
        for handle in handles:
            handle.close()
        if job is not None:
            close_without_new_wait(job)
        result['controller_seconds'] = time.perf_counter() - started
        write(directory / 'controller_result.json', result, True)
    print(json.dumps({key: result[key] for key in ('status', 'arm', 'exit_code',
        'controller_seconds', 'job_active_pids_after_cleanup')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['base', 'sft'], required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker', action='store_true')
    arguments = parser.parse_args()
    if arguments.run:
        run(arguments.arm)
    else:
        worker(arguments.arm)
