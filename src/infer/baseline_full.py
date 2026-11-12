"""Generate the complete internal-validation baseline, with no reference inputs.

--check hashes local evidence and tokenizes 625 inputs on CPU, without loading
model weights. Normal execution freshly generates every row once in a new run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import statistics
import time
from types import SimpleNamespace

from baseline_pilot import (
    GPUSampler, ROOT, extract_sql, failed_prediction, project_path,
    read_json, relative, save_json, sha256,
)

CONFIG = ROOT / "configs/baseline_full.json"
PACKAGE_NAMES = ("torch", "transformers", "accelerate", "bitsandbytes", "tokenizers")
GENERATION_KEYS = ("do_sample", "num_beams", "num_return_sequences", "max_new_tokens",
                   "repetition_penalty", "use_cache", "temperature", "top_p", "top_k")


def verify_hashes(fingerprints, label):
    for filename, expected in fingerprints.items():
        if sha256(project_path(filename)) != expected:
            raise ValueError(f"{label} changed: {filename}")


def validate_full_inputs(config, manifest, rows):
    if config["expected_examples"] != 625 or manifest["validation_examples"] != 625 or len(rows) != 625:
        raise ValueError("Full baseline requires exactly 625 internal-validation inputs")
    required = {"source_index", "source_file", "db_id", "project_split", "messages"}
    indices = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != required or row["project_split"] != "validation":
            raise ValueError("Unexpected generation fields or non-validation split")
        index = row["source_index"]
        if type(index) is not int or index < 0:
            raise ValueError("source_index must be a nonnegative integer")
        if row["source_file"] != "data/raw/spider/train_spider.json" or not isinstance(row["db_id"], str):
            raise ValueError("Unexpected source or database identifier")
        messages = row["messages"]
        if (not isinstance(messages, list) or len(messages) != 2
                or not all(isinstance(message, dict) for message in messages)
                or [message.get("role") for message in messages] != ["system", "user"]):
            raise ValueError("Inputs must have system/user messages only, never reference answers")
        if any(set(message) != {"role", "content"} or not isinstance(message["content"], str)
               or not message["content"].strip() for message in messages):
            raise ValueError("Malformed or unexpected message fields")
        indices.append(index)
    if len(set(indices)) != 625 or indices != sorted(indices):
        raise ValueError("Source indices must be unique and in the frozen ascending order")
    db_ids = sorted({row["db_id"] for row in rows})
    if config["expected_database_count"] != 14 or len(db_ids) != 14 or db_ids != config["expected_database_ids"]:
        raise ValueError("Internal-validation database membership changed")
    return indices, db_ids


def make_generation_config(settings, defaults, pad_token_id):
    from transformers import GenerationConfig
    return GenerationConfig(**{key: settings[key] for key in GENERATION_KEYS},
        max_time=settings["max_time_seconds"], eos_token_id=defaults.eos_token_id,
        pad_token_id=pad_token_id, bos_token_id=defaults.bos_token_id)


def verify_effective_config(effective, pilot):
    actual = effective.to_dict()
    if actual != pilot["generation_config"]:
        differences = sorted(key for key in set(actual) | set(pilot["generation_config"])
                             if actual.get(key) != pilot["generation_config"].get(key))
        raise ValueError(f"Resolved generation settings differ from pilot: {differences}")


def preflight():
    import yaml
    from transformers import AutoTokenizer, GenerationConfig
    from transformers.generation.utils import GenerationMixin

    started = time.perf_counter()
    config = read_json(CONFIG)
    verify_hashes(config["protected_pilot_sha256"], "Protected pilot evidence")
    pilot_path = project_path(config["pilot_generation_report"])
    pilot = read_json(pilot_path)
    protocol_keys = ("seed", "batch_size", "attention_implementation", "quantization", "generation",
                     "max_input_tokens", "gpu_sampling_interval_seconds", "retries_per_example",
                     "warmup_generations", "sql_extraction", "model_config")
    for key in protocol_keys:
        if config[key] != pilot["requested_config"][key]:
            raise ValueError(f"Full baseline changed the pilot inference protocol: {key}")
    if config["batch_size"] != 1 or config["retries_per_example"] != 0 or config["warmup_generations"] != 0:
        raise ValueError("Expected batch size one, zero retries, zero warmup generations")
    if config["generation_loop_budget_seconds"] != 3600.0 or config["generation"]["max_time_seconds"] != 30.0:
        raise ValueError("Expected 3600-second loop guard and unchanged 30-second per-query budget")
    packages = {name: version(name) for name in PACKAGE_NAMES}
    if packages != pilot["packages"]:
        raise ValueError("Runtime package versions differ from the pilot")

    manifest_path = project_path(config["input_manifest"])
    if sha256(manifest_path) != pilot["source_sha256"][relative(manifest_path)]:
        raise ValueError("Prepared input manifest changed since pilot")
    manifest = read_json(manifest_path)
    input_path = project_path(config["validation_inputs"])
    if sha256(input_path) != manifest["output_sha256"][input_path.name]:
        raise ValueError("Validation input SHA256 does not match prepared manifest")
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    indices, db_ids = validate_full_inputs(config, manifest, rows)
    model_path = project_path(config["model_config"])
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    model_manifest_path = ROOT / "reports/model_manifest.json"
    model_manifest = read_json(model_manifest_path)
    if model_config != pilot["model_config"] or any(model_config[key] != model_manifest[key]
                                                   for key in ("model_id", "revision", "local_dir")):
        raise ValueError("Model identity/revision does not match original pilot checkpoint")
    if model_config["revision"] != manifest["model_revision"] or model_config["model_id"] != manifest["model_id"]:
        raise ValueError("Prepared inputs target a different model")
    model_dir = project_path(model_config["local_dir"])
    source_files = [CONFIG, Path(__file__), ROOT / "src/infer/baseline_pilot.py", pilot_path,
                    model_path, model_manifest_path, manifest_path, input_path,
                    ROOT / "configs/prompt.yaml", ROOT / "requirements.lock.txt"]
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "generation_config.json"):
        path = model_dir / name
        data = path.read_bytes()
        recorded = model_manifest["verified_files"][name]
        blob_hash = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
        if recorded["algorithm"] != "sha1" or blob_hash != recorded["hash"] or len(data) != recorded["size"]:
            raise ValueError(f"Model/tokenizer metadata hash mismatch: {name}")
        source_files.append(path)
    for path in source_files:
        name = relative(path)
        if name in pilot["source_sha256"] and sha256(path) != pilot["source_sha256"][name]:
            raise ValueError(f"Model/prompt/dependency/helper source differs from pilot: {name}")
        if name in manifest["input_sha256"] and sha256(path) != manifest["input_sha256"][name]:
            raise ValueError(f"Prepared input dependency changed: {name}")
    weights = [entry for entry in model_manifest["files"] if entry["is_weight"]]
    if len(weights) != 1:
        raise ValueError("Expected the original single safetensors weight file")
    weight = model_dir / weights[0]["name"]
    weight_hash = sha256(weight)
    if weight_hash != weights[0]["sha256"] or weight_hash != pilot["model_weight_sha256"] or weight.stat().st_size != weights[0]["size"]:
        raise ValueError("Original model weight SHA256 mismatch")
    weight_stat = {"size": weight.stat().st_size, "mtime_ns": weight.stat().st_mtime_ns}

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Local tokenizer is missing its chat template")
    context_limit = read_json(model_dir / "config.json")["max_position_embeddings"]
    prepared = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                                return_tensors="pt", return_dict=True)
        count = encoded["input_ids"].shape[1]
        if count > config["max_input_tokens"] or count + config["generation"]["max_new_tokens"] > context_limit:
            raise ValueError("Input exceeds frozen context budget; never truncate")
        prepared.append((row, encoded))
    defaults = GenerationConfig.from_pretrained(model_dir, local_files_only=True)
    request = make_generation_config(config["generation"], defaults, tokenizer.pad_token_id)
    effective, _ = GenerationMixin._prepare_generation_config(SimpleNamespace(generation_config=defaults), generation_config=request)
    verify_effective_config(effective, pilot)
    tokens = [int(encoded["input_ids"].shape[1]) for _, encoded in prepared]
    fingerprints = {relative(path): sha256(path) for path in source_files}
    verify_hashes(config["protected_pilot_sha256"], "Protected pilot evidence after preflight")
    report = {
        "status": "ready", "examples": 625, "database_count": len(db_ids), "database_ids": db_ids,
        "source_indices": indices, "input_tokens_min": min(tokens), "input_tokens_max": max(tokens),
        "input_tokens_total": sum(tokens), "weight_path": relative(weight), "weight_sha256": weight_hash,
        "weight_stat": weight_stat, "model_id": model_config["model_id"], "revision": model_config["revision"],
        "model_context_limit": context_limit, "weights_loaded": False, "packages": packages,
        "generation_config": effective.to_dict(), "generation_protocol_matches_pilot": True,
        "input_sha256": fingerprints, "protected_pilot_sha256": config["protected_pilot_sha256"],
        "protected_pilot_files_unchanged": True, "reference_answers_loaded_for_inference": False,
        "pilot_evidence_hash_policy": "Existing pilot evidence is hashed as opaque bytes only; prior evaluation artifacts may contain pilot gold SQL. No such evidence is parsed as inference data.",
        "validation_reference_files_read": False, "final_examples_read": False, "databases_opened": False,
        "preflight_seconds_excluding_initial_library_import": round(time.perf_counter() - started, 3),
    }
    return config, pilot, model_config, tokenizer, prepared, report


def write_progress(run_dir, run_id, predictions, expected, loop_started, status="generating"):
    generated = sum(row["generation_ok"] for row in predictions)
    progress = {"run_id": run_id, "status": status, "completed_records": len(predictions),
                "expected_examples": expected, "generated_count": generated,
                "failure_or_skip_count": len(predictions) - generated,
                "last_source_index": predictions[-1]["source_index"] if predictions else None,
                "loop_elapsed_seconds": time.perf_counter() - loop_started,
                "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    save_json(run_dir / "progress.tmp.json", progress)
    (run_dir / "progress.tmp.json").replace(run_dir / "progress.json")


def record_failure(source, reason, status="generation_error", seconds=0.0):
    row = failed_prediction(source, reason, status)
    row.update(stop_reason=status, effective_max_time_seconds=30.0,
               peak_allocated_mib=None, peak_reserved_mib=None, generation_seconds=seconds)
    return row


def main(check_only):
    task_started = time.perf_counter()
    config, pilot, model_config, tokenizer, prepared, checked = preflight()
    save_json(ROOT / "reports/baseline_full_preflight.json", checked)
    print(f"Full preflight: 625 inputs / 14 databases; tokens {checked['input_tokens_min']}..{checked['input_tokens_max']}; original weight SHA256 verified", flush=True)
    if check_only:
        print("FULL_PREFLIGHT: PASS; CPU tokenizer checks only; model weights not loaded; no generation", flush=True)
        return

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen configuration requires CUDA and bf16")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_baseline_full"
    run_dir = ROOT / "outputs/preds" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in (relative(CONFIG), relative(Path(__file__)), "src/infer/baseline_pilot.py", config["input_manifest"],
                 config["validation_inputs"], config["model_config"], "configs/prompt.yaml", "requirements.lock.txt",
                 config["pilot_generation_report"]):
        destination = run_dir / "snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project_path(name), destination)
    save_json(run_dir / "preflight.json", checked)
    save_json(run_dir / "status.json", {"run_id": run_id, "status": "started", "expected_examples": 625})
    with (run_dir / "prompts.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for source, encoded in prepared:
            stream.write(json.dumps({"source_index": source["source_index"], "db_id": source["db_id"],
                "input_token_count": int(encoded["input_ids"].shape[1]),
                "rendered_prompt": tokenizer.apply_chat_template(source["messages"], tokenize=False, add_generation_prompt=True)}, ensure_ascii=False) + "\n")
    sampler = GPUSampler(config["gpu_sampling_interval_seconds"])
    sampler.start()
    predictions, cleanup_errors = [], []
    model = None
    try:
        sampler.stage = "load"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info()
        gpu_name = torch.cuda.get_device_name(0)
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(project_path(model_config["local_dir"]), local_files_only=True,
            trust_remote_code=False, device_map={"": "cuda:0"}, dtype=torch.bfloat16,
            attn_implementation=config["attention_implementation"],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        load_allocated = torch.cuda.max_memory_allocated() / 1024**2
        load_reserved = torch.cuda.max_memory_reserved() / 1024**2
        devices = sorted({str(parameter.device) for parameter in model.parameters()})
        if not getattr(model, "is_loaded_in_4bit", False) or devices != ["cuda:0"] or getattr(model, "peft_config", None):
            raise RuntimeError("Expected original NF4 model entirely on cuda:0, without a fine-tuning adapter")
        settings = config["generation"]
        generation_config = make_generation_config(settings, model.generation_config, tokenizer.pad_token_id)
        effective, _ = model._prepare_generation_config(generation_config=generation_config)
        verify_effective_config(effective, pilot)
        save_json(run_dir / "generation_config.json", effective.to_dict())
        eos_ids = effective.eos_token_id if isinstance(effective.eos_token_id, list) else [effective.eos_token_id]
        print(f"Model loaded once in {load_seconds:.3f}s. Starting 625 fresh generations with unchanged pilot settings.", flush=True)
        loop_started = time.perf_counter()
        write_progress(run_dir, run_id, predictions, 625, loop_started)
        blocked_error = None
        sampler.stage = "generation"
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for ordinal, (source, cpu_inputs) in enumerate(prepared, 1):
                if blocked_error is not None:
                    row = record_failure(source, blocked_error, "skipped_after_cuda_error")
                elif time.perf_counter() - loop_started >= config["generation_loop_budget_seconds"]:
                    row = record_failure(source, "Generation loop budget exhausted before starting this query", "skipped_budget")
                else:
                    inputs = output = None
                    begun = time.perf_counter()
                    try:
                        inputs = {key: value.to("cuda:0") for key, value in cpu_inputs.items()}
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        per_query_config = deepcopy(generation_config)
                        per_query_config.max_time = settings["max_time_seconds"]
                        begun = time.perf_counter()
                        with torch.inference_mode():
                            output = model.generate(**inputs, generation_config=per_query_config)
                        torch.cuda.synchronize()
                        seconds = time.perf_counter() - begun
                        new_ids = output[0, cpu_inputs["input_ids"].shape[1]:].tolist()
                        text = tokenizer.decode(new_ids, skip_special_tokens=True)
                        ended = bool(new_ids and new_ids[-1] in eos_ids)
                        reached = len(new_ids) >= settings["max_new_tokens"] and not ended
                        row = {"source_index": source["source_index"], "db_id": source["db_id"],
                            "generation_ok": True, "generation_status": "generated", "predicted_sql": extract_sql(text),
                            "generated_text": text, "raw_output_with_special_tokens": tokenizer.decode(new_ids, skip_special_tokens=False),
                            "new_token_ids": new_ids, "generated_token_count": len(new_ids), "generation_seconds": seconds,
                            "ended_with_eos": ended, "reached_token_limit": reached,
                            "stop_reason": "eos" if ended else "token_limit" if reached else "time_or_other_stopping_criterion",
                            "effective_max_time_seconds": settings["max_time_seconds"],
                            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2}
                    except Exception as exc:
                        row = record_failure(source, f"{type(exc).__name__}: {exc}", seconds=time.perf_counter() - begun)
                        if isinstance(exc, RuntimeError):
                            blocked_error = "Stopped further GPU work conservatively after generation RuntimeError: " + str(exc)
                    finally:
                        del output, inputs
                row["ordinal"] = ordinal
                row["input_token_count"] = int(cpu_inputs["input_ids"].shape[1])
                row["attempt_number"] = 1 if row["generation_status"] in {"generated", "generation_error"} else 0
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                write_progress(run_dir, run_id, predictions, 625, loop_started)
                if ordinal % config["progress_stdout_every"] == 0 or ordinal == 1 or ordinal == 625:
                    print(f"[{ordinal:03d}/625] generated={sum(r['generation_ok'] for r in predictions)} last_id={row['source_index']} status={row['generation_status']} loop_seconds={time.perf_counter()-loop_started:.3f}", flush=True)
        loop_seconds = time.perf_counter() - loop_started
        if [row["source_index"] for row in predictions] != checked["source_indices"]:
            raise ValueError("Missing, duplicated or reordered full-validation records")
        sampler.stage = "after_generation"
        sampler.sample()
    except BaseException as exc:
        save_json(run_dir / "status.json", {"run_id": run_id, "status": "failed",
            "prediction_count": len(predictions), "error": f"{type(exc).__name__}: {exc}",
            "note": "Infrastructure/interruption abort is not a completed 625-example benchmark."})
        raise
    finally:
        if model is not None:
            del model
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        sampler.stage = "after_release"
        sampler.close()
        save_json(run_dir / "gpu_samples.json", {"samples": sampler.samples, "errors": sampler.errors,
            "interval_seconds": sampler.interval, "scope": "Sampled whole-GPU memory, including other applications"})

    try:
        verify_hashes(checked["input_sha256"], "Full baseline input/source after generation")
        verify_hashes(checked["protected_pilot_sha256"], "Protected pilot evidence after generation")
        weight = project_path(checked["weight_path"])
        if {"size": weight.stat().st_size, "mtime_ns": weight.stat().st_mtime_ns} != checked["weight_stat"]:
            raise ValueError("Model weight metadata changed during generation")
        if sha256(weight) != checked["weight_sha256"]:
            raise ValueError("Model weight SHA256 changed during generation")
    except Exception as exc:
        save_json(run_dir / "status.json", {"run_id": run_id, "status": "integrity_failed", "error": str(exc), "prediction_count": len(predictions)})
        raise
    generated = [row for row in predictions if row["generation_ok"]]
    memory_rows = [row for row in predictions if row["peak_allocated_mib"] is not None]
    attempted = [row for row in predictions if row["attempt_number"] == 1]
    summary = {
        "run_id": run_id, "scope": config["scope"], "prediction_count": len(predictions),
        "generated_count": len(generated), "generation_failure_or_skip_count": 625 - len(generated),
        "attempted_count": len(attempted), "pilot_predictions_reused": False, "prediction_selection_performed": False,
        "ended_with_eos_count": sum(row["ended_with_eos"] for row in predictions),
        "token_limit_count": sum(row["reached_token_limit"] for row in predictions),
        "generated_tokens_total": sum(row["generated_token_count"] for row in predictions),
        "generation_seconds_total": sum(row["generation_seconds"] for row in predictions),
        "generation_seconds_median": statistics.median(row["generation_seconds"] for row in attempted) if attempted else None,
        "generation_loop_wall_seconds": loop_seconds, "load_seconds": load_seconds,
        "end_to_end_seconds": time.perf_counter() - task_started,
        "first_example_seconds": predictions[0]["generation_seconds"], "warmup_generations": 0,
        "load_peak_allocated_mib": load_allocated, "load_peak_reserved_mib": load_reserved,
        "generation_peak_allocated_mib": max((row["peak_allocated_mib"] for row in memory_rows), default=None),
        "generation_peak_reserved_mib": max((row["peak_reserved_mib"] for row in memory_rows), default=None),
        "nvidia_smi_sampled_peak_used_mib": max((row["used_mib"] for row in sampler.samples), default=None),
        "nvidia_smi_before_load_used_mib": sampler.samples[0]["used_mib"] if sampler.samples else None,
        "nvidia_smi_sample_count": len(sampler.samples), "nvidia_smi_sampling_error_count": len(sampler.errors),
        "nvidia_smi_scope": "Whole GPU, sampled, includes other processes; not a guaranteed instantaneous peak",
        "pytorch_memory_scope": "This process PyTorch CUDA allocator peaks including loaded model; allocated/reserved are different nested metrics, not additive",
        "gpu_name": gpu_name, "total_gpu_memory_mib": total / 1024**2,
        "free_before_load_mib": free_before / 1024**2, "actual_parameter_devices": devices,
        "model_config": model_config, "model_weight_sha256": checked["weight_sha256"],
        "project_fine_tuning_applied": False, "is_loaded_in_4bit": True, "quantization": config["quantization"],
        "generation_config": effective.to_dict(), "generation_protocol_matches_pilot": True,
        "requested_config": config, "packages": checked["packages"], "source_sha256": checked["input_sha256"],
        "source_files_unchanged": True, "protected_pilot_sha256": checked["protected_pilot_sha256"],
        "protected_pilot_files_unchanged": True, "weight_sha256_unchanged_after_run": True,
        "weight_metadata_unchanged_after_run": True, "reference_answers_loaded_for_inference": False,
        "pilot_evidence_hash_policy": checked["pilot_evidence_hash_policy"],
        "validation_reference_files_read": False, "final_examples_read": False, "databases_opened": False,
        "complete_validation_records": True, "complete_validation_generated": len(generated) == 625,
        "denominator_policy": "All 625 frozen inputs, including generation errors, skips, invalid SQL and later SQL execution errors",
        "cleanup_errors": cleanup_errors, "run_dir": relative(run_dir),
        "prediction_path": relative(run_dir / "predictions.jsonl"),
        "prediction_sha256": sha256(run_dir / "predictions.jsonl"),
        "validation_input_sha256": checked["input_sha256"][config["validation_inputs"]],
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "status.json", {"run_id": run_id, "status": "completed", "prediction_count": 625,
        "generated_count": len(generated), "prediction_sha256": summary["prediction_sha256"]})
    write_progress(run_dir, run_id, predictions, 625, loop_started, "completed")
    save_json(ROOT / "reports/baseline_full_generation.json", summary)
    print(json.dumps({key: summary[key] for key in ("generated_count", "generation_failure_or_skip_count", "token_limit_count",
        "generated_tokens_total", "generation_seconds_total", "load_seconds", "end_to_end_seconds",
        "generation_peak_allocated_mib", "generation_peak_reserved_mib", "nvidia_smi_sampled_peak_used_mib", "prediction_path")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="CPU-only input/tokenizer/hash check, without loading model weights")
    main(parser.parse_args().check)
