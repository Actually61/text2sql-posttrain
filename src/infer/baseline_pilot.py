"""Generate once for each frozen pilot input, without reading reference answers.

--check validates local inputs, tokenizer, model hashes and token limits on CPU.
Without --check, loads the original instruct model in NF4 and generates 32 SQLs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import threading
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/baseline_pilot.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Path escaped project")
    return path


def extract_sql(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:sql)?[ \t]*\r?\n(.*?)```", text, flags=re.S | re.I)
    return fenced.group(1).strip() if fenced else text


def failed_prediction(source, reason, status="generation_error"):
    return {"source_index": source["source_index"], "db_id": source["db_id"],
            "predicted_sql": "", "generation_ok": False, "generation_status": status,
            "generation_error": reason, "generated_token_count": 0,
            "new_token_ids": [], "generated_text": "", "raw_output_with_special_tokens": "",
            "ended_with_eos": False, "reached_token_limit": False, "generation_seconds": 0.0}


def validate_pilot_inputs(manifest, rows):
    if manifest["example_count"] != 32 or manifest["maximum_generation_examples"] != 32 or len(rows) != 32:
        raise ValueError("This generator is restricted to exactly 32 frozen pilot inputs")
    expected = manifest["selected_source_indices"]
    if len(set(expected)) != 32 or [r.get("source_index") for r in rows] != expected:
        raise ValueError("Frozen pilot source indices/order changed")
    selected = {r["source_index"]: r for r in manifest["selected_examples"]}
    permitted = {"source_index", "source_file", "db_id", "project_split", "messages"}
    for row in rows:
        if set(row) - permitted or row.get("project_split") != "validation":
            raise ValueError("Generation row contains unexpected fields or wrong split")
        if row["db_id"] != selected[row["source_index"]]["db_id"]:
            raise ValueError("Pilot database association changed")
        messages = row.get("messages")
        if (not isinstance(messages, list) or len(messages) != 2
                or not all(isinstance(m, dict) for m in messages)
                or [m.get("role") for m in messages] != ["system", "user"]):
            raise ValueError("Generation input must contain only system and user messages")
        if any(set(m) != {"role", "content"} or not isinstance(m["content"], str) or not m["content"] for m in messages):
            raise ValueError("Malformed input messages")


def preflight():
    import yaml
    from transformers import AutoTokenizer
    started = time.perf_counter()
    config = read_json(CONFIG)
    if config["expected_examples"] != 32 or config["batch_size"] != 1 or config["retries_per_example"] != 0:
        raise ValueError("Pilot count/batch/retry protocol changed")
    manifest_path = project_path(config["pilot_manifest"])
    pilot = read_json(manifest_path)
    if manifest_path.read_bytes() != (ROOT / "reports/baseline_pilot_manifest.json").read_bytes():
        raise ValueError("Pilot manifest differs from its published copy")
    input_path = project_path(config["pilot_inputs"])
    if sha256(input_path) != pilot["output_sha256"][input_path.name]:
        raise ValueError("Pilot input bytes changed")
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    validate_pilot_inputs(pilot, rows)
    model_path = project_path(config["model_config"])
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    model_manifest_path = ROOT / "reports/model_manifest.json"
    model_manifest = read_json(model_manifest_path)
    if any(model_config[k] != model_manifest[k] for k in ("model_id", "revision", "local_dir")):
        raise ValueError("Model identity/revision mismatch")
    model_dir = project_path(model_config["local_dir"])
    # Only metadata is read here, not validation examples, reference SQL or SQLite.
    input_manifest_path = ROOT / "data/processed/spider_inputs_v1/manifest.json"
    if sha256(input_manifest_path) != pilot["input_sha256"][relative(input_manifest_path)]:
        raise ValueError("Prepared input version changed")
    input_manifest = read_json(input_manifest_path)
    if model_config["revision"] != input_manifest["model_revision"]:
        raise ValueError("Prepared inputs use another model revision")
    files = [CONFIG, Path(__file__), manifest_path, input_path, model_path, model_manifest_path,
             input_manifest_path, ROOT / "configs/prompt.yaml", ROOT / "requirements.lock.txt"]
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "generation_config.json")
    for name in tokenizer_names:
        path = model_dir / name
        data = path.read_bytes()
        record = model_manifest["verified_files"][name]
        git_hash = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
        if record["algorithm"] != "sha1" or git_hash != record["hash"] or len(data) != record["size"]:
            raise ValueError(f"Model/tokenizer metadata changed: {name}")
        files.append(path)
    for name in ("configs/prompt.yaml", "configs/model.yaml"):
        if sha256(ROOT / name) != input_manifest["input_sha256"][name]:
            raise ValueError("Prompt/model config differs from frozen prepared inputs")
    weights = [entry for entry in model_manifest["files"] if entry["is_weight"]]
    if len(weights) != 1:
        raise ValueError("This pilot expects the known single safetensors weight file")
    weight = model_dir / weights[0]["name"]
    weight_hash = sha256(weight)
    if weight_hash != weights[0]["sha256"] or weight.stat().st_size != weights[0]["size"]:
        raise ValueError("Model weight SHA256 differs from downloaded original")
    weight_stat = {"size": weight.stat().st_size, "mtime_ns": weight.stat().st_mtime_ns}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Missing local chat template")
    context_limit = read_json(model_dir / "config.json")["max_position_embeddings"]
    prepared = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        count = encoded["input_ids"].shape[1]
        if count > config["max_input_tokens"] or count + config["generation"]["max_new_tokens"] > context_limit:
            raise ValueError("Input exceeds fixed budget; do not silently truncate")
        prepared.append((row, encoded))
    fingerprints = {relative(path): sha256(path) for path in files}
    report = {"status": "ready", "examples": 32, "input_tokens_min": min(x[1]["input_ids"].shape[1] for x in prepared),
              "input_tokens_max": max(x[1]["input_ids"].shape[1] for x in prepared),
              "input_tokens_total": sum(x[1]["input_ids"].shape[1] for x in prepared),
              "weight_path": relative(weight), "weight_sha256": weight_hash, "weight_stat": weight_stat,
              "model_id": model_config["model_id"], "revision": model_config["revision"],
              "model_context_limit": context_limit, "reference_answers_read_by_generator": False,
              "final_examples_read": False, "weights_loaded": False, "input_sha256": fingerprints,
              "preflight_seconds_excluding_initial_library_import": round(time.perf_counter()-started, 3)}
    return config, pilot, model_config, tokenizer, prepared, report


class GPUSampler:
    """Whole-device samples, not process-only memory or a guaranteed instant peak."""
    def __init__(self, interval):
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stage = "before_load"
        self.stop_event = threading.Event()
        self.thread = None
        self.started = time.perf_counter()

    def sample(self):
        try:
            result = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            values = next(csv.reader(result.stdout.strip().splitlines()))
            self.samples.append({"elapsed_seconds": round(time.perf_counter()-self.started, 3), "stage": self.stage,
                                 "used_mib": float(values[0].strip()), "total_mib": float(values[1].strip())})
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def start(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
        self.sample()


def main(check_only):
    task_started = time.perf_counter()
    config, pilot, model_config, tokenizer, prepared, checked = preflight()
    save_json(ROOT / "reports/baseline_pilot_preflight.json", checked)
    print(f"Preflight: 32 inputs, tokens {checked['input_tokens_min']}..{checked['input_tokens_max']}; weight SHA256 verified", flush=True)
    if check_only:
        print("PILOT_PREFLIGHT: PASS; model weights not loaded", flush=True)
        return
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This fixed configuration requires CUDA and bf16")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_baseline_pilot"
    run_dir = ROOT / "outputs/preds" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    # Store exact inputs/code/config; large tokenizer files are represented by verified hashes.
    for filename in (relative(CONFIG), relative(Path(__file__)), config["pilot_manifest"], config["pilot_inputs"],
                     config["model_config"], "configs/prompt.yaml", "requirements.lock.txt"):
        destination = run_dir / "snapshot" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, destination)
    save_json(run_dir / "preflight.json", checked)
    save_json(run_dir / "status.json", {"run_id": run_id, "status": "started", "expected_examples": 32})
    with (run_dir / "prompts.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for source, encoded in prepared:
            stream.write(json.dumps({"source_index": source["source_index"], "db_id": source["db_id"],
                "input_token_count": encoded["input_ids"].shape[1], "rendered_prompt": tokenizer.apply_chat_template(
                source["messages"], tokenize=False, add_generation_prompt=True)}, ensure_ascii=False) + "\n")
    sampler = GPUSampler(config["gpu_sampling_interval_seconds"])
    sampler.start()
    predictions = []
    try:
        sampler.stage = "load"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before, total = torch.cuda.mem_get_info()
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
        if not getattr(model, "is_loaded_in_4bit", False) or devices != ["cuda:0"]:
            raise RuntimeError("Model was not loaded entirely on CUDA in 4-bit")
        if getattr(model, "peft_config", None):
            raise RuntimeError("Baseline must not contain a project fine-tuning adapter")
        settings = config["generation"]
        common = {key: settings[key] for key in ("do_sample", "num_beams", "num_return_sequences", "max_new_tokens",
                   "repetition_penalty", "use_cache", "temperature", "top_p", "top_k")}
        generation_config = GenerationConfig(**common, max_time=settings["max_time_seconds"],
            eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            bos_token_id=model.generation_config.bos_token_id)
        eos_ids = generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        effective, _ = model._prepare_generation_config(generation_config=generation_config)
        save_json(run_dir / "generation_config.json", effective.to_dict())
        print(f"Model loaded once: {load_seconds:.3f}s, NF4, CUDA. Generating exactly 32 queries.", flush=True)
        loop_started = time.perf_counter()
        blocked_error = None
        sampler.stage = "generation"
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for ordinal, (source, cpu_inputs) in enumerate(prepared, 1):
                remaining = config["generation_loop_budget_seconds"] - (time.perf_counter()-loop_started)
                if blocked_error is not None:
                    row = failed_prediction(source, blocked_error, "skipped_after_cuda_error")
                elif remaining <= 0:
                    row = failed_prediction(source, "Generation loop budget exhausted", "skipped_budget")
                else:
                    output = None
                    inputs = None
                    begun = time.perf_counter()
                    try:
                        inputs = {key: value.to("cuda:0") for key, value in cpu_inputs.items()}
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        begun = time.perf_counter()
                        with torch.inference_mode():
                            output = model.generate(**inputs, generation_config=generation_config,
                                                    max_time=min(settings["max_time_seconds"], remaining))
                        torch.cuda.synchronize()
                        seconds = time.perf_counter()-begun
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
                            "effective_max_time_seconds": min(settings["max_time_seconds"], remaining),
                            "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                            "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2}
                    except Exception as exc:
                        row = failed_prediction(source, f"{type(exc).__name__}: {exc}")
                        row["generation_seconds"] = time.perf_counter()-begun
                        if isinstance(exc, RuntimeError):
                            blocked_error = "Stopped further GPU work after generation RuntimeError: " + str(exc)
                    finally:
                        del output, inputs
                row["ordinal"] = ordinal
                row["input_token_count"] = cpu_inputs["input_ids"].shape[1]
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                print(f"[{ordinal:02d}/32] id={row['source_index']} status={row['generation_status']} tokens={row['generated_token_count']} seconds={row['generation_seconds']:.3f}", flush=True)
        loop_seconds = time.perf_counter()-loop_started
        sampler.stage = "after_generation"
        sampler.sample()
        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        save_json(run_dir / "status.json", {"run_id": run_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        sampler.stage = "after_release"
        sampler.close()
        save_json(run_dir / "gpu_samples.json", {"samples": sampler.samples, "errors": sampler.errors,
            "interval_seconds": sampler.interval, "scope": "sampled whole GPU memory, including other applications"})
    if [r["source_index"] for r in predictions] != pilot["selected_source_indices"]:
        raise ValueError("Missing or reordered pilot prediction records")
    for filename, expected_hash in checked["input_sha256"].items():
        if sha256(ROOT / filename) != expected_hash:
            raise ValueError(f"Source changed during generation: {filename}")
    weight = ROOT / checked["weight_path"]
    if {"size": weight.stat().st_size, "mtime_ns": weight.stat().st_mtime_ns} != checked["weight_stat"]:
        raise ValueError("Weight file metadata changed during generation")
    generated = [row for row in predictions if row["generation_ok"]]
    memory_rows = [r for r in predictions if "peak_allocated_mib" in r]
    summary = {"run_id": run_id, "scope": "fixed 32-question internal-validation pilot; not full 625 benchmark",
        "prediction_count": len(predictions), "generated_count": len(generated),
        "generation_failure_or_skip_count": 32-len(generated),
        "ended_with_eos_count": sum(r["ended_with_eos"] for r in predictions),
        "token_limit_count": sum(r["reached_token_limit"] for r in predictions),
        "generated_tokens_total": sum(r["generated_token_count"] for r in predictions),
        "generation_seconds_total": sum(r["generation_seconds"] for r in predictions),
        "generation_seconds_median": statistics.median(r["generation_seconds"] for r in predictions),
        "generation_loop_wall_seconds": loop_seconds, "load_seconds": load_seconds,
        "end_to_end_seconds": time.perf_counter()-task_started,
        "first_example_seconds": predictions[0]["generation_seconds"], "warmup_generations": 0,
        "load_peak_allocated_mib": load_allocated, "load_peak_reserved_mib": load_reserved,
        "generation_peak_allocated_mib": max((r["peak_allocated_mib"] for r in memory_rows), default=None),
        "generation_peak_reserved_mib": max((r["peak_reserved_mib"] for r in memory_rows), default=None),
        "nvidia_smi_sampled_peak_used_mib": max((r["used_mib"] for r in sampler.samples), default=None),
        "nvidia_smi_before_load_used_mib": sampler.samples[0]["used_mib"] if sampler.samples else None,
        "nvidia_smi_sample_count": len(sampler.samples), "nvidia_smi_sampling_error_count": len(sampler.errors),
        "nvidia_smi_scope": "whole GPU, sampled, includes other processes; not a guaranteed instantaneous peak",
        "gpu_name": torch.cuda.get_device_name(0), "total_gpu_memory_mib": total/1024**2,
        "free_before_load_mib": free_before/1024**2, "actual_parameter_devices": devices,
        "model_config": model_config, "model_weight_sha256": checked["weight_sha256"],
        "project_fine_tuning_applied": False, "is_loaded_in_4bit": True, "quantization": config["quantization"],
        "generation_config": effective.to_dict(), "requested_config": config,
        "packages": {n: version(n) for n in ("torch", "transformers", "accelerate", "bitsandbytes", "tokenizers")},
        "source_sha256": checked["input_sha256"], "source_files_unchanged": True,
        "weight_metadata_unchanged_after_run": True, "reference_answers_read_by_generator": False,
        "final_examples_read": False, "complete_validation_generated": False,
        "run_dir": relative(run_dir), "prediction_path": relative(run_dir / "predictions.jsonl"),
        "prediction_sha256": sha256(run_dir / "predictions.jsonl"), "pilot_manifest_sha256": sha256(project_path(config["pilot_manifest"]))}
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "status.json", {"run_id": run_id, "status": "completed", "prediction_count": 32,
                                      "generated_count": len(generated), "prediction_sha256": summary["prediction_sha256"]})
    shutil.copyfile(run_dir / "predictions.jsonl", ROOT / "outputs/preds/baseline_pilot.jsonl")
    save_json(ROOT / "reports/baseline_pilot_generation.json", summary)
    print(json.dumps({k: summary[k] for k in ("generated_count", "ended_with_eos_count", "token_limit_count", "generated_tokens_total",
        "generation_seconds_total", "load_seconds", "end_to_end_seconds", "generation_peak_allocated_mib",
        "generation_peak_reserved_mib", "nvidia_smi_sampled_peak_used_mib", "prediction_path")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate the fixed pilot and local tokenizer/weights without loading the model")
    main(parser.parse_args().check)
