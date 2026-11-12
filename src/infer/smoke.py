"""Phase 0：先检查练习与判分器，再逐条生成 20 个 SQL；无需新增依赖。"""

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from eval.smoke_checks import evaluate_prediction, run_self_checks, validate_suite


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_yaml(relative_path):
    return yaml.safe_load((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def make_report(summary, predictions):
    lines = [
        "# Phase 0：20 条合成 SQL 冒烟验证", "",
        "范围：手工合成题；不是 Spider EX、官方测试套件成绩或泛化证明。", "",
        f"- 运行：`{summary['run_id']}`",
        f"- 完成生成：{summary['generated_count']}/{summary['case_count']}",
        f"- 两份数据均能执行：{summary['all_valid_count']}/{summary['case_count']}",
        f"- 两份数据均匹配预期：{summary['all_match_count']}/{summary['case_count']}",
        f"- 数据实例执行次数：{summary['fixture_execution_count']}",
        f"- 实例执行成功：{summary['fixture_valid_count']}；实例结果匹配：{summary['fixture_match_count']}",
        f"- 生成总耗时：{summary['generation_seconds_total']:.3f} 秒；逐题中位数：{summary['generation_seconds_median']:.3f} 秒",
        f"- 模型加载耗时：{summary['load_seconds']:.3f} 秒",
        f"- 生成峰值 allocated / reserved：{summary['generation_peak_allocated_mib']:.1f} / {summary['generation_peak_reserved_mib']:.1f} MiB",
        "", "显存为 PyTorch 分配器统计，含模型，不是整卡占用；allocated 与 reserved 不能相加。", "",
        "每题只生成一次，同一条 SQL 执行于 base/edge 两份数据。valid 只表示本次只读、超时和行数限制内执行成功；all_match 才表示两份数据结果都符合预期。", "",
        "比较规则：要求排序则逐行比较；否则忽略行序但保留重复次数。列顺序保持，NULL 与文本严格区分；整数精确比较，浮点绝对容差 1e-6、相对容差 0。", "",
        "| 题号 | 类型 | 可执行（两份） | 匹配（两份） | 新 token | 生成秒 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in predictions:
        evaluation = row["evaluation"]
        lines.append(f"| {row['id']} | {row['category']} | {evaluation['all_valid']} | {evaluation['all_match']} | {row.get('generated_token_count', 0)} | {row.get('generation_seconds', 0):.3f} |")
    lines += ["", "## 逐题结果", ""]
    for row in predictions:
        lines += [f"### {row['id']} · {row['category']}", "", row["question"], "", row["explanation_zh"], "",
                  "模型 SQL：", "", "```sql", row.get("predicted_sql", ""), "```", "", "参考 SQL：", "", "```sql", row["gold_sql"], "```", ""]
        if row.get("generation_error"):
            lines += [f"生成错误：{row['generation_error']}", ""]
        for result in row["evaluation"]["fixture_results"]:
            lines += [f"**{result['fixture_id']}**：valid={result['valid']}，match={result['matches_expected']}", "",
                      f"实际：`{json.dumps(result['predicted_rows'], ensure_ascii=False)}`", "",
                      f"预期：`{json.dumps(result['expected_rows'], ensure_ascii=False)}`", ""]
            if result.get("error"):
                lines += [f"错误：{result['error']}", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke_batch.yaml")
    parser.add_argument("--validate-only", action="store_true", help="只检查40份标准结果和判分器，不加载GPU模型")
    args = parser.parse_args()
    config = read_yaml(args.config)
    suite = read_yaml(config["suite_config"])
    if len(suite["cases"]) != 20:
        raise ValueError("本次 Phase 0 固定为20题，请检查练习文件。")
    eval_kwargs = {"timeout_seconds": config["evaluation"]["process_timeout_seconds"],
                   "max_rows": config["evaluation"]["max_result_rows"]}
    print("先检查标准答案与判分规则；此阶段不使用 GPU。", flush=True)
    validation = validate_suite(suite, **eval_kwargs)
    self_checks = run_self_checks()
    validation_record = {"suite_id": suite["suite_id"], "suite_sha256": sha256(PROJECT_ROOT / config["suite_config"]),
                         "validation": validation, "self_checks": self_checks}
    save_json(PROJECT_ROOT / "reports" / "smoke_validation.json", validation_record)
    print(f"标准结果：{validation['fixture_execution_count']} 次通过；判分自检：{self_checks['passed_count']} 项通过。", flush=True)
    if args.validate_only:
        print("SMOKE_VALIDATION: PASS", flush=True)
        return

    # 到这里才导入模型库、检查 GPU；下载关闭，复用已经校验的本地模型。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
    from generate import extract_sql

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("本配置需要已验证的 CUDA 和 bf16 支持。")
    torch.cuda.manual_seed_all(config["seed"])
    model_config = read_yaml(config["model_config"])
    prompt_config = read_yaml(config["prompt_config"])
    manifest = json.loads((PROJECT_ROOT / "reports/model_manifest.json").read_text(encoding="utf-8"))
    if any(manifest[key] != model_config[key] for key in ("model_id", "revision", "local_dir")):
        raise ValueError("本地模型清单与配置不一致。")
    model_dir = PROJECT_ROOT / model_config["local_dir"]
    weights = [entry for entry in manifest["files"] if entry["is_weight"]]
    if not weights or any(entry["name"] not in manifest["verified_files"] or
                          not (model_dir / entry["name"]).is_file() or
                          (model_dir / entry["name"]).stat().st_size != entry["size"] for entry in weights):
        raise ValueError("权重未完成校验或文件大小改变，请先检查下载。")
    if config["quantization"]["compute_dtype"] != "bfloat16":
        raise ValueError("本步保持 bf16 计算配置。")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("官方聊天模板缺失。")
    prepared = []
    for case in suite["cases"]:
        schema = suite["databases"][case["database_id"]]["schema"]
        # 数据行、参考 SQL 和预期答案只供执行器使用，绝不进入模型输入。
        messages = [{"role": "system", "content": prompt_config["system"]},
                    {"role": "user", "content": prompt_config["user_template"].format(schema=schema, question=case["question"])}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        if inputs["input_ids"].shape[1] > config["max_input_tokens"]:
            raise ValueError(f"{case['id']} 超出本步输入长度限制。")
        prepared.append((case, inputs, tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = PROJECT_ROOT / "outputs" / "preds" / f"phase0_smoke_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    source_paths = [args.config, config["suite_config"], config["model_config"], config["prompt_config"],
                    "requirements.lock.txt", "reports/model_manifest.json", "src/infer/smoke.py", "src/infer/generate.py",
                    "src/eval/smoke_checks.py", "src/eval/sql_worker.py"]
    source_hashes = {}
    for relative in source_paths:
        source = PROJECT_ROOT / relative
        destination = run_dir / "snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hashes[relative] = sha256(source)
    save_json(run_dir / "validation.json", validation_record)
    save_json(run_dir / "manifest.json", {"run_id": run_id, "status": "started", "source_sha256": source_hashes})
    with (run_dir / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for case, inputs, rendered in prepared:
            handle.write(json.dumps({"id": case["id"], "prompt": rendered, "input_token_count": inputs["input_ids"].shape[1]}, ensure_ascii=False) + "\n")

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, device_map={"": "cuda:0"}, dtype=torch.bfloat16,
        attn_implementation=config["attention_implementation"],
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=config["quantization"]["type"],
            bnb_4bit_use_double_quant=config["quantization"]["double_quant"], bnb_4bit_compute_dtype=torch.bfloat16),
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    load_allocated, load_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if not getattr(model, "is_loaded_in_4bit", False) or parameter_devices != ["cuda:0"]:
        raise RuntimeError("模型没有按本步要求完整地以4-bit放到cuda:0。")
    settings = config["generation"]
    generation_config = GenerationConfig(do_sample=settings["do_sample"], num_beams=settings["num_beams"],
        max_new_tokens=settings["max_new_tokens"], max_time=settings["max_time_seconds"], repetition_penalty=settings["repetition_penalty"],
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=model.generation_config.bos_token_id, use_cache=True, temperature=1.0, top_p=1.0, top_k=50)
    eos_ids = generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    predictions = []
    loop_started = time.perf_counter()
    print(f"模型只加载一次，逐题处理20条，加载耗时 {load_seconds:.3f} 秒。", flush=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, (case, cpu_inputs, rendered) in enumerate(prepared, 1):
            if time.perf_counter() - loop_started > config["generation_budget_seconds"]:
                raise RuntimeError(f"已超过本轮时间预算，停止追加生成；已有结果保存在 {run_dir}")
            row = {key: case[key] for key in ("id", "database_id", "category", "question", "gold_sql", "ordered", "explanation_zh")}
            row["input_token_count"] = cpu_inputs["input_ids"].shape[1]
            inputs = {key: value.to("cuda:0") for key, value in cpu_inputs.items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    output = model.generate(**inputs, generation_config=generation_config)
                torch.cuda.synchronize()
                row["generation_seconds"] = time.perf_counter() - started
                new_ids = output[0, row["input_token_count"]:].tolist()
                row.update(generation_ok=True, generated_token_count=len(new_ids), new_token_ids=new_ids,
                           raw_output_with_special_tokens=tokenizer.decode(new_ids, skip_special_tokens=False),
                           generated_text=tokenizer.decode(new_ids, skip_special_tokens=True),
                           ended_with_eos=bool(new_ids and new_ids[-1] in eos_ids))
                row["predicted_sql"] = extract_sql(row["generated_text"])
                row["reached_token_limit"] = len(new_ids) >= settings["max_new_tokens"] and not row["ended_with_eos"]
                del output
            except RuntimeError as error:
                row.update(generation_ok=False, generation_error=str(error), predicted_sql="", generated_token_count=0)
                row["generation_seconds"] = time.perf_counter() - started
            row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
            row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
            del inputs
            row["evaluation"] = evaluate_prediction(suite, case, row["predicted_sql"], **eval_kwargs)
            predictions.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index:02d}/20] {case['id']} valid={row['evaluation']['all_valid']} match={row['evaluation']['all_match']} tokens={row['generated_token_count']} seconds={row['generation_seconds']:.3f}", flush=True)

    fixtures = [item for row in predictions for item in row["evaluation"]["fixture_results"]]
    summary = {
        "run_id": run_id, "scope": "phase0_synthetic_smoke_not_spider_benchmark", "case_count": len(prepared),
        "generated_count": sum(row["generation_ok"] and bool(row["generated_text"].strip()) for row in predictions),
        "all_valid_count": sum(row["evaluation"]["all_valid"] for row in predictions),
        "all_match_count": sum(row["evaluation"]["all_match"] for row in predictions),
        "fixture_execution_count": len(fixtures), "fixture_valid_count": sum(item["valid"] for item in fixtures),
        "fixture_match_count": sum(item["matches_expected"] for item in fixtures),
        "ended_with_eos_count": sum(row.get("ended_with_eos", False) for row in predictions),
        "token_limit_count": sum(row.get("reached_token_limit", False) for row in predictions),
        "generated_tokens_total": sum(row["generated_token_count"] for row in predictions),
        "generation_seconds_total": sum(row["generation_seconds"] for row in predictions),
        "generation_seconds_median": statistics.median(row["generation_seconds"] for row in predictions),
        "loop_seconds": time.perf_counter() - loop_started, "load_seconds": load_seconds,
        "load_peak_allocated_mib": load_allocated / 1024**2, "load_peak_reserved_mib": load_reserved / 1024**2,
        "generation_peak_allocated_mib": max(row["peak_allocated_mib"] for row in predictions),
        "generation_peak_reserved_mib": max(row["peak_reserved_mib"] for row in predictions),
        "model_config": model_config, "config": config, "generation_config_requested": generation_config.to_dict(),
        "is_loaded_in_4bit": model.is_loaded_in_4bit, "actual_parameter_devices": parameter_devices,
        "gpu_name": torch.cuda.get_device_name(), "free_before_load_mib": free_before / 1024**2,
        "total_gpu_memory_mib": total_memory / 1024**2, "python": sys.version, "platform": platform.platform(),
        "package_versions": {name: version(name) for name in ["torch", "transformers", "accelerate", "bitsandbytes", "PyYAML"]},
        "source_sha256": source_hashes, "run_dir": str(run_dir),
    }
    summary["engineering_acceptance_pass"] = summary["generated_count"] == 20 and summary["fixture_execution_count"] == 40 and summary["ended_with_eos_count"] == 20
    save_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(make_report(summary, predictions), encoding="utf-8")
    save_json(run_dir / "manifest.json", {"run_id": run_id, "status": "completed", "source_sha256": source_hashes,
              "predictions_sha256": sha256(run_dir / "predictions.jsonl")})
    shutil.copy2(run_dir / "predictions.jsonl", PROJECT_ROOT / "outputs/preds/base_smoke.jsonl")
    shutil.copy2(run_dir / "summary.json", PROJECT_ROOT / "reports/phase0_smoke_summary.json")
    shutil.copy2(run_dir / "report.md", PROJECT_ROOT / "reports/phase0_smoke_report.md")
    print(f"生成 {summary['generated_count']}/20；两份均可执行 {summary['all_valid_count']}/20；两份均匹配 {summary['all_match_count']}/20。", flush=True)
    print(f"生成总耗时 {summary['generation_seconds_total']:.3f} 秒；显存峰值 allocated/reserved={summary['generation_peak_allocated_mib']:.1f}/{summary['generation_peak_reserved_mib']:.1f} MiB。", flush=True)
    print(f"逐题报告：{PROJECT_ROOT / 'reports/phase0_smoke_report.md'}", flush=True)
    print(f"运行快照：{run_dir}", flush=True)
    print("PHASE0_PIPELINE: " + ("PASS" if summary["engineering_acceptance_pass"] else "NEEDS_REVIEW") + "（工程链路验收，答案正确数另列）", flush=True)


if __name__ == "__main__":
    main()
