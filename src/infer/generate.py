"""用本地官方模型生成一条 SQL，并记录模板、执行结果、耗时与显存。"""

import argparse
import json
import os
import random
import re
import sqlite3
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# 推理阶段只读已下载的本地文件；联网下载由 download_model.py 负责。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_sql(text):
    # 仅去除首尾空白或完整包住答案的一层代码围栏，不改写 SQL 内容。
    text = text.strip()
    fenced = re.fullmatch(r"```(?:sql)?[ \t]*\r?\n(.*?)```", text, flags=re.S | re.I)
    return fenced.group(1).strip() if fenced else text


def check_sql(example, predicted_sql):
    # 此函数只用于这个合成练习，正式 Spider 评测将使用官方评测工具。
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(example["schema"])
        connection.executemany("INSERT INTO students VALUES (?, ?, ?, ?)", example["rows"])
        connection.commit()
        gold_rows = connection.execute(example["gold_sql"]).fetchall()
        expected_rows = [tuple(row) for row in example["expected_rows"]]
        if gold_rows != expected_rows:
            raise RuntimeError("合成数据库或 gold SQL 与手工预期不一致。")

        connection.execute("PRAGMA query_only = ON")

        def authorize(action, arg1, arg2, database, source):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and database == "main" and arg1 == "students":
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        deadline = time.perf_counter() + example["query_timeout_seconds"]
        connection.set_progress_handler(lambda: int(time.perf_counter() >= deadline), 1000)
        started = time.perf_counter()
        try:
            cursor = connection.execute(predicted_sql)
            rows = cursor.fetchmany(example["max_result_rows"] + 1)
            if len(rows) > example["max_result_rows"]:
                raise RuntimeError("查询结果超过本次练习的行数上限。")
            # 本题要求排序，因此严格比较行序，不能改成集合比较。
            return {"executable": True, "predicted_rows": rows, "gold_rows": gold_rows,
                    "matches_gold": rows == gold_rows, "seconds": time.perf_counter() - started}
        except (sqlite3.Error, RuntimeError) as error:
            return {"executable": False, "predicted_rows": None, "gold_rows": gold_rows,
                    "matches_gold": False, "error": str(error),
                    "seconds": time.perf_counter() - started}
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--inspect-prompt", action="store_true", help="只检查分词器和聊天模板，不加载权重")
    args = parser.parse_args()
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text(encoding="utf-8"))
    model_config = yaml.safe_load((PROJECT_ROOT / config["model_config"]).read_text(encoding="utf-8"))
    prompt_config = yaml.safe_load((PROJECT_ROOT / config["prompt_config"]).read_text(encoding="utf-8"))
    manifest = json.loads((PROJECT_ROOT / "reports" / "model_manifest.json").read_text(encoding="utf-8"))
    if not model_config["revision"] or manifest["revision"] != model_config["revision"]:
        raise RuntimeError("请先下载并固定模型版本。")
    model_dir = PROJECT_ROOT / model_config["local_dir"]
    example = config["example"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise RuntimeError("官方分词器没有可用的聊天模板。")
    # 模型只接收 schema 和 question；不会把 gold SQL 或预期答案放入提示词。
    messages = [
        {"role": "system", "content": prompt_config["system"]},
        {"role": "user", "content": prompt_config["user_template"].format(
            schema=example["schema"], question=example["question"])},
    ]
    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 直接由模板生成 token，避免手动再次添加特殊 token。
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_length = inputs["input_ids"].shape[1]
    if input_length > 1024:
        raise RuntimeError("本次单条冒烟输入超过 1024 token，请先检查提示词。")
    prompt_path = PROJECT_ROOT / "outputs" / "preds" / "first_sql_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(rendered_prompt, encoding="utf-8")
    sample_text = "SELECT name FROM students"
    sample_ids = tokenizer.encode(sample_text, add_special_tokens=False)
    tokenizer_report = {
        "model_id": model_config["model_id"], "revision": model_config["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "sample_text": sample_text, "sample_token_ids": sample_ids,
        "sample_token_pieces": tokenizer.convert_ids_to_tokens(sample_ids),
        "sample_decoded": tokenizer.decode(sample_ids),
        "messages": messages, "rendered_prompt": rendered_prompt,
        "input_token_count": input_length, "add_generation_prompt": True,
    }
    if tokenizer_report["sample_decoded"] != sample_text:
        raise RuntimeError("示例文本的分词与解码往返不一致。")
    save_json(PROJECT_ROOT / "reports" / "tokenizer_check.json", tokenizer_report)
    print(f"分词器: {type(tokenizer).__name__}", flush=True)
    print(f"分词示例: {sample_text}", flush=True)
    print(f"token 编号: {sample_ids}", flush=True)
    print(f"token 片段: {tokenizer_report['sample_token_pieces']}", flush=True)
    print(f"完整输入 token 数: {input_length}", flush=True)
    print(f"渲染后的聊天输入: {prompt_path}", flush=True)
    if args.inspect_prompt:
        print("TOKENIZER_AND_TEMPLATE_CHECK: PASS", flush=True)
        return

    weights = [entry for entry in manifest["files"] if entry["is_weight"]]
    if not weights or any(entry["name"] not in manifest["verified_files"] for entry in weights):
        raise RuntimeError("请先完成 --stage weights 下载和权重哈希校验。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请先运行 src/env_check.py。")
    if config["quantization"]["compute_dtype"] != "bfloat16" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前配置需要已验证的 bf16 GPU 支持。")
    torch.cuda.manual_seed_all(config["seed"])
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=config["quantization"]["type"],
        bnb_4bit_use_double_quant=config["quantization"]["double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_memory = torch.cuda.mem_get_info()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
        quantization_config=quantization_config, device_map={"": "cuda:0"},
        dtype=torch.bfloat16, attn_implementation=config["attention_implementation"],
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    load_peak_allocated = torch.cuda.max_memory_allocated()
    load_peak_reserved = torch.cuda.max_memory_reserved()
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("模型未以 4-bit 方式加载。")
    if not all(parameter.device.type == "cuda" for parameter in model.parameters()):
        raise RuntimeError("发现模型参数没有放在 GPU 上。")
    print(f"4-bit 模型加载完成: {load_seconds:.2f} 秒", flush=True)

    # 新建生成配置，明确使用 greedy，避免继承仓库中的随机采样参数。
    generation_config = GenerationConfig(
        do_sample=config["generation"]["do_sample"], num_beams=config["generation"]["num_beams"],
        max_new_tokens=config["generation"]["max_new_tokens"],
        max_time=config["generation"]["max_time_seconds"],
        repetition_penalty=config["generation"]["repetition_penalty"],
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=model.generation_config.bos_token_id, use_cache=True,
        # 当前 Transformers 将未显式设置的值从仓库配置补齐；用中性值避免继承采样偏好。
        temperature=1.0, top_p=1.0, top_k=50,
    )
    inputs = inputs.to("cuda:0")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=generation_config)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    generation_peak_allocated = torch.cuda.max_memory_allocated()
    generation_peak_reserved = torch.cuda.max_memory_reserved()
    # 输出包含输入前缀；这里只解码模型新生成的部分。
    new_ids = output[0, input_length:].tolist()
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=False)
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    predicted_sql = extract_sql(generated_text)
    execution = check_sql(example, predicted_sql)
    eos_ids = generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    ended_with_eos = bool(new_ids and new_ids[-1] in eos_ids)
    reached_token_limit = len(new_ids) >= generation_config.max_new_tokens and not ended_with_eos
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(), "example_id": example["id"],
        "scope": "one_synthetic_example_not_a_benchmark", "model_id": model_config["model_id"],
        "revision": model_config["revision"], "seed": config["seed"],
        "package_versions": {name: version(name) for name in ["torch", "transformers", "accelerate", "bitsandbytes"]},
        "question": example["question"], "schema": example["schema"],
        "raw_output_with_special_tokens": raw_output, "generated_text": generated_text,
        "predicted_sql": predicted_sql, "new_token_ids": new_ids,
        "input_token_count": input_length, "generated_token_count": len(new_ids),
        "ended_with_eos": ended_with_eos, "reached_token_limit": reached_token_limit,
        "generation_config": generation_config.to_dict(), "execution": execution,
        "quantization": config["quantization"], "is_loaded_in_4bit": model.is_loaded_in_4bit,
        "attention_implementation": model.config._attn_implementation,
        "requested_device_map": {"": "cuda:0"},
        "actual_parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}),
        "load_seconds": load_seconds, "generation_seconds": generation_seconds,
        "gpu_name": torch.cuda.get_device_name(), "gpu_free_before_load_mib": free_before / 1024**2,
        "gpu_total_memory_mib": total_memory / 1024**2,
        "load_peak_allocated_mib": load_peak_allocated / 1024**2,
        "load_peak_reserved_mib": load_peak_reserved / 1024**2,
        "generation_peak_allocated_mib": generation_peak_allocated / 1024**2,
        "generation_peak_reserved_mib": generation_peak_reserved / 1024**2,
        "memory_note": "PyTorch allocator peaks include model memory and exclude some driver/context allocations.",
    }
    result_path = PROJECT_ROOT / "reports" / "first_sql_run.json"
    save_json(result_path, result)
    prediction_dir = PROJECT_ROOT / "outputs" / "preds"
    (prediction_dir / "first_sql.jsonl").write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    (prediction_dir / "first_sql.sql").write_text(predicted_sql + "\n", encoding="utf-8")
    print(f"生成 SQL:\n{predicted_sql}", flush=True)
    print(f"执行结果: {execution['predicted_rows']}", flush=True)
    print(f"与本题预期一致: {execution['matches_gold']}", flush=True)
    if not execution["executable"]:
        print(f"执行失败原因: {execution['error']}", flush=True)
    print(f"生成耗时: {generation_seconds:.3f} 秒；新 token 数: {len(new_ids)}", flush=True)
    print(f"加载峰值 allocated/reserved: {load_peak_allocated / 1024**2:.1f}/{load_peak_reserved / 1024**2:.1f} MiB", flush=True)
    print(f"生成峰值 allocated/reserved: {generation_peak_allocated / 1024**2:.1f}/{generation_peak_reserved / 1024**2:.1f} MiB", flush=True)
    print(f"运行记录: {result_path}", flush=True)


if __name__ == "__main__":
    main()
