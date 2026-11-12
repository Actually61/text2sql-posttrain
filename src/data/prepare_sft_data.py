"""先检查8条训练样本，再发布全部6364条SFT特征；只CPU，不加载模型。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sft_features import encode_sft_example, collate_sft_batch

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/sft_data.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def project_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("路径必须在本项目内")
    return path


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(values):
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in values).encode("utf-8")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def expect_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"冻结文件发生变化: {path}")


def verify_sources(fingerprints):
    for name, expected in fingerprints.items():
        expect_hash(project_path(name), expected)


def keyed(rows, label):
    if len(rows) != 6364 or any(type(r.get("source_index")) is not int for r in rows):
        raise ValueError(f"{label}必须是6364条有整数ID的训练记录")
    result = {r["source_index"]: r for r in rows}
    if len(result) != 6364:
        raise ValueError(f"{label}包含重复ID")
    return result


def load_context():
    """只读训练文件和冻结元数据，不打开验证/最终题目或任何数据库。"""
    import yaml
    from transformers import AutoTokenizer

    config = read(CONFIG)
    if (config["expected_examples"] != 6364 or config["max_sequence_length"] != 4096
            or config["truncate"] or config["filter_long_examples"] or config["packing"]
            or config["ignore_index"] != -100 or config["device"] != "cpu"
            or config["padding_side"] != "right"):
        raise ValueError("当前阶段只支持固定的无裁剪、无过滤、CPU准备协议")
    verification_path = project_path(config["prepared_verification"])
    manifest_path = project_path(config["prepared_manifest"])
    expect_hash(manifest_path, read(verification_path)["manifest_sha256"])
    manifest = read(manifest_path)
    if manifest["train_examples"] != 6364:
        raise ValueError("冻结输入不是清洗后的6364条")
    paths = [CONFIG, Path(__file__), ROOT / "src/data/sft_features.py", verification_path, manifest_path,
             ROOT / "requirements.lock.txt"]
    clean_path = project_path(config["clean_train"])
    expect_hash(clean_path, manifest["input_sha256"][relative(clean_path)])
    paths.append(clean_path)
    clean = keyed(read_rows(clean_path), "clean_train")
    tables = {}
    for name in ("train_messages", "train_inputs", "train_lengths"):
        path = project_path(config[name])
        expect_hash(path, manifest["output_sha256"][path.name])
        paths.append(path)
        tables[name] = keyed(read_rows(path), name)
        if set(tables[name]) != set(clean):
            raise ValueError(f"{name}与清洗训练集成员不一致")
    messages, inputs, lengths = (tables[name] for name in ("train_messages", "train_inputs", "train_lengths"))
    required = {"source_index", "source_file", "db_id", "project_split", "prompt", "completion"}
    for index, source in clean.items():
        row = messages[index]
        if (set(row) != required or any(r.get("project_split") != "train" for r in (source, row, inputs[index], lengths[index]))
                or any(r["db_id"] != source["db_id"] or r["source_file"] != source["source_file"] for r in (row, inputs[index], lengths[index]))
                or row["prompt"] != inputs[index]["messages"]
                or row["completion"] != [{"role": "assistant", "content": source["query"]}]):
            raise ValueError(f"来源、prompt或训练答案不一致: {index}")
    if len({r["db_id"] for r in clean.values()}) != config["expected_database_count"]:
        raise ValueError("训练数据库数量不一致")
    model_path = project_path(config["model_config"])
    model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    if model["model_id"] != manifest["model_id"] or model["revision"] != manifest["model_revision"]:
        raise ValueError("模型版本与冻结分词输入不一致")
    for path in (model_path, ROOT / "configs/prompt.yaml", ROOT / "reports/model_manifest.json"):
        expect_hash(path, manifest["input_sha256"][relative(path)])
        paths.append(path)
    model_dir = project_path(model["local_dir"])
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = model_dir / name
        expect_hash(path, manifest["input_sha256"][relative(path)])
        paths.append(path)
    for name, expected in manifest["packages"].items():
        if version(name) != expected:
            raise ValueError(f"冻结输入的依赖版本已变化: {name}")
    if config["max_sequence_length"] > read(model_dir / "config.json")["max_position_embeddings"]:
        raise ValueError("准备长度超过模型上下文上限")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    fingerprints = {relative(path): digest(path) for path in paths}
    ordered = [messages[i] for i in sorted(messages)]
    return config, ordered, lengths, tokenizer, fingerprints, model


def verify_feature(feature, row, length, tokenizer):
    """对已保存的特征独立检查监督位置和原SQL，不重新运行编码函数。"""
    ids, masks, labels = (feature[key] for key in ("input_ids", "attention_mask", "labels"))
    count, start, end = len(ids), feature["prompt_length"], feature["eos_position"]
    if (count != length["full_reference_tokens"] or start != length["prompt_tokens"]
            or count != len(masks) or count != len(labels) or count != feature["sequence_length"]
            or not 0 < start < end < count or count > 4096 or masks != [1] * count
            or ids[end] != tokenizer.eos_token_id):
        raise ValueError("保存特征的长度、attention或EOS位置不正确")
    if any(feature[key] != row[key] for key in ("source_index", "source_file", "db_id", "project_split")):
        raise ValueError("特征来源与训练记录不一致")
    expected_positions = set(range(start, end + 1))
    actual_positions = {i for i, value in enumerate(labels) if value != -100}
    if actual_positions != expected_positions or any(labels[i] != ids[i] for i in actual_positions):
        raise ValueError("标签错位或监督区域不正确")
    if tokenizer.decode(ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False) != row["completion"][0]["content"]:
        raise ValueError("保存的监督SQL与原训练答案不一致")
    if feature["supervised_token_count"] != len(actual_positions) or feature["trailing_template_token_count"] != count-end-1:
        raise ValueError("监督长度元信息不正确")
    if tokenizer.decode(ids[end+1:], skip_special_tokens=False, clean_up_tokenization_spaces=False) not in ("", "\n"):
        raise ValueError("EOS后出现非预期模板文本")


def publish_files(directory, files):
    # 先检查全部内容；同版本可以复用，但不能悄悄覆盖不同的数据。
    for name, payload in files.items():
        path = directory / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"已有版本与新内容不同，拒绝覆盖: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        path = directory / name
        if not path.exists():
            path.write_bytes(payload)


def feature_summary(features):
    seq = sorted(f["sequence_length"] for f in features)
    return {
        "count": len(features), "database_count": len({f["db_id"] for f in features}),
        "sequence_tokens": {"min": min(seq), "median": statistics.median(seq),
                            "p95": seq[math.ceil(0.95*len(seq))-1], "max": max(seq), "total": sum(seq)},
        "supervised_sql_and_eos_tokens": sum(f["supervised_token_count"] for f in features),
        "masked_prompt_tokens": sum(f["prompt_length"] for f in features),
        "masked_trailing_template_tokens": sum(f["trailing_template_token_count"] for f in features),
        "over_2048_count": sum(f["sequence_length"] > 2048 for f in features),
        "over_4096_count": sum(f["sequence_length"] > 4096 for f in features),
    }


def preview(context):
    config, rows, lengths, tokenizer, fingerprints, model = context
    chosen = rows[:config["preview_first_examples"]]
    longest = sorted(rows, key=lambda r: (-lengths[r["source_index"]]["full_reference_tokens"], r["source_index"]))
    for row in longest:
        if row["source_index"] not in {r["source_index"] for r in chosen}:
            chosen.append(row)
        if len(chosen) == config["preview_first_examples"] + config["preview_longest_examples"]:
            break
    features = [encode_sft_example(row, tokenizer, config["max_sequence_length"]) for row in chosen]
    for feature, row in zip(features, chosen):
        verify_feature(feature, row, lengths[row["source_index"]], tokenizer)
    import torch
    batches = {}
    for name, pair in (("first_two", features[:2]), ("short_and_long", [features[0], features[-2]])):
        batch = collate_sft_batch(pair, tokenizer.pad_token_id, config["max_sequence_length"])
        if any(t.device.type != "cpu" or t.dtype != torch.long or t.requires_grad for t in batch.values()):
            raise ValueError("本步batch必须是无需梯度的CPU long张量")
        for position, feature in enumerate(pair):
            real = feature["sequence_length"]
            if batch["labels"][position, :real].tolist() != feature["labels"]:
                raise ValueError("collator改变了原有labels")
            if (not torch.all(batch["labels"][position, real:] == -100)
                    or not torch.all(batch["attention_mask"][position, real:] == 0)
                    or not torch.all(batch["input_ids"][position, real:] == tokenizer.pad_token_id)):
                raise ValueError("padding位置的值不正确")
        batches[name] = {"source_indices": [f["source_index"] for f in pair], "shape": list(batch["input_ids"].shape),
                         "dtype": str(batch["input_ids"].dtype), "device": str(batch["input_ids"].device),
                         "padding_tokens": int((batch["attention_mask"] == 0).sum().item()),
                         "supervised_tokens": int((batch["labels"] != -100).sum().item())}
        if name == "first_two":
            first_batch = {key: value.tolist() for key, value in batch.items()}
    first = features[0]
    trace = []
    for index in range(first["prompt_length"]-6, len(first_batch["input_ids"][0])):
        token = first_batch["input_ids"][0][index]
        label = first_batch["labels"][0][index]
        region = ("padding" if index >= first["sequence_length"] else "prompt" if index < first["prompt_length"]
                  else "sql" if index < first["eos_position"] else "eos" if index == first["eos_position"] else "template_tail")
        trace.append({"position": index, "region": region, "input_id": token,
                      "token_text": tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                      "attention_mask": first_batch["attention_mask"][0][index], "label": label,
                      "prediction_logit_position": index-1 if label != -100 else None})
    directory = project_path(config["preview_dir"])
    files = {"features.jsonl": jsonl_bytes(features), "first_batch.json": json_bytes(first_batch), "token_trace.json": json_bytes(trace)}
    verify_sources(fingerprints)
    publish_files(directory, files)
    report = {"status": "passed", "selection": "first 6 source IDs plus 2 longest different training examples",
              "source_indices": [f["source_index"] for f in features], "stats": feature_summary(features),
              "first_example": {k: v for k, v in first.items() if k not in ("input_ids", "attention_mask", "labels")},
              "first_example_sql": chosen[0]["completion"][0]["content"], "batches": batches,
              "token_trace_path": relative(directory / "token_trace.json"),
              "output_sha256": {relative(directory / name): hashlib.sha256(value).hexdigest() for name, value in files.items()},
              "source_sha256": fingerprints, "model_revision": model["revision"],
              "model_weights_loaded": False, "optimizer_created": False, "backward_called": False,
              "training_performed": False, "validation_examples_loaded": False, "final_examples_read": False}
    save(ROOT / "reports/sft_data_preview.json", report)
    print(json.dumps({k: report[k] for k in ("status", "source_indices", "first_example", "batches", "token_trace_path")}, ensure_ascii=False, indent=2))


def build(context):
    config, rows, lengths, tokenizer, fingerprints, model = context
    small = read(ROOT / "reports/sft_data_preview.json")
    if small["status"] != "passed" or small["source_sha256"] != fingerprints:
        raise ValueError("必须先用当前源码/配置通过8条CPU预览，再构建全部训练特征")
    verify_sources(small["output_sha256"])
    features = []
    for ordinal, row in enumerate(rows, 1):
        feature = encode_sft_example(row, tokenizer, config["max_sequence_length"])
        verify_feature(feature, row, lengths[row["source_index"]], tokenizer)
        features.append(feature)
        if ordinal % 1000 == 0 or ordinal == len(rows):
            print(f"CPU tokenization: {ordinal}/6364", flush=True)
    stats = feature_summary(features)
    if stats["count"] != 6364 or stats["database_count"] != 126 or stats["over_4096_count"]:
        raise ValueError("数据总量、数据库覆盖或长度预算未通过")
    long_rows = [{k: f[k] for k in ("source_index", "db_id", "sequence_length", "prompt_length", "supervised_token_count")}
                 for f in features if f["sequence_length"] > 2048]
    files = {"train_tokens.jsonl": jsonl_bytes(features), "long_over_2048.jsonl": jsonl_bytes(long_rows), "stats.json": json_bytes(stats)}
    manifest = {"version": config["version"], "config": config, "model_id": model["model_id"], "model_revision": model["revision"],
                "stats": stats, "source_sha256": fingerprints,
                "output_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
                "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
                "stored_arrays_unpadded": True, "labels_shifted_in_preprocessing": False,
                "all_training_ids_preserved": True, "all_saved_sql_decode_to_original_targets": True,
                "truncated_examples": 0, "dropped_examples": 0, "preview_passed": True,
                "packages": {name: version(name) for name in ("torch", "transformers", "tokenizers", "PyYAML")},
                "model_weights_loaded": False, "training_performed": False,
                "validation_examples_loaded": False, "final_examples_read": False}
    files["manifest.json"] = json_bytes(manifest)
    verify_sources(fingerprints)
    publish_files(project_path(config["output_dir"]), files)
    save(ROOT / "reports/sft_data_manifest.json", manifest)
    print(json.dumps({"status": "published", "output_dir": config["output_dir"], "stats": stats}, ensure_ascii=False, indent=2))


def verify(context):
    config, rows, lengths, tokenizer, fingerprints, _ = context
    directory = project_path(config["output_dir"])
    manifest = read(directory / "manifest.json")
    if manifest != read(ROOT / "reports/sft_data_manifest.json") or manifest["source_sha256"] != fingerprints:
        raise ValueError("已发布manifest或当前来源与冻结版本不一致")
    for name, expected in manifest["output_sha256"].items():
        expect_hash(directory / name, expected)
    features = read_rows(directory / "train_tokens.jsonl")
    if [f["source_index"] for f in features] != [r["source_index"] for r in rows]:
        raise ValueError("保存文件缺题、重复或顺序改变")
    for feature, row in zip(features, rows):
        verify_feature(feature, row, lengths[row["source_index"]], tokenizer)
    if feature_summary(features) != manifest["stats"]:
        raise ValueError("保存文件统计不一致")
    verify_sources(fingerprints)
    report = {"status": "passed", "verified_examples": len(features), "training_database_count": manifest["stats"]["database_count"],
              "manifest_sha256": digest(directory / "manifest.json"), "source_files_unchanged": True,
              "all_ids_and_original_targets_match": True, "all_masks_and_eos_checked": True,
              "labels_shifted_in_preprocessing": False, "truncated_examples": 0, "dropped_examples": 0,
              "model_weights_loaded": False, "training_performed": False,
              "validation_examples_loaded": False, "final_examples_read": False}
    save(ROOT / "reports/sft_data_verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(stage):
    started = time.perf_counter()
    context = load_context()
    config, rows, lengths, _, fingerprints, _ = context
    if stage == "check":
        full = [entry["full_reference_tokens"] for entry in lengths.values()]
        report = {"status": "ready", "training_examples": len(rows), "training_databases": 126,
                  "existing_max_length": max(full), "existing_over_2048": sum(x > 2048 for x in full),
                  "configured_max_length": config["max_sequence_length"], "source_sha256": fingerprints,
                  "model_weights_loaded": False, "training_performed": False,
                  "validation_examples_loaded": False, "final_examples_read": False}
        save(ROOT / "reports/sft_data_check.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, ensure_ascii=False, indent=2))
    elif stage == "preview":
        preview(context)
    elif stage == "build":
        build(context)
    else:
        verify(context)
    elapsed = time.perf_counter()-started
    save(ROOT / f"reports/sft_data_{stage}_timing.json", {
        "stage": stage, "wall_seconds": elapsed, "scope": "inside main; excludes interpreter startup", "device": "cpu",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "training_performed": False})
    print(f"SFT_DATA_{stage.upper()}: PASS; CPU wall={elapsed:.3f}s; no model/optimizer/backward", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("check", "preview", "build", "verify"))
    main(parser.parse_args().stage)
