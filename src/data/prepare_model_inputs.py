"""Build schema + question prompts and measure the local tokenizer, CPU only.

--stage preview: inspect three fixed training examples without publishing data.
--stage build: publish all train/validation inputs, targets and token statistics.
--stage verify: rebuild in memory, compare every published byte and check hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from schema_serialization import serialize_schema

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/schema_inputs.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(rows):
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)).encode("utf-8")


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def safe_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Project path escaped workspace")
    return path


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen input changed: {path}")
    return actual


def load_sources():
    """Use only frozen clean train, unchanged validation and selected schema metadata."""
    import yaml
    config = load_json(CONFIG)
    if config["truncate"] is not False:
        raise ValueError("This data version forbids truncation")
    gold_path = safe_path(config["gold_manifest"])
    split_path = safe_path(config["split_manifest"])
    gold = load_json(gold_path)
    require_hash(split_path, gold["input_sha256"][relative(split_path)])
    split = load_json(split_path)
    schema_path = safe_path(config["schema_file"])
    require_hash(schema_path, split["source_sha256"][relative(schema_path)])
    rows_by_split = {}
    paths = [CONFIG, gold_path, split_path, schema_path]
    for role, count in (("train", 6364), ("validation", 625)):
        path = safe_path(config[f"{role}_file"])
        expected = gold["file_sha256"]["train_clean.jsonl"] if role == "train" else gold["input_sha256"][relative(path)]
        require_hash(path, expected)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != count or any(row["project_split"] != role for row in rows):
            raise ValueError(f"Unexpected {role} membership or count")
        if {row["db_id"] for row in rows} != set(split["splits"][role]["db_ids"]):
            raise ValueError(f"Unexpected {role} databases")
        rows_by_split[role] = rows
        paths.append(path)
    train_ids = {row["db_id"] for row in rows_by_split["train"]}
    val_ids = {row["db_id"] for row in rows_by_split["validation"]}
    if train_ids & val_ids or (train_ids | val_ids) & set(split["splits"]["final_eval"]["db_ids"]):
        raise ValueError("Database split overlap")
    all_indices = [row["source_index"] for rows in rows_by_split.values() for row in rows]
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Duplicate source indices across selected records")
    allowed = train_ids | val_ids
    # tables.json is shared metadata. Do not serialize unselected database schemas.
    selected_schemas = [schema for schema in load_json(schema_path) if schema["db_id"] in allowed]
    schemas = {schema["db_id"]: schema for schema in selected_schemas}
    if len(schemas) != len(selected_schemas) or set(schemas) != allowed:
        raise ValueError("Missing or repeated selected schema")
    model_path = safe_path(config["model_config"])
    prompt_path = safe_path(config["prompt_config"])
    model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    prompt = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    model_manifest_path = ROOT / "reports/model_manifest.json"
    model_manifest = load_json(model_manifest_path)
    if (model["model_id"], model["revision"]) != (model_manifest["model_id"], model_manifest["revision"]):
        raise ValueError("Model revision changed")
    model_dir = safe_path(model["local_dir"])
    tokenizer_paths = [model_dir / name for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json")]
    for path in tokenizer_paths:
        record = model_manifest["verified_files"][path.name]
        data = path.read_bytes()
        # Original download manifest uses Git blob SHA1 for these small files.
        digest = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
        if record["algorithm"] != "sha1" or len(data) != record["size"] or digest != record["hash"]:
            raise ValueError(f"Tokenizer/config no longer matches downloaded model: {path.name}")
    paths.extend([model_path, prompt_path, model_manifest_path, *tokenizer_paths,
                  Path(__file__), Path(__file__).with_name("schema_serialization.py")])
    fingerprints = {relative(path): sha256(path) for path in paths}
    return config, rows_by_split, schemas, model, prompt, fingerprints


def build_messages(schema_text, question, prompt):
    """No gold parameter: prediction context is built only from schema/question."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question is empty")
    return [{"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user_template"].format(schema=schema_text, question=question)}]


def token_lengths(tokenizer, messages, gold_sql):
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    full_ids = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": gold_sql}],
                                             tokenize=True, add_generation_prompt=False, return_dict=False)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt tokens are not a prefix of the full reference conversation")
    if len(full_ids) <= len(prompt_ids) or tokenizer.eos_token_id not in full_ids[len(prompt_ids):]:
        raise ValueError("Assistant suffix or EOS is missing")
    return {"prompt_tokens": len(prompt_ids), "full_reference_tokens": len(full_ids),
            "assistant_suffix_tokens": len(full_ids) - len(prompt_ids),
            "sql_text_tokens": len(tokenizer.encode(gold_sql, add_special_tokens=False))}


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize empty lengths")
    result = {"min": ordered[0], "mean": round(sum(ordered) / len(ordered), 3), "max": ordered[-1]}
    for percentile in (50, 90, 95, 99):
        result[f"p{percentile}"] = ordered[max(0, math.ceil(percentile / 100 * len(ordered)) - 1)]
    return result


def summarize(lengths, config):
    summary = {"count": len(lengths), "database_count": len({r["db_id"] for r in lengths}),
               "distributions": {field: distribution([row[field] for row in lengths]) for field in
                                  ("schema_tokens", "question_tokens", "prompt_tokens", "sql_text_tokens", "assistant_suffix_tokens", "full_reference_tokens")}}
    reserve = config["generation_reserve_tokens"]
    summary["budgets"] = []
    for budget in config["length_budgets"]:
        prompt_over = [r["source_index"] for r in lengths if r["prompt_tokens"] > budget]
        reserve_over = [r["source_index"] for r in lengths if r["prompt_tokens"] + reserve > budget]
        full_over = [r["source_index"] for r in lengths if r["full_reference_tokens"] > budget]
        summary["budgets"].append({"total_token_budget": budget, "prompt_over_count": len(prompt_over),
            "prompt_plus_reserve_over_count": len(reserve_over), "full_reference_over_count": len(full_over),
            "prompt_plus_reserve_coverage_percent": round(100 * (1 - len(reserve_over)/len(lengths)), 3),
            "full_reference_coverage_percent": round(100 * (1 - len(full_over)/len(lengths)), 3),
            "prompt_over_source_indices": prompt_over, "prompt_plus_reserve_over_source_indices": reserve_over,
            "full_reference_over_source_indices": full_over})
    summary["sql_text_exceeds_generation_reserve"] = sum(r["sql_text_tokens"] > reserve for r in lengths)
    summary["assistant_suffix_exceeds_generation_reserve"] = sum(r["assistant_suffix_tokens"] > reserve for r in lengths)
    summary["longest_prompt_examples"] = sorted(lengths, key=lambda r: (-r["prompt_tokens"], r["source_index"]))[:5]
    return summary


def load_tokenizer(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(safe_path(model["local_dir"]), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Local tokenizer has no chat template")
    return tokenizer


def preview_examples(rows, schema_texts, tokenizer, prompt):
    # Three training examples from three databases, chosen without scoring model outputs.
    selected, seen = [], set()
    for row in rows:
        if row["db_id"] not in seen:
            selected.append(row)
            seen.add(row["db_id"])
        if len(selected) == 3:
            break
    examples = []
    for row in selected:
        schema_text = schema_texts[row["db_id"]]
        messages = build_messages(schema_text, row["question"], prompt)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        exact = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        if tokenizer.encode(rendered, add_special_tokens=False) != exact:
            raise ValueError("Rendered prompt tokenization disagrees with chat template")
        lengths = token_lengths(tokenizer, messages, row["query"])
        lengths["question_tokens"] = len(tokenizer.encode(row["question"], add_special_tokens=False))
        lengths["schema_tokens"] = len(tokenizer.encode(schema_text, add_special_tokens=False))
        examples.append({"source_index": row["source_index"], "db_id": row["db_id"],
                         "question": row["question"], "schema_text": schema_text, "messages": messages,
                         "rendered_generation_prompt": rendered, "training_target_sql": row["query"],
                         "lengths": lengths,
                         "target_is_not_part_of_generation_prompt": True})
    return examples


def build_artifacts(config, rows_by_split, schemas, model, prompt, fingerprints, tokenizer):
    schema_texts = {db_id: serialize_schema(schema) for db_id, schema in sorted(schemas.items())}
    schema_lengths = {db_id: len(tokenizer.encode(text, add_special_tokens=False)) for db_id, text in schema_texts.items()}
    schemas_out = [{"db_id": db_id, "schema_text": text, "schema_tokens": schema_lengths[db_id],
                    "table_count": len(schemas[db_id]["table_names_original"]),
                    "column_count": len(schemas[db_id]["column_names_original"]) - 1,
                    "primary_key_column_count": len(schemas[db_id]["primary_keys"]),
                    "foreign_key_count": len(schemas[db_id]["foreign_keys"])} for db_id, text in schema_texts.items()]
    payloads = {"schemas.jsonl": jsonl_bytes(schemas_out)}
    stats = {"version": config["version"], "model_id": model["model_id"], "model_revision": model["revision"],
             "schema_database_count": len(schemas), "schema_format": config["schema_format"],
             "generation_reserve_tokens": config["generation_reserve_tokens"],
             "percentile_method": "nearest rank: sorted_values[ceil(p*n)-1]",
             "no_truncation": True, "no_dropped_examples": True,
             "shared_schema_metadata_read": True, "final_examples_read": False,
             "final_database_schemas_serialized": False, "model_weights_loaded": False,
             "all_token_prefixes_match": True, "validation_used_for_gradient_training": False,
             "splits": {}}
    for role, rows in rows_by_split.items():
        inputs, references, sft, lengths = [], [], [], []
        for i, row in enumerate(rows):
            meta = {key: row[key] for key in ("source_index", "source_file", "db_id", "project_split")}
            messages = build_messages(schema_texts[row["db_id"]], row["question"], prompt)
            # Only the messages field is sent to a future generation API.
            inputs.append({**meta, "messages": messages})
            references.append({**meta, "gold_sql": row["query"]})
            measured = token_lengths(tokenizer, messages, row["query"])
            # Measure the original question directly, without parsing prompt markers.
            measured["question_tokens"] = len(tokenizer.encode(row["question"], add_special_tokens=False))
            lengths.append({**meta, **measured, "schema_tokens": schema_lengths[row["db_id"]]})
            if role == "train":
                sft.append({**meta, "prompt": messages, "completion": [{"role": "assistant", "content": row["query"]}]})
            if (i + 1) % 1000 == 0 or i + 1 == len(rows):
                print(f"{role}: {i+1}/{len(rows)} tokenized", flush=True)
        payloads[f"{role}_inputs.jsonl"] = jsonl_bytes(inputs)
        payloads[f"{role}_references.jsonl"] = jsonl_bytes(references)
        payloads[f"{role}_lengths.jsonl"] = jsonl_bytes(lengths)
        if role == "train":
            payloads["train_sft.jsonl"] = jsonl_bytes(sft)
        stats["splits"][role] = summarize(lengths, config)
    examples = preview_examples(rows_by_split["train"], schema_texts, tokenizer, prompt)
    payloads["examples.json"] = json_bytes(examples)
    payloads["token_stats.json"] = json_bytes(stats)
    manifest = {"version": config["version"], "input_sha256": fingerprints,
                "output_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
                "config": config, "model_id": model["model_id"], "model_revision": model["revision"],
                "packages": {name: version(name) for name in ("transformers", "tokenizers", "PyYAML")},
                "train_examples": 6364, "validation_examples": 625,
                "prediction_fields": ["messages"], "train_targets_separate_from_prompt": True,
                "validation_references_separate_from_inputs": True,
                "truncation_applied": False, "schema_pruned_using_gold": False,
                "shared_schema_metadata_read": True, "final_examples_read": False,
                "final_database_schemas_serialized": False,
                "training_note": "train_sft contains prompt/completion messages; future trainer must implement and verify completion-only loss masking"}
    payloads["manifest.json"] = json_bytes(manifest)
    return payloads, stats, examples


def verify_unchanged(fingerprints):
    for name, digest in fingerprints.items():
        require_hash(ROOT / name, digest)


def main(stage):
    started = time.perf_counter()
    config, rows, schemas, model, prompt, fingerprints = load_sources()
    tokenizer = load_tokenizer(model)
    reports = ROOT / "reports"
    if stage == "preview":
        selected_ids = set()
        for row in rows["train"]:
            selected_ids.add(row["db_id"])
            if len(selected_ids) == 3:
                break
        texts = {db_id: serialize_schema(schemas[db_id]) for db_id in selected_ids}
        examples = preview_examples(rows["train"], texts, tokenizer, prompt)
        verify_unchanged(fingerprints)
        (reports / "schema_input_preview.json").write_bytes(json_bytes(examples))
        (reports / "schema_input_preview.txt").write_text(examples[0]["rendered_generation_prompt"], encoding="utf-8")
        print(examples[0]["rendered_generation_prompt"], flush=True)
        print("Training target (separate):", examples[0]["training_target_sql"], flush=True)
        print("Token counts:", examples[0]["lengths"], flush=True)
        return
    payloads, stats, examples = build_artifacts(config, rows, schemas, model, prompt, fingerprints, tokenizer)
    verify_unchanged(fingerprints)
    output = safe_path(config["output_dir"])
    for name, payload in payloads.items():
        path = output / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different frozen output: {path}")
        if stage == "verify" and not path.is_file():
            raise ValueError(f"Missing published file: {path}")
    if stage == "build":
        output.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            path = output / name
            if not path.exists():
                path.write_bytes(payload)
        for name, payload in (("schema_token_stats.json", payloads["token_stats.json"]),
                              ("schema_inputs_manifest.json", payloads["manifest.json"]),
                              ("schema_input_examples.json", payloads["examples.json"])):
            (reports / name).write_bytes(payload)
    else:
        verification = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "all_published_files_rebuilt_identically": True,
                        "file_count": len(payloads), "source_files_unchanged": True, "train_examples": 6364, "validation_examples": 625,
                        "all_6989_token_prefixes_match": True, "no_truncation_or_filtering": True,
                        "shared_schema_metadata_read": True, "final_examples_read": False, "final_database_schemas_serialized": False,
                        "manifest_sha256": sha256(output / "manifest.json")}
        (reports / "schema_inputs_verification.json").write_bytes(json_bytes(verification))
    compact = {"stage": stage, "output_dir": relative(output), "seconds": round(time.perf_counter()-started, 3),
               "databases": len(schemas), "splits": {role: {"count": values["count"], "distributions": values["distributions"],
                      "budgets": [{k: v for k, v in budget.items() if not k.endswith("source_indices")} for budget in values["budgets"]],
                      "assistant_suffix_exceeds_generation_reserve": values["assistant_suffix_exceeds_generation_reserve"]}
                      for role, values in stats["splits"].items()}}
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preview", "build", "verify"), required=True)
    main(parser.parse_args().stage)
