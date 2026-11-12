"""Freeze 32 validation inputs before observing any model predictions.

Hardness quotas use largest remainders from the 625-example population.
For each hardness in easy/medium/hard/extra order, sort by source_index,
then random.Random(42).sample from that group; sort the final 32 by source_index.
This only reads validation data and metadata, never executes SQL or a model.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/eval"))
import evaluate_spider as evaluation

VERSION = "spider_baseline_pilot_v1"
SIZE = 32
SEED = 42
LEVELS = ("easy", "medium", "hard", "extra")
DIRECTORY = ROOT / "data/processed" / VERSION
FULL_INPUTS = ROOT / "data/processed/spider_inputs_v1/validation_inputs.jsonl"
REPORT = ROOT / "reports/baseline_pilot_manifest.json"


def allocate_counts(population, size):
    """Largest remainder, with stable LEVELS order to break ties."""
    if set(population) != set(LEVELS) or any(type(n) is not int or n < 0 for n in population.values()):
        raise ValueError("Expected nonnegative integer counts for four hardness groups")
    total = sum(population.values())
    if type(size) is not int or not 0 < size <= total:
        raise ValueError("Sample size must be positive and no larger than the population")
    quotas = {level: size * population[level] // total for level in LEVELS}
    remainder_order = sorted(LEVELS, key=lambda level: (-(size * population[level] % total), LEVELS.index(level)))
    for level in remainder_order[:size-sum(quotas.values())]:
        quotas[level] += 1
    return quotas


def choose_rows(rows, hardness, size=SIZE, seed=SEED):
    if len({row["source_index"] for row in rows}) != len(rows):
        raise ValueError("Repeated source indices in validation")
    if set(hardness) != {row["source_index"] for row in rows} or any(h not in LEVELS for h in hardness.values()):
        raise ValueError("Hardness mapping does not match validation records")
    population = {level: sum(h == level for h in hardness.values()) for level in LEVELS}
    quotas = allocate_counts(population, size)
    rng = random.Random(seed)
    selected = []
    for level in LEVELS:
        group = sorted((row for row in rows if hardness[row["source_index"]] == level), key=lambda row: row["source_index"])
        selected.extend(rng.sample(group, quotas[level]))
    return sorted(selected, key=lambda row: row["source_index"]), population, quotas


def build_expected():
    config, rows, fingerprints = evaluation.load_validation()
    schema_manifest = evaluation.read_json(evaluation.project_path(config["inputs_manifest"]))
    evaluation.expect_hash(FULL_INPUTS, schema_manifest["output_sha256"][FULL_INPUTS.name])
    full_inputs = evaluation.read_jsonl(FULL_INPUTS)
    expected_sources = {row["source_index"]: row for row in rows}
    inputs_lookup = {}
    for entry in full_inputs:
        index = entry.get("source_index")
        if type(index) is not int or index in inputs_lookup or index not in expected_sources:
            raise ValueError("Invalid or duplicate validation input ID")
        source = expected_sources[index]
        if set(entry) != {"source_index", "source_file", "db_id", "project_split", "messages"}:
            raise ValueError("Unexpected generation input fields; gold is forbidden")
        if entry["db_id"] != source["db_id"] or entry["project_split"] != "validation":
            raise ValueError("Generation input database/split mismatch")
        if [message.get("role") for message in entry["messages"]] != ["system", "user"]:
            raise ValueError("Generation input must contain only system and user messages")
        inputs_lookup[index] = entry
    if set(inputs_lookup) != set(expected_sources):
        raise ValueError("Incomplete validation generation inputs")
    hardness = evaluation.label_hardness(rows)
    selected, population, quotas = choose_rows(rows, hardness)
    if population != {"easy": 138, "medium": 247, "hard": 152, "extra": 88}:
        raise ValueError("The frozen validation population has unexpected hardness counts")
    inputs_payload = evaluation.jsonl_bytes([inputs_lookup[row["source_index"]] for row in selected])
    fingerprints = {**fingerprints, evaluation.relative(FULL_INPUTS): evaluation.digest(FULL_INPUTS),
                    evaluation.relative(Path(__file__)): evaluation.digest(Path(__file__))}
    manifest = {
        "version": VERSION, "seed": SEED, "example_count": SIZE,
        "sampling_method": "largest-remainder hardness quotas; easy/medium/hard/extra order; sorted source_index then shared Random(42).sample; final sorted source_index",
        "selected_source_indices": [row["source_index"] for row in selected],
        "selected_examples": [{"source_index": row["source_index"], "db_id": row["db_id"], "hardness": hardness[row["source_index"]]} for row in selected],
        "input_sha256": fingerprints,
        "output_sha256": {"inputs.jsonl": hashlib.sha256(inputs_payload).hexdigest()},
        "hardness_population_counts": population, "hardness_sample_counts": quotas,
        "validation_population_count": len(rows), "selected_database_count": len({row["db_id"] for row in selected}),
        "sample_scope": "fixed 32-example internal-validation pilot, not the complete 625-example evaluation",
        "maximum_generation_examples": SIZE, "maximum_evaluation_examples": SIZE,
        "prediction_fields": ["messages"], "gold_in_generation_inputs": False,
        "selection_uses_model_predictions": False, "final_examples_read": False,
        "sql_executed_during_preparation": False,
    }
    for filename, expected in fingerprints.items():
        evaluation.expect_hash(ROOT / filename, expected)
    return manifest, inputs_payload, selected, config, fingerprints


def publish():
    manifest, payload, _, _, _ = build_expected()
    files = {DIRECTORY / "manifest.json": evaluation.json_bytes(manifest), DIRECTORY / "inputs.jsonl": payload,
             REPORT: evaluation.json_bytes(manifest)}
    # Preflight all files before writing anything; never silently replace a sample.
    for path, content in files.items():
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"Frozen pilot differs; refusing overwrite: {path}")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        if not path.exists():
            path.write_bytes(content)
    print(json.dumps({"manifest": evaluation.relative(DIRECTORY / "manifest.json"),
                      "inputs": evaluation.relative(DIRECTORY / "inputs.jsonl"), "example_count": SIZE,
                      "hardness_sample_counts": manifest["hardness_sample_counts"],
                      "selected_database_count": manifest["selected_database_count"],
                      "selected_source_indices": manifest["selected_source_indices"],
                      "final_examples_read": False, "sql_executed": False}, ensure_ascii=False, indent=2))
    return manifest


def validate_published():
    """Recompute the deterministic selection and compare both frozen files."""
    expected, inputs_payload, rows, config, fingerprints = build_expected()
    manifest_path = DIRECTORY / "manifest.json"
    if manifest_path.read_bytes() != evaluation.json_bytes(expected):
        raise ValueError("Published pilot manifest differs from the deterministic 32-example selection")
    if (DIRECTORY / "inputs.jsonl").read_bytes() != inputs_payload:
        raise ValueError("Published pilot input bytes differ from frozen validation inputs")
    if REPORT.read_bytes() != manifest_path.read_bytes():
        raise ValueError("Pilot manifest report copy changed")
    fingerprints.update({evaluation.relative(manifest_path): evaluation.digest(manifest_path),
                         evaluation.relative(DIRECTORY / "inputs.jsonl"): evaluation.digest(DIRECTORY / "inputs.jsonl"),
                         evaluation.relative(REPORT): evaluation.digest(REPORT)})
    return expected, rows, config, fingerprints


if __name__ == "__main__":
    publish()
