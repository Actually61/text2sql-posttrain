"""Standard-library-only objective binding and weighted schedule validation.

No fixed project paths, ML imports, checkpoint payload loading, or training.
Callers supply current files and compare this binding BEFORE restoring state.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath


VERSION = "sft_round2_objective_binding_v1"
NORMALIZATION_SPEC = {
    "unit": "accumulation_group",
    "numerator": "sum_example_weight_times_valid_token_loss",
    "denominator": "sum_example_weight_times_valid_token_count",
    "ignore_index": -100,
    "causal_shift": 1,
    "extra_accumulation_divisor": False,
    "valid_positions": "assistant_answer_and_eos_only",
}
_BINDING_KEYS = {"version", "files", "normalization_spec", "normalization_spec_sha256"}


def file_digest(path):
    """SHA256 of actual file bytes, without importing a model or loading pickle."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value, label):
    _require(type(value) is str and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"Invalid SHA256: {label}")


def _relative_path(value):
    _require(type(value) is str and bool(value), "File path must be a nonempty string")
    _require("\x00" not in value, "NUL in file path")
    windows = PureWindowsPath(value)
    _require(not windows.drive and not windows.root, "Only project-relative paths are allowed")
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    _require(all(part not in ("", ".", "..") for part in parts), "Path must be canonical and relative")
    _require(not PurePosixPath(normalized).is_absolute(), "Absolute path is forbidden")
    return normalized


def _spec_copy_and_digest(spec):
    _require(type(spec) is dict and set(spec) == set(NORMALIZATION_SPEC),
             "Unexpected normalization specification fields")
    for key, reference in NORMALIZATION_SPEC.items():
        _require(type(spec[key]) is type(reference), f"Invalid normalization field type: {key}")
        if type(spec[key]) is str:
            _require(bool(spec[key]), f"Empty normalization field: {key}")
    canonical = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.loads(canonical), hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_binding(binding):
    _require(type(binding) is dict and set(binding) == _BINDING_KEYS, "Unexpected or missing binding fields")
    _require(binding["version"] == VERSION, "Unknown objective binding version")
    files = binding["files"]
    _require(type(files) is dict and bool(files), "Binding must contain a nonempty file mapping")
    for role, record in files.items():
        _require(type(role) is str and bool(role), "Invalid file role")
        _require(type(record) is dict and set(record) == {"path", "sha256"}, f"Invalid file record: {role}")
        _require(_relative_path(record["path"]) == record["path"], f"Noncanonical saved path: {role}")
        _sha(record["sha256"], role)
    _, expected = _spec_copy_and_digest(binding["normalization_spec"])
    _sha(binding["normalization_spec_sha256"], "normalization_spec")
    _require(binding["normalization_spec_sha256"] == expected, "Normalization specification digest mismatch")


def build_binding(root, files, *, normalization_spec=None):
    """Read caller-specified project files and return JSON-only objective metadata.

    `files` maps semantic roles to project-relative paths; no role is inferred from
    a checkpoint. Resolved files must remain inside the supplied project root.
    """
    project = Path(root).resolve(strict=True)
    _require(project.is_dir(), "Project root must be a directory")
    _require(type(files) is dict and bool(files), "files must be a nonempty mapping")
    records = {}
    for role, relative in files.items():
        _require(type(role) is str and bool(role), "Invalid file role")
        canonical = _relative_path(relative)
        path = (project / canonical).resolve(strict=True)
        _require(path.is_relative_to(project) and path.is_file(), f"File escapes project or is not a file: {role}")
        records[role] = {"path": canonical, "sha256": file_digest(path)}
    spec, spec_hash = _spec_copy_and_digest(NORMALIZATION_SPEC if normalization_spec is None else normalization_spec)
    result = {"version": VERSION, "files": records, "normalization_spec": spec,
              "normalization_spec_sha256": spec_hash}
    _validate_binding(result)
    return result


def verify_binding(saved, current):
    """Reject missing, malformed, or unequal bindings before any state restoration."""
    _validate_binding(saved)
    _validate_binding(current)
    _require(saved == current, "Checkpoint/current objective binding mismatch")
    return True


def _index_rows(rows, label, expected_examples):
    _require(type(rows) is list and len(rows) == expected_examples, f"Wrong row count: {label}")
    indexed = {}
    for row in rows:
        _require(type(row) is dict, f"Non-object row: {label}")
        index = row.get("source_index")
        _require(type(index) is int and index >= 0, f"Invalid source_index: {label}")
        _require(index not in indexed, f"Duplicate source_index in {label}: {index}")
        _require(row.get("project_split") == "train", f"Non-training row in {label}: {index}")
        _require(type(row.get("db_id")) is str and bool(row["db_id"]), f"Invalid db_id in {label}: {index}")
        indexed[index] = row
    return indexed


def _positive_integer(value, label):
    _require(type(value) is int and value > 0, f"Invalid positive integer: {label}")


def verify_group_manifest(train_rows_metadata, weights_rows, original_groups, weighted_groups,
                          *, expected_examples=6364, expected_updates=796):
    """Validate coverage, train DB identity, order, counts, weights and coefficients.

    Inputs are parsed lists. Training rows need source_index, project_split, db_id,
    supervised_token_count. Schedule rows use the frozen plan's group structure.
    Tiny fixtures may override the two expected counts; no actual files are read.
    """
    _positive_integer(expected_examples, "expected_examples")
    _positive_integer(expected_updates, "expected_updates")
    train = _index_rows(train_rows_metadata, "training metadata", expected_examples)
    weights = _index_rows(weights_rows, "weights", expected_examples)
    _require(set(train) == set(weights), "Training/weight ID coverage differs")
    for index, row in train.items():
        _positive_integer(row.get("supervised_token_count"), f"answer tokens at {index}")
        _require(weights[index]["db_id"] == row["db_id"], f"Training/weight DB mismatch: {index}")
        weight = weights[index].get("loss_weight")
        _require(type(weight) is int and weight in (1, 2), f"Weight must be integer 1 or 2: {index}")
        if "focus_a" in weights[index]:
            _require(type(weights[index]["focus_a"]) is bool
                     and weights[index]["focus_a"] == (weight == 2), f"Weight/focus flag differs: {index}")
    _require(type(original_groups) is list and len(original_groups) == expected_updates, "Original update count differs")
    _require(type(weighted_groups) is list and len(weighted_groups) == expected_updates, "Weighted update count differs")
    visited = []
    compositions = {"mixed": 0, "all_focus": 0, "all_other": 0}
    weighted_total = 0
    for ordinal, (original, weighted) in enumerate(zip(original_groups, weighted_groups), 1):
        _require(type(original) is dict and type(weighted) is dict, f"Invalid group: {ordinal}")
        _require(type(original.get("update")) is int and original["update"] == ordinal
                 and type(weighted.get("update")) is int and weighted["update"] == ordinal,
                 f"Update order differs: {ordinal}")
        ids = original.get("source_indices")
        _require(type(ids) is list and bool(ids) and all(type(index) is int and index in train for index in ids),
                 f"Invalid original group IDs: {ordinal}")
        _require(type(weighted.get("source_indices")) is list
                 and all(type(index) is int for index in weighted["source_indices"])
                 and weighted["source_indices"] == ids, f"Weighted source order differs: {ordinal}")
        counts = [train[index]["supervised_token_count"] for index in ids]
        group_weights = [weights[index]["loss_weight"] for index in ids]
        for owner, name, expected in ((original, "target_counts", counts),
                                      (weighted, "target_counts", counts),
                                      (weighted, "loss_weights", group_weights)):
            values = owner.get(name)
            _require(type(values) is list and all(type(value) is int for value in values)
                     and values == expected, f"Group {name} differs: {ordinal}")
        total = sum(counts)
        denominator = sum(weight * count for weight, count in zip(group_weights, counts))
        for owner, name, expected in ((original, "group_target_tokens", total),
                                      (weighted, "unweighted_target_tokens", total),
                                      (weighted, "weighted_target_denominator", denominator)):
            _require(type(owner.get(name)) is int and owner[name] == expected, f"Group {name} differs: {ordinal}")
        expected_coefficients = [weight * count / denominator for weight, count in zip(group_weights, counts)]
        actual_coefficients = weighted.get("micro_mean_loss_coefficients")
        _require(type(actual_coefficients) is list and len(actual_coefficients) == len(ids),
                 f"Coefficient count differs: {ordinal}")
        for actual, expected in zip(actual_coefficients, expected_coefficients):
            _require(type(actual) in (float, int) and math.isfinite(actual)
                     and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12),
                     f"Group loss coefficient differs: {ordinal}")
        composition = "mixed" if len(set(group_weights)) > 1 else ("all_focus" if group_weights[0] == 2 else "all_other")
        _require(weighted.get("composition") == composition, f"Group composition differs: {ordinal}")
        compositions[composition] += 1
        visited.extend(ids)
        weighted_total += denominator
    _require(len(visited) == len(set(visited)) == expected_examples and set(visited) == set(train),
             "Schedule must cover every training ID exactly once")
    return {"examples": expected_examples, "updates": expected_updates,
            "databases": len({row["db_id"] for row in train.values()}),
            "group_composition_counts": compositions,
            "weighted_target_tokens_total": weighted_total,
            "last_group_examples": len(original_groups[-1]["source_indices"])}
