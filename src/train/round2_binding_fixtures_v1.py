"""Bounded, standard-library fixtures for round-two objective binding.

The caller runs this once inside its CPU acceptance. This module imports no ML
library, reads no real dataset, and never changes the baseline fixture files.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from src.train.sft_round2_binding_v1 import (
    NORMALIZATION_SPEC, build_binding, verify_binding, verify_group_manifest,
)


ROLES = ("data", "weights", "original_schedule", "weighted_groups", "config", "helper", "manifest")


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run_binding_checks(directory):
    """Create independent tiny fixtures, require positive acceptance/negative rejection.

    Refuses an existing directory. A report is saved on success or failure. No
    checkpoint payload, training data, model, SQL engine, or torch is loaded.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    checks = []
    report = {"status": "running", "scope": "standard-library-only synthetic fixtures",
              "checks": checks, "real_training_data_read": False, "model_loaded": False,
              "sql_executed": False, "checkpoint_payload_deserialized": False}
    files = {role: f"inputs/{role}.txt" for role in ROLES}
    content = {role: f"fixture:{role}:version=1\n".encode("ascii") for role in ROLES}

    def fixture(name, *, changed_role=None, omit_role=None):
        location = root / name
        (location / "inputs").mkdir(parents=True, exist_ok=False)
        for role, relative in files.items():
            if role == omit_role:
                continue
            payload = content[role]
            if role == changed_role:
                # A single changed byte; every other fixture byte stays identical.
                payload = b"F" + payload[1:]
            with (location / relative).open("xb") as stream:
                stream.write(payload)
        return location

    def reject(name, operation, *, exception=ValueError):
        try:
            operation()
        except exception as error:
            checks.append({"name": name, "status": "pass", "kind": "expected_rejection",
                           "exception": type(error).__name__, "message": str(error)})
        else:
            raise AssertionError(f"Invalid fixture was accepted: {name}")

    try:
        baseline = fixture("baseline")
        binding = build_binding(baseline, files)
        _write_json(baseline / "checkpoint_metadata.json", {"objective_binding": binding})
        saved = json.loads((baseline / "checkpoint_metadata.json").read_text(encoding="utf-8"))["objective_binding"]
        current = build_binding(baseline, files)
        assert verify_binding(saved, current)
        checks.append({"name": "binding_json_roundtrip_and_current_files", "status": "pass", "kind": "positive"})

        for role in ROLES:
            location = fixture(f"changed_{role}", changed_role=role)
            changed = build_binding(location, files)
            changed_roles = [name for name in ROLES if saved["files"][name] != changed["files"][name]]
            if changed_roles != [role]:
                raise AssertionError(f"Fixture mutated more than the requested role: {role}")
            _write_json(location / "current_binding.json", changed)
            reject(f"changed_actual_file_bytes_{role}", lambda value=changed: verify_binding(saved, value))

        spec_root = fixture("changed_normalization")
        spec = deepcopy(NORMALIZATION_SPEC)
        spec["extra_accumulation_divisor"] = True
        changed_spec = build_binding(spec_root, files, normalization_spec=spec)
        if changed_spec["normalization_spec_sha256"] == saved["normalization_spec_sha256"]:
            raise AssertionError("Changed specification did not change its canonical digest")
        _write_json(spec_root / "current_binding.json", changed_spec)
        reject("changed_normalization_with_valid_digest", lambda: verify_binding(saved, changed_spec))
        reject("missing_saved_binding", lambda: verify_binding(None, current))
        reject("missing_current_binding", lambda: verify_binding(saved, None))
        missing_field = deepcopy(current)
        missing_field.pop("normalization_spec_sha256")
        reject("missing_binding_field", lambda: verify_binding(saved, missing_field))
        missing_role = deepcopy(current)
        missing_role["files"].pop("weights")
        reject("missing_file_role", lambda: verify_binding(saved, missing_role))
        missing_root = fixture("missing_actual_weight_file", omit_role="weights")
        reject("missing_actual_weight_file", lambda: build_binding(missing_root, files), exception=FileNotFoundError)
        invalid_hash = deepcopy(current)
        invalid_hash["files"]["weights"]["sha256"] = "not-a-sha256"
        reject("invalid_file_sha256", lambda: verify_binding(saved, invalid_hash))
        bad_spec_hash = deepcopy(current)
        bad_spec_hash["normalization_spec_sha256"] = "0" * 64
        reject("normalization_digest_inconsistent_with_spec", lambda: verify_binding(saved, bad_spec_hash))
        extra_field = deepcopy(current)
        extra_field["ignored_extra"] = True
        reject("unexpected_binding_field", lambda: verify_binding(saved, extra_field))

        train = [
            {"source_index": 10, "project_split": "train", "db_id": "training_a", "supervised_token_count": 2},
            {"source_index": 11, "project_split": "train", "db_id": "training_b", "supervised_token_count": 3},
        ]
        weights = [
            {"source_index": 10, "project_split": "train", "db_id": "training_a", "loss_weight": 1, "focus_a": False},
            {"source_index": 11, "project_split": "train", "db_id": "training_b", "loss_weight": 2, "focus_a": True},
        ]
        original = [{"update": 1, "source_indices": [10, 11], "target_counts": [2, 3], "group_target_tokens": 5}]
        weighted = [{"update": 1, "source_indices": [10, 11], "target_counts": [2, 3],
                     "loss_weights": [1, 2], "unweighted_target_tokens": 5,
                     "weighted_target_denominator": 8, "composition": "mixed",
                     "micro_mean_loss_coefficients": [0.25, 0.75]}]
        inputs = {"train_rows_metadata": train, "weights_rows": weights,
                  "original_groups": original, "weighted_groups": weighted}
        manifest_root = root / "group_fixtures"
        manifest_root.mkdir(exist_ok=False)
        _write_json(manifest_root / "baseline.json", inputs)

        def verify(values):
            return verify_group_manifest(**values, expected_examples=2, expected_updates=1)

        result = verify(inputs)
        if result["weighted_target_tokens_total"] != 8 or result["last_group_examples"] != 2:
            raise AssertionError("Unexpected successful group audit")
        checks.append({"name": "valid_two_row_group", "status": "pass", "kind": "positive", "result": result})

        def bad_group(name, mutate):
            values = deepcopy(inputs)
            mutate(values)
            _write_json(manifest_root / f"{name}.json", values)
            reject(name, lambda: verify(values))

        bad_group("duplicate_weight_id", lambda value: value["weights_rows"][1].update(source_index=10))
        bad_group("missing_weight_row", lambda value: value["weights_rows"].pop())
        bad_group("weight_db_mismatch", lambda value: value["weights_rows"][1].update(db_id="other_db"))
        bad_group("non_train_weight_row", lambda value: value["weights_rows"][0].update(project_split="validation"))
        bad_group("non_train_metadata_row", lambda value: value["train_rows_metadata"][0].update(project_split="validation"))
        bad_group("boolean_weight", lambda value: value["weights_rows"][0].update(loss_weight=True))
        bad_group("zero_weight", lambda value: value["weights_rows"][0].update(loss_weight=0))
        bad_group("changed_source_order", lambda value: value["weighted_groups"][0].update(source_indices=[11, 10]))
        bad_group("wrong_original_target_count", lambda value: value["original_groups"][0].update(target_counts=[1, 4]))
        bad_group("wrong_weighted_target_count", lambda value: value["weighted_groups"][0].update(target_counts=[1, 4]))
        bad_group("wrong_group_denominator", lambda value: value["weighted_groups"][0].update(weighted_target_denominator=5))
        bad_group("wrong_micro_loss_coefficient", lambda value: value["weighted_groups"][0].update(micro_mean_loss_coefficients=[0.5, 0.5]))
        bad_group("wrong_group_weights", lambda value: value["weighted_groups"][0].update(loss_weights=[1, 1]))
        bad_group("wrong_update_index", lambda value: value["weighted_groups"][0].update(update=2))
        bad_group("wrong_group_composition", lambda value: value["weighted_groups"][0].update(composition="all_focus"))

        # Reading the baseline again establishes that negative cases left it intact.
        assert verify_binding(saved, build_binding(baseline, files))
        assert json.loads((manifest_root / "baseline.json").read_text(encoding="utf-8")) == inputs
        checks.append({"name": "baseline_files_and_group_fixture_preserved", "status": "pass", "kind": "positive"})
        report.update(status="pass", successful_checks=len(checks),
                      expected_rejections=sum(item["kind"] == "expected_rejection" for item in checks),
                      positive_checks=sum(item["kind"] == "positive" for item in checks))
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}",
                      successful_checks=len(checks),
                      expected_rejections=sum(item["kind"] == "expected_rejection" for item in checks),
                      positive_checks=sum(item["kind"] == "positive" for item in checks))
        raise
    finally:
        _write_json(root / "report.json", report)
