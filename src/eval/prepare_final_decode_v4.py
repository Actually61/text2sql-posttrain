"""Authorized CPU freeze for a transparent TEXT-decoding scoring revision.

This entry never imports a model or SQLite, and never executes scoring code.
It refuses to claim a new run before an explicit matching authorization exists.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/final_decode_v4.json"
EXPECTED_GENERATION_ROOTS = {"base": "outputs/final_holdout_v3/base", "sft": "outputs/final_holdout_v4/sft"}
DECODER = "utf8_surrogateescape_text_only_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(name):
    result = (ROOT / name).resolve()
    require(result.is_relative_to(ROOT), f"Path outside project: {name}")
    return result


def rel(filename):
    return Path(filename).resolve().relative_to(ROOT).as_posix()


def read(name):
    return json.loads(path(name).read_text(encoding="utf-8"))


def digest(name):
    with path(name).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(name, value, exclusive=True):
    with path(name).open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def verify(hashes):
    require(isinstance(hashes, dict) and hashes, "Missing source fingerprints")
    for name, expected in hashes.items():
        require(digest(name) == expected, f"Frozen file changed: {name}")


def check_authorization(cfg):
    require(cfg["generation_roots"] == EXPECTED_GENERATION_ROOTS and cfg["decoder_revision"] == DECODER,
            "Unexpected scoring scope")
    require(cfg["expected_examples"] == 1034 and cfg["expected_databases"] == 20
            and cfg["scoring_budget_seconds"] == 300, "Unexpected population/budget")
    require(cfg["training"] is False and cfg["regenerate_predictions"] is False
            and cfg["reuse_old_scores"] is False and cfg["automatic_retries"] == 0,
            "This revision may only freshly score fixed predictions")
    authorization_path = path(cfg["authorization_record"])
    require(authorization_path.is_file(), "Explicit user authorization for the decoder revision is missing; do not freeze or score")
    authorization = read(authorization_path)
    require(authorization.get("authorized") is True and authorization.get("authorization_source") == "user",
            "Decoder revision requires explicit user authorization")
    require(isinstance(authorization.get("user_message"), str) and authorization["user_message"].strip(),
            "Authorization must retain the actual user instruction")
    issued = datetime.fromisoformat(authorization["authorized_at_utc"].replace("Z", "+00:00"))
    require(issued.tzinfo is not None, "Authorization timestamp must include its UTC offset")
    scope = authorization.get("scope", {})
    require(scope.get("decoder") == DECODER and scope.get("models") == ["base", "sft"]
            and scope.get("examples_per_model") == 1034 and scope.get("allow_regeneration") is False
            and scope.get("allow_training") is False, "Authorization does not match this exact narrow scoring change")
    return authorization


def verify_authorized_freeze():
    """Used by the scorer before it creates a scoring directory or worker."""
    cfg = read(CONFIG)
    check_authorization(cfg)
    frozen = read(cfg["frozen_report"])
    require(frozen["status"] == "frozen_after_explicit_decoder_authorization"
            and frozen["config"] == cfg, "Scoring freeze/config mismatch")
    needed = [rel(CONFIG), rel(Path(__file__)), cfg["worker_source"], cfg["scorer_source"],
              cfg["authorization_record"], cfg["decoder_cpu_evidence"], cfg["worker_cpu_evidence"]]
    require(all(name in frozen["source_sha256"] for name in needed), "Scoring freeze is missing a required binding")
    verify(frozen["source_sha256"])
    checked = read(cfg["preflight_report"])
    require(checked["status"] == "pass" and checked["config"] == cfg
            and checked["frozen_report_sha256"] == digest(cfg["frozen_report"]),
            "Matching successful decoder CPU preflight required")
    return frozen


def check_minimal_worker(cfg):
    """Prove the new worker differs only at its description and TEXT boundary."""
    old = path("src/eval/spider_pair_worker.py").read_text(encoding="utf-8")
    expected = old.replace(
        "Read-only SQL execution with the pinned official Spider result comparator.",
        "Read-only SQL execution with lossless TEXT decoding and the pinned comparator.")
    expected = expected.replace('"decode_mode": "strict", "seconds": 0.0}',
        '"decode_mode": "utf-8/surrogateescape", "invalid_text_values": 0, "seconds": 0.0}')
    anchor = '        connection.text_factory = lambda raw: raw.decode("utf-8", errors="strict")'
    replacement = '''        # Only SQLite TEXT passes through text_factory. BLOB remains bytes.
        # Invalid UTF-8 is represented reversibly, never ignored or replaced.
        def decode_text(raw):
            value = raw.decode("utf-8", errors="surrogateescape")
            if any(0xDC80 <= ord(character) <= 0xDCFF for character in value):
                metadata["invalid_text_values"] += 1
            return value
        connection.text_factory = decode_text'''
    require(expected.count(anchor) == 1, "Original strict TEXT boundary is not unique")
    expected = expected.replace(anchor, replacement)
    actual = path(cfg["worker_source"]).read_text(encoding="utf-8")
    require(actual == expected, "New worker contains changes beyond the authorized minimal TEXT patch")
    for name in (cfg["worker_source"], cfg["scorer_source"], rel(Path(__file__))):
        source = path(name).read_text(encoding="utf-8")
        ast.parse(source, filename=name)
        compile(source, name, "exec")


def prepare():
    cfg = read(CONFIG)
    # This check deliberately precedes every directory creation and write.
    authorization = check_authorization(cfg)
    output = path(cfg["run_root"])
    require(not output.exists() and not path(cfg["frozen_report"]).exists()
            and not path(cfg["preflight_report"]).exists(), "Decoder scoring attempt already exists; do not overwrite or retry")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "failed", "config": cfg, "created_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_retries": 0, "CUDA_imported": False, "model_loaded": False, "SQL_executed": False,
        "training_performed": False, "predictions_regenerated": False}
    write(cfg["preflight_report"], report)
    try:
        check_minimal_worker(cfg)
        diagnostic = read(cfg["decoder_cpu_evidence"])
        require(diagnostic["status"] == "pass"
                and diagnostic["invalid_utf8_equivalent_to_original_strict_protocol"] is False
                and diagnostic["final_scoring_or_worker_change_requires_separate_authorization"] is True
                and diagnostic["database_bytes_unchanged"] is True
                and diagnostic["protected_sources_unchanged"] is True,
                "Decoder expression evidence did not pass or misstates equivalence")
        require(diagnostic["source_sha256_before"] == diagnostic["source_sha256_after"]
                and diagnostic["database_sha256_before"] == diagnostic["database_sha256_after"],
                "CPU diagnostic changed its protected evidence")
        verify(diagnostic["source_sha256_after"])
        integration = read(cfg["worker_cpu_evidence"])
        require(integration["status"] == "pass" and integration["protected_sources_unchanged"] is True
                and integration["fixture_bytes_unchanged"] is True
                and integration["final_database_accessed"] is False
                and integration["final_scoring_performed"] is False,
                "Actual worker synthetic integration did not pass")
        require(integration["source_sha256_before"] == integration["source_sha256_after"]
                and integration["source_sha256_after"].get(cfg["worker_source"]) == digest(cfg["worker_source"]),
                "Worker integration does not bind the current exact worker")
        verify(integration["source_sha256_after"])
        hashes = {}

        def remember(name, expected=None):
            filename = rel(path(name))
            value = digest(filename)
            require(expected is None or value == expected, f"Input hash mismatch: {filename}")
            require(filename not in hashes or hashes[filename] == value, f"Hash conflict: {filename}")
            hashes[filename] = value

        freeze_names = ["reports/final_holdout_v1_freeze.json"]
        for arm in ("base", "sft"):
            generation_cfg = read(cfg["generation_configs"][arm])
            generation_checked = read(generation_cfg["preflight_report"])
            require(generation_checked["status"] == "pass" and generation_checked["config"] == generation_cfg,
                    f"Generation preflight incomplete: {arm}")
            require(generation_checked["frozen_report_sha256"] == digest(generation_cfg["frozen_report"]),
                    f"Generation freeze/preflight differs: {arm}")
            freeze_names.append(generation_cfg["frozen_report"])
            remember(cfg["generation_configs"][arm]); remember(generation_cfg["preflight_report"])
        for name in dict.fromkeys(freeze_names):
            old_freeze = read(name)
            verify(old_freeze["source_sha256"])
            for filename, expected in old_freeze["source_sha256"].items():
                remember(filename, expected)
            remember(name)
        parent = read("reports/final_holdout_v1_preflight.json")
        require(parent["status"] == "pass", "Original prepared final data did not pass")
        verify(parent["output_sha256"])
        for name, expected in parent["output_sha256"].items():
            remember(name, expected)
        remember("reports/final_holdout_v1_preflight.json")
        remember("data/processed/spider_final_holdout_v1/manifest.json", parent["input_manifest_sha256"])
        # Preserve all files from the failed strict scoring attempt, without
        # parsing final question/gold contents or accepting its partial score.
        old_score = read(path(cfg["preserved_failed_score_dir"]) / "summary.json")
        require(old_score["status"] == "failed" and old_score["complete_final_scored"] is False,
                "Original strict failure evidence is missing")
        for item in path(cfg["preserved_failed_score_dir"]).rglob("*"):
            if item.is_file():
                remember(item)
        for name in (rel(CONFIG), rel(Path(__file__)), cfg["worker_source"], cfg["scorer_source"],
                     cfg["authorization_record"], cfg["decoder_cpu_evidence"], cfg["worker_cpu_evidence"],
                     "src/diagnostics/check_sql_decode_v1.py"):
            remember(name)
        for name, expected in diagnostic["source_sha256_after"].items():
            remember(name, expected)
        for name, expected in integration["source_sha256_after"].items():
            remember(name, expected)
        frozen = {"status": "frozen_after_explicit_decoder_authorization", "config": cfg,
            "created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes,
            "authorization_sha256": hashes[cfg["authorization_record"]],
            "authorization_source": authorization["authorization_source"],
            "decoder_revision": DECODER, "equivalent_to_original_strict_failure_policy": False,
            "minimal_worker_diff_verified": True, "both_models_use_same_new_scoring_protocol": True,
            "old_failed_strict_scores_reused": False, "models_or_queries_changed": False}
        write(cfg["frozen_report"], frozen)
        for name in (rel(CONFIG), rel(Path(__file__)), cfg["worker_source"], cfg["scorer_source"], cfg["authorization_record"]):
            destination = output / "snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path(name), destination)
        report.update(status="pass", frozen_report_sha256=digest(cfg["frozen_report"]),
            authorization_sha256=digest(cfg["authorization_record"]), protected_file_count=len(hashes),
            decoder_expression_fixture_passed=True, worker_minimal_diff_verified=True,
            actual_worker_synthetic_integration_passed=True,
            worker_or_full_scoring_entry_executed_by_prepare=False, both_generation_freezes_verified=True,
            original_failed_score_preserved=True, generation_roots=cfg["generation_roots"],
            generation_completion_policy="Each arm must be complete with successful controller before scoring; CPU prepare does not generate or require pending SFT outputs",
            scoring_protocol_changed_only_at_TEXT_result_decoding=True,
            original_strict_decoding_equivalence_claimed=False)
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["cpu_preflight_seconds"] = time.perf_counter() - started
        write(cfg["preflight_report"], report, exclusive=False)
    print(json.dumps({key: report[key] for key in ("status", "protected_file_count", "cpu_preflight_seconds")}), flush=True)


if __name__ == "__main__":
    prepare()
