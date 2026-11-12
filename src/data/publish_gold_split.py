"""Publish the audited training subset without changing raw or validation data."""
from __future__ import annotations

import json
from pathlib import Path

from check_gold_sql import ROOT, SPLIT_DIR, read_json, relative, sha256, write_json


def local_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Report path is outside the project")
    return path


def publish():
    summary = read_json(ROOT / "reports/gold_sql_check.json")
    if summary["mode"] != "full" or summary["record_count"] != 7000 or not summary["inputs_unchanged"] or summary["final_data_read"]:
        raise ValueError("A complete, unchanged, train/validation-only audit is required")
    audit = local_path(summary["audit_path"])
    inputs = read_json(audit.parent / "inputs.json")
    for filename, digest in inputs["input_sha256"].items():
        if sha256(local_path(filename)) != digest:
            raise ValueError(f"Audit input has changed: {filename}")
    artifacts = {}
    for name, value in summary["derived_paths"].items():
        path = local_path(value)
        if sha256(path) != summary["derived_sha256"][name]:
            raise ValueError(f"Derived file changed: {name}")
        artifacts[name] = path.read_bytes()
    original = {r["source_index"]: r for r in (json.loads(line) for line in (SPLIT_DIR / "train.jsonl").read_text(encoding="utf-8").splitlines())}
    clean = [json.loads(line) for line in artifacts["train_clean.jsonl"].decode("utf-8").splitlines()]
    quarantine = [json.loads(line) for line in artifacts["train_quarantine.jsonl"].decode("utf-8").splitlines()]
    duplicates = [json.loads(line) for line in artifacts["train_duplicates.jsonl"].decode("utf-8").splitlines()]
    all_ids = [r["source_index"] for r in clean + quarantine + duplicates]
    if sorted(all_ids) != sorted(original):
        raise ValueError("Training partitions are not exhaustive and disjoint")
    for row in clean + quarantine + duplicates:
        if {k: row[k] for k in original[row["source_index"]]} != original[row["source_index"]]:
            raise ValueError("Original training fields were rewritten")
    audit_rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    lookup = {r["source_index"]: r for r in audit_rows}
    if sorted(lookup) != list(range(7000)) or len(audit_rows) != 7000:
        raise ValueError("Audit is incomplete or duplicated")
    if any(lookup[r["source_index"]]["status"] != "ok" or not lookup[r["source_index"]]["row_count_complete"] for r in clean):
        raise ValueError("Clean training contains a failed/incomplete gold query")
    keys = [(r["db_id"], r["question"], r["query"]) for r in clean]
    if len(set(keys)) != len(keys):
        raise ValueError("Clean training still contains exact duplicates")
    destination = ROOT / "data/processed/spider_gold_v1"
    # Check every existing target before writing any new file; never overwrite a change.
    for name, payload in artifacts.items():
        path = destination / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different published data: {path}")
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["file_sha256"] != summary["derived_sha256"] or manifest["policy"] != summary["policy"]:
            raise ValueError("Published policy or hashes differ; create an explicit new dataset version")
        if manifest["input_sha256"] != inputs["input_sha256"]:
            raise ValueError("Published source differs; create an explicit new dataset version")
    else:
        manifest = {"version": "spider_gold_v1", "policy": summary["policy"],
                    "counts": summary["training_disposition"],
                    "input_sha256": inputs["input_sha256"], "file_sha256": summary["derived_sha256"],
                    "audit_path": relative(audit), "audit_sha256": sha256(audit),
                    "run_summary_path": relative(audit.parent / "summary.json"),
                    "train_path": relative(destination / "train_clean.jsonl"),
                    "validation_path": "data/processed/spider_v1/validation.jsonl",
                    "validation_count": 625, "final_data_read": False,
                    "all_original_training_fields_preserved": True,
                    "semantic_correctness_verified": False}
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        path = destination / name
        if not path.exists():
            path.write_bytes(payload)
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    # Versionable copy, while dataset files remain ignored by Git.
    report_path = ROOT / "reports/spider_gold_manifest.json"
    if report_path.exists() and report_path.read_bytes() != manifest_path.read_bytes():
        raise ValueError("Versionable manifest copy differs; refusing overwrite")
    if not report_path.exists():
        report_path.write_bytes(manifest_path.read_bytes())
    print(json.dumps({"published_directory": relative(destination), "counts": manifest["counts"],
                      "validation_count_unchanged": 625, "source_fields_preserved": True,
                      "final_data_read": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    publish()
