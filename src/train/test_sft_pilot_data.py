"""CPU-only selection tests against frozen training rows; no model or CUDA work."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.sft_features import collate_sft_batch
from src.train.sft_pilot_data import BUCKET_UPPER_BOUNDS, select_pilot


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    started = time.perf_counter()
    train_path = ROOT / "data/processed/spider_sft_tokens_v1/train_tokens.jsonl"
    manifest_path = ROOT / "data/processed/spider_sft_tokens_v1/manifest.json"
    report_path = ROOT / "reports/sft_pilot_data_cpu_checks.json"
    report = {"status": "started", "utc": datetime.now(timezone.utc).isoformat(),
              "model_loaded": False, "gpu_used": False, "validation_examples_loaded": False,
              "final_examples_read": False, "source_sha256": {name: sha256(ROOT / name) for name in (
                  "src/train/sft_pilot_data.py", "src/train/test_sft_pilot_data.py", "src/data/sft_features.py",
                  "data/processed/spider_sft_tokens_v1/manifest.json", "data/processed/spider_sft_tokens_v1/train_tokens.jsonl")}}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(train_path) != manifest["output_sha256"]["train_tokens.jsonl"]:
            raise AssertionError("Frozen training data changed")
        with train_path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        selection = select_pilot(rows)
        assert selection == select_pilot(rows)
        assert selection == select_pilot(list(reversed(rows)))
        assert selection["selected_source_indices"] != select_pilot(rows, seed=43)["selected_source_indices"]
        assert selection["training_source_count"] == 6364
        assert selection["candidate_bucket_counts"] == {"512": 3504, "1024": 2144, "2048": 634, "4096": 82}
        assert len(selection["selected_source_indices"]) == len(set(selection["selected_source_indices"])) == 32
        assert selection["bucket_counts"] == {str(b): 8 for b in BUCKET_UPPER_BOUNDS}
        longest, answer = selection["required_examples"]["longest_input"], selection["required_examples"]["longest_answer"]
        assert (longest["source_index"], longest["sequence_length"], longest["group_index"], longest["microbatch_index"]) == (3698, 3761, 2, 7)
        assert (answer["source_index"], answer["supervised_token_count"]) == (3293, 202)
        by_id = {r["source_index"]: r for r in rows}
        verified = []
        for group in selection["groups"]:
            assert len(group["source_indices"]) == 8
            assert group["bucket_counts"] == {str(b): 2 for b in BUCKET_UPPER_BOUNDS}
            target_total = 0
            for identifier in group["source_indices"]:
                row = by_id[identifier]
                batch = collate_sft_batch([row], manifest["pad_token_id"], 4096)
                assert list(batch["input_ids"].shape) == [1, row["sequence_length"]]
                assert batch["input_ids"][0].tolist() == row["input_ids"]
                assert batch["labels"][0].tolist() == row["labels"]
                assert bool((batch["attention_mask"] == 1).all())
                count = int((batch["labels"][:, 1:] != -100).sum())
                assert count == row["supervised_token_count"]
                target_total += count
                verified.append(identifier)
            assert target_total == group["group_target_tokens"]
        invalid = []
        examples = (
            ("empty", []),
            ("duplicate_source_id", rows + [rows[0]]),
            ("insufficient_short_bucket", [r for r in rows if r["sequence_length"] > 512]),
            ("non_training_example", [{**rows[0], "project_split": "validation"}] + rows[1:]),
            ("overlength_not_truncated", [{**rows[0], "sequence_length": 4097}] + rows[1:]),
            ("zero_target_example", [{**rows[0], "supervised_token_count": 0}] + rows[1:]),
        )
        for name, bad_rows in examples:
            try:
                select_pilot(bad_rows)
            except ValueError as error:
                invalid.append({"name": name, "rejected": True, "error": str(error)})
            else:
                raise AssertionError(f"Bad selection input accepted: {name}")
        import torch
        if torch.cuda.is_initialized():
            raise AssertionError("CPU selection check initialized CUDA")
        if sha256(train_path) != manifest["output_sha256"]["train_tokens.jsonl"]:
            raise AssertionError("Training file changed during selection")
        report.update(status="pass", selection=selection, deterministic=True, order_invariant=True,
                      seed_changes_nonrequired_examples=True, selected_features_cpu_verified=len(verified),
                      invalid_inputs=invalid, successful_invalid_input_checks=len(invalid),
                      required_examples_verified=True, selected_arrays_unchanged=True,
                      both_process_halves_cover_all_buckets=True, cuda_initialized=False,
                      training_file_unchanged=True)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "selected_features_cpu_verified", "successful_invalid_input_checks", "seconds")}, ensure_ascii=False))
    print(json.dumps({"required_examples": selection["required_examples"], "input_tokens": selection["input_tokens"],
                      "target_tokens": selection["target_tokens"], "groups": selection["groups"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
