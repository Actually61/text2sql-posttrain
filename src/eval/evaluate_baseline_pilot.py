"""Score only the frozen 32-example baseline pilot; full-validation defaults stay intact."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/data"))
import prepare_baseline_pilot as pilot
import evaluate_spider as evaluation


def pilot_config(config, manifest):
    if manifest["example_count"] != 32 or manifest["maximum_evaluation_examples"] != 32:
        raise ValueError("This wrapper may evaluate exactly the frozen 32-example pilot")
    derived = dict(config)
    derived.update(split="validation_pilot", expected_examples=32,
                   denominator_policy="all 32 frozen pilot examples including errors and timeouts")
    return derived


def evaluate(predictions_path):
    manifest, rows, full_config, fingerprints = pilot.validate_published()
    config = pilot_config(full_config, manifest)
    predictions_path = Path(predictions_path)
    predictions_path = predictions_path.resolve() if predictions_path.is_absolute() else evaluation.project_path(str(predictions_path))
    # This rejects complete-625 files, missing pilot records, duplicate IDs and wrong databases.
    evaluation.validate_predictions(evaluation.read_jsonl(predictions_path), rows)
    fingerprints[evaluation.relative(Path(__file__))] = evaluation.digest(Path(__file__))
    summary = evaluation.run_evaluation(predictions_path, "baseline_pilot", config, rows, fingerprints)
    summary.update(sample_scope=manifest["sample_scope"], sample_size=32, validation_population_count=625,
                   pilot_manifest_sha256=evaluation.digest(pilot.DIRECTORY / "manifest.json"),
                   prediction_file_sha256=evaluation.digest(predictions_path),
                   prediction_file=str(predictions_path),
                   fixed_sample_source_indices=manifest["selected_source_indices"],
                   hardness_population_counts=manifest["hardness_population_counts"],
                   hardness_sample_counts=manifest["hardness_sample_counts"],
                   complete_validation_evaluation=False,
                   model_identity_note="Scorer does not infer model provenance; consult the linked prediction file and its generation report",
                   model_generation_performed=False)
    run_dir = ROOT / summary["run_dir"]
    shutil.copyfile(Path(__file__), run_dir / Path(__file__).name)
    shutil.copyfile(Path(pilot.__file__), run_dir / Path(pilot.__file__).name)
    shutil.copyfile(pilot.DIRECTORY / "manifest.json", run_dir / "pilot_manifest.json")
    # The reused runner snapshots the original config file; also save effective subset protocol.
    (run_dir / "effective_pilot_evaluation.json").write_bytes(evaluation.json_bytes(config))
    (run_dir / "summary.json").write_bytes(evaluation.json_bytes(summary))
    (ROOT / "reports/baseline_pilot_eval.json").write_bytes(evaluation.json_bytes(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Exactly the frozen 32 source_index/db_id/predicted_sql records")
    args = parser.parse_args()
    evaluate(args.predictions)
