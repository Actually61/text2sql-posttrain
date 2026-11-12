"""CPU disk checkpoint test; no real Qwen weights or CUDA initialization."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # This test process must not initialize a CUDA context.
import numpy as np
import torch
from src.train.test_target_loss import digest, make_batch, make_model
from src.train.test_token_accumulation import microbatches, named_trainable, OPTIMIZER_KWARGS
from src.train.token_accumulation import backward_token_normalized_group
from src.train.sft_checkpoint import capture_rng, inspect_checkpoint, load_checkpoint, save_checkpoint, state_fingerprint

REPORT = ROOT / "reports/sft_checkpoint_cpu_checks.json"


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = ROOT / "outputs/checks/sft_checkpoint_cpu" / stamp
    run.mkdir(parents=True, exist_ok=False)
    previous = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else None
    report = {"status": "started", "device": "cpu", "test_dir": run.relative_to(ROOT).as_posix(),
              "gpu_used": False, "real_model_loaded": False, "formal_training_performed": False,
              "previous_attempts": ([] if previous is None else [{"report": previous["test_dir"] + "/report.json",
                  "status": previous["status"], "error": previous.get("error", ""),
                  "note": "Prior CPU fixture unexpectedly initialized CUDA; now hide CUDA devices in this CPU test process before torch import"}]),
              "checks": [], "packages": {name: version(name) for name in ("torch", "peft", "transformers", "safetensors")},
              "source_sha256": {name: digest(ROOT / name) for name in (
                  "src/train/sft_checkpoint.py", "src/train/test_sft_checkpoint.py", "src/train/token_accumulation.py",
                  "src/train/test_target_loss.py", "src/train/test_token_accumulation.py")}}
    try:
        model = make_model(800, tied=True, lora=True)
        first = microbatches(make_batch([(3, 2, True), (6, 4, False)]))
        second = microbatches(make_batch([(2, 1, True), (9, 3, True), (4, 7, False)]))
        optimizer = torch.optim.AdamW(list(named_trainable(model).values()), **OPTIMIZER_KWARGS)

        def update(target, opt, batches, update_index):
            for group in opt.param_groups:
                group["lr"] = 1e-4 * update_index / 24
            opt.zero_grad(set_to_none=True)
            result = backward_token_normalized_group(target, batches, device="cpu")
            torch.nn.utils.clip_grad_norm_(list(named_trainable(target).values()), 1.0, error_if_nonfinite=True)
            opt.step()
            for group in opt.param_groups:
                group["lr"] = 1e-4 * (update_index + 1) / 24
            opt.zero_grad(set_to_none=True)
            return result

        update(model, optimizer, first, 1)
        metadata = {"completed_updates": 1, "next_group_index": 1, "next_update_lr": 1e-4 * 2 / 24,
                    "schedule_sha256": "a" * 64, "run_config_sha256": "b" * 64,
                    "base_hash": "cpu_fixture", "protected_source_sha256": report["source_sha256"]}
        checkpoint = run / "checkpoint_1"
        saved = save_checkpoint(checkpoint, model, optimizer, metadata)
        inspected = inspect_checkpoint(checkpoint)
        assert inspected["metadata"] == metadata and saved["readback_equal"]
        report["checks"].append({"name": "disk_save_readback_publish", "status": "pass", "files": sorted(inspected["manifest"]["files"])})
        expected_draws = (random.random(), float(np.random.rand()), torch.rand(4))
        direct_result = update(model, optimizer, second, 2)
        direct_adapter = {name: p.detach().clone() for name, p in named_trainable(model).items()}
        direct_optimizer = copy.deepcopy(optimizer.state_dict())
        # The base must be identical; recreate seed800 before restoring only its adapter.
        restored = make_model(800, tied=True, lora=True)
        restored_optimizer = torch.optim.AdamW(list(named_trainable(restored).values()), **OPTIMIZER_KWARGS)
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        loaded = load_checkpoint(checkpoint, restored, restored_optimizer, metadata)
        actual_draws = (random.random(), float(np.random.rand()), torch.rand(4))
        assert expected_draws[:2] == actual_draws[:2] and torch.equal(expected_draws[2], actual_draws[2])
        resumed_result = update(restored, restored_optimizer, second, 2)
        max_difference = 0.0
        for name, parameter in named_trainable(restored).items():
            torch.testing.assert_close(parameter, direct_adapter[name], atol=0, rtol=0)
            max_difference = max(max_difference, float((parameter.detach() - direct_adapter[name]).abs().max()))
        assert state_fingerprint(restored_optimizer.state_dict()) == state_fingerprint(direct_optimizer)
        assert direct_result["group_loss"] == resumed_result["group_loss"]
        assert loaded["verification"]["rng_exact"] and not loaded["verification"]["cuda_rng_restored"]
        report["checks"].append({"name": "disk_resume_then_next_update", "status": "pass", "max_parameter_difference": max_difference,
                                 "optimizer_equal": True, "rng_draws_equal": True, "next_group_index": 1,
                                 "verification": loaded["verification"]})

        def rejects(name, operation, exception=(ValueError, FileExistsError)):
            try:
                operation()
            except exception as error:
                report["checks"].append({"name": name, "status": "pass", "rejected": str(error)})
            else:
                raise AssertionError(f"Invalid checkpoint operation accepted: {name}")

        rejects("refuse_existing_destination", lambda: save_checkpoint(checkpoint, model, optimizer, metadata))
        bad_metadata = {**metadata, "schedule_sha256": "c" * 64}
        before_mismatch = state_fingerprint({name: p.detach() for name, p in named_trainable(restored).items()})
        rejects("metadata_mismatch_before_mutation", lambda: load_checkpoint(checkpoint, restored, restored_optimizer, bad_metadata))
        assert before_mismatch == state_fingerprint({name: p.detach() for name, p in named_trainable(restored).items()})
        wrong_order = torch.optim.AdamW(list(reversed(list(named_trainable(restored).values()))), **OPTIMIZER_KWARGS)
        rejects("parameter_order_mismatch", lambda: load_checkpoint(checkpoint, restored, wrong_order, metadata))
        damaged = run / "damaged_copy"
        shutil.copytree(checkpoint, damaged)
        with (damaged / "optimizer.pt").open("ab") as stream:
            stream.write(b"corruption fixture")
        rejects("corrupted_payload_before_pickle_load", lambda: inspect_checkpoint(damaged))
        parameter = next(p for p in restored.parameters() if p.requires_grad)
        parameter.grad = torch.zeros_like(parameter)
        rejects("incomplete_group_gradients_rejected", lambda: save_checkpoint(run / "invalid_gradient_save", restored, restored_optimizer,
                {**metadata, "completed_updates": 2, "next_group_index": 2, "next_update_lr": 1e-4 * 3 / 24}))
        parameter.grad = None
        if torch.cuda.is_initialized():
            raise AssertionError("CPU test initialized CUDA")
        report.update(status="pass", successful_checks=len(report["checks"]), cuda_initialized=False,
                      optimizer_updates_scope="CPU fixtures only: first group once and second group on uninterrupted/restored paths",
                      real_gpu_resume_validated=False)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        report["cuda_initialized_at_exit"] = torch.cuda.is_initialized()
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        (run / "report.json").write_text(text, encoding="utf-8")
        REPORT.write_text(text, encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "successful_checks", "seconds", "cuda_initialized")}), flush=True)


if __name__ == "__main__":
    main()
