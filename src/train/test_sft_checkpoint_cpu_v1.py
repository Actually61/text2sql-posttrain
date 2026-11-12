"""One bounded CPU acceptance of isolated checkpoint checks; no GPU/model/training.

Do not run twice: an exclusive claim and result retain the first attempt, including
failure.  PEFT/bitsandbytes imports, CUDA initialization and non-CPU tensor operations
are intercepted before the helper is imported.  CUDA visibility is not hidden.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / "reports/sft_checkpoint_cpu_v1_checks.json"
CLAIM = ROOT / "outputs/checks/sft_checkpoint_cpu_v1"
CHECKPOINT = ROOT / "outputs/runs/sft_round1_resource_v3/checkpoints/update_000796"
BUDGET_SECONDS = 60.0


def write(path, value, mode="w"):
    with Path(path).open(mode, encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def main():
    if REPORT.exists() or CLAIM.exists():
        raise FileExistsError("CPU acceptance already claimed; no retry or overwrite")
    CLAIM.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "budget_seconds": BUDGET_SECONDS, "automatic_retries": 0, "checks": [],
              "real_model_loaded": False, "training_performed": False, "final_examples_read": False,
              "cuda_visibility_was_hidden": False, "attempted_forbidden_imports": [],
              "attempted_cuda_initializations": 0, "attempted_non_cpu_operations": 0}
    write(REPORT, record, "x")
    write(CLAIM / "claim.json", {"pid": os.getpid(), "started_monotonic": started,
                                "budget_seconds": BUDGET_SECONDS, "automatic_retries": 0}, "x")
    final_lock = threading.Lock()
    finished = False

    def timeout():
        nonlocal finished
        with final_lock:
            if finished:
                return
            finished = True
            record.update(status="failed", error="CPU acceptance exceeded 60-second hard budget",
                          seconds=time.perf_counter() - started)
            try:
                write(REPORT, record)
                write(CLAIM / "result.json", record, "x")
            finally:
                os._exit(124)

    timer = threading.Timer(max(0, BUDGET_SECONDS - (time.perf_counter() - started)), timeout)
    timer.daemon = True
    timer.start()
    blocker = None
    original_lazy_init = None
    try:
        class ForbiddenImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in {"peft", "bitsandbytes"}:
                    record["attempted_forbidden_imports"].append(fullname)
                    raise ImportError(f"CPU acceptance forbids importing {fullname}")
                return None

        if any(name.split(".", 1)[0] in {"peft", "bitsandbytes"} for name in sys.modules):
            raise RuntimeError("Forbidden library was imported before acceptance guards")
        blocker = ForbiddenImport()
        sys.meta_path.insert(0, blocker)
        import numpy as np
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialized while importing plain torch/numpy")
        original_lazy_init = torch.cuda._lazy_init

        def blocked_cuda_init(*args, **kwargs):
            record["attempted_cuda_initializations"] += 1
            raise RuntimeError("CPU acceptance intercepted CUDA initialization")

        torch.cuda._lazy_init = blocked_cuda_init
        from torch.utils._python_dispatch import TorchDispatchMode

        def check_devices(value):
            if isinstance(value, torch.Tensor):
                if value.device.type != "cpu":
                    record["attempted_non_cpu_operations"] += 1
                    raise RuntimeError(f"Non-CPU tensor intercepted: {value.device}")
            elif isinstance(value, torch.device):
                if value.type != "cpu":
                    record["attempted_non_cpu_operations"] += 1
                    raise RuntimeError(f"Non-CPU device intercepted: {value}")
            elif isinstance(value, dict):
                for child in value.values():
                    check_devices(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    check_devices(child)

        class CPUOperationsOnly(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                check_devices(args); check_devices(kwargs)
                device = kwargs.get("device")
                if device is not None and torch.device(device).type != "cpu":
                    record["attempted_non_cpu_operations"] += 1
                    raise RuntimeError(f"Non-CPU operation intercepted: {func}")
                result = func(*args, **kwargs)
                check_devices(result)
                return result

        with CPUOperationsOnly():
            from src.train import sft_checkpoint_cpu_v1 as isolated
            isolated.assert_cpu_only("module import")
            record["checks"].append({"name": "isolated_import_under_cpu_guards", "status": "pass"})
            old_path = ROOT / "src/train/sft_checkpoint.py"
            tree = ast.parse(old_path.read_text(encoding="utf-8"), filename=str(old_path))
            definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name == "state_fingerprint"]
            if len(definitions) != 1:
                raise ValueError("Expected one legacy fingerprint function")
            namespace = {"hashlib": hashlib, "struct": struct, "np": np, "torch": torch}
            code = compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                           str(old_path) + "::state_fingerprint_only", "exec")
            exec(code, namespace)
            legacy = namespace["state_fingerprint"]
            tensor = torch.arange(12, dtype=torch.float32, device="cpu").reshape(3, 4).t()
            nested = {"tensor": tensor, "array": np.array([1, 2], dtype=np.int64),
                      "misc": [None, True, 7, -2.5, "校验", (np.int32(8), np.float32(0.25))]}
            if isolated.state_fingerprint(nested) != legacy(nested):
                raise ValueError("Nested fingerprint differs from legacy AST function")
            record["checks"].append({"name": "legacy_ast_nested_equivalence", "status": "pass"})
            fp = isolated.state_fingerprint(tensor)
            mutated = tensor.clone(); mutated[0, 0] += 1
            if fp == isolated.state_fingerprint(mutated):
                raise ValueError("Value mutation was not detected")
            if fp == isolated.state_fingerprint(tensor.to(dtype=torch.float64)):
                raise ValueError("Dtype mutation was not detected")
            if fp == isolated.state_fingerprint(tensor.reshape(-1)):
                raise ValueError("Shape mutation was not detected")
            record["checks"].append({"name": "value_dtype_shape_changes_detected", "status": "pass"})
            checked = isolated.inspect_final_adapter_cpu(CHECKPOINT)
            if legacy(checked["payload"]) != checked["adapter_fingerprint"]:
                raise ValueError("Real adapter fingerprint differs from legacy AST function")
            record["checks"].append({"name": "final796_files_cpu_payload_and_legacy_fingerprint", "status": "pass",
                "completed_updates": checked["metadata"]["completed_updates"],
                "adapter_tensor_count": checked["adapter_tensor_count"],
                "adapter_parameter_count": checked["adapter_parameter_count"],
                "adapter_fingerprint": checked["adapter_fingerprint"],
                "payload_sha256": checked["manifest"]["files"], "optimizer_or_rng_deserialized": False})
            isolated.assert_cpu_only("acceptance completion")
        if (record["attempted_forbidden_imports"] or record["attempted_cuda_initializations"]
                or record["attempted_non_cpu_operations"]
                or any(name.split(".", 1)[0] in {"peft", "bitsandbytes"} for name in sys.modules)):
            raise RuntimeError("A CPU isolation boundary was attempted")
        record.update(status="pass", cuda_initialized=torch.cuda.is_initialized(),
                      forbidden_modules_loaded=False, optimizer_or_rng_deserialized=False)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        if original_lazy_init is not None:
            torch.cuda._lazy_init = original_lazy_init
        if blocker is not None:
            sys.meta_path.remove(blocker)
        with final_lock:
            finished = True
            timer.cancel()
            record["seconds"] = time.perf_counter() - started
            record["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ("src/train/sft_checkpoint.py", "src/train/sft_checkpoint_cpu_v1.py",
                             "src/train/test_sft_checkpoint_cpu_v1.py")}
            write(REPORT, record)
            write(CLAIM / "result.json", record, "x")
    print(json.dumps({key: record.get(key) for key in ("status", "seconds", "cuda_initialized",
          "attempted_forbidden_imports", "attempted_cuda_initializations", "attempted_non_cpu_operations", "error")},
          ensure_ascii=False), flush=True)
    return 0 if record["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
