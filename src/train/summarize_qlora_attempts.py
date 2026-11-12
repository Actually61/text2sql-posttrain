"""汇总真实尝试；只读旧训练产物，写审计账本，不加载模型或训练。"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


attempts = []
for name in ("qlora_single_batch_v1", "qlora_single_batch_v2"):
    path = ROOT / "outputs/runs" / name
    if not path.exists():
        continue
    preflight = read(path / "preflight.json")
    changes_to_own_source = []
    for filename, expected in preflight["protected_sha256"].items():
        if sha(ROOT / filename) != expected:
            # 仅允许这个冒烟脚本/配置的已记录修订；旧运行快照仍必须逐字匹配。
            if filename not in ("src/train/qlora_smoke.py", "configs/qlora_smoke.json"):
                raise ValueError(f"Protected historical file changed: {filename}")
            if sha(path / "snapshot" / filename) != expected:
                raise ValueError(f"Old source snapshot changed: {filename}")
            changes_to_own_source.append(filename)
    step = read(path / "step.json") if (path / "step.json").exists() else None
    gpu = read(path / "gpu_samples.json")
    status = read(path / "status.json")
    attempts.append({"run_dir": path.relative_to(ROOT).as_posix(), "status": status["status"],
        "optimizer_steps": step["optimizer_steps"] if step else 0,
        "loss": step["loss"] if step else None,
        "gradient_all_finite": step["gradients"]["all_finite"] if step else None,
        "gradient_l2_norm": step["gradients"]["global_l2_norm"] if step else None,
        "timings": step["timings"] if step else None,
        "adapter_saved": (path / "adapter/adapter_model.safetensors").exists(),
        "whole_gpu_sampled_peak_mib": max(r["used_mib"] for r in gpu["samples"]),
        "torch_peak_note": "See summary.json if present; first attempt's allocator peaks were not saved before failure",
        "scope": "Each attempt starts independently from the original base; no adapter continued from another attempt",
        "failure": read(path / "failure.json") if (path / "failure.json").exists() else None,
        "updated_smoke_source_with_original_snapshot_verified": changes_to_own_source,
        "protected_base_data_prompt_tokenizer_baseline_files_unchanged": True})
report = {"attempts": attempts, "total_actual_optimizer_steps": sum(r["optimizer_steps"] for r in attempts),
          "validation_examples_loaded": False, "final_examples_read": False,
          "summary_script_model_load_or_training": False}
output = ROOT / "reports/qlora_smoke_attempts.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
