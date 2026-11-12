"""Checkpoint byte checks and fingerprints without importing PEFT/bitsandbytes.

The legacy checkpoint module stays frozen.  This module does not construct models,
load optimizer/RNG pickle payloads, or query CUDA device capabilities.  Its generic
fingerprint preserves the old format, including copying an already-existing GPU
tensor to CPU when called later by the inference worker.  CPU preflight uses the
explicit assert_cpu_only/inspect_final_adapter_cpu boundary below.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch


REQUIRED_METADATA = {"completed_updates", "next_group_index", "next_update_lr",
                     "schedule_sha256", "run_config_sha256", "base_hash", "protected_source_sha256"}
PAYLOAD_FILES = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "rng.pt", "metadata.json"}


def assert_cpu_only(stage="checkpoint preflight"):
    """Read initialization state without initializing CUDA or querying a device."""
    if torch.cuda.is_initialized():
        raise RuntimeError(f"CUDA was initialized during {stage}")


def state_fingerprint(value):
    """Hash nested values/dtypes/shapes exactly; tensor device is deliberately ignored."""
    hasher = hashlib.sha256()

    def add(tag, payload=b""):
        tag = tag.encode("ascii")
        hasher.update(struct.pack("<Q", len(tag)) + tag + struct.pack("<Q", len(payload)) + payload)

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            add("tensor", str(tensor.dtype).encode())
            visit(tuple(tensor.shape))
            add("bytes", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            add("ndarray", item.dtype.str.encode())
            visit(tuple(item.shape))
            add("bytes", np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            add("dict", str(len(item)).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            add(type(item).__name__, str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None:
            add("none")
        elif isinstance(item, bool):
            add("bool", bytes([item]))
        elif isinstance(item, (int, np.integer)):
            add("int", str(int(item)).encode())
        elif isinstance(item, (float, np.floating)):
            add("float", struct.pack("<d", float(item)))
        elif isinstance(item, str):
            add("str", item.encode("utf-8"))
        else:
            raise TypeError(f"Unsupported checkpoint fingerprint value: {type(item).__name__}")

    visit(value)
    return hasher.hexdigest()


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metadata_check(metadata):
    if not isinstance(metadata, dict) or not REQUIRED_METADATA <= set(metadata):
        raise ValueError("Missing checkpoint training metadata")
    completed, cursor = metadata["completed_updates"], metadata["next_group_index"]
    if type(completed) is not int or type(cursor) is not int or completed < 0 or cursor != completed:
        raise ValueError("Checkpoint must be at a completed optimizer-group boundary")
    if not isinstance(metadata["next_update_lr"], (int, float)) or not math.isfinite(metadata["next_update_lr"]) or metadata["next_update_lr"] < 0:
        raise ValueError("Invalid next update learning rate")
    for name in ("schedule_sha256", "run_config_sha256"):
        value = metadata[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Invalid SHA256 metadata: {name}")
    if not isinstance(metadata["protected_source_sha256"], dict):
        raise ValueError("Protected-source metadata must be a hash mapping")
    if json.loads(json.dumps(metadata, allow_nan=False)) != metadata:
        raise ValueError("Metadata must consist of JSON values")


def inspect_checkpoint(source):
    """Hash all saved payload bytes; do not deserialize optimizer/RNG pickle files."""
    source = Path(source)
    manifest = json.loads((source / "complete.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "sft_complete_peft_v1" or set(manifest.get("files", {})) != PAYLOAD_FILES:
        raise ValueError("Invalid/incomplete checkpoint manifest")
    if {path.name for path in source.iterdir()} != PAYLOAD_FILES | {"complete.json"}:
        raise ValueError("Unexpected checkpoint files or directories")
    for name, expected in manifest["files"].items():
        if _file_digest(source / name) != expected:
            raise ValueError(f"Checkpoint payload hash mismatch: {name}")
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    _metadata_check(metadata)
    return {"metadata": metadata, "manifest": manifest}


def inspect_final_adapter_cpu(source, *, expected_updates=796, expected_parameters=9_232_384,
                              expected_tensors=392):
    """Verify the final adapter on CPU and return the checked payload and evidence."""
    assert_cpu_only("checkpoint preflight entry")
    inspected = inspect_checkpoint(source)
    metadata, manifest = inspected["metadata"], inspected["manifest"]
    if metadata["completed_updates"] != expected_updates or metadata["next_group_index"] != expected_updates:
        raise ValueError("Checkpoint is not the authorized final update")
    from safetensors.torch import load_file
    payload = load_file(str(Path(source) / "adapter_model.safetensors"), device="cpu")
    if (len(payload) != expected_tensors or sum(value.numel() for value in payload.values()) != expected_parameters
            or any("lora_" not in name or value.device.type != "cpu" or value.dtype != torch.float32
                   for name, value in payload.items())):
        raise ValueError("Unexpected final FP32 CPU LoRA payload")
    actual = state_fingerprint(payload)
    if actual != manifest["fingerprints"]["adapter"]:
        raise ValueError("Final adapter tensor fingerprint mismatch")
    assert_cpu_only("checkpoint preflight exit")
    return {**inspected, "payload": payload, "adapter_fingerprint": actual,
            "adapter_tensor_count": len(payload), "adapter_parameter_count": expected_parameters,
            "cuda_initialized": False, "optimizer_or_rng_deserialized": False}
