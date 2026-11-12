"""Complete PEFT checkpoints: adapter, optimizer, cursor/LR metadata and RNG state.

The caller creates the same frozen base and fresh LoRA before loading. This module
does not train, initialize CUDA, merge adapters, or select/resume a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import struct

import numpy as np
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file


REQUIRED_METADATA = {"completed_updates", "next_group_index", "next_update_lr",
                     "schedule_sha256", "run_config_sha256", "base_hash", "protected_source_sha256"}
PAYLOAD_FILES = {"adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "rng.pt", "metadata.json"}


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


def capture_rng():
    """Capture CUDA RNG only if the caller has already initialized CUDA."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else None}


def restore_rng(state):
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Unexpected RNG state fields")
    has_cuda = state["torch_cuda"] is not None
    if has_cuda != torch.cuda.is_initialized():
        raise ValueError("Checkpoint/current CUDA initialization state differs")
    if has_cuda and len(state["torch_cuda"]) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA device count differs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if has_cuda:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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
    # A JSON round trip ensures no live tensors or platform-specific objects hide in metadata.
    if json.loads(json.dumps(metadata, allow_nan=False)) != metadata:
        raise ValueError("Metadata must consist of JSON values")


def _parameter_layout(model, optimizer):
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Checkpoint expects trainable LoRA parameters only")
    names_by_id = {id(parameter): name for name, parameter in trainable.items()}
    group_names = []
    for group in optimizer.param_groups:
        if any(id(parameter) not in names_by_id for parameter in group["params"]):
            raise ValueError("Optimizer contains a frozen or unknown parameter")
        group_names.append([names_by_id[id(parameter)] for parameter in group["params"]])
    flattened = [name for group in group_names for name in group]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(trainable):
        raise ValueError("Optimizer must cover every trainable parameter exactly once")
    inventory = [{"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
                 for name, parameter in sorted(trainable.items())]
    return {"optimizer_parameter_names": group_names, "trainable_inventory": inventory}


def _adapter_state(model):
    if set(model.peft_config) != {"default"}:
        raise ValueError("This checkpoint format expects one default PEFT adapter")
    state = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    if not state or any("lora_" not in name for name in state):
        raise ValueError("Adapter payload must contain only LoRA tensors")
    return _cpu_clone(state)


def _optimizer_devices(optimizer):
    devices = {}
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                key = f"{name}:{value.device}"
                devices[key] = devices.get(key, 0) + 1
    return devices


def inspect_checkpoint(source):
    """Validate all payload bytes and metadata without constructing a model or loading pickle."""
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


def save_checkpoint(destination, model, optimizer, metadata):
    """Save a new complete directory; retain incomplete temp files on any failure."""
    destination = Path(destination).resolve()
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists() or temporary.exists():
        raise FileExistsError("Checkpoint or its incomplete attempt already exists; no overwrite")
    _metadata_check(metadata)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Save only after zero_grad(set_to_none=True) at a complete group boundary")
    if any(group["lr"] != metadata["next_update_lr"] for group in optimizer.param_groups):
        raise ValueError("Advance optimizer LR to next_update_lr before saving")
    layout = _parameter_layout(model, optimizer)
    adapter = _adapter_state(model)
    optimizer_state = _cpu_clone(optimizer.state_dict())
    rng = _cpu_clone(capture_rng())
    fingerprints = {"adapter": state_fingerprint(adapter), "optimizer": state_fingerprint(optimizer_state),
                    "rng": state_fingerprint(rng)}
    temporary.mkdir(parents=True, exist_ok=False)
    save_file(adapter, str(temporary / "adapter_model.safetensors"))
    model.peft_config["default"].save_pretrained(temporary)
    torch.save(optimizer_state, temporary / "optimizer.pt")
    torch.save(rng, temporary / "rng.pt")
    _json_write(temporary / "metadata.json", metadata)
    # These two pickle files were just written by this function; never load external untrusted checkpoints.
    readback = {"adapter": load_file(str(temporary / "adapter_model.safetensors")),
                "optimizer": torch.load(temporary / "optimizer.pt", map_location="cpu", weights_only=True),
                "rng": torch.load(temporary / "rng.pt", map_location="cpu", weights_only=False)}
    if any(state_fingerprint(readback[name]) != expected for name, expected in fingerprints.items()):
        raise ValueError("Checkpoint readback does not equal live training state")
    if state_fingerprint(capture_rng()) != fingerprints["rng"]:
        raise ValueError("Saving unexpectedly consumed RNG state")
    manifest = {"format": "sft_complete_peft_v1", "files": {name: _file_digest(temporary / name) for name in sorted(PAYLOAD_FILES)},
                "fingerprints": fingerprints, **layout}
    _json_write(temporary / "complete.json", manifest)
    inspect_checkpoint(temporary)
    # Same-parent rename publishes only the completely verified checkpoint, without replacing another directory.
    temporary.rename(destination)
    return {"path": str(destination), "metadata": metadata, "fingerprints": fingerprints,
            "readback_equal": True, "optimizer_devices": _optimizer_devices(optimizer),
            "cuda_rng_saved": rng["torch_cuda"] is not None}


def load_checkpoint(source, model, optimizer, expected_metadata):
    """Restore into a fresh identical base/LoRA; RNG restoration is the final operation."""
    source = Path(source)
    inspected = inspect_checkpoint(source)
    metadata, manifest = inspected["metadata"], inspected["manifest"]
    _metadata_check(expected_metadata)
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Checkpoint metadata mismatch: {name}")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("Restore requires an empty-gradient model")
    layout = _parameter_layout(model, optimizer)
    if any(layout[key] != manifest[key] for key in layout):
        raise ValueError("Optimizer parameter name/order/shape/dtype differs from checkpoint")
    adapter = load_file(str(source / "adapter_model.safetensors"))
    current = _adapter_state(model)
    if set(adapter) != set(current) or any(adapter[name].shape != current[name].shape or adapter[name].dtype != current[name].dtype for name in adapter):
        raise ValueError("Adapter checkpoint structure differs from the new model")
    # Hash integrity was checked before loading the project's own trusted optimizer/RNG payloads.
    optimizer_state = torch.load(source / "optimizer.pt", map_location="cpu", weights_only=True)
    rng = torch.load(source / "rng.pt", map_location="cpu", weights_only=False)
    for name, value in (("adapter", adapter), ("optimizer", optimizer_state), ("rng", rng)):
        if state_fingerprint(value) != manifest["fingerprints"][name]:
            raise ValueError(f"Checkpoint value fingerprint mismatch: {name}")
    set_peft_model_state_dict(model, adapter, adapter_name="default")
    optimizer.load_state_dict(optimizer_state)
    adapter_after = state_fingerprint(_adapter_state(model))
    optimizer_after = state_fingerprint(optimizer.state_dict())
    if adapter_after != manifest["fingerprints"]["adapter"] or optimizer_after != manifest["fingerprints"]["optimizer"]:
        raise ValueError("Loaded adapter or optimizer is not exactly equal to saved state")
    if any(group["lr"] != metadata["next_update_lr"] for group in optimizer.param_groups):
        raise ValueError("Restored optimizer LR differs from saved next_update_lr")
    devices = _optimizer_devices(optimizer)
    restore_rng(rng)
    rng_after = state_fingerprint(capture_rng())
    if rng_after != manifest["fingerprints"]["rng"]:
        raise ValueError("Restored random state differs from checkpoint")
    return {"metadata": metadata, "verification": {"adapter_exact": True, "optimizer_exact": True,
            "rng_exact": True, "parameter_order_exact": True, "fingerprints": {
                "adapter": adapter_after, "optimizer": optimizer_after, "rng": rng_after},
            "optimizer_devices": devices, "cuda_rng_restored": rng["torch_cuda"] is not None}}
