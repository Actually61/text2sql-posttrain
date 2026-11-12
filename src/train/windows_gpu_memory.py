"""通过Windows PDH读取指定进程的WDDM专用/共享GPU内存；不调用CUDA。"""
from __future__ import annotations
import ctypes
from ctypes import wintypes
import json
import os
import re
import threading
import time

MORE_DATA = 0x800007D2
FMT_LARGE_NOSCALE = 0x00000400 | 0x00001000
COUNTER_NAMES = {"dedicated_bytes": r"\GPU Process Memory(*)\Dedicated Usage",
                 "shared_bytes": r"\GPU Process Memory(*)\Shared Usage"}
INSTANCE_RE = re.compile(r"^pid_(\d+)_luid_(0x[0-9a-f]+)_(0x[0-9a-f]+)_phys_(\d+)(?:#\d+)?$", re.I)


class _ValueUnion(ctypes.Union):
    _fields_ = [("longValue", ctypes.c_int32), ("doubleValue", ctypes.c_double),
                ("largeValue", ctypes.c_int64), ("stringValue", ctypes.c_void_p)]


class _FormattedValue(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("value", _ValueUnion)]


class _FormattedItem(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", _FormattedValue)]


def _check(status, operation):
    code = int(status) & 0xFFFFFFFF
    if code:
        raise OSError(f"{operation}: PDH 0x{code:08X}")


class _PDHQuery:
    """仅在Windows创建原生计数器查询；所有值在返回前复制为Python对象。"""
    def __init__(self):
        if os.name != "nt":
            raise OSError("Windows PDH is unavailable on this platform")
        self.dll = ctypes.WinDLL("pdh.dll")
        self.query = wintypes.HANDLE()
        self.handles = {}
        api = self.dll
        api.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        api.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        api.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        api.PdhGetFormattedCounterArrayW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_FormattedItem)]
        api.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        for name in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData", "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
            getattr(api, name).restype = ctypes.c_int32
        _check(api.PdhOpenQueryW(None, 0, ctypes.byref(self.query)), "PdhOpenQueryW")
        try:
            for name, path in COUNTER_NAMES.items():
                handle = wintypes.HANDLE()
                _check(api.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle)), f"PdhAddEnglishCounterW({name})")
                self.handles[name] = handle
        except Exception:
            self.close()
            raise

    def _array(self, handle):
        # 实例可能在两次调用间增加；MORE_DATA时重新询问大小，不使用失效缓冲。
        for _ in range(3):
            size, count = wintypes.DWORD(0), wintypes.DWORD(0)
            status = self.dll.PdhGetFormattedCounterArrayW(handle, FMT_LARGE_NOSCALE,
                ctypes.byref(size), ctypes.byref(count), None)
            code = int(status) & 0xFFFFFFFF
            if code == 0 and count.value == 0:
                return []
            if code != MORE_DATA:
                _check(status, "PdhGetFormattedCounterArrayW(size)")
            if size.value == 0:
                return []
            buffer = ctypes.create_string_buffer(size.value)
            pointer = ctypes.cast(buffer, ctypes.POINTER(_FormattedItem))
            status = self.dll.PdhGetFormattedCounterArrayW(handle, FMT_LARGE_NOSCALE,
                ctypes.byref(size), ctypes.byref(count), pointer)
            if (int(status) & 0xFFFFFFFF) == MORE_DATA:
                continue
            _check(status, "PdhGetFormattedCounterArrayW(data)")
            return [{"instance": pointer[i].szName, "counter_status": int(pointer[i].FmtValue.CStatus),
                     "bytes": int(pointer[i].FmtValue.value.largeValue)} for i in range(count.value)]
        raise OSError("PDH instance list kept changing; no valid sample")

    def collect(self):
        _check(self.dll.PdhCollectQueryData(self.query), "PdhCollectQueryData")
        return {name: self._array(handle) for name, handle in self.handles.items()}

    def close(self):
        if self.query:
            self.dll.PdhCloseQuery(self.query)
            self.query = wintypes.HANDLE()


def _extract_pid(raw, pid):
    """只输出目标PID，按原始instance配对，不将缺失计数器伪造成零。"""
    by_instance = {}
    provider_counts = {name: len(values) for name, values in raw.items()}
    for metric, values in raw.items():
        for value in values:
            match = INSTANCE_RE.fullmatch(value["instance"] or "")
            if match is None or int(match.group(1)) != pid:
                continue
            key = value["instance"]
            row = by_instance.setdefault(key, {"instance": key, "pid": pid,
                "adapter_luid": f"{match.group(2).lower()}_{match.group(3).lower()}",
                "physical_adapter_index": int(match.group(4)),
                "dedicated_bytes": None, "shared_bytes": None, "counter_status": {}})
            row["counter_status"][metric] = value["counter_status"]
            if value["counter_status"] in (0, 1) and value["bytes"] >= 0:
                row[metric] = value["bytes"]
    instances = sorted(by_instance.values(), key=lambda row: row["instance"])
    complete = bool(instances) and all(row[k] is not None for row in instances for k in COUNTER_NAMES)
    adapter_keys = [(row["adapter_luid"], row["physical_adapter_index"]) for row in instances]
    duplicate_adapter = len(set(adapter_keys)) != len(adapter_keys)
    # 若同PID/物理适配器有多个实例，不假设它们能相加；仍保留原始实例记录。
    usable = complete and not duplicate_adapter
    return {"status": "ok" if usable else ("not_registered" if not instances else "ambiguous_or_invalid_counter"),
        "matched_instances": len(instances), "instances": instances, "provider_instance_counts": provider_counts,
        "dedicated_bytes": sum(r["dedicated_bytes"] for r in instances) if usable else None,
        "shared_bytes": sum(r["shared_bytes"] for r in instances) if usable else None,
        "duplicate_adapter_instances": duplicate_adapter}


def telemetry_preflight():
    """只读检查现有GPU计数器能力；不输出其他进程的PID、名称或具体用量。"""
    query = None
    try:
        query = _PDHQuery()
        raw = query.collect()
        valid_counts = {metric: sum(r["counter_status"] in (0, 1) and r["bytes"] >= 0 for r in values)
                        for metric, values in raw.items()}
        pids = {int(match.group(1)) for values in raw.values() for row in values
                if (match := INSTANCE_RE.fullmatch(row["instance"] or ""))}
        matched_count = sum(_extract_pid(raw, pid)["status"] == "ok" for pid in pids)
        return {"status": "available" if all(valid_counts.values()) and matched_count else "unavailable",
            "valid_provider_instance_counts": valid_counts, "paired_process_count": matched_count,
            "cuda_context_created": False, "model_loaded": False, "errors": []}
    except Exception as error:
        return {"status": "unavailable", "valid_provider_instance_counts": None, "paired_process_count": None,
                "cuda_context_created": False, "model_loaded": False, "errors": [f"{type(error).__name__}: {error}"]}
    finally:
        if query is not None:
            query.close()


class WindowsGPUMemorySampler:
    """采样当前/指定PID；start/close配对，sample可在阶段边界同步调用。"""
    def __init__(self, pid=None, interval=0.1):
        self.pid = os.getpid() if pid is None else int(pid)
        self.interval = float(interval)
        if self.pid <= 0 or self.interval <= 0:
            raise ValueError("pid and interval must be positive")
        self.stage = "before_model_load"
        self.samples, self.errors = [], []
        self.started = time.perf_counter()
        self._query = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False

    def sample(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Sampler already closed")
            row = {"elapsed_seconds": time.perf_counter() - self.started, "stage": self.stage, "pid": self.pid}
            try:
                if self._query is None:
                    self._query = _PDHQuery()
                row.update(_extract_pid(self._query.collect(), self.pid))
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                self.errors.append({"elapsed_seconds": row["elapsed_seconds"], "stage": row["stage"], "error": message})
                row.update(status="unavailable", dedicated_bytes=None, shared_bytes=None,
                           matched_instances=0, instances=[], error=message)
            row["collection_seconds"] = time.perf_counter() - self.started - row["elapsed_seconds"]
            self.samples.append(row)
            return row

    def start(self):
        if self._thread is not None or self._closed:
            raise RuntimeError("Sampler cannot be restarted")
        self.sample()
        def loop():
            while not self._stop.wait(self.interval):
                self.sample()
        self._thread = threading.Thread(target=loop, name="gpu-memory-pdh", daemon=True)
        self._thread.start()

    def close(self):
        if self._closed:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("PDH sampling thread did not stop")
        self.sample()
        with self._lock:
            if self._query is not None:
                self._query.close()
            self._closed = True

    def report(self):
        with self._lock:
            samples = list(self.samples)
            errors = list(self.errors)
        valid = [r for r in samples if r["status"] == "ok"]
        per_adapter = {}
        for sample in valid:
            for row in sample["instances"]:
                key = f"{row['adapter_luid']}_phys_{row['physical_adapter_index']}"
                entry = per_adapter.setdefault(key, {"adapter_luid": row["adapter_luid"],
                    "physical_adapter_index": row["physical_adapter_index"], "sample_count": 0,
                    "peak_dedicated_bytes": 0, "peak_shared_bytes": 0})
                entry["sample_count"] += 1
                entry["peak_dedicated_bytes"] = max(entry["peak_dedicated_bytes"], row["dedicated_bytes"])
                entry["peak_shared_bytes"] = max(entry["peak_shared_bytes"], row["shared_bytes"])
        for entry in per_adapter.values():
            entry["peak_dedicated_mib"] = entry["peak_dedicated_bytes"] / 2**20
            entry["peak_shared_mib"] = entry["peak_shared_bytes"] / 2**20
        peak_dedicated = max((r["dedicated_bytes"] for r in valid), default=None)
        peak_shared = max((r["shared_bytes"] for r in valid), default=None)
        return {"pid": self.pid, "source": "Windows PDH GPU Process Memory Dedicated Usage / Shared Usage",
            "requested_interval_seconds": self.interval, "sample_count": len(samples), "valid_sample_count": len(valid),
            "matched_sample_count": len(valid),
            "not_registered_count": sum(r["status"] == "not_registered" for r in samples),
            "unavailable_count": sum(r["status"] == "unavailable" for r in samples),
            "status": "available" if valid else "unavailable",
            "peak_dedicated_bytes": peak_dedicated, "peak_shared_bytes": peak_shared,
            "peak_dedicated_mib": None if peak_dedicated is None else peak_dedicated / 2**20,
            "peak_shared_mib": None if peak_shared is None else peak_shared / 2**20,
            "per_adapter": per_adapter, "by_adapter": per_adapter,
            "adapter_instances": sorted({r["instance"] for s in valid for r in s["instances"]}),
            "samples": samples, "errors": errors,
            "limitations": "Per-process WDDM accounting sampled in time; not whole-device occupancy or guaranteed instantaneous peaks. Zero is reported only for valid counter observations. Shared usage is not automatically proof of driver fallback."}


if __name__ == "__main__":
    # 只读探测：此进程不创建CUDA上下文，PID未注册应保持None。
    print(json.dumps({"telemetry_preflight": telemetry_preflight()}, ensure_ascii=False))
    sampler = WindowsGPUMemorySampler()
    sampler.start()
    sampler._stop.wait(0.35)
    sampler.close()
    report = sampler.report()
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=False, indent=2))
    print(json.dumps({"provider_instance_counts": report["samples"][-1].get("provider_instance_counts"),
        "last_status": report["samples"][-1]["status"], "last_collection_seconds": report["samples"][-1]["collection_seconds"]}))
