"""One persistent nvidia-smi helper with bounded freshness and strict cleanup.

No CUDA/model libraries are imported.  A containing Windows Job (provided by
the inference controller) is still required for an OS-level lifetime bound:
Python cannot forcibly interrupt a kernel process-creation call or a thread.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback


class GPUSampler:
    """Sample whole-device memory; no retries, restarts, or final extra query."""
    def __init__(self, interval, *, command_factory=None, first_timeout=5.0,
                 stale_timeout=5.0, cleanup_timeout=2.0, expected_total_mib=None,
                 event_path=None):
        for name, value in (("interval", interval), ("first_timeout", first_timeout),
                            ("stale_timeout", stale_timeout), ("cleanup_timeout", cleanup_timeout)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name}")
        if expected_total_mib is not None and (not math.isfinite(expected_total_mib) or expected_total_mib <= 0):
            raise ValueError("Invalid expected total memory")
        self.interval = float(interval)
        self.first_timeout, self.stale_timeout = float(first_timeout), float(stale_timeout)
        self.cleanup_timeout = float(cleanup_timeout)
        self.command_factory = command_factory
        self.samples, self.errors, self.events = [], [], []
        self.stage = "before_load"
        self.started = None
        self._condition = threading.Condition(threading.RLock())
        self._stop = threading.Event()
        self._threads = []
        self._process = None
        self._last_valid = None
        self._total_mib = expected_total_mib
        self._fatal = None
        self._start_called = False
        self._closing = False
        self._closed = False
        self._close_result = None
        self._stdout_lines = self._stderr_lines = 0
        self._launches = 0
        self._command = None
        self._executable_sha256 = None
        self._max_line_bytes = 8192
        self.event_path = None if event_path is None else Path(event_path)
        self._event_stream = None
        self._log_failed = False

    def _event(self, event, **values):
        with self._condition:
            now = time.perf_counter()
            row = {"event": event, "monotonic": now,
                "elapsed_seconds": None if self.started is None else now-self.started,
                "stage": self.stage, "pid": None if self._process is None else self._process.pid, **values}
            self.events.append(row)
            if self._event_stream is not None and not self._log_failed:
                try:
                    self._event_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    self._event_stream.flush()
                    if event == 'fatal':
                        os.fsync(self._event_stream.fileno())
                except Exception as error:
                    self._log_failed = True
                    self.events.append({'event':'event_log_exception','monotonic':time.perf_counter(),
                        'error':f'{type(error).__name__}: {error}','traceback':traceback.format_exc()})
                    self._fail('event_log_error', f'{type(error).__name__}: {error}')

    def _fail(self, kind, message, **values):
        with self._condition:
            if self._fatal is None:
                self._fatal = {"kind": kind, "message": message, "monotonic": time.perf_counter(), **values}
                self.errors.append(f"{kind}: {message}")
                self._event("fatal", **self._fatal)
                self._stop.set()
                self._condition.notify_all()

    def _thread(self, name, target):
        with self._condition:
            thread = threading.Thread(name=name, target=target, daemon=True)
            self._threads.append(thread)
            thread.start()
            return thread

    def _command_spec(self):
        interval_ms = max(1, round(self.interval * 1000))
        if self.command_factory is not None:
            specification = self.command_factory(interval_ms)
            if isinstance(specification, tuple) and len(specification) == 2:
                command, env = specification
            else:
                command, env = specification, None
        else:
            found = shutil.which("nvidia-smi")
            if found is None:
                raise FileNotFoundError("nvidia-smi was not found on PATH")
            command = [str(Path(found).resolve()), "-i", "0",
                "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits",
                f"--loop-ms={interval_ms}"]
            env = None
        if not isinstance(command, (list, tuple)) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError("Command factory must return an argument list or (argument list, environment)")
        executable = Path(command[0]).resolve(strict=True)
        if not executable.is_file():
            raise ValueError("Helper executable is not a file")
        with executable.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self._executable_sha256 = digest
        return [str(executable), *command[1:]], env

    def start(self):
        with self._condition:
            if self._start_called or self._closing:
                raise RuntimeError("Sampler is one-shot and cannot restart")
            self._start_called = True
            self.started = time.perf_counter()
            if self.event_path is not None:
                self._event_stream = self.event_path.open('x', encoding='utf-8', newline='\n')
            self._event("start_requested", first_timeout_seconds=self.first_timeout,
                stale_timeout_seconds=self.stale_timeout, interval_seconds=self.interval)
        self._thread("gpu-v2-watchdog", self._watchdog)
        self._thread("gpu-v2-launch", self._launch)
        with self._condition:
            deadline = self.started + self.first_timeout
            while not self.samples and self._fatal is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    self._fail("first_sample_timeout", "No valid first sample before deadline")
                    break
                self._condition.wait(remaining)
        # Match the old public interface: caller checks errors immediately.

    def _launch(self):
        begun = time.perf_counter()
        try:
            command, env = self._command_spec()
            with self._condition:
                self._command = command
                if self._closing or self._fatal is not None:
                    self._event("launch_cancelled_before_spawn")
                    return
                self._launches += 1
                self._event("spawn_started", command=command, executable_sha256=self._executable_sha256)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            with self._condition:
                self._process = process
                self._event("spawn_returned", create_seconds=time.perf_counter()-begun)
                late = self._closing or self._fatal is not None
                if not late:
                    self._thread("gpu-v2-stdout", lambda: self._read_stream(process.stdout, "stdout"))
                    self._thread("gpu-v2-stderr", lambda: self._read_stream(process.stderr, "stderr"))
            if late:
                self._event("late_spawn_cleanup_started")
                process.kill()
                process.wait(timeout=self.cleanup_timeout)
                for stream in (process.stdout, process.stderr):
                    stream.close()
                self._event("late_spawn_cleanup_finished", returncode=process.returncode)
        except BaseException as error:
            self._event("launch_exception", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            self._fail("launch_error", f"{type(error).__name__}: {error}")

    def _read_stream(self, stream, channel):
        try:
            while True:
                raw = stream.readline(self._max_line_bytes + 1)
                if not raw:
                    self._event("stream_eof", channel=channel)
                    with self._condition:
                        if not self._closing and self._fatal is None:
                            code = None if self._process is None else self._process.poll()
                            self._fail("unexpected_helper_exit", f"{channel} closed before shutdown; returncode={code}", returncode=code)
                    return
                if len(raw) > self._max_line_bytes:
                    self._fail("oversized_output", f"{channel} line exceeded {self._max_line_bytes} bytes")
                    return
                text = raw.decode("utf-8", errors="strict").rstrip("\r\n")
                with self._condition:
                    if channel == "stdout": self._stdout_lines += 1
                    else: self._stderr_lines += 1
                    self._event("line_received", channel=channel, text=text)
                    if self._closing or self._fatal is not None:
                        return
                if channel == "stdout":
                    self._accept(text)
        except BaseException as error:
            self._event("reader_exception", channel=channel, error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc())
            with self._condition:
                if not self._closing:
                    self._fail("reader_error", f"{channel}: {type(error).__name__}: {error}")

    def _accept(self, text):
        try:
            values = next(csv.reader([text], strict=True))
            if len(values) != 2:
                raise ValueError("Expected exactly two CSV fields")
            used, total = (float(value.strip()) for value in values)
            if not all(math.isfinite(v) for v in (used, total)) or total <= 0 or not 0 <= used <= total:
                raise ValueError("Memory values must be finite and satisfy 0 <= used <= total")
            with self._condition:
                if self._closing or self._fatal is not None:
                    return
                now = time.perf_counter()
                deadline = self.started+self.first_timeout if self._last_valid is None else self._last_valid+self.stale_timeout
                if now > deadline:
                    self._fail("late_sample", "Valid-looking sample arrived after freshness deadline")
                    return
                if self._total_mib is None:
                    self._total_mib = total
                if total != self._total_mib:
                    raise ValueError(f"Device total changed: {total} != {self._total_mib}")
                self._last_valid = now
                self.samples.append({"elapsed_seconds": round(now-self.started, 6), "stage": self.stage,
                    "used_mib": used, "total_mib": total})
                self._event("valid_sample", sample_number=len(self.samples), used_mib=used, total_mib=total)
                self._condition.notify_all()
        except BaseException as error:
            self._event("parse_exception", text=text, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
            self._fail("invalid_sample", f"{type(error).__name__}: {error}")

    def _watchdog(self):
        while not self._stop.wait(0.025):
            with self._condition:
                if self._closing or self._fatal is not None:
                    return
                if self._process is not None:
                    code = self._process.poll()
                    if code is not None:
                        self._event("helper_exit_observed", returncode=code)
                        self._fail("unexpected_helper_exit", f"Helper exited before shutdown: {code}", returncode=code)
                        return
                now = time.perf_counter()
                anchor = self.started if self._last_valid is None else self._last_valid
                timeout = self.first_timeout if self._last_valid is None else self.stale_timeout
                if now-anchor > timeout:
                    kind = "first_sample_timeout" if self._last_valid is None else "stale_sample_timeout"
                    self._fail(kind, f"No valid sample within {timeout} seconds", last_valid_monotonic=self._last_valid)
                    return

    def close(self):
        with self._condition:
            if self._closed:
                if self._close_result["status"] != "pass":
                    raise RuntimeError(self._close_result["error"])
                return
            if self._closing:
                raise RuntimeError("Concurrent close is not permitted")
            self._closing = True
            self._stop.set()
            deadline = time.perf_counter()+self.cleanup_timeout
            self._event("close_started", deadline_monotonic=deadline)
        problems = []
        process = self._process
        if process is not None:
            try:
                if process.poll() is None:
                    self._event("helper_termination_requested")
                    process.kill()
                process.wait(timeout=max(0.0, deadline-time.perf_counter()))
                self._event("helper_exit_confirmed", returncode=process.returncode)
            except BaseException as error:
                problems.append(f"Process cleanup: {type(error).__name__}: {error}")
                self._event("cleanup_exception", error=problems[-1], traceback=traceback.format_exc())
        # The launch thread is included; no new readers can be created after
        # _closing is set. A late spawn performs its own termination and wait.
        for thread in list(self._threads):
            thread.join(timeout=max(0.0, deadline-time.perf_counter()))
            if thread.is_alive():
                problems.append(f"Thread did not exit before common deadline: {thread.name}")
        process = self._process
        if process is not None and process.poll() is None:
            problems.append(f"Helper still alive: {process.pid}")
        if not problems and process is not None:
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except BaseException as error:
                    problems.append(f"Pipe close: {type(error).__name__}: {error}")
        with self._condition:
            self._event('cleanup_resources_finished', problems=list(problems))
            stream, self._event_stream = self._event_stream, None
            if stream is not None:
                try:
                    stream.flush(); os.fsync(stream.fileno()); stream.close()
                except BaseException as error:
                    problems.append(f'Event log close: {type(error).__name__}: {error}')
            completed = time.perf_counter()
            if completed > deadline:
                problems.append(f'Common cleanup deadline exceeded by {completed-deadline:.6f} seconds')
            self._closed = True
            self._close_result = {"status": "failed" if problems else "pass",
                "deadline_monotonic": deadline, "completed_monotonic": completed,
                "process_returncode": None if process is None else process.returncode,
                "alive_threads": [thread.name for thread in self._threads if thread.is_alive()],
                "error": "; ".join(problems) if problems else None}
            self._event("close_finished", **self._close_result)
            if problems:
                self.errors.append("CleanupError: " + "; ".join(problems))
                raise RuntimeError(self.errors[-1])

    def report(self):
        with self._condition:
            return {"source": "one persistent nvidia-smi --loop-ms helper", "interval_seconds": self.interval,
                "first_timeout_seconds": self.first_timeout, "stale_timeout_seconds": self.stale_timeout,
                "cleanup_timeout_seconds": self.cleanup_timeout, "helper_launches": self._launches,
                "helper_pid": None if self._process is None else self._process.pid,
                "helper_returncode": None if self._process is None else self._process.poll(),
                "command": self._command, "executable_sha256": self._executable_sha256,
                "event_log_path": None if self.event_path is None else str(self.event_path),
                "event_log_failed": self._log_failed,
                "sample_count": len(self.samples), "stdout_lines": self._stdout_lines, "stderr_lines": self._stderr_lines,
                "samples": list(self.samples), "errors": list(self.errors), "events": list(self.events),
                "fatal": None if self._fatal is None else dict(self._fatal), "close": self._close_result,
                "automatic_restarts": 0, "automatic_retries": 0,
                "limitation": "Sampled whole-GPU occupancy, not instantaneous process peak. External Job supervision supplies the hard OS lifetime bound."}
