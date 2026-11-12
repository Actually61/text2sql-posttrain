"""One bounded named Windows mutex for every access to a stage snapshot.

The same normalized stage path selects the same Local-session mutex. There is
one wait call, no retry, and abandoned ownership is released before failing.
"""
from __future__ import annotations
from contextlib import contextmanager
import ctypes
from ctypes import wintypes as wt
import errno
import hashlib
import math
import os
from pathlib import Path
import threading
import time
import traceback

if os.name != 'nt':
    raise RuntimeError('windows_stage_mutex requires Windows')

_kernel = ctypes.WinDLL('kernel32', use_last_error=True)
_CreateMutex = _kernel.CreateMutexW
_CreateMutex.argtypes = [ctypes.c_void_p, wt.BOOL, wt.LPCWSTR]
_CreateMutex.restype = wt.HANDLE
_Wait = _kernel.WaitForSingleObject
_Wait.argtypes = [wt.HANDLE, wt.DWORD]
_Wait.restype = wt.DWORD
_Release = _kernel.ReleaseMutex
_Release.argtypes = [wt.HANDLE]
_Release.restype = wt.BOOL
_Close = _kernel.CloseHandle
_Close.argtypes = [wt.HANDLE]
_Close.restype = wt.BOOL
WAIT_OBJECT_0, WAIT_ABANDONED, WAIT_TIMEOUT, WAIT_FAILED = 0, 0x80, 0x102, 0xFFFFFFFF


def mutex_identity(directory):
    """Use one canonical directory; stage itself must not be a symlink/hard link.

    Resolve only the directory, avoiding an unlocked lookup of the live stage file.
    """
    path = os.path.normcase(str(Path(directory).resolve() / 'stage.json'))
    name = 'Local\\Text2SQLStage_' + hashlib.sha256(path.encode('utf-8')).hexdigest()
    return path, name


@contextmanager
def locked_stage(directory, wait_seconds=1.0, evidence=None):
    """Own the stage mutex, or fail after one bounded wait without entering.

    Yield ordinary metadata; the native handle stays private. Every caller must
    close its stage read handle before leaving this context. WAIT_ABANDONED
    grants ownership, but is treated as failure because protected state is unknown.
    """
    if (isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float))
            or not math.isfinite(wait_seconds) or not 0 < wait_seconds <= 1.0):
        raise ValueError('Mutex wait must be finite and in (0, 1] seconds')
    milliseconds = min(1000, math.ceil(wait_seconds * 1000))
    path, name = mutex_identity(directory)
    handle, owned = None, False
    phase, wait_result = 'create', None

    def record(current_phase, status, error=None, **fields):
        if evidence is not None:
            evidence({'operation': 'mutex', 'phase': current_phase, 'status': status,
                'path': path, 'mutex_name': name, 'pid': os.getpid(),
                'thread_id': threading.get_ident(), 'monotonic': time.perf_counter(),
                'wait_timeout_ms': milliseconds, 'wait_result': wait_result,
                'errno': getattr(error, 'errno', None), 'winerror': getattr(error, 'winerror', None),
                'error_type': type(error).__name__ if error else None,
                'error': str(error) if error else None,
                'traceback': traceback.format_exc() if error else None,
                'automatic_retries': 0, **fields})

    try:
        record('create', 'started')
        handle = _CreateMutex(None, False, name)  # No initial ownership or inherited handle.
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        phase = 'wait'
        record('wait', 'started')
        wait_started = time.perf_counter()
        wait_result = int(_Wait(handle, milliseconds))  # Exactly one kernel wait.
        wait_error = ctypes.get_last_error() if wait_result == WAIT_FAILED else None
        waited = time.perf_counter() - wait_started
        if wait_result in (WAIT_OBJECT_0, WAIT_ABANDONED):
            owned = True
        if wait_result == WAIT_TIMEOUT:
            raise TimeoutError(errno.ETIMEDOUT, 'Stage mutex wait timed out; no retry')
        if wait_result == WAIT_FAILED:
            raise ctypes.WinError(wait_error)
        if wait_result == WAIT_ABANDONED:
            raise RuntimeError('Stage mutex was abandoned; reject unknown protected state')
        if wait_result != WAIT_OBJECT_0:
            raise RuntimeError(f'Unexpected mutex wait result: {wait_result}')
        phase = 'acquired'
        record('acquired', 'ok', wait_seconds=waited)
        phase = 'critical_section'
        yield {'path': path, 'mutex_name': name, 'wait_seconds': waited,
               'wait_timeout_ms': milliseconds}
    except BaseException as error:
        record(phase, 'failed', error=error)
        raise
    finally:
        try:
            if owned:
                # A failed diagnostic callback must not prevent releasing ownership.
                try:
                    record('releasing', 'started')
                finally:
                    try:
                        if not _Release(handle):
                            raise ctypes.WinError(ctypes.get_last_error())
                        owned = False
                    except BaseException as error:
                        record('release', 'failed', error=error)
                        raise
                    record('released', 'ok')
        finally:
            if handle:
                try:
                    if not _Close(handle):
                        raise ctypes.WinError(ctypes.get_last_error())
                except BaseException as error:
                    record('close', 'failed', error=error)
                    raise
                record('closed', 'ok')
