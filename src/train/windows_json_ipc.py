"""Windows JSON snapshots: delete-sharing readers and atomic replacement.

One writer owns a target; readers see a complete old or new snapshot. There are
no retries, ACL changes, CUDA imports, or fallbacks that hide malformed JSON.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes as wt
from contextlib import contextmanager
import json
import msvcrt
import os
from pathlib import Path
import threading
import time
import traceback
import uuid

if os.name != 'nt':
    raise RuntimeError('windows_json_ipc requires Windows')

_kernel = ctypes.WinDLL('kernel32', use_last_error=True)
_CreateFile = _kernel.CreateFileW
_CreateFile.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p,
                        wt.DWORD, wt.DWORD, wt.HANDLE]
_CreateFile.restype = wt.HANDLE
_CloseHandle = _kernel.CloseHandle
_CloseHandle.argtypes = [wt.HANDLE]
_CloseHandle.restype = wt.BOOL
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_GENERIC_READ, _OPEN_EXISTING, _NORMAL = 0x80000000, 3, 0x80
_SHARE_READ_WRITE_DELETE = 0x1 | 0x2 | 0x4


def _record(evidence, operation, path, status, *, phase, error=None, **values):
    if evidence is None:
        return
    evidence({'operation': operation, 'phase': phase, 'stage': phase, 'path': str(path),
        'status': status, 'pid': os.getpid(), 'thread_id': threading.get_ident(),
        'monotonic': time.perf_counter(), 'errno': getattr(error, 'errno', None),
        'winerror': getattr(error, 'winerror', None),
        'error_type': type(error).__name__ if error else None,
        'error': str(error) if error else None,
        'traceback': traceback.format_exc() if error else None,
        'automatic_retries': 0, **values})


@contextmanager
def open_shared_binary(path, evidence=None):
    """Yield a binary file owning a READ|WRITE|DELETE-sharing native handle.

    A caller may hold this old snapshot open while the writer replaces its name.
    Ownership moves exactly once: HANDLE -> CRT fd -> Python file object.
    """
    path = Path(os.path.abspath(os.fspath(path)))
    handle = fd = stream = None
    acquired = False
    phase = 'CreateFileW'
    _record(evidence, 'open_shared_binary', path, 'started', phase=phase, share_mode=7)
    try:
        handle = _CreateFile(str(path), _GENERIC_READ, _SHARE_READ_WRITE_DELETE,
                             None, _OPEN_EXISTING, _NORMAL, None)
        if handle == _INVALID_HANDLE:
            code = ctypes.get_last_error()
            handle = None
            raise ctypes.WinError(code)
        phase = 'open_osfhandle'
        fd = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
        handle = None  # The CRT fd now owns the HANDLE.
        phase = 'fdopen'
        stream = os.fdopen(fd, 'rb')
        fd = None  # The Python file now owns the CRT fd.
        _record(evidence, 'open_shared_binary', path, 'ok', phase=phase, share_mode=7)
        acquired = True
    except Exception as error:
        _record(evidence, 'open_shared_binary', path, 'failed', phase=phase, error=error)
        raise
    finally:
        # Failed acquisition still owns either HANDLE or fd, never both.
        if not acquired:
            try:
                if stream is not None:
                    stream.close()
                elif fd is not None:
                    os.close(fd)
                elif handle is not None and not _CloseHandle(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            except Exception as error:
                _record(evidence, 'close_read', path, 'failed', phase='acquisition_cleanup', error=error)
                raise
    try:
        yield stream
    finally:
        try:
            stream.close()
        except Exception as error:
            _record(evidence, 'close_read', path, 'failed', phase='file.close', error=error)
            raise
        _record(evidence, 'close_read', path, 'ok', phase='file.close')


def read_json(path, missing_ok=False, evidence=None):
    """Read one snapshot; only explicit missing_ok + WinError 2 returns None.

    Missing directory, access denied, sharing violations and malformed JSON raise.
    No retries or cached-data fallback; evidence callback errors also propagate.
    """
    path = Path(os.path.abspath(os.fspath(path)))
    phase = 'open_shared_binary'
    _record(evidence, 'read_json', path, 'started', phase=phase, share_mode=7)
    try:
        with open_shared_binary(path, evidence=evidence) as stream:
            phase = 'file.read'
            raw = stream.read()
            phase = 'decode_json'
            value = json.loads(raw.decode('utf-8'))
    except Exception as error:
        allowed_missing = (phase == 'open_shared_binary' and missing_ok
                           and getattr(error, 'winerror', None) == 2)
        _record(evidence, 'read_json', path, 'missing' if allowed_missing else 'failed',
                phase=phase, error=error, missing_allowed=allowed_missing)
        if allowed_missing:
            return None
        raise
    _record(evidence, 'read_json', path, 'ok', phase='decode_json', bytes_read=len(raw), share_mode=7)
    return value


def write_json_atomic(path, value, evidence=None):
    """Write/fsync an exclusive sibling temporary, close it, then os.replace.

    Retain the temporary on any failure. The parent directory must already exist.
    This controls a single target write, not a multi-file transaction or ACLs.
    """
    path = Path(os.path.abspath(os.fspath(path)))
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    phase = 'serialize'
    _record(evidence, 'write_json_atomic', path, 'started', phase=phase, temporary=str(temporary))
    try:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        phase = 'write_temp'
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _record(evidence, 'write_temp', path, 'ok', phase=phase,
                temporary=str(temporary), bytes_written=len(payload))
        phase = 'os.replace'
        os.replace(temporary, path)
    except Exception as error:
        _record(evidence, 'write_json_atomic', path, 'failed', phase=phase,
                error=error, temporary=str(temporary))
        raise
    _record(evidence, 'write_json_atomic', path, 'ok', phase=phase,
            temporary=str(temporary), bytes_written=len(payload))
