"""Windows process-tree ownership: suspend, assign a kill-on-close Job, then run.

This module uses only the standard library. It never imports Torch or creates CUDA.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes as wt
import os
from pathlib import Path
import subprocess
import time

if os.name != 'nt':
    raise OSError('Windows Job Objects are required')

SIZE_T = ctypes.c_size_t
ULONG_PTR = ctypes.c_size_t
WAIT_OBJECT_0, WAIT_TIMEOUT, WAIT_FAILED = 0, 258, 0xFFFFFFFF
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
PROCESS_QUERY_LIMITED_INFORMATION, SYNCHRONIZE = 0x1000, 0x100000
CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT, CREATE_NO_WINDOW = 4, 0x400, 0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


class _STARTUPINFO(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('lpReserved', wt.LPWSTR), ('lpDesktop', wt.LPWSTR),
        ('lpTitle', wt.LPWSTR), ('dwX', wt.DWORD), ('dwY', wt.DWORD), ('dwXSize', wt.DWORD),
        ('dwYSize', wt.DWORD), ('dwXCountChars', wt.DWORD), ('dwYCountChars', wt.DWORD),
        ('dwFillAttribute', wt.DWORD), ('dwFlags', wt.DWORD), ('wShowWindow', wt.WORD),
        ('cbReserved2', wt.WORD), ('lpReserved2', ctypes.POINTER(wt.BYTE)),
        ('hStdInput', wt.HANDLE), ('hStdOutput', wt.HANDLE), ('hStdError', wt.HANDLE)]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [('hProcess', wt.HANDLE), ('hThread', wt.HANDLE), ('dwProcessId', wt.DWORD),
                ('dwThreadId', wt.DWORD)]


class _BASIC_LIMIT(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
        ('LimitFlags', wt.DWORD), ('MinimumWorkingSetSize', SIZE_T), ('MaximumWorkingSetSize', SIZE_T),
        ('ActiveProcessLimit', wt.DWORD), ('Affinity', ULONG_PTR), ('PriorityClass', wt.DWORD),
        ('SchedulingClass', wt.DWORD)]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount',
        'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _EXTENDED_LIMIT(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _BASIC_LIMIT), ('IoInfo', _IO_COUNTERS),
        ('ProcessMemoryLimit', SIZE_T), ('JobMemoryLimit', SIZE_T),
        ('PeakProcessMemoryUsed', SIZE_T), ('PeakJobMemoryUsed', SIZE_T)]


class _PROCESSENTRY32(ctypes.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD), ('th32ProcessID', wt.DWORD),
        ('th32DefaultHeapID', ULONG_PTR), ('th32ModuleID', wt.DWORD), ('cntThreads', wt.DWORD),
        ('th32ParentProcessID', wt.DWORD), ('pcPriClassBase', wt.LONG), ('dwFlags', wt.DWORD),
        ('szExeFile', wt.WCHAR * 260)]


_api = ctypes.WinDLL('kernel32', use_last_error=True)


def _bind(name, args, result):
    fn = getattr(_api, name)
    fn.argtypes, fn.restype = args, result
    return fn


_CreateJob = _bind('CreateJobObjectW', [ctypes.c_void_p, wt.LPCWSTR], wt.HANDLE)
_SetJob = _bind('SetInformationJobObject', [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD], wt.BOOL)
_AssignJob = _bind('AssignProcessToJobObject', [wt.HANDLE, wt.HANDLE], wt.BOOL)
_QueryJob = _bind('QueryInformationJobObject', [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD)], wt.BOOL)
_TerminateJob = _bind('TerminateJobObject', [wt.HANDLE, wt.UINT], wt.BOOL)
_IsInJob = _bind('IsProcessInJob', [wt.HANDLE, wt.HANDLE, ctypes.POINTER(wt.BOOL)], wt.BOOL)
_Close = _bind('CloseHandle', [wt.HANDLE], wt.BOOL)
_CreateProcess = _bind('CreateProcessW', [wt.LPCWSTR, wt.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
    wt.BOOL, wt.DWORD, ctypes.c_void_p, wt.LPCWSTR, ctypes.POINTER(_STARTUPINFO), ctypes.POINTER(_PROCESS_INFORMATION)], wt.BOOL)
_ResumeThread = _bind('ResumeThread', [wt.HANDLE], wt.DWORD)
_TerminateProcess = _bind('TerminateProcess', [wt.HANDLE, wt.UINT], wt.BOOL)
_Wait = _bind('WaitForSingleObject', [wt.HANDLE, wt.DWORD], wt.DWORD)
_ExitCode = _bind('GetExitCodeProcess', [wt.HANDLE, ctypes.POINTER(wt.DWORD)], wt.BOOL)
_OpenProcess = _bind('OpenProcess', [wt.DWORD, wt.BOOL, wt.DWORD], wt.HANDLE)
_GetProcessId = _bind('GetProcessId', [wt.HANDLE], wt.DWORD)
_GetTimes = _bind('GetProcessTimes', [wt.HANDLE] + [ctypes.POINTER(wt.FILETIME)] * 4, wt.BOOL)
_ImageName = _bind('QueryFullProcessImageNameW', [wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)], wt.BOOL)
_Snapshot = _bind('CreateToolhelp32Snapshot', [wt.DWORD, wt.DWORD], wt.HANDLE)
_ProcessFirst = _bind('Process32FirstW', [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32)], wt.BOOL)
_ProcessNext = _bind('Process32NextW', [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32)], wt.BOOL)


def _check(ok, operation):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error(), operation)


def _parent_pid(pid):
    snapshot = _Snapshot(2, 0)  # TH32CS_SNAPPROCESS; no WMI/CIM dependency.
    if snapshot == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error(), 'CreateToolhelp32Snapshot')
    try:
        entry = _PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        found = _ProcessFirst(snapshot, ctypes.byref(entry))
        while found:
            if entry.th32ProcessID == pid:
                return int(entry.th32ParentProcessID)
            found = _ProcessNext(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() not in (0, 18):  # ERROR_NO_MORE_FILES
            raise ctypes.WinError(ctypes.get_last_error(), 'Process32NextW')
        raise ProcessLookupError(f'PID {pid} disappeared before its parent could be checked')
    finally:
        _Close(snapshot)


class ProcessHandle:
    """An opened kernel process object pins identity even if its numeric PID is reused."""
    def __init__(self, pid):
        self.pid = int(pid)
        self.handle = _OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, self.pid)
        _check(self.handle, 'OpenProcess')
        try:
            if _GetProcessId(self.handle) != self.pid or not self.alive():
                raise ProcessLookupError(f'PID {pid} is not a live process')
            creation, exit_time, kernel, user = (wt.FILETIME() for _ in range(4))
            _check(_GetTimes(self.handle, ctypes.byref(creation), ctypes.byref(exit_time),
                            ctypes.byref(kernel), ctypes.byref(user)), 'GetProcessTimes')
            size = wt.DWORD(32768)
            image = ctypes.create_unicode_buffer(size.value)
            _check(_ImageName(self.handle, 0, image, ctypes.byref(size)), 'QueryFullProcessImageNameW')
            self._identity = {'pid': self.pid, 'parent_pid': _parent_pid(self.pid),
                'creation_time_100ns': (int(creation.dwHighDateTime) << 32) | int(creation.dwLowDateTime),
                'executable': image.value}
        except BaseException:
            self.close()
            raise

    def identity(self):
        return dict(self._identity)

    def alive(self):
        if not self.handle:
            return False
        status = _Wait(self.handle, 0)
        if status == WAIT_FAILED:
            raise ctypes.WinError(ctypes.get_last_error(), 'WaitForSingleObject')
        return status == WAIT_TIMEOUT

    def wait(self, timeout=None):
        if not self.handle:
            raise RuntimeError('Process handle is closed')
        milliseconds = 0xFFFFFFFF if timeout is None else min(0xFFFFFFFE, max(0, int(timeout * 1000)))
        status = _Wait(self.handle, milliseconds)
        if status == WAIT_TIMEOUT:
            raise TimeoutError(f'PID {self.pid} did not exit before the deadline')
        if status != WAIT_OBJECT_0:
            raise ctypes.WinError(ctypes.get_last_error(), 'WaitForSingleObject')
        code = wt.DWORD()
        _check(_ExitCode(self.handle, ctypes.byref(code)), 'GetExitCodeProcess')
        return int(code.value)

    def close(self):
        if self.handle:
            _Close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class JobProcess:
    """Own a suspended launcher and all descendants before any user code can run."""
    def __init__(self, command, cwd, env, stdout_path):
        import msvcrt
        if not command or any(not isinstance(part, str) or '\0' in part for part in command):
            raise ValueError('command must be nonempty strings without NUL')
        self.job = self.handle = None
        self.pid = None
        self._closed = False
        self.start_order = []
        stdout_path = Path(stdout_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.job = _CreateJob(None, None)  # Non-inheritable; child must not keep the Job alive.
        _check(self.job, 'CreateJobObjectW')
        info = _PROCESS_INFORMATION()
        try:
            limits = _EXTENDED_LIMIT()
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            _check(_SetJob(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)), 'SetInformationJobObject')
            with stdout_path.open('xb', buffering=0) as output, open(os.devnull, 'rb', buffering=0) as null:
                os.set_inheritable(output.fileno(), True)
                os.set_inheritable(null.fileno(), True)
                startup = _STARTUPINFO()
                startup.cb = ctypes.sizeof(startup)
                startup.dwFlags = 0x100 | 1  # STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW
                startup.wShowWindow = 0
                startup.hStdInput = msvcrt.get_osfhandle(null.fileno())
                startup.hStdOutput = startup.hStdError = msvcrt.get_osfhandle(output.fileno())
                environment = ctypes.create_unicode_buffer('\0'.join(
                    f'{key}={value}' for key, value in sorted(env.items(), key=lambda pair: pair[0].upper())) + '\0\0')
                command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
                _check(_CreateProcess(None, command_line, None, None, True,
                    CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW,
                    environment, str(Path(cwd).resolve()), ctypes.byref(startup), ctypes.byref(info)), 'CreateProcessW')
                self.handle, self.pid = info.hProcess, int(info.dwProcessId)
                self.start_order.append('created_suspended')
                _check(_AssignJob(self.job, self.handle), 'AssignProcessToJobObject')
                self.start_order.append('assigned_to_kill_on_close_job')
                result = _ResumeThread(info.hThread)
                if result == 0xFFFFFFFF:
                    raise ctypes.WinError(ctypes.get_last_error(), 'ResumeThread')
                self.start_order.append('resumed')
        except BaseException:
            if self.handle:
                _TerminateProcess(self.handle, 125)  # On assignment failure it is still suspended.
                _Wait(self.handle, 5000)
                _Close(self.handle)
                self.handle = None
            if self.job:
                _Close(self.job)
                self.job = None
            self._closed = True
            raise
        finally:
            if info.hThread:
                _Close(info.hThread)

    def poll(self):
        status = _Wait(self.handle, 0)
        if status == WAIT_TIMEOUT:
            return None
        if status != WAIT_OBJECT_0:
            raise ctypes.WinError(ctypes.get_last_error(), 'WaitForSingleObject')
        code = wt.DWORD()
        _check(_ExitCode(self.handle, ctypes.byref(code)), 'GetExitCodeProcess')
        return int(code.value)

    def wait(self, timeout=None):
        """Wait for the launcher only; active_pids() separately checks the entire tree."""
        milliseconds = 0xFFFFFFFF if timeout is None else min(0xFFFFFFFE, max(0, int(timeout * 1000)))
        status = _Wait(self.handle, milliseconds)
        if status == WAIT_TIMEOUT:
            raise TimeoutError('Launcher did not exit before the deadline')
        if status != WAIT_OBJECT_0:
            raise ctypes.WinError(ctypes.get_last_error(), 'WaitForSingleObject')
        return self.poll()

    def active_pids(self):
        if not self.job:
            raise RuntimeError('Job handle is closed')
        capacity = 16
        for _ in range(10):
            buffer = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ULONG_PTR))
            used = wt.DWORD()
            ok = _QueryJob(self.job, 3, buffer, ctypes.sizeof(buffer), ctypes.byref(used))
            assigned, count = (wt.DWORD * 2).from_buffer(buffer)
            if ok:
                if count > capacity:
                    raise RuntimeError('Invalid Job process list size')
                return sorted(int(pid) for pid in (ULONG_PTR * count).from_buffer(buffer, 8))
            if ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error(), 'QueryInformationJobObject')
            capacity = max(capacity * 2, int(assigned))
        raise RuntimeError('Job process list did not stabilize')

    def open_process(self, pid):
        process = ProcessHandle(pid)
        try:
            contained = wt.BOOL()
            _check(_IsInJob(process.handle, self.job, ctypes.byref(contained)), 'IsProcessInJob')
            if not contained.value:
                raise ValueError(f'PID {pid} is outside this Job')
            return process
        except BaseException:
            process.close()
            raise

    def contains(self, pid):
        try:
            with self.open_process(pid):
                return True
        except (ProcessLookupError, ValueError):
            return False
        except OSError as error:
            if error.winerror == 87:  # ERROR_INVALID_PARAMETER for a PID that no longer exists.
                return False
            raise

    def identity(self, pid):
        with self.open_process(pid) as process:
            return process.identity()

    def terminate_tree(self, code=124, timeout=5.0):
        _check(_TerminateJob(self.job, int(code)), 'TerminateJobObject')
        deadline = time.monotonic() + timeout
        while self.active_pids():
            if time.monotonic() >= deadline:
                raise TimeoutError('Job still contains active processes after termination')
            time.sleep(0.01)
        self.wait(max(0, deadline - time.monotonic()))

    def close(self):
        if self._closed:
            return
        try:
            if self.job and self.active_pids():
                self.terminate_tree()
        finally:
            if self.handle:
                _Close(self.handle)
                self.handle = None
            if self.job:
                _Close(self.job)  # KILL_ON_JOB_CLOSE is the final containment backstop.
                self.job = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
