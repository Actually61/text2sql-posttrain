"""Frozen CPython 3.11.9 native launch adapter; preserve the existing venv.

Only the launch route changes. The original failed Job implementation stays intact.
__PYVENV_LAUNCHER__ is an internal CPython contract, guarded by exact file hashes.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes as wt
import hashlib
import os
from pathlib import Path

from src.train import windows_job_process as old

ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / '.venv/Scripts/python.exe'
NATIVE_PYTHON = Path(r'C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.11_3.11.2544.0_x64__qbz5n2kfra8p0\python.exe')
BASE_ALIAS = Path(os.environ['LOCALAPPDATA']) / r'Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe'
EXPECTED_FILES = {
    str(NATIVE_PYTHON): 'ea21d2833698f5bf74e44b7125855336a7bc1291be6983532109b016f9859257',
    str(VENV_PYTHON): '21bb438c0d4a6f1f164b9a646f6ee000340185e5871180aec06db8d3f07c0082',
    str(ROOT / '.venv/pyvenv.cfg'): '7e82e4d52c4e0cd207cd33f5236bc1b47db34139d2c277f999c6b843868c7c55',
}
ProcessHandle = old.ProcessHandle


def normalize(path):
    return os.path.normcase(os.path.abspath(str(path)))


def interpreter_evidence():
    rows = []
    for name, expected in EXPECTED_FILES.items():
        path = Path(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        rows.append({'path': name, 'expected_sha256': expected, 'actual_sha256': actual,
                     'size': path.stat().st_size if path.is_file() else None, 'matches': actual == expected})
    return rows


def native_venv_launch(command, env):
    """Adapt every spawn separately; CPython consumes the launcher variable."""
    if not command or normalize(command[0]) != normalize(VENV_PYTHON):
        raise ValueError('Adapter only accepts this frozen project venv entry point')
    if not all(row['matches'] for row in interpreter_evidence()):
        raise ValueError('Frozen interpreter files changed; do not guess a replacement')
    upper_env = {key.upper(): value for key, value in env.items()}
    if any(upper_env.get(key) for key in ('PYTHONHOME', 'PYTHONEXECUTABLE')) or any(flag in command for flag in ('-S', '-E', '-I')):
        raise ValueError('This verified venv launch contract requires normal site/environment initialization')
    child_env = dict(env)
    child_env['__PYVENV_LAUNCHER__'] = str(VENV_PYTHON)
    native_command = [str(NATIVE_PYTHON), *command[1:]]
    return native_command, child_env


class JobProcess(old.JobProcess):
    def __init__(self, command, cwd, env, stdout_path):
        native_command, child_env = native_venv_launch(command, env)
        self.launch_route = {'requested_command': list(command), 'native_command': native_command,
                             'venv_launcher_environment': str(VENV_PYTHON)}
        super().__init__(native_command, cwd, child_env, stdout_path)

    def settings_evidence(self):
        info = old._EXTENDED_LIMIT()
        used = wt.DWORD()
        ok = old._QueryJob(self.job, 9, ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(used))
        error = ctypes.get_last_error() if not ok else 0
        flags = int(info.BasicLimitInformation.LimitFlags) if ok else None
        return {'api_success': bool(ok), 'winerror': error, 'returned_bytes': int(used.value),
                'structure_bytes': ctypes.sizeof(info), 'limit_flags': flags,
                'kill_on_close': bool(flags & 0x2000) if flags is not None else None,
                'breakaway_ok': bool(flags & 0x800) if flags is not None else None,
                'silent_breakaway_ok': bool(flags & 0x1000) if flags is not None else None}

    def process_evidence(self, pid):
        """Collect first, without asserting membership or dropping useful identity."""
        try:
            with ProcessHandle(pid) as handle:
                member, any_job = wt.BOOL(), wt.BOOL()
                success = old._IsInJob(handle.handle, self.job, ctypes.byref(member))
                error = ctypes.get_last_error() if not success else 0
                any_success = old._IsInJob(handle.handle, None, ctypes.byref(any_job))
                any_error = ctypes.get_last_error() if not any_success else 0
                return {'identity': handle.identity(), 'alive': handle.alive(),
                    'is_in_job_api_success': bool(success), 'is_in_job_winerror': error,
                    'is_in_this_job': bool(member.value) if success else None,
                    'is_in_any_job_api_success': bool(any_success), 'is_in_any_job_winerror': any_error,
                    'is_in_any_job': bool(any_job.value) if any_success else None}
        except Exception as error:
            return {'pid': pid, 'error': f'{type(error).__name__}: {error}'}
