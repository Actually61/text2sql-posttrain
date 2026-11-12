"""Two-role CPU-only fixture with a common eight-second independent deadline."""
from __future__ import annotations
import os
import sys
import time
import threading

# Start the safety deadline before importing the rest of the fixture.
DEADLINE = min(time.perf_counter() + 8.0, float(os.environ['PID_CHILD_DEADLINE']))


def self_expire():
    time.sleep(max(0.0, DEADLINE - time.perf_counter()))
    os._exit(92)


threading.Thread(target=self_expire, daemon=True).start()

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from src.train.windows_job_process_v2 import ProcessHandle, VENV_PYTHON, native_venv_launch
from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def event(directory, name, **values):
    row = {'event': name, 'pid': os.getpid(), 'monotonic': time.perf_counter(), **values}
    with (directory / f'fixture_events_{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_optional(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None


def identity(role, mode):
    with ProcessHandle(os.getpid()) as process:
        value = process.identity()
    value.update(role=role, mode=mode, nonce=os.environ['PID_RUN_NONCE'],
                 sys_executable=sys.executable, sys_prefix=sys.prefix,
                 independent_deadline_monotonic=DEADLINE,
                 forbidden_imports=[name for name in ('torch', 'transformers', 'peft', 'bitsandbytes')
                                    if name in sys.modules])
    return value


def wait_membership(directory, expected_pids):
    while time.perf_counter() < DEADLINE:
        ack = read_optional(directory / 'membership_ack.json')
        if ack is not None:
            if ack.get('nonce') != os.environ['PID_RUN_NONCE'] or set(ack.get('retained_pids', [])) != set(expected_pids):
                raise ValueError('Membership acknowledgement does not match this process tree')
            return ack
        time.sleep(0.01)
    raise TimeoutError('Membership acknowledgement deadline')


def launcher(mode):
    directory = Path(os.environ['PID_RUN_DIR'])
    own_identity = identity('launcher', mode)
    write(directory / 'fixture_launcher.json', own_identity)
    event(directory, 'launcher_identity_recorded', self_deadline=DEADLINE)
    command, child_env = native_venv_launch(
        [str(VENV_PYTHON), '-X', 'utf8', str(Path(__file__).resolve()), '--worker', mode], dict(os.environ))
    child = subprocess.Popen(command, cwd=ROOT, env=child_env, creationflags=subprocess.CREATE_NO_WINDOW)
    write(directory / 'fixture_launcher_child.json', {'launcher_pid': os.getpid(), 'popen_child_pid': child.pid,
                                                     'native_command': command})
    event(directory, 'fixture_child_started', child_pid=child.pid)
    # All pre-run launcher logs are durable before the worker can publish an invalid identity.
    write(directory / 'fixture_launcher_ready.json', {'pid': os.getpid(), 'logs_flushed': True})
    wait_membership(directory, [os.getpid(), child.pid])
    code = child.wait()  # The independent daemon deadline remains active while waiting.
    event(directory, 'fixture_child_exited', exit_code=code)
    return code


def valid_samples(directory):
    sample_path = directory / 'pid_memory.jsonl'
    if not sample_path.exists():
        return []
    lines = sample_path.read_text(encoding='utf-8').splitlines()
    valid = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue  # The parent may still be appending the last line.
            raise
        if row.get('status') == 'ok' and row.get('pid') == os.getpid():
            valid.append(row)
    return valid


def finish_after_sampler_closed(directory, nonce, *, status, exit_code, valid_count=None, reason=None):
    write(directory / 'stage.json', {'stage': 'cpu_complete' if exit_code == 0 else 'cpu_stopping',
                                    'micro_started_monotonic': None,
                                    'telemetry_required': True, 'telemetry_complete': True})
    event(directory, 'completion_requested', status=status)
    while time.perf_counter() < DEADLINE:
        closed = read_optional(directory / 'telemetry_closed.json')
        if closed:
            if closed.get('nonce') != nonce or closed.get('worker_pid') != os.getpid():
                raise ValueError('Completion acknowledgement belongs to another worker')
            event(directory, 'completion_acknowledged', status=status)
            write(directory / 'worker_result.json', {'status': status, 'pid': os.getpid(),
                'reason': reason, 'valid_samples_seen': valid_count, 'torch_imported': 'torch' in sys.modules})
            return exit_code
        time.sleep(0.01)
    raise TimeoutError('CPU completion acknowledgement deadline')


def worker(mode):
    directory = Path(os.environ['PID_RUN_DIR'])
    nonce = os.environ['PID_RUN_NONCE']
    own_identity = identity('worker', mode)
    write(directory / 'fixture_worker.json', own_identity)
    event(directory, 'worker_identity_recorded', self_deadline=DEADLINE)
    write(directory / 'stage.json', {'stage': 'cpu_identity_handshake', 'micro_started_monotonic': None,
                                    'telemetry_required': False, 'telemetry_complete': False})
    write(directory / 'fixture_worker_ready.json', {'pid': os.getpid(), 'logs_flushed': True})
    wait_membership(directory, [own_identity['parent_pid'], os.getpid()])
    while not (directory / 'fixture_launcher_ready.json').exists():
        if time.perf_counter() >= DEADLINE:
            raise TimeoutError('Launcher durable log readiness deadline')
        time.sleep(0.01)
    if mode == 'invalid_nonce':
        event(directory, 'publishing_invalid_identity', mode=mode)
        # This publication is the last write: the parent can now reject and terminate us.
        publish_worker_identity(directory, nonce + '-invalid')
        while True:
            time.sleep(0.02)
    publish_worker_identity(directory, nonce)
    event(directory, 'identity_published', mode=mode)
    wait_for_identity_ack(directory, nonce, timeout=max(0.01, DEADLINE - time.perf_counter()))
    event(directory, 'identity_acknowledged')
    write(directory / 'stage.json', {'stage': 'cpu_active', 'micro_started_monotonic': None,
                                    'telemetry_required': True, 'telemetry_complete': False})
    ignored_logged = False
    while time.perf_counter() < DEADLINE:
        stop = read_optional(directory / 'stop_request.json')
        if stop and mode == 'cooperative':
            event(directory, 'cooperative_stop', reason=stop.get('reason'))
            return finish_after_sampler_closed(directory, nonce, status='stopped', exit_code=20,
                                               reason=stop.get('reason'))
        if stop and mode == 'ignore_stop' and not ignored_logged:
            event(directory, 'stop_ignored', reason=stop.get('reason'))
            ignored_logged = True
        ready = read_optional(directory / 'telemetry_ready.json')
        if mode == 'normal' and ready:
            if ready.get('nonce') != nonce or ready.get('worker_pid') != os.getpid():
                raise ValueError('Telemetry readiness belongs to another worker')
            valid = valid_samples(directory)
            if len(valid) >= 3:
                return finish_after_sampler_closed(directory, nonce, status='pass', exit_code=0,
                                                   valid_count=len(valid))
        time.sleep(0.01)
    return 92


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = ['normal', 'cooperative', 'ignore_stop', 'invalid_nonce']
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launcher', choices=modes)
    group.add_argument('--worker', choices=modes)
    args = parser.parse_args()
    raise SystemExit(launcher(args.launcher) if args.launcher else worker(args.worker))
