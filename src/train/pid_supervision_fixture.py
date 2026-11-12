"""Pure CPU launcher/worker fixture: intentionally includes a second process layer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


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


def launcher(mode):
    directory = Path(os.environ['PID_RUN_DIR'])
    write(directory / 'fixture_launcher.json', {'pid': os.getpid(), 'parent_pid': os.getppid(), 'mode': mode})
    child = subprocess.Popen([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()), '--worker', mode],
        cwd=ROOT, env=dict(os.environ), creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    write(directory / 'fixture_launcher_child.json', {'launcher_pid': os.getpid(), 'popen_child_pid': child.pid})
    event(directory, 'fixture_child_started', child_pid=child.pid)
    code = child.wait()
    event(directory, 'fixture_child_exited', exit_code=code)
    return code


def worker(mode):
    directory = Path(os.environ['PID_RUN_DIR'])
    nonce = os.environ['PID_RUN_NONCE']
    # Publish the true fixture PID separately from the protocol input we may corrupt.
    write(directory / 'fixture_worker.json', {'pid': os.getpid(), 'parent_pid': os.getppid(), 'mode': mode})
    write(directory / 'stage.json', {'stage': 'cpu_identity_handshake', 'micro_started_monotonic': None,
                                    'telemetry_required': False, 'telemetry_complete': False})
    from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack
    if mode == 'forged_pid':
        # Never briefly publish a valid identity in the watched location.
        staging = directory / 'forged_identity_staging'
        staging.mkdir(exist_ok=False)
        publish_worker_identity(staging, nonce)
        forged = read_optional(staging / 'worker_identity.json')
        forged['pid'] = int(os.environ['PID_SUPERVISOR_PID'])
        forged['parent_pid'] = os.getpid()
        write(directory / 'worker_identity.json', forged)
    else:
        publish_worker_identity(directory, nonce + '-invalid' if mode == 'invalid_nonce' else nonce)
    event(directory, 'identity_published', mode=mode)
    wait_for_identity_ack(directory, nonce, timeout=8)
    event(directory, 'identity_acknowledged')
    write(directory / 'stage.json', {'stage': 'cpu_active', 'micro_started_monotonic': None,
                                    'telemetry_required': True, 'telemetry_complete': False})
    deadline = time.perf_counter() + 20
    while time.perf_counter() < deadline:
        stop = read_optional(directory / 'stop_request.json')
        if stop and mode == 'cooperative':
            event(directory, 'cooperative_stop', reason=stop.get('reason'))
            write(directory / 'worker_result.json', {'status': 'stopped', 'pid': os.getpid(),
                                                    'torch_imported': 'torch' in sys.modules})
            return 2
        if mode == 'normal' and read_optional(directory / 'telemetry_ready.json'):
            valid = []
            sample_path = directory / 'pid_memory.jsonl'
            if sample_path.exists():
                for line in sample_path.read_text(encoding='utf-8').splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Last append may still be in progress.
                    if row.get('status') == 'ok' and row.get('pid') == os.getpid():
                        valid.append(row)
            if len(valid) >= 3:
                write(directory / 'stage.json', {'stage': 'cpu_complete', 'micro_started_monotonic': None,
                                                'telemetry_required': True, 'telemetry_complete': True})
                event(directory, 'completion_requested')
                while time.perf_counter() < deadline:
                    closed = read_optional(directory / 'telemetry_closed.json')
                    if closed:
                        if closed.get('nonce') != nonce or closed.get('worker_pid') != os.getpid():
                            raise AssertionError('Completion acknowledgement belongs to another worker')
                        event(directory, 'completion_acknowledged')
                        write(directory / 'worker_result.json', {'status': 'pass', 'pid': os.getpid(),
                            'valid_samples_seen': len(valid), 'torch_imported': 'torch' in sys.modules})
                        return 0
                    time.sleep(0.02)
                raise TimeoutError('CPU completion acknowledgement timeout')
        time.sleep(0.02)
    raise TimeoutError('CPU fixture outlived its independent deadline')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launcher', choices=['normal', 'ignore_stop', 'cooperative', 'invalid_nonce', 'forged_pid'])
    group.add_argument('--worker', choices=['normal', 'ignore_stop', 'cooperative', 'invalid_nonce', 'forged_pid'])
    args = parser.parse_args()
    raise SystemExit(launcher(args.launcher) if args.launcher else worker(args.worker))
