"""One model-free GPU telemetry probe, gated by verified worker identity.

The launcher never imports torch. The worker makes exactly one <=1 MiB test
tensor after identity acknowledgement. No model, optimizer or dataset is used.
"""
from __future__ import annotations
import os
import sys
import time
import threading

# Independent deadline also covers imports or CUDA initialization that may block.
DEADLINE = min(time.perf_counter() + 25.0, float(os.environ['PID_CHILD_DEADLINE']))


def self_expire():
    time.sleep(max(0.0, DEADLINE - time.perf_counter()))
    os._exit(92)


threading.Thread(target=self_expire, daemon=True).start()

import argparse
import json
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.train.windows_job_process_v2 import ProcessHandle, VENV_PYTHON, native_venv_launch
from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack

TEST_TENSOR_BYTES = 1024 * 1024
PHASE_SAMPLE_COUNT = 3
PHASE_SECONDS = 5.0


def write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


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


def check_stop(directory):
    if time.perf_counter() >= DEADLINE:
        raise TimeoutError('Worker independent deadline reached')
    request = read_optional(directory / 'stop_request.json')
    if request:
        raise RuntimeError(f"Supervisor stop: {request.get('reason', 'unspecified')}")


def identity(role):
    with ProcessHandle(os.getpid()) as process:
        value = process.identity()
    value.update(role=role, nonce=os.environ['PID_RUN_NONCE'], sys_executable=sys.executable,
                 sys_prefix=sys.prefix, independent_deadline_monotonic=DEADLINE,
                 forbidden_imports=[name for name in ('torch', 'transformers', 'peft', 'bitsandbytes')
                                    if name in sys.modules])
    return value


def wait_membership(directory, expected_pids):
    while time.perf_counter() < DEADLINE:
        check_stop(directory)
        ack = read_optional(directory / 'membership_ack.json')
        if ack is not None:
            if ack.get('nonce') != os.environ['PID_RUN_NONCE'] or set(ack.get('retained_pids', [])) != set(expected_pids):
                raise ValueError('Membership acknowledgement does not match this process tree')
            return
        time.sleep(0.01)
    raise TimeoutError('Membership acknowledgement deadline')


def launcher():
    directory = Path(os.environ['PID_RUN_DIR'])
    write(directory / 'fixture_launcher.json', identity('launcher'))
    event(directory, 'launcher_identity_recorded', self_deadline=DEADLINE)
    command, child_env = native_venv_launch(
        [str(VENV_PYTHON), '-X', 'utf8', str(Path(__file__).resolve()), '--worker'], dict(os.environ))
    child = subprocess.Popen(command, cwd=ROOT, env=child_env, creationflags=subprocess.CREATE_NO_WINDOW)
    write(directory / 'fixture_launcher_child.json', {'launcher_pid': os.getpid(), 'popen_child_pid': child.pid,
                                                     'native_command': command})
    event(directory, 'fixture_child_started', child_pid=child.pid)
    write(directory / 'fixture_launcher_ready.json', {'pid': os.getpid(), 'logs_flushed': True})
    wait_membership(directory, [os.getpid(), child.pid])
    code = child.wait()
    event(directory, 'fixture_child_exited', exit_code=code)
    return code


def set_stage(directory, name, *, required, complete=False):
    started = time.perf_counter()
    write(directory / 'stage.json', {'stage': name, 'phase_started_monotonic': started,
        'micro_started_monotonic': None, 'telemetry_required': required, 'telemetry_complete': complete})
    event(directory, 'stage_started', stage=name, phase_started_monotonic=started,
          telemetry_required=required, telemetry_complete=complete)
    return started


def phase_samples(directory, stage, started):
    path = directory / 'pid_memory.jsonl'
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        if index == len(lines) - 1 and not text.endswith('\n'):
            continue  # Only a trailing append in progress may be ignored.
        row = json.loads(line)
        if row.get('pid') == os.getpid() and row.get('stage') == stage and row.get('sampled_monotonic', -1) >= started:
            if row.get('sample_source') != 'Windows PDH':
                raise ValueError('GPU probe requires real Windows PDH samples')
            rows.append(row)
    return rows


def wait_phase_samples(directory, nonce, stage, started, *, registering=False):
    deadline = min(started + PHASE_SECONDS, DEADLINE)
    while time.perf_counter() < deadline:
        check_stop(directory)
        rows = phase_samples(directory, stage, started)
        valid = []
        seen = set()
        registered = not registering
        for row in rows:
            status = row.get('status')
            if status == 'ok':
                if any(not isinstance(row.get(key), int) or row[key] < 0 for key in ('dedicated_bytes', 'shared_bytes')):
                    raise ValueError('Valid PDH sample contains missing or negative memory values')
                registered = True
                timestamp = row['sampled_monotonic']
                if timestamp not in seen:
                    seen.add(timestamp)
                    valid.append(row)
            elif status == 'not_registered' and not registered:
                continue
            else:
                raise RuntimeError(f'Invalid PDH sample in {stage}: {status}')
        ready = read_optional(directory / 'telemetry_ready.json')
        if ready and (ready.get('nonce') != nonce or ready.get('worker_pid') != os.getpid()):
            raise ValueError('Telemetry readiness belongs to another worker')
        if len(valid) >= PHASE_SAMPLE_COUNT and ready:
            return {'valid_sample_count': len(valid), 'all_stage_sample_count': len(rows),
                    'not_registered_count': sum(row.get('status') == 'not_registered' for row in rows),
                    'sample_timestamps': [row['sampled_monotonic'] for row in valid],
                    'first_valid_monotonic': valid[0]['sampled_monotonic'],
                    'last_valid_monotonic': valid[-1]['sampled_monotonic'],
                    'peak_dedicated_bytes': max(row['dedicated_bytes'] for row in valid),
                    'peak_shared_bytes': max(row['shared_bytes'] for row in valid),
                    'adapter_instances': sorted({instance['instance'] for row in valid for instance in row.get('instances', [])})}
        time.sleep(0.02)
    raise TimeoutError(f'Fewer than three valid same-PID PDH samples in phase {stage}')


def torch_memory(torch):
    return {'allocated_bytes': torch.cuda.memory_allocated(0), 'reserved_bytes': torch.cuda.memory_reserved(0),
            'max_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'max_reserved_bytes': torch.cuda.max_memory_reserved(0)}


def worker():
    directory = Path(os.environ['PID_RUN_DIR'])
    nonce = os.environ['PID_RUN_NONCE']
    own_identity = identity('worker')
    write(directory / 'fixture_worker.json', own_identity)
    event(directory, 'worker_identity_recorded', self_deadline=DEADLINE)
    set_stage(directory, 'before_identity_ack', required=False)
    write(directory / 'fixture_worker_ready.json', {'pid': os.getpid(), 'logs_flushed': True})
    result = {'status': 'running', 'pid': os.getpid(), 'phases': [], 'cuda_initialized': False,
              'cuda_initialization_attempted': False,
              'test_tensor_requested_bytes': TEST_TENSOR_BYTES, 'test_tensor_bytes': 0,
              'allocation_attempts': 0, 'successful_test_allocations': 0,
              'model_loaded': False, 'training_updates': 0, 'final_examples_read': False,
              'automatic_retries': 0, 'independent_deadline_monotonic': DEADLINE}
    test_tensor = None
    try:
        wait_membership(directory, [own_identity['parent_pid'], os.getpid()])
        while not (directory / 'fixture_launcher_ready.json').exists():
            check_stop(directory)
            time.sleep(0.01)
        publish_worker_identity(directory, nonce)
        event(directory, 'identity_published')
        wait_for_identity_ack(directory, nonce, timeout=max(0.01, DEADLINE - time.perf_counter()))
        result['identity_ack_received_monotonic'] = time.perf_counter()
        event(directory, 'identity_acknowledged', acknowledgement=read_optional(directory / 'identity_ack.json'))
        write(directory / 'gpu_gate.json', {'worker_pid': os.getpid(), 'nonce': nonce,
            'identity_ack_received_monotonic': result['identity_ack_received_monotonic'],
            'torch_imported': 'torch' in sys.modules, 'cuda_initialization_requested': False})
        if own_identity['forbidden_imports'] or 'torch' in sys.modules:
            raise RuntimeError('Torch/ML module imported before the identity gate')
        if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
            raise RuntimeError('GPU child inherited CUDA_VISIBLE_DEVICES empty; do not override silently')
        check_stop(directory)
        set_stage(directory, 'torch_import', required=False)
        result['torch_import_started_monotonic'] = time.perf_counter()
        event(directory, 'torch_import_started')
        import torch
        result['torch_import_finished_monotonic'] = time.perf_counter()
        result['torch_version'] = torch.__version__
        result['cuda_initialized_before_explicit_init'] = torch.cuda.is_initialized()
        event(directory, 'torch_import_finished', cuda_initialized=torch.cuda.is_initialized())
        if result['cuda_initialized_before_explicit_init']:
            raise RuntimeError('CUDA initialized before the explicit initialization point')
        check_stop(directory)
        set_stage(directory, 'cuda_initializing', required=False)
        result['cuda_init_started_monotonic'] = time.perf_counter()
        result['cuda_initialization_attempted'] = True
        result['cuda_initialized'] = None  # Until init returns, partial initialization is unknown.
        event(directory, 'cuda_init_started')
        torch.cuda.init()
        torch.cuda.synchronize(0)
        result['cuda_init_finished_monotonic'] = time.perf_counter()
        result['cuda_initialized'] = torch.cuda.is_initialized()
        if not result['cuda_initialized']:
            raise RuntimeError('Explicit CUDA initialization did not complete')
        properties = torch.cuda.get_device_properties(0)
        result['gpu'] = {'device': 'cuda:0', 'name': properties.name, 'total_memory_bytes': properties.total_memory,
                         'compute_capability': [properties.major, properties.minor]}
        event(directory, 'cuda_init_finished', gpu=result['gpu'])
        write(directory / 'probe_progress.json', result)

        # Context baseline: no explicit test tensor has been allocated.
        stage = 'cuda_context_only'
        started = set_stage(directory, stage, required=False)
        memory = torch_memory(torch)
        sample_info = wait_phase_samples(directory, nonce, stage, started, registering=True)
        result['phases'].append({'stage': stage, 'started_monotonic': started,
            'finished_monotonic': time.perf_counter(), 'test_tensor_live_bytes': 0,
            'torch_memory': memory, **sample_info})
        write(directory / 'probe_progress.json', result)
        check_stop(directory)

        # Exactly one extra GPU tensor, uint8[1048576], so its storage is <=1 MiB.
        result['allocation_attempts'] = 1
        event(directory, 'test_tensor_allocation_requested', requested_bytes=TEST_TENSOR_BYTES)
        test_tensor = torch.empty(TEST_TENSOR_BYTES, dtype=torch.uint8, device='cuda:0')
        result['successful_test_allocations'] = 1
        result['test_tensor_bytes'] = test_tensor.numel() * test_tensor.element_size()
        if result['test_tensor_bytes'] != TEST_TENSOR_BYTES:
            raise ValueError('Unexpected test tensor storage size')
        test_tensor.fill_(1)
        torch.cuda.synchronize(0)
        stage = 'tensor_live'
        started = set_stage(directory, stage, required=True)
        memory = torch_memory(torch)
        sample_info = wait_phase_samples(directory, nonce, stage, started)
        result['phases'].append({'stage': stage, 'started_monotonic': started,
            'finished_monotonic': time.perf_counter(), 'test_tensor_live_bytes': result['test_tensor_bytes'],
            'torch_memory': memory, **sample_info})
        write(directory / 'probe_progress.json', result)
        check_stop(directory)

        del test_tensor
        test_tensor = None
        torch.cuda.synchronize(0)
        torch.cuda.empty_cache()
        stage = 'tensor_released'
        started = set_stage(directory, stage, required=True)
        memory = torch_memory(torch)
        sample_info = wait_phase_samples(directory, nonce, stage, started)
        result['phases'].append({'stage': stage, 'started_monotonic': started,
            'finished_monotonic': time.perf_counter(), 'test_tensor_live_bytes': 0,
            'torch_memory': memory, **sample_info})
        write(directory / 'probe_progress.json', result)
        check_stop(directory)

        set_stage(directory, 'gpu_probe_complete', required=True, complete=True)
        event(directory, 'completion_requested')
        while time.perf_counter() < DEADLINE:
            check_stop(directory)
            closed = read_optional(directory / 'telemetry_closed.json')
            if closed:
                if closed.get('nonce') != nonce or closed.get('worker_pid') != os.getpid():
                    raise ValueError('Completion acknowledgement belongs to another worker')
                event(directory, 'completion_acknowledged', valid_sample_count=closed.get('valid_sample_count'))
                result.update(status='pass', completion_ack_received_monotonic=time.perf_counter(),
                              telemetry_closed=closed, test_tensor_released=True,
                              forbidden_imports=[name for name in ('transformers', 'peft', 'bitsandbytes') if name in sys.modules])
                if result['forbidden_imports']:
                    raise RuntimeError('Unexpected model-training library import')
                write(directory / 'worker_result.json', result)
                return 0
            time.sleep(0.01)
        raise TimeoutError('GPU probe completion acknowledgement deadline')
    except Exception as error:
        # Only cleanup and evidence remain; do not retry CUDA initialization or allocation.
        test_tensor = None
        result.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                      failed_monotonic=time.perf_counter())
        event(directory, 'worker_failed', error=result['error'])
        write(directory / 'worker_result.json', result)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launcher', action='store_true')
    group.add_argument('--worker', action='store_true')
    arguments = parser.parse_args()
    raise SystemExit(launcher() if arguments.launcher else worker())
