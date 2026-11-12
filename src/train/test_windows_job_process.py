"""Real Windows CPU-only launcher/worker/grandchild containment test."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def publish(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value), encoding='utf-8')
    pending.replace(path)


def child(role, folder):
    folder = Path(folder)
    publish(folder / f'{role}.json', {'pid': os.getpid(), 'parent_pid': os.getppid()})
    print(f'{role} actual PID {os.getpid()}', flush=True)
    if role == 'grandchild':
        time.sleep(30)
    else:
        next_role = 'worker' if role == 'launcher' else 'grandchild'
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--child', next_role, str(folder)],
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        process.wait()


def main():
    from src.train.windows_job_process import JobProcess
    attempt = ROOT / 'outputs/checks/windows_job_lowlevel' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    attempt.mkdir(parents=True, exist_ok=False)
    job, held = None, []
    report = {'status': 'failed', 'attempt_directory': str(attempt), 'torch_imported': False, 'cuda_initialized': False}
    started = time.perf_counter()
    try:
        job = JobProcess([sys.executable, str(Path(__file__).resolve()), '--child', 'launcher', str(attempt)],
                         ROOT, dict(os.environ), attempt / 'console.log')
        deadline = time.perf_counter() + 10
        while not (attempt / 'grandchild.json').exists():
            if job.poll() is not None:
                raise AssertionError('Launcher exited before grandchild published identity')
            if time.perf_counter() >= deadline:
                raise TimeoutError('CPU descendant identity was not published')
            time.sleep(0.02)
        identities = {}
        for role in ('launcher', 'worker', 'grandchild'):
            reported = json.loads((attempt / f'{role}.json').read_text(encoding='utf-8'))
            handle = job.open_process(reported['pid'])
            held.append(handle)
            info = handle.identity()
            assert info['parent_pid'] == reported['parent_pid']
            assert handle.alive() and job.contains(info['pid'])
            identities[role] = info
        assert job.start_order == ['created_suspended', 'assigned_to_kill_on_close_job', 'resumed']
        assert job.contains(job.pid)
        assert not job.contains(os.getpid())
        before = job.active_pids()
        assert all(info['pid'] in before for info in identities.values())
        job.terminate_tree(124, timeout=5)
        after = job.active_pids()
        assert after == [] and job.poll() == 124
        exits = [handle.wait(0) for handle in held]
        assert all(not handle.alive() for handle in held)
        assert exits == [124, 124, 124]
        assert 'torch' not in sys.modules
        report.update(status='pass', launcher_pid=job.pid, identities=identities,
            active_pids_before=before, active_pids_after=after, retained_handle_exit_codes=exits,
            startup_order=job.start_order, elapsed_seconds=time.perf_counter() - started,
            checks=['suspended_assignment_before_resume', 'real_worker_and_grandchild_in_job',
                    'outside_process_rejected', 'retained_creation_identity', 'whole_tree_termination_and_empty_job'])
    except BaseException as error:
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        for handle in held:
            handle.close()
        if job is not None:
            job.close()
        sources = ['src/train/windows_job_process.py', 'src/train/test_windows_job_process.py']
        report['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
        (attempt / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if report['status'] == 'pass':
            destination = ROOT / 'reports/windows_job_cpu_checks.json'
            with destination.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': report['status'], 'checks': len(report['checks']),
                      'active_pids_after': report['active_pids_after'], 'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        child(sys.argv[2], sys.argv[3])
    else:
        main()
