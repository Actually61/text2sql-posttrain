"""One CPU acceptance run, one three-role tree, no CUDA/ML imports or retries."""
from __future__ import annotations
import os
import sys
import time
import threading

# Every role has an independent deadline; cleanup never relies solely on the Job under test.
if '--child' in sys.argv:
    deadline = min(time.perf_counter() + 8.0, float(os.environ['JOB_CPU_DEADLINE']))
    def self_expire():
        time.sleep(max(0, deadline - time.perf_counter()))
        os._exit(92)
    threading.Thread(target=self_expire, daemon=True).start()

import hashlib
from importlib import metadata
import json
from pathlib import Path
import secrets
import shutil
import site
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from src.train.windows_job_process_v2 import JobProcess, ProcessHandle, VENV_PYTHON, NATIVE_PYTHON, BASE_ALIAS, normalize, native_venv_launch, interpreter_evidence
from src.train.pid_supervision import verify_identity
from src.train import windows_job_process as win
from src.train.windows_job_exit_wait import terminate_and_wait, exit_timing, close_without_new_wait

RUN = ROOT / 'outputs/checks/windows_job_v3_cpu_v1'
REPORT = ROOT / 'reports/windows_job_v3_cpu_checks.json'
SOURCES = ['src/train/windows_job_exit_wait.py', 'src/train/test_windows_job_v3.py',
           'src/train/windows_job_process_v2.py', 'src/train/test_windows_job_v2.py',
           'src/train/windows_job_process.py', 'src/train/pid_supervision.py', 'src/train/cache_release_watchdog.py']


def write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def event(directory, name, **values):
    row = {'event': name, 'pid': os.getpid(), 'monotonic': time.perf_counter(), **values}
    with (directory / f'events_{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def environment_identity(role, nonce):
    with ProcessHandle(os.getpid()) as process:
        info = process.identity()
    package_names = read(ROOT / 'reports/sft_resume_pilot_preflight.json')['packages'].keys()
    info.update(role=role, nonce=nonce, sys_executable=sys.executable,
                sys_base_executable=sys._base_executable, sys_prefix=sys.prefix, sys_base_prefix=sys.base_prefix,
                python_version=list(sys.version_info[:3]), site_packages=site.getsitepackages(),
                package_versions={name: metadata.version(name) for name in package_names},
                package_locations={name: str(metadata.distribution(name).locate_file('')) for name in ('torch', 'peft', 'transformers')},
                launcher_variable_consumed='__PYVENV_LAUNCHER__' not in os.environ,
                forbidden_imports=[name for name in ('torch', 'transformers', 'peft', 'bitsandbytes') if name in sys.modules])
    return info


def child(role, directory):
    directory = Path(directory)
    nonce = os.environ['JOB_CPU_NONCE']
    identity = environment_identity(role, nonce)
    write(directory / f'{role}.json', identity)
    event(directory, 'identity_published', role=role, self_deadline=deadline)
    print(f'{role} identity published PID={os.getpid()}', flush=True)
    if role != 'grandchild':
        next_role = 'worker' if role == 'launcher' else 'grandchild'
        command, child_env = native_venv_launch([str(VENV_PYTHON), '-X', 'utf8', str(Path(__file__).resolve()),
                                               '--child', next_role, str(directory)], dict(os.environ))
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, creationflags=subprocess.CREATE_NO_WINDOW)
        write(directory / f'{role}_spawn.json', {'parent_pid': os.getpid(), 'popen_pid': process.pid,
                                               'native_command': command, 'venv_context': str(VENV_PYTHON)})
        write(directory / f'{role}_ready.json', {'pid': os.getpid(), 'role': role, 'logs_flushed': True})
        return process.wait(timeout=max(0.1, deadline - time.perf_counter()))
    # Alive until the acceptance controller kills the Job; 92 is a distinct self-expiry code.
    write(directory / f'{role}_ready.json', {'pid': os.getpid(), 'role': role, 'logs_flushed': True})
    while time.perf_counter() < deadline:
        time.sleep(0.02)
    return 92


def verify_frozen():
    protected = read(ROOT / 'reports/cache_release_pilot_preflight.json')['protected_sha256']
    protected.update(read(ROOT / 'reports/pid_supervision_repair_result.json')['source_sha256'])
    protected.update(read(ROOT / 'reports/windows_job_v2_cpu_checks.json')['source_sha256'])
    protected.update(read(ROOT / 'reports/windows_job_v3_preservation_manifest.json'))
    for name, expected in protected.items():
        with (ROOT / name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Old frozen file changed: {name}')
    for old_run in ('outputs/runs/cache_release_pilot_v1', 'outputs/checks/windows_job_v2_cpu_v1'):
        manifest = read(ROOT / old_run / 'artifact_sha256.json')
        for name, expected in manifest.items():
            with (ROOT / old_run / name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError(f'Old trial artifact changed: {name}')
    for name, expected in read(ROOT / 'reports/windows_job_v2_before_run.json')['old_cpu_failure_sha256'].items():
        with Path(name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Old CPU failure artifact changed: {name}')
    return len(protected)


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('One acceptance claim already exists; no rerun')
    RUN.mkdir(parents=True, exist_ok=False)
    snapshot = RUN / 'source_snapshot'
    snapshot.mkdir()
    for name in SOURCES:
        shutil.copyfile(ROOT / name, snapshot / Path(name).name)
    report = {'status': 'running', 'acceptance_attempts': 1, 'automatic_retries': 0,
              'cuda_initialized': False, 'model_loaded': False, 'training_updates': 0, 'final_examples_read': False,
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}}
    write(RUN / 'claim.json', report)
    job, handles = None, []
    started = None
    evidence, identities = {}, {}
    code = 1
    try:
        # Prelaunch facts are persisted before checking them.
        prelaunch = {'interpreter_files': interpreter_evidence(), 'controller': environment_identity('controller', 'not-a-child'),
                     'allowed_base_executable': [str(BASE_ALIAS), str(NATIVE_PYTHON)],
                     'base_path_contract': 'CPython 3.11.9 PC/python_uwp.cpp: package alias when packaged, module filename fallback otherwise; actual process image must remain native',
                     'pyvenv_cfg': (ROOT / '.venv/pyvenv.cfg').read_text(encoding='utf-8')}
        write(RUN / 'prelaunch_evidence.json', prelaunch)
        report['old_frozen_files_verified_before'] = verify_frozen()
        nonce = secrets.token_hex(24)
        started = time.perf_counter()
        child_env = dict(os.environ, JOB_CPU_NONCE=nonce, JOB_CPU_DEADLINE=str(started + 8.0), CUDA_VISIBLE_DEVICES='')
        job = JobProcess([str(VENV_PYTHON), '-X', 'utf8', str(Path(__file__).resolve()), '--child', 'launcher', str(RUN)],
                         ROOT, child_env, RUN / 'console.log')
        event(RUN, 'launcher_created', launcher_pid=job.pid, start_order=job.start_order)
        while time.perf_counter() - started < 4.0:
            available = [role for role in ('launcher', 'worker', 'grandchild') if (RUN / f'{role}.json').exists()]
            evidence = {'elapsed_seconds': time.perf_counter() - started, 'native_launcher_pid': job.pid,
                        'launch_route': job.launch_route, 'job_settings': job.settings_evidence(),
                        'job_active_pids': job.active_pids(), 'role_evidence': {}}
            for role in available:
                reported = read(RUN / f'{role}.json')
                evidence['role_evidence'][role] = {'reported': reported, 'observed': job.process_evidence(reported['pid'])}
            write(RUN / 'before_assertions.json', evidence)
            if len(available) == 3 and all((RUN / f'{role}_ready.json').exists() for role in available):
                break
            if job.poll() is not None:
                break
            time.sleep(0.03)
        # All identity, path, membership and Job-flag evidence now survives a failed assertion.
        # Keep actual role handles even when their membership is about to fail.
        for entry in evidence['role_evidence'].values():
            handle = ProcessHandle(entry['reported']['pid'])
            if handle.identity()['creation_time_100ns'] != entry['reported']['creation_time_100ns']:
                handle.close()
                raise ValueError('Reported role PID was reused')
            handles.append(handle)
        assert len(evidence['role_evidence']) == 3, 'Missing role; inspect pre-assertion evidence'
        assert all((RUN / f'{role}_ready.json').exists() for role in evidence['role_evidence'])
        settings = evidence['job_settings']
        assert settings['api_success'] and settings['kill_on_close']
        assert settings['limit_flags'] == 0x2000 and not settings['breakaway_ok'] and not settings['silent_breakaway_ok']
        assert job.start_order == ['created_suspended', 'assigned_to_kill_on_close_job', 'resumed']
        expected_packages = read(ROOT / 'reports/sft_resume_pilot_preflight.json')['packages']
        for role, entry in evidence['role_evidence'].items():
            reported, observed = entry['reported'], entry['observed']
            assert observed.get('is_in_job_api_success') and observed.get('is_in_this_job'), f'{role} outside Job'
            assert reported['nonce'] == nonce and observed['identity']['pid'] == reported['pid']
            assert observed['identity']['parent_pid'] == reported['parent_pid']
            assert observed['identity']['creation_time_100ns'] == reported['creation_time_100ns']
            assert normalize(reported['executable']) == normalize(NATIVE_PYTHON)
            assert normalize(reported['sys_executable']) == normalize(VENV_PYTHON)
            assert normalize(reported['sys_base_executable']) in {normalize(BASE_ALIAS), normalize(NATIVE_PYTHON)}
            assert normalize(reported['sys_prefix']) == normalize(ROOT / '.venv')
            assert normalize(reported['sys_base_prefix']) != normalize(ROOT / '.venv')
            assert normalize(ROOT / '.venv/Lib/site-packages') in [normalize(p) for p in reported['site_packages']]
            assert all(normalize(p) == normalize(ROOT / '.venv/Lib/site-packages') for p in reported['package_locations'].values())
            assert reported['package_versions'] == expected_packages and reported['python_version'] == [3, 11, 9]
            assert reported['launcher_variable_consumed'] and not reported['forbidden_imports']
            identities[role] = reported
        assert identities['launcher']['pid'] == job.pid, 'Native process still redirected'
        assert identities['worker']['parent_pid'] == identities['launcher']['pid']
        assert identities['grandchild']['parent_pid'] == identities['worker']['pid']
        assert set(evidence['job_active_pids']) == {row['pid'] for row in identities.values()}
        # Exercise the already-written identity verifier with the actual worker, not launcher.
        verified_handle, actual, chain = verify_identity(job, identities['worker'], nonce)
        verified_handle.close()  # The independently retained worker handle remains open.
        assert actual['pid'] == identities['worker']['pid'] != job.pid
        write(RUN / 'binding_verification.json', {'identity_verified': True, 'bound_pid': actual['pid'],
              'launcher_pid': job.pid, 'ancestry': chain, 'gpu_sampling_performed': False})
        event(RUN, 'identity_verified', worker_pid=actual['pid'])
        before_kill = {'job_active_pids': job.active_pids(), 'roles_alive': [h.alive() for h in handles],
                       'elapsed_seconds': time.perf_counter() - started}
        write(RUN / 'before_termination.json', before_kill)
        assert all(before_kill['roles_alive']) and before_kill['elapsed_seconds'] < 5.0
        termination = terminate_and_wait(job, handles, lambda name, **values: event(RUN, name, **values),
                                         exit_code=124, timeout=2.0)
        exits = termination['roles']
        after = {'job_active_pids': job.active_pids(), 'retained_handles': exits,
                 'elapsed_seconds': time.perf_counter() - started, 'termination': termination}
        write(RUN / 'after_termination.json', after)
        assert after['job_active_pids'] == [] and all(x['exit_code'] == 124 and not x['alive'] for x in exits)
        assert after['elapsed_seconds'] < 8.0, 'Cannot attribute exit to job termination after self-expiry'
        assert termination['confirmation_seconds'] <= 2.0
        assert all(0 <= row['lifetime_seconds'] <= 10.0 for row in exits)
        for role, identity in identities.items():
            role_events = RUN / f"events_{identity['pid']}.jsonl"
            assert role_events.is_file(), f'Missing durable role log: {role}'
            assert any(json.loads(line)['event'] == 'identity_published' for line in role_events.read_text(encoding='utf-8').splitlines())
        for path in RUN.glob('events_*.jsonl'):
            raw = path.read_text(encoding='utf-8')
            assert raw.endswith('\n')
            for line in raw.splitlines():
                json.loads(line)
        assert 'torch' not in sys.modules
        report.update(status='pass', identities=identities, job_settings=settings,
                      active_pids_before=evidence['job_active_pids'], active_pids_after=after['job_active_pids'],
                      retained_handle_exits=exits, elapsed_seconds=after['elapsed_seconds'],
                      termination=termination, strict_10_second_lifetime_proven=True,
                      checks=['native_python_with_original_venv', 'frozen_package_versions_and_locations',
                          'evidence_persisted_before_assertions', 'three_roles_in_same_job', 'parent_chain_and_creation_identity',
                          'real_worker_binding', 'job_flags_read_back', 'whole_tree_exit_124_before_self_deadline',
                          'shared_2_second_exit_confirmation', 'os_process_lifetimes_below_10_seconds', 'durable_jsonl_complete'])
        code = 0
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        cleanup_errors = []
        if job:
            try:
                if job.active_pids():
                    event(RUN, 'cleanup_termination_requested', exit_code=125, acceptance_retry=False)
                    win._check(win._TerminateJob(job.job, 125), 'Cleanup TerminateJobObject')
            except Exception as error:
                cleanup_errors.append(str(error))
            try:
                close_without_new_wait(job)
            except Exception as error:
                cleanup_errors.append(str(error))
        # If membership failed, Job cleanup alone cannot cover escaped roles.
        # Wait for their independent 8s deadlines before finalizing logs/hashes.
        if report['status'] != 'pass' and started is not None:
            cleanup_deadline = started + 9.5
            known = {handle.pid for handle in handles}
            while time.perf_counter() < cleanup_deadline:
                for role in ('launcher', 'worker', 'grandchild'):
                    path = RUN / f'{role}.json'
                    if path.exists():
                        row = read(path)
                        if row['pid'] not in known:
                            known.add(row['pid'])
                            try:
                                handle = ProcessHandle(row['pid'])
                                if handle.identity()['creation_time_100ns'] != row['creation_time_100ns']:
                                    handle.close()
                                    raise ValueError('Role PID reused during cleanup')
                                handles.append(handle)
                            except ProcessLookupError:
                                pass
                            except OSError as error:
                                if error.winerror != 87:
                                    cleanup_errors.append(str(error))
                            except Exception as error:
                                cleanup_errors.append(str(error))
                if time.perf_counter() >= started + 8.2 and all(not h.alive() for h in handles):
                    break
                time.sleep(0.03)
        role_cleanup = []
        for handle in handles:
            alive = handle.alive()
            cleanup_row = {'pid': handle.pid, 'alive': alive, 'observed_monotonic': time.perf_counter(),
                           'exit_code': None if alive else handle.wait(0)}
            if not alive:
                try:
                    cleanup_row.update(exit_timing(handle))
                except Exception as error:
                    cleanup_errors.append(str(error))
            role_cleanup.append(cleanup_row)
            if alive:
                cleanup_errors.append(f'Known role {handle.pid} still alive')
            handle.close()
        write(RUN / 'cleanup_observations.json', role_cleanup)
        report['role_cleanup'] = role_cleanup
        report['cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            report['status'], code = 'failed', 1
        try:
            report['old_frozen_files_verified_after'] = verify_frozen()
        except Exception as error:
            report.update(status='failed', preservation_error=str(error)); code = 1
        write(RUN / 'report.json', report)
        write(REPORT, report)
        write(RUN / 'artifact_sha256.json', {p.relative_to(RUN).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in RUN.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'})
    print(json.dumps({'status': report['status'], 'error': report.get('error'), 'checks': report.get('checks'),
                      'elapsed_seconds': report.get('elapsed_seconds'), 'cuda_initialized': False}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    if '--child' in sys.argv:
        index = sys.argv.index('--child')
        raise SystemExit(child(sys.argv[index + 1], sys.argv[index + 2]))
    raise SystemExit(main())
