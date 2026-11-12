"""One bounded CPU regression for the formal supervisor's stale-now race.

The real decision statements are extracted from the new source and exercised
with controlled read/clock ordering. Four native workers then validate the
actual Job, mutex, identity and stop wiring, with simulated memory samples.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
RUN = ROOT / 'outputs/checks/sft_clock_v2'
REPORT = ROOT / 'reports/sft_clock_v2_cpu_checks.json'
OLD_RUN = ROOT / 'outputs/runs/sft_round1_ready_v1'
FORBIDDEN = ('torch', 'transformers', 'peft', 'bitsandbytes')
MODES = ('normal', 'future', 'soft', 'hard')
SOURCES = ['src/train/test_sft_clock_v2.py', 'src/train/pid_supervision_formal_v2.py',
    'src/train/cache_release_watchdog.py', 'src/train/pid_supervision.py',
    'src/train/windows_job_process_v2.py', 'src/train/windows_job_process.py',
    'src/train/windows_job_exit_wait_formal.py', 'src/train/stage_file_io_v2.py',
    'src/train/windows_stage_mutex.py', 'src/train/windows_json_ipc.py',
    'src/train/stage_file_io.py']
LIMITS = {'budget_profile': 'cpu', 'total_soft_seconds': 3.0, 'total_hard_seconds': 4.0,
    'micro_soft_seconds': 2.5, 'micro_hard_seconds': 3.5,
    'handshake_seconds': 2.5, 'cleanup_seconds': 2.0, 'expected_process_count': 1,
    'child_self_seconds': 8.0, 'case_seconds': 9.5,
    'sample_interval_seconds': 0.05, 'resource_grace_seconds': 1.0,
    'telemetry_stale_seconds': 1.0}
EXPECTED = {
    'normal': ('pass', 'completed', 'normal_completion', 0, False),
    'future': ('stopped', 'soft_stopped', 'Invalid microbatch clock', 20, False),
    'soft': ('stopped', 'soft_stopped', 'Total soft time limit', 20, False),
    'hard': ('stopped', 'hard_timeout', 'Total hard time limit', 124, True),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    from src.train.windows_json_ipc import write_json_atomic
    write_json_atomic(path, value)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(mapping):
    for name, expected in mapping.items():
        assert digest(ROOT / name) == expected, f'Preserved file changed: {name}'


def preserved_files():
    preflight = read(ROOT / 'reports/sft_round1_ready_preflight.json')
    result = read(ROOT / 'reports/sft_round1_ready_result.json')
    ledger = read(OLD_RUN / 'progress.json')
    assert result['status'] == 'failed' and result['automatic_retries'] == 0
    assert ledger['attempted_updates'] == ledger['completed_updates'] == 142
    assert ledger['total_budget'] == 796 and ledger['failed'] and not ledger['partial_update_possible']
    protected = dict(preflight['protected_sha256'])
    protected.update(preflight['source_sha256'])
    manifest = read(OLD_RUN / 'artifact_sha256.json')
    protected.update({str(OLD_RUN / name): sha for name, sha in manifest.items()})
    for path in [OLD_RUN / 'artifact_sha256.json', ROOT / 'reports/sft_round1_ready_preflight.json',
                 ROOT / 'reports/sft_round1_ready_result.json', ROOT / 'reports/sft_round1_ready_analysis.json',
                 ROOT / 'reports/sft_round1_ready_walkthrough.md']:
        protected[str(path)] = digest(path)
    verify(protected)
    return protected, {'formal_failed_updates': 142, 'formal_failed_ledger_sha256': digest(OLD_RUN / 'progress.json'),
        'formal_run_artifacts_verified': len(manifest), 'protection_entries_verified': len(protected),
        'no_ledger_rewind': True}


def deterministic_policy_checks():
    from src.train.cache_release_watchdog import BudgetWatch
    source_path = ROOT / 'src/train/pid_supervision_formal_v2.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    target = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'supervise')
    assigned = {}
    for node in ast.walk(target):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('worker_stop', 'decision_now', 'decision'):
                assigned.setdefault(node.targets[0].id, []).append(node)
    assert set(assigned) == {'worker_stop', 'decision_now', 'decision'}
    assert all(len(nodes) == 1 for nodes in assigned.values())
    nodes = [assigned[key][0] for key in ('worker_stop', 'decision_now', 'decision')]
    assert [node.lineno for node in nodes] == sorted(node.lineno for node in nodes)
    assert ast.unparse(nodes[0].value) == "read_optional(directory / 'stop_request.json')"
    assert ast.unparse(nodes[1].value) == 'time.perf_counter()'
    assert ast.unparse(nodes[2].value) == 'watch.evaluate(decision_now, policy_stage, fresh, worker_stop)'
    stage_lines = [node.lineno for node in ast.walk(target) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == 'read_stage']
    drain_lines = [node.lineno for node in ast.walk(target) if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'fresh' for t in node.targets)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                   and node.value.func.id == 'drain']
    assert stage_lines and drain_lines and max(stage_lines) < nodes[0].lineno and max(drain_lines) < nodes[0].lineno
    decision_code = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source_path), 'exec')
    old_source = (ROOT / 'src/train/pid_supervision_formal.py').read_text(encoding='utf-8')
    old_tree = ast.parse(old_source)
    old_decisions = [node for node in ast.walk(old_tree) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'decision' for t in node.targets)]
    assert len(old_decisions) == 1
    old_code = compile(ast.fix_missing_locations(ast.Module(body=old_decisions, type_ignores=[])), '<frozen-old-decision>', 'exec')
    calls = []
    state = {'now': 100.0}
    def read_stop(path):
        calls.append('read_stop')
        state['now'] = 100.1  # A blocking read finished after the new stage became visible.
        return None
    def clock():
        calls.append('decision_clock')
        return state['now']
    common = {'directory': Path('cpu-only-fixture'), 'now': 100.0,
        'policy_stage': {'micro_started_monotonic': 100.000348}, 'fresh': [],
        'read_optional': read_stop, 'time': SimpleNamespace(perf_counter=clock)}
    old_scope = dict(common, watch=BudgetWatch(99.0, {}))
    exec(old_code, old_scope)
    assert old_scope['decision']['request']['reason'] == 'Invalid microbatch clock'
    calls.clear(); state['now'] = 100.0
    new_scope = dict(common, watch=BudgetWatch(99.0, {}))
    exec(decision_code, new_scope)
    assert calls == ['read_stop', 'decision_clock']
    assert new_scope['decision']['request'] is None and new_scope['decision']['hard_reason'] is None
    cases = [{'name': 'real_source_stale_now_race', 'status': 'pass', 'old_now': 100.0,
        'new_stage_start': 100.000348, 'new_decision_now': new_scope['decision_now'],
        'old_logic_reproduces_failure': True, 'new_logic_accepts_legal_stage': True,
        'actual_source_statements_executed': True, 'read_and_clock_order': list(calls)}]
    equal_scope = dict(common, policy_stage={'micro_started_monotonic': 100.1}, watch=BudgetWatch(99.0, {}))
    exec(decision_code, equal_scope)
    assert equal_scope['decision']['request'] is None
    cases.append({'name': 'start_equal_to_decision_now_is_legal', 'status': 'pass'})
    for name, stamp in [('future', 100.2), ('before_supervisor_start', 98.9)]:
        scope = dict(common, policy_stage={'micro_started_monotonic': stamp}, watch=BudgetWatch(99.0, {}))
        exec(decision_code, scope)
        assert scope['decision']['request']['reason'] == 'Invalid microbatch clock'
        cases.append({'name': name, 'status': 'pass', 'micro_start': stamp, 'decision_now': scope['decision_now']})
    watch = BudgetWatch(0, {})
    assert watch.evaluate(59.999, {'micro_started_monotonic': 0})['request'] is None
    assert watch.evaluate(60, {'micro_started_monotonic': 0})['request']['reason'] == 'Microbatch soft time limit'
    assert watch.evaluate(89.999, {})['hard_reason'] is None
    assert watch.evaluate(90, {})['hard_reason'] == 'Microbatch hard time limit'
    cases.append({'name': 'micro_soft60_hard90_and_retained_deadline', 'status': 'pass'})
    watch = BudgetWatch(0, {})
    assert watch.evaluate(239.999, {})['request'] is None
    assert watch.evaluate(240, {})['request']['reason'] == 'Total soft time limit'
    assert watch.evaluate(299.999, {})['hard_reason'] is None
    assert watch.evaluate(300, {})['hard_reason'] == 'Total hard time limit'
    cases.append({'name': 'total_soft240_hard300', 'status': 'pass'})
    watch = BudgetWatch(0, {})
    request = watch.evaluate(1, {'micro_started_monotonic': 2})['request']
    assert request['kind'] == 'resource'
    assert watch.evaluate(5.999, {})['hard_reason'] is None
    assert watch.evaluate(6.0, {})['hard_reason'] == 'Resource stop grace expired'
    cases.append({'name': 'invalid_clock_resource_grace5', 'status': 'pass'})
    watch = BudgetWatch(0, {})
    assert watch.evaluate(1, {'telemetry_required': True})['request'] is None
    assert watch.evaluate(3, {'telemetry_required': True})['request']['reason'] == 'PID telemetry stalled'
    cases.append({'name': 'required_telemetry_stale2_still_stops', 'status': 'pass'})
    watch = BudgetWatch(0, {})
    for n in range(11):
        decision = watch.evaluate(n / 10, {}, [{'sampled_monotonic': n / 10, 'status': 'ok',
            'shared_bytes': 513 * 2**20, 'dedicated_bytes': 0}])
    assert decision['request']['reason'] == 'PID shared memory exceeded the sustained limit'
    cases.append({'name': 'shared_above512_for1_second_still_stops', 'status': 'pass'})
    return {'status': 'pass', 'cases': cases, 'case_count': len(cases),
        'decision_source_lines': [node.lineno for node in nodes], 'production_watchdog_unchanged': True}


def fixture(mode):
    from src.train.pid_supervision import publish_worker_identity, wait_for_identity_ack
    from src.train.stage_file_io_v2 import write_stage
    from src.train.windows_json_ipc import read_json
    directory, nonce = Path(os.environ['PID_RUN_DIR']), os.environ['PID_RUN_NONCE']
    deadline = float(os.environ['PID_CHILD_DEADLINE'])
    def expire():
        time.sleep(max(0, deadline - time.perf_counter()))
        os._exit(92)
    threading.Thread(target=expire, daemon=True).start()
    def wait_file(name):
        while time.perf_counter() < deadline:
            value = read_json(directory / name, missing_ok=True)
            if value is not None:
                assert value['nonce'] == nonce
                return value
            time.sleep(0.01)
        raise TimeoutError(f'Fixture acknowledgement deadline: {name}')
    def stage(name, micro=None, complete=False):
        write_stage(directory, {'stage': name, 'telemetry_required': True,
            'micro_started_monotonic': micro, 'telemetry_complete': complete})
    def finish(stopped):
        stage('cpu_complete', complete=True)
        closed = wait_file('telemetry_closed.json')
        assert closed['worker_pid'] == os.getpid() and closed['valid_sample_count'] >= 3
        assert not any(name in sys.modules for name in FORBIDDEN)
        write(directory / 'worker_result.json', {'status': 'stopped' if stopped else 'pass',
            'mode': mode, 'pid': os.getpid(), 'ml_imported': False, 'cuda_initialized': False,
            'cooperative_exit_policy_is_cpu_fixture_only': True})
        return 20 if stopped else 0
    try:
        write_stage(directory, {'stage': 'cpu_before_identity', 'telemetry_required': False,
            'micro_started_monotonic': None})
        membership = wait_file('membership_ack.json')
        assert membership['retained_pids'] == [os.getpid()]
        identity = publish_worker_identity(directory, nonce)
        write(directory / 'fixture_worker.json', {**identity, 'mode': mode, 'sys_executable': sys.executable,
            'sys_prefix': sys.prefix, 'forbidden_imports': [name for name in FORBIDDEN if name in sys.modules]})
        ack = wait_for_identity_ack(directory, nonce, timeout=min(2.5, deadline - time.perf_counter()))
        assert ack['launcher_pid'] == ack['worker_pid'] == os.getpid()
        stage('cpu_registered')
        ready = wait_file('telemetry_ready.json')
        assert ready['worker_pid'] == os.getpid() and ready['valid_sample_count'] >= 3
        if mode == 'normal':
            for index in range(4):
                assert not (directory / 'stop_request.json').exists()
                stage(f'cpu_legal_new_stage_{index}', time.perf_counter())
                time.sleep(0.06)
            return finish(False)
        if mode == 'future':
            stage('cpu_illegal_future', time.perf_counter() + 30.0)
        else:
            stage(f'cpu_wait_{mode}')
        while time.perf_counter() < deadline:
            stop = read_json(directory / 'stop_request.json', missing_ok=True)
            if stop and mode != 'hard':
                expected = 'Invalid microbatch clock' if mode == 'future' else 'Total soft time limit'
                assert stop['reason'] == expected, stop
                return finish(True)
            time.sleep(0.01)
        raise TimeoutError('Fixture reached independent self-expiry')
    except Exception as error:
        write(directory / 'fixture_failure.json', {'status': 'failed', 'mode': mode, 'pid': os.getpid(),
            'error': repr(error), 'traceback': traceback.format_exc()})
        return 2


def run_case(mode, deadline):
    from src.train.pid_supervision_formal_v2 import supervise
    from src.train.windows_job_process_v2 import VENV_PYTHON, NATIVE_PYTHON, normalize
    directory = RUN / mode
    sampled = []
    def provider(pid, stage):
        assert read(directory / 'worker_identity.json')['pid'] == pid
        sampled.append(pid)
        return {'pid': pid, 'status': 'ok', 'dedicated_bytes': 0, 'shared_bytes': 0,
                'fixture': 'cpu_simulated_no_gpu'}
    result = supervise([str(VENV_PYTHON), '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--fixture', mode],
        ROOT, dict(os.environ), directory, dict(LIMITS), sample_provider=provider, suite_deadline=deadline)
    write(directory / 'returned_result.json', result)
    expected_status, outcome, reason, code, killed = EXPECTED[mode]
    assert (result['status'], result['outcome'], result['stop_reason'], result['exit_code'], result['killed']) == (
        expected_status, outcome, reason, code, killed), result
    assert result['identity_verified'] and not result['errors'] and not result['cleanup_errors']
    assert result['automatic_retries'] == 0 and result['worker_launches'] == 1
    worker = read(directory / 'fixture_worker.json')
    pid = worker['pid']
    assert result['bound_pid'] == result['worker_pid'] == result['launcher_pid'] == pid
    assert normalize(worker['executable']) == normalize(NATIVE_PYTHON)
    assert normalize(worker['sys_executable']) == normalize(VENV_PYTHON)
    assert normalize(worker['sys_prefix']) == normalize(ROOT / '.venv') and not worker['forbidden_imports']
    assert result['job_active_pids_after_cleanup'] == [] and len(result['role_exits']) == 1
    assert all(not row['alive'] and row['exit_code'] == code and 0 <= row['lifetime_seconds'] <= 9.5
               for row in result['role_exits'])
    assert sampled and set(sampled) == {pid} and result['observed_sample_pids'] == [pid]
    assert result['valid_sample_count'] >= 3 and result['sample_source'] == 'injected_cpu_fixture'
    if mode == 'hard':
        assert not (directory / 'worker_result.json').exists()
        assert result['termination']['confirmation_seconds'] <= 2.0
    else:
        assert result['telemetry_closed'] and read(directory / 'worker_result.json')['ml_imported'] is False
    log_lines = 0
    for path in directory.glob('*.jsonl'):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                assert line.endswith('\n'), f'Incomplete log: {path}'
                json.loads(line); log_lines += 1
    assert time.perf_counter() < deadline
    accepted = {'mode': mode, 'status': 'pass', 'worker_pid': pid, 'stop_reason': reason,
        'exit_code': code, 'role_exits': result['role_exits'], 'wall_seconds': result['wall_seconds'],
        'complete_jsonl_rows': log_lines, 'supervisor_result': result,
        'sample_values_are_simulated': True, 'cpu_fixture_cooperative_exit_only': True}
    write(directory / 'case_acceptance.json', accepted)
    return accepted


def main():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError('CPU regression already claimed; no replay')
    RUN.mkdir(parents=True, exist_ok=False)
    started, done = time.perf_counter(), threading.Event()
    deadline = started + 60.0
    report = {'status': 'running', 'acceptance_rounds': 1, 'automatic_retries': 0, 'cpu_budget_seconds': 60,
        'case_limits': LIMITS, 'cases': [], 'current_case': None, 'cuda_initialized': False,
        'model_loaded': False, 'training_updates': 0, 'final_examples_read': False,
        'validation_examples_read': False, 'sample_values_are_simulated': True}
    write(RUN / 'claim.json', report)
    def expire():
        if not done.wait(max(0, deadline - time.perf_counter())):
            write(RUN / 'hard_timeout.json', {'status': 'failed', 'seconds': time.perf_counter() - started})
            os._exit(93)
    threading.Thread(target=expire, daemon=True).start()
    protected = {}
    code = 1
    try:
        protected, report['preservation_before'] = preserved_files()
        report['protected_sha256'] = protected
        report['source_sha256'] = {name: digest(ROOT / name) for name in SOURCES}
        snapshot = RUN / 'source_snapshot'
        snapshot.mkdir()
        for name in SOURCES:
            shutil.copyfile(ROOT / name, snapshot / Path(name).name)
        report['checks'] = {'deterministic_policy': deterministic_policy_checks()}
        write(RUN / 'before_native_cases.json', report)
        for mode in MODES:
            if deadline - time.perf_counter() < 12.0:
                raise TimeoutError('Insufficient 12-second reserve for next CPU case and summary')
            report['current_case'] = mode
            write(RUN / f'{mode}_started.json', {'mode': mode, 'monotonic': time.perf_counter()})
            report['cases'].append(run_case(mode, deadline))
            write(RUN / 'progress.json', report)
        verify(report['source_sha256'])
        assert not any(name in sys.modules for name in FORBIDDEN)
        assert len(report['cases']) == 4
        report.update(status='pass', current_case=None)
        code = 0
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    finally:
        try:
            verify(protected)
            report['preservation_after'] = {'status': 'pass', 'verified': len(protected),
                'ledger': read(OLD_RUN / 'progress.json')}
        except Exception as error:
            report.update(status='failed', preservation_error=repr(error)); code = 1
        report['seconds'] = time.perf_counter() - started
        report['ml_imported'] = any(name in sys.modules for name in FORBIDDEN)
        report['completed_cases'] = len(report['cases'])
        report['not_started_cases'] = [mode for mode in MODES if not (RUN / f'{mode}_started.json').exists()]
        if report['seconds'] >= 60 or report['ml_imported']:
            report.update(status='failed', budget_or_import_error=True); code = 1
        write(RUN / 'report.json', report)
        write(REPORT, report)
        write(RUN / 'artifact_sha256.json', {p.relative_to(RUN).as_posix(): digest(p)
            for p in RUN.rglob('*') if p.is_file() and p.name != 'artifact_sha256.json'})
        done.set()
    print(json.dumps({key: report.get(key) for key in ('status', 'seconds', 'completed_cases',
        'current_case', 'error', 'ml_imported')}, ensure_ascii=False), flush=True)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', choices=MODES)
    args = parser.parse_args()
    raise SystemExit(fixture(args.fixture) if args.fixture else main())
