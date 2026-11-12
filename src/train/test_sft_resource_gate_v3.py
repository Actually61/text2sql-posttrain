"""Single CPU-only check of actual readiness code using fake time and samples."""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/checks/sft_resource_gate_v3'
REPORT = ROOT / 'reports/sft_resource_gate_v3_cpu_checks.json'
SOURCE = ROOT / 'src/train/sft_round1_resource_v3.py'


def main():
    if OUT.exists() or REPORT.exists():
        raise FileExistsError('Single-use CPU resource-gate check already exists')
    OUT.mkdir(parents=True)
    began = time.perf_counter()
    report = {'status': 'running', 'cuda_initialized': False, 'training_updates': 0,
              'real_gpu_queries': 0, 'cases': []}
    def write(path, value, exclusive=False):
        with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    code = 1
    try:
        source = SOURCE.read_text(encoding='utf-8')
        tree = ast.parse(source)
        gate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'stable_resource_gate')
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        gates = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'stable_resource_gate']
        claims = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and n.func.attr == 'mkdir']
        assert len(gates) == 1 and claims and gates[0] < min(claims)
        compiled = compile(ast.fix_missing_locations(ast.Module(body=[gate], type_ignores=[])), str(SOURCE), 'exec')
        config = json.loads((ROOT / 'configs/sft_round1_resource_v3.json').read_text(encoding='utf-8'))
        assert config['minimum_free_mib'] == 6656 and config['start_headroom_mib'] == 128
        cases = [('boundary_ready', [6784] * 6, 'ready', 6),
                 ('below_headroom', [6783], 'not_ready', 1),
                 ('late_dip', [6914, 6914, 6700], 'not_ready', 3),
                 ('low_disk', [6914], 'not_ready', 1),
                 ('query_timeout', [], 'not_ready', 1),
                 ('nonfinite_sample', ['nan'], 'not_ready', 1)]
        for name, values, expected, expected_calls in cases:
            state = {'now': 100.0, 'calls': 0, 'timeouts': []}
            def sleep(seconds):
                assert seconds >= 0
                state['now'] += seconds
            def query(command, **kwargs):
                state['calls'] += 1
                state['timeouts'].append(kwargs['timeout'])
                assert 0 < kwargs['timeout'] <= 5
                assert command[0] == 'nvidia-smi'
                if name == 'query_timeout':
                    state['now'] += kwargs['timeout']
                    raise subprocess.TimeoutExpired(command, kwargs['timeout'])
                state['now'] += 0.1
                return SimpleNamespace(stdout=f'GPU, 8188, 1000, {values[state["calls"] - 1]}, driver')
            local_config = dict(config, resource_readiness_report=f'{name}.json')
            scope = {'ROOT': OUT, 'write': write, 'math': math, 'sys': sys,
                     'traceback': traceback, 'time': SimpleNamespace(perf_counter=lambda: state['now'], sleep=sleep),
                     'subprocess': SimpleNamespace(run=query),
                     'shutil': SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=(7 if name == 'low_disk' else 20) * 2**30))}
            exec(compiled, scope)
            observed = scope['stable_resource_gate'](local_config)
            assert observed['status'] == expected and state['calls'] == expected_calls, observed
            assert observed['automatic_retries'] == 0 and observed['run_directory_created'] is False
            assert observed['seconds'] <= 20
            samples = observed['samples']
            assert all(b['sample_started_monotonic'] - a['sample_completed_monotonic'] >= 2
                       for a, b in zip(samples, samples[1:]))
            if expected == 'ready':
                assert observed['minimum_observed_free_mib'] == 6784
                assert observed['observed_sample_span_seconds'] >= 10
            report['cases'].append({'name': name, 'status': 'pass', 'observed': expected,
                                    'query_calls': state['calls'], 'virtual_seconds': observed['seconds']})
        assert not (ROOT / config['run_dir']).exists()
        assert not any(name in sys.modules for name in ('torch', 'transformers', 'peft', 'bitsandbytes'))
        assert time.perf_counter() - began < 10
        report.update(status='pass', run_claim_follows_stable_gate=True,
                      source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        code = 0
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    report['seconds'] = time.perf_counter() - began
    write(REPORT, report, exclusive=True)
    print(json.dumps({'status': report['status'], 'cases': len(report['cases']), 'seconds': report['seconds']}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
