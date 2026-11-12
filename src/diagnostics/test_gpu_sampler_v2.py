"""Six one-shot CPU-only fake-helper cases; run parent under a bounded Windows Job.

The fake child arms an eight-second self-exit. No nvidia-smi, CUDA or model
libraries are used. This file is a separate acceptance run, not inference.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def fake_child(case):
    threading.Timer(8.0, lambda: os._exit(98)).start()
    if case == 'nonzero':
        print('injected helper failure', file=sys.stderr, flush=True)
        os._exit(7)
    if case == 'invalid':
        print('nan, 8188', flush=True)
        time.sleep(7.5)
        return
    if case == 'stall':
        print('100, 8188', flush=True)
        time.sleep(7.5)
        return
    deadline = time.perf_counter()+7.5
    while time.perf_counter() < deadline:
        print('100, 8188', flush=True)
        time.sleep(0.05)


def run(output):
    from src.infer.gpu_sampler_v2 import GPUSampler
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('CPU evidence must stay in the project')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); deadline = started+60.0
    results = []
    write(output/'claim.json', {'maximum_seconds':60, 'maximum_child_seconds':8,
        'cases':['normal','nonzero','invalid','stall','close_while_active','double_close'],
        'real_gpu_commands':False, 'automatic_retries':0})
    for case in ('normal','nonzero','invalid','stall','close_while_active','double_close'):
        if time.perf_counter() > deadline-10.0:
            write(output/'summary.json', {'status':'failed','cases':results,
                'seconds':time.perf_counter()-started,'automatic_retries':0,
                'error':'Insufficient remaining CPU acceptance budget'})
            return 1
        case_started = time.perf_counter(); sampler = None
        close_attempted = False
        item = {'case':case, 'status':'failed'}
        try:
            def command_factory(interval_ms):
                command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--fake-child', case]
                if os.name == 'nt':
                    from src.train.windows_job_process_v2 import native_venv_launch
                    return native_venv_launch(command, dict(os.environ, CUDA_VISIBLE_DEVICES=''))
                return command, dict(os.environ, CUDA_VISIBLE_DEVICES='')
            sampler = GPUSampler(0.05, command_factory=command_factory, first_timeout=5.0,
                stale_timeout=0.3 if case=='stall' else 5.0, cleanup_timeout=2.0, expected_total_mib=8188,
                event_path=output/f'{case}.events.jsonl')
            sampler.start()
            if case in ('normal','double_close'):
                limit = time.perf_counter()+1.0
                while len(sampler.samples)<3 and not sampler.errors and time.perf_counter()<limit:
                    time.sleep(0.01)
                assert len(sampler.samples)>=3 and not sampler.errors
            elif case == 'close_while_active':
                assert sampler.samples and not sampler.errors
            elif case == 'stall':
                assert sampler.samples
                limit = time.perf_counter()+1.0
                while not sampler.errors and time.perf_counter()<limit:
                    time.sleep(0.01)
                assert sampler.errors and sampler.report()['fatal']['kind']=='stale_sample_timeout'
            else:
                assert sampler.errors
                if case=='invalid': assert sampler.report()['fatal']['kind']=='invalid_sample'
            before_close = len(sampler.samples)
            close_attempted = True
            sampler.close()
            report = sampler.report()
            assert report['helper_launches']==1 and report['automatic_retries']==0
            assert report['close']['status']=='pass' and report['close']['alive_threads']==[]
            assert report['helper_returncode'] is not None
            event_rows=[json.loads(line) for line in (output/f'{case}.events.jsonl').read_text(encoding='utf-8').splitlines()]
            assert event_rows and any(row['event']=='spawn_returned' for row in event_rows)
            assert any(row['event']=='cleanup_resources_finished' for row in event_rows)
            if case in ('nonzero','invalid','stall'):
                assert any(row['event']=='fatal' for row in event_rows)
            count = len(sampler.samples)
            time.sleep(0.03)
            assert len(sampler.samples)==count, 'Sampling continued after close'
            if case=='double_close':
                old_event_count=len(sampler.events)
                sampler.close()
                assert len(sampler.events)==old_event_count and len(sampler.samples)==count
            if case=='nonzero': assert report['helper_returncode']==7
            assert not any(name in sys.modules for name in ('torch','transformers','peft','bitsandbytes'))
            item.update(status='pass', samples_before_close=before_close, samples_after_close=count,
                sampler=report, child_self_exit_seconds=8)
        except BaseException as error:
            item.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
            if sampler is not None and not close_attempted:
                try:
                    close_attempted = True
                    sampler.close()
                except BaseException as cleanup:
                    item['cleanup_error']=f'{type(cleanup).__name__}: {cleanup}'
            if sampler is not None:
                item['sampler']=sampler.report()
        item['seconds']=time.perf_counter()-case_started
        results.append(item); write(output/f'{case}.json',item)
        if item['status']!='pass' or item['seconds']>10:
            write(output/'summary.json', {'status':'failed', 'cases':results,
                'seconds':time.perf_counter()-started, 'automatic_retries':0})
            return 1
    summary={'status':'pass','cases':results,'seconds':time.perf_counter()-started,
        'real_gpu_commands':False,'cuda_initialized':False,'automatic_retries':0}
    if summary['seconds']>60:
        summary['status']='failed'; summary['error']='CPU total budget exceeded'
    write(output/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'cases':len(results),'seconds':summary['seconds']}),flush=True)
    return 0 if summary['status']=='pass' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fake-child',choices=('normal','nonzero','invalid','stall','close_while_active','double_close'))
    group.add_argument('--output-dir')
    args=parser.parse_args()
    if args.fake_child: fake_child(args.fake_child)
    else: raise SystemExit(run(args.output_dir))
