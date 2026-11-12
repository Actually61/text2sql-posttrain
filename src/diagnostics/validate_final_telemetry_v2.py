"""Bounded native-Job validation; CPU fixtures and a no-model live telemetry probe."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.final_holdout_common_v1 import read_json, write, sha256, relative
from src.train.windows_job_process_v2 import JobProcess, VENV_PYTHON
from src.infer.final_eval_exit_wait_v2 import terminate_and_wait, close_without_new_wait
from src.train.pid_supervision_formal_v2 import confirm_natural_exit

NEW_FILES = ['src/infer/gpu_sampler_v2.py','src/diagnostics/test_gpu_sampler_v2.py',
    'src/diagnostics/validate_final_telemetry_v2.py','src/infer/final_eval_exit_wait_v2.py',
    'src/infer/final_holdout_resume_v2.py','src/eval/evaluate_final_holdout_v2.py',
    'configs/final_holdout_resume_v2.json']


def child_live(directory):
    from src.infer.gpu_sampler_v2 import GPUSampler
    started = time.perf_counter()
    sampler = GPUSampler(0.5, event_path=directory/'live_telemetry_events.jsonl')
    result = {'status':'failed','model_loaded':False,'cuda_initialized':False,'training':False}
    attempted_close = False
    try:
        sampler.start()
        while len(sampler.samples) < 20:
            if sampler.errors:
                raise RuntimeError(str(sampler.errors))
            if time.perf_counter()-started >= 20:
                raise TimeoutError('Twenty samples were not obtained within20 seconds')
            time.sleep(.05)
        attempted_close = True; sampler.close()
        if sampler.errors:
            raise RuntimeError(str(sampler.errors))
        if any(name in sys.modules for name in ('torch','transformers','peft','bitsandbytes')):
            raise RuntimeError('No-model probe imported a model library')
        result.update(status='pass',sample_count=len(sampler.samples),errors=sampler.errors,
                      telemetry=sampler.report())
    except BaseException as error:
        result.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise
    finally:
        if not attempted_close:
            attempted_close=True
            try:
                sampler.close()
            except BaseException as error:
                result['cleanup_error']=f'{type(error).__name__}: {error}'
        result.update(seconds=time.perf_counter()-started,telemetry=sampler.report())
        write(directory/'worker_result.json',result,True)


def run_job(command, directory, seconds, *, force_terminate=False):
    events=[]; handles=[]; job=None; started=time.perf_counter()
    result={'status':'failed','maximum_seconds':seconds,'events':events}
    def record(event,**values):
        events.append({'event':event,'monotonic':time.perf_counter(),**values})
    try:
        job=JobProcess(command,ROOT,dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),directory/'console.log')
        handles.append(job.open_process(job.pid))
        result['worker_pid']=job.pid
        if force_terminate:
            result['termination']=terminate_and_wait(job,handles,record,timeout=2,max_lifetime_seconds=6306)
            result['exit_code']=124
        else:
            code=job.wait(max(0,started+seconds-2-time.perf_counter()))
            result.update(exit_code=code,exits=confirm_natural_exit(job,handles,record,
                min(time.perf_counter()+2,started+seconds),max_lifetime_seconds=seconds))
            if code != 0:
                raise RuntimeError(f'Validation child exited {code}')
        if job.active_pids():
            raise RuntimeError('Validation Job retained processes')
        result.update(status='pass',job_active_pids_after_cleanup=[])
    except BaseException as error:
        result.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        if job is not None and job.active_pids():
            retained={handle.pid:handle for handle in handles}
            for pid in job.active_pids():
                if pid not in retained:
                    handle=job.open_process(pid); handles.append(handle); retained[pid]=handle
            result['termination']=terminate_and_wait(job,[retained[p] for p in job.active_pids()],record,
                timeout=2,max_lifetime_seconds=6306)
        raise
    finally:
        for handle in handles: handle.close()
        if job is not None: close_without_new_wait(job)
        result['seconds']=time.perf_counter()-started
        write(directory/'controller_result.json',result,True)
    return result


def main(mode):
    directory=ROOT/f'outputs/checks/final_telemetry_v2_{mode}_v1'
    directory.mkdir(parents=True,exist_ok=False)
    sources={name:sha256(ROOT/name) for name in NEW_FILES}
    write(directory/'source_sha256.json',sources,True)
    for name in sources:
        destination=directory/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    result={'status':'failed','mode':mode,'source_sha256':sources,'model_loaded':False,'training':False}
    started=time.perf_counter()
    try:
        if mode=='cpu':
            case=directory/'fixtures'; case.mkdir()
            result['fixtures']=run_job([str(VENV_PYTHON),'-B','-X','utf8',
                str(ROOT/'src/diagnostics/test_gpu_sampler_v2.py'),'--output-dir',str(case/'test_output')],case,55)
            cap=directory/'exit_cap';cap.mkdir()
            result['exit_cap_6306']=run_job([str(VENV_PYTHON),'-B','-c','import time;time.sleep(8)'],cap,4,force_terminate=True)
            if time.perf_counter()-started>60:
                raise TimeoutError('CPU suite exceeded60 seconds')
            result['tests']=read_json(case/'test_output/summary.json')
            if result['tests']['status']!='pass':
                raise RuntimeError('CPU fixture report did not pass')
        else:
            result['probe']=run_job([str(VENV_PYTHON),'-B','-X','utf8',str(Path(__file__)),
                                    '--worker-live','--directory',str(directory)],directory,30)
            result['worker']=read_json(directory/'worker_result.json')
            if result['worker']['status']!='pass':
                raise RuntimeError('No-model live probe did not pass')
        for name,expected in sources.items():
            if sha256(ROOT/name)!=expected:
                raise ValueError(f'Source changed during validation: {name}')
        result.update(status='pass',sources_unchanged=True)
    except BaseException as error:
        result.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise
    finally:
        result['seconds']=time.perf_counter()-started
        write(directory/'result.json',result,True)
    print(json.dumps({key:result[key] for key in ('status','mode','seconds','sources_unchanged')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['cpu','live'])
    parser.add_argument('--worker-live',action='store_true')
    parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    if args.worker_live:
        child_live(args.directory)
    elif args.mode:
        main(args.mode)
    else:
        parser.error('Mode required')
