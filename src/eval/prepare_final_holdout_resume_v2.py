"""Freeze repaired infrastructure after bounded tests; retain the original recipe."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eval.final_holdout_common_v1 import read_json,write,sha256,relative,verify_prepared_binding

CONFIG=ROOT/'configs/final_holdout_resume_v2.json'


def prepare():
    started=time.perf_counter();cfg=read_json(CONFIG)
    pref=ROOT/cfg['preflight_report'];freeze_path=ROOT/cfg['frozen_report']
    if freeze_path.exists() or pref.exists() or (ROOT/cfg['run_root']).exists():
        raise FileExistsError('Resume freeze/preflight already claimed')
    write(pref,{'status':'checking','training':False},True)
    result={'status':'failed'}
    try:
        parent=verify_prepared_binding()
        old_cfg=read_json(ROOT/cfg['parent_config'])
        changed={'version','run_root','frozen_report','preflight_report'}
        for key,value in old_cfg.items():
            if key not in changed and cfg[key]!=value:
                raise ValueError(f'Frozen recipe/runtime budget changed: {key}')
        for package,expected in parent['packages'].items():
            if version(package)!=expected: raise ValueError(f'Package changed: {package}')
        if version('peft')!=parent['peft_version']: raise ValueError('PEFT changed')
        test_sources={}
        test_reports={}
        for mode in ('cpu','live'):
            path=ROOT/f'outputs/checks/final_telemetry_v2_{mode}_v1/result.json'
            report=read_json(path)
            if report['status']!='pass' or not report['sources_unchanged']:
                raise ValueError(f'{mode} validation did not pass')
            for name,expected in report['source_sha256'].items():
                if sha256(ROOT/name)!=expected: raise ValueError(f'Tested file changed: {name}')
                if name in test_sources and test_sources[name]!=expected: raise ValueError('CPU/live tested different source')
                test_sources[name]=expected
            test_reports[mode]={'path':relative(path),'sha256':sha256(path),'seconds':report['seconds']}
        from src.infer.final_holdout_resume_v2 import preserved_prefix,PARENT_PROMPTS
        payload,prefix=preserved_prefix()
        if len(prefix)!=cfg['prefix_examples'] or sha256(ROOT/cfg['prefix_path'])!=cfg['prefix_sha256']:
            raise ValueError('Prefix identity differs')
        from src.train.windows_job_process_v2 import interpreter_evidence
        interpreter=interpreter_evidence()
        if not all(row['matches'] for row in interpreter): raise ValueError('Interpreter files changed')
        protected=read_json(ROOT/'outputs/checks/final_telemetry_audit_v1/preservation.json')['source_sha256']
        protected={name:digest for name,digest in protected.items() if name not in ('README.md','docs/项目总结与简历面试手册.md')}
        for name,expected in protected.items():
            if sha256(ROOT/name)!=expected: raise ValueError(f'Historical evidence changed: {name}')
        protected.update(test_sources)
        for name in ('src/eval/prepare_final_holdout_resume_v2.py',cfg['parent_preflight'],relative(PARENT_PROMPTS),
                     'src/eval/evaluate_final_holdout_v2.py',relative(CONFIG)):
            protected[name]=sha256(ROOT/name)
        for mode in ('cpu','live'):
            for path in (ROOT/f'outputs/checks/final_telemetry_v2_{mode}_v1').rglob('*'):
                if path.is_file(): protected[relative(path)]=sha256(path)
        for name in test_sources:
            if name.endswith('.py'): ast.parse((ROOT/name).read_text(encoding='utf-8-sig'),filename=name)
        run=ROOT/cfg['run_root'];run.mkdir(parents=True,exist_ok=False)
        for name in [*test_sources,'src/eval/prepare_final_holdout_resume_v2.py']:
            dest=run/'frozen_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        frozen={'status':'frozen_repair_after_cpu_and_no_model_checks','created_utc':datetime.now(timezone.utc).isoformat(),
            'config':cfg,'source_sha256':protected,'parent_freeze_sha256':sha256(ROOT/'reports/final_holdout_v1_freeze.json'),
            'model_prompt_generation_EX_unchanged':True,'preserved_prefix_sha256':cfg['prefix_sha256'],
            'preserved_prefix_examples':616,'remaining_base_examples':418,'sft_examples':1034,
            'no_answer_selection':True,'tests':test_reports,'training':False,'final_used_for_tuning':False}
        write(freeze_path,frozen,True)
        live=read_json(ROOT/test_reports['live']['path'])
        result={**parent,'status':'pass','config':cfg,'frozen_report_sha256':sha256(freeze_path),
            'parent_preflight_sha256':sha256(ROOT/cfg['parent_preflight']),
            'parent_cpu_checks_reused_by_unchanged_hash':True,
            'repair_checks':test_reports,'protected_source_count':len(protected),
            'prefix_sha256':cfg['prefix_sha256'],'reused_prefix_count':616,'new_base_examples':418,
            'telemetry_executable_sha256':live['worker']['telemetry']['executable_sha256'],
            'telemetry_executable_path':live['worker']['telemetry']['command'][0],
            'new_interpreter_evidence':interpreter,'cuda_initialized':False,'model_loaded':False,
            'training_performed':False,'final_used_for_tuning':False}
        if any(name in sys.modules for name in ('torch','transformers','peft','bitsandbytes')):
            raise RuntimeError('CPU infrastructure preflight imported model libraries')
    except BaseException as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise
    finally:
        result['resume_cpu_preflight_seconds']=time.perf_counter()-started
        write(pref,result)
    print({key:result[key] for key in ('status','reused_prefix_count','new_base_examples','protected_source_count','resume_cpu_preflight_seconds')},flush=True)


if __name__=='__main__':prepare()
