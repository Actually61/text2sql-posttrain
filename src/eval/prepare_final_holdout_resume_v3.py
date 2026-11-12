"""Freeze a path-only new attempt after v2's zero-worker resource-gate failure.

Reuse the successful v2 CPU/live validation by exact source hashes.  Verify that
config, generator and scorer differ only by the declared attempt-path/version
renames before creating a new freeze.  No CUDA, model or SQL execution occurs.
"""
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

CONFIG=ROOT/'configs/final_holdout_resume_v3.json'
V2_CONFIG=ROOT/'configs/final_holdout_resume_v2.json'
V2_PREFLIGHT=ROOT/'reports/final_holdout_resume_v2_preflight.json'
V2_FREEZE=ROOT/'reports/final_holdout_resume_v2_freeze.json'
V2_GATE=ROOT/'reports/final_holdout_resume_v2_base_start_resources.json'
CLONES=(
    ('configs/final_holdout_resume_v2.json','configs/final_holdout_resume_v3.json'),
    ('src/infer/final_holdout_resume_v2.py','src/infer/final_holdout_resume_v3.py'),
    ('src/eval/evaluate_final_holdout_v2.py','src/eval/evaluate_final_holdout_v3.py'),
)


def verify_path_only_clones():
    """Byte-exact allowed substitutions, so changed computation cannot slip in."""
    evidence=[]
    for old,new in CLONES:
        expected=(ROOT/old).read_bytes().replace(b'final_holdout_resume_v2',b'final_holdout_resume_v3')
        expected=expected.replace(b'outputs/final_holdout_v2',b'outputs/final_holdout_v3')
        if new=='src/eval/evaluate_final_holdout_v3.py':
            expected=expected.replace(b'V2_FREEZE',b'V3_FREEZE').replace(b' v2 ',b' v3 ').replace(b'"v2 ',b'"v3 ')
        if (ROOT/new).read_bytes()!=expected:
            raise ValueError(f'Unapproved change beyond attempt paths/version: {new}')
        if new.endswith('.py'):
            ast.parse(expected.decode('utf-8'),filename=new)
        evidence.append({'source':old,'source_sha256':sha256(ROOT/old),
                         'clone':new,'clone_sha256':sha256(ROOT/new),
                         'approved_path_version_substitutions_only':True})
    return evidence


def prepare():
    started=time.perf_counter();cfg=read_json(CONFIG)
    pref=ROOT/cfg['preflight_report'];freeze_path=ROOT/cfg['frozen_report']
    if freeze_path.exists() or pref.exists() or (ROOT/cfg['run_root']).exists():
        raise FileExistsError('Resume freeze/preflight already claimed')
    write(pref,{'status':'checking','training':False},True)
    result={'status':'failed'}
    try:
        v2_cfg=read_json(V2_CONFIG);v2_pref=read_json(V2_PREFLIGHT);v2_frozen=read_json(V2_FREEZE)
        if (v2_pref['status']!='pass' or v2_pref['config']!=v2_cfg
                or v2_frozen['config']!=v2_cfg
                or v2_pref['frozen_report_sha256']!=sha256(V2_FREEZE)):
            raise ValueError('v2 CPU preflight/freeze binding is not successful and unchanged')
        for name,expected in v2_frozen['source_sha256'].items():
            if sha256(ROOT/name)!=expected: raise ValueError(f'Frozen v2 source changed: {name}')
        gate=read_json(V2_GATE)
        if (gate['status']!='not_ready' or gate['run_directory_created'] is not False
                or gate['cuda_initialized'] is not False or gate['model_loaded'] is not False
                or gate['training_updates']!=0):
            raise ValueError('Expected preserved v2 zero-worker resource-gate failure')
        if any((ROOT/v2_cfg['run_root']/arm).exists() for arm in ('base','sft')):
            raise ValueError('v2 created an inference arm; this path-only zero-worker attempt is not applicable')
        clone_evidence=verify_path_only_clones()
        changed={'version','run_root','frozen_report','preflight_report'}
        if set(cfg)!=set(v2_cfg) or any(cfg[key]!=value for key,value in v2_cfg.items() if key not in changed):
            raise ValueError('v3 changed a v2 configuration value beyond approved version/output paths')
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
                if v2_frozen['source_sha256'].get(name)!=expected:
                    raise ValueError(f'v2 freeze does not bind tested file: {name}')
                if name in test_sources and test_sources[name]!=expected: raise ValueError('CPU/live tested different source')
                test_sources[name]=expected
            test_reports[mode]={'path':relative(path),'sha256':sha256(path),'seconds':report['seconds']}
        from src.infer.final_holdout_resume_v3 import preserved_prefix,PARENT_PROMPTS
        payload,prefix=preserved_prefix()
        if len(prefix)!=cfg['prefix_examples'] or sha256(ROOT/cfg['prefix_path'])!=cfg['prefix_sha256']:
            raise ValueError('Prefix identity differs')
        from src.train.windows_job_process_v2 import interpreter_evidence
        interpreter=interpreter_evidence()
        if not all(row['matches'] for row in interpreter): raise ValueError('Interpreter files changed')
        protected=dict(v2_frozen['source_sha256'])
        for name,expected in protected.items():
            if sha256(ROOT/name)!=expected: raise ValueError(f'Historical evidence changed: {name}')
        protected.update(test_sources)
        for name in ('src/eval/prepare_final_holdout_resume_v3.py',cfg['parent_preflight'],relative(PARENT_PROMPTS),
                     'src/infer/final_holdout_resume_v3.py','src/eval/evaluate_final_holdout_v3.py',relative(CONFIG),
                     relative(V2_CONFIG),relative(V2_PREFLIGHT),relative(V2_FREEZE),relative(V2_GATE)):
            protected[name]=sha256(ROOT/name)
        for mode in ('cpu','live'):
            for path in (ROOT/f'outputs/checks/final_telemetry_v2_{mode}_v1').rglob('*'):
                if path.is_file(): protected[relative(path)]=sha256(path)
        for name in test_sources:
            if name.endswith('.py'): ast.parse((ROOT/name).read_text(encoding='utf-8-sig'),filename=name)
        ast.parse(Path(__file__).read_text(encoding='utf-8'),filename=relative(Path(__file__)))
        run=ROOT/cfg['run_root'];run.mkdir(parents=True,exist_ok=False)
        for name in [*test_sources,*(new for _,new in CLONES),'src/eval/prepare_final_holdout_resume_v3.py']:
            dest=run/'frozen_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        frozen={'status':'frozen_path_only_attempt_after_v2_zero_worker_gate','created_utc':datetime.now(timezone.utc).isoformat(),
            'config':cfg,'source_sha256':protected,'parent_freeze_sha256':sha256(ROOT/'reports/final_holdout_v1_freeze.json'),
            'v2_implementation_freeze_sha256':sha256(V2_FREEZE),'v2_preflight_sha256':sha256(V2_PREFLIGHT),
            'v2_resource_failure_sha256':sha256(V2_GATE),'path_only_clones':clone_evidence,
            'v2_worker_launches':0,'v2_predictions_generated':0,
            'model_prompt_generation_EX_unchanged':True,'preserved_prefix_sha256':cfg['prefix_sha256'],
            'preserved_prefix_examples':616,'remaining_base_examples':418,'sft_examples':1034,
            'no_answer_selection':True,'tests':test_reports,'training':False,'final_used_for_tuning':False}
        write(freeze_path,frozen,True)
        live=read_json(ROOT/test_reports['live']['path'])
        result={**parent,'status':'pass','config':cfg,'frozen_report_sha256':sha256(freeze_path),
            'parent_preflight_sha256':sha256(ROOT/cfg['parent_preflight']),
            'parent_cpu_checks_reused_by_unchanged_hash':True,
            'v2_preflight_sha256':sha256(V2_PREFLIGHT),'path_only_clones':clone_evidence,
            'v2_resource_failure_preserved':True,'fresh_resource_gate_required':True,
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
