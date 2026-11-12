"""One-shot CPU integration of the actual strict/lossless SQL worker interfaces.

Only a new synthetic SQLite file is queried. No final questions or databases
are opened. An external Windows Job MUST impose the 30-second total deadline.
Each native child arms an eight-second self-exit and receives one JSONL batch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
WORKERS = {
    'strict': ROOT / 'src/eval/spider_pair_worker.py',
    'surrogateescape': ROOT / 'src/eval/spider_pair_worker_decode_v2.py',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def write_bytes(path, value):
    with Path(path).open('xb') as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items()
                if key not in ('seconds', 'comparison_seconds')}
    if isinstance(value, list): return [without_timing(item) for item in value]
    return value


def shared_semantics(value):
    """Exclude only timing and the explicitly changed decoding metadata."""
    if isinstance(value, dict):
        return {key: shared_semantics(item) for key, item in value.items()
                if key not in ('seconds', 'comparison_seconds', 'decode_mode', 'invalid_text_values')}
    if isinstance(value, list): return [shared_semantics(item) for item in value]
    return value


def no_rows(value):
    if isinstance(value, dict):
        assert not {'rows', 'gold_rows', 'predicted_rows', 'result_rows'} & value.keys()
        for item in value.values(): no_rows(item)
    elif isinstance(value, list):
        for item in value: no_rows(item)


def fixture(output):
    descriptor, name = tempfile.mkstemp(prefix='worker_fixture_', suffix='.sqlite', dir=output)
    os.close(descriptor)
    path = Path(name)
    connection = sqlite3.connect(path)
    try:
        connection.execute('CREATE TABLE fixture(id INTEGER PRIMARY KEY, txt TEXT, payload BLOB)')
        connection.executemany('INSERT INTO fixture VALUES (?, ?, ?)', [
            (1, 'fixture_secret_ASCII', b'\x00\xff'),
            (2, '中文', b'x\xff'),
            (3, '😀e\u0301', b'x\xfe')])
        for index, value in ((4, b'x\xff'), (5, b'x\xfe')):
            connection.execute('INSERT INTO fixture VALUES (?, CAST(? AS TEXT), ?)', (index, value, value))
        connection.commit()
    finally:
        connection.close()
    return path


def test_cases():
    a = 'SELECT id,txt FROM fixture WHERE id<=3'
    # name, gold SQL, predicted SQL, expected match, gold/pred invalid TEXT counts,
    # shared with strict, expected response status, expected prediction status.
    return [
        ('valid_utf8', a+' ORDER BY id', a+' ORDER BY id', True, 0, 0, True, 'scored', 'ok'),
        ('unordered_rows', a, a+' ORDER BY id DESC', True, 0, 0, True, 'scored', 'ok'),
        ('ordered_rows', a+' ORDER BY id', a+' ORDER BY id DESC', False, 0, 0, True, 'scored', 'ok'),
        ('column_permutation', a+' ORDER BY id', 'SELECT txt,id FROM fixture WHERE id<=3 ORDER BY id', True, 0, 0, True, 'scored', 'ok'),
        ('duplicates', 'SELECT txt FROM fixture WHERE id=1 UNION ALL SELECT txt FROM fixture WHERE id=1',
            'SELECT txt FROM fixture WHERE id=1 UNION ALL SELECT txt FROM fixture WHERE id=2', False, 0, 0, True, 'scored', 'ok'),
        ('null_equal', 'SELECT NULL,1', 'SELECT NULL,1', True, 0, 0, True, 'scored', 'ok'),
        ('null_not_empty', 'SELECT NULL', "SELECT ''", False, 0, 0, True, 'scored', 'ok'),
        ('empty_width', 'SELECT txt FROM fixture WHERE 0', 'SELECT id,txt FROM fixture WHERE 0', True, 0, 0, True, 'scored', 'ok'),
        ('numeric_equality', 'SELECT 1', 'SELECT 1.0', True, 0, 0, True, 'scored', 'ok'),
        ('blob_same', 'SELECT payload FROM fixture WHERE id=4', 'SELECT payload FROM fixture WHERE id=4', True, 0, 0, True, 'scored', 'ok'),
        ('blob_distinct', 'SELECT payload FROM fixture WHERE id=4', 'SELECT payload FROM fixture WHERE id=5', False, 0, 0, True, 'scored', 'ok'),
        ('invalid_same', 'SELECT txt FROM fixture WHERE id=4', 'SELECT txt FROM fixture WHERE id=4', True, 1, 1, False, 'scored', 'ok'),
        ('invalid_distinct', 'SELECT txt FROM fixture WHERE id=4', 'SELECT txt FROM fixture WHERE id=5', False, 1, 1, False, 'scored', 'ok'),
        ('invalid_duplicates', 'SELECT txt FROM fixture WHERE id=4 UNION ALL SELECT txt FROM fixture WHERE id=4',
            'SELECT txt FROM fixture WHERE id=4 UNION ALL SELECT txt FROM fixture WHERE id=5', False, 2, 2, False, 'scored', 'ok'),
        ('text_not_blob', 'SELECT txt FROM fixture WHERE id=4', 'SELECT payload FROM fixture WHERE id=4', False, 1, 0, False, 'scored', 'ok'),
        ('write_rejected', 'SELECT 1', 'DELETE FROM fixture WHERE id=1', None, 0, 0, True, 'prediction_error', 'query_rejected'),
        ('sql_error', 'SELECT 1', 'SELECT missing_column FROM fixture', None, 0, 0, True, 'prediction_error', 'sql_error'),
        ('surrogate_request_id_\udcff', 'SELECT 1', 'SELECT 1', True, 0, 0, True, 'scored', 'ok'),
    ]


def request(case, path):
    return {'request_id': case[0], 'db_path': str(path.resolve()), 'gold_sql': case[1],
            'predicted_sql': case[2], 'query_timeout_seconds': 1.0,
            'comparison_timeout_seconds': 1.0, 'seed': 42}


def load_worker(kind):
    spec = importlib.util.spec_from_file_location('actual_decode_check_'+kind, WORKERS[kind])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_batch(kind, requests, output, total_deadline):
    from src.train.windows_job_process_v2 import native_venv_launch, VENV_PYTHON
    if time.perf_counter() > total_deadline-10.0:
        raise TimeoutError('Insufficient budget for one child plus cleanup; no launch')
    stdin = ''.join(json.dumps(item, ensure_ascii=True, allow_nan=False)+'\n' for item in requests).encode('ascii')
    write_bytes(output/(kind+'.stdin.jsonl'), stdin)
    command, env = native_venv_launch([str(VENV_PYTHON), '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                                     '--child-worker', kind], dict(os.environ, CUDA_VISIBLE_DEVICES='',
                                     TEXT2SQL_PROJECT_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1'))
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
    cleanup = None
    try:
        write_json(output/(kind+'.process.json'), {'pid':process.pid, 'native_command':command,
            'self_exit_seconds':8.0, 'parent_communicate_timeout_seconds':8.0,
            'scope':'actual worker main via runpy, synthetic fixture only'})
        stdout, stderr = process.communicate(input=stdin, timeout=8.0)
    except BaseException:
        cleanup_deadline = min(total_deadline, time.perf_counter()+2.0)
        if process.poll() is None: process.kill()
        remaining = cleanup_deadline-time.perf_counter()
        if remaining <= 0: raise TimeoutError('Worker cleanup exceeded its shared deadline')
        stdout, stderr = process.communicate(timeout=remaining)
        write_bytes(output/(kind+'.stdout.failed.jsonl'), stdout)
        write_bytes(output/(kind+'.stderr.failed.txt'), stderr)
        cleanup = {'returncode':process.returncode, 'finished_monotonic':time.perf_counter(),
                   'deadline_monotonic':cleanup_deadline}
        write_json(output/(kind+'.cleanup.json'), cleanup)
        if time.perf_counter()>cleanup_deadline: raise TimeoutError('Worker cleanup exceeded two seconds')
        raise
    write_bytes(output/(kind+'.stdout.jsonl'), stdout)
    write_bytes(output/(kind+'.stderr.txt'), stderr)
    assert process.poll() is not None and process.returncode==0, (kind,process.returncode)
    assert not stderr, (kind,stderr.decode('utf-8',errors='backslashreplace'))
    text = stdout.decode('ascii')  # The actual CLI must keep all surrogate/Unicode output escaped.
    assert b'\\udcff' in stdout and b'fixture_secret_ASCII' not in stdout
    responses = [json.loads(line) for line in text.splitlines()]
    assert len(responses)==len(requests)
    for original, response in zip(requests,responses,strict=True):
        assert response['request_id']==original['request_id']
        no_rows(response)
    elapsed = time.perf_counter()-started
    assert elapsed<10.0
    return responses, {'pid':process.pid,'returncode':process.returncode,'seconds':elapsed,
        'response_count':len(responses),'ascii_only_stdout':True,'surrogate_request_id_roundtrip':True,
        'actual_worker_main_invoked':True,'fixture_row_values_not_emitted':True,'cleanup':cleanup}


def child(kind):
    timer = threading.Timer(8.0, lambda: os._exit(98)); timer.daemon=True; timer.start()
    try:
        sys.argv = [str(WORKERS[kind])]
        runpy.run_path(str(WORKERS[kind]), run_name='__main__')
        assert not any(name in sys.modules for name in ('torch','transformers','peft','bitsandbytes'))
    finally:
        timer.cancel()


def run(output):
    started=time.perf_counter(); deadline=started+30.0
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT): raise ValueError('Evidence must stay inside project')
    output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),*WORKERS.values(),ROOT/'third_party/test-suite-sql-eval/exec_eval.py',
           ROOT/'src/train/windows_job_process_v2.py',ROOT/'reports/spider_evaluator_source.json']
    before={path.relative_to(ROOT).as_posix():digest(path) for path in paths}
    write_json(output/'claim.json',{'maximum_seconds':30,'maximum_child_seconds':8,
        'child_launches_planned':2,'automatic_retries':0,'final_database_accessed':False,
        'final_scoring_performed':False,'source_sha256_before':before})
    report={'status':'failed','scope':'actual worker synthetic CPU integration only',
        'final_database_accessed':False,'final_scoring_performed':False,'model_loaded':False,
        'training':False,'automatic_retries':0,'maximum_seconds':30}
    db=None; db_before=None
    try:
        provenance=json.loads((ROOT/'reports/spider_evaluator_source.json').read_text(encoding='utf-8'))
        vendor=ROOT/provenance['local_dir']
        for name, meta in provenance['source_files'].items(): assert digest(vendor/name)==meta['sha256']
        db=fixture(output); db_before=digest(db)
        modules={kind:load_worker(kind) for kind in WORKERS}
        cases=test_cases(); requests=[request(case,db) for case in cases]
        direct={kind:[module.execute_request(item) for item in requests] for kind,module in modules.items()}
        cli={}; processes={}
        for kind in WORKERS:
            cli[kind],processes[kind]=cli_batch(kind,requests,output,deadline)
            assert without_timing(cli[kind])==without_timing(direct[kind]), kind
        results=[]
        for index,case in enumerate(cases):
            name,gold,pred,matched,gold_count,pred_count,compatible,status,pred_status=case
            old,new=cli['strict'][index],cli['surrogateescape'][index]
            assert new['status']==status and new['matched'] is matched, (name,new)
            assert new['gold']['status']=='ok' and new['prediction']['status']==pred_status
            assert new['prediction_executable']==(pred_status=='ok')
            for side,count in (('gold',gold_count),('prediction',pred_count)):
                assert new[side]['decode_mode']=='utf-8/surrogateescape'
                assert type(new[side]['invalid_text_values']) is int and new[side]['invalid_text_values']==count
                assert old[side]['decode_mode']=='strict'
            if compatible:
                assert shared_semantics(old)==shared_semantics(new), name
            else:
                assert old['status']==('gold_error' if name=='text_not_blob' else 'both_error')
                assert old['gold']['status']=='decode_error' and old['matched'] is None
                if name!='text_not_blob': assert old['prediction']['status']=='decode_error'
            results.append({'case':name,'status':'pass','matched':new['matched'],
                'response_status':new['status'],'old_strict_status':old['status'],
                'gold_invalid_text_values':gold_count,'prediction_invalid_text_values':pred_count,
                'same_shared_semantics_as_strict':compatible,'direct_vs_actual_cli_equal':True})
        meta,rows=modules['surrogateescape']._execute_query(db,'SELECT txt,payload FROM fixture WHERE id=4',1.0)
        assert meta['status']=='ok' and meta['invalid_text_values']==1
        assert len(rows)==1 and isinstance(rows[0][0],str) and isinstance(rows[0][1],bytes)
        assert rows[0][0].encode('utf-8','surrogateescape')==rows[0][1]==b'x\xff'
        assert rows[0][0]!=rows[0][1]
        assert not any(name in sys.modules for name in ('torch','transformers','peft','bitsandbytes'))
        report.update(status='pass',cases=results,case_count=len(cases),processes=processes,
            native_child_count=2,actual_cli_and_execute_request_equal=True,
            strict_shared_semantics_cases=sum(case[6] for case in cases),
            strict_comparison_exclusions=['seconds','comparison_seconds','decode_mode','invalid_text_values'],
            invalid_text_protocol_revision_not_original_strict_equivalence=True,
            text_and_blob_distinct=True,blob_bytes_preserved=True,invalid_text_roundtrip_bytes_equal=True,
            worker_metadata_contains_no_result_rows=True,
            final_scoring_requires_separate_authorization=True,
            comparator_source_sha256=provenance['source_files']['exec_eval.py']['sha256'])
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        report['source_sha256_before']=before
        report['source_sha256_after']={path.relative_to(ROOT).as_posix():digest(path) for path in paths}
        report['protected_sources_unchanged']=before==report['source_sha256_after']
        if db is not None:
            report['fixture_path']=db.relative_to(ROOT).as_posix()
            report['fixture_sha256_before']=db_before; report['fixture_sha256_after']=digest(db)
            report['fixture_bytes_unchanged']=db_before==report['fixture_sha256_after']
        report['seconds']=time.perf_counter()-started
        if not report['protected_sources_unchanged'] or not report.get('fixture_bytes_unchanged') or report['seconds']>30:
            report['status']='failed'; report['final_validation_error']='Preservation or 30-second budget failed'
        write_json(output/'result.json',report)
    print(json.dumps({'status':report['status'],'seconds':report['seconds'],
        'case_count':report.get('case_count'),'error':report.get('error')},ensure_ascii=True),flush=True)
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output-dir')
    group.add_argument('--child-worker',choices=tuple(WORKERS))
    arguments=parser.parse_args()
    if arguments.child_worker:
        child(arguments.child_worker)
    else:
        raise SystemExit(run(arguments.output_dir))
