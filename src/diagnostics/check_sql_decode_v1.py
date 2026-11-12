"""CPU-only evidence for a lossless SQLite TEXT decoding compatibility proposal.

This is not a replacement evaluator and produces no final EX score.  Only the
already-failed final question 455 gold SQL is read from the original database,
once, in bytes mode. All other SQL operates on a new synthetic temporary file.
Run this program under an external 30-second Job deadline.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
GOLD_SQL = 'SELECT first_name ,  last_name FROM players ORDER BY birth_date'
SOURCE_INDEX = 455
DB_RELATIVE = 'data/raw/spider/database/wta_1/wta_1.sqlite'
SCORED_PATH = ROOT / 'outputs/final_holdout_v3/scores_base/per_example.jsonl'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        # Synthetic surrogate code points must be escaped in evidence JSON.
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def decoder(mode):
    if mode == 'bytes': return bytes
    if mode not in ('strict', 'surrogateescape'): raise ValueError(mode)
    return lambda raw: raw.decode('utf-8', errors=mode)


def open_readonly(path, mode, overall_deadline):
    connection = sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True, timeout=2.0)
    try:
        connection.enable_load_extension(False)
        connection.execute('PRAGMA query_only = ON')
        connection.text_factory = decoder(mode)
        actions = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ}
        if hasattr(sqlite3, 'SQLITE_RECURSIVE'): actions.add(sqlite3.SQLITE_RECURSIVE)
        def authorize(action, first, second, database, source):
            if action in actions: return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (second or first or '').casefold() not in {
                    'load_extension', 'readfile', 'writefile'}:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        connection.set_authorizer(authorize)
        deadline = min(overall_deadline, time.perf_counter()+2.0)
        connection.set_progress_handler(lambda: int(time.perf_counter() >= deadline), 1000)
        return connection, deadline
    except BaseException:
        connection.close()
        raise


def query(path, sql, mode, overall_deadline):
    connection, deadline = open_readonly(path, mode, overall_deadline)
    try:
        cursor = connection.execute(sql)
        rows = []
        while True:
            if time.perf_counter() >= deadline: raise TimeoutError('Read-only query budget exceeded')
            batch = cursor.fetchmany(256)
            rows.extend(batch)
            if not batch: break
        if time.perf_counter() >= deadline: raise TimeoutError('Read-only query budget exceeded')
        return rows
    finally:
        connection.close()


def make_fixture(output):
    descriptor, filename = tempfile.mkstemp(prefix='decode_fixture_', suffix='.sqlite', dir=output)
    os.close(descriptor)
    path = Path(filename)
    connection = sqlite3.connect(path)
    invalid = [b'x\xff', b'x\xfe', b'bad\xc0\xaf', b'cut\xe2\x82', b'\xed\xa0\x80',
               b'\xf0\x28\x8c\x28', bytes(range(256))]
    try:
        connection.execute('CREATE TABLE fixture(id INTEGER PRIMARY KEY, grp TEXT, txt TEXT, value REAL, optional TEXT, payload BLOB)')
        connection.executemany('INSERT INTO fixture VALUES (?, ?, ?, ?, ?, ?)', [
            (1, 'valid', 'ASCII', 1.0, None, b'\x00\xff'),
            (2, 'valid', '中文', 2.25, 'é', b'blob'),
            (3, 'valid', '😀e\u0301', -0.0, '', b'\x80')])
        for offset, raw in enumerate(invalid):
            connection.execute('INSERT INTO fixture VALUES (?, ?, CAST(? AS TEXT), ?, ?, ?)',
                (101+offset, 'invalid', raw, float(offset), None, b'blob'))
        connection.commit()
    finally:
        connection.close()
    return path, invalid


def comparator_checks(module, fixture, overall_deadline):
    # These fixtures test the pinned comparator's existing behaviour, including
    # its intentional limitations. They are not new scoring rules.
    a = 'SELECT id,txt FROM fixture WHERE id<=3 ORDER BY id'
    cases = [
        ('identical_valid_utf8', a, a, False, True),
        ('row_order_ignored', a, a+' DESC', False, True),
        ('row_order_required', a, a+' DESC', True, False),
        ('global_column_permutation', a, 'SELECT txt,id FROM fixture WHERE id<=3 ORDER BY id', True, True),
        ('non_global_column_permutation', a,
            'SELECT CASE WHEN id=1 THEN txt ELSE id END, CASE WHEN id=1 THEN id ELSE txt END FROM fixture WHERE id<=3 ORDER BY id', False, False),
        ('duplicate_multiplicity',
            'SELECT txt FROM fixture WHERE id=1 UNION ALL SELECT txt FROM fixture WHERE id=1 UNION ALL SELECT txt FROM fixture WHERE id=2',
            'SELECT txt FROM fixture WHERE id=1 UNION ALL SELECT txt FROM fixture WHERE id=2 UNION ALL SELECT txt FROM fixture WHERE id=2', False, False),
        ('null_matches_null', 'SELECT NULL,1', 'SELECT NULL,1', False, True),
        ('null_not_empty_string', 'SELECT NULL', "SELECT ''", False, False),
        ('empty_ignores_column_width', 'SELECT txt FROM fixture WHERE 0', 'SELECT id,txt FROM fixture WHERE 0', False, True),
        ('integer_float_python_equality', 'SELECT 1', 'SELECT 1.0', False, True),
        ('no_new_numeric_tolerance', 'SELECT 0.1+0.2', 'SELECT 0.3', False, False),
        ('four_column_permutation', 'SELECT id,txt,value,optional FROM fixture WHERE id<=3 ORDER BY id',
            'SELECT optional,value,txt,id FROM fixture WHERE id<=3 ORDER BY id', True, True),
    ]
    records = []
    for name, gold, prediction, ordered, expected in cases:
        values = {}
        for mode in ('strict', 'surrogateescape'):
            left = query(fixture, gold, mode, overall_deadline)
            right = query(fixture, prediction, mode, overall_deadline)
            random.seed(42)
            values[mode] = bool(module.result_eq(left, right, ordered))
        assert values['strict'] == values['surrogateescape'] == expected, (name, values, expected)
        records.append({'name':name, 'order_matters':ordered, 'expected':expected,
            'strict':values['strict'], 'surrogateescape':values['surrogateescape'], 'status':'pass'})
    # For invalid TEXT there is no strict execution result. Compare only with
    # an explicit bytes-valued oracle, never label this old-protocol equivalence.
    invalid_cases = [
        ('invalid_same_bytes', 'SELECT txt FROM fixture WHERE id=101', 'SELECT txt FROM fixture WHERE id=101', False, True),
        ('invalid_distinct_bytes', 'SELECT txt FROM fixture WHERE id=101', 'SELECT txt FROM fixture WHERE id=102', False, False),
        ('invalid_global_column_permutation', 'SELECT id,txt FROM fixture WHERE id IN (101,102) ORDER BY id',
            'SELECT txt,id FROM fixture WHERE id IN (101,102) ORDER BY id', True, True),
        ('invalid_duplicates', 'SELECT txt FROM fixture WHERE id=101 UNION ALL SELECT txt FROM fixture WHERE id=101',
            'SELECT txt FROM fixture WHERE id=101 UNION ALL SELECT txt FROM fixture WHERE id=102', False, False),
        ('invalid_row_order', 'SELECT txt FROM fixture WHERE id IN (101,102) ORDER BY id',
            'SELECT txt FROM fixture WHERE id IN (101,102) ORDER BY id DESC', True, False),
    ]
    invalid_records = []
    for name, gold, prediction, ordered, expected in invalid_cases:
        values = {}
        for mode in ('bytes','surrogateescape'):
            random.seed(42)
            values[mode] = bool(module.result_eq(query(fixture,gold,mode,overall_deadline),
                query(fixture,prediction,mode,overall_deadline), ordered))
        assert values['bytes'] == values['surrogateescape'] == expected, (name,values)
        invalid_records.append({'name':name,'order_matters':ordered,'expected':expected,
            'bytes_oracle':values['bytes'],'surrogateescape':values['surrogateescape'],'status':'pass',
            'strict_protocol_equivalence_claimed':False})
    return records, invalid_records


def first_bad_original_value(path, sql, overall_deadline):
    connection, deadline = open_readonly(path, 'bytes', overall_deadline)
    try:
        cursor = connection.execute(sql)
        columns = [item[0] for item in cursor.description]
        index = 0
        while True:
            if time.perf_counter() >= deadline: raise TimeoutError('Original-query diagnostic exceeded two seconds')
            row = cursor.fetchone()
            if row is None: raise AssertionError('Expected invalid UTF-8 was not found in the known gold query')
            for column_index, value in enumerate(row):
                if not isinstance(value, bytes): continue
                try:
                    value.decode('utf-8', errors='strict')
                except UnicodeDecodeError as error:
                    decoded = value.decode('utf-8', errors='surrogateescape')
                    assert decoded.encode('utf-8', errors='surrogateescape') == value
                    return {'source_index':SOURCE_INDEX, 'db_id':'wta_1', 'mode':'ro', 'text_factory':'bytes',
                        'gold_query_executions':1, 'row_index_zero_based':index, 'result_column_index_zero_based':column_index,
                        'column_name':columns[column_index], 'value_byte_length':len(value), 'value_byte_hex':value.hex(),
                        'decode_error_start':error.start, 'decode_error_end':error.end, 'decode_error_reason':error.reason,
                        'round_trip_bytes_equal':True, 'scanned_result_rows':index+1, 'stopped_at_first_bad_value':True,
                        'query_result_row_index_not_table_primary_key':True,
                        'storage_type_limitation':'text_factory=bytes alone does not distinguish returned TEXT from BLOB; the existing strict decoder error supplies separate TEXT-failure evidence.',
                        'complete_result_rows_or_names_published':False, 'query_result_not_compared_or_scored':True}
            index += 1
    finally:
        connection.close()


def run(output):
    started = time.perf_counter(); deadline = started+30.0
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT): raise ValueError('Output must stay in the project')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'failed','scope':'CPU diagnostic and synthetic comparator checks; not final scoring',
        'maximum_seconds':30, 'model_loaded':False, 'training':False, 'final_EX_computed':False,
        'strict_worker_replaced':False, 'compatibility_change_authorized_for_final_scoring':False}
    db = ROOT / DB_RELATIVE
    db_before = digest(db)
    source_paths = [Path(__file__), ROOT/'src/eval/spider_pair_worker.py', SCORED_PATH,
        ROOT/'configs/evaluation.json', ROOT/'data/processed/spider_v1/split_manifest.json',
        ROOT/'reports/spider_evaluator_source.json']
    source_before = {str(path.relative_to(ROOT)).replace('\\','/'):digest(path) for path in source_paths}
    write_new(output/'claim.json', {'created_utc':datetime.now(timezone.utc).isoformat(),
        'maximum_seconds':30,'original_gold_query_source_index':455,'original_queries_authorized':1,
        'automatic_retries':0,'protected_source_sha256':source_before,'database_sha256_before':db_before})
    try:
        split = read(ROOT/'data/processed/spider_v1/split_manifest.json')
        assert db_before == split['source_sha256'][DB_RELATIVE]
        recorded = [json.loads(line) for line in SCORED_PATH.read_text(encoding='utf-8').splitlines()]
        matches = [item for item in recorded if item.get('source_index') == SOURCE_INDEX]
        assert len(matches)==1
        failed = matches[0]
        assert failed['db_id']=='wta_1' and failed['gold_sql']==GOLD_SQL
        assert failed['gold']['status']==failed['prediction']['status']=='decode_error'
        provenance = read(ROOT/'reports/spider_evaluator_source.json')
        vendor = ROOT/provenance['local_dir']
        for name, meta in provenance['source_files'].items():
            assert digest(vendor/name)==meta['sha256'], name
        sys.path.insert(0,str(vendor))
        spec=importlib.util.spec_from_file_location('pinned_decode_result_eq_v1',vendor/'exec_eval.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        fixture, invalid_values = make_fixture(output)
        fixture_before = digest(fixture)
        valid_sql="SELECT id,grp,txt,value,optional,payload FROM fixture WHERE grp='valid' ORDER BY id"
        strict_rows=query(fixture,valid_sql,'strict',deadline)
        proposed_rows=query(fixture,valid_sql,'surrogateescape',deadline)
        assert strict_rows==proposed_rows
        assert [[type(value).__name__ for value in row] for row in strict_rows] == [[type(value).__name__ for value in row] for row in proposed_rows]
        assert isinstance(strict_rows[0][-1],bytes), 'BLOB must remain bytes'
        raw_rows=query(fixture,"SELECT id,txt FROM fixture WHERE grp='invalid' ORDER BY id",'bytes',deadline)
        decoded_rows=query(fixture,"SELECT id,txt FROM fixture WHERE grp='invalid' ORDER BY id",'surrogateescape',deadline)
        assert len(raw_rows)==len(invalid_values)==len(decoded_rows)
        roundtrips=[]
        for (row_id,raw),(decoded_id,value),expected_raw in zip(raw_rows,decoded_rows,invalid_values,strict=True):
            assert row_id==decoded_id and raw==expected_raw and isinstance(value,str)
            assert value.encode('utf-8',errors='surrogateescape')==raw
            roundtrips.append({'fixture_id':row_id,'byte_length':len(raw),'round_trip_equal':True})
        strict_invalid = None
        try:
            query(fixture,'SELECT txt FROM fixture WHERE id=101','strict',deadline)
        except UnicodeDecodeError as error:
            strict_invalid={'status':'decode_error','error':str(error)}
        assert strict_invalid is not None, 'Strict protocol must still reject invalid UTF-8'
        first,second=invalid_values[:2]
        collision={'ignore_collides':first.decode('utf-8','ignore')==second.decode('utf-8','ignore'),
            'replace_collides':first.decode('utf-8','replace')==second.decode('utf-8','replace'),
            'surrogateescape_keeps_distinct':first.decode('utf-8','surrogateescape')!=second.decode('utf-8','surrogateescape')}
        assert all(collision.values())
        comparisons, invalid_comparisons=comparator_checks(module,fixture,deadline)
        first_bad=first_bad_original_value(db,GOLD_SQL,deadline)
        assert digest(fixture)==fixture_before, 'Read-only fixture checks changed database bytes'
        assert not any(name in sys.modules for name in ('torch','transformers','peft','bitsandbytes'))
        report.update(status='pass', valid_utf8_row_values_and_types_identical=True, valid_utf8_fixture_rows=len(strict_rows),
            blob_bytes_preserved=True, invalid_byte_roundtrips=roundtrips, collision_demo=collision,
            strict_invalid_fixture_result=strict_invalid, valid_utf8_comparator_cases=comparisons,
            invalid_byte_oracle_cases=invalid_comparisons, original_first_bad_value=first_bad,
            fixture_path=str(fixture.relative_to(ROOT)).replace('\\','/'),fixture_sha256=fixture_before,
            comparator_commit=provenance['commit'], comparator_source_sha256=provenance['source_files']['exec_eval.py']['sha256'],
            rule_unchanged_claim='The pinned result_eq implementation is reused unchanged on synthetic returned values.',
            protocol_boundary='surrogateescape extends SQLite TEXT decoding to bytes rejected by strict. It is an execution compatibility revision, not proof of equivalence to the original strict final scoring protocol.',
            invalid_utf8_equivalent_to_original_strict_protocol=False,
            final_scoring_or_worker_change_requires_separate_authorization=True)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        report['database_sha256_before']=db_before
        report['database_sha256_after']=digest(db)
        report['database_bytes_unchanged']=report['database_sha256_before']==report['database_sha256_after']
        report['source_sha256_before']=source_before
        report['source_sha256_after']={str(path.relative_to(ROOT)).replace('\\','/'):digest(path) for path in source_paths}
        report['protected_sources_unchanged']=report['source_sha256_after']==source_before
        report['seconds']=time.perf_counter()-started
        if not report['database_bytes_unchanged'] or not report['protected_sources_unchanged'] or report['seconds']>30:
            report['status']='failed'; report['final_validation_error']='Preservation or 30-second budget failed'
        write_new(output/'result.json',report)
    print(json.dumps({'status':report['status'],'seconds':report['seconds'],
        'database_bytes_unchanged':report['database_bytes_unchanged'],
        'protocol_equivalence_to_old_strict_claimed':False,'error':report.get('error')},ensure_ascii=True),flush=True)
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    raise SystemExit(run(parser.parse_args().output_dir))
