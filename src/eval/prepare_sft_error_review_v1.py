"""Export existing errors for manual schema/SQL review; never execute SQL/models."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'reports/sft_error_review_v1'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    report_path = ROOT / 'reports/sft_round1_eval_v2_analysis.json'
    schemas_path = ROOT / 'data/processed/spider_inputs_v1/schemas.jsonl'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['status'] == 'pass' and report['final_examples_read'] is False
    pairs = report['paired_examples']
    assert len(pairs) == 625 and len({r['source_index'] for r in pairs}) == 625
    errors = [r for r in pairs if r['sft']['ex_match'] is not True]
    regressions = [r for r in errors if r['ex_transition'] == 'regressed']
    other = [r for r in errors if r['ex_transition'] != 'regressed']
    assert len(errors) == 168 and len(regressions) == 34 and len(other) == 134
    schemas = {r['db_id']: r['schema_text'] for r in
               map(json.loads, schemas_path.read_text(encoding='utf-8').splitlines())}
    DEST.mkdir(exist_ok=False)
    groups = {'regressions': regressions, 'other_a': other[:67], 'other_b': other[67:]}
    for name, rows in groups.items():
        payload = {'scope': 'Existing internal-validation errors only; manual static review, no SQL execution',
            'source_sha256': digest(report_path), 'schemas': {db: schemas[db] for db in sorted({r['db_id'] for r in rows})},
            'examples': rows}
        (DEST / f'{name}_packet.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    manifest = {'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (report_path, schemas_path)},
        'groups': {name: [r['source_index'] for r in rows] for name, rows in groups.items()},
        'error_count': 168, 'regression_count': 34, 'sql_executed': False, 'model_loaded': False,
        'training_performed': False, 'final_examples_read': False}
    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'errors':len(errors),'regressions':len(regressions),'packets':list(groups)}))

if __name__ == '__main__':
    main()
