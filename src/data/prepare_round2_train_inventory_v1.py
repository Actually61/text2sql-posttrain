"""Static training-only feature inventory; no SQL execution or ML imports."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / 'data/processed/spider_gold_v1/train_clean.jsonl'
TOKENS = ROOT / 'data/processed/spider_sft_tokens_v1/train_tokens.jsonl'
SCHEDULE = ROOT / 'data/processed/spider_sft_round1_plan_v1/epoch_0.json'
DEST = ROOT / 'data/processed/spider_round2_train_inventory_v1'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)

def features(row):
    blocks = [node for node in walk(row['sql']) if isinstance(node, dict) and 'select' in node and 'from' in node]
    direct_counts = [sum(unit[0] == 'table_unit' for unit in block['from']['table_units']) for block in blocks]
    col_units = [node for node in walk(row['sql']) if isinstance(node, list) and len(node) == 3
                 and type(node[0]) is int and 0 <= node[0] <= 5 and type(node[1]) is int and type(node[2]) is bool]
    agg_selects = [unit[0] for block in blocks for unit in block['select'][1]]
    literals = []
    for block in blocks:
        for clause in ('where', 'having'):
            for cond in block[clause]:
                if isinstance(cond, list):
                    literals.extend(value for value in cond[3:5] if isinstance(value, str))
    return {
        'sql_block_count': len(blocks), 'max_direct_tables_in_block': max(direct_counts, default=0),
        'multi_table_join': any(n >= 2 for n in direct_counts),
        'three_plus_table_join': any(n >= 3 for n in direct_counts),
        'has_subquery_or_set_block': len(blocks) > 1,
        'has_distinct': any(block['select'][0] for block in blocks) or any(unit[2] for unit in col_units),
        'has_aggregate': any(agg > 0 for agg in agg_selects) or any(unit[0] > 0 for unit in col_units),
        'has_group_by': any(block['groupBy'] for block in blocks),
        'has_having': any(block['having'] for block in blocks),
        'has_where': any(block['where'] for block in blocks),
        'has_string_filter_literal': bool(literals), 'string_filter_literal_count': len(literals),
        'has_order_by': any(block['orderBy'] for block in blocks),
        'has_limit': any(block['limit'] is not None for block in blocks),
        'has_set_operation': any(block[key] is not None for block in blocks for key in ('union','intersect','except')),
        'question_mentions_distinct': bool(re.search(r'\b(distinct|different|unique)\b',row['question'],re.I)),
    }

def main():
    if DEST.exists():
        raise FileExistsError('Inventory already exists; do not overwrite')
    assert digest(TRAIN) == '3aa00c3f1f2056d4d3ca067c9083a17886a57861ab86ecee0450836caeeca024'
    plan = json.loads(SCHEDULE.read_text(encoding='utf-8'))
    assert digest(TOKENS) == plan['train_sha256']
    rows = [json.loads(line) for line in TRAIN.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len({r['source_index'] for r in rows}) == 6364
    assert all(r['project_split'] == 'train' and r['source_file'] == 'data/raw/spider/train_spider.json' for r in rows)
    train_dbs = {r['db_id'] for r in rows}
    validation_dbs = set(json.loads((ROOT/'configs/baseline_full.json').read_text(encoding='utf-8'))['expected_database_ids'])
    assert len(train_dbs) == 126 and len(validation_dbs) == 14 and not train_dbs & validation_dbs
    token_meta = {}
    with TOKENS.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            assert row['project_split'] == 'train'
            token_meta[row['source_index']] = {k:row[k] for k in ('db_id','sequence_length','supervised_token_count')}
    assert set(token_meta) == {r['source_index'] for r in rows}
    ordered = [i for group in plan['groups'] for i in group['source_indices']]
    assert len(ordered) == len(set(ordered)) == 6364 and set(ordered) == set(token_meta)
    records = []
    for row in rows:
        meta=token_meta[row['source_index']]
        assert meta['db_id'] == row['db_id']
        records.append({'source_index':row['source_index'],'project_split':'train',**meta,**features(row)})
    flags=[key for key,value in records[0].items() if type(value) is bool]
    report={'status':'pass','examples':6364,'databases':126,
        'flags':{key:{'examples':sum(r[key] for r in records),
                      'answer_tokens':sum(r['supervised_token_count'] for r in records if r[key])} for key in flags},
        'max_direct_tables_counts':dict(Counter(r['max_direct_tables_in_block'] for r in records)),
        'sequence_length_max':max(r['sequence_length'] for r in records),
        'answer_tokens_total':sum(r['supervised_token_count'] for r in records),
        'source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in (TRAIN,TOKENS,SCHEDULE,ROOT/'configs/baseline_full.json')},
        'selection_inputs':'Training gold SQL AST only; database membership metadata used solely for leakage guard',
        'validation_questions_or_answers_read':False,'final_examples_read':False,
        'sql_executed':False,'model_loaded':False,'training_performed':False}
    DEST.mkdir(parents=True,exist_ok=False)
    with (DEST/'train_structure_features.jsonl').open('x',encoding='utf-8') as stream:
        for row in records: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    report['feature_sha256']=digest(DEST/'train_structure_features.jsonl')
    (DEST/'inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__ == '__main__': main()
