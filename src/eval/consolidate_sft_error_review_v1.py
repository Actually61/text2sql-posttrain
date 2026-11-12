"""Consolidate static human review; never execute SQL or rerun evaluation."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'reports/sft_error_review_v1'
NAMES = ('regressions', 'other_a', 'other_b')
LABELS = {
    'table_column': '表列/语义属性选择', 'join': '关系连接/路径',
    'distinct': '去重', 'aggregation': '聚合/分组粒度', 'filter': '筛选/值映射',
    'ordering_limit': '排序/Top-K', 'set_operation': '集合/存在量词',
    'projection': '输出列完整性', 'reference_ambiguity': '题目与参考口径待复核',
    'unresolved': '未定位',
}
OVERRIDES = {
    4372: ('table_column', '描述值绑定到代码列；与4373采用相同口径，join保留为次类。'),
    4824: ('table_column', '输出飞机ID替代名称；与4821/4823采用相同口径，join保留为次类。'),
    2652: ('table_column', '输出cross-reference ID替代register ID；join保留为次类。'),
    2021: ('set_operation', '不存在银行公司被写成存在非银行公司；与6607采用相同口径。'),
    2022: ('set_operation', '不存在银行公司被写成存在非银行公司；与6607采用相同口径。'),
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def count(rows, field):
    return dict(Counter(row['review'][field] for row in rows))


def families(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['db_id'], row['gold_sql'])].append(row['source_index'])
    return {'exact_db_and_gold_string_groups': len(groups),
            'definition': 'Exact original SQL string plus db_id, not normalized SQL or semantic equivalence.',
            'repeated_groups': [{'db_id': db, 'gold_sql': gold, 'source_indices': ids}
                                for (db, gold), ids in groups.items() if len(ids) > 1]}


def main():
    outputs = ('cases.json', 'summary.json', '34_regressions.md')
    assert not any((DEST / name).exists() for name in outputs), 'Do not overwrite prior review'
    manifest = load(DEST / 'manifest.json')
    for path, expected in manifest['source_sha256'].items():
        assert digest(ROOT / path) == expected, f'Frozen input changed: {path}'
    rows, schemas, sources = [], {}, [DEST / 'manifest.json', Path(__file__)]
    adjustments = []
    for name in NAMES:
        packet_path = DEST / f'{name}_packet.json'
        annotation_path = DEST / f'{name}_annotations.json'
        packet, annotations = load(packet_path), load(annotation_path)
        sources.extend((packet_path, annotation_path))
        by_id = {row['source_index']: row for row in annotations}
        assert len(by_id) == len(annotations)
        assert set(by_id) == set(manifest['groups'][name])
        assert {row['source_index'] for row in packet['examples']} == set(by_id)
        for db, text in packet['schemas'].items():
            assert db not in schemas or schemas[db] == text
            schemas[db] = text
        for row in packet['examples']:
            assert row['sft']['ex_match'] is not True
            annotation = deepcopy(by_id[row['source_index']])
            for key in ('primary_category', 'secondary_categories', 'certainty', 'facts',
                        'interpretation', 'hypothesis', 'reference_caveat'):
                assert key in annotation
            assert annotation['primary_category'] in LABELS
            assert all(item in LABELS for item in annotation['secondary_categories'])
            assert annotation['certainty'] in ('confirmed', 'needs_review') and annotation['facts']
            if row['source_index'] in OVERRIDES:
                category, reason = OVERRIDES[row['source_index']]
                before = annotation['primary_category']
                annotation['primary_category'] = category
                annotation['secondary_categories'] = sorted((set(annotation['secondary_categories']) | {before}) - {category})
                adjustments.append({'source_index': row['source_index'], 'before': before,
                                    'after': category, 'reason': reason})
            rows.append({**row, 'review': annotation})
    rows.sort(key=lambda row: row['source_index'])
    assert len(rows) == len({row['source_index'] for row in rows}) == 168
    regressions = [row for row in rows if row['ex_transition'] == 'regressed']
    assert len(regressions) == 34 and all(row['baseline']['ex_match'] is True for row in regressions)
    schema_family = [row for row in rows if row['review']['primary_category'] in ('table_column', 'join')]
    summary = {
        'method': 'Question + frozen schema + existing gold/base/SFT SQL and prior execution records; static review only.',
        'scope': {'existing_sft_errors': 168, 'metric_regressions': 34,
                  'other_sft_errors': 134, 'new_generations': 0, 'new_sql_executions': 0,
                  'new_training_updates': 0, 'new_evaluations': 0, 'final_read': False},
        'frozen_scores': {'baseline_ex': '372/625 = 59.52%', 'round1_ex': '457/625 = 73.12%',
                          'round1_executable': '556/625 = 88.96%', 'scores_or_denominators_changed': False},
        'all_primary_counts': count(rows, 'primary_category'),
        'regression_primary_counts': count(regressions, 'primary_category'),
        'all_certainty_counts': count(rows, 'certainty'),
        'regression_certainty_counts': count(regressions, 'certainty'),
        'all_secondary_counts_nonexclusive': dict(Counter(tag for row in rows for tag in row['review']['secondary_categories'])),
        'existing_outcomes': dict(Counter(row['sft']['outcome'] for row in rows)),
        'existing_execution_error_types': dict(Counter(row['sft']['first_error_category'] for row in rows if not row['sft']['executable'])),
        'database_counts': dict(Counter(row['db_id'] for row in rows)),
        'schema_family': {'count': len(schema_family), 'certainty_counts': count(schema_family, 'certainty'),
                          'database_counts': dict(Counter(row['db_id'] for row in schema_family))},
        'needs_review_ids': [row['source_index'] for row in rows if row['review']['certainty'] == 'needs_review'],
        'primary_reference_ambiguity_ids': [row['source_index'] for row in rows if row['review']['primary_category'] == 'reference_ambiguity'],
        'any_reference_caveat_count': sum(bool(row['review']['reference_caveat']) for row in rows),
        'exact_gold_groups_all_errors': families(rows), 'exact_gold_groups_regressions': families(regressions),
        'category_labels': LABELS, 'category_adjustments_preserving_raw_annotations': adjustments,
        'rubric': [
            'One primary class per question; secondary classes overlap. Primary is a review convention, not causal proof.',
            'Wrong semantic attribute/id or a nonexistent table/column: table_column; missing a necessary relation/path: join.',
            'Missing lookup can overlap both; wrong requested attribute/id takes table_column, join stays secondary.',
            'Entity-level EXISTS/NOT EXISTS versus row predicate belongs to set_operation even without a set keyword.',
            'confirmed denotes observable structural/logical differences, never a proven training mechanism.',
            'needs_review denotes ambiguous intent, reference conflict, or competing explanations requiring new evidence.',
            'Gold/question caveats are retained separately; no scores or questions are removed.',
        ],
        'priority_families': [
            {'name': 'table_column + join', 'scope': '97/168, including 17/34 regressions',
             'plan': 'Training-only multi-table example loss weight 2; broad proxy, unproven remedy.'},
            {'name': 'filter', 'scope': '33/168, including 10/34 regressions',
             'plan': 'Prepare string-filter training pool, inactive in round2; do not introduce validation literals.'},
        ],
        'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sources},
    }
    assert summary['all_primary_counts'] == {'reference_ambiguity': 18, 'table_column': 41,
        'distinct': 1, 'join': 56, 'projection': 3, 'set_operation': 5, 'filter': 33,
        'aggregation': 10, 'ordering_limit': 1}
    assert summary['existing_outcomes'] == {'executable_mismatch': 99, 'execution_failed': 69}
    save(DEST / 'cases.json', {'scope': summary['scope'], 'schemas_by_db_id': schemas, 'cases': rows})
    save(DEST / 'summary.json', summary)
    text = ['# 34条EX退化：静态证据附录', '',
            '本附录复用已有预测与执行记录，没有执行SQL。完整schema见同目录 cases.json 的 schemas_by_db_id。',
            'EX退化只表示本次数据库结果由通过变为未通过；不自动证明所有语义都退化。主类是人工归因口径。', '']
    for row in regressions:
        review = row['review']
        text += [f"## source_index {row['source_index']} · {row['db_id']}", '',
                 f"问题：{row['question']}", '',
                 f"主类：{LABELS[review['primary_category']]}；证据状态：{review['certainty']}。", '',
                 '**参考 SQL**', '```sql', row['gold_sql'], '```',
                 '**基座 SQL（原EX通过）**', '```sql', row['baseline']['sql'], '```',
                 '**首轮 SFT SQL**', '```sql', row['sft']['sql'], '```', '',
                 f"原记录：{row['sft']['outcome']}；执行错误：{row['sft']['first_execution_error'] or '无'}。", '',
                 '**可见事实（含字段归属/连接证据）**', '']
        text += [f'- {fact}' for fact in review['facts']]
        text += ['', f"归因：{review['interpretation']}", '',
                 f"未验证机制：{review['hypothesis'] or '未提出额外机制假设。'}", '',
                 f"参考/证据边界：{review['reference_caveat'] or '未发现额外参考冲突；仍不等于跨数据实例证明。'}", '']
    with (DEST / '34_regressions.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(text) + '\n')
    print(json.dumps({key: summary[key] for key in ('all_primary_counts', 'regression_primary_counts',
        'all_certainty_counts', 'existing_outcomes')}, ensure_ascii=False))
    print(json.dumps({'exact_gold_groups_errors': summary['exact_gold_groups_all_errors']['exact_db_and_gold_string_groups'],
                      'exact_gold_groups_regressions': summary['exact_gold_groups_regressions']['exact_db_and_gold_string_groups']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
