"""Package existing scores and manual SQL/schema review; never execute SQL or models."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/sft_round2_review_v1'
ANALYSIS = ROOT / 'reports/sft_round2_eval_v1_analysis.json'
PAIRS = ROOT / 'reports/sft_round2_eval_v1_comparisons.jsonl'
PROMPTS = ROOT / 'outputs/preds/sft_round2_eval_v1/prompts.jsonl'
CATEGORIES = {'表列归属', '连接路径', '去重', '聚合分组', '筛选取值', '排序TopK', '集合量词', '输出列', '参考口径待复核'}
FIELDS = ('primary_category', 'secondary_categories', 'schema_evidence',
          'observed_sql_change', 'verified_explanation', 'uncertainty',
          'semantic_assessment', 'confidence')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def capture():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [
        ANALYSIS, PAIRS, PROMPTS,
        ROOT / 'reports/sft_error_review_v1/cases.json',
        ROOT / 'reports/sft_error_review_v1_walkthrough.md',
        ROOT / 'reports/spider_gold_manifest.json',
        ROOT / 'reports/spider_split_manifest.json',
        ROOT / 'reports/sft_data_manifest.json',
        ROOT / 'configs/sft_round2_weight_proposal_v1.json',
        ROOT / 'configs/sft_round2_formal_v1.json',
        ROOT / 'configs/evaluation.json',
        ROOT / 'configs/baseline_full.json',
        ROOT / 'reports/sft_round1_resource_v3_analysis.json',
        ROOT / 'reports/sft_round1_resource_v3_result.json',
        ROOT / 'reports/sft_round2_formal_v1_analysis.json',
        ROOT / 'reports/sft_round2_formal_v1_result.json',
    ]
    for stem in ('baseline_full', 'sft_round1_eval_v2', 'sft_round2_eval_v1'):
        inputs += [ROOT / f'reports/{stem}_generation.json', ROOT / f'reports/{stem}_eval.json']
    analysis = read(ANALYSIS)
    inputs += [ROOT / model['prediction_path'] for model in analysis['models'].values()]
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in inputs}
    write_json(OUT / 'input_snapshot.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': hashes,
        'scope': 'Existing internal-validation artifacts and aggregate training/split metadata only; no final examples or database files.'})
    print(json.dumps({'status': 'captured', 'input_files': len(hashes)}))


def text(value):
    return '；'.join(value) if isinstance(value, list) else str(value)


def build():
    snapshot = read(OUT / 'input_snapshot.json')
    for filename, digest in snapshot['source_sha256'].items():
        require(sha(ROOT / filename) == digest, f'Existing source changed: {filename}')
    analysis = read(ANALYSIS)
    require(sha(PAIRS) == analysis['paired_examples_sha256'], 'Paired results hash mismatch')
    rows = lines(PAIRS)
    require(len(rows) == len({r['source_index'] for r in rows}) == 625, 'Expected 625 unique original records')
    for name, expected in analysis['models'].items():
        require(sha(ROOT / expected['prediction_path']) == expected['prediction_sha256'], f'Prediction hash mismatch: {name}')
        saved = analysis['overall'][name]['evaluation']
        require(sum(r['models'][name]['ex_match'] is True for r in rows) == saved['execution_correct'], f'EX flag count mismatch: {name}')
        require(sum(r['models'][name]['executable'] is True for r in rows) == saved['executable_predictions'], f'Executable flag count mismatch: {name}')
    selected = [r for r in rows if r['round1_to_round2_ex'] in ('improved', 'regressed')]
    require(len(selected) == 33, 'Expected 33 changed scores')
    annotations = []
    for kind, count in (('improved', 16), ('regressed', 17)):
        group = read(OUT / f'{kind}_annotations.json')
        expected_ids = {r['source_index'] for r in selected if r['round1_to_round2_ex'] == kind}
        require(len(group) == count and {r['source_index'] for r in group} == expected_ids, f'Annotation coverage mismatch: {kind}')
        annotations.extend(group)
    by_id = {r['source_index']: r for r in annotations}
    prompts = {r['source_index']: r for r in lines(PROMPTS)}
    cases = []
    for row in selected:
        index = row['source_index']
        annotation = by_id[index]
        require(all(field in annotation for field in FIELDS), f'Missing annotation fields: {index}')
        require(annotation['primary_category'] in CATEGORIES, f'Unknown category: {index}')
        require(all(annotation[field] for field in FIELDS if field != 'secondary_categories'), f'Empty annotation: {index}')
        prompt = prompts[index]
        require(prompt['db_id'] == row['db_id'], f'Schema database mismatch: {index}')
        require(row['question'] in prompt['rendered_prompt'], f'Question not in original prompt: {index}')
        cases.append({**row, 'rendered_prompt_with_schema': prompt['rendered_prompt'],
                      'manual_review': annotation,
                      'semantic_assessment_applies_to': 'round2' if row['round1_to_round2_ex'] == 'improved' else 'round1'})
    counts = {kind: dict(Counter(c['manual_review']['primary_category'] for c in cases if c['round1_to_round2_ex'] == kind))
              for kind in ('improved', 'regressed')}
    exact_groups = {}
    for c in cases:
        exact_groups.setdefault((c['db_id'], c['gold_sql']), []).append(c['source_index'])
    summary = {
        'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Static review of all 16 improved and 17 regressed saved EX outcomes; no new scoring.',
        'count': 33, 'primary_category_counts': counts,
        'category_policy': 'One manually assigned primary description per case; secondary categories may overlap. Not independent or proven training causes.',
        'per_database': {kind: dict(Counter(c['db_id'] for c in cases if c['round1_to_round2_ex'] == kind)) for kind in ('improved', 'regressed')},
        'exact_db_gold_groups': len(exact_groups),
        'repeated_exact_db_gold_groups': [v for v in exact_groups.values() if len(v) > 1],
        'group_limit': 'Exact database plus gold SQL text groups only; not a SQL semantic equivalence test.',
        'all_625_metrics_unchanged': {name: analysis['overall'][name]['evaluation'] for name in ('base', 'round1', 'round2')},
        'round1_to_round2_ex_transitions': analysis['comparisons']['round1_to_round2']['overall']['ex_transitions'],
        'predeclared_point_estimate_gate': analysis['predeclared_point_estimate_gate'],
        'no_new_generation': True, 'no_new_training': True, 'no_sql_execution': True,
        'no_final_examples_read': True, 'source_sha256': snapshot['source_sha256'],
    }
    appendix = [
        '# 第二轮 SFT 的 33 条得分变化证据', '',
        '只读复盘已有记录。EX 改善或退化描述原评分变化，不保证语义正确性同步改变。每题一个人工主类用于概括可见 SQL 差异；次类可重叠，不解释训练层面的唯一根因。', '',
        '“通过侧语义判断”在改善题审查第二轮，在退化题审查第一轮。未运行任何 SQL，也未核查实际数据库值；参考争议保留原评分和 625 题分母。详细 schema 来自生成时保存的原提示词。', '',
    ]
    for c in cases:
        a = c['manual_review']
        kind = '改善' if c['round1_to_round2_ex'] == 'improved' else '退化'
        appendix += [f"## {c['source_index']} {kind} {c['db_id']}", '',
                     f"问题：{c['question']}", '',
                     f"主类：{a['primary_category']}；次类：{text(a['secondary_categories']) or '无'}。", '',
                     f"Schema 依据：{text(a['schema_evidence'])}", '',
                     f"可见变化：{text(a['observed_sql_change'])}", '',
                     f"核实解释：{text(a['verified_explanation'])}", '',
                     f"通过侧语义判断：{a['semantic_assessment']}；置信描述：{a['confidence']}。", '',
                     f"限制或待核实：{text(a['uncertainty'])}", '']
        for name, label in (('base', '基座'), ('round1', '第一轮'), ('round2', '第二轮')):
            detail = c['models'][name]
            appendix += [f"{label}：可执行={detail['executable']}，EX={detail['ex_match']}，原结果行数={detail['result_rows']}。", '',
                         '```sql', detail['sql'], '```', '']
            if detail['first_execution_error']:
                appendix += [f"原错误：`{detail['first_execution_error']}`", '']
        appendix += ['原参考 SQL：', '', '```sql', c['gold_sql'], '```', '',
                     '<details><summary>生成时的完整 schema 与问题</summary>', '',
                     '```text', c['rendered_prompt_with_schema'], '```', '', '</details>', '']
    write_json(OUT / 'cases.json', cases)
    write_json(OUT / 'summary.json', summary)
    with (OUT / '33_cases.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(appendix))
    for filename, digest in snapshot['source_sha256'].items():
        require(sha(ROOT / filename) == digest, f'Source changed during packaging: {filename}')
    artifact_names = ('cases.json', 'summary.json', '33_cases.md', 'improved_annotations.json',
                      'regressed_annotations.json', '01_实验结论与核心教程.md', '02_简历项目初稿.md')
    output_hashes = {name: sha(OUT / name) for name in artifact_names}
    write_json(OUT / 'checks.json', {
        'status': 'pass', 'source_file_count': len(snapshot['source_sha256']),
        'source_files_unchanged': True, 'all_625_saved_flags_consistent': True,
        'review_coverage': {'improved': 16, 'regressed': 17, 'duplicates': 0},
        'question_and_schema_binding': True, 'sql_executed': False, 'model_loaded': False,
        'training_performed': False, 'final_examples_read': False,
        'builder_sha256': sha(Path(__file__)), 'artifact_sha256': output_hashes})
    print(json.dumps({'status': 'pass', 'cases': len(cases), 'primary_categories': counts,
                      'exact_db_gold_groups': len(exact_groups), 'source_files_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('capture', 'build'))
    args = parser.parse_args()
    capture() if args.mode == 'capture' else build()
