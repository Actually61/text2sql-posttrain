"""Read-back provenance and arithmetic checks only; no model, SQL or optimizer."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'data/processed/spider_round2_weight_plan_v1'
REPORT = ROOT / 'reports/sft_error_review_v1/static_artifact_checks.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    assert not REPORT.exists(), 'Preserve prior check'
    manifest = read(DEST / 'manifest.json')
    inventory = read(ROOT / 'data/processed/spider_round2_train_inventory_v1/inventory.json')
    review = read(ROOT / 'reports/sft_error_review_v1/summary.json')
    checked = {}
    for hashes in (manifest['source_sha256'], manifest['output_sha256'],
                   inventory['source_sha256'], review['source_sha256']):
        for path, expected in hashes.items():
            with (ROOT / path).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert actual == expected, path
            checked[path] = actual
    train = rows(ROOT / 'data/processed/spider_gold_v1/train_clean.jsonl')
    by_id = {row['source_index']: row for row in train}
    features = rows(ROOT / 'data/processed/spider_round2_train_inventory_v1/train_structure_features.jsonl')
    a = {row['source_index'] for row in features if row['multi_table_join']}
    b = {row['source_index'] for row in features if row['has_string_filter_literal']}
    for filename, selected in (('focus_a_original_train.jsonl', a), ('focus_b_inactive_original_train.jsonl', b)):
        actual = rows(DEST / filename)
        assert actual == [row for row in train if row['source_index'] in selected]
        assert all(row == by_id[row['source_index']] for row in actual)
    weights = rows(DEST / 'train_loss_weights.jsonl')
    assert [row['source_index'] for row in weights] == [row['source_index'] for row in train]
    assert all(row['loss_weight'] == (2 if row['source_index'] in a else 1) for row in weights)
    original = read(ROOT / 'data/processed/spider_sft_round1_plan_v1/epoch_0.json')['groups']
    weighted = read(DEST / 'weighted_groups.json')['groups']
    assert len(original) == len(weighted) == 796
    for old, new in zip(original, weighted):
        assert old['source_indices'] == new['source_indices'] and old['target_counts'] == new['target_counts']
        assert new['weighted_target_denominator'] == sum(w * n for w, n in zip(new['loss_weights'], new['target_counts']))
        assert abs(sum(new['micro_mean_loss_coefficients']) - 1) < 1e-12
        if new['composition'] != 'mixed':
            assert all(abs(c - n / old['group_target_tokens']) < 1e-12
                       for c, n in zip(new['micro_mean_loss_coefficients'], old['target_counts']))
    config = read(ROOT / 'configs/sft_round2_weight_proposal_v1.json')
    assert not config['gpu_training_enabled'] and not config['evaluation_enabled']
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'sqlite3' not in sys.modules
    result = {
        'status': 'pass', 'verified_hashes': checked, 'train_rows': len(train),
        'a_rows_equal_original_gold': len(a), 'b_rows_equal_original_gold': len(b),
        'weights_rows': len(weights), 'original_order_and_all_796_groups_unchanged': True,
        'valid_weighted_denominators': True, 'homogeneous_group_weights_cancel': True,
        'candidate_pools_are_auxiliary_not_appended_to_training': True,
        'no_validation_examples_or_answers_added_to_training': True,
        'weighted_gradient_equivalence': 'not tested; proposed next step',
        'cuda_initialized': False, 'model_loaded': False, 'sql_executed': False,
        'optimizer_updates': 0, 'evaluation_performed': False, 'final_read': False,
    }
    with REPORT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'pass', 'verified_hash_count': len(checked), 'train_rows': len(train),
                      'a_rows': len(a), 'b_rows': len(b), 'updates': 0, 'final_read': False}))


if __name__ == '__main__':
    main()
