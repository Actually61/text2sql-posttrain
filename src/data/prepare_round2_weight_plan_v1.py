"""Prepare a training-only loss-weight proposal. No ML imports or SQL execution."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / 'data/processed/spider_gold_v1/train_clean.jsonl'
INVENTORY = ROOT / 'data/processed/spider_round2_train_inventory_v1'
SCHEDULE = ROOT / 'data/processed/spider_sft_round1_plan_v1/epoch_0.json'
DEST = ROOT / 'data/processed/spider_round2_weight_plan_v1'
CONFIG = ROOT / 'configs/sft_round2_weight_proposal_v1.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def write_rows(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    if DEST.exists() or CONFIG.exists():
        raise FileExistsError('Preserve existing proposal; refuse overwrite')
    inventory = read_json(INVENTORY / 'inventory.json')
    features_path = INVENTORY / 'train_structure_features.jsonl'
    assert digest(TRAIN) == inventory['source_sha256'][str(TRAIN.relative_to(ROOT))]
    assert digest(SCHEDULE) == inventory['source_sha256'][str(SCHEDULE.relative_to(ROOT))]
    assert digest(features_path) == inventory['feature_sha256']
    train = [json.loads(line) for line in TRAIN.read_text(encoding='utf-8').splitlines()]
    features = [json.loads(line) for line in features_path.read_text(encoding='utf-8').splitlines()]
    plan = read_json(SCHEDULE)
    by_id = {row['source_index']: row for row in features}
    assert len(train) == len(features) == len(by_id) == 6364
    assert {row['source_index'] for row in train} == set(by_id)
    assert all(row['project_split'] == 'train' for row in train + features)
    assert all(row['source_file'] == 'data/raw/spider/train_spider.json' for row in train)
    assert all(row['db_id'] == by_id[row['source_index']]['db_id'] for row in train)
    validation_dbs = set(read_json(ROOT / 'configs/baseline_full.json')['expected_database_ids'])
    train_dbs = {row['db_id'] for row in train}
    assert len(train_dbs) == 126 and len(validation_dbs) == 14 and not train_dbs & validation_dbs

    # A is the sole active intervention. B is an inactive review pool.
    focus_a = {i for i, row in by_id.items() if row['multi_table_join']}
    focus_b = {i for i, row in by_id.items() if row['has_string_filter_literal']}
    weights = {i: (2 if i in focus_a else 1) for i in by_id}
    assert len(focus_a) == 2528 and len(focus_b) == 2189
    ordered = [i for group in plan['groups'] for i in group['source_indices']]
    assert len(ordered) == len(set(ordered)) == 6364 and set(ordered) == set(by_id)
    assert len(plan['groups']) == 796 and len(plan['groups'][-1]['source_indices']) == 4
    groups = []
    for group in plan['groups']:
        ids = group['source_indices']
        counts = [by_id[i]['supervised_token_count'] for i in ids]
        assert counts == group['target_counts']
        assert sum(counts) == group['group_target_tokens']
        group_weights = [weights[i] for i in ids]
        denominator = sum(w * n for w, n in zip(group_weights, counts))
        kind = 'mixed' if len(set(group_weights)) > 1 else ('all_focus' if group_weights[0] == 2 else 'all_other')
        groups.append({
            'update': group['update'], 'source_indices': ids,
            'target_counts': counts, 'loss_weights': group_weights,
            'unweighted_target_tokens': sum(counts),
            'weighted_target_denominator': denominator, 'composition': kind,
            'micro_mean_loss_coefficients': [w * n / denominator for w, n in zip(group_weights, counts)],
        })
        assert abs(sum(groups[-1]['micro_mean_loss_coefficients']) - 1) < 1e-12
    total_tokens = sum(row['supervised_token_count'] for row in features)
    focus_tokens = sum(by_id[i]['supervised_token_count'] for i in focus_a)
    weighted_tokens = sum(weights[i] * by_id[i]['supervised_token_count'] for i in by_id)
    # This is an arithmetic audit only: no loss, gradients or updates are computed.
    assert total_tokens == 206073 and focus_tokens == 138930
    assert weighted_tokens == sum(row['weighted_target_denominator'] for row in groups)

    DEST.mkdir(parents=True, exist_ok=False)
    write_rows(DEST / 'train_loss_weights.jsonl', [
        {'source_index': row['source_index'], 'db_id': row['db_id'], 'project_split': 'train',
         'loss_weight': weights[row['source_index']], 'focus_a': row['source_index'] in focus_a}
        for row in train
    ])
    # These are original, unmodified clean training rows, not new pseudo-labels.
    write_rows(DEST / 'focus_a_original_train.jsonl', [row for row in train if row['source_index'] in focus_a])
    write_rows(DEST / 'focus_b_inactive_original_train.jsonl', [row for row in train if row['source_index'] in focus_b])
    write_json(DEST / 'weighted_groups.json', {'source_schedule_sha256': digest(SCHEDULE), 'groups': groups})
    sources = (TRAIN, features_path, INVENTORY / 'inventory.json', SCHEDULE,
               ROOT / 'configs/baseline_full.json', ROOT / 'configs/sft_round1.json',
               ROOT / 'configs/sft_round1_resource_v3.json', Path(__file__))
    manifest = {
        'status': 'static_checks_passed; weighted-gradient equivalence and trainer integration NOT performed',
        'examples': 6364, 'databases': 126, 'updates_planned': 796,
        'focus_a_examples': len(focus_a), 'focus_b_inactive_examples': len(focus_b),
        'a_b_overlap_examples': len(focus_a & focus_b),
        'answer_tokens_total': total_tokens, 'focus_a_answer_tokens': focus_tokens,
        'focus_a_raw_answer_token_share': focus_tokens / total_tokens,
        'weighted_token_mass_total': weighted_tokens,
        'focus_a_weighted_token_mass_share': 2 * focus_tokens / weighted_tokens,
        'mass_share_caveat': 'Corpus mass only, NOT global gradient contribution: normalization is per accumulation group.',
        'group_composition_counts': dict(Counter(row['composition'] for row in groups)),
        'sequence_length_max': max(row['sequence_length'] for row in features),
        'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sources},
        'output_sha256': {str(path.relative_to(ROOT)): digest(path) for path in DEST.iterdir()},
        'selection': {'A': 'Any training gold AST block has >=2 direct table_unit operands; proxy, not correctness label.',
                      'B_inactive': 'Training gold WHERE/HAVING contains string literal; proxy, not a known error label.'},
        'changed_prompt_or_gold_or_tokens': False, 'validation_rows_added_to_training': False,
        'validation_question_or_answer_read_by_builder': False,
        'validation_metadata_read': 'DB ID list only, to enforce disjoint databases',
        'final_read': False, 'model_loaded': False, 'sql_executed': False,
        'training_performed': False, 'evaluation_performed': False,
    }
    write_json(DEST / 'manifest.json', manifest)
    round1 = read_json(ROOT / 'configs/sft_round1.json')
    resources = read_json(ROOT / 'configs/sft_round1_resource_v3.json')
    proposal = {
        'version': 'sft_round2_weight_proposal_v1', 'status': 'PROPOSAL_ONLY_NOT_A_RUN_CONFIG',
        'gpu_training_enabled': False, 'evaluation_enabled': False,
        'single_active_change': {'focus_a_loss_weight': 2, 'other_loss_weight': 1,
                                 'round1_focus_a_loss_weight': 1, 'focus_b_intervention': False},
        'seed': 42, 'model': deepcopy(round1['model']),
        'start': 'Same original frozen base and fresh LoRA; DO NOT continue round1 adapter.',
        'training': deepcopy(round1['training']),
        'normalization': 'For each group: sum_i(w_i * token_loss_sum_i) / sum_i(w_i * valid_answer_tokens_i).',
        'micro_backward_scale': 'micro_mean_loss * weight_i * valid_answer_tokens_i / weighted_group_denominator; no /8',
        'train_path': round1['data']['train'], 'frozen_schedule': str(SCHEDULE.relative_to(ROOT)),
        'weights': str((DEST / 'train_loss_weights.jsonl').relative_to(ROOT)),
        'weighted_groups': str((DEST / 'weighted_groups.json').relative_to(ROOT)),
        'manifest': str((DEST / 'manifest.json').relative_to(ROOT)),
        'unchanged': ['6364 rows and labels', 'assistant-only mask and EOS', 'prompt/tokenizer',
                      'all sequence lengths; no truncation or deletion', '796 original groups and order',
                      'LoRA/optimizer/LR/clipping/seed', 'one epoch; no early checkpoint selection'],
        'save_after_updates': [200, 400, 600, 796],
        'checkpoint_requirement': 'Bind weights hash and weighted schedule hash; restore optimizer/LR/cursor/RNG exactly.',
        'future_resource_budget': {key: resources[key] for key in (
            'minimum_free_mib', 'start_headroom_mib', 'start_resource_samples',
            'start_resource_interval_seconds', 'minimum_disk_gib', 'allocated_limit_mib',
            'reserved_limit_mib', 'total_gpu_window_seconds', 'limits')},
        'resource_note': 'Copied from successful round1; must recheck live resources before any future GPU launch.',
        'future_comparison': {
            'internal_validation_count': 625, 'round1_ex': {'correct': 457, 'total': 625},
            'round1_executable': {'correct': 556, 'total': 625},
            'primary_metric': 'EX on all 625, no exclusions or metric rule changes',
            'decision': 'Consider a candidate only if EX > 457/625 and executable >= 556/625; report all gains/regressions.',
            'secondary': ['paired transition vs round1', 'difficulty and per-database rates',
                          '34 old regressions and all 168 old failures, without selective score replacement'],
            'protocol': 'Future explicit authorization: evaluate update796 once; same frozen prompt/tokenizer/base/NF4, greedy1, max_new_tokens256, EX.',
            'no_claim': 'One split/seed does not establish generalization or statistical significance; no lambda grid search.',
            'final_set': 'sealed; not accessed',
        },
        'next_gate': 'CPU-only weighted accumulation equivalence, lambda1/all2 invariance, unchanged data/order and checkpoint hash guard; no GPU.',
        'hypothesis': 'Increasing relative emphasis on multi-table training examples may improve field binding/join behavior; unproven.',
        'limitations': ['A is broad and does not cover every schema-linking error.',
                        'Weights cancel in homogeneous groups; effect occurs only in mixed groups.',
                        'B is prepared but inactive; schema lacks cell values and no validation literals are introduced.',
                        'Weighted training loss is not numerically comparable to round1 unweighted training loss.'],
    }
    proposal['training']['normalization'] = proposal['normalization']
    write_json(CONFIG, proposal)
    print(json.dumps({key: manifest[key] for key in ('status', 'examples', 'focus_a_examples',
        'focus_b_inactive_examples', 'a_b_overlap_examples', 'group_composition_counts',
        'focus_a_raw_answer_token_share', 'focus_a_weighted_token_mass_share')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
