"""Reuse the proven 32 full examples/order and attach frozen round-two weights."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'data/processed/spider_sft_round2_pilot_v1'
FULL = ROOT / 'data/processed/spider_sft_tokens_v1/train_tokens.jsonl'
OLD = ROOT / 'data/processed/spider_sft_resume_pilot_v1/selection.json'
WEIGHTS = ROOT / 'data/processed/spider_round2_weight_plan_v1/train_loss_weights.jsonl'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if DEST.exists():
        raise FileExistsError('Preserve existing selection; no overwrite')
    old = read(OLD)
    cpu = read(ROOT / 'reports/sft_round2_weight_cpu_v1_checks.json')
    assert cpu['status'] == 'pass' and cpu['cuda_initialized'] is False
    inventory = read(ROOT / 'data/processed/spider_round2_train_inventory_v1/inventory.json')
    assert digest(FULL) == inventory['source_sha256'][str(FULL.relative_to(ROOT))]
    weights = {r['source_index']: r for r in (json.loads(line) for line in WEIGHTS.read_text(encoding='utf-8').splitlines())}
    assert len(weights) == 6364
    ids = old['selected_source_indices']
    selected, max_length, max_answers, total = {}, 0, 0, 0
    with FULL.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            assert row['project_split'] == 'train'
            max_length = max(max_length, row['sequence_length'])
            max_answers = max(max_answers, row['supervised_token_count'])
            total += 1
            if row['source_index'] in ids:
                selected[row['source_index']] = row
    assert total == 6364 and len(ids) == len(set(ids)) == len(selected) == 32
    assert max(r['sequence_length'] for r in selected.values()) == max_length == 3761
    assert max(r['supervised_token_count'] for r in selected.values()) == max_answers == 202
    assert set(weights[i]['loss_weight'] for i in ids) == {1, 2}
    bounds = (512, 1024, 2048, 4096)
    buckets = Counter(str(next(b for b in bounds if selected[i]['sequence_length'] <= b)) for i in ids)
    assert buckets == {'512': 8, '1024': 8, '2048': 8, '4096': 8}
    groups = deepcopy(old['groups'])
    for group in groups:
        group_ids = group['source_indices']
        counts = [selected[i]['supervised_token_count'] for i in group_ids]
        assert counts == group['target_counts'] and sum(counts) == group['group_target_tokens']
        ws = [weights[i]['loss_weight'] for i in group_ids]
        denominator = sum(w * n for w, n in zip(ws, counts))
        group.update(loss_weights=ws, weighted_target_denominator=denominator,
                     unweighted_target_tokens=sum(counts),
                     micro_mean_loss_coefficients=[w*n/denominator for w,n in zip(ws,counts)],
                     composition='mixed' if len(set(ws)) > 1 else ('all_focus' if ws[0] == 2 else 'all_other'))
    assert [i for g in groups for i in g['source_indices']] == ids
    DEST.mkdir(parents=True, exist_ok=False)
    for name, rows in [('train_tokens.jsonl', [selected[i] for i in ids]), ('weights.jsonl', [weights[i] for i in ids])]:
        with (DEST / name).open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {**old, 'version': 'spider_sft_round2_pilot_v1', 'groups': groups,
              'selection_policy': 'Exact previous successful 32 complete examples and order; attach existing train-only weights.',
              'weight_counts': dict(Counter(weights[i]['loss_weight'] for i in ids)),
              'max_sequence_length': max_length, 'max_answer_tokens': max_answers,
              'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (FULL, OLD, WEIGHTS, Path(__file__))},
              'selected_sha256': digest(DEST / 'train_tokens.jsonl'),
              'weights_sha256': digest(DEST / 'weights.jsonl'),
              'final_read': False, 'validation_read': False, 'arrays_changed': False,
              'gpu_training_started': False}
    with (DEST / 'selection.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('selected_count','weight_counts','bucket_counts','max_sequence_length','max_answer_tokens','input_tokens','target_tokens')}))


if __name__ == '__main__':
    main()
