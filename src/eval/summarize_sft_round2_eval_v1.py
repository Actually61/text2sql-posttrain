"""Compare all saved 625-example base, round-one and round-two results on CPU.

Reads complete saved evidence only. No model imports, SQL connections, inference,
training, prediction selection, source modification, or final-set access.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from summarize_baseline_full import (
    ROOT, LEVELS, POPULATION, Evidence, analyze_group, load_generation,
    load_evaluation, project_path, require,
)
from summarize_sft_round1_eval_v2 import (
    COUNT, FROZEN_SHARED, choose_cases, compare_cost, compare_group,
    load_sft_generation, paired_rows, side_details, transition,
)

CONFIG = 'configs/sft_round2_eval_v1.json'
DESTINATION = 'reports/sft_round2_eval_v1_analysis.json'
PAIRS_DESTINATION = 'reports/sft_round2_eval_v1_comparisons.jsonl'
SPECS = {
    'base': ('reports/baseline_full_generation.json', 'reports/baseline_full_eval.json'),
    'round1': ('reports/sft_round1_eval_v2_generation.json', 'reports/sft_round1_eval_v2_eval.json'),
    'round2': ('reports/sft_round2_eval_v1_generation.json', 'reports/sft_round2_eval_v1_eval.json'),
}
EXPECTED_ADAPTER = 'outputs/runs/sft_round2_formal_v1/checkpoints/update_000796'


def compare(indices, before_name, after_name, predictions, scores):
    result = compare_group(indices, scores[before_name], scores[after_name],
                           predictions[before_name], predictions[after_name])
    result['before'] = result.pop('baseline')
    result['after'] = result.pop('sft')
    result['before_model'], result['after_model'] = before_name, after_name
    return result


def comparison_cases(before_name, pairs, prompts_by_id):
    result = {}
    for category in ('improved', 'regressed'):
        selected = []
        for row in choose_cases(pairs, category):
            case = dict(row)
            case['before'] = case.pop('baseline')
            case['after'] = case.pop('sft')
            case['before_model'], case['after_model'] = before_name, 'round2'
            case['rendered_prompt_with_schema'] = prompts_by_id[row['source_index']]['rendered_prompt']
            case['interpretation'] = 'Observed score transition; SQL/schema inspection is still needed for a semantic explanation.'
            selected.append(case)
        result[category] = selected
    return result


def main():
    started = time.perf_counter()
    output, pairs_output = project_path(DESTINATION), project_path(PAIRS_DESTINATION)
    require(not output.exists() and not pairs_output.exists(), 'Analysis already claimed; no overwrite or retry')
    evidence = Evidence()
    config = evidence.load(CONFIG)
    require(project_path(config['adapter_path']) == project_path(EXPECTED_ADAPTER),
            'Round-two analysis is restricted to its final 796-update adapter')
    generations, evaluations, inputs, predictions, scores, prompts = {}, {}, {}, {}, {}, {}
    for name, (generation_path, evaluation_path) in SPECS.items():
        if name == 'base':
            generations[name], predictions[name] = load_generation(evidence, generation_path, COUNT)
        else:
            generations[name], predictions[name] = load_sft_generation(evidence, generation_path)
            require(generations[name].get('status') == 'pass' and
                    generations[name].get('attempted_count') == COUNT and
                    generations[name].get('automatic_retries') == 0 and
                    generations[name].get('generation_failure_or_skip_count') == 0,
                    f'{name} generation did not finish its single attempt')
            controller = evidence.load(project_path(generations[name]['run_dir']) / 'controller_result.json')
            require(controller.get('status') == 'pass' and controller.get('exit_code') == 0 and
                    controller.get('job_active_pids_after_cleanup') == [], f'{name} worker cleanup did not pass')
        evaluations[name], inputs[name], scores[name] = load_evaluation(
            evidence, evaluation_path, generations[name], predictions[name], COUNT)
        # The older baseline format predates the explicit completion flag;
        # load_evaluation already recomputes its 625 unique rows and all totals.
        if name != 'base':
            require(evaluations[name].get('complete_validation_scored') is True,
                    f'{name} evaluation did not score the full population')
        prompts[name] = evidence.load(project_path(generations[name]['run_dir']) / 'prompts.jsonl', lines=True)

    base = generations['base']
    for name in ('round1', 'round2'):
        generation = generations[name]
        for key in ('model_config', 'model_weight_sha256', 'quantization', 'generation_config'):
            require(generation[key] == base[key], f'{name} differs from base: {key}')
        require(generation['is_loaded_in_4bit'] is True, f'{name} is not 4-bit')
        for key, value in base['packages'].items():
            require(generation['packages'].get(key) == value, f'{name} inference package differs: {key}')
        for path in FROZEN_SHARED:
            require(base['source_sha256'].get(path) is not None and
                    base['source_sha256'][path] == generation['source_sha256'].get(path),
                    f'{name} frozen prompt/tokenizer/input hash differs: {path}')
        require(inputs[name]['protocol'] == inputs['base']['protocol'], f'{name} EX protocol changed')
        require(evaluations[name]['metric_limitations'] == evaluations['base']['metric_limitations'],
                f'{name} metric limitations changed')
        require(prompts[name] == prompts['base'], f'{name} rendered prompts or token counts changed')
    generation_config = base['generation_config']
    require(generation_config['max_new_tokens'] == 256 and generation_config['do_sample'] is False and
            generation_config['num_beams'] == generation_config['num_return_sequences'] == 1,
            'Expected one greedy generation with a 256-token ceiling')
    require(project_path(generations['round2']['adapter_path']) == project_path(EXPECTED_ADAPTER),
            'Generated adapter does not match the round-two final checkpoint')
    require(generations['round2']['adapter_exact_before_and_after'] is True,
            'Round-two adapter changed during inference')
    require(evidence.digest(project_path(EXPECTED_ADAPTER) / 'adapter_model.safetensors') ==
            generations['round2']['adapter_sha256'], 'Final adapter hash differs from inference')

    order = [row['source_index'] for row in prompts['base']]
    require(len(order) == len(set(order)) == COUNT, 'Prompt population has duplicates or omissions')
    for name in SPECS:
        require(list(predictions[name]) == order and set(scores[name]) == set(order),
                f'{name} does not preserve the original population and generation order')
        require(inputs[name]['source_index_order'] == order, f'{name} scoring input order differs')
        require(dict(Counter(row['hardness'] for row in scores[name].values())) == POPULATION,
                f'{name} difficulty population differs')
        for prompt in prompts[name]:
            prediction = predictions[name][prompt['source_index']]
            require(prompt['db_id'] == prediction['db_id'] and
                    prompt['input_token_count'] == prediction['input_token_count'],
                    f'{name} prompt/prediction association differs')
    require(evaluations['base']['overall']['execution_correct'] == 372 and
            evaluations['base']['overall']['executable_predictions'] == 469,
            'Frozen baseline totals changed')
    require(evaluations['round1']['overall']['execution_correct'] == 457 and
            evaluations['round1']['overall']['executable_predictions'] == 556,
            'Frozen round-one totals changed')

    pair_sets = {name: paired_rows(order, scores[name], scores['round2'],
                                   predictions[name], predictions['round2']) for name in ('base', 'round1')}
    rows = []
    for ordinal, index in enumerate(order, 1):
        score = scores['round2'][index]
        rows.append({
            'ordinal': ordinal, 'source_index': index, 'db_id': score['db_id'],
            'hardness': score['hardness'], 'question': score['question'], 'gold_sql': score['gold_sql'],
            'order_matters': score['order_matters'],
            'input_token_count': predictions['round2'][index]['input_token_count'],
            'models': {name: side_details(scores[name][index], predictions[name][index]) for name in SPECS},
            'base_to_round2_ex': transition(scores['base'][index]['matched'], score['matched']),
            'round1_to_round2_ex': transition(scores['round1'][index]['matched'], score['matched']),
            'base_to_round2_executable': transition(scores['base'][index]['prediction_executable'], score['prediction_executable']),
            'round1_to_round2_executable': transition(scores['round1'][index]['prediction_executable'], score['prediction_executable']),
        })
    comparisons = {}
    for name in ('base', 'round1'):
        pairs = pair_sets[name]
        comparisons[f'{name}_to_round2'] = {
            'overall': compare(order, name, 'round2', predictions, scores),
            'by_hardness': {level: compare([i for i in order if scores['round2'][i]['hardness'] == level],
                                          name, 'round2', predictions, scores) for level in LEVELS},
            'improved_source_indices': [row['source_index'] for row in pairs if row['ex_transition'] == 'improved'],
            'regressed_source_indices': [row['source_index'] for row in pairs if row['ex_transition'] == 'regressed'],
        }
    old_failures = [i for i in order if scores['round1'][i]['matched'] is not True]
    old_regressions = [i for i in order if scores['base'][i]['matched'] is True and scores['round1'][i]['matched'] is not True]
    require(len(old_failures) == 168 and len(old_regressions) == 34, 'Previous analysis cohorts changed')
    recovery = {}
    for name, indices in (('round1_168_errors', old_failures), ('round1_34_regressions_vs_base', old_regressions)):
        recovered = [i for i in indices if scores['round2'][i]['matched'] is True]
        recovery[name] = {'count': len(indices), 'recovered_count': len(recovered),
                          'recovered_source_indices': recovered,
                          'remaining_wrong_source_indices': [i for i in indices if i not in recovered],
                          'note': 'Descriptive subsets only; the primary metrics retain all 625 examples.'}
    overall = {name: analyze_group(order, predictions[name], scores[name]) for name in SPECS}
    prompts_by_id = {row['source_index']: row for row in prompts['round2']}
    final_metrics = overall['round2']['evaluation']
    for path in (Path(__file__), ROOT / 'src/eval/summarize_sft_round1_eval_v2.py',
                 ROOT / 'src/eval/summarize_baseline_full.py'):
        evidence.digest(path)
    result = {
        'version': 'sft_round2_eval_v1_analysis', 'status': 'pass',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'One full, once-generated 625-question internal validation comparison of base/round1/round2.',
        'count': COUNT, 'models': {name: {
            'generation_run_id': generations[name]['run_id'], 'evaluation_run_id': evaluations[name]['run_id'],
            'prediction_path': generations[name]['prediction_path'],
            'prediction_sha256': generations[name]['prediction_sha256'],
        } for name in SPECS},
        'checks': {'same_625_ids_and_metadata': True, 'same_original_generation_order': True,
                   'same_base_quantization_prompt_tokenizer_and_generation_config': True,
                   'same_ex_scoring_rules': True, 'one_attempt_per_round2_question': True,
                   'all_saved_metrics_recomputed': True, 'evidence_hashes_verified': True},
        'overall': overall,
        'by_hardness': {level: {name: analyze_group([i for i in order if scores[name][i]['hardness'] == level],
                               predictions[name], scores[name]) for name in SPECS} for level in LEVELS},
        'comparisons': comparisons, 'previous_error_cohorts': recovery,
        'paired_examples_path': PAIRS_DESTINATION,
        'illustrative_cases': {f'{name}_to_round2': comparison_cases(name, pair_sets[name], prompts_by_id)
                               for name in ('base', 'round1')},
        'case_selection_policy': 'Deterministic source-index order, prefer non-empty results and distinct databases. All 625 results remain in metrics; no prediction replacement.',
        'cost': {name: compare_cost(generations[name], evaluations[name]) for name in SPECS},
        'timing_limit': 'Saved stage durations exclude command gaps; runs on different dates are not a controlled speed benchmark.',
        'predeclared_point_estimate_gate': {
            'rule': 'Round2 EX > 457/625 and executable >= 556/625',
            'meets_gate': final_metrics['execution_correct'] > 457 and final_metrics['executable_predictions'] >= 556,
            'interpretation': 'A development decision rule, not statistical significance or proof of generalization.'},
        'interpretation': [
            'EX compares results on one original SQLite database; it is not full test-suite accuracy or proof of SQL semantic equivalence.',
            'Both-empty matches remain counted under the unchanged rule and are explicitly flagged.',
            'Execution failures, unknowns and mismatches remain in the denominator of 625.',
            'Error strings and stop flags are observations, not proven semantic root causes.',
            'One development split and one seed cannot establish generalization; final evaluation remains sealed.',
            'Only the already-fixed final 796-step round2 adapter was evaluated; no checkpoint or prediction selection occurred.'
        ],
        'training_performed': False, 'final_examples_read': False,
        'model_generation_performed_by_analysis': False, 'sql_executed_by_analysis': False,
    }
    with pairs_output.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    result['paired_examples_sha256'] = evidence.digest(pairs_output)
    result['source_sha256'] = evidence.hashes
    result['analysis_seconds'] = time.perf_counter() - started
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'pass', 'report': DESTINATION,
        'overall': {name: overall[name]['evaluation'] for name in SPECS},
        'round1_to_round2': comparisons['round1_to_round2']['overall']['ex_transitions'],
        'base_to_round2': comparisons['base_to_round2']['overall']['ex_transitions'],
        'meets_predeclared_gate': result['predeclared_point_estimate_gate']['meets_gate']},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
