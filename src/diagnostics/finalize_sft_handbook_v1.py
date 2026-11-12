"""Finalize two presentation documents from complete, verified final results.

Pure stdlib, saved-record reads only.  This script never starts a model, SQL
worker, test or installation.  A successful completion audit is mandatory.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/final_holdout_decode_v4'
GENERATION = {'base': ROOT/'outputs/final_holdout_v3/base',
              'sft': ROOT/'outputs/final_holdout_v4/sft'}
AUDIT = ROOT / 'outputs/checks/final_completion_v2/result.json'
REVIEW = ROOT / 'outputs/checks/final_report_review_v1.json'
BACKUP = ROOT / 'outputs/final_holdout_v3/document_before_completion'
LOG = RUN / 'document_finalization.json'
STRICT_SCORE = ROOT / 'outputs/final_holdout_v3/scores_base/summary.json'
DECODE_CONFIG = ROOT / 'configs/final_decode_v4.json'
BOOK = ROOT / 'docs/项目总结与简历面试手册.md'
README = ROOT / 'README.md'
PREFIX = ROOT / 'outputs/final_holdout_v1/base/predictions.jsonl'
COUNT = 1034
LEVELS = ('easy', 'medium', 'hard', 'extra')
TITLES = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard', 'extra': 'Extra'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metric(value, expected_count=None):
    count = value['count']
    correct, executable = value['execution_correct'], value['executable_predictions']
    require(type(count) is int and count > 0, 'Invalid metric denominator')
    require(expected_count is None or count == expected_count, 'Metric denominator differs')
    require(0 <= correct <= executable <= count, 'Invalid correct/executable counts')
    require(math.isclose(value['execution_accuracy'], correct/count, abs_tol=1e-12), 'EX arithmetic differs')
    require(math.isclose(value['prediction_executable_rate'], executable/count, abs_tol=1e-12), 'Executable arithmetic differs')
    require(value['prediction_execution_unknown'] == 0 and value['gold_execution_errors'] == 0,
            'Unresolved infrastructure/gold results cannot be published')


def load_verified():
    """Gate publication on all terminal records, fixed population and prefix bytes."""
    require(not LOG.exists(), 'Document finalization already claimed; do not overwrite')
    audit = read(AUDIT)
    require(audit['status'] == 'pass', 'Final completion audit has not passed')
    review = read(REVIEW)
    expected_ids = {f'H{index:02}' for index in range(1, 10)} | {f'R{index:02}' for index in range(1, 5)}
    require({row['id'] for row in review['mandatory_updates_after_success']} == expected_ids,
            'The reviewed 13-item document checklist differs')
    comparison = read(RUN / 'comparison.json')
    require(comparison['status'] == 'pass', 'Final comparison incomplete')
    for key in ('same_prompts_and_generation_config', 'same_ex_protocol'):
        require(comparison[key] is True, f'Fairness check failed: {key}')
    for key in ('prediction_replay_or_selection_performed', 'training_performed', 'final_result_used_for_tuning'):
        require(comparison[key] is False, f'Publication boundary violated: {key}')
    decode_config = read(DECODE_CONFIG)
    generation_roots = {arm: directory.relative_to(ROOT).as_posix() for arm, directory in GENERATION.items()}
    require(comparison['generation_roots'] == decode_config['generation_roots'] == generation_roots,
            'Final comparison binds different generation directories')
    require(comparison['scoring_protocol_revised'] is True and comparison['predictions_regenerated_for_scoring'] is False,
            'Authorized scoring-revision boundary differs')
    require(comparison['decoder_revision'] == decode_config['decoder_revision'] == 'utf8_surrogateescape_text_only_v1',
            'Unexpected final decoder revision')
    require(comparison['authorization_record'] == decode_config['authorization_record'], 'Authorization binding differs')
    authorization_path = ROOT/decode_config['authorization_record']
    authorization = read(authorization_path)
    require(authorization['authorized'] is True and authorization['authorization_source'] == 'user',
            'Explicit decoder authorization missing')
    strict_score = read(STRICT_SCORE)
    require(strict_score['status'] == 'failed' and strict_score['complete_final_scored'] is False
            and strict_score['partial_record_count'] == 454, 'Historical strict-decoding failure differs')
    historical = audit['report_consumables']['strict_scoring_history']
    require(historical['strict_attempt_returned_records'] == 454
            and historical['strict_attempt_valid_outcomes'] == 453
            and historical['old_partial_scores_reused'] is False
            and math.isclose(historical['strict_attempt_wall_seconds'], strict_score['wall_seconds']),
            'Audited strict-scoring history differs')
    strict_score['audit_history'] = historical
    strict_score['audit_consumables'] = audit['report_consumables']
    for name, expected in comparison['source_sha256'].items():
        require(digest(ROOT/name) == expected, f'Compared source changed: {name}')
    config = read(ROOT/'configs/final_holdout_resume_v3.json')
    require(config['run_root'] == 'outputs/final_holdout_v3', 'Successful version path differs')
    require(digest(PREFIX) == config['prefix_sha256'], 'Original 616-row prefix changed')
    prefix = PREFIX.read_bytes()
    require(len(prefix.splitlines()) == 616 and prefix.endswith(b'\n'), 'Invalid preserved prefix')
    generation, controllers, scoring = {}, {}, {}
    evidence = {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
                for path in (AUDIT, REVIEW, RUN/'comparison.json', PREFIX, STRICT_SCORE, DECODE_CONFIG,
                             authorization_path, ROOT/decode_config['frozen_report'],
                             ROOT/decode_config['decoder_cpu_evidence'])}
    for arm in ('base', 'sft'):
        directory = GENERATION[arm]
        generation[arm] = read(directory/'summary.json')
        controllers[arm] = read(directory/'controller_result.json')
        scoring[arm] = read(RUN/f'scores_{arm}'/'summary.json')
        gen, control, score = generation[arm], controllers[arm], scoring[arm]
        require(gen['status'] == control['status'] == score['status'] == 'pass', f'{arm} has an unsuccessful stage')
        require(control['exit_code'] == 0 and control['job_active_pids_after_cleanup'] == [],
                f'{arm} did not exit cleanly')
        require(score['complete_final_scored'] is True, f'{arm} scoring is partial')
        protocol = read(RUN/f'scores_{arm}'/'protocol.json')
        require(protocol['text_decoder'] == 'utf-8/surrogateescape'
                and protocol['decoder_revision'] == comparison['decoder_revision']
                and protocol['raw_sql_unchanged'] is True
                and protocol['scores_reexecuted_for_both_models'] is True,
                'Successful scorer did not use the authorized lossless TEXT protocol')
        for key in ('prediction_count', 'generated_count', 'attempted_count'):
            require(gen[key] == COUNT, f'{arm}: {key} is not canonical 1034')
        require(gen['generation_failure_or_skip_count'] == 0 and gen['automatic_retries'] == 0,
                f'{arm} has generation failures/retries')
        require(gen['training_performed'] is False and gen['reference_answers_loaded_for_inference'] is False,
                f'{arm} inference used training/reference information')
        require(gen['reused_prefix_count'] == (616 if arm == 'base' else 0), 'Wrong prefix count')
        require(gen['new_generated_count'] == (418 if arm == 'base' else 1034), 'Wrong new-generation count')
        predictions_path = directory/'predictions.jsonl'
        require(digest(predictions_path) == gen['prediction_sha256'] == score['prediction_sha256'], 'Prediction hash differs')
        rows = [json.loads(line) for line in predictions_path.read_text(encoding='utf-8').splitlines()]
        require([row['source_index'] for row in rows] == list(range(COUNT)), 'Canonical IDs are not 0..1033 once')
        require(all(row['attempt_number'] == 1 and row['generation_ok'] is True for row in rows),
                'Repeated/failed generation found')
        if arm == 'base':
            require(predictions_path.read_bytes().startswith(prefix), 'Base prefix not preserved byte-for-byte')
        metric(score['overall'], COUNT)
        require(score['overall'] == comparison['overall'][arm], 'Comparison/scoring overall differs')
        for level in LEVELS:
            metric(score['by_hardness'][level])
            require(score['by_hardness'][level] == comparison['by_hardness'][level][arm], 'Difficulty reports differ')
        for key in ('count', 'execution_correct', 'executable_predictions'):
            require(sum(score['by_hardness'][level][key] for level in LEVELS) == score['overall'][key],
                    'Difficulty totals do not reconcile')
        outcomes = comparison['failure_types'][arm]['outcomes']
        require(sum(outcomes.values()) == COUNT, 'Failure outcomes do not sum to denominator')
        require(outcomes.get('ex_match', 0) == score['overall']['execution_correct'], 'Match outcome differs')
        require(outcomes.get('execution_failed', 0) == COUNT-score['overall']['executable_predictions'], 'Failure outcome differs')
        require(outcomes.get('executable_mismatch', 0) == score['overall']['executable_predictions']-score['overall']['execution_correct'],
                'Mismatch outcome differs')
        require(comparison['failure_types'][arm] == score['failure_types'], 'Failure reports differ')
        for path in (directory/'summary.json', directory/'controller_result.json', predictions_path,
                     RUN/f'scores_{arm}'/'summary.json', RUN/f'scores_{arm}'/'protocol.json', directory/'gpu_samples.json'):
            evidence[path.relative_to(ROOT).as_posix()] = digest(path)
    for name in ('prompts.jsonl', 'generation_config.json'):
        require((GENERATION['base']/name).read_bytes() == (GENERATION['sft']/name).read_bytes(), f'Arm bytes differ: {name}')
    pair = comparison['overall']
    require(pair['count'] == COUNT and sum(pair[key] for key in ('improved', 'regressed', 'both_correct', 'both_wrong')) == COUNT,
            'Paired outcomes do not reconcile')
    require(pair['improved']-pair['regressed'] == pair['sft']['execution_correct']-pair['base']['execution_correct'],
            'Paired net change differs')
    require(len(comparison['improved_source_indices']) == pair['improved'] and
            len(comparison['regressed_source_indices']) == pair['regressed'], 'Paired ID counts differ')
    for path, backup in ((README, BACKUP/'README.md'), (BOOK, BACKUP/BOOK.name)):
        require(path.read_bytes() == backup.read_bytes(), f'Document changed since the preserved backup: {path.name}')
        evidence[backup.relative_to(ROOT).as_posix()] = digest(backup)
    old_control = read(PREFIX.parent/'controller_result.json')
    old_gpu = read(PREFIX.parent/'gpu_samples.json')
    require(old_control['status'] == 'failed', 'Original failure history differs')
    require(math.isclose(old_control['controller_seconds'], generation['base']['prior_controller_seconds'], abs_tol=1e-9),
            'Prior controller time differs')
    evidence[(PREFIX.parent/'controller_result.json').relative_to(ROOT).as_posix()] = digest(PREFIX.parent/'controller_result.json')
    evidence[(PREFIX.parent/'gpu_samples.json').relative_to(ROOT).as_posix()] = digest(PREFIX.parent/'gpu_samples.json')
    return comparison, generation, controllers, scoring, old_control, old_gpu, strict_score, evidence


def rate(count, denominator=COUNT):
    return f'{count}/{denominator}＝{100*count/denominator:.2f}%'


def delta(before, after):
    value = 100*(after-before)/COUNT
    return f'提高{value:.2f}个百分点' if value > 0 else f'下降{-value:.2f}个百分点' if value < 0 else '持平（0.00个百分点）'


def seconds(value):
    return f'{value:.2f}秒（{value/60:.2f}分钟）'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |',
                      *('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)])


def section(text, start, end):
    require(text.count(start) == text.count(end) == 1, f'Document section anchors differ: {start}')
    return text.split(start, 1)[1].split(end, 1)[0]


def replace_once(text, old, new):
    require(text.count(old) == 1, f'Document exact anchor differs: {old[:65]}')
    return text.replace(old, new, 1)


def replace_answer(text, number, answer):
    pattern = rf'(\*\*Q{number:02}[^\n]*\*\*\n\n).*?(?=\n\*\*Q\d|\n### |\Z)'
    result, count = re.subn(pattern, lambda match: match[1]+answer.strip()+'\n', text, flags=re.S)
    require(count == 1, f'Question anchor Q{number:02} differs')
    return result


def build_documents(data):
    comp, gen, ctl, score, old_ctl, old_gpu, strict_score, _ = data
    pair = comp['overall']; base, sft = pair['base'], pair['sft']
    b, s = base['execution_correct'], sft['execution_correct']
    be, se = base['executable_predictions'], sft['executable_predictions']
    change, executable_change = delta(b, s), delta(be, se)
    date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y年%m月%d日')
    brief = f'基座EX为{100*b/COUNT:.2f}%，第一轮SFT为{100*s/COUNT:.2f}%，{change}'
    verdict = ('冻结的第一轮SFT在本项目最终留出集上优于基座' if s > b else
               '冻结的第一轮SFT在本项目最终留出集上未超过基座')
    overall = table(['模型', '可执行率', 'EX'], [
        ['原基座', rate(be), rate(b)], ['第一轮796步SFT', rate(se), rate(s)]])
    difficulty = table(['难度', '题数', '基座可执行率', '基座EX', '首轮可执行率', '首轮EX'], [
        [TITLES[level], comp['by_hardness'][level]['count'],
         rate(comp['by_hardness'][level]['base']['executable_predictions'], comp['by_hardness'][level]['count']),
         rate(comp['by_hardness'][level]['base']['execution_correct'], comp['by_hardness'][level]['count']),
         rate(comp['by_hardness'][level]['sft']['executable_predictions'], comp['by_hardness'][level]['count']),
         rate(comp['by_hardness'][level]['sft']['execution_correct'], comp['by_hardness'][level]['count'])]
        for level in LEVELS])
    outcomes = table(['模型', 'EX匹配', '可执行但不匹配', '执行失败'], [
        [label, comp['failure_types'][arm]['outcomes'].get('ex_match', 0),
         comp['failure_types'][arm]['outcomes'].get('executable_mismatch', 0),
         comp['failure_types'][arm]['outcomes'].get('execution_failed', 0)]
        for arm, label in (('base', '基座'), ('sft', '首轮SFT'))])
    categories = sorted(set(comp['failure_types']['base']['first_error_categories']) |
                        set(comp['failure_types']['sft']['first_error_categories']))
    categories = [name for name in categories if name != 'none']
    errors = table(['执行失败首个错误类型', '基座', '首轮SFT'], [
        [name, comp['failure_types']['base']['first_error_categories'].get(name, 0),
         comp['failure_types']['sft']['first_error_categories'].get(name, 0)] for name in categories])
    old_seconds = old_ctl['controller_seconds']; new_seconds = ctl['base']['controller_seconds']
    sft_seconds = ctl['sft']['controller_seconds']
    activity = (old_seconds+new_seconds+sft_seconds+strict_score['wall_seconds']
                +score['base']['wall_seconds']+score['sft']['wall_seconds'])
    require(math.isclose(activity, strict_score['audit_consumables']['combined_windows_including_failed_strict_scoring_seconds'],
                         abs_tol=1e-6), 'Document activity costs disagree with completion audit')
    historical = strict_score['audit_history']
    changed_old = len(historical['strict_valid_outcomes_changed_in_revised_full_score'])
    strict_comparison = (f"旧454条返回记录包含453条有效评分；与修订后的同题结果核对，{changed_old}条评分结果变化。"
                         '这项检查用于披露兼容性影响，不用于选择答案或调整模型。')
    old_peak = max(row['used_mib'] for row in old_gpu['samples'])
    cost = table(['活动段', '题数', '控制器/评分耗时', '查询生成时间之和', '生成阶段allocated / reserved峰值(MiB)', '整卡采样峰值(MiB)'], [
        ['基座原始段（后因遥测停止）', 616, seconds(old_seconds), seconds(gen['base']['prior_generation_seconds_total']),
         f"{gen['base']['prior_generation_peak_allocated_mib']:.2f} / {gen['base']['prior_generation_peak_reserved_mib']:.2f}", f'{old_peak:.2f}'],
        ['基座续接段', 418, seconds(new_seconds), seconds(gen['base']['new_generation_seconds_total']),
         f"{gen['base']['generation_peak_allocated_mib']:.2f} / {gen['base']['generation_peak_reserved_mib']:.2f}", f"{gen['base']['nvidia_smi_sampled_peak_used_mib']:.2f}"],
        ['首轮SFT完整生成', 1034, seconds(sft_seconds), seconds(gen['sft']['new_generation_seconds_total']),
         f"{gen['sft']['generation_peak_allocated_mib']:.2f} / {gen['sft']['generation_peak_reserved_mib']:.2f}", f"{gen['sft']['nvidia_smi_sampled_peak_used_mib']:.2f}"],
        ['历史严格解码评分（CPU，失败保留）', '454条记录（453条有效评分）', seconds(strict_score['wall_seconds']), '不适用', '不适用', '不适用'],
        ['基座修订协议完整评分（CPU）', 1034, seconds(score['base']['wall_seconds']), '不适用', '不适用', '不适用'],
        ['首轮修订协议完整评分（CPU）', 1034, seconds(score['sft']['wall_seconds']), '不适用', '不适用', '不适用']])
    total_cost = (f'基座两段生成控制器合计{seconds(old_seconds+new_seconds)}；上述生成与SQL评分各段记录时长之和为'
                  f'{seconds(activity)}，含历史严格解码失败评分。部分CPU评分与GPU生成重叠，因此这个求和不是从开始到结束的端到端墙钟历时；不含CPU预检、资源等待、解码兼容性验证、修复排查和两段之间的等待。'
                  f"基座两段逐题生成记录的allocated/reserved峰值为{gen['base']['canonical_generation_peak_allocated_mib']:.2f} / "
                  f"{gen['base']['canonical_generation_peak_reserved_mib']:.2f} MiB，是两段对应峰值取最大，不能相加。")
    intro = f'''# Text-to-SQL 项目总结与简历面试手册

截至{date}，SFT训练、625题开发验证及1034题最终留出集两模型对照均已完成。本手册按“项目与结果 → 简历写法 → 两分钟讲稿 → 面试问答”阅读，集中回答核心与常见追问；不能穷尽所有面试官的延伸问题。

**核心结果：** 在取自官方dev的项目最终留出集1034题上，{brief}。可执行率由{100*be/COUNT:.2f}%变为{100*se/COUNT:.2f}%。这里的EX是一份原SQLite数据库上的结果匹配率，不是官方隐藏test或完整test-suite成绩；最终评分含经用户授权的TEXT无损解码修订，说明见第6节。

**先记住：** 1.5B代码指令模型；Windows单卡8GB；6364条训练题；625题开发验证；1034题最终留出集。第一轮在开发集为73.12%，第二轮72.96%未提升；最终只比较提前选定的第一轮与基座，没有根据最终结果调参。

**复习顺序：** 先读结果、简历和讲稿，再读“必会”题。回答时说明为什么这样做、证据是什么、还不能证明什么，不必背诵基础设施故障的全过程。

## 一 项目做了什么 有什么结果

'''
    old = BOOK.read_text(encoding='utf-8')
    work = '### 1 项目要解决的问题'+section(old, '### 1 项目要解决的问题', '### 4 三模型结果与结论')
    work = replace_once(work, '→ 再评测 → 保留改善、退化和负结果', '→ 再评测 → 保留负结果 → 提前选定首轮 → 最终留出集公平对照')
    work = replace_once(work,
        '| 最终评测准备与尝试 | 预先冻结首轮796步方案，CPU检查1034条完整输入；基座生成616条后因遥测超时停止，完整最终对照尚缺 |',
        '| 最终两模型对照 | 提前冻结首轮796步方案，在1034题/20库上完成基座与首轮的同协议生成、EX评分、逐题配对和难度统计 |')
    work = replace_once(work, '**尚未完成：** DPO、在线强化学习、最终集效果评测、生产部署、跨机器开箱复现。不能把这些写成已完成成果。',
        '**尚未完成：** DPO、在线强化学习、生产部署、跨机器开箱复现。当前SFT阶段已完整收尾，不把扩展方向写成已完成成果。')
    work = replace_once(work,
        '| 最终保留集（本次已解封） | 1034 | 20 | 来自官方dev，不是官方test；基座仅完成616条生成，未评分，不用于训练或调参 |',
        '| 项目最终留出集（已评测） | 1034 | 20 | 来自官方dev，不是官方隐藏test；只用于冻结方案的最终两模型报告，不用于训练、偏好数据或调参 |')
    internal = '### 4 内部开发验证：三模型对照\n'+section(old, '### 4 三模型结果与结论', '### 5 这些分数能说明和不能说明什么')
    internal = internal.replace('保留首轮作为当前候选', '在最终评测前选定首轮')
    limits = '''### 5 分数的含义与边界

可执行率说明查询能否成功运行；EX说明预测与参考在一份原SQLite数据库上的结果是否匹配。EX通过不证明普遍语义等价，也不是完整官方test-suite accuracy。625题参与开发决策；1034题用于方案冻结后的项目最终报告。开发三模型保留旧strict UTF-8执行记录，最终两模型统一使用授权的TEXT无损解码修订；两组题目与解码版本都不同，不能当成同一队列、全相同执行协议直接比较百分比。

早期脚本为数据完整性、schema检查和匿名查重读取过官方dev记录；准确说法是“没有用于模型开发、训练或调参”，不是“从未读取过任何最终题字节”。第一轮方案、输入模板和生成参数在最终推理前冻结；原执行评分后因数据库TEXT解码故障中止，经明确授权做窄范围兼容性修订，并对两模型全部1034题统一重新评分，没有改预测。按库隔离不能排除公开Spider被基座预训练接触；单种子、单机结果不证明统计显著性、真实业务泛化或跨机器可复现。

'''
    final = f'''### 6 最终留出集：1034题、20库，基座对首轮SFT

最终数据来自官方dev，作为本项目留出集，不是官方隐藏test。两模型保持同一基座revision、tokenizer、完整schema模板、NF4双量化、BF16、SDPA、batch1、单次贪心、最多256个新token和每题30秒。所有1034题保留固定分母；第二轮没有参加最终评测。

**评分协议的透明修订。** 旧strict UTF-8执行器保留454条记录后，因wta_1参考查询返回的TEXT无法严格解码而停止，这不是模型答错。用户明确授权后，只把SQLite结果中的TEXT解码改为可逆的`surrogateescape`，BLOB仍保持bytes；SQL、数据库字节、比较器、DISTINCT、排序规则和超时不变。新目录对两模型1034条固定预测全部重新评分，没有复用旧部分评分，也没有重新生成SQL。整个执行协议确实有这一处修订，不能称完全未变；本节两模型使用同一修订版本，旧失败评分保留。{strict_comparison}

{overall}

首轮相对基座：**{pair['improved']}题改善、{pair['regressed']}题退化**；双方均正确{pair['both_correct']}题、均错误{pair['both_wrong']}题。EX{change}，可执行率{executable_change}。**结论：{verdict}。** 这是冻结方案的一次结果，不据此再挑检查点、改prompt或调整评分。

{difficulty}

{outcomes}

{errors}

失败类型依据既有评分记录中的首个执行错误统计；可执行但结果不匹配另计，不把它直接归因于某个未经验证的训练原因。

**历史中断与公平续接。** 最初基座完成616题后，一次`nvidia-smi`遥测超过5秒，按规则停止；没有证据证明是GPU容量不足。修复遥测并完成CPU/无模型验收后，另一次资源门槛检查未通过，未启动模型。最终新目录逐字复用原616条、只生成余下418条；首轮SFT完整生成1034条。每题始终只有一次预测，没有重新生成已完成题、择优或按答案修复SQL。旧失败源码、日志和账本保留。

{cost}

{total_cost}

各段题目、长度及后台/系统状态不同，不能把续接段较快归因于遥测修复，也不是公平的推理速度benchmark。allocated包含于reserved口径，不能相加；整卡采样包含其他应用且不是瞬时进程峰值。旧616题失败段缺少最终allocator计数，不能用新段`num_ooms=0`推断整个基座运行OOM计数为0。两模型生成控制器均正常退出且Job为空，评分与最终配对均通过完整性核验。

'''
    resume = f'''## 二 写简历时如何描述这个项目

### 1 可直接使用的版本

**基于QLoRA的小模型Text-to-SQL微调与评测｜个人学习项目**  
技术栈：Python、PyTorch、Transformers、PEFT、QLoRA、SQLite。

- 基于Qwen2.5-Coder-1.5B-Instruct建立数据处理、微调与执行评测流程，按数据库隔离训练、开发验证和最终留出集；使用6364条清洗数据，在单张8GB显存GPU上完成QLoRA SFT，采用assistant-only监督、梯度累积和完整检查点恢复。
- 在取自官方dev的项目最终留出集1034题上，保持相同生成配置、采用统一原库EX评分（含TEXT无损解码兼容修订），基座EX为{100*b/COUNT:.2f}%，首轮SFT为{100*s/COUNT:.2f}%，{change}；可执行率由{100*be/COUNT:.2f}%变为{100*se/COUNT:.2f}%。该指标为单份原SQLite执行结果匹配率，不是官方隐藏test或完整test-suite成绩。
- 在625题开发集上开展多表样本加权的单变量对照：首轮EX为73.12%，第二轮72.96%，16题改善、17题退化，未带来总体提升；保留负结果和逐题分析，并在最终评测前选定首轮方案。

### 2 篇幅有限时如何取舍

保留任务与技术路线、带口径的最终结果、受控实验与分析。rank、学习率和Windows细节留给面试；不要把73.12%改标成最终集成绩，也不把第二轮写成优化成功。项目日期按实际开展时间填写，不虚构单位、团队或线上收益。

| 不准确的写法 | 应采用的表达 |
| --- | --- |
| Spider官方测试准确率 | 官方dev作为项目最终留出集，单份原SQLite上的EX |
| 准确率提升若干% | 分别给出两模型比例，差值用“个百分点” |
| 第二轮进一步优化成功 | 开发集第二轮未提升，保留16改善、17退化 |
| 基座1034题一口气跑完 | 原616条逐字复用，剩余418条续接，每题一次 |
| 提出全新的QLoRA算法 | 使用QLoRA完成任务微调与受控实验 |
| 完成DPO/RL或线上部署 | 仅完成SFT阶段；扩展尚未实现 |

### 3 个人贡献如何回答

围绕自己能讲清并复核的部分：按库划分、assistant-only监督、整组token归一化、公平评测、SQL与schema错误分析以及负结果。代码实现和排查有AI辅助，应如实说明；文件证明实验发生过，不自动证明自己已独立掌握每行实现。未掌握的部分继续走读，不声称独立从零实现所有框架。

## 三 项目讲解文稿与面试问答

### 1 两分钟项目讲解稿

> 我做的是一个Text-to-SQL个人项目：输入自然语言问题和数据库结构，让Qwen2.5-Coder-1.5B-Instruct生成SQL。使用自己的8GB显卡，按数据库隔离6364条训练数据、625条开发验证和官方dev中的1034条项目最终留出题。
>
> 训练采用QLoRA，冻结4-bit基座，只更新LoRA；只监督assistant答案和结束标记。使用batch1、累积8和梯度检查点，按整组有效答案token数归一化，训练一轮796次更新，并验证保存恢复。
>
> 我先在开发集上得到首轮EX73.12%，然后从原基座独立开展第二轮，只提高多表样本loss权重。第二轮为72.96%，16题改善、17题退化，没有总体提升，因此保留负结果，并提前选定第一轮进入最终评测。
>
> 最终1034题保持相同生成配置，并在同一评分版本下对照，{brief}；可执行率由{100*be/COUNT:.2f}%变为{100*se/COUNT:.2f}%。基座因遥测中断过，保留原616条只补418条，没有重新作答；旧评分还因数据库TEXT解码中止，经授权采用可逆解码，对两模型全部重新评分，保留了协议变更记录。
>
> 我关注的不只是分数，还包括为什么会错。例如开发集有缺少DISTINCT计数却通过当前数据库的预测。因此这里的EX只是单份原SQLite结果匹配，不是语义正确证明或官方隐藏测试成绩；公开数据的预训练接触、单种子和跨机器复现仍是限制。

'''
    qa = '### 2 项目定位与个人工作'+section(old, '### 2 项目定位与个人工作', '### 11 面试前的核对清单')
    answers = {
        1: f'输入自然语言问题和schema，输出单条SQLite SQL。完成数据划分清洗、两轮独立QLoRA SFT、625题开发三模型对照及1034题最终两模型评测。最终{brief}；第二轮加权未提高开发集总体EX，保留负结果。交付还包括adapter、固定配置、逐题证据和本手册，不包括DPO或上线产品。',
        9: 'train_spider按库拆为训练与625题开发验证，清洗后训练6364题；官方dev的1034题/20库作为项目最终留出集，不是隐藏test。开发集用于类别级错误分析与方案选择，不参加梯度更新；最终集在方案冻结后用于最终推理和评分，不用于调参、训练或偏好数据。早期完整性、schema与匿名查重读取不等于用于模型开发，也不能据此声称从未读取过最终题字节。',
        10: '根据开发验证错误提出类别级改进假设属于开发过程，所以625题不能再称独立最终测试。训练数据和权重仍只来自训练集，不把验证答案塞回训练。第一轮是根据开发结果提前选定的，之后只在1034题最终留出集报告一次固定方案对照；最终结果没有用于再选模型或调参。',
        11: '训练原6375题隔离3条未解决执行问题、移除8条完全重复，保留处置记录。没有因为样本长或模型答错删题。开发评测固定625题、最终评测固定1034题，错误仍计入各自分母，不能事后筛掉错误提高分数。',
        53: '不能把所有超时都叫OOM。早期训练试跑观察到长样本耗时和内存压力，但还发现CPU文件并发、PID和计时问题。最终评测原616题后是一次nvidia-smi采样超过5秒触发停止，当时已有allocated/reserved峰值约1284/1434 MiB，没有OOM异常证据；底层驱动或调度原因仍未证实。改用经验证的持续采样后，只续接未完成题并完成最终对照；不能声称换了更大显卡，也不能用新段allocator计数补填旧段缺失计数。',
        57: '开发集三模型、最终集基座与首轮均固定do_sample=False、num_beams=1、每题一个结果和256新token上限。这样比较相同推理协议下的训练变化。最终基座分616+418两段，前616条原字节保留，没重新作答；基础设施版本变更不等于给同一道题新增模型尝试。多候选择优、自动修SQL再答会改变协议，本项目没有这样做。',
        59: f'开发集每模型的分母是625；最终集每模型的分母是1034。最终首轮{se}题可执行，其中{s}题EX匹配、{se-s}题可执行但不匹配，另{COUNT-se}题执行失败。不能把EX算成{s}/{se}再当总体EX。旧strict评分的454条是不完整故障记录；授权解码修订后两模型各1034条全部新评分，不能混拼旧分数。模型SQL错误保留在分母，基础设施失败则停止。',
        66: '开发集是从train_spider按库划出的625题；最终集是官方dev的1034题，作为本项目留出集。两者均逐题只在一份原SQLite数据库上执行，使用固定官方工具的比较实现；最终执行器另有公开记录的TEXT无损解码修订。它们不是官方隐藏test，也没有运行完整多数据库test-suite。使用官方函数不等于遵循整个榜单协议。',
        79: '只对训练问题生成多个候选，确认同题有足够正确与错误候选，再通过执行及更可靠校验构造偏好对。单库EX会因空结果、重复值或参考口径偶然通过，错误chosen会带来错误偏好。625题开发验证与1034题最终留出集的问题/答案都不能加入偏好训练；本项目尚未开展DPO数据生成或训练。',
        82: '先展示最终1034题两模型结果与固定协议，再补625题开发三模型对照和第二轮负结果；用开发集已有SQL案例解释错误，沿代码说明loss、累积和恢复。当前SFT阶段已收尾，接下来优先把关键实现讲清、形成可迁移复现入口。新的算法实验要另定开发/最终验证边界；不把已查看的最终成绩重新当作未见测试。',
    }
    for number, answer in answers.items():
        qa = replace_answer(qa, number, answer)
    qa += '''
**Q83 必会｜基座分两段生成，怎样保证公平而不是重试择优？**

原616条有连续ID、attempt=1、完整EOS与原始文件哈希，续接文件先逐字复制这些字节，再只派发616～1033号问题。每题是独立完整prompt，不跨题保留KV状态；基座、量化、生成参数和输出提取规则不变。核验两模型1034题prompt及生成配置字节一致，不挑答案。时间不能当严格速度对照，因为两段题目、负载与系统状态不同。

**Q84｜最终集已经看过了，后续还能怎样使用？**

可以保留这次固定方案结果做展示与审计；不能根据其错误调参后，再把同一批数据称为独立未见最终测试。若开展下一轮研究，训练与开发材料仍不能混入这些答案，应预先确定新的独立评估方案或明确这是已使用数据上的探索；不删除旧结果，也不挑更好的成绩替代它。

**Q85 必会｜为什么要修订TEXT解码，能不能用ignore或replace解决？**

旧执行器把SQLite TEXT按严格UTF-8转成Python字符串，遇到数据库中的非法字节会失败，参考SQL也因此无法评分。surrogateescape把不能解码的字节可逆映射到代理字符，使用相同方式编码可还原原字节；正常UTF-8值不变，BLOB仍是bytes。ignore会丢字节，replace会把不同坏字节变成相同替代符，可能制造错误匹配，所以没有使用。这个改动只修结果解码，不修改数据库、SQL或比较器，但仍是执行协议修订，必须经授权、保留旧失败并对两模型全量统一评分，不能称原协议完全未变。

'''
    ending = f'''### 11 面试前的核对清单

1. 解释输入、输出，以及1.5B模型和8GB显存的取舍。
2. 说清6364训练、625开发、1034项目最终留出，官方dev不等于隐藏test。
3. 解释LoRA参数、−100、EOS、因果移位和整组token归一化。
4. 区分权重、激活、logits显存优化；说明完整保存与恢复。
5. 准确报告最终{brief}，不把开发集73.12%标成最终成绩。
6. 保留第二轮16改善、17退化、下降0.16个百分点的开发负结果。
7. 能解释1161字段连接、1688分组对象、5839去重计数；这些案例属于开发集。
8. 说明616+418续接与每题一次、公平性及耗时不可直接比较。
9. 说明原库EX限制、预训练接触不确定性、AI协作范围和未完成扩展。
10. 说明454条严格解码失败、授权surrogateescape、BLOB不变及两模型统一重新评分；不混用旧分数。

### 12 需要核实细节时的证据索引

正文已包含主线；以下为原始记录与代码导航。生成/训练/评分入口均绑定一次性目录，不要为了展示再次执行。

| 内容 | 证据 |
| --- | --- |
| 最终1034题整体、难度、配对与失败类型 | [最终对照](D:/大模型项目/text2sql-posttrain/outputs/final_holdout_decode_v4/comparison.json) |
| 基座两段与首轮生成耗时、显存 | [基座](D:/大模型项目/text2sql-posttrain/outputs/final_holdout_v3/base/summary.json)、[首轮](D:/大模型项目/text2sql-posttrain/outputs/final_holdout_v4/sft/summary.json) |
| 最终两模型统一修订评分 | [基座评分](D:/大模型项目/text2sql-posttrain/outputs/final_holdout_decode_v4/scores_base/summary.json)、[首轮评分](D:/大模型项目/text2sql-posttrain/outputs/final_holdout_decode_v4/scores_sft/summary.json) |
| 最终方案、续接与完成审计 | [原方案冻结](D:/大模型项目/text2sql-posttrain/reports/final_holdout_v1_freeze.json)、[基座续接冻结](D:/大模型项目/text2sql-posttrain/reports/final_holdout_resume_v3_freeze.json)、[SFT入口冻结](D:/大模型项目/text2sql-posttrain/reports/final_holdout_resume_v4_freeze.json)、[616条复用记录](D:/大模型项目/text2sql-posttrain/outputs/final_holdout_v3/base/continuation.json)、[完成审计](D:/大模型项目/text2sql-posttrain/outputs/checks/final_completion_v2/result.json) |
| TEXT解码修订授权、CPU证据与冻结 | [用户授权记录](D:/大模型项目/text2sql-posttrain/reports/final_decode_authorization_v1.json)、[无损解码验证](D:/大模型项目/text2sql-posttrain/outputs/checks/final_sql_decode_v1/result.json)、[评分修订冻结](D:/大模型项目/text2sql-posttrain/reports/final_decode_v4_freeze.json) |
| 严格UTF-8评分中止（历史，454条） | [失败评分记录](D:/大模型项目/text2sql-posttrain/outputs/final_holdout_v3/scores_base/summary.json) |
| 原616题遥测故障（历史） | [失败分析](D:/大模型项目/text2sql-posttrain/outputs/final_holdout_v1/failure_analysis.json)、[原控制器](D:/大模型项目/text2sql-posttrain/outputs/final_holdout_v1/base/controller_result.json) |
| 开发集三模型与第二轮负结果 | [开发评测汇总](D:/大模型项目/text2sql-posttrain/reports/sft_round2_eval_v1_analysis.json)、[33题复盘](D:/大模型项目/text2sql-posttrain/reports/sft_round2_review_v1/cases.json) |
| 数据划分与清洗 | [划分](D:/大模型项目/text2sql-posttrain/reports/spider_split_manifest.json)、[清洗](D:/大模型项目/text2sql-posttrain/reports/spider_gold_manifest.json) |
| 早期最终集完整性与匿名查重 | [检查源码 第45行](D:/大模型项目/text2sql-posttrain/src/data/check_spider.py:45) |
| 两轮正式训练 | [首轮](D:/大模型项目/text2sql-posttrain/reports/sft_round1_resource_v3_analysis.json)、[第二轮](D:/大模型项目/text2sql-posttrain/reports/sft_round2_formal_v1_analysis.json) |
| schema、assistant标签、EOS | [schema 第89行](D:/大模型项目/text2sql-posttrain/src/data/schema_serialization.py:89)、[训练特征 第80行](D:/大模型项目/text2sql-posttrain/src/data/sft_features.py:80) |
| 有效位置logits、加权累积、正式更新 | [目标loss 第34行](D:/大模型项目/text2sql-posttrain/src/train/target_loss.py:34)、[加权归一化 第17行](D:/大模型项目/text2sql-posttrain/src/train/sft_round2_weight_core_v1.py:17)、[训练循环 第339行](D:/大模型项目/text2sql-posttrain/src/train/sft_round2_formal_v1_worker.py:339) |
| 原库只读执行与TEXT解码版本 | [开发阶段strict执行器 第64行（历史）](D:/大模型项目/text2sql-posttrain/src/eval/spider_pair_worker.py:64)、[最终surrogateescape执行器 第64行](D:/大模型项目/text2sql-posttrain/src/eval/spider_pair_worker_decode_v2.py:64)；TEXT解码从新执行器第67行起，BLOB保持bytes |
| 两版本共用的固定结果比较器 | [比较器 第77行](D:/大模型项目/text2sql-posttrain/third_party/test-suite-sql-eval/exec_eval.py:77) |

**当前结论：** SFT训练、开发对照、最终两模型报告及项目整理已完成。没有开展DPO、线上强化学习或部署；已使用的最终结果不用于下一轮调参。旧日志、失败源码、原adapter及本次更新前文档备份保留。

**下一步可复制提示词：** 请基于这份项目手册做一次模拟面试，一次只问一个问题，优先考察最终评测口径、assistant-only损失、梯度累积和第二轮负结果。等我作答后指出遗漏、给出参考答案，并带我阅读相关核心代码。不要启动训练、生成或评测，也不要根据最终集调参。
'''
    handbook = intro+work+internal+limits+final+resume+qa+ending
    development = section(README.read_text(encoding='utf-8'), '## 三模型对照', '## 数据来源与划分')
    readme = f'''# QLoRA Text-to-SQL 微调与评测

输入自然语言问题和数据库schema，让Qwen2.5-Coder-1.5B-Instruct生成一条SQLite SQL。个人学习项目已在Windows单卡RTX 4070 Laptop 8GB上完成两轮SFT、开发对照、错误复盘和最终留出集评测。

**统一阅读入口：[项目总结与简历面试手册](docs/项目总结与简历面试手册.md)。** 包含真实结果、简历写法、两分钟讲稿和85道带答案的项目面试题，不能穷尽所有延伸追问。

截至{date}，最终1034题上{brief}；可执行率由{100*be/COUNT:.2f}%变为{100*se/COUNT:.2f}%。最终集取自官方dev，作为项目留出集，采用单份原SQLite EX，不是官方隐藏test或完整test-suite成绩。SFT阶段已完成；DPO、在线RL、生产部署、跨机器开箱复现尚未完成。

## 最终留出集对照：1034题 / 20库

第一轮796步adapter在最终模型推理前由开发结果选定，没有根据最终分数调整模型或prompt。两模型保持同一revision、NF4 4-bit、tokenizer、完整模板、单次贪心和256新token上限。

旧strict UTF-8评分在454条记录后因wta_1参考结果TEXT解码故障停止。经用户明确授权，执行器仅将TEXT改为可逆的surrogateescape解码，BLOB保持bytes；SQL、数据库、比较器、排序及超时不变。新目录对两模型全部1034条固定预测重新评分，不混用旧分数、不重新生成。执行协议有此兼容性修订，不能称完全未变；最终两模型使用同一修订版本。

{overall}

首轮相对基座{pair['improved']}题改善、{pair['regressed']}题退化，EX{change}；第二轮没有参加最终评测。各难度、失败类型、配对和成本完整表见统一手册与[最终对照JSON](outputs/final_holdout_decode_v4/comparison.json)。

基座原616题因遥测超时停止，修复后逐字保留616条，仅续接418条；首轮完整生成1034条。每题一次、不择优，旧失败记录保留。基座生成控制器两段合计{seconds(old_seconds+new_seconds)}，首轮{seconds(sft_seconds)}；生成及评分各段记录时长之和为{seconds(activity)}，含旧严格解码失败评分。部分CPU评分与GPU生成重叠，这个求和不是端到端墙钟历时，也不包含预检、兼容性验证、修复和等待。不同段的题目/负载不同，不能据此声称修复带来性能加速。

## 内部开发验证：625题三模型对照
{development}
内部625题保留原strict UTF-8协议的历史数字；最终1034题既换了题目集合，又有公开的TEXT解码修订，不能将两组百分比当成同一队列、同一完整执行协议直接比较。

## 数据划分与训练方法

| 用途 | 来源 | 题数 / 数据库数 | 边界 |
| --- | --- | --- | --- |
| 训练 | train_spider训练部分；隔离3条未解决执行问题、去掉8条完全重复 | 6364 / 126 | 梯度更新和训练侧改进数据 |
| 内部开发验证 | train_spider按数据库划出，seed42 | 625 / 14 | 错误分析与方案选择，不做梯度更新 |
| 项目最终留出 | 官方dev，不是隐藏test | 1034 / 20 | 完成固定方案两模型报告，不用于训练、偏好数据或调参 |

早期数据完整性、schema和匿名查重检查读取过dev记录，不能说从未读取最终题字节；它没有用于模型开发或调参。使用train_spider，不使用train_others。见[划分清单](reports/spider_split_manifest.json)与[清洗记录](reports/spider_gold_manifest.json)。

固定模型revision为`2e1fd397ee46e1388853d2af2c993145b0f1098a`；NF4双量化、BF16、SDPA。两轮均从原模型新建LoRA：rank8、alpha16、dropout0，9,232,384可训练参数；assistant-only监督，batch1、累积8、1轮796次更新，峰值学习率1e-4、warmup24。最长3761-token训练样本完整保留，通过梯度检查点、答案位置logits和缓存释放控制显存；200/400/600/796步保存并恢复完整训练状态。第二轮唯一主动变化是2528题权重2、其余3836题权重1，并按整组加权有效答案token数归一化。

## 只读查看结果

以下PowerShell命令仅读取现有JSON，不加载模型、执行SQL或重新评分；先进入你自己的项目根目录。

```powershell
$final = Get-Content -LiteralPath .\\outputs\\final_holdout_decode_v4\\comparison.json -Raw -Encoding utf8 | ConvertFrom-Json
$final.overall
$final.by_hardness
$final.failure_types
```

## 代码入口与复现边界

**下表仅供源码导航，不是让已有一次性目录再执行一遍。** 当前入口绑定Windows解释器、Job、冻结文件哈希和历史材料；尚未提供干净机器的一键复现，不能只复制配置就运行。

| 环节 | 入口 |
| --- | --- |
| 第一轮 / 第二轮正式SFT | [首轮](src/train/sft_round1_resource_v3.py)、[第二轮](src/train/sft_round2_formal_v1.py) |
| 最终生成预检与冻结 | [基座续接](src/eval/prepare_final_holdout_resume_v3.py)、[SFT独立入口](src/eval/prepare_final_holdout_resume_v4.py) |
| 最终基座续接 / 首轮生成 | [基座](src/infer/final_holdout_resume_v3.py)、[首轮SFT](src/infer/final_holdout_resume_v4.py) |
| TEXT修订授权与冻结 | [prepare_final_decode_v4.py](src/eval/prepare_final_decode_v4.py) |
| 最终统一EX与已存结果配对 | [evaluate_final_decode_v4.py](src/eval/evaluate_final_decode_v4.py) |

历史[复现说明](docs/reproduction.md)用于理解已验证入口与本机限制；其中运行命令会写记录，`--check`也不一定只读。原数据、模型和部分输出由[.gitignore](.gitignore)排除；旧[依赖锁](requirements.lock.txt)缺PEFT，不能当作完整安装保证。未重新验证另一台机器或Linux/WSL全流程。

## 结果边界与历史材料

- 可执行不等于答对；单库EX匹配也不等于普遍SQL语义等价。固定比较器允许全局列置换，双空结果有局限；没有运行完整多库test-suite。
- 开发集分母625，最终集分母1034；模型SQL错误留在对应分母，基础设施失败停止，完整收尾后才报告全量成绩。
- 最终TEXT使用可逆surrogateescape，BLOB不变；没有用ignore丢字节或replace合并坏字节。开发集旧分数没有暗中重算成新版本。
- 625题参与开发，最终结果不再用于调参。两组不是同题，不能直接用百分比差异声称退化；单种子不证明显著性，不能排除Spider预训练接触。
- allocated与reserved不能相加，整卡采样包含其他应用。旧616段缺少最终allocator计数，不能由续接段推断全程OOM计数。
- 代码和排查有AI辅助，陈述以能解释、能核查的工作为准；不声称原创QLoRA、生产上线或独立手写全部框架。

旧[两分钟讲稿](docs/project_talk.md)、[简历初稿](reports/sft_round2_review_v1/02_简历项目初稿.md)和[核心复盘](reports/sft_round2_review_v1/01_实验结论与核心教程.md)属于最终评测前的开发阶段历史材料，当前展示统一使用本手册。历史[README快照](docs/history/README_before_sft_showcase_20261030.md)、[实验日志](reports/experiment_log.md)、[原616题故障](outputs/final_holdout_v1/failure_analysis.json)及全部旧账本保留。

下一步以模拟面试和关键代码走读为主；新算法研究应另立实验方案和独立评估边界。
'''
    return readme, handbook


def validate_documents(readme, handbook):
    questions = [int(value) for value in re.findall(r'^\*\*Q(\d{2})[^\n]*\*\*$', handbook, re.M)]
    require(questions == list(range(1, 86)), 'All original 82 questions plus Q83/Q84/Q85 must remain in order')
    for value in ('372/625＝59.52%', '457/625＝73.12%', '456/625＝72.96%', '16题改善、17题退化'):
        require(value in handbook and value in readme, f'Internal validation evidence missing: {value}')
    for forbidden in ('在读取最终题目之前', '没有最终EX', '最终对照尚未完成', '完整最终评测尚未完成'):
        require(forbidden not in handbook and forbidden not in readme, f'Obsolete claim remains: {forbidden}')
    links = []
    for text, source in ((readme, README), (handbook, BOOK)):
        require(text.count('```') % 2 == 0, 'Unbalanced code fences')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('http://', 'https://', '#')):
                continue
            match = re.match(r'^(.*?):(\d+)$', target)
            line = int(match[2]) if match else None
            target = match[1] if match else target
            path = Path(target) if re.match(r'^[A-Za-z]:/', target) else source.parent/target
            require(path.is_file(), f'Document link missing: {path}')
            if line is not None:
                require(1 <= line <= len(path.read_text(encoding='utf-8-sig').splitlines()), f'Invalid code line: {path}:{line}')
            links.append(str(path))
    for text in (readme, handbook):
        require(all(value in text for value in ('454', 'surrogateescape', 'BLOB', 'strict UTF-8', '修订')),
                'Lossless decoding revision disclosure missing')
    review_evidence = {
        'H01': (handbook, 'SFT训练、625题开发验证及1034题最终留出集两模型对照均已完成'),
        'H02': (handbook, '### 4 内部开发验证：三模型对照'),
        'H03': (handbook, '### 6 最终留出集：1034题、20库，基座对首轮SFT'),
        'H04': (handbook, '早期脚本为数据完整性、schema检查和匿名查重读取过官方dev记录'),
        'H05': (handbook, '基座两段逐题生成记录的allocated/reserved峰值'),
        'H06': (handbook, '## 二 写简历时如何描述这个项目'),
        'H07': (handbook, '### 1 两分钟项目讲解稿'),
        'H08': (handbook, '**Q85 必会｜为什么要修订TEXT解码'),
        'H09': (handbook, '**下一步可复制提示词：** 请基于这份项目手册做一次模拟面试'),
        'R01': (readme, '**统一阅读入口：'),
        'R02': (readme, '## 最终留出集对照：1034题 / 20库'),
        'R03': (readme, '| 项目最终留出 | 官方dev，不是隐藏test | 1034 / 20 |'),
        'R04': (readme, '**下表仅供源码导航，不是让已有一次性目录再执行一遍。**'),
    }
    for item, (text, anchor) in review_evidence.items():
        require(anchor in text, f'Review item lacks its document section: {item}')
    return {'question_count': 85, 'priority_question_count': handbook.count(' 必会｜'),
            'local_link_count': len(links), 'obsolete_status_checks': 'pass',
            'internal_results_preserved': True, 'all_13_review_items_applied': True,
            'review_section_anchors': {item: anchor for item, (_, anchor) in review_evidence.items()},
            'authorized_decoder_revision_disclosed': True}


def main():
    data = load_verified()
    readme, handbook = build_documents(data)
    checks = validate_documents(readme, handbook)
    before = {path.relative_to(ROOT).as_posix(): digest(path) for path in (README, BOOK)}
    log = {'status': 'writing', 'created_local': datetime.now(timezone(timedelta(hours=8))).isoformat(),
           'completion_audit_sha256': digest(AUDIT), 'input_sha256': data[-1],
           'document_before_sha256': before, 'source_sha256': digest(Path(__file__)),
           'backup_overwritten': False, 'model_or_sql_execution': False, 'checks': checks,
           'review_items': [row['id'] for row in read(REVIEW)['mandatory_updates_after_success']]}
    write_json(LOG, log, True)
    try:
        for path, contents in ((README, readme), (BOOK, handbook)):
            pending = path.with_suffix(path.suffix+'.finalization.pending')
            with pending.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(contents); stream.flush(); os.fsync(stream.fileno())
            os.replace(pending, path)
        require(README.read_text(encoding='utf-8') == readme and BOOK.read_text(encoding='utf-8') == handbook,
                'Written documents differ from validated contents')
        for path, expected in data[-1].items():
            require(digest(ROOT/path) == expected, f'Existing evidence/backup changed during documentation: {path}')
        log.update(status='pass', document_after_sha256={path.relative_to(ROOT).as_posix(): digest(path) for path in (README, BOOK)},
                   evidence_and_backups_unchanged=True)
    except BaseException as error:
        log.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        write_json(LOG, log)
    print(json.dumps({'status': 'pass', 'documents': list(log['document_after_sha256']), **checks}, ensure_ascii=False))


if __name__ == '__main__':
    main()
