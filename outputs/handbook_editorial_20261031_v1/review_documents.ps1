$ErrorActionPreference = 'Stop'
$p = 'D:\大模型项目\text2sql-posttrain'
$d = Join-Path $p 'outputs\handbook_editorial_20261031_v1'
$bookPath = Join-Path $p 'docs\项目总结与简历面试手册.md'
$old = [IO.File]::ReadAllText("$d\handbook_before.md").Replace("`r`n", "`n")
$book = [IO.File]::ReadAllText($bookPath).Replace("`r`n", "`n")
$checks = [Collections.Generic.List[object]]::new()
function Check([string]$name, [bool]$ok, $detail) {
    $checks.Add([ordered]@{name=$name; pass=$ok; detail=$detail})
}
function Region([string]$text, [string]$start, [string]$end) {
    $a = $text.IndexOf($start)
    $b = $text.IndexOf($end, $a + $start.Length)
    if ($a -lt 0 -or $b -lt $a) { throw 'Missing document region' }
    return $text.Substring($a, $b-$a)
}
$before = Get-Content "$d\before_manifest.json" -Raw -Encoding utf8 | ConvertFrom-Json
Check 'original_handbook_backup_sha256' ((Get-FileHash "$d\handbook_before.md").Hash.ToLower() -eq $before.files[0].sha256) $before.files[0].sha256
Check 'original_readme_backup_sha256' ((Get-FileHash "$d\README_before.md").Hash.ToLower() -eq $before.files[1].sha256) $before.files[1].sha256
Check 'results_and_resume_sections_unchanged' ((Region $old '## 一 项目做了什么 有什么结果' '## 三 项目讲解文稿与面试问答') -ceq (Region $book '## 一 项目做了什么 有什么结果' '## 三 项目讲解文稿与面试问答')) 'Only introductory reading guide and section III were edited in the handbook.'
foreach ($row in $before.files[2..3]) {
    Check 'experimental_evidence_sha256_unchanged' ((Get-FileHash -LiteralPath $row.path).Hash.ToLower() -eq $row.sha256) $row.path
}
$qs = [regex]::Matches($book, '(?m)^\*\*Q(\d{2})｜(.+)\*\*$')
$ids = @($qs | ForEach-Object { [int]$_.Groups[1].Value })
Check 'twenty_sequential_unique_questions' (($ids -join ',') -eq ((1..20) -join ',')) $ids
$indexStart = $book.IndexOf('### 7 代码与实验记录')
for ($i=0; $i -lt $qs.Count; $i++) {
    $end = if ($i+1 -lt $qs.Count) { $qs[$i+1].Index } else { $indexStart }
    $answer = $book.Substring($qs[$i].Index+$qs[$i].Length, $end-$qs[$i].Index-$qs[$i].Length).Trim()
    Check ('answer_present_Q{0:D2}' -f ($i+1)) ($answer.Length -gt 60) $answer.Length
}
Check 'old_question_count_and_checklist_removed' (-not ($book -match 'Q(?:2[1-9]|[3-8][0-9])｜|85道|必会｜|### 11 面试前')) '20 questions; no second checklist.'
$readme = [IO.File]::ReadAllText("$p\README.md")
$readmeOld = [IO.File]::ReadAllText("$d\README_before.md")
Check 'readme_only_question_count_changed' ($readme -ceq $readmeOld.Replace('85道带答案的项目面试题','20道带答案的核心项目面试题')) '85 to 20; no other README change.'
$talk = Region $book '### 1 两分钟项目讲解稿' '### 2 项目选择与数据处理'
$paragraphs = @([regex]::Matches($talk, '(?m)^> .+$'))
Check 'talk_five_connected_paragraphs' ($paragraphs.Count -eq 5) $paragraphs.Count
Check 'no_standalone_dataset_paragraph_or_infra_in_talk' (-not ($talk -match '(?m)^> 数据|616|418|surrogateescape|nvidia-smi|PID')) 'Dataset counts are woven into method/results; operational details remain in report.'
Check 'balanced_code_fences' (([regex]::Matches($book,'(?m)^```').Count % 2) -eq 0) ([regex]::Matches($book,'(?m)^```').Count)
$links = [regex]::Matches($book, '\]\((D:/[^)]+)\)')
foreach ($link in $links) {
    $target = $link.Groups[1].Value
    $line = 0
    if ($target -match '^(.+):(\d+)$') { $target=$Matches[1]; $line=[int]$Matches[2] }
    $exists = Test-Path -LiteralPath $target -PathType Leaf
    Check 'local_evidence_link_exists' $exists $target
    if ($line -gt 0 -and $exists) {
        Check 'linked_line_exists' ([IO.File]::ReadAllLines($target).Length -ge $line) "$target`:$line"
    }
}
$comparison = Get-Content "$p\outputs\final_holdout_decode_v4\comparison.json" -Raw -Encoding utf8 | ConvertFrom-Json
$o = $comparison.overall
Check 'final_numeric_evidence' ($o.count -eq 1034 -and $o.base.execution_correct -eq 596 -and $o.sft.execution_correct -eq 749 -and $o.sft.executable_predictions -eq 958 -and $o.improved -eq 205 -and $o.regressed -eq 52) 'Read existing comparison only; no new scoring.'
foreach ($number in @('57.64%','72.44%','14.80个百分点','78.92%','92.65%','73.12%','72.96%')) {
    Check 'oral_script_retains_verified_key_numbers' $talk.Contains($number) $number
}
$cases = Get-Content "$p\reports\sft_round2_review_v1\cases.json" -Raw -Encoding utf8 | ConvertFrom-Json
$case = @($cases | Where-Object { $_.source_index -eq 5839 })
Check 'real_sql_case_matches_existing_review' ($case.Count -eq 1 -and $case[0].models.round2.ex_match -eq $true -and $case[0].models.round2.sql -match 'count\(\*\)' -and $case[0].gold_sql -match 'COUNT\(DISTINCT') '5839: current-database EX pass does not establish distinct-workshop semantics; hypothetical counterexample not newly executed.'
$oldQRegion = Region $old '### 2 项目定位与个人工作' '### 11 面试前的核对清单'
$newQRegion = Region $book '### 2 项目选择与数据处理' '### 7 代码与实验记录'
$result = [ordered]@{
    status = $(if (@($checks | Where-Object {-not $_.pass}).Count -eq 0) {'pass'} else {'fail'})
    client_date='2026-10-31'
    scope='documentation-only editorial revision; no generation, training, scoring, model loading or app control'
    handbook_sha256=(Get-FileHash $bookPath).Hash.ToLower()
    readme_sha256=(Get-FileHash "$p\README.md").Hash.ToLower()
    question_count_before=85; question_count_after=$qs.Count
    qa_characters_before=$oldQRegion.Length; qa_characters_after=$newQRegion.Length
    speech_nonspace_characters=([regex]::Replace(($paragraphs | ForEach-Object {$_.Value}) -join '', '\s|>', '')).Length
    local_evidence_links=$links.Count
    checks=$checks
}
$result | ConvertTo-Json -Depth 8 | Set-Content "$d\review_result.json" -Encoding utf8
[pscustomobject]$result | Select-Object status,handbook_sha256,readme_sha256,question_count_after,qa_characters_before,qa_characters_after,speech_nonspace_characters,local_evidence_links | ConvertTo-Json
if ($result.status -ne 'pass') { throw 'Documentation review failed; see review_result.json' }
