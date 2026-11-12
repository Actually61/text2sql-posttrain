$ErrorActionPreference = 'Stop'
# Read existing logs only. Never import Python/Torch, load a model, or call a worker.
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$runRoot = Join-Path $projectRoot 'outputs\runs\sft_resume_pilot_v1'
$destination = Join-Path $PSScriptRoot 'evidence.json'
if (Test-Path -LiteralPath $destination) { throw 'Evidence already exists; do not overwrite.' }
function Read-Json([string]$path) { Get-Content -LiteralPath $path -Raw | ConvertFrom-Json }
function Hash([string]$path) { (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
$manifest = Read-Json (Join-Path $runRoot 'artifact_sha256.json')
$artifactCount = 0
foreach ($property in $manifest.PSObject.Properties) {
    if ((Hash (Join-Path $runRoot $property.Name)) -ne $property.Value) { throw "Changed artifact: $($property.Name)" }
    $artifactCount++
}
$protected = @{}
foreach ($file in Get-ChildItem -LiteralPath $runRoot -Recurse -File) {
    $protected[$file.FullName] = Hash $file.FullName
}
$sources = @('src/train/token_accumulation.py', 'src/train/target_loss.py',
    'src/train/qlora_long_probe.py', 'src/train/windows_gpu_memory.py',
    'src/train/sft_train.py', 'src/train/sft_resume_pilot.py', 'src/train/sft_update_budget.py',
    'configs/sft_round1.json', 'configs/sft_resume_pilot.json',
    'reports/sft_resume_pilot_analysis.json', 'reports/sft_resume_pilot_walkthrough.md')
foreach ($name in $sources) { $path = Join-Path $projectRoot $name; $protected[$path] = Hash $path }
$controller = Read-Json (Join-Path $runRoot 'result.json')
$ledger = Read-Json (Join-Path $runRoot 'progress.json')
if ($ledger.completed_updates -ne 3 -or $ledger.attempted_updates -ne 3 -or !$ledger.failed) {
    throw 'Unexpected ledger: stop diagnosis rather than reinterpret the failed run.'
}
$segments = @()
foreach ($number in 1, 2) {
    $dir = Join-Path $runRoot "segment_$number"
    $phases = @(Read-Json (Join-Path $dir 'phase_memory.json'))
    $forwards = @(Get-Content -LiteralPath (Join-Path $dir 'micro_forwards.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
    $pdhPath = Join-Path $dir 'gpu_process_memory.json'
    $pdh = if (Test-Path -LiteralPath $pdhPath) { Read-Json $pdhPath } else { $null }
    $phaseRows = @()
    foreach ($phase in $phases) {
        $samples = if ($null -ne $pdh) { @($pdh.samples | Where-Object { $_.stage -eq $phase.phase -and $_.status -eq 'ok' }) } else { @() }
        $forward = $null
        if ($phase.phase -match '^update_(\d+)_micro_(\d+)_L(\d+)$') {
            $updateNumber = [int]$Matches[1]; $microNumber = [int]$Matches[2]
            $forward = @($forwards | Where-Object { $_.update -eq $updateNumber -and $_.microbatch -eq $microNumber })[0]
        }
        $phaseRows += [ordered]@{
            phase = $phase.phase; seconds = $phase.seconds
            peak_allocated_mib = $phase.peak_allocated_mib; peak_reserved_mib = $phase.peak_reserved_mib
            end_allocated_mib = $phase.end_allocated_mib; end_reserved_mib = $phase.end_reserved_mib
            source_index = $(if ($forward) { $forward.source_index } else { $null })
            valid_target_count = $(if ($forward) { $forward.valid_target_count } else { $null })
            pid_valid_samples = $samples.Count
            pid_dedicated_peak_mib = $(if ($samples.Count) { ($samples.dedicated_bytes | Measure-Object -Maximum).Maximum / 1MB } else { $null })
            pid_shared_peak_mib = $(if ($samples.Count) { ($samples.shared_bytes | Measure-Object -Maximum).Maximum / 1MB } else { $null })
            pid_sample_collection_total_seconds = $(if ($samples.Count) { ($samples.collection_seconds | Measure-Object -Sum).Sum } else { $null })
        }
    }
    $timePath = Join-Path $dir 'process_timing.json'
    $segments += [ordered]@{
        number = $number; resources_at_launch = Read-Json (Join-Path $dir 'resources.json')
        wall_seconds = $(if (Test-Path -LiteralPath $timePath) { (Read-Json $timePath).worker_wall_seconds } else { $null })
        complete_result_exists = Test-Path -LiteralPath (Join-Path $dir 'result.json')
        history = @(Read-Json (Join-Path $dir 'history.json')); phases = $phaseRows
        last_recorded_phase = Read-Json (Join-Path $dir 'current_phase.json')
        last_recorded_forward = $forwards[-1]
        memory_coverage = $(if ($null -ne $pdh) { 'Completed phase intervals plus sampled PID memory; not continuous instantaneous PID peaks' } else { 'Completed phase intervals only; no complete PID telemetry; full-run peaks unknown' })
    }
}
$checkpointRoot = Join-Path $runRoot 'checkpoints\update_000002'
$checkpoint = Read-Json (Join-Path $checkpointRoot 'complete.json')
$checkpointCount = 0
foreach ($p in $checkpoint.files.PSObject.Properties) {
    if ((Hash (Join-Path $checkpointRoot $p.Name)) -ne $p.Value) { throw "Checkpoint mismatch: $($p.Name)" }
    $checkpointCount++
}
$restore = Read-Json (Join-Path $runRoot 'segment_2\restore_verification.json')
$slow = $segments[0].phases | Where-Object phase -eq 'update_2_micro_2_L3757'
$ledgerHash = Hash (Join-Path $runRoot 'progress.json')
foreach ($entry in $protected.GetEnumerator()) { if ((Hash $entry.Key) -ne $entry.Value) { throw "Changed during read: $($entry.Key)" } }
$relativeHashes = [ordered]@{}
foreach ($key in ($protected.Keys | Sort-Object)) { $relativeHashes[$key.Substring($projectRoot.Length + 1).Replace('\', '/')] = $protected[$key] }
$report = [ordered]@{
    diagnosis_date = '2026-10-29'; status = 'read_only_audit_pass'
    original_run_status = $controller.status; controller_seconds = $controller.controller_seconds
    termination = $controller.error; ledger = $ledger; ledger_sha256 = $ledgerHash
    artifact_hashes_verified = $artifactCount; checkpoint_payload_hashes_verified = $checkpointCount
    protected_files_unchanged = $protected.Count; protected_sha256 = $relativeHashes
    longest_interval_fraction_of_segment_1 = $slow.seconds / $segments[0].wall_seconds
    latest_durable_update = 2; completed_but_unsaved_updates = 1; restore_verification = $restore.verification
    segments = $segments
    limitations = @('Memory metric peaks can occur at different instants; do not add them.',
        'Phase time contains synchronization and logging boundaries, not pure kernel time.',
        'No allocator fragmentation counters or driver paging events were recorded.',
        'Segment 2 incomplete: finished intervals give lower bounds, missing samples are not zero.',
        'This audit read logs/source/checkpoint bytes for hashes only; no dataset examples or tensor states deserialized.')
    scope = @{ model_loaded = $false; cuda_initialized = $false; new_optimizer_updates = 0;
        datasets_read = $false; final_examples_read = $false; old_run_modified = $false }
}
$json = $report | ConvertTo-Json -Depth 16
$stream = [IO.File]::Open($destination, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try { $bytes = [Text.Encoding]::UTF8.GetBytes($json + "`n"); $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Dispose() }
[pscustomobject]@{ status=$report.status; artifact_hashes_verified=$artifactCount; checkpoint_hashes_verified=$checkpointCount;
    unchanged_files=$protected.Count; completed_updates=$ledger.completed_updates; output=$destination } | ConvertTo-Json
