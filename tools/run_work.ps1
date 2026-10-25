# How to run: a thin wrapper for the work machine. It calls ONLY existing commands, stops at the first error and says what to look at. PowerShell, in the project folder:
#     .\tools\run_work.ps1 -Step full                     # THE WHOLE RUN, no questions: probe, check, prepare, baseline, train, evaluate, compare, report
#     .\tools\run_work.ps1 -Step full -Epochs 3           # the same with 3 epochs (default 2)
#     .\tools\run_work.ps1 -Step full -WithTest           # ... and, after compare, the ONE-TIME Test scoring (only on the final chosen run!)
#     .\tools\run_work.ps1 -Step full -Dry                # print what it would run (and the estimated hours), run nothing
# Run the SAME command again to RESUME after a stop or a crash: finished steps are skipped, an unfinished training continues from its last checkpoint. The state is kept in
# <ExperimentsDir>\full_run_state.json (delete or rename that file to start a fresh run; a run with other -Epochs keeps the numbers of the stored state).
# Single steps (for the details see docs\WORK_RUNBOOK.md): -Step probe | check | prepare | benchmark | baseline | train | evaluate | compare | report | final-test
#     .\tools\run_work.ps1 -Step benchmark -Config configs\my_work_cpu.yaml
# -Config  the YAML you filled in (default configs\work_cpu.yaml; it needs 4 edits: the data folder and three column names).
# -Threads the number of PHYSICAL cores (default: found by tools\probe_env.py).
# UNTESTED on the work machine.
param(
    [Parameter(Mandatory = $true)][ValidateSet('full', 'probe', 'check', 'prepare', 'benchmark', 'baseline', 'train', 'evaluate', 'compare', 'report', 'final-test')][string]$Step,
    [string]$Config = 'configs\work_cpu.yaml',
    [int]$Threads = 0,
    [int]$Epochs = 2,
    [switch]$WithTest,
    [switch]$Dry,
    [string]$Experiment = '',
    [string]$BaselineFrom = '',
    [string]$ExperimentsDir = 'data\work_cpu\experiments',
    [string]$ModelPath = 'models\whisper-large-v3-turbo-ivrit',
    [string[]]$Set = @()
)

$ErrorActionPreference = 'Continue'   # the program writes its log on the error stream; a real failure is detected by the exit code below
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = 'src'
$py = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Host "STOP: $py does not exist. Create the virtual environment first (docs\WORK_RUNBOOK.md, step 0)."; exit 2 }

# The numbers used when no benchmark exists: measured on the home laptop with the turbo model and the encoder cache (docs\WORK_RUNBOOK.md, time calculator).
$ConstStep = 26.7; $ConstValLoss = 0.67; $ConstDecode = 2.9; $ConstEncode = 14.8; $ConstExtras = 600.0

function Invoke-Step {
    param([string[]]$Arguments)
    Write-Host ""
    Write-Host ("RUNNING: python " + ($Arguments -join ' '))
    if ($Dry) { Write-Host "(dry run: not started)"; return }
    Write-Host ""
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "STOP: the step failed with exit code $LASTEXITCODE. Read the last lines above, then docs\WORK_RUNBOOK.md (failure guide). Run the same command again to continue."
        exit $LASTEXITCODE
    }
}

function Resolve-Threads {
    if ($script:Threads -ge 1) { return }
    $script:Threads = 4
    if (-not $Dry) {
        $text = (& $py tools\probe_env.py --model $ModelPath) -join "`n"
        if ($text -match '(\d+) physical cores') { $script:Threads = [int]$Matches[1] }
        elseif ($text -match '(\d+) logical threads') { $script:Threads = [Math]::Max(1, [int]([int]$Matches[1] / 2)) }
    }
    Write-Host "Threads: $script:Threads (the physical cores found by the probe; give -Threads N to choose)."
}

function Experiment-Names { if (Test-Path $ExperimentsDir) { @(Get-ChildItem $ExperimentsDir -Directory | Where-Object { $_.Name -match '^exp_\d+$' } | ForEach-Object { $_.Name }) } else { @() } }
function Latest-Experiment { $n = Experiment-Names; if ($n.Count -gt 0) { ($n | Sort-Object | Select-Object -Last 1) } else { '(none yet)' } }
function Status-Of([string]$id) {
    $f = Join-Path $ExperimentsDir "$id\status.json"
    if (Test-Path $f) { (Get-Content $f -Raw | ConvertFrom-Json).status } else { 'missing' }
}
function Run-And-Capture-Id {
    # Runs a step that creates an experiment and returns its id, even when the step fails (so that a rerun can resume it).
    param([string[]]$Arguments)
    $before = Experiment-Names
    $code = 0
    Write-Host ""
    Write-Host ("RUNNING: python " + ($Arguments -join ' '))
    if ($Dry) { Write-Host "(dry run: not started)"; return '<new experiment>' }
    Write-Host ""
    & $py @Arguments | Out-Host
    $code = $LASTEXITCODE
    $new = @(Experiment-Names | Where-Object { $before -notcontains $_ })
    $script:LastCode = $code
    if ($new.Count -gt 0) { return ($new | Select-Object -Last 1) } else { return '' }
}

$extra = @(); foreach ($s in $Set) { $extra += @('--set', $s) }
$statePath = Join-Path $ExperimentsDir 'full_run_state.json'

function Load-State {
    if (Test-Path $statePath) { return (Get-Content $statePath -Raw | ConvertFrom-Json) }
    return $null
}
function Save-State($state) {
    if ($Dry) { return }
    New-Item -ItemType Directory -Force -Path $ExperimentsDir | Out-Null
    $full = $statePath
    if (-not [System.IO.Path]::IsPathRooted($full)) { $full = Join-Path (Get-Location).Path $statePath }
    [System.IO.File]::WriteAllText($full, ($state | ConvertTo-Json), (New-Object System.Text.UTF8Encoding $false))  # UTF-8 without a BOM, so that every tool can read it
}

function Get-Estimate([int]$trainClips, [int]$valClips, [int]$steps, [int]$evals, [bool]$cache) {
    $s = $ConstStep; $l = $ConstValLoss; $d = $ConstDecode; $e = $ConstEncode; $source = 'the constants of docs\WORK_RUNBOOK.md (home laptop)'
    $bench = @(Get-ChildItem $ExperimentsDir -Recurse -Filter benchmark.json -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1)
    if ($bench.Count -gt 0) {
        $b = (Get-Content $bench[0].FullName -Raw | ConvertFrom-Json).measured
        $s = [double]$b.seconds_per_step; $l = [double]$b.seconds_per_val_loss_example; $d = [double]$b.seconds_per_decode_clip
        $source = "the benchmark $($bench[0].Directory.Parent.Name)"
    }
    $seconds = $steps * $s + $evals * $valClips * $l + 3 * $valClips * $d + $ConstExtras
    if ($cache) { $seconds += ($trainClips + $valClips) * $e }
    return @{ hours = [Math]::Round($seconds / 3600.0, 1); source = $source }
}

switch ($Step) {
    'probe' {
        Invoke-Step @('tools\probe_env.py', '--model', $ModelPath)
        Write-Host "LOOK AT: RAM free (close other programs if it is below about 8 GB), Disk free, the model line (must say OK), the physical cores."
    }
    'check' {
        Invoke-Step (@('-m', 'stt', 'check', '--config', $Config) + $extra)
        Write-Host "LOOK AT: the last line must be 'Everything checked is fine.' Any PROBLEM line names what to fix."
    }
    'prepare' {
        Invoke-Step (@('-m', 'stt', 'prepare', '--config', $Config) + $extra)
        $latest = Latest-Experiment
        Write-Host "LOOK AT (in $ExperimentsDir\$latest\reports): load_report.txt, rejected.csv, split_report.txt, filter_report.txt, text_report.txt. Go / no-go rules: docs\WORK_RUNBOOK.md, step 3."
    }
    'benchmark' {
        Resolve-Threads
        Invoke-Step (@('-m', 'stt', 'benchmark', '--config', $Config, '--set', "train.cpu_threads=$Threads", '--set', 'train.hyperparams.warmup_steps=3') + $extra)
        $latest = Latest-Experiment
        Write-Host "LOOK AT: $ExperimentsDir\$latest\reports\benchmark.txt : the seconds per step, per Val-loss example and per decoded clip (time calculator: docs\WORK_RUNBOOK.md, step 4)."
        Write-Host "NOTE: the benchmark keeps a real model folder (about 7.4 GiB) under candidates\. Nothing is deleted by any command."
    }
    'baseline' {
        Resolve-Threads
        Invoke-Step (@('-m', 'stt', 'evaluate', '--config', $Config, '--set', "train.cpu_threads=$Threads") + $extra)
        $latest = Latest-Experiment
        Write-Host "LOOK AT: $ExperimentsDir\$latest\reports\evaluation_baseline.json (WER, CER) and review_baseline.txt (listen to a few clips). Write down the experiment number: $latest."
    }
    'train' {
        Resolve-Threads
        $expArgs = @(); if ($Experiment -ne '') { $expArgs = @('--experiment', $Experiment) }
        Invoke-Step (@('-m', 'stt', 'train', '--config', $Config, '--set', "train.cpu_threads=$Threads") + $extra + $expArgs)
        $latest = Latest-Experiment
        Write-Host "LOOK AT: $ExperimentsDir\$latest\metrics\trial_000\metrics.csv (the loss). Write down the experiment number: $latest. Next: -Step evaluate -Experiment $latest"
    }
    'evaluate' {
        Resolve-Threads
        if ($Experiment -eq '') { Write-Host "STOP: give -Experiment exp_NNNN, the experiment of the train step."; exit 2 }
        Invoke-Step (@('-m', 'stt', 'evaluate', '--config', $Config, '--set', "train.cpu_threads=$Threads", '--experiment', $Experiment) + $extra)
        Write-Host "LOOK AT: $ExperimentsDir\$Experiment\reports\comparison.json. Next: -Step compare -Experiment $Experiment"
    }
    'compare' {
        if ($Experiment -eq '') { Write-Host "STOP: give -Experiment exp_NNNN, the experiment where evaluate scored the trained model."; exit 2 }
        $compareArgs = @('tools\compare_evals.py', '--dir', $ExperimentsDir, '--experiment', $Experiment)
        if ($BaselineFrom -ne '') { $compareArgs += @('--baseline-from', $BaselineFrom) }
        Invoke-Step $compareArgs
    }
    'report' {
        $exp = $Experiment; if ($exp -eq '') { $exp = Latest-Experiment }
        Invoke-Step (@('-m', 'stt', 'report', '--config', $Config, '--experiment', $exp) + $extra)
        Write-Host "THE REPORT: $ExperimentsDir\$exp\reports\report.html (open it in a browser; it needs no network)."
    }
    'final-test' {
        Resolve-Threads
        if ($Experiment -eq '') { Write-Host "STOP: give -Experiment exp_NNNN, the experiment where train and evaluate ran."; exit 2 }
        Write-Host "The Test split is scored ONCE. A second call is refused (exit code 3)."
        Invoke-Step (@('-m', 'stt', 'final-test', '--config', $Config, '--set', "train.cpu_threads=$Threads", '--experiment', $Experiment) + $extra)
        Write-Host "LOOK AT: $ExperimentsDir\$Experiment\reports\test_result.json, then rebuild the report: -Step report -Experiment $Experiment"
    }
    'full' {
        Write-Host "=== FULL RUN: probe, check, prepare, baseline, train, evaluate, compare$(if ($WithTest) { ', final-test' }), report ==="
        Resolve-Threads
        $state = Load-State
        $resuming = $null -ne $state
        if (-not $resuming) { $state = [ordered]@{ epochs = $Epochs; threads = $Threads; prepare = ''; train_clips = 0; val_clips = 0; max_steps = 0; eval_every = 0; warmup = 0; baseline = ''; train = ''; evaluated = $false; tested = $false } }
        else { Write-Host "A state file was found ($statePath): this run CONTINUES it (epochs $($state.epochs)). To start fresh, rename that file." }
        # 1 probe (information only), 2 check (stops on a problem)
        Invoke-Step @('tools\probe_env.py', '--model', $ModelPath)
        Invoke-Step (@('-m', 'stt', 'check', '--config', $Config) + $extra)
        # 3 prepare: the Train size decides the number of steps
        if ($state.prepare -eq '') {
            $id = Run-And-Capture-Id (@('-m', 'stt', 'prepare', '--config', $Config) + $extra)
            if (-not $Dry) {
                if ($script:LastCode -ne 0) { Write-Host "STOP: prepare failed with exit code $script:LastCode. Read the lines above (docs\WORK_RUNBOOK.md, step 3)."; exit $script:LastCode }
                $summary = (Get-Content (Join-Path $ExperimentsDir "$id\reports\stage_results\prepare.json") -Raw | ConvertFrom-Json).summary
                $state.prepare = $id; $state.train_clips = [int]$summary.kept_train; $state.val_clips = [int]$summary.kept_val
                $perEpoch = [int][Math]::Ceiling($state.train_clips / 16.0)
                $state.max_steps = [int]($state.epochs * $perEpoch)
                $state.eval_every = [int][Math]::Max(1, [Math]::Floor($state.max_steps / 4))
                $state.warmup = [int][Math]::Min(5, [Math]::Floor($state.max_steps / 20))
                Save-State $state
                Write-Host "Read the five reports in $ExperimentsDir\$id\reports before trusting the run (split_report.txt: at least 5 lessons in Val and in Test)."
            }
        }
        if ($Dry) { $state.train_clips = 800; $state.val_clips = 100; $perEpoch = 50; $state.max_steps = $state.epochs * 50; $state.eval_every = [int][Math]::Max(1, [Math]::Floor($state.max_steps / 4)); $state.warmup = 5 }
        $evals = [int][Math]::Ceiling($state.max_steps / [double]$state.eval_every)
        $cacheOn = $false
        if (Test-Path $Config) { $cacheOn = ((Get-Content $Config -Raw) -match '(?s)encoder_cache:\s*\r?\n\s*enabled:\s*true') }
        $estimate = Get-Estimate $state.train_clips $state.val_clips $state.max_steps $evals $cacheOn
        Write-Host ""
        Write-Host "CHOSEN: $($state.train_clips) Train clips, $($state.val_clips) Val clips -> $($state.epochs) epochs = $($state.max_steps) steps (16 clips per step), a Val-loss point every $($state.eval_every) steps ($evals points), warmup $($state.warmup)."
        Write-Host "ESTIMATED TIME: about $($estimate.hours) hours (encoder cache $(if ($cacheOn) { 'ON' } else { 'OFF' }); numbers from $($estimate.source)). The computer must stay on."
        $common = @('--set', "train.cpu_threads=$Threads", '--set', "train.hyperparams.max_steps=$($state.max_steps)", '--set', "train.eval_every_steps=$($state.eval_every)", '--set', "train.hyperparams.warmup_steps=$($state.warmup)") + $extra
        # 4 baseline evaluate (the sanity check and the first fill of the Val part of the encoder cache)
        if ($state.baseline -eq '' -or (Status-Of $state.baseline) -ne 'completed') {
            $id = Run-And-Capture-Id (@('-m', 'stt', 'evaluate', '--config', $Config) + $common)
            if (-not $Dry) {
                if ($id -ne '') { $state.baseline = $id; Save-State $state }
                if ($script:LastCode -ne 0) { Write-Host "STOP: the baseline evaluation failed with exit code $script:LastCode. Run the same command again."; exit $script:LastCode }
                Write-Host "BASELINE: see $ExperimentsDir\$($state.baseline)\reports\evaluation_baseline.json and listen to 3 clips of review_baseline.txt. Stop (Ctrl+C) now if the WER is above 1.5 or the outputs are empty."
            }
        }
        # 5 train (a rerun continues the unfinished training from its last checkpoint)
        if ($state.train -eq '' -or (Status-Of $state.train) -ne 'completed') {
            $arguments = @('-m', 'stt', 'train', '--config', $Config) + $common
            if ($state.train -ne '') { $arguments += @('--experiment', $state.train) }
            $id = Run-And-Capture-Id $arguments
            if (-not $Dry) {
                if ($state.train -eq '' -and $id -ne '') { $state.train = $id; Save-State $state }
                if ($script:LastCode -ne 0) { Write-Host "STOP: the training stopped with exit code $script:LastCode. Run the same command again to continue it."; exit $script:LastCode }
            }
        }
        # 6 evaluate the trained model against the baseline
        if (-not $state.evaluated) {
            $trainId = $state.train; if ($Dry) { $trainId = '<train experiment>' }
            Invoke-Step (@('-m', 'stt', 'evaluate', '--config', $Config, '--experiment', $trainId) + $common)
            if (-not $Dry) { $state.evaluated = $true; Save-State $state }
        }
        # 7 compare
        $trainId = $state.train; $baseId = $state.baseline
        if ($Dry) { $trainId = '<train experiment>'; $baseId = '<baseline experiment>' }
        Invoke-Step @('tools\compare_evals.py', '--dir', $ExperimentsDir, '--experiment', $trainId, '--baseline-from', $baseId)
        # 8 the one-time Test scoring, only when asked
        if ($WithTest -and -not $state.tested) {
            Write-Host "The Test split is scored ONCE now (-WithTest). A second call is refused."
            Invoke-Step (@('-m', 'stt', 'final-test', '--config', $Config, '--experiment', $trainId) + $common)
            if (-not $Dry) { $state.tested = $true; Save-State $state }
        }
        # 9 report
        Invoke-Step (@('-m', 'stt', 'report', '--config', $Config, '--experiment', $trainId))
        Write-Host ""
        Write-Host "DONE. OPEN THE REPORT: $ExperimentsDir\$trainId\reports\report.html (in a browser; no network needed)."
    }
}
