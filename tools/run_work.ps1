# How to run: a thin wrapper for the work machine. It calls ONLY existing commands, one step at a time, stops at the first error and says what to look at.
#   PowerShell, in the project folder (the folder that contains tools\):
#     .\tools\run_work.ps1 -Step probe
#     .\tools\run_work.ps1 -Step check     -Config configs\my_work_cpu.yaml
#     .\tools\run_work.ps1 -Step prepare   -Config configs\my_work_cpu.yaml
#     .\tools\run_work.ps1 -Step benchmark -Config configs\my_work_cpu.yaml -Threads 8
#     .\tools\run_work.ps1 -Step baseline  -Config configs\my_work_cpu.yaml -Threads 8
#     .\tools\run_work.ps1 -Step train     -Config configs\my_work_cpu.yaml -Threads 8
#     .\tools\run_work.ps1 -Step train     -Config configs\my_work_cpu.yaml -Threads 8 -Experiment exp_0003     # CONTINUE a stopped training
#     .\tools\run_work.ps1 -Step evaluate  -Config configs\my_work_cpu.yaml -Threads 8 -Experiment exp_0003     # baseline + trained model of the train step
#     .\tools\run_work.ps1 -Step compare   -Experiment exp_0003 [-BaselineFrom exp_0002]
# -Threads is the number of PHYSICAL cores (tools\probe_env.py prints it). The steps are explained in docs\WORK_RUNBOOK.md. UNTESTED on the work machine.
param(
    [Parameter(Mandatory = $true)][ValidateSet('probe', 'check', 'prepare', 'benchmark', 'baseline', 'train', 'evaluate', 'compare')][string]$Step,
    [string]$Config = 'configs\my_work_cpu.yaml',
    [int]$Threads = 0,
    [string]$Experiment = '',
    [string]$BaselineFrom = '',
    [string]$ExperimentsDir = 'data\work_cpu\experiments'
)

$ErrorActionPreference = 'Continue'   # the program writes its log on the error stream; a real failure is detected by the exit code below
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = 'src'
$py = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Host "STOP: $py does not exist. Create the virtual environment first (docs\WORK_RUNBOOK.md, step 0)."; exit 2 }

function Invoke-Step {
    param([string[]]$Arguments)
    Write-Host ""
    Write-Host ("RUNNING: python " + ($Arguments -join ' '))
    Write-Host ""
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "STOP: the step failed with exit code $LASTEXITCODE. Read the last lines above, then docs\WORK_RUNBOOK.md (failure guide). Do not start the next step."
        exit $LASTEXITCODE
    }
}

function Need-Threads {
    if ($Threads -lt 1) { Write-Host "STOP: give -Threads N, the number of PHYSICAL cores (run -Step probe to see it)."; exit 2 }
}

function Latest-Experiment {
    if (Test-Path $ExperimentsDir) { (Get-ChildItem $ExperimentsDir -Directory | Sort-Object Name | Select-Object -Last 1).Name } else { '(none yet)' }
}

$threadSet = @('--set', "train.cpu_threads=$Threads")
$expArgs = @()
if ($Experiment -ne '') { $expArgs = @('--experiment', $Experiment) }

switch ($Step) {
    'probe' {
        Invoke-Step @('tools\probe_env.py')
        Write-Host "LOOK AT: RAM free (close other programs if it is below about 8 GB), Disk free, the model line (must say OK), the physical cores (use it for -Threads)."
    }
    'check' {
        Invoke-Step @('-m', 'stt', 'check', '--config', $Config)
        Write-Host "LOOK AT: the last line must be 'Everything checked is fine.' Any PROBLEM line names what to fix."
    }
    'prepare' {
        Invoke-Step @('-m', 'stt', 'prepare', '--config', $Config)
        $latest = Latest-Experiment
        Write-Host "LOOK AT (in $ExperimentsDir\$latest\reports): load_report.txt, rejected.csv, split_report.txt, filter_report.txt, text_report.txt. Go / no-go rules: docs\WORK_RUNBOOK.md, step 3."
    }
    'benchmark' {
        Need-Threads
        Invoke-Step (@('-m', 'stt', 'benchmark', '--config', $Config) + $threadSet + @('--set', 'train.hyperparams.warmup_steps=3'))
        $latest = Latest-Experiment
        Write-Host "LOOK AT: $ExperimentsDir\$latest\reports\benchmark.txt : the seconds per step, per Val-loss example and per decoded clip. Put them into the time calculator (docs\WORK_RUNBOOK.md, step 4)."
        Write-Host "NOTE: the benchmark keeps a real model folder (about 7.4 GiB) under candidates\. Nothing is deleted by any command."
    }
    'baseline' {
        Need-Threads
        Invoke-Step (@('-m', 'stt', 'evaluate', '--config', $Config) + $threadSet)
        $latest = Latest-Experiment
        Write-Host "LOOK AT: $ExperimentsDir\$latest\reports\evaluation_baseline.json (WER, CER) and review_baseline.txt (listen to a few clips). Write down the experiment number: $latest."
    }
    'train' {
        Need-Threads
        Invoke-Step (@('-m', 'stt', 'train', '--config', $Config) + $threadSet + $expArgs)
        $latest = Latest-Experiment
        Write-Host "LOOK AT: $ExperimentsDir\$latest\metrics\trial_000\metrics.csv (the loss), and the end of the screen: 'Run ended (max_steps)'. Write down the experiment number: $latest. Next: -Step evaluate -Experiment $latest"
    }
    'evaluate' {
        Need-Threads
        if ($Experiment -eq '') { Write-Host "STOP: give -Experiment exp_NNNN, the experiment of the train step."; exit 2 }
        Invoke-Step (@('-m', 'stt', 'evaluate', '--config', $Config) + $threadSet + $expArgs)
        Write-Host "LOOK AT: $ExperimentsDir\$Experiment\reports\evaluation_baseline.json and evaluation_trained.json, comparison.json. Next: -Step compare -Experiment $Experiment"
    }
    'compare' {
        if ($Experiment -eq '') { Write-Host "STOP: give -Experiment exp_NNNN, the experiment where evaluate scored the trained model."; exit 2 }
        $compareArgs = @('tools\compare_evals.py', '--dir', $ExperimentsDir, '--experiment', $Experiment)
        if ($BaselineFrom -ne '') { $compareArgs += @('--baseline-from', $BaselineFrom) }
        Invoke-Step $compareArgs
        Write-Host "LOOK AT: the relative reduction and the verdict. Copy out the small report files listed in docs\WORK_RUNBOOK.md, step 8."
    }
}
