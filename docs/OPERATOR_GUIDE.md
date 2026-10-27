# Operator guide: train a model and read the result, with a machine that already has the repo, the model and the data

## How to run (this page is the whole guide)

For a person who has never seen the code. Open PowerShell (a normal window, not inside an editor), go to the project folder (the one that contains `src`, `tools`, `configs`) and type the commands below in order. Nothing asks you a question. Anything that the repo's documents do not confirm is marked UNTESTED. The details behind every line are in `docs\WORK_RUNBOOK.md`; what the report means is in `docs\WHAT_YOU_GET.md`.

## Before the first command: edit 5 values in your copy of the yaml

```
Copy-Item configs\turbo_full.yaml configs\my_turbo_full.yaml
notepad configs\my_turbo_full.yaml
```

Search the copy for `<<< FILL IN`. Never edit the original `turbo_full.yaml`.

| Key in the yaml | In plain words | How to find it |
|---|---|---|
| `data.source.path` | the folder that holds `manifest.csv` and the audio files | the folder you were given |
| `data.columns.audio` | the manifest column with the audio file name | first line of `manifest.csv` |
| `data.columns.text` | the manifest column with the correct transcript | first line of `manifest.csv` |
| `data.columns.group` | the column that says which lesson / recording a clip comes from. Clips with the same value never go to different parts (Train / Val / Test). A wrong column here leaks test data into training | first line of `manifest.csv`; ask the data owner if unsure |
| `train.cpu_threads` | the number of PHYSICAL CPU cores (not the logical ones) | `.\.venv\Scripts\python.exe tools\probe_env.py` prints "4 physical cores" |

## The commands

| # | Command | What it does | How long | OK looks like | If not |
|---|---|---|---|---|---|
| 1 | `.\.venv\Scripts\python.exe tools\verify_clone.py --config configs\my_turbo_full.yaml` | checks Python, packages, model folder, config and runs quick unit tests; changes nothing. Add `--no-tests` to skip the tests | about 1 to 2 minutes (the repo says about one minute for the tests) | last line `Everything checked is fine.`, exit code 0 | read the `WHAT IS MISSING` list it prints and fix those lines; run again |
| 2 | `.\tools\run_work.ps1 -Step full -Config configs\my_turbo_full.yaml -Epochs 2 -Threads 4` (put YOUR physical core count in `-Threads`) | THE run: probe, check, prepare, baseline score, train, score again, compare, report | about 5 hours for 900 clips the first time (ESTIMATED from small tests; the script prints its own `ESTIMATED TIME`); a repeat on the same files about 1 hour | it prints `CHOSEN: ... Train clips ... steps`, and at the end `DONE. OPEN THE REPORT: <path>` | the script prints `STOP:` and an exit code; use the table below |
| 3a | watch (second PowerShell window): `$dir='<your run.experiments_dir>'; $run = Get-ChildItem $dir -Directory -Filter 'exp_*' \| Sort-Object LastWriteTime \| Select-Object -Last 1; Get-Content "$($run.FullName)\logs\run.log" -Tail 20 -Wait` | live log of the newest experiment; reading never disturbs the run (Ctrl+C there stops only the watching) | any time | lines `Encoder cache: N of M clips done` (every 5%), later `step N of M ... remaining ...` | no new line for 30 minutes at 100% CPU is possible in the first hours (one line per ~9 minutes); see the table |
| 3b | stop safely: press **Ctrl+C ONCE** in the window that runs the script | saves a checkpoint and ends; exit code 130 | under a minute, up to a few minutes while the 4.6 GB checkpoint is written | `Run ended` / exit code 130 | do not press it twice and do not close the window; if the window is gone, just continue |
| 3c | continue: **the exact same command as #2** | finished steps are skipped, the training continues from its last checkpoint, cached clips are not recomputed | seconds to start | `Resuming from ... (step N)` | if it says `config differs` (exit 2): you changed a setting; restore it |
| 4 | `Start-Process (Get-ChildItem <your run.experiments_dir> -Recurse -Filter report.html \| Sort-Object LastWriteTime \| Select-Object -Last 1).FullName` | opens the newest report in the browser. The folder with all results is your `run.experiments_dir` (in the template `data\work_cpu\experiments`; the home file uses `data\turbo_full\experiments`); the report itself is `exp_NNNN\reports\report.html` | seconds | a page with a coloured verdict line | no page: run `.\tools\run_work.ps1 -Step report -Config ... -ExperimentsDir ... -Experiment exp_NNNN` |
| 5 | `.\tools\run_work.ps1 -Step final-test -Config configs\my_turbo_full.yaml -Experiment exp_NNNN` (exp_NNNN = the TRAINING experiment; `full_run_state.json` lists it as `train`; add `-ExperimentsDir` if the yaml's `run.experiments_dir` is not the script default `data\work_cpu\experiments`) | scores the sealed Test clips ONCE for the baseline and the trained model; then rebuild the report with `-Step report -Experiment exp_NNNN` | roughly half an hour for about 100 clips (ESTIMATED) | `test_result.json` appears and the report gains a "Test: CONFIRMED / NOT CONFIRMED" block | a second call is refused with exit code 3: that is the rule, not a fault |
| 6 | no simple command: the trained model is already a normal folder `exp_NNNN\candidates\exp_NNNN_trial_001\exp_NNNN_trial_001_step_NNNNN\`; copy it as it is | the registry and `export` belong to the advanced `python -m stt run` path (`registry`, `export model_0001`) | | | advanced; see `docs\RUNBOOK.md` and `docs\WORK_RUNBOOK.md` ("the older operator guide") |

Use `-Dry` at the end of command 2 to print the plan without running anything.

## If you see this, do this

| You see | Cause | Do |
|---|---|---|
| `Out of memory`, exit code 4 | less than about 8 GB of RAM free | close browsers and other programs (best: restart the machine), then run command 2 again |
| `Not enough free disk for the encoder cache`, exit code 2; or exit 4 while writing a checkpoint | disk full (a checkpoint is 4.6 GB, the cache 3.84 MB per clip) | free disk (need about 11 GB for 900 clips), run command 2 again. Do not delete the cache folder or the experiment |
| exit code 2 `config differs from the stored config_full.yaml` | a setting was changed between two starts | change it back, run again |
| exit code 3 right after prepare | clips over 30 s, a wrong column name, or fewer than 5 lessons in Val or Test | open `reports\filter_report.txt` and `split_report.txt` in the experiment folder; check the 3 column names |
| exit code 4, other | training or scoring failed (for example a NaN loss) | send the last 40 lines of `logs\run.log` (below); then run command 2 again once |
| exit code 130 | you pressed Ctrl+C | run command 2 again to continue |
| nothing printed for a long time | normal in the encoder passes: one progress line per 5% of the clips, about 9 minutes apart for 700 clips; the baseline scoring prints about every 30 s | wait. Worry only after 30 minutes with the CPU near 0% (Task Manager): press Ctrl+C once, run the same command |
| `BASELINE: ... Stop (Ctrl+C) now if the WER is above 1.5` and the WER is above 1.5 | the text column or the audio is wrong | Ctrl+C once; open `reports\review_baseline.txt` and check 3 clips by ear against the text; fix the column names in the yaml (this makes a NEW run: use a new experiments folder) |
| empty outputs in the baseline report | wrong audio column or unreadable audio | same as above |
| training very slow from the first step (about a minute per step instead of about half a minute) | the encoder cache is not in use | check that your yaml has `features.encoder_cache.enabled: true`, and that you passed the right `-Config` |
| the script says it is blocked (PowerShell policy) | execution policy | `powershell -ExecutionPolicy Bypass -File tools\run_work.ps1 -Step full ...` with the same options |

## Never do

1. Do not change any setting between a start and its continuation (the program refuses; a new setting means a new experiments folder).
2. Do not delete the encoder cache or the experiment folders while the run is not finished.
3. Do not open the Test more than once; do not use the Test to choose between runs (use the Val numbers; do at most about 3 runs before the single final Test).
4. Do not `git pull` or change files in `src`, `tools` or `configs` while a run is going.
5. Do not close the window with the X; do not let the machine sleep (plugged in, sleep off); do not run other heavy programs.
6. Do not move or touch the audio files after the first run (the cache would count them as new).

## Who to ask and what to send

Send ONLY small text files (never the audio, the model, `candidates\`, `checkpoints\` or the cache folder):

| Send | Where |
|---|---|
| the command you typed and the last lines of the screen | the PowerShell window |
| `logs\run.log` (last 40 lines: `Get-Content <exp>\logs\run.log -Tail 40`) | `<experiments_dir>\exp_NNNN\` |
| `status.json`, `run_info.json`, `config_diff.yaml` | the same folder |
| `full_run_state.json` | `<experiments_dir>\` |
| `reports\load_report.txt`, `filter_report.txt`, `split_report.txt`, `text_report.txt` | `exp_NNNN\reports\` |
| `reports\evaluation_baseline.json`, and after training `evaluation_trained.json`, `comparison.json` | `exp_NNNN\reports\` |
| the output of `.\.venv\Scripts\python.exe tools\probe_env.py` | any time |

The reports contain sentences from your data: if they are confidential, send the numbers and file names only.
