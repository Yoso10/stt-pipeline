# Work runbook: one training on the work machine, CPU only, alone

## TOMORROW: one page, from arriving at the work machine to reading the report

Everything here is UNTESTED on the work machine; the commands were run on the home laptop (some on a fresh clone of the repo) and the times are measured or estimated as marked. Run every command from the project folder. If any step fails, do not go on: the "If not" column says what to do. A longer explanation of each step is further down this file.

| # | What | Command | Time | OK looks like | If not |
|---|---|---|---|---|---|
| 1 | **Clone** | `git clone <address> stt` ; `cd stt`. No access to the address from the work machine? At home: `git bundle create stt.bundle --all` (2.9 MB, tested), carry the file, then `git clone stt.bundle stt` (tested). Or copy the whole project folder WITHOUT `.venv`, `data`, `models`. | seconds (the clone is 21 MB, measured) | a folder `stt` with `README.md`, `configs\`, `tools\`, `src\` | no git on the machine: copy the folder from home |
| 2 | **Install the environment** | `python -m venv .venv` ; `.\.venv\Scripts\Activate.ps1` ; `pip install --no-index --find-links wheels -r requirements.lock` ; `pip check` | 3 to 10 min (estimated, not measured) | `pip check` prints `No broken requirements found.` | `no matching distribution`: the `wheels\` folder is missing or was made for another Python version (see "No internet for pip" below) |
| 3 | **Bring the model and the data** | the commands in "Bring these (not in git)" below | model: copying 3.24 GB takes minutes (depends on the USB); data: depends on your data | `verify_clone.py` (step 5) says the model folder passes and the manifest has the 3 columns | `verify_clone.py` prints `WHAT IS MISSING` and says what |
| 4 | **Fill the 4 values** | `Copy-Item configs\work_cpu.yaml configs\my_work_cpu.yaml` ; edit lines 23, 25, 26, 27 of `configs\my_work_cpu.yaml` (data folder, audio column, text column, group column) | 2 min | no `<<< FILL IN` is left in the copy | `verify_clone.py` names the key that is still wrong |
| 5 | **Verify the clone** | `.\.venv\Scripts\python.exe tools\verify_clone.py --config configs\my_work_cpu.yaml` | about 20 s (measured, with the quick tests) | the last line `Everything checked is fine.` | read the `FAIL` lines and `WHAT IS MISSING`; fix; run again. A `WARN` about free RAM: close programs (need about 8 GB free; the run peaks at 6.3 GB, measured) |
| 6 | **Run everything** | `.\tools\run_work.ps1 -Step full -Config configs\my_work_cpu.yaml` | FIRST run, 800 Train + 100 Val clips, 2 epochs: about 4.9 h with the encoder cache (estimated from the home measurements); the one-time encoder pass is 3.7 h of that | `CHOSEN: ...` and `ESTIMATED TIME` lines, then progress lines, then `DONE. OPEN THE REPORT: ...` | it stopped (power, Ctrl+C, an error): run the SAME command again, it resumes. Errors: the Hebrew companion at the end of this file |
| 7 | **Open the report** | open the path printed after `OPEN THE REPORT`: `data\work_cpu\experiments\exp_NNNN\reports\report.html` | seconds | the page starts with the verdict line and the chart of Val and train loss | no `report.html`: `.\tools\run_work.ps1 -Step report -Config configs\my_work_cpu.yaml` |
| 8 | **Decide the next run** | read "What to change next" in the report; the table "I see X -> I do Y" at the end of this file | 10 min | you know whether to add steps, lower the rate, or stop | |
| 9 | **The final run with the Test** | after at most 3 runs: pick the best experiment and run `.\tools\run_work.ps1 -Step final-test -Config configs\my_work_cpu.yaml -Experiment exp_NNNN` (`-WithTest` on `-Step full` does the same at the end of a fresh run) | about 0.6 h (encoder pass of the Test clips 0.4 h + decoding; estimated) | the script prints the Test result (status `confirmed` or `not_confirmed` in `reports\test_result.json`) and the report has a `FINAL - Test, scored once` section (rebuild: `-Step report -Experiment exp_NNNN`) | exit code 3 `REFUSED`: the Test was already scored; this is by design and is final |

**Time table with the encoder cache** (800 Train + 100 Val + 97 Test clips; estimated from the measured home numbers: 14.8 s per clip for the one-time encoder pass, 26.7 s per training step of 16 clips, 2.9 s per decoded clip):

| Hours you have | What fits | Plan |
|---|---|---|
| 4 h | the FIRST run does NOT fit (the encoder pass alone is 3.7 h) | use fewer clips: about 500 Train clips give a first run of about 3.4 h; or wait for a longer slot |
| 8 h | first run 4.9 h (2 epochs) + one rerun 1.2 h (cache ready: 2 epochs; 0.8 h for 1 epoch, 1.5 h for 3) + final Test 0.6 h = about 6.7 h | first run `-Epochs 2`, one corrected rerun, then the Test |
| 12 h | first run 5.2 h (3 epochs) + two reruns 3.0 h + final Test 0.6 h = about 8.8 h | the 3 allowed runs, then the Test; 3 h spare |

**The rule: at most 3 runs before the final Test.** Every run is judged on the same Val clips; the more runs you choose between, the more likely the "best" one is only lucky on those clips (a small Val is noisy: 7 lessons in the rehearsal). So: up to 3 runs (the first one counts), choose the best on Val, then score the Test ONCE for that experiment. The Test is refused a second time in the same experiments folder (exit code 3), so you cannot spend it by accident.

### No internet for pip (wheelhouse)

The work machine installs from the folder `wheels\` (52 files, 232 MB on the home laptop; NOT in git). To make it on a machine WITH internet and the SAME Python version and system as the work machine: `python tools\fetch_wheels.py --requirements requirements.txt --out wheels --lock requirements.lock --python-version 3.12 --platform win_amd64` (use the Python version of the work machine; this tool is in the repo and adds no dependency). Without the tool: `pip download -r requirements.lock -d wheels --only-binary=:all: --python-version 3.12 --platform win_amd64`. Then copy `wheels\` to the work machine (a plain copy; each file is small). UNTESTED for another Python version than 3.12.

### Bring these (NOT in git, by design): what, where, how big, how to move it

| Item | Needed tomorrow? | Path and size on the home laptop | Notes |
|---|---|---|---|
| (a) the turbo model folder | **YES** | `models\whisper-large-v3-turbo-ivrit`, 12 files, 3,240,199,074 bytes + `fetch_info.json` (3,090 MiB) | below |
| (b) YOUR data | **YES** | (yours) | below |
| (c) previous experiments and baselines | **NO** | `data\work_like\experiments`, `data\rehearsal_run`: not needed | the baseline is evaluated again on the work machine in step 6 (`full` runs `evaluate` first), on YOUR data |
| (d) the encoder cache | **NO, and it cannot be reused** | none for your data | rebuilt in step 6; the key holds each clip's id, audio path, size and modification time, so a cache made on other files or paths is never used; the home cache is of Daf Yomi clips |
| (e) `.env` and local configs | **NO** (optional) | `.env` is git-ignored; `configs\my_work_cpu.yaml` is made in step 4 | `.env.example` shows what a `.env` could hold (machine settings only) |
| (f) tokenizer and other files of the model folder | **YES, they are part of (a)** | the same folder: `added_tokens.json`, `config.json`, `generation_config.json`, `merges.txt`, `model.safetensors`, `normalizer.json`, `preprocessor_config.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json` | `verify_clone.py` runs the pipeline's own `check_model_folder` on it and lists what is missing |

**(a) The model: move it.** On the home laptop (PowerShell, project folder). Do NOT use `Compress-Archive` for this: the weights file is 3.2 GB and Windows PowerShell 5.1 cannot compress a file above 2 GB (known limit, not tested here); `tar` is in Windows 10 and 11 (`C:\windows\system32\tar.exe`, tested on a small folder):

```
Get-ChildItem models\whisper-large-v3-turbo-ivrit -Recurse -File | Get-FileHash -Algorithm SHA256 | ForEach-Object { "$($_.Hash)  $(Resolve-Path -Relative $_.Path)" } | Set-Content model_sha256.txt -Encoding ascii
tar -cf whisper-large-v3-turbo-ivrit.tar -C models whisper-large-v3-turbo-ivrit
Get-FileHash whisper-large-v3-turbo-ivrit.tar -Algorithm SHA256
Copy-Item whisper-large-v3-turbo-ivrit.tar, model_sha256.txt E:\          # E:\ = the USB drive or a network folder
```

On the work machine (in the project folder `stt`):

```
New-Item -ItemType Directory -Force models | Out-Null
tar -xf E:\whisper-large-v3-turbo-ivrit.tar -C models
Copy-Item E:\model_sha256.txt .
Get-Content model_sha256.txt | ForEach-Object { $h,$p = $_ -split '  ',2; if ((Get-FileHash $p -Algorithm SHA256).Hash -ne $h) { "MISMATCH $p" } }
```

No `MISMATCH` line = every file is identical (the check was tested on the small test model with 0 mismatches). The SHA-256 of the files on the home laptop (so you can compare by eye): `model.safetensors` `600f71638a2c8e674a9bead39ddc16553efc969541e0b74e74e25572b58e1a58`; `config.json` `22df89eb892c95148f6beaf3eb6ed2a70cf496953923e8e1667730d06efb3a69`; `tokenizer.json` `297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd`; `vocab.json` `e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387`. Linux: `tar -cf ...`, `tar -xf ...`, `sha256sum -c` (UNTESTED).

The layout expected afterwards: `stt\models\whisper-large-v3-turbo-ivrit\model.safetensors` and the 10 small files next to it (`fetch_info.json` is optional). `verify_clone.py` then prints `OK model folder ... passes the pipeline's own check`.

**(a') The model from the internet, if the work machine (or a prep machine) is connected.** Only the tool already in the repo and `huggingface_hub` (installed with transformers): `python tools\fetch_model.py --repo ivrit-ai/whisper-large-v3-turbo --out models\whisper-large-v3-turbo-ivrit` (about 3.24 GB, 11 files plus `fetch_info.json`; it ran on the home laptop on 2026-10-24). The work machine needs the internet ONLY for this, for `git clone` from a remote address and for building `wheels\`; everything else (training, scoring, the report) is offline.

**(b) Your data: what `prepare` expects.** One folder, for example `D:\mydata`, with `manifest.csv` (UTF-8, header row) and the audio files; audio paths in the manifest are relative to the manifest or absolute. Three columns are used (any names; you give them in step 4): the audio file name, the transcript text, and the group (a recording or lesson id; all clips of one group go to the same part). Audio: wav, mp3 or flac (read with soundfile); other sample rates are converted to 16 kHz and stereo is averaged; every clip at most 30 seconds (longer clips are REMOVED by the filter, see "Long recordings"). Text: Hebrew; time-stamp markers like `<|12.34|>` are stripped automatically. More: `docs\DATA_ONBOARDING.md`. Moving it: the same `tar` + `Get-FileHash` commands, or a plain `robocopy D:\mydata E:\mydata /E` and then `Get-FileHash` on the manifest. `verify_clone.py` confirms the folder, `manifest.csv` and the three columns.

### Requirements check (verified on the home laptop on 2026-10-26)

| Item | Needed | How verified | Status |
|---|---|---|---|
| Python | 3.12 (tested); any version for which `wheels\` was built | `python --version` = 3.12.14; `verify_clone.py` | OK here. Work machine: UNKNOWN (ask; build `wheels\` for its version) |
| The 11 packages of `requirements.txt` and the 51 pins of `requirements.lock` | exact versions | all 51 pins compared with the installed versions: 0 mismatches; `pip check`: no broken requirements | OK here |
| torch | 2.14.1, CPU build | `torch 2.14.1+cpu`, CUDA build none | OK here (the Windows wheel from PyPI is CPU-only) |
| Install on the work machine | internet OR `wheels\` | `pip install --no-index --find-links wheels -r requirements.lock` the offline install into a clean venv was NOT repeated in this task | UNTESTED on the work machine |
| git | to clone | `git clone` and `git clone` from a bundle tested | OK here; the work machine: UNKNOWN |
| Disk | repo 21 MB; `.venv` 1.2 GB (measured with `du`); `wheels\` 232 MB; model 3.1 GiB; encoder cache about 3.5 GB per 900 clips in fp16 (3.84 MB per clip, computed and measured); each TRAINING experiment 7.4 GiB measured with one checkpoint, up to about 12.3 GiB if the latest and the best checkpoints differ (estimated); the benchmark keeps another 7.4 GiB | sizes measured with `Get-ChildItem` / `du` on the home laptop | plan for at least 30 GB free |
| RAM | about 8 GB free | the process peaked at 6.3 GiB with the cache and at 6.4 to 6.7 GB without (measured, 8-clip test and benchmarks) | OK here with programs closed |
| CPU | any; time scales with the physical cores | 4 physical / 8 logical cores here | the work machine's speed is UNKNOWN |
| OS | Windows PowerShell 5.1 or later | all scripts run here in Windows PowerShell 5.1 (`powershell -ExecutionPolicy Bypass -File ...`) | Linux: `tools/run_work.sh` passes `bash -n` only (UNTESTED); `probe_env.py` shows `unknown` physical cores on Linux and the script then uses half the logical threads |

**What could still block tomorrow** (UNTESTED unless marked): (1) the work machine's Python version differs from the wheels; (2) `git clone` is impossible from the closed network (use the bundle or copy the folder); (3) the model copy is incomplete (caught by `verify_clone.py` and the hash list); (4) the data columns or clips longer than 30 s (caught by `verify_clone.py` / `prepare`); (5) the PowerShell policy forbids scripts (use `powershell -ExecutionPolicy Bypass -File tools\run_work.ps1 ...` or type each command by hand; the script was run that way here); (6) less than 8 GB of free RAM; (7) antivirus or a sync tool holding the new model folder (the program retries 5 times, O122); (8) the computer sleeps or is locked by a company policy (the program asks Windows to stay awake, which a policy may override); (9) a work machine slower than the home laptop (the benchmark tells; the estimate is the home one).

### Map of the documents: if I see X, or want to understand Y, open this

| I see / I want | Open |
|---|---|
| to start on a fresh clone | `README.md` (English and Hebrew), then this page (TOMORROW) |
| `verify_clone.py` output, `WHAT IS MISSING` | this page, "Bring these (not in git)"; `tools\verify_clone.py` (its header) |
| the script `run_work.ps1 -Step full`: what it does step by step | the Hebrew companion at the end of this file, sections 1 to 12; `docs\DECISIONS.md` item 90 (A) |
| what to type, and what the screen shows | this file: the Hebrew companion ("What I type, what I get, what it means") |
| the report: the convergence chart, the verdict words, the WER band | this file: "Reading the report"; `docs\COMPONENT_GUIDE.md` chapters 17 and 21; `DECISIONS.md` item 90 (B) |
| `final-test`, `test_result.json`, `REFUSED` (exit code 3) | this file, Hebrew section 10; `DECISIONS.md` item 90 (E) |
| an exit code (1, 2, 3, 4, 5, 130) | this file: the exit-code table of the companion and "Failure guide"; `docs\USER_GUIDE.md` section 11 |
| a term I do not know | this file: the glossary of 20 terms; `docs\COMPONENT_GUIDE.md` (glossary at the top) |
| what a component does, its keys, its errors | `docs\COMPONENT_GUIDE.md` (one chapter per component; the table "How each chapter was verified" says how far each chapter is checked) |
| a config key | `docs\CONFIG_REFERENCE.md` (21 sections, 188 keys); `configs\work_cpu.yaml` |
| the encoder cache, what it does, how to switch it off | this file: "The encoder cache" and Hebrew section 12; `DECISIONS.md` item 89 |
| the GPU key `runtime.device` | `docs\USER_GUIDE.md` section 4; `DECISIONS.md` items 87 and 88 (UNTESTED on a real GPU) |
| my data does not load or is rejected | `docs\DATA_ONBOARDING.md`; this file, step 3 and "Long recordings" |
| a symptom (loss, loops, empty outputs) | this file: the table "I see X -> I do Y"; `docs\USER_GUIDE.md` section 9; `docs\TROUBLESHOOTING.md` (per component) |
| why something was decided, what is still open | `docs\DECISIONS.md`, `docs\OPEN_ISSUES.md` |
| the slides cards | `docs\DECK_CARDS.md` |
| the older operator guide | `docs\RUNBOOK.md` (still valid for the commands of the full pipeline: `run`, `registry`, `export`) |

## A full turbo training: the config, when it finishes, what to do when it stops (Task 11)

Written from the code and from the home-laptop measurements. UNTESTED on the work machine and UNTESTED with the full turbo model end to end (the tiny model ran the whole chain; the turbo model ran in the 8-clip speed tests). The config for this run is `configs\turbo_full.yaml`; the script `tools\run_work.ps1` has no key for it, you point it at your copy with `-Config`.

### 4a. The exact commands to start (copy and paste, PowerShell, in the project folder)

```
Copy-Item configs\turbo_full.yaml configs\my_turbo_full.yaml
notepad configs\my_turbo_full.yaml        # fill in the 5 places marked  <<< FILL IN  (data folder, 3 column names, cpu_threads)
.\.venv\Scripts\python.exe tools\probe_env.py --model models\whisper-large-v3-turbo-ivrit     # prints the PHYSICAL cores: use that number below
.\.venv\Scripts\python.exe tools\verify_clone.py --config configs\my_turbo_full.yaml          # must end with: Everything checked is fine.
.\tools\run_work.ps1 -Step full -Config configs\my_turbo_full.yaml -Epochs 2 -Threads 4       # THE RUN. Put your physical core count in -Threads. Add -WithTest ONLY on the final run.
```

- `-Config` is the parameter that points the script at the turbo config (its default is `configs\work_cpu.yaml`, which has the same values). `-Epochs` is 2 by default (1, 2 or 3 are the useful values). `-Threads` is the number of PHYSICAL cores (0 or missing = found automatically). If the script is blocked by the PowerShell policy: `powershell -ExecutionPolicy Bypass -File tools\run_work.ps1 -Step full -Config configs\my_turbo_full.yaml -Epochs 2 -Threads 4`.
- Before you leave the computer: keep it plugged in, switch off sleep (`powercfg /change standby-timeout-ac 0`), close other programs (the run peaks at 6.3 GiB of RAM), and have about 30 GB of free disk.
- Linux: `bash tools/run_work.sh full` (UNTESTED; it reads `configs/work_cpu.yaml`, so copy your filled `my_turbo_full.yaml` over it or read the script's header for its variables).
- The start time is printed by nothing, so write it down, or start with: `$start = Get-Date; $start; .\tools\run_work.ps1 -Step full -Config configs\my_turbo_full.yaml -Epochs 2 -Threads 4`.

### 4b. Watching it from a second PowerShell window

```
cd <the project folder>
$dir  = 'data\work_cpu\experiments'
$run  = Get-ChildItem $dir -Directory -Filter 'exp_*' | Sort-Object LastWriteTime | Select-Object -Last 1      # the experiment that is being written now
Get-Content "$($run.FullName)\logs\run.log" -Tail 15 -Wait | Select-String 'Encoder cache|PROGRESS|step |Resuming|Run ended|ERROR|WARNING'   # live lines; Ctrl+C stops ONLY the watching
Get-Content "$($run.FullName)\metrics\trial_001\metrics.csv" -Tail 3      # during training: step, train_loss, val_loss, seconds_per_step, elapsed_sec, eta_sec
Get-Content "$dir\full_run_state.json"                                    # which steps are done (prepare, baseline, train ids; evaluated; tested)
Get-Process python | Select-Object Id, @{n='RAM_GB';e={[math]::Round($_.WorkingSet64/1GB,1)}}, CPU      # the memory of the run
```

The main window prints the same lines (`PROGRESS ...`, `step 40 of 100 (40.0%) ... elapsed ... remaining ...`, `Encoder cache: N of M clips done, elapsed ... remaining about ... s.`), at most one every 30 seconds. The experiment of the baseline and the experiment of the training are different folders, so run the `$run = ...` line again when the step changes. `remaining` is a rough estimate that is wrong at the beginning.

### 4c. What the screen looks like

| When | What you see (normal) | Worry when |
|---|---|---|
| first 1 to 3 minutes | `=== FULL RUN ...`, probe, `Everything checked is fine.`, prepare: `load_report`, `split_report`, a JSON summary; then `CHOSEN: 800 Train clips, 100 Val clips -> 2 epochs = 100 steps ... ESTIMATED TIME: about 4.9 hours` | `CHOSEN` never appears: prepare failed (exit code 3: read `filter_report.txt`, `split_report.txt`); an exit code and `STOPPED` is printed |
| minutes 3 to 10 | the baseline `evaluate` starts: the model loads (1 to 2 minutes, RAM goes to about 3.4 GiB), then `Encoder cache (val): 100 of 100 clips are missing, about 0.38 GB in ...; computing them once.` then `Encoder cache: 5 of 100 clips done, elapsed ... remaining about ... s.` every 5% | no new line for 15 minutes: one clip takes about 15 s, a 5% step of 100 clips is about 75 s, of 800 clips about 10 minutes (the line is every 5%: with 800 clips about 10 minutes between lines is NORMAL) |
| during the encoder pass (about 3.7 h for 900 clips, one time) | only `Encoder cache: ...` lines every 5%; the CPU is at 100% of the physical cores; the folder `data\work_cpu\encoder_cache\enc_..._float16` grows by 3.84 MB per clip; RAM about 3.6 GiB | RAM above 7 GiB; free disk below 5 GB; no line for 30 minutes at 800+ clips with CPU at 0% (the process hung: Ctrl+C once, run the same command again) |
| after the cache | `Encoder cache self-check passed on 3 clips (largest relative loss difference ...)`, then `PROGRESS WER scoring of the baseline model: clip 12 of 100 ...` (about 2.9 s per clip, 5 minutes in all) and the baseline WER; then `BASELINE: see ... Stop (Ctrl+C) now if the WER is above 1.5 or the outputs are empty.` | a baseline WER above 1.5 or empty outputs: stop and read `review_baseline.txt` (the text column or the audio is wrong) |
| training | `Training trial 1: 800 train and 100 val examples.`, `Freeze policy freeze_encoder: 171909120 of 808878080 parameters are trained.`, then `step 5 of 100 (5.0%) ... train_loss ... 26 s/step ... remaining about 0:42:00` | the first step takes more than 5 minutes (the cache is not used: look for `Encoder cache` lines and `features.encoder_cache.enabled`); RAM above 7.5 GiB; `train_loss` is `nan` (the run stops by itself, exit code 4) |
| every 25 steps | a line with `val_loss` (about 70 s) and a checkpoint save (a few minutes of disk: 4.6 GB) | `val_loss` rises at every point and the train loss falls: overfitting, see the report later |
| at the end | `Run ended (max_steps). Model of step N saved in ...`, evaluation of the trained model (5 minutes), `compare`, the report, `DONE. OPEN THE REPORT: ...` | exit code 4 or `STOPPED`: the first red lines tell why; the table below |

### 4d. The finish-time estimate (formula, blanks, totals)

The constants below are from the 8-clip speed tests of the turbo model on THIS laptop (8 logical / 4 physical cores, 16 GB, CPU). **Replace them with your own** from `.\tools\run_work.ps1 -Step benchmark -Config configs\my_turbo_full.yaml` (it trains 3 steps and decodes a few clips, so it is much shorter than the run; it also writes `reports\benchmark.json`; `-Step full` uses the newest `benchmark.json` for its own estimate when there is one). The work machine may be faster or slower: do not trust the column "this laptop" for it.

| Part | Formula | This laptop (constant) | Your benchmark (fill in) | Status |
|---|---|---|---|---|
| 1. one-time encoder pass (Train + Val clips, once ever for these files) | (Train clips + Val clips) x s/clip | 14.8 s/clip | ____ s/clip | measured on 8 clips (146 to 160 s for 8 clips including the self-check), scaled up |
| 2. training | steps x s/step; steps = epochs x ceil(Train / 16) | 26.7 s/step (16 clips) | ____ s/step | measured per 4-clip micro-step (7.4 to 7.7 s) x 4 = about 30 s, 26.7 s used by the calculator; scaled, ESTIMATED |
| 3. Val-loss evaluations | 4 x Val clips x s/clip | 0.67 s/clip (67 s for 100 clips, 4 times) | ____ s/clip | measured on 8 clips (4.6 to 5.4 s), scaled |
| 4. baseline decode + trained decode | 2 x Val clips x s/clip | 2.9 s/clip | ____ s/clip | measured on 8 clips (23 to 24 s), scaled |
| 5. the rest (model loads, prepare, self-check, compare, report) | fixed | 600 s | ____ s | ESTIMATED |
| 6. final Test (only with `-WithTest`, once) | (Test clips x s/clip encoder) + 2 x Test clips x s/clip decode | about 0.55 h for 97 clips | | ESTIMATED |

**Total = 1 + 2 + 3 + 4 + 5** (hours = seconds / 3600). Val clips are taken as Train / 8 (the split is 0.8 / 0.1 / 0.1 by duration; your real Val comes from `split_report.txt`). Totals on this laptop, FIRST run of a dataset (the encoder pass included), ESTIMATED:

| Train clips | Val | 1 epoch (steps) | 2 epochs (steps) | 3 epochs (steps) |
|---|---|---|---|---|
| 700 | 88 | 3.9 h (44) | 4.3 h (88) | 4.6 h (132) |
| 800 | 100 | 4.5 h (50) | 4.8 h (100) | 5.2 h (150) |
| 900 | 112 | 5.0 h (57) | 5.4 h (114) | 5.9 h (171) |
| 1000 | 125 | 5.6 h (63) | 6.0 h (126) | 6.5 h (189) |

The script prints its own figure at the start (`ESTIMATED TIME: about 4.9 hours` for 800 clips, 2 epochs; the 0.1 h gap to this table is its rounding of the fixed part). Later runs on the SAME files (the cache is full: the encoder pass is skipped) take parts 2 to 5 only: 700 clips 0.7 / 1.0 / 1.4 h, 800 clips 0.8 / 1.1 / 1.5 h, 900 clips 0.85 / 1.3 / 1.7 h, 1000 clips 0.9 / 1.4 / 1.9 h for 1 / 2 / 3 epochs.

**When it will finish:**
- At the start: `$start.AddHours(<total>)`, for example `(Get-Date).AddHours(4.8).ToString('ddd HH:mm')`.
- Later, from the log: finish = now + remaining. The training line says `remaining about H:MM:SS` (the training only). Add the parts still to come (decode of the trained model 2.9 s x Val, report 5 minutes): `(Get-Date).AddSeconds(<remaining seconds> + 2.9 * <Val clips> + 300)`.
- During the encoder pass: `remaining about N s` in the `Encoder cache` line is for that pass only; the training (part 2) still follows.

### 4e. If it stops, or you have only part of the data (from the code)

**(a) What is saved, and when.**

| What | When | Where |
|---|---|---|
| encoder-cache file of a clip (3.84 MB, float16) | right after that clip is encoded: written to a temporary file and renamed, so a file is complete or absent (`encoder_cache.py`) | `data\work_cpu\encoder_cache\enc_<hash>_float16\` |
| checkpoint (model weights of the trained decoder, `optimizer.pt`, scheduler, random state, progress counters), about 4.6 GB | every `train.save_every_steps` (25) steps; also at a new best Val loss, at every stop (Ctrl+C or the time cap) and at the end; only the latest and the best are kept (`hf_trainer.py`) | `data\work_cpu\experiments\exp_NNNN\candidates\exp_NNNN_trial_001\checkpoints\checkpoint-<step>\` |
| Val-loss and train-loss points | every `log_every_steps` / `eval_every_steps` into a CSV line at once; on a resume the earlier lines are read back | `exp_NNNN\metrics\trial_001\metrics.csv` |
| finished stage | a marker with the fingerprint of its settings | `exp_NNNN\reports\stage_results\<stage>.json`, `status.json` |
| where the full run was | after each finished step | `data\work_cpu\experiments\full_run_state.json` |

**(b) The command to continue** (after Ctrl+C, power loss, a crash, out of memory, a full disk, a Windows update): free the cause first (RAM, disk), then run **exactly the same command again**, with the same `-Config`, `-Epochs`, `-Threads` and no changed settings:

```
.\tools\run_work.ps1 -Step full -Config configs\my_turbo_full.yaml -Epochs 2 -Threads 4
```

| Event | Kept | Lost | What the script does |
|---|---|---|---|
| Ctrl+C (once) | everything; a checkpoint is saved at once; exit code 130 | nothing but the unfinished step | the same command resumes from that checkpoint (`Resuming from ... (step N).`) |
| power loss, crash, `Out of memory` (exit code 4) | the last periodic checkpoint (at most 25 steps, about 11 minutes, are lost), all cache files already written, all CSV lines | the steps since that checkpoint; a cache file that was being written is only a temporary file and is ignored | resumes the unfinished training from the last COMPLETE checkpoint; the cache is refilled for the missing clips only (the clips already done are NOT recomputed) |
| power loss in the middle of the encoder pass | every finished clip | the clip in progress | the same command computes only the missing clips |
| disk full | what was written | possibly a half-written checkpoint (ignored: only complete checkpoints count) | free disk (the cache build checks first and refuses with the numbers: exit code 2), run the same command |
| you changed `-Epochs` or a setting | | | a state file keeps the numbers of the first start (`a run with other -Epochs keeps the numbers of the stored state`); a changed config is REFUSED with exit code 2 (`Cannot resume ...: the config differs from the stored config_full.yaml`): restore the setting, or start a new experiment by renaming `full_run_state.json` |
| out of memory again | | | set `train.hyperparams.batch_size=2` and `grad_accum_steps=8` in a NEW run (a new config is a new run, the old checkpoint cannot be used) and `eval.batch_size=1` |

I tested on the tiny model that a changed setting on an existing experiment is refused (exit code 2) and that the same settings skip finished stages. The Ctrl+C and power-loss resume of the turbo model is UNTESTED here (the code path is the same as in the tiny-model resume tests of TrainerCore).

**(c) More data later, or only part of it now.**
- A finished experiment never changes. A NEW `prepare` (a new run of `-Step full` with more files) is a NEW experiment: it loads all the files again, filters, splits and trains **from the base model again**. Training on the old model plus new data is NOT a feature of this project.
- The encoder cache of the clips that are unchanged stays valid and is reused: the key is the clip id, the audio path, the file size and the modification time plus the encoder fingerprint, and only clips whose key is missing are computed (the log says `N of M clips are missing`). Do not move or `touch` the audio files, or they count as new. The cache of the Val clips stays valid too.
- **The split can change.** With `split.mode: grouped_ratio` (the default and the value in `turbo_full.yaml`) the lessons are assigned one by one to the part that is furthest below its target share, in a seeded order; when you add lessons the targets move, and **an old lesson can end up in a different part, including Test**. Then (1) the old Val and Test numbers are not comparable with the new ones, and (2) a lesson that the old model was trained on can be in the new Test. With `split.mode: hash_group` each lesson's part depends only on the seed, the ratios and the lesson itself, so old lessons never move; with few lessons the shares can be uneven and the checks stop the run (the module says so). This is an owner decision (DEFAULT, not LOCKED): I kept `grouped_ratio` and wrote the alternative into `turbo_full.yaml`; if you know data will grow, set `split.mode: hash_group` BEFORE the first run (UNTESTED with your data).
- **Test contamination, in short:** if you already scored the Test (the marker `data\work_cpu\experiments\test_split_scored.json` exists), `final-test` refuses to score again. Adding data and splitting again does not make the old Test result wrong, but it makes the Test clips different: use a NEW experiments folder (`--set run.experiments_dir=data/work_cpu/experiments2`) for the new dataset so that its Test is scored once on its own, and never compare the two Test numbers as if they were the same set. Never train on a lesson that is in the Test of the result you intend to report.
- Only part of the data now: run it, it is a normal run on fewer clips (the table above shows the times). The WER of a small Val is noisy (see the report). Check `split_report.txt`: at least 5 lessons in Val and Test.

**(d) Stop at a chosen point and still evaluate.**
- What works from the code (tested on the tiny model): a **time cap set BEFORE the start**. Add `-Set 'train.max_runtime_minutes=240'` (the `-Set` parameter of `run_work.ps1` passes `--set` words to every step). The training saves a checkpoint and ends cleanly at that time with `stopped_reason: time_limit`; the stage counts as finished, `evaluate` then scores the model of the BEST checkpoint (the latest when no Val-loss point existed yet), and the report is built. I ran this on the tiny model: stage `ok`, `evaluate` gave the trained WER. The cap is per call.
- What does NOT work: pressing Ctrl+C and then asking to evaluate. After Ctrl+C the stage is marked interrupted, `evaluate` finds no finished training and scores only the baseline. The model folder of the best checkpoint IS built (`candidates\exp_NNNN_trial_001\exp_NNNN_trial_001_step_NNNNN\`), but no command in the project scores a model folder by path (that needs a change to the code: ask). Also you cannot continue a time-capped experiment with the cap removed: the config would differ (exit code 2, tested); start a new experiment instead.
- **Which checkpoint is the best:** the one with the lowest Val loss. Read `exp_NNNN\metrics\trial_001\metrics.csv` (rows with a `val_loss`), or the report's chart (the best point is marked), or `pipeline_model_info.json` in the model folder (`best_step`, `best_val_loss`, `stopped_reason`, `steps_completed`). With 4 points per run, `eval_every_steps` = max_steps // 4 means the earliest stop that has a Val-loss point is a quarter of the run.

### 4f. If it fails (the five most likely, for this run)

| Symptom | Likely cause | Minimal fix |
|---|---|---|
| exit code 2, `Not enough free disk for the encoder cache` | disk | free space, or `-Set 'features.encoder_cache.dir=D:/cache'` on a bigger drive; run the same command |
| exit code 4, `Out of memory` | less than 8 GB free, or the other programs | close programs; if it persists, the batch-size change in the table above in a new run |
| exit code 3 after prepare | clips over 30 s, a wrong column, fewer than 5 lessons in Val or Test | `filter_report.txt`, `split_report.txt`, `DATA_ONBOARDING.md` |
| it is slow from the first step (60 s per micro-step) | the encoder cache is not used | look for the `Encoder cache` lines; the config must have `features.encoder_cache.enabled: true`; the `-Config` you gave is the right file |
| exit code 2 `the config differs from the stored config_full.yaml` | a setting was changed between two starts | restore it, or rename `full_run_state.json` and start again |

## A full turbo training on THIS laptop: the 900-clip subset, 2 epochs, no Test (Task 12)

Everything here was prepared on 2026-10-27 on the home laptop. The training itself has NOT been started by the agent. Everything marked UNTESTED has not run with the turbo model on the whole subset.

### 12a. Which file, what it holds

- **`configs\home_turbo_full.yaml`** (new). `configs\turbo_full.yaml` (the work-machine template) was put back with its `<<< FILL IN` places, so that nobody runs it by accident on the whole data.
- It extends `data_gmara.yaml`, the same data file that `work_like.yaml` reaches through `home_full.yaml`: the same parquet source (`data\daf_yomi_raw`, extracted to `data\extracted\daf_yomi`), the same columns (`audio`, `transcript`, `meta_entry_id`), the same filter and the same split. It sets the Subsampler preset `work_like` and `val_samples: null` exactly as `work_like.yaml` does.
- **Split (owner decision, written in the yaml):** the data are fixed, no refresh -> `split.mode` `grouped_ratio`, ratios 0.8 / 0.1 / 0.1 by duration, seed 42 from `runtime.seed`.
- Turbo part: frozen encoder, float32, learning rate 1e-5, batch 4 x accumulation 4, encoder cache ON (`data\turbo_full\encoder_cache`, float16), `runtime.device` cpu, `train.cpu_threads` 4 (the physical cores of this laptop), 2 epochs. New folders (nothing inside `data\work_like`): `data\turbo_full\experiments`, `data\turbo_full\registry`, `data\turbo_full\encoder_cache`.

### 12b. The split: the same one, and what I found about the Test

- **Same split.** The resolved `data`, `filter`, `split`, `subsample` and `text` sections of `home_turbo_full.yaml` and `work_like.yaml` are identical (compared in code), the seed is 42 in both, and a `prepare` of the new config (run into a scratch folder, not into the experiments folder) gave **702 Train / 101 Val / 97 Test clips (58 / 7 / 8 lessons)** and the **split fingerprint `5342b8562205736e`, the same as the work_like experiment `exp_0001`**. The existing split is not stored for reuse; it is recomputed from the data, deterministically, and it comes out the same. No new split is created. (Checking clip by clip against the rehearsal's list was not done; counts, lessons, fingerprint and settings match.)
- **Test has never been scored by a marker in the new folder** (`data\turbo_full\experiments` does not exist yet, so there is no `test_split_scored.json` and `final-test` will not refuse).
- **BUT the Test clips of this exact subset were looked at once before.** `data\rehearsal_run\experiments\exp_0002\reports\test_once.json` says `"looked": ["trial_2", "baseline"]`, with `test_result_trial_2.json` and `test_result_baseline.json`: the rehearsal (`whisper-tiny`, 20 steps, the same preset `work_like`) scored the 97-clip Test once, WER 1.0599 for its winner and 1.1728 for its baseline. No turbo model has seen these Test clips, and the rehearsal numbers did not choose any turbo setting (the learning rate and the steps are fixed here), so the risk is small, but the Test is not "never looked at" in the strict sense. Your call: accept it, or get a different split (a different `split.seed` or subsample seed; then it is NOT the same subset any more). Other folders with Test records: `part5_tmp`, `part8_tmp`, `part9_tmp`, `part11_tmp`, `phasec_tmp`, `phasee_tmp` (older tests; I did not check which data they used). `data\work_like\experiments` has none.

### 12c. Ready to run? (numbers from 2026-10-27, this laptop)

| Item | Need | Now | Verdict |
|---|---|---|---|
| Disk, drive C: (the only drive) | cache about 3.1 GB (803 clips x 3.84 MB) + trial folder about 7.9 GB (7.4 GiB measured: one 4.6 GB checkpoint + the 3.1 GB model) + reports = **about 11 GB**; the worst case (latest and best checkpoint differ: two checkpoints) about 16.3 GB | **17.8 GB free** (97% full) | FITS, with 6.8 GB spare in the normal case and only about 1.5 GB in the worst case. Free more before you start (proposals below). |
| RAM | about 8 GB free (peak 6.3 GiB measured) | **2.0 GB available** (15.8 GB total) | DOES NOT FIT NOW. Close programs (below) or restart the laptop. |
| Cores | 4 physical | 4 physical / 8 logical | OK |
| Power | AC | on AC, battery 100%; AC sleep timeout "never", display "never" (read with `powercfg`) | OK |

What uses the RAM now (working sets, 10.1 GB in all): VS Code 15 processes 1.7 GB, Chrome 14 processes 1.5 GB, Windows services (svchost) 1.1 GB, WebView2 (31 processes) 0.6 GB, Telegram 0.6 GB, Defender 0.3 GB, Notion 0.3 GB, Discord 0.2 GB, ProtonVPN 0.2 GB, and Docker Desktop (with its WSL part, whose memory Windows does not list by name). **Close before the start:** Chrome, Telegram, Notion, Discord, Docker Desktop (also `wsl --shutdown`), every other program; run the training from a normal PowerShell window, not from the VS Code terminal. The simplest way to get a clean 10+ GB: restart the laptop, log in, open only PowerShell, check with the command below, then start.

```
(Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue / 1024        # want 9 or more (GB)
(Get-PSDrive C).Free / 1GB                                                         # want 15 or more (GB)
```

**To free disk (proposals only, nothing was deleted):** the project's own `data\part*_tmp`, `data\phase*_tmp`, `data\rehearsal_run` hold only about 0.8 GB together, so they do not help much. Better: Windows Settings -> System -> Storage -> Temporary files; empty the Downloads folder; Docker images (`docker system df` shows them; `docker system prune` removes unused ones: confirm first). Do NOT delete `models\`, `data\extracted`, `data\daf_yomi_raw`.

### 12d. The exact commands

(a) **Start** (one PowerShell window, from the project folder; the same command resumes after any stop). `-Config` is the config, `-ExperimentsDir` MUST be the same folder as `run.experiments_dir` of that config (the script keeps its state file and its comparison there), `-Epochs` 2, `-Threads` 4 = the physical cores. No `-WithTest`.

```
cd C:\Users\9090\Desktop\Work\stt-pipeline
$start = Get-Date; "START: $start"
.\tools\run_work.ps1 -Step full -Config configs\home_turbo_full.yaml -ExperimentsDir data\turbo_full\experiments -Epochs 2 -Threads 4
```

(b) **Watch, in a second PowerShell window:**

```
cd C:\Users\9090\Desktop\Work\stt-pipeline
$dir = 'data\turbo_full\experiments'
$run = Get-ChildItem $dir -Directory -Filter 'exp_*' | Sort-Object LastWriteTime | Select-Object -Last 1      # run this line again when the step changes (baseline -> training)
Get-Content "$($run.FullName)\logs\run.log" -Tail 15 -Wait | Select-String 'Encoder cache|PROGRESS|step |Resuming|Run ended|ERROR|WARNING'      # live lines; Ctrl+C stops only the watching
Get-Content "$($run.FullName)\logs\run.log" | Select-String 'remaining' | Select-Object -Last 1                # the time left, last estimate
Get-Content "$($run.FullName)\metrics\trial_001\metrics.csv" -Tail 3                                           # during training: step, train_loss, val_loss, seconds_per_step, eta_sec
Get-Content "$dir\full_run_state.json"                                                                          # which steps are done
(Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue / 1024                                      # free RAM in GB
```

The log file is `data\turbo_full\experiments\exp_NNNN\logs\run.log` (the baseline and the training are different `exp_NNNN` folders). The main window prints the same lines.

(c) **Resume after Ctrl+C, a crash, power loss, out of memory or a full disk:** fix the cause, then run exactly the command (a) again. Nothing in the config may change (a changed setting is refused, exit code 2).

(d) **After it ends:**

```
Start-Process (Get-ChildItem data\turbo_full\experiments -Recurse -Filter report.html | Sort-Object LastWriteTime | Select-Object -Last 1).FullName      # opens the newest report in the browser
```

**Later, ONCE, the Test (do not run it now):** `.\tools\run_work.ps1 -Step final-test -Config configs\home_turbo_full.yaml -ExperimentsDir data\turbo_full\experiments -Experiment exp_NNNN` (exp_NNNN = the TRAINING experiment named in the last line of the run, `full_run_state.json` has it as `train`), then `.\tools\run_work.ps1 -Step report -Config configs\home_turbo_full.yaml -ExperimentsDir data\turbo_full\experiments -Experiment exp_NNNN`. A second call is refused (exit code 3).

**About the dry run's exit code 255:** `run_work.ps1 -Step full -Dry` returns **0** (checked). The 255 appeared only when its output was cut off by a pipe that closed early (`| Select-Object -First 30`), which makes PowerShell stop the script. It does not matter for a real run. The dry run assumes 800 Train and 100 Val clips (it does not read your data), so it prints `100 steps, a Val-loss point every 25 steps, ESTIMATED TIME about 4.9 hours`; the real run reads the real sizes after `prepare`.

### 12e. What the real run will choose, and the timeline

Real numbers after `prepare`: 702 Train, 101 Val -> 2 epochs = 2 x ceil(702 / 16) = **88 steps**, a Val-loss point every **22 steps** (22, 44, 66, 88), warmup 4. The script prints `CHOSEN: ...` and `ESTIMATED TIME` itself.

Fill in your start time in the first row (`$start` of the command above); each later clock time is the start plus the cumulative column. Constants: encoder 14.8 s per clip, training 26.7 s per step, Val-loss pass 0.67 s per clip, decoding 2.9 s per clip (all measured on 8 clips of the turbo model on this laptop and scaled up, so every duration below is an ESTIMATE; the model loads and checkpoint writes were not timed).

| # | Phase | How long | Cumulative (start + ...) | Clock time |
|---|---|---|---|---|
| 1 | probe, check, prepare (reads 8,723 rows, takes the subset) | 2 to 5 min | 0:05 | start + 0:05 (`CHOSEN` line appears) |
| 2 | baseline evaluate: load the model, **encoder pass for Val** (101 clips x 14.8 s) | 1 to 2 min + 25 min | 0:33 | start + 0:33 |
| 3 | baseline decode of 101 Val clips (101 x 2.9 s) and its WER | 5 min | 0:38 | start + 0:38 (`BASELINE: see ...` line) |
| 4 | train: load the model, **encoder pass for Train** (702 clips x 14.8 s), self-check | 1 to 2 min + 2 h 53 min | 3:34 | start + 3:34 |
| 5 | training, 88 steps x 26.7 s (CPU, 4 cores) | 39 min | 4:13 | start + 4:13 |
| 6 | 4 Val-loss points (4 x 67 s) and 4 checkpoint writes (4.6 GB each) | 5 min + 4 to 12 min | 4:25 | start + 4:25 |
| 7 | evaluate the trained model (load, 101 clips x 2.9 s from the cache) | 7 min | 4:32 | start + 4:32 |
| 8 | compare, report | 2 to 5 min | 4:36 | start + 4:36 (`DONE. OPEN THE REPORT`) |

**Total about 4.4 to 4.8 hours.** The formula gives 4.36 h before the loads and the writes; the dry run's 4.9 h is for 800 clips. Phases 2 and 4 (the encoder passes, 3.3 h in all) are **one-time**: a second run on the same files finds the cache full and takes about 1.0 h (phases 5 to 8 plus the loads). No log line appears for about 5% of a pass (about 80 s for Val, about 9 minutes for Train): that is normal.

### 12f. What can make it fail or waste hours, with the fix (proposals; the agent changed nothing on the system)

| Risk | State now | Fix |
|---|---|---|
| Sleep or hibernate | AC sleep and display timeouts are "never" (read from `powercfg`); hibernate not read | `powercfg /change standby-timeout-ac 0` ; `powercfg /change hibernate-timeout-ac 0` ; `powercfg /change monitor-timeout-ac 0` |
| Closing the lid | not read | `powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0` then `powercfg /setactive SCHEME_CURRENT` (lid does nothing on AC); or keep the lid open |
| Screen lock | a lock does NOT stop the program; only sleep, shutdown or a restart does | nothing to do; do not sign out |
| Battery | on AC, 100% | keep the charger in; plan for 5 hours |
| Windows Update restart | Update service stopped now, no restart pending | Settings -> Windows Update -> Pause updates for 1 week, before you start; do not restart the laptop during the run |
| Antivirus (Defender real-time protection is ON) | it scans each new cache file (about 800 files of 3.84 MB) and the 4.6 GB checkpoints: slower writes, and it can hold the final model folder briefly (the program retries the rename 5 times) | optional, from an ADMINISTRATOR PowerShell: `Add-MpPreference -ExclusionPath 'C:\Users\9090\Desktop\Work\stt-pipeline\data\turbo_full'` (remove later with `Remove-MpPreference -ExclusionPath ...`); the run is correct without it. I could not read the existing exclusions (needs administrator). |
| RAM | 2.0 GB available | the list in 12c; restart the laptop; check the counter before the start |
| Disk | 17.8 GB free | 12c; the cache build refuses with the numbers if it does not fit (exit code 2); the checkpoint write fails if the disk fills (exit code 4): free space and run the same command |
| VS Code or the terminal closes | a process started from the VS Code terminal can end with it | start from a normal PowerShell window |
| Other CPU load (tests, browsers, Docker) | it makes the estimate longer | close them; run nothing else |
| Thermal slowing on a laptop | not measured | put the laptop on a hard surface, charger in; if `seconds_per_step` climbs above 40, check the temperature |
| Wrong `-ExperimentsDir` | | it must equal `run.experiments_dir` of the config (`data\turbo_full\experiments`), as in command (a) |

### 12g. UNTESTED

The whole run on the 702-clip subset with the turbo model; the encoder pass over 803 clips (only 8 clips were timed); the cache build with the 702 Train clips and the 4.6 GB checkpoint writes on a disk with this little free space; the resume of the turbo training after Ctrl+C or a crash; the memory behaviour with the real batch (peak 6.3 GiB was measured on a 4-clip step); `final-test` on this model; the timeline above; Defender's effect on the cache; the Linux script.

## Task 13: the home turbo run you can watch, the ready check after the disk cleanup, and what I need for the next work run

Written 2026-10-27. Docs only: no file in `src`, no config the work machine uses, was changed. The training has NOT been started by the agent.

### 13a. Ready check, measured again on 2026-10-27 (this laptop, after your disk cleanup)

| Item | Need | Now | Verdict |
|---|---|---|---|
| Disk, drive C: | about 11 GB normal; about 16.3 GB worst case | **92.4 GiB free** (`Get-PSDrive C`; `check` prints the same amount as 99.2 GB, decimal units) | FITS with a large margin in both cases |
| RAM | about 8 GB available (peak 6.3 GiB measured) | **1.8 GB available** of 15.8 GB | **NO-GO until programs are closed** |
| Cores | 4 physical | 4 physical / 8 logical | OK |
| Power | AC, no sleep | Balanced plan; sleep and display timeouts AC and battery = 0 (never); on AC, battery 100% | OK |

Top RAM users now (working sets): Chrome 2.5 GB, Windows services (svchost) 1.2 GB, VS Code 1.0 GB, memory compression 0.7 GB, WebView2 0.6 GB, Notion 0.6 GB, Explorer 0.5 GB, Defender 0.3 GB, Claude 0.2 GB, Teams 0.1 GB. Close Chrome, Notion, Teams, Telegram, Discord, Docker Desktop and every WebView2 app; start from a normal PowerShell window (not the VS Code terminal). Then the counter must show 9 or more: `(Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue / 1024`. A restart is the surest way. **GO = disk fits (yes) and that counter says 9 or more. Today: NO-GO on RAM only.**

### 13b. The config was re-checked

`check --config configs\home_turbo_full.yaml`: "Everything checked is fine." A scratch `prepare` (output in the agent's scratch folder, not in `data`) gave: split fingerprint **`5342b8562205736e`** (same as work_like `exp_0001`), 8,723 samples in 710 groups; subset **702 Train (58 lessons) / 101 Val (7) / 97 Test (8)**. The dry run (`-Dry`, exit code 0) prints its fixed 800/100 numbers (100 steps, 4.9 h); the real run prints `CHOSEN: 702 Train clips, 101 Val clips -> 2 epochs = 88 steps (16 clips per step), a Val-loss point every 22 steps (4 points), warmup 4.` `configs\home_turbo_full.yaml` was not changed in Task 13.

### 13c. Start, watch, resume, report, final test

**(1) Start** (normal PowerShell window, nothing else open):

```
cd C:\Users\9090\Desktop\Work\stt-pipeline
$start = Get-Date; "START: $start"
.\tools\run_work.ps1 -Step full -Config configs\home_turbo_full.yaml -ExperimentsDir data\turbo_full\experiments -Epochs 2 -Threads 4
```

**(2) Watch** (a SECOND PowerShell window; all read-only, none of them disturbs the run). Run the first line again when the step changes (the baseline and the training are different `exp_NNNN` folders):

```
cd C:\Users\9090\Desktop\Work\stt-pipeline
$dir = 'data\turbo_full\experiments'
$run = Get-ChildItem $dir -Directory -Filter 'exp_*' | Sort-Object LastWriteTime | Select-Object -Last 1; $run.Name
Get-Content "$($run.FullName)\logs\run.log" -Tail 20 -Wait                                  # everything, live (Ctrl+C stops only the watching)
Get-Content "$($run.FullName)\logs\run.log" | Select-String 'Encoder cache|PROGRESS|step [0-9]+ of|Resuming|Run ended|ERROR|WARNING' | Select-Object -Last 3    # the last progress lines
Get-Content "$($run.FullName)\logs\run.log" | Select-String 'remaining' | Select-Object -Last 1                      # time left (this phase only)
Get-Content "$($run.FullName)\metrics\trial_001\metrics.csv" -Tail 4                         # training: step, train_loss, val_loss, seconds_per_step
Import-Csv "$($run.FullName)\metrics\trial_001\metrics.csv" | Where-Object { $_.val_loss } | Format-Table step, train_loss, val_loss   # only the Val-loss points
Get-Content "$dir\full_run_state.json"                                                       # which steps are done
Get-Process python | Select-Object Id, @{n='RAM_GB';e={[math]::Round($_.WorkingSet64/1GB,1)}}, CPU       # memory of the run
(Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue / 1024                   # free RAM in GB
(Get-PSDrive C).Free / 1GB                                                                   # free disk in GB
```

**Sample lines.** Lines marked REAL were copied from a real log on this laptop; lines marked FORMAT come from the code's log format with example numbers (no turbo run has logged them yet, so every number there is only an example):

| Phase | Line |
|---|---|
| prepare (REAL stage lines, `work_like` exp_0002) | `Stage prepare starts.` ... `Stage prepare finished.` (it took 73 s there) |
| script (REAL text) | `CHOSEN: 702 Train clips, 101 Val clips -> 2 epochs = 88 steps (16 clips per step), a Val-loss point every 22 steps (4 points), warmup 4.` and `ESTIMATED TIME: about ... hours` |
| baseline start (REAL) | `Evaluating the baseline model (whisper-large-v3-turbo-ivrit) on 101 files of 'val'.` |
| encoder pass (FORMAT) | `Encoder cache (val): 101 of 101 clips are missing, about 0.39 GB in data\turbo_full\encoder_cache\enc_<hash>_float16; computing them once.` then every 5%: `Encoder cache: 5 of 101 clips done, elapsed 75 s, remaining about 1425 s.` |
| self-check (FORMAT) | `Encoder cache self-check passed on 3 clips (largest relative loss difference 1.2e-04).` |
| baseline scoring (REAL format) | `PROGRESS WER scoring of the baseline on 'val': clip 2 of 101 (2.0%), elapsed 0:00:54, remaining about 0:44:13` (that run had no cache: 16.6 s per clip; with the cache about 2.9 s per clip) |
| baseline result (REAL) | `val (eval_version 1, model whisper-large-v3-turbo-ivrit): 101 files, 4697 words, WER 0.8335, mean per-file WER 0.9613, CER 0.3889; ... 1681.0 s` and `1 of 101 outputs hit the length cap (a looping model?)` (the warning is normal in small numbers) |
| script (REAL text) | `BASELINE: see ...\reports\evaluation_baseline.json and listen to 3 clips of review_baseline.txt. Stop (Ctrl+C) now if the WER is above 1.5 or the outputs are empty.` |
| train start (FORMAT) | `Training trial 1: 702 train and 101 val examples.` `Freeze policy freeze_encoder: 171909120 of 808878080 parameters are trained.` |
| train step (REAL format, tiny-model numbers) | `[exp_0002_trial_000] step 20 of 40 (50.0%) (epoch 2.00) train_loss 10.8463 val_loss 10.8524 lr 5.25e-06 grad_norm 0.706 0.06 s/step elapsed 0:00:03 remaining 0:00:01` (turbo: about 26.7 s/step, `val_loss` filled at steps 22, 44, 66, 88; `-` in between) |
| end (REAL format) | `Run ended (max_steps). Model of step 88 saved in ...` ; at the very end the script prints `DONE. OPEN THE REPORT: ...` |

**What to watch in the first hour** (start = 0:00; all durations are ESTIMATES from the 8-clip tests, except the baseline, whose 28 minutes without cache were MEASURED on these 101 clips):

| Time | Expected | Normal pause | Stop and look when |
|---|---|---|---|
| 0:00 to 0:05 | `=== FULL RUN`, probe, `Everything checked is fine.`, prepare, then `CHOSEN: 702 Train ... 88 steps` | up to 2 minutes with no line during prepare | no `CHOSEN` after 10 minutes, or `STOPPED` with an exit code: read the lines above it |
| 0:05 to 0:07 | the model loads (RAM climbs to about 3.4 GiB) | 1 to 2 minutes silent | RAM available below 1 GB |
| 0:07 to 0:33 | `Encoder cache (val): 101 of 101 clips are missing`, then a line every 5% (about 5 clips, about 75 s) | up to 2 minutes between lines | no line for 15 minutes with the CPU near 0%; a clip taking over 30 s (RAM full and swapping, or the laptop is hot) |
| 0:33 to 0:38 | self-check passed, then `PROGRESS WER scoring ...` about every 30 s | none | `self-check failed` (the run stops by itself; from the code, exit code 3; its message says to delete the cache folder, use float32, or switch the cache off); a WER above 1.5, empty outputs (`review_baseline.txt`): Ctrl+C |
| 0:38 to 0:47 | the second encoder pass, `Encoder cache (train): 702 of 702 clips are missing, about 2.70 GB` | the first progress line comes after about 9 minutes (5% of 702 = 35 clips) | free disk below 5 GB; RAM above 7.5 GiB |
| 0:47 to 1:00 | `Encoder cache: 35 of 702 ...`, `70 of 702 ...` (one line per 9 minutes) | 9 minutes between lines is NORMAL | no line for 30 minutes; the python process gone: run the same start command again |

The encoder pass for Train ends at about 3:34; the first training step lines follow. If the laptop sleeps, locks the disk or restarts, run the start command again: finished clips are not recomputed.

**(3) Resume** after Ctrl+C, a crash, a power loss, or a full disk: fix the cause, then run exactly command (1) again (nothing in the config may change).

**(4) Report, then the Test (only after YOU have looked at the report and decided):**

```
Start-Process (Get-ChildItem data\turbo_full\experiments -Recurse -Filter report.html | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
Test-Path data\turbo_full\experiments\test_split_scored.json        # False = the Test has not been scored in this folder
Get-Content data\turbo_full\experiments\full_run_state.json         # "train" = the exp_NNNN to use below
# ONLY after the decision (one time; a second call is refused with exit code 3):
.\tools\run_work.ps1 -Step final-test -Config configs\home_turbo_full.yaml -ExperimentsDir data\turbo_full\experiments -Experiment exp_NNNN
.\tools\run_work.ps1 -Step report -Config configs\home_turbo_full.yaml -ExperimentsDir data\turbo_full\experiments -Experiment exp_NNNN
```

Reminder from 12b: the 97 Test clips were scored once by the whisper-tiny rehearsal; no turbo model has seen them.

### 13d. For the NEXT work run: what I still need to fill `configs\turbo_full.yaml` (5 places) and to judge the running experiment

A work experiment is running from the `main` branch. **Do not `git pull` and do not change any file on the work machine until that run has ended.** Reading logs is safe; copying files out is safe. A changed file under `configs\`, `src\` or `tools\` while the run is going could make a later resume refuse (the stored config would differ) or run other code.

| What I need from you | How to find it on the work machine (read-only, one command each) |
|---|---|
| Data folder layout (where `manifest.csv` and the audio are), number of files | `Get-ChildItem <data folder> \| Select-Object -First 15` and `(Get-ChildItem <data folder> -Recurse -File).Count` |
| Audio format (wav / mp3 / flac) and sample rate; channels; clip length range | `.\.venv\Scripts\python.exe tools\inspect_audio.py --data <folder>` works only for parquet data (it prints sample rate, channels, min / median / max seconds); for a CSV folder: `prepare` writes `load_report.txt` (see below), and `soundfile` info of one file: `.\.venv\Scripts\python.exe -c "import soundfile as s; print(s.info(r'<one audio file>'))"` |
| Text column, audio column, group column, and what the group IS (lesson, recording or speaker; clips of one group never go to two parts) | the header line: `Get-Content <data folder>\manifest.csv -TotalCount 3` (a few rows are enough; do not paste private text if it is sensitive) |
| How many clips and how many hours | `load_report.txt` and `split_report.txt` of the running experiment's `prepare` (below) |
| Whether the transcripts cover the whole audio | your knowledge of how the data were made; also `text_report.txt` and `review_baseline.txt` (compare 3 clips by ear) |
| CPU cores, RAM, free disk | `.\.venv\Scripts\python.exe tools\probe_env.py --model models\whisper-large-v3-turbo-ivrit` (prints the physical cores: `train.cpu_threads` is that number) |
| Which config file and which experiments folder the running experiment uses | `Get-ChildItem configs`, and the folder named in `run.experiments_dir` of that config; the experiment's own copy is `<experiments_dir>\exp_NNNN\config_full.yaml` |
| The exact command used to start it | the PowerShell history of that window: `Get-History` (same window) or `(Get-PSReadLineOption).HistorySavePath` then `Select-String 'run_work' <that file>` |

**The running experiment's progress, without disturbing it** (replace the folder by the one in the work config; reading a file never changes it):

```
$dir = 'data\work_cpu\experiments'
Get-Content "$dir\full_run_state.json"
$run = Get-ChildItem $dir -Directory -Filter 'exp_*' | Sort-Object LastWriteTime | Select-Object -Last 1; $run.Name
Get-Content "$($run.FullName)\logs\run.log" -Tail 15
Get-Content "$($run.FullName)\metrics\trial_001\metrics.csv" -Tail 4        # only once the training has started
```

**Small files to copy out and paste to me** (text only; none is large; do NOT copy `candidates\`, `checkpoints\` or the encoder cache): the output of `tools\probe_env.py`; `full_run_state.json`; from the experiment folder `exp_NNNN\config_full.yaml` (it shows the paths, replace anything private before pasting), `reports\load_report.txt`, `reports\filter_report.txt`, `reports\split_report.txt`, `reports\subsample_report.txt` (if there is one), `reports\text_report.txt`, `reports\rejected.csv` (the first 20 lines only); the last 40 lines of `logs\run.log`; once training runs, `metrics\trial_001\metrics.csv`; at the end `reports\evaluation_baseline.json` and `reports\review_baseline.txt` (first 10 lines). Reports contain Hebrew text of your data: if the data are confidential, paste the numbers only.

### 13e. Task 13: what is safe for the work machine, UNTESTED

- **Safe:** this task changed only `README.md` and `docs\WORK_RUNBOOK.md` (documents). The code, the configs and the tools that the work run uses are untouched; nothing needs to be pulled. Pull later, after the run ends, and only if you want the new text.
- **UNTESTED:** the sample lines marked FORMAT; the timeline; the first-hour pauses of the second encoder pass; the self-check with the real turbo model on 3 Val clips; the work-machine commands in 13d (each is read-only but none was run on that machine); `inspect_audio.py` on CSV data (it reads parquet only); everything listed in 12g.

## How to run (the short version): what I type

Four things, in this order (UNTESTED on the work machine; the same steps ran on the home laptop with the tiny model and, in parts, with the turbo model):

1. **Clone and install** (once): `git clone <address> stt`, `cd stt`, then `python -m venv .venv`, `.\.venv\Scripts\Activate.ps1`, `pip install --no-index --find-links wheels -r requirements.lock` (the `wheels\` folder and the model folder `models\whisper-large-v3-turbo-ivrit` are NOT in git: copy them from home; see `README.md`).
2. **Fill 4 values** in a copy of the config: `Copy-Item configs\work_cpu.yaml configs\my_work_cpu.yaml`, then edit lines 23, 25, 26, 27 of `configs\my_work_cpu.yaml` (the data folder and the three column names). Check: `.\.venv\Scripts\python.exe tools\verify_clone.py --config configs\my_work_cpu.yaml`.
3. **Run everything**: `.\tools\run_work.ps1 -Step full -Config configs\my_work_cpu.yaml` (add `-Epochs 3` for 3 epochs; run the SAME command again to resume after any stop; add `-WithTest` only on the final chosen run).
4. **Open the report**: the last line prints its path, `data\work_cpu\experiments\exp_NNNN\reports\report.html`; open it in a browser.

The script finds the number of CPU cores, computes `max_steps = epochs x ceil(Train clips / 16)` from your data, sets one Val-loss point every `max_steps / 4` steps, prints the estimated hours and asks nothing. What each of its steps does, what the screen shows and what to do when it fails: the Hebrew companion at the end of this file ("What I type, what I get, what it means") and the single steps in the sections below. On Linux: `bash tools/run_work.sh full` (UNTESTED).

### The single steps (for when you want to run one by hand)

```
.\tools\run_work.ps1 -Step probe
.\tools\run_work.ps1 -Step check     -Config configs\my_work_cpu.yaml
.\tools\run_work.ps1 -Step prepare   -Config configs\my_work_cpu.yaml
.\tools\run_work.ps1 -Step benchmark -Config configs\my_work_cpu.yaml
.\tools\run_work.ps1 -Step baseline  -Config configs\my_work_cpu.yaml
.\tools\run_work.ps1 -Step train     -Config configs\my_work_cpu.yaml
.\tools\run_work.ps1 -Step evaluate  -Config configs\my_work_cpu.yaml -Experiment exp_0003
.\tools\run_work.ps1 -Step compare   -Experiment exp_0003 -BaselineFrom exp_0002
.\tools\run_work.ps1 -Step report    -Config configs\my_work_cpu.yaml -Experiment exp_0003
.\tools\run_work.ps1 -Step final-test -Config configs\my_work_cpu.yaml -Experiment exp_0003     # ONCE; a second call is refused
```

`-Threads` (the number of PHYSICAL cores) is found automatically; give `-Threads N` to choose. If the script does not start (a PowerShell policy), run the commands of each step by hand; they are written out in full below.

## Before you start: what the machine needs

- **RAM: close every other program.** Training the turbo model with a frozen encoder used a peak of 6.4 to 6.7 GB of RAM on the home laptop (measured). With the encoder cache ON the measured peaks (the turbo model, 8-clip test) were 3.6 GiB while the cache was built and 6.1 GiB for a training step, 6.3 GiB for the whole process: not higher than without the cache. A machine with less than 8 GB of FREE RAM (the probe prints it) may stop with "out of memory" (exit code 4): close the browser, mail, chat and sync programs, then run the probe again. With 16 GB of RAM and nothing else open it is fine (the home laptop did it with 2 to 4 GB free at the start).
- **Disk:** the benchmark keeps a model folder of about 7.4 GiB (measured); a training keeps one too (7.4 to 7.9 GiB) plus its checkpoints; the encoder cache needs 3.8 MB per clip in float16 (900 clips: 3.5 GB; measured size). Plan at least 30 GB free. Nothing is deleted by any command.
- **Time:** see step 4. On the home laptop, 2 epochs of 800 clips took an estimated 4.9 hours WITH the encoder cache and 8.3 hours WITHOUT it.
- **Do not let the computer sleep** (the program asks Windows to stay awake, but a company policy may override it) and do not close the PowerShell window. Stop a run with Ctrl+C ONCE (see step 6).

## Step 0: once, install (UNTESTED on the work machine)

Put the project at a short path (for example `C:\stt`), create the environment, install from the wheel bundle, copy `configs\work_cpu.yaml` to `configs\my_work_cpu.yaml` and fill in the `<<< ... >>>` places (the data folder, the three column names). Details: `docs\RUNBOOK.md` sections 2 and 3, `docs\DATA_ONBOARDING.md`.

```
cd C:\stt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --no-index --find-links wheels -r requirements.lock
Copy-Item configs\work_cpu.yaml configs\my_work_cpu.yaml
```

Success: `.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"` prints a version. Stop and call me if pip prints an error about a missing wheel (the wheel bundle was made for another Python version).

## Step 1: probe the machine

`.\tools\run_work.ps1 -Step probe` (this prints the Python and torch versions, physical cores, RAM total and free, free disk, and whether the model folder is complete). Takes seconds; changes nothing.

- **Success:** the model line says OK; RAM free is at least 8 GB; disk free is at least 30 GB; torch has no "NOT INSTALLED".
- **What the numbers mean for the plan:** physical cores = the value of `-Threads` and of `train.cpu_threads` in your config (more cores = faster, roughly in proportion; the home laptop has 4 physical cores and used 4). RAM free below 8 GB: close programs; if still below, use `train.hyperparams.batch_size: 2` and `grad_accum_steps: 8` (same 16 clips per step, less memory; UNTESTED) and `eval.batch_size: 1`.
- **Stop and call me** if the model line says PROBLEM (the folder is incomplete: copy `models\whisper-large-v3-turbo-ivrit` again from home).

## Step 2: check

`.\tools\run_work.ps1 -Step check -Config configs\my_work_cpu.yaml` (seconds).

- **Success:** the last line is "Everything checked is fine."
- **Stop and call me** if a PROBLEM line is left after you fixed what it names (an unfilled `<<< FILL IN` marker, a wrong folder, a missing package).

## Step 3: prepare, and the go / no-go decision

`.\tools\run_work.ps1 -Step prepare -Config configs\my_work_cpu.yaml` (about 1 to 2 minutes for 900 clips; the Daf Yomi data of 8,723 clips took 73 s; measured). It loads, splits by lesson, cleans the text and applies the filter. It trains nothing and scores nothing.

Open the newest experiment folder `data\work_cpu\experiments\exp_NNNN\reports\` and read the five files:

| File | Read this | GO if | NO-GO (stop and call me) if |
|---|---|---|---|
| `load_report.txt` | "seen X, loaded Y, rejected Z"; the clip lengths | rejected is a few percent at most (the run stops by itself above 5%) | many rejected: open `rejected.csv`, the reason column says why (missing audio, empty text, bad column names) |
| `rejected.csv` | the rows not used | empty, or a few clear cases | a pattern (all of one lesson, all with one reason) |
| `split_report.txt` | "val: ... groups", "test: ... groups", "group overlap check: ok" | **at least 5 groups (lessons) in Val and at least 5 in Test**, overlap check ok, the shares near 80/10/10 | fewer than 5 groups in Val or Test (the statistics cannot say "improved" with so few lessons: add data or change the group column), or an overlap failure |
| `filter_report.txt` | "mode apply, seen ..., kept ..., would reject ..."; max_duration line | few clips removed | **clips over 30 s are removed**: the number in the `max_duration` line is how many; if it is most of your data, read "Long recordings" below, this is a NO-GO until you provide clips of at most 30 s |
| `text_report.txt` | the character inventory | only Hebrew letters, spaces and expected punctuation | unexpected symbols in large numbers (markup, digits): see `docs\DATA_ONBOARDING.md` |

Also write down: the number of Train clips (split_report) and of Val clips. You need them in step 4.

## Step 4: benchmark, and the time calculator

`.\tools\run_work.ps1 -Step benchmark -Config configs\my_work_cpu.yaml -Threads 8` (with the encoder cache on, the benchmark first computes the encoder output of its clips; the home laptop needed 16 to 20 minutes for the benchmark WITHOUT the cache, measured; with it, not measured). Close other programs first and run it **twice**; use the larger numbers. It keeps a model folder of about 7.4 GiB. Open `data\work_cpu\experiments\exp_NNNN\reports\benchmark.txt`.

**Success:** it prints "BENCHMARK on this machine ... Measured: ... s per training step, ... s per example for the Val loss, ... s per clip for the WER scoring". The printed seconds per training step are up to 1.3 times too high (a known timing error, O121), so your total is on the safe side.

Also read from the screen (the benchmark prints `Encoder cache: ... clips done, elapsed ..., remaining ...` lines): **E = elapsed seconds divided by clips done** = the seconds the frozen encoder needs per clip, one time.

### Time calculator (fill in the blanks)

Write your measured values here:

- Train clips `N` = ______ , Val clips `V` = ______ (step 3)
- `S` = seconds per training step ______ (benchmark; one step = 16 clips)
- `L` = seconds per Val-loss clip ______ (benchmark)
- `D` = seconds per decoded clip ______ (benchmark)
- `E` = seconds per clip for the one-time encoder pass ______ (screen, see above; only with the cache)
- `O` = fixed extras (prepare, model loading and saving three times) ≈ 600 s (estimated)

Steps per epoch `P = ceil(N / 16)`; steps for `k` epochs `= k x P` (this is the value of `train.hyperparams.max_steps`).

```
WITH the encoder cache  :  seconds = (N + V) x E  +  k x P x S  +  V x L  +  3 x V x D  +  O
WITHOUT the cache       :  seconds = k x P x S  +  V x L  +  3 x V x D  +  O
hours = seconds / 3600
```

The `3 x V x D` is the baseline scored in step 5, then again in step 7 together with the trained model. The Test split is not used. The home laptop's numbers (measured, 2026-10-25: S = 26.7 s with the cache and 229 s without; L = 0.67 s and 15.6 s; D = 2.9 s and 16.0 s; E about 14.8 s; the first S, L and D are computed from the measured 8-clip speed test, see the guide) give, for N = 800 and V = 100:

| Epochs (steps) | With the cache | Without the cache |
|---|---|---|
| 1 (50 steps) | about 4.5 h | about 5.1 h |
| 2 (100 steps) | about 4.9 h | about 8.3 h |
| 3 (150 steps) | about 5.2 h | about 11.5 h |

And by available time (same numbers; estimated):

| Hours available | With the cache | Without the cache |
|---|---|---|
| 4 h | does not fit (the one-time encoder pass alone is 3.7 h for 900 clips; use fewer clips) | about 0.65 epoch (32 steps) |
| 8 h | about 10 epochs fit; choose 2 or 3 epochs and stop early | 1 epoch (50 steps); 2 epochs need 8.3 h |
| 12 h | about 20 epochs fit | 3 epochs (150 steps) |

**Decision:** pick `k` (1 to 3 epochs), set `train.hyperparams.max_steps` and `train.eval_every_steps` to `k x P` in `configs\my_work_cpu.yaml` (both the same number: one Val-loss evaluation at the end). **Stop and call me** if the total is more than the hours you have.

The model sees each clip `k` times; with few clips 2 to 3 epochs is a reasonable start (UNTESTED on your data; with the home rehearsal data 20 steps already changed the WER by 9%).

## Step 5: baseline evaluate (the sanity check before the long run)

`.\tools\run_work.ps1 -Step baseline -Config configs\my_work_cpu.yaml -Threads 8`. It scores the UNTRAINED model on the Val clips: about `V x (E + D)` seconds the first time (the encoder outputs of the Val clips go into the cache). Home laptop: 28 minutes for 101 clips WITHOUT the cache (measured). Write down the experiment number it prints: this is your **baseline experiment** (for example exp_0002).

- **Success:** `reports\evaluation_baseline.json` has a WER. On the Daf Yomi data the untrained turbo model gave WER 0.8335 and CER 0.3889 (measured). **Open `reports\review_baseline.txt` and listen to 3 clips**: the model's text should look like Hebrew that resembles the label.
- **Stop and call me** if the WER is above 1.5, or most outputs are empty or the same text repeated (look at `predictions_baseline.csv`): the language, the labels or the audio may be wrong. Do NOT start the training.

## Step 6: train

`.\tools\run_work.ps1 -Step train -Config configs\my_work_cpu.yaml -Threads 8`. This is the long step: `k x P x S` seconds plus the one-time encoder pass of the Train clips (shown as `Encoder cache (train): ... clips done`). It writes a loss file you can open in Excel at any time: `data\work_cpu\experiments\exp_NNNN\metrics\trial_001\metrics.csv`. A progress line appears at most every 30 seconds. The training line `s/step` and `remaining` are too high (known timing error); trust the elapsed time and the percent.

- **Success:** the last lines say `Run ended (max_steps). Model of step ... saved in ...` and the exit code is 0. `train_loss` in `metrics.csv` should go down from the first line to the last (it can be noisy). Write down the experiment number (for example exp_0003).
- **To stop safely:** press Ctrl+C **once** in this window. The training saves a checkpoint and stops after the current step (exit code 130). To continue: `.\tools\run_work.ps1 -Step train -Config configs\my_work_cpu.yaml -Threads 8 -Experiment exp_0003`. Never close the window and never run it inside `| Tee-Object`.
- **Stop and call me** if the loss becomes NaN (exit code 4: the message names the step; lower `learning_rate` to 5e-6), if the loss does not move at all, or if an "Access is denied" message is repeated 6 times (see the failure guide).

## Step 7: evaluate the trained model

`.\tools\run_work.ps1 -Step evaluate -Config configs\my_work_cpu.yaml -Threads 8 -Experiment exp_0003` (use the experiment number of step 6). This scores the baseline AND the trained model on the same Val clips and stores the comparison. Time `2 x V x D` (the encoder outputs are already in the cache, because the trained model has the same frozen encoder). The Test split is refused by this command and is never opened.

- **Success:** `reports\evaluation_trained.json` and `comparison.json` exist.

## Step 8: compare, and what to copy out

`.\tools\run_work.ps1 -Step compare -Experiment exp_0003 -BaselineFrom exp_0002` (seconds). It prints the WER and CER of both models, the relative reduction and the verdict:

- **improved** = the trained model is clearly better (the statistics at 90%, one-sided). **no_clear_difference** = could be chance. **worse**. **insufficient_groups** = fewer than 5 lessons in Val, no statement is possible.
- The aim is a relative reduction of at least 10%. A reduction that is reached but not `improved` is only a point estimate.

**Copy out for analysis, small files only** (never the audio, the data, the models, the encoder cache or anything under `candidates\`): from `data\work_cpu\experiments\exp_NNNN\` of the baseline experiment and the train experiment copy `config_diff.yaml`, `status.json`, `run_info.json`, `logs\run.log`, and from `reports\`: `load_report.txt`, `split_report.txt`, `filter_report.txt`, `text_report.txt`, `benchmark.txt` (benchmark experiment), `evaluation_baseline.json`, `evaluation_trained.json`, `comparison.json`, `predictions_baseline.csv`, `predictions_trained.csv`; and `metrics\trial_001\metrics.csv`. Together they are a few megabytes. Check with `Get-ChildItem -Recurse <folder> | Measure-Object Length -Sum` that you copy less than 50 MB. The predictions files contain the transcripts: copy them out only if the rules of the work site allow it.

## Long recordings (if your audio is not in clips of at most 30 s)

The pipeline reads a manifest of clips, each with its own audio file and its own text, and the Whisper model reads at most 30 s at a time. What happens with longer audio (read from the code):

- `prepare` with `filter.mode: apply` (the setting in `work_cpu.yaml`) REMOVES every clip longer than 30 s (the `max_duration` rule); it does not cut it. The removed number is in `filter_report.txt`. With the default `report` mode nothing is removed and the training stops with exit code 3 and the message "clips ... are longer than the model window".
- **There is no segmentation tool in this project.** You must provide clips of at most 30 s, each with the text that belongs to exactly that audio, in the manifest. The text must match the audio of the clip (a cut in the middle of a sentence is acceptable only if the text is cut at the same place).
- If your labels have time stamps like the Daf Yomi data (`<|0.08|>text<|1.84|>`), you can cut the long recordings at the segment boundaries with any audio tool (for example ffmpeg, if it is installed) and write one manifest row per piece; the marker pattern is the key `text.timestamp_pattern`. UNTESTED: nothing in this project does the cutting.
- A group column (the recording or lesson id) must be the SAME for all pieces of one recording, so the pieces of one recording never land in Train and Test at once.

## Failure guide (one page)

**Exit codes:** 0 success. 1 an unexpected bug (keep `logs\run.log` and the screen). 2 wrong settings (config, a missing folder, a marker not filled in, the cache refusing). 3 data problem (rejected rows, a clip longer than 30 s in report mode, features). 4 training or scoring failed (NaN loss, out of memory, a failed model-folder rename). 5 a registered model does not match its checksums, or an export failed (not used in this path). 130 you pressed Ctrl+C.

**Where the log is:** `data\work_cpu\experiments\exp_NNNN\logs\run.log` (everything, with times), `status.json` in the same folder (which stage failed and why). The last 40 lines: `Get-Content data\work_cpu\experiments\exp_NNNN\logs\run.log -Tail 40`.

**The five most likely problems:**

| # | Problem | What you see | What you do |
|---|---|---|---|
| 1 | Out of memory | exit code 4, "Out of memory" | close every other program; set `train.hyperparams.batch_size: 2` and `grad_accum_steps: 8`, `eval.batch_size: 1`; start a NEW experiment (the config changed, so `--experiment` refuses) |
| 2 | Disk full | a write error, "No space left", or the cache says "Not enough free disk ... short by X GB" | free space (delete only what you are sure of, such as old benchmark folders `candidates\exp_NNNN_benchmark`); start again with `-Experiment` (a stopped cache build or training continues) |
| 3 | The model folder could not be renamed ("Access is denied", O122) | warnings "Attempt 1 of 6 ... failed", then maybe exit code 4 | the program retries 5 times by itself; if it still fails the finished model stays in a `.tmp` folder: close the antivirus or sync program that holds it, then run the same step again with `-Experiment exp_NNNN`. Nothing is lost |
| 4 | NaN loss | exit code 4, "The loss became NaN or infinite at step N" | lower `train.hyperparams.learning_rate` to 5e-6, start a new experiment, read the first lines of `metrics.csv` |
| 5 | The run stopped (power cut, window closed, Ctrl+C) | exit code 130, or no exit message, a stale `run.lock` | check no python is running (`Get-Process python`), rename the file `run.lock` in the experiment folder to `run.lock.stale` (do not delete it), then rerun the same step with `-Experiment exp_NNNN`; the training resumes from its last checkpoint (every 25 steps) |

**Never:** edit `config_full.yaml`; delete an experiment folder in the middle of work; run two commands on the same machine at the same time; open the Test split.

## The encoder cache: on, and how to switch it off

Keys (in `work_cpu.yaml`): `features.encoder_cache.enabled: true`, `features.encoder_cache.dir`, `features.encoder_cache.dtype: float16`, and it needs `train.freeze_policy: freeze_encoder`. It computes the output of the frozen encoder once per clip and stores it; training, the Val loss and the decoding then skip the encoder. It refuses to start (exit code 2, naming the key) if the encoder is not frozen, if the model has dropout or layerdrop above 0, if an augmentation is set, or if there is not enough free disk. Before first use it compares the Val loss with and without the cache on 3 clips and stops if they differ. On the home laptop with the turbo model the Val loss, the decoded text and the WER were identical (float16: loss difference 2e-5 relative; measured, 8 clips).

**To run exactly as before (cache OFF):** start a NEW experiment (the settings changed, so an old experiment cannot be continued) with `--set features.encoder_cache.enabled=false` added to the command, or set `enabled: false` in `my_work_cpu.yaml`. The cache folder stays on disk (it can be reused later or deleted by you). The time without the cache is in the calculator above.

## What to prepare at home before you leave (checklist)

- [ ] The project folder (code, `wheels\`, `requirements.lock`, `tools\`, `configs\`, `docs\`) and, if you can, `MANIFEST.sha256` made from that very folder (`python tools\make_manifest.py --root . --out MANIFEST.sha256`); on the work machine `python tools\make_manifest.py --root . --verify MANIFEST.sha256`.
- [ ] The model folder `models\whisper-large-v3-turbo-ivrit` (11 files, 3.24 GB; run `.\tools\run_work.ps1 -Step probe` at home to prove it is complete).
- [ ] Your data in the layout of `docs\DATA_ONBOARDING.md`: one folder with the audio files and ONE `manifest.csv` (UTF-8) with a column for the audio file name, a column for the transcript, and a column for the group (recording or lesson id). Clips of at most 30 s. The transcript is the text of exactly that audio. Time-stamp markers are removed automatically if they use the `<|12.34|>` form.
- [ ] `configs\my_work_cpu.yaml`: copy of `work_cpu.yaml`; leave the `<<<` places to be filled on the machine if you do not know the paths yet.
- [ ] A Python version that matches the wheel bundle (`python tools\fetch_wheels.py --python-version X.Y` was run for the work machine's version).
- [ ] This runbook printed or on a USB stick.

## UNTESTED on the work machine (everything)

Everything in this runbook was prepared and checked on the home laptop only. Never run on the work machine: the installation from the wheels, `probe_env.py`, `run_work.ps1`, the benchmark, the encoder cache build on your data, the training on your data, `compare_evals.py` on a real train-and-evaluate pair, a stopped-and-resumed training, and the filter in `apply` mode on real data (its effect on the WER was never measured). The times of your machine are unknown until your benchmark.

---

# מה אני מקליד, מה אני מקבל, מה זה אומר (What I type, what I get, what it means)

**איך להשתמש בחלק הזה.** זה המדריך לעבודה לבד. לכל פקודה בריצה המלאה יש כאן: הפקודה (באנגלית), משפט אחד על מה היא עושה, מה אמור להופיע במסך כשהכול תקין (שורות אמיתיות מהמחשב הביתי; כתוב מאיזו ריצה), אילו 2-3 מספרים לקרוא, אילו קבצים נוצרים ואיפה מוסבר עליהם, מה עושים כשנכשל, וההחלטה שמגיעה ממנה. כל מה שלא הורץ עדיין על מחשב העבודה מסומן **UNTESTED**. שמות הקבצים ושורות המסך נשארים באנגלית כי כך הם מופיעים אצלכם.

מקורות לשורות הדוגמה: **[turbo]** = ריצה אמיתית של מודל ה-turbo על המחשב הביתי; **[rehearsal]** = ריצת החזרה עם whisper-tiny; **[tiny]** = ריצה קטנה של אותו סקריפט על המודל הזעיר ונתוני בדיקה סינתטיים (המספרים בה חסרי משמעות, רק צורת המסך אמיתית).

## טבלת קודי היציאה (לכל הפקודות)

| קוד | משמעות | מה עושים |
|---|---|---|
| 0 | הצלחה | ממשיכים לשלב הבא |
| 1 | באג לא צפוי | שומרים את `logs\run.log` ואת המסך, ופונים לעזרה; לא ממשיכים |
| 2 | הגדרות שגויות: קובץ הגדרות, תיקייה חסרה, ערך `<<< FILL IN >>>` שלא מולא, "config differs", המטמון מסרב (המקודד לא קפוא, יש dropout, אין דיסק) | קוראים את ההודעה (היא נקראת את המפתח) ומתקנים; ראו USER_GUIDE סעיף 11 |
| 3 | בעיית נתונים: שורות נדחו, קליפ ארוך מ-30 שניות במצב report, פיצול נכשל; **או** `final-test` סירב כי ה-Test כבר נבדק | נתונים: קוראים `rejected.csv` ו-`load_report.txt`. Test: זה סירוב מכוון, לא תקלה |
| 4 | אימון או בדיקה נכשלו: NaN, זיכרון נגמר, שינוי שם של תיקיית המודל נכשל, כל הפלטים ריקים | טבלת "אני רואה X" למטה; ראו TROUBLESHOOTING.md (TrainerCore, Evaluator) |
| 5 | מודל רשום לא תואם לסכומי הביקורת, או ייצוא נכשל | לא בשימוש בנתיב הזה |
| 130 | עצרתם עם Ctrl+C | מריצים שוב את אותה פקודה כדי להמשיך |

כל הפקודות כותבות למסך שורות `INFO`/`WARNING`, ובסוף JSON אחד. הלוג המלא: `data\work_cpu\experiments\exp_NNNN\logs\run.log`; מצב השלבים: `status.json` באותה תיקייה (`COMPONENT_GUIDE.md` פרק 15; `USER_GUIDE.md` סעיף 7).

---

## 1. probe: מה יש במחשב הזה?

| | |
|---|---|
| פקודה (PowerShell) | `.\tools\run_work.ps1 -Step probe` |
| פקודה (Linux, UNTESTED) | `bash tools/run_work.sh probe` |
| מה זה עושה | מדפיס גרסאות, ליבות, זיכרון, דיסק, ובודק שתיקיית המודל שלמה; לא משנה כלום ותמיד מסתיים בקוד 0. |
| מסך תקין **[turbo, מחשב ביתי]** | `Python 3.12.14 (Windows 11)` / `torch 2.14.1+cpu (CUDA build: none; CUDA available: False)` / `CPU 8 logical threads, 4 physical cores` / `RAM 16.93 GB total, 7.26 GB free` / `Disk 23.9 GB free ...` / `Model folder OK: ... passes the model-folder check` |
| מספרים לקרוא | **RAM free** (צריך לפחות 8), **Disk free** (לפחות 30), **physical cores** (זה מספר החוטים), ושורת Model folder צריכה להתחיל ב-OK |
| קבצים שנוצרים | אין |
| איפה מוסבר | `tools\probe_env.py` (הכותרת שלו); `docs\WORK_RUNBOOK.md` שלב 1 |
| אם נכשל | הוא לא נכשל (קוד 0 תמיד). `PROBLEM` בשורת Model folder: מעתיקים שוב את התיקייה `models\whisper-large-v3-turbo-ivrit` מהבית. `NOT INSTALLED` ליד torch: מפעילים את הפקודה עם ה-Python של `.venv`. |

| אני רואה | אני עושה |
|---|---|
| RAM free מתחת ל-8 | סוגרים דפדפן, מייל, צ'אט וסנכרון, ומריצים שוב; אם עדיין נמוך: `batch_size 2` ו-`grad_accum_steps 8` (UNTESTED) |
| Disk free מתחת ל-30 | מפנים מקום לפני שממשיכים |
| Model folder PROBLEM | מעתיקים את המודל שוב (11 קבצים, 3.24 ג'יגה) |

## 2. check: האם ההגדרות והתיקיות תקינות?

| | |
|---|---|
| פקודה | `.\tools\run_work.ps1 -Step check -Config configs\my_work_cpu.yaml` |
| Linux | `bash tools/run_work.sh check --config configs/my_work_cpu.yaml` |
| מה זה עושה | בודק חבילות, תיקיית מודל, תיקיית נתונים, דיסק וזיכרון, ומציג את כל הבעיות ביחד. |
| מסך תקין **[tiny]** | `OK package torch 2.14.1` / `OK model.path exists: ...` / `OK data.source.path exists: ...` / `INFO free disk space where the experiments go: 22.7 GB` / **`Everything checked is fine.`** |
| מספרים לקרוא | השורה האחרונה; כל שורת `PROBLEM` |
| קבצים שנוצרים | אין |
| איפה מוסבר | `USER_GUIDE.md` סעיף 5 (טבלת הפקודות); `COMPONENT_GUIDE.md` פרק 16 |
| אם נכשל | קוד 2: `PROBLEM ... <<< FILL IN >>>` = לא מילאתם את 4 הערכים; `data.source.path` לא קיים = נתיב שגוי; `Model folder not found` = המודל לא הועתק |

**החלטה:** שורה אחרונה `Everything checked is fine.` = ממשיכים. אחרת מתקנים ומריצים שוב; לא ממשיכים עם PROBLEM.

## 3. prepare: טעינת הנתונים, פיצול וניקוי (בלי אימון)

| | |
|---|---|
| פקודה | `.\tools\run_work.ps1 -Step prepare -Config configs\my_work_cpu.yaml` |
| Linux | `bash tools/run_work.sh prepare --config configs/my_work_cpu.yaml` |
| מה זה עושה | טוען, מפצל לפי שיעור ל-Train/Val/Test, מנקה טקסט ומסנן (קליפים מעל 30 שניות מוסרים), וכותב חמישה דוחות; לא מאמן ולא מריץ מודל. |
| מסך תקין **[turbo, נתוני Daf Yomi, `reports\`]** | `load_report.txt`: `seen 8723, loaded 8723, rejected 0` ; `split_report.txt`: `train: 6973 samples, 566 groups, 57.90 h, ratio 0.800` / `val: 875 samples, 72 groups` / `group overlap check: ok: no group appears in more than one split` |
| מספרים לקרוא | `rejected` (כמה שורות נדחו); מספר ה-`groups` ב-val וב-test (**לפחות 5 בכל אחד**); בקובץ `filter_report.txt` את שורת `max_duration ... N fail` (כמה קליפים ארוכים מ-30 שניות הוסרו) |
| קבצים שנוצרים | `data\work_cpu\experiments\exp_0001\reports\`: `load_report.txt` (מה נטען), `rejected.csv` (מה נדחה ולמה), `split_report.txt` (גדלי הפיצול), `text_report.txt` (תווים חריגים), `filter_report.txt` (מה היה מוסר); וגם `config_full.yaml`, `status.json`, `logs\run.log` |
| איפה מוסבר | `COMPONENT_GUIDE.md` פרקים 2 (DataSource), 3 (Splitter), 4 (TextNormalizer), 5 (SampleFilter); `DATA_ONBOARDING.md` |
| אם נכשל | קוד 3, `Rejected ratio ... is above data.max_rejected_ratio` = שמות עמודות שגויים או קבצים חסרים: קוראים `rejected.csv`; `Column 'x' does not exist` = שם עמודה שגוי, ההודעה מפרטת את העמודות הקיימות; `The ratios cannot be met` / `Split 'val' is empty` = מעט מדי שיעורים: אין מספיק נתונים |

| החלטה | |
|---|---|
| נדחו כמה אחוזים ספורים לכל היותר, 5 שיעורים לפחות ב-Val וב-Test, כמעט שום קליפ לא הוסר | GO |
| הרבה קליפים הוסרו על `max_duration` | STOP: האודיו ארוך מ-30 שניות; ראו "Long recordings" בהמשך; צריך קליפים של עד 30 שניות |
| פחות מ-5 שיעורים ב-Val או ב-Test | STOP: הסטטיסטיקה לא תוכל לומר "improved"; מוסיפים נתונים או משנים את עמודת הקבוצה |

## 4. benchmark: כמה זמן ייקח? (אופציונלי בנתיב `full`)

| | |
|---|---|
| פקודה | `.\tools\run_work.ps1 -Step benchmark -Config configs\my_work_cpu.yaml` |
| Linux | `bash tools/run_work.sh benchmark --config configs/my_work_cpu.yaml` |
| מה זה עושה | מאמן 3 צעדים על 12 קליפים וסורק 12 קליפים, ומדפיס שניות לצעד, ל-Val-loss ולפענוח, ואת הזמן המתוכנן. |
| מסך תקין **[turbo, בלי מטמון]** | `55.41 s per training step (batch 1 x 2 accumulation = 2 examples)` / `17.367 s per example for the Val loss` / `19.64 s per clip for the WER scoring` / `peak memory of this process: 6.7 GB` |
| מספרים לקרוא | שלושת הזמנים (ממלאים אותם במחשבון הזמן, שלב 4), ו-`peak memory` |
| קבצים שנוצרים | `reports\benchmark.txt` ו-`benchmark.json`; `candidates\exp_NNNN_benchmark\` (**תיקיית מודל אמיתית, כ-7.4 ג'יגה, נשארת**); `metrics\benchmark\metrics.csv` |
| איפה מוסבר | `COMPONENT_GUIDE.md` פרק 16 ופרק 21; `USER_GUIDE.md` סעיף 8 |
| אם נכשל | קוד 2: `warmup_steps must be between 0 and max_steps (3)` = הסקריפט כבר מורידו ל-3; קוד 4: `Out of memory` = סוגרים תוכניות |

**החלטה:** הזמן הכולל שיצא קטן מהשעות שיש לכם = ממשיכים; גדול = מקטינים `-Epochs` או את כמות הנתונים. הערה: ה-s/step המודפס עד פי 1.3 גבוה מדי (באג O121), לכן החישוב זהיר. UNTESTED על מחשב העבודה (על הבית: הצלחה).

## 5. baseline: הבדיקה לפני האימון

| | |
|---|---|
| פקודה | `.\tools\run_work.ps1 -Step baseline -Config configs\my_work_cpu.yaml` (בנתיב `full` זה קורה לבד) |
| Linux | `bash tools/run_work.sh baseline --config configs/my_work_cpu.yaml` |
| מה זה עושה | בודק את המודל הלא-מאומן על קליפי ה-Val; זו נקודת ההשוואה לכל השאר, ובמקביל ממלא את החלק של Val במטמון. |
| מסך תקין **[turbo]** | `PROGRESS WER scoring of the baseline on 'val': clip 12 of 101 (11.9%), elapsed 0:03:22, remaining about 0:24:54` ... בסוף `val (eval_version 1, model whisper-large-v3-turbo-ivrit): 101 files, 4697 words, WER 0.8335, mean per-file WER 0.9613, CER 0.3889` |
| מספרים לקרוא | **WER** (בנתוני Daf Yomi: 0.8335), **CER**, וכמה פלטים הגיעו לתקרת האורך (`hit the length cap`: 1 מתוך 101) |
| קבצים שנוצרים | `reports\evaluation_baseline.json` (המספרים), `predictions_baseline.csv` (שורה לכל קליפ), `review_baseline.txt` (קליפים לשמוע), `eval_result_baseline.json` (עותק מלא); בתיקיית המטמון `data\work_cpu\encoder_cache\enc_...\` |
| איפה מוסבר | `COMPONENT_GUIDE.md` פרק 10 (Evaluator) ופרק 21; `USER_GUIDE.md` סעיף 7 |
| אם נכשל | קוד 4: `Every hypothesis is empty` = שפה/מודל שגויים; `Out of memory while transcribing` = `--set eval.batch_size=1`; קוד 2: `features.encoder_cache.enabled is true but train.freeze_policy ...` = שוכתב `train.freeze_policy`; `Not enough free disk for the encoder cache ... short by X GB` = לפנות דיסק |

| החלטה | |
|---|---|
| WER בין 0.3 ל-1.2, ובקליפים ששומעים (3) הטקסט דומה לעברית שנאמרה | GO |
| WER מעל 1.5, או רוב הפלטים ריקים/חוזרים | STOP: לא מתחילים אימון; בודקים שפה, תוויות ואודיו |

## 6. train: האימון

| | |
|---|---|
| פקודה | `.\tools\run_work.ps1 -Step train -Config configs\my_work_cpu.yaml` (בנתיב `full`: לבד, עם `max_steps` מחושב) |
| Linux | `bash tools/run_work.sh train --config configs/my_work_cpu.yaml` |
| מה זה עושה | מאמן את המפענח (המקודד קפוא); קודם בונה את מטמון המקודד לקליפי ה-Train (פעם אחת), ואז צעדים. |
| מסך תקין **[rehearsal, whisper-tiny]** | `[exp_0002_trial_000] step 10 of 20 (50.0%) (epoch 0.65) train_loss 3.7079 val_loss - lr 3.06e-06 grad_norm 33.535 ... elapsed 0:00:49` ... `Run ended (max_steps). Model of step 20 saved in ...`. עם המטמון, לפני כן: `Encoder cache: 400 of 800 clips done, elapsed ..., remaining about ...` |
| מספרים לקרוא | `train_loss` (אמור לרדת, גם אם רועש); `val_loss` בנקודות הבדיקה (4); האחוזים. ה-`s/step` וה-`remaining` של שורות האימון גבוהים מדי (באג O121), סומכים על `elapsed` ועל האחוזים |
| קבצים שנוצרים | `candidates\exp_NNNN_trial_001\` (נקודות שמירה והמודל המאומן `..._step_SSSSS`, כ-7.4 ג'יגה), `metrics\trial_001\metrics.csv` (עקומת ההפסד; נפתח באקסל), `reports\stage_results\train.json` |
| איפה מוסבר | `COMPONENT_GUIDE.md` פרקים 8 (FeaturePipeline, מטמון המקודד) ו-9 (TrainerCore); `TROUBLESHOOTING.md` (TrainerCore) |
| אם נכשל | קוד 4, `The loss became NaN or infinite at step N` = מורידים `learning_rate` ל-5e-6; `Out of memory during training` = `batch_size 2` + `grad_accum_steps 8`; `Attempt 1 of 6 to rename ... Access is denied` = אזהרה, התוכנה מנסה שוב 5 פעמים; קוד 2: `Cannot resume ... the config differs` = שיניתם הגדרה באמצע (השתמשו באותה פקודה בדיוק); `run.lock` ישן = משנים את שמו ל-`run.lock.stale` (לא מוחקים) |

| החלטה | |
|---|---|
| נגמר עם קוד 0 ו-`Run ended (max_steps)` | ממשיכים ל-evaluate |
| נעצר באמצע | מריצים שוב את אותה פקודה: האימון ממשיך מנקודת השמירה האחרונה (כל 25 צעדים) |
| NaN | STOP: מורידים קצב למידה, ניסוי חדש |

## 7. evaluate: בדיקת המודל המאומן מול הבסיס

| | |
|---|---|
| פקודה | `.\tools\run_work.ps1 -Step evaluate -Config configs\my_work_cpu.yaml -Experiment exp_0003` (בנתיב `full`: לבד) |
| Linux | `bash tools/run_work.sh evaluate --config configs/my_work_cpu.yaml --experiment exp_0003` |
| מה זה עושה | בודק גם את הבסיס וגם את המודל המאומן על אותם קליפי Val, ושומר את ההשוואה הסטטיסטית; ה-Test נדחה כאן תמיד. |
| מסך תקין **[turbo]** | אותן שורות `PROGRESS WER scoring ...` פעמיים (הבסיס והמאומן, מהר יותר בזכות המטמון) ובסוף `val (...): 101 files, 4697 words, WER 0.xxxx ...` |
| מספרים לקרוא | WER ו-CER של המאומן |
| קבצים שנוצרים | `reports\evaluation_trained.json`, `predictions_trained.csv`, `review_trained.txt`, `comparison.json` (ההשוואה: הפרש WER, הרווח הסטטיסטי, פסק הדין) |
| איפה מוסבר | `COMPONENT_GUIDE.md` פרק 10 (Evaluator); `USER_GUIDE.md` סעיף 7 |
| אם נכשל | כמו baseline; בנוסף קוד 2 `Cannot resume ... differs` אם השתמשתם בהגדרות שונות מאלה של ה-train |

## 8. compare: בסיס מול מאומן במסך

| | |
|---|---|
| פקודה | `.\tools\run_work.ps1 -Step compare -Experiment exp_0003 -BaselineFrom exp_0002` |
| Linux | `bash tools/run_work.sh compare --experiment exp_0003 --baseline-from exp_0002` |
| מה זה עושה | קורא את הקבצים השמורים ומדפיס WER, CER, ההפחתה היחסית ופסק הדין (שניות, בלי מודל). |
| מסך תקין **[tiny]** | `Baseline : WER 1.0000  CER 2.3617  (3 clips, 12 words)   exp_0002` / `Trained  : WER 1.0000 ...   exp_0003` / `Relative reduction of the WER: 0.0%   (target 10%: not reached)` / `Verdict (paired bootstrap over lessons): insufficient_groups. Only 3 groups ...` |
| מספרים לקרוא | **Relative reduction** (היעד: לפחות 10%), **Verdict** |
| קבצים שנוצרים | אין (רק מסך) |
| איפה מוסבר | `tools\compare_evals.py` (הכותרת); הדף "איך קוראים את הדוח" בהמשך |
| אם נכשל | קוד 2: `A stored file is missing` = עוד לא הרצתם evaluate על ניסוי האימון |

## 9. report: הדוח

| | |
|---|---|
| פקודה | `.\tools\run_work.ps1 -Step report -Config configs\my_work_cpu.yaml -Experiment exp_0003` |
| Linux | `bash tools/run_work.sh report --config configs/my_work_cpu.yaml --experiment exp_0003` |
| מה זה עושה | בונה דף HTML אחד: גרף ההפסד, האם האימון התכנס, בסיס מול מאומן, הקליפים הגרועים והטובים, ומה לשנות. לא מחשב שום דבר מחדש. |
| מסך תקין **[tiny]** | `The report is in: ...\e\exp_0003\reports\report.html` / `Open it in a browser (it needs no network).` |
| מספרים לקרוא | אין; פותחים את הקובץ |
| קבצים שנוצרים | `reports\report.html` (נכתב מחדש בכל הרצה) |
| איפה מוסבר | `COMPONENT_GUIDE.md` פרק 17 (ReportBuilder); הדף "איך קוראים את הדוח" בהמשך |
| אם נכשל | קוד 2: ניסוי לא קיים או נעול (`run.lock`) |

## 10. final-test: בדיקת ה-Test, פעם אחת בלבד

| | |
|---|---|
| פקודה | `.\tools\run_work.ps1 -Step final-test -Config configs\my_work_cpu.yaml -Experiment exp_0003` ; או בריצה מלאה: `-Step full -WithTest` |
| Linux | `bash tools/run_work.sh final-test --config configs/my_work_cpu.yaml --experiment exp_0003` |
| מה זה עושה | בודק את קבוצת ה-Test פעם אחת, למודל המאומן ולבסיס, ושומר `test_result.json`. **הפעלה שנייה נדחית** (קוד 3). עושים את זה רק בריצה הסופית שבחרתם. |
| מסך תקין **[tiny]** | `FINAL TEST: scoring the test split once, for the baseline and for exp_0003_trial_001_step_00004.` ... `FINAL TEST: not_confirmed. Baseline WER 1.0000, trained WER 1.0000, relative reduction 0.0%. ...` |
| מספרים לקרוא | `status`: **confirmed** או **not_confirmed**; ההפחתה היחסית |
| קבצים שנוצרים | `reports\test_result.json`; `data\work_cpu\experiments\test_split_scored.json` (סימון שה-Test נבדק; **לא מוחקים**) |
| איפה מוסבר | `DECISIONS.md` פריט 90; `COMPONENT_GUIDE.md` פרק 21 |
| אם נכשל | קוד 3 `REFUSED: the Test split was already scored (...)` = ה-Test נבדק כבר; אין להריץ שוב. קוד 2 `has no trained model` = קודם train ו-evaluate |

**החלטה:** `confirmed` = השיפור אושר על נתונים שלא שימשו להחלטה. `not_confirmed` = השיפור לא אושר; לא "מתקנים" ולא בודקים שוב את ה-Test, מתחילים ניסוי חדש עם הגדרות אחרות (ה-Test כבר "נשרף" לניסוי הזה).

## 11. המשך אחרי Ctrl+C או קריסה (resume)

| | |
|---|---|
| פקודה | **אותה פקודה בדיוק**: `.\tools\run_work.ps1 -Step full -Config configs\my_work_cpu.yaml` |
| Linux | `bash tools/run_work.sh full --config configs/my_work_cpu.yaml` |
| מה זה עושה | קורא את `data\work_cpu\experiments\full_run_state.json`, מדלג על שלבים שהסתיימו, וממשיך אימון שלא הסתיים מנקודת השמירה האחרונה. |
| מסך תקין | `A state file was found (...full_run_state.json): this run CONTINUES it (epochs 2).` ואחריה רק השלבים שנותרו |
| איך עוצרים נכון | Ctrl+C **פעם אחת** בחלון הרגיל: האימון שומר נקודת שמירה ונעצר אחרי הצעד (קוד 130). לא סוגרים את החלון, לא מריצים בתוך `| Tee-Object`. |
| אם נכשל | `run.lock` ישן אחרי קריסה: בודקים ש-python לא רץ (`Get-Process python`), משנים את שם הקובץ `run.lock` בתיקיית הניסוי ל-`run.lock.stale` (לא מוחקים) ומריצים שוב; ריצה חדשה מאפס: משנים את שם `full_run_state.json` |
| איפה מוסבר | `RUNBOOK.md` סעיף 8; `USER_GUIDE.md` סעיף 5 |

## 12. הרצה בלי מטמון המקודד (כמו קודם)

| | |
|---|---|
| פקודה | קודם משנים את שם הקובץ `data\work_cpu\experiments\full_run_state.json` (ריצה חדשה), ואז: `.\tools\run_work.ps1 -Step full -Config configs\my_work_cpu.yaml -Set features.encoder_cache.enabled=false` |
| Linux | `bash tools/run_work.sh full --config configs/my_work_cpu.yaml --set features.encoder_cache.enabled=false` |
| מה זה עושה | אותה ריצה, אבל המקודד רץ בכל צעד ובכל פענוח (איטי בהרבה: 2 סבבים על 800 קליפים כ-8.3 שעות במקום 4.9, הערכה מהמחשב הביתי). התוצאות אמורות להיות זהות (נבדק: ערכי ההפסד והטקסטים זהים). |
| קבצים | אין מטמון; התיקייה `data\work_cpu\encoder_cache` נשארת בדיסק ואפשר לחזור אליה |
| למה | אם יש חשד שהמטמון משנה תוצאות (בדיקת הצלבה), או במודל שאינו turbo עם dropout (אז המטמון מסרב ממילא) |
| איפה מוסבר | `DECISIONS.md` פריט 89; `COMPONENT_GUIDE.md` פרק 8 |

---

# איך קוראים את הדוח (Reading the report)

פותחים `data\work_cpu\experiments\exp_NNNN\reports\report.html` בדפדפן. אין צורך באינטרנט. הדוח כולו בהסבר דו-לשוני (אנגלית ועברית).

| מה רואים | מה זה אומר |
|---|---|
| **שורת הפסק** בראש (ירוקה/כתומה/אפורה) | `Improved` = המודל המאומן טוב בבירור מהבסיס. `No clear difference` = ייתכן שמקרי. `Worse` = גרוע מהבסיס. `Too few lessons` = פחות מ-5 שיעורים ב-Val, אין קביעה סטטיסטית. |
| **WER** | אחוז המילים השגויות (מוחלפות + חסרות + מיותרות) חלקי מילות הייחוס. 0.30 = 30 שגיאות לכל 100 מילים. יכול לעלות על 1.0 כשהמודל מוסיף הרבה מילים. נמוך יותר = טוב יותר. |
| **CER** | אותו דבר באותיות. |
| **Relative reduction** | (WER בסיס פחות WER מאומן) חלקי WER בסיס. 10% = ירידה של עשירית מהשגיאות. היעד שלכם: לפחות 10%. |
| **הרצועה הירוקה 0.10-0.30** על עמודות ה-WER | יעד ה-WER המוחלט שלכם; עמודה בתוכה או מתחתיה עומדת ביעד. זה ערך תצוגה קבוע, לא תוצאה. |
| **גרף ההפסד** | קו כחול מלא = Val loss (בכל בדיקה); קו כתום מקווקו = train loss. הפסד נמוך יותר = המודל "מופתע" פחות מהטקסט הנכון. **העיגול הירוק = best checkpoint**: הנקודה עם ה-Val loss הנמוך ביותר; זה המודל שנשמר. |
| **Still improving** | הנקודה האחרונה של Val היא הנמוכה ביותר וירדה ביותר מ-1% לעומת הקודמת: האימון קצר מדי. |
| **Plateau** | ה-Val loss הפסיק לרדת: עוד צעדים באותה צורה לא יעזרו הרבה. |
| **Overfitting** | ה-Val loss עלה אחרי צעד N בזמן שה-train loss ירד: המודל משנן את האימון. המודל שנשמר הוא של צעד N. |
| **Bootstrap verdict** | מדמה שוב ושוב הרכבים אחרים של שיעורים כדי לבדוק אם ההפרש עמיד: `improved` = החסם התחתון (ברמת ביטחון 90%) מעל אפס. |
| **confirmed / not_confirmed** | רק בסעיף "FINAL - Test": confirmed = בבדיקת ה-Test החד-פעמית ההשוואה יצאה `improved` באותה רמת ביטחון. not_confirmed = לא. |
| **טבלת 15 הקליפים הגרועים/הטובים** | התווית, מה שכתב הבסיס, מה שכתב המאומן: לקרוא כדי לראות מה השתנה בפועל. |
| **What to change next** | הכללים: משתפר עדיין ← יותר צעדים; overfitting ← פחות צעדים או קצב למידה נמוך; שטוח ← קצב למידה גבוה יותר; הפסד טוב אבל WER רע ← לבדוק פלטים ריקים/חוזרים ואת התוויות. |

הערות: ה-Test מופיע רק אחרי `final-test`. הקביעה "התכנס / שטוח / overfitting" מחושבת מהנתונים (קבועים בקוד: 1% לשיפור ו-1% לעלייה), לא מנחשת.

---

# מילון מונחים (20)

| מונח | משמעות |
|---|---|
| **step (צעד)** | עדכון אחד של המשקלים. כאן: 16 קליפים לצעד (4 × 4). |
| **epoch (סבב)** | מעבר אחד על כל קליפי ה-Train. `max_steps = epochs × ceil(Train / 16)`. |
| **clip (קליפ)** | קובץ אודיו של עד 30 שניות עם הטקסט שלו. |
| **group (קבוצה)** | כל הקליפים של שיעור/הקלטה אחת; לעולם לא מתפצלים בין Train ל-Test. |
| **Train** | החלק שהמודל לומד ממנו. |
| **Val** | החלק שמשווים עליו מודלים והגדרות. |
| **Test** | החלק שנבדק פעם אחת בלבד, בסוף. |
| **baseline (בסיס)** | המודל לפני כל אימון. |
| **trained (מאומן)** | המודל אחרי האימון. |
| **encoder / decoder** | המקודד שומע את האודיו; המפענח כותב את הטקסט. כאן מאמנים רק את המפענח. |
| **freeze (הקפאה)** | חלק במודל שלא מתעדכן באימון (המקודד). |
| **cache (מטמון)** | הפלט של המקודד הקפוא נשמר פעם אחת לכל קליפ ומשמש שוב ושוב; תוצאות זהות, הרבה יותר מהר. |
| **learning rate (קצב למידה)** | גודל כל עדכון; גדול מדי ← NaN/קפיצות; קטן מדי ← לא לומד. |
| **loss (הפסד)** | כמה המודל "מופתע" מהטקסט הנכון; לא אותו דבר כמו WER. |
| **Val loss / train loss** | ההפסד על Val ועל קליפי האימון. |
| **checkpoint (נקודת שמירה)** | תמונת מצב של המודל באימון, כדי להמשיך או לשמור את הטוב ביותר. |
| **overfitting (התאמת יתר)** | המודל משנן את האימון ומתדרדר על נתונים חדשים. |
| **WER / CER** | שיעור שגיאות מילים / אותיות. |
| **bootstrap** | בדיקה סטטיסטית שמדמה הרכבים אחרים של שיעורים כדי לדעת אם שיפור הוא מקרי. |
| **experiment (ניסוי)** | תיקייה `exp_NNNN` שמחזיקה הכול מפקודה אחת (הגדרות, לוגים, דוחות, מודלים). |

---

# טבלת החלטות: אני רואה X ← אני עושה Y ← הפקודה הבאה

| אני רואה | אני עושה | הפקודה הבאה |
|---|---|---|
| WER של הבסיס מעל 1.5 | עוצרים; בודקים שפה, תוויות ואודיו ב-`review_baseline.txt` | לתקן ואז `-Step baseline` |
| פלטים ריקים (`Every hypothesis is empty`, קוד 4) | בודקים `model.language: he` ושהמודל הנכון בנתיב | `-Step baseline` שוב |
| מילים חוזרות (הרבה פלטים `hit the length cap`) | מאמנים (זה מצטמצם באימון על תוויות נקיות); לא מנסים להסתיר | `-Step full` |
| ה-loss עדיין יורד בסוף ("Still improving") | יותר צעדים | `-Step full -Epochs 3` בניסוי חדש (שם קובץ state אחר) |
| ה-loss עולה אחרי צעד N ("Overfitting") | פחות צעדים או קצב למידה נמוך (חצי) | ניסוי חדש: `-Set train.hyperparams.learning_rate=5e-6` או `-Epochs 1` |
| ה-loss שטוח ("Plateau") ו-WER לא השתפר | קצב למידה גבוה יותר (כפול), ואז בודקים תוויות | `-Set train.hyperparams.learning_rate=2e-5` בניסוי חדש |
| NaN (`The loss became NaN`, קוד 4) | קצב למידה נמוך (5e-6) | ניסוי חדש עם `-Set train.hyperparams.learning_rate=5e-6` |
| זיכרון נגמר (`Out of memory`, קוד 4) | סוגרים תוכניות; `batch_size 2` ו-`grad_accum_steps 8`; `eval.batch_size 1` | ניסוי חדש עם `-Set train.hyperparams.batch_size=2 -Set train.hyperparams.grad_accum_steps=8` |
| הדיסק מלא (`No space left` או `short by X GB`) | מפנים (רק מה שבטוח, למשל `candidates\exp_NNNN_benchmark`) | אותה פקודה `full` שוב (ממשיכה) |
| הריצה נקטעה (Ctrl+C, חשמל, קריסה) | בודקים `run.lock` ישן ומשנים שם | אותה פקודה `-Step full` שוב |
| `prepare` דוחה הרבה שורות | קוראים `rejected.csv`; מתקנים שמות עמודות/נתיבים; לא מעלים את הסף | `-Step prepare` |
| פחות מ-5 קבוצות ב-Val/Test | מוסיפים נתונים או משנים את עמודת הקבוצה | `-Step prepare` |
| הרבה קליפים הוסרו על `max_duration` | חותכים אודיו לקליפים של עד 30 שניות עם טקסט תואם | `-Step prepare` |
| `final-test` סירב (קוד 3) | זה בכוונה: ה-Test כבר נבדק | אין |

**"ניסוי חדש"** בטבלה = קודם משנים את שם הקובץ `data\work_cpu\experiments\full_run_state.json` (למשל ל-`full_run_state.json.run1`) ואז מריצים את `-Step full` עם ה-`-Set` המתאים; הניסוי החדש יקבל את המספר הבא. אל תשנו את `-ExperimentsDir` (הוא חייב להיות אותו נתיב כמו `run.experiments_dir` בהגדרות). שימו לב: אם כבר הרצתם `final-test` בתיקיית הניסויים הזו, `final-test` יסרב גם בניסוי חדש (קוד 3), כי ה-Test נבדק פעם אחת בלבד לתיקייה.

**UNTESTED:** כל הפקודות והמסכים בחלק הזה הורצו בבית (מחשב ביתי, מודל turbo בחלקם ומודל זעיר בחלקם). שום דבר מהם לא הורץ על מחשב העבודה.
