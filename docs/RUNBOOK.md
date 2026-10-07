# Runbook (operator guide)

## How to run (the short version)
Open PowerShell in the project folder, then once per window:
```
.\.venv\Scripts\Activate.ps1                 # the project's Python (its packages are installed there)
$env:PYTHONPATH = "src"                      # tells Python where the project code is
```
Then:
```
python -m stt check --config configs/work_like.yaml      # is this machine ready? (changes nothing)
python -m stt smoke --full                               # does the whole chain work? tiny model, synthetic data, about 1 minute
python -m stt run --config configs/work_like.yaml        # the rehearsal on the real data, about 35 minutes to 1 hour
```
The command is `python -m stt` (there is no program called `stt`). Every command prints `--help` with an example. Every command example in this file was checked against the real command line on 2026-10-08 (`python -m stt <command> --help` for every command, an automatic comparison of every option named here with the real options, and the commands run on synthetic data).

## 1. What this is
An offline, CPU-only, non-interactive pipeline that fine-tunes a Whisper model on labeled audio, evaluates it with WER, and returns a winning model, or says that the baseline stays. You run it alone; it needs no network and no admin rights.

## 2. One-time setup on a machine
1. FIRST: put the whole project (code, `wheels/`, `.venv`) at a SHORT path, for example `C:\stt` (the venv is then `C:\stt\.venv`). Keep the path to the project folder at 40 characters or fewer. Windows without long-path support fails to install torch in a deep folder (`pip OSError: No such file ...` with a very long path).
2. Python 3.12 (any version works if the wheel bundle was made for it: `python tools/fetch_wheels.py --python-version X.Y` at home) installed for your user.
3. Create a virtual environment and install offline from the wheel bundle:
   ```
   cd C:\stt
   python -m venv .venv
   .venv\Scripts\activate
   pip install --no-index --find-links wheels -r requirements.lock
   ```
4. Copy `.env.example` to `.env` and fill in the machine settings (folders, core counts) if you use them.
5. Verify: `python -m stt check` (with `$env:PYTHONPATH = "src"`). It lists everything that is wrong at once, says whether `MANIFEST.sha256` is there and prints the code version. To verify the code files against the manifest: `python tools/make_manifest.py --root . --verify MANIFEST.sha256`.

## 3. Moving to the closed machine (checklist)
- [ ] The release folder (code) with `MANIFEST.sha256` (made at home: `python tools/make_manifest.py --root . --out MANIFEST.sha256`; make it from the very folder you copy)
- [ ] The `wheels/` folder and `requirements.lock`
- [ ] The model folder (for example `models/whisper-tiny`)
- [ ] The data folder
- [ ] `python -m stt check` passes there, and `python tools/make_manifest.py --root . --verify MANIFEST.sha256` says VERIFIED
- [ ] `python -m stt smoke --full` passes there

## 4. The first real run (what to do, in this order)
The ready files are in `configs/`: `work_like.yaml` (the REHEARSAL: about one tenth of the Daf Yomi data, 20 training steps, about 35 minutes to 1 hour), `home_full.yaml` (ALL the data, about 3.5 to 5 hours) and `work.yaml` (the template for your own data: copy it to a NEW name first, for example `Copy-Item configs\work.yaml configs\my_work.yaml`, and fill in every `<<< FILL IN ... >>>` marker).
1. `python -m stt check --config configs/work_like.yaml`. You see OK lines for Python, the packages, the model folder and the data folder, the free disk, the memory and the cores, and at the end "Everything checked is fine." A line that starts with `PROBLEM` (or `ERROR`) names exactly what to fix.
2. `python -m stt benchmark --config configs/work_like.yaml`, about 2 minutes. Close the other heavy programs first (a browser, Visual Studio Code, a sync tool), run it TWICE and trust the LARGER planned time: the measurement is short and this computer's speed varies. The benchmark prints the seconds per training step, per Val-loss example and per clip, the planned time of the run part by part, and the resources: the size of one trial folder, the planned disk of the whole run and the peak memory. It does not touch the Test split.
   The seconds per step it prints can be up to twice too high (a known timing error, O121 in docs/OPEN_ISSUES.md), so the planned time is on the safe side.
3. `python -m stt run --config configs/work_like.yaml`. What you see on the screen, in order: `Stage prepare ...`, then `PLAN: planned time of python -m stt run ... about 37 min` with a line per part (or `no benchmark numbers for this model and thread count: run python -m stt benchmark`), then the progress lines described in section 6, and at the end `VERDICT: ...`, `Test: ...` and the improvement line (section 7).
4. Open the report (section 7), listen to the listening list, and only then decide about `home_full.yaml`.
Free disk: a run keeps about 600 MB per trial folder: the rehearsal needs about 4 GB, the full run about 4 to 6.5 GB (the benchmark prints the planned disk for your settings).

## 5. Daily use
| Goal | Command | Example |
|---|---|---|
| everything, one command | `run` | `python -m stt run --config configs/home_full.yaml` |
| only prepare and inspect the data | `prepare` | `python -m stt prepare --config configs/my_work.yaml` |
| one training run | `train` | `python -m stt train --config configs/home_full.yaml` |
| hyperparameter search | `search` | `python -m stt search --config configs/home_full.yaml` |
| evaluate a model | `evaluate` | `python -m stt evaluate --config configs/home_full.yaml --set model.path=models/whisper-tiny` |
| rebuild the report of a past run | `report` | `python -m stt report --config configs/work_like.yaml --experiment exp_0003` |
| export a registered model (the entry is the first word after `export`; default: the latest) | `export` | `python -m stt export model_0002 --config configs/work_like.yaml` |
| list the registry (with the number of exports) | `registry list` | `python -m stt registry list --config configs/work_like.yaml` |
| verify an entry against its checksums (the entry is optional) | `registry verify` | `python -m stt registry verify model_0002 --config configs/work_like.yaml` |
| rebuild the registry index from the entry cards | `registry rebuild-index` | `python -m stt registry rebuild-index --config configs/work_like.yaml` |
| learning curve | `learning-curve` | `python -m stt learning-curve --config configs/home_full.yaml` |
| time per step and projected time | `benchmark` | `python -m stt benchmark --config configs/home_full.yaml` |
| PROPOSE what could be deleted to free disk (it only writes a text file and never deletes) | `cleanup` | `python -m stt cleanup --config configs/work_like.yaml` |
| regenerate the config key reference (`docs/CONFIG_REFERENCE.md`) | `config-docs` | `python -m stt config-docs` |
| what is fine and what is not on this machine | `check` | `python -m stt check --config configs/work_like.yaml` |
Shared options: `--config <file>`, `--set a.b=value` (repeatable; the last one wins), `--parent exp_0003` (the new experiment extends that one), `--experiment exp_0004` (continue that experiment). The registry commands need the same `--config` as the run (they read `registry.dir` from it).

## 6. Following a run
- On the screen, in every long phase, a plain line at most every `progress.interval_seconds` (30 seconds):
  - the clips of a WER scoring: `PROGRESS WER scoring of the baseline on 'val': clip 48 of 101 (47.5%), elapsed 0:01:51, remaining about 0:02:02`;
  - the training steps: `[exp_0002_trial_001] step 10 of 20 (50.0%) (epoch 0.65) train_loss 3.0294 val_loss - ... elapsed 0:00:55 remaining 0:02:44`. The `s/step` and `remaining` of the TRAINING lines are about 3 times too high (O121); the elapsed time and the percent are right. The real speed is about 5 seconds per step on this machine;
  - the search: `Search progress: trial 2 of 4 finished (50.0%), elapsed 0:06:47, remaining about 0:06:47`;
  - the finalists and the learning-curve points: `PROGRESS finalists: starting finalist 2 of 2 (50.0% done) ...`.
  Change the interval with `--set progress.interval_seconds=10`.
- The files: `experiments/exp_NNNN/logs/run.log` is ALWAYS written (every detail, with the time), `status.json` says where the run is, `metrics/` holds a CSV per trial that Excel can open, and `metrics/<trial>/progress.html` is a page that refreshes itself (switch it on in your YAML file with `train:` / `  sinks: [csv, console, html]`; a JSON value given with `--set` on the command line is mangled by PowerShell 5.1). At the end the command prints one JSON summary (experiment id, verdict, key numbers, paths).
- Watching the log in a second window: `Get-Content experiments\exp_0001\logs\run.log -Wait -Tail 20`.
- To also keep a copy of the screen in a file, use ONE of these (verified in Windows PowerShell 5.1; the plain `... 2>&1 | Tee-Object -FilePath x` prints a red `NativeCommandError` block at the first line, because PowerShell 5.1 turns every error-stream line into an error record; the program is fine):
  ```
  python -m stt run --config configs/work_like.yaml 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath data\run.log
  cmd /c "python -m stt run --config configs/work_like.yaml 2>&1" | Tee-Object -FilePath data\run.log
  ```
  The plain command without a pipe shows no red block. IMPORTANT: Ctrl+C inside such a pipe is NOT graceful (see section 8); `logs\run.log` of the experiment is a log file you already get without any pipe.

## 7. Reading the results
- Open `experiments/exp_NNNN/reports/report.html` in a browser (it needs no network). The top shows the verdict in plain words:
  - **Baseline retained**: no finalist beat the untouched base model on Val, so the base model stays.
  - **Proven improvement**: the winner is clearly better on Val AND the one-time Test look confirms it.
  - **No proven improvement**: the best finalist looks better but is not proven (the Val difference is not clear, or the Test does not confirm it).
- THE TARGET LINE: `Relative reduction of the Val WER against the baseline: 9.2% (target: 10%): not reached`. It says by how many percent the Val WER of the best model is below the Val WER of the untouched base model on the SAME Val clips: (baseline - best) / baseline. The target is the key `report.target_relative_wer_reduction` (0.10 = 10%; with no target set the line says `no target set`). It appears under `VERDICT:` at the end of `run` and in the verdict box of the report. If the target is reached but the verdict is not "Proven improvement" the line adds `Point estimate only`: the number looks good but is not proven, read the verdict first. A WER above 1.0 (for example 1.13) is possible: it means more word errors than reference words; whisper-tiny often loops or invents text in Hebrew.
- The history chart (the first section after the verdict) shows, for every registered experiment, the Val WER of the baseline and of the best model, and the dashed target line: one point per run, so you can SEE whether you are improving. A rehearsal and a full run must not share one registry (the `work_like.yaml` preset has its own `data/work_like/registry`).
- Listen to the files in the listening list and judge for yourself: your listening can reveal a model that failed in practice although it passed the test. Hints are hints, not proof.
- The winner is in `registry/model_NNNN/` with `model_card.json`, `MODEL_CARD.md`, `SHA256SUMS` and `exports.json` (the automatic export of the model folder, verified to load; the model is in `registry/model_NNNN/model/`).
- Do not look at the Test results to choose between experiments: the Test is looked at ONCE per model, only for the winner and the baseline (the report says how many times the Test set was already looked at).

## 8. Stopping and resuming
- Ctrl+C on the plain command (no pipe) stops gracefully: the first press says `Stop requested: finishing safely`, a training saves a checkpoint and the run is marked `stopped`; press it again to stop at once. Run the same command with `--experiment exp_NNNN` to continue (the message at the stop tells you the number): finished stages are skipped and an interrupted training resumes from its checkpoint.
- Ctrl+C inside a PowerShell pipe (`... | Tee-Object`) is NOT graceful: the program is killed at once (seen on 2026-10-08: no stop message, the running trial lost, a stale `run.lock` left). Stop such a run with Ctrl+C only if you must; then continue as below.
- If the program says the lock `run.lock` is STALE (a run died without releasing it): check that no run is active (`Get-Process python`), then do NOT delete the file: rename it, `Rename-Item experiments\exp_0002\run.lock run.lock.stale`, and run the same command again.
- A trial that was interrupted cannot be resumed without a complete checkpoint: its folder is renamed to `..._stale_1` (never deleted) and the trial starts fresh.
- On Windows a trial can fail at its very end with `PermissionError: [WinError 5] Access is denied` when saving its model folder (another program, for example the antivirus or a sync tool, held the new folder for a moment; O122 in docs/OPEN_ISSUES.md). The search goes on without that trial. Excluding the project folder from the antivirus scan and from the sync tools prevents it.
- Do not close the console window to stop a run. The machine is kept awake during `run`, `search` and `train` (Windows), unless `run.prevent_sleep: false`.

## 9. Disk and what is deleted (read this once)
- NOTHING is deleted automatically by the pipeline, and `python -m stt cleanup` only PROPOSES: it writes `cleanup_proposal_<time>.txt` next to the experiments folder (what could go, how big, and why; never anything of the registry, a running or locked experiment) and has no option that deletes. You delete the folders you agree with yourself.
- The registry settings you should know (the defaults of `registry.*`): `registry.cleanup: true` and `registry.transfer: move`. They mean: after the winner is registered, its model folder is MOVED into the registry (`registry/model_NNNN/model/`), so the experiment's `candidates/..._final_trial_K` folder no longer holds that model (only its checkpoints stay there), and `cleanup: true` would let the registry delete the other finalists' folders and checkpoints IF the caller named them. The Orchestrator currently names NO folder, so it deletes nothing; the defaults themselves were not changed (O91). If you want the winner COPIED instead of moved, set `registry.transfer: copy` (it needs more disk).
- Each trial folder is about 600 MB for whisper-tiny (a checkpoint holds about 3 times the model size). A run of 6 trials and finalists is 3.7 GB, up to 6.4 GB.

## 10. When something fails
1. Read the last lines of the message: they name the stage, the key or the path, and the fix.
2. Exit code: 2 config, 3 data, 4 training or evaluation, 5 export or verification, 130 interrupted.
3. Open `docs/TROUBLESHOOTING.md` at the component named in the message.
4. If you ask the internal agent for help, give it `CLAUDE.md`, `docs/COMPONENTS.md`, the error and `logs/run.log`. The rule: diagnose first, smallest fix, no interface changes.
5. Same seed gives the same result on the same machine only; do not compare exact numbers between machines.

## 11. Do and do not
- Do keep every experiment folder; never edit `config_full.yaml`; use `--parent` to derive a new experiment.
- Do not delete registry entries; do not turn on online mode on the closed machine; do not look at the Test results to choose between experiments.
- Do not raise a safety gate just to make a run pass: read the report that explains why it failed.
- Do keep `train.hyperparams.warmup_steps` at 3 or less in a config you run `benchmark` on (the benchmark trains only 3 steps and refuses a longer warmup).
