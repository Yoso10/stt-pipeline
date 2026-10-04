# Runbook (operator guide)

## How to run (the short version)
```
stt check                                   # is this machine ready?
stt smoke --config configs/smoke.yaml       # does the whole chain work? (tiny model, synthetic data)
stt run --config configs/home_full.yaml     # the whole pipeline, one command
```
`stt` is a placeholder for the command name (O62). This file describes the TARGET usage; the agent verifies every command against the real CLI and updates this file when the CLI exists. Every command also prints `--help` with an example.

## 1. What this is
An offline, CPU-only, non-interactive pipeline that fine-tunes a Whisper model on labeled audio, evaluates it with WER, and returns a winning model, or says that the baseline stays. You run it alone; it needs no network and no admin rights.

## 2. One-time setup on a machine
1. Python 3.12 (or the version the project settled on, O1) installed for your user.
2. Create a virtual environment and install offline from the wheel bundle:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install --no-index --find-links wheels -r requirements.lock
   ```
3. Copy `.env.example` to `.env` and fill in the machine settings (folders, core counts).
4. Verify: `stt check`. It lists everything that is wrong at once, verifies the code against `MANIFEST.sha256` and prints the code version.

## 3. Moving to the closed machine (checklist)
- [ ] The release folder (code) with `MANIFEST.sha256`
- [ ] The `wheels/` folder and `requirements.lock`
- [ ] The model folder (for example `models/whisper-tiny`)
- [ ] The data folder
- [ ] `stt check` passes there
- [ ] `stt smoke` passes there

## 4. Daily use
| Goal | Command | Example |
|---|---|---|
| everything, one command | `run` | `stt run --config configs/work.yaml` |
| only prepare and inspect the data | `prepare` | `stt prepare --config configs/work.yaml` |
| one training run | `train` | `stt train --config configs/home_full.yaml` |
| hyperparameter search | `search` | `stt search --config configs/home_full.yaml` |
| evaluate a model | `evaluate` | `stt evaluate --config configs/home_full.yaml --set model.path=models/whisper-tiny` |
| rebuild the report of a past run | `report` | `stt report --experiment exp_0003` |
| export a registered model | `export` | `stt export --entry model_0002` |
| list or verify the registry | `registry` | `stt registry verify --entry model_0002` |
| learning curve | `learning-curve` | `stt learning-curve --config configs/home_full.yaml` |
| time per step and projected time | `benchmark` | `stt benchmark --config configs/home_full.yaml` |
| propose old experiments to delete | `cleanup` | `stt cleanup` (lists only; deleting needs an explicit flag) |
| regenerate the config key reference | `config-docs` | `stt config-docs` |
Shared options: `--config <file>`, `--set a.b=value`, `--parent exp_0003`.

## 5. Following a run
- Live numbers: `experiments/exp_NNNN/metrics/` (a CSV you can open in Excel, and a page `progress.html` that refreshes itself).
- The log: `experiments/exp_NNNN/logs/run.log`. Status: `status.json`.
- At the end the command prints one JSON summary (experiment id, verdict, key numbers, paths).

## 6. Reading the results
- Open `experiments/exp_NNNN/reports/report.html` in a browser. The top shows the verdict in plain words; below are the graphs, correlations, hints and the listening list. Hints are hints, not proof.
- Listen to the files in the listening list, and judge for yourself: your listening can reveal a model that failed in practice although it passed the test.
- The winner is in `registry/model_NNNN/` with `model_card.json`, `MODEL_CARD.md` and `SHA256SUMS`.

## 7. Stopping and resuming
- Ctrl+C stops gracefully, saves a checkpoint and marks the run `stopped`. Run the same command again to resume.
- Do not close the console window to stop a run; if you must, the last periodic checkpoint is used on resume.
- The machine is kept awake during `run`, `search` and `train` (Windows), unless `run.prevent_sleep: false`.

## 8. When something fails
1. Read the last lines of the message: they name the stage, the key or the path, and the fix.
2. Exit code: 2 config, 3 data, 4 training or evaluation, 5 export or verification, 130 interrupted.
3. Open `docs/TROUBLESHOOTING.md` at the component named in the message.
4. If you ask the internal agent for help, give it `CLAUDE.md`, `docs/COMPONENTS.md`, the error and `logs/run.log`. The rule: diagnose first, smallest fix, no interface changes.
5. Same seed gives the same result on the same machine only; do not compare exact numbers between machines.

## 9. Do and do not
- Do keep every experiment folder; never edit `config_full.yaml`; use `--parent` to derive a new experiment.
- Do not delete registry entries; do not turn on online mode on the closed machine; do not look at the Test results to choose between experiments (the report shows how many times the test set was already looked at).
- Do not raise a safety gate just to make a run pass: read the report that explains why it failed.
