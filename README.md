# stt-pipeline: an offline Whisper fine-tuning pipeline for Hebrew, on the CPU

## How to run (the short version)

```
git clone <the address of this repository> stt
cd stt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --no-index --find-links wheels -r requirements.lock     # offline; see "Set up the environment"
Copy-Item configs\turbo_full.yaml configs\my_turbo_full.yaml        # then fill the 5 marked lines (<<< FILL IN) in the copy
.\.venv\Scripts\python.exe tools\verify_clone.py --config configs\my_turbo_full.yaml
.\tools\run_work.ps1 -Step full -Config configs\my_turbo_full.yaml -Epochs 2 -Threads 4     # the whole run, no questions
# when it ends it prints the path of the report: open that report.html in a browser
```

Run every command from the project folder (the one that holds `src\`, `tools\`, `configs\`). The same `full` command again resumes a stopped run. Windows is the tested system; `tools/run_work.sh` (Linux) is written but UNTESTED.

## What the project is

A "factory" that takes labeled audio (clips of at most 30 s with their text), splits it by lesson/recording into Train / Val / Test, fine-tunes a Hebrew Whisper model (default `ivrit-ai/whisper-large-v3-turbo`), scores it before and after with WER, and writes one self-contained HTML report. It is built to run on a closed machine: CPU only, no internet at run time, no questions asked, no administrator rights. Python, typed, one component per responsibility (list: `docs/COMPONENTS.md`).

## Current state

- **Components built:** C0 Config, C1 DataSource, C1b SampleFilter, C2 TextNormalizer, C3 Splitter, C3b Subsampler, C4 FeaturePipeline (with the encoder cache), C5 ModelLoader, C6 TrainerCore, C7 Evaluator, C8 HyperparamSearcher, C9 Selector, C10 ModelRegistry, C11 ModelExporter, C12 RunManager, C13 Orchestrator and CLI, C14 ReportBuilder (`docs/COMPONENTS.md`). Open items and their status: `docs/OPEN_ISSUES.md`; every decision: `docs/DECISIONS.md`.
- **Tests:** 885 unit tests, offline (`python -m unittest discover -s tests`, about 8 minutes). Latest full run, 2026-10-27, home laptop: 885 run, all passed (the guard test `test_zz_project_folder_guard` also checks that no test writes into the project folder).
- **Measured with the turbo model on the home laptop (4 physical cores, 16 GB, CPU):**
  - on 8 clips (speed tests): a training step 60 s -> 7.5 s, a Val-loss pass 120 s -> 5 s, decoding 130 s -> 24 s with the encoder cache, identical results (`docs/COMPONENTS.md`, C4);
  - baseline scoring of the turbo model on the 101 Val clips of the 900-clip subset, no cache: 1681 s, WER 0.8335 (`data/work_like/experiments/exp_0002`, not in git);
  - peak RAM of a training step about 6.3 GiB (4-clip step).
- **UNTESTED:** a complete turbo run (baseline, training, evaluation, report) on a whole dataset; the resume of a turbo training after Ctrl+C or a crash (the tiny test model resumes correctly); the 4.6 GB checkpoint writes on a nearly full disk; everything on the work machine (its speed, memory, antivirus); the Linux script; the GPU path of `runtime.device` (the home laptop has no GPU). The time figures for a whole run are ESTIMATES scaled from the 8-clip tests (`docs/WORK_RUNBOOK.md`).

## Set up the environment

1. Python 3.12 for your user; the project at a SHORT path (for example `C:\stt`).
2. `python -m venv .venv`, then `.\.venv\Scripts\Activate.ps1`.
3. Packages (`requirements.lock` has the exact versions):
   - with internet: `pip install -r requirements.lock`;
   - closed machine: make the folder `wheels\` once on a machine WITH internet and the same Python version: `python tools\fetch_wheels.py --requirements requirements.txt --out wheels --lock requirements.lock --python-version 3.12 --platform win_amd64`; copy it to the closed machine; `pip install --no-index --find-links wheels -r requirements.lock`.
4. Check: `.\.venv\Scripts\python.exe tools\verify_clone.py` (read only; it lists what is missing). What the machine has: `.\.venv\Scripts\python.exe tools\probe_env.py`.
5. Needs: RAM about 8 GB free for a turbo training; disk about 4 MB per clip for the encoder cache plus about 8 GB for the trial folder (a second 4.6 GB checkpoint can add to it); 4 or more physical cores; no GPU.

## What is NOT in git, and where to place it

| Item | Where | How to get it |
|---|---|---|
| Data (`manifest.csv` + audio, or parquet files) | any folder; its path goes in `data.source.path` of your config | your own; format: `docs/DATA_ONBOARDING.md` |
| Model | `models\whisper-large-v3-turbo-ivrit` (11 files, 3.24 GB) | on a machine with internet: `python tools\fetch_model.py --repo ivrit-ai/whisper-large-v3-turbo --out models\whisper-large-v3-turbo-ivrit`, then copy the folder |
| Wheels | `wheels\` (about 230 MB) | see "Set up the environment" |
| Experiments, registry, encoder cache | folders named by `run.experiments_dir`, `registry.dir`, `features.encoder_cache.dir` (created by the run) | produced by the run |
| `.venv` | project folder | `python -m venv .venv` |

`tools\verify_clone.py` prints a WHAT IS MISSING list of exactly these.

## The commands

`$env:PYTHONPATH = "src"` once per PowerShell window, then `python -m stt <command> --config FILE`. Every command makes an experiment folder `exp_NNNN` and prints one JSON summary. Exit codes: 0 ok, 2 settings, 3 data (also a refused second Test), 4 training or evaluation, 5 registry/export, 130 Ctrl+C.

| Command | What it does |
|---|---|
| `check` | what is fine and what is not on this machine and in the config (no config: the base settings) |
| `prepare` | load, split, normalize, filter; writes the reports; no training |
| `benchmark` | seconds per step on this machine and the planned time of a run |
| `evaluate` | prepare, then the baseline (and the trained model of the experiment) on Val |
| `train` | prepare, then one training run |
| `search` | hyperparameter search on a screening subset |
| `run` | the whole pipeline: prepare, baseline, search, finalists, selection (Test ONCE), registry, export |
| `learning-curve` | the same settings on growing Train sizes; Val WER per size |
| `report` | rebuild `reports/report.html` of an experiment from its stored files |
| `final-test` | the one-time Test scoring of a finished experiment |
| `export`, `registry list / verify / rebuild-index` | export a registered model; read the registry |
| `cleanup` | WRITES a proposal of what could be deleted; deletes nothing |
| `config-docs` | writes `docs/CONFIG_REFERENCE.md` (every config key) from the code |

`python -m stt --help` lists all of them with their options (`--set a.b=value`, `--experiment exp_NNNN` to continue, `--parent`).

## The one-command path

`.\tools\run_work.ps1 -Step full -Config <your yaml> -Epochs 2 -Threads <physical cores>` runs probe, check, prepare, baseline, train, evaluate, compare and report, and resumes when run again. It sets `max_steps`, the evaluation points and the warmup from the real Train size (`max_steps = epochs x ceil(Train / 16)`), prints `CHOSEN: ...` and `ESTIMATED TIME`, and never opens Test. Options: `-Dry` (print only), `-ExperimentsDir` (must equal `run.experiments_dir` of the config), `-Set 'key=value'`, `-WithTest` (the one-time Test, final run only). Single steps: `-Step probe | check | prepare | benchmark | baseline | train | evaluate | compare | report | final-test`. State of the run: `<experiments_dir>\full_run_state.json`.

## The encoder cache and its safety conditions

With a frozen encoder (`train.freeze_policy: freeze_encoder`) the encoder output is computed once per clip (about 3.84 MB in float16, one atomic file per clip) and reused for training, Val loss and decoding: same results, far faster on the CPU. Key per clip: id, audio path, file size, modification time, data type and the encoder fingerprint; only missing clips are computed, so do not move or `touch` the audio. Safety (from the code, `docs/COMPONENTS.md` C4): a one-time self-check compares the loss with and without the cache on 3 clips and stops the run if it differs; the run is REFUSED (exit 2) when the freeze policy is not `freeze_encoder`, when dropout/layerdrop is above 0, with `apply_spec_augment`, or when the disk is too small. Switch off: `features.encoder_cache.enabled: false`.

## The device key

`runtime.device`: `cpu` (default and the only tested value), `cuda` (fails with exit code 2 when no usable GPU is found, never a silent fall back) or `auto` (cuda when torch reports one, else cpu). GPU use is UNTESTED here.

## The Test split is opened once

Train/Val/Test are split by group (lesson) so no group is in two parts. All choices (settings, checkpoints, early stopping) use Val only. The Test is scored ONCE, for the final chosen model and its baseline, by `final-test` (or `-WithTest`). A marker `test_split_scored.json` in the experiments folder makes a second scoring refuse with exit code 3. After you looked at Test, a new data set needs a new experiments folder.

## Documents

| Need | File |
|---|---|
| Start here, the first task | `docs/START_HERE.md` |
| Running it alone, step by step; what each number means; failures | `docs/WORK_RUNBOOK.md`, `docs/USER_GUIDE.md`, `docs/RUNBOOK.md` |
| A short guide for an operator who does not know the code | `docs/OPERATOR_GUIDE.md` |
| What a finished run gives you; how to read and present the report | `docs/WHAT_YOU_GET.md` |
| What each component does | `docs/COMPONENTS.md`, `docs/COMPONENT_GUIDE.md` |
| Every config key | `docs/CONFIG_REFERENCE.md` |
| Preparing a new dataset | `docs/DATA_ONBOARDING.md` |
| Something failed | `docs/TROUBLESHOOTING.md` |
| Decisions, open issues, requirements, build order | `docs/DECISIONS.md`, `docs/OPEN_ISSUES.md`, `docs/PRD.md`, `docs/BUILD_ORDER.md` |

Folders: `src\stt\` code, `tools\` scripts, `configs\` settings, `docs\` documents, `tests\` tests (`python -m unittest discover -s tests`, about 10 minutes, offline).
