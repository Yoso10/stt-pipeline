# User guide: running the system alone on a closed machine

## How to run (the short version)

This guide tells you how to start the system, how to read what it prints, and what to change when something looks wrong. It is written for one person working alone, with no network. Everything in it was checked against the code on 2026-10-25.

Open PowerShell in the project folder and, once per window:

```
.\.venv\Scripts\Activate.ps1          # the project's Python, with its packages
$env:PYTHONPATH = "src"               # tells Python where the project code is
```

Then the three commands that matter most:

```
python -m stt check --config configs\work_like.yaml       # is this machine ready? (changes nothing)
python -m stt benchmark --config configs\work_like.yaml   # how long will a run take on THIS machine?
python -m stt run --config configs\work_like.yaml         # the whole pipeline, the rehearsal size
```

The program is called `python -m stt <command>`. There is no program called `stt`. Every command shows an example with `--help`, for example `python -m stt run --help`.

How to read this guide: sections 1 to 4 are what you need to start. Sections 5 to 8 explain the screens and the results. Section 9 says what to change when a symptom appears. Section 10 lists what was never tested. The component details are in `docs\COMPONENT_GUIDE.md`; this guide stays short on purpose. The older `docs\RUNBOOK.md` and `docs\TROUBLESHOOTING.md` are still valid and are named where they help (see "Differences from the older documents" at the end).

## 1. What the system does

It takes labeled Hebrew audio (clips of up to 30 seconds with their text), splits it into Train, Val and Test by lesson, fine-tunes a Whisper model on the Train part, measures the word error rate (WER) on the Val part, and keeps the best model only if it is clearly better than the model you started with. The Test part is looked at once, for the winner only. It works offline, on the CPU only, with no questions asked while it runs. "The baseline stays" is a normal result (exit code 0), not a failure.

## 2. Setting up a machine (once)

1. Put the project at a SHORT path such as `C:\stt` (40 characters or fewer). Windows fails to install torch in a deep folder.
2. Create the virtual environment and install from the offline wheel bundle: `python -m venv .venv`, activate it, `pip install --no-index --find-links wheels -r requirements.lock`.
3. Check: `python -m stt check`. It lists everything wrong at once.
4. Prove the chain works: `python -m stt smoke --full` (tiny model, synthetic data, about a minute). It does not prove quality.
5. Optional: verify the files against the manifest: `python tools\make_manifest.py --root . --verify MANIFEST.sha256`.

More detail is in `docs\RUNBOOK.md` sections 2 and 3.

## 3. Getting a model: the turbo model

Only the home machine may use the network. The pipeline itself never downloads anything.

```
python tools\fetch_model.py --repo ivrit-ai/whisper-large-v3-turbo --out models\whisper-large-v3-turbo-ivrit
```

This downloads only the safetensors weights and the small config and tokenizer files: 11 files, 3.24 GB (3,090 MiB on disk), one `model.safetensors`. It is NOT sharded. It writes `fetch_info.json` next to the files. Running it again with a finished folder reuses it.

To use it, point the config at the folder and keep float32 (the CPU setting):

```
--set model.path=models/whisper-large-v3-turbo-ivrit --set model.dtype=float32
```

On the closed machine, copy the folder `models\whisper-large-v3-turbo-ivrit` as a whole. The model has 808,878,080 parameters and 128 mel bins (whisper-tiny: 37,760,640 and 80), so it is about 21 times larger and needs far more time and memory (see section 8). It was used for scoring and for one benchmark; it was never used for a complete `run`.

## 4. The configs

| File | What it is |
|---|---|
| `configs\home_full.yaml` | whisper-tiny on ALL the Daf Yomi data. About 3.5 to 5 hours. |
| `configs\work_like.yaml` | the REHEARSAL: one tenth of the data (about 702 Train, 101 Val, 97 Test clips), 20 steps, its own folders `data\work_like\...`. About 35 minutes to 1 hour with whisper-tiny. It extends `home_full.yaml`. |
| `configs\work.yaml` | the TEMPLATE for your own data. Never edit it: copy it (`Copy-Item configs\work.yaml configs\my_work.yaml`), fill in every `<<< FILL IN ... >>>` marker in the copy. `check` names every marker left. |
| `configs\smoke.yaml` | the tiny synthetic test. |
| `configs\work_cpu.yaml` | the settings for the WORK machine, CPU only: turbo, frozen encoder, `train.cpu_threads` as a placeholder, `filter.mode: apply`, one Val-loss evaluation (`train.eval_every_steps` = `max_steps`), `select.test_confirmation: false`, encoder cache on. Copy it to a new name and fill the `<<< ... >>>` places. The step-by-step path (probe, check, prepare, benchmark, baseline, train, evaluate, compare), the time calculator and the failure guide are in `docs\WORK_RUNBOOK.md`. UNTESTED on the work machine. |

Change one value for one run on the command line, for example `--set train.hyperparams.max_steps=200`. The last `--set` wins. Lists and objects must be written as JSON text. The key names are in `docs\CONFIG_REFERENCE.md` (188 keys); never invent one, an unknown key fails with the closest valid name.

Rule: do not edit a config in the middle of a run, and never edit an experiment's `config_full.yaml`.

**The compute device (`runtime.device`, DECISIONS 87).** `cpu` is the default and is what every number in this guide was measured with. `--set runtime.device=cuda` (or the environment variable `STT__RUNTIME__DEVICE=cuda`) asks for the GPU; `auto` takes the GPU when torch finds one and otherwise the CPU (and says which at INFO). With `cuda` and no usable GPU the command STOPS at once with exit code 2 and a message that names the key, what torch found (version, CUDA build, devices) and the fix; it never continues on the CPU by itself. The installed torch must be a CUDA build for this (the Windows torch from PyPI is CPU-only). UNTESTED on a real GPU: this laptop has none usable. See the open GPU points in section 12.

**The encoder cache (`features.encoder_cache.*`, DECISIONS 89).** Off by default. With the encoder frozen (`train.freeze_policy: freeze_encoder`) it computes the output of the encoder once per clip and reuses it for training, the Val loss and the decoding. Switch it ON: `--set features.encoder_cache.enabled=true --set features.encoder_cache.dir=<folder> --set train.freeze_policy=freeze_encoder` (the dtype is `float16`, half the disk; `float32` is exact). The first command that needs a clip builds its file (about 15 s per clip with turbo on the home laptop, measured), later commands reuse them. Switch it OFF and run exactly as before: remove the keys or `--set features.encoder_cache.enabled=false` in a NEW experiment (the settings changed, so an old experiment cannot be continued); the folder stays on disk. It stops with exit code 2 and names the key if the encoder is not frozen, if the model has dropout or layerdrop, if an augmentation is on, or if the disk is too small. Measured with turbo on 8 clips: a training step 58 to 63 s becomes 7.5 s, a Val-loss pass 111 to 125 s becomes about 5 s, decoding 128 to 139 s becomes 24 s, with identical text; peak RAM 6.3 GiB (not higher than without). Never run: a whole run with it, or on the work machine. The work-machine path is in `docs/WORK_RUNBOOK.md`.

**Work-machine tools (new, read-only).** `tools\probe_env.py` prints the Python and torch versions, physical cores, RAM total and free, free disk and whether the model folder is complete (always exit code 0). `tools\compare_evals.py --dir <experiments folder> --experiment exp_NNNN [--baseline-from exp_MMMM]` prints the baseline against the trained model: WER, CER, the relative reduction and the bootstrap verdict, from the files that `evaluate` stored. `toolsun_work.ps1 -Step probe|check|prepare|benchmark|baseline|train|evaluate|compare` calls only the existing commands, one step at a time, and stops at the first error. All three are UNTESTED on the work machine.

## 5. The commands

All commands accept `--config FILE` and repeatable `--set a.b=value`. Those that create an experiment also accept `--parent exp_NNNN` (the new experiment extends that one) and `--experiment exp_NNNN` (continue that existing experiment: finished stages are skipped, an interrupted training resumes).

| Command | What it does | Example | Changes on disk |
|---|---|---|---|
| `check` | Python, packages, model and data folders, disk, RAM, cores, code version. Lists every problem at once. | `python -m stt check --config configs\work_like.yaml` | nothing |
| `prepare` | load, split, normalize, filter; the data reports. | `python -m stt prepare --config configs\my_work.yaml` | a new experiment with reports |
| `evaluate` | prepare, then score the untrained model on Val. The Test split is refused. | `python -m stt evaluate --config configs\work_like.yaml --set model.path=models/whisper-large-v3-turbo-ivrit --set model.dtype=float32` | a new experiment |
| `train` | prepare, then one training run. | `python -m stt train --config configs\home_full.yaml` | a new experiment |
| `search` | prepare, then the hyperparameter search on a small part of Train. | `python -m stt search --config configs\home_full.yaml` | a new experiment |
| `run` | baseline, search, finalists on the full Train data, selection with the one-time Test look, registration, automatic export. | `python -m stt run --config configs\work_like.yaml` | a new experiment; the registry |
| `benchmark` | measures seconds per training step, per Val-loss example, per decoded clip, and prints the planned time of `run`. | `python -m stt benchmark --config configs\work_like.yaml` | a new experiment that KEEPS a real model folder (`candidates\exp_NNNN_benchmark`) |
| `learning-curve` | the same settings on growing Train sizes; Val WER per size; a plain reading ("still improving" / "flat" / "got worse"). Set `subsample.val_samples` for real use. | `python -m stt learning-curve --config configs\home_full.yaml --params-from exp_0003` | a new experiment with several model folders |
| `report` | rebuilds `reports\report.html` of an existing experiment (the latest if `--experiment` is missing). Recomputes nothing. | `python -m stt report --config configs\work_like.yaml --experiment exp_0003` | that one file |
| `registry list` / `verify [ENTRY]` / `rebuild-index` | read the registry; `verify` re-checks the files against their checksums; `rebuild-index` is the only one that writes. | `python -m stt registry verify model_0001 --config configs\work_like.yaml` | `rebuild-index` only |
| `export [ENTRY]` | export a registered model (default: the latest entry); `--format NAME` is repeatable. | `python -m stt export model_0001 --config configs\work_like.yaml` | that entry's `exports.json` |
| `cleanup` | WRITES A LIST of what could be deleted. It deletes nothing and has no option that deletes. | `python -m stt cleanup --config configs\work_like.yaml` | a text file `cleanup_proposal_<time>.txt` next to the experiments folder |
| `config-docs` | rewrites `docs\CONFIG_REFERENCE.md` from the code. | `python -m stt config-docs` | that one file |
| `smoke`, `smoke --full` | the whole chain on synthetic data and the tiny test model. | `python -m stt smoke --full` | a smoke experiment in `experiments\` |

Note: `run` does NOT build the report page. After a run, build it with `python -m stt report --experiment exp_NNNN`.

Stopping: press Ctrl+C ONCE in the plain command window. The training saves a checkpoint and stops after the current step (exit code 130). Run the same command with `--experiment exp_NNNN` to continue. A second Ctrl+C stops at once. Do not use a PowerShell pipe such as `| Tee-Object`: Ctrl+C inside a pipe kills the program without saving (seen once, 2026-10-08). Do not close the window to stop a run.

## 6. A first real run, in this order

1. `python -m stt check --config configs\work_like.yaml`. Fix every line that says `PROBLEM`.
2. Close other heavy programs (browser, editor, sync tools). The machine's speed varies a lot (open issue O120).
3. `python -m stt benchmark --config configs\work_like.yaml`, twice. Trust the LARGER planned time. The seconds per step it prints can be up to twice too high (a known timing bug, O121), so the plan is on the safe side. The benchmark also prints the size of one trial folder, the planned disk of a whole run and the peak memory.
4. If the plan fits your time and disk, `python -m stt run --config configs\work_like.yaml`.
5. Open the report (section 7), listen to the listening list, and only then decide about `home_full.yaml`.

Warmup rule: keep `train.hyperparams.warmup_steps` at 3 or less in a config you benchmark; the benchmark trains only 3 steps and fails with a longer warmup (O119).

## 7. Reading the screen and the report

**On the screen.** A plain line at most every 30 seconds (`progress.interval_seconds`):
- `PLAN: planned time of python -m stt run ... about 37 min` with a line per part, printed after `prepare` when a benchmark exists for this model and thread count (otherwise `PLAN: no benchmark numbers ...`).
- `PROGRESS WER scoring of the baseline on 'val': clip 48 of 101 (47.5%), elapsed 0:01:51, remaining about 0:02:02`.
- `[exp_0002_trial_001] step 10 of 20 (50.0%) ... train_loss 3.0294 val_loss - ...`. The `s/step` and `remaining` of TRAINING lines are too high (O121); the elapsed time and percent are right.
- `Search progress: trial 2 of 4 finished ...`, `PROGRESS finalists: starting finalist 2 of 2 ...`.
- At the end: `VERDICT: ...`, `Test: ...`, the improvement line, and one JSON summary.

**The files.** Everything is in `<run.experiments_dir>\exp_NNNN\`: `logs\run.log` (always written), `status.json` (where the run is), `reports\` (all reports), `metrics\<trial>\metrics.csv` (open in Excel), `config_full.yaml` (what ran). To follow the log: `Get-Content <path>\logs\run.log -Wait -Tail 20`. A live page `metrics\<trial>\progress.html` appears if you put `html` in `train.sinks` in your YAML file (not on the command line: PowerShell 5.1 mangles the JSON).

**The report** (`reports\report.html`, open in a browser). Top to bottom:
- The verdict line: `Baseline retained` (no finalist beat the untrained model), `Proven improvement` (clearly better on Val AND the one-time Test confirms it), `No proven improvement` (the best finalist looks better but is not proven).
- The target line: `Relative reduction of the Val WER against the baseline: 9.2% (target: 10%): not reached`. The target is `report.target_relative_wer_reduction` (0.10 in the presets). `Point estimate only` means the number reaches the target but is not proven.
- Then: history of registered experiments, the finalists against the baseline, WER bars with the interval of the difference, training curves, the per-file WER histogram, WER by lesson, the search, the learning curve, error-versus-attribute correlations, diagnosis hints, the listening list, run information. A section with no data says "Not available in this run" and names the missing file.
- WER above 1.0 is possible and is not an error: it means more word errors than reference words (the model adds words).
- Hints are hints, not proof. Correlation is not cause.
- Listen to the clips in the listening list. Your ears are the second test: a model can pass the statistics and still fail in practice.
- Do NOT use the Test result to choose between experiments. The Test is looked at once per model; the report says how many times this Test set has been looked at.

**The registry.** The winner is in `registry\model_NNNN\` (or `data\work_like\registry` for the rehearsal) with `model\` (the model), `MODEL_CARD.md` (plain English), `model_card.json`, `review_list.txt`, `SHA256SUMS`, `exports.json`. The winner's folder was MOVED there out of `candidates\`.

## 8. Time, memory and disk (measured on the home laptop; estimates are marked)

The home laptop has 15.77 GB RAM, 8 cores and a single drive.

| Quantity | Value | How known |
|---|---|---|
| whisper-tiny, training step, 2 threads, batch 2 x accumulation 2 | about 5.4 s | measured (rehearsal) |
| whisper-tiny, scoring a 30 s clip | about 2.0 s | measured |
| whisper-tiny, one trial folder | about 600 MB (checkpoint about 456 MB) | measured |
| whisper-tiny, whole rehearsal | 38 min, 4.3 GB | measured |
| turbo, scoring a 30 s clip, 4 threads, batch 2 | about 16.6 s (1,681 s for 101 clips) | measured |
| turbo, peak memory while only scoring | 3.5 GiB working set | measured |
| turbo, prepare stage | 22 s to 73 s | measured |
| turbo, training step (frozen encoder, 4 threads) | 106 s as printed (about 84 s corrected), batch 2 x 2; 55 s as printed, batch 1 x 2 | measured (2026-10-25) |
| turbo, peak memory while training | 6.4 to 6.7 GB | measured |
| turbo, one trial or finalist folder | 7.4 to 7.9 GiB | measured |

Rules of thumb that follow: scoring dominates the time of a full `run` with the turbo model (baseline 28 min, two finalists about 56 min, Test look about 54 min: about 2 h 18 min before any training). A trial or finalist folder of the turbo model is 7.4 to 7.9 GiB (measured), against 600 MB for whisper-tiny. A full `run` with the rehearsal settings is planned at 10 h 24 min to 14 h 49 min on this laptop.

Disk: set `train.save_optimizer_state: false` to keep only the weights in a checkpoint (about one third of the space for a full fine-tune of whisper-tiny; with a frozen encoder the saving is small, because only the decoder has optimizer state). A run made this way can NEVER be resumed: `train.resume: true` is refused. Keep the default (true) for any run that may be interrupted.

#### Turbo training measurements (Task 2, 2026-10-25, `freeze_encoder`, 4 threads, float32, this laptop)

Two benchmarks of 3 training steps on 12 Train clips, plus a probe of 12 Val clips (`exp_0003` batch 2, `exp_0004` batch 1; each kept a 7.4 GiB model folder under `data\work_like\experiments\exp_NNNN\candidates\`):

| Quantity | batch 2 x accumulation 2 (4 clips per step) | batch 1 x accumulation 2 (2 clips per step) |
|---|---|---|
| Trainable parameters | 171,909,120 of 808,878,080 | the same |
| Seconds per training step, as printed by the benchmark | 106.0 | 55.4 |
| Seconds per training step, corrected for the O121 timing bug (estimated: elapsed time of the metrics file minus the Val probe) | about 84 | about 46 |
| Seconds per clip of a step (estimated) | about 21 to 26 | about 23 to 28 |
| Seconds per Val-loss example | 24.4 | 17.4 |
| Seconds per decoded clip (12-clip probe) | 22.1 | 19.6 |
| Seconds per decoded clip (101-clip baseline, `exp_0002`) | 16.6 | |
| Peak memory (working set of the process) | 6.4 GB | 6.7 GB |
| Lowest free RAM seen (of 15.77 GB installed) | 0.03 GB | above 0.2 GB in every sample |
| Model folder of the benchmark | 7.4 GiB on disk (benchmark text: 7.9 GB; one 4.6 GB checkpoint plus the 3.1 GB returned model) | the same |
| Saving the model (the "Writing model shards" step) | about 26 s | similar |

Reading: halving the batch halves the time per step; the time per training CLIP does not improve (batch 1 gives no speed-up). The step is dominated by the forward pass of the frozen encoder through the 30 s window, so freezing the encoder saves memory and optimizer disk but little time. The batch-2 run used the last free RAM of the laptop (the free RAM fell to 0.03 GB for a few seconds); batch 1 needs about the same memory (6.7 GB peak). Both ran to the end; neither was killed by the watchdog.

The planned time that `benchmark` printed for the current settings (4 trials and 2 finalists of 20 steps, Val loss on 101 clips): batch 2: 14 h 49 min; batch 1: 10 h 24 min. Planned disk: 47.1 GB for 6 trial folders, up to 74.8 GB.

#### What a full `run` costs on this laptop (turbo, `freeze_encoder`; every number is marked M = measured, E = estimated)

Common numbers: step 106 s (M, as printed by the benchmark, the pessimistic reading; the corrected 84 s would allow about 25% more steps), Val loss 24.4 s per example on 101 clips (M), decoding 16.6 s per clip (M, 101 clips), 2 Val-loss evaluations per training (E, `eval_every_steps` = half of `max_steps`), 3 minutes per training for model loading and saving (E), `prepare` 1.5 minutes (M). One training of S steps therefore costs 106 S + 4,929 + 180 seconds. Baseline on Val: 28 min (M). Winner's Val WER: 28 min (E, same speed). Test look, winner and baseline: 2 x 97 clips = 54 min (E).

The system cannot run "without a search": the stage `finalists` reads the result of the `search` stage (`finalists.py`, exit code 2 if `search_trials.json` is missing). The smallest `run` is `search.n_trials=1` and `search.finalists.k=1`, which is TWO trainings (the trial on the 61-clip screening set, then the finalist on the full Train split, both with the same `max_steps`). The `train` command followed by `evaluate` is ONE training, but it has no search, no selection, no Test look and no registry.

| Plan | Stage times (E unless marked M) | Largest `max_steps` in 8 h | Largest `max_steps` in 4 h |
|---|---|---|---|
| (a) `run` with `n_trials=2`, `finalists.k=1` (three trainings) | prepare 1.5 min; baseline 28 min (M); search 2 x (S x 106 s + 85 min); finalist S x 106 s + 85 min; finalist Val WER 28 min; Test 54 min. Total = 21,991 s + 318 s x S | **21** steps (S=21: baseline 28 min, search 4 h 04 min, finalist 2 h 02 min, Val WER 28 min, Test 54 min; total 7 h 58 min) | **none**: the fixed part alone (baseline, six Val-loss evaluations of 41 min each, Val WER, Test) is 6 h 06 min; it does not fit even at 0 steps |
| (b) the smallest `run`: `n_trials=1`, `finalists.k=1` (two trainings) | same, with one trial. Total = 16,882 s + 212 s x S | **56** steps (search 3 h 04 min, finalist 3 h 04 min, plus 28 + 28 + 54 min; 7 h 58 min) | **none**: 4 h 41 min at 0 steps |
| (c) one training only: `train` then `evaluate` (no search, no selection, no Test) | prepare 1.5 min; the training S x 106 s + 85 min; `evaluate` scores the baseline and the trained model: 2 x 28 min. Total = 8,553 s + 106 s x S | **191** steps (training 7 h 02 min + 56 min) | **55** steps (training 3 h 02 min + 56 min) |

Notes on the table: the 8 h and 4 h columns use the 106 s step; with the corrected 84 s step the limits are about 27 (a), 70 (b), 240 (c, 8 h) and 69 (c, 4 h) steps. If you use a Val slice of 30 clips for the Val loss (`subsample.val_samples=30`, but not together with the `work_like` preset) each Val-loss evaluation shrinks from 41 min to 12 min, which would let plan (a) fit about 54 steps in 8 h (E). Early stopping and pruning only shorten a run. Disk: every training leaves a 7.4 to 7.9 GiB folder, so (a) needs about 24 GB, (b) about 16 GB and (c) about 8 GB; the laptop has 7.7 GB free after the two benchmark folders (15 GB before the second one), so (a) and (b) do not fit without deleting something. Weights-only checkpoints (`train.save_optimizer_state: false`) save little with a frozen encoder: the optimizer state of the decoder is only about 1.5 GB of the 4.6 GB checkpoint (E). Memory: a training peaks at 6.4 to 6.7 GB; the laptop has 15.77 GB but only 2 to 4 GB are usually free, so other programs must be closed. The numbers say nothing about quality: 20 to 190 steps on this data are a rehearsal.

#### The same budget with fewer Val-loss evaluations (Task 4, estimated from the measured numbers above)

How the Val loss is scheduled (checked in `hf_trainer.py`, `planning.py`, `wiring.py` and the installed Hugging Face `trainer_callback.py`): the Val loss is computed on the Val slice of the screening set, in the search AND in the finalists stage (decision O93): all clips of the prepared Val split (101 in the rehearsal), or exactly `subsample.val_samples` clips when that key is set (not allowed together with a scope `all_splits` preset such as `work_like`). It is computed every `train.eval_every_steps` steps, plus once at the end when `max_steps` is not a multiple of it, so a training makes `ceil(max_steps / eval_every_steps)` evaluations. With `eval_every_steps` equal to `max_steps` there is exactly ONE evaluation, at the last step (Hugging Face does not add a second one, because the last step is already a multiple). The first evaluation always counts as an improvement, so it becomes the best checkpoint and the returned model; early stopping needs `patience` (5) evaluations without improvement, and pruning needs at least 3 evaluations, so neither can happen. The WER of the finalists and the baseline is separate and always uses the whole prepared Val split.

Assumptions as before: step 106 s (measured, as printed), Val loss 24.4 s per example (measured), decoding 16.6 s per clip (measured), 3 minutes per training for loading and saving (estimated), fixed part 6,664 s (baseline 28 min, finalist Val WER 28 min, Test look 54 min, prepare 1.5 min). Largest `max_steps` that fits:

| Plan | Two evaluations of 101 clips (the table above) | (i) ONE evaluation at the end, 101 clips | (ii) a Val slice of 30 clips, two evaluations | (i) + (ii): one evaluation of 30 clips |
|---|---|---|---|---|
| (a) `n_trials=2`, `finalists.k=1` (3 trainings): 8 h / 4 h | 21 / none | 44 / none (4 h 03 min at 0 steps) | 54 / 8 | 61 / 15 |
| (b) `n_trials=1`, `finalists.k=1` (2 trainings): 8 h / 4 h | 56 / none | 79 / 11 | 88 / 20 | 95 / 27 |
| (c) `train` + `evaluate` (1 training, no Test): 8 h / 4 h | 191 / 55 | 214 / 78 | 223 / 87 | 230 / 94 |

Formulas (seconds): a training of S steps costs `106 S + evaluations x clips x 24.4 + 180`; plan (a) adds the fixed part and three trainings, plan (b) two, plan (c) uses `90 + training + 3,354`. With the corrected step of about 84 s (estimated) every limit rises by about a quarter. Caution on (ii): with the `work_like` preset a 30-clip Val slice cannot be set (`subsample.val_samples` needs the scope `train_only`); it works only with a preset or settings like `home_full`, where the WER Val is then the whole Val split (875 clips, 4 h scoring at 16.6 s per clip, estimated), so (ii) is not a free saving. Only (i) needs no new setting: `--set train.eval_every_steps=<max_steps>`.


## 9. What to change when a symptom appears

First read, in this order: the last lines of the screen, `reports\search_report.txt`, `metrics\<trial>\metrics.csv`, the report page. Change ONE thing at a time and write down the experiment number. Changes marked "never tried" are reasonable guidance, not measured results.

| Symptom | What it probably means | What to change |
|---|---|---|
| The training loss does not fall (flat from step 1) | learning rate too low, or too few steps. Note: 20 steps (the rehearsal) show almost nothing | raise `train.hyperparams.max_steps`; try a higher `train.hyperparams.learning_rate` (the search covers 5e-6 to 5e-5 in the presets); check that `warmup_steps` is not close to `max_steps` |
| The loss jumps up, `grad_norm` is huge, or the run stops with "loss became NaN or infinite" (exit 4) | learning rate too high | lower `train.hyperparams.learning_rate` (the error names the last good checkpoint); read the first lines of `metrics.csv` |
| Train loss falls but Val loss rises over the last evaluations (hint `overfitting`) | the model is memorizing the small Train set | fewer steps (`max_steps`), more data, smaller learning rate. The best checkpoint by Val loss is already the one returned. `train.freeze_policy: freeze_encoder` trains fewer parameters (never tried for its effect on WER) |
| The Val loss is the same at every evaluation (hint `no_learning`) | too few steps, or too low a rate | as the first row |
| Many outputs "hit the length cap" (log warning; hint `length_cap_or_empty`) | the model loops, repeating text until the cap | this is mostly a property of the model (whisper-tiny: 17 of 101 clips; turbo untrained: 1 of 101). Training on clean labels reduces it. `eval.generation.max_tokens_per_audio_sec` (8.0) already cuts a loop at about 240 tokens for 30 s; lowering it cuts loops sooner but can cut real long speech (never tried) |
| Empty outputs, or the error "Every hypothesis is empty" (exit 4) | wrong language setting or wrong features | check `model.language` is `he`, that `model.path` is the model you meant, and the feature report |
| WER above 1.0 with many extra words ("insertions") | the model adds text. Some of it may be real speech that is missing from the label | read the review list; see "Labels shorter than the audio" in `docs\COMPONENT_GUIDE.md` (chapter 20). No setting fixes this today |
| Out of memory during training (exit 4) | batch too large for the RAM | `train.hyperparams.batch_size` 1 and a larger `grad_accum_steps`; `train.freeze_policy: freeze_encoder`; close other programs; a smaller model |
| Out of memory while scoring (exit 4) | `eval.batch_size` too large | `--set eval.batch_size=1` |
| The run is too slow | CPU time | fewer `search.n_trials`, smaller `search.finalists.k`, fewer `max_steps`, a smaller `subsample` size; more `train.cpu_threads` if cores are free; decoding is the biggest cost with a large model |
| Disk full | trial and checkpoint folders | `python -m stt cleanup` writes a list; delete the folders YOU agree with (see section 10); `train.save_optimizer_state: false` for new runs |
| A trial "failed ... Access is denied" at the end of its training | Windows or an antivirus held the new model folder | the system now retries (`train.rename_retries` 5, `train.rename_retry_seconds` 2.0); if all tries fail, NOTHING is deleted; the message names the `.tmp` folder with the finished model; close the program that holds it, rename the folder by hand, or rerun with `train.resume: true`. Excluding the project folder from antivirus and sync tools prevents it |
| `run.lock` is stale | a run died without releasing it | check `Get-Process python`; RENAME the file (`Rename-Item ...\run.lock run.lock.stale`), do not delete it; rerun with `--experiment` |
| "Cannot resume ... the config differs" | you changed a setting | use the same settings, or start a new experiment with `--parent exp_NNNN` |
| The baseline stays | no finalist was clearly better than the untrained model | normal. Try more data or steps, read the finalist table in the report, compare with the learning curve |

## 10. What is deleted, and what is not

- The system never deletes anything by itself. `cleanup` only writes a proposal. No command has a deleting option.
- The registry settings `registry.cleanup: true` and `registry.transfer: move` are the defaults. In practice: the winner's model folder is MOVED into the registry (it leaves `candidates\`). The Orchestrator names no other folder, so nothing else is removed (decision O91).
- ONE-TIME EXCEPTION (decision 84, owner, 2026-10-24): when free disk on C: was 13.65 GB, the agent, with your permission, deleted by hand-written commands (not by the program) the model and checkpoint folders under `candidates\` of COMPLETED experiments in `data\rehearsal_run`, `data\part11_real` and `data\part4_real`: 15 folders, 7,978.8 MB. The rules: only a folder directly under `<experiment>\candidates\`, only if that experiment's `status.json` says `completed` and it has no `run.lock`, nothing the registry points into; the list was shown before anything was deleted; each experiments folder has a file `deleted_candidates_20261024.txt` with the names and sizes. Consequences: the rehearsal's trial and finalist models can no longer be re-scored or exported (the winner is in the registry), and those checkpoints can no longer be resumed. The reports, metrics, configs and the registry were kept. The exception does not repeat: any further deletion needs your new permission.
- Sizes: whisper-tiny trial folder about 600 MB; turbo, 7.4 to 7.9 GiB (measured). The two benchmark folders `exp_0003` and `exp_0004` (15 GiB together) are kept on purpose; free disk was 7.7 GB afterwards.

## 11. Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | success (also "the baseline stays") | read the report |
| 1 | an unexpected bug | keep `logs\run.log` and the message; ask for help |
| 2 | wrong settings: config, model folder, locked or stale experiment, a config that differs on resume, the refused Test split, a registry inside the experiments folder | read the message; it names the key |
| 3 | data problem (loading, split, filter, features, subsample) | read `rejected.csv` and the data reports |
| 4 | training, evaluation, search, selection or registry failure (also a second look at the Test) | read `logs\run.log` and `status.json` |
| 5 | a registered model does not match its checksums, or an export or its check failed | `python -m stt registry verify`; see the exporter table in `docs\TROUBLESHOOTING.md` |
| 130 | stopped with Ctrl+C | run the same command with `--experiment exp_NNNN` |

## 12. What is untested (do not rely on these yet)

Never run for real:
- A complete `run` (search, finalists, Test look) with the turbo model, or with any model but whisper-tiny.
- A `run` on ALL the data (`home_full.yaml`): prepared, never started.
- `learning-curve` on real Daf Yomi data (synthetic data only).
- Sharded safetensors models (`model.safetensors.index.json`): accepted by the code, never loaded.
- GPU use: `runtime.device` exists (DECISIONS 87) and its selection logic is tested with a faked GPU, but nothing was ever run on a real GPU. Mixed precision (`float16`, `bfloat16`) is not built.
- `optuna_tpe` on real data; `parallel_trials` above 1 with a real model; pruning firing in a real run.
- Early stopping and `train.max_runtime_minutes` on real data.
- The retry of a real Windows rename failure (`train.rename_retries`): tested with simulated failures only.
- Resume of a real long run (a short search resume was done once by hand), and resume after an interruption of a run made with `train.save_optimizer_state: false` (the stage ends with a clear refusal).
- Any export format other than `hf_safetensors`.
- The SampleFilter in `apply` mode on real data (the effect of removing clips on the WER was never measured).
- `mypy --strict` (not installed here).
- The system on the closed machine itself: its Python version, RAM and disk are unknown (questions for the manager: Python version, cores and RAM, data volume and labeling, sleep policy, the target software of the final model).

## 13. Folders at a glance

| Path | What it is |
|---|---|
| `configs\` | the presets (`home_full`, `work_like`, `work`, `smoke`) and example sections |
| `models\` | model folders (`whisper-tiny`, `whisper-large-v3-turbo-ivrit`) |
| `data\daf_yomi_raw`, `data\extracted\daf_yomi` | the downloaded parquet files and the extracted audio and manifest |
| `<run.experiments_dir>\exp_NNNN\` | one experiment (section 7) |
| `registry\` or `data\work_like\registry` | the registered models |
| `docs\` | this guide, `COMPONENT_GUIDE.md`, `RUNBOOK.md`, `TROUBLESHOOTING.md`, `CONFIG_REFERENCE.md`, `DECISIONS.md`, `OPEN_ISSUES.md` |
| `tools\` | the home-side scripts that use the network |

## Differences from the older documents

Where this guide and the older `docs\RUNBOOK.md` or `docs\TROUBLESHOOTING.md` differ, this guide follows the current code. The known differences are listed under "Mismatches found" in `docs\COMPONENT_GUIDE.md`.
