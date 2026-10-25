# Work runbook: one training on the work machine, CPU only, alone

## How to run (the short version)

This is the exact order of commands for the work machine: a CPU-only computer with no internet and nobody to ask. It trains the turbo model once (the decoder only, the encoder frozen), scores it on the Val clips and compares it with the untrained model. Copy the commands as written (PowerShell, in the project folder). Everything that was NOT run on the work machine is marked **UNTESTED**. Times marked "measured" were measured on the home laptop (8 cores, 15.77 GB RAM, CPU), on 2026-10-25; your machine will be faster or slower, so the time calculator in step 4 uses YOUR benchmark.

```
.\tools\run_work.ps1 -Step probe
.\tools\run_work.ps1 -Step check     -Config configs\my_work_cpu.yaml
.\tools\run_work.ps1 -Step prepare   -Config configs\my_work_cpu.yaml
.\tools\run_work.ps1 -Step benchmark -Config configs\my_work_cpu.yaml -Threads 8
.\tools\run_work.ps1 -Step baseline  -Config configs\my_work_cpu.yaml -Threads 8
.\tools\run_work.ps1 -Step train     -Config configs\my_work_cpu.yaml -Threads 8
.\tools\run_work.ps1 -Step evaluate  -Config configs\my_work_cpu.yaml -Threads 8 -Experiment exp_0003
.\tools\run_work.ps1 -Step compare   -Experiment exp_0003 -BaselineFrom exp_0002
```

`-Threads` is the number of PHYSICAL cores (step 1 prints it). `run_work.ps1` only calls the commands written in each step below, stops at the first error and says what to look at. If the script does not start (a PowerShell policy), run the commands of each step by hand; they are written out in full.

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

`.\tools\run_work.ps1 -Step train -Config configs\my_work_cpu.yaml -Threads 8`. This is the long step: `k x P x S` seconds plus the one-time encoder pass of the Train clips (shown as `Encoder cache (train): ... clips done`). It writes a loss file you can open in Excel at any time: `data\work_cpu\experiments\exp_NNNN\metrics\trial_000\metrics.csv`. A progress line appears at most every 30 seconds. The training line `s/step` and `remaining` are too high (known timing error); trust the elapsed time and the percent.

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

**Copy out for analysis, small files only** (never the audio, the data, the models, the encoder cache or anything under `candidates\`): from `data\work_cpu\experiments\exp_NNNN\` of the baseline experiment and the train experiment copy `config_diff.yaml`, `status.json`, `run_info.json`, `logs\run.log`, and from `reports\`: `load_report.txt`, `split_report.txt`, `filter_report.txt`, `text_report.txt`, `benchmark.txt` (benchmark experiment), `evaluation_baseline.json`, `evaluation_trained.json`, `comparison.json`, `predictions_baseline.csv`, `predictions_trained.csv`; and `metrics\trial_000\metrics.csv`. Together they are a few megabytes. Check with `Get-ChildItem -Recurse <folder> | Measure-Object Length -Sum` that you copy less than 50 MB. The predictions files contain the transcripts: copy them out only if the rules of the work site allow it.

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
