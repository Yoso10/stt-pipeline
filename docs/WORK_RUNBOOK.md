# Work runbook: one training on the work machine, CPU only, alone

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
