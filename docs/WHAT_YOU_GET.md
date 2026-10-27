# What a finished training gives you

## How to run (this is a reading guide, nothing to run)

Written for: the owner, to present the result to colleagues who do not know the code. Read it next to a finished `report.html` (the path is printed on the last line of the run: `DONE. OPEN THE REPORT: ...`). Everything below comes from the report generator (`src/stt/reportbuilder/convergence.py`, `builder.py`), the run script and the stored files of earlier experiments on this laptop. No finished turbo `full` run existed on disk when this was written, so the report layout for that path is read from the code, not from a real page: **UNTESTED against a real turbo report**. The numbers in the examples are placeholders or come from the whisper-tiny rehearsal, and are marked.

## 1. In five lines

1. IN: labeled audio (clips of at most 30 s with their text), a Hebrew Whisper model (`ivrit-ai/whisper-large-v3-turbo`), one config file.
2. The system splits the clips by lesson into Train / Val / Test, scores the untouched model on Val (the baseline), trains the decoder, scores the trained model on the same Val clips.
3. OUT: ONE web page (`report.html`, no internet needed) that says whether the trained model is better, by how much, and whether the training converged.
4. OUT: the trained model folder, the numbers as small text/JSON files, and the log.
5. The Test clips are opened once, at the end, only when you ask.

## 2. Every file a finished run leaves

A `full` run makes two experiment folders under `run.experiments_dir`: the BASELINE one (`exp_A`) and the TRAINING one (`exp_B`; `full_run_state.json` names them). Sizes: small = text/JSON, can be shared (but see the note on data below); LARGE = never share by e-mail.

| Where | What it is | Size |
|---|---|---|
| `full_run_state.json` | which steps are done, the two experiment names | small |
| `exp_B\reports\report.html` | THE REPORT, one self-contained page | small |
| `exp_*\config_full.yaml`, `config_diff.yaml` | the exact settings of that experiment | small |
| `exp_*\run_info.json`, `status.json` | fingerprints, stage results, why it failed if it failed | small |
| `exp_*\logs\run.log` | everything that happened, with times | small |
| `exp_*\reports\load_report.txt`, `filter_report.txt`, `split_report.txt`, `subsample_report.txt` (subset runs), `text_report.txt` | what was read, removed, how the lessons were divided | small |
| `exp_*\reports\rejected.csv` | clips thrown out and why | small |
| `exp_A\reports\evaluation_baseline.json`, `eval_result_baseline.json`, `predictions_baseline.csv`, `review_baseline.txt` | the baseline WER/CER per clip and a list to listen to | small |
| `exp_B\reports\evaluation_trained.json`, `predictions_trained.csv`, `comparison.json` | the trained WER/CER, per clip, and the baseline-vs-trained statistics | small |
| `exp_B\metrics\trial_001\metrics.csv` | train loss and Val loss by step | small |
| `exp_B\reports\test_result.json`, and `<experiments_dir>\test_split_scored.json` | exist ONLY after the final Test; the second is the "already opened" marker | small |
| `exp_B\candidates\exp_B_trial_001\exp_B_trial_001_step_NNNNN\` | THE TRAINED MODEL (a Hugging Face folder + `pipeline_model_info.json`) | LARGE, about 3.2 GB |
| `exp_B\candidates\...\checkpoints\checkpoint-N\` | restart points (only the latest and the best are kept) | LARGE, about 4.6 GB each |
| `<features.encoder_cache.dir>\enc_*_float16\` | the cached encoder output, 3.84 MB per clip | LARGE, never delete during a run |
| `models\...` , the data folder | the base model and your audio | LARGE, not produced by the run |

Note: reports contain clip texts (Hebrew sentences from your data). "Small" means small in size; if the data are confidential, share numbers only.

## 3. The report, section by section

The page opens with a coloured verdict line, then the sections below in this order. Each row has a sentence you can say out loud. Replace the words in `[brackets]` with the numbers on your page.

| Section | What it shows (plain words) | What to say |
|---|---|---|
| **Verdict line** (top) | One sentence from the bootstrap: *Improved*, *No clear difference*, *Worse*, *Too few lessons in Val for a statement*, or *Not comparable* | "The system's own verdict is: [Improved]. That means the gain is larger than what chance across lessons would produce." |
| **Top panel: "Baseline WER x, trained WER y"** | WER = share of words wrong (0 = perfect; it can exceed 1 when the model inserts words). Relative reduction = (x - y) / x. | "On [101] unseen lessons' clips the word error rate went from [0.83] to [0.70], a relative reduction of [16%]." |
| **Target line** | Is the reduction at least the target (config `report.target_relative_wer_reduction`, 10%)? Printed "reached" or "not reached"; it adds "point estimate only" when the verdict is not *Improved* | "Our aim was 10% fewer errors; we got [16%]: [reached]." |
| **"Training:" sentence (convergence verdict)** | Computed from the Val-loss points, not guessed. *Still improving at the end* = more steps should help. *Plateau* = more steps of the same kind will not help much. *Overfitting* = Val loss rose while train loss fell; the saved model is the best earlier step. *Not enough points* = fewer than 2 Val points | "The Val loss fell from [2.0] to [1.2] and was still falling at the last point, so a longer training would probably help." |
| **Chart: Val loss (solid blue) and training loss (dashed orange)** | Loss = how surprised the model is by the correct text; lower is better. Val is data it never trains on. Green ring = the best checkpoint, the one that was saved. | "Both curves go down together, so it is learning and not memorising." Or: "The blue curve turns up after step [66] while the orange keeps falling: that is overfitting." |
| **Bar chart: WER on the Val clips, baseline vs trained** | Two bars, with a green band 0.10 to 0.30. The band is a FIXED display value, not a measurement | "The bar dropped from [0.83] to [0.70]; the green band is our wished range, we are [outside / inside] it." |
| **Table: baseline against trained** | WER, CER (character error rate), relative reduction, empty outputs / outputs that hit the length cap, bootstrap verdict, WER difference with its interval | "CER fell from [0.39] to [0.30] too, so it is not a trick of word boundaries." |
| **Interval of the difference** | Example format `+0.1039 (+0.0759 to +0.1330, 7 lessons, 90%)`. Positive = trained is better. If the whole interval is above 0, the verdict is *Improved* | "We resampled the lessons many times; in 9 of 10 resamples the gain stayed above [+0.076], so it is not luck." (see section 4 for exactly what this does and does not mean) |
| **"What to change next" box** | Plain advice from the verdict: more steps if still improving; fewer steps or a lower learning rate if overfitting; a higher learning rate if flat; check labels if the loss is good but WER is not | "The system itself suggests: [run one more epoch]." |
| **15 worst / 15 best Val clips** | For each: label, what the untouched model wrote, what the trained model wrote, WER before and after | "Here is one sentence the old model got wrong and the new one gets right." Read the worst ones too: they often reveal wrong labels. |
| **Final Test block** (only after `final-test`) | Same table, on the Test clips, marked CONFIRMED or NOT CONFIRMED | "Only after we decided, we opened the sealed Test once: WER [0.84] to [0.72]: [confirmed]." |
| **Run information** | fingerprints of data, split and settings; code version | "Anyone can check it was the same data and settings." |
| Sections that appear in other report types (search, finalists, learning curve, groups, hints, listening list) | Built for the `run` and `search` commands; may be absent in a `full` report (code, UNTESTED) | If a section says "not available in this run", it is normal. |

Words used in the report:

| Word | Meaning |
|---|---|
| Baseline | the model before our training, scored on the same clips |
| Val | clips held out of training, used for decisions (when to stop, which run is better) |
| Test | clips kept sealed; scored once at the very end |
| Lesson / group | a recording; all its clips go to one part only, so no lesson is in both Train and Test |
| Bootstrap | the statistics: redraw the lessons with replacement many times and recompute the difference; the spread of the answers is the interval |
| Epoch | one pass over all Train clips |

## 4. How to state the result honestly

| Say | Because |
|---|---|
| Both numbers, always: baseline WER and trained WER, and the same for CER | one number without the other means nothing |
| The relative reduction, e.g. "[16%] fewer word errors", together with the absolute WER | a drop from 0.83 to 0.70 is large in relative terms but the model is still wrong in 7 of 10 words |
| The bootstrap verdict, and the number of lessons in Val | the Val set is a few lessons ([7] in the 900-clip subset); few lessons means a wide interval |
| What "90%" means | the interval is a 90% confidence statement about the difference over lessons: if we repeated the experiment on other lessons of the same kind, we expect the gain to stay positive about 9 times in 10. It is NOT "90% of words are right" and NOT a guarantee for new audio of another kind |
| What the Test number is | the score on clips that played no role in any decision, scored once. It is the number to quote as the result. It is opened once: the program refuses a second scoring (exit code 3) |
| "Confirmed" / "not confirmed" on the Test | confirmed = the improvement also shows on the sealed clips; not confirmed = do not claim the improvement |
| A "point estimate only" note | the reduction reached the target but the bootstrap did not say *Improved*: report it as "not proven" |

## 5. What the result does NOT prove

| Limit | Meaning |
|---|---|
| Small data | the 900-clip subset is a tenth of the audio; the Val has about 100 clips from [7] lessons, so the WER moves by a few points if other lessons were chosen |
| One run | one seed, one learning rate, one number of epochs. A different seed could give a somewhat different number (the same seed gives the same result on the same machine) |
| Val was used for choices | when to stop and which run to keep were chosen on Val, so the Val gain is slightly optimistic; this is why the Test exists |
| Test was seen before in a rehearsal | the whisper-tiny rehearsal scored the same 97 Test clips once (see `docs\WORK_RUNBOOK.md`, 12b); no turbo model saw them, but the Test is not strictly untouched |
| Same kind of audio only | the result says nothing about other speakers, microphones or topics than your data |
| WER is not meaning | a lower WER does not mean every sentence is usable; read the clip tables |
| The report's time/size estimates | ESTIMATED, not measured, until a full turbo run has been done (UNTESTED) |

## 6. One-slide template (fill from the report)

| Blank | Where to find it |
|---|---|
| **Title:** "Fine-tuning Whisper for [domain/language] on [N] clips ([H] hours)" | `split_report.txt`, `subsample_report.txt` |
| **Model / setup:** `ivrit-ai/whisper-large-v3-turbo`, decoder trained, [2] epochs, CPU, [X] hours | `config_full.yaml`, `run.log` |
| **Data:** Train [___] / Val [___] / Test [___] clips; [___] / [___] / [___] lessons; split by lesson | `split_report.txt` |
| **Result (Val):** WER [___] -> [___] ([__]% fewer errors); CER [___] -> [___] | report, table "baseline against trained" |
| **Statistics:** verdict [Improved / No clear difference], interval [+___ to +___], 90%, [__] lessons | report, "Interval of the difference" |
| **Training health:** [still improving / plateau / overfitting] | report, "Training:" sentence |
| **Final Test (once):** WER [___] -> [___], [CONFIRMED / NOT CONFIRMED] (or "not yet opened") | report, Final Test block |
| **Example:** one sentence: label / before / after | report, worst-and-best tables |
| **Caveats:** [N] clips, one run, Val used for choices | section 5 above |
| **Next step:** [the "What to change next" line] | report box |

## Where the details live

`docs\WORK_RUNBOOK.md` (running it, failures), `docs\OPERATOR_GUIDE.md` (the short guide for an operator), `docs\COMPONENTS.md` (C14 ReportBuilder), `docs\CONFIG_REFERENCE.md` (the `report.*` keys).
