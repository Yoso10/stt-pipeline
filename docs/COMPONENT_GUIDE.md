# Component guide: what every part of the system does

## How to read this file (How to run)

This file explains the system in plain English, one chapter per component, in the order the data flows through them. It is a reading guide, not a program. Open it in any text editor. There is nothing to run.

- Chapters 1 to 18: one component each (the same template every time).
- Chapter 19: the story of one whole run, from raw data to the final report.
- Chapter 20: an analysis question (labels that end before the audio ends).
- Chapter 21: what each command leaves behind and what each experiment type tells you.
- At the end: "Mismatches found" and "Open questions".

How much of each chapter was checked against the actual source code is stated honestly in the table "How each chapter was verified" below. Some chapters rest on the older documents (`docs/PRD.md`, `docs/COMPONENTS.md`, `docs/DECISIONS.md`, `docs/OPEN_ISSUES.md`) plus real output files, not on a full reading of the code. The test files were NOT read: only their number of tests was counted and compared with the documents. Where code and documents disagree, this guide says what the code does, and the disagreement is listed under "Mismatches found".

Run commands in PowerShell, from the project folder, after `$env:PYTHONPATH = "src"`. The program is started with `python -m stt <command>` (there is no installed `stt` program).

Numbers in the examples come from real runs on the home laptop: the rehearsal (`configs/work_like.yaml`, whisper-tiny, 20 steps, finished 2026-10-08), the turbo probe (8 Val clips, `exp_0001` under `data\work_like\experiments`), and the turbo baseline on all 101 rehearsal Val clips (`exp_0002`, 2026-10-25). Each example says which one it uses.

## How each chapter was verified

Legend: **Full** = the main source files of the component were read and the chapter matches them. **Partly** = some source files were read (named), the rest comes from the documents and from real output files. **Docs only** = the code was not read for this chapter; the text comes from `COMPONENTS.md`, `PRD.md`, `DECISIONS.md`, `OPEN_ISSUES.md` and from real output files of the runs (so the statements are as reliable as those, no more). Tests were not read in any chapter.

| # | Chapter | Verified against the source | What was read in the code |
|---|---|---|---|
| 1 | Config | Partly | `config/loader.py` (`load_config`), `config/schema.py` (`RuntimeConfig`), function names in `sources.py`; `validators.py`, `errors.py` and most of `sources.py` not read |
| 2 | DataSource | Partly | `config.py`, `csv_source.py` and `factory.py` in full; `parquet_source.py` head and error messages only; `report.py`, `audio_header.py` not read |
| 3 | Splitter | Partly | `config.py` (keys), `grouped_ratio.py`, part of `checks.py`; `hash_group.py`, `predefined.py`, `time_block.py`, `store.py` not read |
| 4 | TextNormalizer | Partly | `types.py`, `timestamps.py`, `normalizer.py` (outline); `steps.py`, `config.py`, `registry.py` not read; the step list is from the real `text_report.txt` |
| 5 | SampleFilter | Docs only | no source read; the numbers are from the real `filter_report.txt` |
| 6 | Subsampler | Partly | `config.py` (keys and built-in presets), the ranking function in `subsampler.py`; `strategies.py`, `store.py` outlines only |
| 7 | ModelLoader | Partly | `hf_whisper.py` (`load`) in full, the folder checks in `whisper_folder.py`, `config.py` |
| 8 | FeaturePipeline | Partly (the encoder cache part: its own new code, written and tested in Task 8) | `pipeline.py` (`build`), `collator.py`, `config.py`; `dataset.py`, `cache.py`, most of `audio_adapter.py` not read |
| 9 | TrainerCore | Partly | `trainer.py`, `freeze.py` in full; `hf_trainer.py` (`run_training` and the control callback), `monitor.py` (later), `config.py`; `final_model.py`, `sinks.py`, `run_folder.py` only as outlines (`final_model.py` was written in Task 1) |
| 10 | Evaluator | Partly | `evaluator.py` (`evaluate`, `make_decode_settings`), `hf_transcriber.py` (`transcribe`), `comparator.py`; `metrics.py`, `bootstrap.py`, `review.py`, `codec.py` not read |
| 11 | HyperparamSearcher | Partly | `config.py` (keys), `pruning.py`; `searcher.py`, `grid.py`, `study.py`, `parallel.py`, `optuna_strategy.py` not read; the trial table is from the real rehearsal report |
| 12 | Selector | Docs only | no source read; the example is the real rehearsal `selection_reason.txt` |
| 13 | ModelRegistry | Docs only | no source read; the example is the real rehearsal registry folder |
| 14 | ModelExporter | Docs only | no source read; the example is the real `exports.json` |
| 15 | RunManager | Partly | `config.py` (keys), the lock outline in `lock.py`; the folder content is from real experiment folders |
| 16 | Orchestrator and command line | Partly | `cli.py` (parser, plans), `wiring.py` (`exit_code_for`, `DataPipeline.prepare`, `SearchWiring.screening`), `planning.py` (formula), `finalists.py` and `stages.py` headers |
| 17 | ReportBuilder | Partly | `builder.py` (`build`, summary, verdict), names in `hints.py`, `config.py` (keys) |
| 18 | Tools | Partly | the `argparse` options of every tool, the real `fetch_info.json`; the rest from the documents |
| 19 | A run from start to end | Partly | the real logs of the rehearsal and turbo runs and `wiring.py`/`cli.py`; the stage order and file names from the documents |
| 20 | Labels shorter than the audio | Full | `textnormalizer/timestamps.py` and `types.py` read; the analysis ran the project's own `prepare` code on the real data |
| 21 | What you get and what each experiment tells you | Partly | file lists from the stage headers and real experiment folders; times from the stored benchmarks and logs; the turbo times marked estimated were never run |

## Progress table

| # | Component | Chapter written |
|---|-----------|-----------------|
| 1 | Config (C0) | yes |
| 2 | DataSource (C1) | yes |
| 3 | Splitter (C3) | yes |
| 4 | TextNormalizer (C2) | yes |
| 5 | SampleFilter (C1b) | yes |
| 6 | Subsampler (C3b) | yes |
| 7 | ModelLoader (C5) | yes |
| 8 | FeaturePipeline (C4) | yes |
| 9 | TrainerCore (C6) | yes |
| 10 | Evaluator (C7) | yes |
| 11 | HyperparamSearcher (C8) | yes |
| 12 | Selector (C9) | yes |
| 13 | ModelRegistry (C10) | yes |
| 14 | ModelExporter (C11) | yes |
| 15 | RunManager (C12) | yes |
| 16 | Orchestrator and command line (C13) | yes |
| 17 | ReportBuilder (C14) | yes |
| 18 | Tools (T) | yes |
| 19 | A run from start to end | yes |
| 20 | Labels shorter than the audio | yes |
| 21 | What you get and what each experiment tells you | yes |
| - | Mismatches found, Open questions | yes |

## Glossary

- **Clip**: one audio file of at most 30 seconds, with its text label. In the Daf Yomi data almost every clip is exactly 30 s.
- **Label**: the text that belongs to a clip. In this data the label is the reference answer ("the truth").
- **Batch**: a small group of clips handled together in one forward pass of the model. `train.hyperparams.batch_size` is its size.
- **Step**: one update of the model weights (one "optimizer step"). One step uses `batch_size x grad_accum_steps` clips.
- **Epoch**: one pass over all training clips. The system counts in steps, not epochs. A rehearsal of 20 steps with 4 clips per step sees 80 clips: far less than one epoch.
- **Group**: clips that belong together, here all clips of one lesson (`meta_entry_id`). A group is never split between Train, Val and Test, so the model is never tested on the lesson it learned from.
- **Split**: one of the three parts of the data: Train (the model learns from it), Val (we compare models and settings on it), Test (looked at once, for the winner only).
- **Val loss**: how surprised the model is by the correct text of the Val clips (a number from teacher-forced training maths, lower is better). It is cheap to compute. It is not the WER.
- **WER**: word error rate. The wrong words (replaced, missing, extra) divided by the number of words in the reference. 0.30 means 30 errors per 100 reference words. It can be above 1.0 when the model adds many extra words. Lower is better.
- **CER**: the same idea with characters.
- **Baseline**: the model before any training. A trained model must beat it, otherwise the baseline is kept.
- **Trial**: one training run with one set of settings, made during the search. Short, on a small part of the data.
- **Finalist**: one of the best trials, trained again on the full training data and then scored by WER on all of Val.
- **Experiment**: one numbered folder `exp_NNNN` that holds everything one command produced: config, logs, reports, models.

## Chapter template (the same for all chapters)

1. Purpose, 2. Input and output, 3. What it does, 4. Config keys, 5. How it connects, 6. What can go wrong, 7. A real example, 8. Status (tested / run for real / never run), 9. Known open issues.

---

## 1. Config (C0)

**1. Purpose.** Reads every setting from files, the environment and the command line, checks them, and hands the rest of the system one read-only settings object.

**2. Input and output.**
- In: a YAML file (for example `configs/work_like.yaml`), `--set a.b=value` words on the command line, environment variables named `STT__SECTION__KEY`, and an optional `.env` file in the project folder.
- Out: one frozen object `AppConfig`. Nothing is written by this component. (The Orchestrator later saves the final settings of every run as `config_full.yaml` in the experiment folder.)
- Code: `src/stt/config/` (`loader.py`, `sources.py`, `schema.py`, `validators.py`, `errors.py`).

**3. What it does.**
1. Reads the YAML file. If the file says `extends: other.yaml`, it reads that file first and puts the child's values on top. A loop of `extends` is refused.
2. Applies environment values (the real environment beats `.env`).
3. Applies `--set` values last, so they win over everything. Several `--set` for the same key: the later one wins.
4. Checks every value against the schema of its section (a section belongs to one component). An unknown key is an error and the message names the closest valid key. All problems are shown together.
5. Unless `runtime.online` is true, it sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE` to 1, so nothing can download.
6. Freezes the result. No component can change a setting while it runs.

Order of power, weakest first: the defaults in the code, the YAML (with `extends`), the environment, `--set`.

**4. Config keys.** The `runtime` section is the only one defined here.
- `runtime.online` (default `false`): true only at home. It switches the offline variables off. The model loader still never downloads, even when this is true.
- `runtime.seed` (default `42`): the random seed used everywhere. Same seed, same machine, same result.

All other sections are defined by the component that uses them. `docs/CONFIG_REFERENCE.md` lists every key (21 sections, 188 keys); `python -m stt config-docs` rewrites it from the code.

**5. How it connects.** Before it: only files and the command line. After it: every component receives its own section from it (`config.section("name", SchemaClass)`).

**6. What can go wrong.**
| Symptom | Cause | Fix |
|---|---|---|
| `ConfigError`, exit code 2, "unknown key ... closest valid key" | typo in a YAML or `--set` key | use the key the message suggests |
| exit code 2, a value of the wrong type | for example text where a number is needed | fix the value; lists and objects on the command line must be JSON, for example `--set 'train.sinks=["csv","console"]'` |
| exit code 2, "extends cycle" or a missing base file | `extends` points to itself or to a missing file | fix the file name (one file name only, not a list) |
| exit code 2, a `<<< FILL IN ... >>>` marker left | you copied `configs/work.yaml` and did not fill it in | `python -m stt check --config yours.yaml` names every marker left |

**7. A real example.** `configs/work_like.yaml` says `extends: home_full.yaml`; that file says `extends: data_gmara.yaml`; that one says `extends: base.yaml`. So the chain is base, then data_gmara (data paths and column names), then home_full (whisper-tiny, 100 steps), then work_like (rehearsal: 20 steps, own folders). For the turbo baseline run (`exp_0002`, 2026-10-25) the command added `--set model.path=models/whisper-large-v3-turbo-ivrit --set model.dtype=float32 --set train.cpu_threads=4 --set eval.batch_size=2`. The experiment folder stored the difference from the defaults in `config_diff.yaml` and the whole result in `config_full.yaml`.

**8. Status.**
- Tested: 34 tests in `tests/test_config.py` (offline), plus the config tests of the presets (`tests/test_presets.py`) and the generated reference (`tests/test_orchestrator_configdocs.py`).
- Run for real: every command of the system loads its settings this way.
- Never run: `.env` files and `STT__` environment variables are tested but the owner has not used them in a real run. `mypy --strict` was never run (mypy is not installed here), so the "100% type hints" rule is checked by reading only.

**9. Open issues.** None for this component alone. Related: O76 (adding a section changes the stored config of older experiments; fixed by a RunManager rule, see chapter 15) and O89 (the default of `pipeline.stages` changed, so very old experiments cannot be resumed).

---

## 2. DataSource (C1)

**1. Purpose.** Reads a folder of labeled audio into one uniform list of samples and writes down what it could not use.

**2. Input and output.**
- In: the `data` section. Either a folder with a `manifest.csv` (type `csv`) or a folder of Hugging Face parquet files (type `hf_parquet`, as downloaded by `tools/fetch_data.py` into `data\daf_yomi_raw`).
- Out: a list of `AudioSample` objects sorted by id, a `LoadReport`, and the list of rejected rows.
- On disk: for parquet, the audio is extracted once to `data\extracted\daf_yomi\audio\<split>\<id>.mp3`, plus `manifest.csv` and `extract_info.json` in `data\extracted\daf_yomi`. In a run, the file `rejected.csv` (id, reason, detail) is written into the experiment's `reports` folder.
- Code: `src/stt/datasource/` (`csv_source.py`, `parquet_source.py`, `factory.py`, `report.py`, `audio_header.py`).

**3. What it does.**
1. For parquet only: reads every `.parquet` file and writes each audio row to its own file. Ids are `<split>_<row number>`, for example `train_005206` or `eval_000000`. The split name is the start of the parquet file name (`train-0000...` gives `train`). Nested `metadata` fields become columns named `meta_<name>` (for example `meta_entry_id`, `meta_seek`). The manifest is written last, so a manifest on disk means "extraction complete". The source files are never changed.
2. Reads the manifest row by row. For each row it checks the audio file exists, the text is not empty, and the audio header can be read (the length in seconds comes from the file header with the `soundfile` library; audio is never decoded here).
3. Finds the group of the row (here `meta_entry_id`, the lesson).
4. Keeps every other column in `extra`, converted to the type given in `data.extra_types`.
5. A bad row is not loaded and gets one of six reasons: `missing_audio`, `empty_text`, `unreadable_audio`, `missing_group`, `malformed_row`, `bad_extra_value`.
6. Stops the whole run if too many rows were rejected, or none could be loaded.

Note that the label text is passed on exactly as it is in the data, including the timestamp markers such as `<|0.08|>`. The TextNormalizer (chapter 4) removes them. This matters for chapter 20.

**4. Config keys** (section `data`).
- `data.source.type`: `csv` or `hf_parquet`. `data.source.path`: the folder (must exist).
- `data.columns.audio`, `data.columns.text`: required. `data.columns.group`: the column that identifies the lesson. `data.columns.id` and `data.columns.split`: optional.
- `data.group_fallback` (default `fail`): what to do without a group column. `fail` is safe, because a wrong group lets the test lessons leak into training.
- `data.extra_types`: column to type (`str`, `int`, `float`, `bool`).
- `data.extract.dir`: where parquet audio goes (required for `hf_parquet`); `data.extract.overwrite` (default `false`).
- `data.on_invalid` (default `skip`; `fail` stops at the first bad row).
- `data.max_rejected_ratio` (default `0.05`): more rejected than this fails the run.

**5. How it connects.** Before it: the downloaded files. After it: the Splitter (chapter 3) gets the list of samples.

**6. What can go wrong.** All are exit code 3 (`DataSourceError`).
| Symptom | Cause | Fix |
|---|---|---|
| "Manifest not found" | wrong `data.source.path` | point to the folder that holds `manifest.csv` |
| "Column 'x' does not exist ... Existing columns: ..." | wrong column name in `data.columns` | copy a name from the list in the message |
| "No group column is mapped ... group_fallback is 'fail'" | no group column | map one, or set the fallback knowingly |
| "Rejected ratio ... is above data.max_rejected_ratio" | mapping is wrong, or many files are missing | read `rejected.csv`; fix the mapping. Do not just raise the ratio |
| "Duplicate sample ids" | wrong id column | fix `data.columns.id` |
| "extraction ... made from other source files" | the parquet files changed after extraction | set `data.extract.overwrite: true` or move the old folder away yourself |

**7. A real example.** Rehearsal and turbo runs (`load_report.txt` of `exp_0002`): `seen 8723, loaded 8723, rejected 0`; 8,286 rows from the `train` parquet files and 437 from `eval`; 710 distinct lessons; clip length minimum 2.95 s, median 30.00 s, maximum 30.00 s. The extracted folder is 794 MB. The first extraction took about 3 minutes; a later load takes about 10 seconds.

**8. Status.**
- Tested: `tests/test_datasource.py` (35 tests, tiny synthetic wav and parquet files, offline).
- Run for real: the full Daf Yomi data (8,723 rows, 0 rejected) many times.
- Never run: a `csv` source on real data other than the extracted Daf Yomi manifest (the parquet source reuses the same code, so it was exercised through it); a work-site dataset (its column names are unknown, see `docs/DATA_ONBOARDING.md`).

**9. Open issues.** O9 (mp3 on Windows: decided, works), O22 (size and structure of the work data: unknown), O13 (155 clips of 30 s have very little text; maybe silence at the end of a lesson).

---

## 3. Splitter (C3)

**1. Purpose.** Divides the samples into Train, Val and Test so that no group (lesson) is in two parts.

**2. Input and output.**
- In: the list of samples from the DataSource.
- Out: a `SplitResult`: for each split name the list of samples (sorted by id), a `SplitReport`, and two fingerprints (a short code for "this exact data" and "this exact split"). In a run the report is written as `reports\split_report.txt`.
- Optional on disk: with `split.dir` set, the split is saved as `<dir>\<version>\split.json` and reused. The default is "nothing saved": the split is made again each time, and it gives the same answer because it is deterministic.
- Code: `src/stt/splitter/` (`grouped_ratio.py`, `hash_group.py`, `predefined.py`, `time_block.py`, `checks.py`).

**3. What it does** (default mode `grouped_ratio`).
1. Puts the samples into groups by the group id (a different key can be set with `split.group_key.column` or `.regex`).
2. Puts the groups in a seeded, stable random order.
3. Takes the groups one by one. Each whole group goes to the split that is furthest below its target share (80% / 10% / 10% of the audio duration by default).
4. Fails if a requested share is missed by more than the tolerance (5 points).
5. Runs hard checks: no group in two splits, nothing lost and nothing doubled, no empty split. Then the minimum gates: at least 30 samples and 2 groups in every split.
6. Writes a report: the shares per split, the largest groups, the overlap check, a table of how the official `source_split` (train / eval) maps onto the new splits (the official `eval` split is NOT used as the Test: decision 26), and a speaker line.

Other modes (choose with `split.mode`): `hash_group` (each group's split comes from a hash, so adding groups never moves old ones), `predefined` (uses the `source_split` column), `time_block` (cuts one recording into blocks with a gap between them).

**4. Config keys** (section `split`).
- `split.mode` (default `grouped_ratio`), `split.names` (default train, val, test), `split.ratios` (0.8, 0.1, 0.1), `split.ratio_unit` (`duration`, or `samples`), `split.ratio_tolerance` (0.05).
- `split.seed`: empty means use `runtime.seed`.
- `split.min_samples_per_split` (30), `split.min_groups_per_split` (2): these are gates; both are only starting values (O22).
- `split.on_oversized_group` (`fail`), `split.on_overlap` (`fail`, for `predefined`), `split.speaker_column` (none).
- `split.dir`, `split.version` (`v1`), `split.reuse` (true), `split.export_csv` (false): saving a split. A saved split is never overwritten; a new version name is the only way to a new split.

**5. How it connects.** Before it: the DataSource. After it: the TextNormalizer and the SampleFilter work on each split (chapters 4 and 5). The Subsampler (chapter 6) may later shrink the splits, but it never moves a clip from one split to another.

**6. What can go wrong.** `SplitError`, exit code 3.
| Symptom | Cause | Fix |
|---|---|---|
| "The ratios cannot be met within split.ratio_tolerance" | too few or too large groups | use a finer group key; raise the tolerance only on purpose |
| "Split 'x' is empty" or "has N samples (minimum 30)" | too little data | check the data; review the gates (they are DEFAULT values) |
| "a group is oversized" | one group is bigger than a split can hold | finer group key, or `split.on_oversized_group: warn` |
| "saved split ... data fingerprint differs" | the data changed after the split was saved | create a new `split.version` |

**7. A real example.** `split_report.txt` of the turbo run `exp_0002`: 8,723 samples, 710 groups. Train 6,973 samples, 566 groups, 57.90 h (share 0.800). Val 875 samples, 72 groups, 7.25 h. Test 875 samples, 72 groups, 7.26 h. The largest group is 0.35% of the data. Group overlap check: ok. Data fingerprint `349a08f9a17b1400`, split fingerprint `5342b8562205736e`. The report also says: "the Test may share speakers with Train" (there is one speaker in the data, so the Test measures new lessons, not new speakers).

**8. Status.**
- Tested: `tests/test_splitter.py` (41 tests, synthetic samples).
- Run for real: `grouped_ratio` on all the real data, in every run.
- Never run for real: `hash_group`, `predefined` and `time_block` on real data; saving and reusing a split with `split.dir` (tested only).

**9. Open issues.** O16 (ratios: default), O18 (group key by lesson or by tractate), O19 (the same sentence can be in Train and Test; measured, not removed), O20 (a second test set), O22 (work data volume).

---

## 4. TextNormalizer (C2)

**1. Purpose.** Makes two cleaned texts from every raw label: one to train on (`train_text`), one to score against (`eval_text`).

**2. Input and output.**
- In: the samples from the splits.
- Out: for each sample a `NormalizedSample` with `train_text`, `eval_text`, flags, and the parsed timestamp segments. The raw text stays untouched in `sample.text`. Written in a run: `reports\text_report.txt`.
- Also used by the Evaluator (chapter 10) to clean the model's output with the same recipe: `build_profile(config, "eval")`.
- Code: `src/stt/textnormalizer/` (`normalizer.py`, `steps.py`, `registry.py`, `timestamps.py`).

**3. What it does.**
1. Parses the timestamp markers `<|0.08|>` before anything is removed. A label looks like `<|0.08|>text<|1.84|><|2.34|>more text<|3.92|>`. The parsed pieces are kept in `timestamp_segments` (start, end, text), or `None` when the markers are missing or malformed.
2. Applies the train steps (default): `unicode_clean`, `strip_timestamps`, `unify_quotes`, `remove_bracket_markup`, `collapse_whitespace`. Punctuation stays, so the model can learn it.
3. Applies the eval steps (default): `unicode_clean`, `strip_timestamps`, `remove_punctuation`, `unify_quotes`, `remove_nikud`, `remove_bracket_markup`, `lowercase_latin`, `collapse_whitespace`. No punctuation, no vowel marks.
4. Repeats a profile until the text stops changing (at most 10 passes), so the result is stable.
5. Sets flags (`had_timestamps`, `had_brackets`, `had_digits`, `had_latin`, `had_nikud`, `bad_timestamps`, `empty_after_normalization`) and counts every unusual character in a character inventory.
6. Never drops a sample.

**4. Config keys** (section `text`).
- `text.eval_version` (default 1): the version of the eval recipe. Results with different versions are not comparable. Change it whenever you change the eval steps.
- `text.profiles.train.steps` and `text.profiles.eval.steps`: the step lists above.
- `text.bracket.pairs`, `text.bracket.mode` (`chars_only`: remove the bracket characters and keep the words inside), and a per-profile `bracket_mode`.
- `text.punctuation.chars`, `text.punctuation.space_chars` (characters replaced by a space; the Hebrew hyphen `maqaf`), `text.quotes.map`, `text.unicode.invisible_chars`, `text.timestamp_pattern`, `text.inventory.top_n` (50).
- `text.train.timestamps` (`strip` only in v1).

**5. How it connects.** Before it: the split samples. After it: the SampleFilter reads `eval_text` (to measure length); the FeaturePipeline turns `train_text` into label tokens; the Evaluator uses `eval_text` as the reference and the same eval recipe on the model's output.

**6. What can go wrong.** `TextNormalizerError`, exit code 2: an unknown step name (the message lists the valid names); `train.timestamps: keep` is refused at config time.

**7. A real example** (`text_report.txt`, `exp_0002`): all 8,723 labels contain timestamp markers; 345 contain brackets; no text became empty. The first Val clip `eval_000000` starts raw as `<|0.08|>שרץ מטמא את המשקין, <|1.84|><|2.34|>ומשקין מטמאין את הכלי, ...`. Its `eval_text` is `שרץ מטמא את המשקין ומשקין מטמאין את הכלי וכלי מטמא את האוכלין ...` (no commas, no markers). The geresh sign `׳` (2,167 uses in 1,662 labels) is kept in `train_text` and removed in `eval_text` (owner decision 29).

**8. Status.**
- Tested: `tests/test_textnormalizer.py` (36 tests, including a random idempotence test).
- Run for real: all 8,723 real labels, every run.
- Never run: the `keep` timestamp mode (does not exist in v1); the non-speech marker recipe (O47) on real data; abbreviation or number normalization (not built).

**9. Open issues.** O3 / O48 (timestamps: strip only), O19 (repeated text between splits is not detected), O25, O26 (abbreviations), O35, O47.

---

## 5. SampleFilter (C1b)

**1. Purpose.** Decides, per split, which samples are good enough to keep, and reports exactly what would be removed and why.

**2. Input and output.**
- In: the normalized samples of one split and its name.
- Out: a `FilterResult` (`kept`, `rejected`, `would_reject`, a report). In a run: `reports\filter_report.txt`.
- Code: `src/stt/samplefilter/`.

**3. What it does.**
1. Runs every rule that applies to this split. Default rules: `min_duration` (1.0 s), `max_duration` (30.0 s), `min_text_chars` (2, measured on `eval_text`) for all splits. Quality rules (`min_quality_score`, `max_bad_segments`, `require_golden`) are off until a threshold is set, and apply to Train only. An optional rule `max_label_tokens` is off.
2. In `report` mode (the default) nothing is removed: the report lists what WOULD be removed (`would_reject`). In `apply` mode the failing samples are removed.
3. Checks the gate: if a split would keep less than 40% of its samples, the run fails.

The rules read only the numbers already known (the duration comes from the audio header). They never open the audio.

**4. Config keys** (section `filter`).
- `filter.mode` (default `report`; `apply` to remove).
- `filter.rules.<name>.enabled / .applies_to / .kind / .threshold / .field / .measure_on`.
- `filter.min_kept_ratio` (default 0.4).

**5. How it connects.** Before it: the TextNormalizer. After it: the Subsampler (chapter 6), then the FeaturePipeline. Note: in `report` mode the FeaturePipeline would FAIL on a clip longer than 30 s, because it never cuts audio. In the Daf Yomi data no clip is longer than 30 s.

**6. What can go wrong.** `FilterError` and `FilterGateError`, exit code 3.
| Symptom | Cause | Fix |
|---|---|---|
| "unknown rule" (lists the names) | typo in the rule name | use a listed name |
| "enabled rule without threshold or field" | the quality rule has no `field` | set `field` to the extra column name or disable the rule |
| gate error "keeps less than min_kept_ratio" | thresholds too strict, or the data is bad | read the per-rule counts in the message |

**7. A real example** (`filter_report.txt`, `exp_0002`): every split `mode report, seen 6973 / 875 / 875, kept all, would reject 0`. With the defaults, nothing is removed from the Daf Yomi data. For comparison, the O5 measurements showed that `min_quality_score` 0.15 would remove 612 of 6,973 Train clips; this was never applied.

**8. Status.**
- Tested: `tests/test_samplefilter.py` (29 tests).
- Run for real: in `report` mode on the real data, every run.
- Never run: `apply` mode on real data (so the effect of removing clips on the final WER was never measured: it is a planned A/B experiment, O5, O6).

**9. Open issues.** O5, O6 (thresholds and which splits), O12 (gates), O13 (a rule comparing text length with duration), O34 (label length rule lives in the FeaturePipeline), O47.

---

## 6. Subsampler (C3b)

**1. Purpose.** Picks a smaller, repeatable part of the data, so a rehearsal or a search can run in minutes instead of hours.

**2. Input and output.**
- In: the filtered splits, and the size wanted.
- Out: the smaller splits and a `SubsampleReport`. In a run: `reports\subsample_report.txt` (and, in a search, `reports\search_screening.txt`).
- Optional on disk: with `subsample.dir` set, the chosen ids are saved as JSON; a stored file is never overwritten.
- Code: `src/stt/subsampler/` (`subsampler.py`, `strategies.py`, `store.py`, `config.py`).

**3. What it does.**
1. Does nothing unless `subsample.enabled` is true (the default is false; the full runs use all data).
2. Gives every unit (a clip with `by_sample`, a whole lesson with `by_group`) a rank: the first 8 bytes of SHA-256 of `seed|repeat_index|unit key`. The rank does not depend on the order of the data. The same seed always picks the same clips, and a smaller subset is always inside a larger one.
3. Takes units in rank order until the wanted size is reached. The unit that crosses the target is kept, so the result can be a little over (reported as "overshoot").
4. Scope: `train_only` (default) shrinks Train only; Val and Test stay whole. `all_splits` shrinks all three by the same share (used by the rehearsal).
5. Val choices with `train_only`: `val: fixed` (whole Val), `val: follow` (same share as Train), or `val_samples: N` (a fixed slice of N Val clips, whatever the Train size; the same slice for every Train size).
6. Fails if the wanted size is larger than what exists, or fewer than `min_train_samples` (10) Train clips would be left.

Built-in presets (in the code): `smoke` 20 clips; `tiny` 30 minutes; `work_like` 10% of the audio, by lesson, over all three splits; `half` 50% of the audio; `full` everything.

**4. Config keys** (section `subsample`).
`enabled` (false), `size` (`all`, or `minutes`, `samples` or `fraction`), `strategy` (`by_sample`), `scope` (`train_only`), `val` (`fixed`), `val_samples` (none), `repeats` (1), `repeat_index` (0), `min_train_samples` (10), `preset` (none), `presets` (your own), `dir` (none). The seed is `runtime.seed`.

Important: when `subsample.enabled` is true the cut is made once, in the `prepare` stage, and every later stage sees only the cut data. When `enabled` is false (the `home_full` preset), `val_samples` only limits the Val set used for the Val loss in the search and the finalist training; every WER scoring still uses the whole Val.

**5. How it connects.** Before it: the SampleFilter. After it: the model loader and feature pipeline see only the kept clips. The search stage applies a second, smaller preset (`search.screening.preset`, default `tiny`) to Train for its trials.

**6. What can go wrong.** `SubsampleConfigError` (exit code 2: unknown preset or strategy; minutes or samples with `all_splits`; `follow` without a Val split). `SubsampleError` (exit code 3: size larger than the data; fewer than `min_train_samples` clips; a stored file that is stale or damaged).

**7. A real example** (`subsample_report.txt`, `exp_0002`, preset `work_like`): requested 10% of the audio, achieved 350.45 minutes for 347.42 requested (overshoot +3.03). Train 6,973 clips / 566 lessons became 702 clips / 58 lessons (350.4 min). Val 875 / 72 became 101 / 7 lessons (50.3 min). Test 875 / 72 became 97 / 8 lessons (48.3 min). Seven lessons in Val is the point of the rehearsal: it shows how noisy a small Val set is.

**8. Status.**
- Tested: `tests/test_subsampler.py` (26 tests) and one in `tests/test_orchestrator.py`.
- Run for real: `work_like` and `tiny` on the real splits; the rehearsal and the turbo runs.
- Never run for real: `half` and `by_sample` over `all_splits`; the stored-subset JSON files with `subsample.dir`.

**9. Open issues.** O11 and O77 (preset sizes: unchanged until step times are known), O21 (design), O78 (Val slice, done).

---

## 7. ModelLoader (C5)

**1. Purpose.** Loads one Whisper model, its audio feature extractor and its tokenizer from a LOCAL folder, checks them, and describes what was loaded.

**2. Input and output.**
- In: the `model` section; a folder such as `models\whisper-tiny` or `models\whisper-large-v3-turbo-ivrit`.
- Out: a `LoadedModel` with the model, a feature extractor, a tokenizer and a `ModelInfo` (name, path, number of parameters, dtype, mel bins, language, task).
- Code: `src/stt/modelloader/` (`hf_whisper.py`, `whisper_folder.py`, `token_counter.py`). Only `hf_whisper.py` imports torch and transformers.

**3. What it does.**
1. Checks the folder before loading: it exists; `config.json` and `preprocessor_config.json`; weights in safetensors format (`model.safetensors`, or `model.safetensors.index.json` for sharded weights); a tokenizer; `tokenizer_config.json`; `model_type` is `whisper`; the model is not English-only. All missing files are listed together. Old formats (`pytorch_model.bin`) are refused.
2. Loads with `local_files_only=True` and `trust_remote_code=False`, in the requested dtype. It never goes to the internet, even when `runtime.online` is true.
3. Checks the tokenizer knows the language token (`<|he|>`), sets language and task, and puts the model in evaluation mode. (Training switches it to training mode itself.)
4. Counts the parameters and reads the number of mel bins (80 for whisper-tiny, 128 for the large-v3 family).
5. Every `load()` call returns new, independent objects.

**4. Config keys** (section `model`). `model.path` (required: a folder, never a Hub name), `model.loader` (`hf_whisper`), `model.language` (`he`), `model.task` (`transcribe`), `model.dtype` (`float32`; `float16` and `bfloat16` load but are untested for speed on CPU).

**5. How it connects.** Before it: a model folder copied or fetched by `tools/fetch_model.py`. After it: the FeaturePipeline uses its extractor and tokenizer; the TrainerCore trains the model; the Evaluator decodes with it.

**6. What can go wrong.** `ModelLoaderError`, exit code 2.
| Symptom | Fix |
|---|---|
| "Model folder not found" | copy the folder, or run `tools/fetch_model.py` at home. Do not turn on online mode |
| "model folder is incomplete. Missing: ..." | copy the whole folder again |
| "only in an old format" | convert once at home to safetensors |
| "Not enough memory to load" | smaller model, or close programs (no automatic retry) |
| "does not know the language" | wrong `model.language` code, or an English-only model |

**7. A real example.** The turbo model `ivrit-ai/whisper-large-v3-turbo` was fetched to `models\whisper-large-v3-turbo-ivrit` (3,090.1 MiB, one `model.safetensors` file, not sharded). Loaded in float32 it has 808,878,080 parameters and 128 mel bins (`exp_0002`, `eval_result_baseline.json`). In that run the load took about 17 seconds (log: evaluate stage starts 00:31:54, evaluation begins 00:32:11). The tiny model has 37,760,640 parameters and 80 mel bins and loads in about 6 s the first time.

**8. Status.**
- Tested: `tests/test_modelloader.py` (28 tests) with the stored tiny random-weight test model `tests/assets/tiny_whisper`, network blocked.
- Run for real: `whisper-tiny` and the turbo model (float32).
- Never run: sharded safetensors (`model.safetensors.index.json`): the check accepts it and the export and registry copy and hash all files, but no sharded model was ever loaded, trained or exported. GPU, `float16`/`bfloat16` training or decoding on CPU.

**9. Open issues.** O28 and O66 (which base model fits the work machine), O32 (old weights refused), O29 (freeze / LoRA, see chapter 9).

---

## 8. FeaturePipeline (C4)

**1. Purpose.** Turns each kept sample into what the model eats: audio features (the input) and label numbers (the target), and makes batches.

**2. Input and output.**
- In: the kept samples of one split, and the loaded model (only its feature extractor and tokenizer are used).
- Out: a lazy dataset (`dataset[i]` computes the features of clip `i` when asked), a `FeatureReport`, and a batch maker (the collator).
- One example is `input_features` (a table of numbers: mel bins by 3000 frames, for a 30 s window) and `labels` (token numbers). One batch is `input_features` of shape (batch, mel bins, 3000) and `labels` of shape (batch, longest label), padded with -100 (the loss ignores -100).
- On disk: nothing by default. An optional cache of `.npy` files if `features.cache.enabled` is true.
- Code: `src/stt/featurepipeline/` (`pipeline.py`, `dataset.py`, `audio_adapter.py`, `collator.py`, `cache.py`). Only the audio adapter imports soundfile and scipy; only the collator imports torch.

**3. What it does.**
1. Fails at once if any `train_text` is empty.
2. Reads every audio header. Clips at another sample rate are converted to 16 kHz (scipy `resample_poly`) when loaded, stereo is averaged to mono, and both are counted.
3. Fails if any clip is longer than the model window (30 s, plus a tiny tolerance). Audio is NEVER cut. The fix named in the message is to switch the SampleFilter to `apply` mode with its `max_duration` rule.
4. Turns each `train_text` into label tokens with the model's tokenizer: start-of-text, language, task, no-timestamps, the text, end-of-text.
5. Removes labels longer than the model limit (448 tokens), lists them, and fails if more than `features.max_dropped_ratio` (5%) were removed.
6. Returns the lazy dataset. A clip's features are computed only when training or scoring asks for it (about 21 ms per clip). The extractor pads a short clip to the full window, so every clip costs the same to the model.
7. The collator pads labels with -100 and drops the leading start token (the model adds it when it shifts the labels).

**4. Config keys** (section `features`). `features.num_workers` (0), `features.cache.enabled` (false), `features.cache.dir`, `features.max_dropped_ratio` (0.05), `features.downmix` (`mean`), `features.shuffle` (false), `features.seed` (none = `runtime.seed`). Sampling rate, window, mel bins and label limit come from the model, not from Config.

**5. How it connects.** Before it: the SampleFilter and the ModelLoader. After it: the TrainerCore reads the batches; the Evaluator reads the features of the Val clips.

**6. What can go wrong.** `FeaturePipelineError`, exit code 3.
| Symptom | Fix |
|---|---|
| "clip(s) ... longer than the model window" | `--set filter.mode=apply` (the `max_duration` rule is on by default) |
| "labels ... exceed the limit of 448 tokens ... above features.max_dropped_ratio" | check that timestamps are stripped; do not raise the ratio without a decision |
| "Cannot read the audio of sample ..." | the file is missing or damaged; look at `rejected.csv` and the data folder |
| "empty train_text" | `--set filter.mode=apply` so `min_text_chars` removes it |

**7. A real example.** For the Daf Yomi labels with the Whisper tokenizer, label length is 1 to 220 tokens, median 151, 95th percentile 182 (measured in O34 with the whisper-tiny tokenizer), far below 448. One feature table is 80 x 3000 x 4 bytes = 0.96 MB for whisper-tiny and 128 x 3000 x 4 bytes = 1.54 MB for the turbo model (computed). Computing it takes about 21 ms per clip, against about 1.2 s for training on the same clip with whisper-tiny at 2 threads, so the cache would save under 2% and stays off (decided).


**Encoder cache (added in Task 8, DECISIONS 89).** With `features.encoder_cache.enabled: true` (default false) and a frozen encoder, the pipeline computes the output of the encoder once per clip into `<dir>\enc_<hash>_<dtype>\<key>.npy`, and the examples carry that output instead of the mel table. It refuses to start (exit code 2) if the encoder is not frozen, if the model has dropout or layerdrop above 0 or `apply_spec_augment`, or if the disk is too small; files are written atomically so a stopped build continues; a one-time check compares the Val loss with and without the cache on 3 clips. Measured with turbo on 8 clips: a training step of 4 clips 58 to 63 s becomes 7.4 to 7.7 s, a Val-loss pass 111 to 125 s becomes 4.6 to 5.4 s, decoding 128 to 139 s becomes 23 to 24 s; the decoded text and the WER (0.8150) are identical. A trained model with a frozen encoder finds the same cache (same encoder weights). Never run: a whole run with the cache, or on the work machine.

**8. Status.**
- Tested: `tests/test_featurepipeline.py` (32 tests) and `tests/test_encoder_cache.py` (19 tests, the encoder cache).
- Run for real: all real training and scoring runs, with `whisper-tiny` (80 mel bins) and the turbo model (128 mel bins).
- Never run: the cache on real data in a real run; resampling or stereo on real data (the Daf Yomi clips are already 16 kHz mono).

**9. Open issues.** O2 (workers on Windows; not needed), O33 (short clips cost as much as 30 s clips), O34 (label length rule placement), feature cache decision (cache stays off).

---

## 9. TrainerCore (C6)

**1. Purpose.** Runs ONE training of a model on given data with given settings, and returns the trained model folder and a record of the run.

**2. Input and output.**
- In: a `TrainRequest`: the model, the Train and Val datasets, the batch maker, the hyperparameters, the output folder (and optionally a name prefix, a stop signal, a callback after each evaluation).
- Out: a `TrainResult`: the model folder, the history of measurements, the best step and its Val loss, why it stopped, timings, and which parameters were trained.
- On disk, in the output folder: `run_info.json` (what the run started with), `metrics.csv` (live, one row per record), `checkpoints\checkpoint-<step>\` (the latest and the best are kept), and the returned model folder (`model`, or `<prefix>_step_<step>` in a pipeline run) with weights, tokenizer files and `pipeline_model_info.json`.
- Code: `src/stt/trainercore/` (`trainer.py`, `hf_trainer.py`, `freeze.py`, `final_model.py`, `run_folder.py`, `sinks.py`, `monitor.py`). Only `hf_trainer.py` imports torch and transformers.

**3. What it does.**
1. Checks the request (not empty; the hyperparameters make sense; warmup not longer than max steps). Refuses to overwrite a non-empty folder.
2. Applies the freeze policy: `none` (train every parameter) or `freeze_encoder` (every parameter whose name starts with `model.encoder.` gets `requires_grad = False`; the decoder is trained). A frozen encoder is still run in the forward pass, but no gradients or optimizer values are kept for it.
3. Runs the Hugging Face `Seq2SeqTrainer` on the CPU: AdamW, a linear schedule (warmup, then straight down to 0), gradient clipping, float32.
4. One step = `batch_size x grad_accum_steps` clips. After each step it checks the stop signal and the time cap.
5. Every `log_every_steps` it writes a record (loss, learning rate, gradient size, seconds per step). Every `eval_every_steps` and at the end, it computes the Val loss on the whole Val set it was given (no text is generated here).
6. Saves a checkpoint every `save_every_steps`, at each new best Val loss, at every stop, and at the end. Keeps only the latest and the best.
7. Stops on: `max_steps`, early stopping (no improvement for `patience` evaluations), the time cap, or a stop signal. A NaN or infinite loss stops the run with an error that names the step and the last good checkpoint.
8. Builds the returned model folder from the best checkpoint by Val loss (the latest if there was no evaluation), copying tokenizer files from the base model. The folder is built under a `.tmp` name and renamed; if Windows refuses the rename it tries again (`train.rename_retries`, `train.rename_retry_seconds`), and if every try fails it keeps both the `.tmp` folder and the checkpoint and tells you how to continue.
9. Resume (`train.resume: true`) continues from the latest checkpoint only if the model, hyperparameters, freeze policy and data sizes match and the checkpoint holds `optimizer.pt`.

**4. Config keys** (section `train`).
- `train.freeze_policy` (`none`; or `freeze_encoder`).
- `train.hyperparams.learning_rate` (1e-5), `.max_steps` (100), `.batch_size` (4), `.grad_accum_steps` (4), `.warmup_steps` (10), `.weight_decay` (0.0), `.max_grad_norm` (1.0), `.seed` (none = `runtime.seed`).
- `train.log_every_steps` (10), `train.eval_every_steps` (50), `train.save_every_steps` (100).
- `train.early_stopping.patience` (5 evaluations), `.min_delta` (0.0).
- `train.max_runtime_minutes` (none), `train.cpu_threads` (none = half the cores), `train.resume` (false).
- `train.save_optimizer_state` (true): false saves weights only, about one third of the disk, and the run can never be resumed.
- `train.rename_retries` (5), `train.rename_retry_seconds` (2.0).
- `train.sinks` (`csv`, `console`; `html` is opt-in for a live progress page).
- `train.output_dir` is set by the program, per experiment.

**5. How it connects.** Before it: the FeaturePipeline (datasets and batches) and the ModelLoader (model). After it: the Orchestrator uses the returned folder for the Evaluator; the Searcher ranks trials by `best_val_loss`.

**6. What can go wrong.**
| Symptom | Exit code | Fix |
|---|---|---|
| `TrainerConfigError`: output folder not empty, unknown policy or sink, bad hyperparameters | 2 | use a new folder; fix the key named |
| Resume refused ("holds no optimizer.pt", or `resume` with `save_optimizer_state: false`) | 2 | start a new experiment; keep `save_optimizer_state: true` for runs that may need resuming |
| `TrainerCoreError`: loss became NaN or infinite | 4 | lower `learning_rate`; read the first lines of `metrics.csv` |
| Out of memory | 4 | lower `batch_size` and raise `grad_accum_steps`; use `freeze_encoder`; smaller model; close programs. There is no automatic retry |
| Could not rename the finished model folder | 4 | close the program holding it (antivirus, indexer, sync), then rename the `.tmp` folder by hand or rerun with `train.resume: true` |
| Training stopped by Ctrl+C | 130 | rerun the same command with `--experiment exp_NNNN` |

**7. A real example** (the rehearsal, whisper-tiny, finalist 1, 20 steps, batch 2 x accumulation 2, `metrics.csv`): training loss fell from 4.62 (step 1) to 3.24 (step 20); Val loss was 1.71 at step 10 and 1.62 at step 20; the real speed was about 5.4 seconds per step. The `seconds_per_step` column of that file shows 14 to 18 s: that is the known timing bug O121 (the number is too high by up to 3 times; elapsed time and percentages are right). Disk: one whisper-tiny checkpoint is about 456 MB, a trial folder about 600 MB; the whole rehearsal folder was 4.3 GB. For the turbo model (808,878,080 parameters, 171,909,120 of them trained with `freeze_encoder`) see chapter 19 and the benchmark numbers recorded there.

**8. Status.**
- Tested: `tests/test_trainercore.py` (51 tests, tiny test model, offline), including resume equality, NaN, out of memory, simulated Windows rename failures and the weights-only mode.
- Run for real: full fine-tuning of whisper-tiny (many runs); `freeze_encoder` and the turbo model only through the benchmark of 2026-10-25 (Task 2, chapter 19).
- Never run: a complete search or finalist training with the turbo model; the real Windows rename failure retry (simulated only: O122 is "fixed in code, still to be confirmed by a real run"); `train.max_runtime_minutes` and early stopping on real data; resume of a real long run.

**9. Open issues.** O121 (timing numbers too high), O122 (rename retry, unconfirmed on a real failure), O29 (freeze or LoRA), O39 (training defaults are placeholders), O64 and O86 (disk per run), O63 (Val loss cost), O120 (step time varies on this machine).

---

## 10. Evaluator (C7)

**1. Purpose.** Lets a model transcribe a set of clips, scores the text against the references, and compares two scored results with a statistics test.

**2. Input and output.**
- In: an `EvalRequest`: the items (id, group, reference text, audio path, attributes such as duration), the features dataset, the model, the split name, the eval version, the eval text recipe, and the decode settings.
- Out: an `EvalResult`: corpus WER, CER, mean per-file WER, number of files and words, the share of files under 10% and 20% WER, one `FileScore` per clip (reference, cleaned and raw model text, errors, WER, CER, flags), the decode settings, seconds, flags, and a review list for listening.
- `Comparator.compare(baseline, candidate)` gives a `ComparisonResult`: difference of WER (positive = candidate better), a confidence interval, a verdict (`improved`, `no_clear_difference`, `worse`, `not_comparable`, `insufficient_groups`) and one plain sentence.
- On disk: the Evaluator writes nothing. The Orchestrator stores results as JSON (a lossless copy: `eval_result_baseline.json`, `finalists\*.val.json`, `test_result_*.json`) through `stt.evaluator.codec`.
- Code: `src/stt/evaluator/` (`evaluator.py`, `metrics.py`, `comparator.py`, `bootstrap.py`, `hf_transcriber.py`, `review.py`, `codec.py`). Only `hf_transcriber.py` imports torch.

**3. What it does.**
1. The transcriber takes the clips in batches (`eval.batch_size`), and the model generates text greedily (`num_beams` 1). Language and task come from the loaded model; no timestamps are asked for.
2. Each clip gets its own length cap: `min(448 minus the prompt, ceil(8.0 x seconds))` tokens, so about 240 for a 30 s clip. An output that uses its whole cap without ending is cut and flagged `hit_length_cap` (usually a looping model).
3. Each model text is cleaned with the same eval recipe as the reference (chapter 4).
4. WER per clip is edit distance in words divided by the reference words. The corpus WER is the sum of all errors divided by the sum of all reference words (so long clips weigh more). CER is the same with characters, spaces included. The mean per-file WER is the plain average of the per-clip WERs (short or badly wrong clips weigh more); it is shown next to the corpus WER and is not used to decide.
5. A clip with an empty reference is left out and listed. An empty model output counts as all words wrong and is flagged. If every output is empty, the run fails (the likely cause is a wrong language setting).
6. The review list holds the worst clips first, then random ones, with a word-by-word difference (`[a->b]` replaced, `[-a-]` missing, `[+b+]` extra) and the audio path, so you can listen.
7. The comparison resamples whole groups (lessons), not single clips, many times (the paired bootstrap) and checks whether the lower end of the difference is above zero (default: 90% confidence, one-sided, 2000 resamples). With fewer than 5 groups it refuses to make a confidence statement (`insufficient_groups`) and gives only the plain difference. Results with different `eval_version` are `not_comparable`.

**4. Config keys** (section `eval`).
- `eval.metrics` (`wer`, `cer`; `wer` is required).
- `eval.generation.num_beams` (1), `.max_new_tokens` (the model limit), `.max_tokens_per_audio_sec` (8.0; none switches the per-clip cap off).
- `eval.batch_size` (4). The turbo runs used 2 to save memory.
- `eval.thresholds.wer_below` (0.10, 0.20).
- `eval.significance.method` (`paired_group_bootstrap`), `.confidence` (0.90), `.one_sided` (true), `.n_resamples` (2000), `.min_effect_abs` (0.0), `.min_groups` (5).
- `eval.review.n_random` (20), `.n_worst` (10). `eval.sanity.max_baseline_wer` (none).

**5. How it connects.** Before it: the model (ModelLoader or a trained model folder) and the Val or Test features. After it: the Selector (chapter 12) uses the comparisons; the registry stores the numbers; the report shows them.

**6. What can go wrong.**
| Symptom | Exit code | Fix |
|---|---|---|
| `EvaluatorConfigError`: unknown metric or method, `wer` missing | 2 | fix `eval.metrics` / `eval.significance.method` |
| "Every hypothesis is empty" | 4 | check `model.language` and the feature report |
| "Out of memory while transcribing" | 4 | lower `eval.batch_size`; close programs |
| "The two results were not scored on the same files" | 4 | compare results of the same split |
| WER above 1.0 | none | not an error: the model inserted more words than the reference has. Look at the `hit_length_cap` count and the review list |

**7. A real example.** The turbo model (`ivrit-ai/whisper-large-v3-turbo`, no training) on the 101 rehearsal Val clips (`exp_0002`, 2026-10-25, float32, 4 threads, batch 2): WER 0.8335, CER 0.3889, mean per-file WER 0.9613 on 4,697 reference words. Of 3,915 word errors, 3,018 are replaced words, 153 missing and 744 extra (computed from the stored per-clip text). One output hit the length cap. It took 1,681 s, about 16.6 s per clip. The untrained whisper-tiny on the same 101 clips gave WER 1.1318, CER 0.6925, and 17 of 101 outputs hit the cap (rehearsal summary). The 8-clip turbo probe (`exp_0001`) gave WER 0.7974 on only 8 clips: do not compare it with the 101-clip number.

**8. Status.**
- Tested: `tests/test_evaluator.py` (23 tests), `tests/test_evaluator_codec.py` (7 tests).
- Run for real: whisper-tiny and the turbo model, baseline and trained (tiny), on Val clips; the Test split once for the rehearsal winner.
- Never run: the Evaluator on a GPU; beam search (`num_beams` above 1) on real data; `min_effect_abs` above 0; a trained turbo model.

**9. Open issues.** O24 (CER next to WER, decided), O38 (WER during training: not built), O41 (strictness of significance), O66 (decode time of large models), O67 (length cap 8 tokens/s: recheck on work data), O98 (scoring dominates the planned time), O117 (WER over steps: not built).

---

## 11. HyperparamSearcher (C8)

**1. Purpose.** Tries several settings (for example several learning rates), one training per setting, and ranks them by Val loss.

**2. Input and output.**
- In: a `SearchSpec` (the parameters and their ranges, the strategy, the number of trials, pruning, failure limits, how many finalists, the study folder, the seed, the thread budget) and a `TrialRunner` (a function that trains one trial; the Orchestrator supplies it).
- Out: a `SearchResult`: every trial with its parameters and outcome (`complete`, `pruned` or `failed`, Val loss, steps, seconds, model folder), the ranked best trials (at most `finalists.k`), why it stopped, and a report.
- On disk (in a run): `search\study.json` (rewritten after every finished trial), `search\trial_TTT.params.json`, `reports\search_report.txt` (read this first), `reports\search_trials.json`, `reports\search_screening.txt`, and the trial folders `candidates\exp_NNNN_trial_TTT\` with live metrics in `metrics\trial_TTT\`.
- Code: `src/stt/searcher/` (`searcher.py`, `grid.py`, `optuna_strategy.py`, `pruning.py`, `parallel.py`, `study.py`). Optuna is imported only in `optuna_strategy.py`, and only if that strategy is chosen.

**3. What it does.**
1. Builds the list of trials with the strategy. `grid` (the default): the product of the values of each parameter in a fixed order (for a log range with `points: 4`, four values spaced evenly on a log scale). `optuna_tpe`: Optuna's TPE sampler (ask and tell, in memory).
2. Runs the trials one after another (`parallel_trials: 1`), or in separate worker processes (`parallel_trials` above 1; each gets `threads / parallel_trials` threads).
3. Each trial is handed to the runner, which trains a fresh model on the SCREENING data (a small part of Train, preset `tiny` by default) and reports each Val loss back.
4. Pruning (on by default in the code; off in `home_full`): after `min_evals` Val evaluations, a trial is stopped if its Val loss is worse than the median of the other finished trials at the same evaluation. With few evaluations per trial it rarely fires; it is meant for trials that evaluate four times or more. At least two finished trials must exist, so the first trials are never pruned.
5. Records every outcome in `study.json`. A rerun with the same settings continues; a changed space, strategy or seed is refused.
6. Stops after `n_trials`, after the time budget, when too many trials failed (`max_failed_ratio`, checked after `min_trials_before_abort`), or when the grid is used up.
7. Returns the best `finalists.k` completed trials by Val loss. It does not train finalists and does not pick a winner.

**4. Config keys** (section `search`).
- `search.strategy` (`grid`), `search.parameters.<name>` (`kind` = `log_float` / `float` / `int` / `categorical`, `low`, `high`, `choices`, `points` 3). The default space is `learning_rate` (5e-6 to 5e-5, log) and `grad_accum_steps` (2 or 4). Naming `search.parameters` replaces the whole default space. `max_steps` and `seed` cannot be searched.
- `search.n_trials` (6), `search.time_budget_minutes` (none).
- `search.pruning.enabled` (true), `search.pruning.min_evals` (3).
- `search.parallel_trials` (1), `search.max_failed_ratio` (0.5), `search.min_trials_before_abort` (4).
- `search.finalists.k` (3), `search.finalist_repeats` (1; above 1 is refused).
- `search.screening.preset` (`tiny`).

**5. How it connects.** Before it: the `prepare` stage (data) and the TrainerCore (via the runner). After it: the `finalists` stage retrains the best trials on all Train data (chapter 16).

**6. What can go wrong.**
| Symptom | Exit code | Fix |
|---|---|---|
| `SearchConfigError`: unknown strategy; `parallel_trials` above the cores; a stored study with another space or seed | 2 | use the key named; start a new experiment |
| `SearchError`: no trial completed | 4 | read the failure reasons in `search_report.txt`; the reports are still written |
| Ctrl+C | 130 | rerun with `--experiment exp_NNNN`; finished trials are kept |
| A trial failed with `PermissionError ... Access is denied` when its model folder was renamed | (trial marked failed) | the retry from Task 1 is meant to prevent this (`train.rename_retries`); the failed trial is not retried by itself in an old experiment |

**7. A real example** (the rehearsal `exp_0002`, whisper-tiny, 4-point learning-rate grid, 20 steps per trial, screening data 61 Train clips and 101 Val clips): trial 0 (5e-6) Val loss 1.7741; trial 1 (1.08e-5) 1.6248; trial 2 (2.32e-5) 1.4618; trial 3 (5e-5) trained to the end but FAILED when Windows refused the rename of its model folder (O122), though it had the best Val loss of the search (1.29 according to the rehearsal summary). Each trial took about 195 s; the search took 832.8 s. The two best (trials 2 and 1) became the finalists. Because trial 3 was lost, the rehearsal winner had a smaller improvement than it might have had.

**8. Status.**
- Tested: `tests/test_searcher.py` (28 tests, fake runners, real worker processes) and `tests/test_orchestrator_search.py` (21 tests).
- Run for real: grid search with whisper-tiny (sequential); a real two-trial Optuna-free search on the real Daf Yomi data (Part 4).
- Never run: `optuna_tpe` on real data; `parallel_trials` above 1 with a real model (verified with a fake runner and real worker processes only; the speed-up is not measured); pruning firing in a real run (it needs four or more evaluations per trial; the rehearsal had two).

**9. Open issues.** O10 (grid or Optuna), O50, O51 (parallel trials), O79 (grid default), O86 to O88 (trial folders keep checkpoints; parallel stop limits), O122.

---

## 12. Selector (C9)

**1. Purpose.** Decides which finalist is the winner, or decides that the untrained baseline stays.

**2. Input and output.**
- In: a `SelectionRequest`: the baseline's Val result, the list of finalist `Candidate`s (trial id, parameters, model folder, Val result, Val loss, steps), a `Comparator`, and a callable that does the ONE-TIME Test evaluation.
- Out: a `SelectionResult`: verdict (`winner` or `baseline_retained`), the winner, a ranking and a comparison table, the Test status (`confirmed`, `not_confirmed`, `not_evaluated`), the Test numbers, the review list, warnings, and a plain-language reason.
- On disk (written by the Orchestrator, not by the Selector): `reports\selection.json`, `selection_table.csv`, `selection_reason.txt`, `test_once.json`, `test_result_baseline.json`, `test_result_trial_T.json`.
- Code: `src/stt/selector/`.

**3. What it does.**
1. Compares every finalist with the baseline on Val (chapter 10). A finalist is eligible only if the verdict reaches `select.min_verdict` (default `improved`: the lower end of the 90% one-sided interval is above zero).
2. Among the eligible, the lowest Val WER wins. A difference below `select.tie_epsilon` (0.001) is a tie, broken by the lower Val loss, then by the lower trial number. There is no randomness.
3. If nobody is eligible, the baseline stays. That is a result, not an error (exit code 0).
4. For a winner it asks for the Test evaluation once for the winner and once for the baseline. It never asks when the baseline stays or when `select.test_confirmation` is false. A second request for the same model fails (`TestOnceGuard`), even after a failed first try.
5. The Test confirms the winner only if the Test comparison is `improved` at the same level. If not, the winner stays the winner but is flagged `not_confirmed`, with the Test numbers beside it. If the Test evaluation itself fails, the winner stays and the status is `not_evaluated`.
6. The Test results are never used to choose among finalists.

**4. Config keys** (section `select`). `select.min_verdict` (`improved`; or `no_clear_difference`), `select.tie_epsilon` (0.001), `select.on_insufficient_groups` (`plain_difference`; or `reject`), `select.test_confirmation` (true).

**5. How it connects.** Before it: the finalists, scored on Val. After it: the ModelRegistry stores the verdict and the winner (chapter 13).

**6. What can go wrong.** `SelectorError`, exit code 4: no candidates, duplicate trial ids, a Val result that carries the Test split name, a different `eval_version`, different file sets. `TestAlreadyEvaluatedError` (also exit code 4): the Test was requested twice for one model.

**7. A real example** (rehearsal `selection_reason.txt`): "Trial 2 (learning_rate 2.32079e-05) is the winner: its Val WER is 1.0279 against 1.1318 for the baseline (improvement +0.1039 ... lower bound +0.0759, 7 groups). The Test confirms it: Test WER 1.0599 for the winner against 1.1728 for the baseline (improvement +0.1130 ... lower bound +0.0588, 8 groups)." The second finalist (trial 1, Val WER 1.0498) was also eligible but lost on WER. Only 7 Val groups and 8 Test groups: the rehearsal's confidence is weak by design.

**8. Status.**
- Tested: `tests/test_selector.py` (17 tests), `tests/test_orchestrator_run.py`.
- Run for real: the rehearsal (winner, Test confirmed), and the smoke runs.
- Never run for real: a `baseline_retained` outcome on real data; a winner the Test does NOT confirm; a failed Test evaluation.

**9. Open issues.** O53 (rules, built), O81 (Test confirmation level: same as Val), O43 (the owner's listening verdict is not used), O84 (a retained baseline: finalist folders are not deleted by the Orchestrator).

---

## 13. ModelRegistry (C10)

**1. Purpose.** Stores the chosen model with its facts as a numbered, never-changed entry, and keeps an index of all entries.

**2. Input and output.**
- In: a `RegisterRequest`: the selection result, a `ModelCard` (all facts about the model), the experiment id, the finalist folders to remove (the Orchestrator passes none), and the review text.
- Out: a `RegistryEntry` (id, kind `model` or `baseline_retained`, folder, card path, checksum file, number of Test looks).
- On disk, in the folder `registry.dir` (default `registry`, next to the project and OUTSIDE the experiments folder): `index.csv` (one row per entry, open in Excel) and one folder per entry: `model_0001\model\` (the winner's model folder), `model_card.json`, `MODEL_CARD.md` (the same in plain English), `review_list.txt` (clips to listen to), `SHA256SUMS`, and later `exports.json`. The whole registry folder can be copied as one piece.
- Code: `src/stt/registry/` (`registry.py`, `card.py`, `index.py`, `hashing.py`).

**3. What it does.**
1. Numbers the entry one above the highest existing number (`model_0001`, `model_0002`; a gap stays a gap). An entry is never overwritten or edited.
2. Registers in a crash-safe order: reserves the number with a folder `.tmp_model_NNNN`, MOVES the model folder into it (a rename on one drive; across drives a verified copy and then deletion of the source), writes the card, the markdown, the review list and the checksums, renames the folder into place, verifies the checksums in the final place, updates the index.
3. If any step fails, the model folder is put back and the temporary folder removed.
4. A baseline that stays is recorded too (card and index row, no model folder).
5. Counts how many registered entries were evaluated on the same Test set (the Test-set fingerprint is a SHA-256 over sorted lines `sample_id|group_id|hash of reference`), and writes "this Test set has been looked at N time(s)" in the card.
6. Other calls: `list_entries`, `get`, `best` (lowest Val WER among entries of the same eval version and Test set), `verify` (re-checks every file against `SHA256SUMS`), `rebuild_index` (the only repair, never silent).

**4. Config keys** (section `registry`). `registry.dir` (`registry`; must lie outside the experiments folder), `registry.keep_finalists` (false), `registry.cleanup` (true), `registry.transfer` (`move`; or `copy`), `registry.checksums` (true).

Be aware: `cleanup: true` and `transfer: move` are deleting defaults (decision O91). The Orchestrator never asks for a deletion of finalists, so `cleanup` does nothing in a normal run. But the winner's model folder IS MOVED out of `candidates\` into the registry. After registration the winner is at `registry\model_NNNN\model`, not in `candidates\`.

**5. How it connects.** Before it: the Selector's result and the Test results. After it: the ModelExporter (chapter 14) adds an export record; the report builder reads the cards for its history chart.

**6. What can go wrong.**
| Symptom | Exit code | Fix |
|---|---|---|
| `RegistryError`: card does not match the selection; winner folder missing or empty; registry inside the experiments folder; damaged or stale index | 4 | read the message; for the index run `python -m stt registry rebuild-index` |
| A leftover `.tmp_model_NNNN` folder (power loss in the middle) | 4 | look inside; move the `model` folder back; delete the temporary folder yourself |
| `registry verify` finds missing or changed files | 5 | the entry was changed after registration; restore from a copy |

**7. A real example** (rehearsal, `data\rehearsal_run\registry`, 149 MB in all): entry `model_0001`, experiment `exp_0002`, kind `model`, verdict `winner`, trial 2, learning rate 2.32e-5, Val WER 1.0279 against 1.1318, Test WER 1.0599 against 1.1728, Test status `confirmed`, `test_evaluations` 1, 15 files in `SHA256SUMS`. The model file is 151 MB (`model.safetensors`, whisper-tiny).

**8. Status.**
- Tested: `tests/test_registry.py` (24 tests, including crash rollback and a move across drives) and `tests/test_orchestrator_run.py`.
- Run for real: one registration (the rehearsal) plus smoke runs.
- Never run: a registration across two drives with a real model; a retained-baseline entry on real data; a real power loss; a registry with a sharded model.

**9. Open issues.** O84, O85 (leftover temporary folder: handled by hand), O91 (deleting defaults), O59 (Test-set counter, built).

---

## 14. ModelExporter (C11)

**1. Purpose.** Converts a registered model to a target format, checks the result, and records the export.

**2. Input and output.**
- In: an entry id of the registry, and the formats wanted (`export.formats`).
- Out: a list of `ExportRecord`s (entry, exporter, target folder, files, the verification result, status `ok` / `failed` / `unverified`, seconds).
- On disk: the entry's `exports.json` and, for exporters that write files, `exports\<exporter>\` in the entry. These are NOT part of `SHA256SUMS`, so an export never breaks `registry verify`.
- Code: `src/stt/modelexporter/` (`service.py`, `hf_safetensors.py`, `registry.py`, `compare.py`).

**3. What it does.**
1. Refuses an unknown format before touching anything (lists the known ones).
2. Refuses a `baseline_retained` entry when asked explicitly (nothing to export); the automatic step skips it.
3. Checks the entry's files against its checksums first; a mismatch fails before the export.
4. Runs the exporter. The only one in v1, `hf_safetensors`, does not copy anything: it points to the entry's own model folder, because that is already a Hugging Face safetensors folder.
5. Verifies (`export.verify`, true): the folder is a complete Whisper folder, it loads through the ModelLoader, the parameter count equals what the weights file declares in its header, the mel bins match `config.json`, and (when the base model folder exists on this machine) the architecture matches the base model's.
6. Appends the record to `exports.json` (temporary file and rename). A failed verification keeps the export, marks it `failed`, and raises an error after all formats are done.

**4. Config keys** (section `export`). `export.formats` (`hf_safetensors`), `export.verify` (true), `export.verify_samples` (5), `export.max_wer_diff` (0.02). The last two matter only for an exporter verified at the transcription level; none exists yet.

**5. How it connects.** Before it: a registered model. After it: nothing; the exported folder is what you hand to the target software. `python -m stt run` exports the registered winner automatically at the end of `select_and_register`; `python -m stt export` exports any entry.

**6. What can go wrong.** `ExportConfigError` exit 2 (unknown format, empty registry, unknown entry). `ExportError` or `ExportDependencyError` exit 5 (verification failed; a needed package is missing and the machine is offline; the target folder already has files). A failed export still leaves the registration in place. A rerun fails again while the latest record of that format is `failed`.

**7. A real example** (rehearsal `model_0001\exports.json`): exporter `hf_safetensors`, status `ok`, level `load`: "loads through the ModelLoader (37760640 parameters, 80 mel bins); the loaded parameter count equals the 37760640 the weights file declares; the mel bins match config.json; the architecture equals the base model's (whisper-tiny)", 12 files listed.

**8. Status.**
- Tested: `tests/test_modelexporter.py` (23 tests) and `tests/test_orchestrator_export.py` (17 tests).
- Run for real: the tiny-model exports of the rehearsal and smoke runs.
- Never run: any other format (faster-whisper, whisper.cpp, quantization: not built, the target is unknown); transcription-level verification; an export of the turbo model.

**9. Open issues.** O46 (target format unknown), O104 to O107 (decided), manager question about the target software.

---

## 15. RunManager (C12)

**1. Purpose.** Creates and manages the numbered experiment folders, where everything one command produced is kept.

**2. Input and output.**
- In: an `ExperimentSpec` (the settings, the parent experiment, the code version, library versions, creation time).
- Out: an `ExperimentHandle` with the paths, the status, stage markers, a lock, and a method to write report files.
- On disk, in `run.experiments_dir` (default `experiments`; the presets use `data\work_like\experiments`): `exp_0001\`, `exp_0002\`, ... Inside one experiment:
  - `config_full.yaml` (the complete settings that ran; read-only), `config_diff.yaml` (what differs from the defaults or from the parent)
  - `status.json` (status, reason, and for each stage `ok` / `skipped` / `failed` with a fingerprint), `run_info.json` (seed, library versions, code version, creation time)
  - `logs\run.log` (everything, DEBUG level)
  - `reports\` (all reports of the stages), `metrics\<trial>\metrics.csv` (live), `candidates\` (trial, finalist, curve and benchmark model folders), `search\` (study)
  - `run.lock` only while a run is active.
- Code: `src/stt/runmanager/` (`store.py`, `handle.py`, `lock.py`, `diffing.py`, `cleanup.py`, `naming.py`, `logsetup.py`).

**3. What it does.**
1. Numbers experiments with plain numbers, never dates in names. The next number is one above the highest existing; it is created with an exclusive `mkdir`, so two processes never get the same number.
2. Writes `config_full.yaml` read-only and the diff.
3. Opens an existing experiment for a resume. If the settings differ from the stored `config_full.yaml`, it FAILS and shows the first differing keys. Keys that are new in the code and equal to their default are ignored (so older experiments can still be resumed after a new section was added; O76).
4. Takes a lock file with a heartbeat (refreshed every 30 s). A second opener fails. A lock older than 10 minutes is "stale": it also fails, with the instruction to rename it by hand. It is never removed automatically.
5. Writes reports only inside `reports\`; a path that leaves it is refused.
6. Sets up one log file per experiment (DEBUG) and INFO-and-above on the console.
7. Cleanup: it only PROPOSES (failed experiments, finalist folders, checkpoint folders of completed experiments). `delete` removes exactly the folders it is given, after checking that they are inside the experiments folder, are folders, have no control file and no fresh lock. The registry is outside and cannot be reached. The Orchestrator's `cleanup` command never calls `delete`.

**4. Config keys** (section `run`). `run.experiments_dir` (`experiments`), `run.parent` (none), `run.id_digits` (4), `run.config_readonly` (true), `run.lock_heartbeat_seconds` (30), `run.lock_stale_minutes` (10), `run.log.console_level` (`INFO`), `run.log.file_level` (`DEBUG`), `run.prevent_sleep` (true; used by the Orchestrator).

**5. How it connects.** Before it: the settings (chapter 1). After it: the Orchestrator creates or opens an experiment for every command and writes through it.

**6. What can go wrong.** `RunManagerError`, exit code 2.
| Symptom | Fix |
|---|---|
| "Cannot resume ... the config differs from the stored config_full.yaml" | rerun with the same settings, or start a new experiment (`--parent exp_NNNN`) |
| lock held, or lock stale | wait for the other run; for a stale lock RENAME `run.lock` by hand (for example to `run.lock.stale`), never delete it |
| path too long (Windows) | a warning with the fix; keep the project at a short path such as `C:\stt` |

**7. A real example** (turbo baseline `exp_0002`, 2026-10-25, `data\work_like\experiments`): `status.json` shows `completed` with stages `prepare ok` and `evaluate ok`; `run_info.json` records seed 42 and the library versions (torch 2.14.1, transformers 5.18.0, numpy 2.5.3, pydantic 2.13.5, PyYAML 6.0.3, soundfile 0.14.0, scipy 1.18.1, pyarrow 25.0.1, accelerate 1.15.0, safetensors 0.8.0); 17 report files, 0.56 MB. Another real example is the rehearsal `data\rehearsal_run\experiments\exp_0002`: its `candidates\` folder is empty now because on 2026-10-24 the owner authorized a one-time deletion of model and checkpoint folders of completed experiments (DECISIONS 84, list in `deleted_candidates_20261024.txt`, 3,915.3 MB for that data folder). The consequence: the rehearsal's trial and finalist models cannot be re-scored or exported; the winner is safe in the registry.

**8. Status.**
- Tested: `tests/test_runmanager.py` (29 tests, including concurrent creation from threads and processes) and the cleanup tests.
- Run for real: every real command.
- Never run: a long-path failure on Windows; a lock on a network drive (the lock is a file, not a network lock); `delete` on real folders through the program (the deletion in DECISIONS 84 was done by hand-written commands, not by the program).

**9. Open issues.** O72 and O111 (cleanup only proposes), O76 (resume after a new section), O86 (disk), O69, O71.

---

## 16. Orchestrator and command line (C13)

**1. Purpose.** Connects all components and runs the whole pipeline, or any one stage, from one command with a clear exit code.

**2. Input and output.**
- In: a command (`python -m stt <command>`), `--config FILE`, repeatable `--set a.b=value`, and for some commands `--parent`, `--experiment`.
- Out: a numbered experiment folder (chapter 15), a log on the console and in `logs\run.log`, and ONE JSON summary on standard output at the end (experiment id and folder, status, exit code, the result of each stage, the key numbers, the paths).
- Code: `src/stt/orchestrator/` (`cli.py`, `orchestrator.py`, `stages.py`, `finalists.py`, `selection.py`, `benchmark.py`, `planning.py`, `curve.py`, `wiring.py`, `progress.py`, `stopflag.py`, `sleepguard.py`, ...). `wiring.py` is the ONLY place that names concrete classes (the composition root); the stages talk to the services through small interfaces.

**3. What it does.**

The commands:

| Command | What it does |
|---|---|
| `check` | Python version, packages, model and data folders of the config, free disk, memory, cores, code version. Lists ALL problems at once. |
| `prepare` | Stage `prepare`: load, split, normalize, filter (and subsample if on); writes the five data reports. |
| `train` | prepare + one training run (stage `train`). |
| `evaluate` | prepare + score the baseline (and a trained model of the same experiment) on Val. The Test split is refused. |
| `search` | prepare + the hyperparameter search on a screening subset. |
| `run` | the stages of `pipeline.stages`: `prepare`, `evaluate`, `search`, `finalists`, `select_and_register`; then the automatic export. |
| `learning-curve` | the same settings on growing Train sizes; Val WER per size; a plain reading. |
| `benchmark` | measures seconds per training step, per Val-loss example and per decoded clip; prints the PLANNED time of `run`. |
| `report` | rebuilds `reports\report.html` of an existing experiment from its stored files. Recomputes nothing. |
| `registry list / verify / rebuild-index` | read the registry (`rebuild-index` is the only one that writes). |
| `export [ENTRY]` | export a registered model (default: the latest entry). |
| `cleanup` | writes a PROPOSAL file of what could be deleted; deletes nothing. |
| `config-docs` | rewrites `docs\CONFIG_REFERENCE.md` from the code. |
| `smoke` (and `smoke --full`) | the whole chain on synthetic data and the tiny test model, to prove the mechanics. |

How a stage run works: each finished stage writes a marker with a fingerprint of its settings (and of the stages it depends on) into `status.json`. A rerun with `--experiment exp_NNNN` skips stages whose marker is still valid, and reruns a stage whose settings changed and every stage after it. The first failed stage stops the run; the later ones are listed as skipped.

What each stage of `run` does, in order:
1. `prepare`: DataSource, Splitter, TextNormalizer, SampleFilter, Subsampler.
2. `evaluate`: scores the untrained model (baseline) on Val; stores `eval_result_baseline.json`.
3. `search`: the trials (chapter 11) on the screening data.
4. `finalists`: for each of the best `search.finalists.k` trials, trains a fresh model on the FULL Train split with the trial's settings (the Val loss on the screening Val slice picks the best checkpoint), then scores it by WER on the full Val split.
5. `select_and_register`: Selector (chapter 12); the ONE-TIME Test look; registry (chapter 13); export (chapter 14).

Ctrl+C: the first press sets a stop flag; the training saves a checkpoint and stops after the current step; the status becomes `stopped`, exit code 130; the same command with `--experiment` resumes. The second press stops at once. A long scoring does not look at the flag, so the first Ctrl+C is honored after the scoring.

Time planning (`planning.py`, used by `benchmark` and by the `PLAN:` lines printed by `run`, `search` and `learning-curve`): training time = `max_steps x seconds per step + number of Val evaluations x screening Val clips x seconds per Val-loss example`. Baseline = Val clips x seconds per decoded clip. Search = `n_trials` x training. Finalists = `k` x training, plus `k` x Val clips x decode seconds. Test = 2 x Test clips x decode seconds (the winner and the baseline), counted but only spent if a winner is found. NOT counted: `prepare`, model loading, checkpoint saving, export.

Also: Windows sleep is prevented during commands (`run.prevent_sleep`); the progress lines `PROGRESS <phase>: i of N (P%), elapsed, remaining about` appear at most every `progress.interval_seconds` (30).

**4. Config keys.** Sections `pipeline` (`pipeline.stages`, `pipeline.skip_completed` true, `pipeline.eval_split` `val`, `pipeline.test_split` `test`), `progress` (`progress.interval_seconds` 30), `benchmark` (`benchmark.steps` 3), `learning_curve` (`learning_curve.presets`: `smoke`, `tiny`, `half`, `full`; `learning_curve.params_from`), `smoke` (smoke-run settings), and `run.prevent_sleep`. The command line options are listed above; `python -m stt <command> --help` shows an example for each.

**5. How it connects.** Before it: you, the config. After it: the experiment folder and the registry.

**6. What can go wrong** (the exit codes of the whole program):
| Code | Meaning |
|---|---|
| 0 | success (also "the baseline stays": that is a result, not an error) |
| 1 | an unexpected bug |
| 2 | wrong settings: config, model folder, locked or missing experiment, config differs on resume, the refused Test split, registry inside the experiments folder |
| 3 | data problem: DataSource, Splitter, SampleFilter, FeaturePipeline, Subsampler |
| 4 | training, evaluation, search, selection or registry failure (also a second look at the Test) |
| 5 | a registered model does not match its checksums (`registry verify`), or an export or its verification failed |
| 130 | stopped with Ctrl+C |

**7. A real example.** The turbo baseline: `python -m stt evaluate --config configs\work_like.yaml --set model.path=models/whisper-large-v3-turbo-ivrit --set model.dtype=float32 --set train.cpu_threads=4 --set eval.batch_size=2` made `exp_0002` (`data\work_like\experiments`) on 2026-10-25: prepare took 73 s (00:30:41 to 00:31:54), model loading about 17 s, scoring 1,681 s; exit code 0. The rehearsal `run` (whisper-tiny, `exp_0002` of `data\rehearsal_run`) planned 37 minutes and took about 38: baseline 4 min, search 14 min, finalists 14 min, Test look and registration 6 min.

**8. Status.**
- Tested: `tests/test_orchestrator*.py` (175 tests, mostly with fake stages and the tiny model), `tests/test_progress_display.py`, `tests/test_presets.py`, `tests/test_improvement_and_history.py`. The full suite last ran 813 tests, all OK (2026-10-24, before Task 2).
- Run for real: `prepare`, `evaluate`, `search`, `run` (rehearsal, whisper-tiny), `benchmark`, `report`, `registry`, `export`, `cleanup`, `config-docs`, `check`, `smoke --full`.
- Never run for real: `learning-curve` on real Daf Yomi data (synthetic only); a full `run` with all the data (`configs\home_full.yaml`); a complete `run` with the turbo model; `parallel_trials` above 1 with a real model; resuming a real interrupted `run` after a Ctrl+C of the finalists stage (the search resume was done by hand once in the rehearsal).

**9. Open issues.** O89 (new default of `pipeline.stages`), O92, O98 (scoring dominates planned time), O119 (benchmark and warmup above 3), O120, O121 (timing bug inflates the benchmark's seconds per step), O75 (baseline is evaluated again in every experiment), plus a limit from Task 1: after an interruption of a run made with `train.save_optimizer_state: false` the stage ends with a clear refusal instead of restarting by itself (`TrainingWiring.can_resume` still treats a complete checkpoint as resumable).

---

## 17. ReportBuilder (C14)

**1. Purpose.** Turns the stored results of a run into one self-contained HTML page that shows whether the model succeeded, where it failed, and what could be changed; it also writes a live progress page while training.

**2. Input and output.**
- In: `ReportData`, made of "views" of the files an experiment stored: `reports\selection.json`, `eval_result_baseline.json`, `finalists\*.val.json`, `test_result_*.json`, `search_trials.json`, `learning_curve.json`, `benchmark.json`, the metrics files, `run_info.json` and the registry card. Every input is optional.
- Out: one HTML text (`reports\report.html`, written by the `report` command through the RunManager), the list of sections that had data, and warnings. The live page is `metrics\<trial>\progress.html`.
- The page needs no network, no script, no font, no image file. Charts are inline SVG drawn by our own code; no plotting library is used.
- Code: `src/stt/reportbuilder/` (`builder.py`, `hints.py`, `correlation.py`, `svg.py`, `progress.py`, `improvement.py`).

**3. What it does.** The builder only formats. It never recomputes a WER, never changes a result, never decides, and never reads the clock; the same inputs give a byte-identical page.
1. A verdict line at the very top: `Baseline retained`; `Proven improvement` (only when the winner is `improved` on Val AND the one-time Test confirms it); `No proven improvement` in every other case. The word WINNER is used only for a proven improvement, otherwise it says "Best finalist (not proven better than the baseline)".
2. A plain summary: baseline and winner WER, the interval of the difference, the Test confirmation, and the improvement line against the target (`report.target_relative_wer_reduction`, 0.10 in the presets).
3. Sections, in this order: History (Val WER of each registered experiment); Finalists compared with the baseline; WER bars and a forest plot; Training curves; Distribution of per-file WER; WER by group (lesson); the Search; the Learning curve; Per-file errors against file attributes (rank correlation, always with the number of points, with a note that correlation is not cause); Diagnosis hints; the Listening list (worst and random clips, word-level differences, audio paths, Hebrew right-to-left); Run information.
4. A section without its input says "Not available in this run" and names the missing file. It is never an empty chart.
5. Diagnosis hints are rules with thresholds from Config: `overfitting`, `no_learning`, `unstable_training`, `gain_concentrated`, `short_clip_weakness`, `length_cap_or_empty`, `edge_of_range`, `failed_or_pruned`. Each hint shows its evidence and a caveat. They are hints, not proof.
6. The Test split appears only as the one-time confirmation of the winner; no per-clip Test result is shown.

**4. Config keys** (section `report`). `report.title` (none = experiment name), `report.attributes`, `report.min_points_for_correlation` (30), `report.max_points_per_chart` (2000, thinning is deterministic and stated), `report.groups_shown` (15), `report.histogram_bins` (20), `report.loss_noise_range` (0.01: a Val-loss chart that varies less gets the note "Differences this small are noise"), `report.absolute_audio_paths` (false), `report.target_relative_wer_reduction` (none; presets 0.10), `report.hints.enabled_rules` and the hint thresholds (`overfitting_last_k_evals` 3, `no_learning_min_rel_improvement` 0.01, `unstable_spike_factor` 2.0, `gain_top_group_fraction` 0.1, `gain_share` 0.5, `short_clip_quantile` 0.25, `short_clip_wer_ratio` 1.5, `length_cap_share` 0.05, `failed_share` 0.3), `report.progress.refresh_seconds` (10), `report.progress.every_n_records` (1).

**5. How it connects.** Before it: the files stored by the other stages. After it: you, in a browser. `run` does NOT build the report: run `python -m stt report --experiment exp_NNNN` afterwards.

**6. What can go wrong.** `ReportConfigError` exit 2 (an unknown hint rule name, a locked experiment, an unknown experiment); `ReportError` exit 4 (a report that cannot be built). A file that exists but cannot be read is left out and listed as an input problem; a registry that cannot be read gives the Test-set count `unknown`, never a false 0.

**7. A real example.** The rehearsal report `data\rehearsal_run\experiments\exp_0002\reports\report.html` (116 KB) shows the verdict (winner trial 2; Val WER 1.1318 to 1.0279; Test confirmed) and the target line "9.2% against the 10% target: not reached".

**8. Status.**
- Tested: `tests/test_reportbuilder.py` (55 tests), `tests/test_orchestrator_report.py` (21 tests), `tests/test_improvement_and_history.py` (19 tests).
- Run for real: the rehearsal report, and reports of real search-only experiments.
- Never run: a report of a full `run` on all data; a report with a learning curve from real data; a report where the Test does not confirm the winner or the baseline stays.

**9. Open issues.** O100 (the attribute list, the hint thresholds and a one-page print layout stay open), O117 (WER over steps: not built), O103.

---

## 18. Tools (T)

**1. Purpose.** Small standalone scripts that prepare everything the offline pipeline needs. Only these scripts may use the network (at home).

**2. Input and output.** All are run as `python tools\<name>.py ...` with the project's virtual environment. Each has a "How to run" header.

| Tool | What it does | Output |
|---|---|---|
| `fetch_model.py --repo ivrit-ai/whisper-large-v3-turbo --out models\whisper-large-v3-turbo-ivrit` | downloads only the safetensors weights and the small config and tokenizer files | the folder plus `fetch_info.json` |
| `fetch_data.py --repo <dataset> --dry-run`, then `--out data\daf_yomi_raw` | prints the download size; downloads the dataset | the folder plus `fetch_info.json` |
| `fetch_wheels.py --requirements requirements.txt --out wheels --lock requirements.lock [--python-version 3.12 --platform win_amd64]` | downloads all packages as binary wheels, runs a clean offline install test | `wheels\`, `requirements.lock` |
| `make_test_model.py --tokenizer-from models\whisper-tiny --out tests\assets\tiny_whisper` | builds the 3.9 MB random-weight test model with the real tokenizer | already stored with the tests |
| `inspect_audio.py --data data\daf_yomi_raw` | read-only report on a downloaded parquet dataset (schema, distributions, 50 decoded audio rows, 30 raw labels) | text on the screen |
| `make_manifest.py --root . --out MANIFEST.sha256`, and `--verify MANIFEST.sha256` | writes or verifies a list of SHA-256 hashes of the release files | `MANIFEST.sha256` |

**3. What it does (details).**
1. `fetch_model` and `fetch_data` are idempotent: a finished folder is reused, an unfinished one is rebuilt. They download public repositories anonymously.
2. `fetch_wheels` accepts binary wheels only. A package that exists only as source fails and the package is named.
3. `make_manifest --verify` exits 1 for a changed or missing file; a file not in the manifest is only listed as EXTRA. `python -m stt check` finds the manifest and prints the code version with 8 characters of its hash, but does not verify the files.

**4. Config keys.** None; the tools take command-line options.

**5. How it connects.** The tools run before the pipeline and outside it. The pipeline only reads their results (the model folder, the data folder, the wheels).

**6. What can go wrong.** Exit codes: 0 OK; 1 failure; `fetch_wheels` also 2 (a wheel is missing for the target) and 3 (the offline install test failed). The tools need the network; they download public repositories only, anonymously.

**7. A real example.** `fetch_model` brought `ivrit-ai/whisper-large-v3-turbo` on 2026-10-24: 11 files, 3,240,199,074 bytes (3,235,581,408 of them the single `model.safetensors`), stored as 3,090.1 MiB in `models\whisper-large-v3-turbo-ivrit`.

**8. Status.**
- Tested: `tests/test_tools_fetch.py`, `tests/test_fetch_wheels.py` (network faked), `tests/test_inspect_audio.py`, `tests/test_make_test_model.py`, `tests/test_make_manifest.py`.
- Run for real: all of them at home. `fetch_wheels` including the offline install test passed with Python 3.12.
- Never run: `fetch_wheels` for the closed machine's Python version (unknown); `make_manifest --verify` on the closed machine; `fetch_data` for the work dataset.

**9. Open issues.** O1 (Python version at work: any), O57, O58, O109 (`check` does not verify the manifest), O110 (editing `configs\work.yaml` shows as CHANGED: copy it to a new name).

---

## 19. A run from start to end

This chapter follows one `python -m stt run` from raw data to the final report. Times marked "rehearsal" are from the real rehearsal run (whisper-tiny, 2 threads, `configs\work_like.yaml` settings, 2026-10-07 to 08, `data\rehearsal_run\experiments\exp_0002`). Times marked "turbo" are from the turbo model on 2026-10-24 and 25 (`data\work_like\experiments\exp_0001`, the 8-clip probe, and `exp_0002`, the 101-clip baseline). Whatever was never timed is marked "not measured".

### The data flow as a picture

```
 raw parquet files (data\daf_yomi_raw)            model folder (models\...)
        |                                                 |
   [DataSource] -> samples (+ rejected.csv)               |
        |                                                 |
   [Splitter]  -> Train | Val | Test  (by lesson)         |
        |                                                 |
   [TextNormalizer] -> train_text, eval_text              |
        |                                                 |
   [SampleFilter] (report mode: removes nothing)          |
        |                                                 |
   [Subsampler] (off, or the work_like 10% cut)           |
        |  === the "prepare" stage ===                    |
        v                                                 v
   Val -------------------------> [Evaluator] <--- [ModelLoader] = baseline model
        |                              |
        |                       baseline Val WER  (stage "evaluate")
        |
   Train(screening, small) + Val slice
        |--> [HyperparamSearcher] -> trial 0..N-1 (each: FeaturePipeline -> TrainerCore)
        |          ranked by Val loss, best k = finalists      (stage "search")
        |
   Train(full) --> [TrainerCore] per finalist --> finalist models
        |           Val --> [Evaluator] --> finalist Val WER   (stage "finalists")
        |
   Val results ----> [Selector] -- winner? --yes--> Test split opened ONCE here
                          |                           (winner + baseline), compare
                          v
                    [ModelRegistry] -> registry\model_NNNN   (stage "select_and_register")
                          |
                    [ModelExporter] -> exports.json
                          |
   python -m stt report --> [ReportBuilder] -> reports\report.html
```

### The timeline

| # | Step and command | Component | What it reads | What appears (folder) | Time | Decision taken / failure that can stop it |
|---|---|---|---|---|---|---|
| 0 | At home, once: `python tools\fetch_model.py ...`, `fetch_data.py`, `fetch_wheels.py` | Tools | the network | `models\...`, `data\daf_yomi_raw`, `wheels\` | not measured (the turbo download was 3.24 GB) | Only step that uses the network. Exit 1 if it fails |
| 1 | `python -m stt check --config configs\work_like.yaml` | `check` | Python, packages, folders, disk, RAM | nothing | seconds | Exit 2 lists every problem at once. Nothing starts until it is clean |
| 2 | `python -m stt benchmark --config ...` (twice, other programs closed) | Orchestrator `benchmark` | a few Train and Val clips, the model | `exp_NNNN\reports\benchmark.txt/json`, `candidates\exp_NNNN_benchmark\` | tiny rehearsal: 3 steps plus a probe (minutes); turbo: 3 steps plus the Val-loss probe took 9 min 4 s (batch 2) and 5 min 46 s (batch 1), then the 12-clip decoding probe about 5 min, saving 26 s; the benchmark kept a 7.4 GiB folder (2026-10-25) | Prints the PLANNED time. Decision for you: is the run affordable? Exit 2 if warmup is above `benchmark.steps` (O119) |
| 3 | `python -m stt run --config configs\work_like.yaml` | Orchestrator | the config | a new `exp_NNNN\` with `config_full.yaml`, `config_diff.yaml`, `run_info.json`, `status.json`, `logs\run.log`, `run.lock` | seconds | Exit 2 if the config is wrong, the registry folder lies inside the experiments folder, or the experiment is locked. The `PLAN:` lines are printed after `prepare` |
| 4 | Stage `prepare` | DataSource, Splitter, TextNormalizer, SampleFilter, Subsampler | the parquet or the extracted manifest and audio headers | `reports\load_report.txt`, `split_report.txt`, `text_report.txt`, `filter_report.txt`, `subsample_report.txt`, `rejected.csv`; first run also `data\extracted\daf_yomi` | rehearsal 10 s; turbo runs 22 s and 73 s (first extraction about 3 min, once) | Data decisions: rejected rows (gate 5%), split ratios (gate 5 points), filter gate (40%). Exit 3 on any failure. NO model is loaded here |
| 5 | Stage `evaluate` (the baseline) | ModelLoader, FeaturePipeline, Evaluator | Val clips, the untrained model | `reports\eval_result_baseline.json`, `evaluation_baseline.json`, `predictions_baseline.csv`, `review_baseline.txt` | rehearsal tiny: 3 min 48 s for 101 clips (about 2.3 s per clip); turbo: model load 17 s to 110 s (first load from disk), then 16.6 s per clip: 101 clips took 1,681 s (28 min), the 8-clip probe 155 s | The reference point of the whole run. Exit 4 if every output is empty. The Test split is REFUSED here (exit 2) |
| 6 | Stage `search` | HyperparamSearcher, FeaturePipeline, TrainerCore | the screening Train (preset `tiny`) and the Val slice | `search\study.json`, `candidates\exp_NNNN_trial_TTT\`, `metrics\trial_TTT\metrics.csv`, `reports\search_report.txt`, `search_trials.json`, `search_screening.txt` | rehearsal: 4 trials of 20 steps, about 195 s each, 14 min in all; turbo: not run (planned 7 h 50 min for 4 trials of 20 steps with batch 2: see below) | Decision: the best `finalists.k` parameter sets. Exit 4 if no trial completed (the reports are still written). Ctrl+C: exit 130, resume with `--experiment` |
| 7 | Stage `finalists` | TrainerCore, Evaluator | the FULL Train split (702 clips in the rehearsal) and the Val slice for the loss, then all of Val for WER | `candidates\exp_NNNN_final_trial_TTT\`, `metrics\final_trial_TTT\`, `reports\finalists\*.json`, `finalists_report.txt` | rehearsal: 2 finalists, 14 min 17 s in all (about 3 min 15 s training and 3 min scoring each); turbo: not measured | Each finalist gets a Val WER. Exit 2 if `search.finalist_repeats` is above 1 |
| 8 | Stage `select_and_register`, part 1: selection | Selector, Comparator | baseline and finalist Val results | `reports\selection.json`, `selection_table.csv`, `selection_reason.txt` | seconds | THE DECISION: a winner that beats the baseline at the 90% one-sided level on Val, or "the baseline stays" (exit 0) |
| 9 | Part 2: the Test split is opened | Orchestrator (`TestOnceGuard`), Evaluator | the Test clips; the winner's model and the baseline | `reports\test_once.json` (written BEFORE the look), `test_result_baseline.json`, `test_result_trial_T.json` | rehearsal: 97 clips twice, 3 min 9 s each; turbo: not measured (about 17 s per clip, so about 55 min for 2 x 97 clips) | **THE ONLY PLACE THE TEST SPLIT IS EVALUATED.** Once per model, ever; a rerun reads the stored result. Never when the baseline stays. Exit 4 on a second look |
| 10 | Part 3: registration | ModelRegistry | the winner's model folder, the card data | `registry\model_NNNN\...`, `index.csv`; the winner's folder MOVES out of `candidates\` | rehearsal: seconds | Exit 4 if the registry is inside the experiments folder or a step fails (everything is rolled back) |
| 11 | Part 4: automatic export | ModelExporter | the registered model | `registry\model_NNNN\exports.json` | seconds | Exit 5 if the export or its check fails; the registration stays |
| 12 | End of `run`: the console prints `VERDICT:`, the Test status, the improvement line, the registry entry, and the JSON summary | Orchestrator | stored results | status `completed` in `status.json`; the lock is removed | rehearsal total 38 min (planned 37) | Exit code 0, or the code of the first failed stage |
| 13 | `python -m stt report --experiment exp_NNNN` | ReportBuilder | the stored files | `reports\report.html` (116 KB in the rehearsal) | seconds | Nothing is recomputed. Exit 2 for a locked or unknown experiment |
| 14 | You: open `report.html`, read the verdict line, listen to the listening list | you | the report, the audio paths | nothing | your time | The real decision: is the model good enough in practice (the pipeline's own test is statistical; your ears are the second lens) |
| 15 | Optional: `python -m stt registry verify`, `python -m stt export`, `python -m stt cleanup` | Registry, Exporter, RunManager | the registry, the experiments | `cleanup_proposal_<time>.txt` (a list only) | seconds | `registry verify` exit 5 on a mismatch. Nothing is deleted by any command |

Where the run can stop: any step that raises an error stops the run at once (nothing is retried), the stage is marked `failed`, the later stages are listed as skipped, and the experiment folder is kept. Fix the cause, then rerun the same command with `--experiment exp_NNNN`: finished stages are skipped; a stage whose settings changed (and every later one) is run again; an interrupted training resumes from its checkpoint.

### Where the time goes (read this before a long run)

- Training time is steps times seconds per step. Scoring time is clips times seconds per decoded clip. In the rehearsal with whisper-tiny, scoring (baseline 4 min, finalists 6 min, Test 6 min) was about 16 of the 38 minutes.
- With the turbo model on this laptop, the measured decode speed (about 16.6 s per clip) means: baseline on 101 Val clips 28 min (measured), two finalists on 101 Val clips about 56 min (estimated, 2 x 28), the Test look (2 x 97 clips) about 54 min (estimated), together about 2 h 18 min of scoring alone, before any training time. The measured turbo training numbers are in the next section.

### Turbo training measurements (Task 2, 2026-10-25, `freeze_encoder`, 4 threads, float32, this laptop)

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

### What a full `run` costs on this laptop (turbo, `freeze_encoder`; every number is marked M = measured, E = estimated)

Common numbers: step 106 s (M, as printed by the benchmark, the pessimistic reading; the corrected 84 s would allow about 25% more steps), Val loss 24.4 s per example on 101 clips (M), decoding 16.6 s per clip (M, 101 clips), 2 Val-loss evaluations per training (E, `eval_every_steps` = half of `max_steps`), 3 minutes per training for model loading and saving (E), `prepare` 1.5 minutes (M). One training of S steps therefore costs 106 S + 4,929 + 180 seconds. Baseline on Val: 28 min (M). Winner's Val WER: 28 min (E, same speed). Test look, winner and baseline: 2 x 97 clips = 54 min (E).

The system cannot run "without a search": the stage `finalists` reads the result of the `search` stage (`finalists.py`, exit code 2 if `search_trials.json` is missing). The smallest `run` is `search.n_trials=1` and `search.finalists.k=1`, which is TWO trainings (the trial on the 61-clip screening set, then the finalist on the full Train split, both with the same `max_steps`). The `train` command followed by `evaluate` is ONE training, but it has no search, no selection, no Test look and no registry.

| Plan | Stage times (E unless marked M) | Largest `max_steps` in 8 h | Largest `max_steps` in 4 h |
|---|---|---|---|
| (a) `run` with `n_trials=2`, `finalists.k=1` (three trainings) | prepare 1.5 min; baseline 28 min (M); search 2 x (S x 106 s + 85 min); finalist S x 106 s + 85 min; finalist Val WER 28 min; Test 54 min. Total = 21,991 s + 318 s x S | **21** steps (S=21: baseline 28 min, search 4 h 04 min, finalist 2 h 02 min, Val WER 28 min, Test 54 min; total 7 h 58 min) | **none**: the fixed part alone (baseline, six Val-loss evaluations of 41 min each, Val WER, Test) is 6 h 06 min; it does not fit even at 0 steps |
| (b) the smallest `run`: `n_trials=1`, `finalists.k=1` (two trainings) | same, with one trial. Total = 16,882 s + 212 s x S | **56** steps (search 3 h 04 min, finalist 3 h 04 min, plus 28 + 28 + 54 min; 7 h 58 min) | **none**: 4 h 41 min at 0 steps |
| (c) one training only: `train` then `evaluate` (no search, no selection, no Test) | prepare 1.5 min; the training S x 106 s + 85 min; `evaluate` scores the baseline and the trained model: 2 x 28 min. Total = 8,553 s + 106 s x S | **191** steps (training 7 h 02 min + 56 min) | **55** steps (training 3 h 02 min + 56 min) |

Notes on the table: the 8 h and 4 h columns use the 106 s step; with the corrected 84 s step the limits are about 27 (a), 70 (b), 240 (c, 8 h) and 69 (c, 4 h) steps. If you use a Val slice of 30 clips for the Val loss (`subsample.val_samples=30`, but not together with the `work_like` preset) each Val-loss evaluation shrinks from 41 min to 12 min, which would let plan (a) fit about 54 steps in 8 h (E). Early stopping and pruning only shorten a run. Disk: every training leaves a 7.4 to 7.9 GiB folder, so (a) needs about 24 GB, (b) about 16 GB and (c) about 8 GB; the laptop has 7.7 GB free after the two benchmark folders (15 GB before the second one), so (a) and (b) do not fit without deleting something. Weights-only checkpoints (`train.save_optimizer_state: false`) save little with a frozen encoder: the optimizer state of the decoder is only about 1.5 GB of the 4.6 GB checkpoint (E). Memory: a training peaks at 6.4 to 6.7 GB; the laptop has 15.77 GB but only 2 to 4 GB are usually free, so other programs must be closed. The numbers say nothing about quality: 20 to 190 steps on this data are a rehearsal.

### The same budget with fewer Val-loss evaluations (Task 4, estimated from the measured numbers above)

How the Val loss is scheduled (checked in `hf_trainer.py`, `planning.py`, `wiring.py` and the installed Hugging Face `trainer_callback.py`): the Val loss is computed on the Val slice of the screening set, in the search AND in the finalists stage (decision O93): all clips of the prepared Val split (101 in the rehearsal), or exactly `subsample.val_samples` clips when that key is set (not allowed together with a scope `all_splits` preset such as `work_like`). It is computed every `train.eval_every_steps` steps, plus once at the end when `max_steps` is not a multiple of it, so a training makes `ceil(max_steps / eval_every_steps)` evaluations. With `eval_every_steps` equal to `max_steps` there is exactly ONE evaluation, at the last step (Hugging Face does not add a second one, because the last step is already a multiple). The first evaluation always counts as an improvement, so it becomes the best checkpoint and the returned model; early stopping needs `patience` (5) evaluations without improvement, and pruning needs at least 3 evaluations, so neither can happen. The WER of the finalists and the baseline is separate and always uses the whole prepared Val split.

Assumptions as before: step 106 s (measured, as printed), Val loss 24.4 s per example (measured), decoding 16.6 s per clip (measured), 3 minutes per training for loading and saving (estimated), fixed part 6,664 s (baseline 28 min, finalist Val WER 28 min, Test look 54 min, prepare 1.5 min). Largest `max_steps` that fits:

| Plan | Two evaluations of 101 clips (the table above) | (i) ONE evaluation at the end, 101 clips | (ii) a Val slice of 30 clips, two evaluations | (i) + (ii): one evaluation of 30 clips |
|---|---|---|---|---|
| (a) `n_trials=2`, `finalists.k=1` (3 trainings): 8 h / 4 h | 21 / none | 44 / none (4 h 03 min at 0 steps) | 54 / 8 | 61 / 15 |
| (b) `n_trials=1`, `finalists.k=1` (2 trainings): 8 h / 4 h | 56 / none | 79 / 11 | 88 / 20 | 95 / 27 |
| (c) `train` + `evaluate` (1 training, no Test): 8 h / 4 h | 191 / 55 | 214 / 78 | 223 / 87 | 230 / 94 |

Formulas (seconds): a training of S steps costs `106 S + evaluations x clips x 24.4 + 180`; plan (a) adds the fixed part and three trainings, plan (b) two, plan (c) uses `90 + training + 3,354`. With the corrected step of about 84 s (estimated) every limit rises by about a quarter. Caution on (ii): with the `work_like` preset a 30-clip Val slice cannot be set (`subsample.val_samples` needs the scope `train_only`); it works only with a preset or settings like `home_full`, where the WER Val is then the whole Val split (875 clips, 4 h scoring at 16.6 s per clip, estimated), so (ii) is not a free saving. Only (i) needs no new setting: `--set train.eval_every_steps=<max_steps>`.

---

## 20. Labels shorter than the audio

Some clips have a label that ends well before the audio ends. Example: `train_006586`, last labeled sentence ends at 21.88 s of 30 s. The model then writes the real speech that the label does not contain, and every such word counts as an insertion. Everything below uses Train and Val only. The Test split was never read for this analysis (the Test samples were dropped from memory at once). The numbers were computed on 2026-10-25 from the stored labels (read-only), with the rehearsal Train (702 clips) and Val (101 clips) re-created by the same code as the `prepare` stage (the Val ids equal the ids of the stored turbo run). Definition used throughout: the "label end" is the end time of the LAST labeled segment that has text (a lone start marker at the very end, which every label has, is ignored). The "gap" is the clip length minus the label end.

### 1. How many clips have a gap of more than 3 s

Rehearsal (the data the rehearsal runs saw):

| Gap | Train clips | Train share of hours | Val clips | Val share of hours |
|---|---|---|---|---|
| 3 s or less | 475 (67.7%) | 67.8% | 71 (70.3%) | 70.5% |
| more than 3 to 6 s | 156 (22.2%) | 22.2% | 17 (16.8%) | 16.9% |
| more than 6 to 10 s | 38 (5.4%) | 5.4% | 9 (8.9%) | 8.9% |
| more than 10 s | 33 (4.7%) | 4.6% | 4 (4.0%) | 3.6% |
| **more than 3 s, all** | **227 (32.3%), 1.88 h of 5.84 h** | **32.2%** | **30 (29.7%), 0.25 h of 0.84 h** | **29.5%** |

("Share of hours" = the hours of those clips divided by all hours of the split.) The audio that has no label is small in total: in the clips with a gap above 3 s it is 0.41 h of the 5.84 h in rehearsal Train (7.0%) and about 0.06 h of the 0.84 h in Val (7%).

For context, the same table on all the data: Train 2,342 of 6,973 clips (33.6%), 19.35 h of 57.90 h; Val 280 of 875 clips (32.0%). So the rehearsal is representative. All 8,723 labels have usable timestamps, so no clip is left out of this count.

### 2. Is there speech in the trailing audio

Method: the loudness of every clip in 25 ms frames. A frame is "speech-like" if it is clearly above the quiet floor of the clip (floor = the 5th percentile, plus 12 dB, and not more than 30 dB below the loud parts); runs shorter than 0.3 s are ignored. The first 0.7 s after the label end are skipped (timestamps are not exact). This is a simple energy check: it cannot tell speech from music or noise, and nobody listened to the clips.

- **Val (101 clips, with the turbo model's own output):** of the 30 clips with a gap above 3 s, all 30 have 1.0 s or more of speech-like sound after the label end (on average 3.4 s of 4.4 s for gaps of 3 to 10 s, and 12.6 s of 16.0 s for gaps above 10 s). Of the 71 clips with a gap of 3 s or less, 19 (27%) have it, but their trailing part is short (0.65 s on average).
- **The model's insertions:** the 30 gap clips hold 342 of the 744 insertions of the turbo baseline (46%) on 24.5% of the reference words: 0.30 inserted words per reference word, against 0.11 for the other 71 clips. With the "label end before 25 s" split (Task 2, Part B): 16 clips, 256 insertions on 513 reference words (0.50 per word), against 85 clips, 488 insertions on 4,184 words (0.12 per word).
- **But the extra words are NOT mostly at the end of the text:** only 49 of the 744 insertions fall after the last reference word that the alignment matched or replaced. The alignment puts most inserted words inside the text, because the model's sentences also differ in the labeled part. The rank correlation between the seconds of trailing speech-like sound and the number of trailing insertions is only 0.16, and with all insertions of the clip 0.40. So the evidence is: clips with a gap are clearly worse on insertions, and the audio after the label does contain sound; but this analysis does not prove that the missing label is the cause, and it does not say how many of the 744 insertions it explains. A listening check of a few of the 30 clips (the review list of `exp_0002` has the paths) would settle it.
- **Train (rehearsal; no model output exists for Train clips):** a random sample of 40 of the 227 gap clips: all 40 have 1.0 s or more of speech-like sound after the label end.

### 3. Where the timestamps come from, who removes them, who still has them

- **Origin.** In the Daf Yomi dataset the `transcript` column is already written with Whisper-style markers: `<|0.08|>text<|1.84|><|2.34|>text ...` (all 8,723 labels have them; `has_timestamps` is true for every row). A marker pair is start and end of a segment. The last marker is the start of a segment that was cut off by the end of the 30 s window and has no text, so the label end is the end marker before it.
- **DataSource (chapter 2)** passes the text unchanged (`AudioSample.text`).
- **TextNormalizer (chapter 4)** parses the markers first (`timestamp_segments`: start, end, text per segment) and then the step `strip_timestamps` removes them from BOTH `train_text` and `eval_text`.
- **Who still has the last timestamp:** the `NormalizedSample` that the TextNormalizer returns still carries `timestamp_segments` and, through `sample`, the raw text with the markers. These objects travel through the SampleFilter and the Subsampler, and the Orchestrator's `prepare` stage keeps them (this is how the analysis read them). But no component downstream uses them: the FeaturePipeline reads only the audio and `train_text`; the Evaluator receives only `eval_text` and a fixed list of attributes (`duration`, `ref_words`, `label_tokens`, `extra_...`, `flag_...`). A search of the source shows that `timestamp_segments` is used only inside the TextNormalizer.
- **Is a timestamp guaranteed in the generic input format?** No. The generic `csv` / `hf_parquet` input has only an audio path and a text. The marker pattern is a config key (`text.timestamp_pattern`) and the markers are optional: a dataset without them gives `timestamp_segments = None`, and malformed markers also give `None`. The decision O48 says there is no need to deal with timestamps. So a rule based on the label end works only for data that carries the markers, as the Daf Yomi data does.

### 4. The options

| Option | Component that changes | Effect on Train | Effect on Val (and Test) | Comparability with earlier numbers (tiny 1.1318 on 101 Val clips; turbo 0.7974 on 8 clips and 0.8335 on 101 clips) | What must be measured again | Risks |
|---|---|---|---|---|---|
| (a) Cut the audio at the label end | FeaturePipeline / audio adapter. It needs the label end per sample: a new field on the sample objects, so an interface change, and a new config key | the model sees silence where it used to see unlabeled speech, so it stops being taught to write what is unlabeled | if applied to Val and Test too, the model is scored on shortened audio and those insertions disappear; if applied to Train only, Val is unchanged | NOT comparable when Val is cut: the input changes and the baseline must be scored again on the cut clips. Comparable if Train only | with Val cut: the baseline (28 min for 101 clips with the turbo model), every claimed improvement, the Test of any registered model. With Train only: the trained models | Timestamps are not exact: a cut can remove real labeled words. At work the real audio will not be cut, so a model scored on cut Val clips looks better than it is in use. Needs the markers in the data |
| (b) Drop such clips | SampleFilter: a new rule (for example "gap above X s"; needs a new field and a new key) | fewer Train clips: rehearsal Train would lose 227 of 702 (32%) and 1.88 h | if applied to Val the set shrinks by about 30 of 101 clips and the lessons change (7 lessons in rehearsal); the Test the same | NOT comparable: another Val set, and new Val and Test fingerprints, so the registry counts them as other Test sets | everything that used Val: baseline, search, finalists | loses a third of scarce data; keeps only the easier clips; at work the data is about one tenth of Daf Yomi, so every clip counts; the confidence of a small Val gets even weaker |
| (c) Only flag them in the report | Orchestrator (add an attribute such as the label gap to the attributes it builds for each Val clip) and the existing config list `report.attributes` | none | none; the report shows the correlation of the per-file WER with the gap, and the listening list can show it | fully comparable: no number changes | nothing | A correlation is not a cause; it needs the markers in the data; it is a small change of one component |
| (d) Do nothing | none | none | none | fully comparable | nothing | the WER keeps an unknown share of "insertions" that are real speech. Two models are still compared fairly (both are scored on the same clips), but an absolute WER is pessimistic |

### 5. Recommendation

Do (c) first, and decide about (a) or (b) only after listening. The comparison between a baseline and a trained model is fair today, because both are scored on the same clips with the same labels. What is uncertain is the size of the absolute WER and, possibly, a training signal that teaches the model to leave out real speech. Option (c) changes no number and touches one component. With the gap in the report you can see how much of the WER sits in the gap clips, and you can listen to the 30 Val clips. If listening confirms that the trailing audio is real speech that the labels miss, the cleanest fix is better labels at the source. If you must act in the pipeline, (a) applied to Train only is the least risky choice: Val, Test and all earlier numbers stay comparable (a new training is needed). Option (b) shrinks a small dataset, and (a) on Val changes the yardstick. This is a recommendation only; nothing was implemented.

---

## 21. What you get and what each experiment tells you

Every command that makes an experiment writes into one numbered folder `<run.experiments_dir>\exp_NNNN\` (chapter 15). Five things are always there: `config_full.yaml` (the exact settings), `config_diff.yaml` (what differs from the defaults), `run_info.json` (seed, library versions), `status.json` (`completed` / `failed` / `stopped`, and each stage) and `logs\run.log` (everything, with times). If a command fails, open `status.json` and the end of `logs\run.log` first. In the tables below "measured" means a number read from a stored report or log of this laptop, "estimated" means computed from measured numbers, "never run" means the thing was never done for real.

### 21.1 What each command leaves behind

| Command | Creates (under the experiment folder unless noted) | What each tells | Open first |
|---|---|---|---|
| `check` | nothing | prints OK lines and `PROBLEM` lines (Python, packages, folders, disk, RAM, cores) | the screen |
| `prepare` | `reports\load_report.txt`, `rejected.csv`, `split_report.txt`, `text_report.txt`, `filter_report.txt`, `subsample_report.txt` (only when the Subsampler is on) | what was read and how long the clips are; which rows were not used and why; clips and lessons per split and the no-overlap check; the unusual characters and the markers found; what the filter WOULD remove; the size of the cut for a rehearsal | `load_report.txt`, then `rejected.csv` (it should hold only its header line), then `split_report.txt` |
| `benchmark` | `reports\benchmark.txt` and `benchmark.json`, `metrics\benchmark\metrics.csv`, `candidates\exp_NNNN_benchmark\` (a real model folder, 7.4 GiB with the turbo model, kept) | seconds per training step, per Val-loss example and per decoded clip; the planned time and disk of a whole `run`; the peak memory | `benchmark.txt` |
| `evaluate` | `reports\eval_result_baseline.json` (lossless result), `evaluation_baseline.json` (the numbers), `predictions_baseline.csv` (one row per clip: reference, model text, WER), `review_baseline.txt` (worst and random clips with word-level differences and audio paths); with a trained model in the same experiment also the `_trained` files and `comparison.json` | the WER and CER of the untrained model; which clips fail; what to listen to | `evaluation_baseline.json`, then `review_baseline.txt` |
| `train` | `candidates\exp_NNNN_trial_000\` (checkpoints and the model folder `exp_NNNN_trial_000_step_SSSSS`), `metrics\trial_000\metrics.csv` | the training and Val loss curve of one training; the trained model | `metrics\trial_000\metrics.csv` (open in Excel) |
| `search` | `search\study.json`, `search\trial_TTT.params.json`, `reports\search_report.txt`, `search_trials.json`, `search_screening.txt`, `candidates\..._trial_TTT\`, `metrics\trial_TTT\metrics.csv` | one row per trial: settings, Val loss, status (complete / pruned / failed); which settings became finalists; the size of the screening data | `search_report.txt` |
| `run` | everything of `prepare`, `evaluate` and `search`, plus `reports\finalists_report.txt`, `finalists\final_trial_TTT.json` and `.val.json`, predictions and review files per finalist, `selection.json`, `selection_table.csv`, `selection_reason.txt`, `test_once.json`, `test_result_baseline.json`, `test_result_trial_T.json`; and outside the experiment `registry\model_NNNN\` (model, `MODEL_CARD.md`, `model_card.json`, `review_list.txt`, `SHA256SUMS`, `exports.json`) | the Val WER of each finalist; the verdict in words; the one-time Test numbers; the registered model with all its facts | the `VERDICT:` lines on the screen, then `selection_reason.txt`, then the report (next row) |
| `report` | `reports\report.html` | the verdict line, the target line, all charts and the listening list | `report.html` (open in a browser) |
| `learning-curve` | `reports\learning_curve.txt`, `learning_curve.csv`, `learning_curve.json`, `reports\curve\<preset>.json` and `.val.json`, `predictions_<preset>.csv`, `candidates\exp_NNNN_curve_<preset>\`, `metrics\curve_<preset>\` | Val WER for each Train size and a reading: STILL IMPROVING / FLAT / GOT WORSE | `learning_curve.txt` |
| `export` | `registry\model_NNNN\exports.json` (and `exports\<exporter>\` for exporters that copy); one JSON on the screen | whether the export and its check are `ok` | the JSON on the screen |
| `registry list` / `verify` / `rebuild-index` | `list` and `verify` create nothing; `rebuild-index` rewrites `registry\index.csv` | `list`: one row per registered model with its WERs and number of exports; `verify`: every file against its checksum (exit 5 on a mismatch) | `registry list` |

### 21.2 The experiments you actually run

Turbo times use the numbers measured on this laptop: decoding 16.6 s per clip (measured over 101 clips), a training step 106 s as printed by the benchmark (about 84 s after the O121 correction, estimated), 24.4 s per Val-loss example (measured), 4 threads, frozen encoder. Everything marked "estimated" below was computed from those and was never run as such. Nothing in this table has been done for real with the turbo model except the first two rows.

| Experiment | The question it answers | Input | Output | Time on this CPU, turbo | A good result looks like | A bad result looks like | The decision it supports |
|---|---|---|---|---|---|---|---|
| Baseline evaluation (`evaluate`) | How good is the model before any training, on my Val clips? | the Val clips (101 in the rehearsal), the untrained model | `evaluation_baseline.json`, predictions, review list | about 30 min for 101 clips: prepare 73 s, model load 17 s, decoding 1,681 s (measured, `exp_0002`) | a WER well below 1.0 and few clips that hit the length cap: turbo gave WER 0.8335, CER 0.3889, 1 of 101 at the cap (measured) | WER above 1.0, empty outputs, many clips at the cap (whisper-tiny: 17 of 101); an error "Every hypothesis is empty" | whether training is worth trying at all; the reference for every later comparison |
| Benchmark (`benchmark`) | How long and how much memory and disk will a run need on THIS machine? | 12 Train clips and 12 Val clips, the model | `benchmark.txt` with the planned time, disk, memory | 16 to 20 min per run (measured, two runs on 2026-10-25) | a planned total that fits your time and disk; a peak memory below the free RAM | a plan of days (turbo: 10 h 24 min to 14 h 49 min, measured plan), a disk need above the free disk (47 GB planned, 7.7 GB free then), a memory peak (6.4 to 6.7 GB) near the free RAM | whether to run at all, with which model, batch, steps, trials |
| Search trial (stage `search`) | Which learning rate (or other setting) trains best? | a small screening set of Train (preset `tiny`, 61 clips in the rehearsal), a Val slice | one row in `search_report.txt`, a Val loss, a model folder | 20 steps and 2 Val-loss evaluations of 101 clips: about 1 h 58 min per trial (planner, estimated); never run with the turbo model. Whisper-tiny: about 195 s per trial (measured) | the trials differ clearly in Val loss and the best is not at the edge of the range (rehearsal: 1.77, 1.62, 1.46) | all trials within noise (the report says "Differences this small are noise"), failed or pruned trials, the best trial at the edge of the range | which settings become finalists |
| Finalist (stage `finalists`) | Does a full training with those settings beat the baseline on Val WER? | the FULL Train split, the Val slice for the loss, then all of Val for WER | `finalists_report.txt`, `final_trial_TTT.val.json`, a model folder | about 2 h 26 min per finalist for 20 steps (estimated: training 1 h 58 min plus scoring 28 min); never run with turbo. Whisper-tiny: 7 min per finalist (measured) | a Val WER clearly below the baseline (rehearsal: 1.1318 down to 1.0279) | a WER equal to or above the baseline | which finalist is the winner, if any |
| Test look (inside `select_and_register`) | Is the winner really better on clips nobody used to choose it? | the winner and the baseline, the Test clips (97 in the rehearsal) | `test_result_*.json`, the Test status in the verdict | 2 x 97 clips x 16.6 s = about 54 min (estimated); never run with turbo. Whisper-tiny: 3 min 9 s per model (measured) | status `confirmed`: the Test comparison is `improved` too | `not_confirmed` (the winner stays but is flagged), or the Test is not evaluated because the baseline stayed | whether to trust and register the winner. It happens ONCE per model; never use it to choose |
| Train + evaluate (`train`, then `evaluate`) | Does one training help at all, before spending hours on a search? | the full Train split, the Val set | a trained model, `metrics.csv`, `evaluation_baseline.json` and `evaluation_trained.json` with `comparison.json` | for 20 steps about 3 h (estimated: training 2 h, scoring the baseline and the trained model 2 x 28 min); never run with turbo | the Val loss falls and the trained WER is lower than the baseline's, with a clear comparison | the Val loss rises or stays flat; the loss becomes NaN (exit 4); WER not better | whether to invest in a search and finalists. It has no search, no selection and no Test look |
| Learning curve (`learning-curve`) | Would more data help? | the same settings on growing Train sizes (smoke, tiny, half, full), one fixed Val set | `learning_curve.txt` with a reading | per point a training plus a Val scoring (for 20 steps about 2 h 28 min with 101 Val clips, estimated), so about 10 h for four points; never run with turbo and never on real data (synthetic only) | `STILL IMPROVING` between the last two sizes | `FLAT` (more data of this kind unlikely to help) or `GOT WORSE`, or "too few groups for a confidence statement" | whether to collect or label more data |

Reading rule for all of them: a number is evidence only if the Val set is large enough. The rehearsal Val had 7 lessons and the Test 8, so its confidence statements are weak by design (chapter 12).

---

## Mismatches found

Where code and documents disagree, this guide says what the code does. These are listed, not fixed.

1. **Number of config keys.** `docs/COMPONENTS.md` (C13, `config-docs`) says "20 sections, 179 keys". The generated `docs/CONFIG_REFERENCE.md` says "21 sections, 184 keys" (it was regenerated after Task 1). The reference is right; the C13 line is stale.
2. **Test counts in `docs/COMPONENTS.md` are stale in places.** Counted from the test files on 2026-10-25: Subsampler 26 (the document says 20), RunManager 29 (28), `test_orchestrator_report.py` 21 (18), `test_orchestrator_curve.py` 21 (20), `test_orchestrator_exit.py` 7 (6). The other counts match.
3. **Rehearsal time.** `docs/OPEN_ISSUES.md` (status after phase E) and `docs/DECISIONS.md` say "35 min against 37 planned". `configs/work_like.yaml` says "real 38 min", and `data/rehearsal_summary.md` says "about 38 minutes of work", and "the resume itself took 35 minutes". The stage times in `logs/run.log` add up to about 38 minutes of work; the 35 is the resumed part only.
4. **Exit code 1 is missing from two tables.** The code (`exit_code_for`) returns 1 for an unexpected bug. `docs/TROUBLESHOOTING.md` ("Exit codes") and `docs/RUNBOOK.md` section 10 list only 0, 2, 3, 4, 5, 130.
5. **`docs/RUNBOOK.md` section 9 says "A run of 6 trials and finalists is 3.7 GB, up to 6.4 GB".** The presets use 4 trials and 2 finalists (`configs/home_full.yaml` says the same 3.7 GB planned, up to 6.4 GB). The trial count in the RUNBOOK sentence does not match the presets.
6. **`docs/RUNBOOK.md` and `docs/TROUBLESHOOTING.md` call the `s/step` of training lines "about 3 times too high".** The code (O121) makes the window mean (w+1)/2 times the true value, so the factor depends on `train.log_every_steps` (2.5 times for 4-step windows, 3.0 for 5-step windows, 1.33 for a 3-step benchmark). "About 3 times" is true only for the rehearsal's logging interval.
7. **`configs/work_like.yaml` says "(real run: ... disk 4.3 GB)".** That was true on 2026-10-08. After the one-time deletion of 2026-10-24 (DECISIONS 84) `data\rehearsal_run` holds 151 MB.
8. **`docs/PRD.md` (lines 1491, 1541, 1611) says the `cleanup` command deletes when given an explicit flag.** The code and `COMPONENTS.md` say `cleanup` never deletes (O111, decided by the owner). `docs/PRD.md` lines 1209 and 1224 also say the registry deletes the other finalists' folders after a registration (`registry.cleanup` true); the Orchestrator never names such a folder, so nothing is deleted (O91).

9. **The "about one third of the disk" of `train.save_optimizer_state: false` (COMPONENTS C6, RUNBOOK, DECISIONS 85) holds for a full fine-tune.** With `train.freeze_policy: freeze_encoder` and the turbo model the benchmark checkpoint is 4.6 GB, of which the optimizer state of the decoder is only about 1.5 GB (estimated), so the saving is about a third of the checkpoint but only about a fifth of the 7.4 GiB trial folder.

None of these changes behaviour.

## Open questions

1. Task 1 limit: should the Orchestrator's `TrainingWiring.can_resume` learn that a checkpoint without `optimizer.pt` cannot be resumed (so an interrupted run made with `train.save_optimizer_state: false` restarts the trial by itself instead of ending with a refusal)? It needs a change in the Orchestrator and was not part of Task 1.
2. For the work machine: the manager's answers on Python version, RAM and cores, data volume and labeling, sleep policy and the final model format are still unknown (O22, O44, O46, O66). Chapters 9 and 16 use the home laptop's numbers only.
3. Is the turbo model (or any large model) really the target? Its memory and disk need on this laptop (see the measurements in chapter 19 and the user guide, section 8) is much larger than that of whisper-tiny; the base-model order decided in DECISIONS 76 is tiny, base, small.
4. Labels shorter than the audio (chapter 20): which option do you want?

## Suggested next step

Read chapter 20 and decide on the label-end question (a, b, c or d). Do not start any change before that decision.
