# Component map

How to use: one entry per component. The agent writes or updates the entry at the end of every component task; the whole file is checked at the end of the project. Entries below hold the planned purpose and config keys extracted from the PRD. The "How to use" line is filled in with the real command or call and one working example when the component is built. Status values: not built, built, verified.

## C0 Config
- Status: not built (wave 1)
- Purpose: Load, merge, validate and freeze all settings into one typed, read-only `AppConfig` object.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `runtime.*` (defined here): `online` (bool, default false), `seed` (int).
    (Consistency fix after the full design review, proposed to the owner: the earlier draft keys `num_proc`,
    `dataloader_num_workers`, `runs_dir` and `log_level` are replaced by the section keys `features.num_workers`,
    `train.cpu_threads`, `run.experiments_dir` and `run.log.*`, so that no setting has two homes.)
  - Other sections are defined by the component that uses them.
  - Environment variable mapping: `STT__SECTION__KEY` overrides `section.key` (exact prefix to be confirmed in implementation brief).

## C1 DataSource
- Status: not built (wave 1)
- Purpose: Load raw labeled audio + text from a LOCAL folder into one uniform list of `AudioSample`, and report what was loaded and what was rejected.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `data.source.type` (`csv` | `hf_parquet`), `data.source.path` (folder)
  - `data.columns.id` (optional), `data.columns.audio`, `data.columns.text`, `data.columns.group`, `data.columns.split` (optional)
  - `data.group_fallback` (`parent_dir` | `file_stem` | `fail`) [DEFAULT: `fail`]: what to do when there is no group column. `fail` because a wrong group causes train/test leakage.
  - `data.extra_types` (mapping column -> `str|int|float|bool`)
  - `data.extract.dir`, `data.extract.overwrite` [DEFAULT: false]
  - `data.on_invalid` (`skip` | `fail`) [DEFAULT: `skip`]
  - `data.max_rejected_ratio` [DEFAULT: 0.05]: if more than this fraction is rejected, the run FAILS (safety gate against a wrongly configured mapping). The value is a starting point to be reviewed after the first real report.
  - Defaults for the Gmara dataset (example only, in the example YAML, not in code): audio=`audio`, text=`transcript`, group=`meta_entry_id`, split=`source_split` from the parquet file/split name.

## C3 Splitter
- Status: not built (wave 1)
- Purpose: Divide the samples into named splits (default `train`, `val`, `test`) BY GROUP, deterministically, guaranteeing that no group appears in more than one split.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `split.mode` (`predefined` | `grouped_ratio` | `hash_group` | `time_block`) [DEFAULT: `grouped_ratio`; final choice OPEN, depends on the data check]
  - `split.group_key.column` [DEFAULT: the DataSource group column, i.e. one lesson/recording], `split.group_key.regex` [DEFAULT: none]
    - OPEN (O18): a content-level key, for example the tractate code extracted by regex from the `meta_source` column. Decided after the data is inspected.
  - `split.names` [DEFAULT: `train`, `val`, `test`]
  - `split.ratios` [DEFAULT: 0.8 / 0.1 / 0.1], `split.ratio_unit` (`duration` | `samples`) [DEFAULT: `duration`], `split.ratio_tolerance` [DEFAULT: 0.05 absolute]
  - `split.seed` (falls back to `runtime.seed`)
  - `split.min_samples_per_split` [DEFAULT: 30], `split.min_groups_per_split` [DEFAULT: 2]; both to be reviewed once the data volume is known (O22)
  - `split.on_oversized_group` (`fail` | `warn`) [DEFAULT: `fail`]
  - `split.on_overlap` (`fail` | `drop_from_train`) for `predefined` [DEFAULT: `fail`]
  - `split.time_block.order_column` [DEFAULT: `meta_seek`], `split.time_block.guard_gap_sec` [DEFAULT: 30]
  - `split.dir`, `split.version`, `split.reuse` [DEFAULT: true], `split.export_csv` [DEFAULT: false]

## C2 TextNormalizer
- Status: not built (wave 1)
- Purpose: Produce for every sample a training label text and an evaluation reference text from the raw transcript, by applying configurable, ordered normalization steps, and report what was found in the text.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `text.eval_version` [DEFAULT: 1]
  - `text.profiles.train.steps` [DEFAULT: `unicode_clean`, `strip_timestamps`, `unify_quotes`, `remove_bracket_markup`, `collapse_whitespace`]
  - `text.profiles.eval.steps` [DEFAULT: `unicode_clean`, `strip_timestamps`, `remove_punctuation`, `unify_quotes`, `remove_nikud`, `remove_bracket_markup`, `lowercase_latin`, `collapse_whitespace`]
  - `text.train.timestamps` (`strip` in v1; `keep` reserved for a later extension) [DEFAULT: `strip`]
  - `text.bracket.pairs` [DEFAULT: `[` `]` and `(` `)`], `text.bracket.mode` [DEFAULT: `chars_only`]; optional per-profile override `text.profiles.<name>.bracket_mode` [DEFAULT: unset = use `text.bracket.mode`]. Recipe for a non-speech marker (O47): train profile `keep`, eval profile `with_content`, so the marker is learned but never counted in the WER. Defaults are unchanged, so nothing happens unless the config asks for it.
  - `text.punctuation.chars` [DEFAULT: a standard Hebrew and Latin punctuation set, listed in the example YAML]
  - `text.quotes.map` [DEFAULT: ׳ ' ’ unified to one form; ״ " “ ” unified to one form]
  - `text.inventory.top_n` [DEFAULT: 50]

## C1b SampleFilter
- Status: not built (wave 1)
- Purpose: Decide, per split, which `AudioSample`s are kept according to configurable rules, and report exactly what was removed and why.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `filter.mode` (`report` | `apply`) [DEFAULT: `report`]
  - `filter.rules.<rule_name>.enabled`, `.applies_to` (list of splits), `.kind` (`structural` | `quality`), plus the rule's threshold:
    - `min_duration` [DEFAULT: 1.0 s], `max_duration` [DEFAULT: 30.0 s, the Whisper window; changed from 30.5 after the FeaturePipeline discussion: the filter removes longer clips, FeaturePipeline never truncates and FAILS if a longer clip arrives. Apply mode is required for this to take effect]
    - `min_text_chars` [DEFAULT: 2], measured on `eval_text` (after normalization, so timestamp tokens and punctuation do not count); optional key `measure_on` (`eval_text` | `train_text`) [DEFAULT: `eval_text`]. For clips labeled with a non-speech marker (O47) the recipe is `measure_on: train_text` for the train split, so they are not removed as "empty"; their empty reference is excluded from the WER by the Evaluator
    - `min_quality_score` [DEFAULT: null = disabled], `max_bad_segments` [DEFAULT: null = disabled], `require_golden` [DEFAULT: false]
    - defaults are deliberately permissive (reject only obviously broken samples) until the first real report is reviewed.
  - `filter.min_kept_ratio` [DEFAULT: 0.4]: if a split keeps less than this fraction, the run FAILS (safety gate).

## C3b Subsampler
- Status: not built (wave 2)
- Purpose: Select, deterministically and reproducibly, a smaller subset of the samples to the size requested in Config, and report exactly what was selected.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `subsample.enabled` [DEFAULT: false]
  - `subsample.size` [DEFAULT: `all`]
  - `subsample.strategy` [DEFAULT: `by_sample`]
  - `subsample.scope` [DEFAULT: `train_only`]
  - `subsample.val` [DEFAULT: `fixed`]
  - `subsample.repeats` [DEFAULT: 1]
  - `subsample.min_train_samples` [DEFAULT: 10]: if fewer samples are selected, FAIL
  - `subsample.presets` (named sizes for convenience) [DEFAULT: `smoke` = 20 samples, `tiny` = 30 minutes, `work_like` = fraction 0.1 with `scope: all_splits` and `by_group`, `half` = fraction 0.5, `full` = all]
  - seed from the global Config.

## C5 ModelLoader
- Status: not built (wave 1)
- Purpose: Load one Whisper model, its feature extractor and its tokenizer from a LOCAL folder, validate them, and describe what was loaded.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `model.loader` [DEFAULT: `hf_whisper`]
  - `model.path` (folder, required, no default)
  - `model.language` [DEFAULT: `he`]
  - `model.task` [DEFAULT: `transcribe`]
  - `model.dtype` [DEFAULT: `float32`; CPU]
  - Example YAML (examples only, never in code): phase 1 `model.path` points to a local copy of OpenAI `whisper-tiny`
    (`whisper-base` as the next step); a commented-out later example for `ivrit-ai/whisper-large-v3-turbo`.
  - Python version [DEFAULT: 3.12, the conservative choice for torch / transformers wheels on Windows and for an offline machine]. Still to be confirmed by the clean-venv install test (O1).

## C4 FeaturePipeline
- Status: not built (wave 1)
- Purpose: Convert kept samples into model-ready examples (input features from the audio, label ids from `train_text`) and provide the batch collator.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `features.num_workers` [DEFAULT: 0]
  - `features.cache.enabled` [DEFAULT: false], `features.cache.dir`
  - `features.max_dropped_ratio` [DEFAULT: 0.05]: safety gate for dropped labels
  - `features.downmix` [DEFAULT: `mean`]
  - `text.train.timestamps` (owned by TextNormalizer; read here)
  - from ModelLoader's output (not Config): sampling rate, window length, mel bins, label limit

## C6 TrainerCore
- Status: not built (wave 1)
- Purpose: Run ONE training run on a given model and datasets with given hyperparameters, and return the trained model folder and the record of the run.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `train.freeze_policy` [DEFAULT: `none`]
  - `train.hyperparams.*` [DEFAULT: small values for batch size (for example 4) and accumulation; learning rate 1e-5; `max_steps` small for smoke runs; final values decided by experiments]
  - `train.log_every_steps` [DEFAULT: 10], `train.eval_every_steps` [DEFAULT: 50]
  - `train.save_every_steps` [DEFAULT: 100]
  - `train.early_stopping.patience` [DEFAULT: 5 evaluations], `train.early_stopping.min_delta` [DEFAULT: 0.0]
  - `train.max_runtime_minutes` [DEFAULT: null = no cap]
  - `train.cpu_threads` [DEFAULT: half of the cores]
  - `train.resume` [DEFAULT: false]
  - `train.output_dir` (per experiment; set by the RunManager later)
  - `train.sinks` [DEFAULT: `csv`, `console`]
  - The search space of the hyperparameters belongs to the HyperparamSearcher, not here.

## C7 Evaluator
- Status: not built (wave 1)
- Purpose: Transcribe an evaluation set with a given model, score the transcripts against the references, and compare two scored results with a statistical test.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `eval.metrics` [DEFAULT: `wer`, `cer`]
  - `eval.generation.num_beams` [DEFAULT: 1], `eval.generation.max_new_tokens` [DEFAULT: the model's label limit], `eval.generation.max_tokens_per_audio_sec` [DEFAULT: set from the first runs]
  - `eval.batch_size` [DEFAULT: 4]
  - `eval.thresholds.wer_below` [DEFAULT: 0.10, 0.20]
  - `eval.significance.method` [DEFAULT: `paired_group_bootstrap`], `.confidence` [DEFAULT: 0.90], `.one_sided` [DEFAULT: true], `.n_resamples` [DEFAULT: 2000], `.min_effect_abs` [DEFAULT: 0.0], `.min_groups` [DEFAULT: 5]
  - `eval.review.n_random` [DEFAULT: 20], `eval.review.n_worst` [DEFAULT: 10]
  - `eval.sanity.max_baseline_wer` [DEFAULT: null = off]
  - seed from the global Config.

## C8 HyperparamSearcher
- Status: not built (wave 2)
- Purpose: Propose hyperparameter sets, hand each one to a trial runner, record every outcome, and return the ranked candidates within a budget.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `search.strategy` [DEFAULT: `optuna_tpe`]
  - `search.parameters.learning_rate` [DEFAULT: log-uniform range around the training default], `search.parameters.grad_accum_steps` [DEFAULT: a few integer choices]
  - `search.n_trials` [DEFAULT: 10], `search.time_budget_minutes` [DEFAULT: null]
  - `search.pruning.enabled` [DEFAULT: true], `search.pruning.min_evals` [DEFAULT: 3]
  - `search.parallel_trials` [DEFAULT: 1]
  - `search.max_failed_ratio` [DEFAULT: 0.5], `search.min_trials_before_abort` [DEFAULT: 4]
  - `search.finalists.k` [DEFAULT: 3], `search.finalist_repeats` [DEFAULT: 1]
  - `search.screening.preset` [DEFAULT: `tiny`, a Subsampler preset; to revise after the first step-time measurements, O11]
  - `search.study_dir` (inside the experiment folder; set by the RunManager)
  - seed from the global Config.

## C9 Selector
- Status: not built (wave 2)
- Purpose: Decide, from evaluated finalists and the baseline, which model is the winner, or declare that the baseline stays.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `select.min_verdict` [DEFAULT: `improved`]
  - `select.tie_epsilon` [DEFAULT: 0.001 absolute WER]
  - `select.on_insufficient_groups` [DEFAULT: `plain_difference`]
  - `select.test_confirmation` [DEFAULT: true]

## C10 ModelRegistry
- Status: not built (wave 2)
- Purpose: Store the selected model together with its metadata as an immutable, numbered entry, and keep an index of all entries.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `registry.dir`
  - `registry.keep_finalists` [DEFAULT: false]
  - `registry.cleanup` [DEFAULT: true]
  - `registry.transfer` [DEFAULT: `move`; alternative `copy`]
  - `registry.checksums` [DEFAULT: true]

## C14 ReportBuilder
- Status: not built (wave 3)
- Purpose: Turn the results of a run into a self-contained visual report, and into a live progress page while training runs, that show at a glance whether the model succeeded, where it failed and what could be changed.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `report.attributes` [DEFAULT: `duration_sec`, `label_tokens`, `ref_words`, `meta_quality_score` (only if present), `had_digits`, `had_latin`, `had_brackets`]. Open to change after the owner sees the first output.
  - `report.min_points_for_correlation` [DEFAULT: 30]
  - `report.max_points_per_chart` [DEFAULT: 2000; deterministic thinning]
  - `report.hints.enabled_rules` [DEFAULT: all registered]
  - `report.hints.*` thresholds [DEFAULT, to calibrate on real runs]:
    - overfitting: Val loss rising over the last K evaluations while the training loss falls
    - no learning: relative Val loss improvement below a threshold
    - unstable training: loss spikes
    - gain concentrated in few groups
    - weakness on short clips
    - many outputs hit the length cap or are empty
    - best trial at the edge of a searched range
    - many failed or pruned trials
  - `report.progress.refresh_seconds` [DEFAULT: 10], `report.progress.every_n_records` [DEFAULT: 1]
  - `report.title` [DEFAULT: experiment name]

## C11 ModelExporter
- Status: not built (wave 3)
- Purpose: Convert a registered model into a target format, verify the result, and record the export.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `export.formats` [DEFAULT: `hf_safetensors`]
  - `export.verify` [DEFAULT: true]
  - `export.verify_samples` [DEFAULT: 5]
  - `export.max_wer_diff` [DEFAULT: 0.02, absolute WER difference allowed between the exported and the original model on the check files]

## C12 RunManager
- Status: not built (wave 1)
- Purpose: Create and manage numbered experiment folders (config snapshots, diffs, status, lock, logs, and the reports written into them).
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `run.experiments_dir`
  - `run.parent` [DEFAULT: none]
  - `run.id_digits` [DEFAULT: 4]
  - `run.config_readonly` [DEFAULT: true]
  - `run.lock_heartbeat_seconds` [DEFAULT: 30], `run.lock_stale_minutes` [DEFAULT: 10]
  - `run.log.console_level` [DEFAULT: INFO], `run.log.file_level` [DEFAULT: DEBUG]

## C13 Orchestrator and CLI
- Status: not built (wave 1)
- Purpose: Wire the components together and run the whole pipeline, or any single stage of it, from one command with full automation and a clear exit code.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)
- Config keys (planned):
  - `pipeline.stages` [DEFAULT: `prepare`, `search`, `select_and_register`, `export`, `report`]
  - `pipeline.skip_completed` [DEFAULT: true]
  - `run.prevent_sleep` [DEFAULT: true]
  - `smoke.n_samples` [DEFAULT: 12], `smoke.duration_sec` [DEFAULT: 3], `smoke.steps` [DEFAULT: 3], `smoke.model_dir`
  - `learning_curve.presets` [DEFAULT: `smoke`, `tiny`, `half`, `full`], `learning_curve.params_from` [DEFAULT: none]
  - `benchmark.steps` [DEFAULT: 3]

## T Standalone tools
- Status: not built (wave 3)
- Purpose: Prepare, outside the pipeline and with network access allowed ONLY here, everything the offline pipeline needs: models, data, packages, the tiny test model and the code manifest.
- How to use: (to be written when built: exact command or call, and one example)
- Inputs and outputs: see the brief in `docs/PRD.md`
- Known limits: (to be written when built)
- Tests: (to be filled in when built)

