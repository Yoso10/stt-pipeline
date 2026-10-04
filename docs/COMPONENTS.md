# Component map

How to use: one entry per component. The agent writes or updates the entry at the end of every component task; the whole file is checked at the end of the project. Entries below hold the planned purpose and config keys extracted from the PRD. The "How to use" line is filled in with the real command or call and one working example when the component is built. Status values: not built, built, verified.

## C0 Config
- Status: built (wave 1); `mypy --strict` is not installed here, so it was not run
- Purpose: Load, merge, validate and freeze all settings into one typed, read-only `AppConfig` object.
- How to use (project root, PowerShell; set `$env:PYTHONPATH = "src"` once per window):
  ```
  python -c "from pathlib import Path; from stt.config import load_config, dump_config; print(dump_config(load_config(Path('configs/base.yaml'), overrides=['runtime.seed=7'], target_environ={})))"
  ```
  Output: `runtime:` then `online: false` and `seed: 7`. In code: `config = load_config(Path("configs/x.yaml"), overrides=["a.b=3e-5"], registry=registry)`; the Orchestrator can use `add_config_arguments(parser)` (adds `--config PATH` and repeatable `--set A.B=VALUE`) and `load_config_from_args(args)`. Precedence: defaults in code < YAML (`extends: other.yaml` allowed) < environment `STT__SECTION__KEY` (also read from `.env`; the real environment wins) < `--set`. Offline by default: unless `runtime.online` is true, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE` are set to 1. Use `config.section("name", SchemaClass)` to read a component's section.
- Inputs and outputs: a YAML path (optional), `--set` strings, environment, `.env` -> a frozen `AppConfig` (`runtime` plus the registered sections). `dump_config(config)` returns the resolved YAML text. Errors raise `ConfigError` (exit code 2) naming the key and how to fix it; all problems are reported together.
- How a component adds its section (no change to the loader): subclass `SectionModel` (frozen, unknown keys forbidden), then the composition root builds `SectionRegistry({"data": DataSection})` and passes it to `load_config`. `stt.config.validators` offers `ExistingPath` and `check_ratios_sum_to_one` for such schemas.
- Known limits: only the `runtime` section exists until components are built, so any other top-level key in a YAML file fails as unknown. `.env` lines are read only for `STT__` names (nothing else is exported). The dump has no comments (the per-key reference comes from `config-docs`, a later task). `extends` takes one file name, not a list.
- Tests: `tests/test_config.py` (34 tests, offline), `tests/test_package_layout.py`. Files: `src/stt/config/` (`schema.py`, `sources.py`, `loader.py`, `validators.py`, `errors.py`), `configs/base.yaml`, `.env.example`.
- Config keys (defined here):
  - `runtime.online` (bool, default false), `runtime.seed` (int, DEFAULT 42).
    (Consistency fix after the full design review: the earlier draft keys `num_proc`, `dataloader_num_workers`, `runs_dir` and `log_level` are replaced by the section keys `features.num_workers`, `train.cpu_threads`, `run.experiments_dir` and `run.log.*`, so that no setting has two homes.)
  - Other sections are defined by the component that uses them.
  - Environment variable mapping: `STT__SECTION__KEY` overrides `section.key` (double underscore; decided in DECISIONS, section 2, item 31).

## C1 DataSource
- Status: built (wave 1); run once on the real Daf Yomi data (8723 rows, 0 rejected)
- Purpose: Load raw labeled audio + text from a LOCAL folder into one uniform list of `AudioSample`, and report what was loaded and what was rejected.
- How to use (project root, PowerShell, `$env:PYTHONPATH = "src"`; needs `data/daf_yomi_raw` from `tools/fetch_data.py`; the first run extracts about 800 MB of audio to `data/extracted/daf_yomi`, about 3 minutes; later runs reuse it, about 10 seconds):
  ```
  python -c "from pathlib import Path; from stt.config import SectionRegistry, load_config; from stt.datasource import DataConfig, create_data_source; c = load_config(Path('configs/data_gmara.yaml'), registry=SectionRegistry({'data': DataConfig})); r = create_data_source(c.section('data', DataConfig), Path('data/load_out')).load(); print(r.report.summary())"
  ```
  Output starts with `seen 8723, loaded 8723, rejected 0`, then the reasons, the counts per split, the number of groups and the duration summary. `rejected.csv` (id, reason, detail) is written to the folder given as `out_dir` (here `data/load_out`; in a run the RunManager passes the experiment folder). Config: copy `configs/data_gmara.yaml` and change the paths and column names.
- Inputs and outputs: config section `data` (class `DataConfig`, registered by the composition root with `SectionRegistry({"data": DataConfig})`) -> `LoadResult(samples sorted by id, report, rejections)`. `AudioSample`: `id`, `audio_path`, `text` (raw), `group_id`, `duration_sec` (from the audio header), `source_split`, `extra` (all other columns, typed by `data.extra_types`; nested `metadata` fields as `meta_<name>`). `LoadReport` has counts, rejection reasons, per-split counts, duration and numeric-extra distributions, empty values left out of `extra`, and how many `group_id` occur in more than one `source_split` (information only). Errors: `DataSourceError` (exit code 3).
- Sources (selected by `data.source.type` through `DataSourceRegistry`, `default_registry()`): `csv` = a folder with `manifest.csv` (UTF-8, BOM tolerated; relative audio paths are relative to the manifest); `hf_parquet` = a folder of parquet files, extracted to `data.extract.dir` (`audio/<split>/<id>.<ext>`, `manifest.csv` written last, `extract_info.json`), then read by the csv code. Parquet ids are `<split>_<row:06d>` (`<split>` = file name before the first `-`).
- Rejection reasons: `missing_audio`, `empty_text`, `unreadable_audio` (from the brief); `missing_group`, `malformed_row`, `bad_extra_value` (added, see DECISIONS 32). Failures: duplicate or empty id, a mapped column that does not exist (the existing columns are listed), no group column with `group_fallback: fail`, rejected ratio above `data.max_rejected_ratio` (after writing `rejected.csv`), zero samples, no parquet files, a reused extraction made from other source files.
- Known limits: audio arrays are never loaded (headers only). An empty value of a non-text `extra` type is left out of `extra` and counted (not a rejection). Extraction needs free disk space about the size of the parquet files. The duration is read from the header with soundfile, about 20 ms per mp3 file.
- Tests: `tests/test_datasource.py` (35 tests: the 10 acceptance tests of the brief plus edge cases; tiny synthetic wav and parquet files, offline).
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
- Status: built (wave 1); run once on the real Daf Yomi data (8723 samples, 710 groups)
- Purpose: Divide the samples into named splits (default `train`, `val`, `test`) BY GROUP, deterministically, guaranteeing that no group appears in more than one split.
- How to use (project root, PowerShell, `$env:PYTHONPATH = "src"`; needs the extracted audio of C1):
  ```
  python -c "from pathlib import Path; from stt.config import SectionRegistry, load_config; from stt.datasource import DataConfig, create_data_source; from stt.splitter import SplitConfig, resolve_split; c = load_config(Path('configs/data_gmara.yaml'), overrides=['split.speaker_column=meta_creator'], registry=SectionRegistry({'data': DataConfig, 'split': SplitConfig})); s = create_data_source(c.section('data', DataConfig), Path('data/load_out')).load().samples; print(resolve_split(c.section('split', SplitConfig), c.runtime.seed, s).report.summary())"
  ```
  Output (about 15 seconds): `mode grouped_ratio, seed 42 ...`, `samples 8723, groups 710`, then `train: 6973 samples, 566 groups, 57.90 h, ratio 0.800`, the same for `val` and `test`, the largest groups, the overlap check, a table `source_split -> split`, and the speaker line. In code: `resolve_split(split_config, runtime_seed, samples)` returns a `SplitResult`; `default_registry().create(config, seed).split(samples)` splits without saving. Saving: set `split.dir` (the split is written to `<dir>/<version>/split.json` and reused by later runs; a new split needs a new `split.version`; `split.export_csv: true` also writes `split.csv`). Example YAML with every key and its DEFAULT: `configs/split_default.yaml`. In a command-line list, write JSON text, for example `--set 'split.names=["a","b"]'`.
- Inputs and outputs: `list[AudioSample]` (C1) -> `SplitResult`: `splits` (name -> samples sorted by id), `dropped_guard` (time_block only), `dropped_overlap` (predefined with `drop_from_train` only), `report` (`SplitReport`: per split samples, groups, hours, actual and requested ratio; the largest groups; the overlap check; the table source split -> split; a speaker line; both fingerprints), `fingerprint` (this split) and `data_fingerprint` (the input). Errors: `SplitError` (exit code 3).
- Modes (`split.mode`, a registry): `grouped_ratio` (DEFAULT: seeded stable order, each whole group goes to the split furthest below its target; fails when a ratio is missed by more than `split.ratio_tolerance`), `hash_group` (sha256 of seed and group: adding groups never moves old ones), `predefined` (uses `source_split`; missing names such as `val` are carved from the first split by group), `time_block` (cuts each recording into contiguous blocks by `meta_seek` with a guard gap; groups span splits by design). Hard checks after every split: no group in two splits (not for time_block), the union equals the input, no duplicate ids, no empty split; then the minimum gates and the oversized-group gate.
- Known limits: the official `eval` split is NOT used for the Test (decision 26): `source_split` is only reported. The Test measures new lessons, not new speakers; set `split.speaker_column` to see it in the report. No stratification and no text-duplicate check between splits (O19). With `hash_group` and few groups the shares can be uneven (the minimum gates fail instead). A saved split cannot be loaded after the data changed (fingerprint check): use a new `split.version` (with `hash_group` the old groups keep their splits). Neighbouring lessons of the same tractate can still share content (O18).
- Tests: `tests/test_splitter.py` (41 tests: the 10 acceptance tests of the brief plus edge cases; synthetic samples, no files, offline).
- Config keys (defined here, section `split`; `split.predefined.source_map` and `split.speaker_column` were added, see DECISIONS 33):
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
- Status: built (wave 1); run once on the 8723 real labels
- Purpose: Produce for every sample a training label text and an evaluation reference text from the raw transcript, by applying configurable, ordered normalization steps, and report what was found in the text.
- How to use (project root, PowerShell, `$env:PYTHONPATH = "src"; $env:PYTHONIOENCODING = "utf-8"`; needs the extracted audio of C1; about 20 seconds):
  ```
  python -c "from pathlib import Path; from stt.config import SectionRegistry, load_config; from stt.datasource import DataConfig, create_data_source; from stt.textnormalizer import TextConfig, TextNormalizer; c = load_config(Path('configs/data_gmara.yaml'), registry=SectionRegistry({'data': DataConfig, 'text': TextConfig})); s = create_data_source(c.section('data', DataConfig), Path('data/load_out')).load().samples; r = TextNormalizer(c.section('text', TextConfig)).normalize(s); print(r.report.summary()); print(r.samples[0].train_text); print(r.samples[0].eval_text)"
  ```
  Output: `samples 8723, eval_version 1`, the two step lists, the flag counts, and the character inventory (code point, Unicode name, counts in the raw text and in each profile), then the first label as `train_text` and as `eval_text`. For a plain string (the Evaluator does this for references and for model outputs): `from stt.textnormalizer import TextConfig, build_profile; f = build_profile(config.section("text", TextConfig), "eval"); f("<|0.5|>שלום, עולם. <|2.0|>")` gives `שלום עולם`. Example YAML with every key and its DEFAULT: `configs/text_default.yaml`. To see more than 50 characters in the inventory set `text.inventory.top_n`.
- Inputs and outputs: `list[AudioSample]` (C1) -> `NormalizeResult(samples, report)`. `NormalizedSample`: `sample` (unchanged, raw text stays in `sample.text`), `train_text`, `eval_text`, `timestamp_segments` (parsed before stripping; None when there are no tokens or they are malformed), `flags`. Flags: `had_timestamps`, `bad_timestamps`, `had_brackets`, `had_digits`, `had_latin`, `had_nikud`, `empty_after_normalization` (computed on the text with the timestamp tokens masked). `NormalizeReport`: flag counts, texts changed per profile, ids that became empty, the character inventory (every character that is not a Hebrew letter, a digit or whitespace; raw counts ignore the timestamp tokens), the eval version and the step lists. Samples are never dropped and keep the input order. Errors: `TextNormalizerError` (exit code 2) for an unknown step or profile (the registered names are listed); `text.train.timestamps: keep` fails at Config validation.
- Steps (registry, selected by name from the profile lists): `unicode_clean`, `strip_timestamps`, `remove_punctuation`, `unify_quotes`, `remove_nikud`, `remove_bracket_markup` (`chars_only`, `with_content`, `keep`), `lowercase_latin`, `collapse_whitespace`. A new step is a new class plus one line in `registry.py`. The profile applies its steps until the text stops changing (so it is idempotent even when removing characters creates a new pattern).
- Known limits: v1 strips timestamps only (no `keep`). No spelling correction, abbreviation expansion or number normalization. No text-duplicate check between splits (O19: figures were measured once, see OPEN_ISSUES). In `eval_text` the geresh is removed, so `גמ׳` becomes `גמ` and `ה׳` becomes `ה` (owner decision 29). The default punctuation list also removes hyphens and dashes (replaced by nothing, except the maqaf which becomes a space); none occurs in the Daf Yomi data. Changing the eval profile needs a new `text.eval_version`.
- Tests: `tests/test_textnormalizer.py` (36 tests: the 11 acceptance tests of the brief, a random idempotence test, offline).
- Config keys (section `text`; `text.punctuation.space_chars`, `text.unicode.invisible_chars` and `text.timestamp_pattern` were added, see DECISIONS 34):
  - `text.eval_version` [DEFAULT: 1]
  - `text.profiles.train.steps` [DEFAULT: `unicode_clean`, `strip_timestamps`, `unify_quotes`, `remove_bracket_markup`, `collapse_whitespace`]
  - `text.profiles.eval.steps` [DEFAULT: `unicode_clean`, `strip_timestamps`, `remove_punctuation`, `unify_quotes`, `remove_nikud`, `remove_bracket_markup`, `lowercase_latin`, `collapse_whitespace`]
  - `text.train.timestamps` (`strip` in v1; `keep` reserved for a later extension) [DEFAULT: `strip`]
  - `text.bracket.pairs` [DEFAULT: `[` `]` and `(` `)`], `text.bracket.mode` [DEFAULT: `chars_only`]; optional per-profile override `text.profiles.<name>.bracket_mode` [DEFAULT: unset = use `text.bracket.mode`]. Recipe for a non-speech marker (O47): train profile `keep`, eval profile `with_content`, so the marker is learned but never counted in the WER. Defaults are unchanged, so nothing happens unless the config asks for it.
  - `text.punctuation.chars` [DEFAULT: a standard Hebrew and Latin punctuation set, listed in the example YAML]
  - `text.quotes.map` [DEFAULT: ׳ ' ’ unified to one form; ״ " “ ” unified to one form]
  - `text.inventory.top_n` [DEFAULT: 50]

## C1b SampleFilter
- Status: built (wave 1); run once on the real data in report mode
- Purpose: Decide, per split, which `AudioSample`s are kept according to configurable rules, and report exactly what was removed and why.
- How to use (project root, PowerShell, `$env:PYTHONPATH = "src"`; needs the extracted audio of C1; about 25 seconds):
  ```
  python -c "from pathlib import Path; from stt.config import SectionRegistry, load_config; from stt.datasource import DataConfig, create_data_source; from stt.splitter import SplitConfig, resolve_split; from stt.textnormalizer import TextConfig, TextNormalizer; from stt.samplefilter import FilterConfig, SampleFilter; c = load_config(Path('configs/data_gmara.yaml'), registry=SectionRegistry({'data': DataConfig, 'split': SplitConfig, 'text': TextConfig, 'filter': FilterConfig})); s = create_data_source(c.section('data', DataConfig), Path('data/load_out')).load().samples; sp = resolve_split(c.section('split', SplitConfig), c.runtime.seed, s); n = {x.sample.id: x for x in TextNormalizer(c.section('text', TextConfig)).normalize(s).samples}; f = SampleFilter(c.section('filter', FilterConfig)); [print(r.report.summary()) for r in f.filter_splits({k: [n[x.id] for x in v] for k, v in sp.splits.items()}).values()]"
  ```
  Output: for each split `split train: mode report, seen 6973, kept 6973, would reject 0, would-be kept ratio 1.0000` and one line per running rule with its threshold and failure count. Change a threshold with `--set`-style overrides, for example `overrides=["filter.rules.min_duration.threshold=5"]` in `load_config`, or in YAML (`configs/filter_default.yaml`). In code: `SampleFilter(filter_config, token_counter=None).filter(normalized_samples, "train")` returns a `FilterResult`; `filter_splits({"train": [...], "test": [...]})` does every split.
- Inputs and outputs: `NormalizedSample`s of one split (C2) plus the split name -> `FilterResult(kept, rejected, would_reject, report)`. `kept` keeps the input order; in `report` mode (DEFAULT) `kept` is everything and `rejected` is empty, `would_reject` always lists the samples that fail a rule (id, failed rule names, reasons); in `apply` mode `rejected` equals `would_reject`. `FilterReport`: counts, would-be kept ratio, one line per rule run (kind, threshold, failures), the rules not applied to this split. Errors: `FilterError` (exit code 3) for an unknown rule name (the registered names are listed), an enabled rule without threshold or field or token counter, or a field missing from the samples (rule and field named; fix: map the column or set the rule `enabled: false`); `FilterGateError` (a `FilterError`) when a split would keep less than `filter.min_kept_ratio` (per-rule breakdown; the error carries the full `result`) or is empty after `apply`.
- Rules (registry, defaults): structural, all splits: `min_duration` 1.0 s, `max_duration` 30.0 s, `min_text_chars` 2 on `eval_text` (`measure_on` can be `train_text`); quality, train only, OFF until a threshold is set: `min_quality_score`, `max_bad_segments` (both need `field`, the name of the sample field they read), `require_golden` (needs `enabled: true` and `field`); optional structural `max_label_tokens` (OFF; measures `train_text` with a token counter injected as `Callable[[str], int]`; this is the alternative of open issue O34, the brief keeps label-length removal in the FeaturePipeline). A rule you do not mention in YAML keeps its default; setting a threshold on a disabled rule enables it. A new rule is a new class plus one `RuleSpec` in a registry; the filter is not edited.
- Known limits: thresholds are DEFAULT and permissive (they remove nothing from the Daf Yomi data); the final values are decided from the report and an A/B experiment (O5, O6). The gate checks the would-be kept ratio in both modes. The filter imports no tokenizer. No text-length-to-duration rule (O13, a later class). Rules never read audio: the duration is `AudioSample.duration_sec` from the C1 header.
- Tests: `tests/test_samplefilter.py` (29 tests: the 8 acceptance tests of the brief plus config, defaults and the token rule with a fake counter; offline).
- Config keys (section `filter`; the rule keys are `enabled`, `applies_to`, `kind`, `threshold`, `field`, `measure_on`, see DECISIONS 35):
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
- Status: built (wave 1); run once on the real `models/whisper-tiny`
- Purpose: Load one Whisper model, its feature extractor and its tokenizer from a LOCAL folder, validate them, and describe what was loaded.
- How to use (project root, PowerShell, `$env:PYTHONPATH = "src"; $env:PYTHONIOENCODING = "utf-8"`; needs `models/whisper-tiny` from `tools/fetch_model.py`, or use the stored `tests/assets/tiny_whisper`):
  ```
  python -c "from pathlib import Path; from stt.modelloader import ModelConfig, create_loader, make_token_counter; m = create_loader(ModelConfig(path=Path('models/whisper-tiny'))).load(); print(m.info); print(m.feature_extractor.num_mel_bins, make_token_counter(m.tokenizer)('שלום עולם'))"
  ```
  Output: `ModelInfo(name='whisper-tiny', ... num_parameters=37760640, dtype='float32', num_mel_bins=80, language='he', task='transcribe', source_config_ref=None)` then `80 4`. The first load takes about 6 seconds (it imports torch), later ones under a second. In a run the section is `model` (class `ModelConfig`, registered with `SectionRegistry({"model": ModelConfig})`); the model is chosen ONLY by `model.path` (a local folder, never a Hub name). Example YAML: `configs/model_default.yaml`.
- Inputs and outputs: `ModelConfig` -> `create_loader(config).load()` -> `LoadedModel(model, feature_extractor, tokenizer, info)`. `ModelInfo`: `name`, `path`, `num_parameters`, `dtype`, `num_mel_bins`, `language`, `task`, `source_config_ref` (from `pipeline_model_info.json` when it exists). The loaded objects are small Protocols (DECISIONS 37): `FeatureExtractor` (`sampling_rate`, `window_seconds`, `num_mel_bins`, `extract(audio, sampling_rate) -> float32 (mel bins, frames)`, which refuses another rate and audio longer than the window), `LabelTokenizer` (`max_label_tokens`, `encode_label(text)` with the start, language, task and no-timestamps prefix and the end token, `count_tokens(text)` without special tokens, `decode(ids, skip_special_tokens)`), and the model itself is an opaque `object` for the adapter modules of the trainer and evaluator. `make_token_counter(tokenizer)` gives a plain `Callable[[str], int]` for the composition root to pass to the SampleFilter. Errors: `ModelLoaderError` (exit code 2).
- Rules kept: local files only, even when `runtime.online` is true; `trust_remote_code` never; the folder is validated first (all missing files are listed) and never modified; safetensors weights required (old formats fail with the conversion advice); not-Whisper, English-only and unsupported-language models fail with the model and language named; language and task are written into the generation settings and the label prefix; two `load()` calls return independent objects; torch and transformers are imported only in `stt/modelloader/hf_whisper.py`, and only when a loader is created.
- Known limits: one implementation (`hf_whisper`). The freeze policy, dropout and thread count are not the loader's business (TrainerCore). The dtype `float16` and `bfloat16` load but are not tested for speed (CPU phase 1 is float32). Old-format weights (`pytorch_model.bin`) are not accepted.
- Tests: `tests/test_modelloader.py` (28 tests: the 9 acceptance tests of the brief plus the Protocols, the token counter and the config; the stored tiny model, network blocked), `tests/test_make_test_model.py`. Test model: `tests/assets/tiny_whisper` (built by `tools/make_test_model.py`).
- Config keys (section `model`): `model.path` (required), `model.loader` [DEFAULT `hf_whisper`], `model.language` [DEFAULT `he`], `model.task` [DEFAULT `transcribe`], `model.dtype` [DEFAULT `float32`]. Python 3.12 with torch 2.14.1 (CPU) was confirmed (O1).

## C4 FeaturePipeline
- Status: built (wave 1); run once on the real train split (6973 samples)
- Purpose: Convert kept samples into model-ready examples (input features from the audio, label ids from `train_text`) and provide the batch collator.
- How to use (project root, PowerShell, `$env:PYTHONPATH = "src"`; needs the extracted audio of C1 and `models/whisper-tiny`; about 30 seconds):
  ```
  python -c "from pathlib import Path; from stt.config import SectionRegistry, load_config; from stt.datasource import DataConfig, create_data_source; from stt.splitter import SplitConfig, resolve_split; from stt.textnormalizer import TextConfig, TextNormalizer; from stt.modelloader import ModelConfig, create_loader; from stt.featurepipeline import FeaturesConfig, FeaturePipeline; c = load_config(Path('configs/data_gmara.yaml'), overrides=['model.path=models/whisper-tiny'], registry=SectionRegistry({'data': DataConfig, 'split': SplitConfig, 'text': TextConfig, 'model': ModelConfig, 'features': FeaturesConfig})); s = create_data_source(c.section('data', DataConfig), Path('data/load_out')).load().samples; sp = resolve_split(c.section('split', SplitConfig), c.runtime.seed, s); n = {x.sample.id: x for x in TextNormalizer(c.section('text', TextConfig)).normalize(s).samples}; m = create_loader(c.section('model', ModelConfig)).load(); p = FeaturePipeline(c.section('features', FeaturesConfig), m); r = p.build([n[x.id] for x in sp.splits['train']], 'train'); print(r.report.summary()); print(r.dataset[0].input_features.shape, p.build_collator()([r.dataset[0], r.dataset[1]])['labels'].shape)"
  ```
  Output: `split train: seen 6973, built 6973, removed for label length 0`, `resampled to 16000 Hz: 0, downmixed to mono: 0`, the label-length line, then `(80, 3000) torch.Size([2, N])`. In a run the section is `features` (class `FeaturesConfig`); example YAML: `configs/features_default.yaml`.
- Inputs and outputs: the kept `NormalizedSample`s of one split (C1b) plus a `LoadedModel` (C5; only its feature extractor and tokenizer are used) -> `FeatureResult(dataset, report)`. `dataset` is a lazy plain dataset (`len`, `dataset[i]` -> `FeatureExample(sample_id, input_features, labels)`); `input_features` is a float32 numpy array (mel bins, frames) = (80, 3000) for the OpenAI models (about 0.96 MB each); `labels` are the full label ids from `tokenizer.encode_label(train_text)` (prefix tokens, text, end token). `FeaturePipeline.build_collator()` returns the batch maker (Hugging Face adapter, the only module that imports torch): `batch = collate(examples)` is a dict `{"input_features": float32 (batch, 80, 3000), "labels": int64 (batch, longest)}`, labels padded with -100, the leading decoder start token removed (the model adds it when it shifts the labels), batch order = example order. `FeatureReport`: seen, built, removed for label length (ids listed), resampled, downmixed, label-length distribution (ids: min, mean, p95, max), the limits applied (audio 30 s, labels 448, both read from the model), cache counts. Errors: `FeaturePipelineError` (exit code 3).
- Rules kept: the rate, the window, the mel bins and the label limit are read from the model; other sampling rates are converted with scipy `resample_poly` and counted, stereo is averaged and counted (counted from the file headers, no decoding at build); audio longer than the window is NEVER truncated (the run fails with the count, the longest clip and the fix: SampleFilter `max_duration` in `apply` mode); labels over the limit are removed, listed and gated by `features.max_dropped_ratio`; empty `train_text`, unreadable audio, a missing audio library or zero examples fail; the audio files are only read. Cache (`features.cache.enabled`): one `.npy` per sample in `<dir>/<extractor signature>/`, keyed by sample id, audio path and the audio file's size and modification time; a changed extractor gives a new folder, a damaged file is rebuilt, an unwritable folder is skipped. `features.num_workers` threads fill the cache (0 = one at a time).
- Known limits: the cache is large: one example is 0.96 MB, so the train split of the Daf Yomi data needs about 6.7 GB of disk. Without the cache the features are recomputed at every epoch (about 21 ms per item). The threads speed up only the cache fill, not reading items one by one (the trainer's own data loader decides that). No augmentation, no shuffling unless `features.shuffle` is true.
- Tests: `tests/test_featurepipeline.py` (32 tests: the 9 acceptance tests of the brief plus laziness, cache invalidation, damaged and unwritable cache, threads, the collator with the model, config; the stored tiny model and synthetic wav files, offline).
- Config keys (section `features`; `features.shuffle` and `features.seed` were added, see DECISIONS 38):
  - `features.num_workers` [DEFAULT: 0]
  - `features.cache.enabled` [DEFAULT: false], `features.cache.dir`
  - `features.max_dropped_ratio` [DEFAULT: 0.05]: safety gate for dropped labels
  - `features.downmix` [DEFAULT: `mean`]
  - `features.shuffle` [DEFAULT: false], `features.seed` [DEFAULT: none = `runtime.seed`]
  - `text.train.timestamps` (owned by TextNormalizer; read here)
  - from ModelLoader's output (not Config): sampling rate, window length, mel bins, label limit

## C6 TrainerCore
- Status: built (wave 1); run once on real Daf Yomi clips with `models/whisper-tiny` (8 steps, 2 threads; the temporary output was deleted)
- Purpose: Run ONE training run on a given model and datasets with given hyperparameters, and return the trained model folder and the record of the run.
- How to use (project root, PowerShell, `$env:PYTHONPATH = "src"`; needs the data of C1, `models/whisper-tiny`; about 40 seconds; the output folder must be new or empty, so delete `data/c6_example_run` before running it again):
  ```
  python -c "from pathlib import Path; from stt.config import SectionRegistry, load_config; from stt.datasource import DataConfig, create_data_source; from stt.splitter import SplitConfig, resolve_split; from stt.textnormalizer import TextConfig, TextNormalizer; from stt.modelloader import ModelConfig, create_loader; from stt.featurepipeline import FeaturesConfig, FeaturePipeline; from stt.trainercore import TrainConfig, TrainerCore, TrainRequest, hyperparams_from_config; c = load_config(Path('configs/data_gmara.yaml'), overrides=['model.path=models/whisper-tiny', 'train.cpu_threads=2', 'train.hyperparams.batch_size=2', 'train.hyperparams.grad_accum_steps=1', 'train.hyperparams.max_steps=2', 'train.hyperparams.warmup_steps=0', 'train.log_every_steps=1', 'train.eval_every_steps=2', 'train.save_every_steps=2'], registry=SectionRegistry({'data': DataConfig, 'split': SplitConfig, 'text': TextConfig, 'model': ModelConfig, 'features': FeaturesConfig, 'train': TrainConfig})); s = create_data_source(c.section('data', DataConfig), Path('data/load_out')).load().samples; sp = resolve_split(c.section('split', SplitConfig), c.runtime.seed, s); n = {x.sample.id: x for x in TextNormalizer(c.section('text', TextConfig)).normalize(s).samples}; m = create_loader(c.section('model', ModelConfig)).load(); p = FeaturePipeline(c.section('features', FeaturesConfig), m); tr = p.build([n[x.id] for x in sp.splits['train'][:4]], 'train').dataset; va = p.build([n[x.id] for x in sp.splits['val'][:2]], 'val').dataset; t = c.section('train', TrainConfig); r = TrainerCore(t).train(TrainRequest(m, tr, va, p.build_collator(), hyperparams_from_config(t, c.runtime.seed), Path('data/c6_example_run'))); print(r.stopped_reason, r.best_step, r.model_dir)"
  ```
  Output: `max_steps 2 data\c6_example_run\model`. In code: `result = TrainerCore(train_config).train(TrainRequest(model, train_dataset, val_dataset, pipeline.build_collator(), hyperparams, output_dir))` (optional fields: `on_eval`, `source_config_ref`, `model_name` = the prefix of the model folder, which becomes `<model_name>_step_<step:05d>` instead of `model` (owner decision, O70), and `stop_flag`, a callable checked after every step) (the model from C5, the datasets and the collator from C4; `hyperparams_from_config(train_config, runtime_seed)` makes the hyperparameters from the config). In a run the section is `train` (class `TrainConfig`); example YAML: `configs/train_default.yaml`.
- Inputs and outputs: `TrainRequest` (model, Train and Val `FeatureDataset`, collator, `TrainHyperparams`, output folder, optional `on_eval` stop callback, optional `source_config_ref`) -> `TrainResult` (`model_dir`, `history` of `StepMetrics`, `best_step`, `best_val_loss`, `stopped_reason` = `max_steps` | `early_stopping` | `time_limit` | `stopped_by_callback`, `timings`, `freeze_report`, `seed`). One step = one optimizer update = `batch_size * grad_accum_steps` examples. Errors: `TrainerConfigError` (exit code 2: wrong values, unknown policy or sink, output folder not empty, resume without a matching checkpoint) and `TrainerCoreError` (exit code 4: NaN or infinite loss, out of memory, damaged saved state).
- The run folder: `run_info.json` (what the run was started with), `metrics.csv` (the LIVE file, one row per record, closed after each write: open it in Excel at any time or follow it with `Get-Content <folder>/metrics.csv -Wait`), `checkpoints/checkpoint-<step>/` (only the latest and the best are kept), `model/` (the returned model: weights and settings of the best checkpoint in safetensors, the tokenizer and feature-extractor files copied unchanged from the base model folder, and `pipeline_model_info.json`; it loads through the ModelLoader, which reads `source_config_ref` from the info file).
- Rules kept: only the Val LOSS is computed in the loop (no generation); a record every `log_every_steps` and at step 1; the Val loss every `eval_every_steps` and once at the end; a checkpoint every `save_every_steps`, at every new best Val loss, at every stop and at the end; the returned model is the best checkpoint by Val loss (the latest one when no evaluation happened); early stopping after `patience` evaluations without an improvement of more than `min_delta` (a warning when the best loss is the first evaluation); the time cap (`max_runtime_minutes`, counted per call) saves and stops with `time_limit`; a NaN or infinite loss, gradient norm or Val loss stops the run, no checkpoint is saved after it (the weights are also checked for NaN before every save), and the run fails naming the step, the learning rate and the last good checkpoint; resume needs a complete checkpoint and the same model, freeze policy, hyperparameters and data sizes (a resumed run has exactly the losses of an uninterrupted one); a non-empty output folder is never overwritten; the seed is applied to Python, numpy, torch and the data order and written to `run_info.json` and `pipeline_model_info.json`; the thread limit is applied during the run and the previous thread count is restored afterwards; a sink that cannot write (for example the file is open in Excel) logs a warning once and the training goes on.
- Known limits: ONE implementation (`Seq2SeqTrainer`, in `hf_trainer.py`, the only module that imports torch and transformers); no LoRA; float32 on CPU; the data loader runs in the training process (0 workers, fixed, because of Windows spawn, O2). The Val loss is computed on the WHOLE Val set at every evaluation: 0.415 s per Val example for whisper-tiny with 2 threads, so the 875 examples of the Daf Yomi Val set cost about 6 minutes per evaluation (see O63). A checkpoint holds the weights, the optimizer states and more, about 3 times the model size, and the run folder keeps the latest AND the best checkpoint plus the returned model (609 MB for whisper-tiny after one checkpoint; about 40 GB for a 1.5B-parameter model, see O64). Measured with 2 threads: 1.24 s per training example (whisper-tiny, batch 4: 4.97 s per step), including the features computed on demand (21 ms of it). The logged learning rate of step 1 is 0 when there is a warmup (the schedule starts at 0). A graceful stop is supported through the optional `TrainRequest.stop_flag` (checked after every step; the run saves a checkpoint and stops with `stopped_by_callback`).
- Tests: `tests/test_trainercore.py` (35 tests: the 12 acceptance tests of the brief plus resume equality, the fewer-steps-than-an-evaluation case, the failing sink, out of memory, the CSV resume rule, the monitor, the config and the import rules; the stored tiny model and synthetic data, offline, about 40 seconds).
- Config keys (section `train`; no key was added to the brief):
  - `train.freeze_policy` [DEFAULT: `none`; registered: `none`, `freeze_encoder`]
  - `train.hyperparams.*`: `learning_rate` [DEFAULT 1e-5], `max_steps` [placeholder 100], `batch_size` [DEFAULT 4], `grad_accum_steps` [placeholder 4], `warmup_steps` [placeholder 10], `weight_decay` [DEFAULT 0.0], `max_grad_norm` [DEFAULT 1.0], `seed` [DEFAULT none = `runtime.seed`]; the placeholders wait for the owner (O39)
  - `train.log_every_steps` [DEFAULT: 10], `train.eval_every_steps` [DEFAULT: 50]
  - `train.save_every_steps` [DEFAULT: 100]
  - `train.early_stopping.patience` [DEFAULT: 5 evaluations], `train.early_stopping.min_delta` [DEFAULT: 0.0]
  - `train.max_runtime_minutes` [DEFAULT: null = no cap]
  - `train.cpu_threads` [DEFAULT: null = half of the cores, at least 1]
  - `train.resume` [DEFAULT: false]
  - `train.output_dir` (per experiment; set by the RunManager later; the caller copies it into `TrainRequest.output_dir`)
  - `train.sinks` [DEFAULT: `csv`, `console`]
  - The search space of the hyperparameters belongs to the HyperparamSearcher, not here.

## C7 Evaluator
- Status: built (wave 1); checked once on 16 real Daf Yomi VAL clips with `models/whisper-tiny` (baseline and a 4-step trained model; the Test split was never used)
- Purpose: Transcribe an evaluation set with a given model, score the transcripts against the references, and compare two scored results with a statistical test.
- How to use (project root, PowerShell, `$env:PYTHONPATH = "src"; $env:PYTHONIOENCODING = "utf-8"`; needs the data of C1 and `models/whisper-tiny`; scores 4 Val clips with 2 threads, about 40 seconds):
  ```
  "from pathlib import Path; from stt.config import SectionRegistry, load_config; from stt.datasource import DataConfig, create_data_source; from stt.splitter import SplitConfig, resolve_split; from stt.textnormalizer import TextConfig, TextNormalizer, build_profile; from stt.modelloader import ModelConfig, create_loader; from stt.featurepipeline import FeaturesConfig, FeaturePipeline; from stt.evaluator import EvalConfig, EvalItem, EvalRequest, Evaluator, make_decode_settings; from stt.evaluator.hf_transcriber import HfTranscriber; c = load_config(Path('configs/data_gmara.yaml'), overrides=['model.path=models/whisper-tiny', 'eval.batch_size=2'], registry=SectionRegistry({'data': DataConfig, 'split': SplitConfig, 'text': TextConfig, 'model': ModelConfig, 'features': FeaturesConfig, 'eval': EvalConfig})); s = create_data_source(c.section('data', DataConfig), Path('data/load_out')).load().samples; sp = resolve_split(c.section('split', SplitConfig), c.runtime.seed, s); n = {x.sample.id: x for x in TextNormalizer(c.section('text', TextConfig)).normalize(s).samples}; m = create_loader(c.section('model', ModelConfig)).load(); p = FeaturePipeline(c.section('features', FeaturesConfig), m); v = [n[x.id] for x in sp.splits['val'][:4]]; ds = p.build(v, 'val').dataset; e = c.section('eval', EvalConfig); t = c.section('text', TextConfig); items = [EvalItem(x.sample.id, x.sample.group_id, x.eval_text, x.sample.audio_path, {'duration': x.sample.duration_sec}) for x in v]; r = Evaluator(e, HfTranscriber(2), c.runtime.seed).evaluate(EvalRequest(items, ds, m, 'val', t.eval_version, build_profile(t, 'eval'), make_decode_settings(e, m, items))); print(r.summary())"
  ```
  Output (the numbers vary little): `val (eval_version 1, model whisper-tiny): 4 files, 191 words, WER 1.4921, mean per-file WER 1.5286, CER 0.8728; files with WER under 10%: 0.0%, under 20%: 0.0%; flags: ...`. In code: `result = Evaluator(eval_config, HfTranscriber(threads), seed).evaluate(EvalRequest(items, dataset, model, "val", eval_version, eval_profile, make_decode_settings(eval_config, model, items)))` and `Comparator(eval_config, seed).compare(baseline_result, candidate_result)`. The `items` are `EvalItem`s (sample id, group id, `eval_text` as the reference, audio path, attributes such as `duration`); the dataset and the model come from C4 and C5; the eval profile from `build_profile(text_config, "eval")` (C2). In a run the section is `eval` (class `EvalConfig`); example YAML: `configs/eval_default.yaml`.
- Inputs and outputs: `EvalRequest` -> `EvalResult` (corpus `wer`, `wer_mean_per_file`, `cer`, `n_files`, `n_words`, `share_files_wer_below`, `files` = one `FileScore` per file with reference, normalized and raw hypothesis, word errors, WER, CER, flags and the attributes, `decode_settings`, `seconds`, `flags` (`empty_output`, `hit_length_cap`, `excluded_empty_reference`), `excluded_ids`, `review`). `Comparator.compare(baseline, candidate)` -> `ComparisonResult` (`delta_wer` = baseline minus candidate, so positive is better; `ci_low`, `ci_high`, `n_groups`, `verdict` = `improved` | `no_clear_difference` | `worse` | `not_comparable` | `insufficient_groups`, and one plain sentence `explanation`). The review list (`result.review.entries`) holds the worst files first and then random files with reference, hypothesis, a word-level difference (`[a->b]` replaced, `[-a-]` missing, `[+b+]` extra), the WER and the audio path, so that you can listen. Errors: `EvaluatorConfigError` (exit code 2: unknown metric or method) and `EvaluatorError` (exit code 4: all hypotheses empty, hypotheses that do not match the items, different file sets, out of memory, a missing `duration`, the sanity limit).
- Rules kept: WER on `eval_text` at corpus level (all words together), CER next to it (characters, spaces included), the mean of the per-file WER too; hypotheses go through the SAME eval profile (the raw text is kept next to it); every file score is returned; an empty output is a full error and flagged; an empty reference is excluded, listed and counted; a different `eval_version` is `not_comparable` with no numbers; the comparison resamples whole GROUPS (the same groups for both models); the language and task come from the loaded model, no timestamps are generated; an output that uses its whole token cap without ending is flagged `hit_length_cap` and cut at the cap; our own metric code (no `jiwer`, no `evaluate.load`); the Evaluator writes no file and prints nothing; the Test split is not special to this component, the Orchestrator enforces that the Test is scored once, for the winner.
- Known limits: ONE Transcriber (`hf_transcriber.py`, the only module that imports torch). Decoding is greedy by default and sequential in batches of `eval.batch_size`. Measured with whisper-tiny, 2 threads, batch 2: about 1.9 to 2.1 seconds per 30 s clip, so 875 clips take about 27 to 31 minutes (a large model will take far longer, O66). The tiny model has a WER above 1 on Daf Yomi Hebrew (it inserts more words than the reference has), as expected for the baseline. The CER of two 1000-character texts costs a few milliseconds in pure Python; scoring 875 files takes seconds. `eval.generation.max_tokens_per_audio_sec` is 8.0 by default (owner decision, O67); it needs the `duration` attribute of every item (set `null` to switch the cap off). The name of a trained model in the result is the name of its folder, so the RunManager should name the model folders (the C6 folder is called `model`).
- Tests: `tests/test_evaluator.py` (23 tests: the 12 acceptance tests of the brief with a fake Transcriber, the adapter on the stored tiny model, the config, the failures and the import rules; offline, about 7 seconds).
- Config keys (section `eval`; no key was added to the brief):
  - `eval.metrics` [DEFAULT: `wer`, `cer`; `wer` is required]
  - `eval.generation.num_beams` [DEFAULT: 1], `eval.generation.max_new_tokens` [DEFAULT: the model's label limit], `eval.generation.max_tokens_per_audio_sec` [DEFAULT: 8.0, O67; null = off]
  - `eval.batch_size` [DEFAULT: 4]
  - `eval.thresholds.wer_below` [DEFAULT: 0.10, 0.20]
  - `eval.significance.method` [DEFAULT: `paired_group_bootstrap`], `.confidence` [DEFAULT: 0.90], `.one_sided` [DEFAULT: true], `.n_resamples` [DEFAULT: 2000], `.min_effect_abs` [DEFAULT: 0.0], `.min_groups` [DEFAULT: 5]
  - `eval.review.n_random` [DEFAULT: 20], `eval.review.n_worst` [DEFAULT: 10]
  - `eval.sanity.max_baseline_wer` [DEFAULT: null = off]
  - the seed comes from `runtime.seed` (given to the Evaluator and the Comparator by the composition root)

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
- Status: built (wave 1); checked once with three experiments, a resume, the lock and a stale lock in a temporary folder (deleted afterwards)
- Purpose: Create and manage numbered experiment folders (config snapshots, diffs, status, lock, logs, and the reports written into them).
- How to use (project root, PowerShell, `$env:PYTHONPATH = "src"`; no data and no model needed; about 2 seconds):
  ```
  python -c "from pathlib import Path; from stt.config import load_config; from stt.runmanager import ExperimentSpec, ExperimentStore, RunConfig; s = ExperimentStore(RunConfig(experiments_dir=Path('data/c12_example'))); h = s.create(ExperimentSpec(load_config(None, env_file=None), None, '0.0.0', {}, '2026-10-05T12:00:00')); h.set_status('running'); print(h.experiment_id, h.status().status, h.paths.root)"
  python -m stt.runmanager show exp_0001 --dir data/c12_example
  Remove-Item -Recurse -Force data/c12_example
  ```
  Output: `exp_0001 running data\c12_example\exp_0001`, then the overview of the experiment (status, stages, and the paths of the config, status, log, metrics and reports). The last line deletes the example folder (the config file is read-only, `-Force` is needed). In code: `store = ExperimentStore(run_config)`, `handle = store.create(ExperimentSpec(config, parent_id, code_version, library_versions, created_at))`, `with handle.lock(): ...`, `handle.set_status("running")`, `handle.mark_stage("prepare", "ok", fingerprint)`, `handle.write_report("report.html", html)`, and for a resume `store.open("exp_0003", config)` (fails when the config differs from the stored one). In a run the section is `run` (class `RunConfig`); example YAML: `configs/run_default.yaml`.
- WHERE TO FIND EVERYTHING OF EXPERIMENT N, IN ONE PLACE: the folder `experiments/exp_000N/` (the folder is `run.experiments_dir`, default `experiments`). Run `python -m stt.runmanager show exp_000N --dir experiments` (PowerShell, `$env:PYTHONPATH = "src"`): it prints the status, the reason, the finished stages and the full path of each of the following, with the last log lines and the last metrics record. By hand: the config that ran is `config_full.yaml` and what differs from the parent (or from the defaults) is `config_diff.yaml`; the status, the reason and the finished stages are `status.json`; the seed, versions and parent are `run_info.json`; the log is `logs/run.log` (everything, DEBUG); the live metrics of each trial are `metrics/<trial>/metrics.csv` (open in Excel or follow with `Get-Content <file> -Wait`); the reports are in `reports/`; the finalist model folders are in `candidates/`; `python -m stt.runmanager list --dir experiments` lists all experiments with their status. A lock file `run.lock` exists only while a run is active.
- Inputs and outputs: `ExperimentSpec(config, parent_id, code_version, library_versions, created_at)` -> `ExperimentHandle` with `paths` (`ExperimentPaths`), `status()`, `set_status(status, reason)`, `mark_stage(stage, status, fingerprint)`, `stage_mark(stage)`, `run_info()`, `config_ref` (the value for `TrainRequest.source_config_ref`: `exp_0004/config_full.yaml`), `metrics_dir(name)`, `candidate_dir(name)`, `write_report(name, content)`, `lock()` and `locations()`. `ExperimentStore`: `create`, `open`, `list_experiments`, `propose_cleanup(CleanupPolicy)`, `delete(paths)`. Pure helpers: `diff_configs(base, new)`, `model_folder_name(experiment_id, kind, trial, step)`. `LogSetup(run_config).configure(handle)` and `.close()`. Statuses: `created`, `running`, `completed`, `failed`, `stopped` (the last one is prepared for the graceful stop, O65); stage statuses `ok`, `skipped`, `failed`. Errors: `RunManagerError` (the message names the path or the experiment and the fix).
- Rules kept: numbers, never timestamps, in folder names (the next number is one above the highest existing, created with an exclusive `mkdir`, so two processes never get the same number; a gap is never refilled); dates only inside files and passed in; `config_full.yaml` is always stored and read-only (the `run.config_readonly` key) and a resume with a different config FAILS showing the first differing keys; `config_diff.yaml` holds only the changed keys against the parent, or against the defaults of the schemas without a parent; `status.json` and the lock are written atomically or exclusively; a failed run keeps its folder (status `failed` with the reason) and a resume reopens it; the lock is a heartbeat file (no process id probing); a stale lock FAILS with the removal instruction and is never removed; `write_report` refuses any path that leaves `reports/`; nothing is deleted automatically, a cleanup only PROPOSES (folders of failed experiments, finalist folders or checkpoint folders of completed ones, optionally older than N days with a `now` that is passed in) and `delete` removes exactly the folders it is given after checking all of them (inside the experiments folder, folders only, no control file, no experiment with a fresh lock; the model registry is outside and cannot be reached); one log file per experiment (DEBUG) and only INFO and above on the console (stderr); paths with spaces and Hebrew characters work; a path that is too long for Windows gives a warning with the fix.
- Known limits: one machine at a time per experiment (the lock is a file, not a network lock). `status.json` has no time stamps (dates are passed in; the file system shows when it changed). The cleanup does not know about the registry because the registry is not under the experiments folder. The long-path warning uses a fixed margin of 90 characters for the longest file inside an experiment. The Windows long-path behavior on the closed machine is not tested (see the brief's OPEN item).
- Tests: `tests/test_runmanager.py` (28 tests: the 11 acceptance tests of the brief including concurrent creation from 6 threads and from 3 separate Python processes, plus the missing parent, damaged status, number gaps, read-only handling on delete, the model folder names and the helper command; offline, about 3 seconds).
- Config keys (section `run`; `run.prevent_sleep` was added to `RunConfig` for the Orchestrator, C13; `model_name_prefix(experiment_id, kind, trial)` was added to the naming helper, the prefix without the step for `TrainRequest.model_name`):
  - `run.experiments_dir` [DEFAULT: `experiments`]
  - `run.parent` [DEFAULT: none]
  - `run.id_digits` [DEFAULT: 4]
  - `run.config_readonly` [DEFAULT: true]
  - `run.lock_heartbeat_seconds` [DEFAULT: 30], `run.lock_stale_minutes` [DEFAULT: 10]
  - `run.log.console_level` [DEFAULT: INFO], `run.log.file_level` [DEFAULT: DEBUG]

## C13 Orchestrator and CLI
- Status: MINIMAL VERSION built (wave 1): the stages `prepare`, `train`, `evaluate` and the commands `prepare`, `train`, `evaluate`, `smoke`, `check`. Not yet: `run`, `search`, `select_and_register`, `export`, `report`, `learning-curve`, `benchmark`, `cleanup`, `config-docs`, `registry` (later waves).
- Purpose: Wire the components together and run the whole pipeline, or any single stage of it, from one command with full automation and a clear exit code.
- How to use (project root, PowerShell; set the path to the sources once per window with `$env:PYTHONPATH = "src"`):
  ```
  python -m stt smoke                          # the whole chain on synthetic data and the tiny test model, about 15 seconds
  python -m stt check                          # what is fine and what is not on this machine (add --config FILE to check its folders)
  python -m stt prepare  --config my.yaml      # load, split, normalize, filter; the reports
  python -m stt train    --config my.yaml      # prepare + one training run
  python -m stt evaluate --config my.yaml      # prepare + baseline (and trained) model on the Val split
  python -m stt train --config my.yaml --experiment exp_0004     # continue an experiment: finished stages are skipped, an interrupted training is resumed
  ```
  Shared options: `--config FILE`, `--set a.b=value` (repeatable, later wins), `--parent exp_0003` (the new experiment extends that one), `--experiment exp_0004`; `evaluate` also has `--split NAME`. `python -m stt <command> --help` shows an example. Every command creates a numbered experiment folder and prints ONE JSON summary at the end (the only thing on stdout: experiment id and folder, status, exit code, the result of each stage, the key numbers, the paths); the log lines go to the console (stderr) and to `logs/run.log`. After a run it prints where the experiment is and the commands to look at it.
- LOOK AT AN EXPERIMENT (see C12 for the one-place list of files): `python -m stt.runmanager list --dir experiments`, `python -m stt.runmanager show exp_0001 --dir experiments`, `python -m stt check`.
- Inputs and outputs: `Orchestrator(config, services, stage_registry, stop_flag, sleep_guard).run(RunPlan(stages, experiment, options)) -> RunOutcome(exit_code, verdict, experiment_id, stage_results, status, message)`. `PipelineStage` (Protocol): `name`, `depends_on`, `config_slice(config)`, `run(ctx) -> StageResult(status, summary, artifacts, error)`. The stages use the services only through Protocols (`DataService`, `ModelService`, `TrainingService`, `EvaluationService`); `wiring.py` is the only module that names concrete classes. Plans: `prepare` = [prepare]; `train` = [prepare, train]; `evaluate` = [prepare, evaluate]; `smoke` = [prepare, train, evaluate].
- What the stages do: `prepare` loads, splits, normalizes and filters the data and writes `load_report.txt`, `split_report.txt`, `text_report.txt`, `filter_report.txt` and `rejected.csv` into `reports/`. `train` runs one trial: the output folder is `candidates/exp_NNNN_trial_TTT/`, the live metrics are `metrics/trial_TTT/metrics.csv`, the model folder is named `exp_NNNN_trial_TTT_step_SSSSS` and loads through the ModelLoader (`source_config_ref` = `exp_NNNN/config_full.yaml`); a trial folder with a complete checkpoint is RESUMED, otherwise a new trial number is used (nothing is overwritten). `evaluate` scores the baseline model on the Val split and, if the experiment has a trained model, the trained model too, compares them and writes `evaluation_<baseline|trained>.json`, `predictions_<...>.csv` (one row per file), `review_<...>.txt` (the human review list) and `comparison.json` (a stop-gap until the ReportBuilder). The Orchestrator builds the per-file attributes for the Evaluator: `duration`, `ref_words`, `label_tokens`, numeric `extra` fields (`extra_<name>`) and text flags (`flag_<name>`). Every finished stage writes a completion marker (with a fingerprint of its config slice and of the stages it depends on) into `status.json` and its result into `reports/stage_results/<stage>.json`.
- Rules kept: unknown stage or command names fail and list the registered ones; a rerun skips a stage whose marker is valid, and a changed config slice reruns that stage and every later one; the first failed stage stops the run (the later stages are listed as skipped, nothing is retried) and the status becomes `failed` with the reason; exit codes 0 success, 2 config error (also a locked experiment, a missing parent, a config that differs on resume, the refused Test split), 3 data error, 4 training or evaluation failure, 130 interrupted (1 for an unexpected bug); the TEST SPLIT IS REFUSED by the evaluate stage and the message says so (it is scored once, for the winner, through the Selector in Wave 2; `smoke` checks that the refusal happens); Ctrl+C: the first press sets the stop flag, the training saves a checkpoint and stops after the current step, the status becomes `stopped`, the exit code is 130 and the same command with `--experiment` resumes; the second press stops at once (a long evaluation does not look at the flag); sleep prevention on Windows is on by default for the commands (`run.prevent_sleep`, standard library only, no admin rights); no interactive prompt, no network.
- `check`: Python version, every required package (from the installed metadata), the model and data folders of the config (when `--config` is given), free disk, memory, cores, the code version and whether `MANIFEST.sha256` exists. It lists ALL problems at once (exit code 2) and passes on a healthy machine (exit code 0). The code version is the constant `stt.__version__` (owner decision, O71); when a `MANIFEST.sha256` exists the first 8 characters of its hash are appended (the hook for the later hash of the sources; verifying the sources is the job of `tools/make_manifest.py`, not built yet).
- `smoke`: writes 12 synthetic 3 s sine-tone clips with random Hebrew transcripts (each its own group) and a manifest into `smoke.work_dir` (default `data/smoke`), points the data, the model (`smoke.model_dir`, the tiny test model) and `train.hyperparams.max_steps` (`smoke.steps`) at them, runs prepare, train and evaluate, then checks the Test-once rule. The gates of the Splitter are relaxed in `configs/smoke.yaml` on purpose. It proves the MECHANICS, not the quality (the verdict is `insufficient_groups` because Val has one clip). Without the test model folder it fails and names `tools/make_test_model.py`. It leaves a real numbered experiment in `experiments/`; delete a smoke experiment by hand when you do not need it (nothing is deleted automatically).
- Known limits: the minimal chain has one training run per experiment and no search, selection, registry, export or report builder. The prepared data is recomputed by every command (it is deterministic; the load of the real Daf Yomi data takes about 10 seconds) and not stored. The baseline is evaluated again in every experiment. The evaluation does not look at the stop flag (the first Ctrl+C is honored after the evaluation, the second one stops at once). `smoke` uses the process's current folder for its relative paths: run it from the project folder. The Windows sleep guard was tested with the system call faked (the real call is made on Windows).
- Tests: `tests/test_orchestrator.py` (38 tests: fake stages and services for the order, markers, failures, exit codes, interrupt and resume; the real smoke chain with the tiny model including the interrupt and resume of a training trial, the rerun that skips, the refused Test split, the help examples, `check`, the code version and the import rules; offline, about 15 seconds).
- Config keys (sections `pipeline`, `run`, `smoke`; NEW keys marked):
  - `pipeline.stages` [DEFAULT: `prepare`, `train`, `evaluate` for the minimal chain; the brief's later default adds `search`, `select_and_register`, `export`, `report`]
  - `pipeline.skip_completed` [DEFAULT: true]
  - `pipeline.eval_split` [NEW, DEFAULT: `val`], `pipeline.test_split` [NEW, DEFAULT: `test`]
  - `run.prevent_sleep` [DEFAULT: true]
  - `smoke.n_samples` [DEFAULT: 12], `smoke.duration_sec` [DEFAULT: 3], `smoke.steps` [DEFAULT: 3], `smoke.model_dir` [DEFAULT: `tests/assets/tiny_whisper`], `smoke.work_dir` [NEW, DEFAULT: `data/smoke`]
  - later: `learning_curve.presets`, `learning_curve.params_from`, `benchmark.steps`

## T Standalone tools
- Status: PARTLY built (Step 0): `fetch_model`, `fetch_data`, `fetch_wheels`, `make_test_model` and the extra `inspect_audio` done. `make_manifest` is not built yet.
- Purpose: Prepare, outside the pipeline and with network access allowed ONLY here, everything the offline pipeline needs: models, data, packages, the tiny test model and the code manifest.
- How to use (project root, project virtual environment active; every script has a "How to run" header):
  - `python tools/fetch_model.py --repo openai/whisper-tiny --out models/whisper-tiny`
  - `python tools/fetch_data.py --repo <dataset> --dry-run` (prints the size, downloads nothing), then `python tools/fetch_data.py --repo <dataset> --out data/daf_yomi_raw`
  - `python tools/fetch_wheels.py --requirements requirements.txt --out wheels --lock requirements.lock` (add `--python-version 3.12 --platform win_amd64`; the Python running it must be that version)
  - `python tools/inspect_audio.py --data data/daf_yomi_raw` (read-only report on a downloaded parquet dataset: schema and row counts from the metadata, metadata distributions, `entry_id` overlap between splits, about 50 real audio rows decoded, about 30 raw labels with simple counts; options `--seed 0 --audio-samples 50 --text-samples 30`; changes nothing; this tool is an addition requested by the owner, not in the PRD tool list)
  - `python tools/make_test_model.py --tokenizer-from models/whisper-tiny --out tests/assets/tiny_whisper` (builds the tiny random-weight test model with the REAL tokenizer and extractor files, 3.9 MB, loads it through the ModelLoader and runs a forward pass; `--seed 0`, `--force` to replace; already built and stored with the tests, so you only run it again to change the model)
  - Run the tool tests: `python -m unittest discover -s tests -v`
- Inputs and outputs: see the brief in `docs/PRD.md`. `fetch_model` and `fetch_data` write the folder plus `fetch_info.json`; they are idempotent (a finished folder is reused, an unfinished one is rebuilt). `fetch_wheels` writes `wheels/` and `requirements.lock`. Exit codes: 0 OK; 1 failure; `fetch_wheels` also 2 = a wheel is missing for the target, 3 = the offline install test failed.
- Known limits: only public repositories (anonymous download). `fetch_model` downloads only safetensors weights and the small config and tokenizer files. `fetch_wheels` takes only binary wheels (`--only-binary`), so a package that exists only as source fails with the package name. The offline install test runs on the home machine, with the same Python version as the target.
- Tests: `tests/test_tools_fetch.py`, `tests/test_fetch_wheels.py` (network faked), `tests/test_inspect_audio.py` (tiny synthetic parquet), `tests/test_make_test_model.py` (the stored tiny model is rebuilt from itself).

