# PRD: Automated Whisper Fine-Tuning Pipeline

## How to read this document
- Goal: a generic, clean "factory" that ingests labeled audio and text, splits it, fine-tunes a Hebrew Whisper model (ivrit-ai / Hugging Face family), evaluates it with WER, and returns one statistically supported winning model, or states that the baseline stays.
- Target environment: a closed (air-gapped) Windows machine, CPU only, no admin rights, no agent present (a weak internal agent may help), possibly no Hugging Face access. Everything runs OFFLINE and NON-INTERACTIVELY. The owner runs it alone.
- Validation first at home on the public Daf Yomi dataset (all of it), then the closed code moves to the work machine with different data (about one tenth of that size, same language, different jargon).
- Component IDs (C0, C1b, ...) are stable labels, not the order. This document is ordered by the pipeline. "Component N" in a brief means the ID CN.
- Status tags: LOCKED = explicitly decided by the owner (binding; changing it needs approval). DEFAULT = a proposed starting value, not binding, always configurable, never hard-coded. OPEN = undecided: the agent presents 2-3 options with trade-offs and a recommendation, and waits. Anything not tagged in a brief is LOCKED.
- Cross-cutting decisions and the decision log are in `docs/DECISIONS.md`; unresolved items in `docs/OPEN_ISSUES.md`; failure handling in `docs/TROUBLESHOOTING.md`; the component map in `docs/COMPONENTS.md`; the build order in `docs/BUILD_ORDER.md`.
- Each brief follows the project format: priority and dependencies, responsibility, interface, typed inputs and outputs, config keys, out of scope, edge cases, "If it fails", acceptance tests, open items.

## Pipeline
```
Config
  -> DataSource -> Splitter -> TextNormalizer -> SampleFilter -> Subsampler
  -> ModelLoader + FeaturePipeline -> TrainerCore
  -> Evaluator (baseline and candidates) -> HyperparamSearcher (Optuna, screening subset, finalists)
  -> Selector (Val decides, Test once) -> ModelRegistry -> ModelExporter -> ReportBuilder
RunManager (numbered experiment folders) and Orchestrator + CLI wrap everything.
Standalone home-side tools (the only code allowed to use the network): fetch model, fetch data, offline wheels, tiny test model, manifest.
```

## Component index
| ID | Component | Wave | Responsibility |
|---|---|---|---|
| C0 | Config | 1 | Load, merge, validate and freeze all settings into one typed, read-only `AppConfig` object. |
| C1 | DataSource | 1 | Load raw labeled audio + text from a LOCAL folder into one uniform list of `AudioSample`, and report what was loaded and what was rejected. |
| C3 | Splitter | 1 | Divide the samples into named splits (default `train`, `val`, `test`) BY GROUP, deterministically, guaranteeing that no group appears in more than one split. |
| C2 | TextNormalizer | 1 | Produce for every sample a training label text and an evaluation reference text from the raw transcript, by applying configurable, ordered normalization steps, and report what was found in the text. |
| C1b | SampleFilter | 1 | Decide, per split, which `AudioSample`s are kept according to configurable rules, and report exactly what was removed and why. |
| C3b | Subsampler | 2 | Select, deterministically and reproducibly, a smaller subset of the samples to the size requested in Config, and report exactly what was selected. |
| C5 | ModelLoader | 1 | Load one Whisper model, its feature extractor and its tokenizer from a LOCAL folder, validate them, and describe what was loaded. |
| C4 | FeaturePipeline | 1 | Convert kept samples into model-ready examples (input features from the audio, label ids from `train_text`) and provide the batch collator. |
| C6 | TrainerCore | 1 | Run ONE training run on a given model and datasets with given hyperparameters, and return the trained model folder and the record of the run. |
| C7 | Evaluator | 1 | Transcribe an evaluation set with a given model, score the transcripts against the references, and compare two scored results with a statistical test. |
| C8 | HyperparamSearcher | 2 | Propose hyperparameter sets, hand each one to a trial runner, record every outcome, and return the ranked candidates within a budget. |
| C9 | Selector | 2 | Decide, from evaluated finalists and the baseline, which model is the winner, or declare that the baseline stays. |
| C10 | ModelRegistry | 2 | Store the selected model together with its metadata as an immutable, numbered entry, and keep an index of all entries. |
| C14 | ReportBuilder | 3 | Turn the results of a run into a self-contained visual report, and into a live progress page while training runs, that show at a glance whether the model succeeded, where it failed and what could be changed. |
| C11 | ModelExporter | 3 | Convert a registered model into a target format, verify the result, and record the export. |
| C12 | RunManager | 1 | Create and manage numbered experiment folders (config snapshots, diffs, status, lock, logs, and the reports written into them). |
| C13 | Orchestrator and CLI | 1 | Wire the components together and run the whole pipeline, or any single stage of it, from one command with full automation and a clear exit code. |
| T | Standalone tools | 3 | Prepare, outside the pipeline and with network access allowed ONLY here, everything the offline pipeline needs: models, data, packages, the tiny test model and the code manifest. |

---
## PRD Component 0: Config

### Priority / order and dependencies
- Order: first. Every other component depends on it. Config depends on no other component.

### Responsibility
Load, merge, validate and freeze all settings into one typed, read-only `AppConfig` object.

### Inputs -> Outputs
- Inputs: path to a YAML file (optionally with `extends: other.yaml`), environment variables
  and a `.env` file, CLI overrides (`--config path.yaml`, `--set a.b.c=value`, repeatable).
- Output: `AppConfig` (pydantic model, frozen). Sections: `data`, `model`, `training`, `eval`, `runtime`.
- Also provides: a function that dumps the fully resolved config to YAML (used later by RunManager).

### Decisions (locked)
1. YAML for experiment parameters, with an explanatory comment for every parameter.
   `extends:` allows inheritance from a base file.
2. `.env` holds machine-specific settings only (paths, online/offline, core counts).
   Read by our own simple parser (lines of `KEY=VALUE`, `#` comments). No new dependency.
   Only `.env.example` is committed.
3. Precedence (lowest to highest): defaults in code < YAML < environment < CLI.
4. Validation with pydantic; YAML parsing with PyYAML. These are the ONLY dependencies of this component.
5. Offline by default. `runtime.online: true` is an explicit switch for running at home.
   In offline mode the component sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`.
6. Config is read-only, makes no network calls, reads no audio, imports no Hugging Face code.
7. Each component owns the schema of its own config section (so adding a component does not
   edit the Config loader). DECIDED in practice: all component briefs follow it and the owner approved them (the owner may still object).
8. `num_proc` and `dataloader_num_workers` default conservatively (0 or 1) because of Windows
   multiprocessing (spawn). Realized as `features.num_workers` [DEFAULT 0], `train.cpu_threads` and `search.parallel_trials` [DEFAULT 1] in the component briefs (DEFAULT, O2).
9. Python version: DEFAULT 3.12 (the conservative choice); confirmed or changed by the clean-venv install test of all packages at home (O1). Owner: any version that works with the libraries available on the closed machine.

### Moved out of this component (recorded as direction, NOT locked)
- CLI subcommands: now specified in the Orchestrator and CLI brief (component 13).
- Experiment numbering (`exp_0001`), `config_full.yaml`, `config_diff.yaml`, parent: now specified in the RunManager brief (component 12).

### Config keys it reads
- `runtime.*` (defined here): `online` (bool, default false), `seed` (int).
  (Consistency fix after the full design review, proposed to the owner: the earlier draft keys `num_proc`,
  `dataloader_num_workers`, `runs_dir` and `log_level` are replaced by the section keys `features.num_workers`,
  `train.cpu_threads`, `run.experiments_dir` and `run.log.*`, so that no setting has two homes.)
- Other sections are defined by the component that uses them.
- Environment variable mapping: `STT__SECTION__KEY` overrides `section.key` (exact prefix to be confirmed in implementation brief).

### Out of scope (must NOT do)
- No reading of audio or datasets, no downloads, no network.
- No creation of run folders, numbering, or diffs (RunManager).
- No CLI subcommands or pipeline flow.
- No interactive prompts.
- No import of torch / transformers / datasets / optuna / scipy.

### Edge cases and failure behavior
- Unknown key in YAML or `--set`: fail with the key name and the closest valid key.
- Wrong type (`--set training.learning_rate=abc`): fail immediately, name the key and expected type.
- Missing required value: fail, name the key and the three places it can be set.
- Split ratios not summing to 1 (when the data section exists): fail.
- Path that must exist does not exist: fail with the path and how to fix.
- `extends` cycle or missing base file: fail with the chain.
- Missing `.env`: allowed (use environment and defaults). Malformed `.env` line: fail with line number.
- Same input always gives the same `AppConfig` (deterministic).

### Acceptance tests (offline, tiny synthetic files)
1. Defaults only -> valid `AppConfig`.
2. YAML overrides defaults; ENV overrides YAML; CLI overrides ENV.
3. `--set a.b=3e-5` converts to the schema type; a bad value fails with an actionable message.
4. `extends` merges base and child; a cycle is detected.
5. `.env` parsing: comments, blank lines, quotes, malformed line.
6. Unknown key fails.
7. Resolved-config dump, reloaded, equals the original `AppConfig`.
8. Offline mode sets the three env variables; online mode does not.
9. No network or Hugging Face import occurs when importing the module (check `sys.modules`).

### Additions after the full design review (owner approved)
1. LOCKED: a generated key reference. A command (`config-docs`, see the Orchestrator brief) writes `docs/CONFIG_REFERENCE.md` from the pydantic models: every key, its type, its default, its status tag (LOCKED / DEFAULT) and its explanation. It is generated from the code, never written by hand, and a test fails when the file is out of date.
2. LOCKED: ready-made presets in a `configs/` folder, built with `extends`: `smoke.yaml` (tiny model and tiny synthetic data), `home_full.yaml` (all Hub data, small model), `work_like.yaml` (one tenth of the data, Subsampler `work_like` preset), and `work.yaml` (a template with the work machine's paths and column mappings to fill in).
3. LOCKED: the Config component itself stays unchanged by these additions; the generator only reads the schemas.

---

## PRD Component 1: DataSource

### Priority / order and dependencies
- Order: second. Depends on Config (`runtime.*`, `data.*` sections). Everything downstream depends on its output type.

### Responsibility
Load raw labeled audio + text from a LOCAL folder into one uniform list of `AudioSample`, and report what was loaded and what was rejected.

### Interface and implementations
- `DataSource` (typing.Protocol): `load() -> LoadResult`.
- Implementations (selected by name through a registry, never by if/else on type names):
  1. `CsvAudioSource`: a folder of audio files plus a CSV/manifest.
  2. `HfParquetSource`: a LOCAL folder of Hugging Face parquet files (downloaded earlier, at home). It first EXTRACTS the audio bytes to files and writes a `manifest.csv` in the standard layout, then reads that manifest through the same code as `CsvAudioSource` (composition, no duplicated logic).
- Downloading from the Hub is NOT part of this component or of the pipeline. It is a separate standalone script (`HubFetcher`, direction only, not locked) run at home.

### Inputs -> Outputs (typed, dataclasses)
- `AudioSample`: `id: str`, `audio_path: Path`, `text: str` (raw, unmodified), `group_id: str`,
  `duration_sec: float`, `source_split: str | None`, `extra: Mapping[str, str | int | float | bool]`.
- `LoadReport`: counts (seen / loaded / rejected), rejection reasons with counts, distributions
  (min / max / mean / percentiles) of numeric `extra` fields, duration distribution, and an
  informational count of `group_id`s that appear in more than one `source_split` (no decision made here).
- `LoadResult`: `samples: list[AudioSample]` (sorted by `id`, deterministic) and `report: LoadReport`.
- Side outputs: `rejected.csv` (id + reason), and for `HfParquetSource` the extracted folder with
  `manifest.csv` and `extract_info.json` (source file names, sizes, row counts).

### Rules (locked)
1. The input data is never modified. Anything derived is written to a new file under the run/data folder.
2. Extra columns are never lost: every non-mapped column is kept in `extra`.
   Nested `metadata` dicts are flattened with the prefix `meta_` (for example `meta_entry_id`, `meta_quality_score`).
3. Types of extra fields come from Config (`data.extra_types`), so CSV text values become real numbers or bools.
4. The sample `id` is deterministic. For parquet: `<source_split>_<row_index:06d>` taken in sorted file order. Source `audio.path` values are NOT unique and must not be used as ids.
5. Extraction is idempotent and crash-safe: audio files are written first, `manifest.csv` is written last
   through a temp file + rename. A manifest present means extraction completed. With `data.extract.overwrite: false` an existing complete extraction is reused.
6. The duration is read from the audio header, never trusted from metadata (the source metadata says 30 for every row, real audio ranges 3-30 s).
7. Text is passed on raw. DataSource only checks that it is not empty after whitespace stripping.

### Config keys it reads (section `data`)
- `data.source.type` (`csv` | `hf_parquet`), `data.source.path` (folder)
- `data.columns.id` (optional), `data.columns.audio`, `data.columns.text`, `data.columns.group`, `data.columns.split` (optional)
- `data.group_fallback` (`parent_dir` | `file_stem` | `fail`) [DEFAULT: `fail`]: what to do when there is no group column. `fail` because a wrong group causes train/test leakage.
- `data.extra_types` (mapping column -> `str|int|float|bool`)
- `data.extract.dir`, `data.extract.overwrite` [DEFAULT: false]
- `data.on_invalid` (`skip` | `fail`) [DEFAULT: `skip`]
- `data.max_rejected_ratio` [DEFAULT: 0.05]: if more than this fraction is rejected, the run FAILS (safety gate against a wrongly configured mapping). The value is a starting point to be reviewed after the first real report.
- Defaults for the Gmara dataset (example only, in the example YAML, not in code): audio=`audio`, text=`transcript`, group=`meta_entry_id`, split=`source_split` from the parquet file/split name.

### Out of scope (must NOT do)
- No splitting into Train/Val/Test (Splitter).
- No quality or length filtering (SampleFilter). Only structural validation.
- No text cleaning, no timestamp-token handling (TextNormalizer).
- No resampling, no spectrograms, no tokenization (FeaturePipeline).
- No network access, no Hub downloads, no model code.
- No loading of audio arrays into memory (paths only).
- No decisions about which samples to use.

### Edge cases and failure behavior
- Missing audio file: reject sample, reason `missing_audio`.
- Empty or whitespace-only transcript: reject, `empty_text`.
- Unreadable or zero-length audio header: reject, `unreadable_audio`.
- Duplicate `id`: FAIL (it means the mapping is wrong), naming the first duplicates.
- Column mapped in Config does not exist: FAIL naming the column and listing the existing columns.
- No group column and `group_fallback: fail`: FAIL with the explanation of why (leakage) and the three ways to fix it.
- Rejected ratio above `data.max_rejected_ratio`: FAIL after writing `rejected.csv`.
- CSV encoding: read as UTF-8 (with BOM tolerated). Hebrew text, Hebrew or spaced Windows paths, and relative paths (resolved relative to the CSV location) must work.
- Parquet folder with no parquet files: FAIL with the path.
- Empty result (0 samples): FAIL.

### If it fails: minimal safe fixes (for the owner and the agent)
| Symptom | Likely cause | Minimal fix (does not change code structure) |
|---|---|---|
| FAIL column not found | work data uses other column names | change `data.columns.*` in YAML |
| FAIL rejected ratio too high | wrong audio path or mapping | check `rejected.csv` reasons; fix `data.source.path` or columns; do NOT raise the ratio just to pass |
| many `unreadable_audio` for mp3 on Windows | audio library cannot decode mp3 | run the home install test again; convert audio to wav before ingestion; do not rewrite the loader |
| FAIL no group column | work data has no speaker/lesson id | set `data.columns.group`, or consciously set `group_fallback` to `parent_dir` / `file_stem` |
| extraction interrupted | crash during extract | delete the extract folder (no manifest means incomplete) and rerun |

### Acceptance tests (offline, tiny synthetic data; wav files generated in the test, parquet built in the test)
1. `CsvAudioSource`: 3 valid rows load, ids stable and sorted, `extra` preserved and typed.
2. Missing audio, empty text, and unreadable audio are rejected with the correct reasons and appear in `rejected.csv`.
3. Duplicate id FAILS with a clear message.
4. Unknown mapped column FAILS and lists the available columns.
5. `max_rejected_ratio` exceeded FAILS; below it passes.
6. `HfParquetSource`: extraction creates files and a manifest, flattens `metadata`, produces deterministic ids; a second run reuses the extraction; a manifest-less folder is re-extracted.
7. Both sources, given equivalent data, return equal `AudioSample` lists (LSP).
8. Hebrew text and a path with spaces / Hebrew characters load correctly.
9. The input files are byte-identical after loading.
10. No network call; `HfParquetSource` imports pyarrow only inside its own adapter module.

### PENDING (not blocking the brief)
- Audio header/duration library (candidate: `soundfile`; mp3 support to be verified by the home install test on Windows).
- Parquet reader (candidate: `pyarrow`, already a dependency of `datasets`; no new dependency intended). Both need explicit approval per project rule 4.

---

## PRD Component 3: Splitter

### Priority / order and dependencies
- Numbering follows the original component table (Config 0, DataSource 1, SampleFilter 1b, TextNormalizer 2, Splitter 3).
- Pipeline order (agreed): DataSource -> Splitter -> SampleFilter. Splitter does not depend on text.
- Depends on Config (`split.*`, `runtime.seed`) and on `AudioSample` from DataSource.

### Responsibility
Divide the samples into named splits (default `train`, `val`, `test`) BY GROUP, deterministically, guaranteeing that no group appears in more than one split.

### Interface and implementations
- `Splitter` (typing.Protocol): `split(samples: Sequence[AudioSample]) -> SplitResult`. Selected by name through a registry.
- Implementations:
  1. `PredefinedSplitter`: uses the split that came with the data (`AudioSample.source_split`); can carve `val` out of `train` by group.
  2. `GroupedRatioSplitter`: exact, balanced assignment of whole groups to splits by the requested ratios.
  3. `HashGroupSplitter`: stable assignment by `sha256(seed + group)`; a NEW group never moves existing groups (for data that grows over time).
  4. `TimeBlockSplitter`: for few long recordings; contiguous time blocks (ordered by `seek`) with a guard gap between splits.
- Persistence is a separate class, `SplitStore` (SRP): `save(result, dir)`, `load(dir, samples) -> SplitResult`, `export_csv(...)`. Splitters never touch files.

### Inputs -> Outputs (typed, dataclasses)
- `SplitResult`: `splits: Mapping[str, list[AudioSample]]`, `dropped_guard: list[AudioSample]` (only `time_block`), `report: SplitReport`, `fingerprint: str`.
- `SplitReport`: per split: samples, groups, total duration, actual ratio vs requested; the largest group's share; group-overlap check result; parameters used; fingerprint.
- Split file (JSON, schema_version included): parameters (mode, seed, ratios, group key), `data_fingerprint`, and the sample ids of each split. CSV export (id, split) on request.
- `data_fingerprint` = sha256 over the sorted lines `id|group_id|duration` of the input samples.

### Rules (locked)
1. HARD CHECKS after every split (violation = FAIL): no group in two splits; union of splits plus `dropped_guard` equals the input; no duplicate ids; no empty split.
2. Deterministic: input sorted by `id` first; seed from Config; hashing with sha256, never Python `hash()`; same input and params give the same result regardless of input order.
3. A saved split file is reused by every experiment. A new split is created only on explicit request (a new `split.version`). Experiments with different split versions are NOT comparable, and the system marks them as such.
4. The group key is configurable without code changes: a column, optionally with a regex (first capture group) applied to that column. A sample for which the key cannot be computed makes the run FAIL listing examples (never silently skipped).
5. The Splitter never filters, subsamples, normalizes, or modifies samples. Only exception: `time_block` guard-gap samples are removed from all splits and listed in `dropped_guard` and in the report.
6. No stratification in v1; the report shows each split's distribution (durations, groups, top groups) so bias is visible.
7. When the data grows: `hash_group` assigns new groups without moving old ones; the other modes FAIL with the instruction to create a new split version.

### Config keys it reads (section `split`)
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

### Out of scope (must NOT do)
- No quality/duration filtering (SampleFilter). No subsampling (Subsampler, component 3b, to be briefed separately).
- No text cleaning (TextNormalizer). No text-duplicate detection between splits in v1 (OPEN, O19).
- No audio reading, no network, no model code.
- No writing files (SplitStore does it).

### Edge cases and failure behavior
- Empty input: FAIL.
- Fewer groups than `min_groups_per_split` can satisfy: FAIL with the group counts and the options (finer key, `time_block`).
- One group larger than the ratio tolerance allows: FAIL (or warn per config), naming the group and its share.
- Requested ratios not summing to 1: FAIL at Config validation.
- `predefined` without `source_split`: FAIL. Group present in more than one predefined split: per `split.on_overlap`.
- `time_block`: samples without the order column FAIL; guard-gap samples dropped and reported.
- Saved split file whose `data_fingerprint` does not match the current data: FAIL with the instruction to create a new split version.
- Regex key not matching a sample: FAIL with examples (see rule 4).
- `hash_group` with very few groups can give uneven splits: the min gates FAIL rather than continuing silently.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL too few groups | group key too coarse for the data size | use a finer key (e.g. one lesson/recording) or `time_block`; do not just lower `min_groups_per_split` |
| FAIL oversized group | one recording dominates the data | choose another key or `time_block`; raise the tolerance only consciously |
| FAIL fingerprint mismatch | the data changed since the split was saved | create a new `split.version`; keep the old one so past experiments stay interpretable |
| test WER suspiciously good | group key too fine (neighbouring content in train) or text duplicated across splits | try a coarser key; check text overlap (O19) |
| test WER jumps between experiments | test has too few groups/samples | enlarge the test share or the number of groups; keep the test fixed |
| FAIL regex did not match | work data has other file/column naming | change `split.group_key.regex` in YAML only |

### Acceptance tests (offline, tiny synthetic data)
1. `grouped_ratio` on 20 synthetic groups: no group in two splits, union equals input, ratios within tolerance.
2. Determinism: shuffled input gives an identical result; a different seed gives a different result.
3. `hash_group`: adding a new group does not move any existing group.
4. `predefined`: uses `source_split`, carves `val` from `train` by group; overlap behaves per `split.on_overlap`.
5. `time_block`: contiguous blocks; guard-gap samples are dropped and reported; none is within the gap in two splits.
6. Min gates, oversized-group gate, and empty-input FAIL with clear messages.
7. Regex group key extracts the right group; a non-matching sample FAILS with examples.
8. `SplitStore`: save/load round trip; fingerprint mismatch FAILS; CSV export matches the JSON.
9. Parametrized invariants test: every registered splitter satisfies the hard checks (LSP).
10. Inputs unmodified; no file access by Splitter; no network.

### OPEN items (agent presents options, owner decides)
- O7: official `eval` split vs our own split (decided from the `entry_id` overlap in the data report).
- O18: group key level: lesson/recording vs content (tractate).
- Optional second test set: held-out groups plus seen groups, to separate "new content" from "new recordings".
- O19: text duplicates between Train and Test (to be decided with TextNormalizer).
- O22: values of the minimum gates after the real data volume is known.

---

## PRD Component 2: TextNormalizer

### Priority / order and dependencies
- Pipeline order: DataSource -> Splitter -> TextNormalizer -> SampleFilter -> FeaturePipeline.
- Depends on Config (`text.*`) and on `AudioSample` from DataSource.
- The Evaluator (future) reuses the SAME eval profile, built from the same Config.

### Responsibility
Produce for every sample a training label text and an evaluation reference text from the raw transcript, by applying configurable, ordered normalization steps, and report what was found in the text.

### Core idea (locked): two texts, not one
- `train_text`: what the model learns to output. Gentle cleaning: remove only what is certainly not speech.
- `eval_text`: the reference for WER. Aggressive cleaning: remove every difference that is not about words.
- The eval profile is also applied to model outputs (baseline and fine-tuned) before WER, so all three texts go through the identical procedure.
- The eval profile is the "measuring stick": it carries a version (`text.eval_version`) that is recorded in every experiment. WER values from different eval versions are NOT comparable, and the system marks them as such.

### Interface and implementations
- `NormalizationStep` (typing.Protocol): `name: str`, `apply(text: str) -> str`. Selected by name through a registry. A new step is a new class, with no change to existing code.
- Initial steps (each a separate class):
  1. `UnicodeClean`: Unicode normalization (NFC), removal of invisible and direction-control characters.
  2. `StripTimestampTokens`: removes Whisper-style tokens such as `<|0.32|>`.
  3. `RemovePunctuation`: removes the characters listed in Config (replaces Hebrew maqaf with a space).
  4. `UnifyQuotes`: maps the geresh/gershayim and quote variants to one form (mapping from Config).
  5. `RemoveNikud`: removes Hebrew vowel and cantillation marks.
  6. `RemoveBracketMarkup`: handles bracket pairs from Config, in mode `chars_only` | `with_content` | `keep`.
  7. `LowercaseLatin`: lowercases Latin letters only.
  8. `CollapseWhitespace`: single spaces, trimmed.
- NOT in v1 (excluded, each would be a new step class): abbreviation expansion, number/digit normalization.
- Consumers that only need to normalize a string (the Evaluator) depend on a plain `Callable[[str], str]`, not on an interface.

### Inputs -> Outputs (typed, dataclasses)
- `TimestampSegment`: `start_sec: float`, `end_sec: float`, `text: str`.
- `NormalizedSample`: `sample: AudioSample` (unchanged; the raw text stays in `sample.text`), `train_text: str`, `eval_text: str`,
  `timestamp_segments: tuple[TimestampSegment, ...] | None` (parsed BEFORE the tokens are stripped, so no information is lost), `flags: tuple[str, ...]`.
- `NormalizeReport`: number of samples carrying each flag; number of samples whose text changed per profile;
  ids of samples that became empty; and the CHARACTER INVENTORY (see rule 7).
- `NormalizeResult`: `samples: list[NormalizedSample]` (input order preserved) and `report: NormalizeReport`.
- Builder: `build_profile(name: str) -> Callable[[str], str]` from Config, used by the Evaluator.
- Flags: `had_timestamps`, `bad_timestamps`, `had_brackets`, `had_digits`, `had_latin`, `had_nikud`, `empty_after_normalization`.

### Rules
1. LOCKED: two profiles (`train`, `eval`) built from ordered step lists in Config. Step order is meaningful (timestamp tokens are stripped before punctuation).
2. LOCKED: the component only applies the configured steps. It never corrects spelling, splits Hebrew prefixes, changes words, or uses a model.
3. LOCKED: deterministic and idempotent: normalizing an already normalized text returns it unchanged.
4. LOCKED: the component never drops samples. A sample that becomes empty is FLAGGED; SampleFilter decides (`min_text_chars` on `eval_text`).
5. LOCKED: no dataset-specific logic in code. Every pattern, pair, character list and mapping lives in Config (the Gmara values live in an example YAML only).
6. LOCKED (owner decision): punctuation is KEPT in `train_text` and removed in `eval_text`. It is a Config switch (the step is simply listed or not).
7. LOCKED: the report always includes a character inventory: every distinct character that is not a Hebrew letter, a digit or a space, with its count in the raw text and in each profile's output (top `text.inventory.top_n`). This is how unexpected symbols in a new dataset become visible without knowing them in advance.
8. LOCKED: the primary WER is computed on `eval_text` (owner approved).
9. DEFAULT: timestamps in the training label are stripped (`text.train.timestamps: strip`). `keep` is NOT implemented in v1 (the owner delegated the decision; timestamps do not count in the final result). It is a documented later extension, and `timestamp_segments` are preserved so nothing is lost. Setting `keep` FAILS at Config validation with an explanation (O3, O48). `eval_text` never contains timestamp tokens.
10. DEFAULT: brackets: remove the bracket characters and keep the content (O25).
11. DEFAULT: unify quotes and geresh in both profiles; lowercase Latin letters in eval only; remove nikud in eval.

### Config keys it reads (section `text`)
- `text.eval_version` [DEFAULT: 1]
- `text.profiles.train.steps` [DEFAULT: `unicode_clean`, `strip_timestamps`, `unify_quotes`, `remove_bracket_markup`, `collapse_whitespace`]
- `text.profiles.eval.steps` [DEFAULT: `unicode_clean`, `strip_timestamps`, `remove_punctuation`, `unify_quotes`, `remove_nikud`, `remove_bracket_markup`, `lowercase_latin`, `collapse_whitespace`]
- `text.train.timestamps` (`strip` in v1; `keep` reserved for a later extension) [DEFAULT: `strip`]
- `text.bracket.pairs` [DEFAULT: `[` `]` and `(` `)`], `text.bracket.mode` [DEFAULT: `chars_only`]; optional per-profile override `text.profiles.<name>.bracket_mode` [DEFAULT: unset = use `text.bracket.mode`]. Recipe for a non-speech marker (O47): train profile `keep`, eval profile `with_content`, so the marker is learned but never counted in the WER. Defaults are unchanged, so nothing happens unless the config asks for it.
- `text.punctuation.chars` [DEFAULT: a standard Hebrew and Latin punctuation set, listed in the example YAML]
- `text.quotes.map` [DEFAULT: ׳ ' ’ unified to one form; ״ " “ ” unified to one form]
- `text.inventory.top_n` [DEFAULT: 50]

### Out of scope (must NOT do)
- No dropping or filtering of samples (SampleFilter).
- No tokenization (FeaturePipeline), no WER (Evaluator), no audio.
- No abbreviation expansion, no number normalization (excluded from v1).
- No detection of identical text between splits (a separate small reporter later, O19).
- No file access, no network, no model code.

### Edge cases and failure behavior
- Text empty after normalization: flag `empty_after_normalization`, keep the sample.
- Malformed timestamp tokens (missing end, non-monotonic): `timestamp_segments = None`, flag `bad_timestamps`; the tokens are still stripped from the text.
- Unknown step name or profile in Config: FAIL listing the registered names.
- A configured bracket pair with an unbalanced opening: the bracket characters are removed, the text is kept, and the sample is flagged `had_brackets`.
- Very long text or text with only symbols: processed normally and flagged as appropriate.
- `text.train.timestamps: keep`: FAIL at Config validation: not implemented in v1 (a later extension).

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| WER high for BOTH baseline and fine-tuned | reference text contains symbols that were not spoken | read the character inventory; add them to `text.punctuation.chars` or `text.bracket.pairs` in YAML (new `eval_version`); do not edit code |
| WER looks good but outputs look wrong | the eval profile is too aggressive | compare with a stricter profile as a new `eval_version`; remember that versions are not comparable |
| the model outputs timestamp tokens | training labels still contain them | check `text.train.timestamps` and the train profile steps |
| the new dataset has markers like `[noise]` | work data uses its own notation | set `text.bracket.mode: with_content` for those pairs, or add the pair |
| the report flags `had_digits` | digits in the work data | ask the owner; a number-normalization step would be a NEW step class, not an edit |
| many `empty_after_normalization` | transcripts hold only timestamp tokens or symbols | check the inventory; let SampleFilter remove them via `min_text_chars` |

### Acceptance tests (offline, tiny synthetic texts)
1. Each step, table-driven, with Hebrew examples (punctuation, geresh variants, nikud, brackets in all three modes, invisible characters, timestamp tokens).
2. Idempotence: `normalize(normalize(x)) == normalize(x)` for every profile.
3. Profile determinism: the same Config gives the same results.
4. Timestamp parsing: segments extracted correctly; malformed tokens give `None` plus the flag, and the text is still stripped.
5. Flags and counts in the report are correct, including the character inventory.
6. `empty_after_normalization` is flagged, and the sample is NOT dropped.
7. Unknown step or profile FAILS and lists the registered names.
8. `build_profile("eval")` applied to a reference and to a hypothesis gives equal results for equal words with different punctuation.
9. `AudioSample` is unchanged and `sample.text` stays raw.
10. Step order matters: a test shows that stripping timestamps after punctuation removal would leave residue (documents the order rule).
11. No file access and no network; Config is the only source of patterns.

### OPEN items (agent presents options, owner decides)
- O3: strip vs keep timestamps; downstream design for `keep`.
- O25: how bracketed markup should be treated in the work data.
- O19: identical text in Train and Test (separate reporter).
- O26 and number normalization: add steps only if the inventory shows the need.

---

## PRD Component 1b: SampleFilter

### Priority / order and dependencies
- Order (REVISED, owner approved option A after the TextNormalizer design): DataSource -> Splitter -> TextNormalizer -> SampleFilter -> FeaturePipeline. The filter must see normalized text, because raw text still contains timestamp tokens that would count as characters.
- Depends on Config (`filter.*`) and on `NormalizedSample` from TextNormalizer (a wrapper around the untouched `AudioSample`, plus `train_text` and `eval_text`).
- ID C1b is a stable label, not the order.

### Responsibility
Decide, per split, which `AudioSample`s are kept according to configurable rules, and report exactly what was removed and why.

### Interface and implementations
- `SampleRule` (typing.Protocol): `name: str`, `required_fields: tuple[str, ...]`, `check(sample: NormalizedSample) -> RuleVerdict` (pass, or fail with a reason). Duration and metadata rules read the wrapped `sample.sample`; text rules read `eval_text`.
- Rules are registered by name (registry); the filter never branches on rule type names.
- Initial rules (each a separate class): `MinDuration`, `MaxDuration`, `MinTextChars`, `MinQualityScore`, `MaxBadSegments`, `RequireGolden`.
- A new rule (for example text-length-to-duration ratio to detect misaligned transcripts) is added as a new class, with no change to filter logic.
- `SampleFilter.filter(samples: Sequence[NormalizedSample], split: str) -> FilterResult`.
  Only the rules whose `applies_to` contains `split` run.

### Inputs -> Outputs (typed)
- `FilterResult`: `kept: list[NormalizedSample]` (original order preserved), `rejected: list[RejectedSample]`
  (`sample_id`, list of failed rule names), `report: FilterReport` (per rule: how many failed; total kept ratio; mode).
- The component returns objects. Writing reports to disk is not its job (see OPEN: report persistence).

### Rules (locked)
1. Two kinds of rules. STRUCTURAL (duration, empty text) apply to ALL splits. QUALITY (`quality_score`, `bad_segments`, `golden`) apply by default to TRAIN only; Val/Test are controlled by Config. Reason: filtering the Test set by quality makes the WER look better than real-world performance.
2. `mode: report` computes and reports what WOULD be rejected, and keeps everything. `mode: apply` actually removes.
3. A rule whose required field is missing from the samples FAILS the run immediately with a clear message, unless the rule is disabled in Config (`enabled: false`). No silent skipping.
4. The component never modifies a sample and never touches files.

### Config keys it reads (section `filter`)
- `filter.mode` (`report` | `apply`) [DEFAULT: `report`]
- `filter.rules.<rule_name>.enabled`, `.applies_to` (list of splits), `.kind` (`structural` | `quality`), plus the rule's threshold:
  - `min_duration` [DEFAULT: 1.0 s], `max_duration` [DEFAULT: 30.0 s, the Whisper window; changed from 30.5 after the FeaturePipeline discussion: the filter removes longer clips, FeaturePipeline never truncates and FAILS if a longer clip arrives. Apply mode is required for this to take effect]
  - `min_text_chars` [DEFAULT: 2], measured on `eval_text` (after normalization, so timestamp tokens and punctuation do not count); optional key `measure_on` (`eval_text` | `train_text`) [DEFAULT: `eval_text`]. For clips labeled with a non-speech marker (O47) the recipe is `measure_on: train_text` for the train split, so they are not removed as "empty"; their empty reference is excluded from the WER by the Evaluator
  - `min_quality_score` [DEFAULT: null = disabled], `max_bad_segments` [DEFAULT: null = disabled], `require_golden` [DEFAULT: false]
  - defaults are deliberately permissive (reject only obviously broken samples) until the first real report is reviewed.
- `filter.min_kept_ratio` [DEFAULT: 0.4]: if a split keeps less than this fraction, the run FAILS (safety gate).

### Out of scope (must NOT do)
- No splitting, no deciding which split a sample belongs to (Splitter).
- No text cleaning (TextNormalizer), no audio decoding or resampling (FeaturePipeline).
- No reading or writing files, no network.
- No choosing thresholds by itself: thresholds come only from Config.
- No changing what a rule means based on the dataset (rules are generic; dataset specifics live in Config).

### Edge cases and failure behavior
- Empty input list: returns an empty result (a split can legitimately be empty only if the Splitter says so); the pipeline decides.
- Split becomes empty after filtering in `apply` mode: FAIL naming the split and the rule that removed the most samples.
- `kept_ratio` below `filter.min_kept_ratio`: FAIL with the per-rule breakdown.
- Unknown rule name in Config: FAIL listing the registered rule names.
- Required field missing (see rule 3): FAIL naming the rule and the field.
- Threshold of the wrong type or a negative duration: FAIL at Config validation.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL required field missing | work data has no `quality_score` etc. | set that rule `enabled: false` in YAML, or map the column in DataSource config; do not edit the rule |
| FAIL kept ratio too low | thresholds too strict for this dataset | run in `report` mode, read the per-rule counts, relax the dominant rule; do not just lower `min_kept_ratio` |
| model weak on noisy audio after training | quality filter removed the noisy kind of audio | compare an experiment with the filter off (config diff only); see OPEN issue on filter thresholds |
| a split becomes empty | filter applied to a small split | restrict the rule's `applies_to` to `train` |

### Acceptance tests (offline, tiny synthetic samples)
1. Each rule passes and fails the right samples (table-driven, one test per rule).
2. `report` mode keeps all samples but reports the correct would-be-rejected counts; `apply` removes them.
3. A rule with `applies_to: [train]` does not run for `test`.
4. Missing required field FAILS with the rule and field names; `enabled: false` removes the failure.
5. `min_kept_ratio` gate FAILS and shows the per-rule breakdown.
6. Order of kept samples equals the input order; input samples are unmodified.
7. Adding a new test-only rule class through the registry works without editing the filter (OCP).
8. Unknown rule name FAILS and lists registered names.

### OPEN items (agent must present options, not choose)
- Text-length-to-duration ratio rule: in v1 or later (adding it later needs only a new class).
- Report persistence: shared ReportWriter vs RunManager (also affects DataSource `rejected.csv`).
- Final threshold values and which splits quality rules apply to: decided from the first reports and an A/B experiment (train with and without the filter, same test set).

---

## PRD Component 3b: Subsampler

### Priority / order and dependencies
- ID C3b is a stable label, not the order.
- Pipeline order (REVISED): DataSource -> Splitter -> TextNormalizer -> SampleFilter -> Subsampler -> FeaturePipeline -> TrainerCore. The Subsampler comes AFTER the filter so that sizes are measured on the audio that is really kept.
- Depends on Config (`subsample.*`, global seed) and on `NormalizedSample` lists per split.
- Purpose (owner): (1) fast runs for checking the pipeline, (2) rehearsing the SMALL-DATA situation of the work machine (about one tenth of the Daf Yomi data), (3) the learning curve (how the WER changes with the amount of data).
- At home, the main runs use ALL the data (owner decision). Subsampling is off unless Config turns it on.

### Responsibility
Select, deterministically and reproducibly, a smaller subset of the samples to the size requested in Config, and report exactly what was selected.

### Interface and implementations
- `Subsampler.subsample(splits: Mapping[str, Sequence[NormalizedSample]], spec: SubsampleSpec) -> SubsampleResult`.
- `SubsampleStrategy` (typing.Protocol): `key(sample: NormalizedSample) -> str` (what is the unit of selection). Implementations selected by name through a registry: `by_sample` (the unit is the sample) and `by_group` (the unit is the whole group, for example a whole lesson). A new strategy is a new class.
- `SubsampleStore`: reads and writes the result as JSON (ids, spec, seed, fingerprint), like the SplitStore.

### Inputs -> Outputs (typed, dataclasses)
- `SubsampleSpec`: `size` (one of `{minutes: float}` | `{samples: int}` | `{fraction: float}` | `all`), `strategy: str`, `scope: str`, `val: str`, `seed: int`, `repeat_index: int`.
- `SubsampleResult`: `splits: dict[str, list[NormalizedSample]]` (input order preserved inside each split), `report: SubsampleReport`, `fingerprint: str`.
- `SubsampleReport`: requested size, achieved size (minutes, samples, groups) per split, overshoot, strategy, scope, seed, and the ids list location.

### Rules
1. LOCKED: determinism and nesting. Every unit gets a rank from a hash of (seed, repeat index, unit key). Units are taken in rank order until the requested size is reached. Therefore the same inputs and seed always give the same subset, the result does not depend on the input order, and a smaller subset is always contained in a larger one (needed for the learning curve).
2. LOCKED: the size unit is minutes of audio (owner approved), with sample counts also possible. `fraction` is added as a third form so that "one tenth of the data" can be expressed directly (agent's addition for the work-like rehearsal).
3. LOCKED: the unit of selection is configurable: single samples or whole groups (owner approved both).
4. LOCKED: `scope: train_only` (DEFAULT): only the Train split is reduced; Val and Test are never touched. `val: fixed` (DEFAULT) keeps Val as is; `val: follow` reduces Val by the same fraction.
5. DEFAULT (agent's choice, the owner asked the agent to decide): `scope: all_splits` is available for the work-like rehearsal: Train, Val and Test are all reduced with the same fraction, by group, so that the whole small dataset behaves like the work data (small test set, few groups, small Val). It is used for rehearsal only, not for the learning curve, because the Test set then changes with the size.
6. LOCKED: no stratification (owner approved).
7. LOCKED: repeated draws are supported (`subsample.repeats`, DEFAULT 1): the same size with different `repeat_index`, so that variance at small sizes can be measured (mean and standard deviation of the WER are computed by the caller).
8. LOCKED: a requested size larger than what exists FAILS, unless the size is `all`. The component never silently returns less.
9. LOCKED: the achieved size may exceed the requested one by at most one unit (the last sample or group). The overshoot is reported. With `by_group` and large groups the overshoot can be big; it is shown, not hidden.
10. LOCKED: the component never modifies samples and never touches files except through `SubsampleStore`.
11. LOCKED: the automatic learning-curve experiment (runs at several sizes and a plot of WER against the amount of data) is a standard experiment of the Orchestrator, not part of this component (owner approved).

### Config keys it reads (section `subsample`)
- `subsample.enabled` [DEFAULT: false]
- `subsample.size` [DEFAULT: `all`]
- `subsample.strategy` [DEFAULT: `by_sample`]
- `subsample.scope` [DEFAULT: `train_only`]
- `subsample.val` [DEFAULT: `fixed`]
- `subsample.repeats` [DEFAULT: 1]
- `subsample.min_train_samples` [DEFAULT: 10]: if fewer samples are selected, FAIL
- `subsample.presets` (named sizes for convenience) [DEFAULT: `smoke` = 20 samples, `tiny` = 30 minutes, `work_like` = fraction 0.1 with `scope: all_splits` and `by_group`, `half` = fraction 0.5, `full` = all]
- seed from the global Config.

### Out of scope (must NOT do)
- No splitting (Splitter), no filtering (SampleFilter), no text or audio processing.
- No running of experiments or learning curves (Orchestrator).
- No choosing a size by itself.
- No network, no model code.
- No changing of the Test split under `train_only`.

### Edge cases and failure behavior
- Requested size larger than available (and not `all`): FAIL with both numbers.
- Fewer than `subsample.min_train_samples` selected: FAIL naming the number and the fix (a larger size).
- A split that becomes empty: FAIL naming it.
- `by_group` selects only one group in Train: warn in the report (a training set with a single group is risky).
- `fraction` outside 0-1 or a negative size: FAIL at Config validation.
- Unknown strategy, scope or preset: FAIL listing the registered names.
- `enabled: false`: returns the input unchanged and says so in the report.
- Stored result with another fingerprint than the current spec and data: FAIL (never reuse a stale subset silently).

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL size larger than available | requested minutes exceed the filtered train set | choose a smaller size, or `all` |
| FAIL fewer than minimum samples | size too small for a meaningful run | raise the size or lower `subsample.min_train_samples` knowingly |
| WER very different between repeats | small data, high variance | use `repeats` and report mean and spread; do not trust one draw |
| overshoot is large | `by_group` with long lessons | use `by_sample` or accept it as reported |
| learning curve crosses | different Test sets used | use `scope: train_only` for curves |
| Val too small to stop early reliably | `val: follow` on small data | use `val: fixed` or accept noisy early stopping |

### Acceptance tests (offline, tiny synthetic samples)
1. The same seed gives the same subset; a different seed gives a different one; shuffling the input order gives the same subset.
2. Nesting: the subset for a smaller size is contained in the subset for a larger size, for both strategies.
3. Sizes in minutes, samples and fraction reach the target within the one-unit overshoot, which is reported.
4. `by_group` never splits a group; `by_sample` can.
5. `train_only` leaves Val and Test byte-identical (same objects, same order); `val: follow` reduces Val; `all_splits` reduces all three.
6. A size larger than available FAILS; `all` returns everything.
7. Fewer than the minimum samples FAILS; an empty split FAILS.
8. `repeat_index` gives different subsets of the same size.
9. Store round trip: written and read back equal; a stale fingerprint FAILS.
10. Disabled mode returns the input unchanged.
11. No network, no model imports.

### OPEN items (agent presents options, owner decides)
- The sizes of the presets after the first measurements of time per training step (O11).
- Whether `work_like` should use the exact data size reported by the manager once known (O22).
- Where the learning-curve experiment is defined (Orchestrator brief).

---

## PRD Component 5: ModelLoader

### Priority / order and dependencies
- Numbered 5 to keep the original numbering. Its products (feature extractor, tokenizer) are consumed by
  FeaturePipeline (4), so in implementation order ModelLoader comes BEFORE FeaturePipeline.
- Depends on Config (`model.*`, `runtime.*`). Does not depend on any data component.
- Phase 1 scope (owner decision): the pipeline is validated with a SMALL OpenAI Whisper model only.
  Large Hebrew models (ivrit-ai) are a later phase, run by the owner on his own machine. The component
  must support both without code change, because the model is chosen only by Config.

### Responsibility
Load one Whisper model, its feature extractor and its tokenizer from a LOCAL folder, validate them, and describe what was loaded.

### Interface and implementations
- `ModelLoader` (typing.Protocol): `load() -> LoadedModel`.
- Implementations selected by name through a registry (`model.loader`): `HfWhisperLoader` first.
  The Protocol exists because a second implementation is planned (another model family), not for show.
- Hugging Face and torch are imported ONLY inside the adapter module of `HfWhisperLoader`.
- A model folder saved by the pipeline (fine-tuned model) is loaded by the SAME loader, with no special case.
- No caching and no global state: every `load()` returns independent objects. When two models are needed
  (baseline vs fine-tuned), the caller loads them one at a time and releases the first (owner approved).

### Inputs -> Outputs (typed, dataclasses)
- `ModelInfo`: `name: str`, `path: Path`, `num_parameters: int`, `dtype: str`, `num_mel_bins: int`,
  `language: str`, `task: str`, `source_config_ref: str | None`.
  - Identification is by NAME and PATH only. No weight hash (owner decision).
  - `source_config_ref` is a link to the config / config-diff file of the experiment that produced this
    model. It is read from a small `pipeline_model_info.json` in the model folder when that file exists
    (written by the component that saves models, not yet specified; DEFAULT). For a downloaded base model it is `None`.
- `LoadedModel`: `model`, `feature_extractor`, `tokenizer`, `info: ModelInfo`.
  - The exact method signatures that FeaturePipeline and TrainerCore need from these three objects are
    OPEN: they are fixed as typing Protocols when those two briefs are written, so that no Hugging Face type
    leaks into high-level code.

### Rules (locked)
1. The loader NEVER downloads anything. It always loads with `local_files_only`, even when `runtime.online` is on.
2. Downloading from the Hub is a separate standalone script (`fetch_model`, direction only, run at home), not part of the pipeline.
3. The model folder is validated BEFORE loading: the loader checks that the required files exist and fails with the list of missing files.
4. `trust_remote_code` is never enabled.
5. Language and task come only from Config and are written into the model's generation settings.
   The Evaluator may override decoding settings (beams, max length); the loader does not own them.
6. The number of mel bins is READ from the feature extractor, never hard-coded (80 vs 128 differs between model families).
7. Freezing layers, dropout, gradient settings and thread counts are NOT the loader's business (TrainerCore).
8. The loader never modifies the model folder.
9. Saved models use the `safetensors` format (owner approved).
10. Approved dependencies (owner approved): `torch`, `transformers`, `safetensors`, `accelerate`. `peft` is NOT approved and not needed in phase 1.

### Config keys it reads (section `model`)
- `model.loader` [DEFAULT: `hf_whisper`]
- `model.path` (folder, required, no default)
- `model.language` [DEFAULT: `he`]
- `model.task` [DEFAULT: `transcribe`]
- `model.dtype` [DEFAULT: `float32`; CPU]
- Example YAML (examples only, never in code): phase 1 `model.path` points to a local copy of OpenAI `whisper-tiny`
  (`whisper-base` as the next step); a commented-out later example for `ivrit-ai/whisper-large-v3-turbo`.
- Python version [DEFAULT: 3.12, the conservative choice for torch / transformers wheels on Windows and for an offline machine]. Still to be confirmed by the clean-venv install test (O1).

### Out of scope (must NOT do)
- No downloading, no network, no Hub model names as input.
- No training logic, freezing, optimizer, or LoRA.
- No audio processing and no tokenizing of data (FeaturePipeline).
- No decoding or WER (Evaluator).
- No saving of models (separate component; the loader only reads).
- No choosing a model by itself or falling back to a different model when loading fails.

### Edge cases and failure behavior
- Folder missing or not a folder: FAIL with the path and how to fix (copy the model there, or run `fetch_model` at home).
- Missing required files (weights, `config.json`, feature extractor config, tokenizer files): FAIL listing exactly which.
- Weights present only in an old non-safetensors format: FAIL explaining the file, and the fix (convert once at home). DEFAULT behavior, may be relaxed by decision.
- Language not supported by the tokenizer, or an English-only model with `model.language: he`: FAIL naming the model and language.
- Model family without the expected components (not a Whisper model): FAIL naming what is missing.
- `model.dtype` unknown: FAIL at Config validation listing the allowed values.
- Not enough memory while loading: the OS error is surfaced with a hint (smaller model or fewer other programs); no automatic retry.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL folder or files missing | model not copied to the machine | copy the full folder from home, or run `fetch_model` at home; do not enable online mode as a fix |
| FAIL language not supported | English-only model chosen | change `model.path` to a multilingual model |
| FAIL old weights format | community model published as `.bin` only | convert once at home to safetensors, then copy |
| garbage output after loading a fine-tuned folder | wrong or incomplete folder | check `pipeline_model_info.json`, compare with the experiment config |
| out of memory on load | model too large for the machine | use a smaller model; do not change loader code |
| output language is wrong (not Hebrew) | language not set | check `model.language`; the ivrit-ai cards require it to be set explicitly |

### Acceptance tests (offline, no network)
1. A valid local model folder loads and `ModelInfo` is filled (name, path, parameter count, dtype, mel bins from the extractor).
2. Missing folder and missing individual files FAIL with the right messages.
3. Unsupported language or an English-only model FAILS.
4. Non-safetensors-only weights FAIL with the explanation.
5. A saved folder with `pipeline_model_info.json` is loaded by the same loader and exposes `source_config_ref`.
6. Two successive `load()` calls return independent objects.
7. The input folder is byte-identical after loading.
8. Unknown loader name in Config FAILS and lists the registered names.
9. No network call (network blocked in the test); Hugging Face and torch are imported only inside the adapter module.

### OPEN items (agent presents options, owner decides)
- DECIDED by the owner (Orchestrator discussion): option (b) below. How the tests get a model without any download: (a) a tiny random-weights model built inside the test with minimal hand-made tokenizer files, (b) a very small test model folder stored with the tests, (c) an integration test that runs only when a local model folder is provided (cannot be the only test). The agent presents the trade-offs.
- Exact Protocols for the three loaded objects (with FeaturePipeline and TrainerCore).
- Whether old-format weights should be accepted.
- `fetch_model` script details (a separate small brief later).

---

## PRD Component 4: FeaturePipeline

### Priority / order and dependencies
- Pipeline order: DataSource -> Splitter -> TextNormalizer -> SampleFilter -> Subsampler -> FeaturePipeline -> TrainerCore.
- Depends on Config (`features.*`, `text.train.timestamps`), on `NormalizedSample` (from SampleFilter) and on
  `LoadedModel` (feature extractor and tokenizer, from ModelLoader). Implement AFTER ModelLoader.

### Responsibility
Convert kept samples into model-ready examples (input features from the audio, label ids from `train_text`) and provide the batch collator.

### Interface and implementations
- `FeaturePipeline.build(samples: Sequence[NormalizedSample], split: str) -> FeatureResult`.
- `FeatureDataset` (typing.Protocol): `__len__() -> int`, `__getitem__(index: int) -> FeatureExample`.
- `FeatureCollator`: turns a list of `FeatureExample` into one padded batch in the form TrainerCore needs.
  The collator lives inside the Hugging Face adapter module (the trainer's batch format is a Hugging Face concern).
- Hugging Face, torch and the audio library are imported ONLY inside adapter modules.
- No `datasets` library (owner decision: simple loop, no new dependency). The trainer receives a plain dataset object.
- No augmentation in v1 (owner approved). It would be a new class later, with no change to existing code.

### Inputs -> Outputs (typed, dataclasses)
- `FeatureExample`: `sample_id: str`, `input_features` (float32 array: mel bins x frames), `labels: list[int]`.
- `FeatureReport`: counts (seen / built / dropped), number resampled, number downmixed to mono,
  number dropped for label length, label-length distribution (min / mean / p95 / max in tokens), and the
  limits that were applied (audio seconds, label tokens, both read from the loaded model).
- `FeatureResult`: `dataset: FeatureDataset`, `report: FeatureReport`.
- The exact array type of `input_features` and the collator's output form are fixed with TrainerCore (OPEN).

### Rules
1. LOCKED: audio at another sampling rate is CONVERTED to the extractor's rate (read from the feature extractor, never hard-coded), and the report counts how many were converted.
2. LOCKED: no automatic truncation of audio. The SampleFilter must remove clips longer than the model window (`filter.rules.max_duration`) BEFORE this component. If a longer clip still arrives (for example the filter is in `report` mode or the rule is disabled), the run FAILS naming the count, the longest clip, and the fix (enable the rule and use `apply` mode). The window length is read from the feature extractor, never hard-coded. (Owner: long clips are removed by the filter.)
3. LOCKED: shorter audio is padded to the model window by the feature extractor. No action is needed. Reminder for planning: the encoder always processes the full window, so a 3-second clip costs the same CPU time as a 30-second clip (see O33).
4. LOCKED: labels come from `train_text`. Label ids are never edited by hand.
5. LOCKED: labels that exceed the model's label limit are FLAGGED and REMOVED from the dataset, and the report lists their count and ids. The limit is read from the loaded model, never hard-coded. If the removed fraction exceeds `features.max_dropped_ratio`, the run FAILS. (Owner approved flag-and-remove as a starting point.)
6. LOCKED: the collator pads labels with the value the trainer ignores in the loss (-100).
7. LOCKED: determinism: any shuffling uses the seed from Config. The component does not shuffle by itself unless Config says so.
8. LOCKED (owner delegated the decision to the agent; timestamps do not count in the final result): v1 implements `strip` only. `keep` is a documented later extension. `timestamp_segments` are already preserved by the TextNormalizer, so nothing is lost. With `text.train.timestamps: keep` the run FAILS at Config validation with an explanation (O3, O35, O48).
9. DEFAULT: stereo is averaged to mono, and the report counts it.
10. DEFAULT: features are computed per item on demand (lazy), with no cache in phase 1 (owner approved the optional cache, off for now). With the cache on, features are computed once in a simple loop with `features.num_workers` and stored as files; a later run reuses them only when the cache key (audio path, sample id, extractor settings) matches.
11. DEFAULT: `features.num_workers` = 0 (O2).

### Config keys it reads
- `features.num_workers` [DEFAULT: 0]
- `features.cache.enabled` [DEFAULT: false], `features.cache.dir`
- `features.max_dropped_ratio` [DEFAULT: 0.05]: safety gate for dropped labels
- `features.downmix` [DEFAULT: `mean`]
- `text.train.timestamps` (owned by TextNormalizer; read here)
- from ModelLoader's output (not Config): sampling rate, window length, mel bins, label limit

### Out of scope (must NOT do)
- No quality or length filtering of audio (SampleFilter); the label-length removal above is the only removal, because only this component has the tokenizer.
- No text cleaning (TextNormalizer) and no splitting.
- No augmentation, no model code, no training, no WER.
- No network, no downloads.
- No choice of batch size (TrainerCore).

### Edge cases and failure behavior
- Audio file unreadable at build time: FAIL naming the sample id and path (DataSource should have rejected it earlier).
- Audio longer than the model window: FAIL (rule 2).
- Empty `train_text` (not removed by the filter): FAIL naming the count, with the fix (SampleFilter `min_text_chars`).
- Label limit exceeded: remove and report; gate FAIL above `features.max_dropped_ratio`.
- Unsupported sampling rate conversion or missing audio library: FAIL naming the library and the fix.
- Zero examples left: FAIL.
- Cache folder not writable or cache key mismatch: rebuild (never reuse silently a stale cache).

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL audio longer than the window | SampleFilter is in `report` mode or `max_duration` disabled | set `filter.mode: apply` and enable `max_duration` in YAML; do not edit this component |
| FAIL dropped ratio too high | many long transcripts | read the label-length distribution; check that timestamps are stripped; raise nothing without a decision |
| model learns to output timestamps by accident | keep mode on | check `text.train.timestamps` |
| out of memory while building | cache on or too many workers | set `features.cache.enabled: false` and `features.num_workers: 0` |
| very slow training | short clips cost the same as long ones | see O33; reduce the subsample or use a smaller model; no code change |
| FAIL missing audio library | home install differs from work machine | repeat the home install test (O9) |

### Acceptance tests (offline, tiny synthetic audio; the extractor and tokenizer come from the ModelLoader test setup)
1. A wav at the model rate gives features of the expected shape and labels equal to the tokenizer output of `train_text`.
2. A different sampling rate is converted and counted in the report; stereo is downmixed and counted.
3. A clip longer than the window FAILS with the count and the fix; a shorter clip is padded.
4. A label above the limit is removed, listed in the report, and the gate FAILS when the ratio is exceeded.
5. Collator: labels are padded with -100, features stack to the same shape, order is preserved.
6. With the cache on, a second run reuses files; changed extractor settings invalidate the cache.
7. Same seed gives the same order; different seed gives a different order (when shuffling is enabled).
8. Input audio files are byte-identical afterwards.
9. No network; Hugging Face, torch and the audio library are imported only inside adapter modules.

### OPEN items (agent presents options, owner decides)
- LATER EXTENSION, not in v1 (O3, O35, O48): keep-timestamps mode. Options when it is needed: (A) rebuild the label from `timestamp_segments`, rounded to the model's timestamp grid, and tokenize with the tokenizer's timestamp support; (B) pass `train_text` with the tokens as they are written; (C) train on a mix of labels with and without timestamps. Recommendation: A, because it is robust to odd timestamp values. It also requires the Evaluator to generate with timestamps and strip them before WER (to be specified at the Evaluator), and the format and resolution of the work-data timestamps (manager question).
- Resampling and mono conversion library (candidates: scipy, which may be needed anyway for significance tests; torchaudio). Needs approval (O9).
- Numeric array library (numpy is installed together with torch; approval requested for explicit use).
- Exact types shared with TrainerCore (feature array, collator output).
- Whether label-length removal should instead be a SampleFilter rule fed with a token counter (cleaner SRP, more wiring).

---

## PRD Component 6: TrainerCore

### Priority / order and dependencies
- Pipeline order: ... -> FeaturePipeline -> TrainerCore -> Evaluator.
- Depends on Config (`train.*`), on `LoadedModel` (ModelLoader) and on `FeatureDataset` plus the collator (FeaturePipeline).
- Hyperparameter VALUES are received from the caller (the manual config now, the HyperparamSearcher later). Searching is not this component's job.

### Responsibility
Run ONE training run on a given model and datasets with given hyperparameters, and return the trained model folder and the record of the run.

### Interface and implementations
- `TrainerCore.train(request: TrainRequest) -> TrainResult`. One implementation (Hugging Face `Seq2SeqTrainer`, inside its own adapter module). No interface is created for it (only one implementation planned).
- `FreezePolicy` (typing.Protocol): `apply(model) -> FreezeReport`. Implementations selected by name through a registry: `none` (full fine-tuning) and `freeze_encoder`. A new policy is a new class. (LoRA is NOT included: `peft` is not approved.)
- `MetricsSink` (typing.Protocol): `write(record: StepMetrics) -> None`. Implementations: `CsvMetricsSink` (live file) and `ConsoleMetricsSink` (log line). More sinks can be added without touching the trainer.
- Hugging Face and torch are imported only inside adapter modules. No new dependency (the trainer of `transformers`, already approved).

### Inputs -> Outputs (typed, dataclasses)
- `TrainHyperparams`: `learning_rate`, `max_steps`, `batch_size`, `grad_accum_steps`, `warmup_steps`, `weight_decay`, `max_grad_norm`, `seed`. The set of fields is a typed object so that the searcher can produce it.
- `TrainRequest`: `model: LoadedModel`, `train: FeatureDataset`, `val: FeatureDataset`, `collator`, `hyperparams: TrainHyperparams`, `output_dir: Path`, `on_eval: Callable[[StepMetrics], bool] | None` (called after each Val evaluation; returning True stops the run; used by the HyperparamSearcher for pruning; ADDED after the HyperparamSearcher discussion, owner confirmed).
- `StepMetrics`: `step`, `epoch_fraction`, `train_loss`, `val_loss | None`, `learning_rate`, `grad_norm`, `seconds_per_step`, `elapsed_sec`, `eta_sec`.
- `TrainResult`: `model_dir: Path` (final selected model), `history: list[StepMetrics]`, `best_step`, `best_val_loss`, `stopped_reason` (`max_steps` | `early_stopping` | `time_limit` | `stopped_by_callback`), `timings`, `freeze_report`.

### Rules
1. LOCKED: the unit of training is steps (`max_steps`), not epochs.
2. LOCKED: during training only the loss on Val is computed. WER (generation) is NOT computed in the training loop by default, because generation is very expensive on CPU. WER is computed by the Evaluator after training.
3. LOCKED (owner request): LIVE MONITORING. While the run is in progress, metrics are appended to a file and flushed at every logging step, so the owner can follow the trend of loss and learning rate in real time, at any moment, without waiting for the end. The same record is written to the log.
4. LOCKED: checkpoints are saved, and a run can be resumed (`train.resume` from Config). Only the latest and the best checkpoint are kept (disk space).
5. LOCKED: early stopping on the Val loss, patience from Config.
6. LOCKED: an optional time cap for the run; on reaching it the run saves and stops gracefully with `stopped_reason: time_limit`.
7. LOCKED: the number of CPU threads is limited by Config so that the machine stays usable (O31).
8. LOCKED: the seed is applied everywhere (Python, numpy, torch, data order) and written to the record.
9. LOCKED: the output model is saved in `safetensors` format, together with `pipeline_model_info.json` linking to the experiment's config or diff file (read by the ModelLoader as `source_config_ref`). The best checkpoint (by Val loss) is the model that is returned, DEFAULT.
10. LOCKED: the training mode is set by Config (`train.freeze_policy`). The component never decides by itself.
11. DEFAULT: full fine-tuning (`none`), on a small model (phase 1).
12. DEFAULT: float32 (CPU).
13. DEFAULT: AdamW, linear warmup then linear decay, gradient clipping 1.0.

### Config keys it reads (section `train`)
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

### Out of scope (must NOT do)
- No hyperparameter search, no choice of model, no data processing.
- No WER or generation in the loop (Evaluator), no statistical tests.
- No model registry or winner selection.
- No LoRA, no new dependency.
- No network, no downloads, no interactive prompts.
- No overwriting of an existing output folder.

### Edge cases and failure behavior
- Loss becomes NaN or infinite: stop, keep the last good checkpoint, FAIL naming the step and the likely cause (learning rate too high).
- Output folder exists and is not empty while `resume: false`: FAIL (never overwrite).
- `resume: true` but no checkpoint, or the checkpoint belongs to a different config: FAIL naming both.
- Empty Train or Val set: FAIL.
- Out of memory: FAIL with a hint (smaller batch size, more accumulation, smaller model, or freeze the encoder); no automatic retry.
- Early stopping triggers at the first evaluation: warn that the Val loss did not improve.
- Time cap reached: save and stop, status `time_limit` (not an error; the caller decides).
- Unknown freeze policy or sink name: FAIL listing the registered names.
- Fewer steps requested than needed for one evaluation: run and evaluate once at the end.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL NaN loss | learning rate too high, bad batch | lower `learning_rate`; check the first live metrics lines |
| training very slow | too many or too few threads, short clips cost like long ones (O33) | adjust `train.cpu_threads`; use a smaller model or fewer steps |
| out of memory | batch too large for the machine | lower `batch_size`, raise `grad_accum_steps`, or use `freeze_encoder` |
| Val loss does not move | learning rate too low, too few steps | read the live trend; increase `max_steps` or the rate |
| Val loss goes up while train loss goes down | overfitting on small data | rely on early stopping and the best checkpoint; use more data or fewer steps |
| FAIL output folder not empty | rerun without resume | choose a new experiment folder (RunManager does this) or set `resume: true` |
| machine unusable during the run | too many threads | lower `train.cpu_threads` |

### Acceptance tests (offline; tiny model and tiny synthetic data from the ModelLoader test setup)
1. A short run (a few steps) finishes, returns a `TrainResult`, and the saved folder loads through the ModelLoader.
2. Live metrics: the metrics file grows DURING the run (checked from inside a step callback), and each record has all fields.
3. Val loss is recorded every `eval_every_steps` and no generation is executed (spy on the generation call).
4. Early stopping stops the run at the expected step and returns the best checkpoint.
5. Resume continues from the saved step; a mismatching config FAILS.
6. NaN loss stops the run, keeps the last good checkpoint, and FAILS.
7. The time cap stops the run gracefully with the right status.
8. `freeze_encoder` freezes exactly the encoder parameters (counted in `FreezeReport`); `none` freezes nothing.
9. The thread limit from Config is applied.
10. The same seed gives the same loss history; a non-empty output folder is not overwritten.
11. Unknown policy or sink FAILS and lists the registered names.
12. No network; Hugging Face and torch are imported only inside adapter modules.

### OPEN items (agent presents options, owner decides)
- Live viewer. The metrics file can be opened in Excel at any time (manual refresh). Options for something that updates by itself: (a) a small standalone script that writes a `progress.html` page with a line chart drawn in plain SVG, refreshing itself, with no dependency; (b) TensorBoard (new heavy dependency, not approved). Recommendation: (a), as a separate small tool, designed after the main components.
- Optional WER on a small fixed Val subset every N steps (`train.periodic_wer.*`, DEFAULT off because of the CPU cost). It would be injected as a callable from the Evaluator, so TrainerCore does not depend on it. Design at the Evaluator.
- Exact feature-array and collator types shared with FeaturePipeline.
- Whether bfloat16 on CPU is worth testing later (speed vs accuracy), only after the pipeline works.

---

## PRD Component 7: Evaluator

### Priority / order and dependencies
- Pipeline order: ... -> TrainerCore -> Evaluator -> (HyperparamSearcher, ModelRegistry).
- Depends on Config (`eval.*`, `text.*`), on `LoadedModel` (ModelLoader), on `FeatureDataset` (FeaturePipeline) and on the eval profile from TextNormalizer (`build_profile("eval")`).
- The same Evaluator measures the baseline model (before training) and every trained model, with identical settings.

### Responsibility
Transcribe an evaluation set with a given model, score the transcripts against the references, and compare two scored results with a statistical test.

### Interface and implementations
Three small parts (SRP), each replaceable:
- `Transcriber` (typing.Protocol): `transcribe(model: LoadedModel, dataset: FeatureDataset, settings: DecodeSettings) -> list[RawHypothesis]`. Hugging Face adapter inside its own module. (The only part that needs the model.)
- `Evaluator.evaluate(request: EvalRequest) -> EvalResult`: uses an injected `Transcriber` and the eval text profile, and computes the metrics. Metrics are classes selected by name through a registry (`wer`, `cer`); a new metric is a new class.
- `Comparator.compare(baseline: EvalResult, candidate: EvalResult) -> ComparisonResult`: uses a `SignificanceTest` (typing.Protocol), implementation `paired_group_bootstrap`. A second implementation (for example Wilcoxon) is possible later and would need approval for a new dependency.
- The metric code is our own pure-Python implementation (word and character edit distance). No `jiwer`, and never `evaluate.load` (it downloads).
- Hugging Face and torch are imported only inside the Transcriber adapter.

### Inputs -> Outputs (typed, dataclasses)
- `EvalItem`: `sample_id`, `group_id`, `reference: str` (this is `eval_text`), `audio_path: Path` (for the human review list), `attributes: Mapping[str, float | int | bool | str]` (per-file facts such as duration, label length, word count, `extra` fields and text flags, built by the Orchestrator; the Evaluator only passes them through; ADDED for the ReportBuilder correlations, owner confirmed).
- `EvalRequest`: `items: Sequence[EvalItem]`, `dataset: FeatureDataset`, `model: LoadedModel`, `split_name: str`, `eval_version: int`, `normalize: Callable[[str], str]` (the eval profile, applied to hypotheses), `settings: DecodeSettings`.
- `FileScore`: `sample_id`, `group_id`, `reference`, `hypothesis` (normalized), `hypothesis_raw`, `ref_words`, `word_errors`, `wer`, `cer`, `empty_output: bool`, `hit_length_cap: bool`, `attributes` (copied unchanged from the `EvalItem`).
- `EvalResult`: `split_name`, `eval_version`, `model_info`, `wer` (corpus level), `wer_mean_per_file`, `cer`, `n_files`, `n_words`, `share_files_wer_below` (for example the share of files with WER under 10% and under 20%; thresholds from Config), `files: list[FileScore]`, `decode_settings`, `seconds`, `flags`.
- `ComparisonResult`: `delta_wer` (baseline minus candidate, positive means improvement), `ci_low`, `ci_high`, `confidence`, `one_sided`, `n_groups`, `min_effect_met: bool`, `verdict` (`improved` | `no_clear_difference` | `worse` | `not_comparable` | `insufficient_groups`), `explanation: str` (one plain sentence for a human).
- `ReviewList`: the files to listen to (see rule 9).

### Rules
1. LOCKED: the primary metric is WER on `eval_text`, computed at corpus level (all words together). CER is a secondary metric in the report (O24 settled). The mean of per-file WER is also reported.
2. LOCKED: hypotheses go through the SAME eval profile as the references (same version, same function).
3. LOCKED: the score of every file is kept and returned (the table is the basis for finding where the model fails).
4. LOCKED: an empty output is scored as all words deleted (a full error), and flagged.
5. LOCKED: results with different `eval_version` are NOT comparable; the comparison returns `not_comparable` with no numbers.
6. LOCKED: the baseline is evaluated by this same component, on the same set, with the same profile and decode settings.
7. LOCKED: model selection and search use Val; the Test split is evaluated once, at the end, for the winner. The Evaluator labels the split; the Orchestrator enforces this order.
8. LOCKED: the statistical comparison resamples GROUPS (`group_id`, for example whole lessons), not single files, because files from one group are not independent.
9. LOCKED (owner's lens: a human listener judging whether the text is good transcription): the report includes a HUMAN REVIEW LIST: a configurable number of random files and of worst files, with reference, hypothesis, a word-level difference, the WER and the audio path, so the owner can listen and judge. The share of files under the WER thresholds (rule in `EvalResult`) is reported as a plain "how many files are usable" figure.
9b. LOCKED (owner): success is judged through TWO lenses: the automatic test (primary, decides) and the owner's own listening. The owner's listening can reveal a model that "failed in practice although it passed the test", so the report must always let him do it (rule 9). The pipeline stays non-interactive: it never waits for a human verdict. How a human verdict is recorded afterwards is OPEN (O43).
9c. LOCKED (owner): WER only decides. Error types are NOT weighted (names, numbers, jargon count like any other word). Timestamps and special punctuation do not count in the final result.
10. LOCKED: significance does not have to be strict. The confidence level, one-sided or two-sided, and the minimum effect are all Config values, and the verdict is worded as a level of evidence, not as proof.
11. LOCKED: decoding uses language and task from the loaded model (set by ModelLoader). Timestamps are not generated in v1 (strip only, O48). The score is on the text only (no timing accuracy). If keep mode is added later, generation would produce timestamp tokens and the eval profile removes them before scoring.
12. LOCKED: a length cap on output relative to the audio length guards against endless repetition; capped outputs are flagged and counted.
13. DEFAULT: `num_beams` = 1.
14. DEFAULT: confidence 0.90, one-sided (candidate better than baseline), 2000 resamples, minimum effect 0.0 absolute WER.
15. DEFAULT: the first-evaluation sanity limit (O15) is OFF in phase 1 (the small model will have a very high Hebrew WER; the goal is only to check that the pipeline runs end to end).

### Config keys it reads (section `eval`)
- `eval.metrics` [DEFAULT: `wer`, `cer`]
- `eval.generation.num_beams` [DEFAULT: 1], `eval.generation.max_new_tokens` [DEFAULT: the model's label limit], `eval.generation.max_tokens_per_audio_sec` [DEFAULT: set from the first runs]
- `eval.batch_size` [DEFAULT: 4]
- `eval.thresholds.wer_below` [DEFAULT: 0.10, 0.20]
- `eval.significance.method` [DEFAULT: `paired_group_bootstrap`], `.confidence` [DEFAULT: 0.90], `.one_sided` [DEFAULT: true], `.n_resamples` [DEFAULT: 2000], `.min_effect_abs` [DEFAULT: 0.0], `.min_groups` [DEFAULT: 5]
- `eval.review.n_random` [DEFAULT: 20], `eval.review.n_worst` [DEFAULT: 10]
- `eval.sanity.max_baseline_wer` [DEFAULT: null = off]
- seed from the global Config.

### Out of scope (must NOT do)
- No training, no data processing, no text normalization logic (uses the profile it is given).
- No choosing the winner (Selector later) and no search.
- No file writing (the report writer or the RunManager does it, O14), no final JSON printing.
- No network, no downloads, no `evaluate.load`.
- No timing-accuracy metric in v1.

### Edge cases and failure behavior
- A reference empty after normalization: the file is excluded from WER, flagged, and counted in the report (the SampleFilter should have removed it).
- All hypotheses empty: FAIL naming the likely cause (wrong language setting or wrong features).
- Fewer groups than `eval.significance.min_groups`: the comparison returns `insufficient_groups` (not a failure) and still reports the plain WER difference without a confidence statement.
- Baseline and candidate evaluated on different file sets: FAIL naming the difference.
- `eval_version` differs: `not_comparable`.
- Output hits the length cap: flagged, counted; the score uses the capped text.
- Unknown metric, method or sink name: FAIL listing the registered names.
- Sanity limit on and baseline WER above it: FAIL naming both values.
- Out of memory during decoding: FAIL with the hint to lower `eval.batch_size`.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| WER close to 1.0 for baseline and trained | wrong language setting, or references contain unspoken symbols | check `model.language`; read the character inventory and the review list |
| WER improves but the review list looks wrong | profile too aggressive or too gentle | compare under a new `eval_version`; versions are not comparable |
| comparison says `not_comparable` | different `eval_version` | re-evaluate the baseline with the current profile |
| comparison says `insufficient_groups` | few groups in the evaluated set | evaluate on a larger set or choose a smaller group key; read the plain difference only |
| many `hit_length_cap` outputs | model loops | lower `max_new_tokens`, check the training labels; do not hide the flag |
| decoding too slow | beams or batch size | `num_beams: 1`, lower batch size, evaluate a subset in smoke runs |
| FAIL all outputs empty | features or language wrong | check the FeaturePipeline report and `model.language` |

### Acceptance tests (offline, tiny synthetic data, a fake Transcriber injected)
1. WER and CER match hand-computed values on table-driven Hebrew examples (substitution, deletion, insertion, prefix error).
2. Hypothesis and reference with the same words and different punctuation score 0 (same profile).
3. Corpus WER differs correctly from the mean of per-file WER on an uneven example.
4. Empty output scores a full error and is flagged; an empty reference is excluded and counted.
5. The length cap flags a looping output.
6. Comparison: a clearly better candidate gives `improved`; identical results give `no_clear_difference`; a worse candidate gives `worse`.
7. Bootstrap is deterministic for a fixed seed and resamples groups (a test with two groups of unequal size shows it).
8. Different `eval_version` gives `not_comparable` with no numbers; different file sets FAIL.
9. Fewer groups than the minimum gives `insufficient_groups`.
10. The review list contains the configured numbers of random and worst files, includes the audio path, and is deterministic for a fixed seed.
11. The Evaluator runs with the fake Transcriber (no model needed).
12. No network; Hugging Face and torch are imported only inside the Transcriber adapter.

### OPEN items (agent presents options, owner decides)
- A second significance method (Wilcoxon on per-group scores needs scipy, which is not approved).
- Report persistence (O14).
- Second test set (held-out groups plus seen groups, O20): the interface already supports it by calling `evaluate` once per set with a different `split_name`; no extra work now.
- Periodic WER during training (O38): this component would provide the callable on a small fixed Val subset.
- How the owner's listening verdict is recorded and used (O43): options (a) outside the pipeline, in the owner's notes; (b) an optional small file the owner fills in after listening, read by the later Selector/Registry as a note on the winner (never as an automatic gate).
- Small-data behavior (the work data will be much smaller than the home data): `min_groups`, bootstrap resamples and the share-of-files figures must stay meaningful or say "insufficient"; verified with the Subsampler rehearsals.
- The Hebrew prefix problem (CER helps, a prefix-aware metric is not in v1).

---

## PRD Component 8: HyperparamSearcher

### Priority / order and dependencies
- Depends on Config (`search.*`), on `TrainHyperparams` (TrainerCore) and on a `TrialRunner` supplied by the Orchestrator.
- Stage 1 assumption (owner): there is a LOT of data (all the Daf Yomi data on the Hub). Small-data behavior is a later rehearsal stage.
- The winner is NOT chosen here (Selector, a separate component).

### Responsibility
Propose hyperparameter sets, hand each one to a trial runner, record every outcome, and return the ranked candidates within a budget.

### Interface and implementations
- `HyperparamSearcher.search(spec: SearchSpec, run_trial: TrialRunner) -> SearchResult`.
- `TrialRunner` (typing.Protocol): `__call__(trial: Trial, report: Callable[[StepMetrics], bool]) -> TrialOutcome`. Supplied by the Orchestrator, which builds it from the Subsampler, TrainerCore and Evaluator. The searcher never trains or evaluates by itself (DIP).
  `report` is called at every Val evaluation; it returns True when the trial should stop (pruned).
- `SearchStrategy` (typing.Protocol), selected by name through a registry. Implementations: `optuna_tpe` (adapter module, the ONLY place that imports Optuna) and `grid` (own code, no dependency). Two implementations exist because grid vs Optuna is still undecided (O10).

### Inputs -> Outputs (typed, dataclasses)
- `ParamRange`: `kind` (`log_float` | `float` | `int` | `categorical`), `low`, `high`, `choices`.
- `SearchSpec`: `parameters: Mapping[str, ParamRange]`, `strategy`, `n_trials`, `time_budget_minutes`, `pruning`, `parallel_trials`, `max_failed_ratio`, `study_dir`, `seed`.
- `Trial`: `trial_id: int`, `params: Mapping[str, float | int | str]`.
- `TrialOutcome`: `status` (`complete` | `pruned` | `failed`), `val_loss: float | None`, `steps`, `seconds`, `model_dir: Path | None`, `error: str | None`.
- `TrialRecord`: `trial`, `outcome`.
- `SearchResult`: `trials: list[TrialRecord]`, `ranked: list[TrialRecord]` (completed trials by Val loss, best first, at most `search.finalists.k`), `stopped_reason` (`n_trials` | `time_budget` | `too_many_failures` | `space_exhausted`), `report`.

### Rules
1. LOCKED (owner): Optuna is approved ONLY as long as it helps. It is imported only inside its adapter. The dependency-free `grid` strategy stays available as the alternative. Honest note for the owner: with only two searched parameters, Optuna's main benefits are pruning, resuming a stopped search, and bookkeeping; a small grid would find similar values.
2. LOCKED (owner: do not widen the search, thin it out if possible): the searched parameters by default are only `learning_rate` and `grad_accum_steps` (the effective batch size). Everything else is fixed from `train.hyperparams`. `max_steps` is NOT searched: a generous cap plus early stopping decides the length. Adding a parameter is a Config change only.
3. LOCKED: the search objective is the Val loss (cheap). WER is not used inside the search (CPU cost). The finalists are scored by WER on Val afterwards by the Evaluator (run by the Orchestrator) and handed to the Selector.
4. LOCKED: pruning is on by default: a trial that looks clearly worse after a few Val evaluations is stopped (`pruned`).
5. LOCKED: a failed trial (for example NaN loss) is recorded as `failed` and the search continues. If the failed share exceeds `search.max_failed_ratio` (after `search.min_trials_before_abort` trials), the search stops with `too_many_failures`.
6. LOCKED: the budget is `search.n_trials` and/or `search.time_budget_minutes`. When it is reached, the search stops and returns the best so far.
7. LOCKED (owner): trials run one after another by default. A simple switch (`search.parallel_trials`) lets the owner allow more than one at a time if he decides to let the machine work. That mode MUST be tested to work, not only offered.
   - DEFAULT design: parallel trials are separate worker processes that share the study file, each with its own CPU thread limit (total threads divided by `parallel_trials`), because torch thread settings are per process. The Windows process-spawn caveat applies (O2).
8. LOCKED: the study is stored in a file inside the experiment folder (no database server) so that a stopped search can resume. A rerun with the same config continues; a changed search space on an existing study FAILS. (Optuna's file-based storage is to be verified when implementing; if it needs an extra package, the agent asks.)
9. LOCKED: determinism: the sampler is seeded from Config. With pruning or parallel trials, the exact sequence can still vary, and the report says so.
10. LOCKED: finalists: after the search, the best `search.finalists.k` parameter sets are retrained on the full data by the Orchestrator (through the same `TrialRunner` with a full-data setting) and evaluated by the Evaluator. Repeating a finalist with several seeds is allowed by Config (`search.finalist_repeats`, DEFAULT 1) but is meant for a later stage, not for the current training stage (owner).
11. LOCKED: "no winner" is a legitimate outcome of the whole pipeline, decided by the Selector. The searcher never declares a winner.
12. DEFAULT: the search itself runs on a smaller subset of the training data (a Subsampler preset) to save time, then the finalists run on everything. At home in stage 1 the full data is large, so screening on a subset is the main time saver.

### Config keys it reads (section `search`)
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

### Out of scope (must NOT do)
- No training, no evaluation, no WER, no data handling.
- No choosing the winner (Selector) and no saving of models (Registry).
- No searching over the base model, the training mode or the text profile.
- No network, no database server.
- No interactive prompts.

### Edge cases and failure behavior
- Optuna not installed while `optuna_tpe` is selected: FAIL naming the package and the alternative (`grid`).
- Invalid parameter range (low above high, empty choices, unknown parameter name): FAIL at Config validation naming the key.
- Zero completed trials at the end: FAIL with the failure reasons summarized.
- All trials pruned: FAIL with the advice to relax pruning.
- Existing study with a different search space or seed: FAIL naming both.
- Budget of zero trials or zero minutes: FAIL at Config validation.
- `grid` space too large for the budget: runs as many as the budget allows in a fixed order and reports `n_trials` or `time_budget` as the reason.
- The trial runner raises an exception: recorded as `failed` with the message; the search continues.
- `parallel_trials` greater than the available cores: FAIL with the numbers.
- A trial stopped by the time budget in the middle: recorded as `failed` with reason `time_budget`; never counted as a result.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL too many failures | learning rate range too high | lower the upper bound in `search.parameters.learning_rate` |
| FAIL all trials pruned | pruning too aggressive early | raise `search.pruning.min_evals` or disable pruning |
| FAIL study space differs | config changed after the first run | use a new experiment folder, or restore the old space |
| search too slow | trials too long | use a smaller screening preset, fewer trials, or a time budget |
| machine sluggish in parallel mode | too many workers | lower `search.parallel_trials` |
| FAIL Optuna missing | package not installed on this machine | install it in the virtual environment, or set `search.strategy: grid` |
| best trial at the edge of the range | range too narrow | widen that range in YAML for the next experiment |

### Acceptance tests (offline, with a fake TrialRunner returning a synthetic objective)
1. The grid strategy enumerates the expected combinations in a fixed order.
2. The Optuna strategy finds the better region of a simple synthetic objective within the budget.
3. `n_trials` and `time_budget_minutes` each stop the search with the right reason.
4. A failing trial is recorded and the search continues; the failure ratio abort works.
5. The pruning callback stops a trial that is clearly worse, and the outcome is `pruned`.
6. Resume: a rerun continues the study (counts add up); a changed space FAILS.
7. The same seed gives the same sequence in sequential mode.
8. `parallel_trials: 2` runs and records every trial exactly once, with the thread limit divided (the owner requires this to be verified).
9. The result is ranked by Val loss and limited to `finalists.k`.
10. Unknown strategy FAILS listing the registered names; Optuna imported only inside its adapter module; no network.

### OPEN items (agent presents options, owner decides)
- Whether parallel trials are processes (DEFAULT) or threads, to be confirmed with the first parallel test.
- The size of the screening preset (O11) and the range of the learning rate (from the first smoke runs).
- Whether grid is enough in practice with two parameters (O10).
- Cross-validation for tiny validation sets (later stage, small-data rehearsal).

---

## PRD Component 9: Selector

### Priority / order and dependencies
- Depends on Config (`select.*`), on `EvalResult` and `ComparisonResult` (Evaluator) and on the finalists produced by the Orchestrator after the HyperparamSearcher.
- Saving the winner is the ModelRegistry's job, not this component's.

### Responsibility
Decide, from evaluated finalists and the baseline, which model is the winner, or declare that the baseline stays.

### Interface and implementations
- `Selector.select(request: SelectionRequest) -> SelectionResult`. One implementation; no interface is created.
- It uses the Evaluator's `Comparator` (injected) and a test-evaluation callable (injected). It never trains or evaluates by itself (DIP).

### Inputs -> Outputs (typed, dataclasses)
- `Candidate`: `trial_id`, `params: Mapping[str, float | int | str]`, `model_dir: Path`, `val: EvalResult`, `val_loss: float`, `steps: int`.
- `SelectionRequest`: `baseline_val: EvalResult`, `candidates: Sequence[Candidate]`, `comparator: Comparator`, `evaluate_test: Callable[[Candidate | None], EvalResult]` (`None` means the baseline model).
- `RankedCandidate`: `candidate`, `delta_wer_vs_baseline`, `comparison: ComparisonResult`, `eligible: bool`, `reason: str`.
- `SelectionResult`: `verdict` (`winner` | `baseline_retained`), `winner: Candidate | None`, `ranking: list[RankedCandidate]`, `comparison_table: list[ComparisonRow]` (all finalists side by side: Val WER, CER, difference from the baseline, interval, verdict, parameters, Val loss, steps), `test_status` (`confirmed` | `not_confirmed` | `not_evaluated`), `test_comparison: ComparisonResult | None`, `review_list` (passed through from the Evaluator unchanged), `reason: str` (one plain sentence).

### Rules
1. LOCKED (owner): Val decides the choice and the comparison with the baseline. The Test split is evaluated ONCE, for the winner and for the baseline, as the final unbiased confirmation.
2. LOCKED: a candidate is eligible only if its Val comparison with the baseline reaches `select.min_verdict` (DEFAULT `improved`, at the lenient level of evidence set in the Evaluator config).
3. LOCKED: among eligible candidates the lowest Val WER wins. A difference below `select.tie_epsilon` counts as a tie, and the lower Val loss wins the tie.
4. LOCKED (owner): if no candidate is eligible, the result is `baseline_retained`. This is a legitimate outcome, not a failure (exit code 0; the report states that the baseline stays).
5. LOCKED (owner, option a): if Val says "improved" but the Test comparison does not confirm it (no clear difference, or worse), the winner stays the winner, and the result is flagged `not_confirmed` with the Test numbers beside it.
6. LOCKED (owner): the report shows ALL finalists in one comparison table.
7. LOCKED (owner): the listening list produced by the Evaluator is passed through to the report. The Selector does not decide by it (O43).
8. LOCKED: the test callable may be called at most once per model. A second call FAILS (guards against tuning on Test).
9. LOCKED: deterministic; no randomness.
10. DEFAULT: when a comparison returns `insufficient_groups`, the candidate is judged by the plain WER difference, flagged "no confidence statement" (`select.on_insufficient_groups: plain_difference`; the alternative is `reject`).

### Config keys it reads (section `select`)
- `select.min_verdict` [DEFAULT: `improved`]
- `select.tie_epsilon` [DEFAULT: 0.001 absolute WER]
- `select.on_insufficient_groups` [DEFAULT: `plain_difference`]
- `select.test_confirmation` [DEFAULT: true]

### Out of scope (must NOT do)
- No training, no evaluation, no searching.
- No saving, copying or exporting of models (ModelRegistry, ModelExporter).
- No use of the owner's listening verdict as an automatic gate.
- No re-running of the Test set.
- No file writing; no network.

### Edge cases and failure behavior
- No candidates: FAIL naming the cause (the search returned nothing).
- Candidate and baseline evaluated on different file sets or `eval_version`: FAIL (the comparison would be invalid); `not_comparable` candidates are not eligible.
- A candidate without a Val result: FAIL naming it.
- Duplicate trial ids: FAIL.
- Test callable invoked twice for one model: FAIL.
- Test evaluation fails: the winner is kept, `test_status: not_evaluated`, and the error is reported (not hidden).
- `test_confirmation: false`: `test_status: not_evaluated`.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL no candidates | search produced no completed trial | read the search report and its failure reasons |
| FAIL different eval_version | baseline evaluated with an old profile | re-evaluate the baseline with the current profile |
| always `baseline_retained` | thresholds too strict, or training does not help | read the comparison table; check `min_verdict` and the live training curves |
| `not_confirmed` on Test | Val is small or the candidate overfit Val | read the Test numbers beside it, listen to the review list |
| FAIL test called twice | orchestration bug | fix the caller; never bypass the guard |

### Acceptance tests (offline; fake comparator and fake test evaluator)
1. A clearly better candidate wins and appears first in the table.
2. No eligible candidate gives `baseline_retained` (not an error).
3. A tie within `tie_epsilon` is decided by the lower Val loss.
4. Val says improved and Test says no clear difference: the winner stays and `not_confirmed` is set.
5. The test callable is invoked exactly once for the winner and once for the baseline; a second call FAILS.
6. Different `eval_version` or file sets FAIL; `not_comparable` candidates are ineligible.
7. `insufficient_groups` follows the configured policy.
8. The comparison table contains every finalist with all columns, in a fixed order.
9. The listening list is passed through unchanged.
10. No file access, no network, no model imports.

### OPEN items (agent presents options, owner decides)
- Aggregating finalists that were repeated with several seeds (later stage).
- How the owner's listening verdict is recorded (O43).
- Whether the Test confirmation should use a stricter level of evidence than Val.

---

## PRD Component 10: ModelRegistry

### Priority / order and dependencies
- Depends on Config (`registry.*`), on `SelectionResult` (Selector) and on `RunInfo` (Orchestrator / RunManager).
- Runs after the Selector and before the ModelExporter and the ReportBuilder.

### Responsibility
Store the selected model together with its metadata as an immutable, numbered entry, and keep an index of all entries.

### Interface and implementations
- `ModelRegistry.register(request: RegisterRequest) -> RegistryEntry`
- `ModelRegistry.get(entry_id: str) -> RegistryEntry`, `ModelRegistry.list_entries() -> list[RegistryEntry]`
- `ModelRegistry.best(eval_version: int, test_fingerprint: str) -> RegistryEntry | None`
- `ModelRegistry.verify(entry_id: str) -> VerifyReport` (checks the stored files against `SHA256SUMS`).
- `ModelRegistry.rebuild_index() -> None` (the index is derived from the entry cards).
- One implementation, on the file system, with `pathlib` only. No interface is created. No Hugging Face imports.

### Inputs -> Outputs (typed, dataclasses)
- `ModelCard`: base model name and path, data / split / subsample / test-set fingerprints, config or diff reference, `eval_version`, Val and Test summary (WER, CER, interval, verdict), decode settings, searched parameters, library versions, seed, generation time (passed in), the Selector's reason.
- `RegisterRequest`: `selection: SelectionResult`, `card: ModelCard`, `experiment_id: str`, `finalist_dirs: Sequence[Path]`.
- `RegistryEntry`: `entry_id` (for example `model_0001`), `experiment_id`, `kind` (`model` | `baseline_retained`), `model_dir: Path | None`, `card_path: Path`, `checksums_path: Path | None`, `test_evaluations: int` (how many registered entries, this one included, were evaluated on the same test set).
- `VerifyReport`: files checked, missing files, mismatching files, `ok: bool`.

### Rules
1. LOCKED (owner): the registry keeps the WINNER and its metadata. The folders of the other finalists and the training checkpoints are deleted after a successful registration (`registry.cleanup`), unless `registry.keep_finalists` is true.
2. LOCKED (owner): numbering, not timestamps, in all names: entries are `model_0001`, `model_0002`, and so on, each linked to its experiment (`exp_0001`) and to the config or diff file of that experiment.
3. LOCKED (owner): `model_card.json` plus a short human-readable `MODEL_CARD.md`.
4. LOCKED (owner): entries are immutable. An existing entry is never overwritten; a new run creates a new entry.
5. LOCKED (owner): an index file (`index.csv`) lists all entries with their metrics. "Best so far" is computed only among entries with the same `eval_version` and the same test-set fingerprint, because other comparisons are invalid.
6. LOCKED (owner): when the Selector returns "baseline retained", an entry of kind `baseline_retained` is still recorded (card and index row, no model folder), so the history is complete.
7. LOCKED (owner): a `SHA256SUMS` file is written for the stored model files, to detect a corrupted copy after moving the registry between machines on a removable drive. This is a transfer-integrity check and is independent of the model identification by name and path in the ModelLoader.
8. LOCKED: registration is crash-safe: files are written into a temporary folder and renamed into place at the end; the cleanup of finalists and checkpoints happens only after the checksums are written and verified. A crash leaves either a complete entry or none.
9. DEFAULT: the winner folder is MOVED into the registry (a copy when the experiment folder is on another drive), so a large model does not take twice the disk space.
10. LOCKED: no network, no model loading, no evaluation.
11. LOCKED (owner): test-set usage counter. For every test-set fingerprint the registry counts how many entries were evaluated on it and shows the count in the index and the card ("this test set has been looked at N times"). Comparison between experiments is made on Val; the Test numbers are reported but not used to pick among experiments. The count covers registered entries only, so it is a lower bound.

### Config keys it reads (section `registry`)
- `registry.dir`
- `registry.keep_finalists` [DEFAULT: false]
- `registry.cleanup` [DEFAULT: true]
- `registry.transfer` [DEFAULT: `move`; alternative `copy`]
- `registry.checksums` [DEFAULT: true]

### Out of scope (must NOT do)
- No choosing the winner (Selector), no evaluation, no training.
- No format conversion (ModelExporter).
- No loading of models or any Hugging Face / torch import.
- No deletion of anything except the finalist folders and checkpoints named in the request, and only after a successful registration.
- No network; no interactive prompts.

### Edge cases and failure behavior
- Registry folder missing: created. Entry number already exists on disk: FAIL (never overwrite).
- Winner model folder missing or empty: FAIL naming the path.
- Not enough disk space or a crash during registration: the temporary folder is removed and the registry is left unchanged; FAIL with the reason.
- `verify` finds a missing or changed file: FAIL listing the files.
- Index missing or corrupted: FAIL naming the problem, with the fix `rebuild_index()`; the index is never silently rebuilt.
- `best()` with no matching entries: returns `None`.
- `baseline_retained` with no Test result: recorded, flagged `test_not_evaluated`.
- Paths with spaces or Hebrew characters work.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL entry exists | two runs raced, or a leftover temp folder | remove the leftover temp folder named in the message; never delete a finished entry |
| verify says files changed | corrupted copy between machines | copy the entry again from the source |
| index does not match entries | index edited by hand or a crash | run `rebuild_index()` |
| disk full on registration | large model, move across drives | free space, or set `registry.transfer: copy` knowingly |
| `best` returns nothing | different `eval_version` or test set | compare only like with like; re-evaluate under the current profile |

### Acceptance tests (offline, temporary folders, small fake model files)
1. A registration creates `model_0001` with the model files, `model_card.json`, `MODEL_CARD.md` and `SHA256SUMS`, and adds an index row.
2. A second registration creates `model_0002`; the first is unchanged byte for byte.
3. A crash simulated before the final rename leaves no entry and no stray files; cleanup does not run.
4. After success, the finalist folders and checkpoints are removed; with `keep_finalists: true` they stay.
5. `baseline_retained` creates an entry without a model folder and an index row.
6. `verify` passes on an intact entry and FAILS on a changed file, naming it.
7. `best` ignores entries with another `eval_version` or test fingerprint.
8. `rebuild_index` reproduces the index from the cards; a corrupted index FAILS and is not rebuilt silently.
9. Move and copy modes both work; paths with spaces and Hebrew characters work.
10. No network; no Hugging Face or torch import.

### OPEN items (agent presents options, owner decides)
- Whether a registry entry should also store the review list (listening samples) next to the card, or only reference the report.
- Retention of old entries (never deleted automatically in v1).

---

## PRD Component 14: ReportBuilder

### Priority / order and dependencies
- Depends on Config (`report.*`), on `SelectionResult` (Selector), `EvalResult` and `FileScore` with attributes (Evaluator), `StepMetrics` and the `MetricsSink` protocol (TrainerCore), `SearchResult` (HyperparamSearcher), and optionally on a learning-curve result (Orchestrator, type OPEN).
- It is the last stage in the pipeline: it only presents what the other components produced.

### Responsibility
Turn the results of a run into a self-contained visual report, and into a live progress page while training runs, that show at a glance whether the model succeeded, where it failed and what could be changed.

### Interface and implementations
- `ReportBuilder.build(data: ReportData) -> ReportOutput`: the final report as one HTML text.
- `HtmlProgressSink` (implements the TrainerCore `MetricsSink` protocol): writes the live progress page and regenerates it as metric records arrive.
- `DiagnosisRule` (typing.Protocol): `evaluate(data: ReportData) -> DiagnosisHint | None`. Rules are selected by name through a registry; a new rule is a new class. Several implementations exist from the start, so the protocol is justified.
- Chart drawing (line, bars with intervals, scatter, histogram) is internal pure code producing inline SVG. No interface is created for it.
- Correlation is a pure function (rank correlation), in its own module.
- Writing the report to disk is done by the caller (RunManager, O14), except for the progress sink, which writes its own live file like the CSV sink does.

### Inputs -> Outputs (typed, dataclasses)
- `ReportData`: `selection: SelectionResult`, `baseline_val: EvalResult`, `test_results: Mapping[str, EvalResult]`, `histories: Mapping[int, list[StepMetrics]]` (trial id to training history), `search: SearchResult | None`, `learning_curve: LearningCurve | None` (type OPEN), `run_info: RunInfo` (experiment number, config reference, model info, data and split fingerprints, timings, library versions, generation time passed in), `settings: ReportSettings`.
- `DiagnosisHint`: `rule: str`, `severity` (`info` | `warning`), `message: str`, `evidence: Mapping[str, float | str]`, `caveat: str`.
- `CorrelationRow`: `attribute: str`, `n: int`, `rho: float | None`, `note: str` (for example "too few points" or "constant").
- `ReportOutput`: `html: str`, `sections: list[str]` (names of sections that were produced), `warnings: list[str]`.

### Rules
1. LOCKED (owner): the report is ONE self-contained HTML file that opens offline in a browser. No network, no external scripts, fonts or images. Charts are inline SVG drawn by our own code. No new dependency (no matplotlib).
2. LOCKED (owner): the top of the page is a plain-language summary suitable for presenting: the verdict (winner or "the baseline stays"), WER of the baseline and of the winner with the interval, and the Test confirmation. The details follow below it.
3. LOCKED (owner): sections: (a) comparison table of all finalists; (b) WER bars: baseline against each finalist with intervals; (c) training curves (train and Val loss) of the winner and of the other trials; (d) distribution of per-file WER and the share of files under the thresholds; (e) WER by group; (f) search: each searched parameter against Val loss, with rank correlation; (g) learning curve when it was run; (h) correlations between per-file errors and file attributes; (i) diagnosis hints; (j) the listening list with word-level differences and audio paths; (k) run information (experiment number, config reference, fingerprints, versions, timings, and how many times this test set has already been evaluated, passed in by the Orchestrator from the registry).
4. LOCKED (owner): correlations between errors and attributes are shown so that the owner can see where the problem is. They are rank correlations computed in our own code, always shown with the number of points, and always accompanied by the note that a correlation is not a cause.
5. LOCKED: diagnosis hints come from rules whose thresholds are in Config. Every hint shows its evidence and a caveat, and is worded as a hint, never as a stated fact.
6. LOCKED: Hebrew text (references, hypotheses) is displayed right-to-left correctly; labels and headings are English (owner approved).
7. LOCKED: the live progress page refreshes itself, shows the trend of the losses, learning rate, step time and estimated time left, and works while the training is still running.
8. LOCKED: a section whose input is missing (no search, no learning curve, no attributes) says "not available in this run". It is never an error and never an empty chart.
9. LOCKED: the builder never recomputes WER, never changes a result and never decides. It presents and correlates what it is given.
10. LOCKED: the same inputs give the same output, including the generation time (passed in, not read from the clock).
11. LOCKED: all text from the data is HTML-escaped (a transcript may contain markup characters).
12. LOCKED: the page prints to PDF cleanly from the browser (print style).
13. DEFAULT: color-blind-safe palette.
14. DEFAULT: the listening list shows audio paths as text and `file:` links (no audio is embedded, to keep the file small).

### Config keys it reads (section `report`)
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

### Out of scope (must NOT do)
- No computing of WER, CER or significance (Evaluator).
- No choosing winners (Selector).
- No training, no data processing, no network.
- No external libraries for charts, no JavaScript from outside.
- No embedding of audio.
- No claims about causes.

### Edge cases and failure behavior
- Very many files: scatter points are thinned deterministically to `max_points_per_chart` and the page says so.
- Missing values (NaN) or constant attributes: the correlation row says "too few points" or "constant".
- A history without any Val loss: the curves section shows the training loss only and a note.
- Paths with spaces or Hebrew characters: links are URL-encoded; the plain text path is shown too.
- Unknown rule name in Config: FAIL listing the registered names.
- A rule raises an exception: the hint is replaced by a warning in the report (the report is still produced) and the error is logged.
- No finalists at all: the summary says so; the report still shows what exists.
- Progress sink cannot write the file: log the error and continue training (the live page must never stop a run).

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| a chart is empty or missing | input not available in this run | read the "not available" note; check that the search or the learning curve was run |
| Hebrew text appears reversed or garbled | page opened without UTF-8 | the page declares UTF-8; open it in a normal browser; do not edit the file by hand |
| correlation rows say "too few points" | small evaluation set | evaluate a larger set, or lower `report.min_points_for_correlation` knowingly |
| the live page does not refresh | the file is opened through a viewer that does not reload | open it in a browser; the refresh is built into the page |
| a hint looks wrong | threshold not calibrated | change the threshold in YAML; hints are advisory |
| the file is large | many points or long lists | lower `report.max_points_per_chart` |

### Acceptance tests (offline, synthetic results)
1. The summary panel shows the right verdict, numbers and interval for a winner case and for a "baseline retained" case.
2. The page contains no reference to an external address (no remote script, stylesheet, font or image).
3. The rank correlation matches hand-computed values, including ties; constant data gives a note and no number.
4. Each default hint rule fires on a synthetic history built to trigger it and stays silent on a healthy one.
5. Hebrew text cells are marked right-to-left; markup characters in a transcript are escaped.
6. A missing section input produces the "not available" note and no error.
7. Same inputs give byte-identical output.
8. The progress sink regenerates the page as records arrive, the page contains a refresh instruction, and a write failure does not stop the caller.
9. Thinning is deterministic and keeps the cap.
10. The print style is present.
11. Nothing imports a plotting library; no network.

### OPEN items (agent presents options, owner decides)
- The type of the learning-curve data (defined with the Orchestrator).
- The final list of attributes and the look of the report: to be changed together with the agent after the owner sees the first real output.
- Whether to add a second, shorter "one page" print layout for presentations.
- Calibration of the hint thresholds on real runs.

---

## PRD Component 11: ModelExporter

### Priority / order and dependencies
- Depends on Config (`export.*`) and on a `RegistryEntry` (ModelRegistry). Low priority: v1 contains only the identity exporter and the interface.
- Reason for existing (owner): the format in which the final model will be used is not known, so an export layer must exist.

### Responsibility
Convert a registered model into a target format, verify the result, and record the export.

### Interface and implementations
- `ModelExporter` (typing.Protocol): `name: str`, `requires: tuple[str, ...]` (names of the packages it needs), `export(source: Path, target_dir: Path) -> ExportResult`, `verify(result: ExportResult, check: VerifyInput) -> VerifyResult`.
- Implementations selected by name through a registry: v1 has one, `hf_safetensors` (the format the registry already stores). The protocol exists because other targets are planned (for example faster-whisper, whisper.cpp, ONNX); each future target is a new class.
- Format-specific libraries are imported only inside their own adapter module.

### Inputs -> Outputs (typed, dataclasses)
- `ExportResult`: `exporter: str`, `target_dir: Path`, `files: list[Path]`, `seconds: float`.
- `VerifyInput`: `audio_paths: list[Path]`, `expected_texts: list[str]` (transcripts of the same files from the registered HF model), `max_wer_diff: float`.
- `VerifyResult`: `level` (`load` | `transcribe`), `ok: bool`, `detail: str`, `wer_diff: float | None`.
- `ExportRecord`: exporter, target folder, verification result; appended to an `exports.json` inside the registry entry (the model card itself stays immutable).

### Rules
1. LOCKED (owner): v1 has only the `hf_safetensors` exporter and the interface. Other formats are added when the target is known, each with a separate approval for its dependency.
2. LOCKED (owner): export runs automatically after a winner is registered, for the formats listed in `export.formats` (DEFAULT: `hf_safetensors` only), and also on demand for any existing registry entry through a command.
3. LOCKED (owner): an exporter whose dependency is missing FAILS with a message naming the package and saying that the machine is offline, so the package must be brought in advance.
4. LOCKED (owner): every export is verified. The HF exporter verifies at `load` level (the folder loads through the ModelLoader and the parameter count and mel bins match the card). Other exporters verify at `transcribe` level when their runtime is available: a short transcription of a few files, compared with the expected texts.
5. LOCKED: an export never modifies the registered model.
6. LOCKED: the exported files live in `exports/<exporter name>/` inside the registry entry.
7. DEFAULT: the HF exporter does not copy files; its result points to the entry's own model folder.

### Config keys it reads (section `export`)
- `export.formats` [DEFAULT: `hf_safetensors`]
- `export.verify` [DEFAULT: true]
- `export.verify_samples` [DEFAULT: 5]
- `export.max_wer_diff` [DEFAULT: 0.02, absolute WER difference allowed between the exported and the original model on the check files]

### Out of scope (must NOT do)
- No training, evaluation or selection; no changes to the registry entry's model files or card.
- No quantization or optimization in v1 (a format-specific later decision).
- No network, no downloads of converters.
- No interactive prompts.

### Edge cases and failure behavior
- Unknown exporter name: FAIL listing the registered names.
- Missing dependency: FAIL (rule 3).
- Target folder already exists and is not empty: FAIL (never overwrite).
- Verification fails (`ok: false`): the export is kept but marked failed in `exports.json`, and the run exits with a failure.
- Registry entry of kind `baseline_retained`: nothing to export; FAIL with a plain explanation when asked explicitly, skipped (and logged) in the automatic step.
- Source folder corrupted (checksums do not match): FAIL before exporting.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL missing package | converter not installed on the closed machine | bring the package in advance with the other wheels; do not enable online mode |
| verification differs too much | conversion changed the outputs | inspect the difference first; raise `max_wer_diff` only knowingly |
| FAIL target exists | repeated export | export to a new target folder or remove the failed one |
| FAIL nothing to export | the winner was "baseline retained" | no export is needed |

### Acceptance tests (offline, small fake model folders)
1. The HF exporter returns the entry's folder and its `load` verification passes.
2. Unknown exporter name FAILS and lists the registered names.
3. A fake exporter that declares a missing package FAILS with the package name.
4. A fake exporter with a deliberately wrong output fails verification and is recorded as failed.
5. An export is recorded in `exports.json` and the model card is unchanged byte for byte.
6. An existing non-empty target folder FAILS.
7. Automatic export runs for the formats in Config and is skipped for `baseline_retained`.
8. No network; format libraries are imported only inside their adapter modules.

### OPEN items (agent presents options, owner decides)
- The list of real target formats (manager question).
- A dependency approval for each real exporter, when it is added.
- Whether quantization is wanted for the final model (speed and size against accuracy).

---

## PRD Component 12: RunManager

### Priority / order and dependencies
- Depends on Config (`run.*`) and on the frozen `AppConfig` (Config component). Used by the Orchestrator; its `RunInfo` is read by the ModelRegistry.
- Implement before the Orchestrator.

### Responsibility
Create and manage numbered experiment folders (config snapshots, diffs, status, lock, logs, and the reports written into them).

### Interface and implementations
Three small classes in one component, each with one job (no interfaces, one implementation each):
- `ExperimentStore`: `create(spec: ExperimentSpec) -> ExperimentHandle`, `open(experiment_id: str) -> ExperimentHandle` (for resume), `list_experiments() -> list[ExperimentSummary]`, `propose_cleanup(policy) -> list[Path]`, `delete(paths: Sequence[Path])`.
- `ExperimentHandle`: `paths: ExperimentPaths`, `set_status(status, reason)`, `mark_stage(stage, status)`, `write_report(name: str, content: str | bytes)`, `lock()` as a context manager.
- `diff_configs(base: Mapping, new: Mapping) -> Mapping` (pure function) and `LogSetup.configure(handle) -> None`.

### Inputs -> Outputs (typed, dataclasses)
- `ExperimentSpec`: `config: AppConfig`, `parent_id: str | None`, `code_version: str`, `library_versions: Mapping[str, str]`, `created_at: str` (passed in, not read from the clock inside the component).
- `ExperimentPaths`: `root`, `config_full`, `config_diff`, `run_info`, `status`, `logs`, `metrics`, `search`, `candidates`, `reports`.
- `RunInfo`: `experiment_id`, `config_ref: Path`, `parent_id`, `code_version`, `library_versions`, `seed`, `created_at`. Written to `run_info.json`.
- `ExperimentSummary`: `experiment_id`, `status`, `parent_id`, `created_at`.

### Rules
1. LOCKED (owner): the folder structure of one experiment:
   ```
   experiments/exp_0001/
     config_full.yaml      (the full config that ran)
     config_diff.yaml      (the difference from the parent experiment, or from the defaults)
     run_info.json         (seed, library versions, code version, parent, config reference)
     status.json           (status, reason, finished stages)
     run.lock              (heartbeat file while a run is active)
     logs/run.log
     metrics/              (live metrics file, progress page)
     search/               (study file of the hyperparameter search)
     candidates/           (finalist folders, removed after registration)
     reports/              (final HTML, JSON, predictions table)
   ```
2. LOCKED (owner): numbering, not timestamps, in names. The next experiment gets the next free number (one above the highest existing). Dates appear only inside files and are passed in from outside. Creation is race-safe (an exclusive folder creation, so two processes never get the same number).
3. LOCKED (owner): parent and child. A new experiment can name a parent (`extends: exp_0003`). The diff is computed automatically and stored. Without a parent the diff is against the defaults.
4. LOCKED (owner): `config_full.yaml` is always stored, is read-only after the run starts, and is never rewritten. A resume checks that the config matches the stored one; a mismatch FAILS.
5. LOCKED (owner): `run_info.json` records the seed, the versions of the libraries and the code version number, so that the run can be reproduced.
6. LOCKED (owner): nothing is deleted automatically, ever. A cleanup command only PROPOSES what could be deleted (the default is a list only); deletion happens only when the command is given an explicit flag. The registry is never touched by it.
7. LOCKED (owner): RunManager writes ALL the reports into the experiment folder. The other components return objects and do not write files (the metrics sinks are the only exception, by design).
8. LOCKED (owner): logging: one file per experiment with levels (detailed in the file), and only the important lines on the console.
9. LOCKED (owner): a run that fails midway leaves its folder in place with status `failed` and the reason in `status.json`. It is never removed. A resume continues in the same folder, under the rules of the TrainerCore and the HyperparamSearcher.
10. LOCKED (owner): a lock prevents two processes from running in one folder. The lock file is a heartbeat: the running process touches it periodically. A lock whose heartbeat is older than `run.lock_stale_minutes` is reported as stale, and the run FAILS with the exact instruction for removing it by hand. It is never removed automatically. (Probing a process id is not used: on Windows, signalling a process id can terminate that process.)
11. LOCKED: all paths use `pathlib`; paths with spaces and Hebrew characters work; a path longer than the Windows limit produces a warning that names the path and the fix (a shorter root folder).
12. LOCKED: no network, no model code, no training logic.

### Config keys it reads (section `run`)
- `run.experiments_dir`
- `run.parent` [DEFAULT: none]
- `run.id_digits` [DEFAULT: 4]
- `run.config_readonly` [DEFAULT: true]
- `run.lock_heartbeat_seconds` [DEFAULT: 30], `run.lock_stale_minutes` [DEFAULT: 10]
- `run.log.console_level` [DEFAULT: INFO], `run.log.file_level` [DEFAULT: DEBUG]

### Out of scope (must NOT do)
- No loading or merging of config (Config component).
- No training, evaluation, selection or model storage (the registry stores models).
- No automatic deletion; no interactive prompts; no network.
- No computing of reports (it only writes what it is given).

### Edge cases and failure behavior
- Experiments folder missing: created.
- Parent id does not exist: FAIL naming it and listing the existing experiments.
- Resume with a different config than the stored one: FAIL showing the first differing keys.
- Locked experiment: FAIL naming the heartbeat age and, for a stale lock, the removal instruction.
- `status.json` corrupted: FAIL naming the file; the folder is not touched.
- Disk full or folder not writable: FAIL with the path and the cause.
- Two processes creating experiments at once: they get different numbers.
- Number gaps (a deleted folder in the middle): the next number is still the highest plus one, never a reused number.
- Cleanup without the explicit flag: lists only and changes nothing.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| FAIL experiment is locked | another run is active | wait, or stop that run |
| FAIL stale lock | a run died without releasing | check that no run is active, then delete the `run.lock` file named in the message |
| FAIL config differs on resume | config edited after the run started | restore the stored config, or start a new experiment with `extends` |
| warning path too long | experiments folder deep in the tree | move `run.experiments_dir` closer to the drive root |
| no console output from components | console level too high | lower `run.log.console_level`; details are always in `logs/run.log` |

### Acceptance tests (offline, temporary folders)
1. `create` makes `exp_0001` with the full config, the diff, `run_info.json`, `status.json` and the folders; a second call makes `exp_0002`.
2. Creation is race-safe: concurrent creations (threads and processes) produce distinct numbers.
3. A child experiment's diff holds exactly the changed keys; with no parent the diff is against the defaults.
4. `config_full.yaml` is read-only and unchanged after the run; a resume with a different config FAILS.
5. A failed run keeps its folder with status `failed` and the reason; resume reopens it.
6. The lock blocks a second opener; the heartbeat keeps it fresh; a stale lock FAILS with the removal instruction and is not removed.
7. `write_report` writes only inside `reports/` and refuses paths that escape it.
8. Cleanup lists candidates without deleting; with the flag it deletes only those listed and never the registry.
9. Log file receives detailed lines; the console only the configured level.
10. Paths with spaces and Hebrew characters work; a too-long path warns.
11. No network; no model imports.

### OPEN items (agent presents options, owner decides)
- How the code version number is derived (a version constant in the package, or a hash of the source files).
- The exact cleanup policy options (for example "failed experiments older than N", proposals only).
- Windows long-path behavior to be checked on the closed machine.

---

## PRD Component 13: Orchestrator and CLI

### Priority / order and dependencies
- Depends on all other components, through their interfaces only. Implemented LAST among the pipeline components (a minimal first version may come earlier for the first end-to-end run, see the build-order discussion).
- It is the only place where concrete classes are named (composition root, `wiring.py`).

### Responsibility
Wire the components together and run the whole pipeline, or any single stage of it, from one command with full automation and a clear exit code.

### Interface and implementations
- `PipelineStage` (typing.Protocol): `name: str`, `config_slice(config) -> Mapping` (the part of the config the stage depends on), `run(ctx: StageContext) -> StageResult`. Implementations, selected by name through a registry: `prepare`, `train`, `search`, `evaluate`, `select_and_register`, `export`, `report`, `learning_curve`, `smoke`, `benchmark`. A new stage is a new class.
- `Orchestrator.run(plan: RunPlan) -> RunOutcome`: runs the stages in the plan, writes completion markers, skips finished stages, stops at the first failure.
- `wiring.build_services(config) -> Services`: the composition root; the only module that names concrete classes, built from the names in Config through the registries.
- `SleepGuard` (typing.Protocol): `__enter__` / `__exit__`. Implementations: `WindowsSleepGuard` (standard library only, no admin rights) and `NoopSleepGuard`.
- CLI: `cli.py` using the standard library `argparse` (no new dependency). Every subcommand has `--help` with an example.

### Inputs -> Outputs (typed, dataclasses)
- `StageContext`: `config`, `experiment: ExperimentHandle`, `services: Services`, `stop_flag` (set on interrupt), `previous: Mapping[str, StageResult]`.
- `StageResult`: `status` (`ok` | `skipped` | `failed`), `summary: Mapping[str, str | float | int]`, `artifacts: list[Path]`, `error: str | None`.
- `RunPlan`: `stages: list[str]`, `experiment`.
- `RunOutcome`: `exit_code: int`, `verdict` (`winner` | `baseline_retained` | `n/a`), `experiment_id`, `stage_results`.
- Final output: one JSON summary printed at the end (the single permitted print): experiment id, verdict, key numbers (baseline and winner WER with interval, Test status), and the paths of the report and the registry entry.

### Commands (owner approved)
`run`, `prepare`, `train`, `search`, `evaluate`, `report`, `export`, `registry` (list, verify), `learning-curve`, `smoke`, `check`, `benchmark`, `cleanup`, `config-docs`.
Shared options on all commands: `--config <file>`, `--set a.b=value` (already decided in the Config component), `--parent exp_0003`.

### Rules
1. LOCKED (owner): `run` is the single command that does everything: prepare data, baseline evaluation, search on a screening subset, retrain of the finalists on the full data, Val WER of the finalists, selection (with the one-time Test confirmation), registration, export, and the final report. The stage commands exist for debugging and partial work.
2. LOCKED (owner): each stage writes a completion marker into `status.json`. A rerun skips stages that completed, if the part of the config that the stage depends on (its `config_slice`) did not change.
3. LOCKED (owner): the run stops at the first failed stage, exits with a non-zero code and says which stage failed. There is no automatic retry, except that a single failed trial inside the search is tolerated by the HyperparamSearcher.
4. LOCKED (owner): exit codes: 0 success (including "the baseline stays"), 2 config error, 3 data error, 4 failure in a training or evaluation stage, 5 export or verification failure, 130 interrupted by the user.
5. LOCKED (owner): interruption (Ctrl+C or a closing console): the stop flag is set, the TrainerCore saves a checkpoint and stops, the status becomes `stopped`, and a rerun resumes. Because closing a console window on Windows may not allow a graceful stop, checkpoints are also saved at a fixed rate by the TrainerCore.
6. LOCKED (owner): sleep prevention is ON by default for `run`, `search` and `train` on Windows and is released when the process ends. It uses only the standard library and needs no admin rights. It can be turned off in Config (`run.prevent_sleep`).
7. LOCKED (owner): `smoke` generates short synthetic audio (for example sine tones) with random transcripts and runs the whole chain on a tiny model, checking the mechanics, not the quality. The tiny model for tests is a random-weight model with the real tokenizer, created once at home by a script and stored with the tests (option b of O32).
8. LOCKED (owner): `learning-curve` runs the same pipeline on several Subsampler sizes (with `scope: train_only`) with fixed hyperparameters (from the config, or from an existing experiment with `--params-from exp_0003`) and passes the curve data to the ReportBuilder.
9. LOCKED (owner): `benchmark` runs a few training steps and prints seconds per step and the projected time of the configured run, for planning and for answering the time-budget question.
10. LOCKED (owner): `report` rebuilds the report of an existing experiment without recomputing anything (useful after changing the report settings).
11. LOCKED (owner): `check` prints in plain words what is fine and what is not: Python version, packages, model and data folders, free disk space, memory, number of cores. It also verifies the code against the `MANIFEST.sha256` of the release (so the code on the closed machine is the code that was tested at home) and prints the code version number (owner approved).
12. LOCKED: the Orchestrator builds the per-file `attributes` for the Evaluator (duration, label token count, reference word count, `extra` fields and text flags) from the normalized samples and the feature examples.
13. LOCKED: the baseline is evaluated once per experiment on Val; the Test split is evaluated only through the Selector's one-time callable.
14. LOCKED: no interactive prompts, ever; everything comes from Config and flags.
15. LOCKED: only `wiring.py` names concrete classes. Stages talk to interfaces.
16. LOCKED (owner): `prepare` alone (no training) doubles as the data inspection when new data arrives: it prints the structure, the distributions and what was rejected and why, and writes the reports. A short `docs/DATA_ONBOARDING.md` gives the checklist for mapping the columns of a new dataset.
17. LOCKED (owner): `config-docs` generates `docs/CONFIG_REFERENCE.md` from the config schemas (see the Config brief).

### Config keys it reads (sections `pipeline`, `run`, `smoke`, `learning_curve`, `benchmark`)
- `pipeline.stages` [DEFAULT: `prepare`, `search`, `select_and_register`, `export`, `report`]
- `pipeline.skip_completed` [DEFAULT: true]
- `run.prevent_sleep` [DEFAULT: true]
- `smoke.n_samples` [DEFAULT: 12], `smoke.duration_sec` [DEFAULT: 3], `smoke.steps` [DEFAULT: 3], `smoke.model_dir`
- `learning_curve.presets` [DEFAULT: `smoke`, `tiny`, `half`, `full`], `learning_curve.params_from` [DEFAULT: none]
- `benchmark.steps` [DEFAULT: 3]

### Out of scope (must NOT do)
- No component logic (everything is delegated to the components).
- No new metrics, no model code, no file formats of its own.
- No network, no interactive prompts, no automatic retries of failed stages.
- No deletion except through the explicit `cleanup` flag.

### Edge cases and failure behavior
- Unknown stage or subcommand: FAIL listing the registered names.
- A stage's config slice changed after its marker: the stage runs again, and later stages are invalidated and run again.
- Rerun of a completed experiment: nothing runs; the summary of the previous result is printed.
- Interrupt during a stage: marker not written; status `stopped`.
- `--parent` that does not exist: FAIL listing existing experiments.
- `smoke` without the test model folder: FAIL naming the script that creates it.
- `learning-curve` with a size larger than the data: FAIL from the Subsampler, shown with the failing size.
- `check` finds a problem: exit code 2 with the full list of problems (not only the first).
- A stage reports `failed` but a later stage depends on it: not run, listed as skipped.
- Sleep guard unavailable (not Windows): the no-op guard is used, with a log line.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| exit code 2 | invalid or missing config key | read the message (it names the key); fix the YAML |
| exit code 3 | data path or column mapping wrong | read `rejected.csv` and the `prepare` report; fix `data.*` in YAML |
| exit code 4 | training or evaluation failed | read `logs/run.log`, `status.json` and the failing stage; consult the component tables in TROUBLESHOOTING |
| exit code 5 | export dependency missing or verification failed | see the ModelExporter table |
| rerun does nothing | all stages completed | use a new experiment (`--parent`), or change the config |
| rerun repeats a stage | its config slice changed | intended; check the diff in `config_diff.yaml` |
| the machine went to sleep | sleep prevention off or blocked by policy | check `run.prevent_sleep`; ask the machine owner about power policy |

### Acceptance tests (offline; fake services, then the smoke test with the tiny test model)
1. `run` with fake services executes the stages in order and prints the single JSON summary.
2. Completion markers: a rerun skips completed stages; changing one stage's config slice reruns it and the later stages.
3. A failing stage stops the run with the right exit code and names the stage; later stages are not run.
4. Each exit code is produced by a corresponding failure (config, data, training, export, interrupt).
5. A simulated interrupt sets the stop flag, the status becomes `stopped`, and a rerun resumes.
6. The Windows sleep guard calls the system function on entry and restores on exit (with the system call faked); the no-op guard works elsewhere.
7. `smoke` runs the whole chain on synthetic data and the tiny test model and produces every artifact (config files, metrics, report, registry entry).
8. `learning-curve` produces the curve data for the configured sizes with identical Test sets.
9. `benchmark` prints seconds per step and a projection.
10. `report` rebuilds the report without calling the trainer or the evaluator (they are replaced by fakes that fail if called).
11. `check` lists all problems at once; a healthy environment passes.
12. Only `wiring.py` imports concrete classes; stages import interfaces only; no network; no interactive prompts.
13. Each subcommand's `--help` shows a usage example.

### OPEN items (agent presents options, owner decides)
- The build order and the first minimal end-to-end version (see the build-order discussion).
- The exact content of the `check` command after the home install test (O9).
- DECIDED by the owner: no diagnostics bundle (an internal agent exists at the work site and can help). The offline-install helper and the manifest maker are standalone tools (see PRD_tools).

---

## PRD: Standalone home-side tools

### How to run (usage header rule)
Each tool is a small script in `tools/`. Every script starts with a "How to run" header with the exact command and an example. Target usage (to be verified against the real scripts at the end of the project):
```
python tools/fetch_model.py --repo openai/whisper-tiny --out models/whisper-tiny
python tools/fetch_data.py  --repo portal-daf-yomi/daf-yomi-talmud-whisper-training --out data/daf_yomi_raw
python tools/fetch_wheels.py --requirements requirements.txt --out wheels/
python tools/make_test_model.py --tokenizer-from models/whisper-tiny --out tests/assets/tiny_whisper
python tools/make_manifest.py --root . --out MANIFEST.sha256
```

### Priority / order and dependencies
- Independent of the pipeline. Run at home, before the pipeline, by the owner (or by the agent at home).
- `fetch_wheels` and the environment files belong to the bootstrap step (see BUILD_ORDER).

### Responsibility
Prepare, outside the pipeline and with network access allowed ONLY here, everything the offline pipeline needs: models, data, packages, the tiny test model and the code manifest.

### The tools
1. `fetch_model`: downloads a model repository from the Hub into a local folder and checks that the files the ModelLoader requires are present. Writes `fetch_info.json` (repo, files, sizes).
2. `fetch_data`: downloads a dataset repository (the Parquet files) into a local folder. Writes `fetch_info.json`. This is the `HubFetcher` of the DataSource brief.
3. `fetch_wheels`: downloads all the packages from the requirements file, with their dependencies, for the target Python version and Windows, into one folder; then tests the installation into a clean virtual environment with no network (`--no-index --find-links`) and reports the result. Writes the pinned `requirements.lock`.
4. `make_test_model`: builds the tiny random-weight Whisper-like model with the real tokenizer (taken from a local model folder) used by the tests and by `smoke` (option b, O32). Created once; stored with the tests.
5. `make_manifest`: writes `MANIFEST.sha256` with the hash of every code file of a release, used by `check` on the closed machine.

### Rules
1. LOCKED: these tools are the ONLY code allowed to use the network. The pipeline never imports them and never runs them; a test checks that no pipeline module imports anything from `tools/`.
2. LOCKED: they use only libraries that the pipeline already depends on, plus the standard library. The Hub download uses `huggingface_hub`, which is installed together with `transformers`. No new dependency without approval.
3. LOCKED: no interactive prompts. Failures give a clear message and a non-zero exit code.
4. LOCKED: downloads are crash-safe: written into a temporary folder and renamed when complete; an existing complete download is reused (idempotent), never silently overwritten.
5. LOCKED: nothing from the network is executed (no `trust_remote_code`).
6. LOCKED: every tool prints, at the end, what it created and where, and how to use it next.
7. DEFAULT: the tools read repository names and folders from their command line; defaults may come from Config.

### Out of scope (must NOT do)
- No training, no data processing, no evaluation.
- No modification of downloaded files.
- No running on the closed machine (they are home-side).

### Edge cases and failure behavior
- No network or repository not found: FAIL with the repository name and the cause.
- Repository requires a login: FAIL and say that the item is not public; nothing is stored.
- Destination exists and is incomplete (no `fetch_info.json`): treated as interrupted, rebuilt.
- `fetch_wheels` cannot find a wheel for the target Python on Windows: FAIL naming the package and the Python version (this is the O1 signal), and suggests trying another Python version.
- The offline installation test fails: FAIL with the pip output summary and the missing package.
- `make_test_model` without a tokenizer folder: FAIL naming the missing files.

### If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| fetch FAIL not found | wrong repository name | check the name on the Hub; do not guess a similar one |
| fetch FAIL login required | repository is gated | obtain the files another way; the pipeline expects only local folders |
| wheel missing for Windows | package has no build for this Python | try another Python version (O1), or ask before replacing the package |
| offline install test fails | a dependency was not downloaded | rerun `fetch_wheels` (it includes dependencies); read the named package |
| `check` reports a manifest mismatch | a file was changed or the copy is damaged | copy the release again from home |

### Acceptance tests (offline; the network is faked)
1. A fake Hub download writes a complete folder and `fetch_info.json`; a second run reuses it.
2. An interrupted download (no info file) is rebuilt.
3. `fetch_wheels` with a fake downloader produces the folder and the lock file; a missing wheel FAILS naming the package.
4. `make_test_model` creates a model that loads through the ModelLoader and runs a forward pass.
5. `make_manifest` output verifies on an unchanged tree and fails on a changed file.
6. No pipeline module imports from `tools/`.
7. No interactive prompts.

### OPEN items (agent presents options, owner decides)
- The exact requirements split: `requirements.txt` for the pipeline and `requirements-tools.txt` for the home-side tools only.
- Whether `fetch_data` should also verify the data against an expected row count from the dataset card.

---

