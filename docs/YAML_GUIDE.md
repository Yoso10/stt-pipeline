# YAML guide: how to write a config file correctly

## How to run

This page explains how a config file is built. It has no command of its own. To test any config WITHOUT running anything:

```
$env:PYTHONPATH = "src"
python -m stt check --config my.yaml
```

Example output (last line): `Everything checked is fine.` (exit code 0). With problems it lists ALL of them and ends with `N problem(s) found` (exit code 2).
A fully annotated example file: `docs/examples/annotated_example.yaml`. Every key and default: `docs/CONFIG_REFERENCE.md`. Recipes: `docs/RECIPES.md`.

How this page was checked (2026-11-08): against the code (file names are given) and, where it says "run", by running the command on the tiny synthetic data in a scratch folder outside the repo. Anything marked "(from code reading, not run)" was not run.

## 1. One schema, many sizes of file

Every config file uses the same keys. A file does not have to list them all. A short file is normal: it holds only what DIFFERS from the defaults in the code and from the file it extends.
To see what a run REALLY used, open `experiments/exp_NNNN/config_full.yaml` (everything resolved, read-only). `config_diff.yaml` in the same folder shows only the difference from the parent experiment (or from the defaults).

## 2. Where a value comes from (strongest first)

| # | Source | Form | Notes |
|---|---|---|---|
| 1 | command line | `--set a.b=value` (repeatable) | later `--set` wins over earlier |
| 2 | real environment variable | `STT__A__B=value` | beats `.env` |
| 3 | `.env` file | `STT__A__B=value` | read ONLY from the current folder, file name exactly `.env` |
| 4 | your YAML (and the YAML it extends) | `a:` / `b: value` | |
| 5 | defaults in the code | | `docs/CONFIG_REFERENCE.md` |

Source: `src/stt/config/loader.py:63-68`, `sources.py:125-184`.

Rules that cause surprises:

1. The name is `STT__SECTION__KEY`, lower-cased by the program, nested keys joined with `__`. Example: `STT__TRAIN__HYPERPARAMS__LEARNING_RATE=3e-5` is `train.hyperparams.learning_rate`. A section name keeps its single underscore: `STT__LEARNING_CURVE__PARAMS_FROM`.
2. A name WITHOUT the `STT__` prefix is silently ignored (no error).
3. A name WITH the prefix but a wrong key is an error: `Unknown key ...` (exit 2). So a typo in `.env` stops the run.
4. A `.env` or environment value overrides your YAML without any message. If a result looks wrong, open `config_full.yaml` of the experiment: it holds the final values.
5. Values from `--set`, the environment and `.env` are text; the schema converts them (`3e-5`, `true`). A list or a dict must be JSON: `--set split.ratios=[0.7,0.15,0.15]`, `--set 'subsample.size={"minutes": 30}'`.
6. The path of the `.env` file cannot be changed from the command line.

### The PowerShell 5.1 quoting trap (run)

In Windows PowerShell 5.1 the double quotes inside a JSON value are removed before the program sees them. This fails (exit 2, `Invalid configuration:`):

```
python -m stt run --config my.yaml --experiment exp_0001 --set 'pipeline.stages=["prepare","finalists"]'
```

This works, with the inner quotes escaped by a backslash:

```
python -m stt run --config my.yaml --experiment exp_0001 --set 'pipeline.stages=[\"prepare\",\"finalists\"]'
```

(The `--set 'train.sinks=["csv","console","html"]'` form in `python -m stt --help` is written for a shell that keeps the quotes.) The simplest way out: put lists and dicts in the YAML file instead of `--set`. A number such as `3e-5` or `[0.7,0.15,0.15]` without quotes is safe.

## 3. `extends`: one file builds on another

- `extends: other.yaml` is relative to the FILE THAT CONTAINS the line, not to the folder you run from (`sources.py:82`). A file in `docs/examples/` that extends `base.yaml` of `configs/` writes `extends: ../../configs/base.yaml`.
- Only ONE base file (a single text). A list is an error (exit 2). A missing base file or a cycle is an error that names the chain.
- Maps are merged key by key. Everything else, lists included, is replaced whole (`sources.py:31-39`).
- A child CANNOT delete a key of its base; it can only overwrite it. If the base names `search.parameters.grad_accum_steps`, the child cannot remove it by leaving it out.
- The `*_default.yaml` files in `configs/` each hold ONE section (they document the defaults) and extend `base.yaml`.

Which file in `configs/` extends which (from the `extends:` lines):

| File | extends |
|---|---|
| `base.yaml` | nothing (only `runtime.online`, `runtime.seed`) |
| the 16 `*_default.yaml` (data excluded), `data_gmara.yaml`, `smoke.yaml`, `turbo_full.yaml`, `work.yaml`, `work_cpu.yaml` | `base.yaml` |
| `home_full.yaml`, `home_turbo_full.yaml`, `home_turbo_run.yaml` | `data_gmara.yaml` |
| `work_like.yaml` | `home_full.yaml` |

## 4. Which keys MUST be given

Exactly five keys have no default (found by running `check` on an empty config):

| Key | Meaning |
|---|---|
| `data.source.type` | `csv` (a folder with `manifest.csv`) or `hf_parquet` (a folder of parquet files) |
| `data.source.path` | the data folder |
| `data.columns.audio` | the manifest column with the audio path |
| `data.columns.text` | the manifest column with the transcript |
| `model.path` | the local model folder (never a hub name) |

Plus the group rule: `data.columns.group` OR a `data.group_fallback` other than `fail`. The default is `fail`, so without a group column `prepare` stops with exit 3 (`No group column is mapped ... data.group_fallback is ...`) (run). The group keeps the clips of one lesson in one part; a wrong group leaks Test data into training.

Conditional (required only in some modes):

| When | Required key | Source |
|---|---|---|
| `data.source.type: hf_parquet` | `data.extract.dir` | `datasource/config.py:63-69` |
| `features.cache.enabled: true` | `features.cache.dir` | `featurepipeline/config.py:58` |
| `features.encoder_cache.enabled: true` | `features.encoder_cache.dir` | `featurepipeline/config.py:64` |
| `python -m stt run` (stage `select_and_register`) | `registry.dir` outside `run.experiments_dir` (checked before the run starts) | `cli.py:246-250` |
| `search.parameters` | only the six names `learning_rate, grad_accum_steps, batch_size, warmup_steps, weight_decay, max_grad_norm`; `max_steps` and `seed` are refused | `searcher/config.py:25,99-109` |
| `search.finalist_repeats` above 1 | not supported: the finalists stage stops with exit 2 | `finalists.py:94-96` (from code reading, not run) |

`docs/CONFIG_REFERENCE.md` shows `text.profiles.train.steps` and `text.profiles.eval.steps` as "(required)". That is wrong in practice: they have defaults (`textnormalizer/config.py:59-60`), and a YAML with only the five keys passed `check` (run).

### Why some files have no data path

1. The key has no default, but it may come from another place: the file in `extends`, `.env`, the environment or `--set`. If it is nowhere, the run stops with exit 2: `Missing required value 'model.path'. Set it in the YAML file, in the environment as STT__MODEL__PATH=..., or with --set model.path=...` (run). When a whole block is missing, the message names the block (`data.source`).
2. The file is a default sheet (`*_default.yaml`) and holds one section only.
3. The file inherits the paths: `home_turbo_run.yaml` gets `data` from `data_gmara.yaml`.

`smoke.yaml` DOES have `data.source.path` and `model.path` (`data/smoke`, `tests/assets/tiny_whisper`); the `smoke` command overrides them with its own values.

## 5. Required, optional, machine-specific

| Kind | Examples | Where to put it |
|---|---|---|
| Required, no default | the five keys above, and the group | the experiment YAML; machine paths may sit in `.env` |
| Optional, has a default | everything else (188 keys in all) | write it only to change it |
| Machine-specific | folders, `train.cpu_threads`, `runtime.online` | `.env`, or a YAML copy for that machine |
| One-off change | a single value for this run | `--set a.b=value` |

## 6. Create the YAML for a new dataset, step by step

1. Copy a template; never edit a shipped file. `Copy-Item configs\work.yaml configs\my_work.yaml` (its `<<< FILL IN ... >>>` markers show up as `PROBLEM` lines in `check` until you replace them).
2. Fill only: `data.source.type`, `data.source.path`, `data.columns.audio`, `data.columns.text`, `data.columns.group`, `model.path`.
3. `python -m stt check --config configs\my_work.yaml` until it says `Everything checked is fine.`
4. `python -m stt prepare --config configs\my_work.yaml`. Read the reports in `experiments\exp_NNNN\reports\`: `load_report.txt` (rows, rejections, durations), `split_report.txt`, `filter_report.txt`, `text_report.txt` (the character inventory), `rejected.csv`. More: `docs/DATA_ONBOARDING.md`.
5. From the reports set `text.*` (from the character inventory) and `filter.mode: apply` if some clips are longer than 30 s.
6. For a first rehearsal add a Subsampler setting (`docs/RECIPES.md`, Recipe 1). Remove it for the real run.
7. Set nothing else until a report tells you to. Keep `runtime.seed` fixed.
8. Read `config_full.yaml` of the first experiment and check it says what you meant.

## 7. Syntax rules that cause most mistakes

- Indent with 2 spaces. Never tabs (the error says `do not use tab characters`).
- `name: value` (a space after the colon). Booleans `true` / `false`. A list `[0.8, 0.1, 0.1]`. A number `1.0e-5`, `0.00001` or `1e-5` (all accepted; run).
- Windows paths: single quotes `'D:\stt\data'` (in double quotes a backslash is an escape), or forward slashes. Quote paths with spaces or Hebrew letters. Files are read as UTF-8.
- On the command line, keys are joined with dots: `--set train.hyperparams.learning_rate=3e-5`.
- An unknown key stops the run with the closest valid key: `Unknown key 'train.hyperparams.learning_rat'. Closest valid key: 'train.hyperparams.learning_rate'` (run, exit 2). Fix the spelling; do not add keys to the code.

## 8. `check`: the validate-only command

`python -m stt check --config my.yaml [--set a.b=value]` (`cli.py:514-524`, `check.py`). It changes nothing and makes no experiment folder.

It DOES check:

- every section: types, values, unknown keys (with the closest key), the five required keys, the conditional rules above. All problems are listed at once (run).
- that `model.path` and `data.source.path` EXIST as folders. A missing folder is a `PROBLEM` line and exit 2 (`check.py:72-78`).
- that the required Python packages are installed (information about disk, memory, cores).

It does NOT check (these need `prepare`):

- that the manifest has the columns you named, that the audio files exist, how many clips are rejected;
- the split gate (`split.min_samples_per_split`, `split.min_groups_per_split`), the filter, the Subsampler size;
- anything about the model inside its folder.

Example (run): the annotated example with two temporary folders given by `--set data.source.path=... --set model.path=...` gave `Everything checked is fine.`, exit 0. The same file with its own `D:\...` paths gave exit 2, because those folders did not exist on this machine.

## 9. Common failures

| Message / exit code | Meaning | Fix |
|---|---|---|
| exit 2, `Missing required value 'X'. Set it in the YAML file, in the environment as STT__..., or with --set ...` | a required key has no value | give it a value in one of the places |
| exit 2, `Unknown key 'X'. Closest valid key: 'Y'` | a typo, or a key that does not exist | use the key from `docs/CONFIG_REFERENCE.md` |
| exit 2, `Invalid configuration:` after a `--set` with a list | the quotes were removed by PowerShell 5.1 | escape them (section 2) or put the list in the YAML |
| exit 2, `Cannot resume exp_N: the config differs from the stored config_full.yaml` | `--experiment exp_N` with ANY changed key | restore the original settings, or start a new experiment (`--parent` only links it) |
| exit 2, `Cannot parse YAML file ... at line L, column C` | bad indentation or a tab | fix that line |
| exit 3 on `prepare`, `No group column is mapped` | no group column and `group_fallback: fail` | map `data.columns.group` |
| exit 3 on `prepare`, wrong path or columns, or too many rejected clips | the data path or a column name is wrong | read `rejected.csv` and `load_report.txt` |
| exit 3, `'val' has N samples (minimum 30)` | the split gate | more data, a better group column; do NOT just lower the gate for a real result (`smoke.yaml` lowers it on purpose, for mechanics only) |
| exit 4 | training or scoring failed | `experiments\exp_N\logs\run.log` |
| exit 130 | Ctrl+C | run the same command with `--experiment exp_N` |
