# PRD Component 0: Config

Status: LOCKED (except items marked PENDING)

## Priority / order and dependencies
- Order: first. Every other component depends on it. Config depends on no other component.

## Responsibility
Load, merge, validate and freeze all settings into one typed, read-only `AppConfig` object.

## Inputs -> Outputs
- Inputs: path to a YAML file (optionally with `extends: other.yaml`), environment variables
  and a `.env` file, CLI overrides (`--config path.yaml`, `--set a.b.c=value`, repeatable).
- Output: `AppConfig` (pydantic model, frozen). Sections: `data`, `model`, `training`, `eval`, `runtime`.
- Also provides: a function that dumps the fully resolved config to YAML (used later by RunManager).

## Decisions (locked)
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
   edit the Config loader). PENDING: confirm with owner.
8. `num_proc` and `dataloader_num_workers` default conservatively (0 or 1) because of Windows
   multiprocessing (spawn). PENDING: confirm with owner.
9. Python version: PENDING (decided after a clean-venv install test of all packages at home).

## Moved out of this component (recorded as direction, NOT locked)
- CLI subcommands (prepare / baseline / train / evaluate / select / auto / preflight): entry point + Orchestrator.
- Experiment numbering (exp_NNN), `config.resolved.yaml`, `config.diff.yaml`, `parent`, `mode`,
  split hash: future component `RunManager`.

## Config keys it reads
- `runtime.*` (defined here): `online` (bool, default false), `seed` (int), `num_proc`, `dataloader_num_workers`,
  `runs_dir` (path), `log_level`.
- Other sections are defined by the component that uses them.
- Environment variable mapping: `STT__SECTION__KEY` overrides `section.key` (exact prefix to be confirmed in implementation brief).

## Out of scope (must NOT do)
- No reading of audio or datasets, no downloads, no network.
- No creation of run folders, numbering, or diffs (RunManager).
- No CLI subcommands or pipeline flow.
- No interactive prompts.
- No import of torch / transformers / datasets / optuna / scipy.

## Edge cases and failure behavior
- Unknown key in YAML or `--set`: fail with the key name and the closest valid key.
- Wrong type (`--set training.learning_rate=abc`): fail immediately, name the key and expected type.
- Missing required value: fail, name the key and the three places it can be set.
- Split ratios not summing to 1 (when the data section exists): fail.
- Path that must exist does not exist: fail with the path and how to fix.
- `extends` cycle or missing base file: fail with the chain.
- Missing `.env`: allowed (use environment and defaults). Malformed `.env` line: fail with line number.
- Same input always gives the same `AppConfig` (deterministic).

## Acceptance tests (offline, tiny synthetic files)
1. Defaults only -> valid `AppConfig`.
2. YAML overrides defaults; ENV overrides YAML; CLI overrides ENV.
3. `--set a.b=3e-5` converts to the schema type; a bad value fails with an actionable message.
4. `extends` merges base and child; a cycle is detected.
5. `.env` parsing: comments, blank lines, quotes, malformed line.
6. Unknown key fails.
7. Resolved-config dump, reloaded, equals the original `AppConfig`.
8. Offline mode sets the three env variables; online mode does not.
9. No network or Hugging Face import occurs when importing the module (check `sys.modules`).
