# PRD Component 13: Orchestrator and CLI

Status legend: LOCKED = explicitly decided by the owner (binding). DEFAULT = proposed starting value
(not binding, configurable, may change by evidence from experiments). OPEN = undecided: the agent
presents 2-3 options with trade-offs and a recommendation, and does not choose.
Everything not tagged below is LOCKED.
NOTE: the owner will send more answers (manager questions) and the PRD will be revised then. Because this component is the face of the whole system, every command must start from a usage header with an exact example (CLAUDE.md section 9).

## Priority / order and dependencies
- Depends on all other components, through their interfaces only. Implemented LAST among the pipeline components (a minimal first version may come earlier for the first end-to-end run, see the build-order discussion).
- It is the only place where concrete classes are named (composition root, `wiring.py`).

## Responsibility
Wire the components together and run the whole pipeline, or any single stage of it, from one command with full automation and a clear exit code.

## Interface and implementations
- `PipelineStage` (typing.Protocol): `name: str`, `config_slice(config) -> Mapping` (the part of the config the stage depends on), `run(ctx: StageContext) -> StageResult`. Implementations, selected by name through a registry: `prepare`, `train`, `search`, `evaluate`, `select_and_register`, `export`, `report`, `learning_curve`, `smoke`, `benchmark`. A new stage is a new class.
- `Orchestrator.run(plan: RunPlan) -> RunOutcome`: runs the stages in the plan, writes completion markers, skips finished stages, stops at the first failure.
- `wiring.build_services(config) -> Services`: the composition root; the only module that names concrete classes, built from the names in Config through the registries.
- `SleepGuard` (typing.Protocol): `__enter__` / `__exit__`. Implementations: `WindowsSleepGuard` (standard library only, no admin rights) and `NoopSleepGuard`.
- CLI: `cli.py` using the standard library `argparse` (no new dependency). Every subcommand has `--help` with an example.

## Inputs -> Outputs (typed, dataclasses)
- `StageContext`: `config`, `experiment: ExperimentHandle`, `services: Services`, `stop_flag` (set on interrupt), `previous: Mapping[str, StageResult]`.
- `StageResult`: `status` (`ok` | `skipped` | `failed`), `summary: Mapping[str, str | float | int]`, `artifacts: list[Path]`, `error: str | None`.
- `RunPlan`: `stages: list[str]`, `experiment`.
- `RunOutcome`: `exit_code: int`, `verdict` (`winner` | `baseline_retained` | `n/a`), `experiment_id`, `stage_results`.
- Final output: one JSON summary printed at the end (the single permitted print): experiment id, verdict, key numbers (baseline and winner WER with interval, Test status), and the paths of the report and the registry entry.

## Commands (owner approved)
`run`, `prepare`, `train`, `search`, `evaluate`, `report`, `export`, `registry` (list, verify), `learning-curve`, `smoke`, `check`, `benchmark`, `cleanup`, `config-docs`.
Shared options on all commands: `--config <file>`, `--set a.b=value` (already decided in the Config component), `--parent exp_0003`.

## Rules
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

## Config keys it reads (sections `pipeline`, `run`, `smoke`, `learning_curve`, `benchmark`)
- `pipeline.stages` [DEFAULT: `prepare`, `search`, `select_and_register`, `export`, `report`]
- `pipeline.skip_completed` [DEFAULT: true]
- `run.prevent_sleep` [DEFAULT: true]
- `smoke.n_samples` [DEFAULT: 12], `smoke.duration_sec` [DEFAULT: 3], `smoke.steps` [DEFAULT: 3], `smoke.model_dir`
- `learning_curve.presets` [DEFAULT: `smoke`, `tiny`, `half`, `full`], `learning_curve.params_from` [DEFAULT: none]
- `benchmark.steps` [DEFAULT: 3]

## Out of scope (must NOT do)
- No component logic (everything is delegated to the components).
- No new metrics, no model code, no file formats of its own.
- No network, no interactive prompts, no automatic retries of failed stages.
- No deletion except through the explicit `cleanup` flag.

## Edge cases and failure behavior
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

## If it fails: minimal safe fixes
| Symptom | Likely cause | Minimal fix |
|---|---|---|
| exit code 2 | invalid or missing config key | read the message (it names the key); fix the YAML |
| exit code 3 | data path or column mapping wrong | read `rejected.csv` and the `prepare` report; fix `data.*` in YAML |
| exit code 4 | training or evaluation failed | read `logs/run.log`, `status.json` and the failing stage; consult the component tables in TROUBLESHOOTING |
| exit code 5 | export dependency missing or verification failed | see the ModelExporter table |
| rerun does nothing | all stages completed | use a new experiment (`--parent`), or change the config |
| rerun repeats a stage | its config slice changed | intended; check the diff in `config_diff.yaml` |
| the machine went to sleep | sleep prevention off or blocked by policy | check `run.prevent_sleep`; ask the machine owner about power policy |

## Acceptance tests (offline; fake services, then the smoke test with the tiny test model)
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

## OPEN items (agent presents options, owner decides)
- The build order and the first minimal end-to-end version (see the build-order discussion).
- The exact content of the `check` command after the home install test (O9).
- DECIDED by the owner: no diagnostics bundle (an internal agent exists at the work site and can help). The offline-install helper and the manifest maker are standalone tools (see PRD_tools).
