"""The command line of the pipeline: `python -m stt <command>` (argparse only, no interactive prompt, ever).

How to run (PowerShell, project root; set the path to the sources once per window):
    $env:PYTHONPATH = "src"
    python -m stt smoke                       # the whole chain on synthetic data and the tiny test model (about 1 minute)
    python -m stt check                       # what is fine and what is not on this machine
    python -m stt prepare  --config my.yaml   # the data steps only, with the reports
    python -m stt train    --config my.yaml   # prepare + one training run
    python -m stt evaluate --config my.yaml   # prepare + baseline (and trained) model on the Val split
    python -m stt search   --config my.yaml   # prepare + the hyperparameter search (a few small trials on a screening subset, objective: Val loss)
Shared options: `--config FILE`, `--set a.b=value` (repeatable), `--parent exp_0003` (the new experiment extends that one), `--experiment exp_0004`
(continue an existing experiment: finished stages are skipped, an interrupted training is resumed). Every command creates a numbered experiment folder
`experiments/exp_NNNN/` and prints ONE JSON summary at the end (the only thing on stdout); the log lines go to the console (stderr) and to
`experiments/exp_NNNN/logs/run.log`. Exit codes: 0 success, 2 config error, 3 data error, 4 training or evaluation failure, 130 interrupted (Ctrl+C:
the training saves a checkpoint first; run the same command with `--experiment` to continue). Look at an experiment afterwards with
`python -m stt.runmanager list` and `python -m stt.runmanager show exp_0001`. `search` writes `reports/search_report.txt` (the trials in a table and the
finalists) and stores the study in `search/study.json`: after Ctrl+C or a crash run the same command with `--experiment exp_NNNN` and the finished trials are kept
(an interrupted trial is resumed). `search.parallel_trials: 2` runs two trials at once, each with half of the threads. Not in this version: run, report, export,
registry, learning-curve, benchmark, cleanup, config-docs (later parts and waves).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from stt.config import ConfigError, add_config_arguments, load_config
from stt.config.sources import apply_overrides, load_yaml_with_extends, parse_set
from stt.orchestrator import check as check_module
from stt.orchestrator.codeversion import code_version
from stt.orchestrator.config import PipelineConfig, SmokeConfig
from stt.orchestrator.errors import OrchestratorError
from stt.orchestrator.orchestrator import Orchestrator
from stt.orchestrator.sleepguard import default_sleep_guard
from stt.orchestrator.smoke import write_smoke_data
from stt.orchestrator.stopflag import InterruptFlag
from stt.orchestrator.types import RunOutcome, RunPlan, StageContext
from stt.orchestrator.wiring import Runtime, build_runtime, exit_code_for, library_versions, section_registry
from stt.runmanager import ExperimentHandle, ExperimentSpec, RunConfig

logger = logging.getLogger("stt.orchestrator")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLANS = {"prepare": ["prepare"], "train": ["prepare", "train"], "evaluate": ["prepare", "evaluate"], "search": ["prepare", "search"], "smoke": ["prepare", "train", "evaluate"]}
DEFAULT_SMOKE_CONFIG = Path("configs/smoke.yaml")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser (every command has an example in its --help)."""
    parser = argparse.ArgumentParser(prog="python -m stt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    examples = {
        "prepare": "python -m stt prepare --config my.yaml",
        "train": "python -m stt train --config my.yaml --set train.hyperparams.max_steps=50",
        "evaluate": "python -m stt evaluate --config my.yaml --experiment exp_0003",
        "search": "python -m stt search --config my.yaml --set search.n_trials=4",
        "smoke": "python -m stt smoke",
        "check": "python -m stt check --config configs/smoke.yaml",
    }
    helps = {
        "prepare": "load, split, normalize and filter the data and write the reports",
        "train": "prepare, then one training run",
        "evaluate": "prepare, then score the baseline (and the trained model of the experiment) on the Val split",
        "search": "prepare, then the hyperparameter search on a screening subset (objective: the Val loss); a rerun with --experiment continues it",
        "smoke": "run the whole chain on synthetic data and the tiny test model, to prove that the mechanics work",
        "check": "say what is fine and what is not on this machine",
    }
    for name, text in helps.items():
        sub = commands.add_parser(name, help=text, description=f"{text}.\nExample:  {examples[name]}", formatter_class=argparse.RawDescriptionHelpFormatter)
        add_config_arguments(sub)
        if name != "check":
            sub.add_argument("--parent", default=None, help="the new experiment extends this one, for example exp_0003")
            sub.add_argument("--experiment", default=None, help="continue this existing experiment instead of creating a new one, for example exp_0004")
        if name == "evaluate":
            sub.add_argument("--split", default=None, help="the split to score (default: pipeline.eval_split = val; the Test split is refused)")
    return parser


def _final_summary(handle: ExperimentHandle, outcome: RunOutcome, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    where = handle.locations()
    evaluate = outcome.stage_results.get("evaluate") or outcome.stage_results.get("search")  # the numbers a human looks at first
    return {
        "experiment_id": outcome.experiment_id,
        "experiment_dir": str(handle.paths.root),
        "status": outcome.status,
        "exit_code": outcome.exit_code,
        "verdict": outcome.verdict,
        "message": outcome.message,
        "stages": {name: {"status": r.status, "summary": dict(r.summary), "error": r.error} for name, r in outcome.stage_results.items()},
        "key_numbers": dict(evaluate.summary) if evaluate is not None else {},
        "paths": {
            "config": str(where.config_full),
            "status": str(where.status),
            "log": str(where.log),
            "metrics": [str(p) for p in where.metrics_files],
            "reports": str(handle.paths.reports),
        },
        **(extra or {}),
    }


def _how_to_look(handle: ExperimentHandle, runtime_dir: Path) -> None:
    exp, folder = handle.experiment_id, runtime_dir
    logger.info("The experiment is in: %s", handle.paths.root)
    logger.info("To look at it, run in PowerShell from the project folder (after `$env:PYTHONPATH = \"src\"`):")
    logger.info("  python -m stt.runmanager list --dir %s", folder)
    logger.info("  python -m stt.runmanager show %s --dir %s", exp, folder)
    logger.info("  python -m stt check")


def _error(message: str, code: int) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def _run_pipeline(args: argparse.Namespace, config_overrides: Sequence[str], smoke: bool = False) -> int:
    try:
        config = load_config(args.config, overrides=config_overrides, registry=section_registry())
    except ConfigError as exc:
        return _error(str(exc), 2)
    runtime = build_runtime(config, _now)
    run = config.section("run", RunConfig)
    pipeline = config.section("pipeline", PipelineConfig)
    handle = None
    try:
        if args.experiment:
            handle = runtime.store.open(args.experiment, config)
        else:
            handle = runtime.store.create(ExperimentSpec(config, args.parent or run.parent, code_version(PROJECT_ROOT), library_versions(), runtime.now()))
        runtime.log_setup.configure(handle)
        flag = InterruptFlag()
        orchestrator = Orchestrator(config, runtime.services, runtime.stages, flag, default_sleep_guard(run.prevent_sleep), pipeline.skip_completed)
        options = {"split": args.split} if getattr(args, "split", None) else {}
        with flag.installed():
            outcome = orchestrator.run(RunPlan(PLANS[args.command], handle, options))
        extra = _test_once_check(config, runtime, handle, flag) if smoke and outcome.exit_code == 0 else None
        if extra is not None and extra.get("test_once_rule") != "refused":
            outcome = RunOutcome(4, outcome.verdict, outcome.experiment_id, outcome.stage_results, outcome.status, "the Test-once rule was not enforced")
        _how_to_look(handle, run.experiments_dir)
        print(json.dumps(_final_summary(handle, outcome, extra), indent=1, ensure_ascii=False))
        return outcome.exit_code
    except Exception as exc:  # before or between stages: the message names the cause; the exit code comes from the mapping
        return _error(f"{type(exc).__name__}: {exc}", exit_code_for(exc))
    finally:
        runtime.log_setup.close()


def _test_once_check(config: Any, runtime: Runtime, handle: ExperimentHandle, flag: InterruptFlag) -> dict[str, Any]:
    """The smoke check of the Test-once rule: the evaluate stage must REFUSE the Test split and say so."""
    stage = runtime.stages.create("evaluate")
    ctx = StageContext(config, handle, runtime.services, flag, {}, {"split": config.section("pipeline", PipelineConfig).test_split})
    try:
        stage.run(ctx)
    except OrchestratorError as exc:
        if "refuses the Test split" in str(exc):
            logger.info("Test-once rule checked: %s", exc)
            return {"test_once_rule": "refused", "test_once_message": str(exc)}
        return {"test_once_rule": f"refused with an unexpected message: {exc}"}
    return {"test_once_rule": "NOT enforced: the evaluate stage accepted the Test split"}


def _smoke(args: argparse.Namespace) -> int:
    path = args.config or DEFAULT_SMOKE_CONFIG
    if not path.is_file():
        return _error(f"The smoke configuration {path} does not exist. Run the command from the project folder, or pass --config.", 2)
    try:
        tree = apply_overrides(load_yaml_with_extends(path), [parse_set(item) for item in args.config_overrides])
        settings = SmokeConfig.model_validate(tree.get("smoke", {}))
        seed = int(tree.get("runtime", {}).get("seed", 42))
    except (ConfigError, ValueError) as exc:
        return _error(f"The smoke settings are invalid: {exc}", 2)
    if not (settings.model_dir / "config.json").is_file():
        return _error(f"The tiny test model folder {settings.model_dir} does not exist. Create it once at home with: python tools/make_test_model.py --tokenizer-from models/whisper-tiny --out {settings.model_dir}", 2)
    write_smoke_data(settings, seed, settings.work_dir)
    derived = [f"data.source.path={settings.work_dir.as_posix()}", f"model.path={settings.model_dir.as_posix()}", f"train.hyperparams.max_steps={settings.steps}"]
    args.config = path
    return _run_pipeline(args, [*derived, *args.config_overrides], smoke=True)


def _check(args: argparse.Namespace) -> int:
    config = None
    if args.config is not None or args.config_overrides:
        try:
            config = load_config(args.config, overrides=args.config_overrides, registry=section_registry(), apply_environment=False)
        except ConfigError as exc:
            return _error(str(exc), 2)
    disk = config.section("run", RunConfig).experiments_dir if config is not None else Path("experiments")
    text, code = check_module.format_report(check_module.run_checks(config, PROJECT_ROOT, disk.resolve() if disk.exists() else PROJECT_ROOT))
    print(text)
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return the exit code."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Hebrew in the summary on a Windows console
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return _check(args)
    if args.command == "smoke":
        return _smoke(args)
    return _run_pipeline(args, args.config_overrides)


if __name__ == "__main__":
    sys.exit(main())
