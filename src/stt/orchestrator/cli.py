"""The command line of the pipeline: `python -m stt <command>` (argparse only, no interactive prompt, ever).

How to run (PowerShell, project root; set the path to the sources once per window):
    $env:PYTHONPATH = "src"
    python -m stt smoke                       # the whole chain on synthetic data and the tiny test model (about 1 minute)
    python -m stt smoke --full                # the same data, the full `run` chain: search, finalists, selection (Test ONCE, synthetic), registry, `registry verify` (about 1 minute)
    python -m stt check                       # what is fine and what is not on this machine
    python -m stt prepare  --config my.yaml   # the data steps only, with the reports
    python -m stt train    --config my.yaml   # prepare + one training run
    python -m stt evaluate --config my.yaml   # prepare + baseline (and trained) model on the Val split
    python -m stt search   --config my.yaml   # prepare + the hyperparameter search (a few small trials on a screening subset, objective: Val loss)
    python -m stt run      --config my.yaml   # the whole pipeline: prepare, baseline on Val, search, finalists retrained on the full data, selection (Test ONCE), registry
    python -m stt registry list   --config my.yaml    # what is registered, with the number of exports of each entry (read only); also: registry verify [model_0001], registry rebuild-index
    python -m stt export model_0001 --config my.yaml  # export a registered model (default: the latest entry) in the formats of export.formats and record it in the entry's exports.json
    python -m stt learning-curve --config my.yaml     # the same settings on increasing Train sizes, Val WER per size, and a plain reading (still improving or flat)
    python -m stt benchmark --config my.yaml          # seconds per step on THIS machine and the planned time of a run
    python -m stt report --experiment exp_0004        # rebuild reports/report.html of an existing experiment from its stored files (nothing is recomputed)
Shared options: `--config FILE`, `--set a.b=value` (repeatable), `--parent exp_0003` (the new experiment extends that one), `--experiment exp_0004`
(continue an existing experiment: finished stages are skipped, an interrupted training is resumed). Every command creates a numbered experiment folder
`experiments/exp_NNNN/` and prints ONE JSON summary at the end (the only thing on stdout); the log lines go to the console (stderr) and to
`experiments/exp_NNNN/logs/run.log`. Exit codes: 0 success, 2 config error, 3 data error, 4 training or evaluation failure, 130 interrupted (Ctrl+C:
the training saves a checkpoint first; run the same command with `--experiment` to continue). Look at an experiment afterwards with
`python -m stt.runmanager list` and `python -m stt.runmanager show exp_0001`. `search` writes `reports/search_report.txt` (the trials in a table and the
finalists) and stores the study in `search/study.json`: after Ctrl+C or a crash run the same command with `--experiment exp_NNNN` and the finished trials are kept
(an interrupted trial is resumed). `search.parallel_trials: 2` runs two trials at once, each with half of the threads. `run` runs the stages of `pipeline.stages` (default:
prepare, evaluate, search, finalists, select_and_register); a rerun with `--experiment` skips the finished stages and continues; the Test split is looked at ONCE, for the winner and the baseline,
and the verdict (`winner` or `baseline_retained`) is printed in plain words. `registry` reads the registry folder `registry.dir` (it must lie OUTSIDE `run.experiments_dir`); exit code 5 when a
registered model does not match its checksums. `learning-curve` trains the same hyperparameters on each Subsampler preset of `learning_curve.presets` (add `--params-from exp_0003` to take the
winner's values of that experiment) and scores the SAME Val set; `benchmark` trains `benchmark.steps` steps, times a Val probe and prints the planned time of `run`; neither touches the Test split.
`report` reads the stored files of an existing experiment (the latest one without `--experiment`) and writes `reports/report.html`: one self-contained page, no network; name the experiments that hold the
learning curve and the benchmark with `--curve-from` and `--benchmark-from`; the `report.*` settings come from `--config` / `--set`. The live progress page of a trial is switched on with
`--set 'train.sinks=["csv","console","html"]'` and is `metrics/<trial>/progress.html`. `run` also exports the registered winner automatically (the formats of `export.formats`, default hf_safetensors: nothing is
copied, the folder is verified to load; skipped for a retained baseline); `export` repeats or adds an export for any existing registry entry (it only writes `exports.json` and `exports/` of that entry; exit code 2 for
an unknown entry or format, 5 for a failed export or verification). Nothing is deleted by any command. Not in this version: cleanup, config-docs (later waves).
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
from stt.orchestrator.reportcmd import build_report, latest_experiment
from stt.orchestrator.selection import check_config_registry_location
from stt.orchestrator.orchestrator import Orchestrator
from stt.orchestrator.sleepguard import default_sleep_guard
from stt.orchestrator.smoke import FULL_MIN_SAMPLES, full_overrides, write_smoke_data
from stt.orchestrator.stopflag import InterruptFlag
from stt.orchestrator.types import RunOutcome, RunPlan, StageContext
from stt.orchestrator.wiring import ExportWiring, RegistryWiring, ReportWiring, Runtime, experiment_catalog, build_runtime, exit_code_for, library_versions, section_registry
from stt.modelexporter import ExportConfig, ExportConfigError, ExportError
from stt.registry import RegistryConfig
from stt.reportbuilder import ReportConfig
from stt.runmanager import ExperimentHandle, ExperimentSpec, RunConfig

logger = logging.getLogger("stt.orchestrator")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLANS = {"prepare": ["prepare"], "train": ["prepare", "train"], "evaluate": ["prepare", "evaluate"], "search": ["prepare", "search"], "learning-curve": ["prepare", "learning_curve"], "benchmark": ["prepare", "benchmark"], "smoke": ["prepare", "train", "evaluate"]}  # `run` uses pipeline.stages
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
        "run": "python -m stt run --config my.yaml --set subsample.val_samples=100",
        "learning-curve": "python -m stt learning-curve --config my.yaml --params-from exp_0003",
        "report": "python -m stt report --experiment exp_0004 --curve-from exp_0005",
        "benchmark": "python -m stt benchmark --config my.yaml --set benchmark.steps=5",
        "smoke": "python -m stt smoke --full",
        "check": "python -m stt check --config configs/smoke.yaml",
    }
    helps = {
        "prepare": "load, split, normalize and filter the data and write the reports",
        "train": "prepare, then one training run",
        "evaluate": "prepare, then score the baseline (and the trained model of the experiment) on the Val split",
        "search": "prepare, then the hyperparameter search on a screening subset (objective: the Val loss); a rerun with --experiment continues it",
        "run": "the whole pipeline: prepare, baseline on Val, search, finalists on the full data, selection (the Test is looked at once) and registration; a rerun with --experiment continues it",
        "learning-curve": "train the same settings on increasing Train sizes (learning_curve.presets), score the same Val set, print the table and a plain reading",
        "benchmark": "measure seconds per training step and per Val evaluation on this machine and print the planned time of a run",
        "smoke": "run the whole chain on synthetic data and the tiny test model, to prove that the mechanics work",
        "check": "say what is fine and what is not on this machine",
    }
    for name, text in helps.items():
        sub = commands.add_parser(name, help=text, description=f"{text}.\nExample:  {examples[name]}", formatter_class=argparse.RawDescriptionHelpFormatter)
        add_config_arguments(sub)
        if name != "check":
            sub.add_argument("--parent", default=None, help="the new experiment extends this one, for example exp_0003")
            sub.add_argument("--experiment", default=None, help="continue this existing experiment instead of creating a new one, for example exp_0004")
        if name == "smoke":
            sub.add_argument("--full", action="store_true", help="also run the full `run` chain (search, finalists, selection with the one-time Test look on the SYNTHETIC Test clips, registration) and `registry verify`")
        if name == "learning-curve":
            sub.add_argument("--params-from", default=None, help="take the hyperparameters from this experiment (its winner, or its best finalist), for example exp_0003; sets learning_curve.params_from")
        if name == "evaluate":
            sub.add_argument("--split", default=None, help="the split to score (default: pipeline.eval_split = val; the Test split is refused)")
    report = commands.add_parser(
        "report",
        help="rebuild reports/report.html of an existing experiment from its stored files (nothing is recomputed)",
        description="Rebuild the report page of an existing experiment from its stored files.\nExample:  python -m stt report --experiment exp_0004 --curve-from exp_0005 --benchmark-from exp_0006",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arguments(report)
    report.add_argument("--experiment", default=None, help="the experiment to report on, for example exp_0004 (default: the latest one)")
    report.add_argument("--curve-from", default=None, help="the experiment that holds the learning curve (reports/learning_curve.json), for example exp_0005")
    report.add_argument("--benchmark-from", default=None, help="the experiment that holds the benchmark (reports/benchmark.json), for example exp_0006")
    export = commands.add_parser(
        "export",
        help="export a registered model (default: the latest entry) in the formats of export.formats and record it in the entry",
        description="Export a registered model and verify the export. It writes only `exports.json` and `exports/` of that entry.\nExample:  python -m stt export model_0001 --format hf_safetensors --config my.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_arguments(export)
    export.add_argument("entry", nargs="?", default=None, help="the registry entry, for example model_0001 (default: the latest entry)")
    export.add_argument("--format", action="append", default=None, help="an exporter name (repeatable); default: export.formats")
    registry = commands.add_parser(
        "registry",
        help="look at the model registry (read only): list, verify, rebuild-index",
        description="Look at the model registry folder `registry.dir`.\nExamples:  python -m stt registry list --config my.yaml   |   python -m stt registry verify model_0001 --config my.yaml   |   python -m stt registry rebuild-index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = registry.add_subparsers(dest="registry_command", required=True, metavar="action")
    for action, text in (
        ("list", "one line per registered entry"),
        ("verify", "check the files of an entry (or of all entries) against their SHA256SUMS; exit code 5 on a mismatch"),
        ("rebuild-index", "make index.csv again from the entry cards (the only action that writes)"),
    ):
        sub = actions.add_parser(action, help=text, description=f"{text}.\nExample:  python -m stt registry {action} --config my.yaml", formatter_class=argparse.RawDescriptionHelpFormatter)
        add_config_arguments(sub)
        if action == "verify":
            sub.add_argument("entry", nargs="?", default=None, help="the entry to check, for example model_0001 (default: all entries)")
    return parser


def _final_summary(handle: ExperimentHandle, outcome: RunOutcome, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    where = handle.locations()
    evaluate = outcome.stage_results.get("learning_curve") or outcome.stage_results.get("benchmark") or outcome.stage_results.get("select_and_register") or outcome.stage_results.get("evaluate") or outcome.stage_results.get("search")  # the numbers a human looks at first
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


def _say_verdict(outcome: RunOutcome) -> None:
    """The verdict of the selection in plain words, on the console (the log file has it too)."""
    selected = outcome.stage_results.get("select_and_register")
    if selected is None or selected.status == "failed" or outcome.verdict == "n/a":
        return
    s = selected.summary
    logger.info("VERDICT: %s. %s", outcome.verdict, s.get("reason", ""))
    logger.info("Test: %s. Registered as %s in %s.", s.get("test_status", "?"), s.get("entry_id", "?"), s.get("registry_dir", "?"))


def _error(message: str, code: int) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def _run_pipeline(args: argparse.Namespace, config_overrides: Sequence[str], smoke: bool = False, full: bool = False) -> int:
    try:
        config = load_config(args.config, overrides=config_overrides, registry=section_registry())
    except ConfigError as exc:
        return _error(str(exc), 2)
    runtime = build_runtime(config, _now)
    run = config.section("run", RunConfig)
    pipeline = config.section("pipeline", PipelineConfig)
    stages = list(pipeline.stages) if args.command == "run" or full else PLANS[args.command]  # `smoke --full` runs the stages of `run`
    if "select_and_register" in stages:  # before an experiment folder is made and before any hour of work: the registry must be outside the experiments
        try:
            check_config_registry_location(config)
        except OrchestratorError as exc:
            return _error(str(exc), exc.exit_code)
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
            outcome = orchestrator.run(RunPlan(stages, handle, options))
        extra = _test_once_check(config, runtime, handle, flag) if smoke and outcome.exit_code == 0 else None
        if extra is not None and extra.get("test_once_rule") != "refused":
            outcome = RunOutcome(4, outcome.verdict, outcome.experiment_id, outcome.stage_results, outcome.status, "the Test-once rule was not enforced")
        if full and extra is not None and outcome.exit_code == 0:
            checks, problem = _full_checks(config, handle, outcome)
            extra.update(checks)
            if problem:
                outcome = RunOutcome(4, outcome.verdict, outcome.experiment_id, outcome.stage_results, outcome.status, problem)
        _say_verdict(outcome)
        _how_to_look(handle, run.experiments_dir)
        print(json.dumps(_final_summary(handle, outcome, extra), indent=1, ensure_ascii=False))
        return outcome.exit_code
    except Exception as exc:  # before or between stages: the message names the cause; the exit code comes from the mapping
        return _error(f"{type(exc).__name__}: {exc}", exit_code_for(exc))
    finally:
        runtime.log_setup.close()


def _full_checks(config: Any, handle: ExperimentHandle, outcome: RunOutcome) -> tuple[dict[str, Any], str | None]:
    """The extra checks of `smoke --full`: the Test was looked at ONCE per model (and not at all when the baseline stayed), and the registered entry verifies against its checksums.

    Returns (what to print, a problem text or None)."""
    selected = outcome.stage_results["select_and_register"].summary
    entry_id = str(selected["entry_id"])
    reports = handle.paths.reports
    looked: list[str] = []
    if (reports / "test_once.json").is_file():
        looked = list(json.loads((reports / "test_once.json").read_text(encoding="utf-8"))["looked"])
    expected = ["baseline", f"trial_{selected['winner_trial']}"] if outcome.verdict == "winner" else []
    verified = RegistryWiring(config).verify(entry_id)
    checks: dict[str, Any] = {
        "full_chain": {"stages": {name: r.status for name, r in outcome.stage_results.items()}, "verdict": outcome.verdict, "test_status": selected["test_status"], "entry": entry_id, "export": selected.get("export")},
        "test_looked_at": sorted(looked),
        "registry_verify": f"{entry_id}: {verified.files_checked} files match their checksums",
    }
    if sorted(looked) != sorted(expected) or len(looked) != len(set(looked)):
        return checks, f"the Test look is wrong: looked at {sorted(looked)}, expected {sorted(expected)}, each once"
    return checks, None


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
    full = bool(args.full)
    write_smoke_data(settings.model_copy(update={"n_samples": max(settings.n_samples, FULL_MIN_SAMPLES)}) if full else settings, seed, settings.work_dir)
    derived = [f"data.source.path={settings.work_dir.as_posix()}", f"model.path={settings.model_dir.as_posix()}", f"train.hyperparams.max_steps={settings.steps}"]
    if full:
        derived += full_overrides(settings)
    args.config = path
    return _run_pipeline(args, [*derived, *args.config_overrides], smoke=True, full=full)


def _report(args: argparse.Namespace) -> int:
    """`report`: the page of an EXISTING experiment from its stored files. No experiment is made, no status or marker changes, nothing is recomputed or deleted."""
    try:
        tree = apply_overrides(load_yaml_with_extends(args.config) if args.config is not None else {}, [parse_set(item) for item in args.config_overrides])
        run = RunConfig.model_validate(tree.get("run", {}))
        settings = ReportConfig.model_validate(tree.get("report", {}))  # only the sections the report needs: no data or model settings
        store = experiment_catalog(run)
        handle = store.open(args.experiment or latest_experiment(store))
        result = build_report(handle, run.experiments_dir, settings, ReportWiring(), _now(), lambda folder: RegistryWiring(RegistryConfig(dir=Path(folder))), args.curve_from, args.benchmark_from)
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}", exit_code_for(exc))
    print(f"The report is in: {result.report_file}", file=sys.stderr)
    print("Open it in a browser (it needs no network).", file=sys.stderr)
    if result.sections_without_data:
        print("Sections WITHOUT data in this experiment: " + "; ".join(title for _, title in result.sections_without_data) + ".", file=sys.stderr)
    for problem in result.problems:
        print(f"WARNING: an input could not be read and was left out: {problem}", file=sys.stderr)
    print(
        json.dumps(
            {
                "experiment_id": result.experiment_id,
                "report_file": str(result.report_file),
                "sections_with_data": result.sections_with_data,
                "sections_without_data": [title for _, title in result.sections_without_data],
                "inputs": result.found,
                "input_problems": result.problems,
                "warnings": result.warnings,
                "curve_from": result.curve_from,
                "benchmark_from": result.benchmark_from,
                "exit_code": 0,
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0


def _export(args: argparse.Namespace) -> int:
    """`export`: export a registered model and record it. Reads the registry; writes only `exports.json` and `exports/` of the chosen entry; deletes nothing."""
    try:
        tree = apply_overrides(load_yaml_with_extends(args.config) if args.config is not None else {}, [parse_set(item) for item in args.config_overrides])
        registry = RegistryWiring(RegistryConfig.model_validate(tree.get("registry", {})))  # only the sections the command needs (no data or model settings)
        settings = ExportConfig.model_validate(tree.get("export", {}))
        known = [e.entry_id for e in registry.entries()]
        if not known:
            raise OrchestratorError(f"The registry {registry.location()} has no entries to export. Run `python -m stt run` first, or point registry.dir at your registry (--set registry.dir=...).", exit_code=2)
        entry_id = args.entry or known[-1]
        if entry_id not in known:
            raise OrchestratorError(f"The registry entry '{entry_id}' does not exist in {registry.location()}. Existing entries: {', '.join(known)}.", exit_code=2)
        service = ExportWiring(settings, registry)
        before = len(service.records(entry_id))
        error: ExportError | None = None
        try:
            service.export(entry_id, args.format, explicit=True, exported_at=_now())
        except ExportConfigError:
            raise  # a wrong setting: exit code 2, nothing was recorded
        except ExportError as exc:  # a failed export or verification: the records were written, say what they hold, exit code 5
            error = exc
        recorded = service.records(entry_id)[before:]
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}", exit_code_for(exc))
    entry_dir = registry.location() / entry_id
    print(f"Entry {entry_id}: {len(recorded)} export(s) recorded in {entry_dir / 'exports.json'}.", file=sys.stderr)
    for r in recorded:
        print(f"  {r.exporter}: {r.status}" + (f" ({r.verify.level} level) {r.verify.detail}" if r.verify is not None else " (not verified: export.verify is false)"), file=sys.stderr)
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
    body = {
        "entry_id": entry_id,
        "exports_file": str(entry_dir / "exports.json"),
        "recorded": [{"exporter": r.exporter, "status": r.status, "target_dir": r.target_dir, "files": len(r.files), "verify": None if r.verify is None else {"level": r.verify.level, "ok": r.verify.ok, "detail": r.verify.detail}, "exported_at": r.exported_at} for r in recorded],
        "error": None if error is None else str(error),
        "exit_code": 0 if error is None else exit_code_for(error),
    }
    print(json.dumps(body, indent=1, ensure_ascii=False))
    return int(body["exit_code"])  # type: ignore[call-overload]


def _registry(args: argparse.Namespace) -> int:
    """`registry list|verify|rebuild-index`: the registry folder of the config, no experiment is made. Read only except `rebuild-index`."""
    try:
        tree = apply_overrides(load_yaml_with_extends(args.config) if args.config is not None else {}, [parse_set(item) for item in args.config_overrides])
        service = RegistryWiring(RegistryConfig.model_validate(tree.get("registry", {})))  # only the registry section is needed (no data or model settings)
        if args.registry_command == "list":
            entries = service.entries()
            print(f"Registry {service.location()}: {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}.")
            for e in entries:
                card = service.card(e.entry_id)
                val = "-" if card.val is None else f"{card.val.wer:.4f}"
                test = "-" if card.test is None else f"{card.test.wer:.4f}"
                try:
                    exports = service.exports(e.entry_id)
                    shown = f"exports {len(exports)}" + (f" ({sum(1 for r in exports if r.status == 'failed')} FAILED)" if any(r.status == "failed" for r in exports) else "")
                except ExportError:
                    shown = "exports unreadable"
                print(f"  {e.entry_id}  from {e.experiment_id}  {card.kind:<17} verdict {card.verdict:<17} Val WER {val}  Test WER {test}  Test status {card.test_status}  {shown}  {card.generated_at}")
            return 0
        if args.registry_command == "rebuild-index":
            service.rebuild_index()
            print(f"The index of {service.location()} was made again from the entry cards.")
            return 0
        names = [args.entry] if args.entry else [e.entry_id for e in service.entries()]
        code = 0
        for name in names:
            try:
                report = service.verify(name)
                print(f"OK      {name}: {report.files_checked} files match their checksums.")
            except Exception as exc:  # one damaged entry does not hide the others
                print(f"FAILED  {name}: {exc}")
                code = max(code, exit_code_for(exc))
        if not names:
            print(f"The registry {service.location()} has no entries to verify.")
        return code
    except Exception as exc:
        return _error(f"{type(exc).__name__}: {exc}", exit_code_for(exc))


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
    if args.command == "registry":
        return _registry(args)
    if args.command == "report":
        return _report(args)
    if args.command == "export":
        return _export(args)
    overrides = list(args.config_overrides)
    if args.command == "learning-curve" and args.params_from:
        overrides.append(f"learning_curve.params_from={args.params_from}")  # a config value, so it is stored with the experiment and part of the stage's fingerprint
    return _run_pipeline(args, overrides)


if __name__ == "__main__":
    sys.exit(main())
