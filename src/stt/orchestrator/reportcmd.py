"""The `report` command: rebuild the report of an EXISTING experiment from its stored files (C13 brief, rule 10; the ReportBuilder is C14).

How to run (PowerShell, project root, after `$env:PYTHONPATH = "src"`):
    python -m stt report --experiment exp_0004 --set run.experiments_dir=experiments
    python -m stt report                                   # the latest experiment
    python -m stt report --experiment exp_0004 --curve-from exp_0005 --benchmark-from exp_0006
Example output (one JSON summary on stdout; the log lines go to stderr):
    {"experiment_id": "exp_0004", "report_file": "experiments/exp_0004/reports/report.html", "sections_with_data": ["summary", "finalists", ...], "sections_without_data": [], ...}
What it does: reads the stored files of the experiment (`reportdata.py`), builds the page with the ReportBuilder and writes `reports/report.html` through the RunManager. It never calls the trainer or the
evaluator, recomputes nothing, creates no experiment, changes no status or completion marker and deletes nothing (an older `report.html` is replaced by the new one; the rest of the experiment is only
read). It needs the experiment's lock for the write, so it refuses while a run is using the experiment. The report settings (`report.*`) and `run.experiments_dir` come from `--config` and `--set` (change
a threshold and rebuild the page without recomputing anything). The learning curve and the benchmark belong to their own experiments: name them with `--curve-from` and `--benchmark-from` (the
experiment's own files are used when it has them and the options are not given); the page and the summary say which inputs were found. Exit codes: 2 an unknown experiment or option value, 4 a report
that cannot be built.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from stt.orchestrator.errors import OrchestratorError
from stt.orchestrator.reportdata import RegistryReader, load_report_data
from stt.reportbuilder import ReportConfig, ReportData, ReportOutput
from stt.runmanager import ExperimentHandle, ExperimentSummary

logger = logging.getLogger("stt.orchestrator")
REPORT_FILE = "report.html"
REPORT_SECTIONS = (
    ("summary", "Summary"),
    ("history", "History: the Val WER of each registered experiment"),
    ("finalists", "Finalists compared with the baseline"),
    ("wer_bars", "WER of the baseline and the finalists"),
    ("training_curves", "Training curves"),
    ("distribution", "Distribution of the per-file WER"),
    ("groups", "WER by group (lesson)"),
    ("search", "The search: each parameter against the Val loss"),
    ("learning_curve", "Learning curve: Train size against Val WER"),
    ("correlations", "Per-file errors and file attributes"),
    ("hints", "Diagnosis hints"),
    ("listening", "Listening list"),
    ("run_info", "Run information"),
)


class ExperimentCatalog(Protocol):
    """The experiments folder as the command needs it (the real `ExperimentStore` fits; `wiring.experiment_catalog` makes it)."""

    def open(self, experiment_id: str) -> ExperimentHandle:
        """A handle on an existing experiment (no config check). Raises RunManagerError for an unknown one, listing the existing ones."""
        ...

    def list_experiments(self) -> list[ExperimentSummary]:
        """All experiments, lowest number first."""
        ...


class ReportService(Protocol):
    """The ReportBuilder bound to nothing but the data (the real `ReportBuilder` fits)."""

    def build(self, data: ReportData) -> ReportOutput:
        """The report page of `data`."""
        ...


@dataclass(frozen=True)
class ReportResult:
    """What the command did: where the file is, which sections had data and which had none, which inputs were found, the files that could not be read and the builder's warnings."""

    experiment_id: str
    report_file: Path
    sections_with_data: list[str]
    sections_without_data: list[tuple[str, str]]
    found: dict[str, str]
    problems: list[str]
    warnings: list[str]
    curve_from: str | None
    benchmark_from: str | None


def latest_experiment(store: ExperimentCatalog) -> str:
    """The id of the newest experiment. Errors: OrchestratorError (exit code 2) when there is none."""
    experiments = store.list_experiments()
    if not experiments:
        raise OrchestratorError("There is no experiment to report on. Run `python -m stt run` first, or point run.experiments_dir at the folder that holds them (--set run.experiments_dir=...).", exit_code=2)
    return experiments[-1].experiment_id


def other_experiment(experiments_dir: Path, name: str | None, option: str) -> Path | None:
    """The folder of the experiment named by `--curve-from` / `--benchmark-from`. Errors: OrchestratorError (exit code 2) listing the existing experiments."""
    if name is None:
        return None
    root = experiments_dir / name
    if not root.is_dir():
        known = sorted(p.name for p in experiments_dir.glob("exp_*") if p.is_dir()) if experiments_dir.is_dir() else []
        raise OrchestratorError(f"{option} names '{name}', but {root} does not exist. Existing experiments: {', '.join(known) or 'none'}.", exit_code=2)
    return root


def build_report(
    handle: ExperimentHandle,
    experiments_dir: Path,
    settings: ReportConfig,
    service: ReportService,
    generated_at: str,
    registry_for: Callable[[str], RegistryReader] | None,
    curve_from: str | None = None,
    benchmark_from: str | None = None,
    registry_dir: str | None = None,
) -> ReportResult:
    """Load, build and write the report of `handle`. Errors: OrchestratorError (exit code 2) for an unknown `curve_from` / `benchmark_from`; RunManagerError when the experiment is locked;
    ReportConfigError and ReportError from the builder; the report page is written through the RunManager (`reports/report.html`)."""
    curve_root = other_experiment(experiments_dir, curve_from, "--curve-from")
    benchmark_root = other_experiment(experiments_dir, benchmark_from, "--benchmark-from")
    loaded = load_report_data(handle.paths.root, settings, generated_at, registry_for, curve_root, benchmark_root, registry_dir)
    output = service.build(loaded.data)
    with handle.lock():
        path = handle.write_report(REPORT_FILE, output.html)
    produced = set(output.sections)
    without = [(key, title) for key, title in REPORT_SECTIONS if key not in produced]
    logger.info("The report is in: %s", path)
    logger.info("Open it in a browser (it needs no network). Sections with data: %s.", ", ".join(output.sections))
    if without:
        logger.info("Sections WITHOUT data in this experiment: %s.", "; ".join(title for _, title in without))
    for problem in loaded.problems:
        logger.warning("An input could not be read and was left out: %s", problem)
    return ReportResult(handle.experiment_id, path, list(output.sections), without, dict(loaded.found), list(loaded.problems), list(output.warnings), curve_from, benchmark_from)
