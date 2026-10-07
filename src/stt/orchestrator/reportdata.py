"""The loader of the report: the stored files of an experiment become the views of the ReportBuilder (C14). Reads files only; recomputes nothing; calls no trainer and no evaluator.

How to run: this module is not a script; `python -m stt report` uses it. Example (PowerShell, project root; reads one finished experiment, nothing is written):
    $env:PYTHONPATH = "src"
    python -c "from pathlib import Path; from stt.orchestrator.reportdata import load_report_data; from stt.reportbuilder import ReportConfig; r = load_report_data(Path('experiments/exp_0001'), ReportConfig(), '2026-10-07T12:00:00'); print(sorted(r.found))"
Example output (the names of the inputs; a missing one says `missing`):
    ['baseline', 'benchmark', 'config', 'final_histories', 'finalists', 'learning_curve', 'prepare_summary', 'registry_card', 'run_info', 'search', 'search_histories', 'selection', 'test_results']
A missing file is NOT an error: its view is None (or an empty mapping) and the report says `Not available in this run` and names the file. A file that exists but cannot be read (damaged JSON, a result
of another codec version, a row with unknown keys) is also left out of the report and is listed in `problems` (the command prints them and logs a warning), so nothing is silently skipped.
Where the files are: `reports/selection.json`, `search_trials.json`, `eval_result_baseline.json`, `finalists/final_trial_TTT.val.json`, `test_result_<baseline|trial_T>.json`, `learning_curve.json`,
`benchmark.json`, `stage_results/prepare.json` (the fingerprints), `metrics/final_trial_TTT/metrics.csv`, `metrics/trial_TTT/metrics.csv`, `run_info.json`, and the registry card
(`<registry_dir>/<entry>/model_card.json`, where `registry_dir` and the entry are written in `selection.json`; the caller passes `registry_for`). `learning_curve.json` and `benchmark.json` belong to their own experiments: an explicit
`curve_root` / `benchmark_root` (the options `--curve-from` and `--benchmark-from`) wins over the experiment's own file.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from stt.evaluator import ComparisonResult, EvalResult, EvaluatorError, eval_result_from_json
from stt.registry import ModelCard, RegistryEntry, test_set_fingerprint
from stt.reportbuilder import (
    BenchmarkView,
    CurvePointView,
    HistoryPoint,
    LearningCurveView,
    ReportConfig,
    ReportData,
    RunInfoView,
    SearchView,
    SelectionView,
    TrialView,
)
from stt.selector import ComparisonRow
from stt.trainercore import StepMetrics

logger = logging.getLogger("stt.orchestrator")


class RegistryReader(Protocol):
    """What the loader needs from the registry (the real `RegistryWiring` fits): the card of an entry and how many registered entries were evaluated on a Test set."""

    def card(self, entry_id: str) -> ModelCard:
        """The card of an entry. Raises RegistryError."""
        ...

    def test_set_count(self, test_fingerprint: str) -> int:
        """How many registered entries carry this Test-set fingerprint."""
        ...

    def entries(self) -> Sequence[RegistryEntry]:
        """All registered entries in registry order (the history chart reads the card of each)."""
        ...


@dataclass(frozen=True)
class LoadedReport:
    """The views and what happened while reading: `found` maps each input to where it came from (or `missing`), `problems` lists the files that exist but could not be read."""

    data: ReportData
    found: Mapping[str, str]
    problems: tuple[str, ...] = field(default_factory=tuple)


def read_metrics_csv(path: Path, up_to_step: int | None = None) -> list[StepMetrics] | None:
    """The records of a trial's `metrics.csv` (None when the file does not exist). `up_to_step` keeps only the records up to that step (a resumed run).

    Errors: ValueError or OSError for a damaged file (the caller decides what to do).
    """
    if not path.is_file():
        return None
    records = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            def number(key: str) -> float | None:
                return float(row[key]) if row.get(key) not in (None, "") else None

            step = int(row["step"])
            if up_to_step is not None and step > up_to_step:
                continue
            records.append(StepMetrics(step, number("epoch_fraction") or 0.0, number("train_loss"), number("val_loss"), number("learning_rate"), number("grad_norm"), number("seconds_per_step") or 0.0, number("elapsed_sec") or 0.0, number("eta_sec")))
    return records


class _Reader:
    """Reads the files, collecting the `found` map and the `problems`."""

    def __init__(self) -> None:
        self.found: dict[str, str] = {}
        self.problems: list[str] = []

    def attempt(self, name: str, path: Path, read: Callable[[Path], Any], default: Any = None) -> Any:
        """`read(path)` or `default` when the file does not exist; a file that cannot be read is a problem (named), not an error."""
        if not path.is_file():
            self.found.setdefault(name, "missing")
            return default
        try:
            value = read(path)
        except (OSError, ValueError, KeyError, TypeError, EvaluatorError) as exc:
            self.problems.append(f"{path}: {type(exc).__name__}: {exc}")
            logger.warning("The report leaves out %s: %s cannot be read (%s: %s).", name, path, type(exc).__name__, exc)
            self.found.setdefault(name, f"unreadable ({path})")
            return default
        self.found[name] = str(path)
        return value


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _selection(path: Path) -> tuple[SelectionView, dict[str, Any]]:
    raw = _json(path)
    comparison = ComparisonResult(**raw["test_comparison"]) if raw.get("test_comparison") else None
    view = SelectionView(
        raw["verdict"], raw["winner_trial"], raw["test_status"], raw["reason"], tuple(ComparisonRow(**row) for row in raw["table"]), tuple(raw.get("warnings", [])), raw.get("test_error"), comparison,
        comparison.confidence if comparison is not None else None,
    )
    return view, raw


def _search(path: Path) -> SearchView:
    raw = _json(path)
    trials = tuple(TrialView(t["trial_id"], t["params"], t["status"], t["val_loss"], t["steps"], t["seconds"], t["error"]) for t in raw["trials"])
    return SearchView(raw["strategy"], raw["seed"], raw["stopped_reason"], raw["space"], trials, tuple(raw["finalists"]), raw["best_params"], raw["best_val_loss"], tuple(raw["notes"]))


def _curve(path: Path) -> LearningCurveView:
    raw = _json(path)
    points = tuple(CurvePointView(p["preset"], p["train_samples"], p["train_minutes"], p["val_wer"], p["val_cer"], p["val_loss"], p["steps"]) for p in raw["points"])
    return LearningCurveView(points, raw["trend"], raw["reading"], raw["hyperparameters_from"], raw["val_files"])


def _benchmark(path: Path) -> BenchmarkView:
    raw = _json(path)
    m = raw["measured"]
    return BenchmarkView(m["model"], m["threads"], m["seconds_per_step"], m["seconds_per_val_loss_example"], m["seconds_per_decode_clip"], tuple((p["part"], p["seconds"]) for p in raw["parts"]), raw["total_seconds"], tuple(raw["notes"]))


def _histories(reader: _Reader, name: str, folder: Path, prefix: str) -> dict[int, list[StepMetrics]]:
    out: dict[int, list[StepMetrics]] = {}
    if folder.is_dir():
        for sub in sorted(folder.glob(f"{prefix}[0-9][0-9][0-9]")):
            records = reader.attempt(f"{name}:{sub.name}", sub / "metrics.csv", read_metrics_csv)
            if records:
                out[int(sub.name[len(prefix) :])] = records
    reader.found = {k: v for k, v in reader.found.items() if not k.startswith(f"{name}:")}  # one line per kind of input, not one per trial (unreadable ones stay in `problems`)
    reader.found[name] = f"{len(out)} trial(s)" if out else "missing"
    return out


def load_report_data(
    root: Path,
    settings: ReportConfig,
    generated_at: str,
    registry_for: Callable[[str], RegistryReader] | None = None,
    curve_root: Path | None = None,
    benchmark_root: Path | None = None,
    registry_dir: str | None = None,
) -> LoadedReport:
    """The report views of the experiment folder `root` (see the module docstring). `generated_at` is passed in (the builder never reads the clock); `registry` gives the card of the registered entry
    and the number of times the Test set was evaluated: `registry_for(registry_dir)` is called with the registry folder that `selection.json` names (None: the run information says they are not known)."""
    reader = _Reader()
    reports = root / "reports"
    sel_loaded = reader.attempt("selection", reports / "selection.json", _selection)
    selection, sel_raw = (sel_loaded if sel_loaded is not None else (None, {}))
    search = reader.attempt("search", reports / "search_trials.json", _search)
    baseline: EvalResult | None = reader.attempt("baseline", reports / "eval_result_baseline.json", lambda p: eval_result_from_json(p.read_text(encoding="utf-8")))
    finalists: dict[int, EvalResult] = {}
    for path in sorted((reports / "finalists").glob("final_trial_*.val.json")) if (reports / "finalists").is_dir() else []:
        result = reader.attempt(f"finalist:{path.name}", path, lambda p: eval_result_from_json(p.read_text(encoding="utf-8")))
        if result is not None:
            finalists[int(path.name.split("_")[2].split(".")[0])] = result
    reader.found = {k: v for k, v in reader.found.items() if not k.startswith("finalist:")}
    reader.found["finalists"] = f"{len(finalists)} result(s)" if finalists else "missing"
    tests: dict[str, EvalResult] = {}
    for path in sorted(reports.glob("test_result_*.json")):
        result = reader.attempt(f"test:{path.name}", path, lambda p: eval_result_from_json(p.read_text(encoding="utf-8")))
        if result is not None:
            tests[path.stem[len("test_result_") :]] = result
    reader.found = {k: v for k, v in reader.found.items() if not k.startswith("test:")}
    reader.found["test_results"] = f"{len(tests)} result(s)" if tests else "missing"
    final_histories = _histories(reader, "final_histories", root / "metrics", "final_trial_")
    search_histories = _histories(reader, "search_histories", root / "metrics", "trial_")
    curve_path = (curve_root / "reports" / "learning_curve.json") if curve_root is not None else reports / "learning_curve.json"
    if curve_root is None and not curve_path.is_file():
        reader.found["learning_curve"] = "missing (pass --curve-from exp_NNNN to use the curve of another experiment)"
        curve = None
    else:
        curve = reader.attempt("learning_curve", curve_path, _curve)
    bench_path = (benchmark_root / "reports" / "benchmark.json") if benchmark_root is not None else reports / "benchmark.json"
    if benchmark_root is None and not bench_path.is_file():
        reader.found["benchmark"] = "missing (pass --benchmark-from exp_NNNN to use the benchmark of another experiment)"
        benchmark = None
    else:
        benchmark = reader.attempt("benchmark", bench_path, _benchmark)
    prepare = reader.attempt("prepare_summary", reports / "stage_results" / "prepare.json", lambda p: _json(p)["summary"], {})
    run = reader.attempt("run_info", root / "run_info.json", _json, {})
    config = reader.attempt("config", root / "config_full.yaml", lambda p: _read_yaml(p), {})
    card: ModelCard | None = None
    entry = sel_raw.get("entry_id")
    registry = registry_for(str(sel_raw["registry_dir"])) if registry_for is not None and entry and sel_raw.get("registry_dir") else None
    if registry is not None and entry:
        try:
            card = registry.card(str(entry))
            reader.found["registry_card"] = f"{entry}"
        except Exception as exc:  # a registry that cannot be read is a named problem, not a failure of the report
            reader.problems.append(f"registry card {entry}: {type(exc).__name__}: {exc}")
            logger.warning("The report leaves out the registry card of %s (%s: %s).", entry, type(exc).__name__, exc)
            reader.found["registry_card"] = f"unreadable ({entry})"
    else:
        reader.found["registry_card"] = "missing (nothing registered, or the registry was not given)"
    history_dir = sel_raw.get("registry_dir") or registry_dir  # the registry of the selection; else the configured one (an experiment without a selection can still show the history)
    history_registry = registry if (registry is not None and sel_raw.get("registry_dir")) else (registry_for(str(history_dir)) if registry_for is not None and history_dir else None)
    history, history_problem = _history(history_registry, reader)
    test_fp = card.test_fingerprint if card is not None and card.test_fingerprint else (test_set_fingerprint(next(iter(tests.values()))) if tests else None)
    count: int | None = None
    if registry is not None and card is not None and test_fp:  # a registry that could not be read gives no count: `unknown`, never a false 0
        try:
            count = registry.test_set_count(test_fp)
        except Exception as exc:
            reader.problems.append(f"test-set count: {type(exc).__name__}: {exc}")
    model_path = ((config or {}).get("model") or {}).get("path")
    data_section = (config or {}).get("data") or {}
    data_folder = ((data_section.get("extract") or {}).get("dir")) or ((data_section.get("source") or {}).get("path"))  # where the audio lives: the extraction folder, else the source
    info = RunInfoView(
        experiment_id=root.name,
        generated_at=generated_at,
        config_ref=(run or {}).get("config_ref"),
        base_model=(card.base_model_name if card is not None else Path(str(model_path)).name if model_path else None),
        registry_entry=str(entry) if entry else None,
        data_fingerprint=(prepare or {}).get("data_fingerprint") or (card.data_fingerprint if card is not None else None),
        split_fingerprint=(prepare or {}).get("split_fingerprint") or (card.split_fingerprint if card is not None else None),
        subsample_fingerprint=(prepare or {}).get("subsample_fingerprint") or (card.subsample_fingerprint if card is not None else None),
        test_fingerprint=test_fp,
        test_evaluations=count,
        library_versions=dict((run or {}).get("library_versions", {})),
        code_version=(run or {}).get("code_version"),
        seed=(run or {}).get("seed"),
        search_seconds=sum(t.seconds for t in search.trials) if search is not None else None,
        data_folder=str(data_folder) if data_folder else None,
    )
    if selection is not None and selection.confidence is None and card is not None and card.val is not None and card.val.confidence is not None:
        selection = replace(selection, confidence=card.val.confidence)  # the level of the table's intervals, from the card
    data = ReportData(info, settings, selection, baseline, finalists, tests, final_histories, search_histories, search, curve, benchmark, history, history_problem)
    return LoadedReport(data, dict(sorted(reader.found.items())), tuple(reader.problems))


def _history(registry: RegistryReader | None, reader: _Reader) -> tuple[tuple[HistoryPoint, ...], str | None]:
    """The history points of every registered entry (read from the stored cards; nothing is recomputed) and, when there are none, the reason in plain words. A card that cannot be read is a named problem."""
    if registry is None:
        reader.found["history"] = "missing (no registry was given and selection.json names none)"
        return (), "no registry was given: pass --set registry.dir=... or run the select_and_register stage"
    lister = getattr(registry, "entries", None)
    if lister is None:
        reader.found["history"] = "missing (this registry reader cannot list its entries)"
        return (), "this registry reader cannot list its entries"
    try:
        entries = list(lister())
    except Exception as exc:  # a registry that cannot be listed is a named problem
        reader.problems.append(f"registry entries: {type(exc).__name__}: {exc}")
        reader.found["history"] = "unreadable"
        return (), f"the registry could not be listed ({type(exc).__name__}: {exc})"
    points: list[HistoryPoint] = []
    skipped: list[str] = []
    for entry in entries:
        try:
            card = registry.card(entry.entry_id)
        except Exception as exc:
            reader.problems.append(f"registry card {entry.entry_id}: {type(exc).__name__}: {exc}")
            skipped.append(f"{entry.entry_id} (card unreadable)")
            continue
        if card.val is None:
            skipped.append(f"{entry.entry_id} (no Val result in its card)")
            continue
        if card.kind == "model":
            baseline, best = card.val.baseline_wer, card.val.wer
        else:  # a retained baseline: the card's Val result is the baseline's own; the best finalist is in the finalists table
            baseline, best = card.val.wer, min((f.val_wer for f in card.finalists), default=None)
        points.append(HistoryPoint(entry.entry_id, entry.experiment_id, card.kind, baseline, best))
    reader.found["history"] = f"{len(points)} registered experiment(s)" if points else "missing (the registry has no entry with a Val result)"
    if points:
        return tuple(points), ("; ".join(skipped) if skipped else None)
    return (), ("the registry has no entry yet" if not entries else "; ".join(skipped))


def _read_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))
