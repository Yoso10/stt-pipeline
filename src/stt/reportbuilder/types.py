"""Typed data objects of the ReportBuilder component and the DiagnosisRule Protocol.

How to run: this module is not a script; the other modules and the callers import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.reportbuilder.types import ReportData; print(list(ReportData.__dataclass_fields__)[:4])"
Example output:
    ['run_info', 'settings', 'selection', 'baseline_val']
No file access and no model imports. The VIEWS below mirror what an experiment already STORES (the files of `reports/` and the registry card), so that a report is made without recomputing
anything: a stored `selection.json` is a `SelectionView`, `search_trials.json` a `SearchView`, the lossless EvalResult files are `EvalResult` (the codec of C7), `learning_curve.json` a
`LearningCurveView`, `benchmark.json` a `BenchmarkView`. Every input is optional (None or empty = "not available in this run"). DEVIATION FROM THE BRIEF, to be approved (DECISIONS 63): the brief's
`selection: SelectionResult` and `search: SearchResult` are these views, because a SelectionResult holds live objects that cannot be rebuilt from the stored files without recomputing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from stt.evaluator import ComparisonResult, EvalResult
from stt.reportbuilder.config import ReportConfig
from stt.selector import ComparisonRow
from stt.trainercore import StepMetrics


@dataclass(frozen=True)
class SelectionView:
    """The stored selection (`reports/selection.json`): the verdict, the reason, the Test status and comparison, and the table of the finalists (winner first).
    `confidence` is the level of the intervals of the table (for example 0.9), when known."""

    verdict: str
    winner_trial: int | None
    test_status: str
    reason: str
    table: tuple[ComparisonRow, ...]
    warnings: tuple[str, ...] = ()
    test_error: str | None = None
    test_comparison: ComparisonResult | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class TrialView:
    """One search trial (`reports/search_trials.json`)."""

    trial_id: int
    params: Mapping[str, float | int | str]
    status: str
    val_loss: float | None
    steps: int
    seconds: float
    error: str | None = None


@dataclass(frozen=True)
class SearchView:
    """The stored search. `space` maps a parameter name to its range (`kind`, `low`, `high`, `choices`)."""

    strategy: str
    seed: int
    stopped_reason: str
    space: Mapping[str, Mapping[str, Any]]
    trials: tuple[TrialView, ...]
    finalists: tuple[int, ...]
    best_params: Mapping[str, float | int | str] | None
    best_val_loss: float | None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CurvePointView:
    """One size of the learning curve."""

    preset: str
    train_samples: int
    train_minutes: float
    val_wer: float
    val_cer: float | None
    val_loss: float | None
    steps: int


@dataclass(frozen=True)
class LearningCurveView:
    """The stored learning curve (`reports/learning_curve.json`), the points in increasing Train size."""

    points: tuple[CurvePointView, ...]
    trend: str
    reading: str
    hyperparameters_from: str
    val_files: int


@dataclass(frozen=True)
class BenchmarkView:
    """The stored benchmark (`reports/benchmark.json`): the measured times and the planned parts of a run."""

    model: str
    threads: int
    seconds_per_step: float
    seconds_per_val_loss_example: float
    seconds_per_decode_clip: float
    parts: tuple[tuple[str, float], ...]
    total_seconds: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunInfoView:
    """What the run information section shows. `generated_at` is passed in (the builder never reads the clock). `test_evaluations` is how many registered entries have been evaluated
    on this Test set, passed in by the Orchestrator from the registry (None when unknown)."""

    experiment_id: str
    generated_at: str
    config_ref: str | None = None
    base_model: str | None = None
    registry_entry: str | None = None
    data_fingerprint: str | None = None
    split_fingerprint: str | None = None
    subsample_fingerprint: str | None = None
    test_fingerprint: str | None = None
    test_evaluations: int | None = None
    library_versions: Mapping[str, str] = field(default_factory=dict)
    code_version: str | None = None
    seed: int | None = None
    search_seconds: float | None = None
    data_folder: str | None = None  # the folder the audio lives in (`data.extract.dir`, else `data.source.path`): the listening list shows paths relative to it


@dataclass(frozen=True)
class HistoryPoint:
    """One registered experiment in the history chart. The numbers are read from its stored model card (nothing is recomputed): for a registered model `baseline_wer` is the baseline Val WER of the comparison and
    `best_wer` the Val WER of the registered winner; for a retained baseline `baseline_wer` is the baseline's own Val WER and `best_wer` the best finalist's Val WER (None when the card lists no finalist)."""

    entry_id: str
    experiment_id: str
    kind: str
    baseline_wer: float | None
    best_wer: float | None


@dataclass(frozen=True)
class ModelNumbers:
    """The numbers of one scored model (from `evaluation_baseline.json` / `evaluation_trained.json` / `test_result.json`)."""

    wer: float
    cer: float | None
    n_files: int
    n_words: int
    empty_output: int = 0
    hit_length_cap: int = 0


@dataclass(frozen=True)
class ClipRow:
    """One Val clip for the table of best and worst clips: the reference and what the two models wrote (from the two `predictions_*.csv` files)."""

    sample_id: str
    group_id: str
    reference: str
    baseline_text: str
    trained_text: str
    baseline_wer: float
    trained_wer: float


@dataclass(frozen=True)
class FinalTestView:
    """`test_result.json`: the Test split scored ONCE for the trained model and the baseline (the `final-test` command, DECISIONS 90)."""

    split: str
    baseline: ModelNumbers
    trained: ModelNumbers
    relative_reduction: float | None
    delta_wer: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    n_groups: int
    verdict: str
    status: str  # "confirmed" | "not_confirmed"
    explanation: str


@dataclass(frozen=True)
class TrainedRunView:
    """An experiment made by `train` + `evaluate` (no search): the loss history of the training, the baseline and trained numbers, the stored comparison and the per-clip texts (DECISIONS 90)."""

    history: Sequence[StepMetrics]
    baseline: ModelNumbers
    trained: ModelNumbers
    comparison: ComparisonResult | None = None
    clips: Sequence[ClipRow] = ()
    max_steps: int | None = None
    final_test: FinalTestView | None = None


@dataclass(frozen=True)
class ReportData:
    """Everything one report needs. `finalist_val` maps a trial id to the lossless Val result of its retrained model; `test_results` maps `baseline` or `trial_<id>` to the stored Test result
    of the one-time look; `final_histories` and `search_histories` map a trial id to its training records (the metrics file of the trial)."""

    run_info: RunInfoView
    settings: ReportConfig
    selection: SelectionView | None = None
    baseline_val: EvalResult | None = None
    finalist_val: Mapping[int, EvalResult] = field(default_factory=dict)
    test_results: Mapping[str, EvalResult] = field(default_factory=dict)
    final_histories: Mapping[int, Sequence[StepMetrics]] = field(default_factory=dict)
    search_histories: Mapping[int, Sequence[StepMetrics]] = field(default_factory=dict)
    search: SearchView | None = None
    learning_curve: LearningCurveView | None = None
    benchmark: BenchmarkView | None = None
    history: tuple[HistoryPoint, ...] = ()  # one point per registered experiment, in registry order (Wave 3 task 7); empty = no history
    history_problem: str | None = None  # why there is no history, in plain words (named in the report instead of an empty chart)
    trained_run: TrainedRunView | None = None  # a `train` + `evaluate` experiment (DECISIONS 90): the convergence view, the clips and, when it exists, the final Test

    def winner_val(self) -> EvalResult | None:
        """The Val result of the winner, or of the baseline when the baseline stays (the model the per-file sections describe)."""
        if self.selection is not None and self.selection.winner_trial is not None:
            return self.finalist_val.get(self.selection.winner_trial)
        return self.baseline_val

    def winner_history(self) -> Sequence[StepMetrics] | None:
        """The training records of the winner's retraining, when there is a winner and its history was passed."""
        if self.selection is not None and self.selection.winner_trial is not None:
            return self.final_histories.get(self.selection.winner_trial)
        return None


Severity = Literal["info", "warning"]


@dataclass(frozen=True)
class DiagnosisHint:
    """A hint of a rule: what it noticed, the numbers behind it and what it cannot tell. It is worded as a hint, never as a stated fact."""

    rule: str
    severity: Severity
    message: str
    evidence: Mapping[str, float | str]
    caveat: str


@dataclass(frozen=True)
class CorrelationRow:
    """The rank correlation of one attribute with the per-file WER. `rho` is None when there are too few points or the data is constant (the `note` says which)."""

    attribute: str
    n: int
    rho: float | None
    note: str


@dataclass(frozen=True)
class ReportOutput:
    """The report: one HTML text, the names of the sections that were produced with data, and the warnings (a rule that failed, thinned charts)."""

    html: str
    sections: list[str]
    warnings: list[str]


class DiagnosisRule(Protocol):
    """A diagnosis rule. A new rule is a new class in the registry. It returns a hint or None; it must not change the data."""

    name: str

    def evaluate(self, data: ReportData) -> DiagnosisHint | None:
        """Look at the data and return a hint, or None when there is nothing to say (or the input is missing)."""
        ...
