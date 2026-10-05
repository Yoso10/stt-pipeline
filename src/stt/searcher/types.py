"""Typed data objects and Protocols of the HyperparamSearcher component.

How to run: this module is not a script; the other modules and later components import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.searcher.types import ParamRange; print(ParamRange('log_float', low=1e-5, high=1e-4))"
Example output:
    ParamRange(kind='log_float', low=1e-05, high=0.0001, choices=(), points=3)

No Optuna import here (it is imported only by `optuna_strategy.py`). Additions to the brief (DECISIONS 49): `Trial.threads`, `SearchSpec.finalists_k`,
`SearchSpec.total_threads`, `ParamRange.points`, `PruningSpec`, `TrialOutcome.reports`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from stt.trainercore import StepMetrics

ParamValue = float | int | str
Kind = Literal["log_float", "float", "int", "categorical"]
Status = Literal["complete", "pruned", "failed"]
StoppedReason = Literal["n_trials", "time_budget", "too_many_failures", "space_exhausted"]


@dataclass(frozen=True)
class ParamRange:
    """The range of one searched parameter. `points` is how many values the grid strategy takes from a `float` or `log_float` range (evenly or geometrically spaced)."""

    kind: Kind
    low: float | None = None
    high: float | None = None
    choices: tuple[ParamValue, ...] = ()
    points: int = 3


@dataclass(frozen=True)
class PruningSpec:
    """`enabled`, and `min_evals`: no trial is pruned before it has had this many Val evaluations."""

    enabled: bool = True
    min_evals: int = 3


@dataclass(frozen=True)
class SearchSpec:
    """Everything one search needs. `total_threads` is the CPU thread budget of the whole search (the caller resolves `train.cpu_threads`); every trial gets
    `total_threads // parallel_trials` of it (at least 1). `finalists_k` is `search.finalists.k`."""

    parameters: Mapping[str, ParamRange]
    strategy: str
    n_trials: int
    time_budget_minutes: float | None
    pruning: PruningSpec
    parallel_trials: int
    max_failed_ratio: float
    min_trials_before_abort: int
    finalists_k: int
    study_dir: Path | None
    seed: int
    total_threads: int = 1


@dataclass(frozen=True)
class Trial:
    """One hyperparameter set to try. `threads` is the CPU thread limit this trial must use (total threads divided by the parallel trials)."""

    trial_id: int
    params: Mapping[str, ParamValue]
    threads: int = 1


@dataclass(frozen=True)
class TrialOutcome:
    """What a trial returned (the runner's answer) or what the searcher made of it. `reports` is the list of (step, Val loss) of the evaluations the searcher saw."""

    status: Status
    val_loss: float | None = None
    steps: int = 0
    seconds: float = 0.0
    model_dir: Path | None = None
    error: str | None = None
    reports: tuple[tuple[int, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrialRecord:
    """A trial and its outcome."""

    trial: Trial
    outcome: TrialOutcome


class TrialRunner(Protocol):
    """Runs ONE trial (the Orchestrator builds it from the Subsampler, TrainerCore and Evaluator). For `parallel_trials` above 1 the runner must be
    picklable (it is sent to a worker process). `report` is called at every metrics record; it returns True when the trial must stop (pruned, or the
    time budget is over): the runner then stops training and returns its outcome."""

    def __call__(self, trial: Trial, report: Callable[[StepMetrics], bool]) -> TrialOutcome:
        """Run the trial and return its outcome (`complete`, `pruned` or `failed`)."""
        ...


@dataclass(frozen=True)
class Suggestion:
    """What a strategy proposes: the parameters of the next trial."""

    params: Mapping[str, ParamValue]


class SearchStrategy(Protocol):
    """Proposes parameter sets and learns from outcomes. A new strategy is a new class in the registry."""

    def ask(self, trial_id: int) -> Suggestion | None:
        """The next parameter set, or None when the space is exhausted (the grid)."""
        ...

    def record_eval(self, trial_id: int, step: int, val_loss: float) -> None:
        """Tell the strategy about an intermediate Val loss of a running trial (optional use)."""
        ...

    def tell(self, trial_id: int, params: Mapping[str, ParamValue], outcome: TrialOutcome) -> None:
        """Tell the strategy how a trial ended (`complete`, `pruned` or `failed`)."""
        ...

    def replay(self, records: list[TrialRecord]) -> None:
        """Teach the strategy the trials of an earlier part of a stored study (a resume)."""
        ...


@dataclass(frozen=True)
class SearchReport:
    """What the search did. `notes` carry the honest remarks (the sequence can vary with pruning or parallel trials; the best value is at the edge of its range)."""

    strategy: str
    seed: int
    n_started: int
    n_complete: int
    n_pruned: int
    n_failed: int
    n_resumed: int
    best_params: Mapping[str, ParamValue] | None
    best_val_loss: float | None
    seconds: float
    parallel_trials: int
    threads_per_trial: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        """The report as plain text."""
        best = "none" if self.best_val_loss is None else f"{self.best_val_loss:.4f} with {dict(self.best_params or {})}"
        lines = [
            f"search ({self.strategy}, seed {self.seed}): {self.n_started} trials ({self.n_complete} complete, {self.n_pruned} pruned, {self.n_failed} failed; {self.n_resumed} from an earlier run)",
            f"  best Val loss {best}",
            f"  {self.seconds:.1f} s, parallel trials {self.parallel_trials}, {self.threads_per_trial} thread(s) per trial",
        ]
        lines.extend(f"  NOTE: {n}" for n in self.notes)
        return "\n".join(lines)


@dataclass(frozen=True)
class SearchResult:
    """The output. `ranked` holds the completed trials by Val loss, best first, at most `finalists_k`. The searcher never declares a winner."""

    trials: list[TrialRecord]
    ranked: list[TrialRecord]
    stopped_reason: StoppedReason
    report: SearchReport
