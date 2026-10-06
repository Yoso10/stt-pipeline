"""Typed data objects and Protocols of the Orchestrator component.

How to run: this module is not a script; the other modules and later components import it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.types import StageResult; print(list(StageResult.__dataclass_fields__))"
Example output:
    ['status', 'summary', 'artifacts', 'error']

The stages talk to the services only through the small Protocols below; `wiring.py` is the only module that names concrete classes.
The Protocols repeat the shape of the real components, so the real classes fit without an adapter, and a unit test can pass fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from stt.config import AppConfig
from stt.evaluator import ComparisonResult, DecodeSettings, EvalItem, EvalRequest, EvalResult
from stt.featurepipeline import FeatureResult
from stt.modelloader import LoadedModel
from stt.registry import RegisterRequest, RegistryEntry
from stt.runmanager import ExperimentHandle
from stt.searcher import SearchResult, SearchSpec, TrialRunner
from stt.selector import SelectionRequest, SelectionResult
from stt.textnormalizer import NormalizedSample
from stt.trainercore import TrainHyperparams, TrainRequest, TrainResult

Summary = Mapping[str, str | float | int]
StageStatus = Literal["ok", "skipped", "failed"]


@dataclass(frozen=True)
class StageResult:
    """What a stage reports. `skipped` means the completion marker was valid and the stored result of the earlier run is returned."""

    status: StageStatus
    summary: Summary = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class PreparedData:
    """The result of the prepare stage that later stages use: the kept samples per split (normalized), the eval profile, and plain-text reports.

    `reports` maps a report file name to its text (the stage writes them through the RunManager). `counts` are the numbers for the summary.
    `fingerprints` (DECISIONS 55, O83) holds the `data`, `split` and `subsample` fingerprints for the model card; `subsample` is None when the Subsampler is off.
    """

    kept: Mapping[str, Sequence[NormalizedSample]]
    eval_version: int
    normalize: Callable[[str], str]
    reports: Mapping[str, str]
    counts: Summary
    fingerprints: Mapping[str, str | None] = field(default_factory=dict)


class DataService(Protocol):
    """DataSource, Splitter, TextNormalizer and SampleFilter in one call (done in the order of the pipeline)."""

    def prepare(self, out_dir: Path) -> PreparedData:
        """Load, split, normalize and filter. `out_dir` receives `rejected.csv`. Raises the components' errors."""
        ...


class FeatureBuilder(Protocol):
    """FeaturePipeline for one model."""

    def build(self, samples: Sequence[NormalizedSample], split: str) -> FeatureResult:
        """The lazy dataset and the report of one split."""
        ...

    def build_collator(self) -> Callable[[Sequence[Any]], Mapping[str, object]]:
        """The batch maker for the trainer."""
        ...


class ModelService(Protocol):
    """ModelLoader and FeaturePipeline factory, bound to the config."""

    def load(self, path: Path | None = None) -> LoadedModel:
        """Load the model of `model.path` (or the folder `path`). Every call returns an independent model."""
        ...

    def features(self, model: LoadedModel) -> FeatureBuilder:
        """The FeaturePipeline of the config for `model`."""
        ...


class TrainingService(Protocol):
    """TrainerCore bound to the config, with the metrics sink of the trial in the experiment's `metrics/` folder."""

    def hyperparams(self) -> TrainHyperparams:
        """The hyperparameters of the config (the seed falls back to `runtime.seed`)."""
        ...

    def can_resume(self, output_dir: Path) -> bool:
        """True when `output_dir` holds a complete checkpoint to resume from."""
        ...

    def train(self, request: TrainRequest, metrics_dir: Path, resume: bool, threads: int | None = None) -> TrainResult:
        """Run one training. `resume` is set by the Orchestrator itself (never from the YAML, O69). `threads` (the search gives each trial its share, DECISIONS 55)
        replaces `train.cpu_threads` for this run; None keeps the config."""
        ...


class EvaluationService(Protocol):
    """Evaluator and Comparator bound to the config."""

    def decode_settings(self, model: LoadedModel, items: Sequence[EvalItem]) -> DecodeSettings:
        """The decoding settings for these items."""
        ...

    def evaluate(self, request: EvalRequest) -> EvalResult:
        """Score one model on one split."""
        ...

    def compare(self, baseline: EvalResult, candidate: EvalResult) -> ComparisonResult:
        """Compare a candidate with the baseline."""
        ...


@dataclass(frozen=True)
class ScreeningSet:
    """The data one search trial uses: the screening Train samples and the Val samples (the Subsampler preset `search.screening.preset` and `subsample.val_samples`
    applied to the prepared data). `report` is the Subsampler's text; `fingerprint` identifies the choice. The samples are plain data, so the set can be sent to a worker process."""

    train_split: str
    val_split: str
    train: tuple[NormalizedSample, ...]
    val: tuple[NormalizedSample, ...]
    report: str
    fingerprint: str


class SearchService(Protocol):
    """The HyperparamSearcher and the Subsampler's screening choice, bound to the config (DECISIONS 55)."""

    services_provider: Callable[[Path], Services]
    """A picklable function that rebuilds the services in a worker process from the stored `config_full.yaml` of the experiment."""

    def spec(self, study_dir: Path) -> SearchSpec:
        """The typed search spec of `search` (the seed from `runtime.seed`, the thread budget from `train.cpu_threads`), with the study stored in `study_dir`."""
        ...

    def screening(self, prepared: PreparedData) -> ScreeningSet:
        """The screening Train and Val samples of the prepared data."""
        ...

    def search(self, spec: SearchSpec, runner: TrialRunner) -> SearchResult:
        """Run the search. Raises SearchConfigError, SearchError and SearchInterrupted."""
        ...

    def subset(self, prepared: PreparedData, preset: str) -> ScreeningSet:
        """The Train samples of the Subsampler preset `preset` (scope train_only, a fixed Val set that is the same for every preset), for the learning curve. Raises SubsampleError."""
        ...


class SelectionService(Protocol):
    """The Selector bound to the `select` section (DECISIONS 57)."""

    def select(self, request: SelectionRequest) -> SelectionResult:
        """Choose the winner, or keep the baseline. The Test is evaluated through `request.evaluate_test`, at most once per model. Raises SelectorError."""
        ...


class RegistryService(Protocol):
    """The ModelRegistry bound to the `registry` section (DECISIONS 57)."""

    def location(self) -> Path:
        """The registry folder (`registry.dir`)."""
        ...

    def entries(self) -> list[RegistryEntry]:
        """The registered entries, lowest number first (read only)."""
        ...

    def register(self, request: RegisterRequest) -> RegistryEntry:
        """Store the selection result as the next entry. Raises RegistryError."""
        ...


@dataclass(frozen=True)
class Services:
    """Everything the stages may use, built once by `wiring.build_services`. `exit_code_for` maps an exception to the process exit code.
    `search`, `selection`, `registry` and `now` are None where they are not wired (a test with fakes); `now` gives the date and time text of a registration."""

    data: DataService
    models: ModelService
    training: TrainingService
    evaluation: EvaluationService
    exit_code_for: Callable[[BaseException], int]
    search: SearchService | None = None
    selection: SelectionService | None = None
    registry: RegistryService | None = None
    now: Callable[[], str] | None = None


class StopFlag(Protocol):
    """Set when the user interrupts (Ctrl+C or a closing console). Calling it returns True once a stop was requested."""

    def __call__(self) -> bool:
        """True after a stop was requested."""
        ...


@dataclass
class StageContext:
    """What a stage receives. `previous` holds the results of the stages that ran before it in this plan (skipped ones included).
    `options` are the command-line choices of this run, for example `split` for the evaluate stage."""

    config: AppConfig
    experiment: ExperimentHandle
    services: Services
    stop_flag: StopFlag
    previous: Mapping[str, StageResult]
    options: Mapping[str, str] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)


class PipelineStage(Protocol):
    """One stage of the pipeline. A new stage is a new class registered by name."""

    name: str
    depends_on: tuple[str, ...]

    def config_slice(self, config: AppConfig) -> Mapping[str, Any]:
        """The part of the config this stage depends on (as plain data); a change makes the stage run again."""
        ...

    def run(self, ctx: StageContext) -> StageResult:
        """Do the work. Raise on failure (the Orchestrator maps the exception to the exit code); raise StageInterrupted on a requested stop."""
        ...


@dataclass(frozen=True)
class RunPlan:
    """The stages to run, in order, on one experiment."""

    stages: list[str]
    experiment: ExperimentHandle
    options: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOutcome:
    """The result of a run. `verdict` is `n/a` until the Selector exists (Wave 2)."""

    exit_code: int
    verdict: str
    experiment_id: str
    stage_results: Mapping[str, StageResult]
    status: str
    message: str | None = None
