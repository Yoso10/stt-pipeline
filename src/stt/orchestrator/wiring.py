"""The composition root: the ONLY module of the Orchestrator that names concrete component classes.

How to run: this module is not a script; the command line uses it. Example (PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.orchestrator.wiring import section_registry; print(sorted(section_registry().sections))"
Example output:
    ['data', 'eval', 'features', 'filter', 'model', 'pipeline', 'run', 'smoke', 'split', 'subsample', 'text', 'train']
`build_runtime(config)` builds the services from the names in the config (through the registries of the components), the experiment store, the stage
registry and the exit-code mapping. Heavy libraries (torch, transformers) are imported lazily, only when the services that need them are used.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from stt.config import AppConfig, ConfigError, SectionRegistry
from stt.datasource import DataConfig, DataSourceError, create_data_source
from stt.evaluator import Comparator, ComparisonResult, DecodeSettings, EvalConfig, EvalItem, EvalRequest, Evaluator, EvaluatorConfigError, EvaluatorError, EvalResult, make_decode_settings
from stt.featurepipeline import FeaturePipeline, FeaturePipelineError, FeaturesConfig
from stt.modelloader import LoadedModel, ModelConfig, ModelLoaderError, create_loader
from stt.orchestrator.config import PipelineConfig, SmokeConfig
from stt.orchestrator.errors import OrchestratorError
from stt.orchestrator.stages import StageRegistry, default_stage_registry
from stt.orchestrator.types import FeatureBuilder, PreparedData, Services
from stt.runmanager import ExperimentStore, LogSetup, RunConfig, RunManagerError
from stt.samplefilter import FilterConfig, FilterError, SampleFilter
from stt.splitter import SplitConfig, SplitError, resolve_split
from stt.subsampler import SubsampleConfig, SubsampleConfigError, SubsampleError, resolve_subsample
from stt.textnormalizer import TextConfig, TextNormalizer, TextNormalizerError
from stt.trainercore import (
    ConsoleMetricsSink,
    CsvMetricsSink,
    SinkContext,
    SinkRegistry,
    TrainConfig,
    TrainerConfigError,
    TrainerCore,
    TrainerCoreError,
    TrainHyperparams,
    TrainRequest,
    TrainResult,
    hyperparams_from_config,
    resolve_threads,
)
from stt.trainercore.run_folder import checkpoints_dir, list_checkpoints

LIBRARIES = ("torch", "transformers", "numpy", "pydantic", "PyYAML", "soundfile", "scipy", "pyarrow", "accelerate", "safetensors")


def section_registry() -> SectionRegistry:
    """Every config section of the pipeline (the config loader itself is never edited)."""
    return SectionRegistry(
        {
            "data": DataConfig,
            "split": SplitConfig,
            "subsample": SubsampleConfig,
            "text": TextConfig,
            "filter": FilterConfig,
            "model": ModelConfig,
            "features": FeaturesConfig,
            "train": TrainConfig,
            "eval": EvalConfig,
            "run": RunConfig,
            "pipeline": PipelineConfig,
            "smoke": SmokeConfig,
        }
    )


def library_versions() -> dict[str, str]:
    """The installed versions of the libraries (written to `run_info.json`)."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def exit_code_for(exc: BaseException) -> int:
    """The process exit code of an exception: 2 config error, 3 data error, 4 training or evaluation failure, 1 for anything unexpected (a bug)."""
    if isinstance(exc, OrchestratorError):
        return exc.exit_code
    if isinstance(exc, ConfigError | ModelLoaderError | TrainerConfigError | EvaluatorConfigError | RunManagerError | TextNormalizerError | SubsampleConfigError):
        return 2
    if isinstance(exc, DataSourceError | SplitError | FilterError | FeaturePipelineError | SubsampleError):
        return 3
    if isinstance(exc, TrainerCoreError | EvaluatorError):
        return 4
    return 1


class DataPipeline:
    """DataSource, Splitter, TextNormalizer and SampleFilter, in the order of the pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def prepare(self, out_dir: Path) -> PreparedData:
        """Load, split, normalize and filter (the filter in the mode of the config; report mode removes nothing)."""
        c = self._config
        loaded = create_data_source(c.section("data", DataConfig), out_dir).load()
        split = resolve_split(c.section("split", SplitConfig), c.runtime.seed, loaded.samples)
        text = TextNormalizer(c.section("text", TextConfig))
        normalized = text.normalize(loaded.samples)
        by_id = {n.sample.id: n for n in normalized.samples}
        splits = {name: [by_id[s.id] for s in samples] for name, samples in split.splits.items()}
        filtered = SampleFilter(c.section("filter", FilterConfig)).filter_splits(splits)
        kept = {name: result.kept for name, result in filtered.items()}
        names = c.section("split", SplitConfig).names
        pipeline = c.sections.get("pipeline")
        subsampled = resolve_subsample(c.section("subsample", SubsampleConfig), c.runtime.seed, kept, names[0], str(getattr(pipeline, "eval_split", "val")))  # the Subsampler comes AFTER the filter; OFF by default
        kept = subsampled.splits
        counts: dict[str, str | float | int] = {"samples": len(loaded.samples), "rejected_at_load": len(loaded.rejections), "eval_version": c.section("text", TextConfig).eval_version}
        counts.update({f"kept_{name}": len(samples) for name, samples in kept.items()})
        reports = {
            "load_report.txt": loaded.report.summary(),
            "split_report.txt": split.report.summary(),
            "text_report.txt": normalized.report.summary(),
            "filter_report.txt": "\n\n".join(result.report.summary() for result in filtered.values()),
        }
        if subsampled.report.enabled:
            reports["subsample_report.txt"] = subsampled.report.summary()
        return PreparedData(kept, c.section("text", TextConfig).eval_version, text.build_profile("eval"), reports, counts)


class ModelWiring:
    """ModelLoader and FeaturePipeline bound to the config."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def load(self, path: Path | None = None) -> LoadedModel:
        """Load `model.path` (or the folder `path`)."""
        model = self._config.section("model", ModelConfig)
        return create_loader(model if path is None else model.model_copy(update={"path": path})).load()

    def features(self, model: LoadedModel) -> FeatureBuilder:
        """The FeaturePipeline for `model`."""
        return FeaturePipeline(self._config.section("features", FeaturesConfig), model, seed=self._config.runtime.seed)  # type: ignore[return-value]


class TrainingWiring:
    """TrainerCore bound to the config; the CSV sink writes into the experiment's `metrics/` folder."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def hyperparams(self) -> TrainHyperparams:
        """The hyperparameters of the config."""
        return hyperparams_from_config(self._config.section("train", TrainConfig), self._config.runtime.seed)

    def can_resume(self, output_dir: Path) -> bool:
        """True when the folder holds a complete checkpoint."""
        return bool(list_checkpoints(checkpoints_dir(output_dir)))

    def train(self, request: TrainRequest, metrics_dir: Path, resume: bool) -> TrainResult:
        """Run the training with `train.resume` set by the caller, never by the YAML (O69)."""
        config = self._config.section("train", TrainConfig).model_copy(update={"resume": resume})
        sinks = SinkRegistry({"csv": lambda ctx: CsvMetricsSink(SinkContext(metrics_dir, ctx.resume_step)), "console": ConsoleMetricsSink})
        return TrainerCore(config, sink_registry=sinks).train(request)


class EvaluationWiring:
    """Evaluator, Comparator and the Hugging Face transcriber bound to the config (the transcriber is imported when the first evaluation runs)."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._evaluator: Evaluator | None = None

    def decode_settings(self, model: LoadedModel, items: Sequence[EvalItem]) -> DecodeSettings:
        """The decoding settings of `eval`."""
        return make_decode_settings(self._config.section("eval", EvalConfig), model, items)

    def evaluate(self, request: EvalRequest) -> EvalResult:
        """Score one model on one split."""
        if self._evaluator is None:
            from stt.evaluator.hf_transcriber import HfTranscriber  # imported only when an evaluation runs

            threads = resolve_threads(self._config.section("train", TrainConfig))
            self._evaluator = Evaluator(self._config.section("eval", EvalConfig), HfTranscriber(threads), self._config.runtime.seed)
        return self._evaluator.evaluate(request)

    def compare(self, baseline: EvalResult, candidate: EvalResult) -> ComparisonResult:
        """Compare a candidate with the baseline."""
        return Comparator(self._config.section("eval", EvalConfig), self._config.runtime.seed).compare(baseline, candidate)


def build_services(config: AppConfig) -> Services:
    """All the services of a run, built from the config."""
    return Services(DataPipeline(config), ModelWiring(config), TrainingWiring(config), EvaluationWiring(config), exit_code_for)


@dataclass(frozen=True)
class Runtime:
    """What the command line needs: the experiment store, the log setup, the services and the stage registry."""

    store: ExperimentStore
    log_setup: LogSetup
    services: Services
    stages: StageRegistry
    now: Callable[[], str]


def build_runtime(config: AppConfig, now: Callable[[], str]) -> Runtime:
    """Wire everything for `config`. `now` gives the date and time text that is passed into new experiments (the RunManager never reads the clock)."""
    run = config.section("run", RunConfig)
    return Runtime(ExperimentStore(run), LogSetup(run), build_services(config), default_stage_registry(), now)

