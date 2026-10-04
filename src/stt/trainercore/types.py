"""Typed data objects of the TrainerCore component and its two small Protocols.

How to run: this module is not a script; the other modules and later components import it. Example
(PowerShell, project root):
    $env:PYTHONPATH = "src"
    python -c "from stt.trainercore.types import TrainHyperparams; print(list(TrainHyperparams.__dataclass_fields__))"
Example output:
    ['learning_rate', 'max_steps', 'batch_size', 'grad_accum_steps', 'warmup_steps', 'weight_decay', 'max_grad_norm', 'seed']

No torch or transformers import here. `TrainRequest.model` is the opaque model of the ModelLoader.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from stt.featurepipeline import FeatureDataset, FeatureExample
from stt.modelloader import LoadedModel

StoppedReason = Literal["max_steps", "early_stopping", "time_limit", "stopped_by_callback"]
Collator = Callable[[Sequence[FeatureExample]], Mapping[str, object]]


@dataclass(frozen=True)
class TrainHyperparams:
    """The values of one run (produced by the config now, by the HyperparamSearcher later). One step = one optimizer update."""

    learning_rate: float
    max_steps: int
    batch_size: int
    grad_accum_steps: int
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    seed: int


@dataclass(frozen=True)
class StepMetrics:
    """One metrics record. `val_loss` is set only on an evaluation record; the other values of an evaluation record are the
    latest known ones. `seconds_per_step` is the mean of the training steps since the previous log record (evaluation and saving
    excluded); `elapsed_sec` is cumulative over resumed calls; `eta_sec` is an estimate for the remaining steps (None when unknown)."""

    step: int
    epoch_fraction: float
    train_loss: float | None
    val_loss: float | None
    learning_rate: float | None
    grad_norm: float | None
    seconds_per_step: float
    elapsed_sec: float
    eta_sec: float | None


@dataclass(frozen=True)
class FreezeReport:
    """What a freeze policy did: counts of parameters and the groups that were frozen (empty for `none`)."""

    policy: str
    total_params: int
    trainable_params: int
    frozen_params: int
    frozen_groups: tuple[str, ...]


@dataclass(frozen=True)
class TrainTimings:
    """Measured times of the run, for planning (approximate for the evaluation and saving parts)."""

    total_seconds: float
    steps_run: int
    mean_seconds_per_step: float
    eval_runs: int
    eval_seconds: float
    save_seconds: float


@dataclass(frozen=True)
class TrainRequest:
    """Everything ONE run needs. `on_eval` is called after each Val evaluation with its record; True stops the run (pruning).

    `source_config_ref` (optional, an addition to the brief) is written into `pipeline_model_info.json` of the model folder; the
    RunManager will pass the path of the experiment's config or diff file.

    `model_name` (optional, owner approved 2026-10-05, O70) is the prefix of the returned model folder: the folder is `<output_dir>/<model_name>_step_<step:05d>`,
    for example `exp_0004_trial_001_step_00100`; None keeps the folder name `model`. `stop_flag` (optional, O65) is checked after every step: when it returns
    True the run saves a checkpoint and stops with `stopped_by_callback`, so a later call with `train.resume` true continues it.
    """

    model: LoadedModel
    train: FeatureDataset
    val: FeatureDataset
    collator: Collator
    hyperparams: TrainHyperparams
    output_dir: Path
    on_eval: Callable[[StepMetrics], bool] | None = None
    source_config_ref: str | None = None
    model_name: str | None = None
    stop_flag: Callable[[], bool] | None = None


@dataclass(frozen=True)
class TrainResult:
    """The outcome. `model_dir` is the returned model (the best checkpoint by Val loss; the latest one when there was no evaluation).

    `best_step` is the step of that model and `best_val_loss` its Val loss (None when no evaluation happened). `history` holds the
    records of the whole run, including the part before a resume. `seed` is the seed that was applied.
    """

    model_dir: Path
    history: tuple[StepMetrics, ...]
    best_step: int
    best_val_loss: float | None
    stopped_reason: StoppedReason
    timings: TrainTimings
    freeze_report: FreezeReport
    seed: int


class FreezePolicy(Protocol):
    """Decides which parameters are trained. A new policy is a new class in the registry."""

    def apply(self, model: object) -> FreezeReport:
        """Set `requires_grad` of the parameters of `model` IN PLACE and report what was frozen."""
        ...


class MetricsSink(Protocol):
    """Receives every metrics record while the run is in progress."""

    def write(self, record: StepMetrics) -> None:
        """Handle one record. A failing sink must not stop the training."""
        ...


@dataclass(frozen=True)
class SinkContext:
    """What a sink factory may use: the run's output folder and, when resuming, the step that was resumed from."""

    output_dir: Path
    resume_step: int | None = None


@dataclass
class RunState:
    """Bookkeeping of a run that is saved in every checkpoint, so that a resume continues it exactly."""

    best_step: int | None = None
    best_val_loss: float | None = None
    bad_evals: int = 0
    evals_done: int = 0
    first_eval_step: int | None = None
    elapsed_sec: float = 0.0
    history: list[StepMetrics] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingJob:
    """What the Hugging Face adapter needs (made by TrainerCore from the request and the config)."""

    request: TrainRequest
    checkpoints_dir: Path
    resume_checkpoint: Path | None
    state: RunState
    sinks: tuple[MetricsSink, ...]
    threads: int
    log_every_steps: int
    eval_every_steps: int
    save_every_steps: int
    patience: int
    min_delta: float
    max_runtime_seconds: float | None
    clock: Callable[[], float]


@dataclass(frozen=True)
class TrainingOutcome:
    """What the adapter returns: the final bookkeeping and the reason the run ended."""

    state: RunState
    stopped_reason: StoppedReason
    timings: TrainTimings
