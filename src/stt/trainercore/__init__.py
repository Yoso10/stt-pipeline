"""stt.trainercore: run ONE training run on a model and datasets and return the trained model folder (C6 TrainerCore).

How to run: from Python (PowerShell, project root; the working example is in docs/COMPONENTS.md, C6):
    $env:PYTHONPATH = "src"
    python -c "from stt.trainercore import default_freeze_registry; print(default_freeze_registry().names())"
Example output:
    ['freeze_encoder', 'none']
Importing this package does not import torch or transformers (the Hugging Face adapter is imported when a run starts).
"""

from stt.trainercore.config import (
    EarlyStoppingConfig,
    HyperparamsConfig,
    TrainConfig,
    hyperparams_from_config,
    resolve_seed,
    resolve_threads,
)
from stt.trainercore.errors import TrainerConfigError, TrainerCoreError
from stt.trainercore.freeze import FreezeEncoder, NoFreeze
from stt.trainercore.monitor import EvalMonitor, EvalVerdict
from stt.trainercore.registry import FreezeRegistry, SinkRegistry, default_freeze_registry, default_sink_registry
from stt.trainercore.sinks import ConsoleMetricsSink, CsvMetricsSink
from stt.trainercore.trainer import TrainerCore, check_hyperparams
from stt.trainercore.types import (
    FreezePolicy,
    FreezeReport,
    MetricsSink,
    SinkContext,
    StepMetrics,
    StoppedReason,
    TrainHyperparams,
    TrainRequest,
    TrainResult,
    TrainTimings,
)

__all__ = [
    "ConsoleMetricsSink",
    "CsvMetricsSink",
    "EarlyStoppingConfig",
    "EvalMonitor",
    "EvalVerdict",
    "FreezeEncoder",
    "FreezePolicy",
    "FreezeRegistry",
    "FreezeReport",
    "HyperparamsConfig",
    "MetricsSink",
    "NoFreeze",
    "SinkContext",
    "SinkRegistry",
    "StepMetrics",
    "StoppedReason",
    "TrainConfig",
    "TrainHyperparams",
    "TrainRequest",
    "TrainResult",
    "TrainTimings",
    "TrainerConfigError",
    "TrainerCore",
    "TrainerCoreError",
    "check_hyperparams",
    "default_freeze_registry",
    "default_sink_registry",
    "hyperparams_from_config",
    "resolve_seed",
    "resolve_threads",
]
