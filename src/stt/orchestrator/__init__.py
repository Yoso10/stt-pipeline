"""stt.orchestrator: the minimal Orchestrator and command line (C13, first end-to-end version: prepare, train, evaluate, smoke, check).

How to run: from the command line (PowerShell, project root; the working example is in docs/COMPONENTS.md, C13):
    $env:PYTHONPATH = "src"
    python -m stt smoke
Importing this package does not import torch or transformers. Only `wiring.py` names concrete component classes; the stages use Protocols.
"""

from stt.orchestrator.codeversion import MANIFEST_NAME, code_version, manifest_hash
from stt.orchestrator.config import PipelineConfig, SmokeConfig
from stt.orchestrator.errors import OrchestratorError, StageInterrupted
from stt.orchestrator.orchestrator import Orchestrator, fingerprint
from stt.orchestrator.sleepguard import NoopSleepGuard, SleepGuard, WindowsSleepGuard, default_sleep_guard
from stt.orchestrator.smoke import write_smoke_data
from stt.orchestrator.stageio import load_result, save_result
from stt.orchestrator.stages import EvaluateStage, PrepareStage, StageRegistry, TrainStage, default_stage_registry
from stt.orchestrator.stopflag import InterruptFlag
from stt.orchestrator.types import (
    PipelineStage,
    PreparedData,
    RunOutcome,
    RunPlan,
    Services,
    StageContext,
    StageResult,
    StopFlag,
)

__all__ = [
    "MANIFEST_NAME",
    "EvaluateStage",
    "InterruptFlag",
    "NoopSleepGuard",
    "Orchestrator",
    "OrchestratorError",
    "PipelineConfig",
    "PipelineStage",
    "PrepareStage",
    "PreparedData",
    "RunOutcome",
    "RunPlan",
    "Services",
    "SleepGuard",
    "SmokeConfig",
    "StageContext",
    "StageInterrupted",
    "StageRegistry",
    "StageResult",
    "StopFlag",
    "TrainStage",
    "WindowsSleepGuard",
    "code_version",
    "default_sleep_guard",
    "default_stage_registry",
    "fingerprint",
    "load_result",
    "manifest_hash",
    "save_result",
    "write_smoke_data",
]
